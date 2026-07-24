// engine/plugin/wasm/WasmParser.cpp — WASM ParserPlugin host side (Increment B, #10).
// See WasmParser.h. The "netvis" parser import set: window-bounded reads (host-
// marked) + append-only model-mutating commands. NO import returns a weight buffer;
// tensors are declared by (offset,len). Compiles to safe stubs without NETVIS_ENABLE_WASM.
#include "engine/plugin/wasm/WasmParser.h"

#include "engine/plugin/wasm/WasmRuntime.h"

#if defined(NETVIS_ENABLE_WASM)

#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "engine/plugin/Registry.h"
#include "ir/IR.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4200)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
extern "C" {
#include "wasm3.h"
#include "m3_env.h"
}
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4100)  // m3ApiRawFunction fixed-signature unused params
#endif

namespace netvis::plugin::wasm {

namespace {

using netvis::ByteReader;

// SDK caps mirrored (plugins/sdk/netvis_plugin.h).
constexpr int64_t  kSniffHead = 4096;
constexpr int64_t  kSniffTail = 4096;
constexpr int32_t  kReadChunkCap = 4096;
constexpr int32_t  kMaxRank = 8;
constexpr int32_t  kMaxNodeIO = 4096;
constexpr int32_t  kMaxAttrLen = 65536;
constexpr int32_t  kMaxInternLen = 4096;
constexpr int32_t  kErrMsgMax = 1024;

// A recorded tensor byte range (a weight). host_read_range must never return bytes
// overlapping one of these, and no rendered string's source range may overlap one.
struct Range { uint64_t off; uint64_t len; };

// Source range of a HOST-read string that was rendered into the Model (name/meta).
struct StrSrc { uint64_t off; uint64_t len; };

// Threaded to every parser import. Borrows the file; owns the Model being built.
struct ParseHostCtx {
  const MappedFile* file = nullptr;
  ir::Model* model = nullptr;
  ParseLimits limits;
  std::vector<Range> recorded;        // declared tensor ranges (weights)
  std::vector<StrSrc> rendered_srcs;  // source ranges of host-read rendered strings
  std::string error;
  bool aborted = false;
  uint32_t intern_calls = 0;

  // Is [off,off+len) entirely inside the up-front sniff window (head OR tail)?
  bool in_window(uint64_t off, uint64_t len) const {
    if (!file) return false;
    const uint64_t sz = file->size();
    if (len == 0 || off > sz || len > sz - off) return false;   // out of file
    const uint64_t end = off + len;
    if (end <= static_cast<uint64_t>(kSniffHead)) return true;   // head window
    const uint64_t tail_start =
        sz > static_cast<uint64_t>(kSniffTail) ? sz - static_cast<uint64_t>(kSniffTail) : 0;
    if (off >= tail_start) return true;                          // tail window
    return false;
  }
  // Does [off,off+len) overlap any recorded tensor range?
  bool overlaps_recorded(uint64_t off, uint64_t len) const {
    const uint64_t end = off + len;
    for (const Range& r : recorded) {
      const uint64_t rend = r.off + r.len;
      if (off < rend && r.off < end) return true;
    }
    return false;
  }
};

ParseHostCtx* pctx(IM3ImportContext c) { return static_cast<ParseHostCtx*>(c->userdata); }

// Read [off,len) from the file THROUGH a marked ByteReader iff it is inside the
// sniff window and does not overlap a recorded tensor range. Returns the bytes (as
// a string) on success; nullptr-equivalent (empty + ok=false) otherwise. This is
// the ONE host-side file reader; it marks each returned byte so payload_read_counter
// reflects only bounded structural reads (§0.1).
struct HostRead { bool ok = false; std::string bytes; };
HostRead host_windowed_read(ParseHostCtx* h, uint64_t off, uint64_t len) {
  HostRead out;
  if (!h || !h->file || !h->file->valid()) return out;
  if (len == 0 || len > static_cast<uint64_t>(kReadChunkCap)) return out;
  if (!h->in_window(off, len)) return out;
  if (h->overlaps_recorded(off, len)) return out;
  ByteReader br(h->file->data(), h->file->size());
  if (!br.seek(off)) return out;
  auto s = br.bytes(len);
  if (!s) return out;
  // Account the structural read (once per byte) — the witness is "== window", not 0.
  for (uint64_t i = 0; i < len; ++i) ByteReader::mark_payload_read();
  out.ok = true;
  out.bytes = std::move(*s);
  return out;
}

// --- READ imports -----------------------------------------------------------
// I64 host_file_len()
m3ApiRawFunction(host_file_len) {
  m3ApiReturnType(int64_t);
  ParseHostCtx* h = pctx(_ctx);
  int64_t n = (h && h->file) ? static_cast<int64_t>(h->file->size()) : 0;
  m3ApiReturn(n);
}
// i32 host_read_range(I64 off, i32 len, i32 dst) -> bytes copied, or -1
m3ApiRawFunction(host_read_range) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int64_t, off);
  m3ApiGetArg(int32_t, len);
  m3ApiGetArgMem(uint8_t*, dst);
  ParseHostCtx* h = pctx(_ctx);
  int32_t written = -1;
  if (h && off >= 0 && len > 0 && len <= kReadChunkCap) {
    HostRead r = host_windowed_read(h, static_cast<uint64_t>(off), static_cast<uint64_t>(len));
    if (r.ok) {
      m3ApiCheckMem(dst, static_cast<size_t>(len));
      std::memcpy(dst, r.bytes.data(), r.bytes.size());
      written = static_cast<int32_t>(r.bytes.size());
    }
  }
  m3ApiReturn(written);
}

// --- Model-mutating commands (host reads guest ranges; every count capped) ---
// i32 host_intern_range(I64 off, i32 len) -> StringId.id; interns bytes the HOST
// reads from the marked window; records the source range for parse-end re-validation.
m3ApiRawFunction(host_intern_range) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int64_t, off);
  m3ApiGetArg(int32_t, len);
  ParseHostCtx* h = pctx(_ctx);
  int32_t id = -1;
  if (h && h->model && off >= 0 && len > 0 && len <= kMaxInternLen &&
      h->intern_calls < h->limits.max_intern_calls) {
    HostRead r = host_windowed_read(h, static_cast<uint64_t>(off), static_cast<uint64_t>(len));
    if (r.ok) {
      ++h->intern_calls;
      StringId sid = h->model->intern(r.bytes);
      h->rendered_srcs.push_back({static_cast<uint64_t>(off), static_cast<uint64_t>(len)});
      id = static_cast<int32_t>(sid.id);
    }
  }
  m3ApiReturn(id);
}
// void host_set_model_info(i32 fmt_id, i32 prod_id, i32 ver_id)
m3ApiRawFunction(host_set_model_info) {
  m3ApiGetArg(int32_t, fmt_id);
  m3ApiGetArg(int32_t, prod_id);
  m3ApiGetArg(int32_t, ver_id);
  ParseHostCtx* h = pctx(_ctx);
  if (h && h->model) {
    if (fmt_id >= 0) h->model->format_name = StringId{static_cast<uint32_t>(fmt_id)};
    if (prod_id >= 0) h->model->producer = StringId{static_cast<uint32_t>(prod_id)};
    if (ver_id >= 0) h->model->version_info = StringId{static_cast<uint32_t>(ver_id)};
  }
  m3ApiSuccess();
}
// void host_set_metadata(i32 key_id, i32 val_id)
m3ApiRawFunction(host_set_metadata) {
  m3ApiGetArg(int32_t, key_id);
  m3ApiGetArg(int32_t, val_id);
  ParseHostCtx* h = pctx(_ctx);
  if (h && h->model && key_id >= 0 && val_id >= 0 &&
      h->model->metadata.size() < h->limits.max_metadata) {
    h->model->metadata.emplace_back(StringId{static_cast<uint32_t>(key_id)},
                                    StringId{static_cast<uint32_t>(val_id)});
  }
  m3ApiSuccess();
}
// void host_set_error(i32 msg_ptr, i32 len) -> capped, truncated (closes exfil)
m3ApiRawFunction(host_set_error) {
  m3ApiGetArgMem(const char*, msg);
  m3ApiGetArg(int32_t, len);
  ParseHostCtx* h = pctx(_ctx);
  if (h && msg && len > 0) {
    int32_t n = len > kErrMsgMax ? kErrMsgMax : len;
    m3ApiCheckMem(msg, static_cast<size_t>(n));
    h->error.assign(msg, static_cast<size_t>(n));
    h->aborted = true;
  }
  m3ApiSuccess();
}
// i32 host_begin_graph(i32 name_id) -> graph index, or -1
m3ApiRawFunction(host_begin_graph) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, name_id);
  ParseHostCtx* h = pctx(_ctx);
  int32_t gi = -1;
  if (h && h->model && h->model->graphs.size() < h->limits.max_graphs) {
    gi = static_cast<int32_t>(h->model->graphs.size());
    ir::Graph g;
    if (name_id >= 0) g.name = StringId{static_cast<uint32_t>(name_id)};
    h->model->graphs.push_back(std::move(g));
  }
  m3ApiReturn(gi);
}
// Read up to `count` i64 dims from guest [ptr,ptr+count*8); returns rank written to
// `out` (clamped to kMaxRank). Validates the CLAMPED byte span. -1 => reject.
static int32_t read_dims(int64_t* ptr, int32_t rank, SmallVec<int64_t, 6>* out,
                         IM3Runtime runtime, void* _mem) {
  if (rank < 0 || rank > kMaxRank) return -1;
  if (rank == 0) return 0;   // scalar: no pointer check
  if (m3ApiIsNullPtr(ptr) ||
      ((uint64_t)(uintptr_t)(ptr) + static_cast<uint64_t>(rank) * 8u) >
          ((uint64_t)(uintptr_t)(_mem) + m3_GetMemorySize(runtime)))
    return -1;
  for (int32_t i = 0; i < rank; ++i) {
    int64_t d = ptr[i];
    out->push_back((d < 1 && d != -1) ? -1 : d);
  }
  return rank;
}
// i32 host_add_value(i32 g, i32 name_id, i32 dtype, i32 dims_ptr, i32 rank)
m3ApiRawFunction(host_add_value) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, g);
  m3ApiGetArg(int32_t, name_id);
  m3ApiGetArg(int32_t, dtype);
  m3ApiGetArgMem(int64_t*, dims);
  m3ApiGetArg(int32_t, rank);
  ParseHostCtx* h = pctx(_ctx);
  int32_t vi = -1;
  if (h && h->model && g >= 0 && static_cast<size_t>(g) < h->model->graphs.size()) {
    ir::Graph& gr = h->model->graphs[static_cast<size_t>(g)];
    if (gr.values.size() < h->limits.max_values) {
      ir::ValueInfo v;
      if (name_id >= 0) v.name = StringId{static_cast<uint32_t>(name_id)};
      if (dtype >= 0 && dtype <= static_cast<int32_t>(ir::DType::Unknown))
        v.dtype = static_cast<ir::DType>(dtype);
      if (read_dims(dims, rank, &v.shape, runtime, _mem) < 0) { m3ApiReturn(vi); }  // -1 reject
      vi = static_cast<int32_t>(gr.values.size());
      gr.values.push_back(std::move(v));
    }
  }
  m3ApiReturn(vi);
}
// i32 host_add_node(i32 g, i32 op_id, i32 name_id, i32 in_ptr, i32 nin, i32 out_ptr, i32 nout)
m3ApiRawFunction(host_add_node) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, g);
  m3ApiGetArg(int32_t, op_id);
  m3ApiGetArg(int32_t, name_id);
  m3ApiGetArgMem(uint32_t*, in_ptr);
  m3ApiGetArg(int32_t, nin);
  m3ApiGetArgMem(uint32_t*, out_ptr);
  m3ApiGetArg(int32_t, nout);
  ParseHostCtx* h = pctx(_ctx);
  int32_t ni = -1;
  if (h && h->model && g >= 0 && static_cast<size_t>(g) < h->model->graphs.size() &&
      nin >= 0 && nin <= kMaxNodeIO && nout >= 0 && nout <= kMaxNodeIO) {
    ir::Graph& gr = h->model->graphs[static_cast<size_t>(g)];
    if (gr.nodes.size() >= h->limits.max_nodes) { m3ApiReturn(ni); }
    // Validate the (clamped) input/output index arrays before touching them.
    if (nin > 0) {
      if (m3ApiIsNullPtr(in_ptr) ||
          ((uint64_t)(uintptr_t)(in_ptr) + static_cast<uint64_t>(nin) * 4u) >
              ((uint64_t)(uintptr_t)(_mem) + m3_GetMemorySize(runtime))) { m3ApiReturn(ni); }
    }
    if (nout > 0) {
      if (m3ApiIsNullPtr(out_ptr) ||
          ((uint64_t)(uintptr_t)(out_ptr) + static_cast<uint64_t>(nout) * 4u) >
              ((uint64_t)(uintptr_t)(_mem) + m3_GetMemorySize(runtime))) { m3ApiReturn(ni); }
    }
    ni = static_cast<int32_t>(gr.nodes.size());
    ir::Node node;
    if (op_id >= 0) node.op_type = StringId{static_cast<uint32_t>(op_id)};
    if (name_id >= 0) node.name = StringId{static_cast<uint32_t>(name_id)};
    node.inputs.begin = static_cast<uint32_t>(gr.edge_refs.size());
    for (int32_t i = 0; i < nin; ++i) {
      uint32_t v = in_ptr[i];
      if (v < gr.values.size()) gr.edge_refs.push_back(v);   // drop out-of-range refs
    }
    node.inputs.count = static_cast<uint32_t>(gr.edge_refs.size()) - node.inputs.begin;
    node.outputs.begin = static_cast<uint32_t>(gr.edge_refs.size());
    for (int32_t i = 0; i < nout; ++i) {
      uint32_t v = out_ptr[i];
      if (v < gr.values.size()) { gr.edge_refs.push_back(v); gr.values[v].producer = ni; }
    }
    node.outputs.count = static_cast<uint32_t>(gr.edge_refs.size()) - node.outputs.begin;
    gr.nodes.push_back(std::move(node));
  }
  m3ApiReturn(ni);
}
// void host_set_graph_io(i32 g, i32 in_ptr, i32 nin, i32 out_ptr, i32 nout)
m3ApiRawFunction(host_set_graph_io) {
  m3ApiGetArg(int32_t, g);
  m3ApiGetArgMem(uint32_t*, in_ptr);
  m3ApiGetArg(int32_t, nin);
  m3ApiGetArgMem(uint32_t*, out_ptr);
  m3ApiGetArg(int32_t, nout);
  ParseHostCtx* h = pctx(_ctx);
  if (h && h->model && g >= 0 && static_cast<size_t>(g) < h->model->graphs.size() &&
      nin >= 0 && nin <= kMaxNodeIO && nout >= 0 && nout <= kMaxNodeIO) {
    ir::Graph& gr = h->model->graphs[static_cast<size_t>(g)];
    bool ok = true;
    if (nin > 0 && (m3ApiIsNullPtr(in_ptr) ||
        ((uint64_t)(uintptr_t)(in_ptr) + static_cast<uint64_t>(nin) * 4u) >
            ((uint64_t)(uintptr_t)(_mem) + m3_GetMemorySize(runtime)))) ok = false;
    if (nout > 0 && (m3ApiIsNullPtr(out_ptr) ||
        ((uint64_t)(uintptr_t)(out_ptr) + static_cast<uint64_t>(nout) * 4u) >
            ((uint64_t)(uintptr_t)(_mem) + m3_GetMemorySize(runtime)))) ok = false;
    if (ok) {
      for (int32_t i = 0; i < nin; ++i) { uint32_t v = in_ptr[i]; if (v < gr.values.size()) gr.graph_inputs.push_back(v); }
      for (int32_t i = 0; i < nout; ++i) { uint32_t v = out_ptr[i]; if (v < gr.values.size()) gr.graph_outputs.push_back(v); }
    }
  }
  m3ApiSuccess();
}
// Attribute helpers append to a node's attribute range. They require the node to be
// the LAST node added to graph g (append-only build), keeping the Range contiguous.
static ir::Node* last_node(ParseHostCtx* h, int32_t g, int32_t node) {
  if (!h || !h->model || g < 0 || static_cast<size_t>(g) >= h->model->graphs.size()) return nullptr;
  ir::Graph& gr = h->model->graphs[static_cast<size_t>(g)];
  if (node < 0 || static_cast<size_t>(node) != gr.nodes.size() - 1 || gr.nodes.empty()) return nullptr;
  return &gr.nodes.back();
}
// i32 host_add_attr_int(i32 g, i32 node, i32 name_id, I64 v)
m3ApiRawFunction(host_add_attr_int) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, g);
  m3ApiGetArg(int32_t, node);
  m3ApiGetArg(int32_t, name_id);
  m3ApiGetArg(int64_t, val);
  ParseHostCtx* h = pctx(_ctx);
  int32_t ok = -1;
  if (h && name_id >= 0) {
    if (ir::Node* n = last_node(h, g, node)) {
      ir::Graph& gr = h->model->graphs[static_cast<size_t>(g)];
      if (n->attributes.count == 0) n->attributes.begin = static_cast<uint32_t>(gr.attributes.size());
      ir::Attribute a; a.name = StringId{static_cast<uint32_t>(name_id)};
      a.value.kind = ir::AttrValue::Kind::Int; a.value.i = val;
      gr.attributes.push_back(std::move(a));
      n->attributes.count++;
      ok = 0;
    }
  }
  m3ApiReturn(ok);
}
// i32 host_add_attr_float(i32 g, i32 node, i32 name_id, F64 v)
m3ApiRawFunction(host_add_attr_float) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, g);
  m3ApiGetArg(int32_t, node);
  m3ApiGetArg(int32_t, name_id);
  m3ApiGetArg(double, val);
  ParseHostCtx* h = pctx(_ctx);
  int32_t ok = -1;
  if (h && name_id >= 0) {
    if (ir::Node* n = last_node(h, g, node)) {
      ir::Graph& gr = h->model->graphs[static_cast<size_t>(g)];
      if (n->attributes.count == 0) n->attributes.begin = static_cast<uint32_t>(gr.attributes.size());
      ir::Attribute a; a.name = StringId{static_cast<uint32_t>(name_id)};
      a.value.kind = ir::AttrValue::Kind::Float; a.value.f = val;
      gr.attributes.push_back(std::move(a));
      n->attributes.count++;
      ok = 0;
    }
  }
  m3ApiReturn(ok);
}
// i32 host_add_attr_string(i32 g, i32 node, i32 name_id, i32 val_id)
m3ApiRawFunction(host_add_attr_string) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, g);
  m3ApiGetArg(int32_t, node);
  m3ApiGetArg(int32_t, name_id);
  m3ApiGetArg(int32_t, val_id);
  ParseHostCtx* h = pctx(_ctx);
  int32_t ok = -1;
  if (h && name_id >= 0 && val_id >= 0) {
    if (ir::Node* n = last_node(h, g, node)) {
      ir::Graph& gr = h->model->graphs[static_cast<size_t>(g)];
      if (n->attributes.count == 0) n->attributes.begin = static_cast<uint32_t>(gr.attributes.size());
      ir::Attribute a; a.name = StringId{static_cast<uint32_t>(name_id)};
      a.value.kind = ir::AttrValue::Kind::String; a.value.s = StringId{static_cast<uint32_t>(val_id)};
      gr.attributes.push_back(std::move(a));
      n->attributes.count++;
      ok = 0;
    }
  }
  m3ApiReturn(ok);
}
// i32 host_add_attr_ints(i32 g, i32 node, i32 name_id, i32 ptr, i32 count)
m3ApiRawFunction(host_add_attr_ints) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, g);
  m3ApiGetArg(int32_t, node);
  m3ApiGetArg(int32_t, name_id);
  m3ApiGetArgMem(int64_t*, ptr);
  m3ApiGetArg(int32_t, count);
  ParseHostCtx* h = pctx(_ctx);
  int32_t ok = -1;
  if (h && name_id >= 0 && count >= 0 && count <= kMaxAttrLen) {
    if (ir::Node* n = last_node(h, g, node)) {
      bool mem_ok = true;
      if (count > 0 && (m3ApiIsNullPtr(ptr) ||
          ((uint64_t)(uintptr_t)(ptr) + static_cast<uint64_t>(count) * 8u) >
              ((uint64_t)(uintptr_t)(_mem) + m3_GetMemorySize(runtime)))) mem_ok = false;
      if (mem_ok) {
        ir::Graph& gr = h->model->graphs[static_cast<size_t>(g)];
        if (n->attributes.count == 0) n->attributes.begin = static_cast<uint32_t>(gr.attributes.size());
        ir::Attribute a; a.name = StringId{static_cast<uint32_t>(name_id)};
        a.value.kind = ir::AttrValue::Kind::Ints;
        a.value.ints.reserve(static_cast<size_t>(count));
        for (int32_t i = 0; i < count; ++i) a.value.ints.push_back(ptr[i]);
        gr.attributes.push_back(std::move(a));
        n->attributes.count++;
        ok = 0;
      }
    }
  }
  m3ApiReturn(ok);
}
// i32 host_record_tensor(i32 g, i32 name_id, I64 off, I64 len, i32 dtype, i32 dims_ptr, i32 rank)
m3ApiRawFunction(host_record_tensor) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, g);
  m3ApiGetArg(int32_t, name_id);
  m3ApiGetArg(int64_t, off);
  m3ApiGetArg(int64_t, len);
  m3ApiGetArg(int32_t, dtype);
  m3ApiGetArgMem(int64_t*, dims);
  m3ApiGetArg(int32_t, rank);
  ParseHostCtx* h = pctx(_ctx);
  int32_t ok = -1;
  if (h && h->model && off >= 0 && len >= 0) {
    // Range must lie within the file (offset+len bounded); zero bytes are read.
    const uint64_t uoff = static_cast<uint64_t>(off), ulen = static_cast<uint64_t>(len);
    const uint64_t fsz = h->file ? h->file->size() : 0;
    if (uoff <= fsz && ulen <= fsz - uoff) {
      ir::TensorRef tr;
      if (name_id >= 0) tr.name = StringId{static_cast<uint32_t>(name_id)};
      if (dtype >= 0 && dtype <= static_cast<int32_t>(ir::DType::Unknown))
        tr.dtype = static_cast<ir::DType>(dtype);
      tr.file_offset = uoff;
      tr.byte_len = ulen;
      read_dims(dims, rank, &tr.shape, runtime, _mem);  // best-effort; bad -> empty shape
      h->recorded.push_back({uoff, ulen});
      if (g >= 0 && static_cast<size_t>(g) < h->model->graphs.size()) {
        ir::Graph& gr = h->model->graphs[static_cast<size_t>(g)];
        if (gr.initializers.size() < h->limits.max_tensors) { gr.initializers.push_back(std::move(tr)); ok = 0; }
      } else {
        if (h->model->flat_tensors.size() < h->limits.max_tensors) { h->model->flat_tensors.push_back(std::move(tr)); ok = 0; }
      }
    }
  }
  m3ApiReturn(ok);
}

void link_parser_capabilities(IM3Module mod, ParseHostCtx* ctx) {
  const char* ns = "netvis";
  auto L = [&](const char* nm, const char* sig, M3RawCall fn) {
    m3_LinkRawFunctionEx(mod, ns, nm, sig, fn, ctx);
  };
  L("host_file_len", "I()", &host_file_len);
  L("host_read_range", "i(Ii*)", &host_read_range);
  L("host_intern_range", "i(Ii)", &host_intern_range);
  L("host_set_model_info", "v(iii)", &host_set_model_info);
  L("host_set_metadata", "v(ii)", &host_set_metadata);
  L("host_set_error", "v(*i)", &host_set_error);
  L("host_begin_graph", "i(i)", &host_begin_graph);
  L("host_add_value", "i(iii*i)", &host_add_value);
  L("host_add_node", "i(iii*i*i)", &host_add_node);
  L("host_set_graph_io", "v(i*i*i)", &host_set_graph_io);
  L("host_add_attr_int", "i(iiiI)", &host_add_attr_int);
  L("host_add_attr_float", "i(iiiF)", &host_add_attr_float);
  L("host_add_attr_string", "i(iiii)", &host_add_attr_string);
  L("host_add_attr_ints", "i(iii*i)", &host_add_attr_ints);
  L("host_record_tensor", "i(iiIIi*i)", &host_record_tensor);
}

// Parse-end re-validation: no rendered string's source range may overlap a recorded
// tensor range (§0.1). A violation means the plugin tried to smuggle payload bytes
// out via an interned name -> reject the whole Model.
bool rendered_strings_clean(const ParseHostCtx& h) {
  for (const StrSrc& s : h.rendered_srcs) {
    if (h.overlaps_recorded(s.off, s.len)) return false;
  }
  return true;
}

// The WasmParserPlugin adapter.
class WasmParserPlugin final : public ParserPlugin {
 public:
  WasmParserPlugin(std::string name, std::shared_ptr<const std::vector<uint8_t>> image)
      : name_(std::move(name)), image_(std::move(image)) {}

  Format format() const override { return Format::Unknown; }
  std::string_view display_name() const override { return name_; }
  int priority() const override { return 10'000; }  // below all built-ins
  uint32_t api_version() const override { return kParserPluginAbiVersion; }

  bool can_parse(const MappedFile& file, const std::string& ext_hint) const override {
    (void)ext_hint;
    if (!image_) return false;
    WasmEngine& eng = WasmEngine::instance();
    if (!eng.enabled()) return false;
    std::lock_guard<std::mutex> guard(eng.lock());
    ParseHostCtx hc;
    hc.file = &file;
    // No model built during sniff; a tiny sandbox. netvis_can_parse -> i32 (1=yes).
    SandboxLimits lim{4, 500'000};
    RunResult lerr;
    WasmModule mod = eng.load(*image_, lim, &hc, &lerr);
    if (!mod.loaded()) return false;
    link_parser_capabilities(static_cast<IM3Module>(mod.raw_module()), &hc);
    if (!abi_ok(mod)) return false;
    int32_t yes = 0;
    RunResult rr = mod.call_i32("netvis_can_parse", &yes);
    return rr.status == RunStatus::Ok && yes != 0;
  }

  Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) const override {
    (void)progress;
    if (!image_) return err("wasm parser: no image", 0);
    WasmEngine& eng = WasmEngine::instance();
    if (!eng.enabled()) return err("wasm disabled", 0);

    std::lock_guard<std::mutex> guard(eng.lock());
    ir::Model model;
    ParseHostCtx hc;
    hc.file = &file;
    hc.model = &model;

    SandboxLimits lim{256, 200'000'000};   // parser gets more fuel than a pass
    RunResult lerr;
    WasmModule mod = eng.load(*image_, lim, &hc, &lerr);
    if (!mod.loaded()) return err("wasm parser load failed: " + lerr.message, 0);
    link_parser_capabilities(static_cast<IM3Module>(mod.raw_module()), &hc);
    if (!abi_ok(mod)) return err("wasm parser abi mismatch", 0);

    int32_t ret = 0;
    RunResult rr = mod.call_i32("netvis_parse", &ret);
    if (rr.status != RunStatus::Ok)
      return err("wasm parser trapped: " + rr.message, 0);  // partial Model discarded
    if (ret != 0 || hc.aborted)
      return err(hc.error.empty() ? "wasm parser reported failure" : hc.error, 0);
    if (!rendered_strings_clean(hc))
      return err("wasm parser: a rendered string overlaps a recorded tensor range", 0);
    if (model.graphs.empty() && model.flat_tensors.empty())
      return err("wasm parser produced an empty model", 0);

    model.has_graph = !model.graphs.empty();
    if (!model.format_name.valid()) model.format_name = model.intern("WASM");
    return model;
  }

 private:
  static bool abi_ok(WasmModule& mod) {
    int32_t v = -1;
    RunResult rr = mod.call_i32("netvis_parser_abi_version", &v);
    return rr.status == RunStatus::Ok && v == static_cast<int32_t>(kParserPluginAbiVersion);
  }
  std::string name_;
  std::shared_ptr<const std::vector<uint8_t>> image_;
};

}  // namespace

std::unique_ptr<ParserPlugin> make_wasm_parser(
    std::string plugin_name, std::shared_ptr<const std::vector<uint8_t>> image) {
  return std::make_unique<WasmParserPlugin>(std::move(plugin_name), std::move(image));
}

std::string load_wasm_parser_plugin(const std::string& plugin_json_path) {
  std::ifstream f(plugin_json_path);
  if (!f) return "cannot open " + plugin_json_path;
  std::string dir;
  {
    auto slash = plugin_json_path.find_last_of("/\\");
    dir = (slash == std::string::npos) ? std::string() : plugin_json_path.substr(0, slash + 1);
  }
  std::string wasm_rel, plugin_name;
  try {
    nlohmann::json j; f >> j;
    if (!j.is_object()) return "manifest is not an object";
    if (auto v = j.find("api_version"); v != j.end() && v->is_number_integer()) {
      if (v->get<int>() != static_cast<int>(kParserPluginAbiVersion)) return "api_version mismatch";
    }
    if (auto n = j.find("name"); n != j.end() && n->is_string()) plugin_name = n->get<std::string>();
    if (auto w = j.find("parser_wasm"); w != j.end() && w->is_string()) wasm_rel = w->get<std::string>();
    if (wasm_rel.empty()) return "manifest missing \"parser_wasm\"";
  } catch (...) {
    return "manifest parse error";
  }
  std::ifstream wf(dir + wasm_rel, std::ios::binary);
  if (!wf) return "cannot open wasm: " + wasm_rel;
  auto image = std::make_shared<std::vector<uint8_t>>(
      (std::istreambuf_iterator<char>(wf)), std::istreambuf_iterator<char>());
  if (image->size() < 8) return "wasm image too small";
  Registry::instance().register_parser(make_wasm_parser(plugin_name, image));
  return {};
}

}  // namespace netvis::plugin::wasm

#else  // !NETVIS_ENABLE_WASM — safe stubs

#include <memory>

namespace netvis::plugin::wasm {
std::unique_ptr<ParserPlugin> make_wasm_parser(std::string, std::shared_ptr<const std::vector<uint8_t>>) {
  return nullptr;
}
std::string load_wasm_parser_plugin(const std::string&) { return "WASM disabled"; }
}  // namespace netvis::plugin::wasm

#endif
