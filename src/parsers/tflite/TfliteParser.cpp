// parsers/tflite/TfliteParser.cpp — TFLite flatbuffer parser.
//
// TFLite schema v3 (TFL3). We hand-roll flatbuffer traversal (NO flatbuffers
// dependency): the on-disk layout is fully documented by the schema, and a
// bounds-checked reader is all we need. Weights (Buffer.data) are NEVER read —
// we record only file_offset + byte_len and move on (spec §2.1, §4).
//
// Flatbuffer layout recap:
//   * Root: u32 UOFFSET at file start -> root table position.
//   * Table: begins with s32 SOFFSET to its vtable (vtable_pos = table_pos -
//     soffset). vtable = u16 vtable_size, u16 table_size, then u16 field offsets
//     (0 == absent, else field lives at table_pos + off).
//   * Vector: u32 length prefix followed by elements.
//   * String: u32 length prefix followed by bytes.
//   * Sub-object reference: u32 UOFFSET relative to the field's own location.
#include "parsers/Parser.h"

#include <cstdint>
#include <string>
#include <vector>

#include "core/ByteReader.h"

namespace netvis::tflite {
namespace {

// A flatbuffer table view: absolute file position of the table and a cached
// vtable position. All accessors go through a shared ByteReader (bounds-checked)
// so any malformed offset yields a Result error with a byte offset, never a
// crash / OOB read.
struct Table {
  uint64_t pos = 0;         // absolute position of the table
  uint64_t vtable_pos = 0;  // absolute position of the vtable
  uint16_t vtable_size = 0; // bytes in the vtable
};

// Read a u32 UOFFSET at absolute `at`, returning the absolute position it points
// to (UOFFSET is relative to its own storage location).
Result<uint64_t> read_uoffset(ByteReader& r, uint64_t at) {
  if (auto s = r.seek(at); !s) return s.error();
  auto off = r.u32le();
  if (!off) return off.error();
  if (*off == 0) return err("null uoffset", at);
  return at + *off;
}

// Open the table located at absolute `table_pos`, resolving its vtable.
Result<Table> open_table(ByteReader& r, uint64_t table_pos) {
  if (auto s = r.seek(table_pos); !s) return s.error();
  auto soffset = r.i32le();  // signed: vtable_pos = table_pos - soffset
  if (!soffset) return soffset.error();
  int64_t vpos = static_cast<int64_t>(table_pos) - static_cast<int64_t>(*soffset);
  if (vpos < 0 || static_cast<uint64_t>(vpos) + 4 > r.size()) {
    return err("vtable position out of range", table_pos);
  }
  Table t;
  t.pos = table_pos;
  t.vtable_pos = static_cast<uint64_t>(vpos);
  if (auto s = r.seek(t.vtable_pos); !s) return s.error();
  auto vsize = r.u16le();
  if (!vsize) return vsize.error();
  t.vtable_size = *vsize;
  if (t.vtable_pos + t.vtable_size > r.size()) {
    return err("vtable overruns file", t.vtable_pos);
  }
  return t;
}

// Return the absolute position of field `idx` within a table, or 0 if the field
// is absent. Field slots begin at vtable_pos + 4 (after vtable_size,table_size).
Result<uint64_t> table_field(ByteReader& r, const Table& t, uint32_t idx) {
  uint64_t slot = t.vtable_pos + 4 + static_cast<uint64_t>(idx) * 2;
  // Absent field: slot beyond the vtable -> 0.
  if (slot + 2 > t.vtable_pos + t.vtable_size) return uint64_t{0};
  if (auto s = r.seek(slot); !s) return s.error();
  auto off = r.u16le();
  if (!off) return off.error();
  if (*off == 0) return uint64_t{0};  // explicitly absent
  return t.pos + *off;
}

// Scalar field readers (default when absent). Each seeks to the field location.
Result<uint32_t> field_u32(ByteReader& r, const Table& t, uint32_t idx,
                           uint32_t dflt) {
  auto loc = table_field(r, t, idx);
  if (!loc) return loc.error();
  if (*loc == 0) return dflt;
  if (auto s = r.seek(*loc); !s) return s.error();
  return r.u32le();
}
Result<int32_t> field_i32(ByteReader& r, const Table& t, uint32_t idx,
                          int32_t dflt) {
  auto loc = table_field(r, t, idx);
  if (!loc) return loc.error();
  if (*loc == 0) return dflt;
  if (auto s = r.seek(*loc); !s) return s.error();
  return r.i32le();
}
Result<uint8_t> field_u8(ByteReader& r, const Table& t, uint32_t idx,
                         uint8_t dflt) {
  auto loc = table_field(r, t, idx);
  if (!loc) return loc.error();
  if (*loc == 0) return dflt;
  if (auto s = r.seek(*loc); !s) return s.error();
  return r.u8();
}

// Read a string field into `out`. Empty if absent.
Result<bool> field_string(ByteReader& r, const Table& t, uint32_t idx,
                          std::string& out) {
  auto loc = table_field(r, t, idx);
  if (!loc) return loc.error();
  if (*loc == 0) { out.clear(); return true; }
  auto strpos = read_uoffset(r, *loc);
  if (!strpos) return strpos.error();
  if (auto s = r.seek(*strpos); !s) return s.error();
  auto len = r.u32le();
  if (!len) return len.error();
  auto bytes = r.bytes(*len);
  if (!bytes) return bytes.error();
  out = std::move(*bytes);
  return true;
}

// Vector info: absolute position of first element + element count. If the field
// is absent, count == 0 and start == 0.
struct VecInfo {
  uint64_t start = 0;
  uint32_t count = 0;
};

Result<VecInfo> field_vector(ByteReader& r, const Table& t, uint32_t idx) {
  VecInfo v;
  auto loc = table_field(r, t, idx);
  if (!loc) return loc.error();
  if (*loc == 0) return v;
  auto vecpos = read_uoffset(r, *loc);
  if (!vecpos) return vecpos.error();
  if (auto s = r.seek(*vecpos); !s) return s.error();
  auto len = r.u32le();
  if (!len) return len.error();
  v.count = *len;
  v.start = *vecpos + 4;  // elements follow the u32 length prefix
  // SECURITY: bound the count by the file. The smallest possible element is 1
  // byte, so a genuine vector needs at least `count` bytes after v.start;
  // anything larger is corrupt. This rejects a 0xFFFFFFFF length before any
  // caller can reserve() gigabytes. (Callers reading 4-byte elements apply the
  // exact per-element bound via elem_bytes below.)
  if (v.start > r.size() || static_cast<uint64_t>(v.count) > r.size() - v.start)
    return err("tflite vector length exceeds file", v.start);
  return v;
}

// Read a vector of int32 (used for shapes, inputs, outputs).
Result<std::vector<int32_t>> read_i32_vector(ByteReader& r, const Table& t,
                                             uint32_t idx) {
  std::vector<int32_t> out;
  auto v = field_vector(r, t, idx);
  if (!v) return v.error();
  out.reserve(v->count);
  for (uint32_t i = 0; i < v->count; ++i) {
    if (auto s = r.seek(v->start + static_cast<uint64_t>(i) * 4); !s) {
      return s.error();
    }
    auto e = r.i32le();
    if (!e) return e.error();
    out.push_back(*e);
  }
  return out;
}

// Read a vector of table offsets (elements are u32 UOFFSETs). Returns absolute
// positions of each sub-table.
Result<std::vector<uint64_t>> read_offset_vector(ByteReader& r, const Table& t,
                                                 uint32_t idx) {
  std::vector<uint64_t> out;
  auto v = field_vector(r, t, idx);
  if (!v) return v.error();
  out.reserve(v->count);
  for (uint32_t i = 0; i < v->count; ++i) {
    uint64_t elem_at = v->start + static_cast<uint64_t>(i) * 4;
    auto sub = read_uoffset(r, elem_at);
    if (!sub) return sub.error();
    out.push_back(*sub);
  }
  return out;
}

// Open a sub-table referenced by table field `idx` (a UOFFSET). Returns nullopt
// (via *present=false) when the field is absent; a real error otherwise.
Result<bool> field_table(ByteReader& r, const Table& t, uint32_t idx,
                         Table& out, bool& present) {
  present = false;
  auto loc = table_field(r, t, idx);
  if (!loc) return loc.error();
  if (*loc == 0) return true;  // absent
  auto tpos = read_uoffset(r, *loc);
  if (!tpos) return tpos.error();
  auto tab = open_table(r, *tpos);
  if (!tab) return tab.error();
  out = *tab;
  present = true;
  return true;
}

// Map TFLite tensor type code -> ir::DType.
ir::DType map_dtype(uint8_t code) {
  switch (code) {
    case 0:  return ir::DType::F32;
    case 1:  return ir::DType::F16;
    case 2:  return ir::DType::I32;
    case 3:  return ir::DType::U8;
    case 4:  return ir::DType::I64;
    // 5 == STRING -> no numeric dtype.
    case 6:  return ir::DType::Bool;
    case 7:  return ir::DType::I16;
    case 9:  return ir::DType::I8;
    default: return ir::DType::Unknown;
  }
}

// Builtin operator code -> op name. Only the common subset is spelled out;
// anything else becomes BUILTIN_<n>.
std::string builtin_name(int32_t code) {
  switch (code) {
    case 0:  return "ADD";
    case 1:  return "AVERAGE_POOL_2D";
    case 2:  return "CONCATENATION";
    case 3:  return "CONV_2D";
    case 4:  return "DEPTHWISE_CONV_2D";
    case 9:  return "FULLY_CONNECTED";
    case 14: return "LOGISTIC";
    case 17: return "MAX_POOL_2D";
    case 18: return "MUL";
    case 19: return "RELU";
    case 22: return "RESHAPE";
    case 25: return "SOFTMAX";
    case 28: return "TANH";
    case 34: return "PAD";
    case 39: return "TRANSPOSE";
    case 40: return "MEAN";
    default: return "BUILTIN_" + std::to_string(code);
  }
}

// Field indices per the frozen schema.
namespace f_model { enum { version = 0, operator_codes = 1, subgraphs = 2,
                           description = 3, buffers = 4 }; }
namespace f_opcode { enum { deprecated_builtin_code = 0, custom_code = 1,
                            builtin_code = 3 }; }
namespace f_subgraph { enum { tensors = 0, inputs = 1, outputs = 2,
                              operators = 3, name = 4 }; }
namespace f_tensor { enum { shape = 0, type = 1, buffer = 2, name = 3 }; }
namespace f_operator { enum { opcode_index = 0, inputs = 1, outputs = 2,
                              builtin_options_type = 3, builtin_options = 4 }; }
namespace f_buffer { enum { data = 0 }; }

// BuiltinOptions union tag values (schema.fbs BuiltinOptions enum ordinal, NOT
// the operator BuiltinOperator code). Only the control-flow ops that reference
// other subgraphs matter here. These are the ordinals in the canonical TFLite
// BuiltinOptions union: IfOptions=92, WhileOptions=93, CallOnceOptions=103.
// (The earlier 45/62/76 were GreaterEqual/LogicalAnd/SquaredDifference options —
// wrong union, so control-flow linking never fired on a real .tflite file.)
namespace builtin_options_type {
enum {
  IfOptions = 92,
  WhileOptions = 93,
  CallOnceOptions = 103,
};
}
// Field indices inside the control-flow option tables (all i32 subgraph refs).
namespace f_if { enum { then_subgraph_index = 0, else_subgraph_index = 1 }; }
namespace f_while { enum { cond_subgraph_index = 0, body_subgraph_index = 1 }; }
namespace f_call_once { enum { init_subgraph_index = 0 }; }

// Resolved buffer: file offset + length of its `data` vector. offset ==
// UINT64_MAX and len == 0 when the buffer is empty (no weights).
struct BufferRef {
  uint64_t offset = UINT64_MAX;
  uint64_t len = 0;
};

}  // namespace

// Parse a TFLite model file into ir::Model. Structural reads only; tensor
// payloads (Buffer.data) are recorded as offset+len and never dereferenced.
Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "TFLite: opening");
  ByteReader r(file.data(), file.size());

  // Root table offset is a u32 UOFFSET at file start.
  auto root_pos = read_uoffset(r, 0);
  if (!root_pos) return root_pos.error();
  auto model = open_table(r, *root_pos);
  if (!model) return model.error();

  ir::Model out;
  out.has_graph = true;
  out.format_name = out.intern("TFLite");

  // Version -> version_info metadata.
  {
    auto ver = field_u32(r, *model, f_model::version, 0);
    if (!ver) return ver.error();
    out.version_info = out.intern("schema v" + std::to_string(*ver));
  }
  {
    std::string desc;
    if (auto s = field_string(r, *model, f_model::description, desc); !s) {
      return s.error();
    }
    if (!desc.empty()) out.producer = out.intern(desc);
  }

  progress.set(0.15f, "TFLite: operator codes");

  // --- Operator codes -------------------------------------------------------
  // Precompute the op name for each operator_code entry.
  std::vector<std::string> op_names;
  {
    auto codes = read_offset_vector(r, *model, f_model::operator_codes);
    if (!codes) return codes.error();
    op_names.reserve(codes->size());
    for (uint64_t oc_pos : *codes) {
      auto oc = open_table(r, oc_pos);
      if (!oc) return oc.error();
      // Custom op: non-empty custom_code wins.
      std::string custom;
      if (auto s = field_string(r, *oc, f_opcode::custom_code, custom); !s) {
        return s.error();
      }
      if (!custom.empty()) {
        op_names.push_back(std::move(custom));
        continue;
      }
      // builtin_code (int32, field 3) is authoritative; fall back to the
      // deprecated 8-bit builtin code (field 0) for old files.
      auto bc = field_i32(r, *oc, f_opcode::builtin_code, 0);
      if (!bc) return bc.error();
      int32_t code = *bc;
      if (code == 0) {
        auto dep = field_u8(r, *oc, f_opcode::deprecated_builtin_code, 0);
        if (!dep) return dep.error();
        if (*dep != 0) code = *dep;
      }
      op_names.push_back(builtin_name(code));
    }
  }

  progress.set(0.3f, "TFLite: buffers");

  // --- Buffers --------------------------------------------------------------
  // Record only the file offset + byte length of each buffer's `data` vector.
  // We NEVER read the payload bytes themselves (spec §2.1) — the weight
  // inspector does that lazily via mmap.
  std::vector<BufferRef> buffers;
  {
    auto bufs = read_offset_vector(r, *model, f_model::buffers);
    if (!bufs) return bufs.error();
    buffers.reserve(bufs->size());
    for (uint64_t b_pos : *bufs) {
      auto bt = open_table(r, b_pos);
      if (!bt) return bt.error();
      BufferRef ref;
      auto v = field_vector(r, *bt, f_buffer::data);
      if (!v) return v.error();
      if (v->count > 0) {
        // v->start is the absolute file offset of the first data byte; the
        // element is ubyte, so byte_len == count. Validate the range lies in
        // the file without touching it.
        if (v->start + v->count > file.size()) {
          return err("buffer data overruns file", v->start);
        }
        ref.offset = v->start;
        ref.len = v->count;
      }
      buffers.push_back(ref);
    }
  }

  progress.set(0.45f, "TFLite: subgraphs");

  // --- Subgraphs -> Graphs --------------------------------------------------
  auto subgraphs = read_offset_vector(r, *model, f_model::subgraphs);
  if (!subgraphs) return subgraphs.error();

  const uint64_t sg_total = subgraphs->size();
  for (uint64_t sg_i = 0; sg_i < sg_total; ++sg_i) {
    auto sg = open_table(r, (*subgraphs)[sg_i]);
    if (!sg) return sg.error();

    ir::Graph g;
    std::string sg_name;
    if (auto s = field_string(r, *sg, f_subgraph::name, sg_name); !s) {
      return s.error();
    }
    g.name = out.intern(sg_name.empty()
                            ? ("subgraph_" + std::to_string(sg_i))
                            : sg_name);

    // Tensors -> ValueInfo (one per tensor). A tensor whose buffer carries data
    // also becomes an initializer TensorRef.
    auto tensors = read_offset_vector(r, *sg, f_subgraph::tensors);
    if (!tensors) return tensors.error();

    g.values.reserve(tensors->size());
    for (uint64_t ti = 0; ti < tensors->size(); ++ti) {
      auto tt = open_table(r, (*tensors)[ti]);
      if (!tt) return tt.error();

      auto shape = read_i32_vector(r, *tt, f_tensor::shape);
      if (!shape) return shape.error();
      auto type_code = field_u8(r, *tt, f_tensor::type, 0);
      if (!type_code) return type_code.error();
      auto buf_idx = field_u32(r, *tt, f_tensor::buffer, 0);
      if (!buf_idx) return buf_idx.error();
      std::string tname;
      if (auto s = field_string(r, *tt, f_tensor::name, tname); !s) {
        return s.error();
      }

      ir::ValueInfo vi;
      vi.name = out.intern(tname.empty()
                               ? ("tensor_" + std::to_string(ti))
                               : tname);
      vi.dtype = map_dtype(*type_code);
      for (int32_t d : *shape) vi.shape.push_back(static_cast<int64_t>(d));
      vi.producer = -1;
      g.values.push_back(std::move(vi));

      // Initializer: only when the referenced buffer actually holds data.
      if (*buf_idx < buffers.size() && buffers[*buf_idx].len > 0) {
        ir::TensorRef tr;
        tr.name = g.values.back().name;
        tr.dtype = map_dtype(*type_code);
        for (int32_t d : *shape) tr.shape.push_back(static_cast<int64_t>(d));
        tr.file_offset = buffers[*buf_idx].offset;  // recorded, never read
        tr.byte_len = buffers[*buf_idx].len;
        g.initializers.push_back(std::move(tr));
      }
    }

    // Graph inputs/outputs -> value indices.
    {
      auto ins = read_i32_vector(r, *sg, f_subgraph::inputs);
      if (!ins) return ins.error();
      for (int32_t idx : *ins) {
        if (idx >= 0 && static_cast<size_t>(idx) < g.values.size()) {
          g.graph_inputs.push_back(static_cast<uint32_t>(idx));
        }
      }
      auto outs = read_i32_vector(r, *sg, f_subgraph::outputs);
      if (!outs) return outs.error();
      for (int32_t idx : *outs) {
        if (idx >= 0 && static_cast<size_t>(idx) < g.values.size()) {
          g.graph_outputs.push_back(static_cast<uint32_t>(idx));
        }
      }
    }

    // Operators -> Nodes. Node input/output tensor indices become edge_refs
    // entries (which index Graph::values).
    auto operators = read_offset_vector(r, *sg, f_subgraph::operators);
    if (!operators) return operators.error();

    g.nodes.reserve(operators->size());
    for (uint64_t oi = 0; oi < operators->size(); ++oi) {
      auto ot = open_table(r, (*operators)[oi]);
      if (!ot) return ot.error();

      auto opcode_index = field_u32(r, *ot, f_operator::opcode_index, 0);
      if (!opcode_index) return opcode_index.error();
      auto op_inputs = read_i32_vector(r, *ot, f_operator::inputs);
      if (!op_inputs) return op_inputs.error();
      auto op_outputs = read_i32_vector(r, *ot, f_operator::outputs);
      if (!op_outputs) return op_outputs.error();

      ir::Node node;
      node.op_type = out.intern(*opcode_index < op_names.size()
                                    ? op_names[*opcode_index]
                                    : ("BUILTIN_" +
                                       std::to_string(*opcode_index)));
      node.name = out.intern("node_" + std::to_string(oi));

      const uint32_t node_idx = static_cast<uint32_t>(g.nodes.size());

      // Inputs: append value indices to edge_refs; -1 (optional) is skipped.
      node.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
      for (int32_t vidx : *op_inputs) {
        if (vidx >= 0 && static_cast<size_t>(vidx) < g.values.size()) {
          g.edge_refs.push_back(static_cast<uint32_t>(vidx));
        }
      }
      node.inputs.count =
          static_cast<uint32_t>(g.edge_refs.size()) - node.inputs.begin;

      // Outputs: append and mark this node as producer of each output value.
      node.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
      for (int32_t vidx : *op_outputs) {
        if (vidx >= 0 && static_cast<size_t>(vidx) < g.values.size()) {
          g.edge_refs.push_back(static_cast<uint32_t>(vidx));
          g.values[static_cast<size_t>(vidx)].producer =
              static_cast<int32_t>(node_idx);
        }
      }
      node.outputs.count =
          static_cast<uint32_t>(g.edge_refs.size()) - node.outputs.begin;

      // Control-flow linking: If/While/CallOnce carry a builtin_options union
      // (tag = field 3, table = field 4) that references other subgraphs by
      // index. Link the primary referenced subgraph into Node.subgraph so the
      // view can descend. Bounds-checked against the subgraph count.
      {
        auto tag = field_u8(r, *ot, f_operator::builtin_options_type, 0);
        if (!tag) return tag.error();
        if (*tag == builtin_options_type::IfOptions ||
            *tag == builtin_options_type::WhileOptions ||
            *tag == builtin_options_type::CallOnceOptions) {
          Table opts;
          bool present = false;
          if (auto s = field_table(r, *ot, f_operator::builtin_options, opts,
                                   present);
              !s) {
            return s.error();
          }
          if (present) {
            uint32_t sub_field = 0;
            switch (*tag) {
              case builtin_options_type::IfOptions:
                sub_field = f_if::then_subgraph_index;
                break;
              case builtin_options_type::WhileOptions:
                sub_field = f_while::cond_subgraph_index;
                break;
              default:  // CallOnceOptions
                sub_field = f_call_once::init_subgraph_index;
                break;
            }
            auto sidx = field_i32(r, opts, sub_field, -1);
            if (!sidx) return sidx.error();
            // Subgraph indices are stable: out.graphs[i] == subgraphs[i] since
            // we append in order. Bounds-check against the total count.
            if (*sidx >= 0 && static_cast<uint64_t>(*sidx) < sg_total) {
              node.subgraph = *sidx;
            }
          }
        }
      }

      g.nodes.push_back(std::move(node));
    }

    out.graphs.push_back(std::move(g));

    if (sg_total > 0) {
      progress.set(0.45f + 0.5f * static_cast<float>(sg_i + 1) /
                                static_cast<float>(sg_total),
                   "TFLite: subgraphs");
    }
  }

  if (out.graphs.empty()) {
    // A model with no subgraphs is still valid structurally; keep has_graph
    // true but there is nothing to show. Treat as malformed for clarity.
    return err("TFLite model has no subgraphs", *root_pos);
  }

  progress.set(1.0f, "TFLite: done");
  return out;
}

}  // namespace netvis::tflite
