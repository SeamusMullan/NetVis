// SPDX-License-Identifier: Apache-2.0
// parsers/pytorch/PytorchParser.cpp — PyTorch .pt/.pth/.bin loader.
//
// Two entry points (Parser.h):
//   parse_zip    — modern torch.save format: a ZIP archive containing data.pkl
//                  (the object graph) plus data/<key> tensor payload blobs.
//                  TorchScript archives add code/ and constants.pkl; PyTorch
//                  Mobile lite-interpreter archives (.ptl) add bytecode.pkl, whose
//                  main method is rebuilt as a compute graph (#137,
//                  MobileBytecode.h), falling back to its exact op inventory.
//   parse_legacy — a standalone pickle stream at file offset 0.
//
// SECURITY: the pickle graph is interpreted by the restricted, non-executing
// PickleVM (see PickleVM.h). No user code ever runs.
//
// WEIGHTS: tensor payloads are NEVER read. For zip entries we compute the
// absolute file offset of the uncompressed blob from its local file header
// (torch stores tensor data with the ZIP "stored" method, i.e. uncompressed)
// and record offset+length in the TensorRef. We never call
// mz_zip_reader_extract on payload entries — only on the small structural
// entries (data.pkl, constants.pkl, bytecode.pkl, code/), each size-capped.
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/ByteReader.h"
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/pytorch/MobileBytecode.h"
#include "parsers/pytorch/PickleVM.h"
#include "parsers/pytorch/TorchScriptIR.h"

#include "miniz.h"

namespace netvis::pytorch {
namespace {

// ZIP local file header: 30 fixed bytes, then filename_len + extra_len.
// Layout (offsets from local header start): 26 = filename_len (u16),
// 28 = extra_len (u16). Payload begins at header + 30 + fn_len + extra_len.
constexpr uint32_t kLocalHeaderSig = 0x04034b50;

// Compute the absolute file offset of an entry's uncompressed payload from its
// local file header. Bounds-checked via ByteReader; returns false on any error.
bool payload_offset_from_local_header(const uint8_t* base, uint64_t file_size,
                                      uint64_t local_header_ofs,
                                      uint64_t& out_offset) {
  ByteReader r(base, file_size);
  if (!r.seek(local_header_ofs)) return false;
  auto sig = r.u32le();
  if (!sig || *sig != kLocalHeaderSig) return false;
  // Skip to filename_len/extra_len: 26 bytes after the 4-byte signature.
  // We've consumed 4 (sig); skip 22 more to reach offset 26.
  r.skip(22);
  auto fn_len = r.u16le();
  if (!fn_len) return false;
  auto extra_len = r.u16le();
  if (!extra_len) return false;
  out_offset = local_header_ofs + 30ULL + *fn_len + *extra_len;
  if (out_offset > file_size) return false;
  return true;
}

// Directory prefix of a path ("prefix/sub/data.pkl" -> "prefix/sub/").
std::string dir_prefix(const std::string& path) {
  auto slash = path.find_last_of('/');
  if (slash == std::string::npos) return "";
  return path.substr(0, slash + 1);
}

// --- TorchScript op inventory (best-effort, bounded) -------------------------
// Extract method names and op identifiers from serialized TorchScript code
// entries. This is a BOUNDED text scan — NOT a Python parser — that harvests
// simple patterns (def <name>, torch.<op>, aten::<op>, ops.<op>) to surface an
// op/method inventory. Hostile/huge code → truncated result, never crash/hang.

constexpr size_t kMaxCodeBytesPerFile = 512 * 1024;  // cap per code entry
constexpr size_t kMaxIdentifiers = 512;              // cap total ops harvested
constexpr size_t kMaxIdentifierLen = 128;            // cap individual identifier

struct OpInventory {
  std::vector<std::string> methods;
  std::vector<std::string> ops;
};

// Check if a character is valid for a Python identifier continuation.
bool is_ident_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}

// Scan code text for method definitions and op calls. Bounded, non-executing.
void scan_torchscript_code(const uint8_t* data, size_t len, OpInventory& inv) {
  if (len > kMaxCodeBytesPerFile) len = kMaxCodeBytesPerFile;

  std::unordered_set<std::string> methods_seen;
  std::unordered_set<std::string> ops_seen;

  size_t i = 0;
  while (i < len && (methods_seen.size() + ops_seen.size()) < kMaxIdentifiers) {
    // Look for "def <name>(" — method definition
    if (i + 4 <= len && data[i] == 'd' && data[i+1] == 'e' &&
        data[i+2] == 'f' && data[i+3] == ' ') {
      i += 4;
      // skip whitespace
      while (i < len && (data[i] == ' ' || data[i] == '\t')) ++i;
      // extract identifier
      size_t start = i;
      while (i < len && is_ident_char(data[i])) ++i;
      size_t ident_len = i - start;
      if (ident_len > 0 && ident_len <= kMaxIdentifierLen) {
        std::string method(reinterpret_cast<const char*>(data + start), ident_len);
        if (methods_seen.insert(method).second &&
            inv.methods.size() < kMaxIdentifiers) {
          inv.methods.push_back(method);
        }
      }
      continue;
    }

    // Look for op patterns: "torch.", "aten::", "prim::", "ops."
    const char* patterns[] = {"torch.", "aten::", "prim::", "ops."};
    size_t pattern_lens[] = {6, 6, 6, 4};
    bool found_pattern = false;

    for (size_t p = 0; p < 4; ++p) {
      size_t plen = pattern_lens[p];
      if (i + plen <= len &&
          std::memcmp(data + i, patterns[p], plen) == 0) {
        i += plen;
        // extract identifier after pattern
        size_t start = i;
        while (i < len && is_ident_char(data[i])) ++i;
        size_t ident_len = i - start;
        if (ident_len > 0 && ident_len <= kMaxIdentifierLen) {
          // reconstruct full op name (pattern + identifier)
          std::string op = std::string(patterns[p]) +
                          std::string(reinterpret_cast<const char*>(data + start),
                                     ident_len);
          if (ops_seen.insert(op).second && ops_seen.size() <= kMaxIdentifiers) {
            inv.ops.push_back(op);
          }
        }
        found_pattern = true;
        break;
      }
    }

    if (!found_pattern) ++i;
  }
}

// --- state_dict walking -------------------------------------------------------
// Collect (name -> TensorRef) from the unpickled value tree. Keys are joined
// with '.' so nested modules read like "encoder.layers.0.weight".
//
// The traversal is ts::for_each_tensor, the one ModuleEnv also uses for
// canonical tensor names, so a parameter has the same name in the tensor table
// and in a graph. It descends dicts, lists, tuples and -- #137 -- TorchScript
// objects (their recorded BUILD state, in slot order): torch.jit.save and .ptl
// archives pickle the module itself, not a state_dict.
//
// SECURITY: the pickle VM's memo (BINGET/DUP) lets a hostile file build a value
// graph that is cyclic (a list containing itself) or a shared DAG with
// exponentially many root-to-leaf paths (~n opcodes → 2^n paths). A naive
// recursion would stack-overflow or hang. The traversal guards both: a `visited`
// set keyed on Value identity expands each shared container/object at most once
// (collapsing the DAG and breaking cycles), and a depth cap bounds pathological
// nesting. Malformed input therefore yields a partial/empty result, never a
// crash.
void collect_tensors(const ValuePtr& v, ir::Model& model) {
  ts::for_each_tensor(v, [&model](const std::string& path, const Value& tv) {
    ir::TensorRef t = tv.tensor;
    t.name = model.intern(path);
    if (!tv.dtype_label.empty()) t.dtype_label = model.intern(tv.dtype_label);
    model.flat_tensors.push_back(std::move(t));
  });
}

// --- storage record maps --------------------------------------------------------
// key -> payload location for every record directly under `dir` ("<prefix>data/",
// "<prefix>constants/", "<prefix>bytecode/"). Offsets come from local file
// headers; payload bytes are NEVER decompressed or extracted.
struct StorageEntry {
  uint64_t offset;
  uint64_t len;
};
using StorageMap = std::map<std::string, StorageEntry>;

std::shared_ptr<StorageMap> build_storage_map(mz_zip_archive& zip,
                                              const uint8_t* base, uint64_t size,
                                              const std::string& dir) {
  auto key_map = std::make_shared<StorageMap>();
  mz_uint num = mz_zip_reader_get_num_files(&zip);
  for (mz_uint i = 0; i < num; ++i) {
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&zip, i, &st)) continue;
    if (st.m_is_directory) continue;
    std::string name = st.m_filename;
    if (name.rfind(dir, 0) != 0) continue;
    std::string key = name.substr(dir.size());
    if (key.empty() || key.find('/') != std::string::npos) continue;
    uint64_t off = 0;
    if (!payload_offset_from_local_header(base, size, st.m_local_header_ofs, off))
      continue;
    (*key_map)[key] = {off, st.m_uncomp_size};
  }
  return key_map;
}

StorageResolver make_resolver(std::shared_ptr<StorageMap> key_map) {
  StorageResolver resolver;
  resolver.resolve = [key_map = std::move(key_map)](
                         const std::string& key, uint64_t& o, uint64_t& l) -> bool {
    auto it = key_map->find(key);
    if (it == key_map->end()) return false;
    o = it->second.offset;
    l = it->second.len;
    return true;
  };
  return resolver;
}

// If the top-level dict has a "state_dict" entry, descend into it; otherwise
// treat the whole value as the state_dict.
ValuePtr find_state_dict(const ValuePtr& top) {
  if (top && top->kind == Value::Kind::Dict) {
    for (const auto& kv : top->pairs) {
      if (kv.first && kv.first->kind == Value::Kind::Str &&
          kv.first->s == "state_dict" && kv.second &&
          kv.second->kind == Value::Kind::Dict) {
        return kv.second;
      }
    }
  }
  return top;
}

// --- #40: best-effort code/ op inventory (desktop TorchScript archives) --------
void emit_code_inventory(mz_zip_archive& zip, bool has_code, ir::Model& model) {
  // TorchScript archive: scan code entries for best-effort op/method listing
  OpInventory inv;
  if (has_code) {
    mz_uint num = mz_zip_reader_get_num_files(&zip);
    for (mz_uint i = 0; i < num; ++i) {
      mz_zip_archive_file_stat st;
      if (!mz_zip_reader_file_stat(&zip, i, &st)) continue;
      if (st.m_is_directory) continue;
      std::string name = st.m_filename;
      // Look for code/ entries (typically code/*.py)
      if (name.find("/code/") != std::string::npos ||
          name.rfind("code/", 0) == 0) {
        // Extract ONLY the small code file (structural text, not payload)
        size_t code_size = 0;
        void* code_mem = mz_zip_reader_extract_to_heap(&zip, i, &code_size, 0);
        if (code_mem && code_size > 0) {
          scan_torchscript_code(reinterpret_cast<const uint8_t*>(code_mem),
                               code_size, inv);
        }
        if (code_mem) mz_free(code_mem);
      }
    }
  }

  // Emit best-effort metadata
  std::string note = "best-effort op inventory (heuristic scan); parameter table below";
  if (!inv.methods.empty()) {
    std::string methods_str;
    for (size_t i = 0; i < inv.methods.size(); ++i) {
      if (i > 0) methods_str += ", ";
      methods_str += inv.methods[i];
    }
    model.metadata.emplace_back(model.intern("torchscript.methods"),
                                model.intern(methods_str));
  }
  if (!inv.ops.empty()) {
    std::string ops_str;
    for (size_t i = 0; i < inv.ops.size(); ++i) {
      if (i > 0) ops_str += ", ";
      ops_str += inv.ops[i];
    }
    model.metadata.emplace_back(model.intern("torchscript.ops"),
                                model.intern(ops_str));
  }
  model.metadata.emplace_back(model.intern("torchscript"),
                              model.intern(note));
}

// --- #137: PyTorch Mobile lite-interpreter bytecode ----------------------------
enum class MobileStatus : uint8_t { Absent, TablesFailed, Inventory, Graph };

// Where an offset relative to bytecode.pkl lands in the file. A STORED entry's
// bytes are the file's bytes, so the offset becomes absolute; inside a
// compressed entry only the relative offset means anything.
struct Located {
  std::string note_suffix;   // appended to metadata notes
  std::string error_suffix;  // appended to a returned Error message
  uint64_t abs = UINT64_MAX;
};

Located locate(uint64_t rel, bool stored, uint64_t payload_off) {
  Located l;
  if (rel == UINT64_MAX) return l;
  if (stored && payload_off != UINT64_MAX) {
    l.abs = payload_off + rel;
    l.note_suffix = " (byte " + std::to_string(l.abs) + ")";
  } else {
    l.note_suffix = " (in compressed bytecode.pkl at +" + std::to_string(rel) + ")";
    l.error_suffix = l.note_suffix;
  }
  return l;
}

struct MobileOutcome {
  MobileStatus status = MobileStatus::Absent;
  bool tables = false;      // bytecode.pkl decoded
  mobile::BcModule module;  // valid when tables
  int32_t main_fn = -1;
  std::string reason;       // TablesFailed / Inventory
  Located where;
  mobile::BuildResult built;  // Graph
};

// Heap copy of one structural zip entry, freed on scope exit.
struct HeapEntry {
  void* p = nullptr;
  size_t n = 0;
  ~HeapEntry() {
    if (p) mz_free(p);
  }
};

// The mobile path (spec #137 §4.1). Never fails the parse by itself: problems
// become the outcome's status + reason, located in the file.
void run_mobile(mz_zip_archive& zip, const uint8_t* base, uint64_t size,
                const std::vector<std::string>& names, mz_uint bc_index,
                const std::string& prefix, const ts::ModuleEnv& env,
                ProgressSink& progress, ir::Model& model, MobileOutcome& out) {
  const mobile::BytecodeLimits lim;
  mz_zip_archive_file_stat st;
  if (!mz_zip_reader_file_stat(&zip, bc_index, &st)) {
    out.status = MobileStatus::TablesFailed;
    out.reason = "failed to read bytecode.pkl";
    return;
  }
  const bool stored = st.m_method == 0;
  uint64_t payload_off = UINT64_MAX;
  if (!payload_offset_from_local_header(base, size, st.m_local_header_ofs, payload_off))
    payload_off = UINT64_MAX;
  auto fail_tables = [&](std::string msg, uint64_t rel) {
    out.status = MobileStatus::TablesFailed;
    out.reason = std::move(msg);
    out.where = locate(rel, stored, payload_off);
  };

  // Zip-bomb guard BEFORE decompression: bytecode.pkl is structure, not weights.
  if (st.m_uncomp_size > lim.max_pickle_bytes) {
    fail_tables("bytecode.pkl too large: " + std::to_string(st.m_uncomp_size) + " bytes",
                UINT64_MAX);
    return;
  }
  HeapEntry bc;
  bc.p = mz_zip_reader_extract_to_heap(&zip, bc_index, &bc.n, 0);
  if (!bc.p) {
    fail_tables("failed to read bytecode.pkl", UINT64_MAX);
    return;
  }

  // Tensor records (BytecodeDeserializer::readArchive): v<=4 archives keep them
  // in <root>bytecode/<idx>; v5+ share <root>constants/<n>.storage with
  // constants.pkl, so a bytecode tensor constant IS that record.
  bool legacy_dir = false;
  for (const std::string& n : names) {
    if (n.rfind(prefix, 0) == 0 && n.find("bytecode/", prefix.size()) != std::string::npos) {
      legacy_dir = true;
      break;
    }
  }
  StorageResolver bc_resolver = make_resolver(build_storage_map(
      zip, base, size, prefix + (legacy_dir ? "bytecode/" : "constants/")));
  auto decoded = mobile::decode_bytecode_pickle(static_cast<const uint8_t*>(bc.p), bc.n,
                                                bc_resolver, lim);
  if (!decoded) {
    fail_tables(decoded.error().message, decoded.error().offset);
    return;
  }
  out.module = decoded.take();
  out.tables = true;

  // constants.pkl names bytecode tensor constants CONSTANTS.c<i>. Optional and
  // non-fatal: on any problem the names fall back to <method>.c<k>.
  ts::TensorConstNames const_names;
  const std::string constants_path = prefix + "constants.pkl";
  for (mz_uint i = 0; i < names.size(); ++i) {
    if (names[i] != constants_path) continue;
    mz_zip_archive_file_stat cst;
    if (!mz_zip_reader_file_stat(&zip, i, &cst) || cst.m_is_directory ||
        cst.m_uncomp_size > lim.max_pickle_bytes)
      break;
    HeapEntry cp;
    cp.p = mz_zip_reader_extract_to_heap(&zip, i, &cp.n, 0);
    if (!cp.p) break;
    StorageResolver c_resolver =
        make_resolver(build_storage_map(zip, base, size, prefix + "constants/"));
    PickleLimits pl;
    pl.max_values = lim.max_pickle_values;
    PickleVM vm(static_cast<const uint8_t*>(cp.p), cp.n, c_resolver, pl);
    auto cv = vm.run();
    if (cv) const_names.add_from_constants_pkl(*cv);
    break;
  }

  out.main_fn = mobile::select_main_function(out.module, env.root_class());
  if (out.module.version <= 5) {
    out.status = MobileStatus::Inventory;
    out.reason = "bytecode v" + std::to_string(out.module.version) +
                 " does not record operator arity (added in v6)";
    return;
  }
  if (out.main_fn < 0) {
    out.status = MobileStatus::Inventory;
    out.reason = "no forward method";
    return;
  }
  progress.set(0.8f, "graph");
  mobile::BuildContext ctx{model, env, const_names, lim};
  auto built = mobile::build_method_graph(out.module, static_cast<size_t>(out.main_fn), ctx);
  if (!built) {  // build_method_graph left model.graphs empty
    out.status = MobileStatus::Inventory;
    out.reason = built.error().message;
    out.where = locate(built.error().offset, stored, payload_off);
    return;
  }
  out.status = MobileStatus::Graph;
  out.built = built.take();
}

// Metadata for Graph and Inventory modes (spec #137 §4.8), in table order.
void emit_mobile_metadata(const MobileOutcome& mob, const ts::ModuleEnv& env,
                          ir::Model& model) {
  auto put = [&model](const char* key, const std::string& value) {
    model.metadata.emplace_back(model.intern(key), model.intern(value));
  };
  const mobile::BcModule& m = mob.module;
  const std::string ver = std::to_string(m.version);
  const mobile::BcFunction* main_fn =
      mob.main_fn >= 0 ? &m.functions[static_cast<size_t>(mob.main_fn)] : nullptr;
  const bool graph = mob.status == MobileStatus::Graph;

  if (graph && main_fn)
    put("torchscript", "mobile lite-interpreter bytecode v" + ver + ": " +
                           main_fn->qualified_name + " decoded from bytecode.pkl");
  else
    put("torchscript", "op inventory from bytecode.pkl (exact); graph not built: " +
                           mob.reason + mob.where.note_suffix);
  put("torchscript.bytecode_version", ver);

  std::string methods;
  for (size_t i = 0; i < m.functions.size(); ++i) {
    if (i) methods += ", ";
    methods += m.functions[i].qualified_name;
  }
  put("torchscript.methods", methods);

  std::string ops;
  std::unordered_set<std::string> seen;  // lookup only; order is first-seen
  for (const auto& f : m.functions) {
    for (const auto& o : f.operators) {
      std::string id = !o.well_formed      ? std::string("?")
                       : o.overload.empty() ? o.name
                                            : o.name + "." + o.overload;
      if (!seen.insert(id).second) continue;
      if (!ops.empty()) ops += ", ";
      ops += id;
    }
  }
  put("torchscript.ops", ops);

  if (main_fn) {
    std::string sig = mobile::method_signature(*main_fn);
    if (!sig.empty()) put("torchscript.signature", sig);
  }

  if (graph) {
    if (!mob.built.self_note.empty()) put("torchscript.self", mob.built.self_note);
    size_t bound = 0;
    for (const Value* t : mob.built.bound_module_tensors)
      if (!env.tensor_path(t, "").empty()) ++bound;
    const size_t total = env.module_tensor_count();
    if (total > bound)
      put("torchscript.unreferenced_tensors", std::to_string(total - bound));
    std::set<std::pair<std::string, uint64_t>> distinct;
    for (const ir::Graph& g : model.graphs)
      for (const ir::TensorRef& t : g.initializers)
        distinct.emplace(std::string(model.str(t.name)), t.file_offset);
    put("tensors", std::to_string(distinct.size()));
  } else {
    put("tensors", std::to_string(model.flat_tensors.size()));
  }
}

}  // namespace

// ============================================================================
// parse_zip
// ============================================================================
Result<ir::Model> parse_zip(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "opening archive");
  const uint8_t* base = file.data();
  uint64_t size = file.size();
  if (!base || size == 0) return err("empty file", 0);

  mz_zip_archive zip;
  std::memset(&zip, 0, sizeof(zip));
  if (!mz_zip_reader_init_mem(&zip, base, static_cast<size_t>(size), 0)) {
    // PyTorch Mobile's flatbuffer container (_use_flatbuffer=True) carries the
    // file identifier "PTMF" at bytes 4..8 (Detect routes it here on purpose).
    if (size >= 8 && std::memcmp(base + 4, "PTMF", 4) == 0)
      return err("PyTorch Mobile flatbuffer (.ptl, PTMF) is not supported; only the "
                 "zip/pickle lite-interpreter container is",
                 4);
    return err("not a valid zip archive", 0);
  }

  // RAII-ish cleanup guard for early returns.
  struct ZipGuard {
    mz_zip_archive* z;
    ~ZipGuard() { mz_zip_reader_end(z); }
  } guard{&zip};

  mz_uint num = mz_zip_reader_get_num_files(&zip);

  // Locate the pickle entry: data.pkl or */data.pkl. Its directory is the
  // archive prefix under which tensor payloads live (<prefix>data/<key>).
  // Every central-directory name is kept (in CD order) for the bytecode.pkl
  // selection rule.
  mz_uint pkl_index = num;  // sentinel
  std::string pkl_path;
  bool has_constants = false;
  bool has_code = false;
  std::vector<std::string> names(num);
  for (mz_uint i = 0; i < num; ++i) {
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&zip, i, &st)) continue;
    names[i] = st.m_filename;
    if (st.m_is_directory) continue;
    const std::string& name = names[i];
    // strip basename
    std::string bn = name;
    auto sl = name.find_last_of('/');
    if (sl != std::string::npos) bn = name.substr(sl + 1);
    if (bn == "data.pkl" && pkl_index == num) {
      pkl_index = i;
      pkl_path = name;
    }
    if (bn == "constants.pkl") has_constants = true;
    if (name.find("/code/") != std::string::npos ||
        name.rfind("code/", 0) == 0)
      has_code = true;
  }
  const bool has_data_pkl = pkl_index != num;
  // A PyTorch Mobile archive also carries bytecode.pkl (#137).
  const int32_t bc_index =
      mobile::select_bytecode_entry(names, has_data_pkl ? pkl_path : std::string());
  if (!has_data_pkl && bc_index < 0) return err("no data.pkl in archive", 0);

  std::string prefix =
      dir_prefix(has_data_pkl ? pkl_path : names[static_cast<size_t>(bc_index)]);

  // --- data.pkl: the module object (or a state_dict) ---------------------------
  ValuePtr top;
  if (has_data_pkl) {
    progress.set(0.3f, "reading pickle");

    // Extract ONLY the small data.pkl (structure), never tensor payloads.
    size_t pkl_size = 0;
    void* pkl_mem = mz_zip_reader_extract_to_heap(&zip, pkl_index, &pkl_size, 0);
    if (!pkl_mem) return err("failed to read data.pkl", 0);

    struct MemGuard {
      void* p;
      ~MemGuard() { if (p) mz_free(p); }
    } memg{pkl_mem};

    // Storage-key -> (offset,len) resolver for <prefix>data/ records.
    StorageResolver resolver =
        make_resolver(build_storage_map(zip, base, size, prefix + "data/"));

    progress.set(0.6f, "interpreting");
    PickleVM vm(reinterpret_cast<const uint8_t*>(pkl_mem), pkl_size, resolver);
    auto top_r = vm.run();
    if (!top_r) return top_r.error();
    top = *top_r;
  }
  // Keeps the data.pkl tree alive for the whole build (graph syms point into it).
  const ts::ModuleEnv env = top ? ts::ModuleEnv(top) : ts::ModuleEnv();

  ir::Model model;
  MobileOutcome mob;
  if (bc_index >= 0) {
    progress.set(0.65f, "bytecode");
    run_mobile(zip, base, size, names, static_cast<mz_uint>(bc_index), prefix, env,
               progress, model, mob);
  }

  model.format_name = model.intern("PyTorch");
  if (mob.status == MobileStatus::Graph) {
    model.has_graph = true;  // flat_tensors stays empty, like every graph format
  } else {
    model.has_graph = false;
    if (top) collect_tensors(find_state_dict(top), model);
  }
  if (mob.tables)
    model.version_info = model.intern("bytecode v" + std::to_string(mob.module.version));

  if (mob.status == MobileStatus::Graph || mob.status == MobileStatus::Inventory) {
    // Exact inventory from bytecode.pkl: the #40 heuristic code/ scan is skipped.
    emit_mobile_metadata(mob, env, model);
  } else {
    if (has_constants || has_code) emit_code_inventory(zip, has_code, model);
    if (mob.status == MobileStatus::TablesFailed)
      model.metadata.emplace_back(
          model.intern("torchscript.bytecode"),
          model.intern("not decoded: " + mob.reason + mob.where.note_suffix));
    model.metadata.emplace_back(model.intern("tensors"),
                                model.intern(std::to_string(model.flat_tensors.size())));
  }
  if (!has_data_pkl && mob.status == MobileStatus::TablesFailed)
    return err(mob.reason + mob.where.error_suffix, mob.where.abs);

  progress.set(1.0f, "done");
  return model;
}

// ============================================================================
// parse_legacy — standalone pickle at offset 0.
// ============================================================================
Result<ir::Model> parse_legacy(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "reading pickle");
  const uint8_t* base = file.data();
  uint64_t size = file.size();
  if (!base || size < 2) return err("file too small", 0);
  if (base[0] != 0x80) return err("not a pickle stream", 0);
  // proto byte must be 2..5
  if (base[1] < 2 || base[1] > 5) return err("unsupported pickle protocol", 1);

  // Legacy .pt files interleave storage payloads after the pickle; without the
  // torch legacy tar/long-index machinery we cannot resolve payload offsets
  // reliably from the mmap alone. We still emit TensorRefs (shape/dtype) with
  // file_offset == UINT64_MAX (resolver always fails), and note the limitation.
  StorageResolver resolver;
  resolver.resolve = [](const std::string&, uint64_t&, uint64_t&) -> bool {
    return false;
  };

  progress.set(0.5f, "interpreting");
  PickleVM vm(base, size, resolver);
  auto top_r = vm.run();
  if (!top_r) return top_r.error();

  ir::Model model;
  model.format_name = model.intern("PyTorch");
  model.has_graph = false;

  ValuePtr sd = find_state_dict(*top_r);
  collect_tensors(sd, model);

  model.metadata.emplace_back(
      model.intern("legacy"),
      model.intern("legacy pickle; tensor payload offsets unresolved"));
  model.metadata.emplace_back(model.intern("tensors"),
                              model.intern(std::to_string(model.flat_tensors.size())));

  progress.set(1.0f, "done");
  return model;
}

}  // namespace netvis::pytorch
