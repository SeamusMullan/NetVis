// parsers/pytorch/PytorchParser.cpp — PyTorch .pt/.pth/.bin loader.
//
// Two entry points (Parser.h):
//   parse_zip    — modern torch.save format: a ZIP archive containing data.pkl
//                  (the object graph) plus data/<key> tensor payload blobs.
//   parse_legacy — a standalone pickle stream at file offset 0.
//
// SECURITY: the pickle graph is interpreted by the restricted, non-executing
// PickleVM (see PickleVM.h). No user code ever runs.
//
// WEIGHTS: tensor payloads are NEVER read. For zip entries we compute the
// absolute file offset of the uncompressed blob from its local file header
// (torch stores tensor data with the ZIP "stored" method, i.e. uncompressed)
// and record offset+length in the TensorRef. We never call
// mz_zip_reader_extract on payload entries — only on the small data.pkl.
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/ByteReader.h"
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/pytorch/PickleVM.h"

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
// Recursively collect (name -> TensorRef) from the unpickled value tree. Keys
// are joined with '.' so nested modules read like "encoder.layers.0.weight".
//
// SECURITY: the pickle VM's memo (BINGET/DUP) lets a hostile file build a value
// graph that is cyclic (a list containing itself) or a shared DAG with
// exponentially many root-to-leaf paths (~n opcodes → 2^n paths). A naive
// recursion would stack-overflow or hang. We guard both: a `visited` set keyed
// on Value identity visits each shared node at most once (collapsing the DAG and
// breaking cycles), and a depth cap bounds pathological nesting. Malformed input
// therefore yields a partial/empty result, never a crash.
constexpr int kMaxCollectDepth = 256;

void collect_tensors(const ValuePtr& v, const std::string& prefix,
                     ir::Model& model,
                     std::unordered_set<const Value*>& visited, int depth) {
  if (!v || depth > kMaxCollectDepth) return;
  // Only containers can recurse / be shared; gating them on `visited` breaks
  // cycles and dedups shared subtrees without suppressing repeated leaf tensors.
  if (v->kind == Value::Kind::Dict || v->kind == Value::Kind::List ||
      v->kind == Value::Kind::Tuple) {
    if (!visited.insert(v.get()).second) return;  // already expanded this node
  }
  switch (v->kind) {
    case Value::Kind::Tensor: {
      ir::TensorRef t = v->tensor;
      t.name = model.intern(prefix);
      model.flat_tensors.push_back(std::move(t));
      break;
    }
    case Value::Kind::Dict: {
      for (const auto& kv : v->pairs) {
        std::string key;
        if (kv.first && kv.first->kind == Value::Kind::Str)
          key = kv.first->s;
        else if (kv.first && kv.first->kind == Value::Kind::Int)
          key = std::to_string(kv.first->i);
        std::string child = prefix.empty() ? key : prefix + "." + key;
        collect_tensors(kv.second, child, model, visited, depth + 1);
      }
      break;
    }
    case Value::Kind::List:
    case Value::Kind::Tuple: {
      for (size_t k = 0; k < v->items.size(); ++k) {
        std::string child =
            prefix.empty() ? std::to_string(k) : prefix + "." + std::to_string(k);
        collect_tensors(v->items[k], child, model, visited, depth + 1);
      }
      break;
    }
    default:
      break;
  }
}

// Convenience overload: start a fresh traversal.
void collect_tensors(const ValuePtr& v, const std::string& prefix,
                     ir::Model& model) {
  std::unordered_set<const Value*> visited;
  collect_tensors(v, prefix, model, visited, 0);
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
  if (!mz_zip_reader_init_mem(&zip, base, static_cast<size_t>(size), 0))
    return err("not a valid zip archive", 0);

  // RAII-ish cleanup guard for early returns.
  struct ZipGuard {
    mz_zip_archive* z;
    ~ZipGuard() { mz_zip_reader_end(z); }
  } guard{&zip};

  mz_uint num = mz_zip_reader_get_num_files(&zip);

  // Locate the pickle entry: data.pkl or */data.pkl. Its directory is the
  // archive prefix under which tensor payloads live (<prefix>data/<key>).
  mz_uint pkl_index = num;  // sentinel
  std::string pkl_path;
  bool has_constants = false;
  bool has_code = false;
  for (mz_uint i = 0; i < num; ++i) {
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&zip, i, &st)) continue;
    if (st.m_is_directory) continue;
    std::string name = st.m_filename;
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
  if (pkl_index == num) return err("no data.pkl in archive", 0);

  std::string prefix = dir_prefix(pkl_path);
  progress.set(0.3f, "reading pickle");

  // Extract ONLY the small data.pkl (structure), never tensor payloads.
  size_t pkl_size = 0;
  void* pkl_mem = mz_zip_reader_extract_to_heap(&zip, pkl_index, &pkl_size, 0);
  if (!pkl_mem) return err("failed to read data.pkl", 0);

  struct MemGuard {
    void* p;
    ~MemGuard() { if (p) mz_free(p); }
  } memg{pkl_mem};

  // Build a storage-key -> (offset,len) resolver by scanning central-dir
  // entries under <prefix>data/. We record local-header-derived payload
  // offsets; NEVER decompress/extract payload bytes.
  struct StorageEntry { uint64_t offset; uint64_t len; };
  auto key_map = std::make_shared<std::map<std::string, StorageEntry>>();
  std::string data_dir = prefix + "data/";
  for (mz_uint i = 0; i < num; ++i) {
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&zip, i, &st)) continue;
    if (st.m_is_directory) continue;
    std::string name = st.m_filename;
    if (name.rfind(data_dir, 0) != 0) continue;
    std::string key = name.substr(data_dir.size());
    if (key.empty() || key.find('/') != std::string::npos) continue;
    uint64_t off = 0;
    if (!payload_offset_from_local_header(base, size, st.m_local_header_ofs, off))
      continue;
    (*key_map)[key] = {off, st.m_uncomp_size};
  }

  StorageResolver resolver;
  resolver.resolve = [key_map](const std::string& key, uint64_t& o,
                               uint64_t& l) -> bool {
    auto it = key_map->find(key);
    if (it == key_map->end()) return false;
    o = it->second.offset;
    l = it->second.len;
    return true;
  };

  progress.set(0.6f, "interpreting");
  PickleVM vm(reinterpret_cast<const uint8_t*>(pkl_mem), pkl_size, resolver);
  auto top_r = vm.run();
  if (!top_r) return top_r.error();

  ir::Model model;
  model.format_name = model.intern("PyTorch");
  model.has_graph = false;

  ValuePtr sd = find_state_dict(*top_r);
  collect_tensors(sd, "", model);

  if (has_constants || has_code) {
    // TorchScript archive: scan code entries for best-effort op/method listing
    OpInventory inv;
    if (has_code) {
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
            mz_free(code_mem);
          }
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
  model.metadata.emplace_back(model.intern("tensors"),
                              model.intern(std::to_string(model.flat_tensors.size())));

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
  collect_tensors(sd, "", model);

  model.metadata.emplace_back(
      model.intern("legacy"),
      model.intern("legacy pickle; tensor payload offsets unresolved"));
  model.metadata.emplace_back(model.intern("tensors"),
                              model.intern(std::to_string(model.flat_tensors.size())));

  progress.set(1.0f, "done");
  return model;
}

}  // namespace netvis::pytorch
