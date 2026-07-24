// ir/IR.h — the common intermediate representation all parsers target.
//
// DECISION (spec §3.2): cache-friendly struct-of-arrays. Nodes/values are POD
// with 32-bit indices instead of pointers, and every string is a StringId into
// the Model's arena. A Range { begin, count } indexes into a shared flat array
// (edge_refs / attributes) so a Node carries no owned containers — the whole IR
// is a handful of contiguous vectors, which is what lets us hold a 100k-node
// graph without pointer-chasing and hash its structure cheaply for the cache.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/SmallVec.h"
#include "core/StringArena.h"

namespace netvis::ir {

// Element type of a tensor/value. Q4/Q8 cover GGUF quantized blocks (spec §6.4);
// dequantization is out of scope for v1 (spec §12) so these are labels only.
enum class DType : uint8_t {
  F32, F16, BF16, F64,
  I8, I16, I32, I64,
  U8, U16, U32, U64,
  Bool, Q4, Q8, Unknown
};

// Human-readable dtype name (for panels/tables).
const char* dtype_name(DType d);
// Byte size of one element, or 0 for quantized/unknown (block-based).
uint32_t dtype_size(DType d);

// A half-open range [begin, begin+count) into a flat backing array.
struct Range {
  uint32_t begin = 0;
  uint32_t count = 0;
};

// A weight/initializer payload, readable lazily from the mmap. The payload is
// NEVER decoded at parse time — only offset+length are recorded (spec §2.1).
struct TensorRef {
  StringId name;
  DType dtype = DType::Unknown;
  SmallVec<int64_t, 6> shape;   // -1 == dynamic dimension
  uint64_t file_offset = UINT64_MAX;  // into the mmap; UINT64_MAX if external/absent
  uint64_t byte_len = 0;
  StringId external_path;       // ONNX external data path; empty otherwise

  // v0.6.3 (#85), APPEND-ONLY. ir::DType is frozen at 16 enumerators (the plugin
  // SDK static_asserts it), so an exact-but-unmapped element type is carried as a
  // human LABEL rather than a new DType: OpenVINO i4/u4/nf4/u1/f8*, CoreML MIL
  // sub-byte. Empty => panels fall back to dtype_name(dtype). Layout-inert: not in
  // structure_hash, never serialized to the layout cache (no kVersion bump).
  StringId dtype_label;
  // true => file_offset points at a CoreML MIL blob_metadata header (64 bytes,
  // sentinel 0xDEADBEEF), NOT the raw payload. TensorStats follows the header to
  // the real data; structural parsing still records offset+len only (zero reads).
  bool blob_indirect = false;

  // Product of dims (>=0 dims only); 0 if any dim is dynamic/unset.
  int64_t elem_count() const {
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (int64_t d : shape) {
      if (d < 0) return 0;
      n *= d;
    }
    return n;
  }
};

// Tagged-union attribute value (spec §3.2). Kind selects the active member.
struct AttrValue {
  enum class Kind : uint8_t {
    None, Int, Float, String, Ints, Floats, Strings, Tensor, Graph
  } kind = Kind::None;

  int64_t i = 0;
  double f = 0.0;
  StringId s;
  std::vector<int64_t> ints;
  std::vector<double> floats;
  std::vector<StringId> strings;
  TensorRef tensor;        // Kind::Tensor
  int32_t graph = -1;      // Kind::Graph -> index into Model::graphs
};

struct Attribute {
  StringId name;
  AttrValue value;
};

// A compute node. inputs/outputs index Graph::edge_refs; attributes index
// Graph::attributes; subgraph (If/Loop/Scan body) indexes Model::graphs or -1.
struct Node {
  StringId op_type;   // "Conv", "MatMul", ...
  StringId name;
  Range inputs;       // -> Graph::edge_refs -> value indices
  Range outputs;
  Range attributes;   // -> Graph::attributes
  int32_t subgraph = -1;
};

// One SSA value (a graph edge / tensor flowing between nodes).
struct ValueInfo {
  StringId name;
  DType dtype = DType::Unknown;   // Unknown until shape inference (spec §7.3)
  SmallVec<int64_t, 6> shape;
  int32_t producer = -1;          // node index; -1 if graph input / initializer
};

struct Graph {
  StringId name;
  std::vector<Node> nodes;
  std::vector<ValueInfo> values;
  std::vector<uint32_t> edge_refs;   // node input/output slot -> value index
  std::vector<Attribute> attributes;
  std::vector<TensorRef> initializers;
  std::vector<uint32_t> graph_inputs;   // -> value indices
  std::vector<uint32_t> graph_outputs;  // -> value indices
};

struct Model {
  std::vector<Graph> graphs;   // graphs[0] == main
  std::vector<std::pair<StringId, StringId>> metadata;
  StringId format_name;        // "ONNX", "GGUF", ...
  StringId producer;
  StringId version_info;

  // has_graph == false => tensor-table mode (GGUF, SafeTensors, state_dict-only
  // PyTorch). flat_tensors holds every tensor with no compute graph (spec §8.6).
  bool has_graph = true;
  std::vector<TensorRef> flat_tensors;

  StringArena strings;         // all StringIds above resolve here

  // Convenience: intern into this model's arena.
  StringId intern(std::string_view s) { return strings.intern(s); }
  std::string_view str(StringId id) const { return strings.get(id); }
};

}  // namespace netvis::ir
