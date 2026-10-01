// SPDX-License-Identifier: Apache-2.0
// parsers/pytorch/TorchScriptIR.h — the TorchScript -> ir:: mapping layer.
//
// Shared by the PyTorch Mobile bytecode decoder (#137, MobileBytecode.h) and the
// desktop code/*.py front-end (#136), so both produce the same graph for the same
// model: the same node kinds and attribute names, the same module-parameter
// binding and canonical tensor names, the same constant-folding rules, and the
// same prim::If / prim::Loop subgraph + capture convention.
//
// Conventions (spec #137 §4):
//   - Node kinds are the exact TorchScript IR spellings (kind:: below).
//   - Node NAMES are empty: TorchScript carries none, and synthetic names with
//     digits would feed CollapseTree's name-prefix grouping fake "blocks".
//   - Value names: graph inputs use their schema names, initializers their module
//     path ("fc.weight") or "CONSTANTS.c<i>", prim::GetAttr outputs their path,
//     everything else "%<n>" from ONE NameCounter per method shared by the main
//     graph and all its subgraphs. Names are unique per graph ("#2" suffix rule).
//   - A non-tensor constant operand is folded into the consuming node as an
//     attribute "arg<k>" (k = 0-based operand position); it only becomes a
//     prim::Constant node where an edge is required.
//   - prim::If / prim::Loop: inputs are the structural operands followed by the
//     captured outer values; the "captures" attribute counts the captures. Each
//     subgraph's graph_inputs are its own captures (plus a loop body's
//     structural inputs first), named identically to the outer values.
//
// Nothing here reads tensor payloads: TensorRefs carry offset+length only.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ir/IR.h"
#include "parsers/pytorch/PickleVM.h"

namespace netvis::pytorch::ts {

namespace kind {  // canonical TorchScript IR node kinds (exact spellings)
inline constexpr const char* kIf = "prim::If";
inline constexpr const char* kLoop = "prim::Loop";
inline constexpr const char* kConstant = "prim::Constant";
inline constexpr const char* kGetAttr = "prim::GetAttr";
inline constexpr const char* kSetAttr = "prim::SetAttr";
inline constexpr const char* kCallMethod = "prim::CallMethod";
inline constexpr const char* kListConstruct = "prim::ListConstruct";
inline constexpr const char* kListUnpack = "prim::ListUnpack";
inline constexpr const char* kTupleConstruct = "prim::TupleConstruct";
inline constexpr const char* kTupleIndex = "prim::TupleIndex";
inline constexpr const char* kTupleSlice = "prim::TupleSlice";
inline constexpr const char* kDictConstruct = "prim::DictConstruct";
inline constexpr const char* kCreateObject = "prim::CreateObject";
inline constexpr const char* kIsInstance = "prim::isinstance";
inline constexpr const char* kRaiseException = "prim::RaiseException";
inline constexpr const char* kUninitialized = "prim::Uninitialized";
inline constexpr const char* kUncheckedCast = "prim::unchecked_cast";
inline constexpr const char* kNumToTensor = "prim::NumToTensor";
inline constexpr const char* kToList = "prim::tolist";
inline constexpr const char* kDtype = "prim::dtype";
inline constexpr const char* kDevice = "prim::device";
inline constexpr const char* kIsCuda = "prim::is_cuda";
inline constexpr const char* kDim = "aten::dim";
inline constexpr const char* kNot = "aten::__not__";
inline constexpr const char* kIs = "aten::__is__";
inline constexpr const char* kIsNot = "aten::__isnot__";
inline constexpr const char* kFormat = "aten::format";
inline constexpr const char* kWarn = "aten::warn";
inline constexpr const char* kGetItem = "aten::__getitem__";
inline constexpr const char* kUnresolved = "?";  // malformed operator entry
}  // namespace kind

namespace attr {
inline constexpr const char* kOverload = "overload";
inline constexpr const char* kName = "name";
inline constexpr const char* kSlot = "slot";
inline constexpr const char* kType = "type";
inline constexpr const char* kTypes = "types";
inline constexpr const char* kValue = "value";
inline constexpr const char* kThen = "then_branch";
inline constexpr const char* kElse = "else_branch";
inline constexpr const char* kBody = "body";
inline constexpr const char* kCaptures = "captures";
inline constexpr const char* kBeg = "beg";
inline constexpr const char* kEnd = "end";
inline constexpr const char* kUnresolved = "unresolved";
}  // namespace attr

// "arg<position>": the attribute a folded constant operand is stored under.
std::string arg_attr_name(uint32_t position);

// A pickled TorchScript object with dict state: Opaque, module starting with
// "__torch__", and a recorded BUILD state (Value::inner) that is a Dict whose
// pairs are the attributes in class slot order.
bool is_torch_object(const Value& v);

// The deterministic traversal of a data.pkl value tree that both the parameter
// table (PytorchParser collect_tensors) and ModuleEnv's canonical tensor names
// use: dict and object-state insertion order, List/Tuple index order, keys
// joined with '.', containers and objects visited at most once (cycles and
// shared subtrees via the memo), depth <= kMaxTraversalDepth. `fn` is called for
// every tensor leaf with its path; a tensor reachable from two containers is
// reported at each path.
inline constexpr int kMaxTraversalDepth = 256;
void for_each_tensor(
    const ValuePtr& root,
    const std::function<void(const std::string& path, const Value& tensor)>& fn);

struct ConstLimits {
  uint32_t max_list = 1024;  // longer lists render as "[… N items]"
  uint32_t max_repr = 256;   // repr bytes before truncation with "…"
  uint32_t max_depth = 16;   // nested containers deeper than this render "..."
};

// Non-tensor pickled constant -> AttrValue (spec #137 §4.6.4). Returns false
// (and leaves `out` untouched) for Kind::Tensor; true otherwise.
bool constant_to_attr(const Value& v, ir::Model& m, ir::AttrValue& out,
                      const ConstLimits& lim = {});
// Python-style repr of a pickled value, bounded by `lim` (cycles render "...").
std::string constant_repr(const Value& v, const ConstLimits& lim = {});

// Binds module attributes against the data.pkl object tree.
class ModuleEnv {
 public:
  ModuleEnv() = default;                              // no data.pkl: nothing binds
  explicit ModuleEnv(const ValuePtr& data_pkl_root);  // keeps the tree alive

  // nullptr unless the data.pkl root is a TorchScript object (is_torch_object).
  const Value* root() const { return root_; }
  // "<module>.<name>" of the root object (e.g. "__torch__.Model"), or "".
  const std::string& root_class() const { return root_class_; }
  // True when constructed from a data.pkl root, even one that is not a
  // TorchScript object (distinguishes "no data.pkl" in diagnostics).
  bool loaded() const { return loaded_; }

  struct Attr {
    enum class Kind : uint8_t { Unknown, Tensor, Object, Constant, Opaque };
    Kind kind = Kind::Unknown;
    const Value* value = nullptr;
    std::string name;  // the attribute's name; empty for Unknown
  };
  // Attribute `slot` of TorchScript object `obj` (bytecode GET_ATTR X).
  Attr by_slot(const Value* obj, int64_t slot) const;
  // Attribute `name` of `obj` (the #136 entry point: self.conv1.weight).
  Attr by_name(const Value* obj, std::string_view name) const;

  // Canonical name of a module tensor: the first path at which this tensor
  // Value is reached by for_each_tensor from the root (tied weights get ONE
  // name). `fallback` when the tensor is not reachable.
  std::string tensor_path(const Value* tensor, std::string_view fallback) const;
  // Number of distinct tensor Values reachable from the root.
  size_t module_tensor_count() const { return paths_.size(); }

 private:
  ValuePtr keep_;
  const Value* root_ = nullptr;
  std::string root_class_;
  bool loaded_ = false;
  std::unordered_map<const Value*, std::string> paths_;  // lookup only
};

// Names bytecode tensor constants after the constants.pkl entry they share a
// record with ("CONSTANTS.c<i>", the identifier code/*.py uses).
class TensorConstNames {
 public:
  // Tolerant: a non-tuple (or null) value adds no names. Tensors without a
  // located record (file_offset == UINT64_MAX) are never matched.
  void add_from_constants_pkl(const ValuePtr& tuple);
  // "CONSTANTS.c<i>" for the lowest i whose tensor has an equal
  // (file_offset, byte_len, dtype, shape); otherwise `fallback`.
  std::string name_for(const ir::TensorRef& t, std::string_view fallback) const;

 private:
  struct Ent {
    ir::TensorRef ref;
    size_t index;
  };
  std::unordered_map<uint64_t, std::vector<Ent>> by_offset_;  // lookup only
};

// "%1", "%2", ... One per method, shared by the method's graph and subgraphs.
class NameCounter {
 public:
  std::string next();

 private:
  uint32_t n_ = 0;
};

// Builds one ir::Graph incrementally. Nodes are STAGED (inputs, outputs and
// attributes may be added in any order, e.g. after a branch is finalized) and
// finish() lays them out as contiguous Ranges, in node creation order.
class GraphBuilder {
 public:
  GraphBuilder(ir::Model& m, NameCounter& names, std::string_view graph_name,
               uint32_t id, GraphBuilder* parent);

  uint32_t id() const { return id_; }
  GraphBuilder* parent() const { return parent_; }
  const std::string& name() const { return name_; }

  // New graph input (appended to graph_inputs). Empty name -> NameCounter.
  uint32_t add_input(std::string_view name);
  // Initializer value (spec §4.6.3): an existing initializer with this name and
  // the same (file_offset, byte_len) is reused; any other clash takes the first
  // free "#2", "#3", ... suffix. dtype/shape come from `t`.
  uint32_t initializer(std::string_view name, const ir::TensorRef& t);
  // Graph input standing for parent value `parent_value_idx` (cached per parent
  // value), named `name`.
  uint32_t capture(uint32_t parent_value_idx, std::string_view name);
  // Parent value indices captured so far, in capture order.
  const std::vector<uint32_t>& captured_parent_values() const { return captured_; }

  uint32_t node(std::string_view op_type);  // staged node id
  void add_edge(uint32_t node, uint32_t value);
  void add_attr(uint32_t node, ir::Attribute a);
  // New output value of `node`. Empty name -> NameCounter.
  uint32_t add_output(uint32_t node, std::string_view name = {});
  void set_subgraph(uint32_t node, int32_t graph_index);
  void add_output_value(uint32_t value);  // graph_outputs
  // Move `value` (already a graph input) to the front of graph_inputs; used for
  // the lazily-added "self".
  void prepend_input(uint32_t value);

  std::string_view value_name(uint32_t value) const;
  size_t node_count() const { return nodes_.size(); }

  // Nodes in creation order; edge_refs = per node inputs then outputs.
  ir::Graph finish();

 private:
  struct Staged {
    StringId op_type;
    std::vector<uint32_t> in, out;
    std::vector<ir::Attribute> attrs;
    int32_t subgraph = -1;
  };
  std::string unique_name(std::string_view base);
  uint32_t new_value(std::string_view name);  // unique-ified, producer -1

  ir::Model& m_;
  NameCounter& names_;
  std::string name_;
  uint32_t id_;
  GraphBuilder* parent_;

  std::vector<Staged> nodes_;
  std::vector<ir::ValueInfo> values_;
  std::vector<ir::TensorRef> inits_;
  std::vector<uint32_t> inputs_, outputs_;
  std::vector<uint32_t> captured_;
  // Lookup-only maps; no emitted order ever comes from iterating them.
  std::unordered_map<std::string, uint32_t> by_name_;      // name -> value
  std::unordered_map<std::string, uint32_t> next_suffix_;  // base -> next #k
  std::unordered_map<uint32_t, uint32_t> init_of_value_;   // value -> init idx
  std::unordered_map<uint32_t, uint32_t> capture_of_;      // parent value -> value
};

}  // namespace netvis::pytorch::ts
