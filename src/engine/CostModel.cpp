// engine/CostModel.cpp — static compute/memory cost analysis (v0.3.0 analyzer).
//
// DECISION (v0.3.0): compute_cost is a pure structure-reader upholding the
// zero-payload-read thesis (spec §2.1). It computes FLOPs per the formula table in
// CostModel.h (2 FLOPs per MAC, elementwise = |O|, etc.), aggregates params/weight
// bytes from initializers by matching value names, computes peak activation
// liveness via the topological pass specified in the header, and builds a
// per-dtype usage report sorted by bytes desc. All arithmetic is checked for
// overflow (saturates at UINT64_MAX); all indices are bounds-checked (hostile
// input).
//
// RATIONALE: The analyzer mode needs to know "what is expensive" in a model. FLOPs
// estimate compute (inference GFLOP/s * batch size), params quantify storage
// (weight MB), and peak activation liveness estimates GPU/accelerator memory
// pressure. The formula table is best-effort — unsupported ops or unresolved
// shapes yield flops_known=false and are excluded from the total, never faked.
// This implementation follows the header's exact table and liveness algorithm so
// the UI and tests all see one coherent contract.

#include "engine/CostModel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "engine/OpCategory.h"
#include "ir/IR.h"

namespace netvis {
namespace {

// Hash functor for StringId (needed for unordered_map keyed by StringId).
struct StringIdHash {
  std::size_t operator()(StringId sid) const noexcept {
    return std::hash<uint32_t>{}(sid.id);
  }
};

// Saturating uint64_t arithmetic (never overflow, clamp at UINT64_MAX).
uint64_t safe_add(uint64_t a, uint64_t b) {
  if (UINT64_MAX - a < b) return UINT64_MAX;
  return a + b;
}

uint64_t safe_mul(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0) return 0;
  if (UINT64_MAX / a < b) return UINT64_MAX;
  return a * b;
}

// Product of shape dims; returns 0 if any dim < 1 (dynamic/unresolved).
uint64_t elem_count_from_shape(const SmallVec<int64_t, 6>& shape) {
  if (shape.empty()) return 0;
  uint64_t n = 1;
  for (int64_t d : shape) {
    if (d < 1) return 0;  // dynamic or invalid => unresolved
    n = safe_mul(n, static_cast<uint64_t>(d));
  }
  return n;
}

// Product of selected dims [offset, offset+count); returns 0 if any dim < 1.
uint64_t partial_shape_product(const SmallVec<int64_t, 6>& shape,
                                uint32_t offset, uint32_t count) {
  if (offset + count > shape.size()) return 0;
  uint64_t n = 1;
  for (uint32_t i = 0; i < count; ++i) {
    int64_t d = shape[offset + i];
    if (d < 1) return 0;
    n = safe_mul(n, static_cast<uint64_t>(d));
  }
  return n;
}

// Compute value bytes = elem_count(shape) * dtype_size(dtype). Returns 0 if shape
// is unresolved OR dtype_size is 0 (quantized/unknown).
uint64_t value_bytes(const ir::ValueInfo& vi) {
  uint32_t dsize = ir::dtype_size(vi.dtype);
  if (dsize == 0) return 0;  // quantized/unknown => no byte contribution
  uint64_t elems = elem_count_from_shape(vi.shape);
  if (elems == 0) return 0;  // unresolved shape
  return safe_mul(elems, dsize);
}

// Compute initializer bytes: use byte_len when recorded (>0, truth for Q4/Q8),
// else elem_count * dtype_size.
uint64_t initializer_bytes(const ir::TensorRef& init) {
  if (init.byte_len > 0) return init.byte_len;
  uint32_t dsize = ir::dtype_size(init.dtype);
  if (dsize == 0) return 0;  // quantized without byte_len => no fallback
  int64_t ec = init.elem_count();
  if (ec <= 0) return 0;
  return safe_mul(static_cast<uint64_t>(ec), dsize);
}

// Build a name (StringId) -> initializer index lookup for quick "is this value a
// weight?" checks. Returns map<StringId, uint32_t> where value is initializer idx.
std::unordered_map<StringId, uint32_t, StringIdHash> build_initializer_index(
    const ir::Graph& g) {
  std::unordered_map<StringId, uint32_t, StringIdHash> idx;
  idx.reserve(g.initializers.size());
  for (uint32_t i = 0; i < g.initializers.size(); ++i) {
    idx[g.initializers[i].name] = i;
  }
  return idx;
}

// Read an int64_t attribute (e.g. "group", "transA"); returns default_val if the
// attribute is absent or not an Int.
int64_t get_int_attr(const ir::Model& model, const ir::Graph& g,
                     const ir::Node& node, std::string_view attr_name,
                     int64_t default_val) {
  for (uint32_t i = 0; i < node.attributes.count; ++i) {
    uint32_t aidx = node.attributes.begin + i;
    if (aidx >= g.attributes.size()) continue;
    const ir::Attribute& attr = g.attributes[aidx];
    if (model.str(attr.name) == attr_name) {
      if (attr.value.kind == ir::AttrValue::Kind::Int) return attr.value.i;
      return default_val;
    }
  }
  return default_val;
}

// Read an ints attribute (e.g. "kernel_shape"); returns empty if not found/wrong.
std::vector<int64_t> get_ints_attr(const ir::Model& model, const ir::Graph& g,
                                    const ir::Node& node,
                                    std::string_view attr_name) {
  for (uint32_t i = 0; i < node.attributes.count; ++i) {
    uint32_t aidx = node.attributes.begin + i;
    if (aidx >= g.attributes.size()) continue;
    const ir::Attribute& attr = g.attributes[aidx];
    if (model.str(attr.name) == attr_name) {
      if (attr.value.kind == ir::AttrValue::Kind::Ints) {
        return attr.value.ints;
      }
      return {};
    }
  }
  return {};
}

// True if every declared input slot of `node` resolves to an in-range value. A
// node with a corrupt/out-of-range input edge_ref is structurally invalid, so we
// refuse to estimate its FLOPs (flops_known stays false) rather than guessing
// from whatever partial shapes happen to resolve — hostile input stays honest.
bool all_inputs_resolve(const ir::Graph& g, const ir::Node& node) {
  for (uint32_t slot = 0; slot < node.inputs.count; ++slot) {
    uint32_t er = node.inputs.begin + slot;
    if (er >= g.edge_refs.size()) return false;
    if (g.edge_refs[er] >= g.values.size()) return false;
  }
  return true;
}

// Get the ValueInfo for node input slot k. Returns nullptr if out of range.
const ir::ValueInfo* get_input_value(const ir::Graph& g, const ir::Node& node,
                                      uint32_t slot) {
  if (slot >= node.inputs.count) return nullptr;
  uint32_t er = node.inputs.begin + slot;
  if (er >= g.edge_refs.size()) return nullptr;
  uint32_t vidx = g.edge_refs[er];
  if (vidx >= g.values.size()) return nullptr;
  return &g.values[vidx];
}

// Get the ValueInfo for node output slot k. Returns nullptr if out of range.
const ir::ValueInfo* get_output_value(const ir::Graph& g, const ir::Node& node,
                                       uint32_t slot) {
  if (slot >= node.outputs.count) return nullptr;
  uint32_t er = node.outputs.begin + slot;
  if (er >= g.edge_refs.size()) return nullptr;
  uint32_t vidx = g.edge_refs[er];
  if (vidx >= g.values.size()) return nullptr;
  return &g.values[vidx];
}

// Compute node FLOPs via the formula table in CostModel.h. Sets nc.flops and
// nc.flops_known. Never reads payload bytes.
void compute_flops(const ir::Model& model, const ir::Graph& g,
                   const ir::Node& node, NodeCost& nc) {
  std::string_view op = model.str(node.op_type);
  OpCategory cat = categorize_op(op);

  // Structurally invalid node (corrupt input edge_ref) => refuse to estimate.
  if (!all_inputs_resolve(g, node)) return;

  // MatMul: macs = |O| * K, K = last dim of input[0]. Einsum shares the MatMul
  // category (for coloring) but its FLOPs depend on the contraction pattern in
  // its "equation" attribute, not on |O|*K — routing it here would fabricate a
  // number, so leave it flops_known=false (honest unknown) rather than guess.
  if (cat == OpCategory::MatMul && op != "Gemm" && op != "Einsum") {
    const ir::ValueInfo* out = get_output_value(g, node, 0);
    const ir::ValueInfo* in0 = get_input_value(g, node, 0);
    if (!out || !in0) return;  // missing value => unknown
    uint64_t out_elems = elem_count_from_shape(out->shape);
    if (out_elems == 0 || in0->shape.empty()) return;
    int64_t K = in0->shape.back();
    if (K < 1) return;  // unresolved
    uint64_t macs = safe_mul(out_elems, static_cast<uint64_t>(K));
    nc.flops = safe_mul(macs, 2);
    nc.flops_known = true;
    return;
  }

  // Gemm: macs = M * N * K. Output is M × N (= out_elems); K = the shared
  // contraction dim of operand A. Shape inference only writes the OUTPUT shape,
  // NOT a transposed view of the input operands, so in0->shape is A's declared
  // shape: [M, K] when transA=0 (K = last dim) but [K, M] when transA=1 (K =
  // first dim). Reading transA is required — using back() unconditionally
  // computes M*N*M instead of M*N*K whenever transA=1 and M != K.
  if (op == "Gemm") {
    const ir::ValueInfo* out = get_output_value(g, node, 0);
    const ir::ValueInfo* in0 = get_input_value(g, node, 0);
    const ir::ValueInfo* in1 = get_input_value(g, node, 1);
    if (!out || !in0 || !in1) return;
    uint64_t out_elems = elem_count_from_shape(out->shape);
    if (out_elems == 0) return;
    if (in0->shape.size() < 2) return;  // Gemm operands are 2D
    int64_t transA = get_int_attr(model, g, node, "transA", 0);
    int64_t K = (transA != 0) ? in0->shape[0] : in0->shape.back();
    if (K < 1) return;
    uint64_t macs = safe_mul(out_elems, static_cast<uint64_t>(K));
    nc.flops = safe_mul(macs, 2);
    nc.flops_known = true;
    return;
  }

  // Conv: macs = |O| * (Cin/group) * prod(kernel). Weight = [Cout, Cin/g, k...],
  // so Cin/g = weight.shape[1], kernel = weight.shape[2..].
  //
  // ConvTranspose has the TRANSPOSED weight layout [Cin, Cout/g, k...], and its
  // output spatial size differs from the input's, so |O| is the wrong base. The
  // clean MAC count is over the INPUT: each input element scatter-accumulates
  // into prod(kernel) * (Cout/g) outputs, i.e. macs = |input| * (Cout/g) *
  // prod(kernel), where Cout/g = weight.shape[1]. Using the Conv formula here
  // would read shape[1] as Cin/g (it is actually Cout/g) and base on |O| — both
  // wrong. Both ops fall under OpCategory::Conv, so disambiguate on the op string.
  if (cat == OpCategory::Conv) {
    const bool transpose = (op == "ConvTranspose");
    const ir::ValueInfo* weight = get_input_value(g, node, 1);
    if (!weight || weight->shape.size() < 3) return;  // malformed/unresolved
    // Channels-per-group factor sits at weight.shape[1] for BOTH layouts:
    // Conv -> Cin/g, ConvTranspose -> Cout/g. Kernel is weight.shape[2..] for both.
    int64_t chan_per_g = weight->shape[1];
    if (chan_per_g < 1) return;  // unresolved
    uint64_t kernel_prod = partial_shape_product(
        weight->shape, 2, static_cast<uint32_t>(weight->shape.size() - 2));
    if (kernel_prod == 0) return;  // unresolved kernel dims
    // Base: output elems for Conv, input elems for ConvTranspose.
    const ir::ValueInfo* base = transpose ? get_input_value(g, node, 0)
                                          : get_output_value(g, node, 0);
    if (!base) return;
    uint64_t base_elems = elem_count_from_shape(base->shape);
    if (base_elems == 0) return;
    uint64_t macs = safe_mul(base_elems, static_cast<uint64_t>(chan_per_g));
    macs = safe_mul(macs, kernel_prod);
    nc.flops = safe_mul(macs, 2);
    nc.flops_known = true;
    return;
  }

  // Pool: flops = |O| * prod(kernel_spatial) from "kernel_shape" attribute.
  if (cat == OpCategory::Pool) {
    const ir::ValueInfo* out = get_output_value(g, node, 0);
    if (!out) return;
    uint64_t out_elems = elem_count_from_shape(out->shape);
    if (out_elems == 0) return;
    std::vector<int64_t> kernel = get_ints_attr(model, g, node, "kernel_shape");
    if (kernel.empty()) return;  // no kernel => unknown
    uint64_t kernel_prod = 1;
    for (int64_t k : kernel) {
      if (k < 1) return;  // unresolved
      kernel_prod = safe_mul(kernel_prod, static_cast<uint64_t>(k));
    }
    nc.flops = safe_mul(out_elems, kernel_prod);
    nc.flops_known = true;
    return;
  }

  // Norm / Activation / Elementwise: flops = |O| (output element count).
  if (cat == OpCategory::Norm || cat == OpCategory::Activation ||
      cat == OpCategory::Elementwise) {
    const ir::ValueInfo* out = get_output_value(g, node, 0);
    if (!out) return;
    uint64_t elems = elem_count_from_shape(out->shape);
    if (elems == 0) return;  // unresolved
    nc.flops = elems;
    nc.flops_known = true;
    return;
  }

  // Reduce (ReduceSum/Mean/...): flops = |input[0]| (elements reduced).
  if (cat == OpCategory::Reduce) {
    const ir::ValueInfo* in0 = get_input_value(g, node, 0);
    if (!in0) return;
    uint64_t elems = elem_count_from_shape(in0->shape);
    if (elems == 0) return;  // unresolved
    nc.flops = elems;
    nc.flops_known = true;
    return;
  }

  // Everything else (Shape/Tensor/ControlFlow/Other): flops_known = false.
  // Structural ops have ~0 arithmetic; report them honestly as unknown.
}

// Compute params/weight_bytes for a node by summing over its inputs that name
// initializers. Sets nc.params and nc.weight_bytes.
void compute_params(const ir::Graph& g, const ir::Node& node,
                    const std::unordered_map<StringId, uint32_t, StringIdHash>& init_idx,
                    NodeCost& nc) {
  for (uint32_t slot = 0; slot < node.inputs.count; ++slot) {
    const ir::ValueInfo* vi = get_input_value(g, node, slot);
    if (!vi) continue;
    auto it = init_idx.find(vi->name);
    if (it == init_idx.end()) continue;  // not an initializer
    uint32_t idx = it->second;
    if (idx >= g.initializers.size()) continue;  // guard
    const ir::TensorRef& init = g.initializers[idx];
    int64_t ec = init.elem_count();
    if (ec > 0) {
      nc.params = safe_add(nc.params, static_cast<uint64_t>(ec));
    }
    nc.weight_bytes = safe_add(nc.weight_bytes, initializer_bytes(init));
  }
}

// Compute act_bytes = sum over node outputs of elem_count * dtype_size.
void compute_act_bytes(const ir::Graph& g, const ir::Node& node, NodeCost& nc) {
  for (uint32_t slot = 0; slot < node.outputs.count; ++slot) {
    const ir::ValueInfo* vi = get_output_value(g, node, slot);
    if (!vi) continue;
    nc.act_bytes = safe_add(nc.act_bytes, value_bytes(*vi));
  }
}

// Compute per-node costs for all nodes in a graph. Returns per_node vector,
// nodes_flops_known count, and totals for flops/params/weight_bytes.
struct GraphCostSummary {
  std::vector<NodeCost> per_node;
  uint64_t total_flops = 0;
  uint32_t nodes_flops_known = 0;
  // NOTE: no total_params/total_weight_bytes here — those are aggregated over
  // initializers in compute_cost (summing per_node would double-count shared
  // weights). This struct is only the per-node view + the FLOP total.
};

GraphCostSummary compute_graph_costs(const ir::Model& model, const ir::Graph& g) {
  GraphCostSummary summary;
  summary.per_node.resize(g.nodes.size());

  auto init_idx = build_initializer_index(g);

  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    const ir::Node& node = g.nodes[i];
    NodeCost& nc = summary.per_node[i];

    compute_flops(model, g, node, nc);
    compute_params(g, node, init_idx, nc);
    compute_act_bytes(g, node, nc);

    if (nc.flops_known) {
      summary.total_flops = safe_add(summary.total_flops, nc.flops);
      summary.nodes_flops_known++;
    }
  }

  return summary;
}

// Compute peak activation liveness via the topological pass in CostModel.h.
// ONNX node order is topological; the pass is single-pass index order.
uint64_t compute_peak_activation_bytes(const ir::Graph& g) {
  auto init_idx = build_initializer_index(g);

  // Mark which values are activations (not initializers).
  std::vector<bool> is_activation(g.values.size(), false);
  for (uint32_t i = 0; i < g.values.size(); ++i) {
    is_activation[i] = (init_idx.find(g.values[i].name) == init_idx.end());
  }

  // Mark which values are graph outputs (never freed).
  std::unordered_set<uint32_t> output_set;
  output_set.reserve(g.graph_outputs.size());
  for (uint32_t vidx : g.graph_outputs) {
    if (vidx < g.values.size()) output_set.insert(vidx);
  }

  // Compute last_use for each value = largest consumer node index, or -1.
  std::vector<int32_t> last_use(g.values.size(), -1);
  for (uint32_t ni = 0; ni < g.nodes.size(); ++ni) {
    const ir::Node& node = g.nodes[ni];
    for (uint32_t slot = 0; slot < node.inputs.count; ++slot) {
      uint32_t er = node.inputs.begin + slot;
      if (er >= g.edge_refs.size()) continue;
      uint32_t vidx = g.edge_refs[er];
      if (vidx >= g.values.size()) continue;
      last_use[vidx] = static_cast<int32_t>(ni);
    }
  }

  // Bucket each freeable value under the node index that frees it, so the pass
  // below iterates only expiring values (O(V+E) total) instead of rescanning all
  // values per node (which would be O(V^2) — see the "cheap to compute" contract
  // in CostModel.h). A value is freeable when it has a consumer and is neither a
  // graph output nor an initializer.
  std::vector<std::vector<uint32_t>> free_at(g.nodes.size());
  for (uint32_t vidx = 0; vidx < g.values.size(); ++vidx) {
    int32_t lu = last_use[vidx];
    if (lu < 0) continue;
    if (output_set.find(vidx) != output_set.end()) continue;
    if (!is_activation[vidx]) continue;
    free_at[static_cast<uint32_t>(lu)].push_back(vidx);
  }

  // Each activation value contributes its bytes to `live` exactly ONCE. `free_at`
  // removes it exactly once, so adding per output SLOT (or re-adding a graph input
  // later emitted as a node output) would break the add/free symmetry and
  // permanently inflate the reported peak on malformed input. `produced` gates
  // every add site — including the graph_inputs seed below.
  std::vector<bool> produced(g.values.size(), false);

  // Initialize live with graph_inputs bytes.
  uint64_t live = 0;
  for (uint32_t vidx : g.graph_inputs) {
    if (vidx >= g.values.size()) continue;
    if (!is_activation[vidx]) continue;  // shouldn't happen, but guard
    if (produced[vidx]) continue;        // duplicate graph_input entry
    produced[vidx] = true;
    live = safe_add(live, value_bytes(g.values[vidx]));
  }
  uint64_t peak = live;

  // Topological pass: for each node in index order, add outputs, update peak,
  // free values whose last_use == this node (unless graph output).
  for (uint32_t ni = 0; ni < g.nodes.size(); ++ni) {
    const ir::Node& node = g.nodes[ni];

    // Add output activation bytes first produced here.
    for (uint32_t slot = 0; slot < node.outputs.count; ++slot) {
      const ir::ValueInfo* vi = get_output_value(g, node, slot);
      if (!vi) continue;
      uint32_t er = node.outputs.begin + slot;
      if (er >= g.edge_refs.size()) continue;
      uint32_t vidx = g.edge_refs[er];
      if (vidx >= g.values.size()) continue;
      if (!is_activation[vidx]) continue;  // shouldn't happen
      if (produced[vidx]) continue;        // already counted (duplicate slot)
      produced[vidx] = true;
      live = safe_add(live, value_bytes(*vi));
    }

    // Update peak after adding outputs.
    if (live > peak) peak = live;

    // Free values whose last_use is this node (unless graph output). free_at[ni]
    // already excludes outputs/initializers/never-consumed values.
    for (uint32_t vidx : free_at[ni]) {
      uint64_t vb = value_bytes(g.values[vidx]);
      if (vb <= live) {
        live -= vb;
      } else {
        live = 0;  // guard underflow (shouldn't happen with correct liveness)
      }
    }
  }

  return peak;
}

// Aggregate dtype_usage from a list of TensorRefs (either initializers or
// flat_tensors). Returns sorted by bytes desc (ties: dtype enum order).
std::vector<DTypeUsage> compute_dtype_usage(
    const std::vector<ir::TensorRef>& tensors) {
  std::unordered_map<ir::DType, DTypeUsage> usage_map;
  for (const ir::TensorRef& t : tensors) {
    ir::DType dt = t.dtype;
    int64_t ec = t.elem_count();
    uint64_t params = (ec > 0) ? static_cast<uint64_t>(ec) : 0;
    uint64_t bytes = initializer_bytes(t);

    auto& u = usage_map[dt];
    u.dtype = dt;
    u.params = safe_add(u.params, params);
    u.bytes = safe_add(u.bytes, bytes);
  }

  std::vector<DTypeUsage> result;
  result.reserve(usage_map.size());
  for (auto& [dt, u] : usage_map) {
    result.push_back(u);
  }

  // Sort by bytes desc, ties by dtype enum order.
  std::sort(result.begin(), result.end(), [](const DTypeUsage& a,
                                              const DTypeUsage& b) {
    if (a.bytes != b.bytes) return a.bytes > b.bytes;
    return static_cast<uint8_t>(a.dtype) < static_cast<uint8_t>(b.dtype);
  });

  return result;
}

// Build a table-mode report from model.flat_tensors (no compute graph).
CostReport build_table_report(const ir::Model& model) {
  CostReport report;
  report.from_graph = false;
  report.dtype_usage = compute_dtype_usage(model.flat_tensors);

  // Aggregate total params and weight bytes from flat_tensors.
  for (const ir::TensorRef& t : model.flat_tensors) {
    int64_t ec = t.elem_count();
    if (ec > 0) {
      report.total_params = safe_add(report.total_params,
                                      static_cast<uint64_t>(ec));
    }
    report.total_weight_bytes =
        safe_add(report.total_weight_bytes, initializer_bytes(t));
  }

  return report;
}

}  // namespace

// Public API: build the cost report for graphs[graph_index].
CostReport compute_cost(const ir::Model& model, uint32_t graph_index) {
  // Table mode: has_graph == false or graph_index out of range.
  if (!model.has_graph || graph_index >= model.graphs.size()) {
    return build_table_report(model);
  }

  const ir::Graph& g = model.graphs[graph_index];
  CostReport report;
  report.from_graph = true;

  // Compute per-node costs and totals.
  GraphCostSummary summary = compute_graph_costs(model, g);
  report.per_node = std::move(summary.per_node);
  report.total_flops = summary.total_flops;
  report.nodes_total = static_cast<uint32_t>(g.nodes.size());
  report.nodes_flops_known = summary.nodes_flops_known;

  // total_params/total_weight_bytes are aggregated over ALL initializers, not by
  // summing per_node.params: a weight shared by K consumers must be counted once,
  // and an initializer consumed by no node must still count. (per_node.params is
  // the per-node view; this is the model view. Consistent with dtype_usage.)
  for (const ir::TensorRef& t : g.initializers) {
    int64_t ec = t.elem_count();
    if (ec > 0)
      report.total_params =
          safe_add(report.total_params, static_cast<uint64_t>(ec));
    report.total_weight_bytes =
        safe_add(report.total_weight_bytes, initializer_bytes(t));
  }

  // Compute peak activation liveness.
  report.peak_activation_bytes = compute_peak_activation_bytes(g);

  // Compute dtype usage from initializers.
  report.dtype_usage = compute_dtype_usage(g.initializers);

  return report;
}

}  // namespace netvis
