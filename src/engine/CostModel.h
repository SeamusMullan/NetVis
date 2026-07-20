// engine/CostModel.h — static compute/memory cost analysis (v0.3.0 analyzer).
//
// DECISION (v0.3.0): NetVis moves from "see the model" to "understand its cost".
// compute_cost is a PURE headless function over an already-parsed ir::Model — it
// reads ONLY structure (resolved value shapes/dtypes + initializer metadata) and
// NEVER a tensor payload, so it upholds the zero-payload-read thesis (spec §2.1)
// and runs on a worker thread / in tests without a GUI, exactly like infer_shapes
// and GraphAdjacency. It is an ESTIMATE, best-effort: unsupported ops or
// unresolved shapes are reported honestly as flops_known=false and excluded from
// the total (never faked), mirroring shape inference's best-effort contract.
//
// Depends only on ir/IR.h. No engine/view coupling; collapse-group rollup is done
// by the caller via CollapseGroup::member_nodes (see view/CostPanel.cpp).
#pragma once

#include <cstdint>
#include <vector>

#include "ir/IR.h"

namespace netvis {

// Per-IR-node cost. All fields are 0 for a node whose cost could not be derived.
struct NodeCost {
  // Floating-point operations. CONVENTION: a multiply-accumulate counts as 2
  // FLOPs (one multiply + one add); pure-elementwise ops count 1 FLOP/output
  // element (macs = 0). 0 when !flops_known. See the formula table below.
  uint64_t flops = 0;
  // Weight/initializer ELEMENTS feeding this node (sum of elem_count over the
  // node's inputs that name a graph initializer). Not affected by dtype.
  uint64_t params = 0;
  // Bytes of those weights: initializer.byte_len when recorded (>0, the truth for
  // quantized blocks), else elem_count * dtype_size(dtype).
  uint64_t weight_bytes = 0;
  // Output activation bytes: sum over the node's OUTPUT values of
  // elem_count * dtype_size(dtype). dtype_size==0 (quant/unknown) contributes 0.
  uint64_t act_bytes = 0;
  // false => the op is unsupported OR a shape it needs is unresolved. Such a node
  // contributes 0 to total_flops and increments (nodes_total - nodes_flops_known).
  bool flops_known = false;
};

// Aggregate weight usage for one dtype (quant-coverage report).
struct DTypeUsage {
  ir::DType dtype = ir::DType::Unknown;
  uint64_t params = 0;   // element count across initializers/flat_tensors of dtype
  uint64_t bytes = 0;    // byte count across the same
};

// The whole-graph cost report. Cheap to compute (O(V + E) arithmetic).
struct CostReport {
  // Indexed 1:1 by IR node index into graphs[graph_index].nodes. EMPTY in
  // table-mode (from_graph == false: model has no compute graph).
  std::vector<NodeCost> per_node;

  uint64_t total_flops = 0;          // sum of per_node.flops over flops_known nodes
  // total_params/total_weight_bytes aggregate over ALL initializers (graph mode)
  // or flat_tensors (table mode), NOT sum(per_node.*): a weight shared by K
  // consumers counts once and an unconsumed initializer still counts. Matches
  // dtype_usage. (per_node.params is the per-node attribution, which double-counts
  // shared weights by design — that is the node view, this is the model view.)
  uint64_t total_params = 0;
  uint64_t total_weight_bytes = 0;
  uint64_t peak_activation_bytes = 0;// see peak-liveness algorithm below

  uint32_t nodes_total = 0;          // == per_node.size()
  uint32_t nodes_flops_known = 0;    // (total - this) == nodes with unknown FLOPs

  // Per-dtype weight usage, SORTED by bytes descending (ties: dtype enum order).
  // In graph mode this covers graphs[graph_index].initializers; in table mode it
  // covers model.flat_tensors.
  std::vector<DTypeUsage> dtype_usage;

  bool from_graph = true;  // false => report built from model.flat_tensors

  // Quant-coverage derived metrics (safe when total_params == 0 -> return 0).
  double effective_bits_per_param() const {
    if (total_params == 0) return 0.0;
    return static_cast<double>(total_weight_bytes) * 8.0 /
           static_cast<double>(total_params);
  }
  // Size relative to storing every param as fp32 (4 bytes). <1 => compressed.
  double size_vs_fp32() const {
    if (total_params == 0) return 0.0;
    return static_cast<double>(total_weight_bytes) /
           (static_cast<double>(total_params) * 4.0);
  }
};

// Build the cost report for graphs[graph_index]. Out-of-range graph_index, or a
// model with has_graph == false, yields a table-mode report (from_graph == false,
// per_node empty) built from model.flat_tensors. Deterministic; never reads a
// tensor payload; never throws.
//
// ---------------------------------------------------------------------------
//  FLOP FORMULA TABLE (best-effort; |O| = product of a node output's dims).
//  Shapes come from ValueInfo (post shape-inference); a needed shape that is
//  empty / contains a dim < 1 makes the node flops_known = false.
// ---------------------------------------------------------------------------
//  MatMul                : macs = |O| * K, K = last dim of input[0]'s shape.
//                          flops = 2 * macs.
//  Gemm                  : macs = M * N * K from the 2D operands (respect the
//                          output shape for M,N; K = shared dim); flops = 2*macs.
//  Conv / ConvTranspose  : macs = |O| * (Cin / group) * prod(kernel_spatial),
//                          where Cin and kernel_spatial come from the WEIGHT
//                          initializer shape input[1] = [Cout, Cin/g, k...], and
//                          group from the "group" attribute (default 1).
//                          flops = 2 * macs.
//  Pool (Max/Average/...) : flops = |O| * prod(kernel_spatial) (kernel from the
//                          "kernel_shape" attribute; if absent -> flops_known=false).
//  Norm  (Batch/Layer/Group/RMS/Instance) : flops = |O| (elementwise-ish).
//  Activation (Relu/Gelu/Sigmoid/Tanh/Softmax/Elu/...) : flops = |O|.
//  Elementwise (Add/Mul/Sub/Div/Pow/Sqrt/...) : flops = |O|.
//  Reduce (ReduceSum/Mean/...)               : flops = |input[0]| (elems reduced).
//  Everything else (Shape/Cast/Reshape/Concat/Gather/control-flow/unknown)
//                          : flops_known = false (structural, ~0 arithmetic).
//  Category comes from engine/OpCategory.h (categorize_op); the exact op string
//  is used only to disambiguate Conv-vs-ConvTranspose and Reduce.
//
// ---------------------------------------------------------------------------
//  PEAK ACTIVATION LIVENESS (peak_activation_bytes). ONNX node order is
//  topological (spec-guaranteed); the pass assumes it.
// ---------------------------------------------------------------------------
//  Consider only ACTIVATION values (a value whose name is NOT a graph
//  initializer). bytes(v) = elem_count(v.shape) * dtype_size(v.dtype).
//   1. live := bytes of all graph_inputs.  peak := live.
//   2. For node i in index order:
//        live += bytes of each OUTPUT activation value first produced here.
//        peak  = max(peak, live).
//        For each value v with last_use(v) == i AND v not in graph_outputs:
//            live -= bytes(v).
//  where last_use(v) = the largest consumer node index of v (nodes whose inputs
//  reference v), or -1 if never consumed. Values never freed if a graph output.
//  A value with an unresolved shape contributes 0 bytes (not an error).
// ---------------------------------------------------------------------------
CostReport compute_cost(const ir::Model& model, uint32_t graph_index);

}  // namespace netvis
