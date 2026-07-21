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

#include "engine/HeatmapGradient.h"  // HeatmapMetric (engine-leaf, only <cstdint>)
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

  // --- v0.4.0: efficiency / arithmetic-intensity ----------------------------
  // Bytes of this node's INPUT values that are activations (NOT graph
  // initializers): sum of elem_count*dtype_size over input slots whose value name
  // is not an initializer. 0 for dtype_size==0 (quant/unknown) or unresolved
  // shape (same 0-contribution convention as act_bytes). The missing third term
  // of bytes_moved(): weight_bytes covers weight reads, act_bytes covers output
  // writes, this covers activation reads. Per-node attribution (an activation read
  // by K consumers is counted K times) — do NOT sum into a model-level "traffic
  // once" total.
  uint64_t input_act_bytes = 0;

  // Total forward-pass bytes touched by this node = weight reads + activation
  // reads + activation writes. Saturating (clamp UINT64_MAX). Denominator of
  // arithmetic_intensity().
  uint64_t bytes_moved() const {
    uint64_t s = weight_bytes;
    if (UINT64_MAX - s < input_act_bytes) return UINT64_MAX;
    s += input_act_bytes;
    if (UINT64_MAX - s < act_bytes) return UINT64_MAX;
    return s + act_bytes;
  }
  // FLOP per byte moved (roofline arithmetic intensity). 0.0 when flops are
  // unknown or no bytes move (undefined) — gate on intensity_known(), not on the
  // value, to distinguish "honest unknown" from a genuine 0.
  double arithmetic_intensity() const {
    if (!flops_known) return 0.0;
    uint64_t bm = bytes_moved();
    if (bm == 0) return 0.0;
    return static_cast<double>(flops) / static_cast<double>(bm);
  }
  bool intensity_known() const { return flops_known && bytes_moved() > 0; }
};

// --- v0.4.0: roofline classification -------------------------------------
// A node/model is memory-bound when its arithmetic intensity (FLOP/byte) is below
// the machine balance point (ridge), else compute-bound. This is an ESTIMATE
// relative to a chosen machine balance — label it approximate in any UI.
enum class RooflineClass : uint8_t { Unknown, MemoryBound, ComputeBound };

// Documented, approximate machine-balance presets (peak FLOP/s / peak bandwidth,
// FLOP/byte, MAC=2 convention). NOT measurements — see ridge_flop_per_byte().
enum class RooflinePreset : uint8_t { Generic, CpuServer, GpuFp32, GpuTensor, MobileNpu };

// Default ridge (Generic): a representative mixed-precision accelerator balance.
constexpr double kDefaultRidgeFlopPerByte = 40.0;

// Ridge FLOP/byte for a preset (Generic=40, CpuServer=8, GpuFp32=13,
// GpuTensor=200, MobileNpu=30). Approximate; pure; never throws.
double ridge_flop_per_byte(RooflinePreset p);

// Model-level roofline verdict over the flops_known nodes classified at a ridge.
struct RooflineSummary {
  double ridge_flop_per_byte = kDefaultRidgeFlopPerByte;
  uint64_t compute_bound_flops = 0;  // sum of flops over compute-bound nodes
  uint64_t memory_bound_flops = 0;   // sum of flops over memory-bound nodes
  uint32_t compute_bound_nodes = 0;
  uint32_t memory_bound_nodes = 0;
  // Fraction of classified FLOPs that are compute-bound (denominator = compute +
  // memory bound flops, NOT total_flops — Unknown nodes are excluded from both).
  // Guards div-by-zero -> 0.0 and uses a SATURATING denominator add (both buckets
  // can reach UINT64_MAX). Defined in CostModel.cpp to reach the TU-local safe_add.
  double compute_bound_fraction() const;
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

  // --- v0.4.0: efficiency aggregates ----------------------------------------
  // Total forward-pass traffic = sum of per_node.bytes_moved() over ALL nodes.
  // This is an UPPER BOUND: it re-reads a shared weight once per consumer (real
  // forward-pass traffic, the OPPOSITE convention from total_weight_bytes which
  // counts storage once) AND counts the reads/writes of pure view ops
  // (Reshape/Transpose/etc.) that logically move nothing. Saturating. 0 in table mode.
  uint64_t total_bytes_moved = 0;
  // Same sum restricted to flops_known nodes — the matched denominator for
  // overall_arithmetic_intensity (numerator total_flops is also flops-known-only).
  uint64_t bytes_moved_flops_known = 0;
  // Roofline classification at kDefaultRidgeFlopPerByte (the default-ridge
  // snapshot; the view recomputes at a user-selected preset via compute_roofline).
  RooflineSummary roofline;

  // Model-level FLOP/byte over the flops-known node set (matched numerator/
  // denominator). Guards div-by-zero -> 0.0.
  double overall_arithmetic_intensity() const {
    if (bytes_moved_flops_known == 0) return 0.0;
    return static_cast<double>(total_flops) /
           static_cast<double>(bytes_moved_flops_known);
  }
};

// Classify one node against a ridge (FLOP/byte). Unknown when flops_known==false
// or bytes_moved()==0 (intensity undefined — never forced into a bound bucket).
// ai >= ridge => ComputeBound (ridge itself is compute-bound, conventional).
RooflineClass classify_node(const NodeCost& nc, double ridge_flop_per_byte);

// Roofline summary over report.per_node at a ridge (PURE, O(V), never throws).
// compute_cost calls this once at kDefaultRidgeFlopPerByte; the view calls it
// again with ridge_flop_per_byte(preset) at draw time without rebuilding the report.
RooflineSummary compute_roofline(const CostReport& report, double ridge_flop_per_byte);

// --- v0.4.0: selectable heatmap metric extractor --------------------------
// The SINGLE source the heatmap range, per-node tint, and legend all read, so
// they cannot diverge (the v0.3.1 group-scale bug was exactly such a divergence).
struct MetricValue {
  uint64_t value = 0;
  bool known = false;  // false => tint neutral gray (honest unknown), not cold
};
// Fixed-point scale for the ArithIntensity metric so it rides the same uint64
// range/normalize/legend path as the byte/count metrics. value = round(FLOP/byte
// * kArithIntensityScale); the legend divides back out for display.
constexpr uint64_t kArithIntensityScale = 1000;

// Extract the selected metric from a NodeCost — which may be a single node's cost
// OR a group SUM from sum_node_costs(). ArithIntensity is therefore derived from
// the aggregate (flops/bytes = ratio-of-sums, the correct roofline value for a
// fused region) — it must NOT read a precomputed per-node intensity. Pure; never
// throws; clamps before any float->int conversion (hostile saturated flops safe).
//   Flops          -> {flops, flops_known}
//   Params         -> {params, true}          (0 is a real value -> cold, not gray)
//   ActBytes       -> {act_bytes, act_bytes!=0}  (0 => unresolved -> honest unknown)
//   ArithIntensity -> {round(ai*scale), flops_known && bytes_moved()>0}
MetricValue metric_value(const NodeCost& nc, HeatmapMetric m);

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
//  Reduce (ReduceSum/Mean/L1/L2/.../ArgMax/ArgMin/CumSum) : flops = |input[0]|.
//  Everything else (Shape/Cast/Reshape/Concat/Gather/control-flow/unknown)
//                          : flops_known = false (structural, ~0 arithmetic).
//
//  ROUTING (v0.4.0): FLOP correctness is owned by an EXPLICIT op-name dispatch
//  that runs FIRST and returns on match; the category-driven generic formulas
//  above run SECOND, only for ops whose formula matches the generic assumption.
//  Adding an op to a color category never implies its FLOP formula. The Quantize
//  category has NO generic fallback (every quant op needs an explicit handler, so
//  QLinearConv's weight-at-slot-3 layout can never be mis-read as elementwise).
//  Explicit handlers (all MAC=2, all safe_mul/safe_add, honest-unknown on any
//  unresolved dim):
//   Einsum        : constrained 2-operand resolver (no ellipsis / single-char /
//                   no intra-operand repeat); macs = prod(dim over union of LHS
//                   labels); else flops_known=false (empty equation stays unknown).
//   Attention     : proj_out = weights(in[1]).back(); v_hidden = proj_out/3
//                   (require %3==0); macs = B*S*in_hidden*proj_out (QKV proj) +
//                   2*B*S*S*v_hidden (QK^T + PV); flops = 2*macs.
//   MultiHeadAttention : pre-projected; hidden_q from query rank3 [.]=shape[2] or
//                   rank4=shape[2]*shape[3]; S_kv from key(in[1]) or S_q; macs =
//                   2*B*S_q*S_kv*hidden_q; flops = 2*macs (head dim cancels).
//   LSTM/GRU/RNN  : gates=4/3/1; X(in[0])=[seq,batch,input]; hidden from attr
//                   hidden_size or R(in[2]).shape[2] or W(in[1]).shape[1]/gates;
//                   num_directions from 'direction' (bidirectional=2) or W.shape[0];
//                   macs = seq*num_directions*batch*gates*hidden*(input+hidden);
//                   flops = 2*macs. (Dynamic seq => honest-unknown.)
//   QLinearConv   : weight = in[3]; ConvInteger weight = in[1]; else Conv math.
//   QLinearMatMul : A=in[0],B=in[3]; MatMulInteger A=in[0],B=in[1]; QGemm A=in[0],
//                   B=in[3] with transA; macs = |O|*K (K from A), flops=2*macs.
//   QLinearAdd/Mul: |O|.  QLinearGlobalAveragePool: |input0|.  QLinearAveragePool:
//                   |O|*prod(kernel_shape).
//   Quantize/Dequantize/DynamicQuantizeLinear : |O| (pointwise scale/round).
//  Op strings are matched NORMALIZED (lowercase, last dot-segment) so
//  com.microsoft.* contrib ops route correctly. Category (color) still comes from
//  engine/OpCategory.h (categorize_op).
//
// ---------------------------------------------------------------------------
//  EFFICIENCY (v0.4.0). Per node: input_act_bytes = sum over INPUT slots that are
//  NOT initializers of elem_count*dtype_size. bytes_moved() = weight_bytes +
//  input_act_bytes + act_bytes (saturating). arithmetic_intensity() = flops /
//  bytes_moved() (FLOP/byte), defined only when intensity_known(). Roofline:
//  classify_node compares intensity to a ridge FLOP/byte; compute_roofline buckets
//  flops_known nodes. Model: total_bytes_moved (all nodes, upper bound),
//  bytes_moved_flops_known (matched denom), overall_arithmetic_intensity(). All
//  saturating; unresolved/quant dtype contributes 0 bytes (honest, never faked).
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

// Live activation bytes at each node in execution (node index == topological)
// order — the FULL curve behind the single peak_activation_bytes scalar.
// liveness_curve[i] = the memory HIGH-WATER MARK during node i: live activation
// bytes just after node i's outputs are produced but BEFORE the values whose
// last use is node i are freed. That pre-free point is exactly where the peak
// pass takes its max, so max(curve) EQUALS CostReport::peak_activation_bytes by
// construction (this is a doctest-asserted invariant — the two cannot diverge).
// Recording at the post-free point instead would UNDERSTATE the peak whenever the
// peak node frees inputs (e.g. a Concat), so it is deliberately pre-free: this is
// the true resident-memory pressure while node i runs, matching the peak marker
// the view draws on the plot.
//
// Re-walks with the IDENTICAL accounting as compute_peak_activation_bytes (they
// share one implementation): activation = value whose name is NOT a graph
// initializer; bytes(v) = elem_count(v.shape) * dtype_size(v.dtype) (unresolved
// shape or quant/unknown dtype => 0 bytes, never an error); last_use = largest
// consumer node index; graph-output values are never freed; the graph_inputs seed
// contributes to the running `live` but is not itself a recorded per-node point.
// PURE, no payload reads, saturating, never throws. Returns an EMPTY curve in
// table mode (no compute graph) or for an out-of-range graph_index. Its size,
// when non-empty, equals graphs[graph_index].nodes.size().
std::vector<uint64_t> activation_liveness_curve(const ir::Model& model,
                                                uint32_t graph_index);

}  // namespace netvis
