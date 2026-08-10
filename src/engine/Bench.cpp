// engine/Bench.cpp — the benchmark + perf-regression harness (#97).
//
// Bench.h carries the rationale for the harness existing at all; this file is
// the mechanics. Three things drive every choice below:
//
//   1. REPRODUCIBILITY. The fixture is generated, not downloaded, and generated
//      from an in-file PRNG rather than <random> — libstdc++, libc++ and MSVC
//      disagree about the output of every distribution in <random>, so a fixture
//      built from one would differ per platform and a cross-platform baseline
//      could never be compared. splitmix64 is 4 lines and is bit-identical
//      everywhere.
//   2. SINGLE-THREADED PURITY. Every stage below is a direct call into the
//      engine's pure entry point. No JobSystem, no ModelSession, no on-disk
//      layout cache — a cached layout would be reported as a layout speedup
//      (Bench.h's IMPORTANT note), and a thread pool would add exactly the
//      variance a regression gate must not have.
//   3. EQUIVALENT START STATE PER REPEAT. Only one stage mutates the model
//      (infer_shapes_ext writes ValueInfo in place); it is the only one that
//      needs an explicit reset, and it gets one — see measure_case().
#include "engine/Bench.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Rss.h"
#include "engine/CollapseTree.h"
#include "engine/CostModel.h"
#include "engine/GraphAdjacency.h"
#include "engine/LayoutEngine.h"
#include "engine/ModelPath.h"
#include "engine/ShapeInferenceExt.h"
#include "engine/SpatialIndex.h"
#include "parsers/Parser.h"

namespace netvis {

namespace {

using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

// Milliseconds since `t0`, mirroring ModelSession's local ms_since helper so the
// harness and the app's StageTimings are the same unit and the same clock.
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Sink for values whose only purpose is to stop the optimizer deleting the loop
// that produced them. `volatile` forces the store to actually happen; nothing
// ever reads it.
volatile uint64_t g_bench_sink = 0;

// ---------------------------------------------------------------------------
// splitmix64 — the fixture PRNG
// ---------------------------------------------------------------------------
// Deliberately NOT std::mt19937/std::uniform_int_distribution: the distributions
// in <random> are not specified to produce the same sequence across standard
// libraries, so a fixture built on one would not match the baseline recorded on
// another and the gate would compare two different graphs. splitmix64 is fully
// specified by these constants, so every platform generates the same model.
class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}

  uint64_t next() {
    state_ += 0x9E3779B97F4A7C15ull;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  // Uniform-ish in [0, n). The modulo bias is real and irrelevant: this picks
  // fixture topology, not statistics, and determinism is the only property that
  // matters here.
  uint32_t below(uint32_t n) {
    if (n == 0) return 0;
    return static_cast<uint32_t>(next() % n);
  }

 private:
  uint64_t state_;
};

// ---------------------------------------------------------------------------
// Sample reduction
// ---------------------------------------------------------------------------
// Sorts `samples` in place and folds them into a stage. The reported `ms` is the
// MEDIAN (Bench.h): a shared CI runner will hand one repeat to a descheduled
// core, and a mean would carry that hiccup into the gate as a regression.
//
// For an even repeat count we take the UPPER middle rather than averaging the
// two central samples, so the published number is always a sample that was
// actually observed — an interpolated median is a timing no run ever produced.
BenchStage make_stage(const char* name, std::vector<double>& samples) {
  BenchStage s;
  s.name = name;
  s.repeats = static_cast<uint32_t>(samples.size());
  if (samples.empty()) return s;
  std::sort(samples.begin(), samples.end());
  s.min_ms = samples.front();
  s.max_ms = samples.back();
  s.ms = samples[samples.size() / 2];
  return s;
}

// ---------------------------------------------------------------------------
// Synthetic fixture shape
// ---------------------------------------------------------------------------
// The canonical block: a residual conv/attention-ish unit whose ops are all
// recognized by categorize_op AND carry a real formula in CostModel. That is the
// point — a block of invented op names would send every node down the
// honest-unknown path, so the cost stage would measure the "give up" branch
// instead of the arithmetic it is supposed to time.
//
//   Conv/MatMul          -> Conv & MatMul FLOP formulas (need a weight operand)
//   Relu/Gelu/Softmax    -> Activation, flops = |O|
//   LayerNormalization   -> Norm, flops = |O|
//   MaxPool              -> Pool, needs a kernel_shape attribute
//   Add                  -> Elementwise, and the block's residual merge point
constexpr const char* kBlockOps[] = {
    "Conv", "Relu",              "Add",     "MaxPool",
    "MatMul", "Gelu", "LayerNormalization", "Softmax",
};
constexpr uint32_t kBlockOpsN =
    static_cast<uint32_t>(sizeof(kBlockOps) / sizeof(kBlockOps[0]));

// Fixture dimensions. Fixed, not random: the timings must be readable against
// the structural counts, and a per-node random shape would make FLOPs vary
// between rungs for reasons that have nothing to do with graph size.
constexpr int64_t kBatch = 1;
constexpr int64_t kChannels = 64;
constexpr int64_t kSpatial = 32;
constexpr int64_t kKernel = 3;

// Hostile-input discipline: SyntheticSpec is public API, so a caller can ask for
// a graph large enough to exhaust memory. Both knobs are clamped, and because
// the realized node count is `(nodes / block_size) * block_size` the product can
// never exceed kMaxSyntheticNodes — no multiply, so nothing to overflow.
constexpr uint32_t kMaxSyntheticNodes = 2000000;
constexpr uint32_t kMaxBlockSize = 4096;

bool op_takes_weight(std::string_view op) {
  return op == "Conv" || op == "MatMul";
}
bool op_is_merge(std::string_view op) { return op == "Add"; }

void add_ints_attr(ir::Model& m, ir::Graph& g, const char* name,
                   std::initializer_list<int64_t> vals) {
  ir::Attribute a;
  a.name = m.intern(name);
  a.value.kind = ir::AttrValue::Kind::Ints;
  a.value.ints.assign(vals);
  g.attributes.push_back(std::move(a));
}

void add_int_attr(ir::Model& m, ir::Graph& g, const char* name, int64_t v) {
  ir::Attribute a;
  a.name = m.intern(name);
  a.value.kind = ir::AttrValue::Kind::Int;
  a.value.i = v;
  g.attributes.push_back(std::move(a));
}

void add_float_attr(ir::Model& m, ir::Graph& g, const char* name, double v) {
  ir::Attribute a;
  a.name = m.intern(name);
  a.value.kind = ir::AttrValue::Kind::Float;
  a.value.f = v;
  g.attributes.push_back(std::move(a));
}

// Attributes for one op, appended to `g.attributes`; returns how many were
// added so the caller can close the node's Range.
//
// These are CONSTANTS, never seeded values, and that is load-bearing:
// node_fingerprint hashes every attribute name and typed value, so an attribute
// that varied per block instance would give the instances different fingerprints
// and CollapseTree would never see them as repeats.
uint32_t add_op_attrs(ir::Model& m, ir::Graph& g, std::string_view op) {
  if (op == "Conv") {
    add_ints_attr(m, g, "kernel_shape", {kKernel, kKernel});
    add_ints_attr(m, g, "strides", {1, 1});
    add_ints_attr(m, g, "pads", {1, 1, 1, 1});
    add_ints_attr(m, g, "dilations", {1, 1});
    add_int_attr(m, g, "group", 1);
    return 5;
  }
  if (op == "MaxPool") {
    // CostModel's Pool formula reads kernel_shape; without it the node reports
    // an honest unknown and the pool op stops contributing measurable work.
    add_ints_attr(m, g, "kernel_shape", {2, 2});
    add_ints_attr(m, g, "strides", {1, 1});
    add_ints_attr(m, g, "pads", {0, 0, 0, 0});
    return 3;
  }
  if (op == "LayerNormalization") {
    add_int_attr(m, g, "axis", -1);
    add_float_attr(m, g, "epsilon", 1e-5);
    return 2;
  }
  if (op == "Softmax") {
    add_int_attr(m, g, "axis", -1);
    return 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// The headless SizeFn
// ---------------------------------------------------------------------------
// Layout is measurable headlessly ONLY because node sizes are an input
// (LayoutEngine.h): the view measures glyph extents and hands the result in, so
// nothing in compute_layout touches a font atlas. The bench therefore supplies a
// closed-form size — same form as LayoutEngine's own default_size — and gets
// bit-identical boxes on every machine, with no ImGui linked in.
//
// It is passed as a real SizeFn rather than left null (compute_layout falls back
// to default_size for a null one) so the std::function indirection stays in the
// measurement exactly as it is in production.
Vec2 bench_size(const DisplayNode& d) {
  const float chars = d.is_group ? 18.0f : 12.0f;
  return Vec2{40.0f + chars * 7.0f, 40.0f};
}

// Fractions of the world AREA the visible_scan proxies pretend the camera shows.
//
// TWO of them, deliberately (#99). The spatial index that makes culling
// O(visible) is only a win below roughly a quarter of the world: past that the
// grid walk visits nearly every item anyway and costs more than a flat scan, so
// both the index and the canvas fall back to direct enumeration there. Gating on
// one fraction alone would therefore be misleading in whichever direction it was
// chosen —
//   * wide only  => the whole optimization reads as a regression;
//   * zoomed only => a real regression at the zoom level a model FIRST OPENS at
//                    (fit-to-screen, i.e. full coverage) would go unnoticed.
// So the harness reports both and the gate compares both.
constexpr double kVisibleFractionWide = 1.0;
constexpr double kVisibleFractionZoomed = 0.01;

// Lowercased extension without the leading dot ("" if none). Same derivation
// ReportJson uses to route detection; duplicated rather than exported because it
// is three lines and ReportJson's copy is TU-local by design.
std::string ext_of(const std::string& path) {
  auto dot = path.find_last_of('.');
  auto slash = path.find_last_of("/\\");
  if (dot == std::string::npos) return {};
  if (slash != std::string::npos && dot < slash) return {};  // dot in a dir name
  std::string e = path.substr(dot + 1);
  for (char& c : e)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}

// Final path component — BenchCase::label for a real model file. The gate keys
// on the label, so a bare file name keeps a baseline portable between a CI
// checkout and a developer's tree where the absolute paths differ.
std::string base_name(const std::string& path) {
  auto slash = path.find_last_of("/\\");
  if (slash == std::string::npos) return path;
  return path.substr(slash + 1);
}

float min4(float a, float b, float c, float d) {
  return std::min(std::min(a, b), std::min(c, d));
}
float max4(float a, float b, float c, float d) {
  return std::max(std::max(a, b), std::max(c, d));
}

// The same inclusive overlap test GraphCanvas uses (GraphCanvas.cpp's
// aabb_overlap), re-stated over Vec2 so the proxy stays in netvis_core and never
// includes an ImGui type.
bool aabb_overlap(Vec2 amin, Vec2 amax, Vec2 bmin, Vec2 bmax) {
  return amin.x <= bmax.x && amax.x >= bmin.x && amin.y <= bmax.y &&
         amax.y >= bmin.y;
}

// ---------------------------------------------------------------------------
// measure_case — every stage for one already-built model
// ---------------------------------------------------------------------------
// `mmap_base`/`mmap_size` are the model's mapping for a real file, or
// nullptr/0 for a synthetic rung (a generated model has no backing file, so the
// constant-driven shape handlers correctly find nothing to read).
//
// kStageParse is NOT added here: a synthetic rung was never parsed and reporting
// a 0 ms parse for it would put a fabricated sample in the baseline. The caller
// prepends the real parse stage for a file-backed case.
BenchCase measure_case(ir::Model& model, std::string label, uint32_t repeats,
                       const uint8_t* mmap_base, size_t mmap_size) {
  BenchCase c;
  c.label = std::move(label);
  c.peak_rss_bytes = peak_rss_bytes();
  if (model.graphs.empty()) return c;

  const uint32_t gi = 0;
  ir::Graph& g = model.graphs[gi];
  c.ir_nodes = static_cast<uint32_t>(g.nodes.size());

  // Deduped producer->consumer IR edges, the same derivation ReportJson reports
  // and the same one LayoutEngine's display edges are built from. Untimed: this
  // is a structural count the timings are read against, not a stage.
  {
    GraphAdjacency adj;
    adj.build(model, gi);
    c.ir_edges = static_cast<uint32_t>(adj.succ_values().size());
  }

  std::vector<double> samples;
  samples.reserve(repeats);
  uint64_t sink = 0;

  // --- collapse ------------------------------------------------------------
  // build() clears groups_/display_/expanded_ as its first act, so every repeat
  // starts from the same empty tree with no help from us.
  CollapseTree collapse;
  samples.clear();
  for (uint32_t r = 0; r < repeats; ++r) {
    Clock::time_point t0 = Clock::now();
    collapse.build(model, gi);
    samples.push_back(ms_since(t0));
  }
  c.stages.push_back(make_stage(kStageCollapse, samples));

  // The display list is whatever build() left behind — which is EVERY group
  // expanded (CollapseTree's post-build default, chosen for Netron parity). That
  // is deliberate: it is the view a user actually gets when a model opens, so it
  // is the one worth gating on. Measuring the fully collapsed view instead would
  // time a layout of a handful of super-nodes, which no first paint ever does.
  c.display_nodes = static_cast<uint32_t>(collapse.display_nodes().size());

  // --- shape inference -----------------------------------------------------
  // The only stage that mutates the model: infer_shapes_ext writes ValueInfo
  // shapes/dtypes in place and set_shape() only fills a value that is still
  // EMPTY. Timing a second repeat against an already-inferred model would
  // therefore measure the walk with every write short-circuited — a different
  // operation, and a faster one, so repeats 2..N would drag the median down.
  //
  // The reset is a saved pre-inference snapshot of g.values, restored before
  // each repeat and outside the timer. The snapshot clears shape+dtype on every
  // PRODUCED value (producer >= 0) while leaving graph inputs and initializer
  // values intact — which is exactly the state a real ONNX file arrives in
  // (declared inputs and weights, absent value_info for everything between), so
  // each repeat resolves the same real work the app resolves on open.
  {
    const std::vector<ir::ValueInfo> original = g.values;
    std::vector<ir::ValueInfo> seed = original;
    for (ir::ValueInfo& v : seed) {
      if (v.producer < 0) continue;
      v.shape.clear();
      v.dtype = ir::DType::Unknown;
    }

    samples.clear();
    for (uint32_t r = 0; r < repeats; ++r) {
      g.values = seed;  // outside the timer: the reset is not part of the cost
      Clock::time_point t0 = Clock::now();
      uint32_t resolved =
          infer_shapes_ext(model, gi, mmap_base, mmap_size, nullptr);
      samples.push_back(ms_since(t0));
      sink += resolved;
    }
    c.stages.push_back(make_stage(kStageShapes, samples));

    // Leave the model in the most-resolved state available before the cost
    // stage runs: restore the generator's/parser's own shapes, then — only if
    // that still leaves a produced value unresolved — infer once more, UNTIMED.
    // A real ONNX file needs this and gets the same enrichment ModelSession's
    // ShapeInferJob applies, so compute_cost below exercises the real formula
    // table rather than the honest-unknown branch. A synthetic rung built with
    // with_shapes already carries every shape, and the guard is what stops it
    // paying for a whole extra inference pass that would resolve nothing.
    g.values = original;
    bool any_unresolved = false;
    for (const ir::ValueInfo& v : g.values) {
      if (v.producer >= 0 && v.shape.empty()) {
        any_unresolved = true;
        break;
      }
    }
    if (any_unresolved)
      sink += infer_shapes_ext(model, gi, mmap_base, mmap_size, nullptr);
  }

  // --- layout --------------------------------------------------------------
  // compute_layout DIRECTLY (Bench.h): ModelSession::request_layout consults the
  // on-disk .nvl cache keyed on structure_hash+collapse_hash first, so routing
  // through the session would let the second run of the harness report a cache
  // hit as a layout speedup. compute_layout is pure, so every repeat starts from
  // identical state with no reset needed.
  LayoutResult layout;
  samples.clear();
  for (uint32_t r = 0; r < repeats; ++r) {
    Clock::time_point t0 = Clock::now();
    LayoutResult lr =
        compute_layout(model, gi, collapse, bench_size, LayoutParams{}, nullptr);
    samples.push_back(ms_since(t0));
    layout = std::move(lr);  // move-assign after the clock stops
  }
  c.stages.push_back(make_stage(kStageLayout, samples));

  // May EXCEED display_nodes: compute_layout duplicates multi-consumer sources
  // into per-consumer clones and inserts Sugiyama dummies, and emits real+clone
  // nodes as boxes (DECISIONS.md, v0.2.0). The gate needs the box count because
  // that, not display_nodes, is what the per-frame scans below iterate.
  c.layout_boxes = static_cast<uint32_t>(layout.boxes.size());

  // --- cost ----------------------------------------------------------------
  // Pure over the (now shape-resolved) model; no reset needed.
  samples.clear();
  for (uint32_t r = 0; r < repeats; ++r) {
    Clock::time_point t0 = Clock::now();
    CostReport cr = compute_cost(model, gi);
    samples.push_back(ms_since(t0));
    sink += cr.total_flops;  // consumed after the clock stops
  }
  c.stages.push_back(make_stage(kStageCost, samples));

  // --- visible scan --------------------------------------------------------
  // A PROXY for per-frame render cost, never a frame time (Bench.h).
  {
    // visible_scan_ms is frozen to return the median alone, so min/max echo it.
    // Echoing is honest here; inventing a spread the function never exposed
    // would put numbers in the baseline that no repeat produced.
    auto scan_stage = [&](const char* name, double fraction) {
      BenchStage vs;
      vs.name = name;
      vs.repeats = repeats;
      vs.ms = visible_scan_ms(layout, fraction, repeats);
      vs.min_ms = vs.ms;
      vs.max_ms = vs.ms;
      return vs;
    };
    c.stages.push_back(scan_stage(kStageVisibleScan, kVisibleFractionWide));
    c.stages.push_back(
        scan_stage(kStageVisibleScanZoomed, kVisibleFractionZoomed));
  }

  g_bench_sink = sink;

  // Peak RSS is the process high-water mark and cannot be reset (core/Rss.h), so
  // this is "peak across everything up to and including this case", not this
  // case in isolation. Read after the stages so a spike inside one of them is
  // captured; 0 means the platform counter was unavailable, never zero bytes.
  c.peak_rss_bytes = peak_rss_bytes();
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// make_synthetic_model
// ---------------------------------------------------------------------------
// Shape: a chain of `blocks` instances of ONE block template, plus a residual
// skip inside each instance when branch > 1.
//
// WHY CollapseTree COLLAPSES IT. Detection pass (a) normalizes a node name by
// replacing its FIRST digit run with '#' and groups on the prefix through that
// run. Every node here is named "/model/layers.<i>/<Op>_<slot>", whose first
// digit run is the instance index <i>, so all of them share the key
// "/model/layers.#" with `blocks` distinct integer values — over the >= 2
// instances pass (a) requires, giving one group labelled "layers" with
// instances == blocks. That is the same key shape a real exported ONNX
// transformer produces, which is why this fixture exercises the real path.
//
// The instances are also STRUCTURALLY identical, which is what makes the group
// honest rather than a naming coincidence: node_fingerprint hashes op_type,
// input arity, output arity and every attribute name+typed value, and excludes
// the node name. Slot j of every instance is generated from the same template —
// same op, same arity, same constant attributes — so fingerprints match across
// instances by construction. Only the names and the value indices differ, and
// the fingerprint sees neither.
ir::Model make_synthetic_model(const SyntheticSpec& spec) {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("Synthetic");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  g.name = m.intern("main");

  const uint32_t want_nodes = std::min(spec.nodes, kMaxSyntheticNodes);
  if (want_nodes == 0) return m;

  // block_size == 0 means "no repeated blocks" (Bench.h): degenerate to a single
  // instance, whose lone index value fails pass (a)'s >= 2 distinct-instances
  // test, so no name-based group forms. CollapseTree's second pass may still
  // find a period in the op cycle and emit a generic "Block" group — that is the
  // detector doing its job on a genuinely periodic graph, not a naming artifact,
  // and the default ladder never uses block_size == 0 anyway.
  uint32_t block = spec.block_size;
  if (block == 0) block = want_nodes;
  block = std::min(block, kMaxBlockSize);
  block = std::min(block, want_nodes);

  const uint32_t blocks = want_nodes / block;  // rounds DOWN to whole blocks
  const uint32_t total_nodes = blocks * block;
  if (total_nodes == 0) return m;

  SplitMix64 rng(spec.seed);

  // --- The template, drawn ONCE ---------------------------------------------
  // Everything the seed influences is decided here and then stamped out
  // identically for every instance. Drawing per instance instead would give the
  // instances different structure and CollapseTree would stop seeing repeats —
  // which would silently turn the 100k rung into a shape NetVis never displays.
  const uint32_t rot = rng.below(kBlockOpsN);
  std::vector<uint32_t> op_of(block);
  std::vector<uint32_t> skip_back(block, 0);
  for (uint32_t j = 0; j < block; ++j) {
    op_of[j] = (j + rot) % kBlockOpsN;
    // `branch` is the fan-out knob. branch <= 1 is a PURE CHAIN: the merge op's
    // second operand is a bias initializer, so no value ever has two consumers.
    // branch > 1 gives the merge op a residual operand reaching 2..branch slots
    // back, so that source value gains a second consumer and the layout has real
    // skip edges to route (which is what puts Sugiyama dummies in play).
    if (spec.branch > 1 && op_is_merge(kBlockOps[op_of[j]]))
      skip_back[j] = 2 + rng.below(spec.branch - 1);
  }

  // Shapes. Initializers carry theirs unconditionally, even when with_shapes is
  // false: a real file always declares its weight shapes, and they are the seed
  // that shape inference propagates from.
  SmallVec<int64_t, 6> act_shape;
  act_shape.push_back(kBatch);
  act_shape.push_back(kChannels);
  act_shape.push_back(kSpatial);
  act_shape.push_back(kSpatial);
  SmallVec<int64_t, 6> conv_w_shape;
  conv_w_shape.push_back(kChannels);
  conv_w_shape.push_back(kChannels);
  conv_w_shape.push_back(kKernel);
  conv_w_shape.push_back(kKernel);
  SmallVec<int64_t, 6> mm_w_shape;
  mm_w_shape.push_back(kSpatial);
  mm_w_shape.push_back(kSpatial);
  SmallVec<int64_t, 6> bias_shape;
  bias_shape.push_back(kChannels);

  g.nodes.reserve(total_nodes);
  g.values.reserve(static_cast<size_t>(total_nodes) * 2 + 1);
  g.edge_refs.reserve(static_cast<size_t>(total_nodes) * 3);

  // Append an activation value; `producer` is the node index that will emit it.
  auto add_act_value = [&](const std::string& name, int32_t producer) {
    ir::ValueInfo v;
    v.name = m.intern(name);
    v.producer = producer;
    if (spec.with_shapes) {
      v.shape = act_shape;
      v.dtype = ir::DType::F32;
    }
    g.values.push_back(std::move(v));
    return static_cast<uint32_t>(g.values.size() - 1);
  };

  // Append a weight: one ValueInfo (so a node can consume it by slot) plus the
  // matching TensorRef in g.initializers (so CostModel counts params/bytes).
  // file_offset stays UINT64_MAX — a generated model has no backing file, and
  // nothing in the harness ever reads a payload (spec §2.1 holds here too).
  auto add_weight = [&](const std::string& name,
                        const SmallVec<int64_t, 6>& shape) {
    StringId id = m.intern(name);
    ir::ValueInfo v;
    v.name = id;
    v.producer = -1;
    v.shape = shape;
    v.dtype = ir::DType::F32;
    g.values.push_back(std::move(v));
    const uint32_t vi = static_cast<uint32_t>(g.values.size() - 1);

    ir::TensorRef t;
    t.name = id;
    t.dtype = ir::DType::F32;
    t.shape = shape;
    int64_t elems = t.elem_count();
    if (elems < 0) elems = 0;
    t.byte_len = static_cast<uint64_t>(elems) * ir::dtype_size(ir::DType::F32);
    g.initializers.push_back(std::move(t));
    return vi;
  };

  const uint32_t input_val = add_act_value("input", -1);
  g.graph_inputs.push_back(input_val);

  uint32_t cur = input_val;                 // the running chain value
  std::vector<uint32_t> slot_val(block, 0); // outputs of the CURRENT instance

  for (uint32_t i = 0; i < blocks; ++i) {
    const uint32_t block_in = cur;
    const std::string inst = "/model/layers." + std::to_string(i) + "/";
    for (uint32_t j = 0; j < block; ++j) {
      const char* op = kBlockOps[op_of[j]];
      // The first digit run in this name is `i` — see the collapse rationale in
      // the function comment. The trailing "_<slot>" digits come later and are
      // therefore invisible to the grouping key.
      const std::string node_name = inst + op + "_" + std::to_string(j);
      const int32_t node_index = static_cast<int32_t>(g.nodes.size());
      const uint32_t out_val = add_act_value(node_name + ".out", node_index);

      ir::Node n;
      n.op_type = m.intern(op);
      n.name = m.intern(node_name);

      n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
      uint32_t in_count = 1;
      g.edge_refs.push_back(cur);
      if (op_takes_weight(op)) {
        const bool is_conv = (std::string_view(op) == "Conv");
        g.edge_refs.push_back(add_weight(
            node_name + ".weight", is_conv ? conv_w_shape : mm_w_shape));
        ++in_count;
      } else if (op_is_merge(op)) {
        const uint32_t d = skip_back[j];
        if (d == 0) {
          // Pure-chain mode: a bias operand, not a second producer.
          g.edge_refs.push_back(add_weight(node_name + ".bias", bias_shape));
        } else {
          // Residual: reach back within THIS instance, clamped to the block's
          // input so slot j < d still wires to a real, earlier value.
          g.edge_refs.push_back(j >= d ? slot_val[j - d] : block_in);
        }
        ++in_count;
      }
      n.inputs.count = in_count;

      n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
      g.edge_refs.push_back(out_val);
      n.outputs.count = 1;

      n.attributes.begin = static_cast<uint32_t>(g.attributes.size());
      n.attributes.count = add_op_attrs(m, g, op);

      g.nodes.push_back(std::move(n));
      slot_val[j] = out_val;
      cur = out_val;
    }
  }

  g.graph_outputs.push_back(cur);
  return m;
}

// ---------------------------------------------------------------------------
// visible_scan_ms
// ---------------------------------------------------------------------------
// Mirrors the two O(total) sweeps GraphCanvas runs every single frame: the hover
// hit-test over layout->boxes and the cull test over layout->edges, each an AABB
// overlap against the visible world rect. Neither is indexed, so both cost
// O(total) no matter how little is on screen — that is the scaling risk #99
// exists to remove, and this is the number that must stop growing when it does.
//
// The canvas' hover loop breaks on the first box under the pointer; this one
// never breaks. That is on purpose: the full sweep is both the common case (the
// pointer is over empty canvas or over nothing at all) and the worst case, and a
// data-dependent early exit would make the gate's number depend on where a
// hypothetical mouse was.
//
// No ImGui, no GL, no view header — this lives in netvis_core so CI can run it.
double visible_scan_ms(const LayoutResult& layout, double visible_fraction,
                       uint32_t repeats) {
  // `visible_fraction` is a fraction of AREA, so the linear half-extents scale
  // by its square root. Written to reject NaN as well as out-of-range input:
  // !(f > 0.0) is true for NaN, where f <= 0.0 would not be.
  double f = visible_fraction;
  if (!(f > 0.0)) f = 1.0;
  if (f > 1.0) f = 1.0;
  const float lin = static_cast<float>(std::sqrt(f));

  const float cx = 0.5f * (layout.bounds_min.x + layout.bounds_max.x);
  const float cy = 0.5f * (layout.bounds_min.y + layout.bounds_max.y);
  const float hw = 0.5f * (layout.bounds_max.x - layout.bounds_min.x) * lin;
  const float hh = 0.5f * (layout.bounds_max.y - layout.bounds_min.y) * lin;
  const Vec2 vw_min{cx - hw, cy - hh};
  const Vec2 vw_max{cx + hw, cy + hh};

  const uint32_t n = repeats == 0 ? 1u : repeats;
  std::vector<double> samples;
  samples.reserve(n);
  uint64_t sink = 0;

  // #99: the index is built ONCE here, outside the timing loop, because that is
  // how the canvas uses it — a grid is rebuilt when the layout changes, not when
  // a frame is drawn, and layout changes are already the expensive operation.
  // Timing the build inside the per-frame number would report a per-layout cost
  // as a per-frame one. (The build is ~5 ms at the 100k rung, against a layout
  // stage of ~16 ms that must already have run for a grid to exist at all.)
  SpatialIndex index;
  index.build(layout);
  std::vector<uint32_t> candidates;

  const WorldRect rect{vw_min.x, vw_min.y, vw_max.x, vw_max.y};

  for (uint32_t r = 0; r < n; ++r) {
    Clock::time_point t0 = Clock::now();
    uint64_t hits = 0;

    // The index returns CANDIDATES by contract, so the exact AABB test still
    // runs — exactly as at the call site. Dropping it here would measure a
    // cheaper operation than the canvas actually performs.
    index.query_boxes(rect, candidates);
    // Reverse iteration, as the canvas does so topmost boxes win ties. The
    // query returns ascending indices, so walking forward would silently invert
    // the tie-break; this mirrors the fix the canvas needed for the same reason.
    for (size_t k = candidates.size(); k-- > 0;) {
      const uint32_t bi = candidates[k];
      if (bi >= layout.boxes.size()) continue;
      const NodeBox& b = layout.boxes[bi];
      const Vec2 bmin{b.pos.x, b.pos.y};
      const Vec2 bmax{b.pos.x + b.size.x, b.pos.y + b.size.y};
      if (aabb_overlap(bmin, bmax, vw_min, vw_max)) ++hits;
    }

    // Edge cull: the canvas bounds each bezier by its four control points.
    index.query_edges(rect, candidates);
    for (uint32_t ei : candidates) {
      if (ei >= layout.edges.size()) continue;
      const EdgeCurve& e = layout.edges[ei];
      const Vec2 emin{min4(e.p0.x, e.p1.x, e.p2.x, e.p3.x),
                      min4(e.p0.y, e.p1.y, e.p2.y, e.p3.y)};
      const Vec2 emax{max4(e.p0.x, e.p1.x, e.p2.x, e.p3.x),
                      max4(e.p0.y, e.p1.y, e.p2.y, e.p3.y)};
      if (aabb_overlap(emin, emax, vw_min, vw_max)) ++hits;
    }

    samples.push_back(ms_since(t0));
    sink += hits;
  }

  // Both sweeps are side-effect-free over data this TU can see, so the optimizer
  // is entitled to delete them outright. Publishing the accumulator through a
  // volatile store makes the work observable and keeps it.
  g_bench_sink = sink;

  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// ---------------------------------------------------------------------------
// run_bench
// ---------------------------------------------------------------------------
Result<std::vector<BenchCase>> run_bench(const BenchOptions& options) {
  // repeats == 0 would make the median undefined and every stage empty; main.cpp
  // already clamps its CLI value, but BenchOptions is public API.
  const uint32_t repeats = options.repeats == 0 ? 1u : options.repeats;

  std::vector<SyntheticSpec> ladder = options.ladder;
  std::vector<std::string> labels;

  if (ladder.empty()) {
    // The default ladder. block_size 8 is one canonical residual block, so the
    // rungs differ ONLY in how many instances repeat — node count is the single
    // varying dimension and a rung-to-rung comparison isolates scaling.
    auto rung = [](uint32_t nodes) {
      SyntheticSpec s;
      s.nodes = nodes;
      s.branch = 2;
      s.block_size = 8;
      s.seed = 1;
      s.with_shapes = true;
      return s;
    };
    ladder.push_back(rung(1000));
    labels.emplace_back("synthetic-1k");
    ladder.push_back(rung(10000));
    labels.emplace_back("synthetic-10k");
    if (!options.quick) {
      ladder.push_back(rung(100000));
      labels.emplace_back("synthetic-100k");
    }
  } else {
    // A caller-supplied ladder is run verbatim — `quick` is documented as
    // dropping the 100k rung, which is a property of the DEFAULT ladder; second
    // -guessing a custom one would silently discard a rung the caller asked for.
    for (const SyntheticSpec& s : ladder)
      labels.push_back("synthetic-" + std::to_string(s.nodes));
  }

  std::vector<BenchCase> cases;
  cases.reserve(ladder.size() + options.model_paths.size());

  for (size_t i = 0; i < ladder.size(); ++i) {
    ir::Model model = make_synthetic_model(ladder[i]);
    cases.push_back(
        measure_case(model, labels[i], repeats, nullptr, 0));
  }

  // --- real model files ------------------------------------------------------
  // PARTIAL FAILURE POLICY: a path that will not resolve, map or parse is
  // SKIPPED — no case is recorded for it and the run continues. A bench invoked
  // over several files must still publish the numbers for the files that worked;
  // aborting the whole run because one path was stale would throw away the
  // synthetic ladder too. The run only fails when it produced nothing at all
  // (checked after the loop), which is the one outcome with no numbers to gate on.
  for (const std::string& path : options.model_paths) {
    ResolvedModelPath resolved = resolve_model_path(path);
    Result<MappedFile> mapped = MappedFile::open(resolved.map_path);
    if (!mapped) continue;
    MappedFile file = mapped.take();

    const std::string ext = ext_of(resolved.map_path);
    ProgressSink progress;

    // kStageParse. Each repeat parses into a FRESH ir::Model, so every one
    // starts from the same state with no reset needed. The mapping is opened
    // once and stays warm across repeats: this measures parse throughput, not
    // cold first-touch page-fault cost — that is #100's cold-open measurement,
    // which needs a cold page cache the harness cannot create from inside the
    // process.
    std::vector<double> samples;
    samples.reserve(repeats);
    ir::Model model;
    bool parsed_ok = false;
    for (uint32_t r = 0; r < repeats; ++r) {
      Clock::time_point t0 = Clock::now();
      Result<ir::Model> parsed = parse_model(file, ext, progress);
      const double dt = ms_since(t0);
      if (!parsed) {
        parsed_ok = false;
        break;
      }
      samples.push_back(dt);
      model = parsed.take();
      parsed_ok = true;
    }
    if (!parsed_ok) continue;

    BenchCase c = measure_case(model, base_name(path), repeats, file.data(),
                               static_cast<size_t>(file.size()));
    // Parse comes first in the stage list because it comes first in time; the
    // gate matches on (label, stage name) so order is presentation only.
    c.stages.insert(c.stages.begin(), make_stage(kStageParse, samples));
    cases.push_back(std::move(c));
  }

  if (cases.empty())
    return err("bench produced no cases (every requested model failed to open)");
  return cases;
}

// ---------------------------------------------------------------------------
// build_bench_json
// ---------------------------------------------------------------------------
// nlohmann::ordered_json, exactly as ReportJson does, so key order is stable and
// a baseline diff shows real changes instead of hash-order churn.
std::string build_bench_json(const std::vector<BenchCase>& cases) {
  json j;
  j["schema"] = kBenchSchema;
  // Logical cores. Every stage here is single-threaded, so this is context for a
  // human reading two baselines, not a divisor. 0 means the host would not say.
  j["hardware_concurrency"] = std::thread::hardware_concurrency();
  // The gate MUST see this: a Debug build runs these hot loops several times
  // slower, and comparing a Debug run against a Release baseline would fail
  // every threshold for a reason that has nothing to do with the code.
#ifdef NDEBUG
  j["build"] = "release";
#else
  j["build"] = "debug";
#endif

  json arr = json::array();
  for (const BenchCase& c : cases) {
    json cj;
    cj["label"] = c.label;
    cj["ir_nodes"] = c.ir_nodes;
    cj["ir_edges"] = c.ir_edges;
    cj["display_nodes"] = c.display_nodes;
    cj["layout_boxes"] = c.layout_boxes;
    // 0 from peak_rss_bytes() means "the platform counter was unavailable", not
    // "no memory used" (core/Rss.h). Emitting it as 0 would let a gate read an
    // unmeasured host as a perfect memory score, so unknown is emitted as null.
    if (c.peak_rss_bytes != 0)
      cj["peak_rss_bytes"] = c.peak_rss_bytes;
    else
      cj["peak_rss_bytes"] = nullptr;

    json stages = json::array();
    for (const BenchStage& s : c.stages) {
      json sj;
      sj["name"] = s.name;
      sj["ms"] = s.ms;
      sj["min_ms"] = s.min_ms;
      sj["max_ms"] = s.max_ms;
      sj["repeats"] = s.repeats;
      stages.push_back(std::move(sj));
    }
    cj["stages"] = std::move(stages);
    arr.push_back(std::move(cj));
  }
  j["cases"] = std::move(arr);

  // Compact single line, no trailing newline — the CLI adds one, and `jq` gets
  // clean input either way. Same convention as build_report_json.
  return j.dump();
}

// --- CLI flag parsing --------------------------------------------------------
//
// Shared by the GUI binary and the headless netvis_bench target so the two can
// never disagree about what a flag means. Unrecognized arguments are ignored
// rather than rejected, because the GUI passes its own argv through here.
BenchOptions parse_bench_args(int argc, char** argv) {
  BenchOptions opt;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--bench-quick") {
      opt.quick = true;
      continue;
    }
    constexpr std::string_view kRepeats = "--bench-repeats=";
    if (a.substr(0, kRepeats.size()) == kRepeats) {
      // Clamped, not trusted: 0 repeats would divide by zero when taking the
      // median, and an absurd count would hang CI rather than fail it.
      const std::string digits(a.substr(kRepeats.size()));
      const long n = std::strtol(digits.c_str(), nullptr, 10);
      opt.repeats = n > 0 ? static_cast<uint32_t>(n < 999 ? n : 999) : 1;
      continue;
    }
    constexpr std::string_view kModel = "--bench-model=";
    if (a.substr(0, kModel.size()) == kModel)
      opt.model_paths.emplace_back(a.substr(kModel.size()));
  }
  return opt;
}

bool wants_bench(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--bench") return true;
    if (a.substr(0, 8) == "--bench-") return true;
  }
  return false;
}

}  // namespace netvis
