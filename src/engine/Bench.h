// engine/Bench.h — the benchmark + perf-regression harness (#97).
//
// DECISION (v0.9.3, Pillar 2 of docs/v1.0-plan.md): NetVis claims instant
// multi-GB open and smooth 100k-node graphs. Until now nothing measured either
// claim — the README's performance table has no reproducing script in the repo,
// so its numbers cannot be checked or defended. This harness is the first task
// of the optimization pillar because #98–#101 cannot prove anything without it,
// and because it must exist BEFORE the v0.9.5 parsers so each one is measured as
// it lands.
//
// WHY IT IS HEADLESS AND SYNCHRONOUS. Every engine hot path — CollapseTree::build,
// compute_layout, infer_shapes_ext, compute_cost — is already a pure synchronous
// function that takes no JobSystem. Calling them directly (the way ReportJson's
// report_file already does) gives the harness single-threaded determinism for
// free: no thread-pool scheduling noise, no completion-queue ordering, nothing to
// make a run irreproducible. Going through ModelSession would buy nothing and
// add exactly the variance a regression gate must not have.
//
// WHAT IT DELIBERATELY DOES NOT MEASURE. A real rendered frame needs a GL context
// and a window, which CI does not have. Rather than fake a frame or skip the
// render pillar, the harness measures the part of the render cost that is pure
// and is the actual scaling risk: the per-frame scans over the layout that are
// O(TOTAL boxes/edges) rather than O(visible) (GraphCanvas hit-test, edge draw,
// search pulse). `visible_scan_ms` is a PROXY and is named and documented as
// one — it is not a frame time and must never be reported as fps.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/Result.h"
#include "engine/Layout.h"
#include "ir/IR.h"

namespace netvis {

// Schema tag for the emitted JSON, mirroring ReportJson's kReportSchema. Bump
// this when a field's MEANING changes; the gate script keys its comparison on it
// and must refuse a baseline written under a different schema rather than
// silently compare incomparable numbers.
inline constexpr const char* kBenchSchema = "netvis.bench.v1";

// --- Synthetic model generation ---------------------------------------------
//
// The fixture ladder is SYNTHETIC, not a downloaded model. A gate that depends on
// fetching a real multi-GB checkpoint is a gate that breaks in CI, and a
// committed 100k-node fixture would be a huge binary in the repo. Generation is
// deterministic from `seed`, so the same spec always produces byte-identical
// structure and any timing change is a code change, not an input change.
struct SyntheticSpec {
  uint32_t nodes = 1000;        // approximate node count (see note below)
  uint32_t branch = 2;          // fan-out per node; 1 => a pure chain
  uint32_t block_size = 0;      // >0 => emit repeated identical blocks of this
                                // size, so CollapseTree has something to collapse
                                // (the 100k-node case is only fast BECAUSE of
                                // collapsing; a ladder without repeats would
                                // measure a shape NetVis never actually shows)
  uint64_t seed = 1;
  bool with_shapes = true;      // populate ValueInfo shapes so cost/shape-infer
                                // have real work rather than honest-unknowns
};

// Build a deterministic synthetic graph. The node count is approximate: the
// generator rounds to whole blocks when block_size > 0. The exact realized counts
// are reported in BenchResult, and the gate compares against those, not the spec.
ir::Model make_synthetic_model(const SyntheticSpec& spec);

// --- Measurement -------------------------------------------------------------

// One timed stage. `ms` is the MEDIAN across repeats (see BenchOptions::repeats)
// — the median, not the mean, because a single scheduler hiccup on a shared CI
// runner skews a mean and the gate would then flag noise as a regression.
struct BenchStage {
  std::string name;
  double ms = 0.0;
  double min_ms = 0.0;   // fastest observed repeat
  double max_ms = 0.0;   // slowest observed repeat
  uint32_t repeats = 0;
};

// One rung of the ladder: every stage timed against one synthetic (or real)
// model, plus the structural counts the timings must be read against.
struct BenchCase {
  std::string label;             // "synthetic-1k", "synthetic-100k", a file name
  uint32_t ir_nodes = 0;
  uint32_t ir_edges = 0;
  uint32_t display_nodes = 0;    // after collapse — the number layout actually sees
  uint32_t layout_boxes = 0;     // after clone/dummy expansion; may exceed display
  std::vector<BenchStage> stages;
  uint64_t peak_rss_bytes = 0;   // 0 => the platform counter was unavailable
};

// Stage names are FROZEN string constants: the gate matches baseline to current
// by (case label, stage name), so a renamed stage silently drops out of the
// comparison instead of failing. Adding a stage is safe; renaming one is not.
inline constexpr const char* kStageParse = "parse";
inline constexpr const char* kStageCollapse = "collapse";
inline constexpr const char* kStageShapes = "shape_infer";
inline constexpr const char* kStageLayout = "layout";
inline constexpr const char* kStageCost = "cost";
inline constexpr const char* kStageVisibleScan = "visible_scan";
// #99: the same sweep at a ZOOMED-IN viewport. Two stages, not one, because the
// spatial index that makes culling O(visible) is only a win below roughly a
// quarter of the world — past that the grid walk visits nearly every item anyway
// and costs more than a flat scan. Measuring only the wide view would therefore
// report the whole optimization as a regression, and measuring only the zoomed
// view would hide a real regression at the zoom level a model first opens at.
// Both must be gated.
inline constexpr const char* kStageVisibleScanZoomed = "visible_scan_zoomed";

struct BenchOptions {
  // Median-of-N. 5 is the default: enough to reject a single outlier, cheap
  // enough that the 100k rung stays inside a sane CI budget.
  uint32_t repeats = 5;
  // The ladder. Empty => the default 1k/10k/100k rungs.
  std::vector<SyntheticSpec> ladder;
  // Optional real model files to time end-to-end alongside the synthetic ladder.
  std::vector<std::string> model_paths;
  // Skip the 100k rung. CI uses this when a fast signal matters more than the
  // top of the ladder; the gate then compares only the rungs actually present.
  bool quick = false;
};

// Parse the bench CLI flags out of argv:
//   --bench                 run the default ladder
//   --bench-quick           skip the 100k rung
//   --bench-repeats=N       median-of-N (clamped to [1, 999]; 0 would divide by
//                           zero when taking the median)
//   --bench-model=<path>    also time a real model file; repeatable
// Unrecognized arguments are ignored, so the GUI binary can share this parser
// without rejecting its own flags. Lives here rather than in a main() so the GUI
// entry point and the headless netvis_bench target cannot drift apart — two
// copies of a flag parser is two behaviours the gate could disagree about.
BenchOptions parse_bench_args(int argc, char** argv);

// True if any argument selects benchmark mode.
bool wants_bench(int argc, char** argv);

// Run every rung and return the results. Synchronous, single-threaded, no
// JobSystem, no GL. Deterministic for a fixed spec + repeats.
//
// IMPORTANT: the layout stage must measure compute_layout itself, NOT
// ModelSession's cached path — ModelSession consults an on-disk layout cache
// keyed on structure_hash before computing, so a second run would report a cache
// hit as a layout speedup. Call the engine function directly.
Result<std::vector<BenchCase>> run_bench(const BenchOptions& options);

// Serialize to the gate's JSON format (nlohmann::ordered_json, like ReportJson).
// Includes kBenchSchema, the host's logical core count, and the build type, so a
// baseline carries enough context to be rejected when it is not comparable.
std::string build_bench_json(const std::vector<BenchCase>& cases);

// --- The render-cost proxy ---------------------------------------------------
//
// Times the O(total) scans the canvas performs every frame: the box hit-test
// sweep and the edge cull sweep, over a layout, for a camera showing `visible_
// fraction` of the world. This is the number that must NOT grow with total node
// count once #99 lands — today it does, because culling is a linear AABB scan
// with no spatial index. Pure over LayoutResult; no GL, no ImGui, no view code.
//
// Declared here rather than in the view layer precisely so the harness stays in
// netvis_core and CI can run it headlessly.
double visible_scan_ms(const LayoutResult& layout, double visible_fraction,
                       uint32_t repeats);

}  // namespace netvis
