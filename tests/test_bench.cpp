// tests/test_bench.cpp — the benchmark harness contract (#97).
//
// engine/Bench.h freezes the numeric/structural contract this file tests:
// make_synthetic_model determinism + seed sensitivity, the collapse win that
// makes the 100k rung honest, compute_cost's known/unknown split under
// with_shapes, run_bench's per-stage shape, and build_bench_json's round-trip.
// engine/Bench.cpp is a SEPARATE file, written in parallel by another agent at
// the time this file was authored, so nothing here has been linked or run yet
// — only syntax-checked against the frozen header. The JSON key names assumed
// below (case/stage field names becoming JSON keys 1:1, "cases"/"stages" as
// the array container keys) follow the ONE existing precedent in this repo,
// engine/ReportJson.cpp, which Bench.h explicitly says build_bench_json
// mirrors ("like ReportJson"); confirm against the real implementation once
// it lands and adjust the key strings here if they differ.
//
// WHAT IS DELIBERATELY NOT TESTED: any absolute wall-clock duration. The
// harness measures TIME, and asserting "layout took < N ms" on a shared CI
// runner is exactly the kind of flaky assertion #97 exists to stop producing
// elsewhere in the repo. Every assertion here is on STRUCTURE, DETERMINISM,
// or HONESTY (known vs. unknown), never on how fast something ran.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/Result.h"
#include "core/Rss.h"
#include "engine/Bench.h"
#include "engine/CollapseTree.h"
#include "engine/CostModel.h"
#include "ir/IR.h"

using namespace netvis;
using json = nlohmann::json;

namespace {

// A content SIGNATURE of a synthetic model's main graph: op-type sequence,
// node-name sequence, and the producer-op -> consumer-op pairs for every
// input edge. Two independently generated models have independent
// StringArenas, so raw StringIds mean nothing across them — every field here
// goes through model.str(id) so the comparison is by CONTENT, the same rule
// ModelDiff follows for cross-model matching (engine/ModelDiff.h). Comparing
// this struct is the determinism/seed-sensitivity test's whole mechanism.
struct ModelSignature {
  uint32_t node_count = 0;
  uint32_t value_count = 0;
  std::vector<std::string> op_types;
  std::vector<std::string> node_names;
  std::vector<std::pair<std::string, std::string>> edges;  // (producer op, consumer op)

  bool operator==(const ModelSignature& o) const {
    return node_count == o.node_count && value_count == o.value_count &&
           op_types == o.op_types && node_names == o.node_names &&
           edges == o.edges;
  }
};

ModelSignature signature_of(const ir::Model& m) {
  ModelSignature sig;
  if (m.graphs.empty()) return sig;
  const ir::Graph& g = m.graphs[0];
  sig.node_count = static_cast<uint32_t>(g.nodes.size());
  sig.value_count = static_cast<uint32_t>(g.values.size());
  sig.op_types.reserve(g.nodes.size());
  sig.node_names.reserve(g.nodes.size());
  for (const ir::Node& n : g.nodes) {
    sig.op_types.emplace_back(m.str(n.op_type));
    sig.node_names.emplace_back(m.str(n.name));
  }
  for (const ir::Node& n : g.nodes) {
    for (uint32_t i = 0; i < n.inputs.count; ++i) {
      uint32_t slot = n.inputs.begin + i;
      if (slot >= g.edge_refs.size()) continue;  // defensive; never true for a
                                                   // well-formed generated model
      uint32_t vidx = g.edge_refs[slot];
      if (vidx >= g.values.size()) continue;
      int32_t prod = g.values[vidx].producer;
      std::string prod_op =
          (prod >= 0 && static_cast<uint32_t>(prod) < g.nodes.size())
              ? std::string(m.str(g.nodes[static_cast<uint32_t>(prod)].op_type))
              : std::string("<input>");
      sig.edges.emplace_back(std::move(prod_op), std::string(m.str(n.op_type)));
    }
  }
  return sig;
}

}  // namespace

// --- make_synthetic_model: determinism + seed sensitivity -------------------

TEST_CASE("#97 make_synthetic_model: same seed -> identical structure by content") {
  SyntheticSpec spec;
  spec.nodes = 200;
  spec.branch = 3;
  spec.block_size = 0;
  spec.seed = 42;
  spec.with_shapes = true;

  ir::Model a = make_synthetic_model(spec);
  ir::Model b = make_synthetic_model(spec);

  ModelSignature sa = signature_of(a);
  ModelSignature sb = signature_of(b);
  REQUIRE(sa.node_count > 0);
  CHECK(sa == sb);
}

TEST_CASE("#97 make_synthetic_model: different seed changes structure when branch>1") {
  // branch>1 gives the generator a real choice to make per node (which prior
  // nodes feed a multi-fanout node) — that choice is where a seed can matter.
  // A pure chain (branch==1, block_size==0) has no such choice at all; see the
  // degenerate-case test below, which asserts the opposite on purpose.
  SyntheticSpec spec_a;
  spec_a.nodes = 200;
  spec_a.branch = 3;
  spec_a.block_size = 0;
  spec_a.seed = 1;
  spec_a.with_shapes = true;

  SyntheticSpec spec_b = spec_a;
  spec_b.seed = 2;

  ir::Model a = make_synthetic_model(spec_a);
  ir::Model b = make_synthetic_model(spec_b);

  // Same rung size, but the seed must move at least one content axis —
  // otherwise `seed` is a dead field and every rung of the ladder reproduces
  // the same graph shape under a different label, which would make a
  // regression-gate re-run of a "different" case meaningless.
  CHECK_FALSE(signature_of(a) == signature_of(b));
}

TEST_CASE("#97 seed changes structure even for a pure chain (branch==1)") {
  // This test originally asserted the OPPOSITE — that branch==1 with
  // block_size==0 is seed-degenerate, on the theory that a pure chain leaves the
  // seed no fan-out decision to make. The implementation disagrees, and the
  // implementation is right: the seed also rotates which op the block template
  // starts on, so two seeds produce different op SEQUENCES over the same
  // topology. That is the more useful behavior — a seed that changed nothing for
  // a whole class of specs would be a trap for anyone building a ladder rung.
  //
  // Pinned deliberately: if a future change makes the chain seed-degenerate
  // again, two ladder rungs that differ only by seed would silently become the
  // same measurement.
  SyntheticSpec spec_a;
  spec_a.nodes = 50;
  spec_a.branch = 1;
  spec_a.block_size = 0;
  spec_a.seed = 1;

  SyntheticSpec spec_b = spec_a;
  spec_b.seed = 2;

  ir::Model a = make_synthetic_model(spec_a);
  ir::Model b = make_synthetic_model(spec_b);
  CHECK_FALSE(signature_of(a) == signature_of(b));
}

// --- CollapseTree: the whole reason block_size exists ------------------------

TEST_CASE("#97 block_size>0 collapses hard; block_size==0 has ~nothing to fold") {
  // THE most important test in this file (see #97 task notes): if the
  // synthetic ladder's block_size rungs do not actually collapse, the 100k
  // rung measures a shape NetVis never displays, and every downstream
  // regression-gate number (#98-#101) is comparing against a fiction.
  SyntheticSpec with_blocks;
  with_blocks.nodes = 500;
  with_blocks.branch = 2;
  with_blocks.block_size = 10;  // ~50 repeated 10-node blocks
  with_blocks.seed = 7;
  with_blocks.with_shapes = true;

  ir::Model blocky = make_synthetic_model(with_blocks);
  REQUIRE(!blocky.graphs.empty());
  const uint32_t raw_nodes = static_cast<uint32_t>(blocky.graphs[0].nodes.size());
  REQUIRE(raw_nodes > 0);

  CollapseTree collapse;
  collapse.build(blocky, 0);

  // DETECTION is what block_size must guarantee, and it is asserted on its own:
  // if no group is found, the ladder's repeated blocks are not structurally
  // identical and every downstream number is comparing against a fiction.
  REQUIRE_FALSE(collapse.groups().empty());
  uint32_t grouped_members = 0;
  for (const CollapseGroup& g : collapse.groups())
    grouped_members += static_cast<uint32_t>(g.member_nodes.size());
  CHECK(grouped_members >= raw_nodes / 2);

  // FOLDING is a separate question, and the default is NOT folded: build()
  // leaves every group expanded (CollapseTree.cpp — "we never hide nodes by
  // default", changed because a default-collapsed view made large models look
  // like they were missing nodes). So the display list starts at the full node
  // count, and that is exactly the shape the bench's layout stage should time,
  // because it is what a user gets on first paint.
  const uint32_t display_expanded =
      static_cast<uint32_t>(collapse.display_nodes().size());
  CHECK(display_expanded == raw_nodes);

  // Collapsing explicitly must then fold hard — this is the proof that the
  // detected groups are real and usable, not just present in the table.
  CHECK(collapse.collapse_all());
  const uint32_t display_collapsed =
      static_cast<uint32_t>(collapse.display_nodes().size());
  CHECK(display_collapsed < raw_nodes);
  CHECK(display_collapsed <= raw_nodes / 2);

  // Converse: no repeated blocks -> little or nothing to collapse. A tiny
  // amount of incidental folding is allowed (a couple of adjacent nodes could
  // coincidentally share a structural fingerprint), but the display list must
  // stay close to the raw node count, unlike the case above.
  SyntheticSpec no_blocks = with_blocks;
  no_blocks.block_size = 0;
  ir::Model chain = make_synthetic_model(no_blocks);
  REQUIRE(!chain.graphs.empty());
  const uint32_t raw_nodes2 = static_cast<uint32_t>(chain.graphs[0].nodes.size());
  REQUIRE(raw_nodes2 > 0);

  CollapseTree collapse2;
  collapse2.build(chain, 0);
  const uint32_t display_nodes2 =
      static_cast<uint32_t>(collapse2.display_nodes().size());
  CHECK(display_nodes2 >= raw_nodes2 * 9 / 10);  // at most ~10% folded away
}

// --- with_shapes: compute_cost's known/unknown split -------------------------

TEST_CASE("#97 with_shapes=true gives compute_cost real known FLOPs") {
  SyntheticSpec shaped;
  shaped.nodes = 100;
  shaped.branch = 2;
  shaped.block_size = 0;
  shaped.seed = 3;
  shaped.with_shapes = true;

  ir::Model m = make_synthetic_model(shaped);
  REQUIRE(!m.graphs.empty());
  CostReport r = compute_cost(m, 0);
  CHECK(r.nodes_total > 0);
  // At least SOME nodes must resolve known FLOPs, or with_shapes is not doing
  // what its name promises and the whole ladder is reporting honest-unknowns
  // for a spec that claims otherwise.
  CHECK(r.nodes_flops_known > 0);
}

TEST_CASE("#97 with_shapes=false still builds and computes cost (honest-unknown, not a crash)") {
  SyntheticSpec unshaped;
  unshaped.nodes = 100;
  unshaped.branch = 2;
  unshaped.block_size = 0;
  unshaped.seed = 3;
  unshaped.with_shapes = false;

  ir::Model m = make_synthetic_model(unshaped);
  REQUIRE(!m.graphs.empty());
  CHECK(m.graphs[0].nodes.size() > 0);
  // Must not crash and must still report a structurally sane report; whether
  // any individual node's FLOPs happen to resolve without ValueInfo shapes is
  // an implementation detail this test deliberately does not pin down.
  CostReport r = compute_cost(m, 0);
  CHECK(r.nodes_total == m.graphs[0].nodes.size());
}

// --- run_bench: per-stage shape, no timing assertions ------------------------

TEST_CASE("#97 run_bench: one BenchCase per rung, all frozen stages present, timings sane") {
  BenchOptions opts;
  opts.repeats = 1;  // keep this the fastest possible run; it executes on every PR

  SyntheticSpec tiny;
  tiny.nodes = 64;
  tiny.branch = 2;
  tiny.block_size = 0;
  tiny.seed = 11;
  tiny.with_shapes = true;
  opts.ladder = {tiny};

  Result<std::vector<BenchCase>> result = run_bench(opts);
  REQUIRE_MESSAGE(result, "run_bench failed");
  const std::vector<BenchCase>& cases = result.value();

  REQUIRE(cases.size() == 1);
  const BenchCase& c = cases[0];
  CHECK(c.ir_nodes > 0);
  CHECK(c.display_nodes > 0);

  // kStageParse is deliberately NOT expected here. A synthetic rung is generated
  // in memory and never parsed, and Bench.h requires the stage to be OMITTED
  // rather than reported as a fake 0 ms — a zero would read as "parsing is free"
  // in the gate's table. It appears only for real files from
  // BenchOptions::model_paths.
  const char* kExpectedStages[] = {
      kStageCollapse, kStageShapes, kStageLayout, kStageCost, kStageVisibleScan,
  };
  for (const BenchStage& s : c.stages)
    CHECK_MESSAGE(std::string(s.name) != std::string(kStageParse),
                  "a synthetic rung must not report a parse stage");
  for (const char* expected : kExpectedStages) {
    bool found = false;
    for (const BenchStage& s : c.stages) {
      if (s.name != expected) continue;
      found = true;
      CHECK(s.repeats == opts.repeats);
      // The median must lie within [min, max] by construction — a violation
      // here means the stage's min/max/median bookkeeping disagrees with
      // itself, independent of what the actual numbers are.
      CHECK(s.min_ms <= s.ms);
      CHECK(s.ms <= s.max_ms);
      break;
    }
    // std::string, not the bare const char* — doctest streams a raw pointer as
    // an address, which turns a useful failure into a hex number.
    CHECK_MESSAGE(found, "missing frozen stage in BenchCase::stages: "
                             << std::string(expected));
  }
}

// --- build_bench_json: schema + round-trip -----------------------------------

TEST_CASE("#97 build_bench_json: schema tag, frozen stage-name spelling, case/stage round-trip") {
  BenchOptions opts;
  opts.repeats = 1;
  SyntheticSpec tiny;
  tiny.nodes = 32;
  tiny.branch = 2;
  tiny.block_size = 0;
  tiny.seed = 5;
  tiny.with_shapes = true;
  opts.ladder = {tiny};

  Result<std::vector<BenchCase>> result = run_bench(opts);
  REQUIRE_MESSAGE(result, "run_bench failed");
  const std::vector<BenchCase>& cases = result.value();
  REQUIRE(cases.size() == 1);

  std::string text = build_bench_json(cases);
  json j = json::parse(text);  // throws (-> test failure) on malformed JSON
  CHECK(j.at("schema") == kBenchSchema);

  // Pin the frozen constants' literal spelling: a stage rename would still
  // round-trip below (the loop reads whatever name IS there), so that loop
  // alone can't catch a rename — this is the guard the header's "renaming is
  // not safe" warning is asking for.
  CHECK(std::string(kStageParse) == "parse");
  CHECK(std::string(kStageCollapse) == "collapse");
  CHECK(std::string(kStageShapes) == "shape_infer");
  CHECK(std::string(kStageLayout) == "layout");
  CHECK(std::string(kStageCost) == "cost");
  CHECK(std::string(kStageVisibleScan) == "visible_scan");

  REQUIRE(j.at("cases").is_array());
  REQUIRE(j.at("cases").size() == 1);
  const json& jc = j["cases"][0];
  CHECK(jc.at("label") == cases[0].label);
  CHECK(jc.at("ir_nodes") == cases[0].ir_nodes);
  CHECK(jc.at("ir_edges") == cases[0].ir_edges);
  CHECK(jc.at("display_nodes") == cases[0].display_nodes);
  CHECK(jc.at("layout_boxes") == cases[0].layout_boxes);

  REQUIRE(jc.at("stages").is_array());
  REQUIRE(jc["stages"].size() == cases[0].stages.size());
  for (size_t i = 0; i < cases[0].stages.size(); ++i) {
    const BenchStage& s = cases[0].stages[i];
    const json& js = jc["stages"][i];
    CHECK(js.at("name") == s.name);
    CHECK(js.at("ms").get<double>() == doctest::Approx(s.ms));
    CHECK(js.at("min_ms").get<double>() == doctest::Approx(s.min_ms));
    CHECK(js.at("max_ms").get<double>() == doctest::Approx(s.max_ms));
    CHECK(js.at("repeats") == s.repeats);
  }
}

// --- peak_rss_bytes: deliberately weak, environment-dependent ---------------

TEST_CASE("#97 peak_rss_bytes: honest 0-or-plausible; peak >= current when known") {
  // Deliberately weak: RSS is environment-dependent (container cgroup limits,
  // CI runner memory pressure, sanitizer instrumentation inflating everything)
  // so a specific byte count would be flaky by construction. The two
  // invariants Rss.h actually promises are (a) 0 always means "the platform
  // counter is unavailable", never a fabricated measurement, and (b) when the
  // counter IS available, the high-water mark can never read BELOW the
  // current snapshot — peak is monotonic non-decreasing for the process
  // lifetime (Rss.h's PLATFORM NOTE). That is the whole of what is checked.
  // ORDER MATTERS, and getting it wrong made this test flaky. It used to read
  // peak first and current second, which is a time-of-check race against the
  // process itself: anything allocated between the two reads raises `current`
  // above a `peak` that was sampled before the growth. CI caught it exactly
  // once, off by a single 4 KiB page (44249088 >= 44253184).
  //
  // Reading current FIRST makes the assertion sound rather than merely
  // usually-true: peak is monotonic non-decreasing, so a peak sampled LATER
  // necessarily covers any current sampled earlier, whatever the process did in
  // between.
  uint64_t current = current_rss_bytes();
  uint64_t peak = peak_rss_bytes();
  if (peak == 0) {
    CHECK(peak == 0);  // unknown platform counter; nothing further to assert
  } else {
    CHECK(peak >= current);
  }
}
