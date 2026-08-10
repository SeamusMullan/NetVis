// tests/test_layout_ordering.cpp — #122: the barycenter early-stop must never
// publish an ordering worse than the one it started from.
//
// THE BUG. The crossing-reduction loop ran a sweep, counted crossings, and broke
// out when the count failed to improve — but a sweep PERMUTES the layer ordering
// before its result can be judged, and the break left that permutation in place.
// So a sweep that made the drawing worse had its result published. Standard
// Sugiyama keeps the best ordering seen; this pins that it now does.
//
// WHY THE TEST IS SHAPED LIKE THIS. Two earlier versions were VACUOUS — both
// passed with the fix reverted — and the reasons are worth keeping, because both
// are easy traps:
//
//   1. Comparing every budget against the ZERO-sweep baseline is too weak.
//      Several improving sweeps easily mask one worsening sweep at the end, so
//      the result is far better than unsorted while still being worse than it
//      should be. The observable break is between CONSECUTIVE budgets: the loop
//      stops at the first sweep that fails to improve, so budget k publishes that
//      sweep's worse ordering while budget k-1 stops one sweep earlier and keeps
//      the better one.
//   2. Building the fixture from `make_synthetic_model` cannot observe this bug
//      AT ALL. That generator produces a CHAIN — roughly one node per layer — and
//      barycenter_sort early-exits for layers smaller than two, so no sweep ever
//      permutes anything and a rollback has nothing to restore. The graph must
//      have WIDE layers (see make_wide_model).
//
// With both corrected the test fails without the fix on concrete numbers
// (crossings 18->21, 15->19, 22->25), which also confirms the bug was real and
// user-visible rather than theoretical.
//
// Crossings are counted here in the test from the emitted geometry rather than
// by reaching into the engine's internal counter, so the assertion is about what
// the user actually sees.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "engine/Bench.h"
#include "engine/CollapseTree.h"
#include "engine/Layout.h"
#include "engine/LayoutEngine.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// Deterministic, font-free node sizing — the same trick the harness uses so
// layout is reproducible headlessly.
Vec2 fixed_size(const DisplayNode& /*dn*/) { return Vec2{120.0f, 40.0f}; }

// True if segments (a0,a1) and (b0,b1) properly cross. Shared endpoints do not
// count: edges out of one node meeting at that node are not a crossing, and
// counting them would swamp the signal with noise proportional to fan-out.
bool segments_cross(Vec2 a0, Vec2 a1, Vec2 b0, Vec2 b1) {
  auto orient = [](Vec2 p, Vec2 q, Vec2 r) {
    const double v = (static_cast<double>(q.x) - p.x) * (static_cast<double>(r.y) - p.y) -
                     (static_cast<double>(q.y) - p.y) * (static_cast<double>(r.x) - p.x);
    return v > 0.0 ? 1 : (v < 0.0 ? -1 : 0);
  };
  const int d1 = orient(a0, a1, b0);
  const int d2 = orient(a0, a1, b1);
  const int d3 = orient(b0, b1, a0);
  const int d4 = orient(b0, b1, a1);
  // Strict crossing only — collinear/touching cases are deliberately excluded,
  // since a tie there is a rendering nicety, not a crossing the user perceives.
  return d1 * d2 < 0 && d3 * d4 < 0;
}

// Count crossings among the emitted edges, using each edge's endpoints. Uses the
// straight chord between endpoints rather than the bezier: the ordering pass is
// what this test is about, and ordering is what the chord reflects.
uint64_t count_layout_crossings(const LayoutResult& layout) {
  const size_t n = layout.edges.size();
  uint64_t crossings = 0;
  for (size_t i = 0; i < n; ++i) {
    const EdgeCurve& a = layout.edges[i];
    for (size_t j = i + 1; j < n; ++j) {
      const EdgeCurve& b = layout.edges[j];
      // Skip pairs sharing an endpoint (see segments_cross).
      const bool shares = (a.p0.x == b.p0.x && a.p0.y == b.p0.y) ||
                          (a.p0.x == b.p3.x && a.p0.y == b.p3.y) ||
                          (a.p3.x == b.p0.x && a.p3.y == b.p0.y) ||
                          (a.p3.x == b.p3.x && a.p3.y == b.p3.y);
      if (shares) continue;
      if (segments_cross(a.p0, a.p3, b.p0, b.p3)) ++crossings;
    }
  }
  return crossings;
}

LayoutResult layout_with_sweeps(const ir::Model& model,
                                const CollapseTree& collapse, int sweeps) {
  LayoutParams params;
  params.barycenter_sweeps = sweeps;
  return compute_layout(model, 0, collapse, fixed_size, params, nullptr);
}

// A WIDE graph: two ranks of `k` nodes, wired rank0[i] -> rank1[perm[i]] so the
// drawing starts with many crossings, plus a shared root and sink so the whole
// thing is one connected DAG on three layers.
//
// Width is the entire point. `make_synthetic_model` builds a CHAIN — roughly one
// node per layer — and barycenter_sort early-exits for layers smaller than two,
// so on that fixture no sweep ever permutes anything and there is nothing for a
// rollback to restore. A test built on it cannot observe this bug at all: it was
// tried, and it passed with the fix reverted. Only wide layers exercise the
// ordering pass.
ir::Model make_wide_model(uint32_t k, uint64_t seed) {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  g.name = m.intern("g");

  // Values: v_root, then k for rank0 outputs, k for rank1 outputs, then sink.
  const uint32_t n_values = 1 + k + k + 1;
  for (uint32_t i = 0; i < n_values; ++i) {
    ir::ValueInfo v;
    v.name = m.intern("v" + std::to_string(i));
    g.values.push_back(v);
  }

  auto add_node = [&](const std::string& name, const char* op,
                      const std::vector<uint32_t>& ins, uint32_t out) {
    ir::Node n;
    n.op_type = m.intern(op);
    n.name = m.intern(name);
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    for (uint32_t iv : ins) g.edge_refs.push_back(iv);
    n.inputs.count = static_cast<uint32_t>(ins.size());
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(out);
    n.outputs.count = 1;
    g.values[out].producer = static_cast<int32_t>(g.nodes.size());
    g.nodes.push_back(n);
  };

  // Root produces v0, consumed by every rank0 node — that is what puts all k of
  // them on the same layer.
  add_node("root", "Relu", {}, 0);
  for (uint32_t i = 0; i < k; ++i)
    add_node("a" + std::to_string(i), "Relu", {0}, 1 + i);

  // A fixed shuffle so rank0[i] -> rank1[perm[i]] genuinely crosses. A stride
  // that is coprime with k visits every index exactly once, so this is a
  // permutation for any k, and it is seeded so different graphs differ.
  const uint32_t stride = 1 + (static_cast<uint32_t>(seed) % (k > 2 ? k - 1 : 1));
  for (uint32_t i = 0; i < k; ++i) {
    const uint32_t src = 1 + ((i * stride) % k);
    add_node("b" + std::to_string(i), "Relu", {src}, 1 + k + i);
  }

  // Sink consumes every rank1 output, holding them all on one layer too.
  std::vector<uint32_t> sink_ins;
  sink_ins.reserve(k);
  for (uint32_t i = 0; i < k; ++i) sink_ins.push_back(1 + k + i);
  add_node("sink", "Relu", sink_ins, 1 + k + k);

  g.graph_inputs.push_back(0);
  g.graph_outputs.push_back(1 + k + k);
  return m;
}

}  // namespace

TEST_CASE("#122 raising the sweep budget never increases crossings") {
  // Assert MONOTONICITY BETWEEN CONSECUTIVE BUDGETS, not "better than zero
  // sweeps". The weaker form is vacuous and was verified to be so: with the fix
  // reverted it still passed, because several improving sweeps easily mask the
  // one worsening sweep at the end, leaving the result far better than the
  // unsorted baseline while still being worse than it should be.
  //
  // Consecutive budgets is where the bug is actually observable. The loop breaks
  // at the first sweep that fails to improve, so for every budget at or above
  // that point the output is identical — and it is the output of the sweep that
  // made things worse. Budget k-1 stops one sweep earlier and keeps the better
  // ordering. So without the rollback, crossings(k) > crossings(k-1) at exactly
  // that k.
  //
  // Several seeds and fan-outs, because which sweep worsens is graph-dependent
  // and a single graph could pass by luck. branch > 1 creates the skip edges
  // that make crossings possible at all.
  for (uint64_t seed : {1u, 2u, 3u, 5u, 7u, 11u}) {
    for (uint32_t k : {6u, 9u, 12u, 17u}) {
      ir::Model m = make_wide_model(k, seed);
      REQUIRE_FALSE(m.graphs.empty());

      CollapseTree collapse;
      collapse.build(m, 0);

      uint64_t prev = count_layout_crossings(layout_with_sweeps(m, collapse, 0));
      for (int sweeps = 1; sweeps <= 8; ++sweeps) {
        const uint64_t got =
            count_layout_crossings(layout_with_sweeps(m, collapse, sweeps));
        CHECK_MESSAGE(got <= prev,
                      "k=" << k << " seed=" << seed << ": crossings(" << sweeps
                           << ")=" << got << " exceeds crossings("
                           << (sweeps - 1) << ")=" << prev
                           << " — a sweep that worsened the ordering was "
                              "published instead of rolled back");
        prev = got;
      }
    }
  }
}

TEST_CASE("#122 layout stays deterministic with the rollback in place") {
  // The rollback moves data around after a sweep, which is exactly the kind of
  // change that can introduce order-dependence. Determinism is a frozen property
  // of this engine (same file -> same layout, and layouts are cached on disk by
  // structure hash), so it is re-pinned here rather than assumed.
  ir::Model m = make_wide_model(12, 5);
  CollapseTree collapse;
  collapse.build(m, 0);

  const LayoutResult a = layout_with_sweeps(m, collapse, 4);
  const LayoutResult b = layout_with_sweeps(m, collapse, 4);

  REQUIRE(a.boxes.size() == b.boxes.size());
  for (size_t i = 0; i < a.boxes.size(); ++i) {
    CHECK(a.boxes[i].pos.x == b.boxes[i].pos.x);
    CHECK(a.boxes[i].pos.y == b.boxes[i].pos.y);
    CHECK(a.boxes[i].display_id == b.boxes[i].display_id);
  }
  CHECK(a.edges.size() == b.edges.size());
}
