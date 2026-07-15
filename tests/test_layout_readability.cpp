// tests/test_layout_readability.cpp — v0.2.0 layout readability (kVersion v3).
//
// Covers multi-consumer source DUPLICATION and long-edge DUMMY routing added to
// compute_layout: a shared source is cloned once per consumer so each clone sits
// next to a single user (killing the top-row hairball), and edges spanning many
// layers still lay out. Also re-asserts determinism now that the internal
// layout-node model (real + clones + dummies) drives coordinate assignment.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "engine/CollapseTree.h"
#include "engine/Layout.h"
#include "engine/LayoutEngine.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

Vec2 headless_size(const DisplayNode&) { return Vec2{120.0f, 40.0f}; }

// One Constant source feeding N Add consumers (each Add also consumes the prev
// Add's output so the consumers land on distinct layers). This is the exact
// hairball shape: without duplication the single Constant fans out N long edges.
ir::Model make_shared_source_model(int N) {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  auto add_val = [&](const std::string& nm, int32_t prod) {
    ir::ValueInfo v;
    v.name = m.intern(nm);
    v.producer = prod;
    g.values.push_back(v);
    return static_cast<uint32_t>(g.values.size() - 1);
  };

  // value 0 = the shared constant's output (producer = node 0).
  uint32_t cval = add_val("c", 0);
  std::vector<uint32_t> oval(N);
  for (int i = 0; i < N; ++i) oval[i] = add_val("o" + std::to_string(i), 1 + i);

  // node 0 = the shared Constant source (in-degree 0, out-degree N).
  {
    ir::Node n;
    n.op_type = m.intern("Constant");
    n.name = m.intern("shared_const");
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(cval);
    n.outputs.count = 1;
    g.nodes.push_back(n);
  }
  // nodes 1..N = a chain of Adds, each consuming the shared constant + prev out.
  for (int i = 0; i < N; ++i) {
    ir::Node n;
    n.op_type = m.intern("Add");
    n.name = m.intern("add_" + std::to_string(i));
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    if (i > 0) g.edge_refs.push_back(oval[i - 1]);
    g.edge_refs.push_back(cval);
    n.inputs.count = (i > 0) ? 2 : 1;
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(oval[i]);
    n.outputs.count = 1;
    g.nodes.push_back(n);
  }
  return m;
}

}  // namespace

TEST_CASE("multi-consumer source is duplicated per consumer") {
  const int N = 8;
  ir::Model m = make_shared_source_model(N);
  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult r = compute_layout(m, 0, collapse, headless_size, {}, nullptr);

  const size_t disp = collapse.display_nodes().size();
  REQUIRE(disp == static_cast<size_t>(N + 1));  // 1 const + N adds, no collapse

  // The Constant (display id for node 0) fans out to N consumers on distinct
  // layers, so it is cloned N-1 times => N extra boxes total.
  CHECK(r.boxes.size() == disp + static_cast<size_t>(N - 1));

  // Find the Constant's display id (the leaf display node whose ir_node == 0).
  int32_t const_disp = -1;
  for (size_t i = 0; i < collapse.display_nodes().size(); ++i) {
    const DisplayNode& d = collapse.display_nodes()[i];
    if (!d.is_group && d.ir_node == 0) { const_disp = static_cast<int32_t>(i); break; }
  }
  REQUIRE(const_disp >= 0);

  // Every clone box carries the Constant's display id; count them.
  int clones = 0;
  for (const NodeBox& b : r.boxes)
    if (b.display_id == static_cast<uint32_t>(const_disp)) ++clones;
  CHECK(clones == N);  // original + (N-1) clones all share the display id
}

TEST_CASE("duplication collapses the long constant fan-out") {
  const int N = 8;
  ir::Model m = make_shared_source_model(N);
  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult r = compute_layout(m, 0, collapse, headless_size, {}, nullptr);

  // With per-consumer clones, each constant->consumer edge should span ~1 layer.
  // A pinned shared constant would produce an edge spanning ~N layers. One layer
  // gap = node height (40) + rank_sep (60) = 100; allow a few layers of slack.
  REQUIRE(!r.edges.empty());
  float max_span = 0.0f;
  for (const EdgeCurve& e : r.edges) {
    float s = e.p3.y - e.p0.y;
    if (s < 0) s = -s;
    if (s > max_span) max_span = s;
  }
  CHECK(max_span < 100.0f * 3.0f);
}

TEST_CASE("layout with duplication + dummies is deterministic") {
  ir::Model m = make_shared_source_model(8);
  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult a = compute_layout(m, 0, collapse, headless_size, {}, nullptr);
  LayoutResult b = compute_layout(m, 0, collapse, headless_size, {}, nullptr);

  REQUIRE(a.boxes.size() == b.boxes.size());
  for (size_t i = 0; i < a.boxes.size(); ++i) {
    CHECK(a.boxes[i].display_id == b.boxes[i].display_id);
    CHECK(a.boxes[i].pos.x == doctest::Approx(b.boxes[i].pos.x));
    CHECK(a.boxes[i].pos.y == doctest::Approx(b.boxes[i].pos.y));
    CHECK(a.boxes[i].layer == b.boxes[i].layer);
  }
  REQUIRE(a.edges.size() == b.edges.size());
  for (size_t i = 0; i < a.edges.size(); ++i) {
    CHECK(a.edges[i].p0.x == doctest::Approx(b.edges[i].p0.x));
    CHECK(a.edges[i].p3.y == doctest::Approx(b.edges[i].p3.y));
  }
}

TEST_CASE("long-span edge from a NON-source inserts dummy waypoints") {
  // A deep linear chain n0->n1->...->n7 with an EXTRA edge n1->n7. n1 has
  // in-degree 1 (consumes n0), so it is NOT a duplicable source — the n1->n7
  // edge genuinely spans ~6 layers and MUST be routed through dummy nodes. This
  // is the path the earlier fan-out test could not reach (that source got
  // cloned, collapsing its span to 1). Dummies must NOT be emitted as boxes, and
  // the long edge's control points must bend (p1.x/p2.x pulled off the straight
  // p0.x/p3.x line by the waypoints).
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  const int N = 8;
  std::vector<uint32_t> ov(N);
  auto add_val = [&](const std::string& nm, int32_t prod) {
    ir::ValueInfo v;
    v.name = m.intern(nm);
    v.producer = prod;
    g.values.push_back(v);
    return static_cast<uint32_t>(g.values.size() - 1);
  };
  for (int i = 0; i < N; ++i) ov[i] = add_val("v" + std::to_string(i), i);
  for (int i = 0; i < N; ++i) {
    ir::Node n;
    n.op_type = m.intern("Relu");
    n.name = m.intern("n" + std::to_string(i));
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    if (i > 0) g.edge_refs.push_back(ov[i - 1]);  // chain edge
    // node N-1 ALSO consumes node 1's output => a long edge from a NON-source.
    if (i == N - 1) g.edge_refs.push_back(ov[1]);
    n.inputs.count = (i > 0) ? (i == N - 1 ? 2u : 1u) : 0u;
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(ov[i]);
    n.outputs.count = 1;
    g.nodes.push_back(n);
  }

  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult r = compute_layout(m, 0, collapse, headless_size, {}, nullptr);

  // No source is duplicable here (n0's only consumer is n1), so boxes exactly
  // match the display list — dummies are waypoints, never boxes.
  CHECK(r.boxes.size() == collapse.display_nodes().size());
  for (const NodeBox& b : r.boxes)
    CHECK(b.display_id < collapse.display_nodes().size());

  // The long n1->n7 edge must be present and its control points bent by dummies:
  // for a routed-through-dummies edge, p1.x/p2.x are pulled to waypoint centers,
  // so at least one differs from the straight p0.x/p3.x it would otherwise take.
  bool found_bent_long_edge = false;
  for (const EdgeCurve& e : r.edges) {
    float span = e.p3.y - e.p0.y;
    if (span < 0) span = -span;
    if (span > 100.0f * 2.0f) {  // spans more than ~2 layers => had dummies
      if (e.p1.x != doctest::Approx(e.p0.x).epsilon(0.001) ||
          e.p2.x != doctest::Approx(e.p3.x).epsilon(0.001))
        found_bent_long_edge = true;
    }
  }
  CHECK(found_bent_long_edge);

  CHECK(r.bounds_max.x >= r.bounds_min.x);
  CHECK(r.bounds_max.y >= r.bounds_min.y);
}
