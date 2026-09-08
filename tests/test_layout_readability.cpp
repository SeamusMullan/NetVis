// SPDX-License-Identifier: Apache-2.0
// tests/test_layout_readability.cpp — v0.2.0 layout readability (kVersion v3).
//
// Covers multi-consumer source DUPLICATION and long-edge DUMMY routing added to
// compute_layout: a shared source is cloned once per consumer so each clone sits
// next to a single user (killing the top-row hairball), and edges spanning many
// layers still lay out. Also re-asserts determinism now that the internal
// layout-node model (real + clones + dummies) drives coordinate assignment.
//
// v0.9.x (#153) appends the constant-CONE cases at the bottom of this file: the
// duplication above only ever helped a constant that was a single in-degree-0
// node, and a real export's constants arrive as short chains.
#include <doctest/doctest.h>

#include <algorithm>
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

// ---------------------------------------------------------------------------
// #153 — constants belong on the layer BEFORE the node that uses them.
//
// The one-hop version of that rule (move every in-degree-0 source to
// min(consumer layer) - 1) shipped long before this issue, and it is not what a
// real export needs: a constant there is usually a small CONE of nodes
// (Constant -> Cast -> Cast -> ... -> the op that uses the value), and only the
// cone's ROOT is a source. Everything above the root kept its longest-path rank
// and stayed pinned to the top of the drawing — the "all the constants are at
// the start" complaint. These cases pin the cone behaviour, the multi-consumer
// rule that was chosen for it, and the two things it must not break: a Cast that
// sits on the real activation path must NOT be dragged down with the constants,
// and the layering must stay acyclic.
// ---------------------------------------------------------------------------
namespace {

// Builder for the fixtures below. A linear chain op_0 -> ... -> op_(N-1) (op_0
// reads a graph input, so it has no display predecessor but is NOT a constant),
// plus one constant cone `Constant -> Cast -> Cast` whose tail feeds every op in
// `consumers`. Returns the model; `out_cone` receives the cone's IR node indices
// root-first.
ir::Model make_const_cone_model(int N, const std::vector<int>& consumers,
                                std::vector<uint32_t>* out_cone) {
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
  // Node indices are assigned in the order the nodes are pushed: the chain
  // first (0..N-1), then the three cone nodes (N, N+1, N+2).
  const uint32_t k_root = static_cast<uint32_t>(N);
  const uint32_t k_mid = k_root + 1;
  const uint32_t k_tail = k_mid + 1;

  const uint32_t in_val = add_val("input", -1);  // graph input: no producer
  std::vector<uint32_t> oval(N);
  for (int i = 0; i < N; ++i)
    oval[i] = add_val("o" + std::to_string(i), i);
  const uint32_t kv0 = add_val("k0", static_cast<int32_t>(k_root));
  const uint32_t kv1 = add_val("k1", static_cast<int32_t>(k_mid));
  const uint32_t kv2 = add_val("k2", static_cast<int32_t>(k_tail));

  auto add_node = [&](const char* op, const std::string& name,
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
    g.nodes.push_back(n);
  };

  for (int i = 0; i < N; ++i) {
    std::vector<uint32_t> ins;
    ins.push_back(i == 0 ? in_val : oval[i - 1]);
    if (std::find(consumers.begin(), consumers.end(), i) != consumers.end())
      ins.push_back(kv2);  // this op also reads the constant cone's output
    add_node("Add", "op_" + std::to_string(i), ins, oval[i]);
  }
  // The cone. Constant has no inputs; Cast categorizes to OpCategory::Tensor,
  // so both halves of the const-like predicate are exercised.
  add_node("Constant", "k_root", {}, kv0);
  add_node("Cast", "k_mid", {kv0}, kv1);
  add_node("Cast", "k_tail", {kv1}, kv2);

  g.graph_inputs.push_back(in_val);
  g.graph_outputs.push_back(oval[N - 1]);
  if (out_cone) *out_cone = {k_root, k_mid, k_tail};
  return m;
}

// Display id of a leaf display node wrapping IR node `ir_node`, or UINT32_MAX.
uint32_t display_of(const CollapseTree& collapse, uint32_t ir_node) {
  const std::vector<DisplayNode>& d = collapse.display_nodes();
  for (size_t i = 0; i < d.size(); ++i)
    if (!d[i].is_group && d[i].ir_node == ir_node)
      return static_cast<uint32_t>(i);
  return UINT32_MAX;
}

// Layer of the FIRST box carrying `did`. Only used on nodes that are never
// cloned (cone interiors have a predecessor, so duplication skips them).
int32_t layer_of(const LayoutResult& r, uint32_t did) {
  for (const NodeBox& b : r.boxes)
    if (b.display_id == did) return b.layer;
  return -1;
}

}  // namespace

TEST_CASE("#153 a constant CONE sinks to the layer above its consumer") {
  // op_15 is the only consumer of the cone. Longest-path layering puts the cone
  // on layers 0/1/2 while op_15 is on layer 15, so before the fix the cone tail
  // fired a 13-layer edge across the whole drawing. Each cone node must now sit
  // exactly one layer above the next, with the tail one layer above op_15.
  const int N = 16;
  std::vector<uint32_t> cone;
  ir::Model m = make_const_cone_model(N, {N - 1}, &cone);
  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult r = compute_layout(m, 0, collapse, headless_size, {}, nullptr);

  const uint32_t d_consumer = display_of(collapse, static_cast<uint32_t>(N - 1));
  const uint32_t d_root = display_of(collapse, cone[0]);
  const uint32_t d_mid = display_of(collapse, cone[1]);
  const uint32_t d_tail = display_of(collapse, cone[2]);
  REQUIRE(d_consumer != UINT32_MAX);
  REQUIRE(d_tail != UINT32_MAX);

  const int32_t l_consumer = layer_of(r, d_consumer);
  REQUIRE(l_consumer > 3);  // the chain really is deep
  CHECK(layer_of(r, d_tail) == l_consumer - 1);
  CHECK(layer_of(r, d_mid) == l_consumer - 2);
  CHECK(layer_of(r, d_root) == l_consumer - 3);

  // And the user-visible consequence: no edge sprays across the canvas.
  float max_span = 0.0f;
  for (const EdgeCurve& e : r.edges) {
    float s = e.p3.y - e.p0.y;
    if (s < 0) s = -s;
    max_span = std::max(max_span, s);
  }
  // One layer gap = node height (40) + rank_sep (60) = 100. Before the fix the
  // cone-tail edge alone spanned ~13 of those.
  CHECK(max_span < 100.0f * 3.0f);
}

TEST_CASE("#153 a shared constant cone sits above its EARLIEST consumer") {
  // THE MULTI-CONSUMER DECISION, pinned. A cone feeding op_4 and op_12 cannot be
  // adjacent to both. It is placed above the EARLIEST consumer rather than
  // duplicated per consumer: duplication is only safe for in-degree-0 sources
  // (a clone needs no producer above it), and a clone of a cone INTERIOR node
  // would be stranded with no incoming edge. The cost is that op_12 keeps one
  // long edge, which this case also states out loud.
  const int N = 16;
  std::vector<uint32_t> cone;
  ir::Model m = make_const_cone_model(N, {4, 12}, &cone);
  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult r = compute_layout(m, 0, collapse, headless_size, {}, nullptr);

  const int32_t l_early = layer_of(r, display_of(collapse, 4u));
  const int32_t l_late = layer_of(r, display_of(collapse, 12u));
  const int32_t l_tail = layer_of(r, display_of(collapse, cone[2]));
  REQUIRE(l_early > 0);
  REQUIRE(l_late > l_early);
  CHECK(l_tail == l_early - 1);   // adjacent to the earliest consumer
  CHECK(l_tail < l_late - 1);     // and therefore NOT adjacent to the later one

  // The cone is shared, so exactly one box carries each cone node's display id
  // (no clones were made for the interior).
  int tail_boxes = 0;
  for (const NodeBox& b : r.boxes)
    if (b.display_id == display_of(collapse, cone[2])) ++tail_boxes;
  CHECK(tail_boxes == 1);
}

TEST_CASE("#153 a Cast on the activation path is NOT sunk with the constants") {
  // The const-like seed is node_is_const_source(), which is true for ANY op in
  // OpCategory::Tensor — Cast included. That predicate alone would sink a Cast
  // that sits on the real data path, dragging live compute to the bottom of the
  // drawing. The cone rule therefore also demands that every display predecessor
  // be const-like. Here op_4 -> cast -> op_10: the Cast's predecessor is a
  // compute node, so the Cast must keep its longest-path layer (5) and must not
  // slide down to 9, one above op_10.
  const int N = 16;
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
  const uint32_t cast_node = static_cast<uint32_t>(N);
  const uint32_t in_val = add_val("input", -1);
  std::vector<uint32_t> oval(N);
  for (int i = 0; i < N; ++i) oval[i] = add_val("o" + std::to_string(i), i);
  const uint32_t cast_val = add_val("cast", static_cast<int32_t>(cast_node));

  auto add_node = [&](const char* op, const std::string& name,
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
    g.nodes.push_back(n);
  };
  for (int i = 0; i < N; ++i) {
    std::vector<uint32_t> ins;
    ins.push_back(i == 0 ? in_val : oval[i - 1]);
    if (i == 10) ins.push_back(cast_val);
    add_node("Add", "op_" + std::to_string(i), ins, oval[i]);
  }
  add_node("Cast", "live_cast", {oval[4]}, cast_val);

  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult r = compute_layout(m, 0, collapse, headless_size, {}, nullptr);

  const int32_t l_src = layer_of(r, display_of(collapse, 4u));
  const int32_t l_cast = layer_of(r, display_of(collapse, cast_node));
  const int32_t l_dst = layer_of(r, display_of(collapse, 10u));
  REQUIRE(l_dst > l_src + 2);
  CHECK(l_cast == l_src + 1);   // stayed with its producer
  CHECK(l_cast < l_dst - 1);    // was NOT sunk to sit above op_10
}

TEST_CASE("#153 sinking constants keeps the layering acyclic and deterministic") {
  // The pull-down raises layers, so the invariant it could plausibly break is
  // that every edge still runs strictly downward. Assert it geometrically (that
  // is what the user sees) on a fixture with several cones at different depths,
  // and re-pin determinism, since the pass is order-sensitive by construction.
  std::vector<uint32_t> cone;
  ir::Model m = make_const_cone_model(20, {3, 9, 14, 19}, &cone);
  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult a = compute_layout(m, 0, collapse, headless_size, {}, nullptr);
  LayoutResult b = compute_layout(m, 0, collapse, headless_size, {}, nullptr);

  REQUIRE(!a.edges.empty());
  // p0 is always the upper endpoint and p3 the lower one (reversal is carried in
  // the `reversed` flag, not by swapping the points), so a valid layering means
  // p3.y is never above p0.y.
  for (const EdgeCurve& e : a.edges)
    CHECK(e.p3.y >= e.p0.y);

  REQUIRE(a.boxes.size() == b.boxes.size());
  for (size_t i = 0; i < a.boxes.size(); ++i) {
    CHECK(a.boxes[i].display_id == b.boxes[i].display_id);
    CHECK(a.boxes[i].layer == b.boxes[i].layer);
    CHECK(a.boxes[i].pos.x == doctest::Approx(b.boxes[i].pos.x));
    CHECK(a.boxes[i].pos.y == doctest::Approx(b.boxes[i].pos.y));
  }
}
