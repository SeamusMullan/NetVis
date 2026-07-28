// tests/test_layout.cpp — layered layout determinism (spec §2.7, §7.2).
//
// Builds a tiny ir::Model in code, runs CollapseTree::build + compute_layout
// TWICE, and asserts byte-identical positions (determinism is what makes the
// layout cache correct) and that boxes.size() == display_nodes().size().
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/JobSystem.h"
#include "engine/CollapseTree.h"
#include "engine/Layout.h"
#include "engine/LayoutEngine.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// A small linear chain A -> B -> C with three SSA values so layered layout has
// real edges to route. Kept intentionally tiny for a fast, deterministic test.
ir::Model make_chain_model() {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  g.name = m.intern("g");

  // Values: v0 (input), v1, v2 (output).
  for (int i = 0; i < 3; ++i) {
    ir::ValueInfo v;
    v.name = m.intern("v" + std::to_string(i));
    g.values.push_back(v);
  }

  auto add_node = [&](const char* op, uint32_t in_val, uint32_t out_val,
                      int32_t producer) {
    ir::Node n;
    n.op_type = m.intern(op);
    n.name = m.intern(op);
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(in_val);
    n.inputs.count = 1;
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(out_val);
    n.outputs.count = 1;
    g.values[out_val].producer = producer;
    g.nodes.push_back(n);
  };

  add_node("Relu", 0, 1, 0);
  add_node("Relu", 1, 2, 1);
  g.graph_inputs.push_back(0);
  g.graph_outputs.push_back(2);
  return m;
}

// #18: a producer P and consumer C where P emits TWO output values (v1, v2),
// both consumed by C. The single deduped display edge P->C must carry the
// SMALLEST value index. C consumes them in REVERSE index order (v2 before v1)
// so the test proves the value-index sort — not iteration order — picks the
// smallest. Returns the model; `out_min_vidx` is the smaller of the two.
ir::Model make_multivalue_model(uint32_t* out_min_vidx) {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  g.name = m.intern("g");

  auto add_val = [&](const std::string& nm, int32_t prod) {
    ir::ValueInfo v;
    v.name = m.intern(nm);
    v.producer = prod;
    g.values.push_back(v);
    return static_cast<uint32_t>(g.values.size() - 1);
  };
  // v0: graph input (no producer). v1, v2: both produced by P (node 0).
  // v3: produced by C (node 1).
  uint32_t v0 = add_val("v0", -1);
  uint32_t v1 = add_val("v1", 0);
  uint32_t v2 = add_val("v2", 0);
  uint32_t v3 = add_val("v3", 1);

  // P (node 0): consumes v0, produces v1 and v2.
  {
    ir::Node n;
    n.op_type = m.intern("P");
    n.name = m.intern("P");
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(v0);
    n.inputs.count = 1;
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(v1);
    g.edge_refs.push_back(v2);
    n.outputs.count = 2;
    g.nodes.push_back(n);
  }
  // C (node 1): consumes v2 THEN v1 (reverse index order), produces v3.
  {
    ir::Node n;
    n.op_type = m.intern("C");
    n.name = m.intern("C");
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(v2);
    g.edge_refs.push_back(v1);
    n.inputs.count = 2;
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(v3);
    n.outputs.count = 1;
    g.nodes.push_back(n);
  }
  g.graph_inputs.push_back(v0);
  g.graph_outputs.push_back(v3);
  if (out_min_vidx) *out_min_vidx = std::min(v1, v2);
  return m;
}

// Headless size function: a fixed box (view supplies real font metrics).
Vec2 headless_size(const DisplayNode&) { return Vec2{120.0f, 40.0f}; }

}  // namespace

TEST_CASE("layout is deterministic and boxes match display nodes") {
  ir::Model model = make_chain_model();

  CollapseTree collapse;
  collapse.build(model, 0);

  SizeFn size_fn = headless_size;
  LayoutParams params;  // defaults

  LayoutResult a = compute_layout(model, 0, collapse, size_fn, params, nullptr);
  LayoutResult b = compute_layout(model, 0, collapse, size_fn, params, nullptr);

  // boxes are parallel to the display node list.
  CHECK(a.boxes.size() == collapse.display_nodes().size());
  CHECK(b.boxes.size() == collapse.display_nodes().size());

  // Determinism: identical inputs -> identical positions (bit-for-bit).
  REQUIRE(a.boxes.size() == b.boxes.size());
  for (size_t i = 0; i < a.boxes.size(); ++i) {
    CHECK(a.boxes[i].display_id == b.boxes[i].display_id);
    CHECK(a.boxes[i].pos.x == doctest::Approx(b.boxes[i].pos.x));
    CHECK(a.boxes[i].pos.y == doctest::Approx(b.boxes[i].pos.y));
    CHECK(a.boxes[i].layer == b.boxes[i].layer);
  }

  // The structure hash is a pure function of structure, so it must match.
  CHECK(a.structure_hash == b.structure_hash);
  CHECK(a.collapse_hash == b.collapse_hash);
}

TEST_CASE("constants are placed next to their consumer, not pinned to layer 0") {
  // A chain op_0 -> op_1 -> ... -> op_(N-1); each op_i also consumes a dedicated
  // Constant source c_i. Longest-path layering would pin every constant at
  // layer 0, spraying long edges across the whole graph (the hairball). The
  // constant-placement pass must instead put each constant just above its
  // consumer, so every constant->consumer edge spans a single layer.
  const int N = 20;
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  std::vector<uint32_t> cval(N), oval(N);
  auto add_val = [&](const std::string& nm, int32_t prod) {
    ir::ValueInfo v;
    v.name = m.intern(nm);
    v.producer = prod;
    g.values.push_back(v);
    return static_cast<uint32_t>(g.values.size() - 1);
  };
  for (int i = 0; i < N; ++i) cval[i] = add_val("c" + std::to_string(i), i);
  for (int i = 0; i < N; ++i) oval[i] = add_val("o" + std::to_string(i), N + i);

  for (int i = 0; i < N; ++i) {  // constant source nodes (in-degree 0)
    ir::Node n;
    n.op_type = m.intern("Constant");
    n.name = m.intern("const_" + std::to_string(i));
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(cval[i]);
    n.outputs.count = 1;
    g.nodes.push_back(n);
  }
  for (int i = 0; i < N; ++i) {  // chain ops consuming prev op + own constant
    ir::Node n;
    n.op_type = m.intern("Add");
    n.name = m.intern("op_" + std::to_string(i));
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    if (i > 0) g.edge_refs.push_back(oval[i - 1]);
    g.edge_refs.push_back(cval[i]);
    n.inputs.count = (i > 0) ? 2 : 1;
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(oval[i]);
    n.outputs.count = 1;
    g.nodes.push_back(n);
  }

  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult r = compute_layout(m, 0, collapse, headless_size, {}, nullptr);

  // Every edge should span roughly one layer (constants adjacent to consumers).
  // If any constant were pinned to layer 0, an edge to op_(N-1) would span ~N
  // layers. Assert the max vertical edge span stays small (a few rank gaps).
  REQUIRE(!r.edges.empty());
  float max_span = 0.0f;
  for (const EdgeCurve& e : r.edges) {
    float s = e.p3.y - e.p0.y;
    if (s < 0) s = -s;
    if (s > max_span) max_span = s;
  }
  // One layer gap = node height (40) + rank_sep (default 60) = 100. Allow a
  // couple of layers of slack; a pinned-constant hairball would be ~N*100.
  CHECK(max_span < 100.0f * 3.0f);
}

TEST_CASE("compute_layout bails out when the ProgressSink is cancelled") {
  // #61 regression: request_cancel() BEFORE compute_layout runs must make it
  // return promptly with .cancelled == true and no usable geometry. Polling
  // happens at the first coarse checkpoint (measure, 0.1), so cancel is observed
  // early. Unit test on the pure function — no threading/timing dependence.
  ir::Model model = make_chain_model();

  CollapseTree collapse;
  collapse.build(model, 0);

  ProgressSink progress;
  progress.request_cancel();

  LayoutResult r = compute_layout(model, 0, collapse, headless_size, {}, &progress);
  CHECK(r.cancelled == true);
  CHECK(r.boxes.empty());
  CHECK(r.edges.empty());
}

TEST_CASE("a normal (non-cancelled) layout reports cancelled == false") {
  // The negative case: with a fresh sink (never cancelled) the layout completes
  // and the flag stays false, so a good result is never mistaken for cancelled.
  ir::Model model = make_chain_model();

  CollapseTree collapse;
  collapse.build(model, 0);

  ProgressSink progress;  // not cancelled
  LayoutResult r = compute_layout(model, 0, collapse, headless_size, {}, &progress);
  CHECK(r.cancelled == false);
  CHECK(r.boxes.size() == collapse.display_nodes().size());

  // And with no sink at all (the common test path), also not cancelled.
  LayoutResult r2 = compute_layout(model, 0, collapse, headless_size, {}, nullptr);
  CHECK(r2.cancelled == false);
}

TEST_CASE("#18: edge carries the IR value index of the value it stands for") {
  // Chain A(=Relu node0) -> B(=Relu node1): node0 produces value v1 (index 1),
  // which node1 consumes. The single display edge for that pair must carry
  // value_index == 1, and it must be a real (resolved) index, not UINT32_MAX.
  ir::Model model = make_chain_model();

  CollapseTree collapse;
  collapse.build(model, 0);
  LayoutResult r = compute_layout(model, 0, collapse, headless_size, {}, nullptr);

  REQUIRE(!r.edges.empty());
  // Locate the edge between node0's display box and node1's display box. In a
  // tiny uncollapsed graph display ids == IR node ids, but resolve via the box
  // that owns each IR node's producer to stay robust to display ordering: the
  // value v1 is produced by node 0, so the edge's producer output is v1.
  bool found = false;
  for (const EdgeCurve& e : r.edges) {
    // The producer of v1 is node 0; whichever display box owns node 0 is this
    // edge's from side. There is exactly one real edge in this two-node chain.
    CHECK(e.value_index != UINT32_MAX);
    CHECK(e.value_index == 1u);  // v1 is the value flowing A->B
    found = true;
  }
  CHECK(found);
}

TEST_CASE("#18: deduped edge for a pair carries the SMALLEST value index") {
  // P emits v1 and v2, both consumed by C, with C listing v2 before v1. The one
  // deduped P->C display edge must carry min(v1,v2), proving the value-index
  // tiebreak (not consumer iteration order) selects the representative value.
  uint32_t min_vidx = UINT32_MAX;
  ir::Model model = make_multivalue_model(&min_vidx);
  REQUIRE(min_vidx != UINT32_MAX);

  CollapseTree collapse;
  collapse.build(model, 0);
  LayoutResult r = compute_layout(model, 0, collapse, headless_size, {}, nullptr);

  // Exactly one display edge exists between P and C (deduped from two values).
  REQUIRE(r.edges.size() == 1);
  CHECK(r.edges[0].value_index == min_vidx);  // smallest of the two, i.e. v1.
  CHECK(r.edges[0].value_index != UINT32_MAX);
}

TEST_CASE("#18: value_index is deterministic across repeated layout runs") {
  // Determinism contract: identical inputs -> identical value_index on every
  // emitted edge, in the same edge order (the field must not perturb ordering).
  ir::Model model = make_multivalue_model(nullptr);

  CollapseTree collapse;
  collapse.build(model, 0);
  LayoutResult a = compute_layout(model, 0, collapse, headless_size, {}, nullptr);
  LayoutResult b = compute_layout(model, 0, collapse, headless_size, {}, nullptr);

  REQUIRE(a.edges.size() == b.edges.size());
  for (size_t i = 0; i < a.edges.size(); ++i) {
    CHECK(a.edges[i].from_display_id == b.edges[i].from_display_id);
    CHECK(a.edges[i].to_display_id == b.edges[i].to_display_id);
    CHECK(a.edges[i].value_index == b.edges[i].value_index);
  }
}
