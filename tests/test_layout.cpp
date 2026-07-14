// tests/test_layout.cpp — layered layout determinism (spec §2.7, §7.2).
//
// Builds a tiny ir::Model in code, runs CollapseTree::build + compute_layout
// TWICE, and asserts byte-identical positions (determinism is what makes the
// layout cache correct) and that boxes.size() == display_nodes().size().
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

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
