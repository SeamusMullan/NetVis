// tests/test_collapse.cpp — CollapseTree global collapse/expand (#21).
//
// Builds tiny ir::Models in code (one with a detected repeated-block group, one
// with none) and exercises collapse_all()/expand_all(): the change-detection
// return contract, the resulting display-node list, and the collapse_hash.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "engine/CollapseTree.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// A graph with exactly one detected group: nodes named "layers.0/1/2" share the
// name prefix before their first digit run, so CollapseTree's name-prefix pass
// groups them (>= 2 distinct integer instances). Chained v0->v1->v2->v3 so each
// node has a real producer edge, matching the make_chain_model fixture style.
ir::Model make_grouped_model() {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  g.name = m.intern("g");

  // Values: v0 (input), v1, v2, v3 (outputs of each layer).
  for (int i = 0; i < 4; ++i) {
    ir::ValueInfo v;
    v.name = m.intern("v" + std::to_string(i));
    g.values.push_back(v);
  }

  auto add_node = [&](const std::string& name, uint32_t in_val, uint32_t out_val,
                      int32_t producer) {
    ir::Node n;
    n.op_type = m.intern("MatMul");
    n.name = m.intern(name);
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(in_val);
    n.inputs.count = 1;
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(out_val);
    n.outputs.count = 1;
    g.values[out_val].producer = producer;
    g.nodes.push_back(n);
  };

  add_node("layers.0", 0, 1, 0);
  add_node("layers.1", 1, 2, 1);
  add_node("layers.2", 2, 3, 2);
  g.graph_inputs.push_back(0);
  g.graph_outputs.push_back(3);
  return m;
}

// A graph with NO detectable group: two distinct ops, no digit runs in names,
// so neither the name-prefix pass nor the repeated-fingerprint pass fires.
ir::Model make_flat_model() {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  g.name = m.intern("g");

  for (int i = 0; i < 2; ++i) {
    ir::ValueInfo v;
    v.name = m.intern("v" + std::to_string(i));
    g.values.push_back(v);
  }

  auto add_node = [&](const char* op, const char* name, uint32_t in_val,
                      uint32_t out_val, int32_t producer) {
    ir::Node n;
    n.op_type = m.intern(op);
    n.name = m.intern(name);
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(in_val);
    n.inputs.count = 1;
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(out_val);
    n.outputs.count = 1;
    g.values[out_val].producer = producer;
    g.nodes.push_back(n);
  };

  add_node("Relu", "act", 0, 1, 0);
  add_node("Sigmoid", "gate", 1, 1, 0);
  g.graph_inputs.push_back(0);
  g.graph_outputs.push_back(1);
  return m;
}

// Count is_group entries in a display list.
size_t count_group_nodes(const std::vector<DisplayNode>& d) {
  size_t c = 0;
  for (const DisplayNode& n : d)
    if (n.is_group) ++c;
  return c;
}

}  // namespace

TEST_CASE("collapse_all collapses the detected group into a super-node") {
  ir::Model model = make_grouped_model();
  CollapseTree collapse;
  collapse.build(model, 0);

  // Precondition: exactly one group detected, default view fully expanded.
  REQUIRE(collapse.groups().size() == 1);
  REQUIRE(count_group_nodes(collapse.display_nodes()) == 0);
  const uint64_t expanded_hash = collapse.collapse_hash();

  // collapse_all() changes the display -> true, and the group shows collapsed.
  CHECK(collapse.collapse_all() == true);
  CHECK(count_group_nodes(collapse.display_nodes()) == 1);
  // One collapsed super-node stands in for all three member leaves.
  CHECK(collapse.display_nodes().size() == 1);
  CHECK(collapse.display_nodes()[0].is_group == true);
  CHECK(collapse.display_nodes()[0].group_index == 0u);
  // The collapse state (and thus its hash / layout cache key) actually changed.
  CHECK(collapse.collapse_hash() != expanded_hash);
}

TEST_CASE("collapse_all is a no-op when already fully collapsed") {
  ir::Model model = make_grouped_model();
  CollapseTree collapse;
  collapse.build(model, 0);

  REQUIRE(collapse.collapse_all() == true);
  const uint64_t collapsed_hash = collapse.collapse_hash();
  const size_t collapsed_size = collapse.display_nodes().size();

  // Second call: nothing left to collapse -> false, display + hash unchanged.
  CHECK(collapse.collapse_all() == false);
  CHECK(collapse.collapse_hash() == collapsed_hash);
  CHECK(collapse.display_nodes().size() == collapsed_size);
}

TEST_CASE("expand_all restores the default post-build display after a collapse") {
  ir::Model model = make_grouped_model();
  CollapseTree collapse;
  collapse.build(model, 0);

  // Snapshot the default (all-expanded) display and hash.
  const uint64_t default_hash = collapse.collapse_hash();
  const size_t default_size = collapse.display_nodes().size();
  REQUIRE(default_size == 3);  // three member leaves, no group node

  REQUIRE(collapse.collapse_all() == true);

  // expand_all() undoes the collapse -> true, and we're back to the default:
  // every member visible as a leaf, no is_group entries.
  CHECK(collapse.expand_all() == true);
  CHECK(count_group_nodes(collapse.display_nodes()) == 0);
  CHECK(collapse.display_nodes().size() == default_size);
  CHECK(collapse.collapse_hash() == default_hash);
}

TEST_CASE("expand_all is a no-op on a freshly-built (all-expanded) tree") {
  ir::Model model = make_grouped_model();
  CollapseTree collapse;
  collapse.build(model, 0);

  // Fresh build defaults to fully expanded, so there is nothing to expand.
  const uint64_t default_hash = collapse.collapse_hash();
  CHECK(collapse.expand_all() == false);
  CHECK(collapse.collapse_hash() == default_hash);
}

TEST_CASE("collapse_all/expand_all are no-ops on a graph with no groups") {
  ir::Model model = make_flat_model();
  CollapseTree collapse;
  collapse.build(model, 0);

  REQUIRE(collapse.groups().empty());
  const uint64_t hash = collapse.collapse_hash();
  const size_t size = collapse.display_nodes().size();

  CHECK(collapse.collapse_all() == false);
  CHECK(collapse.expand_all() == false);
  CHECK(collapse.collapse_hash() == hash);
  CHECK(collapse.display_nodes().size() == size);
}
