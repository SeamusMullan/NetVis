// tests/test_graph_adjacency.cpp — CSR forward/reverse adjacency + bounded BFS.
//
// Builds a small DAG in code and asserts successor/predecessor sets, that lists
// are ascending + deduped, reachable_* hop bounds and visited caps, and that an
// out-of-range graph yields an empty (but safe) adjacency.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "engine/GraphAdjacency.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// Diamond + tail:  A -> B, A -> C, B -> D, C -> D, D -> E.
// Node indices: A=0 B=1 C=2 D=3 E=4. One value per producer.
ir::Model make_diamond() {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  // One output value per node; value index == producer node index.
  for (int i = 0; i < 5; ++i) {
    ir::ValueInfo v;
    v.name = m.intern("v" + std::to_string(i));
    v.producer = i;
    g.values.push_back(v);
  }

  auto add_node = [&](const char* name, std::vector<uint32_t> in_vals,
                      uint32_t out_val) {
    ir::Node n;
    n.op_type = m.intern("Op");
    n.name = m.intern(name);
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    for (uint32_t v : in_vals) g.edge_refs.push_back(v);
    n.inputs.count = static_cast<uint32_t>(in_vals.size());
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(out_val);
    n.outputs.count = 1;
    g.nodes.push_back(n);
  };

  add_node("A", {}, 0);          // node 0, no inputs
  add_node("B", {0}, 1);         // node 1, consumes A
  add_node("C", {0}, 2);         // node 2, consumes A
  add_node("D", {1, 2}, 3);      // node 3, consumes B,C
  add_node("E", {3}, 4);         // node 4, consumes D
  return m;
}

// Fetch the neighbor row [off[n], off[n+1]) as a vector.
std::vector<uint32_t> row(const std::vector<uint32_t>& off,
                          const std::vector<uint32_t>& val, uint32_t n) {
  return std::vector<uint32_t>(val.begin() + off[n], val.begin() + off[n + 1]);
}

}  // namespace

TEST_CASE("adjacency: successor/predecessor sets on a diamond") {
  ir::Model m = make_diamond();
  GraphAdjacency adj;
  adj.build(m, 0);

  REQUIRE(adj.node_count() == 5);
  const auto& so = adj.succ_offsets();
  const auto& sv = adj.succ_values();
  const auto& po = adj.pred_offsets();
  const auto& pv = adj.pred_values();
  REQUIRE(so.size() == 6);
  REQUIRE(po.size() == 6);

  // Successors.
  CHECK(row(so, sv, 0) == std::vector<uint32_t>{1, 2});  // A -> B,C
  CHECK(row(so, sv, 1) == std::vector<uint32_t>{3});     // B -> D
  CHECK(row(so, sv, 2) == std::vector<uint32_t>{3});     // C -> D
  CHECK(row(so, sv, 3) == std::vector<uint32_t>{4});     // D -> E
  CHECK(row(so, sv, 4).empty());                          // E -> {}

  // Predecessors.
  CHECK(row(po, pv, 0).empty());                          // A <- {}
  CHECK(row(po, pv, 1) == std::vector<uint32_t>{0});     // B <- A
  CHECK(row(po, pv, 2) == std::vector<uint32_t>{0});     // C <- A
  CHECK(row(po, pv, 3) == std::vector<uint32_t>{1, 2});  // D <- B,C (ascending)
  CHECK(row(po, pv, 4) == std::vector<uint32_t>{3});     // E <- D
}

TEST_CASE("adjacency: neighbor lists are deduped on parallel edges") {
  // A node consuming the SAME producer value twice must yield ONE edge.
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  for (int i = 0; i < 2; ++i) {
    ir::ValueInfo v;
    v.producer = i;
    g.values.push_back(v);
  }
  // node 0 produces value 0.
  {
    ir::Node n;
    n.op_type = m.intern("Src");
    n.outputs.begin = 0;
    g.edge_refs.push_back(0);
    n.outputs.count = 1;
    g.nodes.push_back(n);
  }
  // node 1 consumes value 0 TWICE.
  {
    ir::Node n;
    n.op_type = m.intern("Dst");
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(0);
    g.edge_refs.push_back(0);
    n.inputs.count = 2;
    g.nodes.push_back(n);
  }

  GraphAdjacency adj;
  adj.build(m, 0);
  CHECK(row(adj.succ_offsets(), adj.succ_values(), 0) ==
        std::vector<uint32_t>{1});
  CHECK(row(adj.pred_offsets(), adj.pred_values(), 1) ==
        std::vector<uint32_t>{0});
}

TEST_CASE("adjacency: reachable_* hop bounds + cap + start exclusion") {
  ir::Model m = make_diamond();
  GraphAdjacency adj;
  adj.build(m, 0);

  // From A, 1 hop => B,C only (excludes A itself).
  CHECK(adj.reachable_succ(0, 1, 100) == std::vector<uint32_t>{1, 2});
  // 2 hops => B,C,D.
  CHECK(adj.reachable_succ(0, 2, 100) == std::vector<uint32_t>{1, 2, 3});
  // Unbounded => B,C,D,E.
  CHECK(adj.reachable_succ(0, UINT32_MAX, 100) ==
        std::vector<uint32_t>{1, 2, 3, 4});

  // Predecessors of E, unbounded => A,B,C,D.
  CHECK(adj.reachable_pred(4, UINT32_MAX, 100) ==
        std::vector<uint32_t>{0, 1, 2, 3});
  // 1 hop back from E => D only.
  CHECK(adj.reachable_pred(4, 1, 100) == std::vector<uint32_t>{3});

  // Cap bounds the visited count (result never exceeds cap).
  std::vector<uint32_t> capped = adj.reachable_succ(0, UINT32_MAX, 2);
  CHECK(capped.size() <= 2);

  // Zero cap => empty; start excluded even at 0 hops.
  CHECK(adj.reachable_succ(0, UINT32_MAX, 0).empty());
  CHECK(adj.reachable_succ(0, 0, 100).empty());
}

TEST_CASE("adjacency: out-of-range graph is safe and empty") {
  ir::Model m = make_diamond();
  GraphAdjacency adj;
  adj.build(m, 99);  // OOB
  CHECK(adj.node_count() == 0);
  CHECK(adj.succ_values().empty());
  CHECK(adj.pred_values().empty());
  // Queries on an empty adjacency are safe.
  CHECK(adj.reachable_succ(0, 1, 100).empty());
  CHECK(adj.reachable_pred(5, 1, 100).empty());
}
