// tests/test_graph_adjacency.cpp — CSR forward/reverse adjacency + bounded BFS.
//
// Builds a small DAG in code and asserts successor/predecessor sets, that lists
// are ascending + deduped, reachable_* hop bounds and visited caps, and that an
// out-of-range graph yields an empty (but safe) adjacency.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "engine/GraphAdjacency.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// Build a model with `n` nodes where node i produces value i (value idx == node
// idx, matching the make_diamond convention). `edges` are directed (producer,
// consumer) NODE pairs: the consumer gets the producer's value on an input slot.
ir::Model make_model(uint32_t n,
                     const std::vector<std::pair<uint32_t, uint32_t>>& edges) {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  for (uint32_t i = 0; i < n; ++i) {
    ir::ValueInfo v;
    v.name = m.intern("v" + std::to_string(i));
    v.producer = static_cast<int32_t>(i);
    g.values.push_back(v);
  }

  std::vector<std::vector<uint32_t>> ins(n);  // consumer -> producer value idxs
  for (const auto& e : edges) {
    if (e.second < n) ins[e.second].push_back(e.first);
  }

  for (uint32_t i = 0; i < n; ++i) {
    ir::Node nd;
    nd.op_type = m.intern("Op");
    nd.name = m.intern("n" + std::to_string(i));
    nd.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    for (uint32_t p : ins[i]) g.edge_refs.push_back(p);
    nd.inputs.count = static_cast<uint32_t>(ins[i].size());
    nd.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(i);  // node i outputs value i
    nd.outputs.count = 1;
    g.nodes.push_back(nd);
  }
  return m;
}

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

// Chain A->X->Y->B with a side branch X->Z and an isolated node W:
//   A=0 X=1 Y=2 B=3 Z=4 W=5.  One value per producer (value idx == node idx).
// Only A,X,Y,B lie between A and B; Z is a dead-end off the path, W disjoint.
ir::Model make_chain_branch() {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  for (int i = 0; i < 6; ++i) {
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

  add_node("A", {}, 0);       // node 0, source
  add_node("X", {0}, 1);      // node 1, consumes A
  add_node("Y", {1}, 2);      // node 2, consumes X
  add_node("B", {2}, 3);      // node 3, consumes Y (chain end)
  add_node("Z", {1}, 4);      // node 4, consumes X (side branch, off path)
  add_node("W", {}, 5);       // node 5, isolated (no edges)
  return m;
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

TEST_CASE("adjacency: nodes_on_paths_between — chain includes all interior") {
  // Chain A(0)->X(1)->Y(2)->B(3), side branch X->Z(4), isolated W(5).
  ir::Model m = make_chain_branch();
  GraphAdjacency adj;
  adj.build(m, 0);

  // A->..->B path is exactly {A,X,Y,B}; Z (side branch) and W excluded.
  CHECK(adj.nodes_on_paths_between(0, 3, 100) ==
        std::vector<uint32_t>{0, 1, 2, 3});
  // Same result regardless of argument order (B upstream tried too).
  CHECK(adj.nodes_on_paths_between(3, 0, 100) ==
        std::vector<uint32_t>{0, 1, 2, 3});

  // A sub-segment: X..Y => {X,Y}.
  CHECK(adj.nodes_on_paths_between(1, 2, 100) == std::vector<uint32_t>{1, 2});

  // Endpoint adjacent to a side branch: A..X => {A,X} (Z not between them).
  CHECK(adj.nodes_on_paths_between(0, 1, 100) == std::vector<uint32_t>{0, 1});

  // No directed path connects the side branch Z to Y (Z is a leaf off X).
  CHECK(adj.nodes_on_paths_between(4, 2, 100).empty());
  // Isolated node W connects to nothing.
  CHECK(adj.nodes_on_paths_between(0, 5, 100).empty());
}

TEST_CASE("adjacency: nodes_on_paths_between — diamond keeps both branches") {
  // Diamond A(0)->{B(1),C(2)}->D(3)->E(4). Between A and D: {A,B,C,D}.
  ir::Model m = make_diamond();
  GraphAdjacency adj;
  adj.build(m, 0);

  CHECK(adj.nodes_on_paths_between(0, 3, 100) ==
        std::vector<uint32_t>{0, 1, 2, 3});
  CHECK(adj.nodes_on_paths_between(3, 0, 100) ==
        std::vector<uint32_t>{0, 1, 2, 3});
  // A..E spans the whole graph.
  CHECK(adj.nodes_on_paths_between(0, 4, 100) ==
        std::vector<uint32_t>{0, 1, 2, 3, 4});
  // One branch endpoint to the other (B,C) are not on any directed B->C or
  // C->B path (siblings), so empty.
  CHECK(adj.nodes_on_paths_between(1, 2, 100).empty());
}

TEST_CASE("adjacency: nodes_on_paths_between — degenerate + OOB cases") {
  ir::Model m = make_chain_branch();
  GraphAdjacency adj;
  adj.build(m, 0);

  // a == b => empty (no *between* semantics for a single node).
  CHECK(adj.nodes_on_paths_between(0, 0, 100).empty());
  CHECK(adj.nodes_on_paths_between(2, 2, 100).empty());
  // Out-of-range index => empty, either argument.
  CHECK(adj.nodes_on_paths_between(0, 99, 100).empty());
  CHECK(adj.nodes_on_paths_between(99, 0, 100).empty());
  CHECK(adj.nodes_on_paths_between(99, 100, 100).empty());
  // Queries on an empty adjacency are safe.
  GraphAdjacency empty;
  empty.build(m, 99);
  CHECK(empty.nodes_on_paths_between(0, 1, 100).empty());
}

// =============================================================================
//  #14 — longest_cost_path (critical path: source->sink maximizing sum(weight)).
// =============================================================================

TEST_CASE("longest_cost_path: simple chain A->B->C->D returns whole chain") {
  // 0->1->2->3, uniform weights -> the whole chain is the max-cost path.
  ir::Model m = make_model(4, {{0, 1}, {1, 2}, {2, 3}});
  GraphAdjacency adj;
  adj.build(m, 0);

  std::vector<uint64_t> w{5, 5, 5, 5};
  CHECK(adj.longest_cost_path(w) == std::vector<uint32_t>{0, 1, 2, 3});
}

TEST_CASE("longest_cost_path: diamond picks the higher-weight branch") {
  // A(0)->{B(1),C(2)}->D(3). B heavier than C -> path is A,B,D not A,C,D.
  ir::Model m = make_model(4, {{0, 1}, {0, 2}, {1, 3}, {2, 3}});
  GraphAdjacency adj;
  adj.build(m, 0);

  //         A  B  C  D
  std::vector<uint64_t> w{1, 10, 2, 1};
  CHECK(adj.longest_cost_path(w) == std::vector<uint32_t>{0, 1, 3});

  // Flip the bias: C heavier -> path goes through C.
  std::vector<uint64_t> w2{1, 2, 10, 1};
  CHECK(adj.longest_cost_path(w2) == std::vector<uint32_t>{0, 2, 3});
}

TEST_CASE("longest_cost_path: weights bias the sink selection") {
  // A(0)->B(1)->D(3) and A(0)->C(2)->E(4): two sink candidates (D and E). The
  // higher-weight branch's sink wins.
  ir::Model m = make_model(5, {{0, 1}, {1, 3}, {0, 2}, {2, 4}});
  GraphAdjacency adj;
  adj.build(m, 0);

  //         A  B  C  D   E
  std::vector<uint64_t> w{1, 1, 1, 2, 50};  // E-branch far heavier
  CHECK(adj.longest_cost_path(w) == std::vector<uint32_t>{0, 2, 4});

  // Bias the other sink: D-branch wins.
  std::vector<uint64_t> w2{1, 1, 1, 100, 2};
  CHECK(adj.longest_cost_path(w2) == std::vector<uint32_t>{0, 1, 3});
}

TEST_CASE("longest_cost_path: empty graph returns empty") {
  ir::Model m = make_model(0, {});
  GraphAdjacency adj;
  adj.build(m, 0);
  CHECK(adj.node_count() == 0);
  CHECK(adj.longest_cost_path({}).empty());
  CHECK(adj.longest_cost_path({1, 2, 3}).empty());

  // Out-of-range build is also empty and safe.
  GraphAdjacency oob;
  oob.build(m, 99);
  CHECK(oob.longest_cost_path({}).empty());
}

TEST_CASE("longest_cost_path: weight shorter than node_count treats missing as 0") {
  // Chain 0->1->2->3, weight covers only nodes 0,1,2 (size 3) -> node 3 counts as
  // 0. best = [1,2,7,7]: node 3 adds nothing so it ties node 2, and the sink
  // tie-break (smallest index) stops the chain at node 2. Result is a valid,
  // ascending source->sink chain and demonstrates the i<weight.size() guard.
  ir::Model m = make_model(4, {{0, 1}, {1, 2}, {2, 3}});
  GraphAdjacency adj;
  adj.build(m, 0);

  std::vector<uint64_t> w{1, 1, 5};  // shorter than node_count()==4 (node 3 -> 0)
  std::vector<uint32_t> path = adj.longest_cost_path(w);
  CHECK(path == std::vector<uint32_t>{0, 1, 2});
  // Order is source->sink (strictly ascending topo on this chain).
  for (size_t i = 1; i < path.size(); ++i) CHECK(path[i - 1] < path[i]);

  // Empty weight => all zeros; still returns a valid chain (best[v]==0 for all,
  // sink is the smallest-index argmax = node 0, a length-1 source path).
  std::vector<uint32_t> zpath = adj.longest_cost_path({});
  CHECK(zpath == std::vector<uint32_t>{0});
}

TEST_CASE("longest_cost_path: a residual back-edge cycle still terminates") {
  // 0->1->2 plus a back-edge 2->1 forms a cycle {1,2}. indegree(1)=2, indegree(2)
  // =1: after popping 0 and relaxing 0->1, indeg(1) drops 2->1 (never 0), so the
  // queue drains without visiting the cycle body via a second relaxation. The
  // pass must terminate and return a valid (best-effort) chain.
  ir::Model m = make_model(3, {{0, 1}, {1, 2}, {2, 1}});
  GraphAdjacency adj;
  adj.build(m, 0);

  std::vector<uint64_t> w{1, 1, 1};
  std::vector<uint32_t> path = adj.longest_cost_path(w);  // must not hang/crash
  CHECK_FALSE(path.empty());
  // Whatever chain is returned, it is ascending (source->sink) and has no repeats.
  for (size_t i = 1; i < path.size(); ++i) CHECK(path[i - 1] < path[i]);
}
