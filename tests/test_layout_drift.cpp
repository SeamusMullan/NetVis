// tests/test_layout_drift.cpp — v0.2.1 regression: ONNX graphs must not drift
// consistently rightward with depth (the minimap-looks-like-a-diagonal-line bug).
//
// Builds a deep chain with periodic skip/residual edges (2-input nodes, the
// shape that reveals x-alignment bias) and asserts the horizontal spread of node
// centers stays bounded — a healthy layered layout is a mostly-vertical ribbon,
// not a staircase whose x grows ~linearly with layer index.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/CollapseTree.h"
#include "engine/Layout.h"
#include "engine/LayoutEngine.h"
#include "ir/IR.h"

using namespace netvis;

namespace {
Vec2 headless_size(const DisplayNode&) { return Vec2{120.0f, 40.0f}; }

// Chain n0->n1->...->n(N-1); node i (i>=2) ALSO consumes n(i-2) — a skip edge.
// Every node from layer 2 on has two predecessors, exercising median x-align.
ir::Model make_skip_chain(int N) {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
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
    n.op_type = m.intern(i % 2 ? "Add" : "Relu");
    n.name = m.intern("n" + std::to_string(i));
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    uint32_t cnt = 0;
    if (i >= 1) { g.edge_refs.push_back(ov[i - 1]); ++cnt; }
    if (i >= 2) { g.edge_refs.push_back(ov[i - 2]); ++cnt; }
    n.inputs.count = cnt;
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(ov[i]);
    n.outputs.count = 1;
    g.nodes.push_back(n);
  }
  return m;
}
}  // namespace

TEST_CASE("layered layout does not drift consistently rightward with depth") {
  const int N = 40;
  ir::Model m = make_skip_chain(N);
  CollapseTree collapse;
  collapse.build(m, 0);
  LayoutResult r = compute_layout(m, 0, collapse, headless_size, {}, nullptr);
  REQUIRE(!r.boxes.empty());

  // Per-layer mean center-x, and the min/max across layers.
  int max_layer = 0;
  for (const NodeBox& b : r.boxes) max_layer = std::max(max_layer, b.layer);
  std::vector<double> sum(max_layer + 1, 0.0);
  std::vector<int> cnt(max_layer + 1, 0);
  for (const NodeBox& b : r.boxes) {
    sum[b.layer] += b.pos.x + b.size.x * 0.5;
    ++cnt[b.layer];
  }
  double lo = 1e18, hi = -1e18, first = 0, last = 0;
  bool have_first = false;
  for (int l = 0; l <= max_layer; ++l) {
    if (!cnt[l]) continue;
    double c = sum[l] / cnt[l];
    if (!have_first) { first = c; have_first = true; }
    last = c;
    lo = std::min(lo, c);
    hi = std::max(hi, c);
  }
  const double node_w = 120.0;
  std::printf("[drift] layers=%d  center range=%.1f (%.2f node-widths)  "
              "first->last=%.1f\n",
              max_layer + 1, hi - lo, (hi - lo) / node_w, last - first);

  // A vertical ribbon keeps all layer centers within a couple node widths. The
  // pre-fix staircase spanned ~29 node-widths over 40 layers and grew with depth
  // (the minimap-is-a-line bug). Bound the spread tightly so a regression of the
  // de-shear / median-symmetry fix trips this.
  CHECK((hi - lo) < node_w * 3.0);
}
