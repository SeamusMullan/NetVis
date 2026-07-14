// engine/LayoutEngine.cpp — from-scratch layered (Sugiyama) layout.
//
// PERF (spec §7.2): the whole pipeline is O(V+E) plus O(sweeps*E) for ordering,
// with no per-node heap churn in the hot loops. Target: 10k display nodes in
// < 250 ms. Everything works on the CURRENT display-node list (already
// collapsed by CollapseTree, so V is usually small), in world coordinates, with
// no GUI dependency — node sizes come in through SizeFn so this runs headless.
//
// Stages:
//   1. Build DAG over display nodes (edge A->B if an IR node in A produces a
//      value consumed by an IR node in B).
//   2. Cycle break via DFS back-edge detection (reversed edges flagged).
//   3. Longest-path layering.
//   4. Barycenter crossing reduction (down+up sweeps, early stop).
//   5. Coordinate assignment (y per layer top-down, x by order + median align).
//   6. Cubic-bezier edge routing.
#include "engine/LayoutEngine.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "engine/CollapseTree.h"

namespace netvis {

namespace {

// Default headless size: width scales with label length, fixed ~40 height.
Vec2 default_size(const DisplayNode& d) {
  // A representative label length; groups are a bit wider than leaves.
  float chars = d.is_group ? 18.0f : 12.0f;
  float w = 40.0f + chars * 7.0f;
  return Vec2{w, 40.0f};
}

// Internal directed edge between display nodes.
struct DEdge {
  uint32_t from = 0;
  uint32_t to = 0;
  bool reversed = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// compute_layout
// ---------------------------------------------------------------------------
LayoutResult compute_layout(const ir::Model& model, uint32_t graph_index,
                            const CollapseTree& collapse, const SizeFn& size_fn,
                            const LayoutParams& params, ProgressSink* progress) {
  LayoutResult out;
  out.structure_hash = collapse.structure_hash();
  out.collapse_hash = collapse.collapse_hash();

  const std::vector<DisplayNode>& disp = collapse.display_nodes();
  const uint32_t V = static_cast<uint32_t>(disp.size());

  // Node boxes parallel to the display list; sizes measured now.
  out.boxes.resize(V);
  for (uint32_t i = 0; i < V; ++i) {
    out.boxes[i].display_id = i;
    Vec2 sz = size_fn ? size_fn(disp[i]) : default_size(disp[i]);
    if (sz.x <= 0.0f) sz.x = 1.0f;
    if (sz.y <= 0.0f) sz.y = 1.0f;
    out.boxes[i].size = sz;
  }
  if (progress) progress->set(0.1f, "layout: measure");

  if (V == 0 || graph_index >= model.graphs.size()) {
    out.bounds_min = Vec2{0, 0};
    out.bounds_max = Vec2{0, 0};
    return out;
  }
  const ir::Graph& g = model.graphs[graph_index];

  // -- Map each IR node -> its display node index. For a collapsed group all
  // member IR nodes map to the group's display node; for an expanded group only
  // the representative leaves are present as their own display nodes. IR nodes
  // that are not represented in the current view map to UINT32_MAX.
  const size_t nIR = g.nodes.size();
  std::vector<uint32_t> ir_to_disp(nIR, UINT32_MAX);
  const std::vector<CollapseGroup>& groups = collapse.groups();
  for (uint32_t di = 0; di < V; ++di) {
    const DisplayNode& d = disp[di];
    if (d.is_group) {
      if (d.group_index < groups.size())
        for (uint32_t ni : groups[d.group_index].member_nodes)
          if (ni < nIR) ir_to_disp[ni] = di;
    } else if (d.ir_node < nIR) {
      ir_to_disp[d.ir_node] = di;
    }
  }

  // -- 1) Build DAG over display nodes. Edge A->B when an IR node mapped to A
  // produces a value consumed by an IR node mapped to B (A != B). Dedup edges.
  std::vector<DEdge> edges;
  {
    // Build a set of (from<<32|to) for dedup; reserve conservatively.
    std::vector<uint64_t> keys;
    keys.reserve(g.edge_refs.size());
    for (uint32_t ni = 0; ni < nIR; ++ni) {
      uint32_t consumer = ir_to_disp[ni];
      if (consumer == UINT32_MAX) continue;
      const ir::Node& nd = g.nodes[ni];
      uint32_t b = nd.inputs.begin;
      uint32_t e = nd.inputs.begin + nd.inputs.count;
      for (uint32_t k = b; k < e && k < g.edge_refs.size(); ++k) {
        uint32_t vidx = g.edge_refs[k];
        if (vidx >= g.values.size()) continue;
        int32_t p = g.values[vidx].producer;
        if (p < 0 || static_cast<size_t>(p) >= nIR) continue;
        uint32_t producer = ir_to_disp[static_cast<uint32_t>(p)];
        if (producer == UINT32_MAX || producer == consumer) continue;
        keys.push_back((static_cast<uint64_t>(producer) << 32) | consumer);
      }
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    edges.reserve(keys.size());
    for (uint64_t k : keys) {
      DEdge de;
      de.from = static_cast<uint32_t>(k >> 32);
      de.to = static_cast<uint32_t>(k & 0xffffffffu);
      edges.push_back(de);
    }
  }
  if (progress) progress->set(0.3f, "layout: edges");

  // Adjacency (out) for DFS cycle-break and layering.
  std::vector<std::vector<uint32_t>> out_adj(V);  // -> edge indices
  for (uint32_t ei = 0; ei < edges.size(); ++ei)
    out_adj[edges[ei].from].push_back(ei);

  // -- 2) Cycle break: iterative DFS, an edge to a node currently on the DFS
  // stack (gray) is a back-edge and gets reversed. Deterministic: nodes visited
  // in index order, out-edges in insertion (sorted) order.
  enum Color : uint8_t { White = 0, Gray = 1, Black = 2 };
  {
    std::vector<uint8_t> color(V, White);
    // Explicit stack of (node, next out-edge cursor).
    std::vector<std::pair<uint32_t, size_t>> stack;
    stack.reserve(V);
    for (uint32_t s = 0; s < V; ++s) {
      if (color[s] != White) continue;
      stack.push_back({s, 0});
      color[s] = Gray;
      while (!stack.empty()) {
        auto& [u, ci] = stack.back();
        if (ci < out_adj[u].size()) {
          uint32_t ei = out_adj[u][ci++];
          DEdge& de = edges[ei];
          uint32_t w = de.to;
          if (color[w] == Gray) {
            de.reversed = true;  // back-edge -> reverse to make acyclic
          } else if (color[w] == White) {
            color[w] = Gray;
            stack.push_back({w, 0});
          }
        } else {
          color[u] = Black;
          stack.pop_back();
        }
      }
    }
  }

  // Effective (acyclic) direction accessor: reversed edges flip from/to.
  auto eff_from = [&](const DEdge& e) { return e.reversed ? e.to : e.from; };
  auto eff_to = [&](const DEdge& e) { return e.reversed ? e.from : e.to; };

  // -- 3) Longest-path layering. layer(v) = max over in-edges of layer(src)+1.
  // Compute via a topological pass using Kahn on the acyclic (post-reversal)
  // graph. O(V+E).
  std::vector<int32_t> layer(V, 0);
  {
    std::vector<uint32_t> indeg(V, 0);
    std::vector<std::vector<uint32_t>> eff_out(V);
    for (const DEdge& e : edges) {
      eff_out[eff_from(e)].push_back(eff_to(e));
      ++indeg[eff_to(e)];
    }
    std::vector<uint32_t> ready;
    for (uint32_t v = 0; v < V; ++v)
      if (indeg[v] == 0) ready.push_back(v);
    std::sort(ready.begin(), ready.end());
    size_t head = 0;
    std::vector<uint32_t> order;
    order.reserve(V);
    // Process deterministically: maintain sorted-ish by popping in order.
    while (head < ready.size()) {
      uint32_t u = ready[head++];
      order.push_back(u);
      for (uint32_t w : eff_out[u]) {
        if (layer[w] < layer[u] + 1) layer[w] = layer[u] + 1;
        if (--indeg[w] == 0) ready.push_back(w);
      }
    }
    // Cycle remnants (shouldn't happen after reversal): leave layer as-is.
  }

  // -- 3b) Pull "constant-like" source nodes DOWN next to their consumers.
  // Longest-path layering pins every source (in-degree 0) to layer 0. In real
  // models most sources are constants/initializers/weights that feed a node deep
  // in the graph, so they all pile onto the top row and their edges spray across
  // the entire canvas (the classic hairball). Instead, place each source just
  // above its NEAREST consumer: layer = min(consumer layer) - 1. This keeps a
  // constant adjacent to where it is used and collapses those long edges.
  // Only sources are moved (they have no predecessors, so lowering them can
  // never violate an edge from above); non-sources keep their longest-path rank.
  {
    // Effective in/out degree + successor layers.
    std::vector<uint32_t> indeg(V, 0);
    std::vector<std::vector<uint32_t>> eff_out(V);
    for (const DEdge& e : edges) {
      eff_out[eff_from(e)].push_back(eff_to(e));
      ++indeg[eff_to(e)];
    }
    for (uint32_t v = 0; v < V; ++v) {
      if (indeg[v] != 0) continue;      // only sources
      if (eff_out[v].empty()) continue; // isolated node: leave at 0
      int32_t min_consumer = std::numeric_limits<int32_t>::max();
      for (uint32_t w : eff_out[v])
        min_consumer = std::min(min_consumer, layer[w]);
      // Sit one layer above the nearest consumer (never below 0).
      if (min_consumer != std::numeric_limits<int32_t>::max())
        layer[v] = std::max(0, min_consumer - 1);
    }
  }

  int32_t max_layer = 0;
  for (uint32_t v = 0; v < V; ++v) max_layer = std::max(max_layer, layer[v]);
  const size_t L = static_cast<size_t>(max_layer) + 1;
  if (progress) progress->set(0.5f, "layout: layering");

  // Layer membership; `pos_in_layer` = order index within a layer.
  std::vector<std::vector<uint32_t>> layers(L);
  for (uint32_t v = 0; v < V; ++v)
    layers[static_cast<size_t>(layer[v])].push_back(v);
  // Initial order: by display index (deterministic).
  for (auto& lv : layers) std::sort(lv.begin(), lv.end());

  std::vector<uint32_t> order_idx(V, 0);
  auto refresh_order = [&]() {
    for (auto& lv : layers)
      for (uint32_t p = 0; p < lv.size(); ++p) order_idx[lv[p]] = p;
  };
  refresh_order();

  // Predecessor / successor adjacency in effective direction for barycenters.
  std::vector<std::vector<uint32_t>> preds(V), succs(V);
  for (const DEdge& e : edges) {
    uint32_t f = eff_from(e), t = eff_to(e);
    succs[f].push_back(t);
    preds[t].push_back(f);
  }

  // Count total crossings between all adjacent layer pairs (for early-stop).
  // PERF: crossings between two layers = inversions in the lower endpoints once
  // edges are ordered by upper endpoint. We count inversions in O(E log E) with
  // a Fenwick tree (BIT), NOT the naive O(E^2) double loop — a wide graph with
  // no collapse can put thousands of edges between one layer pair, where O(E^2)
  // would blow the sub-250ms budget.
  auto count_crossings = [&]() -> uint64_t {
    uint64_t total = 0;
    std::vector<std::pair<uint32_t, uint32_t>> segs;  // (upper_pos, lower_pos)
    std::vector<uint32_t> bit;
    for (size_t li = 0; li + 1 < L; ++li) {
      segs.clear();
      for (uint32_t u : layers[li])
        for (uint32_t w : succs[u])
          if (static_cast<size_t>(layer[w]) == li + 1)
            segs.push_back({order_idx[u], order_idx[w]});
      if (segs.size() < 2) continue;
      // Order by upper endpoint (ties by lower) so that, scanning left to right,
      // a crossing is exactly a previously-seen edge with a GREATER lower pos.
      std::sort(segs.begin(), segs.end());
      const size_t width = layers[li + 1].size();
      bit.assign(width + 1, 0);
      size_t seen = 0;
      for (const auto& s : segs) {
        // #previously-seen edges with lower_pos <= s.second, via prefix sum.
        uint32_t leq = 0;
        for (size_t x = s.second + 1; x > 0; x -= x & (~x + 1)) leq += bit[x];
        total += static_cast<uint64_t>(seen) - leq;  // those with lower > s.second cross
        for (size_t x = s.second + 1; x <= width; x += x & (~x + 1)) ++bit[x];
        ++seen;
      }
    }
    return total;
  };

  // -- 4) Barycenter ordering. Alternate down (order by predecessor mean) and
  // up (by successor mean) sweeps; early-stop when crossings stop improving.
  auto barycenter_sort = [&](size_t li, bool use_preds) {
    auto& lv = layers[li];
    const auto& nbr = use_preds ? preds : succs;
    // Compute barycenter key per node; nodes with no neighbor keep their pos.
    std::vector<std::pair<float, uint32_t>> keyed;
    keyed.reserve(lv.size());
    for (uint32_t v : lv) {
      float sum = 0.0f;
      uint32_t cnt = 0;
      for (uint32_t nb : nbr[v]) {
        // Only neighbors in the adjacent layer matter for this ordering pass.
        sum += static_cast<float>(order_idx[nb]);
        ++cnt;
      }
      float key = cnt ? sum / static_cast<float>(cnt)
                      : static_cast<float>(order_idx[v]);
      keyed.push_back({key, v});
    }
    std::stable_sort(keyed.begin(), keyed.end(),
                     [](const auto& a, const auto& b) {
                       if (a.first != b.first) return a.first < b.first;
                       return a.second < b.second;  // deterministic tiebreak
                     });
    for (size_t p = 0; p < lv.size(); ++p) lv[p] = keyed[p].second;
  };

  {
    uint64_t best = count_crossings();
    int sweeps = std::max(0, params.barycenter_sweeps);
    for (int s = 0; s < sweeps; ++s) {
      // Down sweep: order each layer by predecessor barycenters, top->bottom.
      for (size_t li = 1; li < L; ++li) barycenter_sort(li, /*use_preds=*/true);
      refresh_order();
      // Up sweep: order by successor barycenters, bottom->top.
      for (size_t li = L; li-- > 0;) barycenter_sort(li, /*use_preds=*/false);
      refresh_order();
      uint64_t c = count_crossings();
      if (c >= best) break;  // early-stop: no improvement
      best = c;
    }
  }
  if (progress) progress->set(0.75f, "layout: ordering");

  // -- 5) Coordinate assignment.
  // y: accumulate per layer top-down (max node height in layer + rank_sep).
  std::vector<float> layer_y(L, 0.0f);
  {
    float y = 0.0f;
    for (size_t li = 0; li < L; ++li) {
      float maxh = 0.0f;
      for (uint32_t v : layers[li]) maxh = std::max(maxh, out.boxes[v].size.y);
      layer_y[li] = y;
      // Center each node vertically within its layer band.
      for (uint32_t v : layers[li])
        out.boxes[v].pos.y = y + (maxh - out.boxes[v].size.y) * 0.5f;
      y += maxh + params.rank_sep;
    }
  }
  // x: pack by ordered position with node_sep gaps.
  auto pack_x = [&]() {
    for (size_t li = 0; li < L; ++li) {
      float x = 0.0f;
      for (uint32_t v : layers[li]) {
        out.boxes[v].pos.x = x;
        x += out.boxes[v].size.x + params.node_sep;
      }
    }
  };
  pack_x();
  // Median alignment pass: shift each node toward the median x-center of its
  // neighbors (both layers), then re-resolve overlaps left-to-right. One down +
  // one up pass keeps chains vertically aligned without expensive optimization.
  auto center_of = [&](uint32_t v) {
    return out.boxes[v].pos.x + out.boxes[v].size.x * 0.5f;
  };
  auto align_pass = [&](bool use_preds) {
    const auto& nbr = use_preds ? preds : succs;
    for (size_t li = 0; li < L; ++li) {
      auto& lv = layers[li];
      // Desired center per node from neighbor median.
      for (uint32_t v : lv) {
        if (nbr[v].empty()) continue;
        std::vector<float> cs;
        cs.reserve(nbr[v].size());
        for (uint32_t nb : nbr[v]) cs.push_back(center_of(nb));
        std::sort(cs.begin(), cs.end());
        float med = cs[cs.size() / 2];
        out.boxes[v].pos.x = med - out.boxes[v].size.x * 0.5f;
      }
      // Resolve overlaps left-to-right preserving order.
      for (size_t p = 1; p < lv.size(); ++p) {
        float min_x = out.boxes[lv[p - 1]].pos.x +
                      out.boxes[lv[p - 1]].size.x + params.node_sep;
        if (out.boxes[lv[p]].pos.x < min_x) out.boxes[lv[p]].pos.x = min_x;
      }
    }
  };
  align_pass(/*use_preds=*/true);
  align_pass(/*use_preds=*/false);

  // -- 6) Edge routing as cubic beziers. p0 producer bottom-center, p3 consumer
  // top-center, p1/p2 vertical control points for a smooth flow. Reversed edges
  // keep their ORIGINAL from/to display ids (visual direction) but flag it.
  out.edges.reserve(edges.size());
  for (const DEdge& e : edges) {
    EdgeCurve c;
    c.from_display_id = e.from;
    c.to_display_id = e.to;
    c.reversed = e.reversed;
    const NodeBox& fb = out.boxes[e.from];
    const NodeBox& tb = out.boxes[e.to];
    c.p0 = Vec2{fb.pos.x + fb.size.x * 0.5f, fb.pos.y + fb.size.y};
    c.p3 = Vec2{tb.pos.x + tb.size.x * 0.5f, tb.pos.y};
    float dy = (c.p3.y - c.p0.y) * 0.5f;
    c.p1 = Vec2{c.p0.x, c.p0.y + dy};
    c.p2 = Vec2{c.p3.x, c.p3.y - dy};
    out.edges.push_back(c);
  }

  // -- Bounds.
  float minx = std::numeric_limits<float>::max();
  float miny = std::numeric_limits<float>::max();
  float maxx = std::numeric_limits<float>::lowest();
  float maxy = std::numeric_limits<float>::lowest();
  for (const NodeBox& b : out.boxes) {
    minx = std::min(minx, b.pos.x);
    miny = std::min(miny, b.pos.y);
    maxx = std::max(maxx, b.pos.x + b.size.x);
    maxy = std::max(maxy, b.pos.y + b.size.y);
  }
  if (V == 0) {
    minx = miny = maxx = maxy = 0.0f;
  }
  out.bounds_min = Vec2{minx, miny};
  out.bounds_max = Vec2{maxx, maxy};
  if (progress) progress->set(1.0f, "layout: done");
  return out;
}

}  // namespace netvis
