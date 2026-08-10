// engine/LayoutEngine.cpp — from-scratch layered (Sugiyama) layout.
//
// PERF (spec §7.2): the whole pipeline is O(V+E) plus O(sweeps*E log E) for
// ordering, and holds no per-node heap allocation in any pass. Target: 10k
// display nodes in < 250 ms. Everything works on the CURRENT display-node list
// (already collapsed by CollapseTree, so V is usually small), in world
// coordinates, with no GUI dependency — node sizes come in through SizeFn so
// this runs headless.
//
// The "no per-node allocation" half of that used to be aspiration rather than
// fact. #98 profiled the 100k rung stage by stage and found the asymptotics were
// fine but the constants were not: six vector-of-vectors adjacencies and two
// scratch buffers rebuilt per layer added up to roughly a million small heap
// allocations, which was two thirds of the stage. The structures below are
// therefore flat (see Csr) and every scratch buffer is hoisted to the outermost
// scope that can own it. Nothing about the OUTPUT changed — positions are
// byte-identical, which is load-bearing because layouts are cached on disk by
// structure hash (LayoutCache).
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
#include <unordered_map>
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

// --- Compressed adjacency ----------------------------------------------------
//
// #98: every adjacency in this file used to be a
// std::vector<std::vector<uint32_t>>. Profiling the 100k rung showed why that is
// the wrong shape here: the synthetic ladder lays out ~112k layout nodes across
// ~100k layers, so each of the six adjacencies built below was ~100k separate
// heap blocks holding roughly one element each. The traversal was never the
// cost — the allocation traffic and the pointer chase were. CSR is two
// allocations for the whole structure and keeps each owner's entries contiguous.
//
// A slice of a Csr, deliberately shaped like the std::vector<uint32_t> it
// replaced (range-for, size, empty, indexing) so the ordering and coordinate
// passes below read unchanged.
struct CsrSpan {
  const uint32_t* first = nullptr;
  const uint32_t* last = nullptr;
  const uint32_t* begin() const { return first; }
  const uint32_t* end() const { return last; }
  size_t size() const { return static_cast<size_t>(last - first); }
  bool empty() const { return first == last; }
  uint32_t operator[](size_t i) const { return first[i]; }
};

// The same slice, writable. Only two passes need it: the clone duplication sorts
// a source's out-route list in place, and barycenter ordering permutes a layer.
// Both stay strictly inside one owner's slice, so the offsets remain valid.
struct CsrSpanMut {
  uint32_t* first = nullptr;
  uint32_t* last = nullptr;
  uint32_t* begin() const { return first; }
  uint32_t* end() const { return last; }
  size_t size() const { return static_cast<size_t>(last - first); }
  uint32_t& operator[](size_t i) const { return first[i]; }
};

struct Csr {
  // n+1 offsets: items[start[v] .. start[v+1]) are owner v's entries.
  std::vector<uint32_t> start;
  std::vector<uint32_t> items;

  CsrSpan operator[](size_t v) const {
    return CsrSpan{items.data() + start[v], items.data() + start[v + 1]};
  }
  CsrSpanMut slice(size_t v) {
    return CsrSpanMut{items.data() + start[v], items.data() + start[v + 1]};
  }
};

// Two-pass counting build. `emit_all(sink)` must call sink(owner, item) once per
// entry and produce the SAME sequence both times it is invoked — once to count,
// once to place. Within an owner the items land in emission order, which is
// byte-identical to the push_back order of the vector-of-vectors this replaces;
// every ordering pass downstream is order-sensitive, so determinism (spec §2.7)
// rests on that equivalence.
//
// Offsets are uint32_t because every entry stored here is an edge, route or
// segment index, all of which this file already carries in a uint32_t — the
// total therefore cannot exceed what those indices can address.
template <typename Fn>
Csr build_csr(uint32_t n, Fn&& emit_all) {
  Csr c;
  c.start.assign(static_cast<size_t>(n) + 1, 0);
  emit_all([&](uint32_t owner, uint32_t) { ++c.start[owner + 1]; });
  for (uint32_t v = 0; v < n; ++v) c.start[v + 1] += c.start[v];
  c.items.resize(c.start[n]);
  // Per-owner write cursor, seeded with each owner's first free slot.
  std::vector<uint32_t> cursor(c.start.begin(), c.start.end() - 1);
  emit_all(
      [&](uint32_t owner, uint32_t item) { c.items[cursor[owner]++] = item; });
  return c;
}

// Internal directed edge between display nodes.
struct DEdge {
  uint32_t from = 0;
  uint32_t to = 0;
  bool reversed = false;
  // #18: representative IR value index for this (from,to) display pair — the
  // SMALLEST value index (ascending) among all values producer->consumer, or
  // UINT32_MAX if none. Carried to the emitted EdgeCurve for the hover tooltip.
  uint32_t value_index = UINT32_MAX;
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

  // #61: cooperative cancellation. The main thread flips the ProgressSink's
  // cancel flag; we poll it ONLY at the coarse `progress->set(...)` checkpoints
  // below (and once per barycenter sweep) — never in the tight per-node inner
  // loops — so the poll stays O(checkpoints) and the layout stays deterministic.
  // On cancel we return a fresh result flagged cancelled (no usable boxes); the
  // caller must not publish it. Poll only when a sink was supplied.
  auto make_cancelled = []() {
    LayoutResult c;
    c.cancelled = true;
    return c;
  };

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
  if (progress && progress->cancelled()) return make_cancelled();

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
    // Each entry: a packed (producer<<32|consumer) dedup key + the IR value
    // index that realizes that display edge. Sorted by (key, vidx) so the FIRST
    // entry for each unique key carries the SMALLEST value index (#18). Edge
    // ORDER is byte-identical to the old key-only sort: the primary sort key
    // (the packed producer/consumer) is unchanged, and we still emit exactly one
    // edge per unique key in ascending key order — vidx is only a tiebreak that
    // never reorders across keys.
    std::vector<std::pair<uint64_t, uint32_t>> refs;  // (key, vidx)
    refs.reserve(g.edge_refs.size());
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
        uint64_t key = (static_cast<uint64_t>(producer) << 32) | consumer;
        refs.push_back({key, vidx});
      }
    }
    std::sort(refs.begin(), refs.end());
    edges.reserve(refs.size());
    // Producer/consumer are never UINT32_MAX here (both filtered above), so a
    // real key can never equal UINT64_MAX — safe "no previous key" sentinel.
    uint64_t prev_key = UINT64_MAX;
    for (const auto& rf : refs) {
      if (rf.first == prev_key) continue;  // dedup: keep first (smallest vidx)
      prev_key = rf.first;
      DEdge de;
      de.from = static_cast<uint32_t>(rf.first >> 32);
      de.to = static_cast<uint32_t>(rf.first & 0xffffffffu);
      de.value_index = rf.second;
      edges.push_back(de);
    }
  }
  if (progress) progress->set(0.3f, "layout: edges");
  if (progress && progress->cancelled()) return make_cancelled();

  // Adjacency (out) for the DFS cycle-break, as edge indices per source node.
  const Csr out_adj = build_csr(V, [&](auto&& emit) {
    for (uint32_t ei = 0; ei < edges.size(); ++ei) emit(edges[ei].from, ei);
  });

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
        const CsrSpan uo = out_adj[u];
        if (ci < uo.size()) {
          uint32_t ei = uo[ci++];
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

  // Effective (post-reversal) successors and in-degrees over the display graph.
  // #98: sections 3, 3b and the clone duplication further down each used to
  // derive this pair for themselves — three identical scans over the same edge
  // list, each materializing its own V-entry vector-of-vectors. Derive it once
  // here; the passes that consume in-degree destructively work on a copy.
  std::vector<uint32_t> indeg(V, 0);
  for (const DEdge& e : edges) ++indeg[eff_to(e)];
  const Csr eff_out = build_csr(V, [&](auto&& emit) {
    for (const DEdge& e : edges) emit(eff_from(e), eff_to(e));
  });

  // -- 3) Longest-path layering. layer(v) = max over in-edges of layer(src)+1.
  // Compute via a topological pass using Kahn on the acyclic (post-reversal)
  // graph. O(V+E).
  std::vector<int32_t> layer(V, 0);
  {
    std::vector<uint32_t> pending = indeg;  // Kahn drains this to zero
    std::vector<uint32_t> ready;
    for (uint32_t v = 0; v < V; ++v)
      if (pending[v] == 0) ready.push_back(v);
    std::sort(ready.begin(), ready.end());
    size_t head = 0;
    // Process deterministically: maintain sorted-ish by popping in order.
    while (head < ready.size()) {
      uint32_t u = ready[head++];
      for (uint32_t w : eff_out[u]) {
        if (layer[w] < layer[u] + 1) layer[w] = layer[u] + 1;
        if (--pending[w] == 0) ready.push_back(w);
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
    for (uint32_t v = 0; v < V; ++v) {
      if (indeg[v] != 0) continue;    // only sources
      const CsrSpan outs = eff_out[v];
      if (outs.empty()) continue;     // isolated node: leave at 0
      int32_t min_consumer = std::numeric_limits<int32_t>::max();
      for (uint32_t w : outs)
        min_consumer = std::min(min_consumer, layer[w]);
      // Sit one layer above the nearest consumer (never below 0).
      if (min_consumer != std::numeric_limits<int32_t>::max())
        layer[v] = std::max(0, min_consumer - 1);
    }
  }

  // ==========================================================================
  //  v0.2.0 LAYOUT-NODE MODEL (LayoutCache kVersion v3). All ordering and
  //  coordinate passes below run on INTERNAL arrays indexed by LAYOUT-node id:
  //  indices [0,V) are the real display nodes, then per-consumer CLONES of
  //  multi-consumer sources, then Sugiyama DUMMY nodes for long edges. Only real
  //  + clone nodes become out.boxes (a clone shares its source's display_id, so
  //  out.boxes may exceed display_nodes().size() and several boxes may share a
  //  display_id — every view consumer keys off box.display_id, not box index).
  //  Dummies are bezier waypoints only and are never emitted.
  // ==========================================================================
  constexpr uint32_t kDummyOwner = UINT32_MAX;
  std::vector<Vec2> nsize;      // per layout-node measured size
  std::vector<int32_t> nlayer;  // per layout-node layer
  std::vector<uint32_t> nowner; // owning display id, or kDummyOwner for a dummy
  nsize.reserve(static_cast<size_t>(V) * 2);
  nlayer.reserve(static_cast<size_t>(V) * 2);
  nowner.reserve(static_cast<size_t>(V) * 2);
  for (uint32_t i = 0; i < V; ++i) {
    nsize.push_back(out.boxes[i].size);
    nlayer.push_back(layer[i]);
    nowner.push_back(i);
  }

  // A routed edge in EFFECTIVE (acyclic) direction u(upper)->v(lower). from_disp/
  // to_disp/reversed carry the ORIGINAL display-space direction for the emitted
  // EdgeCurve; `chain` holds intermediate dummy node ids (top->bottom).
  struct RouteEdge {
    uint32_t u, v;
    uint32_t from_disp, to_disp;
    bool reversed;
    std::vector<uint32_t> chain;
    // #18: representative IR value index for this display edge, carried straight
    // through from DEdge. Independent of layout direction/reversal and of dummy
    // insertion — it only names WHICH value the edge stands for.
    uint32_t value_index;
  };
  std::vector<RouteEdge> routes;
  routes.reserve(edges.size());
  for (const DEdge& e : edges)
    routes.push_back(RouteEdge{eff_from(e), eff_to(e), e.from, e.to, e.reversed,
                               {}, e.value_index});

  // -- Multi-consumer source duplication. A real, non-group source (effective
  // in-degree 0) feeding >=2 consumers emits one long edge per consumer (the
  // hairball). Clone it once per EXTRA consumer so each clone sits just above a
  // single consumer. The first consumer (ascending by (layer, id)) keeps the
  // original node; clones re-point that consumer's edge. Deterministic.
  constexpr uint32_t kMaxDuplicate = 64;  // fan-out cap: above this stay shared
  {
    // Route indices leaving each display node. The effective in-degree this pass
    // needs is exactly `indeg` from the layering section: `routes` was built 1:1
    // from `edges` with (u,v) = (eff_from, eff_to), and no route has been
    // re-pointed yet — only `u` is ever re-pointed, and only in the loop below.
    // So it is reused rather than recounted (#98).
    Csr out_routes = build_csr(V, [&](auto&& emit) {
      for (uint32_t ri = 0; ri < routes.size(); ++ri)
        if (routes[ri].u < V) emit(routes[ri].u, ri);
    });
    for (uint32_t s = 0; s < V; ++s) {
      if (indeg[s] != 0 || disp[s].is_group) continue;  // real sources only
      const CsrSpanMut outs = out_routes.slice(s);
      if (outs.size() < 2 || outs.size() > kMaxDuplicate) continue;
      std::sort(outs.begin(), outs.end(), [&](uint32_t a, uint32_t b) {
        uint32_t va = routes[a].v, vb = routes[b].v;
        if (nlayer[va] != nlayer[vb]) return nlayer[va] < nlayer[vb];
        return va < vb;  // deterministic tiebreak
      });
      nlayer[s] = std::max(0, nlayer[routes[outs[0]].v] - 1);  // keep, place
      for (size_t k = 1; k < outs.size(); ++k) {
        RouteEdge& r = routes[outs[k]];
        uint32_t clone = static_cast<uint32_t>(nsize.size());
        nsize.push_back(nsize[s]);
        nlayer.push_back(std::max(0, nlayer[r.v] - 1));
        nowner.push_back(nowner[s]);  // clone shares the source's display id
        r.u = clone;
      }
    }
  }

  // -- Sugiyama dummy nodes: for a route spanning D>1 layers, insert D-1 dummies
  // on the intermediate layers so crossing reduction converges and the edge can
  // bend around nodes. Capped per edge (above the cap, route straight).
  constexpr uint32_t kMaxDummiesPerEdge = 128;
  for (RouteEdge& r : routes) {
    int32_t lu = nlayer[r.u], lv = nlayer[r.v];
    int32_t d = lv - lu;
    if (d <= 1 || static_cast<uint32_t>(d - 1) > kMaxDummiesPerEdge) continue;
    for (int32_t li = lu + 1; li < lv; ++li) {
      uint32_t dm = static_cast<uint32_t>(nsize.size());
      nsize.push_back(Vec2{1.0f, 1.0f});
      nlayer.push_back(li);
      nowner.push_back(kDummyOwner);
      r.chain.push_back(dm);
    }
  }

  const uint32_t M = static_cast<uint32_t>(nsize.size());

  // Unit segments (each spans exactly one layer) drive ordering/crossings/align.
  // Same-layer degenerate segments (a clone level with its consumer) are dropped
  // from adjacency — they cannot be ordered across a layer — but the route still
  // renders. NB: routing uses `routes`, ordering uses `segs`.
  std::vector<std::pair<uint32_t, uint32_t>> segs;  // (upper, lower)
  segs.reserve(routes.size());
  auto add_seg = [&](uint32_t a, uint32_t b) {
    if (nlayer[b] == nlayer[a] + 1) segs.push_back({a, b});
  };
  for (const RouteEdge& r : routes) {
    if (r.chain.empty()) {
      add_seg(r.u, r.v);
    } else {
      add_seg(r.u, r.chain.front());
      for (size_t i = 0; i + 1 < r.chain.size(); ++i)
        add_seg(r.chain[i], r.chain[i + 1]);
      add_seg(r.chain.back(), r.v);
    }
  }

  int32_t max_layer = 0;
  for (uint32_t v = 0; v < M; ++v) max_layer = std::max(max_layer, nlayer[v]);
  const size_t L = static_cast<size_t>(max_layer) + 1;
  if (progress) progress->set(0.5f, "layout: layering");
  if (progress && progress->cancelled()) return make_cancelled();

  // Layer membership over ALL layout nodes; order index within a layer. Each
  // layer's members are contiguous (#98) and the ordering passes below permute
  // them strictly inside their own slice, so the offsets stay valid for the rest
  // of the pipeline. Emitting v ascending already leaves every slice sorted
  // ascending — which is the "initial order by layout-node id" the old code
  // spelled out with a per-layer std::sort that could never move anything.
  Csr layers = build_csr(static_cast<uint32_t>(L), [&](auto&& emit) {
    for (uint32_t v = 0; v < M; ++v)
      emit(static_cast<uint32_t>(nlayer[v]), v);
  });

  std::vector<uint32_t> order_idx(M, 0);
  auto refresh_order = [&]() {
    for (size_t li = 0; li < L; ++li) {
      const uint32_t b = layers.start[li];
      const uint32_t e = layers.start[li + 1];
      for (uint32_t k = b; k < e; ++k) order_idx[layers.items[k]] = k - b;
    }
  };
  refresh_order();

  // Predecessor / successor adjacency (effective direction) for barycenters.
  const Csr succs = build_csr(M, [&](auto&& emit) {
    for (const auto& s : segs) emit(s.first, s.second);
  });
  const Csr preds = build_csr(M, [&](auto&& emit) {
    for (const auto& s : segs) emit(s.second, s.first);
  });

  // Count total crossings between all adjacent layer pairs (for early-stop).
  // PERF: crossings between two layers = inversions in the lower endpoints once
  // edges are ordered by upper endpoint. We count inversions in O(E log E) with
  // a Fenwick tree (BIT), NOT the naive O(E^2) double loop — a wide graph with
  // no collapse can put thousands of edges between one layer pair, where O(E^2)
  // would blow the sub-250ms budget.
  auto count_crossings = [&]() -> uint64_t {
    uint64_t total = 0;
    std::vector<std::pair<uint32_t, uint32_t>> pairs;  // (upper_pos, lower_pos)
    std::vector<uint32_t> bit;
    for (size_t li = 0; li + 1 < L; ++li) {
      pairs.clear();
      for (uint32_t u : layers[li])
        for (uint32_t w : succs[u])
          if (static_cast<size_t>(nlayer[w]) == li + 1)
            pairs.push_back({order_idx[u], order_idx[w]});
      if (pairs.size() < 2) continue;
      // Order by upper endpoint (ties by lower) so that, scanning left to right,
      // a crossing is exactly a previously-seen edge with a GREATER lower pos.
      std::sort(pairs.begin(), pairs.end());
      const size_t width = layers[li + 1].size();
      bit.assign(width + 1, 0);
      size_t seen = 0;
      for (const auto& s : pairs) {
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
  //
  // #98: `keyed` is hoisted out of the lambda and reused. barycenter_sort runs
  // once per layer per sweep direction — 200k calls at the 100k rung — so a
  // vector constructed inside it was 200k malloc/free pairs, and that, not the
  // sorting, was the single largest line item in the whole layout stage.
  std::vector<std::pair<float, uint32_t>> keyed;
  auto barycenter_sort = [&](size_t li, bool use_preds) {
    const CsrSpanMut lv = layers.slice(li);
    // A layer holding fewer than two nodes has exactly one possible ordering, so
    // everything below is provably a no-op on it. Bailing here is not a micro-
    // optimization: deep graphs are mostly one-node layers, and std::stable_sort
    // heap-allocates its temporary buffer even for a single element.
    if (lv.size() < 2) return;
    const Csr& nbr = use_preds ? preds : succs;
    // Compute barycenter key per node; nodes with no neighbor keep their pos.
    keyed.clear();
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
      // #61: poll cancellation once per sweep (the sweep loop is the dominant
      // cost on large graphs). Coarse-grained: never inside the per-layer loops.
      if (progress && progress->cancelled()) return make_cancelled();
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
  if (progress && progress->cancelled()) return make_cancelled();

  // -- 5) Coordinate assignment, on the internal per-layout-node arrays.
  std::vector<Vec2> npos(M);
  // y: accumulate per layer top-down (max node height in layer + rank_sep).
  {
    float y = 0.0f;
    for (size_t li = 0; li < L; ++li) {
      float maxh = 0.0f;
      for (uint32_t v : layers[li]) maxh = std::max(maxh, nsize[v].y);
      // Center each node vertically within its layer band.
      for (uint32_t v : layers[li])
        npos[v].y = y + (maxh - nsize[v].y) * 0.5f;
      y += maxh + params.rank_sep;
    }
  }
  // x: pack by ordered position with node_sep gaps.
  auto pack_x = [&]() {
    for (size_t li = 0; li < L; ++li) {
      float x = 0.0f;
      for (uint32_t v : layers[li]) {
        npos[v].x = x;
        x += nsize[v].x + params.node_sep;
      }
    }
  };
  pack_x();
  // Median alignment pass: shift each node toward the median x-center of its
  // neighbors (both layers), then re-resolve overlaps. One down + one up pass
  // keeps chains vertically aligned without expensive optimization.
  //
  // SYMMETRY (v0.2.1 drift fix): both the median pick and the overlap resolution
  // must be left/right symmetric, or the layout drifts consistently rightward
  // with depth (minimap collapses to a diagonal line). The two bugs were:
  //   (1) an EVEN neighbor count took the upper median cs[k/2] — the RIGHT
  //       neighbor — so every 2-input node chased its right input. Use the true
  //       median (mean of the two middle values) for even counts.
  //   (2) overlap resolution ran a single left-to-right pass that only ever
  //       PUSHED RIGHT, a ratchet that accumulated layer over layer. Instead
  //       resolve symmetrically: average a left-packed solution (biases right)
  //       with a right-packed one (biases left). Averaging two gap-feasible
  //       monotone solutions is itself gap-feasible and is centered, so there is
  //       no net directional push.
  auto center_of = [&](uint32_t v) { return npos[v].x + nsize[v].x * 0.5f; };
  // Width of the widest layer, which is all the scratch below ever needs.
  size_t widest = 0;
  for (size_t li = 0; li < L; ++li) widest = std::max(widest, layers[li].size());
  // #98: `want`/`lo`/`hi` are sized ONCE for the widest layer instead of being
  // re-assigned per layer, and `cs` is hoisted out of the per-layer loop. Those
  // four buffers were four vector fills per layer per pass, which on a deep
  // graph (the 100k rung is ~100k layers of ~1 node) is the whole cost of this
  // pass. Reading a stale entry is impossible: each of want/lo/hi has every
  // slot in [0,k) written before anything reads it, on every layer.
  auto align_pass = [&](bool use_preds) {
    const Csr& nbr = use_preds ? preds : succs;
    std::vector<float> want(widest, 0.0f), lo(widest, 0.0f), hi(widest, 0.0f);
    std::vector<float> cs;
    for (size_t li = 0; li < L; ++li) {
      const CsrSpan lv = layers[li];
      const size_t k = lv.size();
      if (k == 0) continue;
      // Desired center per node = TRUE median of neighbor centers; a node with
      // no neighbor keeps its current center.
      for (size_t p = 0; p < k; ++p) {
        uint32_t v = lv[p];
        const CsrSpan nb_of_v = nbr[v];
        if (nb_of_v.empty()) { want[p] = center_of(v); continue; }
        cs.clear();
        cs.reserve(nb_of_v.size());
        for (uint32_t nb : nb_of_v) cs.push_back(center_of(nb));
        std::sort(cs.begin(), cs.end());
        const size_t c = cs.size();
        want[p] = (c & 1u) ? cs[c / 2] : (cs[c / 2 - 1] + cs[c / 2]) * 0.5f;
      }
      // Minimum center-to-center gap between ordered neighbors p-1,p.
      auto gap = [&](size_t p) {
        return (nsize[lv[p - 1]].x + nsize[lv[p]].x) * 0.5f + params.node_sep;
      };
      // Left-packed (only pushes right).
      lo[0] = want[0];
      for (size_t p = 1; p < k; ++p)
        lo[p] = std::max(want[p], lo[p - 1] + gap(p));
      // Right-packed (only pushes left).
      hi[k - 1] = want[k - 1];
      for (size_t p = k - 1; p-- > 0;)
        hi[p] = std::min(want[p], hi[p + 1] - gap(p + 1));
      // Centered average of the two feasible solutions.
      for (size_t p = 0; p < k; ++p) {
        float center = (lo[p] + hi[p]) * 0.5f;
        npos[lv[p]].x = center - nsize[lv[p]].x * 0.5f;
      }
    }
  };
  align_pass(/*use_preds=*/true);
  align_pass(/*use_preds=*/false);

  // -- De-shear (v0.2.1). Median alignment against dummy "lanes" from skip/
  // residual edges leaves a systematic horizontal SHEAR: layer centroids grow
  // roughly linearly with layer index, so the whole graph marches down-and-to-
  // the-right and the minimap collapses to a diagonal line. The shear is a
  // linear trend in the per-layer centroid; fit it by least squares over the
  // occupied layers and subtract b*layer from every node. This removes only the
  // global drift — all within-layer order and relative offsets (and thus a
  // straight chain, whose slope contribution is 0) are preserved. O(V+L).
  {
    // Per-layer centroid over ALL layout nodes (dummies included: they are part
    // of the visual flow and drive the shear).
    std::vector<double> csum(L, 0.0);
    std::vector<uint32_t> ccnt(L, 0);
    for (uint32_t v = 0; v < M; ++v) {
      csum[static_cast<size_t>(nlayer[v])] += center_of(v);
      ++ccnt[static_cast<size_t>(nlayer[v])];
    }
    // Least-squares slope of centroid vs layer index over occupied layers.
    double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t li = 0; li < L; ++li) {
      if (!ccnt[li]) continue;
      double x = static_cast<double>(li);
      double y = csum[li] / ccnt[li];
      n += 1; sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    double denom = n * sxx - sx * sx;
    if (n >= 2 && denom > 1e-6) {
      double slope = (n * sxy - sx * sy) / denom;
      for (uint32_t v = 0; v < M; ++v)
        npos[v].x -= static_cast<float>(slope * nlayer[v]);
    }
  }

  // -- Emit boxes: ONLY real + clone layout nodes (dummies are waypoints). Each
  // box carries its OWNING display id; clones share their source's display id so
  // the view's display_id-keyed lookups still resolve. boxes may therefore be
  // longer than display_nodes() and share display ids (documented, kVersion v3).
  out.boxes.clear();
  out.boxes.reserve(M);
  for (uint32_t v = 0; v < M; ++v) {
    if (nowner[v] == kDummyOwner) continue;  // dummy: no box
    NodeBox b;
    b.display_id = nowner[v];
    b.pos = npos[v];
    b.size = nsize[v];
    b.layer = nlayer[v];
    out.boxes.push_back(b);
  }

  // -- 6) Edge routing as cubic beziers through any dummy waypoints. p0 at the
  // source's bottom-center, p3 at the consumer's top-center. With dummies the
  // control points are pulled toward the first/last waypoint x so the edge bends
  // around intervening layers instead of cutting straight through them. Reversed
  // edges keep their ORIGINAL display-space from/to (visual direction) + flag.
  auto bottom_center = [&](uint32_t v) {
    return Vec2{npos[v].x + nsize[v].x * 0.5f, npos[v].y + nsize[v].y};
  };
  auto top_center = [&](uint32_t v) {
    return Vec2{npos[v].x + nsize[v].x * 0.5f, npos[v].y};
  };
  out.edges.reserve(routes.size());
  for (const RouteEdge& r : routes) {
    EdgeCurve c;
    c.from_display_id = r.from_disp;
    c.to_display_id = r.to_disp;
    c.reversed = r.reversed;
    c.value_index = r.value_index;  // #18: representative IR value on this edge
    c.p0 = bottom_center(r.u);
    c.p3 = top_center(r.v);
    float dy = (c.p3.y - c.p0.y) * 0.5f;
    if (r.chain.empty()) {
      c.p1 = Vec2{c.p0.x, c.p0.y + dy};
      c.p2 = Vec2{c.p3.x, c.p3.y - dy};
    } else {
      // Bend toward the first/last dummy waypoint centers.
      float fx = npos[r.chain.front()].x + nsize[r.chain.front()].x * 0.5f;
      float lx = npos[r.chain.back()].x + nsize[r.chain.back()].x * 0.5f;
      c.p1 = Vec2{fx, c.p0.y + dy};
      c.p2 = Vec2{lx, c.p3.y - dy};
    }
    out.edges.push_back(c);
  }

  // -- Bounds (over emitted boxes only).
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
  if (out.boxes.empty()) {
    minx = miny = maxx = maxy = 0.0f;
  }
  out.bounds_min = Vec2{minx, miny};
  out.bounds_max = Vec2{maxx, maxy};
  if (progress) progress->set(1.0f, "layout: done");
  return out;
}

}  // namespace netvis
