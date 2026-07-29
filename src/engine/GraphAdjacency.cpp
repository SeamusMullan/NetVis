// engine/GraphAdjacency.cpp — CSR forward/reverse adjacency over one IR graph.
//
// PERF/CORRECTNESS: one O(V+E) pass builds both directions. We mirror the edge
// derivation LayoutEngine/CollapseTree use — an edge u->v exists when a value
// produced by node u (values[vidx].producer == u) is consumed by node v via one
// of v's input slots (edge_refs[inputs.begin .. inputs.begin+count)). Neighbor
// lists are sorted ascending and de-duplicated so queries are deterministic and
// parallel edges collapse to one. reachable_* do a bounded BFS (max_hops + a
// visited cap) to protect the frame budget on pathological graphs.
#include "engine/GraphAdjacency.h"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <utility>

#include "core/SafeMath.h"  // safe_add (single source for saturating arithmetic)

namespace netvis {

void GraphAdjacency::build(const ir::Model& model, uint32_t graph_index) {
  node_count_ = 0;
  succ_off_.clear();
  succ_val_.clear();
  pred_off_.clear();
  pred_val_.clear();

  if (graph_index >= model.graphs.size()) return;
  const ir::Graph& g = model.graphs[graph_index];
  const size_t n = g.nodes.size();
  node_count_ = static_cast<uint32_t>(n);
  if (n == 0) return;

  // Collect directed edges (u -> v) as pairs, then bucket into CSR. Reserve on
  // an edge-count estimate (sum of input arities) to avoid per-edge reallocation.
  size_t edge_estimate = 0;
  for (size_t i = 0; i < n; ++i) edge_estimate += g.nodes[i].inputs.count;

  std::vector<std::pair<uint32_t, uint32_t>> edges;  // (u=producer, v=consumer)
  edges.reserve(edge_estimate);

  for (uint32_t v = 0; v < n; ++v) {
    const ir::Node& nd = g.nodes[v];
    const uint32_t b = nd.inputs.begin;
    const uint32_t e = nd.inputs.begin + nd.inputs.count;
    for (uint32_t k = b; k < e && k < g.edge_refs.size(); ++k) {
      uint32_t vidx = g.edge_refs[k];
      if (vidx >= g.values.size()) continue;
      int32_t p = g.values[vidx].producer;
      if (p < 0 || static_cast<size_t>(p) >= n) continue;
      uint32_t u = static_cast<uint32_t>(p);
      if (u == v) continue;  // drop self-loops
      edges.emplace_back(u, v);
    }
  }

  // Build forward CSR (succ) via counting sort on u, then dedupe within each row.
  auto build_csr = [&](bool forward, std::vector<uint32_t>& off,
                       std::vector<uint32_t>& val) {
    off.assign(n + 1, 0);
    // Count out-degree per source key.
    for (const auto& pr : edges) {
      uint32_t key = forward ? pr.first : pr.second;
      ++off[key + 1];
    }
    for (size_t i = 0; i < n; ++i) off[i + 1] += off[i];
    val.assign(off[n], 0);
    std::vector<uint32_t> cursor(off.begin(), off.end() - 1);
    for (const auto& pr : edges) {
      uint32_t key = forward ? pr.first : pr.second;
      uint32_t nbr = forward ? pr.second : pr.first;
      val[cursor[key]++] = nbr;
    }
    // Sort + dedupe each row, compacting val and rewriting offsets.
    std::vector<uint32_t> compact;
    compact.reserve(val.size());
    std::vector<uint32_t> new_off(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
      uint32_t row_begin = off[i];
      uint32_t row_end = off[i + 1];
      std::sort(val.begin() + row_begin, val.begin() + row_end);
      new_off[i] = static_cast<uint32_t>(compact.size());
      uint32_t last = UINT32_MAX;
      bool have = false;
      for (uint32_t j = row_begin; j < row_end; ++j) {
        if (have && val[j] == last) continue;
        compact.push_back(val[j]);
        last = val[j];
        have = true;
      }
    }
    new_off[n] = static_cast<uint32_t>(compact.size());
    off = std::move(new_off);
    val = std::move(compact);
  };

  build_csr(true, succ_off_, succ_val_);
  build_csr(false, pred_off_, pred_val_);
}

namespace {

// Bounded BFS over a CSR adjacency. Excludes `start`; result is ascending.
std::vector<uint32_t> bfs(const std::vector<uint32_t>& off,
                          const std::vector<uint32_t>& val, uint32_t node_count,
                          uint32_t start, uint32_t max_hops, uint32_t cap) {
  std::vector<uint32_t> out;
  if (start >= node_count || cap == 0) return out;

  std::vector<bool> visited(node_count, false);
  visited[start] = true;
  std::queue<std::pair<uint32_t, uint32_t>> q;  // (node, hops)
  q.emplace(start, 0u);

  while (!q.empty()) {
    auto [cur, hops] = q.front();
    q.pop();
    if (hops >= max_hops) continue;  // do not expand beyond the hop budget
    uint32_t rb = off[cur];
    uint32_t re = off[cur + 1];
    for (uint32_t j = rb; j < re; ++j) {
      uint32_t nbr = val[j];
      if (visited[nbr]) continue;
      visited[nbr] = true;
      out.push_back(nbr);
      if (out.size() >= cap) {
        std::sort(out.begin(), out.end());
        return out;
      }
      q.emplace(nbr, hops + 1);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

std::vector<uint32_t> GraphAdjacency::reachable_succ(uint32_t start,
                                                     uint32_t max_hops,
                                                     uint32_t cap) const {
  return bfs(succ_off_, succ_val_, node_count_, start, max_hops, cap);
}

std::vector<uint32_t> GraphAdjacency::reachable_pred(uint32_t start,
                                                     uint32_t max_hops,
                                                     uint32_t cap) const {
  return bfs(pred_off_, pred_val_, node_count_, start, max_hops, cap);
}

std::vector<uint32_t> GraphAdjacency::nodes_on_paths_between(uint32_t a,
                                                             uint32_t b,
                                                             uint32_t cap) const {
  std::vector<uint32_t> out;
  if (a == b || a >= node_count_ || b >= node_count_) return out;

  // Mark, in `on_path`, every node lying on some directed path for a given
  // orientation (up -> .. -> down). A node n qualifies iff n is forward-
  // reachable from `up` (inclusive of up) AND backward-reachable from `down`
  // (inclusive of down). We first require `down` to be forward-reachable from
  // `up`, else no connecting path exists for this orientation and it adds
  // nothing. Both BFS walks are cap-bounded to protect the frame budget.
  std::vector<uint8_t> on_path(node_count_, 0);
  auto accumulate = [&](uint32_t up, uint32_t down) {
    std::vector<uint32_t> fwd = reachable_succ(up, UINT32_MAX, cap);
    // Connecting forward path exists only if `down` is among up's successors.
    if (!std::binary_search(fwd.begin(), fwd.end(), down)) return;

    // S = forward-reachable from up, inclusive of up.
    std::vector<uint8_t> in_s(node_count_, 0);
    in_s[up] = 1;
    for (uint32_t n : fwd) in_s[n] = 1;

    // P = backward-reachable from down, inclusive of down. Intersect with S.
    // up and down bound the path and are in S∩P, so include them directly.
    on_path[up] = 1;
    on_path[down] = 1;
    std::vector<uint32_t> bwd = reachable_pred(down, UINT32_MAX, cap);
    for (uint32_t n : bwd)
      if (in_s[n]) on_path[n] = 1;
  };

  accumulate(a, b);  // ordering 1: a upstream, b downstream
  accumulate(b, a);  // ordering 2: b upstream, a downstream

  // Collect set bits: naturally ascending and deduped.
  for (uint32_t n = 0; n < node_count_; ++n)
    if (on_path[n]) out.push_back(n);
  return out;
}

std::vector<uint32_t> GraphAdjacency::longest_cost_path(
    const std::vector<uint64_t>& weight) const {
  std::vector<uint32_t> result;
  const uint32_t n = node_count_;
  if (n == 0) return result;  // empty graph => empty chain

  // Missing weight entries are 0 (weight may be shorter/empty than node_count()).
  auto w_at = [&](uint32_t i) -> uint64_t {
    return i < weight.size() ? weight[i] : 0;
  };
  // Saturating add (core/SafeMath, the codebase's single source for this) so a
  // pathological summed cost clamps at UINT64_MAX rather than wrapping — per-node
  // weights are already saturated flops-style quantities.
  auto sat_add = [](uint64_t a, uint64_t b) -> uint64_t { return safe_add(a, b); };

  // best[v] = max summed per-node weight of a path ENDING at v (v included).
  // parent[v] = the argmax predecessor (smallest index on ties), kNone for a
  // source. Seeded to the no-predecessor value w[v].
  constexpr uint32_t kNone = UINT32_MAX;
  std::vector<uint64_t> best(n);
  std::vector<uint32_t> parent(n, kNone);
  std::vector<uint32_t> indeg(n, 0);
  for (uint32_t v = 0; v < n; ++v) best[v] = w_at(v);

  // Indegree from the forward (succ) edges — dedup already collapsed parallels.
  for (uint32_t u = 0; u < n; ++u)
    for (uint32_t j = succ_off_[u]; j < succ_off_[u + 1]; ++j)
      ++indeg[succ_val_[j]];

  // Kahn queue, seeded with sources in ascending index order (determinism).
  std::queue<uint32_t> q;
  for (uint32_t v = 0; v < n; ++v)
    if (indeg[v] == 0) q.push(v);

  // DP over the Kahn order. Relax only edges out of popped (finalized) nodes, so
  // a residual cycle (indegree never reaching 0) simply never propagates and the
  // pass still terminates — its unvisited nodes keep best = w[v].
  while (!q.empty()) {
    uint32_t u = q.front();
    q.pop();
    for (uint32_t j = succ_off_[u]; j < succ_off_[u + 1]; ++j) {
      uint32_t v = succ_val_[j];
      uint64_t cand = sat_add(best[u], w_at(v));
      // Strictly better cost wins; equal cost keeps the smallest-index pred.
      if (cand > best[v] ||
          (cand == best[v] && (parent[v] == kNone || u < parent[v]))) {
        best[v] = cand;
        parent[v] = u;
      }
      if (indeg[v] > 0 && --indeg[v] == 0) q.push(v);
    }
  }

  // Sink = global argmax best[v] (smallest index on ties via strict >).
  uint32_t sink = 0;
  for (uint32_t v = 1; v < n; ++v)
    if (best[v] > best[sink]) sink = v;

  // Walk parent[] back to a source; the seen-guard bounds it to n steps even on
  // hostile input (parent only ever points at an earlier-finalized node).
  std::vector<bool> seen(n, false);
  for (uint32_t cur = sink; cur != kNone && !seen[cur]; cur = parent[cur]) {
    seen[cur] = true;
    result.push_back(cur);
  }
  std::reverse(result.begin(), result.end());  // source -> sink (ascending topo)
  return result;
}

}  // namespace netvis
