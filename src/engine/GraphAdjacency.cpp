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

}  // namespace netvis
