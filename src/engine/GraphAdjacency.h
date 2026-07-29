// engine/GraphAdjacency.h — CSR forward/reverse adjacency over one graph.
//
// DECISION (v0.2.0 graph navigation): highlight fan-in/out, focus neighborhoods,
// and jump-to-producer all need cheap predecessor/successor queries over the IR
// compute graph. We build a compressed-sparse-row (CSR) adjacency keyed by IR
// node index in a single O(V+E) pass. An edge u->v exists when a value produced
// by node u is consumed by node v (mirrors LayoutEngine's edge derivation, but
// over IR nodes, not display nodes). Cheap enough to build synchronously on the
// main thread on model/graph change — no worker, so no ir::Model lifetime race.
#pragma once

#include <cstdint>
#include <vector>

#include "ir/IR.h"

namespace netvis {

class GraphAdjacency {
 public:
  // Build for graphs[graph_index]. Deterministic; neighbor lists are ascending
  // and deduped. Safe on any input (out-of-range graph => empty adjacency).
  void build(const ir::Model& model, uint32_t graph_index);

  uint32_t node_count() const { return node_count_; }

  // Raw CSR access. Successors of node n are succ_values()[succ_offsets()[n] ..
  // succ_offsets()[n+1]). offsets has node_count()+1 entries (0 when empty).
  const std::vector<uint32_t>& succ_offsets() const { return succ_off_; }
  const std::vector<uint32_t>& succ_values() const { return succ_val_; }
  const std::vector<uint32_t>& pred_offsets() const { return pred_off_; }
  const std::vector<uint32_t>& pred_values() const { return pred_val_; }

  // Transitive successors/predecessors of `start` via BFS, up to `max_hops`
  // (UINT32_MAX = unbounded) and bounded to `cap` visited nodes to protect the
  // frame budget on pathological graphs. Excludes `start`. Result is ascending.
  std::vector<uint32_t> reachable_succ(uint32_t start, uint32_t max_hops,
                                       uint32_t cap) const;
  std::vector<uint32_t> reachable_pred(uint32_t start, uint32_t max_hops,
                                       uint32_t cap) const;

  // #15 (path-between): IR nodes lying on ANY directed path connecting `a` and
  // `b`, in EITHER direction (a->..->b or b->..->a), inclusive of a and b when a
  // connecting path exists. Computed as the intersection of forward reachability
  // from the upstream endpoint with backward reachability from the downstream
  // endpoint, tried both orderings and unioned (the caller need not know which of
  // a/b is upstream). Bounded to `cap` visited nodes per BFS (frame-budget guard,
  // as with reachable_*). Result is ascending and deduped. Empty if a==b, either
  // is out of range, or no directed path connects them. Pure over the built
  // adjacency; safe on the main thread.
  std::vector<uint32_t> nodes_on_paths_between(uint32_t a, uint32_t b,
                                               uint32_t cap) const;

  // #14 (critical path): the node chain, source→sink, that maximizes the summed
  // per-node `weight`. `weight` is indexed by node (size must be node_count();
  // shorter/empty => treated as 0 for missing entries). Computed by DP over a
  // Kahn topological order of the CSR forward edges (a DAG after the layout's
  // cycle handling; any back-edge that would form a cycle is simply not relaxed,
  // so the pass always terminates). Returns the chain in source→sink order
  // (ascending topo), or empty if the graph is empty. Deterministic: ties broken
  // by smallest node index. Pure over the built adjacency; O(V+E); never throws.
  std::vector<uint32_t> longest_cost_path(const std::vector<uint64_t>& weight) const;

 private:
  uint32_t node_count_ = 0;
  std::vector<uint32_t> succ_off_, succ_val_;
  std::vector<uint32_t> pred_off_, pred_val_;
};

}  // namespace netvis
