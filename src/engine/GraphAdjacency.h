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

 private:
  uint32_t node_count_ = 0;
  std::vector<uint32_t> succ_off_, succ_val_;
  std::vector<uint32_t> pred_off_, pred_val_;
};

}  // namespace netvis
