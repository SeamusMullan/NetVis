// engine/CollapseTree.h — repeated-block detection + collapse state.
//
// DECISION (spec §7.1): modern models are dominated by repeated blocks (e.g. 32
// identical decoder layers). We detect them BEFORE layout so the default view
// lays out a handful of super-nodes instead of 100k leaves — this is what makes
// the default layout sub-second. Detection combines a structural fingerprint
// per node with ONNX name-prefix grouping (e.g. "/model/layers.0/...").
//
// The collapse view presents a flat "display node" list: each display node is
// either a single leaf IR node or a collapsed group labeled "Block ×N". The
// layout and canvas operate purely on display nodes.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/Hash.h"
#include "ir/IR.h"

namespace netvis {

// A detected repeated group: a set of node-index members that share a
// structural fingerprint / name-prefix pattern, with `instances` repeats.
struct CollapseGroup {
  std::string label;                 // e.g. "DecoderBlock"
  uint32_t instances = 0;            // N in "×N"
  std::vector<uint32_t> member_nodes;// all IR node indices covered by the group
  // Nodes of a single representative instance, for incremental expand layout.
  std::vector<uint32_t> representative_nodes;
  uint64_t fingerprint = 0;
};

// One entry in the flattened display list the canvas/layout consume.
struct DisplayNode {
  bool is_group = false;
  uint32_t group_index = UINT32_MAX; // into CollapseTree::groups if is_group
  uint32_t ir_node = UINT32_MAX;     // into Graph::nodes if leaf
  bool expanded = false;             // groups only
};

class CollapseTree {
 public:
  // Build groups for one graph. Deterministic: same graph -> same grouping.
  void build(const ir::Model& model, uint32_t graph_index);

  const std::vector<CollapseGroup>& groups() const { return groups_; }

  // The current display-node list given the expand/collapse state.
  const std::vector<DisplayNode>& display_nodes() const { return display_; }

  // Toggle a group's expansion and rebuild the display list. Returns true if the
  // display list changed (view should request a re-layout of the region).
  bool toggle_group(uint32_t group_index);

  // Hash of the current collapse state (which groups are expanded) — part of the
  // layout cache key (spec §7.2.7).
  uint64_t collapse_hash() const;

  // Structural hash of the graph (independent of collapse state) — the other
  // half of the cache key.
  uint64_t structure_hash() const { return structure_hash_; }

 private:
  std::vector<CollapseGroup> groups_;
  std::vector<DisplayNode> display_;
  std::vector<bool> expanded_;       // per group
  uint64_t structure_hash_ = 0;

  void rebuild_display(const ir::Graph& g);
  // Non-owning: valid only during build()/rebuild_display() calls.
  const ir::Model* model_ = nullptr;
  uint32_t graph_index_ = 0;
};

// Structural fingerprint of a single node (spec §7.1.1): hash of op_type,
// attribute names+values (excluding node/doc names), and input/output arity.
uint64_t node_fingerprint(const ir::Model& model, const ir::Graph& g,
                          const ir::Node& n);

}  // namespace netvis
