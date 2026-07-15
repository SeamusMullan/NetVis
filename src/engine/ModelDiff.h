// engine/ModelDiff.h — structural diff between two models (v0.2.0 model diff).
//
// DECISION: compare a primary model+graph against a comparison model+graph and
// classify each node Added / Removed / Changed / Same. Matching reuses the
// CollapseTree structural fingerprint (op_type + attrs + arity) plus node-name
// equality, compared by string CONTENT (model.str(id)) because the two models
// have INDEPENDENT StringArenas — a raw StringId is meaningless across models.
//
// SAFETY: diff reads only op_type / name / attributes / arity / topology
// (producer indices on ValueInfo). It NEVER reads ValueInfo.shape/dtype, so it is
// safe to compute while background shape inference mutates shapes on either model.
#pragma once

#include <cstdint>
#include <vector>

#include "ir/IR.h"

namespace netvis {

enum class DiffStatus : uint8_t {
  Same,     // matched, identical fingerprint
  Added,    // present in B (comparison) only
  Removed,  // present in A (primary) only
  Changed,  // matched by name/position but fingerprint differs
};

// Per-node diff. a_status is parallel to graph A's node list; b_status to B's.
// a_to_b[i] = the matched B node index for A node i, or -1. b_to_a symmetric.
struct ModelDiffResult {
  std::vector<DiffStatus> a_status;
  std::vector<DiffStatus> b_status;
  std::vector<int32_t> a_to_b;
  std::vector<int32_t> b_to_a;
  uint32_t same = 0, added = 0, removed = 0, changed = 0;
  bool valid = false;  // false if either graph index was out of range
};

// Diff graph_a of model_a (primary) against graph_b of model_b (comparison).
// Deterministic. See header notes for what is (not) read.
ModelDiffResult diff_models(const ir::Model& model_a, uint32_t graph_a,
                            const ir::Model& model_b, uint32_t graph_b);

}  // namespace netvis
