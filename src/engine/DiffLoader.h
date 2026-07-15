// engine/DiffLoader.h — loads + diffs a COMPARISON model against the primary.
//
// DECISION (v0.2.0 model diff): the view must never include a parser
// (view -> engine -> parsers, per CONTRACTS.md / DECISIONS.md). So the async
// load of model B and the diff computation live here, in the engine. The view
// (DiffPanel) drives this through load_comparison() and reads results, mirroring
// how it drives ModelSession.
//
// LIFETIME/RACE: the comparison model is held by shared_ptr<const ir::Model> and
// the worker that computes the diff captures that shared_ptr, so the model stays
// alive for the whole job even if a newer load replaces the published one. A
// monotonic token guards completions: only the newest load publishes. The diff
// job reads only fingerprint/topology data (never shape/dtype), so it does not
// race the primary session's in-flight shape inference. Uses its OWN JobSystem
// (passed by App as a SECOND system) so its generation counter cannot
// cross-cancel the primary ModelSession's in-flight jobs.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "engine/ModelDiff.h"
#include "ir/IR.h"

namespace netvis {

class ModelSession;  // primary session (read-only here)

enum class DiffLoadState : uint8_t { Empty, Loading, Ready, Failed };

// Owns the comparison model + the latest diff. Main-thread-only public API,
// exactly like ModelSession. One instance owned by App for the whole run.
class DiffLoader {
 public:
  explicit DiffLoader(JobSystem& jobs);
  ~DiffLoader();

  DiffLoader(const DiffLoader&) = delete;
  DiffLoader& operator=(const DiffLoader&) = delete;

  // Begin loading `path` as the comparison model and diffing it against the
  // primary session's CURRENT graph. Non-blocking. Snapshots the primary graph
  // index at call time; call again after the primary changes graphs to refresh.
  void load_comparison(const ModelSession& primary, const std::string& path);

  // Drain completions (call once per frame from the main thread, after the
  // primary session's update()).
  void update();

  // Clear the comparison (exit diff mode).
  void clear();

  DiffLoadState state() const { return state_; }
  const std::string& error() const { return error_; }
  const std::string& path() const { return path_; }

  const ir::Model* model() const { return model_.get(); }

  // The diff of (primary graph snapshot) vs (comparison graph). Null until Ready.
  const ModelDiffResult* diff() const { return diff_ ? &*diff_ : nullptr; }

  // Which primary graph index the current diff was computed against. The view
  // must only tint when this equals the primary session's current graph AND the
  // node index is in range (guard against push_graph/pop_graph invalidation).
  uint32_t primary_graph() const { return primary_graph_; }

  // The primary session generation the current diff was computed against. The
  // view must require this equals the primary's live generation() before
  // tinting — a graph-index match alone is insufficient because current_graph_
  // resets to 0 on every open, so a reloaded primary would otherwise be painted
  // with the previous model's stale per-node diff status.
  uint64_t primary_generation() const { return primary_generation_; }
  bool active() const { return state_ == DiffLoadState::Ready && diff_; }

 private:
  JobSystem& jobs_;
  MappedFile file_;
  std::shared_ptr<const ir::Model> model_;
  std::unique_ptr<ModelDiffResult> diff_;
  ProgressSink progress_;

  DiffLoadState state_ = DiffLoadState::Empty;
  std::string error_;
  std::string path_;
  uint32_t primary_graph_ = 0;
  uint64_t primary_generation_ = UINT64_MAX;  // primary gen at diff-compute time
  uint64_t token_ = 0;  // newest load wins
};

}  // namespace netvis
