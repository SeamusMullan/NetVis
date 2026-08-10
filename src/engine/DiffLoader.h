// engine/DiffLoader.h — loads + diffs COMPARISON models against the primary.
//
// DECISION (v0.2.0 model diff): the view must never include a parser
// (view -> engine -> parsers, per CONTRACTS.md / DECISIONS.md). So the async
// load of the comparison models and the diff computation live here, in the
// engine. The view (DiffPanel) drives this through add_comparison() and reads
// results, mirroring how it drives ModelSession.
//
// v0.9.1b (#36) — N-WAY. This used to hold exactly one comparison. A quant ladder
// (fp32 -> int8 -> int4) needs several at once, so it now holds up to
// kMaxComparisons of them, each an independent Comparison with its OWN mapping,
// model, diff, state and token. Every comparison is diffed against the SAME
// primary, never against each other: the primary is the fixed baseline and the
// ladder is read as a column per comparison. Cross-comparison diffs would need
// N^2 results and have no obvious baseline, so they are out of scope.
//
// The pre-#36 single-comparison surface is retained and now means "the ACTIVE
// comparison" — the one driving the graph tint. That keeps GraphCanvas and the
// tint path unchanged: there is still exactly one set of node colors on screen.
//
// LIFETIME/RACE (unchanged, now per comparison): the comparison model is held by
// shared_ptr<const ir::Model> and the worker that computes the diff captures that
// shared_ptr, so the model stays alive for the whole job even if a newer load
// replaces the published one. A monotonic token PER COMPARISON SLOT guards
// completions: only the newest load into a slot publishes. The diff job reads
// only fingerprint/topology data (never shape/dtype), so it does not race the
// primary session's in-flight shape inference. Uses its OWN JobSystem (passed by
// App as a SECOND system) so its generation counter cannot cross-cancel the
// primary ModelSession's in-flight jobs.
//
// SLOT STABILITY: a comparison's index is stable for its lifetime except across
// remove_comparison(), which compacts. Callers must not cache an index across a
// removal — re-read it from the panel's own selection state.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "engine/ModelDiff.h"
#include "ir/IR.h"

namespace netvis {

class ModelSession;  // primary session (read-only here)

enum class DiffLoadState : uint8_t { Empty, Loading, Ready, Failed };

// #36: how many comparisons may be loaded at once. Three gives the A/B/C(/D)
// quant ladder the issue asks for while bounding the cost: each comparison holds
// a full mapping + parsed model, and the panel renders one column per slot.
constexpr size_t kMaxComparisons = 3;

// Owns the comparison models + their diffs. Main-thread-only public API,
// exactly like ModelSession. One instance owned by App for the whole run.
class DiffLoader {
 public:
  explicit DiffLoader(JobSystem& jobs);
  ~DiffLoader();

  DiffLoader(const DiffLoader&) = delete;
  DiffLoader& operator=(const DiffLoader&) = delete;

  // --- N-way surface (#36) ---------------------------------------------------

  // Number of comparison slots currently held (loading, ready or failed).
  size_t comparison_count() const { return comps_.size(); }

  // Append `path` as an ADDITIONAL comparison and diff it against the primary's
  // CURRENT graph. Non-blocking. Snapshots the primary graph index + generation +
  // session identity at call time, per slot. No-op (returns false) when already
  // at kMaxComparisons. The new slot becomes active if it is the first one.
  bool add_comparison(const ModelSession& primary, const std::string& path);

  // Replace the comparison in slot `i` (must be < comparison_count()). Bumps that
  // slot's token so any in-flight load into it is dropped.
  void reload_comparison(size_t i, const ModelSession& primary,
                         const std::string& path);

  // Drop slot `i`. Remaining slots COMPACT (indices above `i` shift down), and
  // the active index is adjusted to stay in range. Any in-flight load for the
  // removed slot is invalidated by its token.
  void remove_comparison(size_t i);

  // Which comparison drives the graph tint and the single-comparison accessors
  // below. Always < comparison_count(), or 0 when there are none.
  size_t active_comparison() const { return active_; }
  void set_active_comparison(size_t i);

  // Per-slot accessors. All are bounds-checked: an out-of-range index yields the
  // empty/neutral value (Empty state, nullptr model/diff, empty string), never
  // UB. This matters because the panel holds a selection index across frames.
  DiffLoadState state_of(size_t i) const;
  const std::string& error_of(size_t i) const;
  const std::string& path_of(size_t i) const;
  const ir::Model* model_of(size_t i) const;
  const ModelDiffResult* diff_of(size_t i) const;
  uint32_t primary_graph_of(size_t i) const;
  uint64_t primary_generation_of(size_t i) const;
  const ModelSession* primary_session_of(size_t i) const;
  bool active_of(size_t i) const;

  // The comparison model's MAPPING and directory (#34/#50). Needed because the
  // weight-stat delta reads payload out of model B exactly the way the inspector
  // reads it out of model A, and only DiffLoader owns B's mmap. The returned
  // reference is valid until this slot is reloaded or removed; a worker that
  // outlives that must be guarded by the same token discipline the loader uses
  // internally. Out-of-range yields a default-constructed (invalid) MappedFile.
  const MappedFile& file_of(size_t i) const;
  const std::string& model_dir_of(size_t i) const;

  // An OWNING handle on slot `i`'s model, for a worker that must outlive a
  // possible remove_comparison(). model_of() returns a bare pointer that dies
  // with the slot; a background decode of model B's payload (#50) cannot rely on
  // that, because the user can drop a comparison while its decode is in flight.
  // Taking a copy of this shared_ptr keeps the model alive for the job's life.
  //
  // NOTE the MAPPING is NOT covered by this — MappedFile is move-only and dies
  // with the slot. A worker that needs bytes must open its own mapping of
  // path_of(i) (an mmap is ~1 ms, so this is cheap) rather than borrowing the
  // slot's. Defined inline so it needs no .cpp definition.
  std::shared_ptr<const ir::Model> model_ptr_of(size_t i) const {
    const Comparison* c = slot(i);
    return c ? c->model : nullptr;
  }

  // --- Single-comparison surface == the ACTIVE slot --------------------------
  // Pre-#36 API, unchanged in meaning for the one-comparison case.

  // Clear ALL comparisons and load `path` as the only one (the pre-#36 behavior
  // of this call: newest load wins, replacing whatever was there).
  void load_comparison(const ModelSession& primary, const std::string& path);

  // Drain completions (call once per frame from the main thread, after the
  // primary session's update()). Drains every slot's completions.
  void update();

  // Clear ALL comparisons (exit diff mode).
  void clear();

  // #35: node-matching strategy for the diff. set_match re-runs the diff INLINE
  // against EVERY already-loaded comparison (bounded main-thread work, same as
  // the post-load diff) when the mode changes — no reload. No-op if the mode is
  // unchanged. The strategy is loader-wide: showing a ladder where each rung used
  // a different matching rule would make the columns incomparable.
  DiffMatch match() const { return match_; }
  void set_match(DiffMatch m);

  DiffLoadState state() const { return state_of(active_); }
  const std::string& error() const { return error_of(active_); }
  const std::string& path() const { return path_of(active_); }
  const ir::Model* model() const { return model_of(active_); }

  // The diff of (primary graph snapshot) vs (active comparison graph). Null until
  // that slot is Ready.
  const ModelDiffResult* diff() const { return diff_of(active_); }

  // Which primary graph index the ACTIVE diff was computed against. The view must
  // only tint when this equals the primary session's current graph AND the node
  // index is in range (guard against push_graph/pop_graph invalidation).
  uint32_t primary_graph() const { return primary_graph_of(active_); }

  // The primary session generation the ACTIVE diff was computed against. The view
  // must require this equals the primary's live generation() before tinting — a
  // graph-index match alone is insufficient because current_graph_ resets to 0 on
  // every open, so a reloaded primary would otherwise be painted with the
  // previous model's stale per-node diff status.
  uint64_t primary_generation() const { return primary_generation_of(active_); }
  bool active() const { return active_of(active_); }

  // Identity of the primary ModelSession the ACTIVE diff was loaded against (#62:
  // with multi-model tabs, each tab has its OWN session, and per-session
  // generation counters both start at 0 — so a generation+graph match no longer
  // uniquely identifies the model. The view must additionally require this equals
  // the address of the session it is about to tint, or a diff computed for tab A
  // would paint tab B's unrelated graph). Pointer identity only — NEVER
  // dereferenced by the view; App clears the diff when this session's tab closes
  // so the pointer can't dangle. nullptr when no comparison is loaded.
  const ModelSession* primary_session() const {
    return primary_session_of(active_);
  }

 private:
  // One comparison slot. Non-copyable (MappedFile is move-only), held by
  // unique_ptr so a vector resize can never move a slot a worker's completion
  // lambda is about to write through.
  struct Comparison {
    MappedFile file;
    std::string model_dir;
    std::shared_ptr<const ir::Model> model;
    std::unique_ptr<ModelDiffResult> diff;
    ProgressSink progress;

    DiffLoadState state = DiffLoadState::Empty;
    std::string error;
    std::string path;
    uint32_t primary_graph = 0;
    uint64_t primary_generation = UINT64_MAX;
    const ModelSession* primary_session = nullptr;
    uint64_t token = 0;  // newest load into THIS slot wins
  };

  // Start an async load into an existing slot. Shared by add/reload/load.
  void start_load(Comparison& c, const ModelSession& primary,
                  const std::string& path);
  // Re-run the diff for one loaded slot against the primary it snapshotted.
  // Main-thread, inline, bounded. Used by set_match.
  void rediff(Comparison& c);

  Comparison* slot(size_t i) { return i < comps_.size() ? comps_[i].get() : nullptr; }
  const Comparison* slot(size_t i) const {
    return i < comps_.size() ? comps_[i].get() : nullptr;
  }

  JobSystem& jobs_;
  std::vector<std::unique_ptr<Comparison>> comps_;
  size_t active_ = 0;
  DiffMatch match_ = DiffMatch::NameThenTopology;  // #35, loader-wide
};

}  // namespace netvis
