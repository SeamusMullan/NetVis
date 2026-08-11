// engine/ViewSnapshot.h — bounded undo/redo over view state (#106).
//
// SCOPE DECISION (v0.9.4, user's call): a snapshot captures EVERYTHING the user
// can change about a view — selection, navigation, collapse configuration,
// filters, panel toggles AND the camera.
//
// Including the camera has a known hazard: a pan or a zoom is a continuous
// gesture, and naively snapshotting per change would push a hundred entries for
// one drag and make Ctrl+Z useless. So camera-only changes are COALESCED — a run
// of consecutive camera-only edits collapses into the single entry that started
// it, and a non-camera change closes the run. One drag is therefore one undo
// step, which is what a user means by "undo that pan".
//
// WHAT IS DELIBERATELY NOT CAPTURED: derived caches. `cost`, the heatmap range,
// the nav dim/hidden masks, `adj`, every `*_key_*` field, and the animation
// interpolants are all rebuilt from the state above. Snapshotting them would
// make history entries enormous, and RESTORING them would resurrect a cache for
// state that no longer exists — the precise bug class this codebase has shipped
// three times (a stale cost report, a stale diff tint, a stale nav mask). A
// restore therefore sets intent and lets the caches rebuild themselves.
//
// Selection is stored as a STABLE IR NODE INDEX, never a display index. The
// display list is rebuilt by every collapse/expand/filter/graph change, so a
// display id means something different after an undo than it did before it —
// this is the lesson v0.8.1 learned when path endpoints stored as display ids
// pointed at the wrong node after Collapse-all.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace netvis {


// One restorable point. Deliberately a value type with no pointers: an entry may
// outlive the model it was taken against (a tab reload), and `owner_generation`
// + `owner_graph` let a restore refuse rather than apply nonsense.
struct ViewSnapshot {
  // Identity of the state this was captured against. A snapshot must never be
  // applied across a model reload or a subgraph dive: IR node indices and group
  // indices are only stable within one (generation, graph).
  uint64_t owner_generation = UINT64_MAX;
  uint32_t owner_graph = UINT32_MAX;

  // Camera. Plain floats rather than ImVec2 DELIBERATELY: this header lives in
  // netvis_core so netvis_tests can link and exercise the real ring buffer. An
  // ImGui type here would confine it to the GUI target, and the test would have
  // to hand-copy the algorithm to test it — which is not a test of this code.
  float pan_x = 0.0f;
  float pan_y = 0.0f;
  float zoom = 1.0f;

  // Selection, as STABLE IR indices (see header note). -1 for none.
  int32_t selected_ir_node = -1;
  int32_t selected_value = -1;

  // Collapse configuration, one bool per group, matching CollapseTree::groups().
  // Empty means "no collapse state captured" (a graph-less model).
  std::vector<bool> expanded;

  // Navigation intent (GraphNavState's user-set fields only — never its masks).
  uint8_t nav_mode = 0;
  uint32_t nav_hops = UINT32_MAX;
  bool follow_preds = true;
  bool follow_succs = true;
  uint32_t category_mask = 0xFFFFFFFFu;
  bool filter_active = false;
  int32_t path_a = -1;
  int32_t path_b = -1;
  std::vector<uint32_t> pinned;

  // Filters and panel toggles the user drives.
  std::string search_query;
  std::string attr_filter;
  std::string table_filter;
  bool hide_const_edges = false;
  bool show_layer_bands = false;
  bool show_critical_path = false;
  bool cost_heatmap = false;
  bool show_search_results = false;
  bool diff_panel_open = false;
  int edge_routing = 0;

  // True when this entry differs from `o` ONLY in camera fields. Drives the
  // coalescing rule above.
  bool camera_only_diff(const ViewSnapshot& o) const;
};

// A bounded ring of snapshots with a cursor. Bounded because an unbounded
// history of a 100k-node model's collapse bitsets is a memory leak with a nice
// name; kMaxEntries is small enough to stay trivial and large enough to cover
// the exploration a user actually backtracks through.
inline constexpr size_t kMaxHistoryEntries = 64;

class ViewHistory {
 public:
  // Record the current state. No-op when it is identical to the current entry —
  // every frame would otherwise push a duplicate. Applies the camera-coalescing
  // rule. Pushing after an undo TRUNCATES the redo tail, which is the standard
  // and least surprising behaviour.
  void push(const ViewSnapshot& s);

  bool can_undo() const;
  bool can_redo() const;

  // Step the cursor and return the entry to apply, or nullptr at the end.
  // The returned pointer is invalidated by the next push().
  const ViewSnapshot* undo();
  const ViewSnapshot* redo();

  // Drop everything. Called when the model or graph changes: entries from a
  // previous (generation, graph) can never be applied, so keeping them would
  // only offer the user undo steps that silently do nothing.
  void clear();

  size_t size() const { return entries_.size(); }

  // Read-only inspection of the ring. The UI wants these to show how deep the
  // undo stack is, and tests want them to assert on the ring's shape rather than
  // inferring it by walking undo() — an inference that would pass just as
  // happily against a subtly wrong implementation.
  //
  // `at` is bounds-checked to a null return rather than UB: an index here comes
  // from a caller's own bookkeeping, and this class already refuses every other
  // out-of-range request instead of trusting one.
  size_t cursor() const { return cursor_; }
  const ViewSnapshot* at(size_t i) const {
    return i < entries_.size() ? &entries_[i] : nullptr;
  }

 private:
  std::vector<ViewSnapshot> entries_;
  size_t cursor_ = 0;      // index of the CURRENT state within entries_
  bool coalescing_ = false;  // last push was camera-only
};

}  // namespace netvis
