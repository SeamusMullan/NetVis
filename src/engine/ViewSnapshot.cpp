// engine/ViewSnapshot.cpp — the bounded undo/redo ring (#106).
//
// Lives in netvis_core, not in the view, so netvis_tests links and exercises
// THIS code rather than a hand-copied mirror of it. The capture/apply half needs
// ViewState and stays in view/ViewHistory.cpp.
#include "engine/ViewSnapshot.h"

#include <cstddef>
#include <vector>

namespace netvis {
namespace {

// Bitwise-exact camera equality. Exactness is the point, not a shortcut: the
// question push() asks is "did this frame move the camera at all", and an
// epsilon would silently swallow a slow zoom into a run that never ends.
bool same_camera(const ViewSnapshot& a, const ViewSnapshot& b) {
  return a.pan_x == b.pan_x && a.pan_y == b.pan_y && a.zoom == b.zoom;
}

}  // namespace

// ---------------------------------------------------------------------------
// ViewSnapshot
// ---------------------------------------------------------------------------
bool ViewSnapshot::camera_only_diff(const ViewSnapshot& o) const {
  // Every non-camera field, exhaustively — this is a whitelist by omission, and
  // a field added to ViewSnapshot but forgotten here would be SWALLOWED by the
  // coalescing rule: changing it would look camera-only, so push() would
  // overwrite the open run's entry instead of starting a new undo step, and the
  // user's edit would have no Ctrl+Z of its own.
  //
  // Two identical snapshots also satisfy this predicate ("differ in nothing" is
  // a subset of "differ only in the camera"). That is deliberate and safe
  // because push() tests full identity first; keeping the predicate this way
  // means callers need only one comparison to ask "is this the same step".
  return owner_generation == o.owner_generation &&
         owner_graph == o.owner_graph &&
         selected_ir_node == o.selected_ir_node &&
         selected_value == o.selected_value && expanded == o.expanded &&
         nav_mode == o.nav_mode && nav_hops == o.nav_hops &&
         follow_preds == o.follow_preds && follow_succs == o.follow_succs &&
         category_mask == o.category_mask && filter_active == o.filter_active &&
         path_a == o.path_a && path_b == o.path_b && pinned == o.pinned &&
         search_query == o.search_query && attr_filter == o.attr_filter &&
         table_filter == o.table_filter &&
         hide_const_edges == o.hide_const_edges &&
         show_layer_bands == o.show_layer_bands &&
         show_critical_path == o.show_critical_path &&
         cost_heatmap == o.cost_heatmap &&
         show_search_results == o.show_search_results &&
         diff_panel_open == o.diff_panel_open && edge_routing == o.edge_routing;
}

// ---------------------------------------------------------------------------
// ViewHistory
// ---------------------------------------------------------------------------
void ViewHistory::push(const ViewSnapshot& s) {
  if (entries_.empty()) {
    entries_.push_back(s);
    cursor_ = 0;
    coalescing_ = false;
    return;
  }

  const ViewSnapshot& cur = entries_[cursor_];
  const bool camera_only = s.camera_only_diff(cur);

  // (1) Nothing changed. Dropped BEFORE the truncation in (3) — see ordering
  // note 1 at the top of this file. A no-op frame also leaves `coalescing_`
  // alone: a drag that pauses for a frame with the mouse held still is still one
  // gesture, and resetting the run here would split it into two undo steps.
  if (camera_only && same_camera(s, cur)) return;

  // (2) Continue an OPEN camera run. The run owns exactly one slot — the entry
  // that started it — and that slot tracks the NEWEST camera, so undo returns to
  // the pose from before the gesture and redo returns to where the gesture
  // ended. Overwriting cannot strand a redo tail: `coalescing_` is cleared by
  // undo()/redo() and by every non-camera push, and (3) truncates, so an open
  // run implies cursor_ is already the last entry.
  if (camera_only && coalescing_) {
    entries_[cursor_] = s;
    return;
  }

  // (3) A genuinely new step. Drop the redo tail (standard behaviour: branching
  // off an undone state abandons the branch that was undone), then append.
  entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(cursor_) + 1,
                 entries_.end());
  entries_.push_back(s);
  // A camera-only step OPENS a run: this appended entry is the one the rest of
  // the gesture will collapse into. Anything else CLOSES the run.
  coalescing_ = camera_only;
  cursor_ = entries_.size() - 1;

  // (4) Enforce the bound by dropping the OLDEST entries. Erasing from the front
  // shifts every index down, so cursor_ must move with them — it sits at the
  // back here, but the subtraction is written generally so the arithmetic stays
  // right if this ever runs with the cursor elsewhere. The O(n) shift is fine:
  // n is kMaxHistoryEntries, and it is a move of 64 snapshots, once per push,
  // only once the ring is already full.
  if (entries_.size() > kMaxHistoryEntries) {
    const size_t drop = entries_.size() - kMaxHistoryEntries;
    entries_.erase(entries_.begin(),
                   entries_.begin() + static_cast<std::ptrdiff_t>(drop));
    cursor_ = (cursor_ >= drop) ? cursor_ - drop : 0;
  }
}

bool ViewHistory::can_undo() const { return !entries_.empty() && cursor_ > 0; }

bool ViewHistory::can_redo() const {
  return !entries_.empty() && cursor_ + 1 < entries_.size();
}

const ViewSnapshot* ViewHistory::undo() {
  if (!can_undo()) return nullptr;
  --cursor_;
  coalescing_ = false;  // ordering note 2: a cursor move closes the camera run
  return &entries_[cursor_];
}

const ViewSnapshot* ViewHistory::redo() {
  if (!can_redo()) return nullptr;
  ++cursor_;
  coalescing_ = false;  // ordering note 2
  return &entries_[cursor_];
}

void ViewHistory::clear() {
  entries_.clear();
  cursor_ = 0;
  coalescing_ = false;
}

// ---------------------------------------------------------------------------
// capture_view

}  // namespace netvis
