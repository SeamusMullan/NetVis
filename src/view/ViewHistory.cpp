// view/ViewHistory.cpp — the bounded undo/redo ring + capture/apply (#106).
//
// The RULES are specified in ViewHistory.h; this file is their only
// implementation. Three orderings below are load-bearing and trivially broken by
// "tidying" the code, so they are called out here rather than buried:
//
//   1. push() must test the IDENTICAL-snapshot drop BEFORE it truncates the redo
//      tail. record_view_history() runs every frame, so the first frame after an
//      undo pushes a snapshot equal to the entry the cursor just landed on; a
//      truncate-first ordering would eat the redo tail there and make redo
//      unreachable after a single undo — the failure App.h's `suppress_history`
//      comment already warns about.
//   2. undo()/redo() must close any open camera run. A run overwrites
//      entries_[cursor_] IN PLACE, so leaving it open across a cursor move would
//      let the next pan clobber the very entry the user stepped onto.
//   3. apply_view() must restore the collapse configuration BEFORE the
//      selection. Selection is stored as an IR node index and resolved through
//      the display list, and the display list is what the collapse restore
//      rebuilds.
#include "view/ViewHistory.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h references
// without including it; both must precede view/App.h for it to compile.
#include "engine/CollapseTree.h"
#include "engine/LayoutEngine.h"
#include "engine/ModelSession.h"
#include "view/App.h"
#include "view/GraphNav.h"
#include "view/PanelHelpers.h"


namespace netvis {

// ---------------------------------------------------------------------------
ViewSnapshot capture_view(const ViewState& vs, const ModelSession& session) {
  ViewSnapshot s;
  s.owner_generation = session.generation();
  s.owner_graph = session.current_graph();

  s.pan_x = vs.cam.pan.x;
  s.pan_y = vs.cam.pan.y;
  s.zoom = vs.cam.zoom;

  // Selection: DISPLAY index -> STABLE IR node index. This is the forward
  // direction of panel_detail::display_index_for_node, and it must be a real
  // translation rather than a copy — the display list is rebuilt by every
  // collapse/expand/filter/graph change, so a stored display id denotes a
  // different node after the undo that restores it (the v0.8.1 path-endpoint
  // lesson recorded in GraphNav.h #15).
  //
  // A GROUP display node stands for a whole run of members; take its FIRST
  // member as the representative, exactly as DiffPanel::diff_tint_for_display
  // resolves a group to an A-side node. The round trip is exact: a group node
  // only exists while the group is COLLAPSED (CollapseTree::rebuild_display
  // emits every member as a leaf when expanded), and display_index_for_node maps
  // any member of a collapsed group back to that group.
  const CollapseTree& ct = session.collapse();
  const auto& disp = ct.display_nodes();
  if (vs.selected_display >= 0 &&
      static_cast<size_t>(vs.selected_display) < disp.size()) {
    const DisplayNode& dn = disp[static_cast<size_t>(vs.selected_display)];
    if (dn.is_group) {
      const auto& groups = ct.groups();
      if (dn.group_index < groups.size() &&
          !groups[dn.group_index].member_nodes.empty())
        s.selected_ir_node =
            static_cast<int32_t>(groups[dn.group_index].member_nodes.front());
    } else if (dn.ir_node != UINT32_MAX) {
      s.selected_ir_node = static_cast<int32_t>(dn.ir_node);
    }
  }
  // selected_value is ALREADY an IR value index (App.h: "an edge/value selected
  // via properties jump"), so unlike the node it needs no translation.
  s.selected_value = vs.selected_value;

  s.expanded = ct.expanded_state();

  // Navigation INTENT only — never nav's derived dim/hidden masks or its `adj`,
  // which are rebuilt by ensure_nav() from exactly the fields copied here.
  if (vs.nav) {
    const GraphNavState& nav = *vs.nav;
    s.nav_mode = static_cast<uint8_t>(nav.mode);
    s.nav_hops = nav.hops;
    s.follow_preds = nav.follow_preds;
    s.follow_succs = nav.follow_succs;
    s.category_mask = nav.category_mask;
    s.filter_active = nav.filter_active;
    s.path_a = nav.path_a;
    s.path_b = nav.path_b;
    s.pinned = nav.pinned;
  }
  // No else branch: ViewSnapshot's defaults for the nav block are byte-for-byte
  // GraphNavState's own defaults, so a tab that has never opened the navigation
  // UI captures "no overlay" without this const capture having to force a
  // GraphNavState into existence.

  s.search_query = vs.search_query;
  s.attr_filter = vs.attr_filter;
  s.table_filter = vs.table_filter;
  s.hide_const_edges = vs.hide_const_edges;
  s.show_layer_bands = vs.show_layer_bands;
  s.show_critical_path = vs.show_critical_path;
  s.cost_heatmap = vs.cost_heatmap;
  s.show_search_results = vs.show_search_results;
  s.diff_panel_open = vs.diff_panel_open;
  s.edge_routing = vs.edge_routing;
  return s;
}

// ---------------------------------------------------------------------------
// apply_view
// ---------------------------------------------------------------------------
bool apply_view(const ViewSnapshot& s, ViewState& vs, ModelSession& session) {
  // REFUSE outright rather than apply the model-agnostic half. IR node indices,
  // value indices and group indices are meaningful only within one
  // (generation, graph): a reopen reassigns them and a subgraph dive replaces
  // the graph outright, so a mismatch means every index below denotes a
  // different thing than it did at capture. A half-applied restore would leave
  // the user in a state they never chose and cannot name.
  //
  // (generation, graph) is a sufficient key HERE, where the same pair keyed on
  // its own would not be elsewhere in this codebase: the history is per-tab and
  // is handed its OWN tab's session, so the cross-tab collision that a shared
  // static cache suffers cannot arise. It still relies on the caller clearing
  // the history when the model or graph changes (ViewHistory::clear).
  if (s.owner_generation != session.generation()) return false;
  if (s.owner_graph != session.current_graph()) return false;

  CollapseTree& ct = session.collapse();
  if (s.expanded.size() != ct.groups().size()) return false;

  // --- Collapse configuration (ordering note 3: this comes first) ------------
  // set_expanded_state() is a CollapseTree operation: it rebuilds the display
  // list but does NOT ask the session to re-lay-out, and only ModelSession's
  // toggle_group/collapse_all/expand_all reach the private request_layout().
  // Routing every differing group through session.toggle_group() instead would
  // submit one full LayoutJob per group — undoing a Collapse-all over 32 decoder
  // blocks would queue 32 layouts of the same view.
  //
  // So: stage every differing bit BUT ONE in a single set_expanded_state()
  // rebuild, then flip the last one through ModelSession::toggle_group, which
  // rebuilds the display once more and requests exactly ONE re-layout.
  //
  // The current state is COPIED, not aliased: expanded_state() returns a
  // reference to the member that set_expanded_state() is about to overwrite.
  const std::vector<bool> cur_expanded = ct.expanded_state();
  uint32_t last_diff = UINT32_MAX;
  for (uint32_t i = 0; i < s.expanded.size(); ++i)
    if (s.expanded[i] != cur_expanded[i]) last_diff = i;
  if (last_diff != UINT32_MAX) {
    std::vector<bool> staged = s.expanded;
    staged[last_diff] = cur_expanded[last_diff];  // left for the toggle below
    // A `false` here means "already equal", not a failure — the size was checked
    // above, which is the only rejection set_expanded_state has.
    ct.set_expanded_state(staged);
    session.toggle_group(last_diff);
  }

  // --- Selection -------------------------------------------------------------
  // Resolved against the display list the collapse restore just rebuilt. -1 when
  // the node is not currently displayable: that cannot happen for a snapshot
  // restored against its own collapse configuration, but display_index_for_node
  // is the bounds check on an IR index that came from a value type, so the
  // result is honoured rather than assumed.
  vs.selected_display = -1;
  if (s.selected_ir_node >= 0)
    vs.selected_display = panel_detail::display_index_for_node(
        ct, static_cast<uint32_t>(s.selected_ir_node));
  vs.selected_value = s.selected_value;
  // Hover is a property of where the mouse is, not of history, and it is a
  // DISPLAY index into a list that may have just been rebuilt underneath it.
  // Clear it; the canvas recomputes it from the cursor next frame.
  vs.hovered_display = -1;

  // --- Camera ----------------------------------------------------------------
  vs.cam.pan = ImVec2(s.pan_x, s.pan_y);
  vs.cam.zoom = s.zoom;
  // Cancel any fly-to in flight. The animation interpolants are deliberately not
  // snapshotted, but they WRITE the camera every frame while running, so leaving
  // one active would overwrite the pose restored above on the very next frame.
  vs.animating = false;

  // --- Navigation intent -----------------------------------------------------
  if (!vs.nav) vs.nav = std::make_unique<GraphNavState>();
  GraphNavState& nav = *vs.nav;
  nav.mode = static_cast<NavMode>(s.nav_mode);
  nav.hops = s.nav_hops;
  nav.follow_preds = s.follow_preds;
  nav.follow_succs = s.follow_succs;
  nav.category_mask = s.category_mask;
  nav.filter_active = s.filter_active;
  nav.path_a = s.path_a;
  nav.path_b = s.path_b;
  nav.pinned = s.pinned;
  // The nav collections this snapshot does NOT carry (focus history, bookmarks)
  // are still IR indices, and they belong to whatever graph nav last claimed. If
  // that is not the live graph — the user dove into a subgraph and came back
  // within a frame, before ensure_nav() ran — the claim below would suppress
  // ensure_nav's wipe and leave those indices to be reinterpreted against a
  // different graph. Wipe exactly what ensure_nav would, and only that:
  // path_a/path_b/pinned are restored from the snapshot immediately above.
  if (nav.owner_generation != session.generation() ||
      nav.owner_graph != session.current_graph()) {
    nav.focus_hist.clear();
    nav.focus_pos = -1;
    nav.focus_last_ir = UINT32_MAX;
    nav.bookmarks.clear();
  }
  // Claim nav ownership of the graph these IR indices belong to — which is the
  // live one, checked at the top. ensure_nav() wipes path_a/path_b/pinned
  // whenever nav's owner_* does not match the current (generation, graph), and a
  // GraphNavState constructed a few lines above carries owner_* == UINT*_MAX, so
  // without this claim the endpoints and pins just restored would be erased on
  // the next frame. Same claim App::load_view_state makes for #56.
  nav.owner_generation = session.generation();
  nav.owner_graph = session.current_graph();

  // --- Filters and panel toggles ---------------------------------------------
  vs.search_query = s.search_query;
  vs.attr_filter = s.attr_filter;
  // attr_filter_key is deliberately NOT touched. It is a cache key, so it is not
  // snapshotted, and it must not be forced either: the properties panel clears
  // attr_filter whenever the key stops matching the node it is drawing, which is
  // precisely the rule "a filter belongs to the node it was typed against". An
  // undo that also moves the selection therefore drops the filter, and an undo
  // within one node keeps it — both correct, with no key surgery here.
  vs.table_filter = s.table_filter;
  vs.hide_const_edges = s.hide_const_edges;
  vs.show_layer_bands = s.show_layer_bands;
  vs.show_critical_path = s.show_critical_path;
  vs.cost_heatmap = s.cost_heatmap;
  vs.show_search_results = s.show_search_results;
  vs.diff_panel_open = s.diff_panel_open;
  vs.edge_routing = s.edge_routing;
  return true;
}

}  // namespace netvis
