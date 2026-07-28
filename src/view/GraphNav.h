// view/GraphNav.h — graph-navigation view state + display-space masks (v0.2.0).
//
// Module-private to view/. Holds the navigation UI state (highlight, focus,
// op-category filter) and the derived per-display-node masks the canvas reads to
// dim/cull boxes and edges. The heavy lifting (adjacency BFS) is done by the
// engine's GraphAdjacency; this layer only translates IR-node reachability into
// display-node masks and owns the transient UI toggles.
//
// STATE INVALIDATION: masks are indexed by DISPLAY-node id, which is rebuilt by
// CollapseTree on toggle_group / push_graph / pop_graph. So the mask cache key
// includes (generation, current_graph, collapse_hash, selection, mode, hops,
// filter). ensure_nav() rebuilds adjacency + ir<->display maps + masks whenever
// that key changes — never trusting a stale mask against a shifted display list.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "imgui.h"  // ImVec2 (draw_pinned_strip signature)

#include "engine/GraphAdjacency.h"
#include "engine/OpCategory.h"

namespace netvis {

class App;  // forward; defined in view/App.h

enum class NavMode : uint8_t {
  None,       // no highlight/focus overlay
  Highlight,  // dim non-connected nodes, keep all visible
  Focus,      // cull nodes outside the selected neighborhood
};

// #16 named bookmark: a saved camera pose + selection the user can jump back to.
// Camera is stored as raw pan/zoom (world space, resolution-independent);
// selection is an IR node index (stable across collapse rebuilds) or -1 for none.
// Held in GraphNavState (per-tab, since each tab owns its ViewState — #62). NOT
// persisted across sessions in v0.8.1: transient nav aids for the current session
// (view_prefs holds only global toggles). A later release can serialize them
// keyed by model path.
struct Bookmark {
  std::string name;
  float pan_x = 0.0f, pan_y = 0.0f;
  float zoom = 1.0f;
  int32_t ir_node = -1;   // selected node at save time, or -1
};

// All navigation UI + derived mask state. Owned by ViewState (App.h) via a
// unique_ptr so App.h needs only a forward declaration + this include.
struct GraphNavState {
  NavMode mode = NavMode::None;
  uint32_t hops = UINT32_MAX;  // neighborhood radius (UINT32_MAX = transitive)
  bool follow_preds = true;    // include fan-in
  bool follow_succs = true;    // include fan-out

  // Op-category filter: bit c set => category c is VISIBLE. Default all visible.
  uint32_t category_mask = 0xFFFFFFFFu;
  bool filter_active = false;  // true when category_mask != all

  // #15 path-between: two picked endpoints as STABLE IR node indices (NOT display
  // ids — a display id is invalidated by any collapse/expand/toggle/graph-dive that
  // rebuilds the display list, which would silently repoint the endpoint at a
  // different node; IR indices survive by construction, matching how `pinned` and
  // `focus_hist` store nodes). When BOTH are set the canvas dims everything except
  // nodes on a directed path connecting them (GraphAdjacency::nodes_on_paths_
  // between). -1 == unset. Folded into the mask cache key. path_active() gates it.
  int32_t path_a = -1;
  int32_t path_b = -1;
  bool path_active() const { return path_a >= 0 && path_b >= 0; }

  // #17 focus history: browser-style back/forward through focused nodes. Stored as
  // IR node indices (STABLE across collapse/display rebuilds, unlike display ids).
  // focus_pos indexes the current position; a new focus truncates any forward tail
  // and appends. Navigation state only — never affects dim/hidden masks. A
  // back/forward step updates focus_last_ir to the node it lands on, so the
  // subsequent selection of that same node is deduped (no re-record).
  std::vector<uint32_t> focus_hist;
  int32_t focus_pos = -1;         // -1 == empty
  uint32_t focus_last_ir = UINT32_MAX;  // last recorded IR node (change detector)

  // #19 sticky pins: IR node indices (stable) kept visible in a screen-edge strip
  // while panning. Overlay only — does not affect dim/hidden masks.
  std::vector<uint32_t> pinned;

  // #16 bookmarks: saved camera+selection poses for THIS tab's model. Panel
  // visibility toggled from the View menu.
  std::vector<Bookmark> bookmarks;
  bool show_bookmarks = false;

  // #13 op-type legend/filter panel visibility (View menu).
  bool show_legend = false;

  // The (generation, graph) the IR-index-based nav collections above (path_a/b,
  // focus_hist, pinned, bookmarks) were populated against. IR indices are stable
  // only WITHIN one graph, so ensure_nav() clears these collections when the graph
  // or generation changes (a subgraph dive via push_graph/pop_graph, or a reopen)
  // — else index 42 from graph 0 would be reinterpreted against graph 1. UINT*_MAX
  // == not yet bound.
  uint64_t owner_generation = UINT64_MAX;
  uint32_t owner_graph = UINT32_MAX;

  // --- Derived, rebuilt by ensure_nav(); indexed by display-node id ----------
  // dim[i]==true => draw node/edge i de-emphasized (Highlight mode / filter).
  // hidden[i]==true => cull node/edge i entirely (Focus mode / filter-hide).
  std::vector<uint8_t> dim;
  std::vector<uint8_t> hidden;

  // Engine adjacency over the CURRENT graph (rebuilt on graph/generation change).
  std::shared_ptr<GraphAdjacency> adj;

  // Cache key of the last successful rebuild (see header note).
  uint64_t key_generation = UINT64_MAX;
  uint32_t key_graph = UINT32_MAX;
  uint64_t key_collapse = UINT64_MAX;
  int32_t key_selection = -1;
  NavMode key_mode = NavMode::None;
  uint32_t key_hops = 0;
  uint32_t key_category = 0;
  int32_t key_path_a = -1;  // #15: path endpoints folded into the mask cache key
  int32_t key_path_b = -1;
  // follow_preds<<1 | follow_succs at last rebuild — both drive the neighborhood
  // mask, so a toggle must invalidate the cache (else masks stay stale).
  uint8_t key_follow = 0;
  bool key_valid = false;

  bool any_overlay() const {
    return mode != NavMode::None || filter_active || path_active();
  }
};

// Rebuild adjacency + masks if the nav cache key changed. Cheap no-op otherwise.
// Call once per frame from App::frame() BEFORE draw_graph_canvas. Main thread.
void ensure_nav(App& app);

// Draw the nav control widgets (mode toggles, hop slider, category filter menu).
// Called from the View menu / a small toolbar in App::frame().
void draw_nav_controls(App& app);

// Camera fly-to helpers for jump-to-producer/consumers (Feature 4). Fly to the
// display node for `ir_node`; if hidden in a collapsed group, flies to the group.
void nav_jump_to_ir_node(App& app, uint32_t ir_node);
// Fly to the centroid of a value's consumers (or its single producer).
void nav_jump_to_value_producer(App& app, uint32_t value_index);
void nav_jump_to_value_consumers(App& app, uint32_t value_index);

// #17 focus history: record `ir_node` as the newest focus (truncates any forward
// tail). Called by the canvas/properties when the user selects a node. No-op if it
// equals the current focus or focus_suppress is set (during back/forward).
void nav_record_focus(App& app, uint32_t ir_node);
// Step back / forward through focus history; flies to the node and selects it.
// No-op at the ends. focus_can_back/forward gate the toolbar buttons + keybinds.
void nav_focus_back(App& app);
void nav_focus_forward(App& app);
// Non-const App& — App::view() has no const overload (it indexes the active tab).
bool nav_focus_can_back(App& app);
bool nav_focus_can_forward(App& app);

// #19 sticky pins: toggle `ir_node` in the pin set (add if absent, remove if
// present). Used by the canvas context menu + a keybind on the selected node.
void nav_toggle_pin(App& app, uint32_t ir_node);
// Draw the pinned-node strip along the canvas's top edge (screen-space overlay).
// Called from draw_graph_canvas inside the canvas child. Clicking a chip flies to
// that node; an "x" removes the pin. No-op when the pin set is empty.
void draw_pinned_strip(App& app, ImVec2 canvas_origin, ImVec2 canvas_size);

// #16 bookmarks: draw the bookmark panel (save current view as a named bookmark;
// list saved bookmarks with jump / delete). A small window toggled from the View
// menu (view().show_bookmarks). Persists via App::save_prefs().
void draw_bookmarks_panel(App& app);

// #13 op-type legend: a color key of the op categories. Clicking a row toggles
// that category's visibility bit in the nav filter (category_mask) so the legend
// doubles as a filter. A small window toggled from the View menu
// (nav->show_legend). No-op when no model is loaded.
void draw_legend_panel(App& app);

}  // namespace netvis
