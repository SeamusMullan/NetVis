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
#include <vector>

#include "engine/GraphAdjacency.h"
#include "engine/OpCategory.h"

namespace netvis {

class App;  // forward; defined in view/App.h

enum class NavMode : uint8_t {
  None,       // no highlight/focus overlay
  Highlight,  // dim non-connected nodes, keep all visible
  Focus,      // cull nodes outside the selected neighborhood
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
  // follow_preds<<1 | follow_succs at last rebuild — both drive the neighborhood
  // mask, so a toggle must invalidate the cache (else masks stay stale).
  uint8_t key_follow = 0;
  bool key_valid = false;

  bool any_overlay() const {
    return mode != NavMode::None || filter_active;
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

}  // namespace netvis
