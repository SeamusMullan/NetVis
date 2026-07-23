// view/GraphNav.cpp — graph-navigation view logic (v0.2.0 graph navigation).
//
// Implements ensure_nav() (adjacency + display-space mask rebuild keyed on the
// cache key from GraphNav.h), the nav control widgets, and the jump-to-node /
// jump-to-value helpers. All main-thread only; reads only published, immutable
// engine state (adjacency is built synchronously here from the current graph).
#define IMGUI_DEFINE_MATH_OPERATORS
#include "view/GraphNav.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "imgui.h"

#include "engine/GraphAdjacency.h"
#include "engine/LayoutEngine.h"
#include "engine/OpCategory.h"
#include "engine/plugin/Registry.h"
#include "view/App.h"
#include "view/PanelHelpers.h"

namespace netvis {

namespace {

// Bound any transitive reachability / consumer scan so a pathological graph
// can never blow the frame budget on the main thread.
constexpr uint32_t kReachCap = 250000;

// Op-category of a display node (leaf op, or a group's representative op).
OpCategory category_for_display(ModelSession& s, uint32_t display_id) {
  const auto& disp = s.collapse().display_nodes();
  if (display_id >= disp.size()) return OpCategory::Other;
  const DisplayNode& dn = disp[display_id];
  const ir::Model* m = s.model();
  if (m == nullptr) return OpCategory::Other;
  uint32_t gi = s.current_graph();
  if (gi >= m->graphs.size()) return OpCategory::Other;
  const auto& nodes = m->graphs[gi].nodes;
  if (dn.is_group) {
    const auto& groups = s.collapse().groups();
    if (dn.group_index < groups.size() &&
        !groups[dn.group_index].representative_nodes.empty()) {
      uint32_t ni = groups[dn.group_index].representative_nodes.front();
      if (ni < nodes.size())
        return plugin::resolve_category(*m, m->graphs[gi], nodes[ni]);
    }
    return OpCategory::Other;
  }
  if (dn.ir_node < nodes.size())
    return plugin::resolve_category(*m, m->graphs[gi], nodes[dn.ir_node]);
  return OpCategory::Other;
}

// Collect the IR node indices a display node stands for (a single leaf, or all
// member nodes of a collapsed group).
void ir_nodes_for_display(ModelSession& s, int32_t display_id,
                          std::vector<uint32_t>& out) {
  out.clear();
  const auto& disp = s.collapse().display_nodes();
  if (display_id < 0 || static_cast<size_t>(display_id) >= disp.size()) return;
  const DisplayNode& dn = disp[static_cast<size_t>(display_id)];
  if (dn.is_group) {
    const auto& groups = s.collapse().groups();
    if (dn.group_index < groups.size())
      out = groups[dn.group_index].member_nodes;
  } else {
    out.push_back(dn.ir_node);
  }
}

}  // namespace

void ensure_nav(App& app) {
  ViewState& vs = app.view();
  ModelSession& s = app.session();

  if (!vs.nav) vs.nav = std::make_unique<GraphNavState>();
  GraphNavState& nav = *vs.nav;
  nav.filter_active = nav.category_mask != 0xFFFFFFFFu;

  const ir::Model* model = s.model();
  const uint64_t generation = s.generation();
  const uint32_t graph = s.current_graph();
  const uint64_t collapse_hash = s.collapse().collapse_hash();
  const int32_t selection = vs.selected_display;
  const auto& disp = s.collapse().display_nodes();

  // (1) Rebuild adjacency on (generation, current_graph) change. Cheap O(V+E),
  // synchronous, no worker — so no ir::Model lifetime race.
  if (!nav.adj || nav.key_generation != generation || nav.key_graph != graph) {
    nav.adj = std::make_shared<GraphAdjacency>();
    if (model != nullptr) nav.adj->build(*model, graph);
  }

  // (2) Decide whether the mask cache is still valid.
  const bool overlay = nav.any_overlay();
  const uint8_t follow = static_cast<uint8_t>((nav.follow_preds ? 2 : 0) |
                                              (nav.follow_succs ? 1 : 0));
  const bool key_match =
      nav.key_valid && nav.key_generation == generation &&
      nav.key_graph == graph && nav.key_collapse == collapse_hash &&
      nav.key_selection == selection && nav.key_mode == nav.mode &&
      nav.key_hops == nav.hops && nav.key_category == nav.category_mask &&
      nav.key_follow == follow;
  if (key_match) return;  // nothing changed — keep masks.

  // Record the key we are rebuilding for.
  nav.key_generation = generation;
  nav.key_graph = graph;
  nav.key_collapse = collapse_hash;
  nav.key_selection = selection;
  nav.key_mode = nav.mode;
  nav.key_hops = nav.hops;
  nav.key_category = nav.category_mask;
  nav.key_follow = follow;
  nav.key_valid = true;

  const size_t n = disp.size();
  nav.dim.assign(n, 0);
  nav.hidden.assign(n, 0);

  // No overlay, or nothing selected in a highlight/focus mode with no filter:
  // clear masks and bail (all nodes fully visible).
  const bool neigh_active =
      (nav.mode == NavMode::Highlight || nav.mode == NavMode::Focus) &&
      selection >= 0 && static_cast<size_t>(selection) < n;
  if (!overlay || (!neigh_active && !nav.filter_active)) {
    return;  // dim/hidden already all-zero.
  }

  // (3) Neighborhood mask (Highlight dims / Focus culls nodes outside it).
  if (neigh_active && nav.adj) {
    const uint32_t node_count = nav.adj->node_count();
    std::vector<uint8_t> in_neigh(node_count, 0);
    std::vector<uint32_t> starts;
    ir_nodes_for_display(s, selection, starts);
    for (uint32_t start : starts) {
      if (start < node_count) in_neigh[start] = 1;
      if (nav.follow_preds) {
        for (uint32_t p : nav.adj->reachable_pred(start, nav.hops, kReachCap))
          if (p < node_count) in_neigh[p] = 1;
      }
      if (nav.follow_succs) {
        for (uint32_t su : nav.adj->reachable_succ(start, nav.hops, kReachCap))
          if (su < node_count) in_neigh[su] = 1;
      }
    }
    // Translate to display space: a display node is IN the neighborhood if any
    // IR node it stands for is in_neigh.
    std::vector<uint32_t> members;
    for (size_t i = 0; i < n; ++i) {
      ir_nodes_for_display(s, static_cast<int32_t>(i), members);
      bool inside = false;
      for (uint32_t nd : members) {
        if (nd < node_count && in_neigh[nd]) {
          inside = true;
          break;
        }
      }
      if (!inside) {
        if (nav.mode == NavMode::Focus)
          nav.hidden[i] = 1;
        else
          nav.dim[i] = 1;
      }
    }
  }

  // (4) Category filter: hide display nodes whose category bit is cleared.
  if (nav.filter_active) {
    for (size_t i = 0; i < n; ++i) {
      OpCategory c = category_for_display(s, static_cast<uint32_t>(i));
      uint32_t bit = 1u << static_cast<uint32_t>(c);
      if ((nav.category_mask & bit) == 0u) {
        nav.hidden[i] = 1;  // filtered-out categories are culled outright.
        nav.dim[i] = 0;
      }
    }
  }
}

void draw_nav_controls(App& app) {
  ViewState& vs = app.view();
  if (!vs.nav) vs.nav = std::make_unique<GraphNavState>();
  GraphNavState& nav = *vs.nav;

  // Mode radio.
  int mode = static_cast<int>(nav.mode);
  if (ImGui::RadioButton("Nav: off", &mode, 0)) nav.mode = NavMode::None;
  if (ImGui::RadioButton("Highlight neighborhood", &mode, 1))
    nav.mode = NavMode::Highlight;
  if (ImGui::RadioButton("Focus neighborhood", &mode, 2))
    nav.mode = NavMode::Focus;

  ImGui::Separator();

  // Hop radius: 0 = transitive (UINT32_MAX), else the slider value.
  bool transitive = nav.hops == UINT32_MAX;
  if (ImGui::Checkbox("All hops (transitive)", &transitive))
    nav.hops = transitive ? UINT32_MAX : 1u;
  if (!transitive) {
    int hops = static_cast<int>(std::min<uint32_t>(nav.hops, 32u));
    if (ImGui::SliderInt("Hops", &hops, 1, 32))
      nav.hops = static_cast<uint32_t>(std::max(1, hops));
  }

  bool preds = nav.follow_preds;
  bool succs = nav.follow_succs;
  if (ImGui::Checkbox("Follow inputs (fan-in)", &preds)) nav.follow_preds = preds;
  ImGui::SameLine();
  if (ImGui::Checkbox("Follow outputs (fan-out)", &succs))
    nav.follow_succs = succs;

  ImGui::Separator();

  // Category visibility filter.
  if (ImGui::BeginMenu("Category filter")) {
    for (int c = 0; c <= static_cast<int>(OpCategory::Other); ++c) {
      uint32_t bit = 1u << static_cast<uint32_t>(c);
      bool visible = (nav.category_mask & bit) != 0u;
      if (ImGui::MenuItem(category_name(static_cast<OpCategory>(c)), nullptr,
                          visible)) {
        nav.category_mask ^= bit;
      }
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Show all categories")) nav.category_mask = 0xFFFFFFFFu;
    ImGui::EndMenu();
  }
}

// --- Jump-to helpers ---------------------------------------------------------

void nav_jump_to_ir_node(App& app, uint32_t ir_node) {
  ModelSession& s = app.session();
  int32_t disp = panel_detail::display_index_for_node(s.collapse(), ir_node);
  if (disp < 0) return;
  app.view().selected_display = disp;
  panel_detail::BoxCenter c =
      panel_detail::box_center_for_display(s.layout(), disp);
  ImGuiViewport* vp = ImGui::GetMainViewport();
  animate_camera_to(app.view(), ImVec2(c.x, c.y), vp->WorkSize,
                    app.view().cam.zoom);
}

void nav_jump_to_value_producer(App& app, uint32_t value_index) {
  ModelSession& s = app.session();
  const ir::Model* m = s.model();
  if (m == nullptr) return;
  uint32_t gi = s.current_graph();
  if (gi >= m->graphs.size()) return;
  const ir::Graph& g = m->graphs[gi];
  if (value_index >= g.values.size()) return;
  int32_t producer = g.values[value_index].producer;
  if (producer < 0) return;  // graph input / initializer: no node to fly to.
  nav_jump_to_ir_node(app, static_cast<uint32_t>(producer));
}

void nav_jump_to_value_consumers(App& app, uint32_t value_index) {
  ModelSession& s = app.session();
  const ir::Model* m = s.model();
  if (m == nullptr) return;
  uint32_t gi = s.current_graph();
  if (gi >= m->graphs.size()) return;
  const ir::Graph& g = m->graphs[gi];
  if (value_index >= g.values.size()) return;

  // Find every node consuming value_index (bounded scan of node input slots),
  // map each to a display box, and fly to the centroid of those boxes.
  const LayoutResult* layout = s.layout();
  float sx = 0.0f, sy = 0.0f;
  uint32_t count = 0;
  uint32_t scanned = 0;
  for (uint32_t ni = 0; ni < g.nodes.size() && scanned < kReachCap; ++ni) {
    const ir::Node& node = g.nodes[ni];
    bool consumes = false;
    for (uint32_t slot = 0; slot < node.inputs.count && scanned < kReachCap;
         ++slot, ++scanned) {
      if (panel_detail::resolve_edge_value(g, node.inputs, slot) == value_index) {
        consumes = true;
        break;
      }
    }
    if (!consumes) continue;
    int32_t disp = panel_detail::display_index_for_node(s.collapse(), ni);
    if (disp < 0) continue;
    panel_detail::BoxCenter c = panel_detail::box_center_for_display(layout, disp);
    sx += c.x;
    sy += c.y;
    ++count;
  }
  if (count == 0) return;
  ImVec2 centroid(sx / static_cast<float>(count), sy / static_cast<float>(count));
  ImGuiViewport* vp = ImGui::GetMainViewport();
  animate_camera_to(app.view(), centroid, vp->WorkSize, app.view().cam.zoom);
}

}  // namespace netvis
