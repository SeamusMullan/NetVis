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
#include <string>
#include <string_view>
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

  // (1b) The IR-index-based nav collections are valid only within one graph. On a
  // graph/generation change (subgraph dive, reopen) clear them so a stale IR index
  // is never reinterpreted against a different graph (see GraphNavState::owner_*).
  if (nav.owner_generation != generation || nav.owner_graph != graph) {
    nav.owner_generation = generation;
    nav.owner_graph = graph;
    nav.path_a = -1;
    nav.path_b = -1;
    nav.focus_hist.clear();
    nav.focus_pos = -1;
    nav.focus_last_ir = UINT32_MAX;
    nav.pinned.clear();
    nav.bookmarks.clear();
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
      nav.key_follow == follow && nav.key_path_a == nav.path_a &&
      nav.key_path_b == nav.path_b;
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
  nav.key_path_a = nav.path_a;
  nav.key_path_b = nav.path_b;
  nav.key_valid = true;

  const size_t n = disp.size();
  nav.dim.assign(n, 0);
  nav.hidden.assign(n, 0);

  // No overlay, or nothing to compute (no neighborhood, no filter, no path):
  // clear masks and bail (all nodes fully visible).
  const bool neigh_active =
      (nav.mode == NavMode::Highlight || nav.mode == NavMode::Focus) &&
      selection >= 0 && static_cast<size_t>(selection) < n;
  if (!overlay ||
      (!neigh_active && !nav.filter_active && !nav.path_active())) {
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

  // (5) #15 Path-between: dim every display node NOT on a directed path connecting
  // the two picked endpoints. Endpoints are display ids; map each to a representa-
  // tive IR node (a leaf's node, or a group's first member), ask the engine for the
  // node set on any connecting path, then translate back to display space. Uses the
  // SAME dim mask as Highlight (de-emphasize, keep visible) so it composes with a
  // filter's hidden mask.
  if (nav.path_active() && nav.adj) {
    const uint32_t node_count = nav.adj->node_count();
    // Endpoints are STABLE IR node indices (see GraphNav.h #15). Valid only if
    // both index real nodes of the current graph's adjacency.
    const uint32_t a_ir = static_cast<uint32_t>(nav.path_a);
    const uint32_t b_ir = static_cast<uint32_t>(nav.path_b);
    if (a_ir < node_count && b_ir < node_count) {
      std::vector<uint32_t> on_path =
          nav.adj->nodes_on_paths_between(a_ir, b_ir, kReachCap);
      std::vector<uint8_t> on(node_count, 0);
      for (uint32_t nd : on_path)
        if (nd < node_count) on[nd] = 1;
      // A display node is on-path if ANY IR node it stands for is on-path.
      std::vector<uint32_t> members;
      for (size_t i = 0; i < n; ++i) {
        if (nav.hidden[i]) continue;  // already culled by filter — leave hidden.
        ir_nodes_for_display(s, static_cast<int32_t>(i), members);
        bool inside = false;
        for (uint32_t nd : members) {
          if (nd < node_count && on[nd]) { inside = true; break; }
        }
        if (!inside) nav.dim[i] = 1;
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

  ImGui::Separator();

  // #15 Path-between: endpoints are picked from the canvas context menu ("Path:
  // set A/B"); this shows their status and a clear button. When both are set the
  // canvas dims everything off the connecting path.
  auto endpoint_label = [&](int32_t ir) -> std::string {
    if (ir < 0) return "(unset)";
    return "node #" + std::to_string(ir);
  };
  ImGui::TextDisabled("Path A: %s", endpoint_label(nav.path_a).c_str());
  ImGui::TextDisabled("Path B: %s", endpoint_label(nav.path_b).c_str());
  if (nav.path_active()) {
    if (ImGui::SmallButton("Clear path")) {
      nav.path_a = -1;
      nav.path_b = -1;
    }
  } else {
    ImGui::TextDisabled("(right-click a node -> Path: set A / set B)");
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

// --- #17 focus history -------------------------------------------------------

void nav_record_focus(App& app, uint32_t ir_node) {
  ViewState& vs = app.view();
  if (!vs.nav) vs.nav = std::make_unique<GraphNavState>();
  GraphNavState& nav = *vs.nav;
  // No-op when re-selecting the same node (this also dedupes the selection that a
  // back/forward step induces, since nav_goto_focus_pos sets focus_last_ir first).
  if (ir_node == nav.focus_last_ir) return;
  nav.focus_last_ir = ir_node;
  // Truncate any forward tail, then append and point at the new tip.
  if (nav.focus_pos >= 0 &&
      static_cast<size_t>(nav.focus_pos) + 1 < nav.focus_hist.size())
    nav.focus_hist.resize(static_cast<size_t>(nav.focus_pos) + 1);
  nav.focus_hist.push_back(ir_node);
  nav.focus_pos = static_cast<int32_t>(nav.focus_hist.size()) - 1;
}

bool nav_focus_can_back(App& app) {
  const GraphNavState* nav = app.view().nav.get();
  return nav != nullptr && nav->focus_pos > 0;
}

bool nav_focus_can_forward(App& app) {
  const GraphNavState* nav = app.view().nav.get();
  return nav != nullptr && nav->focus_pos >= 0 &&
         static_cast<size_t>(nav->focus_pos) + 1 < nav->focus_hist.size();
}

// Fly to the history entry at focus_pos, selecting it, without re-recording.
namespace {
void nav_goto_focus_pos(App& app) {
  GraphNavState& nav = *app.view().nav;
  if (nav.focus_pos < 0 ||
      static_cast<size_t>(nav.focus_pos) >= nav.focus_hist.size())
    return;
  uint32_t ir_node = nav.focus_hist[static_cast<size_t>(nav.focus_pos)];
  // Set focus_last_ir BEFORE the jump so the selection it induces is deduped by
  // nav_record_focus (no re-record of the node we're navigating to).
  nav.focus_last_ir = ir_node;
  nav_jump_to_ir_node(app, ir_node);
}
}  // namespace

void nav_focus_back(App& app) {
  if (!nav_focus_can_back(app)) return;
  --app.view().nav->focus_pos;
  nav_goto_focus_pos(app);
}

void nav_focus_forward(App& app) {
  if (!nav_focus_can_forward(app)) return;
  ++app.view().nav->focus_pos;
  nav_goto_focus_pos(app);
}

// --- #19 sticky pins ---------------------------------------------------------

void nav_toggle_pin(App& app, uint32_t ir_node) {
  ViewState& vs = app.view();
  if (!vs.nav) vs.nav = std::make_unique<GraphNavState>();
  auto& pins = vs.nav->pinned;
  auto it = std::find(pins.begin(), pins.end(), ir_node);
  if (it != pins.end())
    pins.erase(it);
  else
    pins.push_back(ir_node);
}

void draw_pinned_strip(App& app, ImVec2 canvas_origin, ImVec2 canvas_size) {
  ViewState& vs = app.view();
  GraphNavState* nav = vs.nav.get();
  if (nav == nullptr || nav->pinned.empty()) return;
  ModelSession& s = app.session();
  const ir::Model* m = s.model();
  if (m == nullptr) return;
  uint32_t gi = s.current_graph();
  if (gi >= m->graphs.size()) return;
  const auto& nodes = m->graphs[gi].nodes;

  ImDrawList* dl = ImGui::GetWindowDrawList();
  const bool dark = vs.dark_theme;
  const ImU32 chip_bg = dark ? IM_COL32(44, 50, 60, 235) : IM_COL32(226, 232, 240, 235);
  const ImU32 chip_bd = dark ? IM_COL32(90, 100, 116, 255) : IM_COL32(150, 160, 175, 255);
  const ImU32 chip_tx = dark ? IM_COL32(226, 232, 240, 255) : IM_COL32(30, 36, 46, 255);
  const ImU32 x_col   = dark ? IM_COL32(200, 120, 120, 255) : IM_COL32(180, 70, 70, 255);

  const float pad = 6.0f;
  const float gap = 6.0f;
  float x = canvas_origin.x + pad;
  const float y = canvas_origin.y + pad;
  const float line_h = ImGui::GetTextLineHeight();
  const float chip_h = line_h + 6.0f;
  const float x_w = line_h;  // click target for the remove "x"

  ImGuiIO& io = ImGui::GetIO();
  const ImVec2 mouse = io.MousePos;
  const bool clicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
  uint32_t to_unpin = UINT32_MAX;
  int32_t to_fly = -1;

  for (uint32_t ir_node : nav->pinned) {
    std::string label = "node";
    if (ir_node < nodes.size()) {
      std::string_view nm = m->str(nodes[ir_node].name);
      if (nm.empty()) nm = m->str(nodes[ir_node].op_type);
      if (!nm.empty()) label.assign(nm);
    }
    // Truncate very long names so the strip stays readable.
    if (label.size() > 22) { label.resize(21); label += "..."; }
    ImVec2 ts = ImGui::CalcTextSize(label.c_str());
    float chip_w = ts.x + 2.0f * pad + x_w;
    // Wrap to a new row if we'd overflow the canvas width.
    if (x + chip_w > canvas_origin.x + canvas_size.x - pad &&
        x > canvas_origin.x + pad) {
      break;  // v0.8.1: single-row strip; extra pins just don't show (rare).
    }
    ImVec2 cmin(x, y), cmax(x + chip_w, y + chip_h);
    dl->AddRectFilled(cmin, cmax, chip_bg, 4.0f);
    dl->AddRect(cmin, cmax, chip_bd, 4.0f, 0, 1.0f);
    dl->AddText(ImVec2(x + pad, y + 3.0f), chip_tx, label.c_str());
    // "x" remove glyph at the right end of the chip.
    ImVec2 xpos(cmax.x - x_w + 2.0f, y + 3.0f);
    dl->AddText(xpos, x_col, "x");

    if (clicked && mouse.y >= cmin.y && mouse.y <= cmax.y) {
      if (mouse.x >= cmax.x - x_w && mouse.x <= cmax.x)
        to_unpin = ir_node;                 // clicked the x
      else if (mouse.x >= cmin.x && mouse.x < cmax.x - x_w)
        to_fly = static_cast<int32_t>(ir_node);  // clicked the label
    }
    x += chip_w + gap;
  }

  if (to_unpin != UINT32_MAX) nav_toggle_pin(app, to_unpin);
  else if (to_fly >= 0) nav_jump_to_ir_node(app, static_cast<uint32_t>(to_fly));
}

// --- #16 bookmarks -----------------------------------------------------------

void draw_bookmarks_panel(App& app) {
  ViewState& vs = app.view();
  if (!vs.nav) vs.nav = std::make_unique<GraphNavState>();
  GraphNavState& nav = *vs.nav;
  if (!nav.show_bookmarks) return;

  if (!ImGui::Begin("Bookmarks", &nav.show_bookmarks)) {
    ImGui::End();
    return;
  }

  ModelSession& s = app.session();
  const bool have_model = s.model() != nullptr && s.has_graph();

  // Save current view as a new bookmark.
  static char name_buf[48] = {0};
  ImGui::SetNextItemWidth(-90.0f);
  ImGui::InputTextWithHint("##bm_name", "bookmark name...", name_buf,
                           sizeof(name_buf));
  ImGui::SameLine();
  if (ImGui::Button("Save", ImVec2(80.0f, 0.0f)) && have_model) {
    Bookmark b;
    b.name = name_buf[0] ? name_buf
                         : ("view " + std::to_string(nav.bookmarks.size() + 1));
    const Camera& cam = vs.cam;
    b.pan_x = cam.pan.x;
    b.pan_y = cam.pan.y;
    b.zoom = cam.zoom;
    // Record the selected node as an IR index (stable across collapse rebuilds).
    b.ir_node = -1;
    if (vs.selected_display >= 0) {
      const auto& disp = s.collapse().display_nodes();
      if (static_cast<size_t>(vs.selected_display) < disp.size()) {
        const DisplayNode& dn = disp[static_cast<size_t>(vs.selected_display)];
        if (!dn.is_group) b.ir_node = static_cast<int32_t>(dn.ir_node);
      }
    }
    nav.bookmarks.push_back(std::move(b));
    name_buf[0] = '\0';
  }

  ImGui::Separator();

  if (nav.bookmarks.empty()) {
    ImGui::TextDisabled("No bookmarks. Save the current camera + selection above.");
    ImGui::End();
    return;
  }

  // List with jump / delete.
  int32_t to_delete = -1;
  for (size_t i = 0; i < nav.bookmarks.size(); ++i) {
    Bookmark& b = nav.bookmarks[i];
    ImGui::PushID(static_cast<int>(i));
    if (ImGui::Button(b.name.c_str())) {
      // Jump: restore camera pose; re-select + fly if a node was saved.
      vs.cam.pan = ImVec2(b.pan_x, b.pan_y);
      vs.cam.zoom = b.zoom;
      vs.animating = false;
      if (b.ir_node >= 0) {
        int32_t disp = panel_detail::display_index_for_node(
            s.collapse(), static_cast<uint32_t>(b.ir_node));
        if (disp >= 0) vs.selected_display = disp;
      }
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("del")) to_delete = static_cast<int32_t>(i);
    ImGui::PopID();
  }
  if (to_delete >= 0)
    nav.bookmarks.erase(nav.bookmarks.begin() + to_delete);

  ImGui::End();
}

// --- #13 op-type legend / filter --------------------------------------------

void draw_legend_panel(App& app) {
  ViewState& vs = app.view();
  if (!vs.nav) vs.nav = std::make_unique<GraphNavState>();
  GraphNavState& nav = *vs.nav;
  if (!nav.show_legend) return;

  if (!ImGui::Begin("Legend", &nav.show_legend)) {
    ImGui::End();
    return;
  }

  ImGui::TextDisabled("Click a category to show/hide it");
  ImGui::Separator();

  const bool dark = vs.dark_theme;
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const float sw = ImGui::GetTextLineHeight();  // swatch size

  for (int c = 0; c <= static_cast<int>(OpCategory::Other); ++c) {
    OpCategory cat = static_cast<OpCategory>(c);
    uint32_t bit = 1u << static_cast<uint32_t>(c);
    bool visible = (nav.category_mask & bit) != 0u;

    ImGui::PushID(c);
    // Color swatch.
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImU32 col = App::category_color(cat, dark);
    if (!visible) col = (col & 0x00FFFFFFu) | 0x40000000u;  // faded when hidden
    dl->AddRectFilled(p, ImVec2(p.x + sw, p.y + sw), col, 3.0f);
    ImGui::Dummy(ImVec2(sw + 6.0f, sw));
    ImGui::SameLine();
    // Clicking the row (Selectable spanning the label) toggles the category bit.
    if (ImGui::Selectable(category_name(cat), false)) nav.category_mask ^= bit;
    ImGui::PopID();
  }

  ImGui::Separator();
  if (ImGui::SmallButton("Show all")) nav.category_mask = 0xFFFFFFFFu;
  ImGui::SameLine();
  if (ImGui::SmallButton("Hide all")) nav.category_mask = 0u;

  ImGui::End();
}

}  // namespace netvis
