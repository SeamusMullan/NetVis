// view/Minimap.cpp — overview inset in the graph canvas (spec §8.5).
//
// DECISION (spec §8.5): the minimap is drawn with the SAME single ImDrawList as
// the canvas, inset bottom-right — no separate window, no per-node widgets. It
// shows the whole layout bounds scaled to fit a small box, plus a draggable
// viewport rectangle marking the current visible world rect; dragging it pans
// the camera. This is drawn AFTER the graph so it overlays on top.
#define IMGUI_DEFINE_MATH_OPERATORS
#include "engine/LayoutEngine.h"
#include "view/App.h"

#include <algorithm>
#include <cstdint>

#include "imgui.h"

namespace netvis {

// Draw the minimap. Called from draw_graph_canvas() while its child region and
// draw list are still active.
void draw_minimap(App& app) {
  ViewState& vs = app.view();
  if (!vs.show_minimap) return;

  ModelSession& session = app.session();
  const LayoutResult* layout = session.layout();
  // Guard: nothing to overview until a layout exists with real bounds.
  if (layout == nullptr || layout->boxes.empty()) return;

  const float world_w = layout->bounds_max.x - layout->bounds_min.x;
  const float world_h = layout->bounds_max.y - layout->bounds_min.y;
  if (world_w < 1e-3f || world_h < 1e-3f) return;

  ImDrawList* dl = ImGui::GetWindowDrawList();

  // Position the minimap inset in the bottom-right corner of the canvas child.
  const ImVec2 win_min = ImGui::GetWindowPos();
  const ImVec2 win_sz = ImGui::GetWindowSize();
  const float kPad = 12.0f;
  const float kMaxW = 220.0f, kMaxH = 160.0f;

  // Fit world aspect ratio inside the max box.
  float scale = std::min(kMaxW / world_w, kMaxH / world_h);
  float map_w = world_w * scale;
  float map_h = world_h * scale;

  ImVec2 map_max(win_min.x + win_sz.x - kPad, win_min.y + win_sz.y - kPad);
  ImVec2 map_min(map_max.x - map_w, map_max.y - map_h);

  const bool dark = vs.dark_theme;
  ImU32 bg = dark ? IM_COL32(18, 20, 24, 220) : IM_COL32(240, 242, 246, 230);
  ImU32 frame = dark ? IM_COL32(70, 76, 86, 255) : IM_COL32(150, 156, 168, 255);
  ImU32 node_col = dark ? IM_COL32(120, 150, 200, 200)
                        : IM_COL32(90, 120, 180, 200);
  ImU32 view_col = IM_COL32(90, 170, 255, 255);

  // Backing panel.
  dl->AddRectFilled(map_min, map_max, bg, 4.0f);
  dl->AddRect(map_min, map_max, frame, 4.0f, 0, 1.0f);

  // world -> minimap transform.
  auto to_map = [&](float wx, float wy) -> ImVec2 {
    return ImVec2(map_min.x + (wx - layout->bounds_min.x) * scale,
                  map_min.y + (wy - layout->bounds_min.y) * scale);
  };

  // Draw every box as a tiny tinted rect. Cheap: minimap boxes are ~1px, and the
  // whole graph is small on screen so no culling is needed here.
  for (const NodeBox& b : layout->boxes) {
    ImVec2 a = to_map(b.pos.x, b.pos.y);
    ImVec2 c = to_map(b.pos.x + b.size.x, b.pos.y + b.size.y);
    // Ensure at least 1px so tiny nodes remain visible.
    if (c.x - a.x < 1.0f) c.x = a.x + 1.0f;
    if (c.y - a.y < 1.0f) c.y = a.y + 1.0f;
    dl->AddRectFilled(a, c, node_col);
  }

  // Current visible world rect: invert the camera at the canvas corners.
  const Camera& cam = vs.cam;
  const ImVec2 canvas_origin = win_min;  // child region top-left in screen space.
  ImVec2 vw_min = screen_to_world(cam, canvas_origin, win_min);
  ImVec2 vw_max = screen_to_world(
      cam, canvas_origin, ImVec2(win_min.x + win_sz.x, win_min.y + win_sz.y));
  ImVec2 r0 = to_map(vw_min.x, vw_min.y);
  ImVec2 r1 = to_map(vw_max.x, vw_max.y);
  // Clamp the viewport rect to the minimap panel so it never spills out.
  r0.x = std::clamp(r0.x, map_min.x, map_max.x);
  r0.y = std::clamp(r0.y, map_min.y, map_max.y);
  r1.x = std::clamp(r1.x, map_min.x, map_max.x);
  r1.y = std::clamp(r1.y, map_min.y, map_max.y);
  dl->AddRect(r0, r1, view_col, 2.0f, 0, 2.0f);

  // --- Drag-to-pan -----------------------------------------------------------
  // Hit-test the minimap panel manually (no widget) and, on drag, recenter the
  // camera on the world point under the cursor.
  ImGuiIO& io = ImGui::GetIO();
  const ImVec2 m = io.MousePos;
  const bool over_map = m.x >= map_min.x && m.x <= map_max.x &&
                        m.y >= map_min.y && m.y <= map_max.y;
  if (over_map &&
      (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
       ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f))) {
    // Map click position back to world coordinates.
    float wx = layout->bounds_min.x + (m.x - map_min.x) / scale;
    float wy = layout->bounds_min.y + (m.y - map_min.y) / scale;
    // Center the canvas on (wx,wy): pan = view_center - world*zoom.
    Camera& mut = vs.cam;
    mut.pan.x = win_sz.x * 0.5f - wx * mut.zoom;
    mut.pan.y = win_sz.y * 0.5f - wy * mut.zoom;
    vs.animating = false;  // manual navigation cancels any fly-to.
  }
}

}  // namespace netvis
