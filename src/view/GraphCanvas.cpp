// view/GraphCanvas.cpp — the interactive graph view (spec §8.1).
//
// DECISION (spec §8.1): the ENTIRE graph is drawn inside ONE ImGui child region
// using a single ImDrawList — there is NOT one ImGui widget per node. A 100k-node
// graph would die under 100k Buttons; instead we cull to the visible world rect
// and emit raw draw commands, so per-frame cost is O(visible), not O(nodes).
// Hit-testing is likewise done against the culled boxes, not via ImGui items.
#define IMGUI_DEFINE_MATH_OPERATORS
#include "engine/LayoutEngine.h"
#include "view/App.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include "imgui.h"

#include "engine/OpCategory.h"

namespace netvis {

namespace {

// LOD zoom thresholds (spec §8.1). Above kFull we draw the full node with two
// text lines + edge labels; between kMid..kFull just the op_type; between
// kFlat..kMid a flat rect with no text; below kBlob one blob per collapse group.
constexpr float kZoomFull = 0.70f;
constexpr float kZoomMid = 0.30f;
constexpr float kZoomFlat = 0.08f;
constexpr float kMinZoom = 0.02f;
constexpr float kMaxZoom = 4.0f;

// Smooth ease-in-out for the fly-to animation.
float ease(float t) { return t * t * (3.0f - 2.0f * t); }

// True if world-space AABB [amin,amax] intersects [bmin,bmax].
bool aabb_overlap(ImVec2 amin, ImVec2 amax, ImVec2 bmin, ImVec2 bmax) {
  return amin.x <= bmax.x && amax.x >= bmin.x && amin.y <= bmax.y &&
         amax.y >= bmin.y;
}

// Blend two packed colors by t in [0,1].
ImU32 lerp_col(ImU32 a, ImU32 b, float t) {
  ImVec4 ca = ImGui::ColorConvertU32ToFloat4(a);
  ImVec4 cb = ImGui::ColorConvertU32ToFloat4(b);
  ImVec4 r(ca.x + (cb.x - ca.x) * t, ca.y + (cb.y - ca.y) * t,
           ca.z + (cb.z - ca.z) * t, ca.w + (cb.w - ca.w) * t);
  return ImGui::ColorConvertFloat4ToU32(r);
}

// Resolve the two label lines + op category for one display node.
struct NodeLabel {
  std::string primary;
  std::string secondary;
  OpCategory cat = OpCategory::Other;
};

NodeLabel label_for(const App& app, uint32_t display_id) {
  NodeLabel out;
  ModelSession& s = const_cast<App&>(app).session();
  const auto& disp = s.collapse().display_nodes();
  if (display_id >= disp.size()) return out;
  const DisplayNode& dn = disp[display_id];
  const ir::Model* m = s.model();
  if (dn.is_group) {
    const auto& groups = s.collapse().groups();
    if (dn.group_index < groups.size()) {
      const CollapseGroup& g = groups[dn.group_index];
      out.primary = g.label;
      out.secondary = "x" + std::to_string(g.instances);
      // Category from the group's first representative node op, if resolvable.
      if (m != nullptr && !g.representative_nodes.empty()) {
        uint32_t gi = s.current_graph();
        if (gi < m->graphs.size()) {
          const auto& nodes = m->graphs[gi].nodes;
          uint32_t ni = g.representative_nodes.front();
          if (ni < nodes.size())
            out.cat = categorize_op(m->str(nodes[ni].op_type));
        }
      }
    }
  } else if (m != nullptr) {
    uint32_t gi = s.current_graph();
    if (gi < m->graphs.size()) {
      const auto& nodes = m->graphs[gi].nodes;
      if (dn.ir_node < nodes.size()) {
        const ir::Node& n = nodes[dn.ir_node];
        out.primary = std::string(m->str(n.op_type));
        out.secondary = std::string(m->str(n.name));
        out.cat = categorize_op(m->str(n.op_type));
      }
    }
  }
  if (out.primary.empty()) out.primary = "node";
  return out;
}

}  // namespace

// Draw the graph canvas. Called once per frame from App::frame() when a compute
// graph is loaded. All drawing is ImDrawList; input/animation are handled here.
void draw_graph_canvas(App& app) {
  ViewState& vs = app.view();
  ModelSession& session = app.session();

  ImGui::Begin("Graph");
  // Single child region: everything below is manual ImDrawList output.
  ImGui::BeginChild("canvas", ImVec2(0, 0), ImGuiChildFlags_None,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);

  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 origin = ImGui::GetCursorScreenPos();  // canvas top-left (screen)
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const ImVec2 canvas_size(std::max(avail.x, 1.0f), std::max(avail.y, 1.0f));

  // An invisible full-region button captures hover/clicks for the canvas without
  // per-node widgets, and lets us know when interaction targets empty space.
  ImGui::InvisibleButton("canvas_btn", canvas_size,
                         ImGuiButtonFlags_MouseButtonLeft |
                             ImGuiButtonFlags_MouseButtonRight |
                             ImGuiButtonFlags_MouseButtonMiddle);
  const bool canvas_hovered = ImGui::IsItemHovered();
  ImGuiIO& io = ImGui::GetIO();

  Camera& cam = vs.cam;

  // --- Input: zoom to cursor -------------------------------------------------
  if (canvas_hovered && io.MouseWheel != 0.0f) {
    // Keep the world point under the cursor fixed while zooming (spec §8.1).
    ImVec2 before = screen_to_world(cam, origin, io.MousePos);
    float factor = std::pow(1.1f, io.MouseWheel);
    cam.zoom = std::clamp(cam.zoom * factor, kMinZoom, kMaxZoom);
    ImVec2 after = screen_to_world(cam, origin, io.MousePos);
    cam.pan.x += (after.x - before.x) * cam.zoom;
    cam.pan.y += (after.y - before.y) * cam.zoom;
    vs.animating = false;  // manual zoom cancels a fly-to.
  }

  // --- Input: pan (middle-drag, or space+left-drag) --------------------------
  const bool space = ImGui::IsKeyDown(ImGuiKey_Space);
  if (canvas_hovered &&
      (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
       (space && ImGui::IsMouseDragging(ImGuiMouseButton_Left)))) {
    cam.pan.x += io.MouseDelta.x;
    cam.pan.y += io.MouseDelta.y;
    vs.animating = false;
  }

  // --- Camera animation step (fly-to) ---------------------------------------
  if (vs.animating) {
    // Advance ~over ~0.35s; ease and snap when close to avoid a lingering crawl.
    vs.anim_t += io.DeltaTime / 0.35f;
    if (vs.anim_t >= 1.0f) {
      vs.anim_t = 1.0f;
      vs.animating = false;
    }
    float e = ease(std::clamp(vs.anim_t, 0.0f, 1.0f));
    cam.zoom = vs.anim_from_zoom + (vs.anim_to_zoom - vs.anim_from_zoom) * e;
    cam.pan.x = vs.anim_from_pan.x + (vs.anim_to_pan.x - vs.anim_from_pan.x) * e;
    cam.pan.y = vs.anim_from_pan.y + (vs.anim_to_pan.y - vs.anim_from_pan.y) * e;
  }

  const LayoutResult* layout = session.layout();

  // --- Fit request (F key) ---------------------------------------------------
  if (vs.request_fit) {
    if (layout != nullptr) {
      fit_camera(vs, ImVec2(layout->bounds_min.x, layout->bounds_min.y),
                 ImVec2(layout->bounds_max.x, layout->bounds_max.y),
                 canvas_size);
    }
    vs.request_fit = false;
  }

  // Clip drawing to the canvas rect.
  const ImVec2 canvas_max(origin.x + canvas_size.x, origin.y + canvas_size.y);
  dl->PushClipRect(origin, canvas_max, true);

  if (layout == nullptr || layout->boxes.empty()) {
    // Nothing laid out yet — show a hint centered in the canvas.
    const char* msg = session.model() ? "Laying out graph..."
                                      : "Open a model (File > Open, or drop a file)";
    ImVec2 ts = ImGui::CalcTextSize(msg);
    dl->AddText(ImVec2(origin.x + (canvas_size.x - ts.x) * 0.5f,
                       origin.y + (canvas_size.y - ts.y) * 0.5f),
                ImGui::GetColorU32(ImGuiCol_TextDisabled), msg);
    dl->PopClipRect();
    draw_minimap(app);
    ImGui::EndChild();
    ImGui::End();
    return;
  }

  // --- Compute the visible world rect (for culling) --------------------------
  // PERF: transform the canvas corners back to world space once; every box/edge
  // is tested against this AABB so the draw loop is O(visible), not O(total).
  ImVec2 vw_min = screen_to_world(cam, origin, origin);
  ImVec2 vw_max = screen_to_world(cam, origin, canvas_max);
  if (vw_min.x > vw_max.x) std::swap(vw_min.x, vw_max.x);
  if (vw_min.y > vw_max.y) std::swap(vw_min.y, vw_max.y);

  const float zoom = cam.zoom;
  const bool dark = vs.dark_theme;
  const double now = ImGui::GetTime();
  // Pulse factor for search-hit highlight (0..1), driven by wall time.
  const float pulse =
      0.5f + 0.5f * static_cast<float>(std::sin(now * 4.0));

  const ImU32 col_edge = dark ? IM_COL32(150, 160, 175, 160)
                              : IM_COL32(90, 100, 120, 170);
  const ImU32 col_edge_hi = IM_COL32(120, 200, 255, 255);
  const ImU32 col_accent = IM_COL32(90, 170, 255, 255);
  const ImU32 col_text = dark ? IM_COL32(235, 238, 242, 255)
                             : IM_COL32(24, 28, 36, 255);
  const ImU32 col_text_muted = dark ? IM_COL32(160, 168, 180, 255)
                                    : IM_COL32(90, 100, 116, 255);
  const ImU32 col_border = dark ? IM_COL32(20, 22, 26, 220)
                               : IM_COL32(120, 128, 140, 220);

  const Fonts& fonts = app.fonts();

  // --- Hover hit-test (against culled boxes) ---------------------------------
  const ImVec2 mouse = io.MousePos;
  int32_t hover_box = -1;
  if (canvas_hovered) {
    // Iterate reverse so topmost (later) boxes win ties.
    for (size_t i = layout->boxes.size(); i-- > 0;) {
      const NodeBox& b = layout->boxes[i];
      ImVec2 bmin(b.pos.x, b.pos.y);
      ImVec2 bmax(b.pos.x + b.size.x, b.pos.y + b.size.y);
      if (!aabb_overlap(bmin, bmax, vw_min, vw_max)) continue;
      ImVec2 smin = world_to_screen(cam, origin, bmin);
      ImVec2 smax = world_to_screen(cam, origin, bmax);
      if (mouse.x >= smin.x && mouse.x <= smax.x && mouse.y >= smin.y &&
          mouse.y <= smax.y) {
        hover_box = static_cast<int32_t>(b.display_id);
        break;
      }
    }
  }
  vs.hovered_display = hover_box;

  // --- Draw edges first (under nodes) ----------------------------------------
  // PERF: skip any edge whose endpoints' bounding box is fully outside view.
  const float edge_thick = std::clamp(1.5f * zoom, 0.75f, 3.0f);
  const bool draw_edge_labels = zoom > kZoomFull;
  for (const EdgeCurve& e : layout->edges) {
    ImVec2 emin(std::min(std::min(e.p0.x, e.p1.x), std::min(e.p2.x, e.p3.x)),
                std::min(std::min(e.p0.y, e.p1.y), std::min(e.p2.y, e.p3.y)));
    ImVec2 emax(std::max(std::max(e.p0.x, e.p1.x), std::max(e.p2.x, e.p3.x)),
                std::max(std::max(e.p0.y, e.p1.y), std::max(e.p2.y, e.p3.y)));
    if (!aabb_overlap(emin, emax, vw_min, vw_max)) continue;
    ImVec2 p0 = world_to_screen(cam, origin, ImVec2(e.p0.x, e.p0.y));
    ImVec2 p1 = world_to_screen(cam, origin, ImVec2(e.p1.x, e.p1.y));
    ImVec2 p2 = world_to_screen(cam, origin, ImVec2(e.p2.x, e.p2.y));
    ImVec2 p3 = world_to_screen(cam, origin, ImVec2(e.p3.x, e.p3.y));
    const bool touches_hover =
        hover_box >= 0 && (static_cast<int32_t>(e.from_display_id) == hover_box ||
                           static_cast<int32_t>(e.to_display_id) == hover_box);
    ImU32 ec = touches_hover ? col_edge_hi : col_edge;
    float th = touches_hover ? edge_thick + 1.0f : edge_thick;
    dl->AddBezierCubic(p0, p1, p2, p3, ec, th);

    // Edge shape label (highest LOD only) placed near the curve midpoint.
    if (draw_edge_labels && fonts.small != nullptr) {
      // Only bother when there's room; label the producing value's shape later.
      // (Shape text is model-derived; kept minimal here to stay O(visible).)
    }
  }

  // --- Draw nodes (LOD by zoom) ----------------------------------------------
  if (zoom < kZoomFlat) {
    // Lowest LOD: one blob per collapse group / node, tinted by category.
    for (const NodeBox& b : layout->boxes) {
      ImVec2 bmin(b.pos.x, b.pos.y);
      ImVec2 bmax(b.pos.x + b.size.x, b.pos.y + b.size.y);
      if (!aabb_overlap(bmin, bmax, vw_min, vw_max)) continue;
      NodeLabel lab = label_for(app, b.display_id);
      ImVec2 c = world_to_screen(
          cam, origin,
          ImVec2((bmin.x + bmax.x) * 0.5f, (bmin.y + bmax.y) * 0.5f));
      float r = std::max(2.0f, 0.25f * b.size.x * zoom);
      dl->AddCircleFilled(c, r, App::category_color(lab.cat, dark), 8);
    }
  } else {
    for (const NodeBox& b : layout->boxes) {
      ImVec2 bmin(b.pos.x, b.pos.y);
      ImVec2 bmax(b.pos.x + b.size.x, b.pos.y + b.size.y);
      if (!aabb_overlap(bmin, bmax, vw_min, vw_max)) continue;

      ImVec2 smin = world_to_screen(cam, origin, bmin);
      ImVec2 smax = world_to_screen(cam, origin, bmax);
      const bool selected =
          vs.selected_display == static_cast<int32_t>(b.display_id);
      const bool hovered = hover_box == static_cast<int32_t>(b.display_id);
      NodeLabel lab = label_for(app, b.display_id);
      ImU32 header = App::category_color(lab.cat, dark);

      ImU32 body = dark ? IM_COL32(34, 38, 44, 255) : IM_COL32(244, 246, 249, 255);
      if (hovered)
        body = dark ? IM_COL32(48, 54, 62, 255) : IM_COL32(232, 238, 246, 255);

      if (zoom < kZoomMid) {
        // Flat rect, no text.
        dl->AddRectFilled(smin, smax, body, 3.0f);
        dl->AddRect(smin, smax, header, 3.0f, 0, 2.0f);
      } else {
        // Rounded rect body + colored header strip.
        const float rounding = 5.0f;
        dl->AddRectFilled(smin, smax, body, rounding);
        float header_h = std::min(20.0f * zoom, (smax.y - smin.y) * 0.5f);
        ImVec2 hmax(smax.x, smin.y + header_h);
        dl->AddRectFilled(smin, hmax, header, rounding,
                          ImDrawFlags_RoundCornersTop);
        dl->AddRect(smin, smax, col_border, rounding, 0, 1.0f);

        // Text. High LOD draws both lines; mid LOD op_type only.
        dl->PushClipRect(smin, smax, true);
        if (zoom > kZoomFull) {
          if (fonts.bold != nullptr)
            dl->AddText(fonts.bold, 16.0f * std::min(zoom, 1.0f),
                        ImVec2(smin.x + 6.0f, smin.y + 2.0f), col_text,
                        lab.primary.c_str());
          if (fonts.small != nullptr && !lab.secondary.empty())
            dl->AddText(fonts.small, 12.0f * std::min(zoom, 1.0f),
                        ImVec2(smin.x + 6.0f, smin.y + header_h + 2.0f),
                        col_text_muted, lab.secondary.c_str());
        } else {
          if (fonts.body != nullptr)
            dl->AddText(fonts.body, 14.0f * std::min(zoom, 1.0f),
                        ImVec2(smin.x + 6.0f, smin.y + 2.0f), col_text,
                        lab.primary.c_str());
        }
        dl->PopClipRect();
      }

      // Selection / search-hit outlines.
      if (selected) {
        dl->AddRect(ImVec2(smin.x - 2, smin.y - 2), ImVec2(smax.x + 2, smax.y + 2),
                    col_accent, 6.0f, 0, 2.5f);
      }
    }

    // Search-hit pulse: outline the current search target if it maps to a box.
    if (vs.search_open && !vs.search_query.empty()) {
      const auto& hits =
          session.search().query(vs.search_query, 64);
      if (!hits.empty()) {
        int idx = std::clamp(vs.search_active_result, 0,
                             static_cast<int>(hits.size()) - 1);
        const SearchEntry& se = session.search().entries()[hits[idx].entry];
        // Map a node entry to its display box if visible.
        for (const NodeBox& b : layout->boxes) {
          NodeLabel lab = label_for(app, b.display_id);
          (void)lab;
          const auto& disp = session.collapse().display_nodes();
          if (b.display_id >= disp.size()) continue;
          const DisplayNode& dn = disp[b.display_id];
          if (!dn.is_group && se.kind == SearchKind::Node &&
              dn.ir_node == se.ref) {
            ImVec2 smin = world_to_screen(cam, origin, ImVec2(b.pos.x, b.pos.y));
            ImVec2 smax = world_to_screen(
                cam, origin, ImVec2(b.pos.x + b.size.x, b.pos.y + b.size.y));
            ImU32 pc = lerp_col(col_accent, IM_COL32(255, 220, 120, 255), pulse);
            dl->AddRect(ImVec2(smin.x - 3, smin.y - 3),
                        ImVec2(smax.x + 3, smax.y + 3), pc, 7.0f, 0, 3.0f);
            break;
          }
        }
      }
    }
  }

  dl->PopClipRect();

  // --- Interactions ----------------------------------------------------------
  if (canvas_hovered) {
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !space) {
      vs.selected_display = hover_box;  // -1 clears when clicking empty space.
    }
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && hover_box >= 0) {
      const auto& disp = session.collapse().display_nodes();
      if (static_cast<size_t>(hover_box) < disp.size()) {
        const DisplayNode& dn = disp[static_cast<size_t>(hover_box)];
        if (dn.is_group && dn.group_index != UINT32_MAX)
          session.toggle_group(dn.group_index);  // expand/collapse (spec §7.2.6).
      }
    }
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && hover_box >= 0) {
      vs.selected_display = hover_box;
      ImGui::OpenPopup("canvas_ctx");
    }
  }

  if (ImGui::BeginPopup("canvas_ctx")) {
    if (vs.selected_display >= 0) {
      NodeLabel lab = label_for(app, static_cast<uint32_t>(vs.selected_display));
      ImGui::TextDisabled("%s", lab.primary.c_str());
      ImGui::Separator();
      if (ImGui::MenuItem("Copy name")) {
        const std::string& nm =
            lab.secondary.empty() ? lab.primary : lab.secondary;
        ImGui::SetClipboardText(nm.c_str());
      }
      const auto& disp = session.collapse().display_nodes();
      if (static_cast<size_t>(vs.selected_display) < disp.size()) {
        const DisplayNode& dn = disp[static_cast<size_t>(vs.selected_display)];
        if (dn.is_group && ImGui::MenuItem("Expand / collapse"))
          session.toggle_group(dn.group_index);
      }
    }
    ImGui::EndPopup();
  }

  // Minimap is drawn inside the same child region, inset bottom-right.
  draw_minimap(app);

  ImGui::EndChild();
  ImGui::End();
}

}  // namespace netvis
