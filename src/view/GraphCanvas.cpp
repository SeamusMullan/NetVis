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
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "imgui.h"

#include "engine/OpCategory.h"
#include "view/DiffPanel.h"
#include "view/GraphNav.h"
#include "view/PanelHelpers.h"

namespace netvis {

namespace {

// Per-display-node readability classification for the "hide constant edges"
// toggle (Feature 2). Recomputed only when the session (generation / graph /
// collapse) changes — NOT every frame. is_const_source[i] marks a leaf display
// node that is a constant/initializer source (inputs.count==0 or op categorizes
// to OpCategory::Tensor); const_badge[i] counts a consumer's hidden constant/
// initializer inputs (for the "+N" badge).
struct ReadabilityCache {
  uint64_t key_generation = UINT64_MAX;
  uint32_t key_graph = UINT32_MAX;
  uint64_t key_collapse = UINT64_MAX;
  bool valid = false;
  std::vector<uint8_t> is_const_source;   // indexed by display id
  std::vector<uint16_t> const_badge;      // indexed by display id
};

// True if IR node `n` is a constant/initializer *source* (a leaf producing a
// constant with no compute inputs).
bool node_is_const_source(const ir::Model& m, const ir::Node& n) {
  if (n.inputs.count == 0) return true;
  return categorize_op(m.str(n.op_type)) == OpCategory::Tensor;
}

// Recompute the readability cache if the session key changed. Returns it.
const ReadabilityCache& readability_cache(App& app) {
  static ReadabilityCache cache;
  ModelSession& s = app.session();
  const uint64_t gen = s.generation();
  const uint32_t gi = s.current_graph();
  const uint64_t ch = s.collapse().collapse_hash();
  if (cache.valid && cache.key_generation == gen && cache.key_graph == gi &&
      cache.key_collapse == ch)
    return cache;

  cache.key_generation = gen;
  cache.key_graph = gi;
  cache.key_collapse = ch;
  cache.valid = true;

  const auto& disp = s.collapse().display_nodes();
  const size_t n = disp.size();
  cache.is_const_source.assign(n, 0);
  cache.const_badge.assign(n, 0);

  const ir::Model* m = s.model();
  if (m == nullptr || gi >= m->graphs.size()) return cache;
  const ir::Graph& g = m->graphs[gi];

  // Per-IR-node const-source classification + set of initializer value names.
  std::vector<uint8_t> node_const(g.nodes.size(), 0);
  for (size_t i = 0; i < g.nodes.size(); ++i)
    node_const[i] = node_is_const_source(*m, g.nodes[i]) ? 1 : 0;
  // Initializer value NAMES: a consumer input with no producer whose value name
  // matches an initializer is a hidden constant input (counted in the badge).
  std::vector<StringId> init_names;
  init_names.reserve(g.initializers.size());
  for (const ir::TensorRef& t : g.initializers) init_names.push_back(t.name);
  auto is_init_name = [&](StringId id) {
    for (StringId in : init_names)
      if (in == id) return true;
    return false;
  };

  for (size_t i = 0; i < n; ++i) {
    const DisplayNode& dn = disp[i];
    if (dn.is_group) continue;  // only leaf nodes are const sources.
    if (dn.ir_node >= g.nodes.size()) continue;
    if (node_const[dn.ir_node]) cache.is_const_source[i] = 1;

    // Count this consumer's hidden constant/initializer inputs.
    const ir::Node& node = g.nodes[dn.ir_node];
    uint32_t badge = 0;
    for (uint32_t slot = 0; slot < node.inputs.count; ++slot) {
      uint32_t vidx = panel_detail::resolve_edge_value(g, node.inputs, slot);
      if (vidx == UINT32_MAX || vidx >= g.values.size()) continue;
      const ir::ValueInfo& vi = g.values[vidx];
      if (vi.producer >= 0) {
        if (static_cast<size_t>(vi.producer) < node_const.size() &&
            node_const[vi.producer])
          ++badge;
      } else if (is_init_name(vi.name)) {
        ++badge;  // initializer input (no producing node / no edge).
      }
    }
    if (badge > 0)
      cache.const_badge[i] = static_cast<uint16_t>(std::min<uint32_t>(badge, 0xFFFFu));
  }
  return cache;
}

// Badge glyph size tracks zoom like the node label text (clamped).
float text_px_for_badge(float zoom) {
  return std::clamp(11.0f * zoom, 11.0f * 0.5f, 11.0f * 3.0f);
}

// Apply an alpha multiplier to a packed color (for nav "dim" de-emphasis).
ImU32 with_alpha_mul(ImU32 col, float mul) {
  ImVec4 c = ImGui::ColorConvertU32ToFloat4(col);
  c.w *= mul;
  return ImGui::ColorConvertFloat4ToU32(c);
}

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

  // --- Navigation masks + readability cache (display-id indexed) -------------
  // ensure_nav() (called from App::frame before us) keeps these in sync; guard
  // for a null nav or a stale-size mask across a display-list rebuild this frame.
  const GraphNavState* nav = vs.nav.get();
  const size_t disp_count = session.collapse().display_nodes().size();
  auto nav_hidden = [&](uint32_t did) -> bool {
    return nav != nullptr && did < nav->hidden.size() &&
           nav->hidden.size() == disp_count && nav->hidden[did] != 0;
  };
  auto nav_dim = [&](uint32_t did) -> bool {
    return nav != nullptr && did < nav->dim.size() &&
           nav->dim.size() == disp_count && nav->dim[did] != 0;
  };
  const bool hide_const = vs.hide_const_edges;
  const ReadabilityCache& rc = readability_cache(app);
  auto is_const_source = [&](uint32_t did) -> bool {
    return hide_const && did < rc.is_const_source.size() &&
           rc.is_const_source.size() == disp_count &&
           rc.is_const_source[did] != 0;
  };
  // A display box is culled entirely when nav hides it OR it is a hidden const
  // source. A box is de-emphasized when nav dims it.
  auto box_culled = [&](uint32_t did) -> bool {
    return nav_hidden(did) || is_const_source(did);
  };
  const float kDimAlpha = 0.18f;

  // --- Hover hit-test (against culled boxes) ---------------------------------
  const ImVec2 mouse = io.MousePos;
  int32_t hover_box = -1;
  if (canvas_hovered) {
    // Iterate reverse so topmost (later) boxes win ties.
    for (size_t i = layout->boxes.size(); i-- > 0;) {
      const NodeBox& b = layout->boxes[i];
      if (box_culled(b.display_id)) continue;  // can't hover a culled box.
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
    // Feature 2: skip edges whose source is a hidden constant/initializer box.
    // Also cull edges touching a nav-hidden endpoint.
    if (is_const_source(e.from_display_id)) continue;
    if (nav_hidden(e.from_display_id) || nav_hidden(e.to_display_id)) continue;
    const bool edge_dim =
        nav_dim(e.from_display_id) || nav_dim(e.to_display_id);
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
    if (edge_dim && !touches_hover) ec = with_alpha_mul(ec, kDimAlpha);
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
      if (box_culled(b.display_id)) continue;
      ImVec2 bmin(b.pos.x, b.pos.y);
      ImVec2 bmax(b.pos.x + b.size.x, b.pos.y + b.size.y);
      if (!aabb_overlap(bmin, bmax, vw_min, vw_max)) continue;
      NodeLabel lab = label_for(app, b.display_id);
      ImVec2 c = world_to_screen(
          cam, origin,
          ImVec2((bmin.x + bmax.x) * 0.5f, (bmin.y + bmax.y) * 0.5f));
      float r = std::max(2.0f, 0.25f * b.size.x * zoom);
      // Diff overlay overrides the category color at the low-LOD blob site.
      ImU32 blob = App::category_color(lab.cat, dark);
      DiffTint tint = diff_tint_for_display(app, static_cast<int32_t>(b.display_id));
      if (tint.active) blob = tint.color;
      if (nav_dim(b.display_id)) blob = with_alpha_mul(blob, kDimAlpha);
      dl->AddCircleFilled(c, r, blob, 8);
    }
  } else {
    for (const NodeBox& b : layout->boxes) {
      if (box_culled(b.display_id)) continue;
      ImVec2 bmin(b.pos.x, b.pos.y);
      ImVec2 bmax(b.pos.x + b.size.x, b.pos.y + b.size.y);
      if (!aabb_overlap(bmin, bmax, vw_min, vw_max)) continue;

      ImVec2 smin = world_to_screen(cam, origin, bmin);
      ImVec2 smax = world_to_screen(cam, origin, bmax);
      const bool selected =
          vs.selected_display == static_cast<int32_t>(b.display_id);
      const bool hovered = hover_box == static_cast<int32_t>(b.display_id);
      const bool dimmed = nav_dim(b.display_id);
      NodeLabel lab = label_for(app, b.display_id);
      ImU32 header = App::category_color(lab.cat, dark);
      // Diff overlay overrides the header color when a diff is active.
      DiffTint tint = diff_tint_for_display(app, static_cast<int32_t>(b.display_id));
      if (tint.active) header = tint.color;
      if (dimmed) header = with_alpha_mul(header, kDimAlpha);

      ImU32 body = dark ? IM_COL32(34, 38, 44, 255) : IM_COL32(244, 246, 249, 255);
      if (hovered)
        body = dark ? IM_COL32(48, 54, 62, 255) : IM_COL32(232, 238, 246, 255);
      if (dimmed) body = with_alpha_mul(body, kDimAlpha);

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

        // Text. High LOD draws both lines; mid LOD op_type only. Label sizes
        // scale WITH zoom so text tracks the node box — the world-space node
        // grows by `zoom`, so the glyphs must too. (A previous cap of
        // min(zoom,1.0) froze text at its baked pixel size, making it shrink to
        // an invisible speck relative to a zoomed-in node.) Clamp the upper size
        // so the font atlas doesn't blur badly at extreme zoom.
        auto text_px = [zoom](float base) {
          return std::clamp(base * zoom, base * 0.5f, base * 3.0f);
        };
        dl->PushClipRect(smin, smax, true);
        if (zoom > kZoomFull) {
          if (fonts.bold != nullptr)
            dl->AddText(fonts.bold, text_px(16.0f),
                        ImVec2(smin.x + 6.0f * zoom, smin.y + 2.0f * zoom),
                        col_text, lab.primary.c_str());
          if (fonts.small != nullptr && !lab.secondary.empty())
            dl->AddText(fonts.small, text_px(12.0f),
                        ImVec2(smin.x + 6.0f * zoom, smin.y + header_h + 2.0f),
                        col_text_muted, lab.secondary.c_str());
        } else {
          if (fonts.body != nullptr)
            dl->AddText(fonts.body, text_px(14.0f),
                        ImVec2(smin.x + 6.0f * zoom, smin.y + 2.0f * zoom),
                        col_text, lab.primary.c_str());
        }
        dl->PopClipRect();
      }

      // Selection / search-hit outlines.
      if (selected) {
        dl->AddRect(ImVec2(smin.x - 2, smin.y - 2), ImVec2(smax.x + 2, smax.y + 2),
                    col_accent, 6.0f, 0, 2.5f);
      }

      // Feature 2: "+N" badge counting this consumer's hidden constant/
      // initializer inputs (only when the hide-const toggle is on and we're at a
      // legible LOD).
      if (hide_const && zoom >= kZoomFlat &&
          b.display_id < rc.const_badge.size() &&
          rc.const_badge.size() == disp_count && rc.const_badge[b.display_id] > 0 &&
          fonts.small != nullptr) {
        char badge[16];
        std::snprintf(badge, sizeof(badge), "+%u",
                      static_cast<unsigned>(rc.const_badge[b.display_id]));
        float bs = text_px_for_badge(zoom);
        ImVec2 ts = fonts.small->CalcTextSizeA(bs, FLT_MAX, 0.0f, badge);
        ImVec2 bp(smin.x - ts.x - 4.0f, smin.y);
        dl->AddRectFilled(ImVec2(bp.x - 2, bp.y - 1),
                          ImVec2(bp.x + ts.x + 2, bp.y + ts.y + 1),
                          IM_COL32(60, 66, 78, 230), 3.0f);
        dl->AddText(fonts.small, bs, bp, col_text_muted, badge);
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
