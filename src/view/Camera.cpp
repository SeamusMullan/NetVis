// view/Camera.cpp — world<->screen transform + camera animation helpers.
//
// DECISION (spec §8.1): layout lives in world space; the canvas applies a single
// pan+zoom transform. Keeping the transform here (a handful of tiny pure
// functions) means the canvas, search fly-to, and minimap all agree on exactly
// one mapping — there is no second, subtly-different copy to drift out of sync.
#define IMGUI_DEFINE_MATH_OPERATORS
// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h (pulled in by
// App.h) references but does not itself include; pre-include it here.
#include "engine/LayoutEngine.h"
#include "view/App.h"

#include <algorithm>
#include <cmath>

namespace netvis {

// Zoom is clamped to this range everywhere a zoom is produced (scroll, fit,
// animation targets) so the transform can never invert or explode.
static constexpr float kMinZoom = 0.02f;
static constexpr float kMaxZoom = 4.0f;

// world -> screen: origin + world*zoom + pan. `origin` is the canvas top-left in
// screen pixels; pan is an additional screen-space offset.
ImVec2 world_to_screen(const Camera& c, ImVec2 origin, ImVec2 world) {
  return ImVec2(origin.x + world.x * c.zoom + c.pan.x,
                origin.y + world.y * c.zoom + c.pan.y);
}

// screen -> world: exact inverse of world_to_screen.
ImVec2 screen_to_world(const Camera& c, ImVec2 origin, ImVec2 screen) {
  // Guard against a degenerate zoom (never expected, but avoids a div-by-zero).
  float z = c.zoom != 0.0f ? c.zoom : 1.0f;
  return ImVec2((screen.x - origin.x - c.pan.x) / z,
                (screen.y - origin.y - c.pan.y) / z);
}

// Begin a smooth fly-to so `world_pt` ends up centered in a viewport of
// `view_size` at `target_zoom` (spec §8.4). The canvas steps anim_t each frame.
void animate_camera_to(ViewState& vs, ImVec2 world_pt, ImVec2 view_size,
                       float target_zoom) {
  target_zoom = std::clamp(target_zoom, kMinZoom, kMaxZoom);
  vs.animating = true;
  vs.anim_from_pan = vs.cam.pan;
  vs.anim_from_zoom = vs.cam.zoom;
  vs.anim_to_zoom = target_zoom;
  // We want world_to_screen(origin, world_pt) == origin + view_size/2, i.e.
  // world_pt*zoom + pan == view_size/2  ->  pan = view_size/2 - world_pt*zoom.
  vs.anim_to_pan = ImVec2(view_size.x * 0.5f - world_pt.x * target_zoom,
                          view_size.y * 0.5f - world_pt.y * target_zoom);
  vs.anim_t = 0.0f;
}

// Fit the whole layout bounds into `view_size` with a small margin (spec §8.1
// 'F' key). Sets zoom (clamped) and pans so the bounds center is view-centered.
void fit_camera(ViewState& vs, ImVec2 bounds_min, ImVec2 bounds_max,
                ImVec2 view_size) {
  float w = bounds_max.x - bounds_min.x;
  float h = bounds_max.y - bounds_min.y;
  // Empty / degenerate bounds: reset to identity rather than dividing by ~0.
  if (w < 1e-3f || h < 1e-3f || view_size.x < 1e-3f || view_size.y < 1e-3f) {
    vs.cam.zoom = 1.0f;
    vs.cam.pan = ImVec2(0, 0);
    return;
  }
  // Leave a 10% margin so nodes at the edge are not flush against the border.
  const float kMargin = 0.9f;
  float zoom = std::min(view_size.x / w, view_size.y / h) * kMargin;
  zoom = std::clamp(zoom, kMinZoom, kMaxZoom);
  ImVec2 center((bounds_min.x + bounds_max.x) * 0.5f,
                (bounds_min.y + bounds_max.y) * 0.5f);
  vs.cam.zoom = zoom;
  vs.cam.pan = ImVec2(view_size.x * 0.5f - center.x * zoom,
                      view_size.y * 0.5f - center.y * zoom);
  vs.animating = false;  // an explicit fit cancels any in-flight animation.
}

}  // namespace netvis
