// SPDX-License-Identifier: Apache-2.0
// view/CanvasInput.cpp — see CanvasInput.h. Pure float maths; only <cmath> and
// <algorithm>. No float is ever cast to an integer here (UBSan's
// float-cast-overflow), and std::clamp is never called on an unvalidated float
// because it passes NaN straight through: bounds checks are written as
// `!(x >= lo)`, the same idiom ViewPrefs.cpp uses for ui_scale, so a NaN fails
// the check instead of slipping past it.
#include "view/CanvasInput.h"

#include <algorithm>
#include <cmath>

namespace netvis {

namespace {

// Non-finite becomes 0; finite is clamped to +-kMaxWheelUnitsPerFrame.
float sanitize_wheel_axis(float v) {
  if (!std::isfinite(v)) return 0.0f;
  if (v > kMaxWheelUnitsPerFrame) return kMaxWheelUnitsPerFrame;
  if (v < -kMaxWheelUnitsPerFrame) return -kMaxWheelUnitsPerFrame;
  return v;
}

// The zoom used for ratio maths. A camera with zoom 0, negative or NaN is
// treated as 1, the same recovery screen_to_world applies for a degenerate zoom.
float sane_zoom(float z) { return (std::isfinite(z) && z > 0.0f) ? z : 1.0f; }

// A pan component that is not finite becomes 0 (recovery from a poisoned camera).
float sane_pan(float p) { return std::isfinite(p) ? p : 0.0f; }

// Clamp a FINITE target into [kMinZoom, kMaxZoom]; the callers reject non-finite
// targets first. NaN-safe in form regardless.
float clamp_zoom(float t) {
  if (!(t >= kMinZoom)) return kMinZoom;
  if (!(t <= kMaxZoom)) return kMaxZoom;
  return t;
}

}  // namespace

const char* wheel_mode_name(WheelMode m) {
  return m == WheelMode::Zoom ? "zoom" : "pan";
}

bool wheel_mode_from_name(std::string_view s, WheelMode& out) {
  if (s == "pan") {
    out = WheelMode::Pan;
    return true;
  }
  if (s == "zoom") {
    out = WheelMode::Zoom;
    return true;
  }
  return false;
}

bool wheel_zoom_modifier(bool key_ctrl, bool key_super, bool is_macos) {
  return key_ctrl || (is_macos && key_super);
}

WheelResult map_wheel(const WheelInput& in, WheelMode mode) {
  WheelResult r;
  const float x = sanitize_wheel_axis(in.wheel_x);
  const float y = sanitize_wheel_axis(in.wheel_y);

  // Legacy (the behaviour of every release before #158): the vertical wheel
  // zooms, the horizontal axis and every modifier are ignored. The factor is the
  // exact pre-#158 expression, std::pow(1.1f, io.MouseWheel).
  if (mode == WheelMode::Zoom) {
    if (y == 0.0f) return r;
    r.action = WheelAction::Zoom;
    r.zoom_factor = std::pow(kWheelZoomBase, y);
    return r;
  }

  // Pan mode, zoom modifier held (Ctrl, or Cmd/Control on macOS): zoom. The
  // modifier beats Shift. x is ignored so a diagonal trackpad swipe does not
  // drift while zooming. The rate is the same for precise input: GLFW's 0.1
  // scaling already makes 10 pt of finger travel equal to one notch.
  if (in.zoom_mod) {
    if (y == 0.0f) return r;
    r.action = WheelAction::Zoom;
    r.zoom_factor = std::pow(kWheelZoomBase, y);
    return r;
  }

  // Pan mode: scroll. Shift turns the vertical wheel into a horizontal pan. On
  // Windows and Linux ImGui does not swap, so the delta arrives in y. On macOS
  // the OS has already moved it to x and y is 0, so only swap when x is 0 — that
  // guard is what prevents a double swap.
  float h = x;
  float v = y;
  if (in.shift && h == 0.0f) {
    h = v;
    v = 0.0f;
  }
  if (h == 0.0f && v == 0.0f) return r;
  const float px = in.precise ? kPanPxPerPreciseUnit : kPanPxPerNotch;
  r.action = WheelAction::Pan;
  r.pan_dx = h * px;
  r.pan_dy = v * px;
  return r;
}

CamPose zoom_to(const CamPose& cam, float anchor_x, float anchor_y, float target_zoom) {
  if (!std::isfinite(anchor_x) || !std::isfinite(anchor_y) || !std::isfinite(target_zoom) ||
      !(target_zoom > 0.0f))
    return cam;
  const float z = sane_zoom(cam.zoom);
  const float z2 = clamp_zoom(target_zoom);
  const float r = z2 / z;
  CamPose out;
  out.zoom = z2;
  if (r == 1.0f) {
    // At a clamp limit (or a no-op request) the pan is left alone: a - (a - p)
    // is not bit-exactly p, and a camera that wobbles at the limit would jitter.
    out.pan_x = sane_pan(cam.pan_x);
    out.pan_y = sane_pan(cam.pan_y);
  } else {
    // The world point under the anchor stays fixed:
    //   anchor = world * z + pan = world * z2 + pan'   =>   pan' = a - (a - pan) * r
    out.pan_x = sane_pan(anchor_x - (anchor_x - cam.pan_x) * r);
    out.pan_y = sane_pan(anchor_y - (anchor_y - cam.pan_y) * r);
  }
  return out;
}

CamPose zoom_about(const CamPose& cam, float anchor_x, float anchor_y, float factor) {
  if (!std::isfinite(factor) || !(factor > 0.0f)) return cam;
  // The product is formed in double and clamped before narrowing, so a finite
  // factor on a huge (poisoned) camera zoom cannot overflow to Inf and be
  // rejected; it lands on the clamp limit instead. Both operands are finite and
  // positive, so the double product is finite and never NaN.
  const double t = static_cast<double>(sane_zoom(cam.zoom)) * static_cast<double>(factor);
  const double lo = static_cast<double>(kMinZoom);
  const double hi = static_cast<double>(kMaxZoom);
  const double tc = t < lo ? lo : (t > hi ? hi : t);
  return zoom_to(cam, anchor_x, anchor_y, static_cast<float>(tc));
}

float key_zoom_factor(int steps) {
  const int s = std::clamp(steps, -kMaxKeyZoomSteps, kMaxKeyZoomSteps);
  return std::pow(kKeyZoomStep, static_cast<float>(s));
}

PanDelta arrow_pan(bool up, bool down, bool left, bool right) {
  PanDelta d;
  if (up) d.dy += kArrowPanPx;
  if (down) d.dy -= kArrowPanPx;
  if (left) d.dx += kArrowPanPx;
  if (right) d.dx -= kArrowPanPx;
  return d;
}

float accumulate_pinch_log(float acc, double magnification) {
  if (!std::isfinite(magnification) || magnification <= -1.0) return acc;
  return static_cast<float>(static_cast<double>(acc) + std::log1p(magnification));
}

float pinch_factor(float acc_log) {
  if (!std::isfinite(acc_log)) return 1.0f;
  float l = acc_log;
  if (l > kMaxPinchLogPerFrame) l = kMaxPinchLogPerFrame;
  if (l < -kMaxPinchLogPerFrame) l = -kMaxPinchLogPerFrame;
  return std::exp(l);
}

PanDelta drag_pan_step(DragPanState& st, bool button_down, bool past_threshold,
                       float total_dx, float total_dy, float frame_dx, float frame_dy) {
  PanDelta d;
  if (!button_down) {
    st.active = false;
    return d;
  }
  if (!st.active) {
    if (!past_threshold) return d;
    // Catch up the distance travelled while below the threshold so the content
    // is glued to the cursor from here on.
    st.active = true;
    d.dx = total_dx;
    d.dy = total_dy;
    return d;
  }
  d.dx = frame_dx;
  d.dy = frame_dy;
  return d;
}

bool rect_contains(const ScreenRect& r, float x, float y) {
  return x >= r.min_x && x <= r.max_x && y >= r.min_y && y <= r.max_y;
}

bool PinnedStripLayout::next(float text_w, PinnedChip& out) {
  const float chip_w = text_w + 2.0f * kPinnedStripPad + line_h_;  // label + "x" target
  // Same overflow rule the strip always had: stop at the first chip that would
  // cross the right padding, but never refuse the first one.
  if (next_x_ + chip_w > origin_x_ + canvas_w_ - kPinnedStripPad &&
      next_x_ > origin_x_ + kPinnedStripPad)
    return false;
  const float y = origin_y_ + kPinnedStripPad;
  out.box = ScreenRect{next_x_, y, next_x_ + chip_w, y + line_h_ + 6.0f};
  out.remove_x = out.box.max_x - line_h_;
  next_x_ += chip_w + kPinnedStripGap;
  return true;
}

bool canvas_click_selects(bool released, bool past_drag_threshold,
                          bool press_owned_by_overlay, bool space_held) {
  return released && !past_drag_threshold && !press_owned_by_overlay && !space_held;
}

}  // namespace netvis
