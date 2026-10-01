// SPDX-License-Identifier: Apache-2.0
// tests/test_canvas_input.cpp — #158 Netron-style canvas navigation, the pure part.
//
// What is covered: view/CanvasInput.{h,cpp}, which lives in netvis_core (see
// CMakeLists.txt) and holds every DECISION the canvas makes about input: how a
// wheel delta becomes a pan or a zoom, which modifier wins, how the zoom anchor
// moves the camera, how keys, a pinch and a drag map to the camera, and how each
// of those survives a NaN, an Inf or a huge delta from a driver.
//
// What is NOT covered, and why: the ImGui and GLFW glue in GraphCanvas.cpp and
// App.cpp (reading io.MouseWheel, IsItemActive, the menu items) and the macOS
// NSEvent bridge. netvis_tests links netvis_core only, and PR CI does not compile
// the GUI target. That glue is deliberately thin: it reads a signal, calls a
// function below, and applies the result. It is verified by building the app and
// the manual matrix in the PR description.
//
// No fixtures: this change parses no file format. Every input is a literal here.
// Exact == is used only where the expression is bit-identical by construction;
// everything else uses doctest::Approx.
#include <doctest/doctest.h>

#include <cmath>
#include <limits>
#include <string>

#include "view/CanvasInput.h"

using namespace netvis;
using doctest::Approx;

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

WheelInput wheel(float x, float y, bool shift = false, bool zoom_mod = false,
                 bool precise = false) {
  WheelInput in;
  in.wheel_x = x;
  in.wheel_y = y;
  in.shift = shift;
  in.zoom_mod = zoom_mod;
  in.precise = precise;
  return in;
}

// The world point under a canvas-relative screen point for a pose, i.e.
// Camera.cpp's screen_to_world with a zero origin.
float world_x(const CamPose& c, float sx) { return (sx - c.pan_x) / c.zoom; }
float world_y(const CamPose& c, float sy) { return (sy - c.pan_y) / c.zoom; }

}  // namespace

TEST_CASE("CanvasInput: constants are pinned") {
  // kMinZoom/kMaxZoom replaced three private copies (Camera.cpp, GraphCanvas.cpp,
  // SessionStore.cpp). SessionStore validates a persisted zoom against them, so a
  // drift here silently changes which saved tabs restore.
  CHECK(kMinZoom == 0.02f);
  CHECK(kMaxZoom == 4.0f);
  CHECK(kWheelZoomBase == 1.1f);
  CHECK(kKeyZoomStep == 1.1f);
  CHECK(kPanPxPerNotch == 80.0f);
  CHECK(kPanPxPerPreciseUnit == 10.0f);
  CHECK(kArrowPanPx == 40.0f);
  CHECK(kMaxWheelUnitsPerFrame == 200.0f);
  CHECK(kMaxPinchLogPerFrame == 1.0f);
  CHECK(kMaxKeyZoomSteps == 50);
}

TEST_CASE("CanvasInput: WheelMode names round-trip and reject near-misses") {
  CHECK(std::string(wheel_mode_name(WheelMode::Pan)) == "pan");
  CHECK(std::string(wheel_mode_name(WheelMode::Zoom)) == "zoom");

  WheelMode m = WheelMode::Zoom;
  CHECK(wheel_mode_from_name("pan", m));
  CHECK(m == WheelMode::Pan);
  CHECK(wheel_mode_from_name("zoom", m));
  CHECK(m == WheelMode::Zoom);

  // Case-sensitive, exact: a hand-edited file must not guess.
  for (const char* bad : {"Zoom", "", "scroll", "pan ", "PAN", " zoom"}) {
    WheelMode keep = WheelMode::Pan;
    CHECK_FALSE(wheel_mode_from_name(bad, keep));
    CHECK(keep == WheelMode::Pan);  // out untouched
    keep = WheelMode::Zoom;
    CHECK_FALSE(wheel_mode_from_name(bad, keep));
    CHECK(keep == WheelMode::Zoom);
  }
}

TEST_CASE("CanvasInput: wheel_zoom_modifier") {
  // (key_ctrl, key_super, is_macos). On macOS ImGui has swapped Cmd into KeyCtrl
  // and Control into KeySuper, so both zoom there; elsewhere the Super/Windows
  // key must not.
  CHECK(wheel_zoom_modifier(true, false, false));
  CHECK_FALSE(wheel_zoom_modifier(false, true, false));
  CHECK(wheel_zoom_modifier(false, true, true));
  CHECK(wheel_zoom_modifier(true, false, true));
  CHECK_FALSE(wheel_zoom_modifier(false, false, true));
  CHECK_FALSE(wheel_zoom_modifier(false, false, false));
}

TEST_CASE("CanvasInput: map_wheel Pan mode scrolls") {
  WheelResult r = map_wheel(wheel(0, +1), WheelMode::Pan);
  CHECK(r.action == WheelAction::Pan);
  CHECK(r.pan_dx == 0.0f);
  CHECK(r.pan_dy == 80.0f);

  r = map_wheel(wheel(0, -1), WheelMode::Pan);
  CHECK(r.action == WheelAction::Pan);
  CHECK(r.pan_dy == -80.0f);

  r = map_wheel(wheel(0, 0.25f), WheelMode::Pan);
  CHECK(r.pan_dy == 20.0f);

  r = map_wheel(wheel(-1, 0), WheelMode::Pan);
  CHECK(r.action == WheelAction::Pan);
  CHECK(r.pan_dx == -80.0f);
  CHECK(r.pan_dy == 0.0f);

  r = map_wheel(wheel(0.5f, -0.25f), WheelMode::Pan);
  CHECK(r.pan_dx == 40.0f);
  CHECK(r.pan_dy == -20.0f);

  CHECK(map_wheel(wheel(0, 0), WheelMode::Pan).action == WheelAction::None);

  // Linear in the delta, so fractional trackpad deltas are smooth and additive.
  const float a = map_wheel(wheel(0, 0.1f), WheelMode::Pan).pan_dy;
  const float b = map_wheel(wheel(0, 0.2f), WheelMode::Pan).pan_dy;
  const float c = map_wheel(wheel(0, 0.3f), WheelMode::Pan).pan_dy;
  CHECK(a + b == Approx(c));
}

TEST_CASE("CanvasInput: map_wheel precise deltas pan 1:1") {
  WheelResult r = map_wheel(wheel(0, 0.3f, false, false, true), WheelMode::Pan);
  CHECK(r.action == WheelAction::Pan);
  CHECK(r.pan_dy == Approx(3.0f));
  r = map_wheel(wheel(-0.5f, 0, false, false, true), WheelMode::Pan);
  CHECK(r.pan_dx == Approx(-5.0f));
  CHECK(r.pan_dy == Approx(0.0f));
}

TEST_CASE("CanvasInput: Shift+wheel pans horizontally, no double swap") {
  // Windows/Linux: ImGui leaves the delta in the vertical wheel.
  WheelResult r = map_wheel(wheel(0, +1, true), WheelMode::Pan);
  CHECK(r.action == WheelAction::Pan);
  CHECK(r.pan_dx == 80.0f);
  CHECK(r.pan_dy == 0.0f);
  // Wheel down scrolls right, matching ImGui's own swap.
  r = map_wheel(wheel(0, -1, true), WheelMode::Pan);
  CHECK(r.pan_dx == -80.0f);
  CHECK(r.pan_dy == 0.0f);

  // macOS: the OS already moved the delta to the horizontal axis. Swapping again
  // would zero it.
  r = map_wheel(wheel(+1, 0, true), WheelMode::Pan);
  CHECK(r.action == WheelAction::Pan);
  CHECK(r.pan_dx == 80.0f);
  CHECK(r.pan_dy == 0.0f);

  // Both axes present: no swap.
  r = map_wheel(wheel(0.5f, 1, true), WheelMode::Pan);
  CHECK(r.pan_dx == 40.0f);
  CHECK(r.pan_dy == 80.0f);

  CHECK(map_wheel(wheel(0, 0, true), WheelMode::Pan).action == WheelAction::None);
}

TEST_CASE("CanvasInput: zoom modifier zooms in Pan mode and beats Shift") {
  WheelResult r = map_wheel(wheel(0, +1, false, true), WheelMode::Pan);
  CHECK(r.action == WheelAction::Zoom);
  CHECK(r.zoom_factor == std::pow(1.1f, 1.0f));  // exact: same expression as pre-#158

  r = map_wheel(wheel(0, -2, false, true), WheelMode::Pan);
  CHECK(r.action == WheelAction::Zoom);
  CHECK(r.zoom_factor == Approx(1.0f / 1.21f));

  // Horizontal-only input does not zoom, and does not pan either.
  CHECK(map_wheel(wheel(3, 0, false, true), WheelMode::Pan).action == WheelAction::None);

  // The modifier beats Shift.
  r = map_wheel(wheel(0, 1, true, true), WheelMode::Pan);
  CHECK(r.action == WheelAction::Zoom);
  CHECK(r.zoom_factor == Approx(1.1f));

  // Precise input zooms at the same rate: GLFW's 0.1 scaling already makes 10 pt
  // of finger travel equal to one notch.
  r = map_wheel(wheel(0, 0.5f, false, true, true), WheelMode::Pan);
  CHECK(r.action == WheelAction::Zoom);
  CHECK(r.zoom_factor == Approx(std::pow(1.1f, 0.5f)));

  // A diagonal zoom does not drift: x is ignored.
  r = map_wheel(wheel(5, 1, false, true), WheelMode::Pan);
  CHECK(r.action == WheelAction::Zoom);
  CHECK(r.pan_dx == 0.0f);
  CHECK(r.pan_dy == 0.0f);
}

TEST_CASE("CanvasInput: map_wheel Zoom mode is the legacy behaviour") {
  WheelResult r = map_wheel(wheel(0, +1), WheelMode::Zoom);
  CHECK(r.action == WheelAction::Zoom);
  // Byte-identical to the pre-#158 expression std::pow(1.1f, io.MouseWheel).
  CHECK(r.zoom_factor == std::pow(1.1f, 1.0f));

  r = map_wheel(wheel(0, 1, true), WheelMode::Zoom);  // Shift is ignored
  CHECK(r.action == WheelAction::Zoom);
  CHECK(r.zoom_factor == Approx(1.1f));

  // The horizontal wheel was ignored before #158 and still is.
  CHECK(map_wheel(wheel(2, 0), WheelMode::Zoom).action == WheelAction::None);

  r = map_wheel(wheel(0, -1, false, true), WheelMode::Zoom);
  CHECK(r.action == WheelAction::Zoom);
  CHECK(r.zoom_factor == Approx(1.0f / 1.1f));

  // Zoom mode never pans.
  r = map_wheel(wheel(1, 1, true), WheelMode::Zoom);
  CHECK(r.action == WheelAction::Zoom);
  CHECK(r.pan_dx == 0.0f);
  CHECK(r.pan_dy == 0.0f);
}

TEST_CASE("CanvasInput: map_wheel survives hostile deltas") {
  // NaN on one axis becomes 0 for that axis only.
  WheelResult r = map_wheel(wheel(kNaN, 1), WheelMode::Pan);
  CHECK(r.action == WheelAction::Pan);
  CHECK(r.pan_dx == 0.0f);
  CHECK(r.pan_dy == 80.0f);

  CHECK(map_wheel(wheel(kInf, 0), WheelMode::Pan).action == WheelAction::None);
  CHECK(map_wheel(wheel(0, -kInf), WheelMode::Pan).action == WheelAction::None);
  CHECK(map_wheel(wheel(kNaN, kNaN), WheelMode::Pan).action == WheelAction::None);
  CHECK(map_wheel(wheel(0, -kInf), WheelMode::Zoom).action == WheelAction::None);
  CHECK(map_wheel(wheel(0, kNaN, false, true), WheelMode::Pan).action == WheelAction::None);

  // A million-notch burst is clamped to 200 units: 16000 px, never more.
  r = map_wheel(wheel(0, 1e9f), WheelMode::Pan);
  CHECK(r.action == WheelAction::Pan);
  CHECK(r.pan_dy == 16000.0f);
  r = map_wheel(wheel(-1e9f, 0), WheelMode::Pan);
  CHECK(r.pan_dx == -16000.0f);

  // In Zoom mode the factor is finite, and applying it lands on the clamp.
  r = map_wheel(wheel(0, 1e9f), WheelMode::Zoom);
  CHECK(r.action == WheelAction::Zoom);
  CHECK(std::isfinite(r.zoom_factor));
  CHECK(zoom_about(CamPose{0, 0, 1}, 0, 0, r.zoom_factor).zoom == kMaxZoom);
  r = map_wheel(wheel(0, -1e9f), WheelMode::Zoom);
  CHECK(std::isfinite(r.zoom_factor));
  CHECK(zoom_about(CamPose{0, 0, 1}, 0, 0, r.zoom_factor).zoom == kMinZoom);
}

TEST_CASE("CanvasInput: zoom_about keeps the anchor's world point fixed") {
  // pan' = a - (a - pan) * r. World under the anchor is (200, 150) before and
  // after.
  const CamPose before{100, 50, 1};
  const CamPose after = zoom_about(before, 300, 200, 2);
  CHECK(after.pan_x == -100.0f);
  CHECK(after.pan_y == -100.0f);
  CHECK(after.zoom == 2.0f);
  CHECK(world_x(before, 300) == Approx(200.0f));
  CHECK(world_y(before, 200) == Approx(150.0f));
  CHECK(world_x(after, 300) == Approx(200.0f));
  CHECK(world_y(after, 200) == Approx(150.0f));
}

TEST_CASE("CanvasInput: zoom clamps at both limits without moving the pan") {
  // Max: 3.9 * 1.1 = 4.29 clamps to 4.0, and the anchor's world point holds.
  const CamPose start{0, 0, 3.9f};
  const CamPose hi = zoom_about(start, 10, 10, 1.1f);
  CHECK(hi.zoom == kMaxZoom);
  CHECK(world_x(hi, 10) == Approx(world_x(start, 10)));
  CHECK(world_y(hi, 10) == Approx(world_y(start, 10)));

  // Already AT the limit: the ratio is exactly 1, so the pan is left alone.
  const CamPose at_max = zoom_about(CamPose{12.3f, -4.5f, kMaxZoom}, 10, 10, 1.1f);
  CHECK(at_max.zoom == kMaxZoom);
  CHECK(at_max.pan_x == 12.3f);
  CHECK(at_max.pan_y == -4.5f);

  // Min.
  const CamPose lo = zoom_about(CamPose{5, 5, kMinZoom}, 7, 7, 0.5f);
  CHECK(lo.zoom == 0.02f);
  CHECK(lo.pan_x == Approx(5.0f));
  CHECK(lo.pan_y == Approx(5.0f));
}

TEST_CASE("CanvasInput: zoom_to actual size is exact") {
  const CamPose r = zoom_to(CamPose{12, 34, 0.37f}, 50, 60, 1.0f);
  CHECK(r.zoom == 1.0f);
  CHECK(world_x(r, 50) == Approx(world_x(CamPose{12, 34, 0.37f}, 50)));
  CHECK(world_y(r, 60) == Approx(world_y(CamPose{12, 34, 0.37f}, 60)));

  // The target is clamped like any other zoom.
  CHECK(zoom_to(CamPose{}, 0, 0, 100.0f).zoom == kMaxZoom);
  CHECK(zoom_to(CamPose{}, 0, 0, 0.0001f).zoom == kMinZoom);
}

TEST_CASE("CanvasInput: a degenerate camera recovers on the next zoom") {
  // Zoom 0, negative and NaN are treated as 1 for the ratio, the guard
  // screen_to_world already applies.
  for (float bad_zoom : {0.0f, -2.0f, kNaN}) {
    const CamPose r = zoom_about(CamPose{0, 0, bad_zoom}, 0, 0, 2);
    CHECK(r.zoom == 2.0f);
    CHECK(r.pan_x == 0.0f);
    CHECK(r.pan_y == 0.0f);
  }
  // A poisoned pan component becomes 0; the rest stay finite.
  const CamPose p = zoom_about(CamPose{kNaN, 3, 1}, 0, 0, 2);
  CHECK(p.pan_x == 0.0f);
  CHECK(std::isfinite(p.pan_y));
  CHECK(std::isfinite(p.zoom));
  const CamPose q = zoom_about(CamPose{kInf, -kInf, 1}, 0, 0, 1.5f);
  CHECK(q.pan_x == 0.0f);
  CHECK(q.pan_y == 0.0f);

  // A zoom far outside the range, times a finite factor, lands on the limit
  // instead of overflowing to Inf and being rejected.
  const CamPose huge = zoom_about(CamPose{0, 0, 3e38f}, 0, 0, 1e30f);
  CHECK(huge.zoom == kMaxZoom);
}

TEST_CASE("CanvasInput: bad factors and anchors leave the pose unchanged") {
  const CamPose c{7, 8, 1.5f};
  for (float bad : {kNaN, 0.0f, -1.0f, kInf, -kInf}) {
    const CamPose r = zoom_about(c, 1, 2, bad);
    CHECK(r.pan_x == c.pan_x);
    CHECK(r.pan_y == c.pan_y);
    CHECK(r.zoom == c.zoom);
  }
  for (float bad : {kNaN, kInf, -kInf}) {
    const CamPose a = zoom_about(c, bad, 2, 2);
    CHECK(a.pan_x == c.pan_x);
    CHECK(a.zoom == c.zoom);
    const CamPose b = zoom_about(c, 1, bad, 2);
    CHECK(b.pan_y == c.pan_y);
    CHECK(b.zoom == c.zoom);
    const CamPose d = zoom_to(c, 1, 2, bad);
    CHECK(d.zoom == c.zoom);
  }
  // zoom_to also rejects a non-positive target.
  CHECK(zoom_to(c, 1, 2, 0.0f).zoom == c.zoom);
  CHECK(zoom_to(c, 1, 2, -3.0f).zoom == c.zoom);
}

TEST_CASE("CanvasInput: key_zoom_factor is symmetric and capped") {
  CHECK(key_zoom_factor(1) == 1.1f);
  CHECK(key_zoom_factor(-1) == Approx(1.0f / 1.1f));
  CHECK(key_zoom_factor(0) == 1.0f);
  CHECK(key_zoom_factor(2) == Approx(1.21f));
  // An auto-repeat flood is capped at 50 steps per frame.
  CHECK(key_zoom_factor(1000) == key_zoom_factor(50));
  CHECK(key_zoom_factor(-1000) == key_zoom_factor(-50));
  CHECK(std::isfinite(key_zoom_factor(2147483647)));
  CHECK(std::isfinite(key_zoom_factor(-2147483647 - 1)));

  // In then out returns to the start (Netron's 1.1 / 0.9 pair would not).
  const CamPose c{20, 30, 0.7f};
  const float cx = 400, cy = 300;
  const CamPose in = zoom_about(c, cx, cy, key_zoom_factor(+1));
  const CamPose back = zoom_about(in, cx, cy, key_zoom_factor(-1));
  CHECK(back.zoom == Approx(c.zoom).epsilon(1e-6));
  CHECK(back.pan_x == Approx(c.pan_x).epsilon(1e-4));
  CHECK(back.pan_y == Approx(c.pan_y).epsilon(1e-4));
}

TEST_CASE("CanvasInput: arrow_pan") {
  PanDelta d = arrow_pan(true, false, false, false);
  CHECK(d.dx == 0.0f);
  CHECK(d.dy == 40.0f);
  d = arrow_pan(false, true, false, false);
  CHECK(d.dx == 0.0f);
  CHECK(d.dy == -40.0f);
  d = arrow_pan(false, false, true, false);
  CHECK(d.dx == 40.0f);
  CHECK(d.dy == 0.0f);
  d = arrow_pan(false, false, false, true);
  CHECK(d.dx == -40.0f);
  CHECK(d.dy == 0.0f);
  // Opposite keys cancel.
  d = arrow_pan(true, true, false, false);
  CHECK(d.dx == 0.0f);
  CHECK(d.dy == 0.0f);
  d = arrow_pan(false, false, true, true);
  CHECK(d.dx == 0.0f);
  CHECK(d.dy == 0.0f);
  d = arrow_pan(false, false, false, false);
  CHECK(d.dx == 0.0f);
  CHECK(d.dy == 0.0f);
}

TEST_CASE("CanvasInput: pinch accumulates multiplicatively") {
  // Two +10% events are 1.1 * 1.1 = 1.21, because the accumulator is a sum of
  // logs: the zoom follows finger spread 1:1, as Safari's e.scale does.
  float acc = 0.0f;
  acc = accumulate_pinch_log(acc, 0.1);
  acc = accumulate_pinch_log(acc, 0.1);
  CHECK(pinch_factor(acc) == Approx(1.21f));

  // A pinch out then back in cancels.
  acc = accumulate_pinch_log(0.0f, 0.25);
  acc = accumulate_pinch_log(acc, 1.0 / 1.25 - 1.0);
  CHECK(pinch_factor(acc) == Approx(1.0f).epsilon(1e-5));
}

TEST_CASE("CanvasInput: pinch ignores hostile magnifications") {
  const float start = 0.25f;
  CHECK(accumulate_pinch_log(start, -1.0) == start);   // scale 0
  CHECK(accumulate_pinch_log(start, -2.0) == start);   // negative scale
  CHECK(accumulate_pinch_log(start, std::numeric_limits<double>::quiet_NaN()) == start);
  CHECK(accumulate_pinch_log(start, std::numeric_limits<double>::infinity()) == start);
  CHECK(accumulate_pinch_log(start, -std::numeric_limits<double>::infinity()) == start);
  // A huge but legal magnification stays finite.
  CHECK(std::isfinite(accumulate_pinch_log(start, 1e300)));
}

TEST_CASE("CanvasInput: pinch_factor") {
  CHECK(pinch_factor(0.0f) == 1.0f);
  CHECK(pinch_factor(10.0f) == Approx(std::exp(1.0f)));   // capped at |ln f| <= 1
  CHECK(pinch_factor(-10.0f) == Approx(std::exp(-1.0f)));
  CHECK(pinch_factor(kNaN) == 1.0f);
  CHECK(pinch_factor(kInf) == 1.0f);
  CHECK(pinch_factor(-kInf) == 1.0f);
}

TEST_CASE("CanvasInput: drag_pan_step glues the content after the threshold") {
  DragPanState st;
  float sum_x = 0, sum_y = 0;
  auto step = [&](bool down, bool past, float tx, float ty, float fx, float fy) {
    const PanDelta d = drag_pan_step(st, down, past, tx, ty, fx, fy);
    sum_x += d.dx;
    sum_y += d.dy;
    return d;
  };

  // Below the threshold: nothing moves, not active.
  PanDelta d = step(true, false, 3, 0, 3, 0);
  CHECK(d.dx == 0.0f);
  CHECK(d.dy == 0.0f);
  CHECK_FALSE(st.active);

  // Crossing the threshold catches up the whole distance so far.
  d = step(true, true, 8, 2, 5, 2);
  CHECK(d.dx == 8.0f);
  CHECK(d.dy == 2.0f);
  CHECK(st.active);

  // From then on, the frame delta.
  d = step(true, true, 10, 5, 2, 3);
  CHECK(d.dx == 2.0f);
  CHECK(d.dy == 3.0f);
  CHECK(st.active);

  // Release ends the drag.
  d = step(false, true, 10, 5, 0, 0);
  CHECK(d.dx == 0.0f);
  CHECK(d.dy == 0.0f);
  CHECK_FALSE(st.active);

  // The deltas over the whole drag sum to the final total drag delta.
  CHECK(sum_x == 10.0f);
  CHECK(sum_y == 5.0f);
}

TEST_CASE("CanvasInput: a new drag starts fresh") {
  DragPanState st;
  st.active = true;  // left over from a drag that ended off-frame

  // past_threshold with no button down pans nothing and resets.
  PanDelta d = drag_pan_step(st, false, true, 50, 50, 5, 5);
  CHECK(d.dx == 0.0f);
  CHECK(d.dy == 0.0f);
  CHECK_FALSE(st.active);

  // A fresh press below the threshold does nothing...
  d = drag_pan_step(st, true, false, 1, 1, 1, 1);
  CHECK(d.dx == 0.0f);
  CHECK(d.dy == 0.0f);
  CHECK_FALSE(st.active);

  // ...and a fresh drag past it catches up from the new press, not the old one.
  d = drag_pan_step(st, true, true, 7, 0, 7, 0);
  CHECK(d.dx == 7.0f);
  CHECK(st.active);
}
