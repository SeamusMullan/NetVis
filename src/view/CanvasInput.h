// SPDX-License-Identifier: Apache-2.0
// view/CanvasInput.h — the pure maths of canvas navigation (#158): how a wheel
// tick, a pinch, an arrow key or a mouse drag turns into a camera change.
//
// WHY THIS IS ITS OWN FILE: GraphCanvas.cpp owns the ImGui glue (it reads the
// wheel, the modifiers and the mouse), but every DECISION in that glue — "does
// this wheel tick pan or zoom", "which point stays fixed while zooming", "what
// does a Shift+wheel do on a Mac where the OS already swapped the axis" — is
// arithmetic on plain floats and booleans. Keeping that arithmetic here means it
// has no ImGui, GLFW or view dependency, lives in netvis_core, and is covered by
// netvis_tests (tests/test_canvas_input.cpp). The CMake list that puts this file
// in netvis_core is the same one SessionStore.cpp and ViewPrefs.cpp use; the bar
// for that list is strict, so this header includes only <cstdint> and
// <string_view>.
//
// Everything here is a pure function of its arguments. There is no time-based
// smoothing and no hidden state (DragPanState is passed in by the caller), so the
// same input sequence always produces the same camera.
//
// Input deltas come from OS drivers and are treated like any untrusted number: a
// NaN, an Inf or a million-notch burst must never poison the camera, so every
// function sanitizes before it multiplies. See docs/canvas-controls.md for the
// user-facing description of the gestures these functions implement.
#pragma once

#include <cstdint>
#include <string_view>

namespace netvis {

// --- Constants ---------------------------------------------------------------

// Zoom is clamped to this range everywhere a zoom is produced (wheel, pinch,
// keys, fit, fly-to targets, session restore) so the transform can never invert
// or explode. This is the single definition: Camera.cpp, GraphCanvas.cpp and
// SessionStore.cpp each used to carry a private copy (#158).
inline constexpr float kMinZoom = 0.02f;
inline constexpr float kMaxZoom = 4.0f;

// Zoom per wheel unit. 1.1^wheel, unchanged from before #158, so fractional
// trackpad deltas zoom smoothly and a notch is a 10% step.
inline constexpr float kWheelZoomBase = 1.1f;
// Shift+Up multiplies the zoom by this; Shift+Down divides by it (symmetric, so
// in-then-out returns to the start, unlike Netron's 1.1 / 0.9 pair).
inline constexpr float kKeyZoomStep = 1.1f;

// Screen pixels per 1.0 wheel unit for line-based devices (a mouse notch). About
// ImGui's own "one unit is about five lines" at NetVis's 16 px body font, and
// between Firefox (about 57 px) and Chromium (100 px) per notch.
inline constexpr float kPanPxPerNotch = 80.0f;
// Screen pixels per 1.0 wheel unit when the OS says the delta is precise
// (macOS trackpads and Magic Mouse). GLFW scales those deltas by 0.1, so 10 is
// its exact inverse and a two-finger swipe tracks the fingers 1:1.
inline constexpr float kPanPxPerPreciseUnit = 10.0f;
// Screen pixels per arrow-key press or key repeat.
inline constexpr float kArrowPanPx = 40.0f;

// Per-axis sanity clamp on one frame's wheel delta (200 notches). Bounds the pan
// at kMaxWheelUnitsPerFrame * kPanPxPerNotch = 16000 px and keeps the zoom
// exponent finite.
inline constexpr float kMaxWheelUnitsPerFrame = 200.0f;
// |ln factor| cap for one frame's accumulated pinch, so at most e-fold per frame.
inline constexpr float kMaxPinchLogPerFrame = 1.0f;
// Cap on keyboard zoom steps accumulated in one frame (auto-repeat floods).
inline constexpr int kMaxKeyZoomSteps = 50;

// --- Wheel mode --------------------------------------------------------------

// What a plain scroll does. Pan is the Netron default; Zoom is the behaviour of
// every release before #158 and is kept as a preference.
enum class WheelMode : uint8_t { Pan = 0, Zoom = 1 };

// "pan" | "zoom" — the strings persisted in view_prefs.json.
const char* wheel_mode_name(WheelMode m);
// Exact, case-sensitive inverse of wheel_mode_name. Returns false and leaves
// `out` untouched for anything else (so a hand-edited file degrades to the base).
bool wheel_mode_from_name(std::string_view s, WheelMode& out);

// --- Camera pose -------------------------------------------------------------

// The camera as plain floats, mirroring view/Camera.cpp's transform
//   screen = origin + world * zoom + pan
// ImVec2 is deliberately not used (same reason ViewSnapshot uses plain floats).
struct CamPose {
  float pan_x = 0.0f, pan_y = 0.0f, zoom = 1.0f;
};

// --- Wheel -------------------------------------------------------------------

struct WheelInput {
  float wheel_x = 0.0f;  // ImGui io.MouseWheelH (+ = scroll left)
  float wheel_y = 0.0f;  // ImGui io.MouseWheel  (+ = scroll up)
  bool shift = false;
  bool zoom_mod = false;  // see wheel_zoom_modifier
  bool precise = false;   // the OS says this delta is in (scaled) pixels
};
enum class WheelAction : uint8_t { None, Pan, Zoom };
struct WheelResult {
  WheelAction action = WheelAction::None;
  float pan_dx = 0.0f, pan_dy = 0.0f;  // screen px to add to the camera pan
  float zoom_factor = 1.0f;            // multiplier for the camera zoom
};
struct PanDelta {
  float dx = 0.0f, dy = 0.0f;
};

// Drag-to-pan bookkeeping owned by the caller (a file static in GraphCanvas.cpp).
struct DragPanState {
  bool active = false;
};

// True when the held modifier means "zoom, not scroll". `key_ctrl`/`key_super`
// are ImGui's io.KeyCtrl/io.KeySuper, which on macOS are ALREADY swapped (ImGui
// maps Command to KeyCtrl and Control to KeySuper under ConfigMacOSXBehaviors).
// So on macOS both Cmd and Control zoom; elsewhere only Ctrl does, and the
// Windows/Super key does not.
bool wheel_zoom_modifier(bool key_ctrl, bool key_super, bool is_macos);

// Decide what one frame's wheel delta does. Pure; see the .cpp for the rules.
WheelResult map_wheel(const WheelInput& in, WheelMode mode);

// --- Zoom --------------------------------------------------------------------

// Set the zoom to `target_zoom` (clamped to [kMinZoom, kMaxZoom]) keeping the
// world point under the canvas-relative screen anchor fixed. A non-finite anchor
// or an unusable target returns `cam` unchanged.
CamPose zoom_to(const CamPose& cam, float anchor_x, float anchor_y, float target_zoom);
// Multiply the zoom by `factor` about the anchor. A non-finite or non-positive
// factor returns `cam` unchanged.
CamPose zoom_about(const CamPose& cam, float anchor_x, float anchor_y, float factor);
// Zoom factor for an accumulated number of Shift+Up(+)/Shift+Down(-) steps.
float key_zoom_factor(int steps);

// --- Keyboard pan ------------------------------------------------------------

// Pan for one frame's arrow keys. Same convention as the wheel: Up moves the
// content down (+y), Left moves it right (+x). Opposite keys cancel.
PanDelta arrow_pan(bool up, bool down, bool left, bool right);

// --- Pinch -------------------------------------------------------------------

// Fold one trackpad magnification event (NSEvent.magnification, a fractional
// change in scale) into a running log-scale accumulator. Events that are
// non-finite or <= -1 (a scale that is zero or negative) are ignored.
float accumulate_pinch_log(float acc, double magnification);
// Zoom factor for an accumulated log: exp(acc), |acc| capped at
// kMaxPinchLogPerFrame; 1 for a non-finite accumulator. The product of
// (1 + m_i) equals exp(sum of log1p(m_i)), so the zoom follows finger spread 1:1.
float pinch_factor(float acc_log);

// --- Drag --------------------------------------------------------------------

// Advance a drag-to-pan by one frame and return the pan to apply. The press
// itself does not pan; once the drag passes ImGui's threshold the whole distance
// travelled so far is applied at once (so the content is glued to the cursor from
// then on), and each later frame applies that frame's mouse delta. The deltas
// returned over one drag sum to the final total drag delta. Releasing the button
// resets the state.
PanDelta drag_pan_step(DragPanState& st, bool button_down, bool past_threshold,
                       float total_dx, float total_dy, float frame_dx, float frame_dy);

}  // namespace netvis
