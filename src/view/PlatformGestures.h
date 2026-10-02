// SPDX-License-Identifier: Apache-2.0
// view/PlatformGestures.h — the one platform-specific input the canvas needs that
// GLFW 3.4 does not deliver (#158).
//
// Two signals reach the canvas from macOS that GLFW drops or flattens:
//   - a trackpad pinch (NSEventTypeMagnify), which GLFW has no handler for, and
//   - whether a scroll delta is "precise" (a trackpad or Magic Mouse, in scaled
//     pixels) rather than line-based (a mouse wheel), which GLFW flattens into the
//     same io.MouseWheel number.
// PlatformGestures_mac.mm taps both with an NSEvent local monitor. On every other
// platform the stub in PlatformGestures_stub.cpp answers "no pinch, not precise",
// which is the honest answer: GLFW 3.4 gives Windows and Linux no way to tell, and
// NetVis does not guess from the size of a delta.
//
// Everything here is main-thread only. The macOS monitor runs inside
// glfwPollEvents on the same thread as the ImGui frame, so no atomics are needed.
#pragma once

struct GLFWwindow;

namespace netvis {

// Start observing pinch and scroll events for `window`. Idempotent. Call once,
// right after the ImGui GLFW backend is initialised.
void platform_gestures_install(GLFWwindow* window);
// Stop observing. Call before glfwDestroyWindow so the monitor never sees a dead
// window. Safe to call without a prior install.
void platform_gestures_uninstall();

// Sum of log1p(magnification) of every pinch event since the last call, then
// reset to 0 (see accumulate_pinch_log in view/CanvasInput.h). 0 where there is no
// bridge.
float platform_take_pinch_log();
// Whether the most recent scroll event reported precise deltas. False before any
// scroll event has been seen, and always false where there is no bridge.
bool platform_last_scroll_precise();

}  // namespace netvis
