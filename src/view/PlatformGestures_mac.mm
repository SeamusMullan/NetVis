// SPDX-License-Identifier: Apache-2.0
// view/PlatformGestures_mac.mm — the macOS side of PlatformGestures.h (#158).
//
// GLFW 3.4 drops NSEventTypeMagnify (its content view has no magnifyWithEvent:),
// so a trackpad pinch never reaches the app, and it flattens every scroll into one
// io.MouseWheel number whether it came from a wheel or a trackpad. An NSEvent LOCAL
// monitor sees both before GLFW does and lets every event through untouched.
//
// Why this is small and safe:
//   - A local monitor runs on the main thread during glfwPollEvents(), in the same
//     thread and phase as the ImGui frame, so it needs no atomics. It sees a scroll
//     event before GLFWContentView forwards it, so the precise flag is current by
//     the time ImGui receives the delta.
//   - It allocates nothing per event and does O(1) work.
//   - It returns every event unchanged, so GLFW's behaviour is untouched.
//   - Uninstalling before glfwDestroyWindow means the block never sees a dead window.
//   - If the monitor never fires, pinch is a no-op and `precise` stays false, so a
//     trackpad pans at the notch scale: a graceful fallback, not a failure.
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#import <AppKit/AppKit.h>
#include <cmath>
#include <cstdio>
#include "view/CanvasInput.h"
#include "view/PlatformGestures.h"

namespace {
id s_monitor = nil;            // ARC-managed; main thread only
NSWindow* s_window = nil;
float s_pinch_log = 0.0f;      // sum of log1p(magnification) since last take
bool s_last_precise = false;   // hasPreciseScrollingDeltas of the latest scroll event
}  // namespace

void netvis::platform_gestures_install(GLFWwindow* w) {
  if (s_monitor != nil || w == nullptr) return;              // idempotent
  s_window = glfwGetCocoaWindow(w);
  if (s_window == nil) {
    std::fprintf(stderr, "netvis: trackpad pinch unavailable (no NSWindow)\n");
    return;
  }
  s_monitor = [NSEvent
      addLocalMonitorForEventsMatchingMask:(NSEventMaskMagnify | NSEventMaskScrollWheel)
                                   handler:^NSEvent*(NSEvent* e) {
    if (e.window != s_window) return e;
    if (e.type == NSEventTypeMagnify)
      s_pinch_log = netvis::accumulate_pinch_log(s_pinch_log, e.magnification);
    else if (e.type == NSEventTypeScrollWheel)
      s_last_precise = e.hasPreciseScrollingDeltas;
    return e;                                                // never swallow: GLFW still sees scrolls
  }];
}

void netvis::platform_gestures_uninstall() {
  if (s_monitor != nil) {
    [NSEvent removeMonitor:s_monitor];
    s_monitor = nil;
  }
  s_window = nil;
}

float netvis::platform_take_pinch_log() {
  const float v = s_pinch_log;
  s_pinch_log = 0.0f;
  return v;
}

bool netvis::platform_last_scroll_precise() { return s_last_precise; }
