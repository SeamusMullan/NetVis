// SPDX-License-Identifier: Apache-2.0
// view/PlatformGestures_stub.cpp — the no-bridge implementation of
// PlatformGestures.h for every platform except macOS (#158).
//
// Globbed into the GUI target with the other src/view/*.cpp files. On Apple the
// whole body is compiled out and PlatformGestures_mac.mm provides the symbols
// instead, so this translation unit holds only the header's declarations there —
// which is a valid, empty C++ TU.
#include "view/PlatformGestures.h"

#ifndef __APPLE__

namespace netvis {

void platform_gestures_install(GLFWwindow*) {}
void platform_gestures_uninstall() {}
float platform_take_pinch_log() { return 0.0f; }
bool platform_last_scroll_precise() { return false; }

}  // namespace netvis

#endif  // !__APPLE__
