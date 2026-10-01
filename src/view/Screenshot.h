// SPDX-License-Identifier: Apache-2.0
// view/Screenshot.h — the offscreen render target and PNG writer behind
// `netvis --screenshot` (#170).
//
// This is the ONLY file pair in the PR that contains OpenGL calls, and it includes
// no system GL header: it declares the handful of types and constants it needs
// locally and loads the entry points with glfwGetProcAddress. That keeps the
// platform `#if` maze of <OpenGL/gl.h> / <windows.h>+<GL/gl.h> out of the capture
// path entirely, and the same code builds on macOS, Windows and Linux. (ImGui's own
// loader has no framebuffer-object functions, which is why it is not reused.)
//
// The rules around it (argument limits, the capture gate, the RGBA->RGB flip) are
// core and unit-tested in engine/ScreenshotCli.h. What is here cannot be linked into
// netvis_tests, so it is deliberately thin.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace netvis {

// An RGBA8 renderbuffer framebuffer of exactly width x height pixels, whatever the
// window's HiDPI scale. A GL context must be current for every call, including
// destruction (release() before the context goes away).
class OffscreenTarget {
 public:
  OffscreenTarget();
  ~OffscreenTarget();
  OffscreenTarget(const OffscreenTarget&) = delete;
  OffscreenTarget& operator=(const OffscreenTarget&) = delete;

  // Load the GL entry points, check the size against the driver's limits
  // (GL_MAX_RENDERBUFFER_SIZE, GL_MAX_VIEWPORT_DIMS), create the framebuffer and
  // check it is complete. On failure returns false and error() says why; nothing is
  // left allocated.
  bool init(uint32_t width, uint32_t height);

  // Bind the framebuffer, set the viewport to the full target and clear it. Call
  // before each ImGui_ImplOpenGL3_RenderDrawData.
  void begin_frame(float r, float g, float b, float a);

  // Read the target back as top-down 8-bit RGB (width*height*3 bytes). False and
  // error() on a GL error or an impossible size.
  bool read_rgb(std::vector<uint8_t>& out);

  void release();

  const std::string& error() const { return error_; }
  uint32_t width() const { return width_; }
  uint32_t height() const { return height_; }

 private:
  struct Gl;  // the loaded entry points (defined in Screenshot.cpp)
  Gl* gl_ = nullptr;
  uint32_t fbo_ = 0, rbo_ = 0;
  uint32_t width_ = 0, height_ = 0;
  std::string error_;
};

// Encode `rgb` (width*height*3, top-down) as a PNG and write it to `path`
// ATOMICALLY: the bytes go to `<path>.netvis-tmp`, which is then renamed over the
// target, so a failed run never truncates or replaces a good PNG. On failure
// returns false, sets `error`, and removes the temp file.
bool write_png_rgb(const std::string& path, const std::vector<uint8_t>& rgb, uint32_t width,
                   uint32_t height, std::string& error);

}  // namespace netvis
