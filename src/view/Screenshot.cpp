// SPDX-License-Identifier: Apache-2.0
// view/Screenshot.cpp — see Screenshot.h. The offscreen target and the PNG writer.
//
// NO SYSTEM GL HEADER. The types and constants below are declared locally (values
// checked against GLFW's bundled glad header) and every entry point comes from
// glfwGetProcAddress, which on WGL/CGL/GLX/EGL also resolves the GL 1.1 functions
// through GLFW's fallbacks. So this file has no `#if __APPLE__` / `#if _WIN32`
// include dance, only the calling convention below.
#include "view/Screenshot.h"

#include <cstddef>
#include <cstdio>
#include <climits>
#include <filesystem>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <system_error>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "core/SafeMath.h"
#include "engine/ScreenshotCli.h"

// stb_image_write: declarations only. The single STB_IMAGE_WRITE_IMPLEMENTATION is
// compiled in App.cpp.
#include "stb_image_write.h"

namespace netvis {

namespace {

// --- Local GL vocabulary -------------------------------------------------------
using NvGLenum = unsigned int;
using NvGLuint = unsigned int;
using NvGLint = int;
using NvGLsizei = int;
using NvGLbitfield = unsigned int;
using NvGLfloat = float;

#if defined(_WIN32)
#define NETVIS_GLAPI __stdcall
#else
#define NETVIS_GLAPI
#endif

constexpr NvGLenum kGlFramebuffer = 0x8D40;
constexpr NvGLenum kGlRenderbuffer = 0x8D41;
constexpr NvGLenum kGlColorAttachment0 = 0x8CE0;
constexpr NvGLenum kGlFramebufferComplete = 0x8CD5;
constexpr NvGLenum kGlRgba8 = 0x8058;
constexpr NvGLenum kGlRgba = 0x1908;
constexpr NvGLenum kGlUnsignedByte = 0x1401;
constexpr NvGLbitfield kGlColorBufferBit = 0x4000;
constexpr NvGLenum kGlPackAlignment = 0x0D05;
constexpr NvGLenum kGlMaxRenderbufferSize = 0x84E8;
constexpr NvGLenum kGlMaxViewportDims = 0x0D3A;

std::string hex(unsigned v) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "0x%X", v);
  return buf;
}

}  // namespace

// The sixteen entry points the target needs.
struct OffscreenTarget::Gl {
  void(NETVIS_GLAPI* GenFramebuffers)(NvGLsizei, NvGLuint*) = nullptr;
  void(NETVIS_GLAPI* DeleteFramebuffers)(NvGLsizei, const NvGLuint*) = nullptr;
  void(NETVIS_GLAPI* BindFramebuffer)(NvGLenum, NvGLuint) = nullptr;
  void(NETVIS_GLAPI* FramebufferRenderbuffer)(NvGLenum, NvGLenum, NvGLenum, NvGLuint) = nullptr;
  NvGLenum(NETVIS_GLAPI* CheckFramebufferStatus)(NvGLenum) = nullptr;
  void(NETVIS_GLAPI* GenRenderbuffers)(NvGLsizei, NvGLuint*) = nullptr;
  void(NETVIS_GLAPI* DeleteRenderbuffers)(NvGLsizei, const NvGLuint*) = nullptr;
  void(NETVIS_GLAPI* BindRenderbuffer)(NvGLenum, NvGLuint) = nullptr;
  void(NETVIS_GLAPI* RenderbufferStorage)(NvGLenum, NvGLenum, NvGLsizei, NvGLsizei) = nullptr;
  void(NETVIS_GLAPI* Viewport)(NvGLint, NvGLint, NvGLsizei, NvGLsizei) = nullptr;
  void(NETVIS_GLAPI* ClearColor)(NvGLfloat, NvGLfloat, NvGLfloat, NvGLfloat) = nullptr;
  void(NETVIS_GLAPI* Clear)(NvGLbitfield) = nullptr;
  void(NETVIS_GLAPI* ReadPixels)(NvGLint, NvGLint, NvGLsizei, NvGLsizei, NvGLenum, NvGLenum,
                                 void*) = nullptr;
  void(NETVIS_GLAPI* PixelStorei)(NvGLenum, NvGLint) = nullptr;
  void(NETVIS_GLAPI* GetIntegerv)(NvGLenum, NvGLint*) = nullptr;
  NvGLenum(NETVIS_GLAPI* GetError)() = nullptr;

  // Clear stale error flags (ImGui's loader and backend may have left one) so a
  // later non-zero glGetError is really ours. Bounded: a lost context returns the
  // same error forever.
  void drain_errors() const {
    for (int i = 0; i < 16 && GetError() != 0; ++i) {
    }
  }
};

namespace {

template <typename Fn>
bool load_fn(Fn& fn, const char* name, std::string& error) {
  const GLFWglproc p = glfwGetProcAddress(name);
  if (p == nullptr) {
    error = std::string("OpenGL entry point ") + name + " unavailable";
    return false;
  }
  fn = reinterpret_cast<Fn>(p);
  return true;
}

}  // namespace

OffscreenTarget::OffscreenTarget() = default;
OffscreenTarget::~OffscreenTarget() { release(); }

bool OffscreenTarget::init(uint32_t width, uint32_t height) {
  release();
  error_.clear();
  if (width == 0 || height == 0) {
    error_ = "capture size is zero";
    return false;
  }
  if (width > static_cast<uint32_t>(INT_MAX) || height > static_cast<uint32_t>(INT_MAX)) {
    error_ = "capture size is too large";
    return false;
  }

  auto g = std::make_unique<Gl>();
#define NV_LOAD(member) \
  if (!load_fn(g->member, "gl" #member, error_)) return false
  NV_LOAD(GenFramebuffers);
  NV_LOAD(DeleteFramebuffers);
  NV_LOAD(BindFramebuffer);
  NV_LOAD(FramebufferRenderbuffer);
  NV_LOAD(CheckFramebufferStatus);
  NV_LOAD(GenRenderbuffers);
  NV_LOAD(DeleteRenderbuffers);
  NV_LOAD(BindRenderbuffer);
  NV_LOAD(RenderbufferStorage);
  NV_LOAD(Viewport);
  NV_LOAD(ClearColor);
  NV_LOAD(Clear);
  NV_LOAD(ReadPixels);
  NV_LOAD(PixelStorei);
  NV_LOAD(GetIntegerv);
  NV_LOAD(GetError);
#undef NV_LOAD

  // The driver's limits, named in the message so a user knows what to lower.
  NvGLint max_rb = 0;
  NvGLint max_vp[2] = {0, 0};
  g->GetIntegerv(kGlMaxRenderbufferSize, &max_rb);
  g->GetIntegerv(kGlMaxViewportDims, max_vp);
  if (max_rb > 0 && (width > static_cast<uint32_t>(max_rb) || height > static_cast<uint32_t>(max_rb))) {
    error_ = "size " + std::to_string(width) + "x" + std::to_string(height) +
             " exceeds this driver's GL_MAX_RENDERBUFFER_SIZE (" + std::to_string(max_rb) + ")";
    return false;
  }
  if (max_vp[0] > 0 && max_vp[1] > 0 &&
      (width > static_cast<uint32_t>(max_vp[0]) || height > static_cast<uint32_t>(max_vp[1]))) {
    error_ = "size " + std::to_string(width) + "x" + std::to_string(height) +
             " exceeds this driver's GL_MAX_VIEWPORT_DIMS (" + std::to_string(max_vp[0]) + "x" +
             std::to_string(max_vp[1]) + ")";
    return false;
  }

  gl_ = g.release();
  width_ = width;
  height_ = height;

  gl_->drain_errors();
  gl_->GenFramebuffers(1, &fbo_);
  gl_->GenRenderbuffers(1, &rbo_);
  gl_->BindRenderbuffer(kGlRenderbuffer, rbo_);
  gl_->RenderbufferStorage(kGlRenderbuffer, kGlRgba8, static_cast<NvGLsizei>(width),
                           static_cast<NvGLsizei>(height));
  const NvGLenum storage_err = gl_->GetError();
  gl_->BindFramebuffer(kGlFramebuffer, fbo_);
  gl_->FramebufferRenderbuffer(kGlFramebuffer, kGlColorAttachment0, kGlRenderbuffer, rbo_);
  const NvGLenum status = gl_->CheckFramebufferStatus(kGlFramebuffer);
  if (storage_err != 0 || status != kGlFramebufferComplete) {
    error_ = storage_err != 0
                 ? "could not allocate a " + std::to_string(width) + "x" + std::to_string(height) +
                       " render target (GL error " + hex(storage_err) + ")"
                 : "offscreen framebuffer is incomplete (status " + hex(status) + ")";
    release();
    return false;
  }
  return true;
}

void OffscreenTarget::begin_frame(float r, float g, float b, float a) {
  if (gl_ == nullptr) return;
  gl_->BindFramebuffer(kGlFramebuffer, fbo_);
  gl_->Viewport(0, 0, static_cast<NvGLsizei>(width_), static_cast<NvGLsizei>(height_));
  gl_->ClearColor(r, g, b, a);
  gl_->Clear(kGlColorBufferBit);
}

bool OffscreenTarget::read_rgb(std::vector<uint8_t>& out) {
  if (gl_ == nullptr) {
    error_ = "no offscreen target";
    return false;
  }
  const uint64_t px = safe_mul(width_, height_);
  const uint64_t bytes = safe_mul(px, 4);
  if (bytes == UINT64_MAX || bytes > static_cast<uint64_t>(SIZE_MAX)) {
    error_ = "capture size overflows the readback buffer";
    return false;
  }

  std::vector<uint8_t> rgba;
  try {
    rgba.resize(static_cast<size_t>(bytes));
  } catch (const std::bad_alloc&) {
    error_ = "out of memory reading back the capture";
    return false;
  }

  gl_->BindFramebuffer(kGlFramebuffer, fbo_);
  gl_->PixelStorei(kGlPackAlignment, 1);
  gl_->drain_errors();
  gl_->ReadPixels(0, 0, static_cast<NvGLsizei>(width_), static_cast<NvGLsizei>(height_), kGlRgba,
                  kGlUnsignedByte, rgba.data());
  const NvGLenum err = gl_->GetError();
  if (err != 0) {
    error_ = "OpenGL error " + hex(err) + " reading back the capture";
    return false;
  }
  if (!rgba_bottom_up_to_rgb_top_down(rgba.data(), rgba.size(), width_, height_, out)) {
    error_ = "could not convert the capture to RGB";
    return false;
  }
  return true;
}

void OffscreenTarget::release() {
  if (gl_ != nullptr) {
    if (fbo_ != 0) gl_->DeleteFramebuffers(1, &fbo_);
    if (rbo_ != 0) gl_->DeleteRenderbuffers(1, &rbo_);
    delete gl_;
    gl_ = nullptr;
  }
  fbo_ = rbo_ = 0;
  width_ = height_ = 0;
}

// ---------------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------------
bool write_png_rgb(const std::string& path, const std::vector<uint8_t>& rgb, uint32_t width,
                   uint32_t height, std::string& error) {
  namespace fs = std::filesystem;
  if (width == 0 || height == 0 || width > static_cast<uint32_t>(INT_MAX / 3) ||
      height > static_cast<uint32_t>(INT_MAX)) {
    error = "PNG size is out of range";
    return false;
  }
  const uint64_t need = safe_mul(safe_mul(width, height), 3);
  if (need == UINT64_MAX || rgb.size() < need) {
    error = "pixel buffer is smaller than the image";
    return false;
  }

  std::vector<uint8_t> png;
  auto sink = [](void* ctx, void* data, int size) {
    if (size <= 0) return;
    auto* v = static_cast<std::vector<uint8_t>*>(ctx);
    const auto* p = static_cast<const uint8_t*>(data);
    v->insert(v->end(), p, p + size);
  };
  if (stbi_write_png_to_func(sink, &png, static_cast<int>(width), static_cast<int>(height), 3,
                             rgb.data(), static_cast<int>(width) * 3) == 0 ||
      png.empty()) {
    error = "PNG encoding failed";
    return false;
  }

  const std::string tmp = path + ".netvis-tmp";
  auto cleanup = [&tmp] {
    std::error_code ec;
    fs::remove(fs::path(tmp), ec);
  };
  {
    std::ofstream f(fs::path(tmp), std::ios::binary | std::ios::trunc);
    if (!f) {
      error = "cannot create '" + tmp + "'";
      return false;
    }
    f.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    if (!f.good()) {
      error = "failed writing '" + tmp + "'";
      f.close();
      cleanup();
      return false;
    }
    f.close();
    if (f.fail()) {
      error = "failed closing '" + tmp + "'";
      cleanup();
      return false;
    }
  }
  // Atomic on POSIX (rename(2) replaces); MSVC's rename replaces an existing file.
  std::error_code ec;
  fs::rename(fs::path(tmp), fs::path(path), ec);
  if (ec) {
    error = "cannot move the PNG into place at '" + path + "': " + ec.message();
    cleanup();
    return false;
  }
  return true;
}

}  // namespace netvis
