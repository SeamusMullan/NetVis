// SPDX-License-Identifier: Apache-2.0
// engine/ScreenshotCli.h — the pure, testable half of `netvis --screenshot` (#170).
//
// The capture itself (hidden window, offscreen FBO, PNG encode) needs OpenGL and
// ImGui, so it lives in the GUI target (view/AppScreenshot.cpp, view/Screenshot.cpp).
// Every RULE that capture depends on is factored out here, into netvis_core, so
// headless ctest can pin it down:
//   * argument parsing and limits            (parse_screenshot_args, parse_screenshot_size)
//   * output/model/view path validation      (resolve_screenshot_paths)
//   * the "is the model finished?" decision  (capture_gate)
//   * the pixel flip/convert                 (rgba_bottom_up_to_rgb_top_down)
//   * the PNG-or-refuse-to-overwrite check   (is_png_signature)
//
// No GL, no ImGui, no exceptions. Hostile input is the norm for a CLI: every
// numeric argument is digit-capped before it is converted, so no conversion can
// overflow, and every argument is length-capped.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace netvis {

// Process exit codes. Each failure class has its own so scripts can branch on it;
// the reason is always on stderr as one line prefixed "netvis --screenshot: error:".
enum class ScreenshotExit : int {
  Ok = 0,
  Usage = 2,      // bad/missing arguments, conflicting mode, bad output path
  Load = 3,       // model missing / failed to open, parse or lay out
  View = 4,       // bad view file
  Timeout = 5,    // --timeout expired
  Graphics = 6,   // window / OpenGL / framebuffer failure
  Write = 7,      // PNG could not be written
};

constexpr int exit_code(ScreenshotExit e) { return static_cast<int>(e); }

inline constexpr uint32_t kScreenshotDefaultWidth = 1600;
inline constexpr uint32_t kScreenshotDefaultHeight = 1000;
inline constexpr uint32_t kScreenshotMinSide = 64;
// 8192^2 RGBA is ~268 MB of readback; peak (readback + RGB copy + PNG) is ~450 MB.
inline constexpr uint32_t kScreenshotMaxSide = 8192;
inline constexpr uint32_t kScreenshotDefaultTimeoutS = 120;
inline constexpr uint32_t kScreenshotMaxTimeoutS = 3600;
// Frames rendered (at a fixed DeltaTime) before the capture is read back, and the
// number of extra rounds allowed if a panel schedules background work.
inline constexpr int kScreenshotSettleFrames = 4;
inline constexpr int kScreenshotMaxSettleRounds = 3;
inline constexpr float kScreenshotFrameDt = 1.0f / 60.0f;
inline constexpr size_t kScreenshotMaxArgBytes = 4096;

inline constexpr std::string_view kScreenshotUsage =
    "usage: netvis --screenshot <out.png> [--size WxH] [--view <file.netvis-view>]\n"
    "              [--canvas-only] [--fit] [--theme dark|light] [--timeout <seconds>]\n"
    "              [--no-layout-cache] [--] <model>\n";

struct ScreenshotOptions {
  // Absolute after resolve_screenshot_paths(); view_path "" = no view file.
  std::string out_path, model_path, view_path;
  uint32_t width = kScreenshotDefaultWidth, height = kScreenshotDefaultHeight;
  bool canvas_only = false;    // only the graph canvas (+ minimap), no chrome
  bool fit = false;            // fit the graph even when the view file stores a camera
  bool dark_theme = true;      // --theme
  uint32_t timeout_s = kScreenshotDefaultTimeoutS;
  bool use_layout_cache = true;  // --no-layout-cache clears this
};

struct ScreenshotArgs {
  ScreenshotOptions options;
  ScreenshotExit status = ScreenshotExit::Ok;
  std::string error;  // one line, no prefix; empty when status == Ok
};

// True iff some argv[i] (i >= 1) is `--screenshot` or starts with `--screenshot=`.
// Checked FIRST by main(), so every other invocation is untouched.
bool wants_screenshot(int argc, char** argv);

// Today `--report` silently wins over a second mode flag. Screenshot mode refuses
// the combination instead. Returns the first conflicting selector as written
// (`--report`, `--report=x`, `--bench`, `--bench-quick`, `query`, `mcp`) or "".
std::string screenshot_conflict(int argc, char** argv);

// Parse argv[1..argc). Never throws, never touches the filesystem. On any problem
// status is Usage and `error` names the offending flag.
ScreenshotArgs parse_screenshot_args(int argc, char** argv);

// `WxH` / `WXH`: 1..5 ASCII digits each, no sign/space/exponent, each side in
// [kScreenshotMinSide, kScreenshotMaxSide]. The digit count is capped BEFORE
// conversion, so the conversion cannot overflow.
bool parse_screenshot_size(std::string_view text, uint32_t& width, uint32_t& height);

// Make every path absolute and validate it against the filesystem (error_code
// overloads only; never throws). MUST run before glfwInit: GLFW chdirs into
// Contents/Resources inside a macOS bundle, which would break relative paths.
//   * output: parent must exist and be a directory; the output must not be a
//     directory; an existing output is replaced only if it already is a PNG
//     (so swapped arguments cannot clobber the model).                  -> Usage
//   * model:  must exist (a directory is fine: .mlpackage, saved_model/). -> Load
//   * view:   must exist and be a regular file.                           -> View
ScreenshotArgs resolve_screenshot_paths(ScreenshotOptions options);

// --- The capture gate ----------------------------------------------------------
//
// "Is the model finished loading, and is there something to draw?" decided from
// pool quiescence, not from LoadStage labels (which have been wrong before: the
// Enriching race this PR fixes). LoadStage matters only as `load_failed`.
enum class CaptureGate : uint8_t {
  Wait,         // work still queued/running/undrained: keep pumping
  Ready,        // capture
  LoadFailed,   // the load failed (or ended without a model/layout)
  NoCanvas,     // --canvas-only on a weights-only model
  EmptyGraph,   // a graph with zero display nodes
};

struct CaptureGateInputs {
  bool load_failed;  // session.stage() == LoadStage::Failed
  bool jobs_idle;    // tab JobSystem::idle(), sampled AFTER session.update()
  bool has_model;    // session.model() != nullptr
  bool has_graph;    // session.has_graph()
  bool has_layout;   // session.layout() != nullptr
  bool layout_empty; // layout has no boxes (meaningful only if has_layout)
  bool canvas_only;
};

// Decision order (first match wins):
//   1. load_failed                 -> LoadFailed
//   2. !jobs_idle                  -> Wait
//   3. !has_model                  -> LoadFailed   (defensive: idle and not failed
//                                                   means a model must exist)
//   4. canvas_only && !has_graph   -> NoCanvas
//   5. has_graph && !has_layout    -> LoadFailed   (defensive)
//   6. has_graph && layout_empty   -> EmptyGraph
//   7. otherwise                   -> Ready
CaptureGate capture_gate(const CaptureGateInputs& in);

// --- Pixels --------------------------------------------------------------------

// glReadPixels returns bottom-up RGBA; a PNG wants top-down RGB. Writes row h-1-y
// of `src` to row y of `out` and drops alpha (ImGui keeps destination alpha at 1
// in practice; dropping it removes any chance of a semi-transparent PNG).
// Returns false and leaves `out` untouched when w or h is 0, when w*h*4 saturates,
// or when src_len < w*h*4.
bool rgba_bottom_up_to_rgb_top_down(const uint8_t* src, size_t src_len,
                                    uint32_t w, uint32_t h, std::vector<uint8_t>& out);

// True iff the first 8 bytes are the PNG signature (89 50 4E 47 0D 0A 1A 0A).
bool is_png_signature(const uint8_t* data, size_t len);

}  // namespace netvis
