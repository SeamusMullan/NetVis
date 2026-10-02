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
//   * the PNG-or-refuse-to-overwrite check   (is_png_signature, screenshot_output_replaceable)
//   * the atomic, symlink-safe PNG file write (write_screenshot_png)
//   * the name shown in a capture            (screenshot_display_name)
//
// No GL, no ImGui, no exceptions. Hostile input is the norm for a CLI: every
// numeric argument is digit-capped before it is converted, so no conversion can
// overflow, and every argument is length-capped.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
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
// 8192^2 RGBA is ~268 MB of readback; peak (readback + RGB copy + PNG + the
// framebuffer itself) was measured at ~590 MB max RSS.
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
    "              [--no-layout-cache] [--] <model>\n"
    "       netvis --screenshot --help\n";

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
  // `--help` / `-h` was given: print kScreenshotUsage on stdout and exit 0. Set only
  // with status == Ok, and nothing else in `options` is meaningful then.
  bool help = false;
};

// True iff some argv[i] (i >= 1) is `--screenshot` or starts with `--screenshot=`.
// Checked FIRST by main(), so every other invocation is untouched.
bool wants_screenshot(int argc, char** argv);

// Today `--report` silently wins over a second mode flag. Screenshot mode refuses
// the combination instead. Returns the first conflicting selector as written
// (`--report`, `--report=x`, `--bench`, `--bench-quick`, `query`, `mcp`) or "".
std::string screenshot_conflict(int argc, char** argv);

// Parse argv[1..argc). Never throws, never touches the filesystem. On any problem
// status is Usage and `error` names the offending flag. `--help` / `-h` anywhere
// before a `--` wins over every other problem and sets `help`.
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
//     (so swapped arguments cannot clobber the model); the name must end in
//     `.png`, because the file is always a PNG whatever it is called.   -> Usage
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

// --- The output file -----------------------------------------------------------

// May a capture write to `path`? The ONE rule behind both the up-front check in
// resolve_screenshot_paths and the re-check write_screenshot_png makes just before
// it moves the PNG into place (a capture can run for up to --timeout seconds, and
// the file system is not frozen meanwhile):
//   * a directory                      -> no
//   * exists, not a regular file       -> no
//   * exists, regular, not a PNG       -> no  (swapped arguments, a model, notes)
//   * absent (or a dangling symlink)   -> yes
//   * exists and is a PNG              -> yes
// A symbolic link is judged by what it points at, as before; replacing it replaces
// the LINK, never the file it points at (rename(2), see write_screenshot_png).
// On "no", `error` is one line naming the path and the reason.
bool screenshot_output_replaceable(const std::string& path, std::string& error);

// Create `p` for writing (binary), failing with errno EEXIST if ANYTHING is already
// there: a file, a directory, or a symbolic link (live or dangling), which is never
// opened through. fopen "wbx": O_CREAT|O_EXCL on POSIX, CREATE_NEW on Windows; the
// permission bits are the usual 0666 & ~umask. Returns the open FILE* (the caller
// fclose()s it) or nullptr with `err` set to the errno value. This is the step that
// makes write_screenshot_png's temp file safe in a shared directory; it is public so
// that can be tested directly.
std::FILE* open_new_file_exclusive(const std::filesystem::path& p, int& err);

// Write `png` (a complete PNG file: must start with the PNG signature) to `path`
// ATOMICALLY and without following a planted link:
//   * the bytes go to a temp file in the SAME directory with an unpredictable name
//     (`<path>.<8 hex digits>.netvis-tmp`), created EXCLUSIVELY (fopen "wbx":
//     O_CREAT|O_EXCL on POSIX, which never opens through a symbolic link;
//     CREATE_NEW on Windows), so another user cannot make the capture truncate or
//     overwrite some other file by pre-creating the temp name as a symlink;
//   * `path` is re-checked with screenshot_output_replaceable just before the
//     rename, which then replaces it in one step, so a failed run never truncates
//     or replaces a good PNG and never leaves a partial file;
//   * on any failure the temp file is removed, `path` is untouched, and `error`
//     says why.
// Never throws.
bool write_screenshot_png(const std::string& path, const uint8_t* png, size_t len,
                          std::string& error);

// What a capture shows for a model path: the final path component (a trailing
// separator, as on a `.mlpackage/` bundle, is ignored). The full absolute path is
// used to LOAD the model, but never drawn: it leaks the user's name and directory
// layout into pictures meant for public docs, and it makes the pixels depend on
// where the repository happens to be checked out.
std::string screenshot_display_name(const std::string& model_path);

}  // namespace netvis
