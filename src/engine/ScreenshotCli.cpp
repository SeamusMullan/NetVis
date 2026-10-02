// SPDX-License-Identifier: Apache-2.0
// engine/ScreenshotCli.cpp — see ScreenshotCli.h. Pure rules for `--screenshot`.
#include "engine/ScreenshotCli.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/SafeMath.h"
#include "engine/Bench.h"        // wants_bench
#include "engine/McpServer.h"    // wants_mcp
#include "engine/QueryCli.h"     // wants_query

namespace netvis {

namespace {

constexpr std::string_view kShotFlag = "--screenshot";

// A bounded copy of an argument for an error message (arguments can be 4 KiB).
std::string shown(std::string_view s) {
  constexpr size_t kMax = 64;
  std::string out(s.substr(0, kMax));
  if (s.size() > kMax) out += "...";
  return out;
}

ScreenshotArgs usage_error(std::string msg) {
  ScreenshotArgs r;
  r.status = ScreenshotExit::Usage;
  r.error = std::move(msg);
  return r;
}

// `^[0-9]{1,max_digits}$` -> value. The digit cap comes first, so `v` cannot overflow.
bool parse_digits(std::string_view s, size_t max_digits, uint32_t& v) {
  if (s.empty() || s.size() > max_digits) return false;
  uint32_t acc = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    acc = acc * 10u + static_cast<uint32_t>(c - '0');
  }
  v = acc;
  return true;
}

bool is_value_flag(std::string_view n) {
  return n == "--screenshot" || n == "--size" || n == "--view" || n == "--theme" ||
         n == "--timeout";
}
bool is_bool_flag(std::string_view n) {
  return n == "--canvas-only" || n == "--fit" || n == "--no-layout-cache";
}

// An unpredictable 32-bit value for temp-file names. Never throws: a missing
// entropy source (std::random_device can throw) falls back to the clock and a
// process-wide counter, which still makes consecutive names differ.
uint32_t random_u32() {
  static std::atomic<uint32_t> counter{0};
  uint64_t v = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  v ^= static_cast<uint64_t>(counter.fetch_add(1, std::memory_order_relaxed) + 1) * 0x9E3779B97F4A7C15ull;
  try {
    std::random_device rd;
    v ^= (static_cast<uint64_t>(rd()) << 32) | rd();
  } catch (...) {
  }
  v ^= v >> 33;
  v *= 0xFF51AFD7ED558CCDull;
  v ^= v >> 33;
  return static_cast<uint32_t>(v);
}

}  // namespace

bool wants_screenshot(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == nullptr) continue;
    const std::string_view a = argv[i];
    if (a == kShotFlag) return true;
    if (a.size() > kShotFlag.size() && a.substr(0, kShotFlag.size()) == kShotFlag &&
        a[kShotFlag.size()] == '=')
      return true;
  }
  return false;
}

std::string screenshot_conflict(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == nullptr) continue;
    const std::string_view a = argv[i];
    if (a == "--report" || a.substr(0, 9) == "--report=") return std::string(a);
  }
  // wants_bench scans every argument; wants_query / wants_mcp look at argv[1].
  if (wants_bench(argc, argv)) {
    for (int i = 1; i < argc; ++i) {
      if (argv[i] == nullptr) continue;
      const std::string_view a = argv[i];
      if (a == "--bench" || a.substr(0, 8) == "--bench-") return std::string(a);
    }
  }
  if (wants_query(argc, argv)) return "query";
  if (wants_mcp(argc, argv)) return "mcp";
  return std::string();
}

bool parse_screenshot_size(std::string_view text, uint32_t& width, uint32_t& height) {
  const size_t x = text.find_first_of("xX");
  if (x == std::string_view::npos) return false;
  uint32_t w = 0, h = 0;
  if (!parse_digits(text.substr(0, x), 5, w)) return false;
  if (!parse_digits(text.substr(x + 1), 5, h)) return false;  // a second 'x' is a non-digit
  if (w < kScreenshotMinSide || w > kScreenshotMaxSide) return false;
  if (h < kScreenshotMinSide || h > kScreenshotMaxSide) return false;
  width = w;
  height = h;
  return true;
}

ScreenshotArgs parse_screenshot_args(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (argv[i] == nullptr) return usage_error("null argument");
    if (std::string_view(argv[i]).size() > kScreenshotMaxArgBytes)
      return usage_error("argument " + std::to_string(i) + " is longer than " +
                         std::to_string(kScreenshotMaxArgBytes) + " bytes");
  }

  // --help / -h wins over every other problem (a half-typed command line is exactly
  // when it is asked for), but a model literally named "-h" can still follow `--`.
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--") break;
    if (a == "--help" || a == "-h") {
      ScreenshotArgs h;
      h.help = true;
      return h;
    }
  }

  ScreenshotOptions o;
  bool seen_out = false, seen_size = false, seen_view = false, seen_theme = false,
       seen_timeout = false, seen_canvas = false, seen_fit = false, seen_nocache = false;
  std::vector<std::string> positionals;
  bool only_positionals = false;

  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (only_positionals) {
      positionals.emplace_back(a);
      continue;
    }
    if (a == "--") {
      only_positionals = true;
      continue;
    }
    if (a.empty() || a[0] != '-') {
      positionals.emplace_back(a);
      continue;
    }

    // A flag: `--name`, `--name=value`, or an unknown dash-argument.
    const bool long_flag = a.size() >= 2 && a[1] == '-';
    const size_t eq = long_flag ? a.find('=') : std::string_view::npos;
    const std::string_view name = (eq == std::string_view::npos) ? a : a.substr(0, eq);
    const bool has_eq = eq != std::string_view::npos;

    if (long_flag && is_value_flag(name)) {
      const std::string n(name);
      bool* seen = nullptr;
      if (name == "--screenshot") seen = &seen_out;
      else if (name == "--size") seen = &seen_size;
      else if (name == "--view") seen = &seen_view;
      else if (name == "--theme") seen = &seen_theme;
      else seen = &seen_timeout;
      if (*seen) return usage_error(n + " given twice");
      *seen = true;

      std::string_view value;
      if (has_eq) {
        value = a.substr(eq + 1);
        if (value.empty()) return usage_error(n + " needs a value");
      } else {
        if (i + 1 >= argc) return usage_error(n + " needs a value");
        const std::string_view next = argv[i + 1];
        // A flag where a value belongs (`--view --canvas-only`) is a mistake, not a
        // file named "--canvas-only"; use the `--view=<file>` form for that.
        if (next.empty() || next.substr(0, 2) == "--") return usage_error(n + " needs a value");
        value = next;
        ++i;
      }

      if (name == "--screenshot") {
        o.out_path = std::string(value);
      } else if (name == "--size") {
        if (!parse_screenshot_size(value, o.width, o.height))
          return usage_error("--size must be WxH with each side " +
                             std::to_string(kScreenshotMinSide) + ".." +
                             std::to_string(kScreenshotMaxSide) + ", got '" + shown(value) + "'");
      } else if (name == "--view") {
        o.view_path = std::string(value);
      } else if (name == "--theme") {
        if (value == "dark") o.dark_theme = true;
        else if (value == "light") o.dark_theme = false;
        else return usage_error("--theme must be 'dark' or 'light', got '" + shown(value) + "'");
      } else {  // --timeout
        uint32_t t = 0;
        if (!parse_digits(value, 5, t) || t < 1 || t > kScreenshotMaxTimeoutS)
          return usage_error("--timeout must be 1.." + std::to_string(kScreenshotMaxTimeoutS) +
                             " seconds, got '" + shown(value) + "'");
        o.timeout_s = t;
      }
      continue;
    }

    if (long_flag && is_bool_flag(name)) {
      const std::string n(name);
      if (has_eq) return usage_error(n + " takes no value");
      bool* seen = nullptr;
      if (name == "--canvas-only") seen = &seen_canvas;
      else if (name == "--fit") seen = &seen_fit;
      else seen = &seen_nocache;
      if (*seen) return usage_error(n + " given twice");
      *seen = true;
      if (name == "--canvas-only") o.canvas_only = true;
      else if (name == "--fit") o.fit = true;
      else o.use_layout_cache = false;
      continue;
    }

    return usage_error("unknown argument '" + shown(a) + "' (use -- before a model whose name starts with '-')");
  }

  if (!seen_out) return usage_error("--screenshot <out.png> is required");
  if (positionals.empty()) return usage_error("expected a model file");
  if (positionals.size() > 1)
    return usage_error("expected one model file, got " + std::to_string(positionals.size()));
  if (positionals[0].empty()) return usage_error("the model path is empty");
  o.model_path = positionals[0];

  ScreenshotArgs r;
  r.options = std::move(o);
  return r;
}

ScreenshotArgs resolve_screenshot_paths(ScreenshotOptions options) {
  namespace fs = std::filesystem;
  ScreenshotArgs r;
  r.options = std::move(options);
  ScreenshotOptions& o = r.options;

  auto fail = [&r](ScreenshotExit s, std::string msg) {
    r.status = s;
    r.error = std::move(msg);
    return r;
  };

  // Absolute first: everything below, and the capture itself, must not depend on
  // the working directory (GLFW may chdir inside a macOS bundle).
  std::error_code ec;
  for (std::string* p : {&o.out_path, &o.model_path, &o.view_path}) {
    if (p->empty()) continue;
    const fs::path abs = fs::absolute(fs::path(*p), ec);
    if (ec) return fail(ScreenshotExit::Usage, "cannot resolve path '" + shown(*p) + "': " + ec.message());
    *p = abs.string();
  }

  // --- output (Usage) ---------------------------------------------------------
  {
    const fs::path out(o.out_path);
    ec.clear();
    if (o.out_path.empty()) return fail(ScreenshotExit::Usage, "the output path is empty");
    const fs::path parent = out.parent_path();
    if (!fs::is_directory(parent, ec))
      return fail(ScreenshotExit::Usage,
                  "output directory does not exist: '" + parent.string() + "'");
    std::string why;
    if (!screenshot_output_replaceable(o.out_path, why)) return fail(ScreenshotExit::Usage, why);
    // The file is always a PNG whatever it is called, so `--screenshot a.jpg` would
    // write PNG bytes into a .jpg and fool every tool that trusts the extension.
    std::string ext = out.extension().string();
    for (char& c : ext) {
      if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    if (ext != ".png")
      return fail(ScreenshotExit::Usage,
                  "the output file must end in .png (the capture is always a PNG), got '" +
                      shown(out.filename().string()) + "'");
  }

  // --- model (Load): a directory is fine (.mlpackage, saved_model/) ------------
  ec.clear();
  if (!fs::exists(fs::path(o.model_path), ec))
    return fail(ScreenshotExit::Load, "model not found: '" + o.model_path + "'");

  // --- view (View) -------------------------------------------------------------
  if (!o.view_path.empty()) {
    ec.clear();
    if (!fs::is_regular_file(fs::path(o.view_path), ec))
      return fail(ScreenshotExit::View,
                  "view file not found or not a regular file: '" + o.view_path + "'");
  }
  return r;
}

// See the header. fopen's "x" is O_CREAT|O_EXCL on POSIX (which never opens through a symbolic link, even a
// dangling one) and CREATE_NEW on Windows. The permission bits are the usual
// 0666 & ~umask.
std::FILE* open_new_file_exclusive(const std::filesystem::path& p, int& err) {
  std::FILE* f = nullptr;
#if defined(_WIN32)
  err = static_cast<int>(_wfopen_s(&f, p.c_str(), L"wbx"));
  if (f != nullptr) err = 0;
  else if (err == 0) err = EIO;
#else
  errno = 0;
  f = std::fopen(p.c_str(), "wbx");
  err = (f != nullptr) ? 0 : (errno != 0 ? errno : EIO);
#endif
  return f;
}

bool screenshot_output_replaceable(const std::string& path, std::string& error) {
  namespace fs = std::filesystem;
  const fs::path out(path);
  std::error_code ec;
  const fs::file_status st = fs::status(out, ec);
  if (fs::is_directory(st)) {
    error = "output '" + path + "' is a directory";
    return false;
  }
  if (st.type() == fs::file_type::none) {  // an error other than "not found"
    error = "cannot examine output '" + path + "': " + ec.message();
    return false;
  }
  if (!fs::exists(st)) return true;  // absent, or a dangling link (the link is replaced)
  if (!fs::is_regular_file(st)) {
    error = "refusing to overwrite '" + path + "': it exists and is not a regular file";
    return false;
  }
  // Swapped arguments (`--screenshot model.onnx other.onnx`) must not clobber the
  // model: an existing output is replaced only if it already is a PNG.
  uint8_t head[8] = {};
  size_t got = 0;
  {
    std::ifstream f(out, std::ios::binary);
    if (f) {
      f.read(reinterpret_cast<char*>(head), sizeof head);
      got = static_cast<size_t>(f.gcount());
    }
  }
  if (!is_png_signature(head, got)) {
    error = "refusing to overwrite '" + path + "': it exists and is not a PNG";
    return false;
  }
  return true;
}

bool write_screenshot_png(const std::string& path, const uint8_t* png, size_t len,
                          std::string& error) {
  namespace fs = std::filesystem;
  if (png == nullptr || !is_png_signature(png, len)) {
    error = "internal error: the encoded image is not a PNG";
    return false;
  }
  if (path.empty()) {
    error = "the output path is empty";
    return false;
  }
  // Fail fast, before anything is created.
  if (!screenshot_output_replaceable(path, error)) return false;

  // An unpredictable name, created exclusively (see the header). A name that is
  // already taken (EEXIST: a leftover, or somebody's planted file or link) is simply
  // skipped; any other error is real.
  fs::path tmp;
  std::FILE* f = nullptr;
  int err = 0;
  for (int attempt = 0; attempt < 16 && f == nullptr; ++attempt) {
    char suffix[40];
    std::snprintf(suffix, sizeof suffix, ".%08x.netvis-tmp", static_cast<unsigned>(random_u32()));
    tmp = fs::path(path);
    tmp += suffix;
    f = open_new_file_exclusive(tmp, err);
    if (f == nullptr && err != EEXIST) break;
  }
  if (f == nullptr) {
    error = "cannot create a temporary file next to '" + path +
            "': " + std::error_code(err, std::generic_category()).message();
    return false;
  }

  auto discard_tmp = [&tmp] {
    std::error_code ec;
    fs::remove(tmp, ec);
  };
  const size_t wrote = len == 0 ? 0 : std::fwrite(png, 1, len, f);
  const bool flushed = std::fflush(f) == 0;
  const bool closed = std::fclose(f) == 0;
  if (wrote != len || !flushed || !closed) {
    error = "failed writing '" + tmp.string() + "'";
    discard_tmp();
    return false;
  }

  // The capture may have run for minutes since the first check: look again, as late
  // as possible, so a file somebody created at `path` meanwhile is not clobbered.
  if (!screenshot_output_replaceable(path, error)) {
    discard_tmp();
    return false;
  }
  // rename(2) replaces atomically on POSIX; MSVC's std::filesystem::rename replaces
  // an existing file too. A symbolic link at `path` is replaced, not followed.
  std::error_code ec;
  fs::rename(tmp, fs::path(path), ec);
  if (ec) {
    error = "cannot move the PNG into place at '" + path + "': " + ec.message();
    discard_tmp();
    return false;
  }
  return true;
}

std::string screenshot_display_name(const std::string& model_path) {
  namespace fs = std::filesystem;
  fs::path p(model_path);
  // `bundle.mlpackage/` has an empty filename(); the bundle's name is its parent's.
  if (!p.has_filename() && p.has_relative_path()) p = p.parent_path();
  std::string name = p.filename().string();
  return name.empty() ? model_path : name;
}

CaptureGate capture_gate(const CaptureGateInputs& in) {
  if (in.load_failed) return CaptureGate::LoadFailed;
  if (!in.jobs_idle) return CaptureGate::Wait;
  if (!in.has_model) return CaptureGate::LoadFailed;
  if (in.canvas_only && !in.has_graph) return CaptureGate::NoCanvas;
  if (in.has_graph && !in.has_layout) return CaptureGate::LoadFailed;
  if (in.has_graph && in.layout_empty) return CaptureGate::EmptyGraph;
  return CaptureGate::Ready;
}

bool rgba_bottom_up_to_rgb_top_down(const uint8_t* src, size_t src_len, uint32_t w, uint32_t h,
                                    std::vector<uint8_t>& out) {
  if (src == nullptr || w == 0 || h == 0) return false;
  const uint64_t px = safe_mul(w, h);
  const uint64_t need = safe_mul(px, 4);
  if (px == UINT64_MAX || need == UINT64_MAX) return false;  // saturated
  if (need > SIZE_MAX) return false;                         // 32-bit size_t
  if (src_len < need) return false;
  const size_t row_in = static_cast<size_t>(w) * 4;
  const size_t row_out = static_cast<size_t>(w) * 3;
  try {
    out.resize(static_cast<size_t>(px) * 3);
  } catch (const std::bad_alloc&) {
    return false;
  }
  for (uint32_t y = 0; y < h; ++y) {
    const uint8_t* s = src + static_cast<size_t>(h - 1 - y) * row_in;
    uint8_t* d = out.data() + static_cast<size_t>(y) * row_out;
    for (uint32_t x = 0; x < w; ++x) {
      d[0] = s[0];
      d[1] = s[1];
      d[2] = s[2];
      s += 4;
      d += 3;
    }
  }
  return true;
}

bool is_png_signature(const uint8_t* data, size_t len) {
  static constexpr uint8_t kSig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
  if (data == nullptr || len < sizeof kSig) return false;
  return std::memcmp(data, kSig, sizeof kSig) == 0;
}

}  // namespace netvis
