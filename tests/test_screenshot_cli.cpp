// SPDX-License-Identifier: Apache-2.0
// tests/test_screenshot_cli.cpp — the pure rules behind `netvis --screenshot` (#170).
//
// The capture itself needs a window and OpenGL, so netvis_tests (which links
// netvis_core only) cannot run it. Everything it DECIDES is in
// engine/ScreenshotCli.h, and is pinned down here: argument parsing and limits,
// path validation, the capture gate, the pixel flip/convert, the PNG-signature
// guard. See also test_screenshot_gate.cpp (the gate against a real session).
#include <doctest/doctest.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "engine/ScreenshotCli.h"

using namespace netvis;

namespace {

namespace fs = std::filesystem;

// Owns the strings behind a C argv. args[0] is the program name, as in main().
struct Argv {
  std::vector<std::string> store;
  std::vector<char*> ptrs;
  explicit Argv(std::vector<std::string> a) : store(std::move(a)) {
    for (std::string& s : store) ptrs.push_back(s.data());
    ptrs.push_back(nullptr);
  }
  int argc() const { return static_cast<int>(store.size()); }
  char** argv() { return ptrs.data(); }
};

ScreenshotArgs parse(std::vector<std::string> args) {
  args.insert(args.begin(), "netvis");
  Argv a(std::move(args));
  return parse_screenshot_args(a.argc(), a.argv());
}

bool wants(std::vector<std::string> args) {
  args.insert(args.begin(), "netvis");
  Argv a(std::move(args));
  return wants_screenshot(a.argc(), a.argv());
}

std::string conflict(std::vector<std::string> args) {
  args.insert(args.begin(), "netvis");
  Argv a(std::move(args));
  return screenshot_conflict(a.argc(), a.argv());
}

bool has(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

// A scratch directory that is removed even when a CHECK fails halfway.
struct TempDir {
  fs::path path;
  explicit TempDir(const char* name) {
    path = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove_all(path, ec);
    fs::create_directories(path, ec);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

// Restores the working directory on scope exit.
struct CwdGuard {
  fs::path saved = fs::current_path();
  ~CwdGuard() {
    std::error_code ec;
    fs::current_path(saved, ec);
  }
};

void write_file(const fs::path& p, const std::string& bytes) {
  std::ofstream f(p, std::ios::binary);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

const std::string kPngHead = std::string("\x89PNG\r\n\x1a\n", 8);

}  // namespace

TEST_CASE("wants_screenshot: recognises --screenshot in both forms, anywhere") {
  CHECK(wants({"--screenshot", "x.png", "m.onnx"}));
  CHECK(wants({"--screenshot=x.png", "m.onnx"}));
  CHECK(wants({"m.onnx", "--screenshot", "x.png"}));
  CHECK_FALSE(wants({"--screenshots", "x.png"}));
  CHECK_FALSE(wants({"screenshot", "m.onnx"}));
  CHECK_FALSE(wants({"m.onnx"}));
  CHECK_FALSE(wants_screenshot(0, nullptr));
}

TEST_CASE("parse_screenshot_args: a minimal invocation uses the documented defaults") {
  const ScreenshotArgs r = parse({"--screenshot", "out.png", "model.onnx"});
  REQUIRE(r.status == ScreenshotExit::Ok);
  CHECK(r.error.empty());
  CHECK(r.options.out_path == "out.png");
  CHECK(r.options.model_path == "model.onnx");
  CHECK(r.options.view_path.empty());
  CHECK(r.options.width == 1600);
  CHECK(r.options.height == 1000);
  CHECK_FALSE(r.options.canvas_only);
  CHECK_FALSE(r.options.fit);
  CHECK(r.options.dark_theme);
  CHECK(r.options.timeout_s == 120);
  CHECK(r.options.use_layout_cache);
}

TEST_CASE("parse_screenshot_args: every flag works as `--k v` and as `--k=v`") {
  for (const bool eq : {false, true}) {
    auto kv = [eq](const std::string& k, const std::string& v) {
      return eq ? std::vector<std::string>{k + "=" + v} : std::vector<std::string>{k, v};
    };
    std::vector<std::string> args;
    auto add = [&args](std::vector<std::string> more) {
      args.insert(args.end(), more.begin(), more.end());
    };
    add(kv("--screenshot", "o.png"));
    add(kv("--size", "800x600"));
    add(kv("--view", "v.netvis-view"));
    add(kv("--theme", "light"));
    add(kv("--timeout", "30"));
    add({"--canvas-only", "--fit", "--no-layout-cache", "m.onnx"});

    const ScreenshotArgs r = parse(args);
    INFO("eq form: " << eq << "  error: " << r.error);
    REQUIRE(r.status == ScreenshotExit::Ok);
    CHECK(r.options.out_path == "o.png");
    CHECK(r.options.width == 800);
    CHECK(r.options.height == 600);
    CHECK(r.options.view_path == "v.netvis-view");
    CHECK_FALSE(r.options.dark_theme);
    CHECK(r.options.timeout_s == 30);
    CHECK(r.options.canvas_only);
    CHECK(r.options.fit);
    CHECK_FALSE(r.options.use_layout_cache);
    CHECK(r.options.model_path == "m.onnx");
  }
}

TEST_CASE("parse_screenshot_size: accepts WxH inside [64, 8192], rejects everything else") {
  uint32_t w = 0, h = 0;
  CHECK(parse_screenshot_size("1600x1000", w, h));
  CHECK((w == 1600 && h == 1000));
  CHECK(parse_screenshot_size("1600X1000", w, h));
  CHECK((w == 1600 && h == 1000));
  CHECK(parse_screenshot_size("64x64", w, h));
  CHECK((w == 64 && h == 64));
  CHECK(parse_screenshot_size("8192x8192", w, h));
  CHECK((w == 8192 && h == 8192));

  // Rejections leave the outputs alone.
  w = 7;
  h = 9;
  for (const char* bad : {"63x100", "100x63", "8193x100", "100x8193", "0x0", "x100", "100x",
                          "100x100x1", "+100x100", " 100x100", "100 x100", "1e3x100", "-5x10",
                          "123456x1", "100x123456", ""}) {
    INFO("size: '" << bad << "'");
    CHECK_FALSE(parse_screenshot_size(bad, w, h));
  }
  CHECK((w == 7 && h == 9));
}

TEST_CASE("parse_screenshot_args: --timeout is 1..3600 whole seconds") {
  for (const char* good : {"1", "3600", "120"}) {
    INFO("timeout: " << good);
    CHECK(parse({"--screenshot", "o.png", "--timeout", good, "m"}).status == ScreenshotExit::Ok);
  }
  CHECK(parse({"--screenshot", "o.png", "--timeout", "3600", "m"}).options.timeout_s == 3600);
  for (const char* bad : {"0", "3601", "-1", "1.5", "abc", "99999999999", "123456", "+5"}) {
    INFO("timeout: " << bad);
    const ScreenshotArgs r = parse({"--screenshot", "o.png", std::string("--timeout=") + bad, "m"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--timeout"));
  }
}

TEST_CASE("parse_screenshot_args: --theme is exactly dark or light") {
  CHECK(parse({"--screenshot", "o.png", "--theme", "dark", "m"}).options.dark_theme);
  CHECK_FALSE(parse({"--screenshot", "o.png", "--theme", "light", "m"}).options.dark_theme);
  for (const char* bad : {"Dark", "blue", "LIGHT", "dark "}) {
    INFO("theme: '" << bad << "'");
    const ScreenshotArgs r = parse({"--screenshot", "o.png", std::string("--theme=") + bad, "m"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--theme"));
  }
  // `--theme=` (empty) is a missing value.
  const ScreenshotArgs empty = parse({"--screenshot", "o.png", "--theme=", "m"});
  CHECK(empty.status == ScreenshotExit::Usage);
  CHECK(has(empty.error, "--theme"));
}

TEST_CASE("parse_screenshot_args: bad flag use is Usage and names the flag") {
  {
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "--size", "100x100", "--size", "200x200", "m"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--size"));
    CHECK(has(r.error, "twice"));
  }
  {
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "--frobnicate", "m"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--frobnicate"));
  }
  {
    // A flag where a value belongs.
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "--view", "--canvas-only", "m"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--view"));
  }
  {
    // Trailing value flag with no value.
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "m", "--size"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--size"));
  }
  {
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "--fit=1", "m"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--fit"));
  }
  {
    const ScreenshotArgs r = parse({"--screenshot=", "m"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--screenshot"));
  }
  {
    // A duplicate boolean is also a mistake.
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "--fit", "--fit", "m"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--fit"));
  }
  {
    // --screenshot itself is mandatory even if parse is called anyway.
    const ScreenshotArgs r = parse({"m.onnx"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "--screenshot"));
  }
  {
    // A single-dash argument is unknown, not a positional.
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "-x", "m"});
    CHECK(r.status == ScreenshotExit::Usage);
  }
}

TEST_CASE("parse_screenshot_args: exactly one positional model") {
  {
    const ScreenshotArgs r = parse({"--screenshot", "o.png"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "expected a model file"));
  }
  {
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "a.onnx", "b.onnx"});
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "got 2"));
  }
  {
    // `--` ends flag parsing: a model whose name starts with '-' is accepted.
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "--", "-weird.onnx"});
    REQUIRE(r.status == ScreenshotExit::Ok);
    CHECK(r.options.model_path == "-weird.onnx");
  }
  {
    // After `--` even a flag-looking name is positional.
    const ScreenshotArgs r = parse({"--screenshot", "o.png", "--", "--fit"});
    REQUIRE(r.status == ScreenshotExit::Ok);
    CHECK(r.options.model_path == "--fit");
    CHECK_FALSE(r.options.fit);
  }
  {
    const ScreenshotArgs r = parse({"--screenshot", "o.png", ""});
    CHECK(r.status == ScreenshotExit::Usage);
  }
}

TEST_CASE("parse_screenshot_args: an over-long argument is Usage") {
  const std::string ok(kScreenshotMaxArgBytes, 'a');
  const std::string too_long(kScreenshotMaxArgBytes + 1, 'a');
  CHECK(parse({"--screenshot", "o.png", ok}).status == ScreenshotExit::Ok);
  CHECK(parse({"--screenshot", "o.png", too_long}).status == ScreenshotExit::Usage);
  CHECK(parse({"--screenshot", too_long, "m"}).status == ScreenshotExit::Usage);
  // The error message must stay bounded (it is echoed to stderr).
  const ScreenshotArgs r = parse({"--screenshot", "o.png", "--bogus" + too_long});
  CHECK(r.status == ScreenshotExit::Usage);
}

TEST_CASE("screenshot_conflict: another mode selector is refused, a plain capture is not") {
  CHECK(conflict({"--screenshot", "x.png", "m.onnx"}) == "");
  CHECK(conflict({"--report", "m.onnx", "--screenshot", "x.png"}) == "--report");
  CHECK(conflict({"--report=m.onnx", "--screenshot", "x.png"}) == "--report=m.onnx");
  CHECK(conflict({"--screenshot", "x.png", "--bench"}) == "--bench");
  CHECK(conflict({"--screenshot", "x.png", "--bench-quick"}) == "--bench-quick");
  CHECK(conflict({"query", "summary", "m.onnx", "--screenshot", "x.png"}) == "query");
  CHECK(conflict({"mcp", "--screenshot", "x.png"}) == "mcp");
  // `query` / `mcp` only select a mode as argv[1]; as a file name they do not.
  CHECK(conflict({"--screenshot", "x.png", "query"}) == "");
}

TEST_CASE("resolve_screenshot_paths: validates output, model and view before any window exists") {
  TempDir td("nv170_resolve");
  const fs::path dir = td.path;
  const fs::path model = dir / "m.onnx";
  write_file(model, "model-bytes");
  const fs::path view = dir / "v.netvis-view";
  write_file(view, "{}");

  auto opts = [&](const fs::path& out) {
    ScreenshotOptions o;
    o.out_path = out.string();
    o.model_path = model.string();
    return o;
  };

  SUBCASE("relative paths become absolute") {
    CwdGuard guard;
    fs::current_path(dir);
    ScreenshotOptions o;
    o.out_path = "out.png";
    o.model_path = "m.onnx";
    o.view_path = "v.netvis-view";
    const ScreenshotArgs r = resolve_screenshot_paths(o);
    REQUIRE(r.status == ScreenshotExit::Ok);
    CHECK(fs::path(r.options.out_path).is_absolute());
    CHECK(fs::path(r.options.model_path).is_absolute());
    CHECK(fs::path(r.options.view_path).is_absolute());
    CHECK(fs::path(r.options.out_path).filename() == "out.png");
  }

  SUBCASE("a fresh output in an existing directory is fine") {
    CHECK(resolve_screenshot_paths(opts(dir / "fresh.png")).status == ScreenshotExit::Ok);
  }

  SUBCASE("an output in a missing directory is Usage") {
    const ScreenshotArgs r = resolve_screenshot_paths(opts(dir / "no" / "such" / "x.png"));
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK_FALSE(r.error.empty());
  }

  SUBCASE("an output that is a directory is Usage") {
    const fs::path d = dir / "adir";
    fs::create_directories(d);
    CHECK(resolve_screenshot_paths(opts(d)).status == ScreenshotExit::Usage);
  }

  SUBCASE("an existing non-PNG output is refused and left untouched") {
    const fs::path out = dir / "precious.txt";
    write_file(out, "hello");
    const ScreenshotArgs r = resolve_screenshot_paths(opts(out));
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, "not a PNG"));
    CHECK(read_file(out) == "hello");
  }

  SUBCASE("swapped arguments cannot clobber the model") {
    // `--screenshot m.onnx other.onnx`: the output IS the model.
    const ScreenshotArgs r = resolve_screenshot_paths(opts(model));
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(read_file(model) == "model-bytes");
  }

  SUBCASE("an existing PNG output is replaceable") {
    const fs::path out = dir / "old.png";
    write_file(out, kPngHead + "junk-after-the-signature");
    CHECK(resolve_screenshot_paths(opts(out)).status == ScreenshotExit::Ok);
  }

  SUBCASE("a 0-byte or short existing output is not a PNG") {
    const fs::path empty = dir / "empty.png";
    write_file(empty, "");
    CHECK(resolve_screenshot_paths(opts(empty)).status == ScreenshotExit::Usage);
    const fs::path shorty = dir / "short.png";
    write_file(shorty, kPngHead.substr(0, 7));
    CHECK(resolve_screenshot_paths(opts(shorty)).status == ScreenshotExit::Usage);
  }

  SUBCASE("a missing model is Load") {
    ScreenshotOptions o = opts(dir / "o.png");
    o.model_path = (dir / "nope.onnx").string();
    const ScreenshotArgs r = resolve_screenshot_paths(o);
    CHECK(r.status == ScreenshotExit::Load);
    CHECK_FALSE(r.error.empty());
  }

  SUBCASE("an existing directory model is allowed (.mlpackage, saved_model/)") {
    const fs::path d = dir / "bundle.mlpackage";
    fs::create_directories(d);
    ScreenshotOptions o = opts(dir / "o.png");
    o.model_path = d.string();
    CHECK(resolve_screenshot_paths(o).status == ScreenshotExit::Ok);
  }

  SUBCASE("view: present is fine, missing is View, a directory is View") {
    ScreenshotOptions o = opts(dir / "o.png");
    o.view_path = view.string();
    CHECK(resolve_screenshot_paths(o).status == ScreenshotExit::Ok);
    o.view_path = (dir / "gone.netvis-view").string();
    CHECK(resolve_screenshot_paths(o).status == ScreenshotExit::View);
    o.view_path = dir.string();
    CHECK(resolve_screenshot_paths(o).status == ScreenshotExit::View);
  }

  SUBCASE("a usage problem is reported before a model problem") {
    ScreenshotOptions o = opts(dir / "no" / "x.png");
    o.model_path = (dir / "nope.onnx").string();
    CHECK(resolve_screenshot_paths(o).status == ScreenshotExit::Usage);
  }
}

namespace {

CaptureGateInputs ready_inputs() {
  CaptureGateInputs in{};
  in.load_failed = false;
  in.jobs_idle = true;
  in.has_model = true;
  in.has_graph = true;
  in.has_layout = true;
  in.layout_empty = false;
  in.canvas_only = false;
  return in;
}

// Visit all 2^6 combinations of the non-load_failed inputs.
template <typename F>
void for_every_combination(F&& f) {
  for (unsigned bits = 0; bits < 64; ++bits) {
    CaptureGateInputs in{};
    in.load_failed = false;
    in.jobs_idle = bits & 1u;
    in.has_model = bits & 2u;
    in.has_graph = bits & 4u;
    in.has_layout = bits & 8u;
    in.layout_empty = bits & 16u;
    in.canvas_only = bits & 32u;
    f(in);
  }
}

}  // namespace

TEST_CASE("capture_gate: Ready needs an idle pool, a model, a graph and a non-empty layout") {
  CHECK(capture_gate(ready_inputs()) == CaptureGate::Ready);

  CaptureGateInputs c = ready_inputs();
  c.canvas_only = true;
  CHECK(capture_gate(c) == CaptureGate::Ready);
}

TEST_CASE("capture_gate: load_failed beats every other input") {
  for_every_combination([](CaptureGateInputs in) {
    in.load_failed = true;
    CHECK(capture_gate(in) == CaptureGate::LoadFailed);
  });
}

TEST_CASE("capture_gate: a busy pool waits, even when a layout is already present") {
  for_every_combination([](CaptureGateInputs in) {
    in.jobs_idle = false;
    CHECK(capture_gate(in) == CaptureGate::Wait);
  });
  CaptureGateInputs in = ready_inputs();
  in.jobs_idle = false;
  CHECK(in.has_layout);
  CHECK(capture_gate(in) == CaptureGate::Wait);
}

TEST_CASE("capture_gate: idle but no model is a failed load (defensive)") {
  CaptureGateInputs in = ready_inputs();
  in.has_model = false;
  CHECK(capture_gate(in) == CaptureGate::LoadFailed);
  // Rule 3 precedes rule 4: no model AND canvas-only is LoadFailed, not NoCanvas.
  in.has_graph = false;
  in.has_layout = false;
  in.canvas_only = true;
  CHECK(capture_gate(in) == CaptureGate::LoadFailed);
}

TEST_CASE("capture_gate: weights-only model") {
  CaptureGateInputs in = ready_inputs();
  in.has_graph = false;
  in.has_layout = false;
  in.canvas_only = true;
  CHECK(capture_gate(in) == CaptureGate::NoCanvas);   // nothing to draw on a canvas
  in.canvas_only = false;
  CHECK(capture_gate(in) == CaptureGate::Ready);      // the tensor table is the picture
}

TEST_CASE("capture_gate: graph without a layout (defensive) and an empty graph") {
  CaptureGateInputs in = ready_inputs();
  in.has_layout = false;
  CHECK(capture_gate(in) == CaptureGate::LoadFailed);

  in = ready_inputs();
  in.layout_empty = true;
  CHECK(capture_gate(in) == CaptureGate::EmptyGraph);
  in.canvas_only = true;
  CHECK(capture_gate(in) == CaptureGate::EmptyGraph);
}

TEST_CASE("rgba_bottom_up_to_rgb_top_down: flips rows and drops alpha") {
  // glReadPixels order: row 0 is the BOTTOM of the picture.
  //   bottom row: A B C        top row: D E F
  const std::vector<uint8_t> src = {
      /* A */ 1, 2, 3, 255,   /* B */ 4, 5, 6, 254,   /* C */ 7, 8, 9, 253,
      /* D */ 10, 11, 12, 252, /* E */ 13, 14, 15, 251, /* F */ 16, 17, 18, 250,
  };
  std::vector<uint8_t> out;
  REQUIRE(rgba_bottom_up_to_rgb_top_down(src.data(), src.size(), 3, 2, out));
  const std::vector<uint8_t> expect = {
      10, 11, 12, 13, 14, 15, 16, 17, 18,   // top row first: D E F
      1,  2,  3,  4,  5,  6,  7,  8,  9,    // then the bottom row: A B C
  };
  CHECK(out == expect);
  CHECK(out.size() == 18);

  // A longer source is fine; only w*h*4 bytes are read.
  std::vector<uint8_t> longer = src;
  longer.push_back(99);
  std::vector<uint8_t> out2;
  REQUIRE(rgba_bottom_up_to_rgb_top_down(longer.data(), longer.size(), 3, 2, out2));
  CHECK(out2 == expect);
}

TEST_CASE("rgba_bottom_up_to_rgb_top_down: refuses bad sizes and leaves out untouched") {
  const std::vector<uint8_t> src(24, 7);
  const std::vector<uint8_t> sentinel = {1, 2, 3};
  std::vector<uint8_t> out = sentinel;

  CHECK_FALSE(rgba_bottom_up_to_rgb_top_down(src.data(), src.size(), 0, 2, out));
  CHECK(out == sentinel);
  CHECK_FALSE(rgba_bottom_up_to_rgb_top_down(src.data(), src.size(), 3, 0, out));
  CHECK(out == sentinel);
  // One byte short of w*h*4.
  CHECK_FALSE(rgba_bottom_up_to_rgb_top_down(src.data(), 3 * 2 * 4 - 1, 3, 2, out));
  CHECK(out == sentinel);
  // w*h*4 saturates: must refuse, not wrap to a small number and over-read.
  CHECK_FALSE(rgba_bottom_up_to_rgb_top_down(src.data(), src.size(), UINT32_MAX, UINT32_MAX, out));
  CHECK(out == sentinel);
  CHECK_FALSE(rgba_bottom_up_to_rgb_top_down(nullptr, 0, 1, 1, out));
  CHECK(out == sentinel);
}

TEST_CASE("is_png_signature: exactly the 8-byte PNG signature") {
  const uint8_t sig[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
  CHECK(is_png_signature(sig, 8));
  CHECK(is_png_signature(sig, 8 + 100));          // more bytes after it is fine
  CHECK_FALSE(is_png_signature(sig, 7));
  uint8_t flipped[8];
  for (int i = 0; i < 8; ++i) flipped[i] = sig[i];
  flipped[3] ^= 0x01;
  CHECK_FALSE(is_png_signature(flipped, 8));
  CHECK_FALSE(is_png_signature(nullptr, 0));
  CHECK_FALSE(is_png_signature(nullptr, 8));
}

// ---------------------------------------------------------------------------
// --help, the output extension, the display name (review fixes for #178)
// ---------------------------------------------------------------------------
TEST_CASE("parse_screenshot_args: --help and -h ask for the usage, and win over other problems") {
  for (const char* h : {"--help", "-h"}) {
    {
      const ScreenshotArgs r = parse({"--screenshot", h});
      CHECK(r.help);
      CHECK(r.status == ScreenshotExit::Ok);
      CHECK(r.error.empty());
    }
    {
      // A half-typed command line is exactly when help is asked for.
      const ScreenshotArgs r = parse({"--screenshot", "o.png", "--size", "bogus", h});
      CHECK(r.help);
      CHECK(r.status == ScreenshotExit::Ok);
    }
    {
      // After `--` it is a (strange) model name, not a request for help.
      const ScreenshotArgs r = parse({"--screenshot", "o.png", "--", h});
      CHECK_FALSE(r.help);
      REQUIRE(r.status == ScreenshotExit::Ok);
      CHECK(r.options.model_path == h);
    }
  }
  // An ordinary invocation does not ask for help.
  CHECK_FALSE(parse({"--screenshot", "o.png", "m.onnx"}).help);
  CHECK_FALSE(parse({"--screenshot", "o.png"}).help);  // a usage error, not help
  CHECK(kScreenshotUsage.find("--help") != std::string_view::npos);
}

TEST_CASE("resolve_screenshot_paths: the output must end in .png") {
  TempDir td("nv170_resolve_ext");
  const fs::path model = td.path / "m.onnx";
  write_file(model, "model-bytes");
  auto resolve = [&](const std::string& name) {
    ScreenshotOptions o;
    o.out_path = (td.path / name).string();
    o.model_path = model.string();
    return resolve_screenshot_paths(o);
  };

  CHECK(resolve("a.png").status == ScreenshotExit::Ok);
  CHECK(resolve("a.PNG").status == ScreenshotExit::Ok);  // case-insensitive
  CHECK(resolve("a.b.png").status == ScreenshotExit::Ok);
  for (const char* bad : {"a.jpg", "a.jpeg", "a.svg", "a", "a.png.txt", ".png2"}) {
    const ScreenshotArgs r = resolve(bad);
    INFO(bad);
    CHECK(r.status == ScreenshotExit::Usage);
    CHECK(has(r.error, ".png"));
    // Nothing was created by the check.
    CHECK_FALSE(fs::exists(td.path / bad));
  }
  // An existing non-PNG is still reported as that (the more specific message wins).
  write_file(td.path / "keep.jpg", "hello");
  const ScreenshotArgs keep = resolve("keep.jpg");
  CHECK(keep.status == ScreenshotExit::Usage);
  CHECK(has(keep.error, "not a PNG"));
  CHECK(read_file(td.path / "keep.jpg") == "hello");
}

TEST_CASE("screenshot_display_name: the last path component, never the directory") {
  CHECK(screenshot_display_name("/Users/someone/work/repo/tests/fixtures/model.onnx") == "model.onnx");
  CHECK(screenshot_display_name("model.onnx") == "model.onnx");
  CHECK(screenshot_display_name("./a/../b/model.onnx") == "model.onnx");
  // A bundle directory with a trailing separator has an empty filename(): the
  // bundle's own name is still what is meant.
  CHECK(screenshot_display_name("/x/y/bundle.mlpackage/") == "bundle.mlpackage");
  CHECK(screenshot_display_name("/x/y/saved_model/") == "saved_model");
  // Nothing to take a name from: the string itself, not an empty label.
  CHECK(screenshot_display_name("/") == "/");
  CHECK(screenshot_display_name("").empty());
  // No separator, however long, leaks through.
  const std::string shown = screenshot_display_name("/home/alice/secret-project/m.gguf");
  CHECK_FALSE(has(shown, "alice"));
  CHECK_FALSE(has(shown, "/"));
}

// ---------------------------------------------------------------------------
// The output file (F2/F9 of the review): atomic, symlink-safe, never clobbers
// ---------------------------------------------------------------------------
namespace {

std::vector<uint8_t> png_bytes(const std::string& tail = "pixels") {
  const std::string s = kPngHead + tail;
  return std::vector<uint8_t>(s.begin(), s.end());
}

bool write_png(const fs::path& p, const std::vector<uint8_t>& b, std::string& err) {
  return write_screenshot_png(p.string(), b.data(), b.size(), err);
}

// Every directory entry whose name carries the temp marker.
std::vector<std::string> temp_leftovers(const fs::path& dir) {
  std::vector<std::string> out;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    const std::string n = e.path().filename().string();
    if (n.find("netvis-tmp") != std::string::npos) out.push_back(n);
  }
  return out;
}

size_t entry_count(const fs::path& dir) {
  size_t n = 0;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(dir, ec)) {
    (void)e;
    ++n;
  }
  return n;
}

}  // namespace

TEST_CASE("write_screenshot_png: a new PNG is written whole and no temp file is left") {
  TempDir td("nv170_pngwrite_new");
  const fs::path out = td.path / "shot.png";
  const std::vector<uint8_t> b = png_bytes("hello-pixels");
  std::string err;
  REQUIRE(write_png(out, b, err));
  CHECK(err.empty());
  CHECK(read_file(out) == std::string(b.begin(), b.end()));
  CHECK(temp_leftovers(td.path).empty());
  CHECK(entry_count(td.path) == 1);
}

TEST_CASE("write_screenshot_png: an existing PNG is replaced") {
  TempDir td("nv170_pngwrite_replace");
  const fs::path out = td.path / "shot.png";
  write_file(out, kPngHead + "old-image");
  const std::vector<uint8_t> b = png_bytes("new-image");
  std::string err;
  REQUIRE(write_png(out, b, err));
  CHECK(read_file(out) == std::string(b.begin(), b.end()));
  CHECK(temp_leftovers(td.path).empty());
}

TEST_CASE("write_screenshot_png: an existing non-PNG is untouched and nothing is left behind") {
  TempDir td("nv170_pngwrite_nonpng");
  const fs::path out = td.path / "model.onnx";
  write_file(out, "precious model bytes");
  std::string err;
  CHECK_FALSE(write_png(out, png_bytes(), err));
  CHECK(has(err, "not a PNG"));
  CHECK(read_file(out) == "precious model bytes");
  CHECK(temp_leftovers(td.path).empty());
  CHECK(entry_count(td.path) == 1);
}

TEST_CASE("write_screenshot_png: a directory at the output, or a missing directory, is a failure") {
  TempDir td("nv170_pngwrite_bad");
  std::string err;

  const fs::path d = td.path / "adir";
  fs::create_directories(d);
  CHECK_FALSE(write_png(d, png_bytes(), err));
  CHECK(has(err, "directory"));
  CHECK(entry_count(d) == 0);

  err.clear();
  const fs::path missing = td.path / "no" / "such" / "dir" / "x.png";
  CHECK_FALSE(write_png(missing, png_bytes(), err));
  CHECK_FALSE(err.empty());
  CHECK_FALSE(fs::exists(td.path / "no"));  // nothing was created on the way
  CHECK(temp_leftovers(td.path).empty());
}

TEST_CASE("write_screenshot_png: bytes that are not a PNG are refused before any file is made") {
  TempDir td("nv170_pngwrite_notpng");
  const fs::path out = td.path / "shot.png";
  std::string err;
  const std::vector<uint8_t> junk = {'n', 'o', 't', ' ', 'a', ' ', 'p', 'n', 'g', '!'};
  CHECK_FALSE(write_png(out, junk, err));
  CHECK_FALSE(err.empty());
  CHECK_FALSE(write_screenshot_png(out.string(), nullptr, 0, err));
  const std::vector<uint8_t> short_sig(kPngHead.begin(), kPngHead.begin() + 7);
  CHECK_FALSE(write_png(out, short_sig, err));
  CHECK_FALSE(fs::exists(out));
  CHECK(entry_count(td.path) == 0);
  CHECK_FALSE(write_screenshot_png("", png_bytes().data(), png_bytes().size(), err));
}

TEST_CASE("write_screenshot_png: a file created at the output after the first check is not clobbered") {
  // The capture can run for minutes between resolve_screenshot_paths' check and the
  // write, and the file system is not frozen meanwhile. The writer applies the SAME rule
  // itself (screenshot_output_replaceable), so a non-PNG that appears at the output after
  // the caller's own check is never replaced.
  TempDir td("nv170_pngwrite_late");
  const fs::path out = td.path / "shot.png";
  std::string err;
  CHECK(screenshot_output_replaceable(out.string(), err));  // absent: fine at check time
  write_file(out, "someone else's notes");                  // ... then it appears
  CHECK_FALSE(write_png(out, png_bytes(), err));
  CHECK(has(err, "not a PNG"));
  CHECK(read_file(out) == "someone else's notes");
  CHECK(temp_leftovers(td.path).empty());
}

TEST_CASE("open_new_file_exclusive: creates only what is not there, and never opens through a link") {
  // This is the step that makes the writer's temp file safe in a shared directory: it
  // must fail, not truncate or follow, when anything already has the name.
  TempDir td("nv170_exclusive");
  int err = 0;

  // A fresh name is created, empty, and writable.
  const fs::path fresh = td.path / "fresh.tmp";
  std::FILE* f = open_new_file_exclusive(fresh, err);
  REQUIRE(f != nullptr);
  CHECK(err == 0);
  CHECK(std::fputs("abc", f) >= 0);
  CHECK(std::fclose(f) == 0);
  CHECK(read_file(fresh) == "abc");

  // The same name again: refused, and the first file is untouched (not truncated).
  err = 0;
  CHECK(open_new_file_exclusive(fresh, err) == nullptr);
  CHECK(err == EEXIST);
  CHECK(read_file(fresh) == "abc");

  // A directory at the name.
  const fs::path dir = td.path / "adir";
  fs::create_directories(dir);
  err = 0;
  CHECK(open_new_file_exclusive(dir, err) == nullptr);
  CHECK(err != 0);

  // A missing parent directory is a different error, not EEXIST.
  err = 0;
  CHECK(open_new_file_exclusive(td.path / "no" / "such" / "x.tmp", err) == nullptr);
  CHECK(err != 0);
  CHECK(err != EEXIST);
  CHECK_FALSE(fs::exists(td.path / "no"));

#if !defined(_WIN32)
  // A live symlink and a dangling one: both refused as "exists", and nothing behind
  // either is created or truncated.
  const fs::path victim = td.path / "victim.txt";
  write_file(victim, "precious\n");
  std::error_code ec;
  fs::create_symlink(victim, td.path / "live.tmp", ec);
  REQUIRE_FALSE(ec);
  err = 0;
  CHECK(open_new_file_exclusive(td.path / "live.tmp", err) == nullptr);
  CHECK(err == EEXIST);
  CHECK(read_file(victim) == "precious\n");

  const fs::path will_be_created = td.path / "created-through-link.txt";
  fs::create_symlink(will_be_created, td.path / "dangling.tmp", ec);
  REQUIRE_FALSE(ec);
  err = 0;
  CHECK(open_new_file_exclusive(td.path / "dangling.tmp", err) == nullptr);
  CHECK(err == EEXIST);
  CHECK_FALSE(fs::exists(will_be_created));  // the link's target was NOT created
#endif
}

TEST_CASE("screenshot_output_replaceable: the one rule") {
  TempDir td("nv170_replaceable");
  std::string err;
  CHECK(screenshot_output_replaceable((td.path / "absent.png").string(), err));
  write_file(td.path / "old.png", kPngHead + "x");
  CHECK(screenshot_output_replaceable((td.path / "old.png").string(), err));
  write_file(td.path / "text.png", "not really");
  CHECK_FALSE(screenshot_output_replaceable((td.path / "text.png").string(), err));
  CHECK(has(err, "not a PNG"));
  fs::create_directories(td.path / "d.png");
  CHECK_FALSE(screenshot_output_replaceable((td.path / "d.png").string(), err));
  CHECK(has(err, "directory"));
}

#if !defined(_WIN32)
// Symbolic links need no privilege on POSIX; on Windows creating one does, and the
// attack below (a planted link at a predictable name in a shared temp directory) is
// the POSIX one, so these two cases are POSIX-only by design, not skipped for want
// of a feature.
TEST_CASE("write_screenshot_png: a symlink planted at the old fixed temp name is not followed") {
  // Before the fix the writer used the fixed name `<out>.netvis-tmp` and opened it
  // through a symlink, truncating whatever the link pointed at and then renaming over
  // the output. A local attacker could aim it at the victim's ~/.zshrc.
  TempDir td("nv170_pngwrite_symlink_tmp");
  const fs::path victim = td.path / "victim.txt";
  write_file(victim, "precious\n");
  const fs::path out = td.path / "y.png";
  std::error_code ec;
  fs::create_symlink(victim, td.path / "y.png.netvis-tmp", ec);
  REQUIRE_FALSE(ec);

  std::string err;
  REQUIRE(write_png(out, png_bytes("real-capture"), err));
  CHECK(read_file(victim) == "precious\n");  // not a PNG, not truncated
  CHECK(read_file(out) == kPngHead + "real-capture");
  // The planted link is still there, untouched (it is not ours to remove).
  CHECK(fs::is_symlink(td.path / "y.png.netvis-tmp"));
}

TEST_CASE("write_screenshot_png: an output that is a symlink to a PNG is replaced, not followed") {
  TempDir td("nv170_pngwrite_symlink_out");
  const fs::path real = td.path / "real.png";
  write_file(real, kPngHead + "the-old-image");
  const fs::path out = td.path / "latest.png";
  std::error_code ec;
  fs::create_symlink(real, out, ec);
  REQUIRE_FALSE(ec);

  std::string err;
  REQUIRE(write_png(out, png_bytes("the-new-image"), err));
  // rename(2) replaced the LINK; the file it pointed at keeps its bytes.
  CHECK_FALSE(fs::is_symlink(out));
  CHECK(read_file(out) == kPngHead + "the-new-image");
  CHECK(read_file(real) == kPngHead + "the-old-image");
}

TEST_CASE("write_screenshot_png: an output symlink to a non-PNG is refused and its target untouched") {
  TempDir td("nv170_pngwrite_symlink_nonpng");
  const fs::path victim = td.path / "victim.txt";
  write_file(victim, "precious\n");
  const fs::path out = td.path / "shot.png";
  std::error_code ec;
  fs::create_symlink(victim, out, ec);
  REQUIRE_FALSE(ec);

  std::string err;
  CHECK_FALSE(write_png(out, png_bytes(), err));
  CHECK(read_file(victim) == "precious\n");
  CHECK(fs::is_symlink(out));
  CHECK(temp_leftovers(td.path).empty());
}
#endif
