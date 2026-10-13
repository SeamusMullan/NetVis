// SPDX-License-Identifier: Apache-2.0
// main.cpp — process entry point.
//
// Modes:
//   * GUI (default): create the App, forward an optional CLI path (argv[1]) as
//     the initial file, and run the main loop. All real work lives in App;
//     keeping main tiny means the same App can be driven from a test harness.
//   * Headless report (--report <path>, issue #58): parse + analyze + print JSON
//     to stdout and exit — NO window, NO ImGui. Uses only netvis_core, so no GUI
//     is ever spun up in this path.
//   * Headless benchmark (--bench, issue #97): time the engine hot paths across a
//     synthetic fixture ladder and print JSON to stdout — same headless contract
//     as --report, so CI can gate on it without a display.
//   * Headless query (`netvis query <verb> ...`): the agent-facing analysis CLI —
//     one structural question per invocation, one line of JSON out. Same
//     headless contract; see engine/QueryCli.h and docs/agent-cli.md.
//   * MCP server (`netvis mcp`): serve the same query surface to MCP clients
//     over stdio until EOF. See engine/McpServer.h.
//   * Screenshot (`netvis --screenshot out.png [...] model`, issue #170): open the
//     model through the normal pipeline in a hidden window, wait until it has
//     finished loading, render it offscreen at an exact size, write a PNG and exit.
//     Needs a window system (it is not headless like the modes above), so it lives
//     only in this binary. Checked FIRST; combining it with another mode is refused.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h (via
// App.h) but not included there; pre-include it so App.h compiles.
#include "engine/Bench.h"
#include "engine/LayoutEngine.h"
#include "engine/McpServer.h"
#include "engine/QueryCli.h"
#include "engine/ReportJson.h"
#include "engine/ScreenshotCli.h"
#include "engine/ViewFile.h"
#include "view/App.h"

namespace {

// Recognize `--report <path>` and `--report=<path>`. Returns true and fills
// `path` when the report flag is present; `path` may stay empty (missing arg),
// which we surface as an error rather than silently opening the GUI.
bool parse_report_arg(int argc, char** argv, std::string& path, bool& found) {
  found = false;
  path.clear();
  for (int i = 1; i < argc; ++i) {
    std::string_view a = argv[i];
    if (a == "--report") {
      found = true;
      if (i + 1 < argc) path = argv[i + 1];
      return true;
    }
    constexpr std::string_view kEq = "--report=";
    if (a.substr(0, kEq.size()) == kEq) {
      found = true;
      path = std::string(a.substr(kEq.size()));
      return true;
    }
  }
  return false;
}

// Headless report path: print JSON to stdout, errors to stderr. Returns the
// process exit code (0 ok, 1 on missing arg / parse failure).
int run_report(const std::string& path) {
  if (path.empty()) {
    std::fprintf(stderr, "netvis --report: expected a model file path\n");
    return 1;
  }
  netvis::Result<std::string> json = netvis::report_file(path);
  if (!json) {
    std::fprintf(stderr, "netvis --report: %s\n", json.error().message.c_str());
    return 1;
  }
  std::printf("%s\n", json->c_str());
  return 0;
}

// Headless benchmark path (#97). The flags and their parsing live in
// engine/Bench.h so this binary and the dedicated headless netvis_bench target
// cannot drift apart — CI gates on netvis_bench's numbers, and two flag parsers
// would be two behaviours the gate could silently disagree about.
//
// Prints the harness JSON to stdout and exits. Never creates the App, so it
// needs no display.
int run_bench_cli(int argc, char** argv) {
  const netvis::BenchOptions opt = netvis::parse_bench_args(argc, argv);
  netvis::Result<std::vector<netvis::BenchCase>> cases = netvis::run_bench(opt);
  if (!cases) {
    std::fprintf(stderr, "netvis --bench: %s\n", cases.error().message.c_str());
    return 1;
  }
  std::printf("%s\n", netvis::build_bench_json(*cases).c_str());
  return 0;
}

// `netvis --screenshot` (#170). The pure rules (argument parsing, path validation,
// the capture gate) live in engine/ScreenshotCli.h and are unit-tested; the capture
// itself is App::run_screenshot. Paths are made absolute HERE, before glfwInit,
// because GLFW chdirs into Contents/Resources inside a macOS bundle.
//
// Returns a ScreenshotExit code. stdout stays empty; every failure is one line on
// stderr prefixed "netvis --screenshot: error:".
int run_screenshot_cli(int argc, char** argv) {
  using namespace netvis;
  auto report = [](const ScreenshotArgs& a) {
    std::fprintf(stderr, "netvis --screenshot: error: %s\n", a.error.c_str());
    if (a.status == ScreenshotExit::Usage)
      std::fwrite(kScreenshotUsage.data(), 1, kScreenshotUsage.size(), stderr);
    return exit_code(a.status);
  };

  ScreenshotArgs args = parse_screenshot_args(argc, argv);
  if (args.help) {  // an explicit request for help is the one thing printed on stdout
    std::fwrite(kScreenshotUsage.data(), 1, kScreenshotUsage.size(), stdout);
    return 0;
  }
  if (args.status != ScreenshotExit::Ok) return report(args);
  args = resolve_screenshot_paths(args.options);
  if (args.status != ScreenshotExit::Ok) return report(args);
  const ScreenshotOptions& opts = args.options;

  // The view file is read BEFORE any window exists, so a bad one fails fast.
  ViewFileLoad view;
  const ViewFile* view_ptr = nullptr;
  if (!opts.view_path.empty()) {
    view = read_view_file(opts.view_path);
    if (!view.ok()) {
      std::fprintf(stderr, "netvis --screenshot: error: %s\n", view.error.c_str());
      return exit_code(ScreenshotExit::View);
    }
    for (const std::string& w : view.warnings)
      std::fprintf(stderr, "netvis --screenshot: note: %s\n", w.c_str());
    view_ptr = &view.file;
  }

  netvis::App app;
  return app.run_screenshot(opts, view_ptr);
}

}  // namespace

int main(int argc, char** argv) {
  // #170: checked FIRST. Today `--report` would silently win over a second mode
  // flag; screenshot mode refuses the combination instead of guessing.
  if (netvis::wants_screenshot(argc, argv)) {
    const std::string conflict = netvis::screenshot_conflict(argc, argv);
    if (!conflict.empty()) {
      std::fprintf(stderr, "netvis --screenshot: error: cannot be combined with %s\n",
                   conflict.c_str());
      // A usage error like any other (exit 2), so it prints the usage too.
      std::fwrite(netvis::kScreenshotUsage.data(), 1, netvis::kScreenshotUsage.size(), stderr);
      return netvis::exit_code(netvis::ScreenshotExit::Usage);
    }
    return run_screenshot_cli(argc, argv);
  }

  std::string report_path;
  bool report_mode = false;
  if (parse_report_arg(argc, argv, report_path, report_mode) && report_mode) {
    return run_report(report_path);  // headless: never creates the App
  }
  if (netvis::wants_bench(argc, argv)) {
    return run_bench_cli(argc, argv);  // headless: never creates the App
  }
  if (netvis::wants_query(argc, argv)) {
    return netvis::run_query_cli(argc, argv);  // headless: never creates the App
  }
  if (netvis::wants_mcp(argc, argv)) {
    return netvis::run_mcp_stdio();  // headless: never creates the App
  }

  // GUI path (unchanged): argv[1] is an optional initial file to open.
  netvis::App app;
  std::string path = argc > 1 ? argv[1] : std::string();
  if (!app.init(path)) return 1;
  return app.run();
}
