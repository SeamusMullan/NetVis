// main.cpp — process entry point.
//
// Two modes:
//   * GUI (default): create the App, forward an optional CLI path (argv[1]) as
//     the initial file, and run the main loop. All real work lives in App;
//     keeping main tiny means the same App can be driven from a test harness.
//   * Headless report (--report <path>, issue #58): parse + analyze + print JSON
//     to stdout and exit — NO window, NO ImGui. Uses only netvis_core, so no GUI
//     is ever spun up in this path.
//   * Headless benchmark (--bench, issue #97): time the engine hot paths across a
//     synthetic fixture ladder and print JSON to stdout — same headless contract
//     as --report, so CI can gate on it without a display.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h (via
// App.h) but not included there; pre-include it so App.h compiles.
#include "engine/Bench.h"
#include "engine/LayoutEngine.h"
#include "engine/ReportJson.h"
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

}  // namespace

int main(int argc, char** argv) {
  std::string report_path;
  bool report_mode = false;
  if (parse_report_arg(argc, argv, report_path, report_mode) && report_mode) {
    return run_report(report_path);  // headless: never creates the App
  }
  if (netvis::wants_bench(argc, argv)) {
    return run_bench_cli(argc, argv);  // headless: never creates the App
  }

  // GUI path (unchanged): argv[1] is an optional initial file to open.
  netvis::App app;
  std::string path = argc > 1 ? argv[1] : std::string();
  if (!app.init(path)) return 1;
  return app.run();
}
