// bench_main.cpp — entry point for the headless benchmark CLI (#97).
//
// Links netvis_core ONLY: no GLFW, no OpenGL, no ImGui, no window. That is the
// whole reason this target exists separately from the GUI binary — CI runs the
// perf gate on every push and PR, and making it build the full graphics stack to
// time a benchmark that never draws anything would cost minutes per run for no
// benefit. The GUI binary keeps its own --bench flags for local convenience;
// both share engine/Bench.h's parser so the two cannot drift.
#include <cstdio>
#include <string>
#include <vector>

#include "engine/Bench.h"

namespace {

void print_usage() {
  std::fprintf(stderr,
               "netvis_bench — headless performance harness (#97)\n"
               "\n"
               "  --bench                 run the default 1k/10k/100k ladder\n"
               "  --bench-quick           skip the 100k rung (the CI signal)\n"
               "  --bench-repeats=N       median-of-N (default 5)\n"
               "  --bench-model=<path>    also time a real model file "
               "(repeatable)\n"
               "\n"
               "Writes the harness JSON to stdout. Compare it against a "
               "committed\nbaseline with tools/bench_gate.py.\n");
}

}  // namespace

int main(int argc, char** argv) {
  // Running with no flags at all is a usage error rather than an implicit full
  // ladder: the 100k rung is expensive, and a bare invocation is far more likely
  // to be a mistake than a request for the slowest possible run.
  if (!netvis::wants_bench(argc, argv)) {
    print_usage();
    return 2;
  }

  const netvis::BenchOptions opt = netvis::parse_bench_args(argc, argv);
  netvis::Result<std::vector<netvis::BenchCase>> cases = netvis::run_bench(opt);
  if (!cases) {
    std::fprintf(stderr, "netvis_bench: %s\n", cases.error().message.c_str());
    return 1;
  }
  std::printf("%s\n", netvis::build_bench_json(*cases).c_str());
  return 0;
}
