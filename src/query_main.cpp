// query_main.cpp — entry point for the headless query CLI.
//
// Links netvis_core ONLY: no GLFW, no OpenGL, no ImGui, no window. The GUI
// binary also dispatches `netvis query ...` for local convenience, but the
// audience of this CLI is automated tooling on machines that often have no
// display stack at all — CI runners, remote debug boxes, agent sandboxes. Both
// entries call engine/QueryCli.h's run_query_cli, so the two cannot drift.
//
// Accepts the verb either bare (`netvis_query nodes model.onnx`) or behind the
// GUI binary's `query` word (`netvis_query query nodes model.onnx`), so a
// script can swap one binary name for the other without editing arguments.
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "engine/QueryCli.h"

int main(int argc, char** argv) {
  std::vector<std::string> args;
  int start = 1;
  if (argc > 1 && std::string_view(argv[1]) == "query") start = 2;
  for (int i = start; i < argc; ++i) args.emplace_back(argv[i]);

  netvis::Result<std::string> out = netvis::run_query(args);
  if (!out) {
    std::fprintf(stderr, "netvis query: %s\n", out.error().message.c_str());
    return 1;
  }
  std::printf("%s\n", out->c_str());
  return 0;
}
