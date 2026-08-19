// engine/QueryCli.h — headless agent-facing query CLI.
//
// DECISION: `netvis query <verb> <model> ...` answers one structural question
// about a model as a single line of JSON on stdout, then exits. It exists so
// automated tooling (coding agents, scripts, CI) can interrogate a model the
// way the GUI does — nodes, tensors, shapes, costs, connectivity, search —
// without a window, a display, or a long-lived process.
//
// Stateless by design: the whole point of the zero-payload thesis is that
// opening a model costs milliseconds (mmap + structural parse), so every
// invocation simply re-opens the file. No daemon, no session, no cache to
// invalidate; an agent composes queries as ordinary subprocess calls and each
// answer is reproducible from the file alone.
//
// Every verb except `tensor` upholds the zero-payload rule (asserted by the
// test suite via the ByteReader counter). `tensor` is the one deliberate
// exception: it streams a payload through the same compute_tensor_stats path
// the weight inspector uses, because "what is in this weight" is exactly the
// question a debugging agent needs answered.
//
// ---------------------------------------------------------------------------
//  ENVELOPE ("netvis.query.v1")
// ---------------------------------------------------------------------------
// Every success prints ONE compact JSON object to stdout:
//   { "schema": "netvis.query.v1", "verb": "<verb>", "model": "<path>", ... }
// Errors go to stderr as plain text with exit code 1; stdout stays silent, so
// a consumer never has to parse a half-written object. Verb payloads are
// documented in docs/agent-cli.md next to worked examples.
//
// Verbs:
//   summary   <model>                     — the netvis.report.v1 report, wrapped
//   io        <model> [--graph N]         — graph inputs/outputs w/ dtype+shape
//   nodes     <model> [--graph N] [--op T] [--contains S] [--limit N] [--offset N]
//   node      <model> <name|#index> [--graph N]
//   tensors   <model> [--graph N] [--sort bytes|params|name] [--limit N] [--offset N]
//   tensor    <model> <name> [--graph N]  — payload stats (the ONE payload reader)
//   search    <model> <query> [--limit N] — fuzzy + field query (op:/name:/dtype:/
//                                           shape:/params:), same as the GUI bar
//   neighbors <model> <name|#index> [--graph N] [--dir in|out|both] [--hops N] [--cap N]
//   cost      <model> [--graph N] [--by flops|params|weight_bytes|act_bytes|intensity]
//                     [--limit N]         — the analyzer's per-node ranking
//   diff      <model-a> <model-b> [--graph N] [--match name|topology] [--limit N]
//
// Lives in netvis_core (no view/ImGui deps) so it runs in ctest and anywhere a
// display is unavailable, exactly like ReportJson and Bench.
#pragma once

#include <string>
#include <vector>

#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"

namespace netvis {

// Schema identifier for every query response. Bump when any verb's payload
// shape changes so consumers can gate on it.
inline constexpr const char* kQuerySchema = "netvis.query.v1";

// A model opened through the same synchronous pipeline the GUI worker drives:
// resolve bundle path -> mmap -> parse -> ONNX shape inference on the main
// graph. `model_dir` is the directory of the mapped file, for resolving ONNX
// external data when a payload is (explicitly) read.
struct HeadlessModel {
  MappedFile file;
  ir::Model model;
  std::string display_path;  // what the caller asked to open
  std::string model_dir;     // directory of the mapped file
};

// Open `path` headlessly. Shared by the report CLI and every query verb so the
// two can never disagree about how a model is loaded. Returns an Error Result
// (message suitable for stderr) on a map/parse failure.
Result<HeadlessModel> load_model_headless(const std::string& path);

// One parsed query invocation. `args` are the raw tokens after `query`.
// run_query resolves the verb, loads the model, and builds the JSON response.
// PURE aside from file I/O; never throws; no window.
Result<std::string> run_query(const std::vector<std::string>& args);

// Process entry for `netvis query ...`: prints run_query's JSON (plus newline)
// to stdout, or the error to stderr. Returns the process exit code (0 ok,
// 1 on any error, including an unknown verb or missing argument).
int run_query_cli(int argc, char** argv);

// True when argv selects query mode (`argv[1] == "query"`). Kept next to the
// other headless-mode predicates so main.cpp stays a thin dispatcher.
bool wants_query(int argc, char** argv);

}  // namespace netvis
