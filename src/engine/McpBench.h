// engine/McpBench.h — footprint harness for the MCP server.
//
// DECISION: the server's performance claims follow the same rule as the
// engine's (README "Performance"): numbers come from a reproducible harness,
// not from a one-off measurement. `netvis_bench --bench-mcp` generates a
// deterministic ladder of synthetic chain models AS FILES (the cache is keyed
// by path and revalidated by stat, so an in-memory graph would bypass the very
// layer under test), then measures through the real handle_line entry:
//
//   * protocol overhead   — ping and tools/list, microseconds, no model;
//   * cold tool call      — first query against a path (load + analyze);
//   * warm tool calls     — repeat queries per verb family (nodes / search /
//                           cost / neighbors), the steady state an agent
//                           session actually lives in;
//   * session footprint   — current-RSS deltas after filling the cache to its
//                           cap and after touching 2x cap distinct models,
//                           demonstrating the LRU bound in bytes.
//
// Output is one JSON document ("netvis.mcpbench.v1") to stdout, same contract
// as the engine harness. Wall-clock numbers do not transfer between machines;
// compare runs from the same machine only.
#pragma once

#include <string>

#include "core/Result.h"

namespace netvis {

// True when argv selects the MCP footprint harness (`--bench-mcp`).
bool wants_mcp_bench(int argc, char** argv);

// Run the harness and return the JSON document. Writes its synthetic models
// under the system temp directory and removes them before returning.
Result<std::string> run_mcp_bench();

}  // namespace netvis
