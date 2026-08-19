// engine/McpServer.h — Model Context Protocol server over the query engine.
//
// DECISION: agents that speak MCP get the same ten answers the query CLI
// gives, as tools. The server is a PURE ADAPTER: every tool call is translated
// into the exact argument vector `netvis query` would take and dispatched
// through the same run_query, so the tool surface and the CLI cannot drift.
//
// Transport is stdio (one JSON-RPC 2.0 message per line, responses flushed per
// message, logs to stderr only) because that is how MCP clients launch and own
// their servers: the process lives exactly as long as the client session and
// can never be leaked, orphaned, or double-started — the resource-management
// property a background socket daemon would have to earn with lifecycle code.
// The server core is transport-agnostic (handle_line in, response line out),
// so a socket front-end stays a straightforward later addition.
//
// Resources inside the session are managed by ModelCache: an LRU of parsed
// models keyed by path, capped at a few entries and revalidated by file size
// and mtime, so a burst of queries against one multi-gigabyte file parses it
// once but never serves a stale or unbounded set of models. Tensor payloads
// stay on disk as always; the cache holds structure only.
//
// The GUI binary (`netvis mcp`), the query binary (`netvis_query mcp`) and the
// dedicated netvis_mcp target all start this same loop, so any of them can be
// wired into an agent's server configuration.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/Result.h"
#include "engine/QueryCli.h"

namespace netvis {

// LRU cache of headlessly-loaded models. `get` revalidates an entry against
// the file's current size and mtime, so an overwritten model is reloaded and a
// hit is never stale. Not thread-safe; the stdio loop is single-threaded.
class ModelCache {
 public:
  explicit ModelCache(size_t max_models = kDefaultMaxModels);

  // Default cap. A parsed multi-hundred-thousand-node graph can hold on the
  // order of 100 MB of structure, so the bound is deliberately small; a diff
  // plus a couple of working models fit, and everything else recycles.
  static constexpr size_t kDefaultMaxModels = 4;

  Result<std::shared_ptr<HeadlessModel>> get(const std::string& path);

  size_t size() const { return entries_.size(); }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }

 private:
  struct Entry {
    std::string path;
    int64_t mtime_ns = 0;
    uint64_t bytes = 0;
    std::shared_ptr<HeadlessModel> model;
  };
  std::vector<Entry> entries_;  // front = most recently used; N is tiny
  size_t max_models_;
  uint64_t hits_ = 0, misses_ = 0;
};

// One MCP session: the initialize handshake state plus the model cache.
// handle_line consumes a single JSON-RPC message (one line, no trailing
// newline) and returns the response line, or "" when no response is due (a
// notification, or malformed input that names no id). Never throws.
class McpServer {
 public:
  explicit McpServer(size_t cache_models = ModelCache::kDefaultMaxModels);

  std::string handle_line(const std::string& line);

  ModelCache& cache() { return cache_; }

 private:
  ModelCache cache_;
};

// Run the stdio loop until EOF: read one message per line from stdin, write
// one response per line to stdout (flushed per message). Returns the process
// exit code. The cache cap can be overridden with NETVIS_MCP_CACHE_MODELS.
int run_mcp_stdio();

// True when argv selects MCP mode (`argv[1] == "mcp"`), for the thin
// dispatchers in the GUI and query binaries.
bool wants_mcp(int argc, char** argv);

}  // namespace netvis
