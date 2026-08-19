// engine/McpServer.cpp — Model Context Protocol server over the query engine.
//
// See McpServer.h for the design. Protocol surface: initialize, ping,
// tools/list, tools/call, and silence for notifications. Tool arguments are
// translated into the exact `netvis query` argument vector and dispatched
// through run_query with the cache-backed model provider, so the MCP tools and
// the CLI verbs are one implementation with two front doors.
#include "engine/McpServer.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

#include <nlohmann/json.hpp>

namespace netvis {

namespace {

using json = nlohmann::ordered_json;

// The protocol revision this server implements. Clients negotiate downward,
// and every mainstream client speaks this revision.
constexpr const char* kProtocolVersion = "2024-11-05";

#ifdef NETVIS_VERSION_STRING
constexpr const char* kServerVersion = NETVIS_VERSION_STRING;
#else
constexpr const char* kServerVersion = "0.0.0";
#endif

// ---------------------------------------------------------------------------
// Tool table. One entry per query verb; the schema and the argument-to-CLI
// translation are generated from the same rows, so they cannot disagree.
// ---------------------------------------------------------------------------
struct ArgDef {
  const char* key;    // MCP argument name
  const char* type;   // "string" | "integer"
  bool required;      // listed in the schema's `required`
  const char* flag;   // nullptr = positional (appended in table order)
  const char* desc;
};

struct ToolDef {
  const char* name;   // MCP tool name
  const char* verb;   // query verb it dispatches to
  const char* desc;
  std::vector<ArgDef> args;
};

const std::vector<ToolDef>& tools() {
  static const std::vector<ToolDef> t = {
      {"netvis_summary", "summary",
       "Summarize a model file: format, graphs, parameter/FLOP totals, "
       "quantization profile and roofline. Reads no tensor payloads.",
       {{"model", "string", true, nullptr, "Path to the model file"}}},
      {"netvis_io", "io",
       "List a graph's declared inputs and outputs with dtype and shape.",
       {{"model", "string", true, nullptr, "Path to the model file"},
        {"graph", "integer", false, "graph", "Graph index (default 0)"}}},
      {"netvis_nodes", "nodes",
       "List nodes with op, category and cost. Filter by op type or name "
       "substring; paged, with an exact total_matched.",
       {{"model", "string", true, nullptr, "Path to the model file"},
        {"op", "string", false, "op", "Exact op type filter (case-insensitive)"},
        {"contains", "string", false, "contains", "Name substring filter"},
        {"limit", "integer", false, "limit", "Rows to return (default 100)"},
        {"offset", "integer", false, "offset", "Rows to skip"},
        {"graph", "integer", false, "graph", "Graph index (default 0)"}}},
      {"netvis_node", "node",
       "Everything about one node: attributes, input/output values with "
       "shapes, per-node cost, direct predecessors and successors. Target is "
       "an exact node name or '#<index>'.",
       {{"model", "string", true, nullptr, "Path to the model file"},
        {"target", "string", true, nullptr, "Exact node name or '#<index>'"},
        {"graph", "integer", false, "graph", "Graph index (default 0)"}}},
      {"netvis_tensors", "tensors",
       "List weight tensors with dtype, shape and size. Sortable by name, "
       "bytes or params; paged. Reads no payloads.",
       {{"model", "string", true, nullptr, "Path to the model file"},
        {"sort", "string", false, "sort", "name | bytes | params"},
        {"limit", "integer", false, "limit", "Rows to return (default 100)"},
        {"offset", "integer", false, "offset", "Rows to skip"},
        {"graph", "integer", false, "graph", "Graph index (default 0)"}}},
      {"netvis_tensor", "tensor",
       "Stream one weight tensor's payload into stats: min/max/mean/std, zero "
       "and NaN/Inf counts, 64-bucket histogram. The only tool that reads "
       "tensor bytes.",
       {{"model", "string", true, nullptr, "Path to the model file"},
        {"name", "string", true, nullptr, "Exact tensor name"},
        {"graph", "integer", false, "graph", "Graph index (default 0)"}}},
      {"netvis_search", "search",
       "Fuzzy and field-scoped search over names, ops, values and tensors. "
       "Supports op:/name:/dtype:/shape:/params: prefixes, '*' globs, and "
       "K/M/G numeric suffixes.",
       {{"model", "string", true, nullptr, "Path to the model file"},
        {"query", "string", true, nullptr, "Search query"},
        {"limit", "integer", false, "limit", "Hits to return (default 50)"}}},
      {"netvis_neighbors", "neighbors",
       "Fan-in/fan-out of a node, N hops out. Target is an exact node name or "
       "'#<index>'.",
       {{"model", "string", true, nullptr, "Path to the model file"},
        {"target", "string", true, nullptr, "Exact node name or '#<index>'"},
        {"dir", "string", false, "dir", "in | out | both (default both)"},
        {"hops", "integer", false, "hops", "Hop count (default 1)"},
        {"cap", "integer", false, "cap", "Max nodes per direction (default 256)"},
        {"graph", "integer", false, "graph", "Graph index (default 0)"}}},
      {"netvis_cost", "cost",
       "Per-node cost ranking: where the FLOPs, parameters or bytes are.",
       {{"model", "string", true, nullptr, "Path to the model file"},
        {"by", "string", false, "by",
         "flops | params | weight_bytes | act_bytes | intensity"},
        {"limit", "integer", false, "limit", "Rows to return (default 25)"},
        {"graph", "integer", false, "graph", "Graph index (default 0)"}}},
      {"netvis_diff", "diff",
       "Compare two models: exact added/removed/changed counts plus capped "
       "change lists.",
       {{"model_a", "string", true, nullptr, "Path to the primary model"},
        {"model_b", "string", true, nullptr, "Path to the comparison model"},
        {"match", "string", false, "match", "name | topology (default name)"},
        {"limit", "integer", false, "limit", "Entries per change list"},
        {"graph", "integer", false, "graph", "Graph index of the primary"}}},
  };
  return t;
}

json tool_schema(const ToolDef& t) {
  json props = json::object();
  json required = json::array();
  for (const ArgDef& a : t.args) {
    json p;
    p["type"] = a.type;
    p["description"] = a.desc;
    props[a.key] = std::move(p);
    if (a.required) required.push_back(a.key);
  }
  json schema;
  schema["type"] = "object";
  schema["properties"] = std::move(props);
  schema["required"] = std::move(required);
  json out;
  out["name"] = t.name;
  out["description"] = t.desc;
  out["inputSchema"] = std::move(schema);
  return out;
}

// Translate MCP arguments into the CLI argument vector, in table order:
// positionals first, then flags. A JSON integer becomes its decimal string.
Result<std::vector<std::string>> tool_args_to_cli(const ToolDef& t, const json& args) {
  std::vector<std::string> cli;
  cli.push_back(t.verb);
  for (const ArgDef& a : t.args) {
    auto it = args.find(a.key);
    if (it == args.end() || it->is_null()) {
      if (a.required) return err(std::string("missing required argument '") + a.key + "'");
      continue;
    }
    std::string value;
    if (it->is_string()) {
      value = it->get<std::string>();
    } else if (it->is_number_integer()) {
      value = std::to_string(it->get<int64_t>());
    } else {
      return err(std::string("argument '") + a.key + "' must be a " + a.type);
    }
    if (a.flag) {
      cli.push_back(std::string("--") + a.flag);
      cli.push_back(std::move(value));
    } else {
      cli.push_back(std::move(value));
    }
  }
  return cli;
}

json rpc_result(const json& id, json result) {
  json j;
  j["jsonrpc"] = "2.0";
  j["id"] = id;
  j["result"] = std::move(result);
  return j;
}

json rpc_error(const json& id, int code, const std::string& message) {
  json j;
  j["jsonrpc"] = "2.0";
  j["id"] = id;
  json e;
  e["code"] = code;
  e["message"] = message;
  j["error"] = std::move(e);
  return j;
}

// Tool-execution failures are results with isError, per the protocol; only
// malformed requests become JSON-RPC errors.
json tool_text_result(const json& id, const std::string& text, bool is_error) {
  json content = json::array();
  json block;
  block["type"] = "text";
  block["text"] = text;
  content.push_back(std::move(block));
  json r;
  r["content"] = std::move(content);
  r["isError"] = is_error;
  return rpc_result(id, std::move(r));
}

}  // namespace

// ---------------------------------------------------------------------------
// ModelCache
// ---------------------------------------------------------------------------
ModelCache::ModelCache(size_t max_models) : max_models_(max_models == 0 ? 1 : max_models) {}

Result<std::shared_ptr<HeadlessModel>> ModelCache::get(const std::string& path) {
  // Revalidation identity: byte size + mtime. Both come from one stat and
  // together catch every ordinary re-export; content hashing would cost a full
  // read, which is exactly what this tool exists to avoid.
  std::error_code ec;
  const uint64_t bytes = std::filesystem::file_size(path, ec);
  int64_t mtime_ns = 0;
  if (!ec) {
    const auto t = std::filesystem::last_write_time(path, ec);
    if (!ec) mtime_ns = static_cast<int64_t>(t.time_since_epoch().count());
  }
  // A stat failure falls through to the loader, whose error message names the
  // real problem (missing file, permissions, bundle resolution).

  for (size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].path != path) continue;
    if (!ec && entries_[i].bytes == bytes && entries_[i].mtime_ns == mtime_ns) {
      ++hits_;
      Entry hit = std::move(entries_[i]);
      entries_.erase(entries_.begin() + static_cast<ptrdiff_t>(i));
      entries_.insert(entries_.begin(), std::move(hit));
      return entries_.front().model;
    }
    // Stale (or unstattable): drop and reload below.
    entries_.erase(entries_.begin() + static_cast<ptrdiff_t>(i));
    break;
  }

  ++misses_;
  Result<HeadlessModel> loaded = load_model_headless(path);
  if (!loaded) return loaded.error();

  Entry e;
  e.path = path;
  e.mtime_ns = mtime_ns;
  e.bytes = ec ? 0 : bytes;
  e.model = std::make_shared<HeadlessModel>(loaded.take());
  entries_.insert(entries_.begin(), std::move(e));
  if (entries_.size() > max_models_) entries_.pop_back();
  return entries_.front().model;
}

// ---------------------------------------------------------------------------
// McpServer
// ---------------------------------------------------------------------------
McpServer::McpServer(size_t cache_models) : cache_(cache_models) {}

std::string McpServer::handle_line(const std::string& line) {
  if (line.empty()) return {};

  json msg = json::parse(line, nullptr, false);
  if (msg.is_discarded() || !msg.is_object()) {
    return rpc_error(nullptr, -32700, "parse error").dump();
  }

  const bool has_id = msg.contains("id") && !msg["id"].is_null();
  const json id = has_id ? msg["id"] : json();
  const std::string method = msg.value("method", "");

  // Notifications (no id) get no response, whatever the method.
  if (!has_id) return {};

  if (method == "initialize") {
    json caps;
    caps["tools"] = json::object();
    json info;
    info["name"] = "netvis";
    info["version"] = kServerVersion;
    json r;
    r["protocolVersion"] = kProtocolVersion;
    r["capabilities"] = std::move(caps);
    r["serverInfo"] = std::move(info);
    r["instructions"] =
        "Query neural-network model files headlessly. Every tool returns one "
        "netvis.query.v1 JSON document as text. Only netvis_tensor reads "
        "tensor payloads; everything else is structural and instant.";
    return rpc_result(id, std::move(r)).dump();
  }

  if (method == "ping") {
    return rpc_result(id, json::object()).dump();
  }

  if (method == "tools/list") {
    json arr = json::array();
    for (const ToolDef& t : tools()) arr.push_back(tool_schema(t));
    json r;
    r["tools"] = std::move(arr);
    return rpc_result(id, std::move(r)).dump();
  }

  if (method == "tools/call") {
    const json params = msg.value("params", json::object());
    const std::string name = params.value("name", "");
    const json args = params.value("arguments", json::object());

    const ToolDef* def = nullptr;
    for (const ToolDef& t : tools())
      if (name == t.name) def = &t;
    if (!def) return rpc_error(id, -32602, "unknown tool: " + name).dump();

    Result<std::vector<std::string>> cli = tool_args_to_cli(*def, args);
    if (!cli) return tool_text_result(id, cli.error().message, true).dump();

    Result<std::string> out = run_query(*cli, [this](const std::string& path) {
      return cache_.get(path);
    });
    if (!out) return tool_text_result(id, out.error().message, true).dump();
    return tool_text_result(id, *out, false).dump();
  }

  return rpc_error(id, -32601, "method not found: " + method).dump();
}

bool wants_mcp(int argc, char** argv) {
  return argc > 1 && std::string_view(argv[1]) == "mcp";
}

int run_mcp_stdio() {
  size_t cache_models = ModelCache::kDefaultMaxModels;
  if (const char* env = std::getenv("NETVIS_MCP_CACHE_MODELS")) {
    const long v = std::strtol(env, nullptr, 10);
    if (v > 0) cache_models = static_cast<size_t>(v);
  }
  McpServer server(cache_models);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const std::string response = server.handle_line(line);
    if (!response.empty()) {
      std::fwrite(response.data(), 1, response.size(), stdout);
      std::fputc('\n', stdout);
      std::fflush(stdout);
    }
  }
  return 0;
}

}  // namespace netvis
