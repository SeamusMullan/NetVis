// tests/test_mcp.cpp — the MCP server contract.
//
// McpServer::handle_line is the whole protocol surface: one JSON-RPC message
// in, one response line out (or none for notifications). These tests drive the
// real handshake and tool calls against the generated fixtures without a
// process boundary, plus the ModelCache's reuse, eviction and staleness rules.
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

#include "engine/McpServer.h"

using namespace netvis;
using json = nlohmann::json;

namespace {

constexpr const char* kOnnx = "tests/fixtures/model.onnx";

bool have(const char* path) {
  if (std::filesystem::exists(path)) return true;
  WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
  return false;
}

json rpc(McpServer& s, const std::string& line) {
  const std::string out = s.handle_line(line);
  REQUIRE_MESSAGE(!out.empty(), "expected a response for: " << line);
  json j = json::parse(out, nullptr, false);
  REQUIRE(!j.is_discarded());
  CHECK(j["jsonrpc"] == "2.0");
  return j;
}

}  // namespace

TEST_CASE("MCP: initialize handshake, ping, and notification silence") {
  McpServer s;

  json init = rpc(s, R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})");
  CHECK(init["id"] == 1);
  CHECK(init["result"]["protocolVersion"] == "2024-11-05");
  CHECK(init["result"]["serverInfo"]["name"] == "netvis");
  CHECK(init["result"]["capabilities"].contains("tools"));

  // Notifications get no response at all.
  CHECK(s.handle_line(R"({"jsonrpc":"2.0","method":"notifications/initialized"})").empty());

  json ping = rpc(s, R"({"jsonrpc":"2.0","id":2,"method":"ping"})");
  CHECK(ping["result"].is_object());
}

TEST_CASE("MCP: tools/list exposes the whole query surface with schemas") {
  McpServer s;
  json r = rpc(s, R"({"jsonrpc":"2.0","id":1,"method":"tools/list"})");
  const json& tools = r["result"]["tools"];
  CHECK(tools.size() == 10);
  bool found_nodes = false;
  for (const auto& t : tools) {
    CHECK(t.contains("name"));
    CHECK(t.contains("description"));
    CHECK(t["inputSchema"]["type"] == "object");
    if (t["name"] == "netvis_nodes") {
      found_nodes = true;
      CHECK(t["inputSchema"]["required"] == json::array({"model"}));
      CHECK(t["inputSchema"]["properties"].contains("op"));
    }
  }
  CHECK(found_nodes);
}

TEST_CASE("MCP: tools/call dispatches to the query engine") {
  if (!have(kOnnx)) return;
  McpServer s;

  json call = json::parse(R"({"jsonrpc":"2.0","id":7,"method":"tools/call","params":
      {"name":"netvis_nodes","arguments":{"model":"","limit":2}}})");
  call["params"]["arguments"]["model"] = kOnnx;
  json r = rpc(s, call.dump());
  CHECK(r["result"]["isError"] == false);
  json payload = json::parse(r["result"]["content"][0]["text"].get<std::string>());
  CHECK(payload["schema"] == "netvis.query.v1");
  CHECK(payload["verb"] == "nodes");
  CHECK(payload["total_matched"] == 3);
  CHECK(payload["returned"] == 2);  // the integer argument reached --limit
}

TEST_CASE("MCP: failures are loud in the right channel") {
  McpServer s;

  // Tool-execution failures (bad arguments, bad model) are isError results.
  json missing = rpc(s, R"({"jsonrpc":"2.0","id":1,"method":"tools/call","params":
      {"name":"netvis_nodes","arguments":{}}})");
  CHECK(missing["result"]["isError"] == true);

  json bad_model = rpc(s, R"({"jsonrpc":"2.0","id":2,"method":"tools/call","params":
      {"name":"netvis_summary","arguments":{"model":"definitely-missing.onnx"}}})");
  CHECK(bad_model["result"]["isError"] == true);

  // Protocol-level failures are JSON-RPC errors.
  json unknown_tool = rpc(s, R"({"jsonrpc":"2.0","id":3,"method":"tools/call","params":
      {"name":"netvis_frobnicate","arguments":{}}})");
  CHECK(unknown_tool["error"]["code"] == -32602);

  json unknown_method = rpc(s, R"({"jsonrpc":"2.0","id":4,"method":"resources/list"})");
  CHECK(unknown_method["error"]["code"] == -32601);

  json parse_error = rpc(s, "this is not json");
  CHECK(parse_error["error"]["code"] == -32700);
}

TEST_CASE("MCP: the model cache reuses, revalidates and evicts") {
  if (!have(kOnnx)) return;

  // A private mutable copy, so staleness can be exercised without touching the
  // shared fixture other tests read.
  const std::string copy =
      (std::filesystem::temp_directory_path() / "nv_mcp_cache.onnx").string();
  std::filesystem::copy_file(kOnnx, copy,
                             std::filesystem::copy_options::overwrite_existing);

  ModelCache cache(1);
  auto a = cache.get(copy);
  REQUIRE(a);
  auto b = cache.get(copy);
  REQUIRE(b);
  CHECK(a->get() == b->get());  // same parsed model, not a reload
  CHECK(cache.hits() == 1);
  CHECK(cache.misses() == 1);

  // A capacity of one means a second path evicts the first.
  auto other = cache.get(kOnnx);
  REQUIRE(other);
  CHECK(cache.size() == 1);
  auto a2 = cache.get(copy);
  REQUIRE(a2);
  CHECK(cache.misses() == 3);

  // A newer mtime (a re-export writing identical bytes) must invalidate.
  std::filesystem::last_write_time(
      copy, std::filesystem::last_write_time(copy) + std::chrono::seconds(2));
  auto a3 = cache.get(copy);
  REQUIRE(a3);
  CHECK(cache.misses() == 4);

  std::filesystem::remove(copy);
}
