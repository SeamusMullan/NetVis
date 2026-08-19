// engine/McpBench.cpp — footprint harness for the MCP server.
//
// See McpBench.h for what is measured and why. The synthetic models are ONNX
// chain graphs hand-encoded on the protobuf wire (the same approach the test
// fixture generator takes), written to temp files so the path-keyed,
// stat-revalidated ModelCache is exercised for real.
#include "engine/McpBench.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/Rss.h"
#include "engine/McpServer.h"

namespace netvis {

namespace {

using json = nlohmann::ordered_json;
using clock_ = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Minimal ONNX wire encoding — just enough for a deterministic chain graph.
// ---------------------------------------------------------------------------
void put_varint(std::string& b, uint64_t v) {
  while (v >= 0x80) {
    b.push_back(static_cast<char>((v & 0x7f) | 0x80));
    v >>= 7;
  }
  b.push_back(static_cast<char>(v));
}

void put_tag(std::string& b, uint32_t field, uint32_t wire) {
  put_varint(b, (static_cast<uint64_t>(field) << 3) | wire);
}

void put_str(std::string& b, uint32_t field, std::string_view s) {
  put_tag(b, field, 2);
  put_varint(b, s.size());
  b.append(s);
}

void put_msg(std::string& b, uint32_t field, const std::string& m) {
  put_tag(b, field, 2);
  put_varint(b, m.size());
  b.append(m);
}

// NodeProto: input(1) output(2) name(3) op_type(4).
std::string chain_node(uint32_t i) {
  std::string n;
  put_str(n, 1, "v" + std::to_string(i));
  put_str(n, 2, "v" + std::to_string(i + 1));
  put_str(n, 3, "n" + std::to_string(i));
  put_str(n, 4, "Relu");
  return n;
}

// A linear Relu chain of `nodes` nodes: v0 -> n0 -> v1 -> n1 -> ... Written to
// `path`. Deterministic, structure-only (no initializers, no payloads).
Result<bool> write_chain_model(const std::string& path, uint32_t nodes) {
  std::string graph;
  for (uint32_t i = 0; i < nodes; ++i) put_msg(graph, 1, chain_node(i));
  put_str(graph, 2, "bench_chain");

  std::string model;
  put_tag(model, 1, 0);  // ir_version
  put_varint(model, 7);
  put_str(model, 2, "netvis-mcp-bench");
  put_msg(model, 7, graph);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return err("mcp-bench: cannot write " + path);
  out.write(model.data(), static_cast<std::streamsize>(model.size()));
  out.close();
  return true;
}

// ---------------------------------------------------------------------------
// Timing helpers. Median-of-N over the real handle_line entry.
// ---------------------------------------------------------------------------
double time_ms(const std::function<void()>& fn) {
  const auto t0 = clock_::now();
  fn();
  const auto t1 = clock_::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double median_ms(int repeats, const std::function<void()>& fn) {
  std::vector<double> runs;
  runs.reserve(static_cast<size_t>(repeats));
  for (int i = 0; i < repeats; ++i) runs.push_back(time_ms(fn));
  std::sort(runs.begin(), runs.end());
  return runs[runs.size() / 2];
}

std::string call_line(int id, const std::string& tool, const json& args) {
  json j;
  j["jsonrpc"] = "2.0";
  j["id"] = id;
  j["method"] = "tools/call";
  json p;
  p["name"] = tool;
  p["arguments"] = args;
  j["params"] = std::move(p);
  return j.dump();
}

// A response that failed the tool call invalidates the whole run: a harness
// that times error paths reports fiction.
Result<bool> expect_ok(const std::string& response) {
  json j = json::parse(response, nullptr, false);
  if (j.is_discarded() || j.contains("error") ||
      (j.contains("result") && j["result"].value("isError", false))) {
    return err("mcp-bench: tool call failed: " + response.substr(0, 200));
  }
  return true;
}

}  // namespace

bool wants_mcp_bench(int argc, char** argv) {
  for (int i = 1; i < argc; ++i)
    if (std::string_view(argv[i]) == "--bench-mcp") return true;
  return false;
}

Result<std::string> run_mcp_bench() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "netvis_mcp_bench";
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) return err("mcp-bench: cannot create " + dir.string());

  constexpr int kRepeats = 5;
  const std::vector<uint32_t> ladder = {1'000, 10'000, 100'000};

  json cases = json::array();
  for (uint32_t nodes : ladder) {
    const std::string path = (dir / ("chain_" + std::to_string(nodes) + ".onnx")).string();
    auto wrote = write_chain_model(path, nodes);
    if (!wrote) return wrote.error();

    McpServer server;
    std::string last;

    json nodes_args{{"model", path}, {"limit", 1}};
    json search_args{{"model", path}, {"query", "op:relu"}, {"limit", 5}};
    json cost_args{{"model", path}, {"limit", 5}};
    json nbr_args{{"model", path}, {"target", "#0"}};

    // Cold: the first query a session makes against a path (load + analyze).
    const double cold_ms =
        time_ms([&] { last = server.handle_line(call_line(1, "netvis_nodes", nodes_args)); });
    auto ok = expect_ok(last);
    if (!ok) return ok.error();

    // Warm: the steady state, one median per verb family.
    const double warm_nodes_ms = median_ms(kRepeats, [&] {
      last = server.handle_line(call_line(2, "netvis_nodes", nodes_args));
    });
    if (auto r = expect_ok(last); !r) return r.error();
    const double warm_search_ms = median_ms(kRepeats, [&] {
      last = server.handle_line(call_line(3, "netvis_search", search_args));
    });
    if (auto r = expect_ok(last); !r) return r.error();
    const double warm_cost_ms = median_ms(kRepeats, [&] {
      last = server.handle_line(call_line(4, "netvis_cost", cost_args));
    });
    if (auto r = expect_ok(last); !r) return r.error();
    const double warm_neighbors_ms = median_ms(kRepeats, [&] {
      last = server.handle_line(call_line(5, "netvis_neighbors", nbr_args));
    });
    if (auto r = expect_ok(last); !r) return r.error();

    json c;
    c["nodes"] = nodes;
    c["cold_call_ms"] = cold_ms;
    c["warm_nodes_ms"] = warm_nodes_ms;
    c["warm_search_ms"] = warm_search_ms;
    c["warm_cost_ms"] = warm_cost_ms;
    c["warm_neighbors_ms"] = warm_neighbors_ms;
    cases.push_back(std::move(c));
  }

  // Protocol-only overhead, measured without any model in play.
  {
    McpServer server;
    const double ping_us =
        median_ms(kRepeats, [&] {
          (void)server.handle_line(R"({"jsonrpc":"2.0","id":1,"method":"ping"})");
        }) * 1000.0;
    const double list_us =
        median_ms(kRepeats, [&] {
          (void)server.handle_line(R"({"jsonrpc":"2.0","id":2,"method":"tools/list"})");
        }) * 1000.0;
    json p;
    p["ping_us"] = ping_us;
    p["tools_list_us"] = list_us;
    cases.push_back(json{{"protocol", std::move(p)}});
  }

  // Session footprint: RSS growth is bounded by the cache cap, not by how many
  // distinct models a session touches. Eight distinct 10k-node models through a
  // cap-4 cache must cost roughly what four cost.
  json footprint;
  {
    std::vector<std::string> paths;
    for (int i = 0; i < 8; ++i) {
      const std::string p = (dir / ("rss_" + std::to_string(i) + ".onnx")).string();
      auto wrote = write_chain_model(p, 10'000);
      if (!wrote) return wrote.error();
      paths.push_back(p);
    }
    McpServer server;
    const uint64_t rss0 = current_rss_bytes();
    for (int i = 0; i < 4; ++i) {
      auto r = expect_ok(server.handle_line(
          call_line(10 + i, "netvis_nodes", json{{"model", paths[static_cast<size_t>(i)]}, {"limit", 1}})));
      if (!r) return r.error();
    }
    const uint64_t rss_at_cap = current_rss_bytes();
    for (int i = 4; i < 8; ++i) {
      auto r = expect_ok(server.handle_line(
          call_line(10 + i, "netvis_nodes", json{{"model", paths[static_cast<size_t>(i)]}, {"limit", 1}})));
      if (!r) return r.error();
    }
    const uint64_t rss_after_churn = current_rss_bytes();
    footprint["cache_cap"] = ModelCache::kDefaultMaxModels;
    footprint["rss_before_bytes"] = rss0;
    footprint["rss_at_cap_bytes"] = rss_at_cap;
    footprint["rss_after_2x_cap_bytes"] = rss_after_churn;
    footprint["peak_rss_bytes"] = peak_rss_bytes();
  }

  fs::remove_all(dir, ec);

  json out;
  out["schema"] = "netvis.mcpbench.v1";
  out["repeats"] = kRepeats;
  out["cases"] = std::move(cases);
  out["footprint"] = std::move(footprint);
  return out.dump();
}

}  // namespace netvis
