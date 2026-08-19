// tests/test_query.cpp — the agent-facing query CLI contract.
//
// run_query is the whole surface: verb dispatch, option parsing, model loading
// and the JSON projections. These tests run every verb end-to-end against the
// generated fixtures (graceful WARN skip when absent, like test_report.cpp) and
// assert the two properties the CLI promises on top of correct payloads:
//   1. every verb except `tensor` performs ZERO payload reads (ByteReader
//      counter), upholding the thesis the whole tool is built on;
//   2. errors are loud — unknown verbs, unknown flags and missing targets all
//      fail with a Result error, never a silent empty answer.
#include <doctest/doctest.h>

#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/ByteReader.h"
#include "engine/QueryCli.h"

using namespace netvis;
using json = nlohmann::json;

namespace {

constexpr const char* kOnnx = "tests/fixtures/model.onnx";
constexpr const char* kTflite = "tests/fixtures/model.tflite";
constexpr const char* kSafetensors = "tests/fixtures/model.safetensors";

bool have(const char* path) {
  if (std::filesystem::exists(path)) return true;
  WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
  return false;
}

// Run a query and parse its JSON, asserting success and the envelope fields.
json run_ok(const std::vector<std::string>& args) {
  Result<std::string> r = run_query(args);
  REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
  json j = json::parse(*r, nullptr, false);
  REQUIRE_MESSAGE(!j.is_discarded(), "verb emitted invalid JSON");
  CHECK(j["schema"] == std::string(kQuerySchema));
  CHECK(j["verb"] == args[0]);
  return j;
}

}  // namespace

TEST_CASE("Query: structural verbs answer correctly and read zero payload bytes") {
  if (!have(kOnnx)) return;
  ByteReader::payload_read_counter() = 0;

  // summary embeds the report v1 whole — same pipeline, same numbers.
  json summary = run_ok({"summary", kOnnx});
  CHECK(summary["report"]["schema"] == "netvis.report.v1");
  CHECK(summary["report"]["graphs"][0]["nodes"] == 3);

  // io: the ONNX fixture declares no graph input/output value_info, and the
  // verb reports that honestly as empty arrays rather than inventing entries.
  json io = run_ok({"io", kOnnx});
  CHECK(io["inputs"].is_array());
  CHECK(io["inputs"].empty());
  CHECK(io["outputs"].empty());

  // nodes: all three, then filtered to the one Conv (case-insensitive).
  json nodes = run_ok({"nodes", kOnnx});
  CHECK(nodes["total_matched"] == 3);
  CHECK(nodes["nodes"].size() == 3);
  json convs = run_ok({"nodes", kOnnx, "--op", "conv"});
  CHECK(convs["total_matched"] == 1);
  CHECK(convs["nodes"][0]["op"] == "Conv");

  // Pagination is exact: 3 matches, window [1, 2).
  json page = run_ok({"nodes", kOnnx, "--limit", "1", "--offset", "1"});
  CHECK(page["total_matched"] == 3);
  CHECK(page["returned"] == 1);
  CHECK(page["nodes"][0]["index"] == 1);

  // node by index: full detail incl. attrs, io values and adjacency.
  json node = run_ok({"node", kOnnx, "#0"});
  CHECK(node["op"] == "Conv");
  CHECK(node["inputs"].size() > 0);
  CHECK(node["attributes"].is_array());
  CHECK(node["predecessors"].is_array());
  REQUIRE(node["successors"].size() == 1);
  CHECK(node["successors"][0]["op"] == "Relu");

  // node by exact name round-trips to the same node.
  const std::string name = node["name"].get<std::string>();
  if (!name.empty()) {
    json by_name = run_ok({"node", kOnnx, name});
    CHECK(by_name["index"] == 0);
  }

  // neighbors: 2 hops downstream of the Conv reaches Relu and MatMul.
  json nb = run_ok({"neighbors", kOnnx, "#0", "--hops", "2", "--dir", "out"});
  CHECK(nb["successors"].size() == 2);
  CHECK(!nb.contains("predecessors"));

  // tensors: initializer listing, largest-first when sorted by bytes.
  json tensors = run_ok({"tensors", kOnnx, "--sort", "bytes"});
  REQUIRE(tensors["total"].get<uint64_t>() > 0);
  if (tensors["tensors"].size() >= 2) {
    CHECK(tensors["tensors"][0]["bytes"].get<uint64_t>() >=
          tensors["tensors"][1]["bytes"].get<uint64_t>());
  }

  // search: field-scoped query hits the Conv node.
  json hits = run_ok({"search", kOnnx, "op:conv"});
  bool found_conv = false;
  for (const auto& h : hits["hits"]) found_conv = found_conv || h["kind"] == "node";
  CHECK(found_conv);

  // cost: ranking covers every node and orders by the requested metric.
  json cost = run_ok({"cost", kOnnx, "--by", "flops"});
  CHECK(cost["nodes_total"] == 3);
  REQUIRE(cost["nodes"].size() > 0);
  CHECK(cost["nodes"][0].contains("flops"));

  // diff against itself: nothing added, removed or changed.
  json diff = run_ok({"diff", kOnnx, kOnnx});
  CHECK(diff["same"] == 3);
  CHECK(diff["added"] == 0);
  CHECK(diff["removed"] == 0);
  CHECK(diff["changed"] == 0);

  // The property all of the above hangs on: not one payload byte was read.
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Query: tensor is the one payload-reading verb and says what it read") {
  if (!have(kOnnx)) return;

  // List structurally first (still zero payload reads); pick the in-file
  // initializer W, whose payload the fixture actually carries.
  ByteReader::payload_read_counter() = 0;
  json tensors = run_ok({"tensors", kOnnx});
  REQUIRE(tensors["tensors"].size() > 0);
  CHECK(ByteReader::payload_read_counter() == 0);

  json t = run_ok({"tensor", kOnnx, "W"});
  CHECK(t["tensor"]["name"] == "W");
  CHECK(t["stats"]["count"].get<uint64_t>() == 4);
  CHECK(t["stats"]["min"].get<double>() == doctest::Approx(1.0));
  CHECK(t["stats"]["max"].get<double>() == doctest::Approx(4.0));
  CHECK(t["stats"]["histogram"].size() == 64);
  // Exactly one payload read, through the same path the weight inspector uses.
  CHECK(ByteReader::payload_read_counter() == 1);

  // The fixture's B initializer points 5 GB into a sibling file on purpose;
  // the verb must surface that as an error, never fabricate stats.
  Result<std::string> bogus = run_query({"tensor", kOnnx, "B"});
  CHECK(!bogus);
}

TEST_CASE("Query: io reports declared graph inputs and outputs (TFLite)") {
  if (!have(kTflite)) return;
  json io = run_ok({"io", kTflite});
  CHECK(io["inputs"].size() > 0);
  CHECK(io["outputs"].size() > 0);
  CHECK(io["inputs"][0].contains("dtype"));
  CHECK(io["inputs"][0]["shape"].is_array());
}

TEST_CASE("Query: table-mode models answer through flat_tensors") {
  if (!have(kSafetensors)) return;
  json tensors = run_ok({"tensors", kSafetensors});
  CHECK(tensors["graph"].is_null());
  CHECK(tensors["total"].get<uint64_t>() > 0);

  // Graph verbs on a graph-less model fail loudly instead of faking a graph.
  Result<std::string> io = run_query({"io", kSafetensors});
  CHECK(!io);
}

TEST_CASE("Query: malformed invocations fail loudly, never silently") {
  Result<std::string> r = run_query({});
  CHECK(!r);
  r = run_query({"frobnicate", kOnnx});
  CHECK(!r);
  r = run_query({"nodes", kOnnx, "--no-such-flag", "1"});
  CHECK(!r);
  r = run_query({"nodes", kOnnx, "--limit", "abc"});
  CHECK(!r);
  r = run_query({"node", kOnnx});  // missing target
  CHECK(!r);
  r = run_query({"node", kOnnx, "#999"});  // out of range
  CHECK(!r);
  r = run_query({"node", kOnnx, "no-such-node"});
  CHECK(!r);
  r = run_query({"summary", "tests/fixtures/definitely-missing.onnx"});
  CHECK(!r);
}
