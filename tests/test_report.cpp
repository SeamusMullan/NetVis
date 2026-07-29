// tests/test_report.cpp — headless JSON report contract (issue #58).
//
// build_report_json is a PURE structural summary over an ir::Model (parse +
// compute_cost, zero payload reads). These tests build a tiny model in code (no
// file dependency), parse the emitted JSON back with nlohmann, and assert the
// documented schema keys plus the correct totals for known-value graphs. A
// file-parse smoke test (report_file) covers the end-to-end path on the ONNX
// fixture when present.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/ByteReader.h"
#include "engine/ReportJson.h"
#include "ir/IR.h"

using namespace netvis;
using json = nlohmann::json;

namespace {

// Minimal graph builders mirroring test_cost.cpp's helpers.
uint32_t add_value(ir::Model& m, ir::Graph& g, const std::string& name,
                   ir::DType dt, const std::vector<int64_t>& shape) {
  ir::ValueInfo v;
  v.name = m.intern(name);
  v.dtype = dt;
  for (int64_t d : shape) v.shape.push_back(d);
  v.producer = -1;
  uint32_t idx = static_cast<uint32_t>(g.values.size());
  g.values.push_back(std::move(v));
  return idx;
}

uint32_t add_node(ir::Model& m, ir::Graph& g, const std::string& op,
                  const std::vector<uint32_t>& ins,
                  const std::vector<uint32_t>& outs) {
  ir::Node n;
  n.op_type = m.intern(op);
  n.name = m.intern(op + "0");
  uint32_t node_idx = static_cast<uint32_t>(g.nodes.size());
  n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  for (uint32_t vi : ins) g.edge_refs.push_back(vi);
  n.inputs.count = static_cast<uint32_t>(ins.size());
  n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  for (uint32_t vi : outs) {
    g.edge_refs.push_back(vi);
    g.values[vi].producer = static_cast<int32_t>(node_idx);
  }
  n.outputs.count = static_cast<uint32_t>(outs.size());
  g.nodes.push_back(std::move(n));
  return node_idx;
}

void add_initializer(ir::Model& m, ir::Graph& g, const std::string& name,
                     ir::DType dtype, const std::vector<int64_t>& shape) {
  ir::TensorRef tr;
  tr.name = m.intern(name);
  tr.dtype = dtype;
  for (int64_t d : shape) tr.shape.push_back(d);
  tr.file_offset = 0;
  g.initializers.push_back(std::move(tr));
}

}  // namespace

TEST_CASE("Report: valid JSON with schema keys for a MatMul graph") {
  // MatMul [8,16]x[16,32]->[8,32]: total_flops = 2*8*32*16 = 8192; no params.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.producer = m.intern("netvis-test");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  g.name = m.intern("main");
  uint32_t a = add_value(m, g, "a", ir::DType::F32, {8, 16});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {16, 32});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {8, 32});
  g.graph_inputs.push_back(a);
  g.graph_inputs.push_back(b);
  g.graph_outputs.push_back(out);
  add_node(m, g, "MatMul", {a, b}, {out});

  ByteReader::payload_read_counter() = 0;
  std::string s = build_report_json(m);
  CHECK(ByteReader::payload_read_counter() == 0);  // zero payload reads

  // Must be parseable JSON.
  json j = json::parse(s);  // throws on malformed -> test failure

  // Top-level schema keys.
  CHECK(j.at("schema") == kReportSchema);
  CHECK(j.at("format") == "ONNX");
  CHECK(j.at("producer") == "netvis-test");
  CHECK(j.at("has_graph") == true);
  CHECK(j.at("graph_count") == 1);

  REQUIRE(j.at("graphs").is_array());
  REQUIRE(j.at("graphs").size() == 1);
  const json& g0 = j["graphs"][0];
  CHECK(g0.at("index") == 0);
  CHECK(g0.at("name") == "main");
  CHECK(g0.at("nodes") == 1);
  CHECK(g0.at("edges") == 0);  // single node, no producer->consumer edge
  CHECK(g0.at("inputs") == 2);
  CHECK(g0.at("outputs") == 1);
  CHECK(g0.at("initializers") == 0);

  const json& cost = j.at("cost");
  CHECK(cost.at("graph_index") == 0);
  CHECK(cost.at("from_graph") == true);
  CHECK(cost.at("total_flops") == 2 * 8 * 32 * 16);  // 8192
  CHECK(cost.at("total_params") == 0);
  CHECK(cost.at("nodes_total") == 1);
  CHECK(cost.at("nodes_flops_known") == 1);
  CHECK(cost.contains("total_weight_bytes"));
  CHECK(cost.contains("peak_activation_bytes"));
  CHECK(cost.contains("effective_bits_per_param"));
  CHECK(cost.contains("size_vs_fp32"));
  REQUIRE(cost.at("dtype_usage").is_array());

  const json& roof = cost.at("roofline");
  CHECK(roof.at("ridge_flop_per_byte") == doctest::Approx(40.0));
  CHECK(roof.contains("compute_bound_fraction"));
}

TEST_CASE("Report: Conv weights -> correct params/weight_bytes + dtype_usage") {
  // Conv out [1,64,28,28], weight [64,32,3,3] fp32, group=1 (numbers from
  // test_cost.cpp): params = 18432, weight_bytes = 73728, flops = 28901376.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  g.name = m.intern("g");
  uint32_t inp = add_value(m, g, "inp", ir::DType::F32, {1, 32, 30, 30});
  uint32_t w = add_value(m, g, "w", ir::DType::F32, {64, 32, 3, 3});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {1, 64, 28, 28});
  add_initializer(m, g, "w", ir::DType::F32, {64, 32, 3, 3});
  add_node(m, g, "Conv", {inp, w}, {out});
  // group attr on the node.
  {
    ir::Attribute at;
    at.name = m.intern("group");
    at.value.kind = ir::AttrValue::Kind::Int;
    at.value.i = 1;
    uint32_t ai = static_cast<uint32_t>(g.attributes.size());
    g.attributes.push_back(std::move(at));
    g.nodes.back().attributes.begin = ai;
    g.nodes.back().attributes.count = 1;
  }

  ByteReader::payload_read_counter() = 0;
  std::string s = build_report_json(m);
  CHECK(ByteReader::payload_read_counter() == 0);

  json j = json::parse(s);
  const json& cost = j.at("cost");
  CHECK(cost.at("total_flops") == 28901376);
  CHECK(cost.at("total_params") == 18432);
  CHECK(cost.at("total_weight_bytes") == 73728);
  CHECK(cost.at("nodes_flops_known") == 1);

  REQUIRE(cost.at("dtype_usage").size() == 1);
  CHECK(cost["dtype_usage"][0].at("dtype") == "f32");
  CHECK(cost["dtype_usage"][0].at("params") == 18432);
  CHECK(cost["dtype_usage"][0].at("bytes") == 73728);
  // effective bits/param = 8 * 73728 / 18432 = 32 (pure fp32).
  CHECK(cost.at("effective_bits_per_param").get<double>() == doctest::Approx(32.0));
}

TEST_CASE("Report: table-mode model (no compute graph) reports from flat_tensors") {
  ir::Model m;
  m.format_name = m.intern("GGUF");
  m.has_graph = false;
  ir::TensorRef t;
  t.name = m.intern("t");
  t.dtype = ir::DType::F32;
  t.shape.push_back(100);
  t.byte_len = 400;
  m.flat_tensors.push_back(std::move(t));

  ByteReader::payload_read_counter() = 0;
  std::string s = build_report_json(m);
  CHECK(ByteReader::payload_read_counter() == 0);

  json j = json::parse(s);
  CHECK(j.at("format") == "GGUF");
  CHECK(j.at("has_graph") == false);
  CHECK(j.at("graph_count") == 0);
  CHECK(j.at("graphs").is_array());
  CHECK(j.at("graphs").empty());

  const json& cost = j.at("cost");
  CHECK(cost.at("from_graph") == false);
  CHECK(cost.at("total_params") == 100);
  CHECK(cost.at("total_weight_bytes") == 400);
  CHECK(cost.at("nodes_total") == 0);
}

TEST_CASE("Report: report_file on the ONNX fixture parses + reports end-to-end") {
  const char* kFixture = "tests/fixtures/model.onnx";
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  ByteReader::payload_read_counter() = 0;
  Result<std::string> r = report_file(kFixture);
  REQUIRE_MESSAGE(r, "report_file returned an error");
  CHECK(ByteReader::payload_read_counter() == 0);  // zero payload reads

  json j = json::parse(*r);
  CHECK(j.at("schema") == kReportSchema);
  CHECK(j.at("format") == "ONNX");
  CHECK(j.at("has_graph") == true);
  REQUIRE(j.at("graphs").size() >= 1);
  // The fixture has 3 nodes (Conv/Relu/MatMul) per test_onnx.cpp.
  CHECK(j["graphs"][0].at("nodes") == 3);
  CHECK(j.at("cost").at("nodes_total") == 3);
}

TEST_CASE("Report: report_file on a missing path returns an error, not a crash") {
  Result<std::string> r = report_file("tests/fixtures/does_not_exist.onnx");
  CHECK_FALSE(r);
  CHECK_FALSE(r.error().message.empty());
}
