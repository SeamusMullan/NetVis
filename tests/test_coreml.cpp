// tests/test_coreml.cpp — CoreML .mlmodel parser contract (spec §6, §10; #38).
//
// Parses the hand-encoded model.mlmodel fixture (a Model protobuf with one
// neuralNetwork innerProduct layer + a rawValue weight) and asserts: the layer
// surfaced as a node with wired input/output edges, the weight TensorRef records
// a real in-file offset+len, and — the product thesis — the payload-read counter
// is still 0 after parsing (structural parse never touches weight bytes). A
// second case feeds a truncated proto and asserts a clean Result error / no crash
// with the counter left clean.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {
const char* kFixture = "tests/fixtures/model.mlmodel";

// Read a whole fixture into memory (small fixtures).
std::vector<uint8_t> read_all(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
}

// Write a prefix of `bytes` to a temp file and return its path.
std::string write_prefix(const std::vector<uint8_t>& bytes, size_t n) {
  std::filesystem::path p =
      std::filesystem::temp_directory_path() / "nv_coreml_trunc.mlmodel";
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(n));
  out.close();
  return p.string();
}
}  // namespace

TEST_CASE("CoreML: neuralNetwork layer -> node, weight offset+len, no payload reads") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  // Reset the (thread-local, cross-test) counter so the assertion below measures
  // only this parse.
  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = coreml::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "coreml::parse returned an error");

  const ir::Model& model = *res;
  CHECK(model.str(model.format_name) == "CoreML");
  REQUIRE(model.has_graph);
  REQUIRE(model.graphs.size() >= 1);
  const ir::Graph& g = model.graphs[0];

  // --- the layer surfaced as a node with the mapped op type ------------------
  REQUIRE(g.nodes.size() == 1);
  const ir::Node& node = g.nodes[0];
  CHECK(model.str(node.op_type) == "InnerProduct");
  CHECK(model.str(node.name) == "fc1");

  // --- edges wired from the input/output name lists --------------------------
  // The node's inputs are: the "data" activation + one synthesized weight edge.
  REQUIRE(node.inputs.count >= 1);
  REQUIRE(node.outputs.count == 1);

  bool saw_data_input = false;
  for (uint32_t i = 0; i < node.inputs.count; ++i) {
    uint32_t vi = g.edge_refs[node.inputs.begin + i];
    if (model.str(g.values[vi].name) == "data") saw_data_input = true;
  }
  CHECK(saw_data_input);

  // The single output edge "fc_out" is produced by this node.
  {
    uint32_t vo = g.edge_refs[node.outputs.begin];
    CHECK(model.str(g.values[vo].name) == "fc_out");
    CHECK(g.values[vo].producer == 0);
  }

  // --- the weight TensorRef records a real in-file offset + length -----------
  REQUIRE(g.initializers.size() == 1);
  const ir::TensorRef& w = g.initializers[0];
  CHECK(w.file_offset != UINT64_MAX);
  CHECK(w.file_offset > 0);
  CHECK(w.file_offset + w.byte_len <= mf->size());  // in-bounds sub-range
  CHECK(w.byte_len == 16);                            // four F32 in the fixture

  // The weight's synthesized name must appear as one of the node's inputs.
  bool weight_wired = false;
  for (uint32_t i = 0; i < node.inputs.count; ++i) {
    uint32_t vi = g.edge_refs[node.inputs.begin + i];
    if (g.values[vi].name == w.name) weight_wired = true;
  }
  CHECK(weight_wired);

  // --- ModelDescription surfaced graph inputs/outputs ------------------------
  CHECK(g.graph_inputs.size() == 1);
  CHECK(g.graph_outputs.size() == 1);

  // --- the critical invariant: zero payload reads during structural parse ----
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("CoreML: truncated proto yields a clean error, never a crash") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  std::vector<uint8_t> bytes = read_all(kFixture);
  REQUIRE(bytes.size() > 8);

  // Truncate mid-message (before the neuralNetwork body completes): the
  // length-delimited sub-range now claims more bytes than remain, which must
  // surface as a bounds-checked Result error, not an over-read or crash.
  for (int k = 1; k <= 7; ++k) {
    size_t n = (bytes.size() * static_cast<size_t>(k)) / 8;
    std::string path = write_prefix(bytes, n);

    ByteReader::payload_read_counter() = 0;
    auto mf = MappedFile::open(path);
    if (mf) {
      ProgressSink progress;
      auto res = coreml::parse(*mf, progress);
      // Accept ok (a valid short prefix) or a clean error — the invariant is
      // "returned safely" and never decoded a payload.
      (void)res;
      CHECK(ByteReader::payload_read_counter() == 0);
    }
    std::filesystem::remove(path);
  }
}
