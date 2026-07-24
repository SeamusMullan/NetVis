// tests/test_mlpackage.cpp — mlpackage bundle + mlProgram weight blob (#85).
//
// Tests the resolve_model_path contract: a `.mlpackage` directory resolves to
// its inner Data/com.apple.CoreML/model.mlmodel; a plain file passes through.
// Tests the mlProgram parser: field 502 mlProgram proto -> graph with wired
// nodes/edges/initializers; blob_indirect weight TensorRef with real offset 64.
// Tests the blob_metadata header follow in TensorStats (the sanctioned payload
// read). All must assert payload_read_counter()==0 after structural parse.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <string>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "engine/ModelPath.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {
const char* kBundlePath = "tests/fixtures/model.mlpackage";
const char* kPlainPath = "tests/fixtures/model.mlmodel";
}  // namespace

TEST_CASE("mlpackage: resolve_model_path finds inner model.mlmodel") {
  if (!std::filesystem::exists(kBundlePath)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  auto resolved = resolve_model_path(kBundlePath);

  // display_path == what the user opened (the bundle directory).
  CHECK(resolved.display_path == kBundlePath);

  // map_path must end with Data/com.apple.CoreML/model.mlmodel and exist.
  CHECK(resolved.map_path.ends_with("Data/com.apple.CoreML/model.mlmodel"));
  CHECK(std::filesystem::exists(resolved.map_path));

  // The map_path must be an actual file, not a directory.
  CHECK(std::filesystem::is_regular_file(resolved.map_path));
}

TEST_CASE("mlpackage: resolve_model_path passes a plain file through") {
  if (!std::filesystem::exists(kPlainPath)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  auto resolved = resolve_model_path(kPlainPath);

  // Both display_path and map_path should equal the input for a plain file.
  CHECK(resolved.display_path == kPlainPath);
  CHECK(resolved.map_path == kPlainPath);
}

TEST_CASE("mlProgram: parses to a linear-op graph with a blob_indirect weight, zero payload reads") {
  if (!std::filesystem::exists(kBundlePath)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  auto resolved = resolve_model_path(kBundlePath);
  REQUIRE(std::filesystem::exists(resolved.map_path));

  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(resolved.map_path);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = coreml::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "coreml::parse returned an error");

  const ir::Model& model = *res;
  CHECK(model.str(model.format_name) == "CoreML");
  REQUIRE(model.has_graph);
  REQUIRE(model.graphs.size() >= 1);
  const ir::Graph& g = model.graphs[0];

  // --- the mlProgram linear op surfaced as a node ----------------------------
  REQUIRE(g.nodes.size() == 1);
  const ir::Node& node = g.nodes[0];
  CHECK(model.str(node.op_type) == "linear");

  // --- edges wired: input "x", output "out" produced by node 0 ---------------
  bool saw_x_input = false;
  for (uint32_t i = 0; i < node.inputs.count; ++i) {
    uint32_t vi = g.edge_refs[node.inputs.begin + i];
    if (model.str(g.values[vi].name) == "x") saw_x_input = true;
  }
  CHECK(saw_x_input);

  REQUIRE(node.outputs.count == 1);
  uint32_t vo = g.edge_refs[node.outputs.begin];
  CHECK(model.str(g.values[vo].name) == "out");
  CHECK(g.values[vo].producer == 0);

  // --- graph inputs/outputs populated ----------------------------------------
  CHECK(g.graph_inputs.size() >= 1);
  CHECK(g.graph_outputs.size() >= 1);

  // --- initializer: blob_indirect weight with external_path, offset 64 -------
  REQUIRE(g.initializers.size() == 1);
  const ir::TensorRef& w = g.initializers[0];
  CHECK(w.external_path.valid());
  std::string ext_path(model.str(w.external_path));
  CHECK(ext_path == "weights/weight.bin");
  CHECK(w.file_offset == 64);  // offset to blob_metadata header
  CHECK(w.blob_indirect == true);
  CHECK(w.dtype == ir::DType::F32);

  // --- the critical invariant: zero payload reads during structural parse ----
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("mlProgram weight: TensorStats follows the blob_metadata header to real data") {
  if (!std::filesystem::exists(kBundlePath)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  auto resolved = resolve_model_path(kBundlePath);
  REQUIRE(std::filesystem::exists(resolved.map_path));

  // Parse the mlProgram to get the initializer TensorRef.
  auto mf = MappedFile::open(resolved.map_path);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = coreml::parse(*mf, progress);
  REQUIRE(res);

  const ir::Model& model = *res;
  REQUIRE(model.graphs.size() >= 1);
  const ir::Graph& g = model.graphs[0];
  REQUIRE(g.initializers.size() == 1);
  const ir::TensorRef& w = g.initializers[0];

  // Compute tensor stats: model_dir must be the parent directory of map_path
  // so external_path "weights/weight.bin" resolves correctly.
  std::filesystem::path map_path_fs(resolved.map_path);
  std::string model_dir = map_path_fs.parent_path().string();

  ByteReader::payload_read_counter() = 0;
  auto stats_res = compute_tensor_stats(w, *mf, model_dir, &model);
  REQUIRE_MESSAGE(stats_res, "compute_tensor_stats failed");

  const TensorStats& stats = *stats_res;
  // The fixture weight.bin contains 4 F32 values: 1.0, 2.0, 3.0, 4.0 (at offset
  // 64 after the blob_metadata header; the header's data_offset points to them).
  CHECK(stats.count == 4);
  CHECK(stats.min == 1.0);
  CHECK(stats.max == 4.0);

  // The blob_metadata header follow is the ONE accounted payload read.
  CHECK(ByteReader::payload_read_counter() == 1);
}

TEST_CASE("mlProgram: pathologically deep block nesting hits the cap, no crash (#85)") {
  // Block -> Operation{blocks:[Block -> ...]} nested 2000 deep. The parser's
  // build_mil_block <-> parse_mil_operation mutual recursion must be depth-capped
  // (kMaxMilBlockDepth): a clean Result error or bounded model, NEVER a stack
  // overflow. Reaching the assertions below is itself the survival test.
  const char* kDeep = "tests/fixtures/model_mlprogram_deep.mlmodel";
  if (!std::filesystem::exists(kDeep)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  ByteReader::payload_read_counter() = 0;
  auto mf = MappedFile::open(kDeep);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = coreml::parse(*mf, progress);
  // Either a clean error or a bounded best-effort model is acceptable; the only
  // hard requirement is that we returned safely and read no payload.
  if (res) {
    // If it built a (partial) model, has_graph/graphs must be self-consistent.
    CHECK((res->has_graph ? !res->graphs.empty() : res->graphs.empty()));
  }
  CHECK(ByteReader::payload_read_counter() == 0);
}
