// tests/test_onnx_external.cpp — ONNX external-data resolution (issue #43).
//
// Parses model.onnx and exercises resolve_payload() on the small resolvable
// external initializer (C: offset 8, length 8 into weights.bin), asserting:
// (1) structural parse leaves payload_read_counter at 0, (2) resolve_payload
// opens the sibling and bounds-checks correctly, (3) a negative case (length
// exceeding the sibling file) returns an error rather than over-reading.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <string>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {
const char* kFixture = "tests/fixtures/model.onnx";
const char* kWeightsBin = "tests/fixtures/weights.bin";
}

TEST_CASE("ONNX external-data: resolve_payload positive") {
  if (!std::filesystem::exists(kFixture) || !std::filesystem::exists(kWeightsBin)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  // Reset payload counter before parse.
  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = onnx::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "onnx::parse returned an error");

  const ir::Model& model = *res;
  REQUIRE(model.has_graph);
  REQUIRE(model.graphs.size() >= 1);
  const ir::Graph& g = model.graphs[0];

  // --- Structural parse must leave payload_read_counter at 0 ---------------
  CHECK(ByteReader::payload_read_counter() == 0);

  // --- Find the small resolvable external initializer (C: offset 8, len 8) --
  REQUIRE(g.initializers.size() >= 3);
  const ir::TensorRef* ext_resolvable = nullptr;
  for (const auto& t : g.initializers) {
    if (t.external_path.valid() && t.file_offset == 8 && t.byte_len == 8) {
      ext_resolvable = &t;
      break;
    }
  }
  REQUIRE(ext_resolvable != nullptr);
  CHECK(model.str(ext_resolvable->external_path) == "weights.bin");

  // --- Call compute_tensor_stats: it internally calls resolve_payload ------
  // model_dir must be the fixtures directory so the sibling is found.
  // Pass &model so external_path StringId can be resolved.
  std::string model_dir = "tests/fixtures";
  auto stats = compute_tensor_stats(*ext_resolvable, *mf, model_dir, &model);
  REQUIRE_MESSAGE(stats, "resolve_payload / compute_tensor_stats failed");

  // The resolvable external data is 2 F32s: 5.0, 6.0 (see gen_fixtures.py).
  CHECK(stats->count == 2);
  // Calling compute_tensor_stats marks exactly one payload read.
  CHECK(ByteReader::payload_read_counter() == 1);

  // Verify the stats make sense: min=5, max=6, mean=5.5.
  CHECK(stats->min == 5.0);
  CHECK(stats->max == 6.0);
  CHECK(stats->mean == doctest::Approx(5.5));
}

TEST_CASE("ONNX external-data: bounds error when length exceeds sibling") {
  if (!std::filesystem::exists(kFixture) || !std::filesystem::exists(kWeightsBin)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);

  // Construct a TensorRef with external_path="weights.bin" but a length that
  // exceeds the file size (weights.bin is 16 bytes: 8 pad + 8 data).
  ir::Model temp_model;
  ir::TensorRef bad_ref;
  bad_ref.external_path = temp_model.intern("weights.bin");
  bad_ref.file_offset = 0;
  bad_ref.byte_len = 1000;  // way past the 16-byte file end
  bad_ref.dtype = ir::DType::F32;
  bad_ref.shape = {250};  // 250 * 4 = 1000 bytes

  std::string model_dir = "tests/fixtures";
  auto stats = compute_tensor_stats(bad_ref, *mf, model_dir, &temp_model);
  // resolve_payload should return an error, not crash or over-read.
  CHECK_FALSE(stats);
  // The error message should mention "out of range" (from TensorStats.cpp line 60).
}

TEST_CASE("ONNX external-data: large-offset external survives as 64-bit") {
  // The existing external initializer (B) with offset 5e9 (>4GB) must carry
  // that offset and length correctly. This test is redundant with test_onnx.cpp
  // but confirms the 64-bit handling explicitly.
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = onnx::parse(*mf, progress);
  REQUIRE(res);

  const ir::Model& model = *res;
  const ir::Graph& g = model.graphs[0];
  REQUIRE(g.initializers.size() >= 3);

  const ir::TensorRef* ext_large = nullptr;
  for (const auto& t : g.initializers) {
    if (t.external_path.valid() && t.file_offset == 5000000000ULL) {
      ext_large = &t;
      break;
    }
  }
  REQUIRE(ext_large != nullptr);
  CHECK(ext_large->byte_len == 16);
  CHECK(model.str(ext_large->external_path) == "weights.bin");
}
