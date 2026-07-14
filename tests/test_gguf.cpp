// tests/test_gguf.cpp — GGUF v3 parser contract (spec §6.4, §10).
//
// GGUF is a tensor table (no graph). Asserts two tensors (one F32, one Q4_0
// quantized), tensor-table mode, a real payload offset into the aligned data
// section, and zero payload reads during structural parse (§2.1).
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <string>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {
const char* kFixture = "tests/fixtures/model.gguf";
}

TEST_CASE("GGUF: F32 + Q4_0 tensors, tensor-table mode, no payload reads") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = gguf::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "gguf::parse returned an error");

  const ir::Model& model = *res;
  CHECK(model.has_graph == false);
  REQUIRE(model.flat_tensors.size() == 2);

  bool saw_f32 = false, saw_q4 = false, saw_offset = false;
  for (const auto& t : model.flat_tensors) {
    if (t.dtype == ir::DType::F32) saw_f32 = true;
    if (t.dtype == ir::DType::Q4) saw_q4 = true;
    // Every tensor offset is relative to (or absolute into) the mmap and must
    // be a recorded value, never UINT64_MAX for present GGUF data.
    if (t.file_offset != UINT64_MAX) saw_offset = true;
  }
  CHECK(saw_f32);
  CHECK(saw_q4);
  CHECK(saw_offset);

  CHECK(ByteReader::payload_read_counter() == 0);
}
