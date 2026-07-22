// tests/test_npz.cpp — NumPy .npz parser contract (spec §6, §10, issue #41).
//
// NumPy .npz has no compute graph, so has_graph must be false and tensors land
// in flat_tensors. Asserts two F32 tensors (w, b), correct shapes, real payload
// offsets, and zero payload reads (offsets recorded, bytes untouched — §2.1).
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
const char* kFixture = "tests/fixtures/model.npz";
}

TEST_CASE("NumPy .npz: two F32 tensors, tensor-table mode, no payload reads") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = npz::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "npz::parse returned an error");

  const ir::Model& model = *res;
  // No compute graph -> tensor-table mode.
  CHECK(model.has_graph == false);

  // Expect exactly w and b (2 tensors).
  CHECK(model.flat_tensors.size() == 2);

  bool found_w = false, found_b = false;
  bool saw_offset = false;
  for (const auto& t : model.flat_tensors) {
    std::string name(model.str(t.name));
    CHECK(t.dtype == ir::DType::F32);
    if (t.file_offset != UINT64_MAX) saw_offset = true;
    if (name == "w") {
      found_w = true;
      REQUIRE(t.shape.size() == 2);
      CHECK(t.shape[0] == 2);
      CHECK(t.shape[1] == 3);
      CHECK(t.byte_len == 24);  // 6 floats * 4 bytes
    } else if (name == "b") {
      found_b = true;
      REQUIRE(t.shape.size() == 1);
      CHECK(t.shape[0] == 3);
      CHECK(t.byte_len == 12);  // 3 floats * 4 bytes
    }
  }
  CHECK(found_w);
  CHECK(found_b);
  CHECK(saw_offset);

  // The invariant: structural parsing never touched payload bytes.
  CHECK(ByteReader::payload_read_counter() == 0);
}
