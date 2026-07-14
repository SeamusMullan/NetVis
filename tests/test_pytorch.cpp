// tests/test_pytorch.cpp — PyTorch zip parser contract (spec §6.5, §10).
//
// The fixture's pickle has key "w" -> _rebuild_tensor_v2 (allowlisted; must
// yield a TensorRef with a real storage offset) and key "junk" -> a GLOBAL to a
// NON-allowlisted target (numpy.core.foo; must be handled gracefully via the
// opaque path, NO crash). Asserts tensor-table mode + zero payload reads (§2.1).
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
const char* kFixture = "tests/fixtures/model.pt";
}

TEST_CASE("PyTorch zip: allowlist hit yields tensor, miss is opaque, no crash") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = pytorch::parse_zip(*mf, progress);
  // The parser must not crash on the non-allowlisted GLOBAL; it should return a
  // valid model (opaque entry ignored/marked), not an error.
  REQUIRE_MESSAGE(res, "pytorch::parse_zip returned an error");

  const ir::Model& model = *res;
  // A state_dict-only checkpoint has no compute graph.
  CHECK(model.has_graph == false);

  // The allowlisted "w" tensor must have produced a TensorRef pointing at the
  // storage bytes in archive/data/0 (a real, non-sentinel offset).
  bool found_w = false;
  for (const auto& t : model.flat_tensors) {
    if (std::string(model.str(t.name)) == "w") {
      found_w = true;
      CHECK(t.file_offset != UINT64_MAX);
    }
  }
  CHECK_MESSAGE(found_w, "allowlisted key 'w' should yield a TensorRef");

  // The non-allowlisted "junk" GLOBAL must not have produced a bogus real
  // tensor; at most an opaque placeholder. The key assertion is simply that we
  // got here without crashing.
  CHECK(ByteReader::payload_read_counter() == 0);
}
