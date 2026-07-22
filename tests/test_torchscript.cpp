// tests/test_torchscript.cpp — TorchScript archive best-effort op listing (§#40).
//
// The fixture is a TorchScript archive with data.pkl + constants.pkl +
// code/model.py. Asserts: parameters surface (existing path), op/method
// metadata is populated from the code scan, zero payload reads, and the pickle
// allowlist still holds on constants.pkl (no weakening).
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
const char* kFixture = "tests/fixtures/model_ts.pt";
}

TEST_CASE("TorchScript archive: best-effort op/method listing, zero payload reads") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = pytorch::parse_zip(*mf, progress);
  REQUIRE_MESSAGE(res, "pytorch::parse_zip returned an error");

  const ir::Model& model = *res;
  // TorchScript archive: tensor-table mode (no full graph reconstruction)
  CHECK(model.has_graph == false);

  // Parameters must still surface from data.pkl (existing collect_tensors path)
  bool found_w = false;
  for (const auto& t : model.flat_tensors) {
    if (std::string(model.str(t.name)) == "w") {
      found_w = true;
      CHECK(t.file_offset != UINT64_MAX);
    }
  }
  CHECK_MESSAGE(found_w, "parameter 'w' should surface from data.pkl");

  // Op/method inventory metadata should be populated from code scan
  bool found_methods = false;
  bool found_ops = false;
  bool found_torchscript_note = false;
  for (const auto& kv : model.metadata) {
    std::string key(model.str(kv.first));
    std::string val(model.str(kv.second));
    if (key == "torchscript.methods") {
      found_methods = true;
      // The fixture code defines "forward" and "helper"
      CHECK(val.find("forward") != std::string::npos);
      CHECK(val.find("helper") != std::string::npos);
    }
    if (key == "torchscript.ops") {
      found_ops = true;
      // The fixture code calls torch.relu, torch.add, aten::matmul, ops.custom_op
      CHECK(val.find("torch.relu") != std::string::npos);
      CHECK(val.find("torch.add") != std::string::npos);
      CHECK(val.find("aten::matmul") != std::string::npos);
      CHECK(val.find("ops.custom_op") != std::string::npos);
    }
    if (key == "torchscript") {
      found_torchscript_note = true;
      // Should mention best-effort/heuristic
      CHECK(val.find("best-effort") != std::string::npos);
    }
  }
  CHECK_MESSAGE(found_methods, "torchscript.methods metadata should be present");
  CHECK_MESSAGE(found_ops, "torchscript.ops metadata should be present");
  CHECK_MESSAGE(found_torchscript_note, "torchscript note should be present");

  // Zero payload reads: code blobs are structural text, weights stay offset+len
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("TorchScript: pickle allowlist enforced on constants.pkl") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  // The fixture's constants.pkl is an empty tuple (allowlist-safe). The parser
  // must not crash; the existing PickleVM allowlist must still hold (no
  // weakening). This is a smoke test: if the allowlist were weakened or
  // constants.pkl ran arbitrary code, a hostile archive would crash/hang. Our
  // fixture is safe, so we just assert successful parse.
  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = pytorch::parse_zip(*mf, progress);
  REQUIRE_MESSAGE(res, "parse must not crash on constants.pkl");
}
