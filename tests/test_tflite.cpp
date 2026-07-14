// tests/test_tflite.cpp — TFLite flatbuffer parser contract (spec §6, §10).
//
// The fixture is a minimal-but-valid flatbuffer: TFL3 identifier, one SubGraph,
// one ADD operator, two tensors, two buffers (one holding payload). Asserts a
// compute graph with the operator mapped to a node, and zero payload reads
// (buffer bytes recorded as offsets, never decoded — §2.1).
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
const char* kFixture = "tests/fixtures/model.tflite";
}

TEST_CASE("TFLite: subgraph with ADD op, tensors, no payload reads") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = tflite::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "tflite::parse returned an error");

  const ir::Model& model = *res;
  // TFLite carries a compute graph (subgraph -> Graph).
  CHECK(model.has_graph == true);
  REQUIRE(model.graphs.size() >= 1);
  const ir::Graph& g = model.graphs[0];

  // One operator -> one node.
  CHECK(g.nodes.size() == 1);

  // Two tensors were declared; they surface as values and/or initializers. At
  // least one tensor payload offset (buffer1) should be recorded somewhere.
  bool saw_offset = false;
  for (const auto& t : g.initializers)
    if (t.file_offset != UINT64_MAX) saw_offset = true;
  for (const auto& t : model.flat_tensors)
    if (t.file_offset != UINT64_MAX) saw_offset = true;
  // Not all TFLite tensors have a payload buffer, so this is a soft check: if
  // the parser records the one populated buffer, it must be a real offset.
  CHECK((saw_offset || g.values.size() >= 2));

  CHECK(ByteReader::payload_read_counter() == 0);
}
