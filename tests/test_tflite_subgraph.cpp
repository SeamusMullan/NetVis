// tests/test_tflite_subgraph.cpp — TFLite control-flow subgraph linking.
//
// Verifies the f_operator field-number fix (builtin_options_type=field 3,
// builtin_options=field 4) AND that an IF operator links its referenced
// then-subgraph into Node.subgraph. Also re-checks the plain fixture still
// parses so the field-number change is not a regression.
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
const char* kCtrlFlow = "tests/fixtures/model_ctrlflow.tflite";
const char* kPlain = "tests/fixtures/model.tflite";
}  // namespace

TEST_CASE("TFLite: IF operator links a subgraph, no payload reads") {
  if (!std::filesystem::exists(kCtrlFlow)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kCtrlFlow);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = tflite::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "tflite::parse returned an error");

  const ir::Model& model = *res;
  REQUIRE(model.has_graph);
  // Two subgraphs: main (0) + then-branch (1).
  REQUIRE(model.graphs.size() == 2);

  const ir::Graph& main = model.graphs[0];
  REQUIRE(main.nodes.size() == 1);
  const ir::Node& if_node = main.nodes[0];

  // The IF op must link the then-subgraph (index 1) via Node.subgraph.
  CHECK(if_node.subgraph == 1);

  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("TFLite: plain fixture still parses (field-number fix regression)") {
  if (!std::filesystem::exists(kPlain)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kPlain);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = tflite::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "tflite::parse returned an error");

  const ir::Model& model = *res;
  REQUIRE(model.graphs.size() >= 1);
  const ir::Graph& g = model.graphs[0];
  // One ADD operator; no control flow so no subgraph link.
  CHECK(g.nodes.size() == 1);
  CHECK(g.nodes[0].subgraph == -1);

  CHECK(ByteReader::payload_read_counter() == 0);
}
