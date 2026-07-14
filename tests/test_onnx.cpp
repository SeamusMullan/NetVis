// tests/test_onnx.cpp — ONNX parser contract (spec §6, §10).
//
// Parses the hand-encoded model.onnx fixture and asserts: 3 nodes with op_types
// Conv/Relu/MatMul, a raw_data initializer whose TensorRef records a file_offset,
// an external_data initializer carrying external_path + byte_len, AND that the
// payload-read counter is still 0 after parsing (the whole product thesis:
// structural parse never touches tensor payloads — spec §2.1).
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
const char* kFixture = "tests/fixtures/model.onnx";
}

TEST_CASE("ONNX: 3 nodes, initializers, no payload reads") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  // Reset the payload counter BEFORE the parse so the assertion below measures
  // only this parse (the counter is thread-local and shared across tests).
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

  // --- node count + op types -----------------------------------------------
  CHECK(g.nodes.size() == 3);
  REQUIRE(g.nodes.size() == 3);
  CHECK(model.str(g.nodes[0].op_type) == "Conv");
  CHECK(model.str(g.nodes[1].op_type) == "Relu");
  CHECK(model.str(g.nodes[2].op_type) == "MatMul");

  // --- AttributeProto int-lists parse as ints, not strings (regression) -----
  // The Conv node carries kernel_shape=[3,3] and strides=[1,1] on field 8. A
  // field-number transposition (the original bug) would read these as strings
  // and leave value.ints empty. Assert they survive as Ints with correct values.
  {
    const ir::Node& conv = g.nodes[0];
    REQUIRE(conv.attributes.count == 2);
    bool saw_kernel = false, saw_strides = false;
    for (uint32_t i = 0; i < conv.attributes.count; ++i) {
      const ir::Attribute& a = g.attributes[conv.attributes.begin + i];
      std::string an(model.str(a.name));
      CHECK(a.value.kind == ir::AttrValue::Kind::Ints);
      REQUIRE(a.value.ints.size() == 2);
      if (an == "kernel_shape") {
        saw_kernel = true;
        CHECK(a.value.ints[0] == 3);
        CHECK(a.value.ints[1] == 3);
      } else if (an == "strides") {
        saw_strides = true;
        CHECK(a.value.ints[0] == 1);
        CHECK(a.value.ints[1] == 1);
      }
    }
    CHECK(saw_kernel);
    CHECK(saw_strides);
  }

  // --- initializers: one raw_data (real offset), one external ---------------
  REQUIRE(g.initializers.size() == 2);
  const ir::TensorRef* raw = nullptr;
  const ir::TensorRef* ext = nullptr;
  for (const auto& t : g.initializers) {
    if (t.external_path.valid() || t.file_offset == UINT64_MAX) {
      ext = &t;
    } else {
      raw = &t;
    }
  }
  REQUIRE(raw != nullptr);
  REQUIRE(ext != nullptr);

  // The raw_data initializer must record a concrete in-file offset and length,
  // and must NOT have been decoded.
  CHECK(raw->file_offset != UINT64_MAX);
  CHECK(raw->byte_len > 0);

  // The external initializer must carry the external path + declared byte_len
  // and a >2GB offset survives as a 64-bit value (offset stored via file_offset
  // or dedicated field; we only require the external_path + byte_len contract).
  CHECK(ext->external_path.valid());
  CHECK(model.str(ext->external_path).size() > 0);
  CHECK(ext->byte_len > 0);

  // --- the critical invariant: zero payload reads during structural parse ---
  CHECK(ByteReader::payload_read_counter() == 0);
}
