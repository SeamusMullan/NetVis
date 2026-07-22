// tests/test_openvino.cpp — OpenVINO IR (.xml + .bin) parser contract (#39).
//
// Parses the hand-written model.xml fixture (Parameter -> Convolution(Const) ->
// ReLU -> Result) and asserts: node count + op types, that edges wired the ports
// (a value shape resolved from <dim>s), that the Const initializer records
// external_path ending ".bin" + byte_len==16 + file_offset==0, and — the whole
// product thesis — payload_read_counter()==0 after a structural parse. Plus a
// truncated-<net> test (clean Result error, no crash) and a deep-nesting test
// that must hit the XmlReader depth cap and error rather than overflow the stack.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/openvino/XmlReader.h"

using namespace netvis;

namespace {
const char* kFixture = "tests/fixtures/model.xml";

// Write `content` to a uniquely-named temp file and return its path.
std::string write_temp(const std::string& stem, const std::string& content) {
  std::filesystem::path p =
      std::filesystem::temp_directory_path() / ("nv_ov_" + stem);
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(content.data(), static_cast<std::streamsize>(content.size()));
  out.close();
  return p.string();
}
}  // namespace

TEST_CASE("OpenVINO: nodes + wired edges + Const offset/len, no payload reads") {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }

  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kFixture);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = openvino::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "openvino::parse returned an error");

  const ir::Model& model = *res;
  CHECK(model.str(model.format_name) == "OpenVINO");
  REQUIRE(model.has_graph);
  REQUIRE(model.graphs.size() >= 1);
  const ir::Graph& g = model.graphs[0];

  // --- node count + op types (normalized) ----------------------------------
  // Parameter, Const, Convolution->Conv, ReLU->Relu, Result = 5 nodes.
  CHECK(g.nodes.size() == 5);

  bool saw_conv = false, saw_relu = false, saw_param = false, saw_result = false;
  const ir::Node* conv = nullptr;
  for (const ir::Node& n : g.nodes) {
    std::string op(model.str(n.op_type));
    if (op == "Conv") { saw_conv = true; conv = &n; }
    if (op == "Relu") saw_relu = true;
    if (op == "Parameter") saw_param = true;
    if (op == "Result") saw_result = true;
  }
  CHECK(saw_conv);
  CHECK(saw_relu);
  CHECK(saw_param);
  CHECK(saw_result);

  // Data attributes on the Conv layer must have been captured (strides etc.).
  REQUIRE(conv != nullptr);
  CHECK(conv->attributes.count > 0);

  // Graph IO markers wired: Parameter -> graph_inputs, Result -> graph_outputs.
  CHECK(g.graph_inputs.size() == 1);
  CHECK(g.graph_outputs.size() == 1);

  // --- an edge wired a value whose shape came from the <dim> list -----------
  // The Conv consumes the Parameter's output value; that value's shape must be
  // the 4-D [1,1,2,2] read from the port <dim>s.
  REQUIRE(conv->inputs.count >= 1);
  uint32_t vi = g.edge_refs[conv->inputs.begin];
  const ir::ValueInfo& conv_in = g.values[vi];
  REQUIRE(conv_in.shape.size() == 4);
  CHECK(conv_in.shape[0] == 1);
  CHECK(conv_in.shape[1] == 1);
  CHECK(conv_in.shape[2] == 2);
  CHECK(conv_in.shape[3] == 2);
  CHECK(conv_in.dtype == ir::DType::F32);
  // The producing node is the Parameter (edge was actually wired, not dangling).
  CHECK(conv_in.producer >= 0);

  // --- Const initializer: external_path -> .bin, offset 0, len 16 -----------
  REQUIRE(g.initializers.size() == 1);
  const ir::TensorRef& w = g.initializers[0];
  CHECK(w.external_path.valid());
  std::string ext(model.str(w.external_path));
  CHECK(ext.size() >= 4);
  CHECK(ext.compare(ext.size() - 4, 4, ".bin") == 0);
  CHECK(w.file_offset == 0);
  CHECK(w.byte_len == 16);
  CHECK(w.dtype == ir::DType::F32);
  REQUIRE(w.shape.size() == 2);
  CHECK(w.shape[0] == 2);
  CHECK(w.shape[1] == 2);

  // --- the critical invariant: zero payload reads during structural parse ---
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("OpenVINO: truncated <net> yields a clean error, no crash") {
  ByteReader::payload_read_counter() = 0;

  // A <net> opened but never closed (cut mid-document). The XML reader must
  // report an unclosed-element error rather than loop or over-read.
  const std::string truncated =
      "<?xml version=\"1.0\"?>\n"
      "<net name=\"trunc\" version=\"11\">\n"
      "  <layers>\n"
      "    <layer id=\"0\" name=\"in\" type=\"Parameter\">\n";
  std::string path = write_temp("trunc.xml", truncated);
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = openvino::parse(*mf, progress);
  CHECK_FALSE(res);  // must be an error, not a crash / not a bogus model
  std::filesystem::remove(path);

  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("OpenVINO XmlReader: deep nesting hits the depth cap and errors") {
  // Build a document nested far past kMaxXmlDepth. Construction is iterative, so
  // this must return a clean Result error (never a stack overflow).
  const int kDepth = openvino::kMaxXmlDepth + 50;
  std::string doc = "<root>";
  for (int i = 0; i < kDepth; ++i) doc += "<a>";
  for (int i = 0; i < kDepth; ++i) doc += "</a>";
  doc += "</root>";

  auto parsed = openvino::XmlDocument::parse(
      reinterpret_cast<const uint8_t*>(doc.data()), doc.size(), 0);
  CHECK_FALSE(parsed);  // depth cap tripped -> error, no crash
}

TEST_CASE("OpenVINO XmlReader: no entity expansion (billion-laughs immunity)") {
  // A DOCTYPE declaring entities must NOT be expanded; the reader skips the DTD
  // structurally and copies unknown &entities; verbatim. This proves immunity to
  // the entity-expansion (billion-laughs / XXE) class by construction.
  const std::string doc =
      "<!DOCTYPE net [ <!ENTITY lol \"boom\"> ]>\n"
      "<net name=\"&lol;\" version=\"11\"><layers/></net>";
  auto parsed = openvino::XmlDocument::parse(
      reinterpret_cast<const uint8_t*>(doc.data()), doc.size(), 0);
  REQUIRE(parsed);
  const auto& root = parsed->root();
  CHECK(root.name == "net");
  const std::string* nm = root.attr("name");
  REQUIRE(nm != nullptr);
  // "&lol;" is copied verbatim (NOT expanded to "boom") — no entity expansion.
  CHECK(*nm == "&lol;");
}
