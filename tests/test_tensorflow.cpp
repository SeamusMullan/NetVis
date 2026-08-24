// tests/test_tensorflow.cpp — TensorFlow GraphDef / SavedModel parser (#107).
//
// Covers the three things that make this parser trustworthy: (1) a frozen
// GraphDef becomes a wired compute graph — including a node whose output arity
// is only knowable from a ":1" consumer reference, and a control dependency;
// (2) the Const payload is recorded as an in-file offset+length and the
// payload-read counter is still 0 afterwards (the product thesis); (3) a
// SavedModel unwraps to meta_graphs[0].graph_def and its FunctionDef library
// becomes drill-down subgraphs, with the undecoded checkpoint stated honestly
// rather than faked. A truncated file must yield a clean Result error.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "engine/ModelPath.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {
const char* kFrozen = "tests/fixtures/model_frozen.pb";
const char* kSavedDir = "tests/fixtures/saved_model";

const ir::Node* find_node(const ir::Model& m, const ir::Graph& g,
                          std::string_view name) {
  for (const ir::Node& n : g.nodes) {
    if (m.str(n.name) == name) return &n;
  }
  return nullptr;
}

const ir::ValueInfo* find_value(const ir::Model& m, const ir::Graph& g,
                                std::string_view name) {
  for (const ir::ValueInfo& v : g.values) {
    if (m.str(v.name) == name) return &v;
  }
  return nullptr;
}

std::string meta(const ir::Model& m, std::string_view key) {
  for (const auto& kv : m.metadata) {
    if (m.str(kv.first) == key) return std::string(m.str(kv.second));
  }
  return {};
}
}  // namespace

TEST_CASE("TensorFlow: frozen GraphDef -> wired graph, Const offset+len, no payload reads") {
  if (!std::filesystem::exists(kFrozen)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kFrozen);
  REQUIRE(mf);
  CHECK(detect_format(*mf, "pb") == Format::TensorFlow);
  CHECK(detect_format(*mf, "") == Format::TensorFlow);  // content, not extension

  ProgressSink progress;
  auto res = tensorflow::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "tensorflow::parse returned an error");

  const ir::Model& model = *res;
  CHECK(model.str(model.format_name) == "TensorFlow");
  REQUIRE(model.has_graph);
  REQUIRE(model.graphs.size() == 1);  // no function library in a frozen graph
  const ir::Graph& g = model.graphs[0];
  CHECK(model.str(model.version_info).find("frozen GraphDef") != std::string::npos);
  CHECK(model.str(model.version_info).find("1286") != std::string::npos);

  REQUIRE(g.nodes.size() == 7);
  const ir::Node* x = find_node(model, g, "x");
  const ir::Node* mm = find_node(model, g, "matmul");
  const ir::Node* split = find_node(model, g, "split");
  const ir::Node* relu = find_node(model, g, "relu");
  REQUIRE(x != nullptr);
  REQUIRE(mm != nullptr);
  REQUIRE(split != nullptr);
  REQUIRE(relu != nullptr);
  CHECK(model.str(x->op_type) == "Placeholder");
  CHECK(model.str(mm->op_type) == "MatMul");

  // --- edges: "x" and "W" feed matmul; the Placeholder is a graph input ------
  REQUIRE(mm->inputs.count == 2);
  CHECK(model.str(g.values[g.edge_refs[mm->inputs.begin]].name) == "x");
  CHECK(model.str(g.values[g.edge_refs[mm->inputs.begin + 1]].name) == "W");
  REQUIRE(g.graph_inputs.size() == 1);
  CHECK(model.str(g.values[g.graph_inputs[0]].name) == "x");

  // --- output arity derived from a ":1" consumer reference -------------------
  CHECK(split->outputs.count == 2);
  CHECK(model.str(g.values[g.edge_refs[split->outputs.begin]].name) == "split");
  CHECK(model.str(g.values[g.edge_refs[split->outputs.begin + 1]].name) == "split:1");
  // relu consumes split:1 and carries a control dependency on biasadd; both are
  // edges, so the dependency stays visible in the graph.
  REQUIRE(relu->inputs.count == 2);
  CHECK(model.str(g.values[g.edge_refs[relu->inputs.begin]].name) == "split:1");
  CHECK(model.str(g.values[g.edge_refs[relu->inputs.begin + 1]].name) == "biasadd");

  // --- honest-unknown: the batch dim stays -1, never guessed -----------------
  const ir::ValueInfo* vx = find_value(model, g, "x");
  REQUIRE(vx != nullptr);
  REQUIRE(vx->shape.size() == 2);
  CHECK(vx->shape[0] == -1);
  CHECK(vx->shape[1] == 4);
  CHECK(vx->dtype == ir::DType::F32);
  // _output_shapes fills the intermediate edge.
  const ir::ValueInfo* vmm = find_value(model, g, "matmul");
  REQUIRE(vmm != nullptr);
  REQUIRE(vmm->shape.size() == 2);
  CHECK(vmm->shape[0] == -1);
  CHECK(vmm->shape[1] == 2);

  // --- the Const payload: offset+len only, bytes untouched -------------------
  REQUIRE(g.initializers.size() == 2);
  const ir::TensorRef* w = nullptr;
  for (const ir::TensorRef& t : g.initializers) {
    if (model.str(t.name) == "W") w = &t;
  }
  REQUIRE(w != nullptr);
  CHECK(w->dtype == ir::DType::F32);
  REQUIRE(w->shape.size() == 2);
  CHECK(w->shape[0] == 4);
  CHECK(w->shape[1] == 2);
  CHECK(w->byte_len == 32);  // 8 floats
  CHECK(w->file_offset != UINT64_MAX);
  CHECK(w->file_offset + w->byte_len <= mf->size());
  CHECK(w->elem_count() == 8);

  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("TensorFlow: SavedModel -> meta_graph + FunctionDef subgraphs, honest checkpoint") {
  const std::string pb = std::string(kSavedDir) + "/saved_model.pb";
  if (!std::filesystem::exists(pb)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  ByteReader::payload_read_counter() = 0;

  // The user opens the DIRECTORY; ModelPath resolves it to saved_model.pb so
  // variables/ stays a sibling of the mapped file.
  ResolvedModelPath rp = resolve_model_path(kSavedDir);
  CHECK(rp.display_path == kSavedDir);
  CHECK(std::filesystem::path(rp.map_path).filename() == "saved_model.pb");

  auto mf = MappedFile::open(rp.map_path);
  REQUIRE(mf);
  // A SavedModel's top level mimics an ONNX ModelProto; detection must not
  // hand it to the ONNX parser.
  CHECK(detect_format(*mf, "pb") == Format::TensorFlow);
  CHECK(detect_format(*mf, "") == Format::TensorFlow);

  ProgressSink progress;
  auto res = tensorflow::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "tensorflow::parse returned an error");
  const ir::Model& model = *res;

  CHECK(model.str(model.version_info).find("SavedModel") != std::string::npos);
  CHECK(meta(model, "tags") == "serve");
  CHECK(meta(model, "tensorflow_version") == "2.15.0");
  CHECK(model.str(model.producer) == "TensorFlow 2.15.0");
  CHECK(meta(model, "saved_model_schema_version") == "1");
  CHECK(meta(model, "functions") == "1");
  // The checkpoint is present and NOT decoded — the parser says so rather than
  // inventing offsets for tensors it cannot locate.
  CHECK(meta(model, "variables") == "checkpoint present (payloads not decoded)");

  // main graph + the one library FunctionDef.
  REQUIRE(model.graphs.size() == 2);
  const ir::Graph& main = model.graphs[0];
  const ir::Node* call = find_node(model, main, "StatefulPartitionedCall");
  REQUIRE(call != nullptr);
  // The `f` attribute links the stub to the graph holding the real body.
  REQUIRE(call->subgraph == 1);

  const ir::Graph& fn = model.graphs[1];
  CHECK(model.str(fn.name) == "__inference_serve_17");
  REQUIRE(fn.nodes.size() == 2);
  // The signature's input_arg has no producing node: it is a graph input.
  REQUIRE(fn.graph_inputs.size() == 1);
  CHECK(model.str(fn.values[fn.graph_inputs[0]].name) == "inp");
  CHECK(fn.values[fn.graph_inputs[0]].producer == -1);
  // ret maps "mul:z:0" -> node `mul`, slot 0 (the FunctionDef three-part form).
  REQUIRE(fn.graph_outputs.size() == 1);
  CHECK(model.str(fn.values[fn.graph_outputs[0]].name) == "mul");
  // The function body's Const payload is recorded, not read.
  REQUIRE(fn.initializers.size() == 1);
  CHECK(fn.initializers[0].byte_len == 8);
  CHECK(fn.initializers[0].file_offset + 8 <= mf->size());

  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("TensorFlow: truncated GraphDef -> clean error, no crash") {
  if (!std::filesystem::exists(kFrozen)) return;

  std::ifstream in(kFrozen, std::ios::binary);
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
  REQUIRE(bytes.size() > 64);

  ByteReader::payload_read_counter() = 0;
  const std::filesystem::path p =
      std::filesystem::temp_directory_path() / "nv_tf_trunc.pb";
  // Cut MID-MESSAGE at a few points; every one must be an error, never a crash.
  // (Protobuf has no terminator, so a cut that lands exactly on a top-level
  // record boundary is a legitimately shorter GraphDef, not a truncation — these
  // offsets all fall inside a NodeDef's declared length.)
  for (size_t n : {size_t{16}, bytes.size() / 3, bytes.size() - 10}) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(n));
    out.close();
    auto mf = MappedFile::open(p.string());
    REQUIRE(mf);
    ProgressSink progress;
    auto res = tensorflow::parse(*mf, progress);
    CHECK_FALSE(res);  // a truncated protobuf must not parse "successfully"
  }
  std::filesystem::remove(p);
  CHECK(ByteReader::payload_read_counter() == 0);
}
