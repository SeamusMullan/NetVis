// SPDX-License-Identifier: Apache-2.0
// tests/test_caffe.cpp — Caffe .prototxt + .caffemodel parser (#138, #109).
//
// Covers: the modern pair (text topology + sibling binary weights) end to end —
// SSA chaining of in-place layers, exact op mapping, derived attributes, CEIL
// pooling shapes, initializers pointing into the sibling, lazy weight decode;
// the .caffemodel alone; the prototxt with no pairable sibling; the legacy V1
// schema with its phase filter and `_deploy` pairing, and text/binary
// equivalence; every pairing rule and refusal; determinism; fork / unknown
// types; the collision guard proving no unvetted layer inherits ONNX semantics;
// and hostile inputs. Every structural parse leaves the payload-read counter at 0.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "engine/CostModel.h"
#include "engine/OpCategory.h"
#include "engine/ShapeInference.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/caffe/CaffeNet.h"
#include "parsers/caffe/CaffeOps.h"
#include "parsers/caffe/CaffeSchema.h"
#include "temp_file_guard.h"

using namespace netvis;
namespace fs = std::filesystem;

namespace {

const char* kProto = "tests/fixtures/model_caffe.prototxt";
const char* kModel = "tests/fixtures/model_caffe.caffemodel";
const char* kAlone = "tests/fixtures/model_caffe_alone.prototxt";
const char* kV1Proto = "tests/fixtures/model_caffe_v1_deploy.prototxt";
const char* kV1Model = "tests/fixtures/model_caffe_v1.caffemodel";

bool have_fixtures() {
  for (const char* p : {kProto, kModel, kAlone, kV1Proto, kV1Model}) {
    if (!fs::exists(p)) {
      WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
      return false;
    }
  }
  return true;
}

const ir::Node* find_node(const ir::Model& m, const ir::Graph& g, std::string_view name) {
  for (const ir::Node& n : g.nodes)
    if (m.str(n.name) == name) return &n;
  return nullptr;
}

const ir::ValueInfo* find_value(const ir::Model& m, const ir::Graph& g, std::string_view name) {
  for (const ir::ValueInfo& v : g.values)
    if (m.str(v.name) == name) return &v;
  return nullptr;
}

std::string meta(const ir::Model& m, std::string_view key) {
  for (const auto& kv : m.metadata)
    if (m.str(kv.first) == key) return std::string(m.str(kv.second));
  return {};
}

bool starts_with(const std::string& s, const std::string& p) { return s.rfind(p, 0) == 0; }

std::vector<int64_t> shape_of(const ir::ValueInfo* v) {
  if (!v) return {-999};
  return std::vector<int64_t>(v->shape.begin(), v->shape.end());
}

std::vector<int64_t> shape_named(const ir::Model& m, std::string_view name) {
  return shape_of(find_value(m, m.graphs[0], name));
}

const ir::AttrValue* attr(const ir::Model& m, const ir::Graph& g, const ir::Node& n,
                          std::string_view name) {
  for (uint32_t i = 0; i < n.attributes.count; ++i) {
    const ir::Attribute& a = g.attributes[n.attributes.begin + i];
    if (m.str(a.name) == name) return &a.value;
  }
  return nullptr;
}

std::vector<std::string> input_names(const ir::Model& m, const ir::Graph& g, const ir::Node& n) {
  std::vector<std::string> out;
  for (uint32_t k = 0; k < n.inputs.count; ++k)
    out.emplace_back(m.str(g.values[g.edge_refs[n.inputs.begin + k]].name));
  return out;
}

std::vector<std::string> output_names(const ir::Model& m, const ir::Graph& g, const ir::Node& n) {
  std::vector<std::string> out;
  for (uint32_t k = 0; k < n.outputs.count; ++k)
    out.emplace_back(m.str(g.values[g.edge_refs[n.outputs.begin + k]].name));
  return out;
}

std::vector<std::string> value_names(const ir::Model& m, const std::vector<uint32_t>& idx) {
  std::vector<std::string> out;
  for (uint32_t vi : idx) out.emplace_back(m.str(m.graphs[0].values[vi].name));
  return out;
}

const ir::TensorRef* find_init(const ir::Model& m, std::string_view name) {
  for (const ir::TensorRef& t : m.graphs[0].initializers)
    if (m.str(t.name) == name) return &t;
  return nullptr;
}

std::string read_file(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_file(const fs::path& p, const std::string& bytes) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

Result<ir::Model> parse_path(const std::string& path) {
  auto mf = MappedFile::open(path);
  if (!mf) return mf.error();
  ProgressSink progress;
  return caffe::parse(*mf, progress);
}

// A fresh directory per test, so pairing rule 3 never sees stray files.
//
// Every file placed here gets a TempFileGuard, registered BEFORE the file is
// written: Windows cannot delete a file that is still mapped, so declare the
// TempDir before any MappedFile (or parse) that reads its files, and the guards
// delete each file only after those mappings are gone. Members are destroyed in
// reverse order, so the file guards run first and the (by then empty) directory
// is removed last.
struct TempDir {
 private:
  struct DirRemover {
    fs::path dir;
    ~DirRemover() {
      std::error_code ec;
      fs::remove_all(dir, ec);   // best effort, never throws
    }
  };
  DirRemover remover_;                               // destroyed last
  std::vector<netvis_test::TempFileGuard> files_;   // destroyed before it

 public:
  fs::path path;

  explicit TempDir(const std::string& tag) {
    static int counter = 0;
    path = fs::temp_directory_path() / ("nv_caffe_" + tag + "_" + std::to_string(++counter));
    remover_.dir = path;
    std::error_code ec;
    fs::remove_all(path, ec);
    fs::create_directories(path, ec);
  }
  std::string file(const std::string& name) const { return (path / name).string(); }
  void copy(const char* src, const std::string& name) {
    files_.emplace_back(file(name));
    std::error_code ec;
    fs::copy_file(src, path / name, fs::copy_options::overwrite_existing, ec);
    REQUIRE_FALSE(ec);
  }
  void put(const std::string& name, const std::string& bytes) {
    files_.emplace_back(file(name));
    write_file(path / name, bytes);
  }
};

// ---- protobuf wire builders (tests only) ------------------------------------------
std::string varint(uint64_t v) {
  std::string s;
  do {
    uint8_t b = static_cast<uint8_t>(v & 0x7F);
    v >>= 7;
    if (v) b |= 0x80;
    s.push_back(static_cast<char>(b));
  } while (v);
  return s;
}
std::string tag(uint32_t field, uint32_t wire) { return varint((uint64_t{field} << 3) | wire); }
std::string pb_varint(uint32_t f, uint64_t v) { return tag(f, 0) + varint(v); }
std::string pb_len(uint32_t f, const std::string& body) {
  return tag(f, 2) + varint(body.size()) + body;
}
std::string pb_f32(uint32_t f, float x) {
  std::string s = tag(f, 5);
  char b[4];
  std::memcpy(b, &x, 4);
  s.append(b, 4);
  return s;
}
std::string layer_bin(const std::string& name, const std::string& type,
                      const std::vector<std::string>& bottoms,
                      const std::vector<std::string>& tops, const std::string& extra = {}) {
  std::string b = pb_len(1, name) + pb_len(2, type);
  for (const auto& x : bottoms) b += pb_len(3, x);
  for (const auto& x : tops) b += pb_len(4, x);
  return b + extra;
}

// Everything a viewer could observe, as one string (determinism / equivalence).
std::string node_attrs(const ir::Model& m, const ir::Graph& g, const ir::Node& n) {
  std::vector<std::string> out;
  for (uint32_t i = 0; i < n.attributes.count; ++i) {
    const ir::Attribute& a = g.attributes[n.attributes.begin + i];
    std::ostringstream s;
    s << m.str(a.name) << "=" << static_cast<int>(a.value.kind) << ":";
    switch (a.value.kind) {
      case ir::AttrValue::Kind::Int: s << a.value.i; break;
      case ir::AttrValue::Kind::Float: s << a.value.f; break;
      case ir::AttrValue::Kind::String: s << m.str(a.value.s); break;
      case ir::AttrValue::Kind::Ints:
        for (int64_t v : a.value.ints) s << v << ",";
        break;
      case ir::AttrValue::Kind::Floats:
        for (double v : a.value.floats) s << v << ",";
        break;
      case ir::AttrValue::Kind::Strings:
        for (StringId v : a.value.strings) s << m.str(v) << ",";
        break;
      default: break;
    }
    out.push_back(s.str());
  }
  return [&] {
    std::string r;
    for (const auto& x : out) r += x + ";";
    return r;
  }();
}

std::string signature(const ir::Model& m) {
  const ir::Graph& g = m.graphs[0];
  std::ostringstream s;
  s << m.str(m.format_name) << "|" << m.str(m.version_info) << "\n";
  for (const ir::Node& n : g.nodes) {
    s << m.str(n.op_type) << " " << m.str(n.name) << " in:";
    for (const auto& x : input_names(m, g, n)) s << x << ",";
    s << " out:";
    for (const auto& x : output_names(m, g, n)) s << x << ",";
    s << " " << node_attrs(m, g, n) << "\n";
  }
  for (const ir::ValueInfo& v : g.values) {
    s << m.str(v.name) << ":" << static_cast<int>(v.dtype) << ":" << v.producer << ":";
    for (int64_t d : v.shape) s << d << ",";
    s << "\n";
  }
  for (const ir::TensorRef& t : g.initializers) {
    s << m.str(t.name) << "@" << t.file_offset << "+" << t.byte_len << ":"
      << m.str(t.external_path) << ":" << static_cast<int>(t.dtype) << "\n";
  }
  for (uint32_t vi : g.graph_inputs) s << "in " << vi << "\n";
  for (uint32_t vi : g.graph_outputs) s << "out " << vi << "\n";
  for (const auto& kv : m.metadata) s << m.str(kv.first) << "=" << m.str(kv.second) << "\n";
  return s.str();
}

const std::vector<int64_t> kNone = {};

}  // namespace

// ---------------------------------------------------------------------------
// A. Modern prototxt, paired with its sibling caffemodel
// ---------------------------------------------------------------------------
TEST_CASE("Caffe A: modern prototxt pairs with model_caffe.caffemodel") {
  if (!have_fixtures()) return;
  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kProto);
  REQUIRE(mf);
  DetectReason reason = DetectReason::None;
  CHECK(detect_format(*mf, "prototxt", reason) == Format::Caffe);
  CHECK(reason == DetectReason::Structure);
  CHECK(detect_format(*mf, "", reason) == Format::Caffe);
  CHECK(reason == DetectReason::Structure);
  CHECK(std::string(format_name(Format::Caffe)) == "Caffe");

  ProgressSink progress;
  auto res = parse_model(*mf, "prototxt", progress);
  REQUIRE_MESSAGE(res, (res ? "" : res.error().message));
  const ir::Model& m = *res;
  const ir::Graph& g = m.graphs[0];

  // A.2 labels
  CHECK(m.str(m.format_name) == "Caffe");
  CHECK(m.graphs.size() == 1);
  CHECK(m.has_graph);
  CHECK(m.str(m.version_info).find("prototxt") != std::string::npos);
  CHECK(m.str(m.version_info).find("LayerParameter") != std::string::npos);
  CHECK(meta(m, "name") == "nv_caffe_tiny");
  CHECK(meta(m, "schema") == "LayerParameter (layer)");
  CHECK(meta(m, "source") == "prototxt (text)");
  CHECK_MESSAGE(starts_with(meta(m, "weights"), "paired with model_caffe.caffemodel: 4 layers"),
                meta(m, "weights"));
  CHECK(starts_with(meta(m, "activation_dtype"), "f32"));

  // A.3 ops
  const std::vector<std::string> ops = {"Input", "Conv", "Relu", "MaxPool", "LRN", "Conv",
                                        "Conv", "Concat", "Add", "Flatten", "Gemm", "Softmax"};
  REQUIRE(g.nodes.size() == ops.size());
  for (size_t i = 0; i < ops.size(); ++i) CHECK(m.str(g.nodes[i].op_type) == ops[i]);
  const ir::AttrValue* ct = attr(m, g, g.nodes[1], "caffe.type");
  REQUIRE(ct != nullptr);
  CHECK(m.str(ct->s) == "Convolution");

  // A.4 SSA: the in-place ReLU consumes conv1 and produces conv1#1.
  const ir::Node* conv1 = find_node(m, g, "conv1");
  const ir::Node* relu1 = find_node(m, g, "relu1");
  const ir::Node* pool1 = find_node(m, g, "pool1");
  REQUIRE(conv1 != nullptr);
  REQUIRE(relu1 != nullptr);
  REQUIRE(pool1 != nullptr);
  CHECK(input_names(m, g, *relu1) == std::vector<std::string>{"conv1"});
  CHECK(output_names(m, g, *relu1) == std::vector<std::string>{"conv1#1"});
  CHECK(find_value(m, g, "conv1")->producer == 1);
  CHECK(find_value(m, g, "conv1#1")->producer == 2);
  CHECK(input_names(m, g, *pool1) == std::vector<std::string>{"conv1#1"});

  // A.5 global invariants: no self-loop, one producer per value.
  std::vector<int> producers(g.values.size(), 0);
  for (uint32_t ni = 0; ni < g.nodes.size(); ++ni) {
    const ir::Node& n = g.nodes[ni];
    for (uint32_t o = 0; o < n.outputs.count; ++o) {
      const uint32_t ov = g.edge_refs[n.outputs.begin + o];
      ++producers[ov];
      CHECK(g.values[ov].producer == static_cast<int32_t>(ni));
      for (uint32_t i = 0; i < n.inputs.count; ++i)
        CHECK(g.edge_refs[n.inputs.begin + i] != ov);
    }
  }
  for (uint32_t vi = 0; vi < g.values.size(); ++vi) {
    CHECK(producers[vi] <= 1);
    if (g.values[vi].producer >= 0) CHECK(producers[vi] == 1);
  }

  // A.6 params follow the bottoms.
  CHECK(input_names(m, g, *conv1) ==
        std::vector<std::string>{"data", "conv1/weight", "conv1/bias"});

  // A.7 shapes (pool1 [1,4,4,4] proves CEIL rounding; FLOOR would give 3).
  CHECK(shape_named(m, "data") == std::vector<int64_t>{1, 3, 8, 8});
  CHECK(shape_named(m, "conv1") == std::vector<int64_t>{1, 4, 8, 8});
  CHECK(shape_named(m, "conv1#1") == std::vector<int64_t>{1, 4, 8, 8});
  CHECK(shape_named(m, "pool1") == std::vector<int64_t>{1, 4, 4, 4});
  CHECK(shape_named(m, "norm1") == std::vector<int64_t>{1, 4, 4, 4});
  CHECK(shape_named(m, "b1") == std::vector<int64_t>{1, 2, 4, 4});
  CHECK(shape_named(m, "b2") == std::vector<int64_t>{1, 2, 4, 4});
  CHECK(shape_named(m, "cat") == std::vector<int64_t>{1, 4, 4, 4});
  CHECK(shape_named(m, "sum") == std::vector<int64_t>{1, 4, 4, 4});
  CHECK(shape_named(m, "flat") == std::vector<int64_t>{1, 64});
  CHECK(shape_named(m, "fc") == std::vector<int64_t>{1, 3});
  CHECK(shape_named(m, "prob") == std::vector<int64_t>{1, 3});
  for (const char* v : {"data", "conv1", "conv1#1", "pool1", "norm1", "cat", "sum", "flat",
                        "fc", "prob"})
    CHECK_MESSAGE(find_value(m, g, v)->dtype == ir::DType::F32, v);

  // A.8 derived + native attributes.
  auto ints = [&](const ir::Node* n, const char* a) {
    const ir::AttrValue* v = attr(m, g, *n, a);
    return v && v->kind == ir::AttrValue::Kind::Ints ? v->ints : std::vector<int64_t>{-999};
  };
  auto int1 = [&](const ir::Node* n, const char* a) -> int64_t {
    const ir::AttrValue* v = attr(m, g, *n, a);
    return v && v->kind == ir::AttrValue::Kind::Int ? v->i : -999;
  };
  CHECK(ints(conv1, "kernel_shape") == std::vector<int64_t>{3, 3});
  CHECK(ints(conv1, "strides") == std::vector<int64_t>{1, 1});
  CHECK(ints(conv1, "pads") == std::vector<int64_t>{1, 1, 1, 1});
  CHECK(ints(conv1, "dilations") == std::vector<int64_t>{1, 1});
  CHECK(int1(conv1, "group") == 1);
  CHECK(int1(pool1, "ceil_mode") == 1);
  CHECK(int1(find_node(m, g, "fc"), "transB") == 1);
  CHECK(int1(find_node(m, g, "cat"), "axis") == 1);
  CHECK(int1(conv1, "convolution_param.num_output") == 4);
  const ir::AttrValue* pool = attr(m, g, *pool1, "pooling_param.pool");
  REQUIRE(pool != nullptr);
  CHECK(pool->kind == ir::AttrValue::Kind::String);
  CHECK(m.str(pool->s) == "MAX");
  const ir::AttrValue* alpha = attr(m, g, *find_node(m, g, "norm1"), "lrn_param.alpha");
  REQUIRE(alpha != nullptr);
  CHECK(alpha->kind == ir::AttrValue::Kind::Float);
  CHECK(alpha->f == doctest::Approx(1e-4).epsilon(1e-6));
  // Repeated scalars are lists even with one element.
  CHECK(ints(conv1, "convolution_param.kernel_size") == std::vector<int64_t>{3});

  // A.9 initializers point into the sibling.
  const std::vector<std::string> inits = {"conv1/weight", "conv1/bias", "b1/weight", "b1/bias",
                                          "b2/weight",    "b2/bias",    "fc/weight", "fc/bias"};
  REQUIRE(g.initializers.size() == inits.size());
  const uint64_t sib_size = fs::file_size(kModel);
  for (size_t i = 0; i < inits.size(); ++i) {
    const ir::TensorRef& t = g.initializers[i];
    CHECK(m.str(t.name) == inits[i]);
    CHECK(m.str(t.external_path) == "model_caffe.caffemodel");
    CHECK(t.file_offset != UINT64_MAX);
    CHECK(t.file_offset + t.byte_len <= sib_size);
    CHECK(t.byte_len == 4 * static_cast<uint64_t>(t.elem_count()));
    CHECK(t.dtype == ir::DType::F32);
  }
  CHECK(std::vector<int64_t>(g.initializers[0].shape.begin(), g.initializers[0].shape.end()) ==
        std::vector<int64_t>{4, 3, 3, 3});

  // A.10 graph boundary.
  CHECK(value_names(m, g.graph_inputs) == std::vector<std::string>{"data"});
  CHECK(value_names(m, g.graph_outputs) == std::vector<std::string>{"prob"});

  // A.11 zero payload reads.
  CHECK(ByteReader::payload_read_counter() == 0);

  // E. The weights are readable through the pair: exactly one decode, from the sibling.
  const ir::TensorRef* bias = find_init(m, "conv1/bias");
  REQUIRE(bias != nullptr);
  auto st = compute_tensor_stats(*bias, *mf, "tests/fixtures", &m);
  REQUIRE_MESSAGE(st, (st ? "" : st.error().message));
  CHECK(st->count == 4);
  CHECK(st->min == 1.0);
  CHECK(st->max == 4.0);
  CHECK(st->mean == doctest::Approx(2.5));
  CHECK(ByteReader::payload_read_counter() == 1);

  // F. Cost through the mapped ops.
  const CostReport cr = compute_cost(m, 0);
  REQUIRE(cr.per_node.size() == g.nodes.size());
  auto node_index = [&](const char* name) {
    return static_cast<size_t>(find_node(m, g, name) - g.nodes.data());
  };
  CHECK(cr.per_node[node_index("conv1")].flops_known);
  CHECK(cr.per_node[node_index("conv1")].flops == 13824);   // 2 x 256 x 3 x 9
  CHECK(cr.per_node[node_index("pool1")].flops_known);
  CHECK(cr.per_node[node_index("pool1")].flops == 576);     // 64 x 9
  CHECK(cr.per_node[node_index("fc")].flops_known);
  CHECK(cr.per_node[node_index("fc")].flops == 384);        // 2 x 3 x 64
}

// ---------------------------------------------------------------------------
// B. The caffemodel alone
// ---------------------------------------------------------------------------
TEST_CASE("Caffe B: model_caffe.caffemodel opens on its own") {
  if (!have_fixtures()) return;
  ByteReader::payload_read_counter() = 0;
  auto mf = MappedFile::open(kModel);
  REQUIRE(mf);
  for (const char* ext : {"", "caffemodel", "onnx", "pb"}) {
    DetectReason reason = DetectReason::None;
    CHECK_MESSAGE(detect_format(*mf, ext, reason) == Format::Caffe, ext);
    CHECK(reason == DetectReason::Structure);
  }
  ProgressSink progress;
  auto res = caffe::parse(*mf, progress);
  REQUIRE_MESSAGE(res, (res ? "" : res.error().message));
  const ir::Model& m = *res;
  const ir::Graph& g = m.graphs[0];

  CHECK(g.nodes.size() == 11);   // no Input node: the input is net-level
  REQUIRE(g.graph_inputs.size() == 1);
  const ir::ValueInfo& data = g.values[g.graph_inputs[0]];
  CHECK(m.str(data.name) == "data");
  CHECK(data.producer == -1);
  CHECK(shape_of(&data) == std::vector<int64_t>{1, 3, 8, 8});
  CHECK(data.dtype == ir::DType::F32);
  CHECK(shape_named(m, "conv1#1") == std::vector<int64_t>{1, 4, 8, 8});
  CHECK(shape_named(m, "pool1") == std::vector<int64_t>{1, 4, 4, 4});
  CHECK(shape_named(m, "cat") == std::vector<int64_t>{1, 4, 4, 4});
  CHECK(shape_named(m, "flat") == std::vector<int64_t>{1, 64});
  CHECK(shape_named(m, "prob") == std::vector<int64_t>{1, 3});
  CHECK(value_names(m, g.graph_outputs) == std::vector<std::string>{"prob"});

  REQUIRE(g.initializers.size() == 8);
  for (const ir::TensorRef& t : g.initializers) {
    CHECK_FALSE(t.external_path.valid());
    CHECK(t.file_offset + t.byte_len <= mf->size());
  }
  CHECK(meta(m, "source") == "caffemodel (binary)");
  CHECK(meta(m, "weights") == "embedded (8 blobs)");
  CHECK(m.str(m.version_info) == "Caffe caffemodel, LayerParameter");
  for (const ir::Node& n : g.nodes) {
    const ir::AttrValue* ph = attr(m, g, n, "phase");
    REQUIRE(ph != nullptr);
    CHECK(m.str(ph->s) == "TEST");
  }
  CHECK(ByteReader::payload_read_counter() == 0);

  // The embedded bias decodes from the file itself.
  auto st = compute_tensor_stats(*find_init(m, "conv1/bias"), *mf, "tests/fixtures", &m);
  REQUIRE(st);
  CHECK(st->max == 4.0);
}

// ---------------------------------------------------------------------------
// C. A prototxt with no pairable sibling
// ---------------------------------------------------------------------------
TEST_CASE("Caffe C: a prototxt alone has shapes only up to the first convolution") {
  if (!have_fixtures()) return;
  ByteReader::payload_read_counter() = 0;
  auto res = parse_path(kAlone);
  REQUIRE_MESSAGE(res, (res ? "" : res.error().message));
  const ir::Model& m = *res;
  const ir::Graph& g = m.graphs[0];
  CHECK(g.nodes.size() == 12);
  CHECK(g.initializers.empty());
  CHECK(find_node(m, g, "conv1")->inputs.count == 1);
  CHECK(shape_named(m, "data") == std::vector<int64_t>{1, 3, 8, 8});
  CHECK(find_value(m, g, "data")->dtype == ir::DType::Unknown);
  CHECK(shape_named(m, "conv1") == kNone);
  CHECK(shape_named(m, "pool1") == kNone);
  CHECK(shape_named(m, "prob") == kNone);
  CHECK_MESSAGE(starts_with(meta(m, "weights"), "ambiguous: 2 .caffemodel files"),
                meta(m, "weights"));
  CHECK(meta(m, "activation_dtype") == "unknown (no weights)");
  const CostReport cr = compute_cost(m, 0);
  CHECK_FALSE(cr.per_node[1].flops_known);
  CHECK(ByteReader::payload_read_counter() == 0);
}

// ---------------------------------------------------------------------------
// D. Legacy V1 pair
// ---------------------------------------------------------------------------
TEST_CASE("Caffe D: V1 prototxt + caffemodel, phase filter, _deploy pairing, equivalence") {
  if (!have_fixtures()) return;
  ByteReader::payload_read_counter() = 0;
  auto res = parse_path(kV1Proto);
  REQUIRE_MESSAGE(res, (res ? "" : res.error().message));
  const ir::Model& m = *res;
  const ir::Graph& g = m.graphs[0];

  CHECK(meta(m, "schema") == "V1LayerParameter (layers)");
  const ir::Node* conv1 = find_node(m, g, "conv1");
  REQUIRE(conv1 != nullptr);
  CHECK(m.str(attr(m, g, *conv1, "caffe.v1_type")->s) == "CONVOLUTION");
  CHECK(m.str(attr(m, g, *conv1, "caffe.type")->s) == "Convolution");
  const ir::Node* prob = find_node(m, g, "prob");
  REQUIRE(prob != nullptr);
  CHECK(m.str(prob->op_type) == "Softmax");
  CHECK(m.str(attr(m, g, *prob, "caffe.v1_type")->s) == "SOFTMAX");

  CHECK_MESSAGE(starts_with(meta(m, "phase_filter"), "TEST (excluded 1 layer: loss)"),
                meta(m, "phase_filter"));
  CHECK(g.nodes.size() == 5);
  CHECK(find_node(m, g, "loss") == nullptr);
  CHECK(find_value(m, g, "label") == nullptr);

  CHECK(value_names(m, g.graph_inputs) == std::vector<std::string>{"data"});
  CHECK(find_value(m, g, "data")->producer == -1);
  CHECK(shape_named(m, "data") == std::vector<int64_t>{1, 3, 8, 8});
  CHECK(shape_named(m, "conv1") == std::vector<int64_t>{1, 2, 6, 6});
  CHECK(shape_named(m, "conv1#1") == std::vector<int64_t>{1, 2, 6, 6});
  CHECK(shape_named(m, "pool1") == std::vector<int64_t>{1, 2, 3, 3});
  const ir::Node* pool1 = find_node(m, g, "pool1");
  CHECK(m.str(pool1->op_type) == "AveragePool");
  CHECK(attr(m, g, *pool1, "ceil_mode")->i == 1);
  CHECK(shape_named(m, "fc1") == kNone);   // Gemm needs a rank-2 input
  CHECK(shape_named(m, "prob") == kNone);

  CHECK_MESSAGE(starts_with(meta(m, "weights"), "paired with model_caffe_v1.caffemodel: 2 layers"),
                meta(m, "weights"));
  CHECK(std::vector<int64_t>(find_init(m, "fc1/weight")->shape.begin(),
                             find_init(m, "fc1/weight")->shape.end()) ==
        std::vector<int64_t>{1, 1, 3, 18});
  CHECK(std::vector<int64_t>(find_init(m, "conv1/weight")->shape.begin(),
                             find_init(m, "conv1/weight")->shape.end()) ==
        std::vector<int64_t>{2, 3, 3, 3});

  // F. V1 costs.
  const CostReport cr = compute_cost(m, 0);
  auto idx = [&](const char* name) {
    return static_cast<size_t>(find_node(m, g, name) - g.nodes.data());
  };
  CHECK(cr.per_node[idx("conv1")].flops_known);
  CHECK(cr.per_node[idx("conv1")].flops == 3888);   // 2 x 72 x 3 x 9
  CHECK(cr.per_node[idx("pool1")].flops_known);
  CHECK(cr.per_node[idx("pool1")].flops == 72);     // 18 x 4
  CHECK_FALSE(cr.per_node[idx("fc1")].flops_known);

  // D.5 the caffemodel alone describes the same graph.
  auto bres = parse_path(kV1Model);
  REQUIRE_MESSAGE(bres, (bres ? "" : bres.error().message));
  const ir::Model& b = *bres;
  const ir::Graph& bg = b.graphs[0];
  REQUIRE(bg.nodes.size() == g.nodes.size());
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    CHECK(m.str(g.nodes[i].op_type) == b.str(bg.nodes[i].op_type));
    CHECK(m.str(g.nodes[i].name) == b.str(bg.nodes[i].name));
    CHECK(input_names(m, g, g.nodes[i]) == input_names(b, bg, bg.nodes[i]));
    CHECK(output_names(m, g, g.nodes[i]) == output_names(b, bg, bg.nodes[i]));
    auto set_of = [](const std::string& s) {
      std::set<std::string> out;
      std::string cur;
      for (char c : s) {
        if (c == ';') { out.insert(cur); cur.clear(); }
        else cur.push_back(c);
      }
      return out;
    };
    CHECK(set_of(node_attrs(m, g, g.nodes[i])) == set_of(node_attrs(b, bg, bg.nodes[i])));
  }
  REQUIRE(bg.values.size() == g.values.size());
  for (size_t i = 0; i < g.values.size(); ++i) {
    CHECK(m.str(g.values[i].name) == b.str(bg.values[i].name));
    CHECK(shape_of(&g.values[i]) == shape_of(&bg.values[i]));
  }
  REQUIRE(bg.initializers.size() == g.initializers.size());
  for (size_t i = 0; i < g.initializers.size(); ++i) {
    const ir::TensorRef& x = g.initializers[i];
    const ir::TensorRef& y = bg.initializers[i];
    CHECK(m.str(x.name) == b.str(y.name));
    CHECK(x.file_offset == y.file_offset);
    CHECK(x.byte_len == y.byte_len);
    CHECK(std::vector<int64_t>(x.shape.begin(), x.shape.end()) ==
          std::vector<int64_t>(y.shape.begin(), y.shape.end()));
    CHECK(m.str(x.external_path) == "model_caffe_v1.caffemodel");
    CHECK_FALSE(y.external_path.valid());
  }
  CHECK(ByteReader::payload_read_counter() == 0);
}

// ---------------------------------------------------------------------------
// G. Pairing rules, each in a fresh directory
// ---------------------------------------------------------------------------
TEST_CASE("Caffe G: sibling pairing rules and refusals") {
  if (!have_fixtures()) return;
  auto weights_of = [](const std::string& path) -> std::string {
    auto r = parse_path(path);
    REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
    return meta(*r, "weights");
  };

  SUBCASE("rule 1: same stem") {
    TempDir d("rule1");
    d.copy(kProto, "net.prototxt");
    d.copy(kModel, "net.caffemodel");
    CHECK(starts_with(weights_of(d.file("net.prototxt")), "paired with net.caffemodel: 4 layers"));
  }
  SUBCASE("rule 2: _deploy / -deploy / .deploy") {
    for (const char* stem : {"net_deploy", "net-deploy", "net.deploy"}) {
      TempDir d("rule2");
      d.copy(kProto, std::string(stem) + ".prototxt");
      d.copy(kModel, "net.caffemodel");
      d.copy(kModel, "other.caffemodel");   // rule 3 alone would be ambiguous
      CHECK_MESSAGE(starts_with(weights_of(d.file(std::string(stem) + ".prototxt")),
                                "paired with net.caffemodel: 4 layers"),
                    stem);
    }
  }
  SUBCASE("rule 3: the only .caffemodel in the folder") {
    TempDir d("rule3");
    d.copy(kProto, "deploy.prototxt");
    d.copy(kModel, "weights.caffemodel");
    CHECK(starts_with(weights_of(d.file("deploy.prototxt")),
                      "paired with weights.caffemodel: 4 layers"));
  }
  SUBCASE("ambiguous: two candidates, none named after the prototxt") {
    TempDir d("ambig");
    d.copy(kProto, "deploy.prototxt");
    d.copy(kModel, "a.caffemodel");
    d.copy(kModel, "b.caffemodel");
    auto r = parse_path(d.file("deploy.prototxt"));
    REQUIRE(r);
    CHECK(r->graphs[0].initializers.empty());
    CHECK(starts_with(meta(*r, "weights"), "ambiguous: 2"));
  }
  SUBCASE("none found") {
    TempDir d("none");
    d.copy(kProto, "deploy.prototxt");
    CHECK(weights_of(d.file("deploy.prototxt")) == "none found");
  }
  SUBCASE("a sibling that is not a caffemodel is a note, not an error") {
    TempDir d("notcm");
    d.copy(kProto, "deploy.prototxt");
    d.copy(kProto, "x.caffemodel");
    CHECK(weights_of(d.file("deploy.prototxt")) ==
          "x.caffemodel is not a Caffe binary NetParameter");
  }
  SUBCASE("a truncated sibling never fails the prototxt") {
    TempDir d("trunc");
    d.copy(kProto, "deploy.prototxt");
    d.put("net.caffemodel", read_file(kModel).substr(0, 300));
    ByteReader::payload_read_counter() = 0;
    auto r = parse_path(d.file("deploy.prototxt"));
    REQUIRE(r);
    CHECK(ByteReader::payload_read_counter() == 0);
  }
  SUBCASE("a layer whose weights do not fit is left unpaired and listed") {
    TempDir d("mismatch");
    std::string text = read_file(kProto);
    const std::string from = "num_output: 4 kernel_size: 3";
    const size_t at = text.find(from);
    REQUIRE(at != std::string::npos);
    text.replace(at, from.size(), "num_output: 5 kernel_size: 3");
    d.put("net.prototxt", text);
    d.copy(kModel, "net.caffemodel");
    auto r = parse_path(d.file("net.prototxt"));
    REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
    const ir::Model& m = *r;
    const ir::Graph& g = m.graphs[0];
    CHECK(find_node(m, g, "conv1")->inputs.count == 1);
    CHECK(meta(m, "unpaired_layers") == "conv1");
    CHECK(starts_with(meta(m, "weights"), "paired with net.caffemodel: 3 layers; 1 rejected"));
    for (const char* l : {"b1", "b2", "fc"})
      CHECK_MESSAGE(find_node(m, g, l)->inputs.count == 3, l);
    CHECK(shape_named(m, "conv1") == kNone);   // no weight: nothing guessed
  }
}

// ---------------------------------------------------------------------------
// H. Determinism
// ---------------------------------------------------------------------------
TEST_CASE("Caffe H: the same file gives the same model") {
  if (!have_fixtures()) return;
  for (const char* p : {kProto, kModel, kV1Proto, kV1Model, kAlone}) {
    auto a = parse_path(p);
    auto b = parse_path(p);
    REQUIRE(a);
    REQUIRE(b);
    CHECK_MESSAGE(signature(*a) == signature(*b), p);
  }
}

// ---------------------------------------------------------------------------
// J. Unknown and fork types
// ---------------------------------------------------------------------------
TEST_CASE("Caffe J: fork and unknown layer types stay unresolved") {
  TempDir d("fork");
  const std::string input =
      "layer { name: \"data\" type: \"Input\" top: \"data\" "
      "input_param { shape { dim: 1 dim: 4 dim: 8 dim: 8 } } }\n";
  d.put("up.prototxt", input +
                           "layer { name: \"up\" type: \"Upsample\" bottom: \"data\" top: \"up\" "
                           "upsample_param { scale: 2 } }\n"
                           "layer { name: \"t\" type: \"Transpose\" bottom: \"up\" top: \"t\" }\n");
  auto r = parse_path(d.file("up.prototxt"));
  REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
  const ir::Model& m = *r;
  const ir::Graph& g = m.graphs[0];
  const ir::Node* up = find_node(m, g, "up");
  REQUIRE(up != nullptr);
  CHECK(m.str(up->op_type) == "Upsample");
  const ir::AttrValue* sc = attr(m, g, *up, "upsample_param.scale");
  REQUIRE(sc != nullptr);
  CHECK(sc->kind == ir::AttrValue::Kind::Int);
  CHECK(sc->i == 2);
  CHECK(shape_named(m, "up") == kNone);
  CHECK(m.str(find_node(m, g, "t")->op_type) == "CaffeTranspose");

  d.put("v1.prototxt",
        "input: \"data\"\nlayers { name: \"x\" type: FOO_BAR bottom: \"data\" top: \"x\" }\n");
  auto v = parse_path(d.file("v1.prototxt"));
  REQUIRE_MESSAGE(v, (v ? "" : v.error().message));
  const ir::Node* x = find_node(*v, v->graphs[0], "x");
  REQUIRE(x != nullptr);
  CHECK(v->str(x->op_type) == "V1TYPE_FOO_BAR");
  CHECK(v->str(attr(*v, v->graphs[0], *x, "caffe.v1_type")->s) == "FOO_BAR");

  // Binary V1 layer with LayerType 77.
  const std::string v1 = pb_len(4, "x") + pb_varint(5, 77) + pb_len(2, "data") + pb_len(3, "x");
  d.put("v1.caffemodel", pb_len(1, "n") + pb_len(3, "data") + pb_len(2, v1));
  auto b = parse_path(d.file("v1.caffemodel"));
  REQUIRE_MESSAGE(b, (b ? "" : b.error().message));
  CHECK(b->str(find_node(*b, b->graphs[0], "x")->op_type) == "V1TYPE_77");

  // Binary modern layer with an unknown field 222.
  const std::string lay = layer_bin("x", "ReLU", {"data"}, {"x"}, pb_len(222, "zz"));
  d.put("u.caffemodel", pb_len(1, "n") + pb_len(3, "data") + pb_len(100, lay));
  auto u = parse_path(d.file("u.caffemodel"));
  REQUIRE_MESSAGE(u, (u ? "" : u.error().message));
  CHECK(meta(*u, "undecoded_fields") == "1 binary fields not decoded");
  CHECK(u->str(find_node(*u, u->graphs[0], "x")->op_type) == "Relu");

  // Display names for types NetVis does not know.
  CHECK(caffe::safe_display_name("Upsample") == "Upsample");
  CHECK(caffe::safe_display_name("Permute") == "Permute");
  CHECK(caffe::safe_display_name("Transpose") == "CaffeTranspose");
  CHECK(caffe::safe_display_name("Identity") == "CaffeIdentity");
  CHECK(caffe::safe_display_name("ReduceAll") == "CaffeReduceAll");
  CHECK(caffe::safe_display_name("fork.Conv") == "Caffefork_Conv");
}

// ---------------------------------------------------------------------------
// K. Collision guard: no unvetted layer inherits ONNX semantics
// ---------------------------------------------------------------------------
namespace {
// One node of `op` over a known [1,4,8,8] F32 input: inference and cost must
// both stay unknown.
void check_inert(const std::string& op) {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  ir::ValueInfo x;
  x.name = m.intern("x");
  x.dtype = ir::DType::F32;
  x.shape = {1, 4, 8, 8};
  g.values.push_back(x);
  ir::ValueInfo y;
  y.name = m.intern("y");
  y.producer = 0;
  g.values.push_back(y);
  ir::Node n;
  n.op_type = m.intern(op);
  n.name = m.intern("n");
  n.inputs = {0, 1};
  n.outputs = {1, 1};
  g.edge_refs = {0, 1};
  g.nodes.push_back(n);
  g.graph_inputs = {0};
  g.graph_outputs = {1};
  infer_shapes(m, 0, nullptr);
  CHECK_MESSAGE(g.values[1].shape.empty(), op);
  const CostReport cr = compute_cost(m, 0);
  REQUIRE(cr.per_node.size() == 1);
  CHECK_FALSE_MESSAGE(cr.per_node[0].flops_known, op);
}
}  // namespace

TEST_CASE("Caffe K: every unvetted mapping is inert in shape inference and cost") {
  size_t checked = 0;
  for (size_t i = 0; i < caffe::kBvlcLayerTypeCount; ++i) {
    const caffe::BvlcLayerType& row = caffe::kBvlcLayerTypes[i];
    if (!row.fallback) continue;   // Input: always vetted
    caffe::LayerRaw L;
    L.type = row.type;
    L.has_type = true;
    for (int b = 0; b < row.test_bottoms; ++b) L.bottom.push_back("b" + std::to_string(b));
    for (int t = 0; t < row.test_tops; ++t) L.top.push_back("t" + std::to_string(t));
    ir::Model scratch;
    const caffe::MappedOp mo = caffe::map_layer(L, scratch, -1);
    CHECK_FALSE_MESSAGE(mo.vetted, row.type);
    CHECK_MESSAGE(mo.op_type == row.fallback, row.type);
    CHECK_MESSAGE(mo.derived.empty(), row.type);
    check_inert(mo.op_type);
    ++checked;
  }
  CHECK(checked + 1 == caffe::kBvlcLayerTypeCount);
  for (const char* fork : {"Upsample", "Transpose", "Identity", "Resize", "LSTM", "ReduceAll",
                           "x.Conv", "Sum"})
    check_inert(caffe::safe_display_name(fork));
  // The unresolved placeholders too.
  check_inert("V1TYPE_77");
  check_inert("UNTYPED");
}

// ---------------------------------------------------------------------------
// L. Negative and hostile inputs
// ---------------------------------------------------------------------------
TEST_CASE("Caffe L: hostile and malformed inputs") {
  TempDir d("neg");
  const std::string input =
      "layer { name: \"data\" type: \"Input\" top: \"data\" "
      "input_param { shape { dim: 1 dim: 3 dim: 8 dim: 8 } } }\n";

  SUBCASE("both generations in one net (text and binary)") {
    const std::string text = input + "layers { name: \"r\" type: RELU bottom: \"data\" top: \"r\" }\n";
    d.put("both.prototxt", text);
    auto r = parse_path(d.file("both.prototxt"));
    REQUIRE_FALSE(r);
    CHECK(r.error().message.find("both 'layer' and 'layers'") != std::string::npos);
    CHECK(r.error().offset == text.find("layers"));

    const std::string first = pb_len(1, "n") + pb_len(3, "data") +
                              pb_len(100, layer_bin("a", "ReLU", {"data"}, {"a"}));
    const std::string v1 = pb_len(4, "b") + pb_varint(5, 18) + pb_len(2, "a") + pb_len(3, "b");
    d.put("both.caffemodel", first + pb_len(2, v1));
    auto b = parse_path(d.file("both.caffemodel"));
    REQUIRE_FALSE(b);
    CHECK(b.error().message.find("both 'layer' and 'layers'") != std::string::npos);
    CHECK(b.error().offset == first.size());
  }
  SUBCASE("no layers / a solver file") {
    d.put("empty.prototxt", "name: \"x\"\n");
    auto r = parse_path(d.file("empty.prototxt"));
    REQUIRE_FALSE(r);
    CHECK(r.error().message.find("no 'layer' or 'layers'") != std::string::npos);

    d.put("solver.prototxt", "net: \"train.prototxt\"\nbase_lr: 0.01\n");
    auto mf = MappedFile::open(d.file("solver.prototxt"));
    REQUIRE(mf);
    ProgressSink progress;
    auto s = parse_model(*mf, "prototxt", progress);
    REQUIRE_FALSE(s);
    CHECK(s.error().message.find("solver") != std::string::npos);
  }
  SUBCASE("a known field with a value of the wrong kind is an error at that value") {
    const std::string big = "99999999999999999999";
    const std::string text = input +
                             "layer { name: \"c\" type: \"Convolution\" bottom: \"data\" top: \"c\"\n"
                             "  convolution_param { num_output: 2 kernel_size: " + big + " } }\n";
    d.put("big.prototxt", text);
    auto r = parse_path(d.file("big.prototxt"));
    REQUIRE_FALSE(r);
    CHECK(r.error().offset == text.find(big));
    CHECK(r.error().message.find("kernel_size") != std::string::npos);
    CHECK(r.error().message.find("line 3") != std::string::npos);

    d.put("neg.prototxt", input +
                              "layer { name: \"c\" type: \"Convolution\" bottom: \"data\" top: \"c\" "
                              "convolution_param { num_output: -1 } }\n");
    auto n = parse_path(d.file("neg.prototxt"));
    REQUIRE_FALSE(n);
    CHECK(n.error().message.find("unsigned") != std::string::npos);
  }
  SUBCASE("packed float payload with a ragged length") {
    const std::string payload = "\x01\x02\x03\x04\x05\x06";
    const std::string blob = pb_len(5, payload) + pb_len(7, pb_len(1, varint(6)));
    const std::string net = pb_len(1, "n") + pb_len(3, "data") +
                            pb_len(100, layer_bin("x", "PReLU", {"data"}, {"x"}, pb_len(7, blob)));
    d.put("ragged.caffemodel", net);
    auto r = parse_path(d.file("ragged.caffemodel"));
    REQUIRE_FALSE(r);
    CHECK(r.error().message.find("multiple of 4") != std::string::npos);
    CHECK(r.error().offset == net.find(payload));
  }
  SUBCASE("too many axes, too long a name") {
    std::string dims;
    for (int i = 0; i < 33; ++i) dims += varint(1);
    const std::string blob = pb_len(7, pb_len(1, dims));
    d.put("axes.caffemodel",
          pb_len(1, "n") + pb_len(3, "data") +
              pb_len(100, layer_bin("x", "PReLU", {"data"}, {"x"}, pb_len(7, blob))));
    auto r = parse_path(d.file("axes.caffemodel"));
    REQUIRE_FALSE(r);
    CHECK(r.error().message.find("32 axes") != std::string::npos);

    const std::string name(70000, 'n');
    d.put("name.caffemodel", pb_len(1, "n") + pb_len(3, "data") +
                                 pb_len(100, layer_bin(name, "ReLU", {"data"}, {"x"})));
    auto nm = parse_path(d.file("name.caffemodel"));
    REQUIRE_FALSE(nm);
    CHECK(nm.error().message.find("too long") != std::string::npos);
  }
  SUBCASE("an unpacked payload is abandoned, never walked or offered") {
    // data as three wire-5 floats, THEN shape: the shape is never read.
    const std::string blob =
        pb_f32(5, 1.0f) + pb_f32(5, 2.0f) + pb_f32(5, 3.0f) + pb_len(7, pb_len(1, varint(3)));
    const std::string net = pb_len(1, "n") + pb_len(3, "data") +
                            pb_len(100, layer_bin("x", "PReLU", {"data"}, {"x"}, pb_len(7, blob)));
    d.put("unpacked.caffemodel", net);
    ByteReader::payload_read_counter() = 0;
    auto r = parse_path(d.file("unpacked.caffemodel"));
    REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
    const ir::TensorRef* t = find_init(*r, "x/slope");
    REQUIRE(t != nullptr);
    CHECK(t->file_offset == UINT64_MAX);
    CHECK(t->shape.empty());
    CHECK(starts_with(meta(*r, "unpacked_blobs"), "1"));
    CHECK(ByteReader::payload_read_counter() == 0);

    // The same blob delivered by pairing: no offset AND no external path, so the
    // inspector refuses instead of decoding the sibling from byte 0.
    TempDir p("unpacked_pair");
    p.put("net.caffemodel", net);
    p.put("net.prototxt",
          "input: \"data\"\ninput_shape { dim: 1 dim: 3 }\n"
          "layer { name: \"x\" type: \"PReLU\" bottom: \"data\" top: \"x\" }\n");
    auto pr = parse_path(p.file("net.prototxt"));
    REQUIRE_MESSAGE(pr, (pr ? "" : pr.error().message));
    const ir::TensorRef* pt = find_init(*pr, "x/slope");
    REQUIRE(pt != nullptr);
    CHECK(pt->file_offset == UINT64_MAX);
    CHECK_FALSE(pt->external_path.valid());
    auto mf = MappedFile::open(p.file("net.prototxt"));
    REQUIRE(mf);
    auto st = compute_tensor_stats(*pt, *mf, p.path.string(), &*pr);
    CHECK_FALSE(st);
  }
  SUBCASE("oversized and negative declared dims") {
    d.put("huge.prototxt",
          "layer { name: \"data\" type: \"Input\" top: \"data\" "
          "input_param { shape { dim: 4611686018427387904 dim: 4 } } }\n");
    auto r = parse_path(d.file("huge.prototxt"));
    REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
    CHECK(shape_named(*r, "data") == kNone);
    CHECK(starts_with(meta(*r, "oversized_shapes"), "1"));

    d.put("negdim.prototxt",
          "layer { name: \"data\" type: \"Input\" top: \"data\" "
          "input_param { shape { dim: -5 dim: 4 } } }\n");
    auto n = parse_path(d.file("negdim.prototxt"));
    REQUIRE_MESSAGE(n, (n ? "" : n.error().message));
    CHECK(shape_named(*n, "data") == std::vector<int64_t>{-1, 4});
  }
  SUBCASE("an unknown enum identifier is kept verbatim, the layer unvetted") {
    d.put("maxx.prototxt", input +
                               "layer { name: \"p\" type: \"Pooling\" bottom: \"data\" top: \"p\" "
                               "pooling_param { pool: MAXX kernel_size: 2 stride: 2 } }\n");
    auto r = parse_path(d.file("maxx.prototxt"));
    REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
    const ir::Node* p = find_node(*r, r->graphs[0], "p");
    REQUIRE(p != nullptr);
    CHECK(r->str(p->op_type) == "Pooling");
    CHECK(r->str(attr(*r, r->graphs[0], *p, "pooling_param.pool")->s) == "MAXX");
    CHECK(shape_named(*r, "p") == kNone);
  }
  SUBCASE("inline text blobs: shaped, typed, never addressable; names uniquified") {
    d.put("inline.prototxt",
          input +
              "layer { name: \"s\" type: \"Scale\" bottom: \"data\" top: \"conv1/weight\"\n"
              "  blobs { shape { dim: 3 } data: 1 data: 2 data: [3] } }\n"
              "layer { name: \"conv1\" type: \"Convolution\" bottom: \"conv1/weight\" top: \"conv1\"\n"
              "  convolution_param { num_output: 2 kernel_size: 1 }\n"
              "  blobs { shape { dim: 2 dim: 3 dim: 1 dim: 1 } data: [1, 2, 3, 4, 5, 6] } }\n");
    ByteReader::payload_read_counter() = 0;
    auto r = parse_path(d.file("inline.prototxt"));
    REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
    const ir::TensorRef* s = find_init(*r, "s/scale");
    REQUIRE(s != nullptr);
    CHECK(s->dtype == ir::DType::F32);
    CHECK(std::vector<int64_t>(s->shape.begin(), s->shape.end()) == std::vector<int64_t>{3});
    CHECK(s->file_offset == UINT64_MAX);
    CHECK(starts_with(meta(*r, "text_blobs"), "2"));
    const ir::Node* c = find_node(*r, r->graphs[0], "conv1");
    REQUIRE(c != nullptr);
    CHECK(input_names(*r, r->graphs[0], *c) ==
          std::vector<std::string>{"conv1/weight", "conv1/weight#2"});
    CHECK(ByteReader::payload_read_counter() == 0);
  }
  SUBCASE("the net's own state picks the phase") {
    d.put("state.prototxt",
          "state { phase: TRAIN }\n" + input +
              "layer { name: \"tr\" type: \"ReLU\" bottom: \"data\" top: \"tr\" "
              "include { phase: TRAIN } }\n"
              "layer { name: \"te\" type: \"ReLU\" bottom: \"data\" top: \"te\" "
              "include { phase: TEST } }\n");
    auto r = parse_path(d.file("state.prototxt"));
    REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
    CHECK(find_node(*r, r->graphs[0], "tr") != nullptr);
    CHECK(find_node(*r, r->graphs[0], "te") == nullptr);
    CHECK(starts_with(meta(*r, "phase_filter"), "TRAIN (excluded 1 layer: te)"));
  }
}

// ---------------------------------------------------------------------------
// Schema tables: unique field numbers / names, enum round trips
// ---------------------------------------------------------------------------
namespace {
void check_message(const caffe::MessageSpec& m, std::set<const caffe::MessageSpec*>& seen) {
  if (!seen.insert(&m).second) return;
  std::set<uint32_t> numbers;
  std::set<std::string> names;
  for (size_t i = 0; i < m.count; ++i) {
    const caffe::FieldSpec& f = m.fields[i];
    CHECK_MESSAGE(numbers.insert(f.number).second, m.name);
    CHECK_MESSAGE(names.insert(f.name).second, m.name);
    CHECK(caffe::find_field(m, f.number) == &f);
    CHECK(caffe::find_field(m, std::string_view(f.name)) == &f);
    if (f.kind == caffe::FieldKind::Enum) {
      REQUIRE(f.enum_spec != nullptr);
      for (size_t k = 0; k < f.enum_spec->count; ++k) {
        const caffe::EnumValue& e = f.enum_spec->values[k];
        CHECK(caffe::enum_number(*f.enum_spec, e.name) == e.number);
        CHECK(std::string(caffe::enum_name(*f.enum_spec, e.number)) == e.name);
      }
    }
    if (f.kind == caffe::FieldKind::Message && f.message) check_message(*f.message, seen);
  }
}
}  // namespace

TEST_CASE("Caffe schema: field numbers and names are unique, enums round-trip") {
  std::set<const caffe::MessageSpec*> seen;
  check_message(caffe::net_parameter_spec(), seen);
  check_message(caffe::layer_parameter_spec(), seen);
  check_message(caffe::v1_layer_parameter_spec(), seen);
  CHECK(seen.size() > 30);
  CHECK(std::string(caffe::v1_modern_type(4)) == "Convolution");
  CHECK(std::string(caffe::v1_modern_type(21)) == "SoftmaxWithLoss");
  CHECK(std::string(caffe::v1_modern_type(39)) == "Deconvolution");
  CHECK(std::string(caffe::v1_modern_type(40)).empty());
  CHECK(std::string(caffe::v1_modern_type(0)).empty());
}

TEST_CASE("Caffe K2: an unresolvable convolution geometry is never defaulted") {
  TempDir d("geom");
  const std::string input =
      "layer { name: \"data\" type: \"Input\" top: \"data\" "
      "input_param { shape { dim: 1 dim: 3 dim: 8 dim: 8 } } }\n";
  const std::string weight = " blobs { shape { dim: 2 dim: 3 dim: 3 dim: 3 } } ";
  struct Case {
    const char* name;
    std::string params;
    bool with_weight;
  };
  const std::vector<Case> cases = {
      // three kernel sizes for a 2-D weight
      {"k3", "num_output: 2 kernel_size: 3 kernel_size: 3 kernel_size: 3", true},
      // stride_h without stride_w
      {"sh", "num_output: 2 kernel_size: 3 stride_h: 2", true},
      // kernel_size and kernel_h together (Caffe refuses this)
      {"kk", "num_output: 2 kernel_size: 3 kernel_h: 3 kernel_w: 3", true},
      // no weight: R = 2 from the kernel list, but three strides
      {"s3", "num_output: 2 kernel_size: 3 kernel_size: 3 stride: 1 stride: 1 stride: 1", false},
      // a zero stride
      {"s0", "num_output: 2 kernel_size: 3 stride: 0", true},
  };
  for (const Case& c : cases) {
    const std::string text = input + "layer { name: \"c\" type: \"Convolution\" bottom: \"data\" "
                                     "top: \"c\" convolution_param { " +
                             c.params + " }" + (c.with_weight ? weight : std::string()) + "}\n";
    d.put(std::string(c.name) + ".prototxt", text);
    auto r = parse_path(d.file(std::string(c.name) + ".prototxt"));
    REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
    const ir::Graph& g = r->graphs[0];
    const ir::Node* n = find_node(*r, g, "c");
    REQUIRE(n != nullptr);
    CHECK_MESSAGE(r->str(n->op_type) == "Convolution", c.name);
    CHECK_MESSAGE(attr(*r, g, *n, "kernel_shape") == nullptr, c.name);
    CHECK_MESSAGE(attr(*r, g, *n, "strides") == nullptr, c.name);
    CHECK_MESSAGE(shape_named(*r, "c") == kNone, c.name);
    CHECK_FALSE_MESSAGE(compute_cost(*r, 0).per_node[1].flops_known, c.name);
  }
}
