// tests/test_hardening.cpp — regression tests for the hostile-input defects
// found by the adversarial review. Each asserts that a malicious value produces
// a graceful outcome (error / clamp / bounded work), never a crash, OOB read,
// SIGFPE, or unbounded allocation. Run under ASan/UBSan these also prove no UB.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/MappedFile.h"
#include "engine/ShapeInference.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {
std::string write_temp(const std::string& stem, const std::vector<uint8_t>& bytes) {
  std::string p = (std::filesystem::temp_directory_path() / ("nv_hard_" + stem)).string();
  std::ofstream o(p, std::ios::binary);
  o.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  return p;
}
}  // namespace

TEST_CASE("TensorStats: overflowing shape*dtype clamps to payload, no OOB") {
  // A tiny 16-byte payload but a declared shape of 2^61 F64 elements. The
  // element-count computation must clamp to what the payload holds (2 elems),
  // never return the huge count and stream past the buffer.
  std::vector<uint8_t> payload(16, 0);
  std::string path = write_temp("ovf.bin", payload);
  auto mf = MappedFile::open(path);
  REQUIRE(mf);

  ir::TensorRef t;
  t.dtype = ir::DType::F64;
  t.shape = {static_cast<int64_t>(1) << 61};  // 2^61 elements
  t.file_offset = 0;
  t.byte_len = 16;

  auto stats = compute_tensor_stats(t, *mf, "");
  // Either an error or a clamped result — but it must return, not crash/OOB.
  if (stats) CHECK(stats->count <= 2);  // 16 bytes / 8 = at most 2 F64
  std::filesystem::remove(path);
}

TEST_CASE("ShapeInference: zero stride does not divide-by-zero (Conv)") {
  // Conv with strides=[0,0] must not SIGFPE; inference bails on that output.
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  auto& g = m.graphs[0];

  // value 0 = X (input, NCHW), value 1 = W (weight), value 2 = Y (output)
  ir::ValueInfo x; x.name = m.intern("X"); x.dtype = ir::DType::F32;
  x.shape = {1, 3, 8, 8}; x.producer = -1;
  ir::ValueInfo w; w.name = m.intern("W"); w.dtype = ir::DType::F32;
  w.shape = {4, 3, 3, 3}; w.producer = -1;
  ir::ValueInfo y; y.name = m.intern("Y"); y.producer = 0;
  g.values = {x, w, y};

  ir::Node n;
  n.op_type = m.intern("Conv");
  n.name = m.intern("conv0");
  n.inputs.begin = 0; n.inputs.count = 2;
  n.outputs.begin = 2; n.outputs.count = 1;
  // attribute: strides = [0, 0]
  ir::Attribute a; a.name = m.intern("strides");
  a.value.kind = ir::AttrValue::Kind::Ints; a.value.ints = {0, 0};
  n.attributes.begin = 0; n.attributes.count = 1;
  g.attributes = {a};
  g.edge_refs = {0, 1, 2};
  g.nodes = {n};

  // Must return (no SIGFPE). Y stays unresolved, which is fine.
  uint32_t resolved = infer_shapes(m, 0, nullptr);
  CHECK(resolved == 0);  // could not resolve Conv with stride 0 — but did not crash
}

TEST_CASE("SafeTensors: near-2^64 data_offset end is rejected, not wrapped") {
  // Header declares end = 2^64-8 which would wrap data_base+end; the parser must
  // reject the tensor rather than accept an out-of-file range.
  std::string json =
      "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],"
      "\"data_offsets\":[0,18446744073709551608]}}";
  std::vector<uint8_t> bytes(8);
  uint64_t n = json.size();
  std::memcpy(bytes.data(), &n, 8);  // little-endian on our targets
  bytes.insert(bytes.end(), json.begin(), json.end());
  bytes.push_back(0);  // one payload byte so file isn't just the header

  std::string path = write_temp("st_ovf.safetensors", bytes);
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink ps;
  auto r = safetensors::parse(*mf, ps);
  CHECK_FALSE(r);  // must be an error, not an accepted out-of-range tensor
  std::filesystem::remove(path);
}
