// SPDX-License-Identifier: Apache-2.0
// tests/test_fp4.cpp — MXFP4 / NVFP4 element and scale types across formats.
//
// MXFP4 and NVFP4 share one element type (FP4 E2M1) and differ in the block
// scale: MXFP4 uses one E8M0 power of two per 32 elements, NVFP4 one FP8 E4M3
// per 16. GGUF has dedicated block types for both (test_gguf_blocks.cpp,
// test_quant_preview.cpp). The other formats store the elements and scales as
// separate tensors, so what they need is honest element-type labels and an
// explanation of why the inspector has no histogram for them; those and the
// scalar minifloat decoders in core/Half.h are covered here.
//
// Expected values are hand-derived from the OCP MX v1.0 spec, not copied from
// the implementation.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/ByteReader.h"
#include "core/Half.h"
#include "core/MappedFile.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/pytorch/PickleVM.h"

using namespace netvis;

namespace {

std::string write_temp(const std::string& name, const std::string& bytes) {
  std::filesystem::path p = std::filesystem::temp_directory_path() / ("nv_fp4_" + name);
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return p.string();
}

const ir::TensorRef* find_tensor(const std::vector<ir::TensorRef>& ts,
                                 const ir::Model& m, std::string_view name) {
  for (const ir::TensorRef& t : ts)
    if (m.str(t.name) == name) return &t;
  return nullptr;
}

// --- protobuf wire helpers (ONNX) ---
void varint(std::string& b, uint64_t v) {
  while (v >= 0x80) { b.push_back(static_cast<char>((v & 0x7F) | 0x80)); v >>= 7; }
  b.push_back(static_cast<char>(v));
}
void field_varint(std::string& b, uint32_t field, uint64_t v) {
  varint(b, (static_cast<uint64_t>(field) << 3) | 0);
  varint(b, v);
}
void field_bytes(std::string& b, uint32_t field, const std::string& v) {
  varint(b, (static_cast<uint64_t>(field) << 3) | 2);
  varint(b, v.size());
  b += v;
}

// --- pickle helpers ---
void global_(std::string& b, const std::string& mod, const std::string& nm) {
  b += 'c'; b += mod; b += '\n'; b += nm; b += '\n';
}
void unicode(std::string& b, const std::string& s) {
  b += static_cast<char>(0x8c);  // SHORT_BINUNICODE
  b += static_cast<char>(s.size());
  b += s;
}
void binint1(std::string& b, uint8_t v) { b += 'K'; b += static_cast<char>(v); }

}  // namespace

// --- core/Half.h minifloat decoders ------------------------------------------

TEST_CASE("fp4_e2m1_to_f32: all 16 codes") {
  const float expect[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  for (uint8_t c = 0; c < 16; ++c) CHECK(fp4_e2m1_to_f32(c) == expect[c]);
  CHECK(std::signbit(fp4_e2m1_to_f32(0x8)));  // -0 keeps its sign
  CHECK(fp4_e2m1_to_f32(0xF7) == 6.0f);       // high bits ignored
}

TEST_CASE("e8m0_to_f32: pure powers of two, 0xFF is NaN") {
  CHECK(e8m0_to_f32(127) == 1.0f);
  CHECK(e8m0_to_f32(128) == 2.0f);
  CHECK(e8m0_to_f32(126) == 0.5f);
  CHECK(e8m0_to_f32(254) == std::ldexp(1.0f, 127));
  CHECK(e8m0_to_f32(0) == std::ldexp(1.0f, -127));  // subnormal in float
  CHECK(std::isnan(e8m0_to_f32(0xFF)));
}

TEST_CASE("fp8_e4m3fn_to_f32: normals, subnormals, max, NaN") {
  CHECK(fp8_e4m3fn_to_f32(0x38) == 1.0f);    // exp 7 (bias 7), mantissa 0
  CHECK(fp8_e4m3fn_to_f32(0x3C) == 1.5f);    // mantissa 4/8
  CHECK(fp8_e4m3fn_to_f32(0xB8) == -1.0f);   // sign bit
  CHECK(fp8_e4m3fn_to_f32(0x7E) == 448.0f);  // largest finite: 1.75 * 2^8
  CHECK(fp8_e4m3fn_to_f32(0x01) == std::ldexp(1.0f, -9));  // smallest subnormal
  CHECK(fp8_e4m3fn_to_f32(0x08) == std::ldexp(1.0f, -6));  // smallest normal
  CHECK(fp8_e4m3fn_to_f32(0x00) == 0.0f);
  CHECK(std::isnan(fp8_e4m3fn_to_f32(0x7F)));
  CHECK(std::isnan(fp8_e4m3fn_to_f32(0xFF)));
  CHECK(fp8_e4m3fn_to_f32(0x78) == 256.0f);  // exp 15 is finite in the fn variant
}

// --- SafeTensors: F4 / F8_* dtype strings keep their exact label -------------

TEST_CASE("SafeTensors: F4 and F8_E8M0 tensors are labeled, not '?'") {
  // An MXFP4 checkpoint layout: packed E2M1 weights + one E8M0 scale per 32.
  const std::string header =
      R"({"w":{"dtype":"F4","shape":[2,32],"data_offsets":[0,32]},)"
      R"("w_scale":{"dtype":"F8_E8M0","shape":[2,1],"data_offsets":[32,34]},)"
      R"("b":{"dtype":"F32","shape":[1],"data_offsets":[34,38]}})";
  std::string file;
  uint64_t n = header.size();
  for (int i = 0; i < 8; ++i) file.push_back(static_cast<char>((n >> (8 * i)) & 0xFF));
  file += header;
  file += std::string(38, '\0');
  const std::string path = write_temp("st.safetensors", file);

  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  ByteReader::payload_read_counter() = 0;
  auto res = safetensors::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "safetensors::parse returned an error");
  CHECK(ByteReader::payload_read_counter() == 0);
  const ir::Model& m = *res;

  const ir::TensorRef* w = find_tensor(m.flat_tensors, m, "w");
  const ir::TensorRef* sc = find_tensor(m.flat_tensors, m, "w_scale");
  const ir::TensorRef* b = find_tensor(m.flat_tensors, m, "b");
  REQUIRE(w);
  REQUIRE(sc);
  REQUIRE(b);
  CHECK(w->dtype == ir::DType::Unknown);
  CHECK(m.str(w->dtype_label) == "F4");
  CHECK(w->byte_len == 32);
  CHECK(m.str(sc->dtype_label) == "F8_E8M0");
  CHECK(b->dtype == ir::DType::F32);
  CHECK_FALSE(b->dtype_label.valid());  // mapped types need no label
  std::filesystem::remove(path);
}

// --- ONNX: FLOAT4E2M1 / FLOAT8E4M3FN initializers keep their exact label -----

TEST_CASE("ONNX: FLOAT4E2M1 weight + FLOAT8E4M3FN scale initializers are labeled") {
  // An NVFP4 layout: 32 E2M1 values (16 packed bytes) + two E4M3 scales.
  std::string w;
  field_varint(w, 1, 32);                      // dims
  field_varint(w, 2, 23);                      // data_type = FLOAT4E2M1
  field_bytes(w, 8, "w");                      // name
  field_bytes(w, 9, std::string(16, '\x11'));  // raw_data
  std::string s;
  field_varint(s, 1, 2);
  field_varint(s, 2, 17);                      // FLOAT8E4M3FN
  field_bytes(s, 8, "w_scale");
  field_bytes(s, 9, std::string("\x38\x40", 2));
  std::string graph;
  field_bytes(graph, 2, "g");
  field_bytes(graph, 5, w);
  field_bytes(graph, 5, s);
  std::string model;
  field_varint(model, 1, 10);                  // ir_version
  field_bytes(model, 7, graph);
  const std::string path = write_temp("m.onnx", model);

  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  ByteReader::payload_read_counter() = 0;
  auto res = onnx::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "onnx::parse returned an error");
  CHECK(ByteReader::payload_read_counter() == 0);
  const ir::Model& m = *res;
  REQUIRE_FALSE(m.graphs.empty());

  const ir::TensorRef* wt = find_tensor(m.graphs[0].initializers, m, "w");
  const ir::TensorRef* st = find_tensor(m.graphs[0].initializers, m, "w_scale");
  REQUIRE(wt);
  REQUIRE(st);
  CHECK(wt->dtype == ir::DType::Unknown);
  CHECK(m.str(wt->dtype_label) == "float4e2m1");
  CHECK(wt->byte_len == 16);
  CHECK(m.str(st->dtype_label) == "float8e4m3fn");
  std::filesystem::remove(path);
}

// --- PyTorch: _rebuild_tensor_v3 carries the dtype as a trailing global -------

TEST_CASE("PickleVM: _rebuild_tensor_v3 with torch.float4_e2m1fn_x2") {
  // torch.save of a float4_e2m1fn_x2 tensor of shape [4, 2] (8 bytes, 16 FP4
  // values): _rebuild_tensor_v3(UntypedStorage pid, 0, (4, 2), (2, 1), False,
  // OrderedDict(), torch.float4_e2m1fn_x2).
  std::string b;
  b += "\x80\x02";
  global_(b, "torch._utils", "_rebuild_tensor_v3");
  b += '(';                                            // MARK (args)
  b += '(';                                            //   MARK (pid)
  unicode(b, "storage");
  global_(b, "torch", "UntypedStorage");
  unicode(b, "0");
  unicode(b, "cpu");
  binint1(b, 8);
  b += 't';                                            //   TUPLE
  b += 'Q';                                            //   BINPERSID
  binint1(b, 0);                                       //   storage_offset
  binint1(b, 4); binint1(b, 2); b += static_cast<char>(0x86);  // size
  binint1(b, 2); binint1(b, 1); b += static_cast<char>(0x86);  // stride
  b += static_cast<char>(0x89);                        //   requires_grad
  global_(b, "collections", "OrderedDict");
  b += ')'; b += 'R';                                  //   backward_hooks
  global_(b, "torch", "float4_e2m1fn_x2");             //   dtype
  b += 't';                                            // TUPLE
  b += 'R';                                            // REDUCE
  b += '.';

  pytorch::StorageResolver resolver;
  resolver.resolve = [](const std::string& key, uint64_t& off, uint64_t& len) {
    if (key != "0") return false;
    off = 1000;
    len = 8;
    return true;
  };
  pytorch::PickleVM vm(reinterpret_cast<const uint8_t*>(b.data()), b.size(), resolver);
  auto res = vm.run();
  REQUIRE_MESSAGE(res, "PickleVM::run returned an error");
  const pytorch::ValuePtr& v = *res;
  REQUIRE(v);
  REQUIRE(v->kind == pytorch::Value::Kind::Tensor);
  CHECK(v->tensor.dtype == ir::DType::Unknown);
  CHECK(v->dtype_label == "float4_e2m1fn_x2");
  REQUIRE(v->tensor.shape.size() == 2);
  CHECK(v->tensor.shape[0] == 4);
  CHECK(v->tensor.shape[1] == 2);
  CHECK(v->tensor.file_offset == 1000);
  CHECK(v->tensor.byte_len == 8);  // 1-byte elements, each two FP4 values
}

TEST_CASE("PickleVM: _rebuild_tensor_v3 with torch.uint16 maps to a real DType") {
  std::string b;
  b += "\x80\x02";
  global_(b, "torch._utils", "_rebuild_tensor_v3");
  b += '(';
  b += '(';
  unicode(b, "storage");
  global_(b, "torch", "UntypedStorage");
  unicode(b, "0");
  unicode(b, "cpu");
  binint1(b, 6);
  b += 't';
  b += 'Q';
  binint1(b, 1);                                       // storage_offset = 1 elem
  binint1(b, 2); b += static_cast<char>(0x85);         // size (2,)
  binint1(b, 1); b += static_cast<char>(0x85);         // stride (1,)
  b += static_cast<char>(0x89);
  global_(b, "collections", "OrderedDict");
  b += ')'; b += 'R';
  global_(b, "torch", "uint16");
  b += 't';
  b += 'R';
  b += '.';

  pytorch::StorageResolver resolver;
  resolver.resolve = [](const std::string&, uint64_t& off, uint64_t& len) {
    off = 64;
    len = 6;
    return true;
  };
  pytorch::PickleVM vm(reinterpret_cast<const uint8_t*>(b.data()), b.size(), resolver);
  auto res = vm.run();
  REQUIRE(res);
  const pytorch::ValuePtr& v = *res;
  REQUIRE(v->kind == pytorch::Value::Kind::Tensor);
  CHECK(v->tensor.dtype == ir::DType::U16);
  CHECK(v->dtype_label.empty());
  CHECK(v->tensor.file_offset == 66);  // 64 + 1 element * 2 bytes
  CHECK(v->tensor.byte_len == 4);
}

// --- stats_unavailable_reason: why the inspector shows no histogram -----------

namespace {
std::string reason_for(ir::Model& m, ir::DType dt, std::string_view label) {
  ir::TensorRef t;
  t.dtype = dt;
  if (!label.empty()) t.dtype_label = m.intern(label);
  return stats_unavailable_reason(t, &m);
}
}  // namespace

TEST_CASE("stats_unavailable_reason: FP4 codes explain the missing block scales") {
  ir::Model m;
  // Every parser's spelling of the FP4 element type gets the same explanation.
  for (const char* lbl : {"F4", "float4e2m1", "float4_e2m1fn_x2", "f4e2m1"}) {
    const std::string r = reason_for(m, ir::DType::Unknown, lbl);
    CAPTURE(lbl);
    CHECK(r.find("FP4 (E2M1)") != std::string::npos);
    CHECK(r.find("block scale") != std::string::npos);
  }
}

TEST_CASE("stats_unavailable_reason: E8M0 and other FP8 scale tensors") {
  ir::Model m;
  for (const char* lbl : {"F8_E8M0", "float8e8m0", "float8_e8m0fnu", "f8e8m0"}) {
    CAPTURE(lbl);
    CHECK(reason_for(m, ir::DType::Unknown, lbl).find("E8M0") != std::string::npos);
  }
  const std::string e4m3 = reason_for(m, ir::DType::Unknown, "F8_E4M3");
  CHECK(e4m3.find("FP8 (F8_E4M3)") != std::string::npos);
  CHECK(e4m3.find("NVFP4") != std::string::npos);
}

TEST_CASE("stats_unavailable_reason: other labels, no label, and decodable types") {
  ir::Model m;
  CHECK(reason_for(m, ir::DType::Unknown, "complex64").find("complex64") !=
        std::string::npos);
  CHECK_FALSE(reason_for(m, ir::DType::Unknown, "").empty());
  // Decodable types and GGUF block quants (explained elsewhere) get no reason.
  CHECK(reason_for(m, ir::DType::F32, "").empty());
  CHECK(reason_for(m, ir::DType::Q4, "MXFP4").empty());
}
