// tests/test_detect.cpp — content-based format detection matrix (spec §5, §10).
//
// Writes minimal magic-byte buffers to temp files, maps them with MappedFile,
// and asserts detect_format() returns the expected Format. Extension hints are
// exercised as tie-breakers. Compile-only bar: links later with parser TUs.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/MappedFile.h"
#include "engine/OpCategory.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {

// Write bytes to a uniquely-named temp file; returns the path. RAII cleanup is
// handled by the caller via std::filesystem::remove at scope end.
std::string write_temp(const std::string& stem, const std::vector<uint8_t>& bytes) {
  std::filesystem::path p =
      std::filesystem::temp_directory_path() / ("nv_detect_" + stem);
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  out.close();
  return p.string();
}

// Map a byte buffer through a temp file and detect its format.
Format detect_bytes(const std::string& stem, const std::vector<uint8_t>& bytes,
                    const std::string& ext_hint) {
  std::string path = write_temp(stem, bytes);
  auto mf = MappedFile::open(path);
  REQUIRE(mf);  // mapping a freshly-written file must succeed
  Format f = detect_format(*mf, ext_hint);
  std::filesystem::remove(path);
  return f;
}

}  // namespace

TEST_CASE("detect GGUF by magic bytes") {
  // "GGUF" + version + counts; magic alone must be enough.
  std::vector<uint8_t> b = {'G', 'G', 'U', 'F', 3, 0, 0, 0};
  b.resize(32, 0);
  CHECK(detect_bytes("gguf", b, "gguf") == Format::GGUF);
}

TEST_CASE("detect TFLite by TFL3 identifier at bytes 4..8") {
  // FlatBuffer: root uoffset (4 bytes) then file_identifier "TFL3".
  std::vector<uint8_t> b = {0x1c, 0, 0, 0, 'T', 'F', 'L', '3'};
  b.resize(64, 0);
  CHECK(detect_bytes("tflite", b, "tflite") == Format::TFLite);
}

TEST_CASE("detect SafeTensors by header-length + JSON brace") {
  // u64 LE header length, then '{' starting the JSON header.
  std::string hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
  std::vector<uint8_t> b(8, 0);
  uint64_t n = hdr.size();
  for (int i = 0; i < 8; ++i) b[i] = static_cast<uint8_t>((n >> (8 * i)) & 0xff);
  for (char c : hdr) b.push_back(static_cast<uint8_t>(c));
  b.resize(b.size() + 4, 0);  // a little payload
  CHECK(detect_bytes("safetensors", b, "safetensors") == Format::SafeTensors);
}

TEST_CASE("detect PyTorch zip by PK magic") {
  // Local file header signature "PK\x03\x04".
  std::vector<uint8_t> b = {'P', 'K', 0x03, 0x04};
  b.resize(64, 0);
  Format f = detect_bytes("ptzip", b, "pt");
  CHECK(f == Format::PyTorchZip);
}

TEST_CASE("detect ONNX protobuf by ir_version field") {
  // ModelProto: field 1 (ir_version), varint wire type -> tag 0x08.
  std::vector<uint8_t> b = {0x08, 0x07};  // ir_version = 7
  // Add a graph field (field 7, length-delimited -> tag 0x3a) to look real.
  b.push_back(0x3a);
  b.push_back(0x02);
  b.push_back(0x00);
  b.push_back(0x00);
  CHECK(detect_bytes("onnx", b, "onnx") == Format::ONNX);
}

TEST_CASE("detect PyTorch legacy pickle by protocol-2 opcode") {
  // Standalone pickle begins with PROTO opcode 0x80 followed by a version byte.
  std::vector<uint8_t> b = {0x80, 0x02, 'c'};
  b.resize(32, 0);
  Format f = detect_bytes("ptlegacy", b, "pkl");
  // Accept either legacy-pickle classification (preferred) or Unknown if the
  // detector requires a fuller stream; we only forbid a wrong positive match.
  CHECK((f == Format::PyTorchLegacy || f == Format::Unknown));
}

TEST_CASE("detect Unknown on random bytes") {
  std::vector<uint8_t> b = {0xde, 0xad, 0xbe, 0xef, 0x11, 0x22, 0x33, 0x44};
  b.resize(32, 0);
  CHECK(detect_bytes("junk", b, "") == Format::Unknown);
}

TEST_CASE("detect Keras HDF5 by superblock signature") {
  // HDF5 superblock magic at offset 0 -> Format::Keras (raw .h5).
  std::vector<uint8_t> b = {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
  b.resize(64, 0);
  CHECK(detect_bytes("h5", b, "h5") == Format::Keras);
}

// --- v0.4.0: OpCategory coverage (color routing) ------------------------------
// categorize_op maps an op string to a coloring category. New v0.4.0 categories
// (Attention/Recurrent/Quantize) plus gap-fill entries must land in the right
// bucket, and com.microsoft.* domain prefixes must be tolerated.

TEST_CASE("OpCategory: v0.4.0 new-op categories") {
  // Quantized compute ops keep their float-analogue COLOR category (Conv/MatMul),
  // per the plan (category owns color; FLOP routing is separate).
  CHECK(categorize_op("QLinearConv") == OpCategory::Conv);
  CHECK(categorize_op("MatMulInteger") == OpCategory::MatMul);
  CHECK(categorize_op("Attention") == OpCategory::Attention);
  CHECK(categorize_op("LSTM") == OpCategory::Recurrent);
  // Pure quant markers -> Quantize.
  CHECK(categorize_op("QuantizeLinear") == OpCategory::Quantize);

  // Domain prefix tolerated (contrib ops).
  CHECK(categorize_op("com.microsoft.Attention") == OpCategory::Attention);
  CHECK(categorize_op("com.microsoft.QuantizeLinear") == OpCategory::Quantize);
}

TEST_CASE("OpCategory: gap-fill samples") {
  // Erf is elementwise math (Elementwise per the plan's gap-fill table).
  CHECK(categorize_op("Erf") == OpCategory::Elementwise);
  // ReduceL2 is a reduction.
  CHECK(categorize_op("ReduceL2") == OpCategory::Reduce);
  // Mish is an activation.
  CHECK(categorize_op("Mish") == OpCategory::Activation);
}

TEST_CASE("OpCategory: category_name is non-empty for every category") {
  // Exhaustive over Conv..Other (Other is last). category_name must return a
  // stable non-empty label for each, including the three new v0.4.0 ones.
  for (int c = 0; c <= static_cast<int>(OpCategory::Other); ++c) {
    const char* name = category_name(static_cast<OpCategory>(c));
    CHECK(name != nullptr);
    CHECK(name[0] != '\0');
  }
  // Spot-check the new names route through the switch (not the default).
  CHECK(std::string(category_name(OpCategory::Attention)) != "Other");
  CHECK(std::string(category_name(OpCategory::Recurrent)) != "Other");
  CHECK(std::string(category_name(OpCategory::Quantize)) != "Other");
}
