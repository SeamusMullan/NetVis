// SPDX-License-Identifier: Apache-2.0
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

// #45: same, but also captures WHY detection decided. Verifies the 3-arg
// overload and that the 2-arg wrapper stays consistent with it.
Format detect_bytes_reason(const std::string& stem,
                           const std::vector<uint8_t>& bytes,
                           const std::string& ext_hint, DetectReason& reason) {
  std::string path = write_temp(stem, bytes);
  auto mf = MappedFile::open(path);
  REQUIRE(mf);  // mapping a freshly-written file must succeed
  Format f = detect_format(*mf, ext_hint, reason);
  CHECK(detect_format(*mf, ext_hint) == f);  // wrapper delegates identically
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

TEST_CASE("detect CoreML by .mlmodel extension tiebreaker") {
  // A bare CoreML Model protobuf begins with field 1 (specificationVersion,
  // varint), which structurally looks like ONNX's ir_version. The .mlmodel
  // extension is the decisive tiebreaker and must route to CoreML, not ONNX.
  std::vector<uint8_t> b = {0x08, 0x04};  // specificationVersion = 4
  // Add a neuralNetwork field (500, length-delimited): tag=(500<<3)|2=4002.
  b.push_back(0xa2);
  b.push_back(0x1f);
  b.push_back(0x00);  // zero-length body
  CHECK(detect_bytes("coreml", b, "mlmodel") == Format::CoreML);
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

// --- #45: format-detection reason (confidence signal) -------------------------
// The 3-arg detect_format reports WHICH signal decided the format, surfaced in
// the status bar. Magic > structure > extension > content-default confidence.

TEST_CASE("detect reason: GGUF magic -> Magic") {
  std::vector<uint8_t> b = {'G', 'G', 'U', 'F', 3, 0, 0, 0};
  b.resize(32, 0);
  DetectReason r = DetectReason::None;
  CHECK(detect_bytes_reason("r_gguf", b, "gguf", r) == Format::GGUF);
  CHECK(r == DetectReason::Magic);
}

TEST_CASE("detect reason: ONNX-shaped protobuf -> Structure") {
  // field 1 (ir_version, varint) + field 7 (graph, length-delimited). No .onnx
  // extension, so only the structural sniff can decide it.
  std::vector<uint8_t> b = {0x08, 0x07, 0x3a, 0x02, 0x00, 0x00};
  DetectReason r = DetectReason::None;
  CHECK(detect_bytes_reason("r_onnx", b, "", r) == Format::ONNX);
  CHECK(r == DetectReason::Structure);
}

TEST_CASE("detect reason: ambiguous zip + ext_hint pt -> Extension") {
  // A bare zip (PK local-file-header, no central directory the scan recognizes)
  // carries no specific content signal; the "pt" extension breaks the tie.
  std::vector<uint8_t> b = {'P', 'K', 0x03, 0x04};
  b.resize(64, 0);
  DetectReason r = DetectReason::None;
  CHECK(detect_bytes_reason("r_zip_pt", b, "pt", r) == Format::PyTorchZip);
  CHECK(r == DetectReason::Extension);
}

TEST_CASE("detect reason: ambiguous zip + no ext -> ContentDefault") {
  // Same zip with no extension hint: the family matched but the specific format
  // falls back to the PyTorch-zip default (lower confidence).
  std::vector<uint8_t> b = {'P', 'K', 0x03, 0x04};
  b.resize(64, 0);
  DetectReason r = DetectReason::None;
  CHECK(detect_bytes_reason("r_zip_none", b, "", r) == Format::PyTorchZip);
  CHECK(r == DetectReason::ContentDefault);
}

TEST_CASE("detect reason: random bytes -> None (Unknown)") {
  std::vector<uint8_t> b = {0xde, 0xad, 0xbe, 0xef, 0x11, 0x22, 0x33, 0x44};
  b.resize(32, 0);
  DetectReason r = DetectReason::Magic;  // seed non-None to prove it is cleared
  CHECK(detect_bytes_reason("r_junk", b, "", r) == Format::Unknown);
  CHECK(r == DetectReason::None);
}

TEST_CASE("detect reason: CoreML .mlmodel tiebreak -> Extension") {
  std::vector<uint8_t> b = {0x08, 0x04, 0xa2, 0x1f, 0x00};
  DetectReason r = DetectReason::None;
  CHECK(detect_bytes_reason("r_coreml", b, "mlmodel", r) == Format::CoreML);
  CHECK(r == DetectReason::Extension);
}

TEST_CASE("detect_reason_name: non-empty label per reason") {
  CHECK(std::string(detect_reason_name(DetectReason::None)) == "none");
  CHECK(std::string(detect_reason_name(DetectReason::Magic)) == "magic");
  CHECK(std::string(detect_reason_name(DetectReason::Structure)) == "structure");
  CHECK(std::string(detect_reason_name(DetectReason::Extension)) == "extension");
  CHECK(std::string(detect_reason_name(DetectReason::ContentDefault)) ==
        "content default");
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
  // Common exported-graph spellings of the same concepts.
  CHECK(categorize_op("Convert") == OpCategory::Tensor);
  CHECK(categorize_op("Broadcast") == OpCategory::Shape);
  CHECK(categorize_op("ScaledDotProductAttention") == OpCategory::Attention);
  CHECK(categorize_op("ShapeOf") == OpCategory::Shape);
  CHECK(categorize_op("Select") == OpCategory::Elementwise);
  CHECK(categorize_op("LessEqual") == OpCategory::Elementwise);
  CHECK(categorize_op("GreaterEqual") == OpCategory::Elementwise);
  CHECK(categorize_op("TopK") == OpCategory::Reduce);
  CHECK(categorize_op("Range") == OpCategory::Tensor);
  // Graph state and I/O boundary markers.
  CHECK(categorize_op("ReadValue") == OpCategory::IO);
  CHECK(categorize_op("Assign") == OpCategory::IO);
  CHECK(categorize_op("Parameter") == OpCategory::IO);
  CHECK(categorize_op("Result") == OpCategory::IO);
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

TEST_CASE("OpCategory #137: TorchScript ns::name[.overload] op types") {
  CHECK(categorize_op("aten::conv2d") == OpCategory::Conv);
  CHECK(categorize_op("aten::_convolution") == OpCategory::Conv);
  CHECK(categorize_op("aten::linear") == OpCategory::MatMul);
  CHECK(categorize_op("aten::bmm") == OpCategory::MatMul);
  CHECK(categorize_op("aten::relu_") == OpCategory::Activation);  // in-place
  CHECK(categorize_op("aten::hardtanh_") == OpCategory::Activation);
  CHECK(categorize_op("aten::batch_norm") == OpCategory::Norm);
  CHECK(categorize_op("aten::adaptive_avg_pool2d") == OpCategory::Pool);
  CHECK(categorize_op("aten::add.Tensor") == OpCategory::Elementwise);  // overload
  CHECK(categorize_op("aten::add_.Tensor") == OpCategory::Elementwise);
  CHECK(categorize_op("aten::cat") == OpCategory::Shape);
  CHECK(categorize_op("aten::view") == OpCategory::Shape);
  CHECK(categorize_op("prim::If") == OpCategory::ControlFlow);
  CHECK(categorize_op("prim::Loop") == OpCategory::ControlFlow);
  CHECK(categorize_op("prim::Constant") == OpCategory::Tensor);
  // Dunder names keep their underscores; deliberately unmapped ops stay Other.
  CHECK(categorize_op("aten::__is__") == OpCategory::Other);
  CHECK(categorize_op("aten::addmm") == OpCategory::Other);
  CHECK(categorize_op("aten::conv_transpose2d") == OpCategory::Other);
  CHECK(categorize_op("custom::my_op") == OpCategory::Other);
  CHECK(categorize_op("?") == OpCategory::Other);
  // Regression: the dotted-domain rule is unchanged.
  CHECK(categorize_op("com.microsoft.Gelu") == OpCategory::Activation);
  CHECK(categorize_op("Conv") == OpCategory::Conv);
  CHECK(categorize_op("ai.onnx.Relu") == OpCategory::Activation);
}

// --- #137: PyTorch Mobile (.ptl) detection --------------------------------------
namespace {

void le16(std::vector<uint8_t>& b, uint16_t v) {
  b.push_back(static_cast<uint8_t>(v & 0xff));
  b.push_back(static_cast<uint8_t>(v >> 8));
}
void le32(std::vector<uint8_t>& b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
}

// A minimal STORED zip with zero-size entries named `names`: per name a local
// file header (+ name), then one central-directory record per name, then the
// EOCD. Detection only reads the central directory and EOCD (CRCs are 0).
std::vector<uint8_t> make_zip(const std::vector<std::string>& names) {
  std::vector<uint8_t> b;
  std::vector<uint32_t> local_ofs;
  for (const std::string& n : names) {
    local_ofs.push_back(static_cast<uint32_t>(b.size()));
    le32(b, 0x04034b50);
    le16(b, 20);  // version needed
    le16(b, 0);   // flags
    le16(b, 0);   // STORED
    le16(b, 0);   // time
    le16(b, 0);   // date
    le32(b, 0);   // crc
    le32(b, 0);   // compressed size
    le32(b, 0);   // uncompressed size
    le16(b, static_cast<uint16_t>(n.size()));
    le16(b, 0);   // extra
    b.insert(b.end(), n.begin(), n.end());
  }
  const auto cd_off = static_cast<uint32_t>(b.size());
  for (size_t i = 0; i < names.size(); ++i) {
    const std::string& n = names[i];
    le32(b, 0x02014b50);
    le16(b, 20);  // made by
    le16(b, 20);  // needed
    le16(b, 0);
    le16(b, 0);
    le16(b, 0);
    le16(b, 0);
    le32(b, 0);
    le32(b, 0);
    le32(b, 0);
    le16(b, static_cast<uint16_t>(n.size()));
    le16(b, 0);   // extra
    le16(b, 0);   // comment
    le16(b, 0);   // disk
    le16(b, 0);   // internal attrs
    le32(b, 0);   // external attrs
    le32(b, local_ofs[i]);
    b.insert(b.end(), n.begin(), n.end());
  }
  const auto cd_size = static_cast<uint32_t>(b.size() - cd_off);
  le32(b, 0x06054b50);
  le16(b, 0);
  le16(b, 0);
  le16(b, static_cast<uint16_t>(names.size()));
  le16(b, static_cast<uint16_t>(names.size()));
  le32(b, cd_size);
  le32(b, cd_off);
  le16(b, 0);
  return b;
}

Format zip_detect(const std::string& stem, const std::vector<std::string>& names,
                  const std::string& ext, DetectReason& r) {
  return detect_bytes_reason(stem, make_zip(names), ext, r);
}

}  // namespace

TEST_CASE("detect #137 T-D1/T-D2: bytecode.pkl is a PyTorch content signal") {
  DetectReason r = DetectReason::None;
  CHECK(zip_detect("d1a", {"bytecode.pkl"}, "", r) == Format::PyTorchZip);
  CHECK(r == DetectReason::Magic);
  CHECK(zip_detect("d1b", {"data.pkl", "bytecode.pkl"}, "", r) == Format::PyTorchZip);
  CHECK(r == DetectReason::Magic);
  CHECK(zip_detect("d2", {"model/bytecode.pkl"}, "", r) == Format::PyTorchZip);
  CHECK(r == DetectReason::Magic);
}

TEST_CASE("detect #137 T-D3: only the exact basename counts") {
  DetectReason r = DetectReason::None;
  CHECK(zip_detect("d3a", {"bytecode.pkl.bak"}, "", r) == Format::PyTorchZip);
  CHECK(r == DetectReason::ContentDefault);
  CHECK(zip_detect("d3b", {"xbytecode.pkl"}, "", r) == Format::PyTorchZip);
  CHECK(r == DetectReason::ContentDefault);
}

TEST_CASE("detect #137 T-D4/T-D5: the PyTorch tier wins over npz and keras") {
  DetectReason r = DetectReason::None;
  CHECK(zip_detect("d4", {"bytecode.pkl", "w.npy"}, "", r) == Format::PyTorchZip);
  CHECK(zip_detect("d5", {"bytecode.pkl", "config.json", "model.weights.h5"}, "", r) ==
        Format::PyTorchZip);
  CHECK(r == DetectReason::Magic);
}

TEST_CASE("detect #137 T-D6/T-D7/T-D8: .ptl is a tiebreak, never beats content") {
  DetectReason r = DetectReason::None;
  CHECK(zip_detect("d6", {"foo.txt"}, "ptl", r) == Format::PyTorchZip);
  CHECK(r == DetectReason::Extension);
  CHECK(zip_detect("d7", {"w.npy"}, "ptl", r) == Format::Npz);
  CHECK(r == DetectReason::Magic);
  std::vector<uint8_t> legacy = {0x80, 0x02, 'K', 0x01, '.'};
  legacy.resize(32, 0);
  CHECK(detect_bytes_reason("d8", legacy, "ptl", r) == Format::PyTorchLegacy);
  // Content nothing recognises: the extension tiebreak routes .ptl to PyTorch.
  std::vector<uint8_t> junk = {0xde, 0xad, 0xbe, 0xef, 0x11, 0x22, 0x33, 0x44};
  junk.resize(32, 0);
  CHECK(detect_bytes_reason("d8b", junk, "ptl", r) == Format::PyTorchZip);
  CHECK(r == DetectReason::Extension);
}

TEST_CASE("detect #137 T-D9: a PTMF flatbuffer is routed to an honest error") {
  std::vector<uint8_t> b = {0x1c, 0, 0, 0, 'P', 'T', 'M', 'F'};
  b.resize(64, 0);
  for (const char* ext : {"ptl", ""}) {
    DetectReason r = DetectReason::None;
    CHECK(detect_bytes_reason(std::string("d9") + ext, b, ext, r) == Format::PyTorchZip);
    CHECK(r == DetectReason::Magic);
  }
  std::string path = write_temp("d9_parse", b);
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = pytorch::parse_zip(*mf, progress);
  REQUIRE_FALSE(res);
  CHECK(res.error().message.find("flatbuffer") != std::string::npos);
  CHECK(res.error().offset == 4);
  std::filesystem::remove(path);
}

TEST_CASE("detect #137: a zip with neither pickle keeps the no-data.pkl error") {
  std::string path = write_temp("nopkl", make_zip({"foo.txt"}));
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = pytorch::parse_zip(*mf, progress);
  REQUIRE_FALSE(res);
  CHECK(res.error().message == "no data.pkl in archive");
  std::filesystem::remove(path);
}
