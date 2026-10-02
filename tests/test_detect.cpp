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
#include "parsers/caffe/CaffeSniff.h"
#include "temp_file_guard.h"

using namespace netvis;

namespace {

// Write bytes to a uniquely-named temp file; returns the path. RAII cleanup is
// handled by the caller with a TempFileGuard declared before the mapping.
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
  netvis_test::TempFileGuard cleanup(path);  // before mf: unmap first, then delete
  auto mf = MappedFile::open(path);
  REQUIRE(mf);  // mapping a freshly-written file must succeed
  Format f = detect_format(*mf, ext_hint);
  return f;
}

// #45: same, but also captures WHY detection decided. Verifies the 3-arg
// overload and that the 2-arg wrapper stays consistent with it.
Format detect_bytes_reason(const std::string& stem,
                           const std::vector<uint8_t>& bytes,
                           const std::string& ext_hint, DetectReason& reason) {
  std::string path = write_temp(stem, bytes);
  netvis_test::TempFileGuard cleanup(path);  // before mf: unmap first, then delete
  auto mf = MappedFile::open(path);
  REQUIRE(mf);  // mapping a freshly-written file must succeed
  Format f = detect_format(*mf, ext_hint, reason);
  CHECK(detect_format(*mf, ext_hint) == f);  // wrapper delegates identically
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

TEST_CASE("detect CoreML wins over the SavedModel sniff on a real .mlmodel (#114)") {
  // REGRESSION. The hand-built buffer above kept passing while every REAL
  // .mlmodel broke, because it carries no `description` (field 2):
  // looks_like_saved_model rejects it at its second tag (field 500, not field 2)
  // and so never got as far as being mistaken for a SavedModel. A real CoreML
  // Model always has one:
  //   field 1 specificationVersion (varint)  ->  SavedModel's schema_version
  //   field 2 description (len-delimited)    ->  SavedModel's meta_graphs[0]
  //   ... whose own first field is len-delimited (input) -> MetaGraphDef's
  // which satisfies the SavedModel prefix check exactly. #107 placed the
  // TensorFlow sniff ahead of the .mlmodel guard, so those files were handed to
  // the TensorFlow parser and died with "SavedModel meta_graph carries no
  // graph_def". Assert against the shipped fixture, not a synthetic buffer: the
  // whole point is that the synthetic one was not representative.
  auto mf = MappedFile::open("tests/fixtures/model.mlmodel");
  REQUIRE_MESSAGE(mf, "fixture missing; run tools/gen_fixtures.py");
  CHECK(detect_format(*mf, "mlmodel") == Format::CoreML);

  DetectReason reason = DetectReason::None;
  CHECK(detect_format(*mf, "mlmodel", reason) == Format::CoreML);
  CHECK(reason == DetectReason::Extension);
}

TEST_CASE("detect CoreML from content alone, without the .mlmodel suffix (#114)") {
  // The extension is a tiebreaker, not the signal. A CoreML Model carries its
  // `oneof Type` at a field number >= 200 that no SavedModel / GraphDef / ONNX
  // ModelProto ever uses, so a renamed file is still recognised. Before the
  // content check, every one of these hints sent the shipped fixture to the
  // TensorFlow parser ("SavedModel meta_graph carries no graph_def").
  auto mf = MappedFile::open("tests/fixtures/model.mlmodel");
  REQUIRE_MESSAGE(mf, "fixture missing; run tools/gen_fixtures.py");
  for (const char* ext : {"", "bin", "pb", "onnx", "txt"}) {
    CAPTURE(ext);
    DetectReason reason = DetectReason::None;
    CHECK(detect_format(*mf, ext, reason) == Format::CoreML);
    CHECK(reason == DetectReason::Structure);
  }
}

TEST_CASE("detect CoreML mlProgram spec without the suffix is CoreML, not ONNX (#114)") {
  // The .mlpackage's inner spec has no `description` at all - (1,varint) then the
  // mlProgram at field 502 - so it matched the ONNX sniff (field 1 varint) when
  // the extension was absent. A Manifest.json may name its inner spec anything.
  auto mf = MappedFile::open(
      "tests/fixtures/model.mlpackage/Data/com.apple.CoreML/model.mlmodel");
  REQUIRE_MESSAGE(mf, "fixture missing; run tools/gen_fixtures.py");
  DetectReason reason = DetectReason::None;
  CHECK(detect_format(*mf, "", reason) == Format::CoreML);
  CHECK(reason == DetectReason::Structure);
}

TEST_CASE("detect CoreML with a description field, synthetic (#114)") {
  // The smallest buffer with the shape of a real .mlmodel: a version, a
  // non-empty description whose first field is length-delimited (the shape the
  // SavedModel prefix check accepts), then the neuralNetwork at field 500.
  std::vector<uint8_t> b = {0x08, 0x04,                   // specificationVersion = 4
                            0x12, 0x03, 0x0a, 0x01, 'x',  // description{ input: "x" }
                            0xa2, 0x1f, 0x00};            // neuralNetwork (500), empty
  DetectReason r = DetectReason::None;
  CHECK(detect_bytes_reason("coreml_desc_ext", b, "mlmodel", r) == Format::CoreML);
  CHECK(r == DetectReason::Extension);
  CHECK(detect_bytes_reason("coreml_desc_noext", b, "", r) == Format::CoreML);
  CHECK(r == DetectReason::Structure);
}

TEST_CASE("detect frozen GraphDef is TensorFlow even when named .mlmodel") {
  // GraphDef's field 1 is a length-delimited NodeDef and CoreML's is a varint, so
  // the GraphDef sniff cannot collide with CoreML and keeps winning over the
  // extension: the content is unambiguous.
  auto mf = MappedFile::open("tests/fixtures/model_frozen.pb");
  REQUIRE_MESSAGE(mf, "fixture missing");
  DetectReason reason = DetectReason::None;
  CHECK(detect_format(*mf, "mlmodel", reason) == Format::TensorFlow);
  CHECK(reason == DetectReason::Structure);
}

TEST_CASE("detect: a SavedModel-shaped prefix with another top-level field is not TensorFlow") {
  // SavedModel's top level is only fields 1 and 2. A buffer that opens like one
  // but then carries an ONNX ModelProto's graph (field 7) is ONNX.
  std::vector<uint8_t> b = {0x08, 0x01,                   // schema_version / ir_version = 1
                            0x12, 0x03, 0x0a, 0x01, 'x',  // field 2: looks like meta_graphs[0]
                            0x3a, 0x02, 0x00, 0x00};      // field 7 (graph), 2 bytes
  CHECK(detect_bytes("saved_model_shaped_onnx", b, "") == Format::ONNX);
}

TEST_CASE("detect: hostile length varints do not wrap past the bounds check") {
  // A length of 2^64-1 makes `offset + length` wrap to a small number, which the
  // naive `offset + length > size` check lets through. The NodeDef sniff would
  // then run with a "size" of 2^64-1 and trust the next length it finds.
  // Detection must reject these instead of guessing a format.
  const std::vector<uint8_t> huge = {0xff, 0xff, 0xff, 0xff, 0xff,
                                     0xff, 0xff, 0xff, 0xff, 0x01};  // 2^64-1
  std::vector<uint8_t> graph_def = {0x0a};  // field 1, length-delimited
  graph_def.insert(graph_def.end(), huge.begin(), huge.end());
  graph_def.insert(graph_def.end(), {0x0a, 0x01, 'a', 0x12, 0x01, 'b'});
  CHECK(detect_bytes("huge_len_graphdef", graph_def, "") == Format::Unknown);

  std::vector<uint8_t> onnx_like = {0x12};  // field 2 (producer_name), length-delimited
  onnx_like.insert(onnx_like.end(), huge.begin(), huge.end());
  onnx_like.insert(onnx_like.end(), {0x3a, 0x02, 0x00, 0x00});
  CHECK(detect_bytes("huge_len_onnx", onnx_like, "") == Format::Unknown);
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

// #138/#109: Caffe display names. Unvetted Caffe layers keep their own names
// (or a `Caffe` prefix) and these keys colour them; none may land in a category
// whose CostModel rule would invent FLOPs without an output shape.
TEST_CASE("OpCategory: Caffe display names") {
  CHECK(categorize_op("Convolution") == OpCategory::Conv);
  CHECK(categorize_op("Deconvolution") == OpCategory::Conv);
  CHECK(categorize_op("InnerProduct") == OpCategory::MatMul);
  CHECK(categorize_op("Pooling") == OpCategory::Pool);
  CHECK(categorize_op("SPP") == OpCategory::Pool);
  CHECK(categorize_op("LRN") == OpCategory::Norm);
  CHECK(categorize_op("MeanVarianceNormalization") == OpCategory::Norm);
  CHECK(categorize_op("Scale") == OpCategory::Elementwise);
  CHECK(categorize_op("Bias") == OpCategory::Elementwise);
  CHECK(categorize_op("Power") == OpCategory::Elementwise);
  CHECK(categorize_op("Eltwise") == OpCategory::Elementwise);
  CHECK(categorize_op("Threshold") == OpCategory::Activation);
  CHECK(categorize_op("Crop") == OpCategory::Shape);
  CHECK(categorize_op("CaffeSplit") == OpCategory::Shape);
  CHECK(categorize_op("CaffeSlice") == OpCategory::Shape);
  CHECK(categorize_op("CaffeFlatten") == OpCategory::Shape);
  CHECK(categorize_op("CaffeReshape") == OpCategory::Shape);
  CHECK(categorize_op("CaffeTile") == OpCategory::Shape);
  CHECK(categorize_op("CaffeLSTM") == OpCategory::Recurrent);
  CHECK(categorize_op("CaffeRNN") == OpCategory::Recurrent);
  CHECK(categorize_op("CaffeParameter") == OpCategory::Tensor);
  CHECK(categorize_op("Input") == OpCategory::IO);
  CHECK(categorize_op("Data") == OpCategory::IO);
  CHECK(categorize_op("ImageData") == OpCategory::IO);
  CHECK(categorize_op("MemoryData") == OpCategory::IO);
  CHECK(categorize_op("HDF5Data") == OpCategory::IO);
  CHECK(categorize_op("WindowData") == OpCategory::IO);
  CHECK(categorize_op("DummyData") == OpCategory::IO);
  // Reduce counts |input|, which would give these FLOPs with no output shape.
  CHECK(categorize_op("Reduction") == OpCategory::Other);
  CHECK(categorize_op("CaffeArgMax") == OpCategory::Other);
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

// ---------------------------------------------------------------------------
// #138: Caffe detection collisions. Caffe text and binary NetParameters both
// look like protobuf to the loose ONNX sniff; these pin the sniff ORDER and the
// one-level-deeper peeks against the real fixtures of every protobuf format.
// ---------------------------------------------------------------------------
namespace {

std::vector<uint8_t> bytes_of(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }

// Detect a fixture file (WARN + nullopt-ish Unknown when the fixture is missing).
bool fixture_format(const char* path, const std::string& ext, Format& f, DetectReason& r) {
  if (!std::filesystem::exists(path)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return false;
  }
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  f = detect_format(*mf, ext, r);
  return true;
}

}  // namespace

TEST_CASE("detect Caffe D1: a prototxt that walks the ONNX sniff onto ':'") {
  // The ONNX sniff reads this as 0x0A (field 1, len 110) -> 'i' (field 13,
  // fixed64) -> ':' (field 7, length-delimited, len 32) -> "graph": an ONNX
  // signal. The Caffe text sniff runs ahead of every protobuf sniff.
  const std::string kOnnxTrapPrototxt =
      "\nname: \"AlexNet\"\nlayer {\n    name: \"data\"\n    type: \"Input\"\n    top: \"data\"\n"
      "    input_param { shape: { dim: 64 dim: 1 dim: 28 dim: 28 } }\n}\n"
      "layer {\n    name: \"conv1\"\n    type: \"Convolution\"\n    bottom: \"data\"\n    top: \"conv1\"\n"
      "    convolution_param {\n        num_output: 20\n        kernel_size: 5\n        stride: 1\n    }\n}\n";
  for (const char* ext : {"", "prototxt"}) {
    DetectReason r = DetectReason::None;
    CHECK_MESSAGE(detect_bytes_reason("caffe_trap", bytes_of(kOnnxTrapPrototxt), ext, r) ==
                      Format::Caffe,
                  ext);
    CHECK(r == DetectReason::Structure);
  }
}

TEST_CASE("detect Caffe D2: both fixtures by content, whatever the extension") {
  Format f = Format::Unknown;
  DetectReason r = DetectReason::None;
  for (const char* ext : {"", "caffemodel", "onnx"}) {
    if (!fixture_format("tests/fixtures/model_caffe.caffemodel", ext, f, r)) return;
    CHECK_MESSAGE(f == Format::Caffe, ext);   // carries input_shape: ONNX-shaped
    CHECK(r == DetectReason::Structure);
  }
  if (!fixture_format("tests/fixtures/model_caffe_v1.caffemodel", "", f, r)) return;
  CHECK(f == Format::Caffe);
  CHECK(r == DetectReason::Structure);
  for (const char* p : {"tests/fixtures/model_caffe.prototxt",
                        "tests/fixtures/model_caffe_v1_deploy.prototxt"}) {
    for (const char* ext : {"", "txt"}) {
      if (!fixture_format(p, ext, f, r)) return;
      CHECK_MESSAGE(f == Format::Caffe, p);
      CHECK(r == DetectReason::Structure);
    }
  }
}

TEST_CASE("detect Caffe D3: real protobuf fixtures keep their formats") {
  Format f = Format::Unknown;
  DetectReason r = DetectReason::None;
  for (const char* ext : {"", "onnx", "caffemodel"}) {
    if (!fixture_format("tests/fixtures/model.onnx", ext, f, r)) return;
    CHECK_MESSAGE(f == Format::ONNX, ext);
  }
  for (const char* ext : {"", "pb", "caffemodel"}) {
    if (!fixture_format("tests/fixtures/model_frozen.pb", ext, f, r)) return;
    CHECK_MESSAGE(f == Format::TensorFlow, ext);
  }
  if (!fixture_format("tests/fixtures/saved_model/saved_model.pb", "", f, r)) return;
  CHECK(f == Format::TensorFlow);
  // CoreML's extension-less verdict belongs to #166's tests; here it only must
  // never become Caffe.
  for (const char* ext : {"", "mlmodel", "caffemodel"}) {
    if (!fixture_format("tests/fixtures/model.mlmodel", ext, f, r)) return;
    CHECK_MESSAGE(f != Format::Caffe, ext);
  }
  if (!fixture_format("tests/fixtures/model.xml", "", f, r)) return;
  CHECK(f == Format::OpenVINO);

  // The sniffs themselves reject every non-Caffe fixture (detection order aside).
  for (const char* p :
       {"tests/fixtures/model.onnx", "tests/fixtures/model_frozen.pb",
        "tests/fixtures/saved_model/saved_model.pb", "tests/fixtures/model.mlmodel",
        "tests/fixtures/model_mlprogram_deep.mlmodel", "tests/fixtures/model.tflite",
        "tests/fixtures/model.xml", "tests/fixtures/model.gguf",
        "tests/fixtures/model.safetensors"}) {
    if (!std::filesystem::exists(p)) continue;
    auto mf = MappedFile::open(p);
    REQUIRE(mf);
    CHECK_FALSE_MESSAGE(caffe::looks_like_caffemodel(mf->data(), mf->size()), p);
    CHECK_FALSE_MESSAGE(caffe::looks_like_prototxt(mf->data(), mf->size()), p);
  }

  // The synthetic ONNX and CoreML buffers used above are not Caffe either.
  const std::vector<uint8_t> onnx = {0x08, 0x07, 0x3a, 0x02, 0x00, 0x00};
  const std::vector<uint8_t> coreml = {0x08, 0x04, 0xa2, 0x1f, 0x00};
  CHECK(detect_bytes("caffe_onnx", onnx, "") != Format::Caffe);
  CHECK(detect_bytes("caffe_onnx2", onnx, "caffemodel") == Format::ONNX);
  CHECK(detect_bytes("caffe_coreml", coreml, "") != Format::Caffe);
}

TEST_CASE("detect Caffe D4: near-misses are not Caffe by content") {
  const std::string tf_text = "node { name: \"x\" op: \"Placeholder\" }\n";
  CHECK(detect_bytes("caffe_pbtxt", bytes_of(tf_text), "") != Format::Caffe);
  CHECK(detect_bytes("caffe_pbtxt2", bytes_of(tf_text), "pbtxt") != Format::Caffe);

  const std::string solver = "net: \"train.prototxt\"\nbase_lr: 0.01\n";
  DetectReason r = DetectReason::None;
  CHECK(detect_bytes_reason("caffe_solver", bytes_of(solver), "", r) == Format::Unknown);
  CHECK(detect_bytes_reason("caffe_solver2", bytes_of(solver), "prototxt", r) == Format::Caffe);
  CHECK(r == DetectReason::Extension);

  CHECK(detect_bytes("caffe_comment", bytes_of("# nothing\n"), "") == Format::Unknown);
  CHECK(detect_bytes("caffe_emptylayer", bytes_of("layer { }\n"), "") != Format::Caffe);
  CHECK(detect_bytes("caffe_badtype", bytes_of("layer { type: \"a b\" }\n"), "") != Format::Caffe);
  CHECK(detect_bytes("caffe_v1bad", bytes_of("layers { type: 40 }\n"), "") != Format::Caffe);
  CHECK(detect_bytes("caffe_v1ok", bytes_of("layers { type: 39 }\n"), "") == Format::Caffe);
  CHECK(detect_bytes("caffe_v0", bytes_of("layers { layer { name: \"a\" } }\n"), "") ==
        Format::Caffe);

  // name + field 2 holding {field 1 varint} (a GraphDef VersionDef's shape).
  const std::vector<uint8_t> versions = {0x0A, 0x01, 'n', 0x12, 0x02, 0x08, 0x01};
  CHECK(detect_bytes("caffe_versions", versions, "") != Format::Caffe);
}
