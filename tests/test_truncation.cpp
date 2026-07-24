// tests/test_truncation.cpp — malformed/truncated input safety (spec §6).
//
// For each fixture that exists, truncate it at every 1/8th of its length, map
// the prefix, and run the matching parser. The contract: a truncated file must
// yield a Result error (or at worst a best-effort partial model) but NEVER
// crash and NEVER read out of bounds — every read goes through the bounds-
// checked ByteReader. We also assert the payload-read counter stays 0.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "engine/ModelPath.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {

using ParseFn = std::function<Result<ir::Model>(const MappedFile&, ProgressSink&)>;

// Read a whole fixture into memory (small fixtures).
std::vector<uint8_t> read_all(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
  return bytes;
}

// Write a prefix of `bytes` to a temp file and return its path.
std::string write_prefix(const std::string& stem, const std::vector<uint8_t>& bytes,
                         size_t n) {
  std::filesystem::path p =
      std::filesystem::temp_directory_path() / ("nv_trunc_" + stem);
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(n));
  out.close();
  return p.string();
}

// Truncate `full_path` at each 1/8th and run `parse`. Asserts no crash; a
// truncated stream must not leave the payload counter dirty.
void truncation_sweep(const std::string& stem, const std::string& full_path,
                      const ParseFn& parse) {
  if (!std::filesystem::exists(full_path)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  std::vector<uint8_t> bytes = read_all(full_path);
  REQUIRE(bytes.size() > 0);

  // Truncate at 1/8..7/8 of the length (a full-length parse is covered by the
  // per-format success tests). Include 0 bytes as the degenerate empty case.
  for (int k = 0; k <= 7; ++k) {
    size_t n = (bytes.size() * static_cast<size_t>(k)) / 8;
    std::string path = write_prefix(stem, bytes, n);

    ByteReader::payload_read_counter() = 0;
    auto mf = MappedFile::open(path);
    // A zero-byte file may fail to map on some platforms; that is itself a
    // graceful failure, not a crash.
    if (mf) {
      ProgressSink progress;
      auto res = parse(*mf, progress);
      // We accept ok or error — the invariant is "returned safely". Malformed
      // input that survives must not have decoded any payload.
      (void)res;
      CHECK(ByteReader::payload_read_counter() == 0);
    }
    std::filesystem::remove(path);
  }
}

std::string fixture(const char* name) { return std::string("tests/fixtures/") + name; }

}  // namespace

TEST_CASE("truncation: ONNX parser stays safe at every 1/8th") {
  truncation_sweep("onnx", fixture("model.onnx"),
                   [](const MappedFile& f, ProgressSink& p) { return onnx::parse(f, p); });
}

TEST_CASE("truncation: SafeTensors parser stays safe at every 1/8th") {
  truncation_sweep("safetensors", fixture("model.safetensors"),
                   [](const MappedFile& f, ProgressSink& p) { return safetensors::parse(f, p); });
}

TEST_CASE("truncation: GGUF parser stays safe at every 1/8th") {
  truncation_sweep("gguf", fixture("model.gguf"),
                   [](const MappedFile& f, ProgressSink& p) { return gguf::parse(f, p); });
}

TEST_CASE("truncation: PyTorch zip parser stays safe at every 1/8th") {
  truncation_sweep("pt", fixture("model.pt"),
                   [](const MappedFile& f, ProgressSink& p) { return pytorch::parse_zip(f, p); });
}

TEST_CASE("truncation: TFLite parser stays safe at every 1/8th") {
  truncation_sweep("tflite", fixture("model.tflite"),
                   [](const MappedFile& f, ProgressSink& p) { return tflite::parse(f, p); });
}

TEST_CASE("truncation: OpenVINO IR parser stays safe at every 1/8th") {
  truncation_sweep("ovxml", fixture("model.xml"),
                   [](const MappedFile& f, ProgressSink& p) { return openvino::parse(f, p); });
}

TEST_CASE("truncation: NumPy .npz parser stays safe at every 1/8th") {
  truncation_sweep("npz", fixture("model.npz"),
                   [](const MappedFile& f, ProgressSink& p) { return npz::parse(f, p); });
}

TEST_CASE("truncation: CoreML parser stays safe at every 1/8th") {
  truncation_sweep("mlmodel", fixture("model.mlmodel"),
                   [](const MappedFile& f, ProgressSink& p) { return coreml::parse(f, p); });
}

TEST_CASE("truncation: Keras HDF5 parser stays safe at every 1/8th") {
  truncation_sweep("keras_h5", fixture("model.h5"),
                   [](const MappedFile& f, ProgressSink& p) { return keras::parse(f, p); });
  truncation_sweep("keras_v3", fixture("model.keras"),
                   [](const MappedFile& f, ProgressSink& p) { return keras::parse(f, p); });
}

TEST_CASE("truncation: mlPackage inner model.mlmodel stays safe at every 1/8th") {
  // The mlProgram fixture: must truncate the INNER model.mlmodel file, not the
  // bundle directory. resolve_model_path gives us the inner file path.
  const std::string bundle_path = fixture("model.mlpackage");
  if (!std::filesystem::exists(bundle_path)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  auto resolved = resolve_model_path(bundle_path);
  truncation_sweep("mlprogram_inner", resolved.map_path,
                   [](const MappedFile& f, ProgressSink& p) { return coreml::parse(f, p); });
}

TEST_CASE("truncation: mlPackage weight.bin stays safe at every 1/8th") {
  // Truncate the external weight.bin file. Open it with a pass-through parser
  // (we don't parse weight.bin standalone; just test that truncating it doesn't
  // crash when compute_tensor_stats attempts to follow the blob_metadata header).
  const std::string bundle_path = fixture("model.mlpackage");
  if (!std::filesystem::exists(bundle_path)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  // Construct the weight.bin path from the bundle structure.
  std::filesystem::path weight_bin_path =
      std::filesystem::path(bundle_path) / "Data" / "com.apple.CoreML" / "weights" / "weight.bin";
  if (!std::filesystem::exists(weight_bin_path)) {
    WARN_MESSAGE(false, "weight.bin fixture missing; run tools/gen_fixtures.py");
    return;
  }

  std::vector<uint8_t> bytes = read_all(weight_bin_path.string());
  REQUIRE(bytes.size() > 0);

  // Drive the ACTUAL blob-header follow (follow_blob_header via resolve_payload)
  // against each truncated prefix: a blob_indirect TensorRef pointing at offset
  // 64 (the metadata header). A truncated/short weight.bin must yield a clean
  // Result error, never an out-of-bounds read (ASan/UBSan catch a regression).
  for (int k = 0; k <= 8; ++k) {
    size_t n = (bytes.size() * static_cast<size_t>(k)) / 8;
    std::string path = write_prefix("weight_bin", bytes, n);

    auto mf = MappedFile::open(path);
    if (mf) {
      ir::Model m;
      ir::TensorRef t;
      t.name = m.intern("w");
      t.external_path = m.intern(path);   // absolute path -> used verbatim
      t.file_offset = 64;                  // the blob_metadata header offset
      t.blob_indirect = true;
      t.dtype = ir::DType::F32;
      ByteReader::payload_read_counter() = 0;
      // model_dir empty: external_path is absolute, resolved as-is.
      auto stats = compute_tensor_stats(t, *mf, std::string(), &m);
      // Full-length (k==8) should succeed; truncated prefixes error cleanly.
      // The invariant under test is "no crash / no UB", not a specific verdict.
      (void)stats;
    }
    std::filesystem::remove(path);
  }
}
