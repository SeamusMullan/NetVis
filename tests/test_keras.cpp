// tests/test_keras.cpp — Keras .h5 / .keras parser contract (spec §6, §10).
//
// Keras has no compute graph, so has_graph must be false and datasets land in
// flat_tensors. Asserts the hand-encoded classic-form HDF5 fixture surfaces one
// F32 [2,2] dataset "kernel" with a real contiguous offset+len, that the .keras
// v3 zip resolves its embedded model.weights.h5 to the same, and — the whole
// product thesis — zero payload reads during structural parse (§2.1).
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {
const char* kH5 = "tests/fixtures/model.h5";
const char* kKeras = "tests/fixtures/model.keras";

const ir::TensorRef* find_named(const ir::Model& m, const std::string& want) {
  for (const auto& t : m.flat_tensors) {
    if (std::string(m.str(t.name)) == want) return &t;
  }
  return nullptr;
}
}  // namespace

TEST_CASE("Keras raw .h5: one contiguous F32 dataset, tensor-table, no payload reads") {
  if (!std::filesystem::exists(kH5)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kH5);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = keras::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "keras::parse returned an error");

  const ir::Model& model = *res;
  CHECK(model.has_graph == false);

  const ir::TensorRef* k = find_named(model, "kernel");
  REQUIRE_MESSAGE(k != nullptr, "expected a dataset named 'kernel'");
  CHECK(k->dtype == ir::DType::F32);
  REQUIRE(k->shape.size() == 2);
  CHECK(k->shape[0] == 2);
  CHECK(k->shape[1] == 2);
  // Contiguous layout -> a real, addressable payload of 16 bytes (4 * f32).
  CHECK(k->file_offset != UINT64_MAX);
  CHECK(k->byte_len == 16);
  // The recorded offset must point inside the mapped file.
  CHECK(k->file_offset + k->byte_len <= mf->size());

  // The critical invariant: structural parse never touched a payload byte.
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Keras v3 .keras: embedded weights.h5 resolves to absolute offset") {
  if (!std::filesystem::exists(kKeras)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  ByteReader::payload_read_counter() = 0;

  auto mf = MappedFile::open(kKeras);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = keras::parse(*mf, progress);
  REQUIRE_MESSAGE(res, "keras::parse returned an error on .keras archive");

  const ir::Model& model = *res;
  CHECK(model.has_graph == false);

  const ir::TensorRef* k = find_named(model, "kernel");
  REQUIRE_MESSAGE(k != nullptr, "expected 'kernel' from embedded model.weights.h5");
  CHECK(k->dtype == ir::DType::F32);
  CHECK(k->byte_len == 16);
  // Offset was rebased to an absolute position inside the OUTER .keras zip.
  CHECK(k->file_offset != UINT64_MAX);
  CHECK(k->file_offset + k->byte_len <= mf->size());
  // It must be past the zip local file header (i.e. genuinely rebased, not 0).
  CHECK(k->file_offset > 8);

  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Keras: truncated / garbage HDF5 fails cleanly, never crashes") {
  ByteReader::payload_read_counter() = 0;
  ProgressSink progress;

  // A buffer with the HDF5 signature but nothing else valid must not crash and
  // must not fabricate datasets.
  {
    std::string path =
        (std::filesystem::temp_directory_path() / "nv_keras_trunc.h5").string();
    {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      const char sig[9] = {(char)0x89, 'H', 'D', 'F', '\r', '\n', (char)0x1a, '\n', 0};
      out.write(sig, 8);
      char junk[32] = {0};
      out.write(junk, sizeof(junk));
    }
    auto mf = MappedFile::open(path);
    REQUIRE(mf);
    auto res = keras::parse(*mf, progress);
    // Either an honest error or an empty/partial model — never a crash.
    if (res) {
      CHECK((*res).flat_tensors.empty());
    }
    std::filesystem::remove(path);
  }

  CHECK(ByteReader::payload_read_counter() == 0);
}
