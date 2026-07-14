// tests/test_npy.cpp — NumPy .npy export round-trip (spec §7.5).
//
// Creates a temp file of known F32 values, maps it, builds a TensorRef over it,
// calls export_npy, then re-reads the .npy header + data and asserts the magic,
// dtype descr, shape, and float values all round-trip. export_npy is the one
// place that reads payloads, so it may bump the payload-read counter.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/MappedFile.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

std::string temp_path(const std::string& stem) {
  return (std::filesystem::temp_directory_path() / ("nv_npy_" + stem)).string();
}

}  // namespace

TEST_CASE("export_npy round-trips a small F32 tensor") {
  // 6 known F32 values as a 2x3 tensor.
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Write the raw payload to a source file and map it (the "model" mmap).
  std::string src = temp_path("src.bin");
  {
    std::ofstream out(src, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
  }
  auto mf = MappedFile::open(src);
  REQUIRE(mf);

  // A TensorRef pointing at offset 0 of that mmap.
  ir::TensorRef t;
  t.dtype = ir::DType::F32;
  t.shape = {2, 3};
  t.file_offset = 0;
  t.byte_len = values.size() * sizeof(float);

  std::string out_npy = temp_path("out.npy");
  auto r = export_npy(t, *mf, /*model_dir=*/"", out_npy);
  REQUIRE_MESSAGE(r, "export_npy returned an error");
  CHECK(*r);

  // --- re-read and validate the .npy file ----------------------------------
  std::ifstream in(out_npy, std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
  in.close();
  REQUIRE(bytes.size() >= 10);

  // Magic string "\x93NUMPY" + version 1.0.
  CHECK(bytes[0] == static_cast<char>(0x93));
  CHECK(std::string(bytes.data() + 1, 5) == "NUMPY");
  CHECK(bytes[6] == 0x01);  // major version
  CHECK(bytes[7] == 0x00);  // minor version

  // Header length is a u16 LE at bytes 8..10; header text follows.
  uint16_t hlen = 0;
  std::memcpy(&hlen, bytes.data() + 8, 2);
  size_t header_start = 10;
  REQUIRE(bytes.size() >= header_start + hlen);
  std::string header(bytes.data() + header_start, hlen);

  // Descr must be little-endian float32 ('<f4' or '|f4'/'=f4' acceptable variants)
  // and the shape must serialize (2, 3).
  CHECK(header.find("f4") != std::string::npos);
  CHECK(header.find("'descr'") != std::string::npos);
  CHECK(header.find("'shape'") != std::string::npos);
  CHECK(header.find("(2, 3)") != std::string::npos);
  CHECK(header.find("'fortran_order': False") != std::string::npos);

  // Data region begins right after the header and must equal the source floats.
  size_t data_off = header_start + hlen;
  REQUIRE(bytes.size() >= data_off + values.size() * sizeof(float));
  std::vector<float> got(values.size());
  std::memcpy(got.data(), bytes.data() + data_off, values.size() * sizeof(float));
  for (size_t i = 0; i < values.size(); ++i) {
    CHECK(got[i] == doctest::Approx(values[i]));
  }

  std::filesystem::remove(src);
  std::filesystem::remove(out_npy);
}
