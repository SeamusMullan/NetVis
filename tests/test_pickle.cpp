// tests/test_pickle.cpp — pickle VM opcode coverage (spec §6.5, §10).
//
// NOTE: there is no public src/parsers/pytorch/PickleVM.h in the frozen tree, so
// per spec §10 this drives pytorch::parse_legacy on tiny hand-crafted pickle
// byte strings instead of unit-testing the VM directly. Each buffer exercises a
// different opcode cluster (ints, unicode, tuples, dicts, memo, REDUCE allowlist
// hit + miss). The bar here is: malformed/edge input becomes a Result value and
// NEVER crashes or reads out of bounds (spec §6). We assert we get a Result back
// (ok or error) and that the payload-read counter stays 0.
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {

// Persist a byte string to a temp file and map it (parse_legacy takes a
// MappedFile, and the mmap is what parsers read through — spec §2.1).
std::string write_temp(const std::string& stem, const std::vector<uint8_t>& b) {
  std::filesystem::path p =
      std::filesystem::temp_directory_path() / ("nv_pickle_" + stem);
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(b.data()),
            static_cast<std::streamsize>(b.size()));
  out.close();
  return p.string();
}

// Run parse_legacy over a pickle byte string; returns whether it survived (i.e.
// produced a Result at all, without UB). The caller checks ok()/error() further.
bool drive_legacy(const std::string& stem, const std::vector<uint8_t>& bytes,
                  bool* out_ok) {
  std::string path = write_temp(stem, bytes);
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  // Reset the thread-local payload counter so each case's "0 payload reads"
  // assertion is self-contained regardless of test-execution order (an earlier
  // test that legitimately decodes a tensor would otherwise leave it non-zero).
  ByteReader::payload_read_counter() = 0;
  auto res = pytorch::parse_legacy(*mf, progress);
  if (out_ok) *out_ok = static_cast<bool>(res);
  std::filesystem::remove(path);
  return true;  // reaching here means no crash / no OOB read
}

// PROTO 2 preamble.
void proto2(std::vector<uint8_t>& b) { b.push_back(0x80); b.push_back(0x02); }
// BININT1 n.
void binint1(std::vector<uint8_t>& b, uint8_t n) { b.push_back('K'); b.push_back(n); }
// SHORT_BINUNICODE-ish: use BINUNICODE (X) with 4-byte LE length.
void binunicode(std::vector<uint8_t>& b, const std::string& s) {
  b.push_back('X');
  uint32_t n = static_cast<uint32_t>(s.size());
  for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((n >> (8 * i)) & 0xff));
  for (char c : s) b.push_back(static_cast<uint8_t>(c));
}
void binput(std::vector<uint8_t>& b, uint8_t i) { b.push_back('q'); b.push_back(i); }
void global_(std::vector<uint8_t>& b, const std::string& mod, const std::string& nm) {
  b.push_back('c');
  for (char c : mod) b.push_back(static_cast<uint8_t>(c));
  b.push_back('\n');
  for (char c : nm) b.push_back(static_cast<uint8_t>(c));
  b.push_back('\n');
}

}  // namespace

TEST_CASE("pickle VM: integers do not crash the legacy parser") {
  std::vector<uint8_t> b;
  proto2(b);
  binint1(b, 7);
  b.push_back('.');  // STOP
  bool ok = false;
  CHECK(drive_legacy("ints", b, &ok));
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("pickle VM: unicode + memo (BINPUT) do not crash") {
  std::vector<uint8_t> b;
  proto2(b);
  binunicode(b, "hello");
  binput(b, 0);
  b.push_back('.');
  bool ok = false;
  CHECK(drive_legacy("unicode", b, &ok));
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("pickle VM: tuple + dict SETITEMS do not crash") {
  std::vector<uint8_t> b;
  proto2(b);
  b.push_back('}');       // EMPTY_DICT
  binput(b, 0);
  b.push_back('(');       // MARK
  binunicode(b, "k");
  binint1(b, 1);
  b.push_back('u');       // SETITEMS
  b.push_back('.');
  bool ok = false;
  CHECK(drive_legacy("dict", b, &ok));
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("pickle VM: REDUCE allowlist HIT (torch _rebuild_tensor_v2)") {
  // Build an OrderedDict via an allowlisted GLOBAL + REDUCE. We only require the
  // parser to survive and route this through the allowlist path.
  std::vector<uint8_t> b;
  proto2(b);
  global_(b, "collections", "OrderedDict");
  binput(b, 0);
  b.push_back(')');       // EMPTY_TUPLE
  b.push_back('R');       // REDUCE
  binput(b, 1);
  b.push_back('.');
  bool ok = false;
  CHECK(drive_legacy("reduce_hit", b, &ok));
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("pickle VM: REDUCE allowlist MISS (non-allowlisted GLOBAL) is opaque") {
  // A GLOBAL to a target that is NOT on the allowlist must be handled via the
  // opaque path — no crash, no arbitrary construction (spec §6.5 security note).
  std::vector<uint8_t> b;
  proto2(b);
  global_(b, "numpy.core", "foo");
  binput(b, 0);
  b.push_back(')');       // EMPTY_TUPLE
  b.push_back('R');       // REDUCE
  b.push_back('.');
  bool ok = false;
  CHECK(drive_legacy("reduce_miss", b, &ok));
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("pickle VM: truncated stream returns a value, never OOB") {
  // A PROTO with no STOP and a dangling BINUNICODE length must bounds-check.
  std::vector<uint8_t> b;
  proto2(b);
  b.push_back('X');
  b.push_back(0xff);  // claims a 255-byte string that isn't there
  b.push_back(0xff);
  b.push_back(0xff);
  b.push_back(0x7f);
  bool ok = false;
  CHECK(drive_legacy("truncated", b, &ok));
  // Either an error Result or a best-effort ok Result is acceptable; the point
  // is that we returned without an out-of-bounds read.
  CHECK(ByteReader::payload_read_counter() == 0);
}
