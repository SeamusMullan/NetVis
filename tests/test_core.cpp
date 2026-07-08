// tests/test_core.cpp — foundation contract tests (core + ir).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "core/ByteReader.h"
#include "core/Hash.h"
#include "core/Result.h"
#include "core/SmallVec.h"
#include "core/StringArena.h"
#include "ir/IR.h"

using namespace netvis;

TEST_CASE("Result carries value or error") {
  Result<int> ok = 42;
  CHECK(ok);
  CHECK(*ok == 42);
  Result<int> bad = err("nope", 7);
  CHECK_FALSE(bad);
  CHECK(bad.error().offset == 7);
}

TEST_CASE("SmallVec inline then spill") {
  SmallVec<int64_t, 6> v;
  for (int i = 0; i < 3; ++i) v.push_back(i);
  CHECK(v.size() == 3);
  CHECK(v[2] == 2);
  for (int i = 3; i < 20; ++i) v.push_back(i);  // force heap spill
  CHECK(v.size() == 20);
  CHECK(v[19] == 19);
  SmallVec<int64_t, 6> w = v;  // copy of spilled
  CHECK(w == v);
  SmallVec<int64_t, 6> m = std::move(w);  // move of spilled
  CHECK(m.size() == 20);
  CHECK(m[0] == 0);
}

TEST_CASE("StringArena interns and stays stable across growth") {
  StringArena a;
  CHECK(a.get(StringId{0}).empty());
  StringId x = a.intern("Conv");
  StringId y = a.intern("Conv");
  CHECK(x == y);  // deduped
  StringId first = a.intern("first");
  // Force many inserts; earlier views must remain valid (deque stability).
  for (int i = 0; i < 5000; ++i) a.intern("k" + std::to_string(i));
  CHECK(a.get(first) == "first");
  CHECK(a.get(x) == "Conv");
}

TEST_CASE("ByteReader bounds-checks and never over-reads") {
  const uint8_t buf[] = {0x01, 0x00, 0x00, 0x00, 0xff};
  ByteReader r(buf, sizeof(buf));
  auto v = r.u32le();
  CHECK(v);
  CHECK(*v == 1u);
  CHECK(r.u8());       // 0xff, ok
  CHECK_FALSE(r.u8()); // eof -> error, not crash
}

TEST_CASE("ByteReader payload counter isolates structural reads") {
  ByteReader::payload_read_counter() = 0;
  const uint8_t buf[] = {1, 2, 3, 4};
  ByteReader r(buf, 4);
  (void)r.u32le();  // structural
  CHECK(ByteReader::payload_read_counter() == 0);
  ByteReader::mark_payload_read();
  CHECK(ByteReader::payload_read_counter() == 1);
}

TEST_CASE("FNV-1a is order sensitive and deterministic") {
  CHECK(fnv1a("ab") == fnv1a("ab"));
  CHECK(fnv1a("ab") != fnv1a("ba"));
}

TEST_CASE("IR TensorRef element count handles dynamic dims") {
  ir::TensorRef t;
  t.shape = {1, 3, 224, 224};
  CHECK(t.elem_count() == 1 * 3 * 224 * 224);
  t.shape = {-1, 3};
  CHECK(t.elem_count() == 0);  // dynamic -> 0
}

TEST_CASE("IR dtype tables") {
  CHECK(std::string(ir::dtype_name(ir::DType::F32)) == "f32");
  CHECK(ir::dtype_size(ir::DType::F64) == 8);
  CHECK(ir::dtype_size(ir::DType::Q4) == 0);  // quantized: block-based
}
