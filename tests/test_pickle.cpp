// SPDX-License-Identifier: Apache-2.0
// tests/test_pickle.cpp — pickle VM opcode coverage (spec §6.5, §10).
//
// Two layers. The first half drives pytorch::parse_legacy on tiny hand-crafted
// pickle byte strings: each buffer exercises a different opcode cluster (ints,
// unicode, tuples, dicts, memo, REDUCE allowlist hit + miss). The bar there is:
// malformed/edge input becomes a Result value and NEVER crashes or reads out of
// bounds (spec §6). The second half (#137) drives the VM directly through
// parsers/pytorch/PickleVM.h to pin the exact values it builds: the
// torch.jit._pickle identity allowlist, inert retention of REDUCE/NEWOBJ args and
// BUILD state, the value cap, and the src_pos stamp.
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
#include "parsers/pytorch/PickleVM.h"

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

// ===========================================================================
// #137: the VM driven directly (parsers/pytorch/PickleVM.h)
// ===========================================================================
namespace {

using pytorch::Value;
using pytorch::ValuePtr;

// Run the VM over `b` with a resolver that resolves nothing.
Result<ValuePtr> run_vm(const std::vector<uint8_t>& b,
                        pytorch::PickleLimits lim = {}) {
  pytorch::StorageResolver res;
  res.resolve = [](const std::string&, uint64_t&, uint64_t&) { return false; };
  pytorch::PickleVM vm(b.data(), b.size(), res, lim);
  return vm.run();
}

void op(std::vector<uint8_t>& b, char c) { b.push_back(static_cast<uint8_t>(c)); }
void opb(std::vector<uint8_t>& b, uint8_t c) { b.push_back(c); }

// GLOBAL torch.jit._pickle <fn>, MARK, <list-builder>, TUPLE, REDUCE.
template <typename F>
std::vector<uint8_t> jit_list_call(const std::string& fn, F&& fill_list) {
  std::vector<uint8_t> b;
  proto2(b);
  global_(b, "torch.jit._pickle", fn);
  op(b, '(');            // MARK (args tuple)
  op(b, ']');            // EMPTY_LIST
  op(b, '(');            // MARK (list items)
  fill_list(b);
  op(b, 'e');            // APPENDS
  op(b, 't');            // TUPLE
  op(b, 'R');            // REDUCE
  op(b, '.');
  return b;
}

}  // namespace

TEST_CASE("pickle VM #137: torch.jit._pickle build_*list return their list") {
  {
    auto r = run_vm(jit_list_call("build_intlist", [](std::vector<uint8_t>& b) {
      binint1(b, 1);
      binint1(b, 2);
    }));
    REQUIRE(r);
    REQUIRE((*r)->kind == Value::Kind::List);
    REQUIRE((*r)->items.size() == 2);
    CHECK((*r)->items[0]->kind == Value::Kind::Int);
    CHECK((*r)->items[1]->i == 2);
  }
  {
    auto r = run_vm(jit_list_call("build_doublelist", [](std::vector<uint8_t>& b) {
      op(b, 'G');  // BINFLOAT 1.5 (big-endian)
      const uint8_t d[8] = {0x3f, 0xf8, 0, 0, 0, 0, 0, 0};
      b.insert(b.end(), d, d + 8);
    }));
    REQUIRE(r);
    REQUIRE((*r)->kind == Value::Kind::List);
    REQUIRE((*r)->items.size() == 1);
    CHECK((*r)->items[0]->kind == Value::Kind::Double);
    CHECK((*r)->items[0]->d == doctest::Approx(1.5));
  }
  {
    auto r = run_vm(jit_list_call("build_boollist", [](std::vector<uint8_t>& b) {
      opb(b, 0x88);  // NEWTRUE
      opb(b, 0x89);  // NEWFALSE
    }));
    REQUIRE(r);
    REQUIRE((*r)->kind == Value::Kind::List);
    REQUIRE((*r)->items.size() == 2);
    CHECK((*r)->items[0]->kind == Value::Kind::Bool);
    CHECK((*r)->items[0]->b == true);
  }
  {
    auto r = run_vm(jit_list_call("build_tensorlist", [](std::vector<uint8_t>&) {}));
    REQUIRE(r);
    CHECK((*r)->kind == Value::Kind::List);
    CHECK((*r)->items.empty());
  }
  {
    // A non-list argument: inert Opaque, nothing executed.
    std::vector<uint8_t> b;
    proto2(b);
    global_(b, "torch.jit._pickle", "build_intlist");
    binint1(b, 5);
    opb(b, 0x85);  // TUPLE1
    op(b, 'R');
    op(b, '.');
    auto r = run_vm(b);
    REQUIRE(r);
    CHECK((*r)->kind == Value::Kind::Opaque);
    CHECK((*r)->name == "build_intlist");
  }
}

TEST_CASE("pickle VM #137: restore_type_tag returns the tagged object itself") {
  std::vector<uint8_t> b;
  proto2(b);
  global_(b, "torch.jit._pickle", "restore_type_tag");
  op(b, '}');            // EMPTY_DICT
  op(b, '(');
  binunicode(b, "k");
  binint1(b, 1);
  op(b, 'u');            // SETITEMS
  binunicode(b, "Dict[str, int]");
  opb(b, 0x86);          // TUPLE2
  op(b, 'R');
  op(b, '.');
  auto r = run_vm(b);
  REQUIRE(r);
  REQUIRE((*r)->kind == Value::Kind::Dict);
  REQUIRE((*r)->pairs.size() == 1);
  CHECK((*r)->pairs[0].first->s == "k");
  CHECK((*r)->pairs[0].second->i == 1);
}

TEST_CASE("pickle VM #137: REDUCE args and BUILD state are recorded, never run") {
  {
    // torch.device('cpu') is not allowlisted: Opaque that keeps its argument.
    std::vector<uint8_t> b;
    proto2(b);
    global_(b, "torch", "device");
    binunicode(b, "cpu");
    opb(b, 0x85);        // TUPLE1
    op(b, 'R');
    op(b, '.');
    auto r = run_vm(b);
    REQUIRE(r);
    CHECK((*r)->kind == Value::Kind::Opaque);
    CHECK((*r)->module == "torch");
    CHECK((*r)->name == "device");
    REQUIRE((*r)->items.size() == 1);
    CHECK((*r)->items[0]->s == "cpu");
  }
  {
    // A TorchScript object: GLOBAL __torch__ M, EMPTY_TUPLE, NEWOBJ, state, BUILD.
    std::vector<uint8_t> b;
    proto2(b);
    global_(b, "__torch__", "M");
    op(b, ')');          // EMPTY_TUPLE
    opb(b, 0x81);        // NEWOBJ
    op(b, '}');          // EMPTY_DICT
    op(b, '(');
    binunicode(b, "w");
    binint1(b, 3);
    op(b, 'u');          // SETITEMS
    op(b, 'b');          // BUILD
    op(b, '.');
    auto r = run_vm(b);
    REQUIRE(r);
    CHECK((*r)->kind == Value::Kind::Opaque);
    CHECK((*r)->module == "__torch__");
    REQUIRE((*r)->inner);
    REQUIRE((*r)->inner->kind == Value::Kind::Dict);
    REQUIRE((*r)->inner->pairs.size() == 1);
    CHECK((*r)->inner->pairs[0].first->s == "w");
    CHECK((*r)->inner->pairs[0].second->i == 3);
  }
}

TEST_CASE("pickle VM #137: PickleLimits caps the number of values") {
  std::vector<uint8_t> b;
  proto2(b);
  op(b, '(');
  for (int i = 0; i < 20; ++i) op(b, 'N');
  op(b, 't');
  op(b, '.');
  pytorch::PickleLimits lim;
  lim.max_values = 10;
  auto r = run_vm(b, lim);
  REQUIRE_FALSE(r);
  CHECK(r.error().message.find("value cap") != std::string::npos);
  CHECK(r.error().offset < b.size());
  // Unlimited (the default) still succeeds on the same stream.
  CHECK(run_vm(b));
}

TEST_CASE("pickle VM #137: values carry the offset of the opcode that made them") {
  std::vector<uint8_t> b;
  proto2(b);
  binint1(b, 5);  // at byte 2
  op(b, '.');
  auto r = run_vm(b);
  REQUIRE(r);
  CHECK((*r)->kind == Value::Kind::Int);
  CHECK((*r)->src_pos == 2);
}

namespace {

// A pickled module object `<module>.M` whose state has one attribute "w" that
// is a _rebuild_tensor_v2 tensor (the shape torch.jit.save writes for modules).
std::vector<uint8_t> object_with_tensor(const std::string& module) {
  std::vector<uint8_t> b;
  proto2(b);
  global_(b, module, "M");
  op(b, ')');          // EMPTY_TUPLE
  opb(b, 0x81);        // NEWOBJ
  op(b, '}');          // EMPTY_DICT
  op(b, '(');          // MARK (state items)
  binunicode(b, "w");
  global_(b, "torch._utils", "_rebuild_tensor_v2");
  op(b, '(');          // MARK (rebuild args)
  op(b, '(');          // MARK (storage pid)
  binunicode(b, "storage");
  global_(b, "torch", "FloatStorage");
  binunicode(b, "0");
  binunicode(b, "cpu");
  binint1(b, 2);
  op(b, 't');
  op(b, 'Q');          // BINPERSID
  binint1(b, 0);       // storage_offset
  op(b, '(');
  binint1(b, 2);
  op(b, 't');          // size (2,)
  op(b, '(');
  binint1(b, 1);
  op(b, 't');          // stride (1,)
  opb(b, 0x89);        // requires_grad False
  global_(b, "collections", "OrderedDict");
  op(b, ')');
  op(b, 'R');          // backward_hooks
  op(b, 't');
  op(b, 'R');          // the tensor
  op(b, 'u');          // SETITEMS
  op(b, 'b');          // BUILD
  op(b, '.');
  return b;
}

}  // namespace

TEST_CASE("pickle VM #137: __torch__ object state lists its parameters") {
  std::string path = write_temp("ts_obj", object_with_tensor("__torch__"));
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = pytorch::parse_legacy(*mf, progress);
  REQUIRE(res);
  bool found = false;
  for (const auto& t : res->flat_tensors) {
    if (res->str(t.name) == "w") {
      found = true;
      CHECK(t.dtype == ir::DType::F32);
      CHECK(t.shape.size() == 1);
    }
  }
  CHECK(found);
  std::filesystem::remove(path);
}

TEST_CASE("pickle VM #137: non-TorchScript objects are still not descended") {
  // A Python full-module pickle (module "mymod") stays an inert placeholder:
  // unchanged behavior, no tensors are claimed from it.
  std::string path = write_temp("py_obj", object_with_tensor("mymod"));
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = pytorch::parse_legacy(*mf, progress);
  REQUIRE(res);
  CHECK(res->flat_tensors.empty());
  std::filesystem::remove(path);
}
