// tests/test_tensor_diff.cpp — cross-model tensor matching (#34/#50), PHASE 1
// only (enumerate_tensors / resolve_tensor / find_tensor_by_name /
// match_tensors). PHASE 2 (compute_tensor_stat_delta) reads payload through
// the same compute_tensor_stats path already covered by test_tensor_stats.cpp,
// so it is not re-tested here; this file is about the free, payload-less
// matching logic that fills the comparison table every frame.
//
// Mirrors test_modeldiff.cpp's idiom: hand-build ir::Models directly (no
// parser), using INDEPENDENT StringArenas for the "model A" / "model B" pairs
// so matching-by-content (never by raw StringId) is actually exercised.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "engine/TensorDiff.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// Append a tensor to `m` (either flat_tensors or, if `graph` is given, that
// graph's initializers) and return nothing — callers read the model back
// through enumerate_tensors/resolve_tensor like production code does, rather
// than trusting the index they just pushed at.
ir::TensorRef make_tensor(ir::Model& m, std::string_view name, ir::DType dtype,
                          const std::vector<int64_t>& shape, uint64_t byte_len) {
  ir::TensorRef t;
  t.name = m.intern(name);
  t.dtype = dtype;
  for (int64_t d : shape) t.shape.push_back(d);
  t.byte_len = byte_len;
  return t;
}

// A model with tensors in BOTH places TensorDiff addresses: one flat tensor
// (graph == -1), two initializers in graph 0, one in graph 1, and one more
// UNNAMED initializer in graph 1 — the unnamed one exists specifically so the
// find_tensor_by_name("") test has a real empty-named tensor to (fail to)
// match against, not just an absent one.
ir::Model make_mixed_model() {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");

  m.flat_tensors.push_back(make_tensor(m, "flat0", ir::DType::F32, {2}, 8));

  m.graphs.emplace_back();  // graph 0
  m.graphs[0].initializers.push_back(
      make_tensor(m, "g0a", ir::DType::F32, {3}, 12));
  m.graphs[0].initializers.push_back(
      make_tensor(m, "g0b", ir::DType::F32, {4}, 16));

  m.graphs.emplace_back();  // graph 1
  m.graphs[1].initializers.push_back(
      make_tensor(m, "g1a", ir::DType::F32, {5}, 20));
  m.graphs[1].initializers.push_back(
      make_tensor(m, "", ir::DType::F32, {1}, 4));  // unnamed

  return m;
}

}  // namespace

// --- enumerate_tensors ---------------------------------------------------------

TEST_CASE("#34 enumerate_tensors: flat tensors first, then graphs ascending") {
  ir::Model m = make_mixed_model();
  std::vector<TensorLocator> locs = enumerate_tensors(m);
  REQUIRE(locs.size() == 5);
  CHECK(locs[0] == TensorLocator{-1, 0});  // flat0
  CHECK(locs[1] == TensorLocator{0, 0});   // g0a
  CHECK(locs[2] == TensorLocator{0, 1});   // g0b
  CHECK(locs[3] == TensorLocator{1, 0});   // g1a
  CHECK(locs[4] == TensorLocator{1, 1});   // unnamed

  // The order is what the panel iterates in, so the names must line up too.
  CHECK(m.str(resolve_tensor(m, locs[0])->name) == "flat0");
  CHECK(m.str(resolve_tensor(m, locs[1])->name) == "g0a");
  CHECK(m.str(resolve_tensor(m, locs[2])->name) == "g0b");
  CHECK(m.str(resolve_tensor(m, locs[3])->name) == "g1a");
}

// --- resolve_tensor: OOB in every direction is nullptr, never a crash ----------

TEST_CASE("#34 resolve_tensor: valid locators resolve, OOB locators are nullptr") {
  ir::Model m = make_mixed_model();

  // Valid: one from flat_tensors, one from a graph's initializers.
  REQUIRE(resolve_tensor(m, TensorLocator{-1, 0}) != nullptr);
  CHECK(m.str(resolve_tensor(m, TensorLocator{-1, 0})->name) == "flat0");
  REQUIRE(resolve_tensor(m, TensorLocator{0, 1}) != nullptr);
  CHECK(m.str(resolve_tensor(m, TensorLocator{0, 1})->name) == "g0b");

  // Index out of range into flat_tensors (only 1 present).
  CHECK(resolve_tensor(m, TensorLocator{-1, 1}) == nullptr);
  // Index out of range into a graph's initializers (graph 0 has only 2).
  CHECK(resolve_tensor(m, TensorLocator{0, 2}) == nullptr);
  // Graph index out of range (only graphs 0 and 1 exist).
  CHECK(resolve_tensor(m, TensorLocator{2, 0}) == nullptr);
  CHECK(resolve_tensor(m, TensorLocator{99, 0}) == nullptr);
  // A default/invalid locator (negative index) must never be dereferenced.
  CHECK(resolve_tensor(m, TensorLocator{}) == nullptr);
}

// --- find_tensor_by_name: content match across independent arenas --------------

TEST_CASE("#34 find_tensor_by_name: matches by CONTENT, empty name never matches") {
  ir::Model a = make_mixed_model();
  // `b` is a SEPARATE Model/arena that happens to share one tensor's name —
  // find_tensor_by_name must compare bytes, not StringId (which is meaningless
  // across the two arenas).
  ir::Model b;
  b.has_graph = false;
  b.flat_tensors.push_back(make_tensor(b, "g0b", ir::DType::F32, {4}, 16));

  TensorLocator loc = find_tensor_by_name(a, "g0b");
  REQUIRE(loc.valid());
  CHECK(loc == TensorLocator{0, 1});
  CHECK(a.str(resolve_tensor(a, loc)->name) == "g0b");

  // A name absent from the model.
  CHECK_FALSE(find_tensor_by_name(a, "does_not_exist").valid());

  // An empty query never matches, even though `a` has a real unnamed tensor
  // at TensorLocator{1,1} — otherwise every unnamed tensor in one model would
  // spuriously "match" every unnamed tensor in another.
  CHECK_FALSE(find_tensor_by_name(a, "").valid());

  // Cross-arena sanity: b's locator for "g0b" is unrelated to a's.
  TensorLocator loc_b = find_tensor_by_name(b, "g0b");
  REQUIRE(loc_b.valid());
  CHECK(loc_b == TensorLocator{-1, 0});
}

// --- match_tensors ---------------------------------------------------------------

TEST_CASE("#34 match_tensors: matched pairs only, A's order, shape/dtype flags") {
  // Model A: 6 flat tensors covering every case the header documents.
  ir::Model a;
  a.has_graph = false;
  a.flat_tensors.push_back(
      make_tensor(a, "same", ir::DType::F32, {2, 3}, 24));
  a.flat_tensors.push_back(
      make_tensor(a, "rank_mismatch", ir::DType::F32, {2, 3}, 24));
  a.flat_tensors.push_back(
      make_tensor(a, "dims_mismatch", ir::DType::F32, {2, 3}, 24));
  a.flat_tensors.push_back(
      make_tensor(a, "dtype_mismatch", ir::DType::F32, {2, 2}, 16));
  a.flat_tensors.push_back(
      make_tensor(a, "only_in_a", ir::DType::F32, {1}, 4));
  a.flat_tensors.push_back(make_tensor(a, "", ir::DType::F32, {1}, 4));

  // Model B: independent arena. Same names for the first four (with the
  // deltas the test wants to see), a B-only tensor, and its own unnamed one.
  ir::Model b;
  b.has_graph = false;
  b.flat_tensors.push_back(
      make_tensor(b, "same", ir::DType::F32, {2, 3}, 24));
  b.flat_tensors.push_back(          // rank 2 -> 3: shape_equal must be false
      make_tensor(b, "rank_mismatch", ir::DType::F32, {2, 3, 4}, 96));
  b.flat_tensors.push_back(          // same rank, dim1 3 -> 5: shape_equal false
      make_tensor(b, "dims_mismatch", ir::DType::F32, {2, 5}, 40));
  b.flat_tensors.push_back(          // same shape, dtype F32 -> F16
      make_tensor(b, "dtype_mismatch", ir::DType::F16, {2, 2}, 8));
  b.flat_tensors.push_back(
      make_tensor(b, "only_in_b", ir::DType::F32, {1}, 4));
  b.flat_tensors.push_back(make_tensor(b, "", ir::DType::F32, {1}, 4));

  std::vector<TensorMatch> matches = match_tensors(a, b);

  // Exactly the 4 commonly-named tensors; only_in_a/only_in_b/unnamed omitted.
  REQUIRE(matches.size() == 4);

  // In A's enumeration order (== insertion order here, all flat_tensors).
  CHECK(a.str(resolve_tensor(a, matches[0].a)->name) == "same");
  CHECK(a.str(resolve_tensor(a, matches[1].a)->name) == "rank_mismatch");
  CHECK(a.str(resolve_tensor(a, matches[2].a)->name) == "dims_mismatch");
  CHECK(a.str(resolve_tensor(a, matches[3].a)->name) == "dtype_mismatch");

  const TensorMatch& same = matches[0];
  CHECK(same.shape_equal);
  CHECK(same.dtype_equal);
  CHECK(same.a_elems == 6);
  CHECK(same.b_elems == 6);
  CHECK(same.a_bytes == 24);
  CHECK(same.b_bytes == 24);

  const TensorMatch& rank_mm = matches[1];
  CHECK_FALSE(rank_mm.shape_equal);  // differing RANK (2 vs 3 dims)
  CHECK(rank_mm.dtype_equal);
  CHECK(rank_mm.a_elems == 6);
  CHECK(rank_mm.b_elems == 24);
  CHECK(rank_mm.a_bytes == 24);
  CHECK(rank_mm.b_bytes == 96);

  const TensorMatch& dims_mm = matches[2];
  CHECK_FALSE(dims_mm.shape_equal);  // same rank, differing DIM
  CHECK(dims_mm.dtype_equal);
  CHECK(dims_mm.a_elems == 6);
  CHECK(dims_mm.b_elems == 10);
  CHECK(dims_mm.a_bytes == 24);
  CHECK(dims_mm.b_bytes == 40);

  const TensorMatch& dtype_mm = matches[3];
  CHECK(dtype_mm.shape_equal);
  CHECK_FALSE(dtype_mm.dtype_equal);  // F32 vs F16
  CHECK(dtype_mm.a_elems == 4);
  CHECK(dtype_mm.b_elems == 4);
  CHECK(dtype_mm.a_bytes == 16);
  CHECK(dtype_mm.b_bytes == 8);

  // Determinism: re-running on the same models yields the same order and the
  // same per-pair results (a stale UI table must never see a match reshuffle).
  std::vector<TensorMatch> matches2 = match_tensors(a, b);
  REQUIRE(matches2.size() == matches.size());
  for (size_t i = 0; i < matches.size(); ++i) {
    CHECK(matches2[i].a == matches[i].a);
    CHECK(matches2[i].b == matches[i].b);
    CHECK(matches2[i].shape_equal == matches[i].shape_equal);
    CHECK(matches2[i].dtype_equal == matches[i].dtype_equal);
    CHECK(matches2[i].a_elems == matches[i].a_elems);
    CHECK(matches2[i].b_elems == matches[i].b_elems);
    CHECK(matches2[i].a_bytes == matches[i].a_bytes);
    CHECK(matches2[i].b_bytes == matches[i].b_bytes);
  }
}
