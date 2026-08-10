// tests/test_spatial_index.cpp — uniform-grid culling index (#99).
//
// The load-bearing property is EXHAUSTIVENESS: a culling structure that misses
// a visible node produces a rendering bug that reads as data loss, and it does
// so only at some zoom levels, on some models. So the headline test is a
// differential one — the index must return a SUPERSET of what a brute-force
// linear AABB scan finds, over a pile of random rects with a fixed seed. The
// remaining cases pin the contract the header promises (ascending, deduped,
// candidates-not-hits) and the degenerate geometry that hostile or half-broken
// layouts produce.
//
// LayoutResult is plain data, so these build it directly rather than running
// compute_layout: the point is to control the geometry precisely (coincident
// boxes, one world-sized box, a zero-extent layout), not to re-test the layout
// engine.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "engine/Layout.h"
#include "engine/SpatialIndex.h"

using namespace netvis;

namespace {

// Deterministic 32-bit LCG (Numerical Recipes constants). Hand-rolled rather
// than <random> so the sequence is identical on every platform and compiler —
// a differential test that samples different rects per toolchain is not a test.
struct Rng {
  uint32_t s;
  explicit Rng(uint32_t seed) : s(seed) {}
  uint32_t next() {
    s = s * 1664525u + 1013904223u;
    return s;
  }
  // Uniform-ish float in [lo, hi).
  float uniform(float lo, float hi) {
    const float t = static_cast<float>(next() >> 8) / 16777216.0f;
    return lo + t * (hi - lo);
  }
};

NodeBox make_box(float x, float y, float w, float h) {
  NodeBox b;
  b.pos = {x, y};
  b.size = {w, h};
  return b;
}

EdgeCurve make_edge(float x0, float y0, float x1, float y1) {
  EdgeCurve e;
  e.p0 = {x0, y0};
  e.p1 = {x0, (y0 + y1) * 0.5f};
  e.p2 = {x1, (y0 + y1) * 0.5f};
  e.p3 = {x1, y1};
  return e;
}

// The AABB the index is expected to have filed each item under.
void box_bounds(const NodeBox& b, float& x0, float& y0, float& x1, float& y1) {
  x0 = std::min(b.pos.x, b.pos.x + b.size.x);
  y0 = std::min(b.pos.y, b.pos.y + b.size.y);
  x1 = std::max(b.pos.x, b.pos.x + b.size.x);
  y1 = std::max(b.pos.y, b.pos.y + b.size.y);
}

void edge_bounds(const EdgeCurve& e, float& x0, float& y0, float& x1,
                 float& y1) {
  x0 = std::min(std::min(e.p0.x, e.p1.x), std::min(e.p2.x, e.p3.x));
  y0 = std::min(std::min(e.p0.y, e.p1.y), std::min(e.p2.y, e.p3.y));
  x1 = std::max(std::max(e.p0.x, e.p1.x), std::max(e.p2.x, e.p3.x));
  y1 = std::max(std::max(e.p0.y, e.p1.y), std::max(e.p2.y, e.p3.y));
}

bool overlaps(float ax0, float ay0, float ax1, float ay1, const WorldRect& r) {
  return ax0 <= r.x1 && ax1 >= r.x0 && ay0 <= r.y1 && ay1 >= r.y0;
}

// The reference implementation: the same inclusive AABB test GraphCanvas and
// Bench's visible_scan proxy do today, over every item.
std::vector<uint32_t> brute_force_boxes(const LayoutResult& l,
                                        const WorldRect& r) {
  std::vector<uint32_t> out;
  for (uint32_t i = 0; i < l.boxes.size(); ++i) {
    float x0, y0, x1, y1;
    box_bounds(l.boxes[i], x0, y0, x1, y1);
    if (overlaps(x0, y0, x1, y1, r)) out.push_back(i);
  }
  return out;
}

std::vector<uint32_t> brute_force_edges(const LayoutResult& l,
                                        const WorldRect& r) {
  std::vector<uint32_t> out;
  for (uint32_t i = 0; i < l.edges.size(); ++i) {
    float x0, y0, x1, y1;
    edge_bounds(l.edges[i], x0, y0, x1, y1);
    if (overlaps(x0, y0, x1, y1, r)) out.push_back(i);
  }
  return out;
}

// `got` is ascending and duplicate-free (the contract), and contains every
// index in `want` (exhaustiveness). Extra entries are allowed: the header
// promises candidates, not hits.
//
// Folded into three asserts rather than one per element: this runs inside a
// 400-rect loop, and a per-element REQUIRE would spend the whole test budget in
// doctest's assert machinery. The first missing index is captured so a failure
// still names the item that was culled.
void check_superset(const std::vector<uint32_t>& got,
                    const std::vector<uint32_t>& want) {
  const bool sorted = std::is_sorted(got.begin(), got.end());
  const bool unique = std::adjacent_find(got.begin(), got.end()) == got.end();
  uint32_t missing = UINT32_MAX;
  for (uint32_t w : want) {
    if (!std::binary_search(got.begin(), got.end(), w)) {
      missing = w;
      break;
    }
  }
  REQUIRE(sorted);
  REQUIRE(unique);
  CAPTURE(missing);
  REQUIRE(missing == UINT32_MAX);
}

// A scattered layout with roughly layer-like structure: boxes on a jittered
// grid, edges linking a box to one below it. Big enough that the index really
// has to be multi-cell.
LayoutResult make_scatter(uint32_t n_boxes, uint32_t seed) {
  Rng rng(seed);
  LayoutResult l;
  l.boxes.reserve(n_boxes);
  for (uint32_t i = 0; i < n_boxes; ++i) {
    const float x = rng.uniform(-500.0f, 4500.0f);
    const float y = rng.uniform(-200.0f, 9800.0f);
    l.boxes.push_back(make_box(x, y, rng.uniform(20.0f, 160.0f),
                               rng.uniform(10.0f, 40.0f)));
  }
  for (uint32_t i = 0; i + 1 < n_boxes; i += 2) {
    const NodeBox& a = l.boxes[i];
    const NodeBox& b = l.boxes[i + 1];
    l.edges.push_back(make_edge(a.pos.x, a.pos.y + a.size.y, b.pos.x, b.pos.y));
  }
  l.bounds_min = {-500.0f, -200.0f};
  l.bounds_max = {4700.0f, 9840.0f};
  return l;
}

}  // namespace

TEST_CASE("#99 spatial index returns a superset of a brute-force scan") {
  const LayoutResult l = make_scatter(4000, 12345u);
  SpatialIndex idx;
  idx.build(l);
  REQUIRE(idx.valid());

  Rng rng(0xC0FFEEu);
  std::vector<uint32_t> got;
  for (int trial = 0; trial < 400; ++trial) {
    // Rects of wildly different scales: a few cells, a screenful, the whole
    // world, and some that fall partly or entirely outside the extent.
    const float cx = rng.uniform(-1500.0f, 5500.0f);
    const float cy = rng.uniform(-1500.0f, 11000.0f);
    const float half_w = rng.uniform(1.0f, 3000.0f);
    const float half_h = rng.uniform(1.0f, 6000.0f);
    WorldRect r{cx - half_w, cy - half_h, cx + half_w, cy + half_h};

    idx.query_boxes(r, got);
    check_superset(got, brute_force_boxes(l, r));

    idx.query_edges(r, got);
    check_superset(got, brute_force_edges(l, r));
  }
}

TEST_CASE("#99 an item spanning many cells is returned once, in order") {
  LayoutResult l = make_scatter(2000, 777u);
  // One box covering the entire world, so it is filed in every single cell.
  const uint32_t giant = static_cast<uint32_t>(l.boxes.size());
  l.boxes.push_back(make_box(-500.0f, -200.0f, 5200.0f, 10040.0f));

  SpatialIndex idx;
  idx.build(l);
  REQUIRE(idx.valid());
  // The grid must genuinely be multi-cell, else this proves nothing.
  REQUIRE(idx.cells_x() * idx.cells_y() > 1);

  std::vector<uint32_t> got;
  // A rect spanning tens of cells — the giant box is filed in every one of
  // them — but small enough that the wide-view shortcut does NOT fire, so this
  // really exercises the de-duplication and not the whole-list path.
  WorldRect r{0.0f, 0.0f, 1000.0f, 2000.0f};
  idx.query_boxes(r, got);
  REQUIRE(got.size() < l.boxes.size());
  CHECK(std::is_sorted(got.begin(), got.end()));
  CHECK((std::adjacent_find(got.begin(), got.end()) == got.end()));
  CHECK(std::count(got.begin(), got.end(), giant) == 1);
  check_superset(got, brute_force_boxes(l, r));
}

TEST_CASE("#99 a full-extent query returns every item exactly once") {
  const LayoutResult l = make_scatter(1500, 42u);
  SpatialIndex idx;
  idx.build(l);

  WorldRect all{-1e6f, -1e6f, 1e6f, 1e6f};
  std::vector<uint32_t> got;

  idx.query_boxes(all, got);
  REQUIRE(got.size() == l.boxes.size());
  for (uint32_t i = 0; i < got.size(); ++i) CHECK(got[i] == i);

  idx.query_edges(all, got);
  REQUIRE(got.size() == l.edges.size());
  for (uint32_t i = 0; i < got.size(); ++i) CHECK(got[i] == i);
}

TEST_CASE("#99 a query outside the extent returns nothing") {
  const LayoutResult l = make_scatter(500, 9u);
  SpatialIndex idx;
  idx.build(l);

  std::vector<uint32_t> got;
  const WorldRect far_right{100000.0f, 100000.0f, 200000.0f, 200000.0f};
  idx.query_boxes(far_right, got);
  CHECK(got.empty());
  idx.query_edges(far_right, got);
  CHECK(got.empty());

  const WorldRect far_left{-200000.0f, -200000.0f, -100000.0f, -100000.0f};
  idx.query_boxes(far_left, got);
  CHECK(got.empty());
  idx.query_edges(far_left, got);
  CHECK(got.empty());

  // An inverted rect selects nothing rather than wrapping around the grid.
  const WorldRect inverted{500.0f, 500.0f, -500.0f, -500.0f};
  idx.query_boxes(inverted, got);
  CHECK(got.empty());
}

TEST_CASE("#99 spatial index survives degenerate layouts") {
  std::vector<uint32_t> got;
  const WorldRect anywhere{-1e6f, -1e6f, 1e6f, 1e6f};

  SUBCASE("empty layout is not valid and answers empty") {
    const LayoutResult l;
    SpatialIndex idx;
    idx.build(l);
    CHECK_FALSE(idx.valid());
    CHECK(idx.indexed_boxes() == 0);
    CHECK(idx.indexed_edges() == 0);
    idx.query_boxes(anywhere, got);
    CHECK(got.empty());
    idx.query_edges(anywhere, got);
    CHECK(got.empty());
  }

  SUBCASE("single box gets a 1x1 grid") {
    LayoutResult l;
    l.boxes.push_back(make_box(10.0f, 20.0f, 30.0f, 40.0f));
    SpatialIndex idx;
    idx.build(l);
    REQUIRE(idx.valid());
    CHECK(idx.cells_x() == 1);
    CHECK(idx.cells_y() == 1);
    idx.query_boxes(anywhere, got);
    REQUIRE(got.size() == 1);
    CHECK(got[0] == 0);
    // Exactly on the box: a hit. Well clear of it: nothing.
    idx.query_boxes(WorldRect{15.0f, 25.0f, 16.0f, 26.0f}, got);
    CHECK(got.size() == 1);
    idx.query_boxes(WorldRect{1000.0f, 1000.0f, 2000.0f, 2000.0f}, got);
    CHECK(got.empty());
  }

  SUBCASE("all boxes coincident: zero extent, still exhaustive") {
    LayoutResult l;
    for (int i = 0; i < 64; ++i) l.boxes.push_back(make_box(5.0f, 7.0f, 0, 0));
    SpatialIndex idx;
    idx.build(l);
    REQUIRE(idx.valid());
    CHECK(idx.cells_x() == 1);
    CHECK(idx.cells_y() == 1);
    idx.query_boxes(WorldRect{0.0f, 0.0f, 10.0f, 10.0f}, got);
    CHECK(got.size() == 64);
    CHECK(std::is_sorted(got.begin(), got.end()));
    // The degenerate axis still culls exactly in world coordinates.
    idx.query_boxes(WorldRect{100.0f, 100.0f, 200.0f, 200.0f}, got);
    CHECK(got.empty());
  }

  SUBCASE("zero-height world (all boxes on one row)") {
    LayoutResult l;
    for (int i = 0; i < 200; ++i)
      l.boxes.push_back(make_box(static_cast<float>(i) * 50.0f, 0.0f, 40.0f,
                                 0.0f));
    SpatialIndex idx;
    idx.build(l);
    REQUIRE(idx.valid());
    CHECK(idx.cells_y() == 1);
    CHECK(idx.cells_x() > 1);  // the budget goes to the axis that has extent
    const WorldRect r{-10.0f, -1.0f, 500.0f, 1.0f};
    idx.query_boxes(r, got);
    check_superset(got, brute_force_boxes(l, r));
  }

  SUBCASE("non-finite coordinates do not hang or crash") {
    LayoutResult l;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    l.boxes.push_back(make_box(0.0f, 0.0f, 100.0f, 100.0f));
    l.boxes.push_back(make_box(nan, nan, 10.0f, 10.0f));
    l.boxes.push_back(make_box(inf, -inf, 10.0f, 10.0f));
    l.boxes.push_back(make_box(50.0f, 50.0f, nan, nan));
    l.edges.push_back(make_edge(nan, 0.0f, inf, 10.0f));
    SpatialIndex idx;
    idx.build(l);
    REQUIRE(idx.valid());
    idx.query_boxes(anywhere, got);
    CHECK(got.size() == 4);  // every box is reachable from some cell
    CHECK(std::is_sorted(got.begin(), got.end()));
    // A NaN query rect selects nothing rather than reading a garbage range.
    idx.query_boxes(WorldRect{nan, nan, nan, nan}, got);
    CHECK(got.empty());
  }

  SUBCASE("world-sized boxes coarsen the grid instead of exploding it") {
    // Every item covers the whole world, so a resolution derived from the item
    // COUNT alone would file each of them in every cell — 4000 x ~1000 entries.
    // The insertion budget has to notice and collapse the grid.
    LayoutResult l;
    for (int i = 0; i < 4000; ++i)
      l.boxes.push_back(make_box(-1000.0f, -1000.0f, 2000.0f, 2000.0f));
    l.boxes.push_back(make_box(-1000.0f, -1000.0f, 1.0f, 1.0f));
    SpatialIndex idx;
    idx.build(l);
    REQUIRE(idx.valid());
    CHECK(idx.cells_x() * idx.cells_y() <= 4);
    const WorldRect r{0.0f, 0.0f, 10.0f, 10.0f};
    idx.query_boxes(r, got);
    check_superset(got, brute_force_boxes(l, r));
    CHECK(got.size() >= 4000);
  }

  SUBCASE("negative box sizes are normalised, not dropped") {
    LayoutResult l;
    l.boxes.push_back(make_box(100.0f, 100.0f, -60.0f, -60.0f));
    l.boxes.push_back(make_box(500.0f, 500.0f, 10.0f, 10.0f));
    SpatialIndex idx;
    idx.build(l);
    const WorldRect r{50.0f, 50.0f, 60.0f, 60.0f};
    idx.query_boxes(r, got);
    check_superset(got, brute_force_boxes(l, r));
    CHECK_FALSE(got.empty());
  }
}

TEST_CASE("#99 indexed sizes match the layout the index was built for") {
  const LayoutResult a = make_scatter(300, 3u);
  SpatialIndex idx;
  idx.build(a);
  CHECK(idx.indexed_boxes() == a.boxes.size());
  CHECK(idx.indexed_edges() == a.edges.size());

  // Rebuilding for a different layout re-reports, and never leaves the previous
  // layout's sizes behind — a stale pair is the caller's only defence against
  // indexing out of range.
  const LayoutResult b = make_scatter(37, 4u);
  idx.build(b);
  CHECK(idx.indexed_boxes() == b.boxes.size());
  CHECK(idx.indexed_edges() == b.edges.size());

  idx.clear();
  CHECK_FALSE(idx.valid());
  CHECK(idx.indexed_boxes() == 0);
  CHECK(idx.indexed_edges() == 0);
}

TEST_CASE("#99 reusing the caller's out vector does not leak results") {
  const LayoutResult l = make_scatter(800, 5150u);
  SpatialIndex idx;
  idx.build(l);

  std::vector<uint32_t> out;
  const WorldRect all{-1e6f, -1e6f, 1e6f, 1e6f};
  idx.query_boxes(all, out);
  const size_t everything = out.size();
  REQUIRE(everything == l.boxes.size());

  // A tiny rect immediately after a full-world one must not inherit any of it.
  const WorldRect tiny{1000.0f, 2000.0f, 1010.0f, 2010.0f};
  idx.query_boxes(tiny, out);
  CHECK(out.size() < everything);
  check_superset(out, brute_force_boxes(l, tiny));

  // A miss clears the vector rather than leaving the previous answer in place.
  idx.query_boxes(WorldRect{9e5f, 9e5f, 9.1e5f, 9.1e5f}, out);
  CHECK(out.empty());

  // Boxes then edges through the same vector: the second call must not carry
  // box indices, which would be read as edge indices by the caller.
  idx.query_boxes(all, out);
  idx.query_edges(all, out);
  CHECK(out.size() == l.edges.size());

  // And an invalid index empties it too.
  idx.clear();
  out.assign(4, 7u);
  idx.query_boxes(all, out);
  CHECK(out.empty());
}
