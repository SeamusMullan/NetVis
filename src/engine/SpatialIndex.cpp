// engine/SpatialIndex.cpp — uniform-grid build + query (#99).
//
// The header argues WHY a grid; this file is the HOW, and every number in it is
// derived rather than picked. Three things carry the design:
//
//   1. RESOLUTION IS COMPUTED. Cell count comes from the item count (target a
//      few items per cell) and the split between axes comes from the extent's
//      aspect ratio, so a wide layered drawing gets a wide grid instead of a
//      square one full of empty cells. Both the per-axis count and the product
//      are clamped, so no layout — however pathological — can talk us into
//      allocating an enormous grid.
//   2. THE INSERTION COUNT IS BUDGETED. An item is filed in every cell its AABB
//      touches, so ONE box the size of the world costs cell_count entries. A
//      count-derived resolution knows nothing about item SIZE, so we measure the
//      total insertion count before committing and coarsen the grid until it
//      fits a budget. That is what makes rule 1 safe on hostile geometry.
//   3. QUERIES DE-DUPLICATE THROUGH A BITMASK, NOT A SET. An item spanning four
//      cells is found four times; the contract promises it once, ascending. A
//      per-thread mark buffer (grown once, reused forever, left all-zero) turns
//      "seen?" into one shift and one OR and makes ascending order fall out of
//      scanning the words low-to-high. See mark_words() for the invariant.
#include "engine/SpatialIndex.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <numeric>

namespace netvis {
namespace {

// --- tuning ----------------------------------------------------------------
// Target items per cell. Too high and each query drags in neighbours it must
// then reject; too low and the grid is mostly empty cells and the `starts`
// array dwarfs the data it indexes. 4 keeps a query's candidate set within a
// small constant factor of the true visible set while holding the grid at one
// uint32 per 4 items.
constexpr uint64_t kItemsPerCell = 4;

// Hard caps. kMaxCellsPerAxis stops an extreme aspect ratio (a 1-pixel-tall
// world) from asking for a million columns; kMaxCells caps the `starts`
// allocation at 256K entries = 1 MB regardless of model size, which is already
// generous next to the 136 MB peak RSS of the 100k rung.
constexpr uint32_t kMaxCellsPerAxis = 1024;
constexpr uint64_t kMaxCells = 1u << 18;

// Items are addressed by uint32 index and insertion offsets are uint32, so
// refuse to index a layout that cannot be addressed that way. A caller holding
// an invalid index falls back to its linear scan — correct, just slower — which
// is a far better failure than a truncated index that silently loses nodes.
constexpr size_t kMaxItems = 1u << 26;

// Insertion budget: how many (cell, item) entries the CSR may hold. Boxes and
// short edges normally straddle one or two cells each, so ~8 per item is slack,
// not a limit anyone reaches; the flat 8M ceiling (32 MB) bounds the truly
// degenerate case where every item covers the world.
constexpr uint64_t kInsertionsPerItem = 8;
constexpr uint64_t kMaxInsertions = 8u << 20;

// --- geometry --------------------------------------------------------------

// A normalised world AABB (x0 <= x1, y0 <= y1).
struct Aabb {
  float x0, y0, x1, y1;
};

// Fold `v` into a running min/max, ignoring non-finite values so one NaN
// coordinate cannot swallow the whole extent (min/max against NaN is
// order-dependent, and an infinite extent would collapse every cell to zero
// width). Items with unusable coordinates still get indexed — see cell_of.
void widen(float v, float& lo, float& hi) {
  if (!std::isfinite(v)) return;
  if (v < lo) lo = v;
  if (v > hi) hi = v;
}

// NodeBox stores top-left + size. Nothing guarantees a positive size (a plugin
// or a degenerate label measurement can produce zero or negative), so normalise
// rather than assume. std::min/std::max with the finite value FIRST returns the
// finite value when the other is NaN, which quietly repairs a half-NaN box.
Aabb item_aabb(const NodeBox& b) {
  const float ax = b.pos.x;
  const float ay = b.pos.y;
  const float bx = b.pos.x + b.size.x;
  const float by = b.pos.y + b.size.y;
  return {std::min(ax, bx), std::min(ay, by), std::max(ax, bx),
          std::max(ay, by)};
}

// A cubic Bezier lies entirely within the convex hull of its four control
// points: every point on the curve is a Bernstein-weighted combination of
// p0..p3, and those weights are non-negative and sum to 1. So the AABB of the
// control points CONTAINS the AABB of the curve. That makes the control-point
// hull a valid conservative bound — never misses a hit, occasionally offers a
// candidate whose curve does not really reach the rect — and it costs four
// point reads instead of sampling or solving the derivative for extrema.
Aabb item_aabb(const EdgeCurve& e) {
  const float lox = std::min(std::min(e.p0.x, e.p1.x), std::min(e.p2.x, e.p3.x));
  const float loy = std::min(std::min(e.p0.y, e.p1.y), std::min(e.p2.y, e.p3.y));
  const float hix = std::max(std::max(e.p0.x, e.p1.x), std::max(e.p2.x, e.p3.x));
  const float hiy = std::max(std::max(e.p0.y, e.p1.y), std::max(e.p2.y, e.p3.y));
  return {lox, loy, hix, hiy};
}

// --- the grid --------------------------------------------------------------

// World-to-cell mapping. `inv` is cells-per-world-unit; it is exactly 0 on a
// degenerate axis (zero or unusable extent), which collapses that axis to the
// single cell 0 without ever dividing by anything.
struct Grid {
  float min_x = 0, min_y = 0;
  float inv_w = 0, inv_h = 0;
  uint32_t cx = 1, cy = 1;

  // Cells-per-unit from an axis extent. A subnormal extent can push the ratio
  // past float range; treat that as degenerate (one cell) rather than shipping
  // an infinite scale that maps everything to the two end cells.
  static float scale(uint32_t cells, double extent) {
    if (!(extent > 0.0)) return 0.0f;
    const double s = static_cast<double>(cells) / extent;
    if (!std::isfinite(s) || s > 3.0e38) return 0.0f;
    return static_cast<float>(s);
  }

  void set_cells(uint32_t nx, uint32_t ny, double w, double h) {
    cx = nx;
    cy = ny;
    inv_w = scale(nx, w);
    inv_h = scale(ny, h);
    // A degenerate scale means one usable column/row; keep the declared shape
    // honest so query and build agree on how many cells exist.
    if (inv_w == 0.0f) cx = 1;
    if (inv_h == 0.0f) cy = 1;
  }

  // Clamping world->cell map. The comparison order is chosen so that NaN — for
  // which every comparison is false — falls into cell 0, and so that +inf is
  // caught by the upper clamp before any cast: converting a float outside the
  // uint32 range is undefined, and this is the one place hostile coordinates
  // reach a cast.
  static uint32_t cell_of(float v, float origin, float inv, uint32_t cells) {
    if (cells <= 1) return 0;
    const float t = (v - origin) * inv;
    if (!(t > 0.0f)) return 0;
    const float last = static_cast<float>(cells - 1);
    if (t >= last) return cells - 1;
    return static_cast<uint32_t>(t);
  }

  uint32_t cell_x(float v) const { return cell_of(v, min_x, inv_w, cx); }
  uint32_t cell_y(float v) const { return cell_of(v, min_y, inv_h, cy); }

  // Inclusive cell span of an AABB. cell_of is monotone for finite input, but
  // the c1 < c0 repair keeps every loop below bounded even if a coordinate pair
  // arrives inverted through some NaN path.
  void span(const Aabb& a, uint32_t& x0, uint32_t& x1, uint32_t& y0,
            uint32_t& y1) const {
    x0 = cell_x(a.x0);
    x1 = cell_x(a.x1);
    y0 = cell_y(a.y0);
    y1 = cell_y(a.y1);
    if (x1 < x0) x1 = x0;
    if (y1 < y0) y1 = y0;
  }
};

// Clamp a real-valued axis count into [1, kMaxCellsPerAxis]. Written as a
// negated comparison so NaN (from an infinite or zero-division aspect ratio)
// lands on 1 rather than on the cap.
uint32_t clamp_axis(double v) {
  if (!(v >= 1.0)) return 1;
  if (v >= static_cast<double>(kMaxCellsPerAxis)) return kMaxCellsPerAxis;
  return static_cast<uint32_t>(v);
}

// Split `target` cells between the axes in proportion to the extent's aspect
// ratio: for a w x h world we want cx/cy ~ w/h and cx*cy ~ target, giving
// cx = sqrt(target * w/h). A degenerate axis contributes nothing to divide, so
// the whole budget goes to the other one; both degenerate means one cell.
void choose_cells(uint64_t target, double w, double h, uint32_t& cx,
                  uint32_t& cy) {
  const bool wide = w > 0.0;
  const bool tall = h > 0.0;
  const double t = static_cast<double>(target);
  if (wide && tall) {
    cx = clamp_axis(std::sqrt(t * (w / h)));
    cy = clamp_axis(t / static_cast<double>(cx));
  } else if (wide) {
    cx = clamp_axis(t);
    cy = 1;
  } else if (tall) {
    cx = 1;
    cy = clamp_axis(t);
  } else {
    cx = 1;
    cy = 1;
  }
  // Both axes are individually capped at kMaxCellsPerAxis, so their product can
  // still reach 1M; trim the second axis to hold the total under kMaxCells.
  // cx <= 1024 and kMaxCells >= 256K, so the quotient is never 0.
  const uint64_t max_y = kMaxCells / cx;
  if (static_cast<uint64_t>(cy) > max_y) cy = static_cast<uint32_t>(max_y);
  if (cy == 0) cy = 1;
}

// Total (cell, item) entries this grid would file, stopping early once the
// budget is blown — the point is to reject a resolution, not to measure how
// badly it loses, and the pathological case is exactly the one we must not walk
// in full. Multiplying the spans is O(1) per item; nothing iterates cells here.
template <typename T>
uint64_t count_insertions(const std::vector<T>& src, const Grid& g,
                          uint64_t budget) {
  uint64_t total = 0;
  for (const T& it : src) {
    uint32_t x0, x1, y0, y1;
    g.span(item_aabb(it), x0, x1, y0, y1);
    total += static_cast<uint64_t>(x1 - x0 + 1) *
             static_cast<uint64_t>(y1 - y0 + 1);
    if (total > budget) break;
  }
  return total;
}

// Two-pass counting fill of one CSR. No per-item allocation: two vectors, sized
// once, and `starts` doubles as the pass-2 write cursor.
template <typename T>
void fill_csr(const std::vector<T>& src, const Grid& g,
              std::vector<uint32_t>& starts, std::vector<uint32_t>& items) {
  const size_t cells = static_cast<size_t>(g.cx) * g.cy;
  starts.assign(cells + 1, 0);
  items.clear();
  if (src.empty()) return;

  // Pass 1 — per-cell counts, parked in starts[cell].
  for (const T& it : src) {
    uint32_t x0, x1, y0, y1;
    g.span(item_aabb(it), x0, x1, y0, y1);
    for (uint32_t y = y0; y <= y1; ++y) {
      const size_t row = static_cast<size_t>(y) * g.cx;
      for (uint32_t x = x0; x <= x1; ++x) ++starts[row + x];
    }
  }

  // Exclusive prefix sum: starts[c] becomes cell c's begin and starts[cells]
  // the grand total. The count budget already guaranteed this fits uint32.
  uint32_t run = 0;
  for (size_t c = 0; c <= cells; ++c) {
    const uint32_t n = starts[c];
    starts[c] = run;
    run += n;
  }

  // Pass 2 — place, using starts[] itself as the per-cell write cursor. Items
  // are visited in ascending index order, so each cell's slice comes out sorted
  // ascending, which is what makes the query's word scan emit ascending too.
  items.assign(run, 0);
  const uint32_t n_items = static_cast<uint32_t>(src.size());
  for (uint32_t i = 0; i < n_items; ++i) {
    uint32_t x0, x1, y0, y1;
    g.span(item_aabb(src[i]), x0, x1, y0, y1);
    for (uint32_t y = y0; y <= y1; ++y) {
      const size_t row = static_cast<size_t>(y) * g.cx;
      for (uint32_t x = x0; x <= x1; ++x) items[starts[row + x]++] = i;
    }
  }

  // Pass 2 left starts[c] holding cell c's END, i.e. the array is shifted one
  // slot left. Shift it back; starts[0] is 0 by construction.
  for (size_t c = cells; c > 0; --c) starts[c] = starts[c - 1];
  starts[0] = 0;
}

// --- query -----------------------------------------------------------------

// Inclusive cell range covered by [lo, hi] on one axis. Returns false when the
// span misses the grid entirely — the clamping cell_of alone would map an
// off-world rect onto the edge cells and hand back their contents, so the
// outside-the-extent rejection has to happen here, in world units.
bool axis_range(float lo, float hi, float origin, float inv, uint32_t cells,
                uint32_t& c0, uint32_t& c1) {
  if (!(hi >= lo)) return false;      // inverted, or NaN anywhere in the pair
  if (!(hi >= origin)) return false;  // entirely below the first cell
  if (inv > 0.0f) {
    const float t0 = (lo - origin) * inv;
    const float t1 = (hi - origin) * inv;
    // The grid spans t in [0, cells]; t0 == cells is the far edge, where items
    // clamped into the last cell live, so only a strictly greater t0 misses.
    if (t0 > static_cast<float>(cells)) return false;
    const float last = static_cast<float>(cells - 1);
    c0 = (t0 <= 0.0f) ? 0u
                      : (t0 >= last ? cells - 1 : static_cast<uint32_t>(t0));
    c1 = (t1 >= last) ? cells - 1 : static_cast<uint32_t>(t1);
  } else {
    // Degenerate axis: the whole layout sits at `origin`, so the rect either
    // straddles that line or it misses everything. This stays exact — a
    // zero-width extent is the one case where the grid cannot cull but the
    // world coordinates still can.
    if (lo > origin) return false;
    c0 = 0;
    c1 = 0;
  }
  if (c1 < c0) c1 = c0;
  return true;
}

// Per-thread de-duplication scratch, one bit per item.
//
// INVARIANT: every word is zero on entry and on exit — the emit loop clears
// each word as it consumes it, so the buffer never needs a bulk memset and a
// query costs O(candidates + words actually touched), not O(items). That also
// makes it safe to share between instances and between the box and edge
// queries: it carries no state across calls, only capacity.
//
// thread_local rather than a member because the header's layout is fixed, and
// per-thread is strictly better than per-instance anyway: two threads querying
// two indices never collide. One thread must not run two queries at once on any
// index (there is no reentrancy here to make that possible) — the canvas is
// main-thread only, so this costs nothing.
std::vector<uint64_t>& mark_words(size_t words) {
  static thread_local std::vector<uint64_t> buf;
  if (buf.size() < words) buf.resize(words, 0);
  return buf;
}

void query_grid(const Grid& g, const std::vector<uint32_t>& starts,
                const std::vector<uint32_t>& items, uint32_t n,
                const WorldRect& rect, std::vector<uint32_t>& out) {
  out.clear();
  if (n == 0 || starts.empty()) return;

  uint32_t x0, x1, y0, y1;
  if (!axis_range(rect.x0, rect.x1, g.min_x, g.inv_w, g.cx, x0, x1)) return;
  if (!axis_range(rect.y0, rect.y1, g.min_y, g.inv_h, g.cy, y0, y1)) return;

  // Wide-view shortcut. Measured on a 100k-box synthetic layout: walking the
  // grid costs ~3.3 ns per candidate (CSR read, mark, emit) while the linear
  // AABB scan it replaces costs ~1 ns per item, so the walk stops paying once
  // the query covers more than roughly a quarter of the grid. Past that point
  // hand back the whole list instead: still a valid candidate set (the header
  // promises candidates, not hits), still ascending and duplicate-free, and it
  // pins the zoomed-OUT case at parity with today's scan rather than losing to
  // it by ~2x. At full coverage it is not even an approximation — every item is
  // filed in at least one cell, so the whole list IS the answer.
  const uint64_t covered = static_cast<uint64_t>(x1 - x0 + 1) *
                           static_cast<uint64_t>(y1 - y0 + 1);
  if (covered * 4 >= static_cast<uint64_t>(g.cx) * g.cy) {
    out.resize(n);
    std::iota(out.begin(), out.end(), 0u);
    return;
  }

  std::vector<uint64_t>& mark = mark_words((static_cast<size_t>(n) + 63) / 64);
  size_t lo_word = mark.size();
  size_t hi_word = 0;
  size_t found = 0;

  // Cells within a row are consecutive in the CSR, so the whole x-range of a
  // row is ONE contiguous slice [starts[row+x0], starts[row+x1+1]). That drops
  // all per-cell bookkeeping from the hot loop: one slice per visible row.
  for (uint32_t y = y0; y <= y1; ++y) {
    const size_t row = static_cast<size_t>(y) * g.cx;
    const uint32_t k0 = starts[row + x0];
    const uint32_t k1 = starts[row + x1 + 1];
    for (uint32_t k = k0; k < k1; ++k) {
      const uint32_t idx = items[k];
      const size_t w = idx >> 6;
      const uint64_t bit = uint64_t{1} << (idx & 63u);
      // The word is loaded for the OR regardless, so counting the 0->1
      // transitions here is nearly free and lets the emit loop reserve exactly
      // once instead of growing while it writes.
      found += (mark[w] & bit) == 0 ? 1u : 0u;
      mark[w] |= bit;
      if (w < lo_word) lo_word = w;
      if (w > hi_word) hi_word = w;
    }
  }
  if (lo_word > hi_word) return;  // nothing marked
  out.reserve(found);

  // Emit low word to high word, low bit to high bit: ascending index order and,
  // because a bit can only be set once, no duplicates. Clearing here is what
  // restores the all-zero invariant.
  for (size_t w = lo_word; w <= hi_word; ++w) {
    uint64_t bits = mark[w];
    if (bits == 0) continue;
    mark[w] = 0;
    const uint32_t base = static_cast<uint32_t>(w) << 6;
    while (bits != 0) {
      out.push_back(base + static_cast<uint32_t>(std::countr_zero(bits)));
      bits &= bits - 1;  // clear the lowest set bit
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------
void SpatialIndex::build(const LayoutResult& layout) {
  clear();

  const size_t nb = layout.boxes.size();
  const size_t ne = layout.edges.size();
  if (nb == 0 && ne == 0) return;             // empty layout => stays !valid()
  if (nb > kMaxItems || ne > kMaxItems) return;  // unaddressable; see kMaxItems

  // World extent from what we actually index, NOT from layout.bounds_*: those
  // bound the node boxes, while edge control points routinely swing outside
  // them, and a stale or degenerate bounds pair would silently misplace every
  // item. One pass over both arrays is cheap next to the two fill passes.
  float min_x = HUGE_VALF, min_y = HUGE_VALF;
  float max_x = -HUGE_VALF, max_y = -HUGE_VALF;
  for (const NodeBox& b : layout.boxes) {
    const Aabb a = item_aabb(b);
    widen(a.x0, min_x, max_x);
    widen(a.x1, min_x, max_x);
    widen(a.y0, min_y, max_y);
    widen(a.y1, min_y, max_y);
  }
  for (const EdgeCurve& e : layout.edges) {
    const Aabb a = item_aabb(e);
    widen(a.x0, min_x, max_x);
    widen(a.x1, min_x, max_x);
    widen(a.y0, min_y, max_y);
    widen(a.y1, min_y, max_y);
  }

  // An axis on which nothing finite was seen (every coordinate NaN/inf) reads
  // as a zero extent at the origin: one cell holding everything on that axis.
  // Degenerate, but a valid index — every query overlapping that line gets the
  // items as candidates and the caller's exact test does the rest, which beats
  // reporting an empty scene. Handled per axis so a layout that is sane in x
  // and broken in y keeps its x culling.
  if (!(min_x <= max_x)) min_x = max_x = 0.0f;
  if (!(min_y <= max_y)) min_y = max_y = 0.0f;
  // Widths in double: max - min can overflow float to +inf for coordinates near
  // FLT_MAX, and an infinite width would zero every cell scale.
  const double world_w = static_cast<double>(max_x) - static_cast<double>(min_x);
  const double world_h = static_cast<double>(max_y) - static_cast<double>(min_y);

  // Resolution: ceil(items / kItemsPerCell) cells, capped, split by aspect.
  // The empty-layout return above guarantees items >= 1, so the ceiling
  // division cannot produce a zero target.
  const uint64_t items = static_cast<uint64_t>(nb) + static_cast<uint64_t>(ne);
  uint64_t target = (items + kItemsPerCell - 1) / kItemsPerCell;
  if (target > kMaxCells) target = kMaxCells;

  Grid g;
  {
    uint32_t nx = 1, ny = 1;
    choose_cells(target, world_w, world_h, nx, ny);
    g.min_x = min_x;
    g.min_y = min_y;
    g.set_cells(nx, ny, world_w, world_h);
  }

  // Coarsen until the CSR fits its budget. The item count says how many things
  // there are, never how BIG they are, so a layout of world-sized boxes would
  // otherwise file every item in every cell. Halving both axes quarters the
  // insertion count and terminates at 1x1, where the total is exactly `items`.
  const uint64_t budget =
      std::min(kInsertionsPerItem * items + 64, kMaxInsertions);
  while (g.cx > 1 || g.cy > 1) {
    uint64_t total = count_insertions(layout.boxes, g, budget);
    if (total <= budget) total += count_insertions(layout.edges, g, budget);
    if (total <= budget) break;
    const uint32_t nx = g.cx > 1 ? g.cx / 2 : 1;
    const uint32_t ny = g.cy > 1 ? g.cy / 2 : 1;
    g.set_cells(nx, ny, world_w, world_h);
  }

  min_x_ = g.min_x;
  min_y_ = g.min_y;
  inv_cell_w_ = g.inv_w;
  inv_cell_h_ = g.inv_h;
  cells_x_ = g.cx;
  cells_y_ = g.cy;
  indexed_boxes_ = static_cast<uint32_t>(nb);
  indexed_edges_ = static_cast<uint32_t>(ne);

  fill_csr(layout.boxes, g, boxes_.starts, boxes_.items);
  fill_csr(layout.edges, g, edges_.starts, edges_.items);
}

void SpatialIndex::clear() {
  // clear(), not shrink: the index is rebuilt on every re-layout and keeping the
  // capacity means a rebuild reuses the same blocks instead of re-allocating a
  // grid the same size as the one just dropped.
  boxes_.starts.clear();
  boxes_.items.clear();
  edges_.starts.clear();
  edges_.items.clear();
  min_x_ = min_y_ = 0.0f;
  inv_cell_w_ = inv_cell_h_ = 0.0f;
  cells_x_ = cells_y_ = 0;
  indexed_boxes_ = indexed_edges_ = 0;
}

// ---------------------------------------------------------------------------
// queries
// ---------------------------------------------------------------------------
namespace {

Grid grid_of(float min_x, float min_y, float inv_w, float inv_h, uint32_t cx,
             uint32_t cy) {
  Grid g;
  g.min_x = min_x;
  g.min_y = min_y;
  g.inv_w = inv_w;
  g.inv_h = inv_h;
  g.cx = cx;
  g.cy = cy;
  return g;
}

}  // namespace

void SpatialIndex::query_boxes(const WorldRect& rect,
                               std::vector<uint32_t>& out) const {
  out.clear();
  if (!valid()) return;
  query_grid(grid_of(min_x_, min_y_, inv_cell_w_, inv_cell_h_, cells_x_,
                     cells_y_),
             boxes_.starts, boxes_.items, indexed_boxes_, rect, out);
}

void SpatialIndex::query_edges(const WorldRect& rect,
                               std::vector<uint32_t>& out) const {
  out.clear();
  if (!valid()) return;
  query_grid(grid_of(min_x_, min_y_, inv_cell_w_, inv_cell_h_, cells_x_,
                     cells_y_),
             edges_.starts, edges_.items, indexed_edges_, rect, out);
}

}  // namespace netvis
