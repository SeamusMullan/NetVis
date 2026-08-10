// engine/SpatialIndex.h — uniform-grid spatial index over a laid-out graph (#99).
//
// DECISION (v0.9.3, Pillar 2): the canvas culls by scanning EVERY box and EVERY
// edge each frame and AABB-testing it against the viewport. That is O(total), not
// O(visible), so zooming into a corner of a 100k-node graph costs exactly as much
// as showing the whole thing — the opposite of what culling is for. The #97
// harness measures this as `visible_scan`: 0.58 ms at 100k, paid several times
// per frame (hit-test, edge draw, search pulse), and growing linearly with model
// size rather than with what is on screen.
//
// A uniform grid is the right structure here, not a quadtree or BVH:
//   * Layout output is roughly uniform in density — a layered Sugiyama drawing
//     has no deep empty regions for a quadtree's subdivision to pay off on.
//   * It builds in one O(N) pass with no recursion, no rebalancing, no
//     allocation per node — it is rebuilt whenever the layout changes, and
//     layout changes are already the expensive operation.
//   * Query is a rect-to-cell-range walk: no tree descent, no per-query
//     allocation, trivially deterministic.
// A tree would be more code, more allocation, and better only for the input
// distribution this problem does not have.
//
// SCOPE: this indexes a LayoutResult, which is pure engine data (boxes + edge
// curves). It lives in netvis_core so the #97 harness can measure it headlessly —
// the same reason visible_scan lives in Bench.h rather than in the view.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/Layout.h"

namespace netvis {

// A world-space axis-aligned query rectangle. Members are named to match the
// canvas's existing vw_min/vw_max convention so the call site reads the same.
struct WorldRect {
  float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  bool valid() const { return x1 >= x0 && y1 >= y0; }
};

class SpatialIndex {
 public:
  // Rebuild for `layout`. O(boxes + edges), one pass, no per-item allocation
  // beyond the two index vectors. Safe to call on an empty layout (leaves the
  // index empty and !valid()).
  //
  // The caller must rebuild whenever the LayoutResult changes. The index stores
  // INDICES into layout.boxes / layout.edges, never pointers, so a reallocated
  // LayoutResult cannot leave it holding dangling memory — but a stale index
  // would return indices into the wrong layout, so the view must key its rebuild
  // on the same signal it uses to re-layout.
  void build(const LayoutResult& layout);

  void clear();
  bool valid() const { return cells_x_ > 0 && cells_y_ > 0; }

  // Indices of boxes / edges whose AABB may overlap `rect`.
  //
  // CANDIDATES, not exact hits: a returned item overlaps the queried CELLS, so
  // the caller must still do its own precise test. That is deliberate — the
  // caller already does an exact AABB test today, and duplicating it here would
  // be two places to get it wrong.
  //
  // `out` is cleared and refilled. Pass the SAME vector every frame: the whole
  // point is to stop allocating per frame, and a fresh vector per call would
  // reintroduce exactly the cost this removes. Results are in ascending index
  // order and contain no duplicates, so draw order stays deterministic — a
  // grid can otherwise return an item once per cell it spans.
  void query_boxes(const WorldRect& rect, std::vector<uint32_t>& out) const;
  void query_edges(const WorldRect& rect, std::vector<uint32_t>& out) const;

  // Sizes the index was built for. A caller that finds these disagree with the
  // live LayoutResult is holding a stale index and must rebuild rather than
  // index out of range.
  uint32_t indexed_boxes() const { return indexed_boxes_; }
  uint32_t indexed_edges() const { return indexed_edges_; }

  // Grid shape, for tests and for the harness to report. Not load-bearing.
  uint32_t cells_x() const { return cells_x_; }
  uint32_t cells_y() const { return cells_y_; }

 private:
  // CSR-style buckets: `starts` has cell_count+1 entries, `items` is the
  // concatenated per-cell index lists. Two flat vectors instead of a
  // vector-of-vectors — one allocation each, cache-friendly to walk, and no
  // per-cell heap block for a grid that is mostly small buckets.
  struct Buckets {
    std::vector<uint32_t> starts;
    std::vector<uint32_t> items;
  };

  Buckets boxes_;
  Buckets edges_;

  float min_x_ = 0, min_y_ = 0;
  float inv_cell_w_ = 0, inv_cell_h_ = 0;
  uint32_t cells_x_ = 0, cells_y_ = 0;
  uint32_t indexed_boxes_ = 0, indexed_edges_ = 0;
};

}  // namespace netvis
