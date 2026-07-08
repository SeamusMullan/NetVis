// engine/Layout.h — layered (Sugiyama-style) graph layout output types.
//
// DECISION (spec §7.2): from-scratch layered layout, no OGDF/Graphviz. The
// engine produces flat position arrays indexed by the *display* node id (which,
// after collapse, may be a super-node — see CollapseTree). The view reads only
// these POD arrays and never re-runs layout on the UI thread.
#pragma once

#include <cstdint>
#include <vector>

#include "core/SmallVec.h"

namespace netvis {

// Screen-independent world coordinates. Layout works in world space; the canvas
// applies pan+zoom (spec §8.1).
struct Vec2 {
  float x = 0.0f;
  float y = 0.0f;
};

// A laid-out node box in world space. `display_id` maps back to the collapse
// view's node list (a leaf IR node or a collapsed super-node).
struct NodeBox {
  uint32_t display_id = 0;
  Vec2 pos;      // top-left
  Vec2 size;     // width/height from measured label extents
  int32_t layer = 0;
};

// A routed edge as a cubic bezier (spec §7.2.5): p0 at producer's bottom port,
// p3 at consumer's top port, p1/p2 control points for the vertical-ish curve.
struct EdgeCurve {
  uint32_t from_display_id = 0;
  uint32_t to_display_id = 0;
  Vec2 p0, p1, p2, p3;
  bool reversed = false;   // set if this edge was reversed to break a cycle
};

// A completed layout for one collapse level. Published atomically by LayoutJob;
// the canvas swaps to it in one assignment (spec §7.2.6).
struct LayoutResult {
  std::vector<NodeBox> boxes;   // indexed parallel to the display node list
  std::vector<EdgeCurve> edges;
  Vec2 bounds_min;              // world-space bounding box of the whole layout
  Vec2 bounds_max;
  uint64_t structure_hash = 0;  // key this layout was computed for
  uint64_t collapse_hash = 0;
  bool from_cache = false;
};

}  // namespace netvis
