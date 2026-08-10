// view/GraphCanvas.cpp — the interactive graph view (spec §8.1).
//
// DECISION (spec §8.1): the ENTIRE graph is drawn inside ONE ImGui child region
// using a single ImDrawList — there is NOT one ImGui widget per node. A 100k-node
// graph would die under 100k Buttons; instead we cull to the visible world rect
// and emit raw draw commands, so per-frame cost is O(visible), not O(nodes).
// Hit-testing is likewise done against the culled boxes, not via ImGui items.
//
// #99 (v0.9.3): "cull to the visible rect" originally meant TESTING every box and
// every edge against that rect each frame — the emit was O(visible) but the sweep
// was O(total), so zooming into a corner of a 100k-node graph cost exactly what
// showing all of it did. A SpatialIndex over the published layout now supplies the
// candidates, and the per-frame label allocations are gone with it.
#define IMGUI_DEFINE_MATH_OPERATORS
#include "engine/LayoutEngine.h"
#include "view/App.h"
#include "view/CostPanel.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"  // ImBezierCubicClosestPoint (#18 edge hit-test)

#include "engine/CostModel.h"    // #14: CostReport per-node flops (path weights)
#include "engine/GraphAdjacency.h"  // #14: longest_cost_path
#include "engine/OpCategory.h"
#include "engine/SpatialIndex.h"  // #99: O(visible) box/edge queries
#include "engine/plugin/Registry.h"
#include "view/DiffPanel.h"
#include "view/GraphNav.h"
#include "view/PanelHelpers.h"

namespace netvis {

namespace {

// Per-display-node readability classification for the "hide constant edges"
// toggle (Feature 2). Recomputed only when the session (generation / graph /
// collapse) changes — NOT every frame. is_const_source[i] marks a leaf display
// node that is a constant/initializer source (inputs.count==0 or op categorizes
// to OpCategory::Tensor); const_badge[i] counts a consumer's hidden constant/
// initializer inputs (for the "+N" badge).
struct ReadabilityCache {
  uint64_t key_generation = UINT64_MAX;
  uint32_t key_graph = UINT32_MAX;
  uint64_t key_collapse = UINT64_MAX;
  bool valid = false;
  std::vector<uint8_t> is_const_source;   // indexed by display id
  std::vector<uint16_t> const_badge;      // indexed by display id
};

// True if IR node `n` is a constant/initializer *source* (a leaf producing a
// constant with no compute inputs).
bool node_is_const_source(const ir::Model& m, const ir::Node& n) {
  if (n.inputs.count == 0) return true;
  return categorize_op(m.str(n.op_type)) == OpCategory::Tensor;
}

// Recompute the readability cache if the session key changed. Returns it.
const ReadabilityCache& readability_cache(App& app) {
  static ReadabilityCache cache;
  ModelSession& s = app.session();
  const uint64_t gen = s.generation();
  const uint32_t gi = s.current_graph();
  const uint64_t ch = s.collapse().collapse_hash();
  if (cache.valid && cache.key_generation == gen && cache.key_graph == gi &&
      cache.key_collapse == ch)
    return cache;

  cache.key_generation = gen;
  cache.key_graph = gi;
  cache.key_collapse = ch;
  cache.valid = true;

  const auto& disp = s.collapse().display_nodes();
  const size_t n = disp.size();
  cache.is_const_source.assign(n, 0);
  cache.const_badge.assign(n, 0);

  const ir::Model* m = s.model();
  if (m == nullptr || gi >= m->graphs.size()) return cache;
  const ir::Graph& g = m->graphs[gi];

  // Per-IR-node const-source classification + set of initializer value names.
  std::vector<uint8_t> node_const(g.nodes.size(), 0);
  for (size_t i = 0; i < g.nodes.size(); ++i)
    node_const[i] = node_is_const_source(*m, g.nodes[i]) ? 1 : 0;
  // Initializer value NAMES: a consumer input with no producer whose value name
  // matches an initializer is a hidden constant input (counted in the badge).
  std::vector<StringId> init_names;
  init_names.reserve(g.initializers.size());
  for (const ir::TensorRef& t : g.initializers) init_names.push_back(t.name);
  auto is_init_name = [&](StringId id) {
    for (StringId in : init_names)
      if (in == id) return true;
    return false;
  };

  for (size_t i = 0; i < n; ++i) {
    const DisplayNode& dn = disp[i];
    if (dn.is_group) continue;  // only leaf nodes are const sources.
    if (dn.ir_node >= g.nodes.size()) continue;
    if (node_const[dn.ir_node]) cache.is_const_source[i] = 1;

    // Count this consumer's hidden constant/initializer inputs.
    const ir::Node& node = g.nodes[dn.ir_node];
    uint32_t badge = 0;
    for (uint32_t slot = 0; slot < node.inputs.count; ++slot) {
      uint32_t vidx = panel_detail::resolve_edge_value(g, node.inputs, slot);
      if (vidx == UINT32_MAX || vidx >= g.values.size()) continue;
      const ir::ValueInfo& vi = g.values[vidx];
      if (vi.producer >= 0) {
        if (static_cast<size_t>(vi.producer) < node_const.size() &&
            node_const[vi.producer])
          ++badge;
      } else if (is_init_name(vi.name)) {
        ++badge;  // initializer input (no producing node / no edge).
      }
    }
    if (badge > 0)
      cache.const_badge[i] = static_cast<uint16_t>(std::min<uint32_t>(badge, 0xFFFFu));
  }
  return cache;
}

// #20 depth-ruler cache: per-Sugiyama-layer world-space vertical extents [lo,hi].
// These are a pure function of the published layout (independent of the camera),
// so we rebuild them ONLY when a new layout is published — keyed by the layout's
// structure+collapse hashes — rather than re-scanning all boxes every frame.
struct LayerBandCache {
  uint64_t key_structure = UINT64_MAX;
  uint64_t key_collapse = UINT64_MAX;
  bool valid = false;
  std::vector<float> lo, hi;  // indexed by layer
};

const LayerBandCache& layer_band_cache(const LayoutResult& layout) {
  static LayerBandCache cache;
  if (cache.valid && cache.key_structure == layout.structure_hash &&
      cache.key_collapse == layout.collapse_hash)
    return cache;
  cache.key_structure = layout.structure_hash;
  cache.key_collapse = layout.collapse_hash;
  cache.valid = true;
  cache.lo.clear();
  cache.hi.clear();
  for (const NodeBox& b : layout.boxes) {
    if (b.layer < 0) continue;
    size_t L = static_cast<size_t>(b.layer);
    if (L >= cache.lo.size()) {
      cache.lo.resize(L + 1, FLT_MAX);
      cache.hi.resize(L + 1, -FLT_MAX);
    }
    cache.lo[L] = std::min(cache.lo[L], b.pos.y);
    cache.hi[L] = std::max(cache.hi[L], b.pos.y + b.size.y);
  }
  return cache;
}

// #14 critical-path cache: the set of DISPLAY nodes on the heaviest cumulative-
// FLOP chain source→sink. Computed from the cost report's per-node FLOPs as edge
// weights over the graph adjacency (longest_cost_path), then mapped IR→display.
// Rebuilt only when the session key or the report identity changes — not per
// frame. `on[display_id] != 0` => draw a glow.
struct CriticalPathCache {
  uint64_t key_generation = UINT64_MAX;
  uint32_t key_graph = UINT32_MAX;
  uint64_t key_collapse = UINT64_MAX;
  const void* key_report = nullptr;   // CostReport pointer identity
  bool valid = false;
  std::vector<uint8_t> on;            // indexed by display id
};

const CriticalPathCache& critical_path_cache(App& app, const CostReport* report,
                                             const GraphAdjacency* adj) {
  static CriticalPathCache cache;
  ModelSession& s = app.session();
  const uint64_t gen = s.generation();
  const uint32_t gi = s.current_graph();
  const uint64_t ch = s.collapse().collapse_hash();
  if (cache.valid && cache.key_generation == gen && cache.key_graph == gi &&
      cache.key_collapse == ch && cache.key_report == report)
    return cache;
  cache.key_generation = gen;
  cache.key_graph = gi;
  cache.key_collapse = ch;
  cache.key_report = report;
  cache.valid = true;

  const auto& disp = s.collapse().display_nodes();
  cache.on.assign(disp.size(), 0);
  if (report == nullptr || adj == nullptr || report->per_node.empty()) return cache;

  // Per-node FLOP weights (0 where unknown). longest_cost_path returns the IR-node
  // chain maximizing summed weight; map each to its display node.
  std::vector<uint64_t> weight(report->per_node.size());
  for (size_t i = 0; i < report->per_node.size(); ++i)
    weight[i] = report->per_node[i].flops_known ? report->per_node[i].flops : 0;
  std::vector<uint32_t> chain = adj->longest_cost_path(weight);

  // Build a reverse index (IR node -> display id) in ONE O(display) pass rather
  // than a per-chain-node linear scan (which would be O(chain × display) — a
  // multi-second freeze on a deep 100k-node graph). A leaf display node maps its
  // ir_node directly; a collapsed group maps every member to the group's display
  // id (so a chain node hidden in a group lights the group, matching the old
  // display_index_for_node fallback).
  const auto& groups = s.collapse().groups();
  std::vector<int32_t> rev(report->per_node.size(), -1);
  for (size_t i = 0; i < disp.size(); ++i) {
    const DisplayNode& dn = disp[i];
    if (!dn.is_group) {
      if (dn.ir_node < rev.size()) rev[dn.ir_node] = static_cast<int32_t>(i);
    } else if (dn.group_index < groups.size()) {
      for (uint32_t member : groups[dn.group_index].member_nodes)
        if (member < rev.size() && rev[member] < 0)
          rev[member] = static_cast<int32_t>(i);
    }
  }
  for (uint32_t ir_node : chain) {
    if (ir_node >= rev.size()) continue;
    int32_t d = rev[ir_node];
    if (d >= 0 && static_cast<size_t>(d) < cache.on.size())
      cache.on[static_cast<size_t>(d)] = 1;
  }
  return cache;
}

// Badge glyph size tracks zoom like the node label text (clamped).
float text_px_for_badge(float zoom) {
  return std::clamp(11.0f * zoom, 11.0f * 0.5f, 11.0f * 3.0f);
}

// Apply an alpha multiplier to a packed color (for nav "dim" de-emphasis).
ImU32 with_alpha_mul(ImU32 col, float mul) {
  ImVec4 c = ImGui::ColorConvertU32ToFloat4(col);
  c.w *= mul;
  return ImGui::ColorConvertFloat4ToU32(c);
}

// LOD zoom thresholds (spec §8.1). Above kFull we draw the full node with two
// text lines + edge labels; between kMid..kFull just the op_type; between
// kFlat..kMid a flat rect with no text; below kBlob one blob per collapse group.
constexpr float kZoomFull = 0.70f;
constexpr float kZoomMid = 0.30f;
constexpr float kZoomFlat = 0.08f;
constexpr float kMinZoom = 0.02f;
constexpr float kMaxZoom = 4.0f;

// Smooth ease-in-out for the fly-to animation.
float ease(float t) { return t * t * (3.0f - 2.0f * t); }

// True if world-space AABB [amin,amax] intersects [bmin,bmax].
bool aabb_overlap(ImVec2 amin, ImVec2 amax, ImVec2 bmin, ImVec2 bmax) {
  return amin.x <= bmax.x && amax.x >= bmin.x && amin.y <= bmax.y &&
         amax.y >= bmin.y;
}

// Squared distance from `p` to the cubic bezier (p0..p3), via ImGui's
// closest-point helper (samples `steps` segments). Used by the #18 edge-hover
// hit-test; only called for visible edges when the tooltip is enabled and no node
// is under the cursor.
float dist_sq_point_bezier(ImVec2 p, ImVec2 p0, ImVec2 p1, ImVec2 p2, ImVec2 p3,
                           int steps) {
  ImVec2 c = ImBezierCubicClosestPoint(p0, p1, p2, p3, p, steps);
  float dx = c.x - p.x, dy = c.y - p.y;
  return dx * dx + dy * dy;
}

// Blend two packed colors by t in [0,1].
ImU32 lerp_col(ImU32 a, ImU32 b, float t) {
  ImVec4 ca = ImGui::ColorConvertU32ToFloat4(a);
  ImVec4 cb = ImGui::ColorConvertU32ToFloat4(b);
  ImVec4 r(ca.x + (cb.x - ca.x) * t, ca.y + (cb.y - ca.y) * t,
           ca.z + (cb.z - ca.z) * t, ca.w + (cb.w - ca.w) * t);
  return ImGui::ColorConvertFloat4ToU32(r);
}

// An empty label whose data() is never null, so `%.*s` and ImGui's begin/end
// AddText overload stay well-defined for a node with no resolvable text.
constexpr std::string_view kEmptyLabel{""};

// Resolve the two label lines + op category for one display node.
//
// #99: this used to be `NodeLabel label_for()` returning two std::strings BY
// VALUE, called for every VISIBLE node at BOTH LOD tiers — two heap allocations
// per visible node per frame, and by far the largest per-frame allocation source
// in the canvas. Nothing here needs to OWN bytes:
//   * a leaf's op_type/name already live in the model's StringArena, whose
//     backing std::deque keeps element addresses stable across growth
//     (core/StringArena.h), so a view into it is as good as a copy;
//   * a collapsed group's label is a std::string owned by the CollapseTree, alive
//     as long as the display list we just read it from;
//   * the only line we SYNTHESIZE — "xN" for a group's instance count — is
//     printed into `scratch`, a buffer inside the caller's own NodeLabel.
//
// LIFETIME: the views are valid until the next fill_label() on this NodeLabel,
// and only while the model / collapse tree that produced them is alive. Every
// call site below uses one stack NodeLabel reused across a draw loop and never
// stores it, so a view cannot outlive its source. Copying is deleted rather than
// merely discouraged: `secondary` may point into `scratch`, so a copy would
// silently alias the ORIGINAL's buffer.
struct NodeLabel {
  std::string_view primary = kEmptyLabel;
  std::string_view secondary = kEmptyLabel;
  OpCategory cat = OpCategory::Other;
  char scratch[16] = {};  // "x" + a uint32 decimal (10 digits) + NUL needs 12

  NodeLabel() = default;
  NodeLabel(const NodeLabel&) = delete;
  NodeLabel& operator=(const NodeLabel&) = delete;
};

void fill_label(const App& app, uint32_t display_id, NodeLabel& out) {
  out.primary = kEmptyLabel;
  out.secondary = kEmptyLabel;
  out.cat = OpCategory::Other;
  ModelSession& s = const_cast<App&>(app).session();
  const auto& disp = s.collapse().display_nodes();
  // An out-of-range id keeps the EMPTY primary rather than the "node" fallback
  // below — matching label_for's early return, which skipped that fallback.
  if (display_id >= disp.size()) return;
  const DisplayNode& dn = disp[display_id];
  const ir::Model* m = s.model();
  if (dn.is_group) {
    const auto& groups = s.collapse().groups();
    if (dn.group_index < groups.size()) {
      const CollapseGroup& g = groups[dn.group_index];
      out.primary = g.label;
      int n = std::snprintf(out.scratch, sizeof(out.scratch), "x%u",
                            static_cast<unsigned>(g.instances));
      // snprintf returns the length it WANTED to write; a negative or truncated
      // result must never become a view running off the end of the buffer.
      if (n > 0 && static_cast<size_t>(n) < sizeof(out.scratch))
        out.secondary = std::string_view(out.scratch, static_cast<size_t>(n));
      // Category from the group's first representative node op, if resolvable.
      if (m != nullptr && !g.representative_nodes.empty()) {
        uint32_t gi = s.current_graph();
        if (gi < m->graphs.size()) {
          const auto& nodes = m->graphs[gi].nodes;
          uint32_t ni = g.representative_nodes.front();
          if (ni < nodes.size())
            out.cat = plugin::resolve_category(*m, m->graphs[gi], nodes[ni]);
        }
      }
    }
  } else if (m != nullptr) {
    uint32_t gi = s.current_graph();
    if (gi < m->graphs.size()) {
      const auto& nodes = m->graphs[gi].nodes;
      if (dn.ir_node < nodes.size()) {
        const ir::Node& n = nodes[dn.ir_node];
        out.primary = m->str(n.op_type);
        out.secondary = m->str(n.name);
        out.cat = plugin::resolve_category(*m, m->graphs[gi], n);
      }
    }
  }
  if (out.primary.empty()) out.primary = "node";
}

// #99: the canvas's viewport index, so the hit-test / edge draw / node draw are
// O(visible) instead of O(total). The grid is a pure function of the published
// LayoutResult, so it is rebuilt only when that layout changes — never per frame.
//
// KEYING. A LayoutResult is reachable only through ModelSession::layout(), and
// this cache is a process-global static shared by every tab, so its key must
// separate two DIFFERENT tabs' layouts as reliably as two successive layouts of
// one tab. (structure_hash, collapse_hash) alone does NOT: open one file in two
// tabs and both layouts hash identically, so a hash-only key would serve tab A's
// grid for tab B's boxes — the same collision per-tab JobSystems already inflict
// on generation counters. The layout POINTER alone does not either: layout_ is a
// unique_ptr a re-layout replaces, and the allocator may hand the replacement the
// address the old one just freed. The key is therefore pointer AND both hashes
// AND both container sizes. Matching all five requires the previous layout to be
// destroyed and the new one to describe the same graph at the same collapse state
// with identical box/edge counts — and layout being deterministic in its inputs,
// that is the same drawing, so the grid is correct for it either way.
// indexed_boxes()/indexed_edges() are re-checked on every hit as a final backstop
// against ever indexing an array the grid was not built for.
struct CanvasIndexCache {
  SpatialIndex index;
  const LayoutResult* key_layout = nullptr;
  uint64_t key_structure = UINT64_MAX;
  uint64_t key_collapse = UINT64_MAX;
  size_t key_boxes = 0;
  size_t key_edges = 0;
  bool valid = false;
};

const SpatialIndex& canvas_index(const LayoutResult& layout) {
  static CanvasIndexCache c;
  if (c.valid && c.key_layout == &layout &&
      c.key_structure == layout.structure_hash &&
      c.key_collapse == layout.collapse_hash &&
      c.key_boxes == layout.boxes.size() &&
      c.key_edges == layout.edges.size() &&
      static_cast<size_t>(c.index.indexed_boxes()) == layout.boxes.size() &&
      static_cast<size_t>(c.index.indexed_edges()) == layout.edges.size())
    return c.index;
  c.key_layout = &layout;
  c.key_structure = layout.structure_hash;
  c.key_collapse = layout.collapse_hash;
  c.key_boxes = layout.boxes.size();
  c.key_edges = layout.edges.size();
  c.valid = true;
  c.index.build(layout);
  return c.index;
}

// Candidate list 0..n-1 — every item, which is exactly what the pre-#99 loops
// walked. Used when the grid cannot help (see kIndexAreaFraction) or reports
// !valid() on a degenerate zero-extent layout.
//
// The contents are a pure function of n, and n changes only when a new layout is
// published, so a list that is ALREADY 0..n-1 is left alone: the steady state is
// one size compare, not another 400 KB of stores per frame at the 100k rung. The
// size + last-element test is conclusive because the vector only ever holds
// ascending de-duplicated indices below n — n of those with max n-1 can only be
// the full run.
void fill_all_indices(std::vector<uint32_t>& out, size_t n) {
  if (out.size() == n &&
      (n == 0 || out.back() == static_cast<uint32_t>(n - 1)))
    return;
  out.resize(n);
  for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint32_t>(i);
}

// #99: the search-hit pulse used to LINEAR-SCAN every box every frame — building
// a NodeLabel, i.e. allocating, per box — purely to locate ONE box, and to re-run
// the whole fuzzy query over every search entry alongside it. Both are pure
// functions of (query, active result, search index, layout), none of which
// changes per frame, so the RESOLVED box is cached instead and the steady-state
// cost drops to a key comparison.
//
// Keyed on the search index's entries pointer + count (a rebuild replaces that
// vector) plus the same layout identity canvas_index() uses, for the same
// anti-collision reason — AND on the LIVE collapse hash, because the resolution
// walks the current display list, which toggle_group() rebuilds immediately
// while the matching layout is still being computed on a worker. `box` is the
// index into layout.boxes of the first match in ASCENDING order — the one the
// old loop's `break` picked — or -1 for "this target has no box in this layout"
// (it is inside a collapsed group, say), a result worth caching so a miss does
// not re-scan every frame either.
struct SearchPulseCache {
  const void* key_entries = nullptr;
  size_t key_entry_count = 0;
  std::string key_query;
  int key_active = -1;
  const LayoutResult* key_layout = nullptr;
  uint64_t key_structure = UINT64_MAX;
  uint64_t key_collapse = UINT64_MAX;
  uint64_t key_live_collapse = UINT64_MAX;
  size_t key_boxes = 0;
  bool valid = false;
  int64_t box = -1;
};

int64_t search_pulse_box(ModelSession& session, const LayoutResult& layout,
                         const std::string& query, int active_result) {
  static SearchPulseCache c;
  const SearchIndex& si = session.search();
  const void* entries = static_cast<const void*>(si.entries().data());
  const uint64_t live_collapse = session.collapse().collapse_hash();
  if (c.valid && c.key_entries == entries &&
      c.key_entry_count == si.entries().size() && c.key_query == query &&
      c.key_active == active_result && c.key_layout == &layout &&
      c.key_structure == layout.structure_hash &&
      c.key_collapse == layout.collapse_hash &&
      c.key_live_collapse == live_collapse &&
      c.key_boxes == layout.boxes.size())
    return c.box;
  c.key_entries = entries;
  c.key_entry_count = si.entries().size();
  c.key_query = query;
  c.key_active = active_result;
  c.key_layout = &layout;
  c.key_structure = layout.structure_hash;
  c.key_collapse = layout.collapse_hash;
  c.key_live_collapse = live_collapse;
  c.key_boxes = layout.boxes.size();
  c.valid = true;
  c.box = -1;

  std::vector<SearchHit> hits = si.query(query, 64);
  if (hits.empty()) return c.box;
  int idx = std::clamp(active_result, 0, static_cast<int>(hits.size()) - 1);
  const uint32_t entry = hits[static_cast<size_t>(idx)].entry;
  if (entry >= si.entries().size()) return c.box;
  const SearchEntry& se = si.entries()[entry];
  if (se.kind != SearchKind::Node) return c.box;

  // Map the hit's IR node to its display box. Only leaf display nodes carry an
  // ir_node, which is the old loop's `!dn.is_group` test unchanged.
  const auto& disp = session.collapse().display_nodes();
  for (size_t i = 0; i < layout.boxes.size(); ++i) {
    const uint32_t did = layout.boxes[i].display_id;
    if (did >= disp.size()) continue;
    const DisplayNode& dn = disp[did];
    if (!dn.is_group && dn.ir_node == se.ref) {
      c.box = static_cast<int64_t>(i);
      break;
    }
  }
  return c.box;
}

}  // namespace

// Draw the graph canvas. Called once per frame from App::frame() when a compute
// graph is loaded. All drawing is ImDrawList; input/animation are handled here.
void draw_graph_canvas(App& app) {
  ViewState& vs = app.view();
  ModelSession& session = app.session();

  ImGui::Begin("Graph");
  // Single child region: everything below is manual ImDrawList output.
  ImGui::BeginChild("canvas", ImVec2(0, 0), ImGuiChildFlags_None,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);

  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 origin = ImGui::GetCursorScreenPos();  // canvas top-left (screen)
  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const ImVec2 canvas_size(std::max(avail.x, 1.0f), std::max(avail.y, 1.0f));

  // An invisible full-region button captures hover/clicks for the canvas without
  // per-node widgets, and lets us know when interaction targets empty space.
  ImGui::InvisibleButton("canvas_btn", canvas_size,
                         ImGuiButtonFlags_MouseButtonLeft |
                             ImGuiButtonFlags_MouseButtonRight |
                             ImGuiButtonFlags_MouseButtonMiddle);
  const bool canvas_hovered = ImGui::IsItemHovered();
  ImGuiIO& io = ImGui::GetIO();

  Camera& cam = vs.cam;

  // --- Input: zoom to cursor -------------------------------------------------
  if (canvas_hovered && io.MouseWheel != 0.0f) {
    // Keep the world point under the cursor fixed while zooming (spec §8.1).
    ImVec2 before = screen_to_world(cam, origin, io.MousePos);
    float factor = std::pow(1.1f, io.MouseWheel);
    cam.zoom = std::clamp(cam.zoom * factor, kMinZoom, kMaxZoom);
    ImVec2 after = screen_to_world(cam, origin, io.MousePos);
    cam.pan.x += (after.x - before.x) * cam.zoom;
    cam.pan.y += (after.y - before.y) * cam.zoom;
    vs.animating = false;  // manual zoom cancels a fly-to.
  }

  // --- Input: pan (middle-drag, or space+left-drag) --------------------------
  const bool space = ImGui::IsKeyDown(ImGuiKey_Space);
  if (canvas_hovered &&
      (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) ||
       (space && ImGui::IsMouseDragging(ImGuiMouseButton_Left)))) {
    cam.pan.x += io.MouseDelta.x;
    cam.pan.y += io.MouseDelta.y;
    vs.animating = false;
  }

  // --- Camera animation step (fly-to) ---------------------------------------
  if (vs.animating) {
    // Advance ~over ~0.35s; ease and snap when close to avoid a lingering crawl.
    vs.anim_t += io.DeltaTime / 0.35f;
    if (vs.anim_t >= 1.0f) {
      vs.anim_t = 1.0f;
      vs.animating = false;
    }
    float e = ease(std::clamp(vs.anim_t, 0.0f, 1.0f));
    cam.zoom = vs.anim_from_zoom + (vs.anim_to_zoom - vs.anim_from_zoom) * e;
    cam.pan.x = vs.anim_from_pan.x + (vs.anim_to_pan.x - vs.anim_from_pan.x) * e;
    cam.pan.y = vs.anim_from_pan.y + (vs.anim_to_pan.y - vs.anim_from_pan.y) * e;
  }

  const LayoutResult* layout = session.layout();

  // --- Fit request (F key) ---------------------------------------------------
  if (vs.request_fit) {
    if (layout != nullptr) {
      fit_camera(vs, ImVec2(layout->bounds_min.x, layout->bounds_min.y),
                 ImVec2(layout->bounds_max.x, layout->bounds_max.y),
                 canvas_size);
    }
    vs.request_fit = false;
  }

  // Clip drawing to the canvas rect.
  const ImVec2 canvas_max(origin.x + canvas_size.x, origin.y + canvas_size.y);
  dl->PushClipRect(origin, canvas_max, true);

  if (layout == nullptr || layout->boxes.empty()) {
    // Nothing laid out yet — show a hint centered in the canvas.
    const char* msg = session.model() ? "Laying out graph..."
                                      : "Open a model (File > Open, or drop a file)";
    ImVec2 ts = ImGui::CalcTextSize(msg);
    dl->AddText(ImVec2(origin.x + (canvas_size.x - ts.x) * 0.5f,
                       origin.y + (canvas_size.y - ts.y) * 0.5f),
                ImGui::GetColorU32(ImGuiCol_TextDisabled), msg);
    dl->PopClipRect();
    draw_minimap(app);
    ImGui::EndChild();
    ImGui::End();
    return;
  }

  // --- Compute the visible world rect (for culling) --------------------------
  // PERF: transform the canvas corners back to world space once. This AABB is
  // both the precise per-item cull test AND the rect handed to the #99 spatial
  // index, which is what makes the loops below O(visible) rather than O(total) —
  // testing every item against this rect was still a full sweep of the arrays.
  ImVec2 vw_min = screen_to_world(cam, origin, origin);
  ImVec2 vw_max = screen_to_world(cam, origin, canvas_max);
  if (vw_min.x > vw_max.x) std::swap(vw_min.x, vw_max.x);
  if (vw_min.y > vw_max.y) std::swap(vw_min.y, vw_max.y);

  const float zoom = cam.zoom;
  const bool dark = vs.dark_theme;
  const double now = ImGui::GetTime();
  // Pulse factor for search-hit highlight (0..1), driven by wall time.
  const float pulse =
      0.5f + 0.5f * static_cast<float>(std::sin(now * 4.0));

  const ImU32 col_edge = dark ? IM_COL32(150, 160, 175, 160)
                              : IM_COL32(90, 100, 120, 170);
  const ImU32 col_edge_hi = IM_COL32(120, 200, 255, 255);
  const ImU32 col_accent = IM_COL32(90, 170, 255, 255);
  const ImU32 col_text = dark ? IM_COL32(235, 238, 242, 255)
                             : IM_COL32(24, 28, 36, 255);
  const ImU32 col_text_muted = dark ? IM_COL32(160, 168, 180, 255)
                                    : IM_COL32(90, 100, 116, 255);
  const ImU32 col_border = dark ? IM_COL32(20, 22, 26, 220)
                               : IM_COL32(120, 128, 140, 220);

  const Fonts& fonts = app.fonts();

  // --- #14 critical path: display nodes on the heaviest FLOP chain -----------
  // Computed (cached) from the cost report + graph adjacency when the toggle is on.
  const CriticalPathCache* crit = nullptr;
  if (vs.show_critical_path) {
    const GraphAdjacency* adj = vs.nav ? vs.nav->adj.get() : nullptr;
    crit = &critical_path_cache(app, vs.cost.get(), adj);
  }
  auto on_critical = [&](uint32_t did) -> bool {
    return crit != nullptr && did < crit->on.size() && crit->on[did] != 0;
  };

  // --- #20 Depth ruler: faint per-layer bands --------------------------------
  // Reads NodeBox::layer (already produced by the Sugiyama layout). We derive each
  // layer's world-space vertical extent [top,bottom] in one O(boxes) pass, then
  // draw alternating faint bands so graph DEPTH is legible at a glance. Toggle via
  // view().show_layer_bands (View menu). Skipped at very low zoom (bands would be
  // sub-pixel) and drawn UNDER edges/nodes.
  if (vs.show_layer_bands && zoom >= kZoomFlat) {
    // Per-layer extents are cached (rebuilt only on a new layout), so per frame we
    // just transform + draw O(layers) bands — no O(boxes) rescan.
    const LayerBandCache& lb = layer_band_cache(*layout);
    const ImU32 band = dark ? IM_COL32(255, 255, 255, 8)
                            : IM_COL32(20, 30, 50, 10);
    for (size_t L = 0; L < lb.lo.size(); ++L) {
      if (L % 2 == 1) continue;             // every other layer gets a tint.
      if (lb.lo[L] == FLT_MAX) continue;    // empty layer.
      // World y-range -> screen; the band spans the full canvas width.
      ImVec2 top = world_to_screen(cam, origin, ImVec2(vw_min.x, lb.lo[L]));
      ImVec2 bot = world_to_screen(cam, origin, ImVec2(vw_max.x, lb.hi[L]));
      // Cull bands fully outside the canvas vertically.
      if (bot.y < origin.y || top.y > canvas_max.y) continue;
      dl->AddRectFilled(ImVec2(origin.x, top.y), ImVec2(canvas_max.x, bot.y), band);
    }
  }

  // --- Navigation masks + readability cache (display-id indexed) -------------
  // ensure_nav() (called from App::frame before us) keeps these in sync; guard
  // for a null nav or a stale-size mask across a display-list rebuild this frame.
  const GraphNavState* nav = vs.nav.get();
  const size_t disp_count = session.collapse().display_nodes().size();
  auto nav_hidden = [&](uint32_t did) -> bool {
    return nav != nullptr && did < nav->hidden.size() &&
           nav->hidden.size() == disp_count && nav->hidden[did] != 0;
  };
  auto nav_dim = [&](uint32_t did) -> bool {
    return nav != nullptr && did < nav->dim.size() &&
           nav->dim.size() == disp_count && nav->dim[did] != 0;
  };
  const bool hide_const = vs.hide_const_edges;
  const ReadabilityCache& rc = readability_cache(app);
  auto is_const_source = [&](uint32_t did) -> bool {
    return hide_const && did < rc.is_const_source.size() &&
           rc.is_const_source.size() == disp_count &&
           rc.is_const_source[did] != 0;
  };
  // A display box is culled entirely when nav hides it OR it is a hidden const
  // source. A box is de-emphasized when nav dims it.
  auto box_culled = [&](uint32_t did) -> bool {
    return nav_hidden(did) || is_const_source(did);
  };
  const float kDimAlpha = 0.18f;

  // --- #99 Viewport query: the candidate boxes/edges for THIS frame -----------
  // Everything below (hover hit-test, edge draw, node draw) used to walk the FULL
  // boxes/edges arrays and AABB-test each — O(total), so zooming into a corner of
  // a 100k-node graph cost exactly as much as showing all of it. The grid narrows
  // that to the items overlapping the view rect; the precise AABB tests are kept
  // verbatim below, because a grid returns CANDIDATES (an item overlapping the
  // queried CELLS), never exact hits.
  //
  // The two candidate vectors are function-local statics so their heap buffers
  // survive between frames — SpatialIndex::query_* clears and refills whatever it
  // is handed, and a fresh vector per frame would reintroduce precisely the
  // allocation this change removes. Main-thread only, like the whole canvas.
  static std::vector<uint32_t> vis_boxes;
  static std::vector<uint32_t> vis_edges;
  {
    // A viewport that already covers most of the layout has almost nothing to
    // cull, and there the grid is pure overhead — it walks every bucket, visits
    // each item once per cell it spans, and hands back nearly the whole array
    // regardless. Measured on a 100k-box synthetic layout, the crossover is
    // ~25% of the bounds AREA: below it the grid runs 1.1x-670x faster than the
    // old full scan, above it up to 3x SLOWER. So past the threshold we
    // enumerate everything, which is what the pre-#99 loops did anyway. Both
    // paths feed the same ascending candidate list into the same precise tests,
    // so which one runs is invisible in the picture.
    constexpr double kIndexAreaFraction = 0.25;
    const double lw = static_cast<double>(layout->bounds_max.x) -
                      static_cast<double>(layout->bounds_min.x);
    const double lh = static_cast<double>(layout->bounds_max.y) -
                      static_cast<double>(layout->bounds_min.y);
    const double ow = std::min(static_cast<double>(vw_max.x),
                               static_cast<double>(layout->bounds_max.x)) -
                      std::max(static_cast<double>(vw_min.x),
                               static_cast<double>(layout->bounds_min.x));
    const double oh = std::min(static_cast<double>(vw_max.y),
                               static_cast<double>(layout->bounds_max.y)) -
                      std::max(static_cast<double>(vw_min.y),
                               static_cast<double>(layout->bounds_min.y));
    // Zero-extent bounds count as fully covered (and would divide by zero).
    const bool covers_most =
        !(lw > 0.0) || !(lh > 0.0) ||
        std::max(ow, 0.0) * std::max(oh, 0.0) >= kIndexAreaFraction * lw * lh;

    // Building the grid is deferred to the first frame that actually queries it,
    // so opening a model fit-to-screen never pays for an index it never reads.
    bool queried = false;
    if (!covers_most) {
      const SpatialIndex& sindex = canvas_index(*layout);
      if (sindex.valid()) {
        const WorldRect view_rect{vw_min.x, vw_min.y, vw_max.x, vw_max.y};
        sindex.query_boxes(view_rect, vis_boxes);
        sindex.query_edges(view_rect, vis_edges);
        queried = true;
      }
    }
    if (!queried) {
      fill_all_indices(vis_boxes, layout->boxes.size());
      fill_all_indices(vis_edges, layout->edges.size());
    }
  }

  // --- Hover hit-test (against culled boxes) ---------------------------------
  const ImVec2 mouse = io.MousePos;
  int32_t hover_box = -1;
  if (canvas_hovered) {
    // Iterate reverse so topmost (later) boxes win ties. The index returns
    // candidates in ASCENDING order, so this walks the candidate list BACKWARDS
    // to reproduce the old `for (i = boxes.size(); i-- > 0;)` tie-break exactly.
    // Walking it forwards would silently hand the hover to the BOTTOM box of any
    // overlapping stack — a visible behaviour change, not a perf detail.
    for (size_t k = vis_boxes.size(); k-- > 0;) {
      const uint32_t bi = vis_boxes[k];
      if (bi >= layout->boxes.size()) continue;  // never index a stale grid
      const NodeBox& b = layout->boxes[bi];
      if (box_culled(b.display_id)) continue;  // can't hover a culled box.
      ImVec2 bmin(b.pos.x, b.pos.y);
      ImVec2 bmax(b.pos.x + b.size.x, b.pos.y + b.size.y);
      if (!aabb_overlap(bmin, bmax, vw_min, vw_max)) continue;
      ImVec2 smin = world_to_screen(cam, origin, bmin);
      ImVec2 smax = world_to_screen(cam, origin, bmax);
      if (mouse.x >= smin.x && mouse.x <= smax.x && mouse.y >= smin.y &&
          mouse.y <= smax.y) {
        hover_box = static_cast<int32_t>(b.display_id);
        break;
      }
    }
  }
  vs.hovered_display = hover_box;

  // --- Draw edges first (under nodes) ----------------------------------------
  // PERF: skip any edge whose endpoints' bounding box is fully outside view.
  const float edge_thick = std::clamp(1.5f * zoom, 0.75f, 3.0f);
  // #18 edge hover: only hit-test edges when the tooltip is enabled, the canvas is
  // hovered, and NO node is under the cursor (nodes win). Track the nearest edge.
  const bool want_edge_hover =
      vs.edge_tooltips && canvas_hovered && hover_box < 0 && zoom >= kZoomFlat;
  int32_t hover_edge = -1;
  float hover_edge_d2 = FLT_MAX;
  const float kEdgePickPx = 6.0f;  // pointer must be within this many px of a curve
  // #99: ascending candidate indices, so edges still emit in layout order — the
  // grid de-duplicates an edge spanning several cells precisely so this holds.
  for (const uint32_t edge_idx : vis_edges) {
    if (edge_idx >= layout->edges.size()) continue;  // never index a stale grid
    const EdgeCurve& e = layout->edges[edge_idx];
    // Feature 2: skip edges whose source is a hidden constant/initializer box.
    // Also cull edges touching a nav-hidden endpoint.
    if (is_const_source(e.from_display_id)) continue;
    if (nav_hidden(e.from_display_id) || nav_hidden(e.to_display_id)) continue;
    const bool edge_dim =
        nav_dim(e.from_display_id) || nav_dim(e.to_display_id);
    ImVec2 emin(std::min(std::min(e.p0.x, e.p1.x), std::min(e.p2.x, e.p3.x)),
                std::min(std::min(e.p0.y, e.p1.y), std::min(e.p2.y, e.p3.y)));
    ImVec2 emax(std::max(std::max(e.p0.x, e.p1.x), std::max(e.p2.x, e.p3.x)),
                std::max(std::max(e.p0.y, e.p1.y), std::max(e.p2.y, e.p3.y)));
    if (!aabb_overlap(emin, emax, vw_min, vw_max)) continue;
    ImVec2 p0 = world_to_screen(cam, origin, ImVec2(e.p0.x, e.p0.y));
    ImVec2 p1 = world_to_screen(cam, origin, ImVec2(e.p1.x, e.p1.y));
    ImVec2 p2 = world_to_screen(cam, origin, ImVec2(e.p2.x, e.p2.y));
    ImVec2 p3 = world_to_screen(cam, origin, ImVec2(e.p3.x, e.p3.y));
    const bool touches_hover =
        hover_box >= 0 && (static_cast<int32_t>(e.from_display_id) == hover_box ||
                           static_cast<int32_t>(e.to_display_id) == hover_box);
    ImU32 ec = touches_hover ? col_edge_hi : col_edge;
    float th = touches_hover ? edge_thick + 1.0f : edge_thick;
    if (edge_dim && !touches_hover) ec = with_alpha_mul(ec, kDimAlpha);
    // #22: route the edge per the chosen style. Bezier uses the layout control
    // points; Orthogonal is a 3-segment Manhattan path (down / across / down);
    // Straight is a direct line. All share the same p0->p3 endpoints.
    switch (vs.edge_routing) {
      case 1: {  // orthogonal (Manhattan)
        float midy = (p0.y + p3.y) * 0.5f;
        dl->AddLine(p0, ImVec2(p0.x, midy), ec, th);
        dl->AddLine(ImVec2(p0.x, midy), ImVec2(p3.x, midy), ec, th);
        dl->AddLine(ImVec2(p3.x, midy), p3, ec, th);
        break;
      }
      case 2:  // straight
        dl->AddLine(p0, p3, ec, th);
        break;
      default:  // bezier
        dl->AddBezierCubic(p0, p1, p2, p3, ec, th);
        break;
    }

    // #18: track the nearest edge under the cursor (screen space). Hit-test uses
    // the bezier approximation regardless of routing (close enough for picking).
    if (want_edge_hover) {
      float d2 = dist_sq_point_bezier(mouse, p0, p1, p2, p3, 12);
      if (d2 < hover_edge_d2) {
        hover_edge_d2 = d2;
        hover_edge = static_cast<int32_t>(edge_idx);
      }
    }
  }

  // #18: shape/dtype tooltip for the hovered edge. Uses the value_index the layout
  // recorded on the edge (frozen Layout.h contract) to look up the ValueInfo in the
  // current graph — pure structural read, no payload bytes.
  if (want_edge_hover && hover_edge >= 0 &&
      hover_edge_d2 <= kEdgePickPx * kEdgePickPx) {
    const EdgeCurve& e = layout->edges[static_cast<size_t>(hover_edge)];
    const ir::Model* m = session.model();
    uint32_t gi = session.current_graph();
    if (m != nullptr && gi < m->graphs.size() &&
        e.value_index != UINT32_MAX &&
        e.value_index < m->graphs[gi].values.size()) {
      const ir::ValueInfo& vi = m->graphs[gi].values[e.value_index];
      std::string_view nm = m->str(vi.name);
      ImGui::BeginTooltip();
      if (!nm.empty()) ImGui::TextUnformatted(std::string(nm).c_str());
      ImGui::Text("%s  %s", panel_detail::shape_string(vi.shape).c_str(),
                  ir::dtype_name(vi.dtype));
      ImGui::EndTooltip();
    }
  }

  // --- Draw nodes (LOD by zoom) ----------------------------------------------
  // #99: both tiers walk the ascending candidate list (layout order preserved)
  // and reuse ONE NodeLabel across the loop, so no node costs an allocation.
  NodeLabel lab;
  if (zoom < kZoomFlat) {
    // Lowest LOD: one blob per collapse group / node, tinted by category.
    for (const uint32_t bi : vis_boxes) {
      if (bi >= layout->boxes.size()) continue;  // never index a stale grid
      const NodeBox& b = layout->boxes[bi];
      if (box_culled(b.display_id)) continue;
      ImVec2 bmin(b.pos.x, b.pos.y);
      ImVec2 bmax(b.pos.x + b.size.x, b.pos.y + b.size.y);
      if (!aabb_overlap(bmin, bmax, vw_min, vw_max)) continue;
      fill_label(app, b.display_id, lab);
      ImVec2 c = world_to_screen(
          cam, origin,
          ImVec2((bmin.x + bmax.x) * 0.5f, (bmin.y + bmax.y) * 0.5f));
      float r = std::max(2.0f, 0.25f * b.size.x * zoom);
      // Diff overlay overrides the category color at the low-LOD blob site.
      ImU32 blob = App::category_color(lab.cat, dark);
      // Diff overlay wins; else cost heatmap; else category color.
      DiffTint tint = diff_tint_for_display(app, static_cast<int32_t>(b.display_id));
      if (tint.active) {
        blob = tint.color;
      } else {
        CostTint ct = cost_tint_for_display(app, static_cast<int32_t>(b.display_id));
        if (ct.active) blob = ct.color;
      }
      if (nav_dim(b.display_id)) blob = with_alpha_mul(blob, kDimAlpha);
      dl->AddCircleFilled(c, r, blob, 8);
    }
  } else {
    for (const uint32_t bi : vis_boxes) {
      if (bi >= layout->boxes.size()) continue;  // never index a stale grid
      const NodeBox& b = layout->boxes[bi];
      if (box_culled(b.display_id)) continue;
      ImVec2 bmin(b.pos.x, b.pos.y);
      ImVec2 bmax(b.pos.x + b.size.x, b.pos.y + b.size.y);
      if (!aabb_overlap(bmin, bmax, vw_min, vw_max)) continue;

      ImVec2 smin = world_to_screen(cam, origin, bmin);
      ImVec2 smax = world_to_screen(cam, origin, bmax);
      const bool selected =
          vs.selected_display == static_cast<int32_t>(b.display_id);
      const bool hovered = hover_box == static_cast<int32_t>(b.display_id);
      const bool dimmed = nav_dim(b.display_id);
      fill_label(app, b.display_id, lab);
      ImU32 header = App::category_color(lab.cat, dark);
      // Diff overlay wins; else cost heatmap; else category color.
      DiffTint tint = diff_tint_for_display(app, static_cast<int32_t>(b.display_id));
      if (tint.active) {
        header = tint.color;
      } else {
        CostTint ct = cost_tint_for_display(app, static_cast<int32_t>(b.display_id));
        if (ct.active) header = ct.color;
      }
      if (dimmed) header = with_alpha_mul(header, kDimAlpha);

      ImU32 body = dark ? IM_COL32(34, 38, 44, 255) : IM_COL32(244, 246, 249, 255);
      if (hovered)
        body = dark ? IM_COL32(48, 54, 62, 255) : IM_COL32(232, 238, 246, 255);
      if (dimmed) body = with_alpha_mul(body, kDimAlpha);

      if (zoom < kZoomMid) {
        // Flat rect, no text.
        dl->AddRectFilled(smin, smax, body, 3.0f);
        dl->AddRect(smin, smax, header, 3.0f, 0, 2.0f);
      } else {
        // Rounded rect body + colored header strip.
        const float rounding = 5.0f;
        dl->AddRectFilled(smin, smax, body, rounding);
        float header_h = std::min(20.0f * zoom, (smax.y - smin.y) * 0.5f);
        ImVec2 hmax(smax.x, smin.y + header_h);
        dl->AddRectFilled(smin, hmax, header, rounding,
                          ImDrawFlags_RoundCornersTop);
        dl->AddRect(smin, smax, col_border, rounding, 0, 1.0f);

        // Text. High LOD draws both lines; mid LOD op_type only. Label sizes
        // scale WITH zoom so text tracks the node box — the world-space node
        // grows by `zoom`, so the glyphs must too. (A previous cap of
        // min(zoom,1.0) froze text at its baked pixel size, making it shrink to
        // an invisible speck relative to a zoomed-in node.) Clamp the upper size
        // so the font atlas doesn't blur badly at extreme zoom.
        auto text_px = [zoom](float base) {
          return std::clamp(base * zoom, base * 0.5f, base * 3.0f);
        };
        // #99: the begin/end AddText overload takes the label views directly, so
        // no NUL-terminated copy is materialized to draw a node. ImDrawList
        // returns early when begin == end, so an empty line still draws nothing.
        dl->PushClipRect(smin, smax, true);
        if (zoom > kZoomFull) {
          if (fonts.bold != nullptr)
            dl->AddText(fonts.bold, text_px(16.0f),
                        ImVec2(smin.x + 6.0f * zoom, smin.y + 2.0f * zoom),
                        col_text, lab.primary.data(),
                        lab.primary.data() + lab.primary.size());
          if (fonts.small != nullptr && !lab.secondary.empty())
            dl->AddText(fonts.small, text_px(12.0f),
                        ImVec2(smin.x + 6.0f * zoom, smin.y + header_h + 2.0f),
                        col_text_muted, lab.secondary.data(),
                        lab.secondary.data() + lab.secondary.size());
        } else {
          if (fonts.body != nullptr)
            dl->AddText(fonts.body, text_px(14.0f),
                        ImVec2(smin.x + 6.0f * zoom, smin.y + 2.0f * zoom),
                        col_text, lab.primary.data(),
                        lab.primary.data() + lab.primary.size());
        }
        dl->PopClipRect();
      }

      // #14: critical-path glow — a warm outline on nodes of the heaviest FLOP
      // chain (drawn under the selection outline so selection still reads on top).
      if (on_critical(b.display_id)) {
        const ImU32 glow = IM_COL32(255, 170, 60, 235);
        dl->AddRect(ImVec2(smin.x - 3, smin.y - 3), ImVec2(smax.x + 3, smax.y + 3),
                    glow, 7.0f, 0, 3.0f);
      }

      // Selection / search-hit outlines.
      if (selected) {
        dl->AddRect(ImVec2(smin.x - 2, smin.y - 2), ImVec2(smax.x + 2, smax.y + 2),
                    col_accent, 6.0f, 0, 2.5f);
      }

      // Feature 2: "+N" badge counting this consumer's hidden constant/
      // initializer inputs (only when the hide-const toggle is on and we're at a
      // legible LOD).
      if (hide_const && zoom >= kZoomFlat &&
          b.display_id < rc.const_badge.size() &&
          rc.const_badge.size() == disp_count && rc.const_badge[b.display_id] > 0 &&
          fonts.small != nullptr) {
        char badge[16];
        std::snprintf(badge, sizeof(badge), "+%u",
                      static_cast<unsigned>(rc.const_badge[b.display_id]));
        float bs = text_px_for_badge(zoom);
        ImVec2 ts = fonts.small->CalcTextSizeA(bs, FLT_MAX, 0.0f, badge);
        ImVec2 bp(smin.x - ts.x - 4.0f, smin.y);
        dl->AddRectFilled(ImVec2(bp.x - 2, bp.y - 1),
                          ImVec2(bp.x + ts.x + 2, bp.y + ts.y + 1),
                          IM_COL32(60, 66, 78, 230), 3.0f);
        dl->AddText(fonts.small, bs, bp, col_text_muted, badge);
      }
    }

    // Search-hit pulse: outline the current search target if it maps to a box.
    // #99: the target box is resolved once and cached (see search_pulse_box);
    // this used to re-run the fuzzy query AND scan every box, allocating a
    // NodeLabel per box, every single frame the overlay was open.
    if (vs.search_open && !vs.search_query.empty()) {
      const int64_t target = search_pulse_box(session, *layout, vs.search_query,
                                              vs.search_active_result);
      if (target >= 0 && static_cast<size_t>(target) < layout->boxes.size()) {
        const NodeBox& b = layout->boxes[static_cast<size_t>(target)];
        ImVec2 smin = world_to_screen(cam, origin, ImVec2(b.pos.x, b.pos.y));
        ImVec2 smax = world_to_screen(
            cam, origin, ImVec2(b.pos.x + b.size.x, b.pos.y + b.size.y));
        ImU32 pc = lerp_col(col_accent, IM_COL32(255, 220, 120, 255), pulse);
        dl->AddRect(ImVec2(smin.x - 3, smin.y - 3),
                    ImVec2(smax.x + 3, smax.y + 3), pc, 7.0f, 0, 3.0f);
      }
    }
  }

  dl->PopClipRect();

  // --- Interactions ----------------------------------------------------------
  // #17: record a leaf-node selection in the focus history so back/forward can
  // step through visited nodes. Maps a display id to its stable IR node index.
  auto record_focus_for_display = [&](int32_t disp_id) {
    if (disp_id < 0) return;
    const auto& disp = session.collapse().display_nodes();
    if (static_cast<size_t>(disp_id) >= disp.size()) return;
    const DisplayNode& dn = disp[static_cast<size_t>(disp_id)];
    if (!dn.is_group) nav_record_focus(app, dn.ir_node);
  };

  if (canvas_hovered) {
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !space) {
      vs.selected_display = hover_box;  // -1 clears when clicking empty space.
      record_focus_for_display(hover_box);
    }
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && hover_box >= 0) {
      const auto& disp = session.collapse().display_nodes();
      if (static_cast<size_t>(hover_box) < disp.size()) {
        const DisplayNode& dn = disp[static_cast<size_t>(hover_box)];
        if (dn.is_group && dn.group_index != UINT32_MAX)
          session.toggle_group(dn.group_index);  // expand/collapse (spec §7.2.6).
      }
    }
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Right) && hover_box >= 0) {
      vs.selected_display = hover_box;
      ImGui::OpenPopup("canvas_ctx");
    }
  }

  if (ImGui::BeginPopup("canvas_ctx")) {
    if (vs.selected_display >= 0) {
      NodeLabel ctx_lab;
      fill_label(app, static_cast<uint32_t>(vs.selected_display), ctx_lab);
      ImGui::TextDisabled("%.*s", static_cast<int>(ctx_lab.primary.size()),
                          ctx_lab.primary.data());
      ImGui::Separator();
      if (ImGui::MenuItem("Copy name")) {
        // The clipboard API needs a NUL-terminated buffer, so this ONE call site
        // materializes a string — on a click, not per frame, so it is free.
        std::string_view nm =
            ctx_lab.secondary.empty() ? ctx_lab.primary : ctx_lab.secondary;
        ImGui::SetClipboardText(std::string(nm).c_str());
      }
      // #57: copy the node's op/attrs/input-output shapes as JSON to the clipboard.
      const ir::Model* cm = session.model();
      uint32_t cgi = session.current_graph();
      const auto& cdisp = session.collapse().display_nodes();
      if (cm != nullptr && cgi < cm->graphs.size() &&
          static_cast<size_t>(vs.selected_display) < cdisp.size()) {
        const DisplayNode& cdn = cdisp[static_cast<size_t>(vs.selected_display)];
        if (!cdn.is_group && cdn.ir_node < cm->graphs[cgi].nodes.size() &&
            ImGui::MenuItem("Copy as JSON")) {
          const ir::Graph& cg = cm->graphs[cgi];
          std::string js =
              panel_detail::node_to_json(*cm, cg, cg.nodes[cdn.ir_node]);
          ImGui::SetClipboardText(js.c_str());
        }
      }
      const auto& disp = session.collapse().display_nodes();
      if (static_cast<size_t>(vs.selected_display) < disp.size()) {
        const DisplayNode& dn = disp[static_cast<size_t>(vs.selected_display)];
        if (dn.is_group && ImGui::MenuItem("Expand / collapse"))
          session.toggle_group(dn.group_index);

        // #15 path-between + #19 pin operate on a leaf node's STABLE IR index (a
        // display id would be repointed by a later collapse/expand — see #15 in
        // GraphNav.h). vs.nav is non-null here (ensure_nav ran before the canvas).
        if (!dn.is_group && vs.nav) {
          ImGui::Separator();
          if (ImGui::MenuItem("Path: set as A"))
            vs.nav->path_a = static_cast<int32_t>(dn.ir_node);
          if (ImGui::MenuItem("Path: set as B"))
            vs.nav->path_b = static_cast<int32_t>(dn.ir_node);
          if (vs.nav->path_active() && ImGui::MenuItem("Path: clear")) {
            vs.nav->path_a = -1;
            vs.nav->path_b = -1;
          }
          ImGui::Separator();
          const bool pinned =
              std::find(vs.nav->pinned.begin(), vs.nav->pinned.end(),
                        dn.ir_node) != vs.nav->pinned.end();
          if (ImGui::MenuItem(pinned ? "Unpin node" : "Pin node"))
            nav_toggle_pin(app, dn.ir_node);
        }
      }
    }
    ImGui::EndPopup();
  }

  // #19: pinned-node strip along the canvas top edge (screen-space overlay). Drawn
  // over nodes so pins stay visible while panning a huge graph.
  draw_pinned_strip(app, origin, canvas_size);

  // Minimap is drawn inside the same child region, inset bottom-right.
  draw_minimap(app);

  // Heatmap legend, inset top-left (only when the cost heatmap is active).
  draw_heatmap_legend(app);

  ImGui::EndChild();
  ImGui::End();
}

}  // namespace netvis
