// view/CostPanel.h — analyzer / cost-report view surface (v0.3.0 analyzer mode).
//
// Module-private to view/. Mirrors the GraphNav pattern: the heavy work
// (compute_cost) lives in the engine and is rebuilt lazily/keyed by
// ensure_cost(); this file is pure ImGui + reading the published CostReport, so
// the view never includes a parser and never reads a payload. The cost-heatmap
// overlay mirrors DiffPanel's DiffTint: GraphCanvas consults cost_tint_for_display
// at the header-color and low-LOD blob sites and overrides the category color
// when the heatmap toggle is on.
#pragma once

#include <cstdint>

#include "imgui.h"

#include "engine/HeatmapGradient.h"

namespace netvis {

class App;  // forward; defined in view/App.h

// Convert an engine-side Rgba8 to an ImGui packed color.
inline ImU32 rgba8_to_imu32(const Rgba8& c) {
  return IM_COL32(c.r, c.g, c.b, c.a);
}
inline Rgba8 imu32_to_rgba8(ImU32 c) {
  return Rgba8{static_cast<uint8_t>((c >> IM_COL32_R_SHIFT) & 0xFF),
               static_cast<uint8_t>((c >> IM_COL32_G_SHIFT) & 0xFF),
               static_cast<uint8_t>((c >> IM_COL32_B_SHIFT) & 0xFF),
               static_cast<uint8_t>((c >> IM_COL32_A_SHIFT) & 0xFF)};
}

// Normalized [min,max] FLOPs the heatmap maps its gradient across, computed over
// the SAME aggregation unit the tint uses (per display node), so a collapsed group
// and a leaf share one scale. valid==false when no display node has known FLOPs.
struct HeatmapRange {
  bool valid = false;
  uint64_t min_flops = 0;
  uint64_t max_flops = 0;
};
// Compute the heatmap range for the current model/report/collapse state. Cheap
// (O(display nodes)); used by both cost_tint_for_display and the legend so they
// never diverge (the v0.3.1 group-scale bug was exactly such a divergence).
HeatmapRange heatmap_range(App& app);

// Draw the heatmap legend (gradient bar + min/max FLOP labels) as a canvas overlay
// — call from draw_graph_canvas while its draw list is active, AFTER the nodes.
// No-op when the heatmap toggle is off or the range is invalid.
void draw_heatmap_legend(App& app);

// Rebuild App.view().cost (a CostReport) if stale. Keyed on the primary session's
// generation() + current_graph() + collapse().collapse_hash() so it recomputes on
// reopen, subgraph dive, and expand/collapse — same rebuild discipline as
// ensure_nav(). No-op (and leaves cost null) until a model with resolved shapes is
// published. Cheap enough to call once per frame from App::frame().
void ensure_cost(App& app);

// The cost-heatmap tint for a display node, or {active=false} when the heatmap
// toggle is off / no cost report / the node index is out of range. Color ramps by
// LOG10(node flops) between the report's min/max known FLOPs (cool = cheap, hot =
// expensive); a flops_known==false node returns a neutral "unknown" gray tint.
// Returns {active=false} (NOT a tint) whenever view().cost_heatmap is false.
struct CostTint {
  bool active = false;
  ImU32 color = 0;
};
CostTint cost_tint_for_display(App& app, int32_t display_id);

// Draw the cost/analyzer section INTO the currently-open Properties panel window
// (called from draw_properties_panel, not a standalone window). Shows: the
// selected node's FLOPs/params/weight+activation bytes; a collapse GROUP's
// per-instance and rolled-up (×N) totals; and — for the model root / always at
// the bottom — the model totals (FLOPs, params, weight MB, peak activation MB),
// the quant-coverage table (per-dtype params/bytes/%, effective bits/param, size
// vs fp32), a "shapes unresolved: k/N nodes" honesty line, and the heatmap toggle
// (App.view().cost_heatmap). Reads only App.view().cost + the current selection.
void draw_cost_section(App& app);

}  // namespace netvis
