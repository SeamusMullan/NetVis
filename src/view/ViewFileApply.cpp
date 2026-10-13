// SPDX-License-Identifier: Apache-2.0
// view/ViewFileApply.cpp — see ViewFileApply.h.

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h references
// without including; it must precede view/App.h for that to compile.
#include "engine/LayoutEngine.h"

#include "view/ViewFileApply.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "engine/ModelSession.h"
#include "view/App.h"
#include "view/GraphNav.h"
#include "view/PanelHelpers.h"
#include "view/ViewHistory.h"

namespace netvis {

ViewFileApplier make_view_file_applier(const ViewFile& f, std::vector<std::string> notes,
                                       const ModelSession& s) {
  ViewFileApplier a;
  a.file = f;
  a.notes = std::move(notes);
  a.generation = s.generation();
  a.has_model = s.model() != nullptr;
  a.same_model = a.has_model && same_model_path(f.model, s.path(), f.source_dir);
  return a;
}

void apply_view_file_agnostic(const ViewFile& f, ViewState& vs) {
  if (f.pan_x) vs.cam.pan.x = *f.pan_x;
  if (f.pan_y) vs.cam.pan.y = *f.pan_y;
  if (f.zoom) vs.cam.zoom = std::clamp(*f.zoom, kViewFileMinZoom, kViewFileMaxZoom);
  // A fly-to in flight would overwrite the pose every frame.
  if (f.has_camera()) vs.animating = false;

  if (f.hide_const_edges) vs.hide_const_edges = *f.hide_const_edges;
  if (f.show_layer_bands) vs.show_layer_bands = *f.show_layer_bands;
  if (f.show_critical_path) vs.show_critical_path = *f.show_critical_path;
  if (f.cost_heatmap) vs.cost_heatmap = *f.cost_heatmap;
  if (f.show_search_results) vs.show_search_results = *f.show_search_results;
  if (f.diff_panel_open) vs.diff_panel_open = *f.diff_panel_open;
  if (f.edge_routing && *f.edge_routing >= 0 && *f.edge_routing <= 2)
    vs.edge_routing = *f.edge_routing;
  if (f.heatmap_metric) vs.heatmap_metric = *f.heatmap_metric;
  if (f.heatmap_log_scale) vs.heatmap_log_scale = *f.heatmap_log_scale;
  if (f.search_query) vs.search_query = *f.search_query;
  if (f.attr_filter) vs.attr_filter = *f.attr_filter;
  if (f.table_filter) vs.table_filter = *f.table_filter;

  // Navigation intent. The state is created on demand, as the #56 loader did, so a
  // category mask in the file lands even on a tab that never opened the nav UI.
  if (!vs.nav) vs.nav = std::make_unique<GraphNavState>();
  GraphNavState& nav = *vs.nav;
  if (f.nav_mode && *f.nav_mode <= 2) nav.mode = static_cast<NavMode>(*f.nav_mode);
  if (f.nav_hops) nav.hops = *f.nav_hops;
  if (f.follow_preds) nav.follow_preds = *f.follow_preds;
  if (f.follow_succs) nav.follow_succs = *f.follow_succs;
  if (f.category_mask) nav.category_mask = *f.category_mask;
  if (f.filter_active) nav.filter_active = *f.filter_active;
}

ViewStep step_view_file(ViewFileApplier& a, ViewState& vs, ModelSession& s, bool jobs_idle) {
  using Phase = ViewFileApplier::Phase;
  for (;;) {
    if (a.phase == Phase::Done) return ViewStep::Done;
    // The model was reopened since this load began: every index in the file would
    // now denote something else. Stop; do not half-apply.
    if (s.generation() != a.generation) return ViewStep::Aborted;

    switch (a.phase) {
      case Phase::Agnostic: {
        apply_view_file_agnostic(a.file, vs);
        if (!a.has_model) {
          a.outcome = ViewApplyOutcome::NoModel;
          a.phase = Phase::Done;
          return ViewStep::Done;
        }
        a.phase = Phase::Dive;
        break;
      }

      case Phase::Dive: {
        // A subgraph dive is model-specific, and queues a layout.
        if (!(a.same_model && a.file.graph)) {
          a.phase = Phase::Specific;
          break;
        }
        if (!jobs_idle) return ViewStep::Pending;
        const ir::Model* m = s.model();
        const uint32_t g = *a.file.graph;
        if (m == nullptr || g >= m->graphs.size()) {
          a.notes.push_back("graph " + std::to_string(g) + " is out of range (the model has " +
                            std::to_string(m ? m->graphs.size() : 0) + " graphs); ignored");
          a.phase = Phase::Specific;
          break;
        }
        a.phase = Phase::Specific;
        if (g != s.current_graph()) {
          s.push_graph(g);
          return ViewStep::Pending;  // the dive queued a layout: wait for it
        }
        break;
      }

      case Phase::Specific: {
        // The collapse change (one re-layout) and the selection both run against a
        // quiescent session, so no worker reads the collapse state under us.
        if (!jobs_idle) return ViewStep::Pending;
        const ir::Model* m = s.model();
        uint32_t nodes = 0, values = 0;
        if (m != nullptr && s.current_graph() < m->graphs.size()) {
          const ir::Graph& g = m->graphs[s.current_graph()];
          nodes = static_cast<uint32_t>(std::min<size_t>(g.nodes.size(), UINT32_MAX));
          values = static_cast<uint32_t>(std::min<size_t>(g.values.size(), UINT32_MAX));
        }
        const ViewSnapshot live = capture_view(vs, s);
        ViewOverlayResult r = overlay_view_file(a.file, live, a.same_model, nodes, values);
        for (std::string& n : r.notes) a.notes.push_back(std::move(n));
        // apply_view does the hard ordering (collapse before selection, at most ONE
        // re-layout, nav ownership claim, hover cleared). It refuses only when the
        // (generation, graph) moved or the bitset size does not match, neither of
        // which can happen on a snapshot overlaid onto a capture taken just above, but
        // if it ever does, that is a failure, not a "View loaded".
        if (!apply_view(r.snapshot, vs, s)) {
          a.outcome = ViewApplyOutcome::Failed;
          a.phase = Phase::Done;
          return ViewStep::Done;
        }
        // The filter text came from the file, but the Properties panel keeps an
        // attribute filter only for the node it was typed against and clears it the
        // first time it draws a different one (a fresh key is UINT64_MAX). Tie the
        // restored filter to the restored selection, or it would be dropped on the
        // very next frame.
        if (a.file.attr_filter && vs.selected_display >= 0) {
          const auto& display = s.collapse().display_nodes();
          const size_t sel = static_cast<size_t>(vs.selected_display);
          if (sel < display.size() && !display[sel].is_group &&
              display[sel].ir_node != UINT32_MAX)
            vs.attr_filter_key =
                panel_detail::attr_filter_key(s.current_graph(), display[sel].ir_node);
        }
        a.outcome = a.same_model ? ViewApplyOutcome::Full : ViewApplyOutcome::OtherModel;
        a.phase = Phase::Done;
        return ViewStep::Done;
      }

      case Phase::Done:
        return ViewStep::Done;
    }
  }
}

}  // namespace netvis
