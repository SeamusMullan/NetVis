// view/DiffPanel.h — comparison / model-diff panel (v0.2.0 model diff).
//
// Module-private to view/. The panel lets the user load a SECOND model as a
// comparison, shows the added/removed/changed summary, and drives the diff-color
// overlay on the graph canvas. All heavy work (parse + diff) is done in the
// engine (DiffLoader) — this file is pure ImGui + reading published results, so
// the view never includes a parser.
#pragma once

#include <cstdint>

#include "imgui.h"

namespace netvis {

class App;  // forward; defined in view/App.h

// The diff-color overlay for a display node, or a neutral value when diff is
// inactive / the node has no diff status. GraphCanvas consults this at the
// header-color and low-LOD blob sites and overrides the category color.
struct DiffTint {
  bool active = false;   // false => use the normal category color
  ImU32 color = 0;       // added=green, removed=red, changed=amber
};

// Resolve the diff tint for a display-node id. Returns {active=false} when no
// diff is loaded, the diff was computed for a different primary graph, or the
// node index is out of the diff result's range (post push_graph/pop_graph).
DiffTint diff_tint_for_display(App& app, int32_t display_id);

// Draw the diff panel (load button, summary counts, add/remove/change lists).
// Called once per frame from App::frame().
void draw_diff_panel(App& app);

}  // namespace netvis
