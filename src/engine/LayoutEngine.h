// engine/LayoutEngine.h — computes a LayoutResult for a collapse view.
//
// DECISION (spec §7.2): pure function of (graph, collapse view, node sizes) ->
// LayoutResult. No GUI dependency: node sizes are passed in (measured by the
// view from font metrics) so layout runs headless on a worker thread and in
// tests. Deterministic: identical inputs -> identical positions (spec §2.7).
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "core/JobSystem.h"
#include "engine/CollapseTree.h"
#include "engine/Layout.h"
#include "ir/IR.h"

namespace netvis {

// Measures a display node's box size. Provided by the view (font metrics); a
// headless default (fixed size from label length) is used in tests.
using SizeFn = std::function<Vec2(const DisplayNode&)>;

struct LayoutParams {
  float rank_sep = 60.0f;   // vertical gap between layers (Netron-like flow)
  float node_sep = 30.0f;   // horizontal gap between nodes in a layer
  int barycenter_sweeps = 4;// down + up sweeps budget (spec §7.2.3)
};

// Compute layout for the current collapse view of a graph. `progress` optional.
LayoutResult compute_layout(const ir::Model& model, uint32_t graph_index,
                            const CollapseTree& collapse, const SizeFn& size_fn,
                            const LayoutParams& params = {},
                            ProgressSink* progress = nullptr);

}  // namespace netvis
