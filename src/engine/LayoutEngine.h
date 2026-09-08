// SPDX-License-Identifier: Apache-2.0
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

// True if IR node `n` is a constant/initializer *source*: it consumes nothing,
// or its op categorizes as OpCategory::Tensor (Constant/Cast/...).
//
// #153: this predicate used to live only in the view (GraphCanvas's "hide
// constants" toggle). Layout now needs the SAME notion to pull constants down
// next to their consumers, and two independent definitions would have let the
// canvas hide a box the layout had not treated as a constant (or the reverse),
// so it is declared here — engine-side, GUI-free — and the view calls it.
bool node_is_const_source(const ir::Model& m, const ir::Node& n);

// Compute layout for the current collapse view of a graph. `progress` optional.
LayoutResult compute_layout(const ir::Model& model, uint32_t graph_index,
                            const CollapseTree& collapse, const SizeFn& size_fn,
                            const LayoutParams& params = {},
                            ProgressSink* progress = nullptr);

}  // namespace netvis
