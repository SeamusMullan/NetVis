// view/PanelHelpers.h — module-PRIVATE helpers shared by the ImGui panels.
//
// Nothing OUTSIDE view/ includes this. It exists so PropertiesPanel /
// WeightInspector / TensorTable / SearchBar / StatusBar can share small
// formatting utilities (shape/dtype/byte strings, graph-edge resolution)
// without each re-deriving them. All functions are main-thread-only (ImGui is
// single threaded) and touch only already-published, immutable ir::Model data.
#pragma once

#include <cstdint>
#include <string>

#include "core/SmallVec.h"
#include "engine/CollapseTree.h"
#include "engine/Layout.h"
#include "ir/IR.h"

namespace netvis {
namespace panel_detail {

// Shapes in the IR are SmallVec<int64_t, 6> (spec §3.2).
using Shape = SmallVec<int64_t, 6>;

// Format a shape as "[a, b, c]" (dynamic dim -1 shown as "?"), "[]" if empty.
std::string shape_string(const Shape& shape);

// Human byte size, e.g. "3.5 MB". Always 64-bit input.
std::string human_bytes(uint64_t bytes);

// Grouping-separator decimal for large counts, e.g. "12,345,678".
std::string grouped_count(int64_t n);

// Resolve a Node input/output slot Range at index `slot` to the value index it
// points at (edge_refs indirection, spec §3.2). Returns UINT32_MAX if OOB.
uint32_t resolve_edge_value(const ir::Graph& g, const ir::Range& r, uint32_t slot);

// Find the display-node index (into collapse.display_nodes()) whose leaf ir_node
// == `ir_node`, or, if that node is hidden inside a collapsed group, the display
// node for that group. Returns -1 if not currently displayed.
int32_t display_index_for_node(const CollapseTree& collapse, uint32_t ir_node);

// World-space center of the layout box for `display_id`, or {0,0} if the id is
// out of range / no layout. Hoisted here (was private to SearchBar.cpp) so
// search, navigation jump-to, and the diff panel can share one fly-to helper.
// `layout` may be null (returns {0,0}). Defined in PanelHelpers.cpp.
struct BoxCenter { float x = 0.0f; float y = 0.0f; };
BoxCenter box_center_for_display(const LayoutResult* layout, int32_t display_id);

}  // namespace panel_detail
}  // namespace netvis
