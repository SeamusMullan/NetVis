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
#include <string_view>

#include "imgui.h"

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

// Case-insensitive ASCII substring test (locale-free; model/attr names are ASCII
// identifiers). Empty needle matches everything. Shared by the tensor-table
// filter (§8.6) and the attribute-inspector filter (#60).
bool icontains(std::string_view hay, std::string_view needle);

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

// #57 (v0.8.2): serialize one leaf node to a JSON string — op_type, name,
// attributes (typed), and input/output value shapes+dtypes. Structural only (no
// payload bytes; a Tensor attribute is emitted as its dtype+shape, not values).
// Hand-built (no nlohmann in the view hot-path headers); values are JSON-escaped.
// `gi` is the graph index the node lives in; returns "{}" if indices are OOB.
std::string node_to_json(const ir::Model& model, const ir::Graph& g,
                         const ir::Node& node);

// --- Copyable panel text (#152) ----------------------------------------------
//
// Every string a panel shows used to be an ImGui::Text, and ImGui::Text draws
// glyphs — it produces no selection and no clipboard. A user who wanted a tensor
// name, a dtype, a file path or a stat out of NetVis had to retype it by eye,
// which for a 90-character transformer weight name is where the bug reports came
// from.
//
// ONE idiom, used at every converted call site: the string draws as ordinary text
// until it is DOUBLE-CLICKED, at which point that single label is replaced in
// place by a read-only ImGui::InputText (ReadOnly | AutoSelectAll) that is
// keyboard-focused and fully selected, so Ctrl+C works immediately and a drag can
// still take a substring.
//
// WHY NOT an always-live read-only InputText (the other common ImGui recipe): an
// active InputText raises io.WantCaptureKeyboard, so every app hotkey (command
// palette, search, tab switching) would die the moment a *label* took focus.
// Gating the box behind a deliberate double-click means the keyboard is only
// captured when the user actually asked to copy something.
//
// WHY NOT Selectable + SetClipboardText: it can only ever copy the whole string,
// shows no selection to confirm what was taken, and eats the single click that
// PropertiesPanel/DiffPanel already bind to jump-to-node.
//
// The box is drawn with FramePadding (0,0), no border and a transparent frame, so
// it occupies EXACTLY the rect the text occupied; otherwise activating one label
// would grow its row by 2*FramePadding.y and shove the whole panel below it down.
// Only one label is expanded at a time (a single static id), and the id is
// dropped if the panel that owns it stops drawing.
//
// Main-thread only, like everything else here. `id` must be unique within the
// current ImGui ID stack (loops already PushID, so a plain literal is enough).
void copyable_text(const char* id, std::string_view text);
void copyable_text_disabled(const char* id, std::string_view text);
void copyable_text_colored(const char* id, const ImVec4& color, std::string_view text);
void copyable_text_fmt(const char* id, const char* fmt, ...) IM_FMTARGS(2);
void copyable_text_disabled_fmt(const char* id, const char* fmt, ...) IM_FMTARGS(2);

}  // namespace panel_detail
}  // namespace netvis
