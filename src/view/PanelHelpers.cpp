// view/PanelHelpers.cpp — implementations of the module-private panel helpers.
#include "view/PanelHelpers.h"

#include <cstdio>

namespace netvis {
namespace panel_detail {

std::string shape_string(const Shape& shape) {
  if (shape.empty()) return "[]";
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    int64_t d = shape[i];
    if (d < 0)
      out += "?";  // dynamic/unresolved dim
    else
      out += std::to_string(d);
  }
  out += "]";
  return out;
}

std::string human_bytes(uint64_t bytes) {
  // 64-bit throughout: model tensors routinely exceed 4 GB (spec §2.1).
  static const char* units[] = {"B", "KB", "MB", "GB", "TB", "PB"};
  double v = static_cast<double>(bytes);
  int u = 0;
  while (v >= 1024.0 && u < 5) {
    v /= 1024.0;
    ++u;
  }
  char buf[64];
  if (u == 0)
    std::snprintf(buf, sizeof(buf), "%llu B",
                  static_cast<unsigned long long>(bytes));
  else
    std::snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
  return buf;
}

std::string grouped_count(int64_t n) {
  std::string digits = std::to_string(n < 0 ? -n : n);
  std::string out;
  int c = 0;
  for (size_t i = digits.size(); i-- > 0;) {
    out.push_back(digits[i]);
    if (++c % 3 == 0 && i != 0) out.push_back(',');
  }
  if (n < 0) out.push_back('-');
  // We built it reversed; flip back.
  std::string rev(out.rbegin(), out.rend());
  return rev;
}

uint32_t resolve_edge_value(const ir::Graph& g, const ir::Range& r, uint32_t slot) {
  if (slot >= r.count) return UINT32_MAX;
  uint32_t idx = r.begin + slot;
  if (idx >= g.edge_refs.size()) return UINT32_MAX;
  uint32_t value_idx = g.edge_refs[idx];
  if (value_idx >= g.values.size()) return UINT32_MAX;
  return value_idx;
}

BoxCenter box_center_for_display(const LayoutResult* layout, int32_t display_id) {
  // Boxes carry their own display_id, so match on that rather than trusting index
  // parity (the layout box array is not guaranteed 1:1 with the display list).
  BoxCenter out;
  if (layout == nullptr || display_id < 0) return out;
  for (const NodeBox& b : layout->boxes) {
    if (static_cast<int32_t>(b.display_id) == display_id) {
      out.x = b.pos.x + b.size.x * 0.5f;
      out.y = b.pos.y + b.size.y * 0.5f;
      return out;
    }
  }
  return out;
}

int32_t display_index_for_node(const CollapseTree& collapse, uint32_t ir_node) {
  const auto& display = collapse.display_nodes();
  // First pass: a directly-visible leaf. Linear scan is fine — display lists are
  // bounded by the collapsed view size, not the full 100k-node graph (spec §7.1).
  for (size_t i = 0; i < display.size(); ++i) {
    const DisplayNode& dn = display[i];
    if (!dn.is_group && dn.ir_node == ir_node) return static_cast<int32_t>(i);
  }
  // Second pass: the node may be a member of a currently-collapsed group; jump to
  // the group instead so selection always lands on something on screen.
  const auto& groups = collapse.groups();
  for (size_t i = 0; i < display.size(); ++i) {
    const DisplayNode& dn = display[i];
    if (!dn.is_group || dn.group_index >= groups.size()) continue;
    for (uint32_t member : groups[dn.group_index].member_nodes) {
      if (member == ir_node) return static_cast<int32_t>(i);
    }
  }
  return -1;
}

}  // namespace panel_detail
}  // namespace netvis
