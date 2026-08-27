// view/PanelHelpers.cpp — implementations of the module-private panel helpers.
#include "view/PanelHelpers.h"

#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>

namespace netvis {
namespace panel_detail {

bool icontains(std::string_view hay, std::string_view needle) {
  if (needle.empty()) return true;
  if (needle.size() > hay.size()) return false;
  auto lower = [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  };
  for (size_t i = 0; i + needle.size() <= hay.size(); ++i) {
    bool m = true;
    for (size_t j = 0; j < needle.size(); ++j) {
      if (lower(hay[i + j]) != lower(needle[j])) { m = false; break; }
    }
    if (m) return true;
  }
  return false;
}

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

namespace {

// Append `s` as a JSON string literal (quotes + minimal escaping) to `out`.
void json_escape_to(std::string& out, std::string_view s) {
  out.push_back('"');
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

// "dtype [shape]" — a value's type summary as a JSON object {"name","dtype","shape"}.
void append_value_json(std::string& out, const ir::Model& model,
                       const ir::Graph& g, uint32_t value_idx) {
  out += "{";
  if (value_idx < g.values.size()) {
    const ir::ValueInfo& vi = g.values[value_idx];
    out += "\"name\":";
    json_escape_to(out, model.str(vi.name));
    out += ",\"dtype\":";
    json_escape_to(out, ir::dtype_name(vi.dtype));
    out += ",\"shape\":";
    json_escape_to(out, shape_string(vi.shape));
  } else {
    out += "\"name\":null";
  }
  out += "}";
}

}  // namespace

std::string node_to_json(const ir::Model& model, const ir::Graph& g,
                         const ir::Node& node) {
  std::string out;
  out.reserve(256);
  out += "{\"op\":";
  json_escape_to(out, model.str(node.op_type));
  out += ",\"name\":";
  json_escape_to(out, model.str(node.name));

  // Attributes (typed). Reuse the same one-line rendering rules as the panel.
  out += ",\"attributes\":{";
  bool first = true;
  for (uint32_t a = 0; a < node.attributes.count; ++a) {
    uint32_t ai = node.attributes.begin + a;
    if (ai >= g.attributes.size()) break;
    const ir::Attribute& attr = g.attributes[ai];
    if (!first) out += ",";
    first = false;
    json_escape_to(out, model.str(attr.name));
    out += ":";
    const ir::AttrValue& v = attr.value;
    char buf[64];
    switch (v.kind) {
      case ir::AttrValue::Kind::None: out += "null"; break;
      case ir::AttrValue::Kind::Int:
        out += std::to_string(v.i);
        break;
      case ir::AttrValue::Kind::Float:
        // JSON has no inf/nan literal; emit null for non-finite so the output
        // stays parseable.
        if (std::isfinite(v.f)) {
          std::snprintf(buf, sizeof(buf), "%g", v.f);
          out += buf;
        } else {
          out += "null";
        }
        break;
      case ir::AttrValue::Kind::String:
        json_escape_to(out, model.str(v.s));
        break;
      case ir::AttrValue::Kind::Ints: {
        out += "[";
        for (size_t i = 0; i < v.ints.size(); ++i) {
          if (i) out += ",";
          out += std::to_string(v.ints[i]);
        }
        out += "]";
        break;
      }
      case ir::AttrValue::Kind::Floats: {
        out += "[";
        for (size_t i = 0; i < v.floats.size(); ++i) {
          if (i) out += ",";
          if (std::isfinite(v.floats[i])) {
            std::snprintf(buf, sizeof(buf), "%g", v.floats[i]);
            out += buf;
          } else {
            out += "null";
          }
        }
        out += "]";
        break;
      }
      case ir::AttrValue::Kind::Strings: {
        out += "[";
        for (size_t i = 0; i < v.strings.size(); ++i) {
          if (i) out += ",";
          json_escape_to(out, model.str(v.strings[i]));
        }
        out += "]";
        break;
      }
      case ir::AttrValue::Kind::Tensor: {
        out += "{\"dtype\":";
        json_escape_to(out, ir::dtype_name(v.tensor.dtype));
        out += ",\"shape\":";
        json_escape_to(out, shape_string(v.tensor.shape));
        out += "}";
        break;
      }
      case ir::AttrValue::Kind::Graph:
        out += "\"<subgraph>\"";
        break;
    }
  }
  out += "}";

  // Input / output value type summaries (structural — no payload reads).
  out += ",\"inputs\":[";
  for (uint32_t s = 0; s < node.inputs.count; ++s) {
    if (s) out += ",";
    append_value_json(out, model, g, resolve_edge_value(g, node.inputs, s));
  }
  out += "],\"outputs\":[";
  for (uint32_t s = 0; s < node.outputs.count; ++s) {
    if (s) out += ",";
    append_value_json(out, model, g, resolve_edge_value(g, node.outputs, s));
  }
  out += "]}";
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

// --- Copyable panel text (#152) ----------------------------------------------
// Rationale for the idiom lives on the declarations in PanelHelpers.h.
namespace {

// At most ONE label is expanded into a read-only box at a time, so a single
// static id is all the state this needs (ImGui, and therefore every panel, is
// main-thread only).
ImGuiID g_copy_id = 0;
// Frame the expanded label was last drawn on. Without it, closing the panel that
// owns the open box would leave g_copy_id set forever, and the next panel that
// happened to hash to the same ImGui id would open as a text box on its own.
int g_copy_seen_frame = -1;
bool g_copy_focus_pending = false;
// InputText wants a writable char* even under ReadOnly; refill it every frame so
// it always mirrors the caller's (possibly changing) string.
std::string g_copy_buf;

void copyable_impl(const char* id, std::string_view text, const ImVec4* color) {
  const int frame = ImGui::GetFrameCount();
  if (g_copy_id != 0 && frame - g_copy_seen_frame > 1) g_copy_id = 0;

  // One buffer feeds both GetID and InputText's label so the two can never
  // disagree; "##" keeps InputText from drawing the id as a visible label.
  char label[192];
  std::snprintf(label, sizeof(label), "##cp_%s", id);
  const ImGuiID this_id = ImGui::GetID(label);

  if (this_id == g_copy_id) {
    g_copy_seen_frame = frame;
    // assign(nullptr, 0) on a default-constructed string_view is UB; several
    // callers pass ir::Model::str() results that can be exactly that.
    if (text.empty())
      g_copy_buf.clear();
    else
      g_copy_buf.assign(text);
    // Zero padding, no border, transparent frame: the box lands on EXACTLY the
    // rect the text had, so expanding a label never reflows the panel below it.
    // The select-all highlight is the only thing that changes, which doubles as
    // the "this is now selectable" feedback.
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(0, 0, 0, 0));
    if (color != nullptr) ImGui::PushStyleColor(ImGuiCol_Text, *color);
    // +2px so the caret past the last glyph is not clipped. Anything larger and
    // an auto-fit table column visibly widens while the box is open.
    ImGui::SetNextItemWidth(ImGui::CalcTextSize(g_copy_buf.c_str()).x + 2.0f);
    if (g_copy_focus_pending) {
      ImGui::SetKeyboardFocusHere();
      g_copy_focus_pending = false;
    }
    ImGui::InputText(label, g_copy_buf.data(), g_copy_buf.size() + 1,
                     ImGuiInputTextFlags_ReadOnly |
                         ImGuiInputTextFlags_AutoSelectAll);
    // Clicking away / Escape collapses it back to plain text.
    if (ImGui::IsItemDeactivated()) g_copy_id = 0;
    if (color != nullptr) ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);
    return;
  }

  if (color != nullptr) ImGui::PushStyleColor(ImGuiCol_Text, *color);
  // An empty string_view's data() may be null, and TextUnformatted reads a null
  // text_end as "call strlen" — which would then strlen(nullptr).
  if (text.empty())
    ImGui::TextUnformatted("");
  else
    ImGui::TextUnformatted(text.data(), text.data() + text.size());
  if (color != nullptr) ImGui::PopStyleColor();
  if (ImGui::IsItemHovered()) {
    // The I-beam cursor is the whole discoverability story: a tooltip on every
    // label in a panel this dense would be unreadable noise.
    ImGui::SetMouseCursor(ImGuiMouseCursor_TextInput);
    if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      g_copy_id = this_id;
      g_copy_seen_frame = frame;
      g_copy_focus_pending = true;
    }
  }
}

// Shared vsnprintf tail for the _fmt entry points. The scratch string is static
// (main-thread only) so a panel full of formatted labels does not allocate once
// per label per frame.
const std::string& format_scratch(const char* fmt, va_list args) {
  static std::string scratch;
  va_list copy;
  va_copy(copy, args);
  const int n = std::vsnprintf(nullptr, 0, fmt, copy);
  va_end(copy);
  scratch.clear();
  if (n > 0) {
    scratch.resize(static_cast<size_t>(n));
    std::vsnprintf(scratch.data(), static_cast<size_t>(n) + 1, fmt, args);
  }
  return scratch;
}

}  // namespace

void copyable_text(const char* id, std::string_view text) {
  copyable_impl(id, text, nullptr);
}

void copyable_text_disabled(const char* id, std::string_view text) {
  const ImVec4 col = ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
  copyable_impl(id, text, &col);
}

void copyable_text_colored(const char* id, const ImVec4& color,
                           std::string_view text) {
  copyable_impl(id, text, &color);
}

void copyable_text_fmt(const char* id, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  copyable_impl(id, format_scratch(fmt, args), nullptr);
  va_end(args);
}

void copyable_text_disabled_fmt(const char* id, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  const ImVec4 col = ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
  copyable_impl(id, format_scratch(fmt, args), &col);
  va_end(args);
}

}  // namespace panel_detail
}  // namespace netvis
