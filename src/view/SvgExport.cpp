// view/SvgExport.cpp — #55 vector (SVG) export of the current graph.
//
// DECISION: SVG is emitted from the WORLD-SPACE layout (boxes + routed edges +
// node labels), NOT the screen framebuffer — so the export is a clean, resolution-
// independent vector of the ENTIRE graph regardless of the current pan/zoom (unlike
// export_view_png, which reads back the visible window). It reuses the same
// per-category header color the canvas draws (App::category_color) so the vector
// matches the on-screen palette. Pure over published engine state (layout +
// collapse + model); no ImGui, no payload reads.
#include <cstdio>
#include <string>

#include "engine/LayoutEngine.h"
#include "engine/OpCategory.h"
#include "engine/plugin/Registry.h"
#include "ir/IR.h"
#include "view/App.h"

namespace netvis {

namespace {

// Escape the five XML text characters so a node name can't break the SVG.
std::string xml_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&apos;"; break;
      default: out.push_back(c);
    }
  }
  return out;
}

// "#rrggbb" for a packed ImU32 (ignores alpha — SVG fills are opaque here).
std::string hex_color(ImU32 c) {
  unsigned r = (c >> IM_COL32_R_SHIFT) & 0xFF;
  unsigned g = (c >> IM_COL32_G_SHIFT) & 0xFF;
  unsigned b = (c >> IM_COL32_B_SHIFT) & 0xFF;
  char buf[8];
  std::snprintf(buf, sizeof(buf), "#%02x%02x%02x", r, g, b);
  return buf;
}

}  // namespace

void App::export_view_svg(const std::string& path) {
  ModelSession& s = session();
  const LayoutResult* layout = s.layout();
  const ir::Model* model = s.model();
  if (layout == nullptr || layout->boxes.empty() || model == nullptr) {
    add_toast("Nothing to export (no layout)", true);
    return;
  }

  const uint32_t gi = s.current_graph();
  const auto& disp = s.collapse().display_nodes();
  const bool dark = view().dark_theme;

  // World bounds -> SVG canvas with a small margin.
  const float margin = 40.0f;
  const float minx = layout->bounds_min.x - margin;
  const float miny = layout->bounds_min.y - margin;
  const float w = (layout->bounds_max.x - layout->bounds_min.x) + 2 * margin;
  const float h = (layout->bounds_max.y - layout->bounds_min.y) + 2 * margin;

  std::string svg;
  svg.reserve(64 * 1024);
  char buf[256];
  std::snprintf(buf, sizeof(buf),
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" "
                "height=\"%.0f\" viewBox=\"%.1f %.1f %.1f %.1f\">\n",
                w, h, minx, miny, w, h);
  svg += buf;
  // Background.
  svg += "<rect x=\"";
  std::snprintf(buf, sizeof(buf), "%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\"",
                minx, miny, w, h);
  svg += buf;
  svg += dark ? " fill=\"#1e2228\"/>\n" : " fill=\"#f4f6f9\"/>\n";

  const char* edge_col = dark ? "#96a0af" : "#5a6478";
  // Edges as cubic bezier paths (matches the default on-screen routing).
  for (const EdgeCurve& e : layout->edges) {
    std::snprintf(buf, sizeof(buf),
                  "<path d=\"M %.1f %.1f C %.1f %.1f %.1f %.1f %.1f %.1f\" "
                  "fill=\"none\" stroke=\"%s\" stroke-width=\"1.5\"/>\n",
                  e.p0.x, e.p0.y, e.p1.x, e.p1.y, e.p2.x, e.p2.y, e.p3.x, e.p3.y,
                  edge_col);
    svg += buf;
  }

  // Node boxes: body rect + colored header strip + op label.
  const char* body_fill = dark ? "#22262c" : "#f4f6f9";
  const char* border = dark ? "#14161a" : "#788090";
  const char* text_col = dark ? "#ebeef2" : "#181c24";
  for (const NodeBox& b : layout->boxes) {
    // Resolve the op category + label for this display node.
    OpCategory cat = OpCategory::Other;
    std::string label = "node";
    if (b.display_id < disp.size()) {
      const DisplayNode& dn = disp[b.display_id];
      if (dn.is_group) {
        const auto& groups = s.collapse().groups();
        if (dn.group_index < groups.size()) {
          label = groups[dn.group_index].label.empty() ? "group"
                                                       : groups[dn.group_index].label;
          if (gi < model->graphs.size() &&
              !groups[dn.group_index].representative_nodes.empty()) {
            uint32_t ni = groups[dn.group_index].representative_nodes.front();
            if (ni < model->graphs[gi].nodes.size())
              cat = plugin::resolve_category(*model, model->graphs[gi],
                                             model->graphs[gi].nodes[ni]);
          }
        }
      } else if (gi < model->graphs.size() &&
                 dn.ir_node < model->graphs[gi].nodes.size()) {
        const ir::Node& node = model->graphs[gi].nodes[dn.ir_node];
        std::string_view op = model->str(node.op_type);
        if (!op.empty()) label = std::string(op);
        cat = plugin::resolve_category(*model, model->graphs[gi], node);
      }
    }
    const std::string header = hex_color(App::category_color(cat, dark));
    const float hdr_h = b.size.y < 40.0f ? b.size.y * 0.4f : 16.0f;

    // Body.
    std::snprintf(buf, sizeof(buf),
                  "<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" "
                  "rx=\"4\" fill=\"%s\" stroke=\"%s\" stroke-width=\"1\"/>\n",
                  b.pos.x, b.pos.y, b.size.x, b.size.y, body_fill, border);
    svg += buf;
    // Header strip.
    std::snprintf(buf, sizeof(buf),
                  "<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" "
                  "rx=\"4\" fill=\"%s\"/>\n",
                  b.pos.x, b.pos.y, b.size.x, hdr_h, header.c_str());
    svg += buf;
    // Op label (clipped visually by the box in a viewer; kept short here).
    std::snprintf(buf, sizeof(buf),
                  "<text x=\"%.1f\" y=\"%.1f\" font-family=\"sans-serif\" "
                  "font-size=\"11\" fill=\"%s\">",
                  b.pos.x + 5.0f, b.pos.y + hdr_h + 12.0f, text_col);
    svg += buf;
    svg += xml_escape(label);
    svg += "</text>\n";
  }

  svg += "</svg>\n";

  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    add_toast("Could not write SVG", true);
    return;
  }
  const size_t wrote = std::fwrite(svg.data(), 1, svg.size(), f);
  const bool close_ok = std::fclose(f) == 0;
  if (wrote != svg.size() || !close_ok) {
    add_toast("SVG write incomplete (disk full?)", true);
    return;
  }
  add_toast(std::string("Exported ") + path, false);
}

}  // namespace netvis
