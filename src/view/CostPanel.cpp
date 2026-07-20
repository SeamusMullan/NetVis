// view/CostPanel.cpp — analyzer / cost-report view (v0.3.0 analyzer mode).
//
// DECISION (v0.3.0): NetVis v0.3.0 adds analyzer-mode overlays (FLOPs, params,
// quant profile, activation memory) for understanding model compute and memory
// footprint at a glance. This panel mirrors the GraphNav pattern: ensure_cost()
// rebuilds the CostReport keyed on (generation, current_graph, collapse_hash) so
// it recomputes lazily on reopen/subgraph-dive/expand. cost_tint_for_display()
// mirrors diff_tint_for_display() to color the canvas by log(FLOPs) when the
// heatmap toggle is on. draw_cost_section() emits ImGui widgets into the
// Properties panel (not a standalone window) showing per-node and model-wide
// cost metrics, the quant-coverage table, and the heatmap toggle.
//
// Reads ONLY published, immutable ir::Model and CostReport data; never touches
// a tensor payload (the cost engine guarantees zero payload reads). Main-thread
// only, no async work — the cost analysis is fast enough (O(V+E) arithmetic) to
// run synchronously once per graph/collapse change.
#define IMGUI_DEFINE_MATH_OPERATORS
#include "view/CostPanel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h (pulled in by
// App.h) references without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/CollapseTree.h"
#include "engine/CostModel.h"
#include "ir/IR.h"
#include "view/App.h"
#include "view/PanelHelpers.h"

namespace netvis {

namespace {

using panel_detail::grouped_count;
using panel_detail::human_bytes;

// Cost-heatmap overlay colors (cool -> neutral -> hot, log scale).
// Chosen distinct from diff/category colors; neutral gray for flops_known==false.
constexpr ImU32 kColCool = IM_COL32(100, 150, 240, 255);     // blue (cheap)
constexpr ImU32 kColNeutral = IM_COL32(150, 150, 150, 255); // gray (unknown)
constexpr ImU32 kColWarm = IM_COL32(240, 180, 60, 255);     // amber (mid)
constexpr ImU32 kColHot = IM_COL32(240, 80, 80, 255);       // red (expensive)

// Interpolate between two colors (RGBA32).
ImU32 lerp_color(ImU32 a, ImU32 b, float t) {
  if (t <= 0.0f) return a;
  if (t >= 1.0f) return b;
  auto ar = static_cast<uint8_t>((a >> IM_COL32_R_SHIFT) & 0xFF);
  auto ag = static_cast<uint8_t>((a >> IM_COL32_G_SHIFT) & 0xFF);
  auto ab = static_cast<uint8_t>((a >> IM_COL32_B_SHIFT) & 0xFF);
  auto aa = static_cast<uint8_t>((a >> IM_COL32_A_SHIFT) & 0xFF);
  auto br = static_cast<uint8_t>((b >> IM_COL32_R_SHIFT) & 0xFF);
  auto bg = static_cast<uint8_t>((b >> IM_COL32_G_SHIFT) & 0xFF);
  auto bb = static_cast<uint8_t>((b >> IM_COL32_B_SHIFT) & 0xFF);
  auto ba = static_cast<uint8_t>((b >> IM_COL32_A_SHIFT) & 0xFF);
  auto r = static_cast<uint8_t>(ar + (br - ar) * t);
  auto g = static_cast<uint8_t>(ag + (bg - ag) * t);
  auto blu = static_cast<uint8_t>(ab + (bb - ab) * t);
  auto alp = static_cast<uint8_t>(aa + (ba - aa) * t);
  return IM_COL32(r, g, blu, alp);
}

// Ramp cool -> warm -> hot by normalized t in [0, 1].
ImU32 cost_ramp(float t) {
  if (t < 0.5f) {
    return lerp_color(kColCool, kColWarm, t * 2.0f);
  }
  return lerp_color(kColWarm, kColHot, (t - 0.5f) * 2.0f);
}

// Collect IR node indices a display node stands for (mirroring GraphNav pattern).
void ir_nodes_for_display(ModelSession& s, int32_t display_id,
                          std::vector<uint32_t>& out) {
  out.clear();
  const auto& disp = s.collapse().display_nodes();
  if (display_id < 0 || static_cast<size_t>(display_id) >= disp.size()) return;
  const DisplayNode& dn = disp[static_cast<size_t>(display_id)];
  if (dn.is_group) {
    const auto& groups = s.collapse().groups();
    if (dn.group_index < groups.size())
      out = groups[dn.group_index].member_nodes;
  } else {
    out.push_back(dn.ir_node);
  }
}

// Aggregate NodeCost over a set of IR node indices (safe for out-of-range).
NodeCost sum_node_costs(const CostReport& report,
                        const std::vector<uint32_t>& nodes) {
  NodeCost sum;
  for (uint32_t ni : nodes) {
    if (ni < report.per_node.size()) {
      const NodeCost& nc = report.per_node[ni];
      sum.flops += nc.flops;
      sum.params += nc.params;
      sum.weight_bytes += nc.weight_bytes;
      sum.act_bytes += nc.act_bytes;
      // A group has flops_known=true iff at least one member does.
      if (nc.flops_known) sum.flops_known = true;
    }
  }
  return sum;
}

}  // namespace

void ensure_cost(App& app) {
  ViewState& vs = app.view();
  ModelSession& s = app.session();

  const ir::Model* model = s.model();
  // No model loaded yet — leave cost null, keys unset.
  if (model == nullptr) return;

  const uint64_t generation = s.generation();
  const uint32_t graph = s.current_graph();
  const uint64_t collapse_hash = s.collapse().collapse_hash();
  // Shape-enrichment epoch: ONNX shape inference mutates ValueInfo shapes in
  // place after publish without bumping generation, so the report must recompute
  // when this advances or FLOPs/peak stay at their pre-inference (empty) values.
  const uint64_t enrich = s.enrich_generation();

  // Check if the cache key matches (same discipline as ensure_nav).
  const bool key_match =
      vs.cost && vs.cost_key_generation == generation &&
      vs.cost_key_graph == graph && vs.cost_key_collapse == collapse_hash &&
      vs.cost_key_enrich == enrich;
  if (key_match) return;  // nothing changed — keep the cached report.

  // Record the key we are rebuilding for.
  vs.cost_key_generation = generation;
  vs.cost_key_graph = graph;
  vs.cost_key_collapse = collapse_hash;
  vs.cost_key_enrich = enrich;

  // Compute the cost report (pure headless call, no payload reads).
  CostReport report = compute_cost(*model, graph);
  vs.cost = std::make_unique<CostReport>(std::move(report));
}

CostTint cost_tint_for_display(App& app, int32_t display_id) {
  CostTint out;
  ViewState& vs = app.view();
  // Only tint when the heatmap toggle is on AND we have a valid cost report.
  if (!vs.cost_heatmap || !vs.cost) return out;

  ModelSession& s = app.session();
  const auto& disp = s.collapse().display_nodes();
  if (display_id < 0 || static_cast<size_t>(display_id) >= disp.size())
    return out;

  // Collect IR node indices this display node represents.
  std::vector<uint32_t> nodes;
  ir_nodes_for_display(s, display_id, nodes);
  if (nodes.empty()) return out;

  // Aggregate cost over those nodes.
  NodeCost nc = sum_node_costs(*vs.cost, nodes);
  if (!nc.flops_known) {
    // Unknown FLOPs -> neutral gray tint.
    out.active = true;
    out.color = kColNeutral;
    return out;
  }

  // Find min/max known FLOPs for normalization over the SAME aggregation unit as
  // the numerator: per DISPLAY node (a collapsed group's tint is the sum over its
  // members, so it must be scaled against other display nodes' sums, not against
  // individual per-node FLOPs — otherwise every multi-node group's sum exceeds the
  // largest single node and saturates to max/hot, which is the default view).
  uint64_t min_flops = UINT64_MAX, max_flops = 0;
  std::vector<uint32_t> scale_nodes;
  for (int32_t d = 0; d < static_cast<int32_t>(disp.size()); ++d) {
    ir_nodes_for_display(s, d, scale_nodes);
    if (scale_nodes.empty()) continue;
    NodeCost agg = sum_node_costs(*vs.cost, scale_nodes);
    if (!agg.flops_known || agg.flops == 0) continue;
    if (agg.flops < min_flops) min_flops = agg.flops;
    if (agg.flops > max_flops) max_flops = agg.flops;
  }
  // No known FLOPs in the whole report -> neutral.
  if (max_flops == 0 || min_flops == UINT64_MAX) {
    out.active = true;
    out.color = kColNeutral;
    return out;
  }

  // Log-scale ramp: log10(flops) between log10(min_flops) and log10(max_flops).
  // Clamp to [1, max_flops] to avoid log10(0).
  uint64_t flops = nc.flops > 0 ? nc.flops : 1;
  if (flops < min_flops) flops = min_flops;
  if (flops > max_flops) flops = max_flops;
  double log_min = std::log10(static_cast<double>(min_flops));
  double log_max = std::log10(static_cast<double>(max_flops));
  double log_flops = std::log10(static_cast<double>(flops));
  float t = 0.0f;
  if (log_max > log_min) {
    t = static_cast<float>((log_flops - log_min) / (log_max - log_min));
  }
  t = std::clamp(t, 0.0f, 1.0f);

  out.active = true;
  out.color = cost_ramp(t);
  return out;
}

void draw_cost_section(App& app) {
  ViewState& vs = app.view();
  ModelSession& s = app.session();

  const ir::Model* model = s.model();
  if (!model) return;  // no model loaded

  // Cost report may be null until ensure_cost() is called once (from App::frame).
  const CostReport* report = vs.cost.get();
  if (!report) {
    ImGui::SeparatorText("Cost / Analyzer");
    ImGui::TextDisabled("Cost report pending...");
    return;
  }

  ImGui::SeparatorText("Cost / Analyzer");

  const int32_t sel = vs.selected_display;
  const CollapseTree& collapse = s.collapse();
  const auto& display = collapse.display_nodes();

  // Helper: render one row in the cost table (label, value).
  auto kv = [](const char* label, const std::string& val) {
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine(160.0f);
    ImGui::TextUnformatted(val.c_str());
  };
  auto kv_u64 = [&kv](const char* label, uint64_t val) {
    kv(label, grouped_count(static_cast<int64_t>(val)));
  };
  auto kv_bytes = [&kv](const char* label, uint64_t bytes) {
    kv(label, human_bytes(bytes));
  };

  // --- Selected node/group cost (if a valid selection exists) -----------------
  bool has_selection = sel >= 0 && static_cast<size_t>(sel) < display.size();
  if (has_selection) {
    std::vector<uint32_t> nodes;
    ir_nodes_for_display(s, sel, nodes);
    if (!nodes.empty()) {
      NodeCost nc = sum_node_costs(*report, nodes);

      // If a collapsed group, show per-instance cost + rolled-up total.
      const DisplayNode& dn = display[static_cast<size_t>(sel)];
      if (dn.is_group) {
        const auto& groups = collapse.groups();
        if (dn.group_index < groups.size()) {
          const CollapseGroup& grp = groups[dn.group_index];
          uint32_t instances = grp.instances > 0 ? grp.instances : 1;

          // Per-instance: divide aggregate by instances (for a representative).
          NodeCost per_inst;
          per_inst.flops = nc.flops / instances;
          per_inst.params = nc.params / instances;
          per_inst.weight_bytes = nc.weight_bytes / instances;
          per_inst.act_bytes = nc.act_bytes / instances;
          per_inst.flops_known = nc.flops_known;

          ImGui::Text("Group: %s (×%u)", grp.label.c_str(), instances);
          ImGui::Separator();
          ImGui::TextUnformatted("Per instance:");
          if (per_inst.flops_known) {
            kv_u64("  FLOPs", per_inst.flops);
          } else {
            kv("  FLOPs", "(unknown)");
          }
          kv_u64("  params", per_inst.params);
          kv_bytes("  weights", per_inst.weight_bytes);
          kv_bytes("  activations", per_inst.act_bytes);

          ImGui::Separator();
          ImGui::TextUnformatted("Rolled up (all instances):");
          if (nc.flops_known) {
            kv_u64("  FLOPs", nc.flops);
          } else {
            kv("  FLOPs", "(unknown)");
          }
          kv_u64("  params", nc.params);
          kv_bytes("  weights", nc.weight_bytes);
          kv_bytes("  activations", nc.act_bytes);
        }
      } else {
        // Leaf node: show its cost directly.
        if (nc.flops_known) {
          kv_u64("FLOPs", nc.flops);
        } else {
          kv("FLOPs", "(unknown)");
        }
        kv_u64("params", nc.params);
        kv_bytes("weights", nc.weight_bytes);
        kv_bytes("activations", nc.act_bytes);
      }
      ImGui::Separator();
    }
  }

  // --- Model totals (always shown at bottom) ----------------------------------
  ImGui::SeparatorText("Model totals");
  if (report->from_graph) {
    if (report->nodes_flops_known > 0) {
      kv_u64("total FLOPs", report->total_flops);
    } else {
      kv("total FLOPs", "(none known)");
    }
  } else {
    // Table mode: no compute graph.
    ImGui::TextDisabled("(no compute graph)");
  }
  kv_u64("total params", report->total_params);
  kv_bytes("total weights", report->total_weight_bytes);
  kv_bytes("peak activations", report->peak_activation_bytes);

  // Honesty line: "FLOPs unknown for k / N nodes".
  if (report->from_graph && report->nodes_total > 0) {
    uint32_t unknown = report->nodes_total - report->nodes_flops_known;
    if (unknown > 0) {
      ImGui::TextDisabled("FLOPs unknown for %u / %u nodes", unknown,
                          report->nodes_total);
    }
  }

  // --- Quant-coverage table ---------------------------------------------------
  if (!report->dtype_usage.empty()) {
    ImGui::SeparatorText("Quantization profile");
    const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("quant", 4, flags)) {
      ImGui::TableSetupColumn("dtype", ImGuiTableColumnFlags_WidthFixed, 60.0f);
      ImGui::TableSetupColumn("params");
      ImGui::TableSetupColumn("bytes");
      ImGui::TableSetupColumn("% of total");
      ImGui::TableHeadersRow();

      uint64_t total_bytes = report->total_weight_bytes;
      for (const DTypeUsage& dtu : report->dtype_usage) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(ir::dtype_name(dtu.dtype));
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%s", grouped_count(static_cast<int64_t>(dtu.params)).c_str());
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%s", human_bytes(dtu.bytes).c_str());
        ImGui::TableSetColumnIndex(3);
        if (total_bytes > 0) {
          double pct = (static_cast<double>(dtu.bytes) /
                        static_cast<double>(total_bytes)) *
                       100.0;
          ImGui::Text("%.1f%%", pct);
        } else {
          ImGui::TextDisabled("-");
        }
      }
      ImGui::EndTable();
    }

    // Derived quant metrics.
    ImGui::Separator();
    double eff_bits = report->effective_bits_per_param();
    double vs_fp32 = report->size_vs_fp32();
    ImGui::Text("Effective bits/param: %.2f", eff_bits);
    ImGui::Text("Size vs fp32: %.2f×", vs_fp32);
  }

  // --- Heatmap toggle ---------------------------------------------------------
  ImGui::Separator();
  bool heatmap = vs.cost_heatmap;
  if (ImGui::Checkbox("Cost heatmap overlay", &heatmap)) {
    vs.cost_heatmap = heatmap;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Color nodes by log(FLOPs): cool (cheap) -> warm -> hot (expensive).");
  }
}

}  // namespace netvis
