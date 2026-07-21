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

// Neutral gray for flops_known==false nodes (independent of the chosen gradient).
constexpr ImU32 kColNeutral = IM_COL32(150, 150, 150, 255);

// Ramp by normalized t in [0,1] using the user's active gradient (engine-side
// pure sampler; the view only maps the result to ImU32).
ImU32 cost_ramp(const HeatmapGradient& gradient, float t) {
  return rgba8_to_imu32(gradient_sample(gradient, t));
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
      // Activation-read bytes must roll up too, else a collapsed group's
      // arithmetic intensity (flops / bytes_moved) is computed against a
      // too-small denominator and reads artificially compute-bound.
      sum.input_act_bytes += nc.input_act_bytes;
      // A group has flops_known=true iff at least one member does.
      if (nc.flops_known) sum.flops_known = true;
    }
  }
  return sum;
}

// Compact FLOPs label for the legend, e.g. "3.2 GF", "812 MF". Deterministic.
std::string human_flops(uint64_t f) {
  const char* unit = "F";
  double v = static_cast<double>(f);
  if (f >= 1000000000000ULL) { v = f / 1e12; unit = "TF"; }
  else if (f >= 1000000000ULL) { v = f / 1e9; unit = "GF"; }
  else if (f >= 1000000ULL) { v = f / 1e6; unit = "MF"; }
  else if (f >= 1000ULL) { v = f / 1e3; unit = "KF"; }
  char buf[48];
  if (unit[0] == 'F')
    std::snprintf(buf, sizeof(buf), "%llu F", static_cast<unsigned long long>(f));
  else
    std::snprintf(buf, sizeof(buf), "%.1f %s", v, unit);
  return buf;
}

// Format a metric value for the legend, keyed on which metric it is. FLOPs use
// the F/KF/MF/GF ladder; Params are a grouped count; ActBytes are human bytes;
// ArithIntensity is the fixed-point value divided back out to FLOP/byte. Labeling
// bytes as "MF" (or intensity as a count) would be a category error, so each
// metric formats in its own unit.
std::string format_metric_value(double v, HeatmapMetric m) {
  char buf[64];
  switch (m) {
    case HeatmapMetric::Flops:
      return human_flops(static_cast<uint64_t>(v));
    case HeatmapMetric::Params:
      return grouped_count(static_cast<int64_t>(v)) + " params";
    case HeatmapMetric::ActBytes:
      return human_bytes(static_cast<uint64_t>(v));
    case HeatmapMetric::ArithIntensity:
      std::snprintf(buf, sizeof(buf), "%.2f F/B",
                    v / static_cast<double>(kArithIntensityScale));
      return buf;
  }
  return human_flops(static_cast<uint64_t>(v));
}

// Human label for a roofline preset (the ridge presets are datasheet estimates,
// not measurements — the panel labels the whole roofline "approximate").
const char* roofline_preset_name(RooflinePreset p) {
  switch (p) {
    case RooflinePreset::Generic:    return "Generic (~40)";
    case RooflinePreset::CpuServer:  return "CPU server (~8)";
    case RooflinePreset::GpuFp32:    return "GPU fp32 (~13)";
    case RooflinePreset::GpuTensor:  return "GPU tensor (~200)";
    case RooflinePreset::MobileNpu:  return "Mobile NPU (~30)";
  }
  return "Generic (~40)";
}
constexpr int kRooflinePresetCount = 5;

}  // namespace

// Forward decls for helpers defined later in this TU.
static std::string cost_summary_text(App& app, const CostReport& report);
static void recompute_heatmap_range(App& app);

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
  if (!key_match) {
    // Record the key we are rebuilding for.
    vs.cost_key_generation = generation;
    vs.cost_key_graph = graph;
    vs.cost_key_collapse = collapse_hash;
    vs.cost_key_enrich = enrich;

    // Compute the cost report (pure headless call, no payload reads).
    CostReport report = compute_cost(*model, graph);
    vs.cost = std::make_unique<CostReport>(std::move(report));
  }

  // Refresh the cached heatmap range once per frame (cheap; O(display nodes)) so
  // the per-node tint and the legend read a scalar cache instead of re-scanning.
  recompute_heatmap_range(app);
}

// Normalize a metric value into [0,1] across [min,max] using the chosen scale.
// Metric-agnostic: the selected heatmap metric (FLOPs/Params/ActBytes/AI) is
// already reduced to a scalar by metric_value before this is called.
// log_scale: interpolate in log10 space (clamped to >=1 to avoid log10(0)), the
// default — magnitudes span orders of magnitude. Otherwise linear. Both guard the
// degenerate min==max case (single distinct value) to 0.
float normalize_metric_value(double value, double min_value, double max_value,
                             bool log_scale) {
  if (value < min_value) value = min_value;
  if (value > max_value) value = max_value;
  float t = 0.0f;
  if (log_scale) {
    double lmin = std::log10(min_value < 1.0 ? 1.0 : min_value);
    double lmax = std::log10(max_value < 1.0 ? 1.0 : max_value);
    double lf = std::log10(value < 1.0 ? 1.0 : value);
    if (lmax > lmin) t = static_cast<float>((lf - lmin) / (lmax - lmin));
  } else {
    if (max_value > min_value) {
      t = static_cast<float>((value - min_value) / (max_value - min_value));
    }
  }
  return std::clamp(t, 0.0f, 1.0f);
}

// Scan display nodes to (re)compute the heatmap FLOPs range, writing the cached
// scalars on ViewState. Called once per frame from ensure_cost — NOT per node —
// so cost_tint_for_display and the legend read the cache instead of each
// re-scanning + heap-allocating. Scale is over the SAME aggregation unit the tint
// uses: per DISPLAY node (a collapsed group's tint is the sum over its members, so
// it must be scaled against other display nodes' sums — scaling a group sum
// against individual per-node FLOPs saturated every group to max/hot, the v0.3.1
// bug).
static void recompute_heatmap_range(App& app) {
  ViewState& vs = app.view();
  vs.heatmap_range_valid = false;
  vs.heatmap_range_min = 0;
  vs.heatmap_range_max = 0;
  if (!vs.cost) return;
  ModelSession& s = app.session();
  const auto& disp = s.collapse().display_nodes();
  double min_value = 0.0, max_value = 0.0;
  bool any = false;
  std::vector<uint32_t> scale_nodes;
  for (int32_t d = 0; d < static_cast<int32_t>(disp.size()); ++d) {
    ir_nodes_for_display(s, d, scale_nodes);
    if (scale_nodes.empty()) continue;
    NodeCost agg = sum_node_costs(*vs.cost, scale_nodes);
    MetricValue mv = metric_value(agg, vs.heatmap_metric);
    // Include KNOWN values even when 0: for Params/ActBytes a 0 is a real value
    // that must tint at the cold end, not render gray (the honest-unknown color).
    // Skipping known-0 made an all-zero graph (e.g. a Params heatmap over a graph
    // with no initializers) produce an invalid range -> every node gray, which
    // reads as "unknown" for a value that is known to be zero. (For the FLOPs
    // metric flops_known implies flops>0, so this changes nothing there.)
    if (!mv.known) continue;
    double v = static_cast<double>(mv.value);
    if (!any || v < min_value) min_value = v;
    if (!any || v > max_value) max_value = v;
    any = true;
  }
  if (!any) return;  // no display node yields a known metric value
  vs.heatmap_range_valid = true;
  vs.heatmap_range_min = min_value;
  vs.heatmap_range_max = max_value;
}

HeatmapRange heatmap_range(App& app) {
  ViewState& vs = app.view();
  HeatmapRange r;
  r.valid = vs.heatmap_range_valid;
  r.min_value = vs.heatmap_range_min;
  r.max_value = vs.heatmap_range_max;
  return r;
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

  // Aggregate cost over those nodes, then extract the SELECTED metric — the gray
  // gate keys on that metric's known-ness, not always flops_known (e.g. an
  // unresolved ActBytes node is honestly gray even if its FLOPs are known).
  NodeCost nc = sum_node_costs(*vs.cost, nodes);
  MetricValue mv = metric_value(nc, vs.heatmap_metric);
  if (!mv.known) {
    // Honest unknown for the selected metric -> neutral gray tint.
    out.active = true;
    out.color = kColNeutral;
    return out;
  }

  HeatmapRange range = heatmap_range(app);
  if (!range.valid) {
    out.active = true;
    out.color = kColNeutral;
    return out;
  }

  float t = normalize_metric_value(static_cast<double>(mv.value),
                                   range.min_value, range.max_value,
                                   vs.heatmap_log_scale);
  out.active = true;
  out.color = cost_ramp(vs.heatmap_gradient, t);
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

  // --- Activation memory timeline (#28) ---------------------------------------
  // The full liveness curve behind peak_activation_bytes: resident activation
  // bytes at each node's high-water point in exec (topological) order. Graph-mode
  // only; skipped when the curve is empty (table mode / no nodes). Recomputed per
  // frame — cheap O(V+E) for modest graphs; huge graphs are DOWNSAMPLED for the
  // plotted array only (the true peak + peak-node index still come from the full
  // curve). max(curve) == peak_activation_bytes by construction (engine invariant).
  if (report->from_graph) {
    std::vector<uint64_t> curve =
        activation_liveness_curve(*model, s.current_graph());
    if (!curve.empty()) {
      ImGui::SeparatorText("Activation memory timeline");

      // True peak + its node index from the FULL curve (before any downsample).
      uint64_t peak_bytes = 0;
      size_t peak_node = 0;
      for (size_t i = 0; i < curve.size(); ++i) {
        if (curve[i] > peak_bytes) {
          peak_bytes = curve[i];
          peak_node = i;
        }
      }
      const double kMB = 1024.0 * 1024.0;
      auto to_mb = [kMB](uint64_t bytes) -> float {
        return static_cast<float>(static_cast<double>(bytes) / kMB);
      };

      // Build the plotted float array (in MB). For very large graphs, downsample
      // to ~kMaxPlotPoints buckets, taking the MAX live-bytes within each bucket so
      // the visual peak is preserved (never averaged away). Small graphs plot 1:1.
      constexpr size_t kDownsampleThreshold = 20000;
      constexpr size_t kMaxPlotPoints = 2000;
      std::vector<float> plot;
      bool downsampled = false;
      if (curve.size() > kDownsampleThreshold) {
        downsampled = true;
        plot.reserve(kMaxPlotPoints);
        // Ceil-divide the node count into kMaxPlotPoints buckets; each plotted
        // point is the bucket's max (guards against hiding the peak).
        const size_t bucket = (curve.size() + kMaxPlotPoints - 1) / kMaxPlotPoints;
        for (size_t start = 0; start < curve.size(); start += bucket) {
          uint64_t bmax = 0;
          for (size_t j = start; j < start + bucket && j < curve.size(); ++j) {
            if (curve[j] > bmax) bmax = curve[j];
          }
          plot.push_back(to_mb(bmax));
        }
      } else {
        plot.reserve(curve.size());
        for (uint64_t v : curve) plot.push_back(to_mb(v));
      }

      const float peak_mb = to_mb(peak_bytes);
      // scale_max = peak so the curve fills the plot height; a hair of headroom
      // keeps the peak from clipping exactly at the top edge.
      const float scale_max = (peak_mb > 0.0f) ? peak_mb * 1.02f : 1.0f;
      ImGui::PlotLines("##act_liveness", plot.data(),
                       static_cast<int>(plot.size()), 0, nullptr, 0.0f,
                       scale_max, ImVec2(0.0f, 60.0f));

      ImGui::Text("peak %s (%.2f MB) at node %zu of %zu",
                  human_bytes(peak_bytes).c_str(), peak_mb, peak_node,
                  curve.size());
      // Estimate note: pure over shapes, unresolved shapes contribute 0.
      ImGui::SameLine();
      ImGui::TextDisabled("(?)");
      if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Estimate from ValueInfo shapes/dtypes (no tensor payloads read).\n"
            "Live activation bytes at each node in execution order; unresolved\n"
            "shapes contribute 0. Curve max equals the reported peak activations.");
      }
      if (downsampled) {
        ImGui::TextDisabled("(plot downsampled to %d buckets; peak is exact)",
                            static_cast<int>(plot.size()));
      }
    }
  }

  // --- Efficiency / roofline (approximate) ------------------------------------
  if (report->from_graph && report->nodes_flops_known > 0) {
    ImGui::SeparatorText("Efficiency (approximate)");
    ImGui::Text("Arithmetic intensity: %.2f FLOP/byte",
                report->overall_arithmetic_intensity());
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip(
          "Total known FLOPs / bytes moved (weight + activation reads + writes).");

    // Roofline verdict at a user-selectable machine-balance preset. Recomputed at
    // draw time (cheap O(V)); the stored CostReport::roofline is the default-ridge
    // snapshot. The preset need not persist — a local static is enough.
    static RooflinePreset roof_preset = RooflinePreset::Generic;
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::BeginCombo("Machine balance",
                          roofline_preset_name(roof_preset))) {
      for (int i = 0; i < kRooflinePresetCount; ++i) {
        auto p = static_cast<RooflinePreset>(i);
        bool is_sel = (roof_preset == p);
        if (ImGui::Selectable(roofline_preset_name(p), is_sel)) roof_preset = p;
        if (is_sel) ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
    RooflineSummary roof =
        compute_roofline(*report, ridge_flop_per_byte(roof_preset));
    double cbf = roof.compute_bound_fraction() * 100.0;
    ImGui::Text("Roofline: %.0f%% compute-bound (%u nodes), %.0f%% memory-bound "
                "(%u nodes)",
                cbf, roof.compute_bound_nodes, 100.0 - cbf,
                roof.memory_bound_nodes);
    if (ImGui::IsItemHovered())
      ImGui::SetTooltip(
          "Approximate: node intensity vs the ridge FLOP/byte of the selected "
          "machine balance (datasheet estimates, not measurements).");
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

  // --- Copy-to-clipboard ------------------------------------------------------
  ImGui::Separator();
  if (ImGui::Button("Copy cost summary")) {
    ImGui::SetClipboardText(cost_summary_text(app, *report).c_str());
  }
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Copy model totals + quant table as tab-separated text.");

  // --- Heatmap toggle + gradient controls -------------------------------------
  ImGui::Separator();
  bool dirty = false;  // any pref changed this frame -> persist
  bool heatmap = vs.cost_heatmap;
  if (ImGui::Checkbox("Cost heatmap overlay", &heatmap)) {
    vs.cost_heatmap = heatmap;
    dirty = true;
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Color nodes by FLOPs using the gradient below (cheap -> expensive).");
  }

  if (vs.cost_heatmap) {
    HeatmapGradient& grad = vs.heatmap_gradient;

    // Metric selector: which scalar the gradient ramps across (FLOPs / Params /
    // Act bytes / Arith intensity). Changing it invalidates the cached range,
    // which self-heals next frame via recompute_heatmap_range in ensure_cost.
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::BeginCombo("Metric",
                          heatmap_metric_name(vs.heatmap_metric))) {
      for (int i = 0; i < kHeatmapMetricCount; ++i) {
        auto m = static_cast<HeatmapMetric>(i);
        bool is_sel = (vs.heatmap_metric == m);
        if (ImGui::Selectable(heatmap_metric_name(m), is_sel)) {
          vs.heatmap_metric = m;
          dirty = true;
        }
        if (is_sel) ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }

    // Preset dropdown.
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::BeginCombo("Gradient",
                          gradient_preset_name(grad.preset))) {
      for (int i = 0; i < kGradientPresetCount; ++i) {
        auto p = static_cast<GradientPreset>(i);
        bool is_sel = (grad.preset == p);
        if (ImGui::Selectable(gradient_preset_name(p), is_sel)) {
          gradient_set_preset(grad, p);
          dirty = true;
        }
        if (is_sel) ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }

    // Live preview bar (samples the active gradient across its width).
    {
      ImDrawList* dl = ImGui::GetWindowDrawList();
      ImVec2 p0 = ImGui::GetCursorScreenPos();
      const float bar_w = 220.0f, bar_h = 14.0f;
      const int segs = 48;
      for (int i = 0; i < segs; ++i) {
        float t0 = static_cast<float>(i) / segs;
        float t1 = static_cast<float>(i + 1) / segs;
        ImU32 c0 = rgba8_to_imu32(gradient_sample(grad, t0));
        ImU32 c1 = rgba8_to_imu32(gradient_sample(grad, t1));
        ImVec2 s0(p0.x + bar_w * t0, p0.y);
        ImVec2 s1(p0.x + bar_w * t1, p0.y + bar_h);
        dl->AddRectFilledMultiColor(s0, s1, c0, c1, c1, c0);
      }
      dl->AddRect(p0, ImVec2(p0.x + bar_w, p0.y + bar_h),
                  IM_COL32(90, 90, 90, 255));
      ImGui::Dummy(ImVec2(bar_w, bar_h));
    }

    // Editable stops. Applying the value live keeps the preview in sync every
    // frame of a drag, but persistence is coalesced to IsItemDeactivatedAfterEdit
    // so we don't rewrite view_prefs.json ~60x/sec while the user drags a swatch.
    auto edit_stop = [&](const char* label, Rgba8& stop) {
      float col[3] = {stop.r / 255.0f, stop.g / 255.0f, stop.b / 255.0f};
      if (ImGui::ColorEdit3(label, col,
                            ImGuiColorEditFlags_NoInputs |
                                ImGuiColorEditFlags_NoLabel)) {
        stop.r = static_cast<uint8_t>(std::clamp(col[0], 0.0f, 1.0f) * 255.0f + 0.5f);
        stop.g = static_cast<uint8_t>(std::clamp(col[1], 0.0f, 1.0f) * 255.0f + 0.5f);
        stop.b = static_cast<uint8_t>(std::clamp(col[2], 0.0f, 1.0f) * 255.0f + 0.5f);
        stop.a = 255;
        grad.preset = GradientPreset::Custom;  // live: switches tag immediately
      }
      if (ImGui::IsItemDeactivatedAfterEdit()) dirty = true;  // persist once
      ImGui::SameLine();
      ImGui::TextUnformatted(label);
    };
    edit_stop("low", grad.low);
    edit_stop("mid", grad.mid);
    edit_stop("high", grad.high);

    if (ImGui::Checkbox("Reverse", &grad.reverse)) dirty = true;

    // Scale: log vs linear.
    ImGui::TextUnformatted("Scale:");
    ImGui::SameLine();
    if (ImGui::RadioButton("log", vs.heatmap_log_scale)) {
      vs.heatmap_log_scale = true;
      dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("linear", !vs.heatmap_log_scale)) {
      vs.heatmap_log_scale = false;
      dirty = true;
    }
  }

  if (dirty) app.save_prefs();
}

// Build the tab-separated "copy summary" text: model totals + quant table.
static std::string cost_summary_text(App& app, const CostReport& report) {
  ModelSession& s = app.session();
  const ir::Model* model = s.model();
  std::string out = "NetVis cost summary\n";
  if (model) {
    out += "model\t";
    out += std::string(model->str(model->format_name));
    out += "\n";
  }
  char line[128];
  if (report.from_graph) {
    if (report.nodes_flops_known > 0) {
      std::snprintf(line, sizeof(line), "total FLOPs\t%llu\n",
                    static_cast<unsigned long long>(report.total_flops));
      out += line;
    } else {
      out += "total FLOPs\t(none known)\n";
    }
    std::snprintf(line, sizeof(line), "FLOPs known nodes\t%u / %u\n",
                  report.nodes_flops_known, report.nodes_total);
    out += line;
  }
  std::snprintf(line, sizeof(line), "total params\t%llu\n",
                static_cast<unsigned long long>(report.total_params));
  out += line;
  std::snprintf(line, sizeof(line), "total weight bytes\t%llu\n",
                static_cast<unsigned long long>(report.total_weight_bytes));
  out += line;
  std::snprintf(line, sizeof(line), "peak activation bytes\t%llu\n",
                static_cast<unsigned long long>(report.peak_activation_bytes));
  out += line;
  std::snprintf(line, sizeof(line), "effective bits/param\t%.3f\n",
                report.effective_bits_per_param());
  out += line;
  std::snprintf(line, sizeof(line), "size vs fp32\t%.4f\n",
                report.size_vs_fp32());
  out += line;
  if (!report.dtype_usage.empty()) {
    out += "\ndtype\tparams\tbytes\n";
    for (const DTypeUsage& d : report.dtype_usage) {
      std::snprintf(line, sizeof(line), "%s\t%llu\t%llu\n",
                    ir::dtype_name(d.dtype),
                    static_cast<unsigned long long>(d.params),
                    static_cast<unsigned long long>(d.bytes));
      out += line;
    }
  }
  return out;
}

void draw_heatmap_legend(App& app) {
  ViewState& vs = app.view();
  if (!vs.cost_heatmap || !vs.cost) return;
  HeatmapRange range = heatmap_range(app);
  if (!range.valid) return;

  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 win_min = ImGui::GetWindowPos();
  const ImVec2 win_sz = ImGui::GetWindowSize();

  // Inset the legend in the TOP-LEFT of the canvas (the minimap owns bottom-right).
  const float kPad = 12.0f;
  const float bar_w = 160.0f, bar_h = 12.0f;
  ImVec2 p0(win_min.x + kPad, win_min.y + kPad + 16.0f);  // leave room for a title

  // Title.
  const ImU32 text_col = IM_COL32(220, 220, 220, 255);
  const ImU32 shadow = IM_COL32(0, 0, 0, 180);
  auto label = [&](ImVec2 at, const char* txt) {
    dl->AddText(ImVec2(at.x + 1, at.y + 1), shadow, txt);
    dl->AddText(at, text_col, txt);
  };
  label(ImVec2(win_min.x + kPad, win_min.y + kPad),
        heatmap_metric_name(vs.heatmap_metric));

  // Gradient bar (respects the active gradient + reverse).
  const int segs = 64;
  for (int i = 0; i < segs; ++i) {
    float t0 = static_cast<float>(i) / segs;
    float t1 = static_cast<float>(i + 1) / segs;
    ImU32 c0 = rgba8_to_imu32(gradient_sample(vs.heatmap_gradient, t0));
    ImU32 c1 = rgba8_to_imu32(gradient_sample(vs.heatmap_gradient, t1));
    dl->AddRectFilledMultiColor(ImVec2(p0.x + bar_w * t0, p0.y),
                                ImVec2(p0.x + bar_w * t1, p0.y + bar_h),
                                c0, c1, c1, c0);
  }
  dl->AddRect(p0, ImVec2(p0.x + bar_w, p0.y + bar_h), IM_COL32(90, 90, 90, 255));

  // Min/max labels under the bar ends (formatted in the selected metric's unit).
  std::string lo = format_metric_value(range.min_value, vs.heatmap_metric);
  std::string hi = format_metric_value(range.max_value, vs.heatmap_metric);
  ImVec2 lbl_pos(p0.x, p0.y + bar_h + 2.0f);
  label(lbl_pos, lo.c_str());
  float hi_w = ImGui::CalcTextSize(hi.c_str()).x;
  label(ImVec2(p0.x + bar_w - hi_w, p0.y + bar_h + 2.0f), hi.c_str());
  // Scale note.
  const char* scale = vs.heatmap_log_scale ? "log" : "linear";
  label(ImVec2(p0.x, p0.y + bar_h + 18.0f), scale);
  (void)win_sz;
}

}  // namespace netvis
