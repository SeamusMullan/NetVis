// view/PropertiesPanel.cpp — the "Properties" inspector panel (spec §8.2).
//
// Shows details for whatever is selected: a leaf IR node (op_type, name,
// category, typed attribute table, click-to-jump inputs/outputs), a collapse
// GROUP (label / instances / expand toggle), or — when nothing is selected —
// the MODEL ROOT (format, producer, metadata, and cheap graph stats).
//
// PERF/DECISION: this panel reads ONLY structural IR (already parsed and
// published on the main thread) — never a tensor payload. The Inspect button on
// a Tensor attribute defers to App::inspect_tensor(), which kicks the single
// payload-reading job (spec §2.1). Param counts sum TensorRef::elem_count(),
// which is pure shape arithmetic, so "total params" costs nothing at the bytes
// level even for multi-GB models.
#include <cctype>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h (pulled in by
// App.h) references without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/CostModel.h"   // #32: compute_receptive_fields / NodeReceptiveField
#include "engine/OpCategory.h"
#include "engine/plugin/Registry.h"
#include "ir/IR.h"
#include "parsers/gguf/GgufMetadata.h"  // #44: well-known GGUF KV grouping
#include "view/App.h"
#include "view/CostPanel.h"
#include "view/PanelHelpers.h"

namespace netvis {

namespace {

using panel_detail::display_index_for_node;
using panel_detail::grouped_count;
using panel_detail::human_bytes;
using panel_detail::node_to_json;
using panel_detail::resolve_edge_value;
using panel_detail::shape_string;

// Render one value's name/shape/dtype row with click-to-jump. `value_idx` is an
// index into `g.values`. Jumping selects the display node containing that
// value's producer node (spec §8.2 click-to-jump).
void draw_value_row(App& app, const ir::Model& model, const ir::Graph& g,
                    uint32_t value_idx, int row_id) {
  ImGui::PushID(row_id);
  if (value_idx >= g.values.size()) {
    ImGui::TextDisabled("(unresolved edge)");
    ImGui::PopID();
    return;
  }
  const ir::ValueInfo& vi = g.values[value_idx];
  std::string_view name = model.str(vi.name);
  std::string label(name.empty() ? "(anon)" : name);
  label += "  ";
  label += shape_string(vi.shape);
  label += "  ";
  label += ir::dtype_name(vi.dtype);

  // Selectable so the whole row is a jump target.
  if (ImGui::Selectable(label.c_str(), false)) {
    if (vi.producer >= 0) {
      int32_t disp =
          display_index_for_node(app.session().collapse(),
                                 static_cast<uint32_t>(vi.producer));
      if (disp >= 0) {
        app.view().selected_display = disp;
        app.view().selected_value = static_cast<int32_t>(value_idx);
      }
    } else {
      // Graph input / initializer: no producer node to jump to; record the value
      // selection so the canvas can still highlight the edge.
      app.view().selected_value = static_cast<int32_t>(value_idx);
    }
  }
  ImGui::PopID();
}

// Render the typed attribute table for a node (spec §8.2). #60: a filter box lets
// the user narrow long attribute lists by name (match on attr name; a common ask
// for big fused ops with dozens of attrs). Filter state lives in the active tab's
// ViewState (per-tab, v0.8.0 #62) so switching tabs never carries a stale filter
// onto another model; it is also cleared whenever the selected node changes.
void draw_attributes(App& app, const ir::Model& model, const ir::Graph& g,
                     const ir::Node& node) {
  if (node.attributes.count == 0) {
    ImGui::TextDisabled("(no attributes)");
    return;
  }

  ViewState& vs = app.view();

  // Reset the filter when the selected node changes so the box never carries a
  // stale filter onto a different node. Keyed by (graph,node).
  const uint64_t this_key =
      (static_cast<uint64_t>(app.session().current_graph()) << 32) |
      static_cast<uint64_t>(&node - g.nodes.data());
  if (this_key != vs.attr_filter_key) {
    vs.attr_filter_key = this_key;
    vs.attr_filter.clear();
  }

  // Only show the filter box once there are enough attributes to warrant it.
  // Char buffer synced to vs.attr_filter (imgui_stdlib not built), as in the
  // tensor table.
  const bool show_filter = node.attributes.count > 6;
  if (show_filter) {
    char filter_buf[64];
    std::snprintf(filter_buf, sizeof(filter_buf), "%s", vs.attr_filter.c_str());
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::InputTextWithHint("##attr_filter", "filter attributes...",
                                 filter_buf, sizeof(filter_buf)))
      vs.attr_filter = filter_buf;
  }
  const std::string_view needle(vs.attr_filter);  // icontains folds case both sides

  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_SizingStretchProp;
  if (!ImGui::BeginTable("attrs", 2, flags)) return;
  ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthFixed, 140.0f);
  ImGui::TableSetupColumn("value");
  ImGui::TableHeadersRow();

  uint32_t shown = 0;
  for (uint32_t a = 0; a < node.attributes.count; ++a) {
    uint32_t ai = node.attributes.begin + a;
    if (ai >= g.attributes.size()) break;  // defensive: malformed range
    const ir::Attribute& attr = g.attributes[ai];
    if (!needle.empty() && !panel_detail::icontains(model.str(attr.name), needle))
      continue;  // #60: filtered out by name
    ++shown;
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    std::string_view aname = model.str(attr.name);
    ImGui::TextUnformatted(aname.empty() ? "(unnamed)" : std::string(aname).c_str());

    ImGui::TableSetColumnIndex(1);
    ImGui::PushID(static_cast<int>(ai));
    const ir::AttrValue& v = attr.value;
    switch (v.kind) {
      case ir::AttrValue::Kind::None:
        ImGui::TextDisabled("-");
        break;
      case ir::AttrValue::Kind::Int:
        ImGui::Text("%lld", static_cast<long long>(v.i));
        break;
      case ir::AttrValue::Kind::Float:
        ImGui::Text("%g", v.f);
        break;
      case ir::AttrValue::Kind::String: {
        std::string_view s = model.str(v.s);
        ImGui::TextUnformatted(std::string(s).c_str());
        break;
      }
      case ir::AttrValue::Kind::Ints: {
        std::string s = "[";
        for (size_t i = 0; i < v.ints.size(); ++i) {
          if (i) s += ", ";
          s += std::to_string(v.ints[i]);
        }
        s += "]";
        ImGui::TextWrapped("%s", s.c_str());
        break;
      }
      case ir::AttrValue::Kind::Floats: {
        std::string s = "[";
        char buf[32];
        for (size_t i = 0; i < v.floats.size(); ++i) {
          if (i) s += ", ";
          std::snprintf(buf, sizeof(buf), "%g", v.floats[i]);
          s += buf;
        }
        s += "]";
        ImGui::TextWrapped("%s", s.c_str());
        break;
      }
      case ir::AttrValue::Kind::Strings: {
        std::string s = "[";
        for (size_t i = 0; i < v.strings.size(); ++i) {
          if (i) s += ", ";
          s += std::string(model.str(v.strings[i]));
        }
        s += "]";
        ImGui::TextWrapped("%s", s.c_str());
        break;
      }
      case ir::AttrValue::Kind::Tensor: {
        // One-line "dtype shape" + an Inspect button (the ONLY route to payload
        // bytes — deferred to the async decode job, spec §2.1).
        std::string line = ir::dtype_name(v.tensor.dtype);
        line += " ";
        line += shape_string(v.tensor.shape);
        ImGui::TextUnformatted(line.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Inspect")) app.inspect_tensor(v.tensor);
        break;
      }
      case ir::AttrValue::Kind::Graph: {
        ImGui::TextDisabled("subgraph");
        ImGui::SameLine();
        if (ImGui::SmallButton("Dive in") && v.graph >= 0) {
          app.session().push_graph(static_cast<uint32_t>(v.graph));
        }
        break;
      }
    }
    ImGui::PopID();
  }
  ImGui::EndTable();
  if (shown == 0 && !needle.empty())
    ImGui::TextDisabled("(no attributes match \"%s\")", vs.attr_filter.c_str());
}

// The MODEL ROOT view shown when nothing is selected (spec §8.2).
void draw_model_root(App& app, const ir::Model& model) {
  ImGui::SeparatorText("Model");
  auto kv = [](const char* k, std::string_view val) {
    ImGui::TextDisabled("%s", k);
    ImGui::SameLine(160.0f);
    ImGui::TextUnformatted(val.empty() ? "-" : std::string(val).c_str());
  };
  kv("format", model.str(model.format_name));
  kv("producer", model.str(model.producer));
  kv("version", model.str(model.version_info));

  // #44: GGUF metadata summary. GGUF is table-mode (has_graph==false), so this
  // model-root view is what a GGUF user sees in Properties (the tensor table
  // never selects a node). The parser already dropped every KV into
  // model.metadata; here we pull the well-known keys (arch / quant / rope /
  // attention / context length / block count …) into labelled groups ABOVE the
  // raw KV table so the model's shape reads at a glance. Empty when not GGUF or
  // when none of the recognised keys are present, so it self-hides.
  {
    // Cache the grouped summary per (model ptr, generation) — it's a pure function
    // of the immutable metadata, so rebuilding its ~28×N key scans + string allocs
    // every frame (this is the view a GGUF user sits on) is pure waste. Same
    // (pointer, generation) cache pattern as the cost/detect caches.
    static const ir::Model* gg_model = nullptr;
    static uint64_t gg_gen = UINT64_MAX;
    static std::vector<gguf::GgufSummaryGroup> gg_groups;
    const uint64_t gen = app.session().generation();
    if (gg_model != &model || gg_gen != gen) {
      gg_model = &model;
      gg_gen = gen;
      gg_groups = gguf::summarize_gguf_metadata(model);
    }
    const std::vector<gguf::GgufSummaryGroup>& groups = gg_groups;
    if (!groups.empty()) {
      ImGui::SeparatorText("GGUF");
      const ImGuiTableFlags gflags =
          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
      for (const gguf::GgufSummaryGroup& grp : groups) {
        // Groups open by default so the summary is visible without a click.
        if (!ImGui::CollapsingHeader(grp.title.c_str(),
                                     ImGuiTreeNodeFlags_DefaultOpen))
          continue;
        ImGui::PushID(grp.title.c_str());
        if (ImGui::BeginTable("ggufgrp", 2, gflags)) {
          ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthFixed, 180.0f);
          ImGui::TableSetupColumn("value");
          for (const gguf::GgufSummaryItem& it : grp.items) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(it.label.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextWrapped("%s", it.value.c_str());
          }
          ImGui::EndTable();
        }
        ImGui::PopID();
      }
    }
  }

  // Metadata table (opset / ir version etc. live here per parser convention).
  if (!model.metadata.empty()) {
    ImGui::SeparatorText("Metadata");
    const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
    if (ImGui::BeginTable("meta", 2, flags)) {
      ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthFixed, 160.0f);
      ImGui::TableSetupColumn("value");
      ImGui::TableHeadersRow();
      for (const auto& [k, val] : model.metadata) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(std::string(model.str(k)).c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::TextWrapped("%s", std::string(model.str(val)).c_str());
      }
      ImGui::EndTable();
    }
  }

  // Cheap graph stats: no payload reads — elem_count() is shape arithmetic only.
  ImGui::SeparatorText("Graph stats");
  uint64_t node_count = 0, edge_count = 0;
  int64_t total_params = 0;
  for (const ir::Graph& g : model.graphs) {
    node_count += g.nodes.size();
    edge_count += g.edge_refs.size();
    for (const ir::TensorRef& t : g.initializers) total_params += t.elem_count();
  }
  // has_graph == false models keep their tensors flat (spec §8.6).
  for (const ir::TensorRef& t : model.flat_tensors) total_params += t.elem_count();

  ImGui::Text("graphs:  %zu", model.graphs.size());
  ImGui::Text("nodes:   %s", grouped_count(static_cast<int64_t>(node_count)).c_str());
  ImGui::Text("edges:   %s", grouped_count(static_cast<int64_t>(edge_count)).c_str());
  ImGui::Text("params:  %s", grouped_count(total_params).c_str());

  // v0.3.0 analyzer: model-wide cost totals + quant-coverage + heatmap toggle.
  draw_cost_section(app);
}

}  // namespace

// Draw the Properties panel. Called once per frame from App::frame() (spec §8.2).
void draw_properties_panel(App& app) {
  if (!ImGui::Begin("Properties")) {
    ImGui::End();
    return;
  }

  ModelSession& session = app.session();
  const ir::Model* model = session.model();
  if (!model) {
    ImGui::TextDisabled("No model loaded.");
    ImGui::End();
    return;
  }

  const int32_t sel = app.view().selected_display;
  const CollapseTree& collapse = session.collapse();
  const auto& display = collapse.display_nodes();

  // Nothing (validly) selected -> MODEL ROOT.
  if (sel < 0 || static_cast<size_t>(sel) >= display.size()) {
    draw_model_root(app, *model);
    ImGui::End();
    return;
  }

  const DisplayNode& dn = display[static_cast<size_t>(sel)];

  // GROUP selection (spec §8.2): label / instances / member count / toggle.
  if (dn.is_group) {
    const auto& groups = collapse.groups();
    if (dn.group_index >= groups.size()) {
      draw_model_root(app, *model);
      ImGui::End();
      return;
    }
    const CollapseGroup& grp = groups[dn.group_index];
    ImGui::SeparatorText("Collapsed group");
    ImGui::Text("label:    %s", grp.label.empty() ? "(group)" : grp.label.c_str());
    ImGui::Text("instances: %u", grp.instances);
    ImGui::Text("members:  %zu nodes", grp.member_nodes.size());
    const char* btn = dn.expanded ? "Collapse" : "Expand";
    if (ImGui::Button(btn)) session.toggle_group(dn.group_index);
    draw_cost_section(app);  // per-instance + rolled-up (xN) group cost + totals
    ImGui::End();
    return;
  }

  // LEAF NODE selection: op_type / name / category / attrs / inputs / outputs.
  uint32_t gi = session.current_graph();
  if (gi >= model->graphs.size() || dn.ir_node >= model->graphs[gi].nodes.size()) {
    draw_model_root(app, *model);
    ImGui::End();
    return;
  }
  const ir::Graph& g = model->graphs[gi];
  const ir::Node& node = g.nodes[dn.ir_node];

  std::string_view op = model->str(node.op_type);
  std::string_view name = model->str(node.name);
  OpCategory cat = plugin::resolve_category(*model, g, node);  // v0.6.0 #8

  ImGui::SeparatorText("Node");
  ImGui::Text("op:       %s", op.empty() ? "?" : std::string(op).c_str());
  ImGui::Text("name:     %s", name.empty() ? "(unnamed)" : std::string(name).c_str());
  ImGui::Text("category: %s", category_name(cat));

  // #32: per-node receptive field (conv/pool stacks). Cached per (generation,
  // graph, enrich) since it's a pure O(V+E) pass over shapes — recompute only when
  // the model/graph/shape-epoch changes, not per selection or per frame.
  {
    // Key includes the MODEL POINTER: per-tab JobSystems make (generation, graph,
    // enrich) collide across tabs (each starts at gen 0), so without the model
    // identity a second tab's node would be looked up in the first tab's RF table
    // (the #62 multi-tab collision that #14's cache avoids via its report ptr).
    static std::vector<NodeReceptiveField> rf_cache;
    static const ir::Model* rf_model = nullptr;
    static uint64_t rf_gen = UINT64_MAX;
    static uint32_t rf_graph = UINT32_MAX;
    static uint64_t rf_enrich = UINT64_MAX;
    const uint64_t gen = session.generation();
    const uint64_t enrich = session.enrich_generation();
    if (rf_model != model || rf_gen != gen || rf_graph != gi || rf_enrich != enrich) {
      rf_cache = compute_receptive_fields(*model, gi);
      rf_model = model;
      rf_gen = gen;
      rf_graph = gi;
      rf_enrich = enrich;
    }
    if (dn.ir_node < rf_cache.size() && rf_cache[dn.ir_node].known) {
      ImGui::Text("recept. field: %llu",
                  static_cast<unsigned long long>(rf_cache[dn.ir_node].rf));
      if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Receptive field along the first spatial axis: input pixels that "
            "influence one output pixel (conv/pool stack, shape-derived).");
    }
  }

  // #57: copy the whole node (op/attrs/input-output shapes) as JSON.
  if (ImGui::SmallButton("Copy as JSON")) {
    std::string js = panel_detail::node_to_json(*model, g, node);
    ImGui::SetClipboardText(js.c_str());
  }

  ImGui::SeparatorText("Attributes");
  draw_attributes(app, *model, g, node);

  ImGui::SeparatorText("Inputs");
  if (node.inputs.count == 0) {
    ImGui::TextDisabled("(none)");
  } else {
    for (uint32_t s = 0; s < node.inputs.count; ++s) {
      uint32_t vidx = resolve_edge_value(g, node.inputs, s);
      draw_value_row(app, *model, g, vidx, static_cast<int>(s));
    }
  }

  ImGui::SeparatorText("Outputs");
  if (node.outputs.count == 0) {
    ImGui::TextDisabled("(none)");
  } else {
    for (uint32_t s = 0; s < node.outputs.count; ++s) {
      uint32_t vidx = resolve_edge_value(g, node.outputs, s);
      // Offset ids so input/output selectables never collide in the ID stack.
      draw_value_row(app, *model, g, vidx, static_cast<int>(s) + 100000);
    }
  }

  draw_cost_section(app);  // this node's FLOPs/params/bytes + model totals
  ImGui::End();
}

}  // namespace netvis
