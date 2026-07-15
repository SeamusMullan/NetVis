// view/DiffPanel.cpp — comparison / model-diff panel (v0.2.0 model diff).
//
// Pure ImGui + reading published DiffLoader results (view -> engine only; no
// parser include). The panel loads a second model as the comparison, shows the
// added/removed/changed summary + lists, and drives the diff-color overlay
// resolved by diff_tint_for_display() (consulted by GraphCanvas).
#define IMGUI_DEFINE_MATH_OPERATORS
#include "view/DiffPanel.h"

#include <cstdint>
#include <string>

#include "imgui.h"

#include "engine/DiffLoader.h"
#include "engine/LayoutEngine.h"
#include "engine/ModelDiff.h"
#include "ir/IR.h"
#include "view/App.h"
#include "view/PanelHelpers.h"

// tinyfiledialogs: declared here (same as App.cpp) so the panel can open the
// comparison-model file dialog. Definition provided by the tinyfiledialogs TU.
extern "C" {
char* tinyfd_openFileDialog(const char* aTitle, const char* aDefaultPathAndFile,
                            int aNumOfFilterPatterns,
                            const char* const* aFilterPatterns,
                            const char* aSingleFilterDescription,
                            int aAllowMultipleSelects);
}

namespace netvis {

namespace {

// Diff overlay colors (dark-first, chosen distinct from the op-category palette).
constexpr ImU32 kColAdded = IM_COL32(76, 201, 120, 255);    // green
constexpr ImU32 kColRemoved = IM_COL32(224, 92, 92, 255);   // red
constexpr ImU32 kColChanged = IM_COL32(232, 168, 56, 255);  // amber

}  // namespace

DiffTint diff_tint_for_display(App& app, int32_t display_id) {
  DiffTint out;
  DiffLoader& dl = app.diff_loader();
  if (!dl.active()) return out;

  ModelSession& s = app.session();
  // Only tint if the diff was computed against the CURRENT primary model+graph.
  // Generation guards a primary RELOAD (current_graph resets to 0 on open, so a
  // graph-index match alone would paint a fresh model with stale diff status).
  if (dl.primary_generation() != s.generation()) return out;
  if (dl.primary_graph() != s.current_graph()) return out;

  const ModelDiffResult* diff = dl.diff();
  if (diff == nullptr || !diff->valid) return out;

  const auto& disp = s.collapse().display_nodes();
  if (display_id < 0 || static_cast<size_t>(display_id) >= disp.size())
    return out;
  const DisplayNode& dn = disp[static_cast<size_t>(display_id)];

  // Resolve the A-side (primary) node index this display node maps to. For a
  // collapsed group, use its first member as the representative.
  uint32_t a_node = UINT32_MAX;
  if (dn.is_group) {
    const auto& groups = s.collapse().groups();
    if (dn.group_index < groups.size() &&
        !groups[dn.group_index].member_nodes.empty())
      a_node = groups[dn.group_index].member_nodes.front();
  } else {
    a_node = dn.ir_node;
  }
  if (a_node == UINT32_MAX || a_node >= diff->a_status.size()) return out;

  switch (diff->a_status[a_node]) {
    case DiffStatus::Removed:
      out.active = true;
      out.color = kColRemoved;
      break;
    case DiffStatus::Changed:
      out.active = true;
      out.color = kColChanged;
      break;
    case DiffStatus::Added:  // A-side nodes are never "Added" (that's B-only).
    case DiffStatus::Same:
    default:
      out.active = false;
      break;
  }
  return out;
}

void draw_diff_panel(App& app) {
  ViewState& vs = app.view();
  if (!vs.diff_panel_open) return;

  if (!ImGui::Begin("Model Diff", &vs.diff_panel_open)) {
    ImGui::End();
    return;
  }

  DiffLoader& dl = app.diff_loader();
  ModelSession& s = app.session();

  if (ImGui::Button("Load comparison model...")) {
    const char* filters[] = {"*.onnx", "*.tflite", "*.safetensors",
                             "*.gguf", "*.pt",     "*.pth",
                             "*.bin"};
    char* picked =
        tinyfd_openFileDialog("Open comparison model", "", 7, filters,
                              "Model files", 0);
    if (picked != nullptr) dl.load_comparison(s, picked);
  }
  if (dl.state() != DiffLoadState::Empty) {
    ImGui::SameLine();
    if (ImGui::Button("Clear")) dl.clear();
  }

  ImGui::Separator();

  switch (dl.state()) {
    case DiffLoadState::Empty:
      ImGui::TextDisabled("No comparison loaded.");
      ImGui::End();
      return;
    case DiffLoadState::Loading:
      ImGui::TextUnformatted("Loading comparison...");
      ImGui::End();
      return;
    case DiffLoadState::Failed:
      ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f), "Load failed:");
      ImGui::TextWrapped("%s", dl.error().c_str());
      ImGui::End();
      return;
    case DiffLoadState::Ready:
      break;
  }

  ImGui::Text("Comparison: %s", dl.path().c_str());
  const ModelDiffResult* diff = dl.diff();
  if (diff == nullptr) {
    ImGui::TextDisabled("No diff result.");
    ImGui::End();
    return;
  }

  if (dl.primary_graph() != s.current_graph()) {
    ImGui::TextDisabled(
        "Diff was computed for graph %u; showing summary only "
        "(re-load to diff the current graph).",
        dl.primary_graph());
  }

  ImGui::SeparatorText("Summary");
  ImGui::Text("same:    %u", diff->same);
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColAdded), "added:   %u",
                     diff->added);
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColRemoved), "removed: %u",
                     diff->removed);
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColChanged), "changed: %u",
                     diff->changed);

  const ir::Model* a_model = s.model();
  const ir::Model* b_model = dl.model();
  const uint32_t a_gi = dl.primary_graph();

  // Helper to render a scrollable list of A-side nodes matching a status.
  auto list_a = [&](const char* title, DiffStatus want, ImU32 col) {
    if (a_model == nullptr || a_gi >= a_model->graphs.size()) return;
    const auto& nodes = a_model->graphs[a_gi].nodes;
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    bool open = ImGui::CollapsingHeader(title);
    ImGui::PopStyleColor();
    if (!open) return;
    if (ImGui::BeginChild(title, ImVec2(0, 140), ImGuiChildFlags_Borders)) {
      for (uint32_t i = 0; i < diff->a_status.size() && i < nodes.size(); ++i) {
        if (diff->a_status[i] != want) continue;
        const ir::Node& node = nodes[i];
        std::string_view nm = a_model->str(node.name);
        std::string_view op = a_model->str(node.op_type);
        std::string label(op.empty() ? "?" : std::string(op));
        if (!nm.empty()) {
          label += "  ";
          label += std::string(nm);
        }
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Selectable(label.c_str())) {
          // Jump the canvas to this (still-present) A-side node.
          int32_t disp =
              panel_detail::display_index_for_node(s.collapse(), i);
          if (disp >= 0) vs.selected_display = disp;
        }
        ImGui::PopID();
      }
    }
    ImGui::EndChild();
  };

  // B-side list for "added" nodes (present only in comparison).
  auto list_b_added = [&]() {
    if (b_model == nullptr) return;
    // DiffLoader always diffs against comparison graph 0 (diff_models(...,*B,0)),
    // so b_status is parallel to graph 0's node list — index that, NOT a_gi.
    constexpr uint32_t b_gi = 0u;
    if (b_gi >= b_model->graphs.size()) return;
    const auto& nodes = b_model->graphs[b_gi].nodes;
    ImGui::PushStyleColor(ImGuiCol_Text, kColAdded);
    bool open = ImGui::CollapsingHeader("Added (comparison only)");
    ImGui::PopStyleColor();
    if (!open) return;
    if (ImGui::BeginChild("added_b", ImVec2(0, 140), ImGuiChildFlags_Borders)) {
      for (uint32_t i = 0; i < diff->b_status.size() && i < nodes.size(); ++i) {
        if (diff->b_status[i] != DiffStatus::Added) continue;
        const ir::Node& node = nodes[i];
        std::string_view nm = b_model->str(node.name);
        std::string_view op = b_model->str(node.op_type);
        std::string label(op.empty() ? "?" : std::string(op));
        if (!nm.empty()) {
          label += "  ";
          label += std::string(nm);
        }
        ImGui::TextUnformatted(label.c_str());
      }
    }
    ImGui::EndChild();
  };

  ImGui::SeparatorText("Details");
  list_a("Removed (primary only)", DiffStatus::Removed, kColRemoved);
  list_a("Changed", DiffStatus::Changed, kColChanged);
  list_b_added();

  ImGui::End();
}

}  // namespace netvis
