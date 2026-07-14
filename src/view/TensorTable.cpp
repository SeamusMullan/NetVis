// view/TensorTable.cpp — flat-tensor browsing mode (spec §8.6, has_graph==false).
//
// For formats with no compute graph (GGUF, SafeTensors, state_dict-only
// PyTorch) the model is just a big list of tensors. This panel shows them two
// ways: a left-hand module HIERARCHY tree (names split on '.' and '/'), and a
// right-hand VIRTUALIZED table (ImGuiListClipper — only the visible rows are
// submitted, so a 100k-tensor model scrolls smoothly) with sortable columns and
// a case-insensitive name filter.
//
// PERF/DECISION: every column is derived from TensorRef metadata (name, dtype,
// shape, elem_count, byte_len, file_offset). NONE of it reads payload bytes; a
// row click routes through App::inspect_tensor(), which is the only path to the
// decode job (spec §2.1). Sorting reorders an index vector, never the tensors.
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h that
// App.h pulls in without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "ir/IR.h"
#include "view/App.h"
#include "view/PanelHelpers.h"

namespace netvis {

namespace {

using panel_detail::human_bytes;
using panel_detail::shape_string;

// Case-insensitive ASCII substring test (spec §8.6 filter). Cheap and locale
// free — model names are ASCII identifiers.
bool icontains(std::string_view hay, std::string_view needle) {
  if (needle.empty()) return true;
  if (needle.size() > hay.size()) return false;
  auto lower = [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  };
  for (size_t i = 0; i + needle.size() <= hay.size(); ++i) {
    bool m = true;
    for (size_t j = 0; j < needle.size(); ++j) {
      if (lower(hay[i + j]) != lower(needle[j])) {
        m = false;
        break;
      }
    }
    if (m) return true;
  }
  return false;
}

// A node in the name hierarchy tree. Children keyed by path segment; `tensor`
// is set on leaves. Built each frame from flat_tensors — cheap for typical
// counts and avoids caching invalidation.
struct TreeNode {
  std::map<std::string, TreeNode> children;
  int64_t tensor_index = -1;  // leaf tensor index, or -1 for interior nodes
};

// Split a tensor name on '.' and '/' into path segments.
std::vector<std::string> split_name(std::string_view name) {
  std::vector<std::string> parts;
  std::string cur;
  for (char c : name) {
    if (c == '.' || c == '/') {
      if (!cur.empty()) {
        parts.push_back(cur);
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) parts.push_back(cur);
  return parts;
}

// Recursively draw the hierarchy tree. Clicking an interior node sets the prefix
// filter (stored in view().table_filter) to that subtree; clicking a leaf
// inspects the tensor.
void draw_tree(App& app, const ir::Model& model, const std::string& prefix,
               const std::string& seg, const TreeNode& node) {
  if (node.tensor_index >= 0) {
    // Leaf: selectable tensor.
    if (ImGui::Selectable(seg.c_str(),
                          app.view().table_selected_row == node.tensor_index)) {
      app.view().table_selected_row = node.tensor_index;
      if (static_cast<size_t>(node.tensor_index) < model.flat_tensors.size())
        app.inspect_tensor(model.flat_tensors[node.tensor_index]);
    }
    return;
  }
  // Interior: a tree node. Left-click the arrow to expand; click the label to
  // filter the table to this prefix (spec §8.6 "selecting a subtree filters").
  ImGui::PushID(seg.c_str());
  bool open = ImGui::TreeNodeEx(seg.c_str(), ImGuiTreeNodeFlags_SpanAvailWidth);
  if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
    app.view().table_filter = prefix.empty() ? seg : (prefix + "." + seg);
  }
  if (open) {
    for (const auto& [child_seg, child] : node.children) {
      std::string child_prefix = prefix.empty() ? seg : (prefix + "." + seg);
      draw_tree(app, model, child_prefix, child_seg, child);
    }
    ImGui::TreePop();
  }
  ImGui::PopID();
}

// Compare two tensors on the active sort column (view().table_sort_col).
bool less_by_col(const ir::Model& model, const ir::TensorRef& a,
                 const ir::TensorRef& b, int col) {
  switch (col) {
    case 0:  // name
      return model.str(a.name) < model.str(b.name);
    case 1:  // dtype
      return std::string_view(ir::dtype_name(a.dtype)) < ir::dtype_name(b.dtype);
    case 2:  // shape: compare by rank then dims
      if (a.shape.size() != b.shape.size()) return a.shape.size() < b.shape.size();
      for (size_t i = 0; i < a.shape.size(); ++i)
        if (a.shape[i] != b.shape[i]) return a.shape[i] < b.shape[i];
      return false;
    case 3:  // params
      return a.elem_count() < b.elem_count();
    case 4:  // bytes
      return a.byte_len < b.byte_len;
    case 5:  // offset
      return a.file_offset < b.file_offset;
    default:
      return false;
  }
}

}  // namespace

// Draw the flat-tensor table (spec §8.6). Only meaningful when has_graph==false,
// but harmless otherwise (shows an empty table).
void draw_tensor_table(App& app) {
  if (!ImGui::Begin("Tensors")) {
    ImGui::End();
    return;
  }

  const ir::Model* model = app.session().model();
  if (!model) {
    ImGui::TextDisabled("No model loaded.");
    ImGui::End();
    return;
  }
  const std::vector<ir::TensorRef>& tensors = model->flat_tensors;
  ViewState& vs = app.view();

  // --- Left region: module hierarchy tree ------------------------------------
  const float tree_w = 240.0f;
  if (ImGui::BeginChild("##tree", ImVec2(tree_w, 0), ImGuiChildFlags_ResizeX)) {
    ImGui::TextDisabled("Modules");
    if (ImGui::SmallButton("clear filter")) vs.table_filter.clear();
    ImGui::Separator();
    // Build the tree fresh (interior nodes have tensor_index<0).
    TreeNode root;
    for (size_t i = 0; i < tensors.size(); ++i) {
      std::vector<std::string> parts = split_name(model->str(tensors[i].name));
      TreeNode* cur = &root;
      for (const std::string& p : parts) cur = &cur->children[p];
      cur->tensor_index = static_cast<int64_t>(i);
    }
    for (const auto& [seg, child] : root.children)
      draw_tree(app, *model, "", seg, child);
  }
  ImGui::EndChild();

  ImGui::SameLine();

  // --- Main region: filter + virtualized table -------------------------------
  ImGui::BeginGroup();

  // Name filter (char buffer synced to vs.table_filter; imgui_stdlib not built).
  static char filter_buf[256];
  std::snprintf(filter_buf, sizeof(filter_buf), "%s", vs.table_filter.c_str());
  ImGui::SetNextItemWidth(240.0f);
  if (ImGui::InputTextWithHint("##filter", "filter name...", filter_buf,
                               sizeof(filter_buf))) {
    vs.table_filter = filter_buf;
  }

  // Build the filtered + sorted index vector. Filtering is O(n) substring; sort
  // touches only indices so the underlying tensors never move (spec §8.6).
  std::vector<uint32_t> order;
  order.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (icontains(model->str(tensors[i].name), vs.table_filter))
      order.push_back(static_cast<uint32_t>(i));
  }
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    // Descending: swap operands once (equal elements keep their stable order).
    // One comparison per pair — no redundant call, matters when sorting 100k rows.
    if (vs.table_sort_asc)
      return less_by_col(*model, tensors[a], tensors[b], vs.table_sort_col);
    return less_by_col(*model, tensors[b], tensors[a], vs.table_sort_col);
  });

  const ImGuiTableFlags tflags =
      ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_Sortable |
      ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;

  // Leave room at the bottom for the summary strip.
  ImVec2 table_size(0.0f, ImGui::GetContentRegionAvail().y -
                              ImGui::GetTextLineHeightWithSpacing() - 6.0f);
  if (ImGui::BeginTable("tensors", 6, tflags, table_size)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_DefaultSort, 0, 0);
    ImGui::TableSetupColumn("dtype", 0, 0, 1);
    ImGui::TableSetupColumn("shape", ImGuiTableColumnFlags_NoSort, 0, 2);
    ImGui::TableSetupColumn("params", 0, 0, 3);
    ImGui::TableSetupColumn("bytes", 0, 0, 4);
    ImGui::TableSetupColumn("offset", 0, 0, 5);
    ImGui::TableHeadersRow();

    // Pull sort spec back into view state (persisted across frames per contract).
    if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs()) {
      if (specs->SpecsCount > 0) {
        vs.table_sort_col = specs->Specs[0].ColumnIndex;
        vs.table_sort_asc =
            (specs->Specs[0].SortDirection != ImGuiSortDirection_Descending);
      }
    }

    // Virtualize: only submit visible rows (spec §8.6).
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(order.size()));
    while (clipper.Step()) {
      for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
        uint32_t ti = order[static_cast<size_t>(r)];
        const ir::TensorRef& t = tensors[ti];
        ImGui::TableNextRow();
        ImGui::PushID(static_cast<int>(ti));

        ImGui::TableSetColumnIndex(0);
        std::string_view name = model->str(t.name);
        std::string nm(name.empty() ? "(anon)" : name);
        bool selected = (vs.table_selected_row == static_cast<int64_t>(ti));
        if (ImGui::Selectable(nm.c_str(), selected,
                              ImGuiSelectableFlags_SpanAllColumns)) {
          vs.table_selected_row = static_cast<int64_t>(ti);
          app.inspect_tensor(t);  // sole route to payload bytes (spec §2.1)
        }

        ImGui::TableSetColumnIndex(1);
        ImGui::TextUnformatted(ir::dtype_name(t.dtype));

        ImGui::TableSetColumnIndex(2);
        ImGui::TextUnformatted(shape_string(t.shape).c_str());

        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%lld", static_cast<long long>(t.elem_count()));

        ImGui::TableSetColumnIndex(4);
        ImGui::TextUnformatted(human_bytes(t.byte_len).c_str());

        ImGui::TableSetColumnIndex(5);
        if (t.file_offset == UINT64_MAX)
          ImGui::TextDisabled("external");
        else
          ImGui::Text("0x%llx", static_cast<unsigned long long>(t.file_offset));

        ImGui::PopID();
      }
    }
    ImGui::EndTable();
  }

  // --- Summary strip ---------------------------------------------------------
  uint64_t total_bytes = 0;
  int64_t total_params = 0;
  for (uint32_t ti : order) {
    total_bytes += tensors[ti].byte_len;
    total_params += tensors[ti].elem_count();
  }
  ImGui::Text("%zu tensors   %s params   %s", order.size(),
              panel_detail::grouped_count(total_params).c_str(),
              human_bytes(total_bytes).c_str());

  ImGui::EndGroup();
  ImGui::End();
}

}  // namespace netvis
