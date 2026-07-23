// view/PluginsPanel.cpp — the Plugins management panel (v0.6.0 #11).
#include "view/PluginsPanel.h"

#include <string>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h (pulled in by
// App.h) references without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/plugin/OpHandler.h"        // kOpHandlerAbiVersion
#include "engine/plugin/declarative/Manifest.h"
#include "view/App.h"

namespace netvis {

void draw_plugins_panel(App& app) {
  ViewState& vs = app.view();
  if (!vs.show_plugins) return;

  if (!ImGui::Begin("Plugins", &vs.show_plugins)) {
    ImGui::End();
    return;
  }

  const auto& manifests = plugin::loaded_manifests();

  ImGui::TextWrapped(
      "Plugins extend NetVis's op coverage without a release. Declarative plugins "
      "(JSON + expression DSL) are safe by construction and load freely. WASM "
      "plugins run sandboxed and are disabled by default.");
  ImGui::Text("Discovered from: %s", plugin::plugin_dir().c_str());
  ImGui::Text("Host op-handler API version: %u",
              static_cast<unsigned>(plugin::kOpHandlerAbiVersion));
  ImGui::Separator();

  if (manifests.empty()) {
    ImGui::TextDisabled("No plugins found. Drop a <name>/plugin.json under the "
                        "directory above and restart.");
    ImGui::End();
    return;
  }

  const ImVec4 kGreen(0.38f, 0.75f, 0.44f, 1.0f);
  const ImVec4 kRed(0.90f, 0.42f, 0.38f, 1.0f);
  const ImVec4 kAmber(0.84f, 0.70f, 0.28f, 1.0f);

  for (size_t i = 0; i < manifests.size(); ++i) {
    const plugin::LoadedManifest& lm = manifests[i];
    ImGui::PushID(static_cast<int>(i));

    std::string title = lm.name.empty() ? lm.path : lm.name;
    bool open = ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen);

    // Trust badge (declarative = safe/green). Same line as the header.
    ImGui::SameLine();
    ImGui::TextColored(kGreen, "[declarative \xE2\x80\xA2 safe]");

    // Status.
    ImGui::SameLine();
    if (lm.ok)
      ImGui::TextColored(kGreen, "loaded");
    else
      ImGui::TextColored(kRed, "rejected: %s", lm.error.c_str());

    if (open) {
      if (!lm.author.empty()) ImGui::TextDisabled("author: %s", lm.author.c_str());
      ImGui::TextDisabled("api_version: %u", static_cast<unsigned>(lm.api_version));
      ImGui::TextDisabled("path: %s", lm.path.c_str());

      if (lm.ok && !lm.ops.empty()) {
        if (ImGui::BeginTable("ops", 3,
                              ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_SizingStretchProp)) {
          ImGui::TableSetupColumn("op");
          ImGui::TableSetupColumn("category");
          ImGui::TableSetupColumn("status");
          ImGui::TableHeadersRow();
          for (const plugin::LoadedOp& op : lm.ops) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(op.op_name.empty() ? "(unnamed)" : op.op_name.c_str());
            ImGui::TableSetColumnIndex(1);
            ImGui::TextUnformatted(op.category.empty() ? "-" : op.category.c_str());
            ImGui::TableSetColumnIndex(2);
            if (!op.ok)
              ImGui::TextColored(kRed, "rejected: %s", op.error.c_str());
            else if (op.overrides_builtin)
              ImGui::TextColored(kAmber, "overrides built-in");
            else
              ImGui::TextColored(kGreen, "ok");
          }
          ImGui::EndTable();
        }
      }
      ImGui::TreePop();
    }
    ImGui::PopID();
  }

  ImGui::End();
}

}  // namespace netvis
