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

namespace {
// §C.4 DEFERRED TOGGLE (no UAF): applying a toggle calls reload_plugins(), which
// replaces the loaded_manifests() vector we iterate by const&. So we NEVER mutate
// inside the loop — we record the intended action in file-static state and apply it
// exactly once AFTER the per-row loop + modal block, when no reference into the
// cache is live. (Mirrors the existing pending-modal pattern.)
bool g_have_pending = false;
std::string g_pending_id;
bool g_pending_val = false;
// A WASM off->on that needs the one-time confirm dialog.
bool g_confirm_open = false;
std::string g_confirm_id;
}  // namespace

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
      "(JSON + expression DSL) are safe by construction and enabled by default. WASM "
      "plugins run sandboxed (memory/fuel-capped, no filesystem/network, cannot read "
      "tensor bytes) but are still arbitrary code, so they are disabled by default.");
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

    const bool is_wasm = (lm.kind == plugin::PluginKind::Wasm);

    // Enable checkbox — the FIRST widget on the row. Records a DEFERRED action; the
    // actual reload happens after the loop (§C.4). Disabled (greyed) for a rejected
    // plugin (nothing to enable).
    bool enabled = lm.enabled;
    ImGui::BeginDisabled(!lm.ok && !is_wasm);   // a rejected declarative can't enable
    if (ImGui::Checkbox("##en", &enabled)) {
      if (is_wasm && enabled && !lm.enabled) {
        // WASM off->on: require the one-time confirm dialog before enabling.
        g_confirm_open = true;
        g_confirm_id = lm.id;
      } else {
        g_have_pending = true;
        g_pending_id = lm.id;
        g_pending_val = enabled;
      }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();

    std::string title = lm.name.empty() ? (lm.id.empty() ? lm.path : lm.id) : lm.name;
    bool open = ImGui::TreeNodeEx(title.c_str(), ImGuiTreeNodeFlags_DefaultOpen);

    // Trust badge: amber for WASM (sandboxed code), green for declarative (safe).
    ImGui::SameLine();
    if (is_wasm)
      ImGui::TextColored(kAmber, "[wasm \xE2\x80\xA2 sandboxed]");
    else
      ImGui::TextColored(kGreen, "[declarative \xE2\x80\xA2 safe]");

    // Status.
    ImGui::SameLine();
    if (!lm.ok)
      ImGui::TextColored(kRed, "rejected: %s", lm.error.c_str());
    else if (!lm.enabled)
      ImGui::TextDisabled("disabled");
    else if (lm.registered)
      ImGui::TextColored(kGreen, "enabled");
    else
      ImGui::TextColored(kAmber, "enabled (not registered)");

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

  // --- WASM enable confirm modal (opened by an off->on toggle above) ----------
  if (g_confirm_open) { ImGui::OpenPopup("Enable WASM plugin?"); g_confirm_open = false; }
  if (ImGui::BeginPopupModal("Enable WASM plugin?", nullptr,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::TextWrapped(
        "This plugin runs sandboxed WebAssembly: memory- and fuel-capped, no "
        "filesystem/network access, and it cannot read tensor weight bytes. It is "
        "still arbitrary code. Enable it?");
    ImGui::Separator();
    if (ImGui::Button("Enable")) {
      g_have_pending = true; g_pending_id = g_confirm_id; g_pending_val = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }

  ImGui::End();

  // --- DEFERRED apply (§C.4): now NO reference into loaded_manifests() is live, so
  // it is safe to persist the choice + reload (which replaces that vector). --------
  if (g_have_pending) {
    app.plugin_enabled().set(g_pending_id, g_pending_val);
    app.save_prefs();
    app.reload_plugins();
    g_have_pending = false;
    g_pending_id.clear();
  }
}

}  // namespace netvis
