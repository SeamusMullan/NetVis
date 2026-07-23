// view/PluginsPanel.h — the Plugins management panel (v0.6.0 #11).
//
// Lists discovered plugins (name / author / api_version / status), the ops each
// declares and which built-ins they override, and surfaces the trust model:
// declarative plugins load freely (safe by construction, green); WASM plugins are
// sandboxed-but-arbitrary and disabled by default (amber). Module-private to view/;
// pure ImGui over the process-wide loaded_manifests() from the plugin loader.
#pragma once

namespace netvis {

class App;  // forward; defined in view/App.h

// Draw the Plugins window when app.view().show_plugins is set. No-op otherwise.
void draw_plugins_panel(App& app);

}  // namespace netvis
