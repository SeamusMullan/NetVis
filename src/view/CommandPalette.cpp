// view/CommandPalette.cpp — Ctrl+P fuzzy command palette (#59). See header.
#include "view/CommandPalette.h"

#include <cctype>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h that
// App.h pulls in without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/SearchIndex.h"   // fuzzy_score — the SAME matcher as the search bar
#include "view/App.h"

namespace netvis {

namespace {

// One palette action: a label the user searches, an optional category prefix for
// grouping in the label, and the effect. `enabled` greys out + blocks actions
// that make no sense right now (e.g. "Close tab" logic still always valid, but
// "Fit graph" needs a graph).
struct Command {
  std::string label;          // e.g. "View: Toggle dark theme"
  std::function<void()> run;
  bool enabled = true;
};

// Build the live action list from current App state. Rebuilt each frame so
// toggles/labels and per-tab items are always current (cheap: a few dozen items).
std::vector<Command> build_commands(App& app) {
  std::vector<Command> cmds;
  auto add = [&](std::string label, std::function<void()> fn, bool en = true) {
    cmds.push_back({std::move(label), std::move(fn), en});
  };

  ViewState& v = app.view();
  const bool has_model = app.session().model() != nullptr;
  const bool has_graph = app.session().has_graph();

  // File.
  add("File: Open model...", [&app] { app.open_file_dialog(); });
  add("File: New tab", [&app] { app.new_tab(); });
  add("File: Close current tab",
      [&app] { app.close_tab(app.active_tab()); }, app.tab_count() > 0);

  // View toggles (reflect current value in the label so the palette doubles as a
  // status readout).
  add(std::string("View: Theme -> ") + (v.dark_theme ? "Light" : "Dark"),
      [&app, &v] { app.set_theme(!v.dark_theme); });
  add(std::string("View: Minimap ") + (v.show_minimap ? "OFF" : "ON"),
      [&app, &v] { v.show_minimap = !v.show_minimap; app.save_prefs(); });
  add(std::string("View: Hide constant edges ") + (v.hide_const_edges ? "OFF" : "ON"),
      [&v] { v.hide_const_edges = !v.hide_const_edges; });
  add(std::string("View: Cost heatmap ") + (v.cost_heatmap ? "OFF" : "ON"),
      [&app, &v] { v.cost_heatmap = !v.cost_heatmap; app.save_prefs(); });
  add(std::string("View: Model diff panel ") + (v.diff_panel_open ? "OFF" : "ON"),
      [&v] { v.diff_panel_open = !v.diff_panel_open; });
  add(std::string("View: Plugins panel ") + (v.show_plugins ? "OFF" : "ON"),
      [&v] { v.show_plugins = !v.show_plugins; });
  add(std::string("View: Search overlay ") + (v.search_open ? "OFF" : "ON"),
      [&v] { v.search_open = !v.search_open; });

  // Graph navigation.
  add("Graph: Fit to view", [&v] { v.request_fit = true; }, has_graph);
  add("Graph: Reset camera (Home)",
      [&v] { v.cam = Camera{}; v.animating = false; }, has_model);
  add("Graph: Reload plugins", [&app] { app.reload_plugins(); });

  // Export.
  add("Export: Save view as PNG...", [&app] { app.export_view_dialog(); },
      has_model);

  // Switch-to-tab items (#62): one per open tab except the active one.
  for (size_t i = 0; i < app.tab_count(); ++i) {
    if (i == app.active_tab()) continue;
    add("Tab: Switch to " + app.tab_title(i),
        [&app, i] { app.switch_tab(i); });
  }
  return cmds;
}

}  // namespace

void draw_command_palette(App& app) {
  bool& open = app.command_palette_open();
  if (!open) return;

  // Persistent-across-frames palette state (single palette, main thread).
  static char query[128] = {0};
  static int selected = 0;
  static bool was_open = false;

  // On the frame it first opens: clear the query, reset selection, and request
  // keyboard focus into the input next frame.
  const bool just_opened = open && !was_open;
  was_open = open;
  if (just_opened) {
    query[0] = '\0';
    selected = 0;
  }

  // Center a fixed-width overlay near the top third of the viewport.
  ImGuiViewport* vp = ImGui::GetMainViewport();
  const float w = 560.0f;
  ImVec2 pos(vp->WorkPos.x + (vp->WorkSize.x - w) * 0.5f,
             vp->WorkPos.y + vp->WorkSize.y * 0.18f);
  ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(w, 0), ImGuiCond_Always);

  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoCollapse;

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
  if (!ImGui::Begin("##command_palette", nullptr, flags)) {
    ImGui::End();
    ImGui::PopStyleVar();
    return;
  }

  // Build + fuzzy-rank the actions against the current query. to_lower/fuzzy_score
  // are the SAME matcher the model search bar uses (engine/SearchIndex.h).
  std::vector<Command> cmds = build_commands(app);
  const std::string q = to_lower(query);

  // Rank: keep only matches (or everything on empty query), best score first.
  struct Ranked { int cmd; int score; };
  std::vector<Ranked> ranked;
  ranked.reserve(cmds.size());
  for (int i = 0; i < static_cast<int>(cmds.size()); ++i) {
    if (q.empty()) {
      ranked.push_back({i, 0});
      continue;
    }
    int s = fuzzy_score(q, to_lower(cmds[i].label));  // fuzzy_score wants lowered inputs
    if (s >= 0) ranked.push_back({i, s});
  }
  // Stable, best-first (simple insertion since the list is tiny).
  for (size_t i = 1; i < ranked.size(); ++i) {
    Ranked key = ranked[i];
    size_t j = i;
    while (j > 0 && ranked[j - 1].score < key.score) {
      ranked[j] = ranked[j - 1];
      --j;
    }
    ranked[j] = key;
  }
  if (selected >= static_cast<int>(ranked.size())) selected = 0;
  if (selected < 0) selected = 0;

  // Keyboard nav (before drawing the input so arrows are not eaten by it).
  const int n = static_cast<int>(ranked.size());
  bool moved = false;
  if (n > 0) {
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, true)) { selected = (selected + 1) % n; moved = true; }
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, true)) { selected = (selected + n - 1) % n; moved = true; }
  }
  // Auto-scroll to the selected row only when it moved (or on open) — computed
  // once here rather than re-testing the arrow keys inside the per-row loop.
  const bool scroll_to_sel = moved || just_opened;

  // The input field. Focus it on open. Enter runs the selection.
  if (just_opened) ImGui::SetKeyboardFocusHere();
  ImGui::SetNextItemWidth(-FLT_MIN);
  const bool entered = ImGui::InputTextWithHint(
      "##palette_query", "Type a command...  (Enter to run, Esc to close)", query,
      sizeof(query), ImGuiInputTextFlags_EnterReturnsTrue);

  int to_run = -1;
  if (entered && n > 0) to_run = ranked[selected].cmd;

  // Results list (scroll region, bounded height).
  ImGui::BeginChild("##palette_results", ImVec2(0, 320), true);
  for (int r = 0; r < n; ++r) {
    const Command& c = cmds[ranked[r].cmd];
    const bool is_sel = (r == selected);
    if (!c.enabled) ImGui::BeginDisabled();
    if (ImGui::Selectable(c.label.c_str(), is_sel) && c.enabled) {
      to_run = ranked[r].cmd;
    }
    if (!c.enabled) ImGui::EndDisabled();
    // Keep the keyboard-selected row visible when the selection just moved.
    if (is_sel && scroll_to_sel) ImGui::SetScrollHereY(0.5f);
  }
  if (n == 0) ImGui::TextDisabled("(no matching commands)");
  ImGui::EndChild();

  ImGui::End();
  ImGui::PopStyleVar();

  // Run AFTER End() so the action (which may open dialogs / mutate tabs) is not
  // sandwiched inside the palette window scope. Close the palette on run.
  if (to_run >= 0 && cmds[to_run].enabled) {
    cmds[to_run].run();
    open = false;
    was_open = false;
  }
}

}  // namespace netvis
