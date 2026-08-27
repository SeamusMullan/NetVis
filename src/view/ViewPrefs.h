// view/ViewPrefs.h — the user-PREFERENCE half of ViewState, and its file (#151).
//
// WHY THIS EXISTS (#151): ViewState is per-tab, and App::new_tab() default-
// constructs one. So every "Open" past the first — which opens a new tab (#62) —
// dropped the user back to the shipped defaults: light theme went dark again, the
// minimap came back, edge routing reverted to Bezier. load_prefs() ran once at
// startup and wrote into the tab that existed THEN, so nothing carried the choices
// forward. The fix needs a name for "the subset of ViewState that belongs to the
// USER, not to the model in front of them", and this struct is that name.
//
// WHAT IS IN HERE is exactly what view_prefs.json already persisted before #151 —
// no more. That file is the pre-existing definition of "preference", and widening
// it silently would change what survives a restart, which is a different decision
// from the one this issue asks for. Deliberately ABSENT, each for its own reason:
//   - camera, selection, search/table/attr filters, nav state, collapse: per-MODEL
//     view state. Indices and queries mean nothing against a different graph.
//   - show_critical_path: App.h calls it a transient analysis overlay, and it is.
//   - show_preferences / show_shortcuts / show_about: App.h's reasoning stands —
//     a settings window that reopens itself every launch is a nuisance.
//   - hide_const_edges, show_search_results, diff_panel_open, show_plugins,
//     nav->show_legend / show_bookmarks: View-menu toggles that were never
//     persisted. Whether they are preferences is a UX call nobody has made, and
//     guessing would change behaviour under cover of a bug fix.
//
// Lives in netvis_core (see CMakeLists.txt's explicit source list) for the same
// reason SessionStore.cpp and PluginPrefs do: netvis_tests links netvis_core ONLY,
// so persistence that lives in a GUI translation unit is persistence that cannot
// be tested. Nothing here includes ImGui or a view header, and it must stay that
// way. The ViewState <-> ViewPrefs mapping is what needs ImGui, so THAT stays in
// App.cpp (the same split ViewHistory.h uses for capture_view/apply_view).
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "engine/HeatmapGradient.h"
#include "engine/plugin/PluginPrefs.h"

namespace netvis {

// One field per view_prefs.json key. The initializers here are a SECOND copy of
// ViewState's, which is a drift hazard — so load_view_prefs() takes the live
// values as its fallback base and App::load_prefs() passes the real ViewState in.
// That makes App.h the single source of default truth at runtime; these exist so
// a default-constructed ViewPrefs is still sane for a caller that has no
// ViewState to hand (the tests).
struct ViewPrefs {
  bool dark_theme = true;
  bool show_minimap = true;
  bool show_layer_bands = false;
  bool edge_tooltips = true;
  bool cost_heatmap = false;
  bool heatmap_log_scale = true;
  HeatmapMetric heatmap_metric = HeatmapMetric::Flops;
  HeatmapGradient heatmap_gradient;
  int edge_routing = 0;
  bool accessible_palette = false;
  float ui_scale = 1.0f;
  bool restore_session = false;
  double custom_ridge = 0.0;
  std::vector<std::pair<std::string, double>> machine_profiles;

  // Per-plugin enable overrides (#11). Not a ViewState field — it is App-owned —
  // but it shares view_prefs.json, and a writer that did not round-trip it would
  // erase the user's plugin choices the first time any other preference changed.
  plugin::PluginEnableSet plugins;
};

// Path of the preferences file, next to session.json in layout_cache_dir().
std::string view_prefs_file_path();

// Write. Best-effort and silent, exactly like save_session/save_recent: failing
// to persist a preference must never interrupt the user.
void save_view_prefs(const ViewPrefs& p);

// Read view_prefs.json OVER `base`. Every key that is absent, of the wrong type,
// or out of range leaves the corresponding field of `base` untouched — so an
// older prefs file written before a key existed loads as "keep the current
// default" rather than as a zeroed field, and a corrupt file degrades to `base`
// entirely instead of throwing across the view/engine boundary.
ViewPrefs load_view_prefs(const ViewPrefs& base);
inline ViewPrefs load_view_prefs() { return load_view_prefs(ViewPrefs{}); }

}  // namespace netvis
