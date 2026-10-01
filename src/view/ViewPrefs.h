// SPDX-License-Identifier: Apache-2.0
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
// from the one this issue asks for. The one deliberate addition since is
// `wheel_mode` (#158): whether the scroll wheel pans or zooms the canvas. It is
// the first key added since #151, and it is a preference rather than per-model
// state because it describes the user's hardware and habits, not the model in
// front of them. Deliberately ABSENT, each for its own reason:
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

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "engine/HeatmapGradient.h"
#include "engine/plugin/PluginPrefs.h"
#include "view/CanvasInput.h"  // WheelMode (#158); ImGui-free

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
  // #158: what a plain scroll does. Pan is the Netron default.
  WheelMode wheel_mode = WheelMode::Pan;
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

// What load_view_prefs() learned about the file itself, beyond the values. Used by
// App::load_prefs for the one-time #158 upgrade notice: a file that exists and
// parsed, but has no `wheel_mode`, was written by a release where the wheel always
// zoomed, and that user deserves to be told the default changed.
struct ViewPrefsLoadInfo {
  // The file exists, whether or not it could be read or parsed. Distinguishes "no
  // file" (a fresh install, or a user who never changed a setting) from "a file we
  // could not use", which must not be overwritten at startup.
  bool file_present = false;
  // The file was opened AND parsed as a JSON object. False for a missing,
  // unreadable, malformed or non-object file.
  bool file_read = false;
  // `wheel_mode` was a string that wheel_mode_from_name() accepts. A wrong-typed,
  // unknown or wrong-case value is reported as absent, so it is rewritten too.
  bool wheel_mode_present = false;
};

// Whether the cache directory shows NetVis has been used here before: a
// recent.json, a session.json, or a cached layout (`*.nvl`). Needed because
// view_prefs.json is written ONLY when a preference changes (never at startup), so
// a long-time user who never touched a setting has no prefs file at all and looks
// exactly like a fresh install through it. Only the cheap checks above; it never
// reads or parses any of them. `cache_dir` is a parameter so tests can use a temp
// directory; production passes layout_cache_dir().
bool has_prior_user_data(const std::string& cache_dir);

// What App::load_prefs should do about the #158 change of the wheel default.
enum class WheelDefaultAction : uint8_t {
  // Nothing to say and nothing to write.
  None,
  // An existing user, whose wheel used to zoom and now pans: show the one-time
  // notice and save the prefs, which writes `wheel_mode` so it appears only once.
  Notify,
  // A fresh install: no notice (nothing changed for them), but save the prefs so
  // the NEXT launch finds a file with `wheel_mode` and does not mistake a user who
  // has since opened a model (and so has a recent.json) for an upgrade.
  Stamp,
};

// Pure decision from what load_view_prefs() reported and whether
// has_prior_user_data() found anything. `prior_user_data` only matters when there
// is no prefs file, so a caller may pass false without probing when
// `info.file_present`.
WheelDefaultAction wheel_default_action(const ViewPrefsLoadInfo& info,
                                        bool prior_user_data);

// Read view_prefs.json OVER `base`. Every key that is absent, of the wrong type,
// or out of range leaves the corresponding field of `base` untouched — so an
// older prefs file written before a key existed loads as "keep the current
// default" rather than as a zeroed field, and a corrupt file degrades to `base`
// entirely instead of throwing across the view/engine boundary.
ViewPrefs load_view_prefs(const ViewPrefs& base);
// Same, and reports what it found in `info` (may be null). There is deliberately
// no default argument: `load_view_prefs()` below must stay unambiguous.
ViewPrefs load_view_prefs(const ViewPrefs& base, ViewPrefsLoadInfo* info);
inline ViewPrefs load_view_prefs() { return load_view_prefs(ViewPrefs{}); }

}  // namespace netvis
