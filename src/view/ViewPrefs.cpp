// SPDX-License-Identifier: Apache-2.0
// view/ViewPrefs.cpp — view_prefs.json I/O. See ViewPrefs.h for the contract
// (what is a preference and what is not, and why this file is in netvis_core).
//
// The on-disk format is UNCHANGED by #151: same filename, same keys, same
// shapes. This file is the pre-existing App::save_prefs/load_prefs body moved
// out of the GUI target so it can be tested, not a new format. The format is
// append-only in practice — releases have added keys and never removed one —
// so every read is guarded individually and a file written by an older NetVis
// must keep loading.
#include "view/ViewPrefs.h"

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "engine/LayoutCache.h"

namespace netvis {

namespace {

nlohmann::json rgba_to_json(const Rgba8& c) {
  return nlohmann::json::array({c.r, c.g, c.b});
}

Rgba8 rgba_from_json(const nlohmann::json& j, Rgba8 fallback) {
  if (!j.is_array() || j.size() < 3) return fallback;
  auto byte = [](const nlohmann::json& e, uint8_t f) -> uint8_t {
    if (!e.is_number_integer() && !e.is_number_unsigned()) return f;
    int64_t v = e.get<int64_t>();
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    return static_cast<uint8_t>(v);
  };
  return Rgba8{byte(j[0], fallback.r), byte(j[1], fallback.g),
               byte(j[2], fallback.b), 255};
}

GradientPreset preset_from_name(const std::string& s) {
  for (int i = 0; i < kGradientPresetCount; ++i) {
    auto p = static_cast<GradientPreset>(i);
    if (s == gradient_preset_name(p)) return p;
  }
  return GradientPreset::Viridis;
}

// Read one boolean key, leaving `dst` alone when the key is missing or is not a
// boolean. Every persisted flag goes through this so "absent means keep the
// current default" is one rule in one place rather than fifteen copies of an
// `if (contains && is_boolean)` that can each be got subtly wrong.
void read_bool(const nlohmann::json& j, const char* key, bool& dst) {
  if (j.contains(key) && j[key].is_boolean()) dst = j[key].get<bool>();
}

// UI-scale bounds. Duplicated from PreferencesPanel.cpp's kUiScaleMin/Max, which
// are file-local to that GUI translation unit — this file cannot include a view
// header without leaving netvis_core (ViewPrefs.h).
constexpr float kUiScaleMin = 0.75f;
constexpr float kUiScaleMax = 2.0f;

// Whether `p` exists. Any error (a permissions problem, a dangling link) counts as
// "no": the notice this feeds is a courtesy, and a probe that cannot tell must not
// invent an upgrade.
bool path_exists(const std::filesystem::path& p) {
  std::error_code ec;
  return std::filesystem::exists(p, ec) && !ec;
}

}  // namespace

std::string view_prefs_file_path() {
  return layout_cache_dir() + "/view_prefs.json";
}

void save_view_prefs(const ViewPrefs& p) {
  const HeatmapGradient& g = p.heatmap_gradient;
  nlohmann::json j;
  j["dark_theme"] = p.dark_theme;
  j["show_minimap"] = p.show_minimap;
  j["show_layer_bands"] = p.show_layer_bands;  // #20 (v0.8.1)
  j["edge_tooltips"] = p.edge_tooltips;        // #18 (v0.8.1)
  j["cost_heatmap"] = p.cost_heatmap;
  j["heatmap_log_scale"] = p.heatmap_log_scale;
  j["heatmap_metric"] = heatmap_metric_name(p.heatmap_metric);
  j["gradient_preset"] = gradient_preset_name(g.preset);
  j["gradient_reverse"] = g.reverse;
  j["gradient_low"] = rgba_to_json(g.low);
  j["gradient_mid"] = rgba_to_json(g.mid);
  j["gradient_high"] = rgba_to_json(g.high);
  // #11: per-plugin enable overrides (empty object if the user changed nothing).
  j["plugins"] = p.plugins.to_json();
  j["edge_routing"] = p.edge_routing;  // #22 (v0.9.0)
  j["wheel_mode"] = wheel_mode_name(p.wheel_mode);  // #158
  // v0.9.4: the settings a user sets once and expects to survive a restart. The
  // WINDOW toggles (show_preferences/show_shortcuts/show_about) are deliberately
  // NOT here — a settings window that reopens itself every launch is a nuisance,
  // not a restored preference.
  j["accessible_palette"] = p.accessible_palette;  // #104
  j["ui_scale"] = p.ui_scale;                      // #104
  j["restore_session"] = p.restore_session;        // #103
  // #4/#30 (v0.8.3): custom roofline ridge + named machine profiles.
  if (p.custom_ridge > 0.0) j["custom_ridge"] = p.custom_ridge;
  if (!p.machine_profiles.empty()) {
    nlohmann::json profs = nlohmann::json::array();
    for (const auto& [name, ridge] : p.machine_profiles)
      profs.push_back({{"name", name}, {"ridge", ridge}});
    j["machine_profiles"] = profs;
  }
  std::ofstream f(view_prefs_file_path());
  // `replace` for the same reason as save_recent: machine-profile names are free
  // text the user can paste into, so strict UTF-8 could throw out of a routine
  // preference save.
  if (f) f << j.dump(2, ' ', false, nlohmann::json::error_handler_t::replace);
}

bool stamp_wheel_mode(WheelMode mode) {
  const std::string path = view_prefs_file_path();
  nlohmann::json j = nlohmann::json::object();
  {
    std::ifstream in(path);
    if (in) {
      // Merge into what is there, so every key the user already has (and any this
      // build does not know) survives untouched.
      try {
        in >> j;
      } catch (...) {
        return false;  // malformed: leave it, exactly as wheel_default_action says
      }
      if (!j.is_object()) return false;
    } else if (path_exists(path)) {
      return false;  // present but unreadable: not a fresh install, so not ours
    }
  }
  j["wheel_mode"] = wheel_mode_name(mode);
  std::ofstream out(path);
  if (!out) return false;
  out << j.dump(2, ' ', false, nlohmann::json::error_handler_t::replace);
  return static_cast<bool>(out);
}

bool has_prior_user_data(const std::string& cache_dir) {
  namespace fs = std::filesystem;
  // An empty path would make "recent.json" resolve against the working directory.
  if (cache_dir.empty()) return false;
  const fs::path dir(cache_dir);
  // The two small files NetVis keeps next to the prefs. recent.json is written the
  // first time a model is opened; session.json by the opt-in session restore.
  if (path_exists(dir / "recent.json") || path_exists(dir / "session.json"))
    return true;
  // A cached layout (`<hash>_<hash>.nvl`) means a model was opened here too, even
  // if recent.json was since deleted. The scan stops at the first hit, so a cache
  // with thousands of layouts costs one directory entry, not one per layout.
  std::error_code ec;
  fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec);
  if (ec) return false;
  for (const fs::directory_iterator end; it != end; it.increment(ec)) {
    if (it->path().extension() == ".nvl") return true;
  }
  return false;  // also reached when an increment fails: the iterator becomes end
}

WheelDefaultAction wheel_default_action(const ViewPrefsLoadInfo& info,
                                        bool prior_user_data) {
  // A usable file that already says what the wheel does: nothing changed for them.
  if (info.file_read && info.wheel_mode_present) return WheelDefaultAction::None;
  // A usable file from before wheel_mode existed, or with a bad wheel_mode value:
  // the user has been here, and the wheel used to zoom.
  if (info.file_read) return WheelDefaultAction::Notify;
  // A file that exists but could not be used (malformed, not an object). Leave it
  // alone: rewriting it at startup would destroy whatever the user was editing, and
  // the first preference they change rewrites it anyway.
  if (info.file_present) return WheelDefaultAction::None;
  // No prefs file. view_prefs.json is only written when a preference CHANGES, so a
  // user who never touched a setting has none, and their wheel still switches from
  // zoom to pan. Other NetVis files say whether they have been here before.
  return prior_user_data ? WheelDefaultAction::Notify : WheelDefaultAction::Stamp;
}

ViewPrefs load_view_prefs(const ViewPrefs& base) {
  return load_view_prefs(base, nullptr);
}

ViewPrefs load_view_prefs(const ViewPrefs& base, ViewPrefsLoadInfo* info) {
  // Reported through locals and published once at the end, so every early return
  // and the catch below leave `info` at "nothing read".
  if (info) *info = ViewPrefsLoadInfo{};
  ViewPrefs p = base;
  std::ifstream f(view_prefs_file_path());
  if (!f) {
    // Missing, or present but unreadable (permissions): only the first is a fresh
    // install, and an unreadable file must not be mistaken for it.
    if (info) info->file_present = path_exists(view_prefs_file_path());
    return p;
  }
  if (info) info->file_present = true;
  try {
    nlohmann::json j;
    f >> j;
    if (!j.is_object()) return base;
    ViewPrefsLoadInfo seen;
    seen.file_present = true;
    seen.file_read = true;

    read_bool(j, "dark_theme", p.dark_theme);
    read_bool(j, "show_minimap", p.show_minimap);
    read_bool(j, "show_layer_bands", p.show_layer_bands);  // #20
    read_bool(j, "edge_tooltips", p.edge_tooltips);        // #18
    read_bool(j, "cost_heatmap", p.cost_heatmap);
    read_bool(j, "heatmap_log_scale", p.heatmap_log_scale);
    read_bool(j, "accessible_palette", p.accessible_palette);  // #104
    read_bool(j, "restore_session", p.restore_session);        // #103

    if (j.contains("heatmap_metric") && j["heatmap_metric"].is_string())
      p.heatmap_metric =
          heatmap_metric_from_name(j["heatmap_metric"].get<std::string>().c_str());

    HeatmapGradient& g = p.heatmap_gradient;
    if (j.contains("gradient_preset") && j["gradient_preset"].is_string()) {
      GradientPreset preset = preset_from_name(j["gradient_preset"].get<std::string>());
      gradient_set_preset(g, preset);  // fills stops for a built-in preset
    }
    read_bool(j, "gradient_reverse", g.reverse);
    // Only a Custom gradient carries its own stops; for a built-in preset the
    // preset's stops (just filled by gradient_set_preset) are authoritative, so a
    // "Viridis" tag always shows Viridis colors and a future change to the preset
    // constants isn't pinned to a stale persisted copy.
    if (g.preset == GradientPreset::Custom) {
      if (j.contains("gradient_low")) g.low = rgba_from_json(j["gradient_low"], g.low);
      if (j.contains("gradient_mid")) g.mid = rgba_from_json(j["gradient_mid"], g.mid);
      if (j.contains("gradient_high"))
        g.high = rgba_from_json(j["gradient_high"], g.high);
    }

    // #11: per-plugin enable overrides (guarded; never prunes, ignores non-bool).
    if (j.contains("plugins") && j["plugins"].is_object())
      p.plugins.load_json(j["plugins"]);

    if (j.contains("edge_routing") && j["edge_routing"].is_number_integer()) {
      const int er = j["edge_routing"].get<int>();
      if (er >= 0 && er <= 2) p.edge_routing = er;
    }
    // #158: only an exact "pan" / "zoom" string counts. Anything else (a number,
    // null, "Zoom", "scroll") keeps the base and is reported as absent.
    if (j.contains("wheel_mode") && j["wheel_mode"].is_string()) {
      WheelMode wm = p.wheel_mode;
      if (wheel_mode_from_name(j["wheel_mode"].get<std::string>(), wm)) {
        p.wheel_mode = wm;
        seen.wheel_mode_present = true;
      }
    }
    // CLAMPED on load, not merely on edit. A persisted 0, a negative, or a NaN
    // would render an unusable window — and the setting that caused it lives
    // inside that window, so the user could not reach it to undo the damage.
    // Written as !(in range) so NaN is rejected too, where a naive comparison
    // would let it through. This is the one key that does NOT fall back to
    // `base` on a bad value: base is whatever the running app currently has,
    // and if that is itself the unusable scale the file just wrote, keeping it
    // would preserve the very state this clamp exists to escape.
    if (j.contains("ui_scale") && j["ui_scale"].is_number()) {
      const float sc = j["ui_scale"].get<float>();
      p.ui_scale = !(sc >= kUiScaleMin && sc <= kUiScaleMax) ? 1.0f : sc;
    }
    if (j.contains("custom_ridge") && j["custom_ridge"].is_number())
      p.custom_ridge = j["custom_ridge"].get<double>();
    if (j.contains("machine_profiles") && j["machine_profiles"].is_array()) {
      p.machine_profiles.clear();
      for (const auto& e : j["machine_profiles"]) {
        if (e.is_object() && e.contains("name") && e["name"].is_string() &&
            e.contains("ridge") && e["ridge"].is_number())
          p.machine_profiles.emplace_back(e["name"].get<std::string>(),
                                          e["ridge"].get<double>());
      }
    }
    if (info) *info = seen;
    return p;
  } catch (...) {
    // Corrupt prefs -> keep what the caller already had. Returning `base` rather
    // than the half-filled `p` matters: a document that parses far enough to set
    // three fields and then throws would otherwise leave the user with a
    // partially-applied set of preferences they never chose.
    return base;
  }
}

}  // namespace netvis
