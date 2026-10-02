// SPDX-License-Identifier: Apache-2.0
// tests/test_view_prefs.cpp — #151 view-preference persistence.
//
// What this file can and cannot reach: ViewPrefs and view_prefs.json live in
// netvis_core (CMakeLists.txt lists src/view/ViewPrefs.cpp explicitly, next to
// SessionStore.cpp and CategoryStyle.cpp), so everything below exercises the
// REAL reader and writer. The other half of #151 — App::new_tab() carrying a
// ViewPrefs into the tab it just created — is NOT reachable from here: it needs
// ViewState, which is defined in view/App.h and drags in ImGui, and netvis_tests
// links netvis_core only. What is tested here is the contract that half depends
// on; the carry itself is a two-line mapping verified by building the GUI target.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

#include <nlohmann/json.hpp>

#include "view/ViewPrefs.h"

using namespace netvis;

namespace {

// view_prefs_file_path() resolves to the user's real cache dir — there is no
// override hook — so back up whatever is there and restore it afterwards rather
// than clobbering a developer's actual preferences. Same shape as
// test_session_store.cpp's SessionFileBackup, for the same reason.
struct PrefsFileBackup {
  std::string path = view_prefs_file_path();
  bool existed = false;
  std::string content;

  PrefsFileBackup() {
    std::ifstream f(path, std::ios::binary);
    if (f) {
      existed = true;
      content.assign(std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>());
    }
    std::remove(path.c_str());
  }
  ~PrefsFileBackup() {
    if (existed) {
      std::ofstream f(path, std::ios::binary | std::ios::trunc);
      f << content;
    } else {
      std::remove(path.c_str());
    }
  }
};

// Writes exactly `text`, bypassing save_view_prefs() — used for the old-file and
// corrupt-file cases a well-behaved writer never produces.
void write_raw_prefs_file(const std::string& text) {
  std::ofstream f(view_prefs_file_path(), std::ios::binary | std::ios::trunc);
  f << text;
}

// A ViewPrefs with every field moved off its default, so a round-trip that
// silently drops a field fails instead of passing by coincidence.
ViewPrefs all_non_default() {
  ViewPrefs p;
  p.dark_theme = false;
  p.show_minimap = false;
  p.show_layer_bands = true;
  p.edge_tooltips = false;
  p.cost_heatmap = true;
  p.heatmap_log_scale = false;
  p.heatmap_metric = HeatmapMetric::ArithIntensity;
  gradient_set_preset(p.heatmap_gradient, GradientPreset::Magma);
  p.heatmap_gradient.reverse = true;
  p.edge_routing = 2;
  p.wheel_mode = WheelMode::Zoom;  // #158: the default is Pan
  p.accessible_palette = true;
  p.ui_scale = 1.25f;
  p.restore_session = true;
  p.custom_ridge = 37.5;
  p.machine_profiles = {{"M2 Max", 12.5}, {"A100", 141.0}};
  return p;
}

}  // namespace

TEST_CASE("ViewPrefs: save then load round-trips every persisted preference") {
  PrefsFileBackup backup;

  const ViewPrefs in = all_non_default();
  save_view_prefs(in);
  const ViewPrefs out = load_view_prefs();

  CHECK(out.dark_theme == false);
  CHECK(out.show_minimap == false);
  CHECK(out.show_layer_bands == true);
  CHECK(out.edge_tooltips == false);
  CHECK(out.cost_heatmap == true);
  CHECK(out.heatmap_log_scale == false);
  CHECK(out.heatmap_metric == HeatmapMetric::ArithIntensity);
  CHECK(out.heatmap_gradient.preset == GradientPreset::Magma);
  CHECK(out.heatmap_gradient.reverse == true);
  CHECK(out.edge_routing == 2);
  CHECK(out.wheel_mode == WheelMode::Zoom);
  CHECK(out.accessible_palette == true);
  CHECK(out.ui_scale == doctest::Approx(1.25f));
  CHECK(out.restore_session == true);
  CHECK(out.custom_ridge == doctest::Approx(37.5));
  REQUIRE(out.machine_profiles.size() == 2);
  CHECK(out.machine_profiles[0].first == "M2 Max");
  CHECK(out.machine_profiles[1].second == doctest::Approx(141.0));
}

TEST_CASE("ViewPrefs: a Custom gradient's stops survive, a preset's are re-derived") {
  PrefsFileBackup backup;

  ViewPrefs in;
  in.heatmap_gradient.preset = GradientPreset::Custom;
  in.heatmap_gradient.low = Rgba8{1, 2, 3, 255};
  in.heatmap_gradient.mid = Rgba8{4, 5, 6, 255};
  in.heatmap_gradient.high = Rgba8{7, 8, 9, 255};
  save_view_prefs(in);

  const ViewPrefs out = load_view_prefs();
  CHECK(out.heatmap_gradient.preset == GradientPreset::Custom);
  CHECK(out.heatmap_gradient.low == Rgba8{1, 2, 3, 255});
  CHECK(out.heatmap_gradient.high == Rgba8{7, 8, 9, 255});

  // A built-in preset takes its stops from the preset table, never from the
  // persisted copy — so a change to the table is not pinned to a stale file.
  ViewPrefs preset_in;
  gradient_set_preset(preset_in.heatmap_gradient, GradientPreset::Grayscale);
  save_view_prefs(preset_in);
  Rgba8 lo, mid, hi;
  gradient_preset_stops(GradientPreset::Grayscale, lo, mid, hi);
  const ViewPrefs preset_out = load_view_prefs();
  CHECK(preset_out.heatmap_gradient.preset == GradientPreset::Grayscale);
  CHECK(preset_out.heatmap_gradient.low == lo);
  CHECK(preset_out.heatmap_gradient.high == hi);
}

TEST_CASE("ViewPrefs: a missing file leaves the caller's base untouched") {
  PrefsFileBackup backup;
  std::remove(view_prefs_file_path().c_str());

  const ViewPrefs base = all_non_default();
  const ViewPrefs out = load_view_prefs(base);

  CHECK(out.dark_theme == base.dark_theme);
  CHECK(out.edge_routing == base.edge_routing);
  CHECK(out.ui_scale == doctest::Approx(base.ui_scale));
  CHECK(out.machine_profiles.size() == base.machine_profiles.size());
}

TEST_CASE("ViewPrefs: keys absent from an older file degrade to the base default") {
  PrefsFileBackup backup;
  // A prefs file as an early v0.3.x NetVis would have written it: the theme and
  // the gradient existed, none of the toggles added since did. Every key added
  // after it must load as "keep what the caller has", not as a zeroed field.
  write_raw_prefs_file(R"({
    "dark_theme": false,
    "gradient_preset": "Magma"
  })");

  const ViewPrefs base = all_non_default();
  const ViewPrefs out = load_view_prefs(base);

  CHECK(out.dark_theme == false);                       // present: file wins
  CHECK(out.heatmap_gradient.preset == GradientPreset::Magma);
  CHECK(out.show_layer_bands == base.show_layer_bands);  // absent: base wins
  CHECK(out.edge_tooltips == base.edge_tooltips);
  CHECK(out.edge_routing == base.edge_routing);
  CHECK(out.wheel_mode == base.wheel_mode);  // #158
  CHECK(out.accessible_palette == base.accessible_palette);
  CHECK(out.restore_session == base.restore_session);
  CHECK(out.ui_scale == doctest::Approx(base.ui_scale));
  CHECK(out.custom_ridge == doctest::Approx(base.custom_ridge));
  CHECK(out.machine_profiles.size() == base.machine_profiles.size());
}

TEST_CASE("ViewPrefs: a wrong-typed key degrades to the base, never throws") {
  PrefsFileBackup backup;
  write_raw_prefs_file(R"({
    "dark_theme": "yes",
    "show_minimap": 3,
    "edge_routing": "orthogonal",
    "heatmap_metric": 7,
    "machine_profiles": {"name": "not an array"},
    "plugins": 12
  })");

  ViewPrefs base;
  base.dark_theme = false;
  base.show_minimap = false;
  base.edge_routing = 1;
  base.heatmap_metric = HeatmapMetric::Params;
  base.machine_profiles = {{"kept", 1.0}};

  const ViewPrefs out = load_view_prefs(base);
  CHECK(out.dark_theme == false);
  CHECK(out.show_minimap == false);
  CHECK(out.edge_routing == 1);
  CHECK(out.heatmap_metric == HeatmapMetric::Params);
  REQUIRE(out.machine_profiles.size() == 1);
  CHECK(out.machine_profiles[0].first == "kept");
}

TEST_CASE("ViewPrefs: malformed JSON degrades to the base, never crashes") {
  PrefsFileBackup backup;

  const ViewPrefs base = all_non_default();
  for (const char* text : {"{", "not json at all", "[1, 2, 3]", "",
                           R"({"dark_theme": true, "edge_routing": )"}) {
    write_raw_prefs_file(text);
    const ViewPrefs out = load_view_prefs(base);
    // Including dark_theme: a document that parses far enough to set a field and
    // then throws must not leave a half-applied set of preferences behind.
    CHECK(out.dark_theme == base.dark_theme);
    CHECK(out.edge_routing == base.edge_routing);
    CHECK(out.custom_ridge == doctest::Approx(base.custom_ridge));
  }
}

TEST_CASE("ViewPrefs: an out-of-range edge_routing is ignored") {
  PrefsFileBackup backup;
  for (const char* text : {R"({"edge_routing": -1})", R"({"edge_routing": 3})",
                           R"({"edge_routing": 99})"}) {
    write_raw_prefs_file(text);
    ViewPrefs base;
    base.edge_routing = 1;
    CHECK(load_view_prefs(base).edge_routing == 1);
  }
}

TEST_CASE("ViewPrefs: an unusable ui_scale is clamped back to 1.0, not kept") {
  PrefsFileBackup backup;
  // 0 / negative renders every window at zero size INCLUDING Preferences, so the
  // control that caused it becomes unreachable. This key deliberately does not
  // fall back to the base, which could itself be the unusable value.
  ViewPrefs base;
  base.ui_scale = 1.5f;
  for (const char* text : {R"({"ui_scale": 0})", R"({"ui_scale": -2.5})",
                           R"({"ui_scale": 40})", R"({"ui_scale": 0.1})"}) {
    write_raw_prefs_file(text);
    CHECK(load_view_prefs(base).ui_scale == doctest::Approx(1.0f));
  }
  // In range: honoured as written.
  write_raw_prefs_file(R"({"ui_scale": 1.75})");
  CHECK(load_view_prefs(base).ui_scale == doctest::Approx(1.75f));
}

TEST_CASE("ViewPrefs: plugin enable overrides round-trip alongside the view prefs") {
  PrefsFileBackup backup;
  // #11's set shares view_prefs.json. A writer that dropped it would erase the
  // user's plugin choices the first time any unrelated preference changed.
  ViewPrefs in;
  in.plugins.set("my_wasm_plugin", true);
  in.plugins.set("some_declarative", false);
  in.dark_theme = false;
  save_view_prefs(in);

  const ViewPrefs out = load_view_prefs();
  CHECK(out.plugins.has_explicit("my_wasm_plugin"));
  CHECK(out.plugins.effective("my_wasm_plugin", plugin::PluginKind::Wasm) == true);
  CHECK(out.plugins.effective("some_declarative",
                              plugin::PluginKind::Declarative) == false);
  CHECK(out.dark_theme == false);
}

TEST_CASE("ViewPrefs: a malformed machine_profiles entry drops only that entry") {
  PrefsFileBackup backup;
  write_raw_prefs_file(R"({
    "machine_profiles": [
      {"name": "good", "ridge": 10.0},
      {"name": "no ridge"},
      {"ridge": 5.0},
      "not an object",
      {"name": 42, "ridge": 5.0},
      {"name": "also good", "ridge": 20.0}
    ]
  })");

  const ViewPrefs out = load_view_prefs();
  REQUIRE(out.machine_profiles.size() == 2);
  CHECK(out.machine_profiles[0].first == "good");
  CHECK(out.machine_profiles[1].first == "also good");
  CHECK(out.machine_profiles[1].second == doctest::Approx(20.0));
}

// --- #158: wheel_mode ---------------------------------------------------------
//
// The key is new, so the interesting cases are the ones where it is not there or
// is wrong, and the ViewPrefsLoadInfo that App::load_prefs uses to decide whether
// to show the one-time "scrolling now pans" notice.

TEST_CASE("ViewPrefs: wheel_mode absent from an older file keeps the base and is reported") {
  PrefsFileBackup backup;
  // A file from a release where the wheel always zoomed: it exists and parses,
  // but has no wheel_mode.
  write_raw_prefs_file(R"({"dark_theme": false})");

  ViewPrefs base;  // wheel_mode defaults to Pan
  ViewPrefsLoadInfo info;
  const ViewPrefs out = load_view_prefs(base, &info);

  CHECK(out.wheel_mode == WheelMode::Pan);
  CHECK(info.file_read);
  CHECK_FALSE(info.wheel_mode_present);
  CHECK(out.dark_theme == false);  // the rest of the file still loads
}

TEST_CASE("ViewPrefs: wheel_mode zoom is read and reported present") {
  PrefsFileBackup backup;
  write_raw_prefs_file(R"({"wheel_mode": "zoom"})");

  ViewPrefsLoadInfo info;
  const ViewPrefs out = load_view_prefs(ViewPrefs{}, &info);
  CHECK(out.wheel_mode == WheelMode::Zoom);
  CHECK(info.file_read);
  CHECK(info.wheel_mode_present);

  write_raw_prefs_file(R"({"wheel_mode": "pan"})");
  ViewPrefs zoom_base;
  zoom_base.wheel_mode = WheelMode::Zoom;
  const ViewPrefs out2 = load_view_prefs(zoom_base, &info);
  CHECK(out2.wheel_mode == WheelMode::Pan);  // the file wins over the base
  CHECK(info.wheel_mode_present);
}

TEST_CASE("ViewPrefs: a wrong-typed or unknown wheel_mode degrades to the base") {
  PrefsFileBackup backup;
  ViewPrefs base;
  base.wheel_mode = WheelMode::Zoom;

  for (const char* text : {R"({"wheel_mode": 1})", R"({"wheel_mode": "scroll"})",
                           R"({"wheel_mode": "Zoom"})", R"({"wheel_mode": null})"}) {
    write_raw_prefs_file(text);
    ViewPrefsLoadInfo info;
    const ViewPrefs out = load_view_prefs(base, &info);
    INFO(text);
    CHECK(out.wheel_mode == WheelMode::Zoom);
    CHECK(info.file_read);
    // Reported absent, so the next save rewrites a valid value.
    CHECK_FALSE(info.wheel_mode_present);
  }
}

TEST_CASE("ViewPrefs: no file / malformed / non-object reports file_read=false") {
  PrefsFileBackup backup;
  const ViewPrefs base = all_non_default();  // wheel_mode Zoom, dark_theme false

  // `expect_present`: whether a file was there to open at all (a missing file is
  // not "present"; a malformed one is, but unusable).
  auto check_unread = [&](const char* label, bool expect_present) {
    // Start `info` dirty to prove load_view_prefs resets it rather than leaving
    // the caller's value in place.
    ViewPrefsLoadInfo info;
    info.file_present = !expect_present;
    info.file_read = true;
    info.wheel_mode_present = true;
    const ViewPrefs out = load_view_prefs(base, &info);
    INFO(label);
    CHECK(out.wheel_mode == base.wheel_mode);
    CHECK(out.dark_theme == base.dark_theme);
    CHECK(info.file_present == expect_present);
    CHECK_FALSE(info.file_read);
    CHECK_FALSE(info.wheel_mode_present);
  };

  std::remove(view_prefs_file_path().c_str());
  check_unread("missing file", false);
  write_raw_prefs_file("{not json");
  check_unread("malformed", true);
  write_raw_prefs_file("[1,2]");
  check_unread("non-object", true);
  // A document cut off after a wheel_mode key. nlohmann's `f >> j` parses the whole
  // document before any key is read, so this fails in the parser, the same path as
  // "{not json": it is NOT a throw part-way through reading keys. Every read after
  // the parse is type-guarded and cannot throw, so no file content reaches the
  // catch from there; the catch exists for the parser and for a future unguarded
  // read.
  write_raw_prefs_file(R"({"wheel_mode": "zoom", "dark_theme": )");
  check_unread("truncated document", true);
}

TEST_CASE("ViewPrefs: a usable file reports file_present too") {
  PrefsFileBackup backup;
  write_raw_prefs_file(R"({"dark_theme": false})");
  ViewPrefsLoadInfo info;
  load_view_prefs(ViewPrefs{}, &info);
  CHECK(info.file_present);
  CHECK(info.file_read);
}

// --- #158 upgrade notice: who counts as an existing user ------------------------
//
// view_prefs.json is written only when a preference changes, never at startup, so
// "has a prefs file" is not "has used NetVis": a long-time user who never changed
// a setting has none, and their wheel still switches from zoom to pan.

namespace {

// A fresh, empty directory under the system temp dir, removed on destruction. The
// real cache dir is never touched.
struct TempCacheDir {
  std::filesystem::path path;
  TempCacheDir() {
    static int counter = 0;
    path = std::filesystem::temp_directory_path() /
           ("nv_prior_data_" + std::to_string(counter++) + "_" +
            std::to_string(reinterpret_cast<uintptr_t>(this)));
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path);
  }
  ~TempCacheDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  void touch(const std::string& name) const {
    std::ofstream(path / name, std::ios::binary) << "x";
  }
  std::string str() const { return path.string(); }
};

}  // namespace

TEST_CASE("ViewPrefs: has_prior_user_data sees the files an existing user has") {
  {
    TempCacheDir d;
    CHECK_FALSE(has_prior_user_data(d.str()));  // a fresh install: an empty dir
  }
  {
    TempCacheDir d;
    d.touch("recent.json");  // written the first time a model is opened
    CHECK(has_prior_user_data(d.str()));
  }
  {
    TempCacheDir d;
    d.touch("session.json");
    CHECK(has_prior_user_data(d.str()));
  }
  {
    TempCacheDir d;
    d.touch("123456789_987654321.nvl");  // a cached layout, recent.json since deleted
    CHECK(has_prior_user_data(d.str()));
  }
}

TEST_CASE("ViewPrefs: has_prior_user_data ignores unrelated files and bad directories") {
  {
    TempCacheDir d;
    d.touch("view_prefs.json");  // the prefs file is the OTHER signal, not this one
    d.touch("notes.txt");
    d.touch("layout.nvl.tmp");  // extension is .tmp, not .nvl
    CHECK_FALSE(has_prior_user_data(d.str()));
  }
  {
    TempCacheDir d;
    // A directory that does not exist, and one that is a file, are "no evidence"
    // rather than an error.
    CHECK_FALSE(has_prior_user_data((d.path / "missing").string()));
    d.touch("plain_file");
    CHECK_FALSE(has_prior_user_data((d.path / "plain_file").string()));
  }
  CHECK_FALSE(has_prior_user_data(""));
}

TEST_CASE("ViewPrefs: wheel_default_action tells existing users once") {
  auto info = [](bool present, bool read, bool wheel) {
    ViewPrefsLoadInfo i;
    i.file_present = present;
    i.file_read = read;
    i.wheel_mode_present = wheel;
    return i;
  };

  // A file that already records the wheel mode: nothing changed for them.
  CHECK(wheel_default_action(info(true, true, true), false) == WheelDefaultAction::None);
  CHECK(wheel_default_action(info(true, true, true), true) == WheelDefaultAction::None);

  // A file from before wheel_mode (or with a bad value): they are told.
  CHECK(wheel_default_action(info(true, true, false), false) == WheelDefaultAction::Notify);
  CHECK(wheel_default_action(info(true, true, false), true) == WheelDefaultAction::Notify);

  // THE BUG: no prefs file, but other NetVis files exist. Before the fix this
  // user was never told and the wheel silently switched to pan.
  CHECK(wheel_default_action(info(false, false, false), true) == WheelDefaultAction::Notify);

  // No prefs file and nothing else: a fresh install. Not told (nothing changed),
  // but the prefs are written so the next launch is not mistaken for an upgrade.
  CHECK(wheel_default_action(info(false, false, false), false) == WheelDefaultAction::Stamp);

  // A file that exists but cannot be used is left alone, whatever else exists:
  // overwriting it at startup would destroy what the user was editing.
  CHECK(wheel_default_action(info(true, false, false), false) == WheelDefaultAction::None);
  CHECK(wheel_default_action(info(true, false, false), true) == WheelDefaultAction::None);
}

TEST_CASE("ViewPrefs: the notice fires once across launches (load, act, save, load)") {
  PrefsFileBackup backup;  // starts with no prefs file

  // One launch as App::load_prefs sees it: load, decide, and save when asked.
  // `cache` stands in for the directory holding recent.json and friends.
  auto launch = [&](const TempCacheDir& cache) {
    ViewPrefsLoadInfo info;
    const ViewPrefs p = load_view_prefs(ViewPrefs{}, &info);
    const bool prior = !info.file_present && has_prior_user_data(cache.str());
    const WheelDefaultAction act = wheel_default_action(info, prior);
    if (act != WheelDefaultAction::None) stamp_wheel_mode(p.wheel_mode);
    return act;
  };

  {
    // An upgrading user who never changed a setting: recent.json, no prefs file.
    TempCacheDir cache;
    cache.touch("recent.json");
    CHECK(launch(cache) == WheelDefaultAction::Notify);
    CHECK(launch(cache) == WheelDefaultAction::None);  // wheel_mode is on disk now
    CHECK(launch(cache) == WheelDefaultAction::None);
  }

  {
    // A fresh install that opens a model during its first session: the first
    // launch stamps the prefs, so the recent.json written afterwards never reads
    // as an upgrade.
    std::remove(view_prefs_file_path().c_str());
    TempCacheDir cache;
    CHECK(launch(cache) == WheelDefaultAction::Stamp);
    cache.touch("recent.json");  // the user opens a model
    CHECK(launch(cache) == WheelDefaultAction::None);
  }
}

TEST_CASE("ViewPrefs: a null info pointer is accepted") {
  PrefsFileBackup backup;
  write_raw_prefs_file(R"({"wheel_mode": "zoom"})");
  CHECK(load_view_prefs(ViewPrefs{}, nullptr).wheel_mode == WheelMode::Zoom);
  CHECK(load_view_prefs(ViewPrefs{}).wheel_mode == WheelMode::Zoom);
}

TEST_CASE("ViewPrefs: save writes wheel_mode so the upgrade notice fires once") {
  PrefsFileBackup backup;
  // A user changing any preference saves the whole file, which carries wheel_mode,
  // so the next launch must not show the notice.
  save_view_prefs(ViewPrefs{});

  ViewPrefsLoadInfo info;
  const ViewPrefs out = load_view_prefs(ViewPrefs{}, &info);
  CHECK(info.file_read);
  CHECK(info.wheel_mode_present);
  CHECK(out.wheel_mode == WheelMode::Pan);
}

// --- #158: the startup stamp writes ONLY wheel_mode -----------------------------
//
// App::load_prefs used to call save_prefs() for Stamp and Notify, which writes a
// value for every preference. After one launch every key was on disk, so a later
// release that changed a default (show_layer_bands, edge_routing, ...) never
// reached a user who had not chosen anything. The stamp must pin nothing else.

namespace {

nlohmann::json read_prefs_json() {
  std::ifstream f(view_prefs_file_path());
  nlohmann::json j;
  f >> j;
  return j;
}

}  // namespace

TEST_CASE("ViewPrefs: stamp_wheel_mode on a fresh install writes only wheel_mode") {
  PrefsFileBackup backup;  // starts with no prefs file
  REQUIRE(stamp_wheel_mode(WheelMode::Pan));

  const nlohmann::json j = read_prefs_json();
  REQUIRE(j.is_object());
  CHECK(j.size() == 1);
  CHECK(j.value("wheel_mode", "") == "pan");
}

TEST_CASE("ViewPrefs: a stamped file leaves every other default to the next load") {
  PrefsFileBackup backup;
  REQUIRE(stamp_wheel_mode(WheelMode::Pan));

  // The running app's defaults changed between releases: the base the next launch
  // passes in differs from anything an earlier launch could have written. Every
  // such value must come through, because the stamp pinned none of them.
  ViewPrefs base = all_non_default();
  base.wheel_mode = WheelMode::Pan;
  ViewPrefsLoadInfo info;
  const ViewPrefs out = load_view_prefs(base, &info);

  CHECK(info.file_read);
  CHECK(info.wheel_mode_present);  // so the notice does not fire again
  CHECK(out.wheel_mode == WheelMode::Pan);
  CHECK(out.dark_theme == base.dark_theme);
  CHECK(out.show_minimap == base.show_minimap);
  CHECK(out.show_layer_bands == base.show_layer_bands);
  CHECK(out.edge_tooltips == base.edge_tooltips);
  CHECK(out.cost_heatmap == base.cost_heatmap);
  CHECK(out.heatmap_log_scale == base.heatmap_log_scale);
  CHECK(out.heatmap_metric == base.heatmap_metric);
  CHECK(out.heatmap_gradient.preset == base.heatmap_gradient.preset);
  CHECK(out.heatmap_gradient.reverse == base.heatmap_gradient.reverse);
  CHECK(out.edge_routing == base.edge_routing);
  CHECK(out.accessible_palette == base.accessible_palette);
  CHECK(out.ui_scale == doctest::Approx(base.ui_scale));
  CHECK(out.restore_session == base.restore_session);
  CHECK(out.custom_ridge == doctest::Approx(base.custom_ridge));
  CHECK(out.machine_profiles == base.machine_profiles);
}

TEST_CASE("ViewPrefs: stamp_wheel_mode merges into an older file without touching it") {
  PrefsFileBackup backup;
  // A file from before wheel_mode, carrying one key this build knows and one it
  // does not (written by some other release): both must survive the stamp, and the
  // keys the file lacks must stay absent rather than being filled in.
  write_raw_prefs_file(R"({"dark_theme": false, "future_key": [1, 2, 3]})");
  REQUIRE(stamp_wheel_mode(WheelMode::Pan));

  const nlohmann::json j = read_prefs_json();
  REQUIRE(j.is_object());
  CHECK(j.size() == 3);
  CHECK(j.value("dark_theme", true) == false);
  CHECK(j.contains("future_key"));
  CHECK(j.value("wheel_mode", "") == "pan");
  CHECK_FALSE(j.contains("show_layer_bands"));
  CHECK_FALSE(j.contains("edge_routing"));
  CHECK_FALSE(j.contains("ui_scale"));

  // And it replaces a bad wheel_mode (which load reports as absent) rather than
  // leaving it to trigger the notice on every launch.
  write_raw_prefs_file(R"({"wheel_mode": "Zoom", "dark_theme": false})");
  REQUIRE(stamp_wheel_mode(WheelMode::Pan));
  const nlohmann::json k = read_prefs_json();
  CHECK(k.value("wheel_mode", "") == "pan");
  CHECK(k.value("dark_theme", true) == false);
  CHECK(k.size() == 2);
}

TEST_CASE("ViewPrefs: stamp_wheel_mode records the mode it is given") {
  PrefsFileBackup backup;
  REQUIRE(stamp_wheel_mode(WheelMode::Zoom));
  ViewPrefsLoadInfo info;
  const ViewPrefs out = load_view_prefs(ViewPrefs{}, &info);
  CHECK(info.wheel_mode_present);
  CHECK(out.wheel_mode == WheelMode::Zoom);
}

TEST_CASE("ViewPrefs: stamp_wheel_mode leaves an unusable file alone") {
  PrefsFileBackup backup;
  // The same files wheel_default_action says to leave alone: rewriting one at
  // startup would destroy whatever the user was editing.
  for (const char* text : {"", "{ not json", "[1, 2, 3]", "42",
                           R"({"wheel_mode": "zoom", "dark_theme": )"}) {
    write_raw_prefs_file(text);
    CHECK_FALSE(stamp_wheel_mode(WheelMode::Pan));
    std::ifstream f(view_prefs_file_path(), std::ios::binary);
    const std::string after((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    CHECK(after == text);
  }
}
