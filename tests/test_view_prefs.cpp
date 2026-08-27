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

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

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
