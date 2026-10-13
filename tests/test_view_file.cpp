// SPDX-License-Identifier: Apache-2.0
// tests/test_view_file.cpp — the `.netvis-view` format and its overlay (#56, #170).
//
// The GUI-side applier (view/ViewFileApply.cpp) needs ImGui and cannot be linked
// here, so everything it DECIDES is core and tested below: the parse policy (errors
// vs warnings), the limits, the round trip against ViewSnapshot, and the overlay
// ("which parts of this file apply to THIS model?").
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/JsonNesting.h"
#include "core/MappedFile.h"
#include "engine/ViewFile.h"
#include "parsers/Parser.h"
#include "temp_file_guard.h"

using namespace netvis;

namespace {

namespace fs = std::filesystem;

ViewFileLoad parse(const std::string& s) { return parse_view_file(s); }

// A view-file document with the given extra members spliced in.
std::string doc(const std::string& members) {
  return std::string("{\"kind\":\"netvis-view\"") + (members.empty() ? "" : "," + members) + "}";
}

bool has(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

bool any_note_has(const std::vector<std::string>& notes, const std::string& needle) {
  for (const std::string& n : notes)
    if (has(n, needle)) return true;
  return false;
}

struct TempDir {
  fs::path path;
  explicit TempDir(const char* name) {
    path = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove_all(path, ec);
    fs::create_directories(path, ec);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

void write_file(const fs::path& p, const std::string& bytes) {
  std::ofstream f(p, std::ios::binary);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// `n` nested arrays, as a JSON value.
std::string nested(int n) { return std::string(static_cast<size_t>(n), '[') + std::string(static_cast<size_t>(n), ']'); }

}  // namespace

// ---------------------------------------------------------------------------
TEST_CASE("view file: the exact JSON NetVis <= 0.9.x writes parses identically") {
  // Verbatim what the #56 App::save_view_state produced.
  const std::string legacy =
      R"({"cam":{"pan_x":12.5,"pan_y":-40.0,"zoom":0.75},"category_mask":4294967295,"graph":0,
          "hide_const_edges":false,"kind":"netvis-view","model":"/tmp/m.onnx","path_a":-1,
          "path_b":-1,"selected_ir_node":3,"show_layer_bands":true,"version":1})";
  const ViewFileLoad r = parse(legacy);
  REQUIRE(r.ok());
  CHECK(r.warnings.empty());
  const ViewFile& f = r.file;
  CHECK(f.model == "/tmp/m.onnx");
  REQUIRE(f.graph);
  CHECK(*f.graph == 0);
  CHECK(*f.pan_x == 12.5f);
  CHECK(*f.pan_y == -40.0f);
  CHECK(*f.zoom == 0.75f);
  CHECK(*f.hide_const_edges == false);
  CHECK(*f.show_layer_bands == true);
  CHECK(*f.category_mask == 4294967295u);
  CHECK(*f.path_a == -1);
  CHECK(*f.path_b == -1);
  CHECK(*f.selected_ir_node == 3);

  // Every key #170 added is absent, i.e. "keep the live value".
  CHECK_FALSE(f.show_critical_path);
  CHECK_FALSE(f.cost_heatmap);
  CHECK_FALSE(f.show_search_results);
  CHECK_FALSE(f.diff_panel_open);
  CHECK_FALSE(f.heatmap_log_scale);
  CHECK_FALSE(f.follow_preds);
  CHECK_FALSE(f.follow_succs);
  CHECK_FALSE(f.filter_active);
  CHECK_FALSE(f.edge_routing);
  CHECK_FALSE(f.heatmap_metric);
  CHECK_FALSE(f.nav_hops);
  CHECK_FALSE(f.nav_mode);
  CHECK_FALSE(f.search_query);
  CHECK_FALSE(f.attr_filter);
  CHECK_FALSE(f.table_filter);
  CHECK_FALSE(f.collapse);
  CHECK_FALSE(f.expanded);
  CHECK_FALSE(f.selected_value);
  CHECK_FALSE(f.pinned);
}

TEST_CASE("view file: the legacy variant without nav keys (vs.nav was null) parses") {
  const std::string legacy =
      R"({"cam":{"pan_x":1.0,"pan_y":2.0,"zoom":1.5},"graph":1,"hide_const_edges":true,
          "kind":"netvis-view","model":"m.onnx","selected_ir_node":-1,
          "show_layer_bands":false,"version":1})";
  const ViewFileLoad r = parse(legacy);
  REQUIRE(r.ok());
  CHECK(r.warnings.empty());
  CHECK_FALSE(r.file.category_mask);
  CHECK_FALSE(r.file.path_a);
  CHECK_FALSE(r.file.path_b);
  CHECK(*r.file.selected_ir_node == -1);
  CHECK(*r.file.graph == 1);
}

TEST_CASE("view file: round trip through ViewSnapshot, with a drift guard") {
  // EVERY ViewSnapshot field moved off its default.
  ViewSnapshot orig;
  orig.pan_x = 12.5f;
  orig.pan_y = -40.25f;
  orig.zoom = 0.75f;
  orig.selected_ir_node = 5;
  orig.selected_value = 9;
  orig.expanded = {true, false, true};
  orig.nav_mode = 2;
  orig.nav_hops = 3;
  orig.follow_preds = false;
  orig.follow_succs = false;
  orig.category_mask = 0x0F0Fu;
  orig.filter_active = true;
  orig.path_a = 1;
  orig.path_b = 4;
  orig.pinned = {2, 7, 5};
  orig.search_query = "conv";
  orig.attr_filter = "kernel";
  orig.table_filter = "weight";
  orig.hide_const_edges = true;
  orig.show_layer_bands = true;
  orig.show_critical_path = true;
  orig.cost_heatmap = true;
  orig.show_search_results = true;
  orig.diff_panel_open = true;
  orig.edge_routing = 2;

  ViewSnapshot base;                       // all defaults...
  base.expanded = {false, false, false};   // ...except the live group count
  orig.owner_generation = base.owner_generation;  // runtime-only identity, not in the file
  orig.owner_graph = base.owner_graph;

  const ViewFile f = view_file_from_snapshot(orig, "/models/m.onnx", 2,
                                             HeatmapMetric::ArithIntensity, false);
  const std::string text = serialize_view_file(f);
  const ViewFileLoad loaded = parse(text);
  REQUIRE(loaded.ok());
  CHECK(loaded.warnings.empty());
  CHECK(loaded.file.model == "/models/m.onnx");
  CHECK(*loaded.file.graph == 2);
  REQUIRE(loaded.file.heatmap_metric);
  CHECK(*loaded.file.heatmap_metric == HeatmapMetric::ArithIntensity);
  CHECK(*loaded.file.heatmap_log_scale == false);
  CHECK_FALSE(loaded.file.collapse);  // a snapshot has the full bitset; no shorthand

  const ViewOverlayResult r =
      overlay_view_file(loaded.file, base, /*same_model=*/true, 100000, 100000);
  CHECK(r.notes.empty());
  // camera_only_diff enumerates EVERY non-camera ViewSnapshot field: a field added to
  // ViewSnapshot later and not carried by the view file fails right here.
  CHECK(r.snapshot.camera_only_diff(orig));
  CHECK(r.snapshot.pan_x == orig.pan_x);
  CHECK(r.snapshot.pan_y == orig.pan_y);
  CHECK(r.snapshot.zoom == orig.zoom);
  CHECK(r.snapshot.pinned == orig.pinned);  // order preserved

  // Deterministic and idempotent.
  CHECK(serialize_view_file(f) == text);
  CHECK(serialize_view_file(loaded.file) == text);
}

TEST_CASE("view file: a snapshot of an untouched view round-trips too") {
  ViewSnapshot s;  // all defaults
  const ViewFile f = view_file_from_snapshot(s, "", 0, HeatmapMetric::Flops, true);
  const ViewFileLoad l = parse(serialize_view_file(f));
  REQUIRE(l.ok());
  CHECK(l.warnings.empty());
  const ViewOverlayResult r = overlay_view_file(l.file, s, true, 10, 10);
  CHECK(r.notes.empty());
  CHECK(r.snapshot.camera_only_diff(s));
  CHECK(r.snapshot.zoom == 1.0f);
}

TEST_CASE("view file: absent keys stay absent and the overlay returns base unchanged") {
  const ViewFileLoad r = parse("{\"kind\":\"netvis-view\"}");
  REQUIRE(r.ok());
  CHECK(r.warnings.empty());
  CHECK(r.file.model.empty());
  CHECK_FALSE(r.file.has_camera());
  CHECK_FALSE(r.file.has_model_specific());

  ViewSnapshot base;
  base.pan_x = 3.0f;
  base.zoom = 2.0f;
  base.expanded = {true, false};
  base.search_query = "keep";
  const ViewOverlayResult o = overlay_view_file(r.file, base, true, 10, 10);
  CHECK(o.notes.empty());
  CHECK(o.snapshot.camera_only_diff(base));
  CHECK(o.snapshot.pan_x == 3.0f);
  CHECK(o.snapshot.zoom == 2.0f);
  const ViewOverlayResult o2 = overlay_view_file(r.file, base, false, 10, 10);
  CHECK(o2.notes.empty());  // no model-specific keys, so nothing to say
  CHECK(o2.snapshot.camera_only_diff(base));
}

TEST_CASE("view file: NotAViewFile for a missing kind, a wrong kind, and a non-object root") {
  CHECK(parse("{}").error_kind == ViewFileErrorKind::NotAViewFile);
  CHECK(parse("{\"version\":1}").error_kind == ViewFileErrorKind::NotAViewFile);
  CHECK(parse("{\"kind\":\"other\"}").error_kind == ViewFileErrorKind::NotAViewFile);
  CHECK(parse("{\"kind\":1}").error_kind == ViewFileErrorKind::NotAViewFile);
  CHECK(parse("[1,2,3]").error_kind == ViewFileErrorKind::NotAViewFile);
  CHECK(parse("\"netvis-view\"").error_kind == ViewFileErrorKind::NotAViewFile);
  CHECK(parse("42").error_kind == ViewFileErrorKind::NotAViewFile);
}

TEST_CASE("view file: version handling") {
  CHECK(parse(doc("\"version\":2")).error_kind == ViewFileErrorKind::NewerVersion);
  CHECK(parse(doc("\"version\":18446744073709551615")).error_kind == ViewFileErrorKind::NewerVersion);
  CHECK(parse(doc("\"version\":0")).error_kind == ViewFileErrorKind::Invalid);
  CHECK(parse(doc("\"version\":-1")).error_kind == ViewFileErrorKind::Invalid);
  CHECK(parse(doc("\"version\":\"1\"")).error_kind == ViewFileErrorKind::Invalid);
  CHECK(parse(doc("\"version\":1.0")).error_kind == ViewFileErrorKind::Invalid);
  CHECK(parse(doc("\"version\":1")).ok());
  CHECK(parse(doc("")).ok());  // missing version is fine (a hand-written file)
}

TEST_CASE("view file: malformed JSON is NotJson and carries a byte offset") {
  const ViewFileLoad r = parse("{\"kind\":\"netvis-view\",\"cam\":{");
  CHECK(r.error_kind == ViewFileErrorKind::NotJson);
  CHECK(has(r.error, "byte "));
  CHECK_FALSE(r.ok());
  CHECK(parse("").error_kind == ViewFileErrorKind::NotJson);
  CHECK(parse("not json at all").error_kind == ViewFileErrorKind::NotJson);
  CHECK(parse("{\"kind\":\"netvis-view\"} trailing").error_kind == ViewFileErrorKind::NotJson);
  // Nothing is half-applied on an error.
  CHECK_FALSE(r.file.has_camera());
}

TEST_CASE("view file: nesting depth is capped before nlohmann sees it") {
  // 10,000 levels would overflow a recursive parser; the pre-scan rejects it first.
  const ViewFileLoad deep = parse(doc("\"search_query\":" + nested(10000)));
  CHECK(deep.error_kind == ViewFileErrorKind::TooDeep);

  // Total depth counts the root object: root (1) + 7 arrays = 8 is accepted ...
  const ViewFileLoad ok8 = parse("{\"kind\":\"netvis-view\",\"search_query\":" + nested(7) + "}");
  CHECK(ok8.ok());
  // ... and 8 arrays = 9 deep is rejected.
  const ViewFileLoad bad9 = parse("{\"kind\":\"netvis-view\",\"search_query\":" + nested(8) + "}");
  CHECK(bad9.error_kind == ViewFileErrorKind::TooDeep);

  // Brackets inside strings, including after an escaped quote, do not count.
  const ViewFileLoad in_str =
      parse("{\"kind\":\"netvis-view\",\"search_query\":\"[[[[[[[[[[[[ \\\" [[[[[[[[[[[[\"}");
  REQUIRE(in_str.ok());
  CHECK(in_str.file.search_query.has_value());
}

TEST_CASE("view file: read_view_file size limit, success and Io") {
  TempDir td("nv170_viewfile_size");
  const fs::path p = td.path / "big.netvis-view";

  // One byte over the cap: TooLarge, and never read.
  write_file(p, std::string(static_cast<size_t>(kMaxViewFileBytes) + 1, ' '));
  CHECK(read_view_file(p.string()).error_kind == ViewFileErrorKind::TooLarge);

  // Exactly the cap, valid JSON padded with whitespace: fine.
  std::string exact = "{\"kind\":\"netvis-view\"}";
  exact.append(static_cast<size_t>(kMaxViewFileBytes) - exact.size(), ' ');
  REQUIRE(exact.size() == kMaxViewFileBytes);
  write_file(p, exact);
  const ViewFileLoad ok = read_view_file(p.string());
  CHECK(ok.ok());

  // A missing path and a directory are Io.
  const ViewFileLoad missing = read_view_file((td.path / "nope.netvis-view").string());
  CHECK(missing.error_kind == ViewFileErrorKind::Io);
  CHECK_FALSE(missing.error.empty());
  CHECK(read_view_file(td.path.string()).error_kind == ViewFileErrorKind::Io);

  // A normal small file.
  write_file(p, doc("\"cam\":{\"zoom\":2.0}"));
  const ViewFileLoad small = read_view_file(p.string());
  REQUIRE(small.ok());
  CHECK(*small.file.zoom == 2.0f);
}

TEST_CASE("view file: a wrong-typed value is ignored with exactly one warning") {
  struct Case {
    const char* json;
    const char* key;
  };
  const Case cases[] = {
      {"\"cam\":{\"zoom\":\"big\"}", "cam.zoom"},
      {"\"cam\":{\"pan_x\":true}", "cam.pan_x"},
      {"\"cam\":[1,2]", "cam"},
      {"\"hide_const_edges\":1", "hide_const_edges"},
      {"\"show_layer_bands\":\"yes\"", "show_layer_bands"},
      {"\"expanded\":[1,0]", "expanded"},
      {"\"expanded\":\"all\"", "expanded"},
      {"\"pinned\":[\"a\"]", "pinned"},
      {"\"pinned\":[-1]", "pinned"},
      {"\"edge_routing\":1.5", "edge_routing"},
      {"\"graph\":\"0\"", "graph"},
      {"\"search_query\":5", "search_query"},
      {"\"heatmap_metric\":3", "heatmap_metric"},
      {"\"collapse\":true", "collapse"},
      {"\"selected_ir_node\":\"3\"", "selected_ir_node"},
  };
  for (const Case& c : cases) {
    INFO("case: " << c.json);
    const ViewFileLoad r = parse(doc(c.json));
    REQUIRE(r.ok());
    REQUIRE(r.warnings.size() == 1);
    CHECK(has(r.warnings[0], c.key));
    CHECK_FALSE(r.file.has_camera());
    CHECK_FALSE(r.file.expanded);
    CHECK_FALSE(r.file.pinned);
    CHECK_FALSE(r.file.graph);
    CHECK_FALSE(r.file.edge_routing);
    CHECK_FALSE(r.file.hide_const_edges);
    CHECK_FALSE(r.file.selected_ir_node);
    CHECK_FALSE(r.file.search_query);
    CHECK_FALSE(r.file.heatmap_metric);
    CHECK_FALSE(r.file.collapse);
  }
}

TEST_CASE("view file: one bad value does not cost the rest of the file") {
  const ViewFileLoad r = parse(doc("\"cam\":{\"zoom\":\"big\",\"pan_x\":7.0},\"hide_const_edges\":true,"
                                   "\"edge_routing\":9,\"show_layer_bands\":true"));
  REQUIRE(r.ok());
  CHECK(r.warnings.size() == 2);
  CHECK(*r.file.pan_x == 7.0f);
  CHECK_FALSE(r.file.zoom);
  CHECK(*r.file.hide_const_edges == true);
  CHECK(*r.file.show_layer_bands == true);
  CHECK_FALSE(r.file.edge_routing);
}

TEST_CASE("view file: an out-of-range value is ignored with a warning") {
  const char* bad[] = {
      "\"edge_routing\":3",
      "\"edge_routing\":-1",
      "\"nav_mode\":3",
      "\"cam\":{\"zoom\":0}",
      "\"cam\":{\"zoom\":-1}",
      "\"path_a\":-2",
      "\"path_b\":2147483648",
      "\"selected_value\":2147483648",
      "\"selected_ir_node\":-5",
      "\"category_mask\":4294967296",
      "\"category_mask\":-1",
      "\"graph\":4294967296",
      "\"nav_hops\":4294967296",
      "\"heatmap_metric\":\"Bogus\"",
      "\"collapse\":\"sideways\"",
      "\"cam\":{\"pan_x\":1e300}",
      "\"selected_ir_node\":18446744073709551615",
  };
  for (const char* b : bad) {
    INFO("case: " << b);
    const ViewFileLoad r = parse(doc(b));
    REQUIRE(r.ok());
    REQUIRE(r.warnings.size() == 1);
    CHECK_FALSE(r.file.edge_routing);
    CHECK_FALSE(r.file.nav_mode);
    CHECK_FALSE(r.file.has_camera());
    CHECK_FALSE(r.file.path_a);
    CHECK_FALSE(r.file.path_b);
    CHECK_FALSE(r.file.selected_value);
    CHECK_FALSE(r.file.selected_ir_node);
    CHECK_FALSE(r.file.category_mask);
    CHECK_FALSE(r.file.graph);
    CHECK_FALSE(r.file.nav_hops);
    CHECK_FALSE(r.file.heatmap_metric);
    CHECK_FALSE(r.file.collapse);
  }
  // The boundary values themselves are accepted.
  const ViewFileLoad edge = parse(doc("\"edge_routing\":2,\"nav_mode\":2,\"path_a\":2147483647,"
                                      "\"selected_value\":-1,\"category_mask\":0,\"graph\":4294967295,"
                                      "\"nav_hops\":4294967295"));
  REQUIRE(edge.ok());
  CHECK(edge.warnings.empty());
  CHECK(*edge.file.edge_routing == 2);
  CHECK(*edge.file.nav_mode == 2);
  CHECK(*edge.file.path_a == 2147483647);
  CHECK(*edge.file.graph == 4294967295u);
  CHECK(*edge.file.nav_hops == 4294967295u);
}

TEST_CASE("view file: zoom is clamped to the camera's own range") {
  const ViewFileLoad hi = parse(doc("\"cam\":{\"zoom\":100}"));
  REQUIRE(hi.ok());
  CHECK(*hi.file.zoom == 4.0f);
  const ViewFileLoad lo = parse(doc("\"cam\":{\"zoom\":0.001}"));
  REQUIRE(lo.ok());
  CHECK(*lo.file.zoom == 0.02f);
  CHECK(kViewFileMinZoom == 0.02f);  // Camera.cpp kMinZoom
  CHECK(kViewFileMaxZoom == 4.0f);   // Camera.cpp kMaxZoom

  // The overlay clamps too: a ViewFile built by code is not trusted either.
  ViewFile f;
  f.zoom = 100.0f;
  const ViewOverlayResult r = overlay_view_file(f, ViewSnapshot{}, true, 1, 1);
  CHECK(r.snapshot.zoom == 4.0f);
}

TEST_CASE("view file: string, pinned and expanded limits") {
  // search_query: 4096 bytes is fine, 4097 is an error.
  CHECK(parse(doc("\"search_query\":\"" + std::string(kMaxViewFileString, 'a') + "\"")).ok());
  const ViewFileLoad s = parse(doc("\"search_query\":\"" + std::string(kMaxViewFileString + 1, 'a') + "\""));
  CHECK(s.error_kind == ViewFileErrorKind::Invalid);
  CHECK(parse(doc("\"model\":\"" + std::string(kMaxViewFileString + 1, 'a') + "\"")).error_kind ==
        ViewFileErrorKind::Invalid);
  CHECK(parse(doc("\"attr_filter\":\"" + std::string(kMaxViewFileString + 1, 'a') + "\"")).error_kind ==
        ViewFileErrorKind::Invalid);
  CHECK(parse(doc("\"table_filter\":\"" + std::string(kMaxViewFileString + 1, 'a') + "\"")).error_kind ==
        ViewFileErrorKind::Invalid);

  // pinned: 4096 entries fine, 4097 an error.
  auto pins = [](size_t n) {
    std::string s = "\"pinned\":[";
    for (size_t i = 0; i < n; ++i) s += (i ? ",1" : "1");
    return s + "]";
  };
  const ViewFileLoad p_ok = parse(doc(pins(kMaxViewFilePinned)));
  REQUIRE(p_ok.ok());
  CHECK(p_ok.file.pinned->size() == kMaxViewFilePinned);
  CHECK(parse(doc(pins(kMaxViewFilePinned + 1))).error_kind == ViewFileErrorKind::Invalid);

  // expanded: the byte cap (4 MiB; "true," is 5 bytes) bites at ~838k entries, BEFORE
  // kMaxViewFileGroups (1,048,576), so the per-key limit is defence in depth and the
  // file-size limit is what a hostile file hits. Below the byte cap it parses ...
  auto bits = [](size_t n) {
    std::string s = "\"expanded\":[";
    s.reserve(n * 5 + 16);
    for (size_t i = 0; i < n; ++i) s += (i ? ",true" : "true");
    return s + "]";
  };
  const ViewFileLoad e_ok = parse(doc(bits(800000)));
  REQUIRE(e_ok.ok());
  CHECK(e_ok.file.expanded->size() == 800000);
  // ... and at kMaxViewFileGroups + 1 entries the file is refused outright.
  const ViewFileLoad e_bad = parse(doc(bits(kMaxViewFileGroups + 1)));
  CHECK_FALSE(e_bad.ok());
  CHECK((e_bad.error_kind == ViewFileErrorKind::TooLarge ||
         e_bad.error_kind == ViewFileErrorKind::Invalid));
}

TEST_CASE("view file: unknown keys are ignored silently") {
  const ViewFileLoad r = parse(doc("\"future_key\":{\"x\":1},\"another\":[1,2],\"cam\":{\"zoom\":2.0,\"roll\":9}"));
  REQUIRE(r.ok());
  CHECK(r.warnings.empty());
  CHECK(*r.file.zoom == 2.0f);
}

TEST_CASE("view file: a model passed as --view is NotJson") {
  const char* model = "tests/fixtures/model.onnx";
  if (!fs::exists(model)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return;
  }
  const ViewFileLoad r = read_view_file(model);
  CHECK_FALSE(r.ok());
  CHECK((r.error_kind == ViewFileErrorKind::NotJson || r.error_kind == ViewFileErrorKind::NotAViewFile));
  // And the same bytes through parse_view_file.
  std::ifstream f(model, std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  CHECK(parse_view_file(bytes).error_kind == ViewFileErrorKind::NotJson);
}

TEST_CASE("view file: a view file passed as the model is Format::Unknown") {
  // The guard comes BEFORE the mapping so the mapping is released first: Windows cannot
  // delete a file that is still mapped (tests/temp_file_guard.h).
  const fs::path p = fs::temp_directory_path() / "nv170_viewfile_detect.netvis-view";
  netvis_test::TempFileGuard cleanup(p.string());
  write_file(p, serialize_view_file(view_file_from_snapshot(ViewSnapshot{}, "/m.onnx", 0,
                                                            HeatmapMetric::Flops, true)));
  auto mf = MappedFile::open(p.string());
  REQUIRE(mf);
  CHECK(detect_format(*mf, "netvis-view") == Format::Unknown);
}

TEST_CASE("same_model_path") {
  const fs::path abs = fs::absolute("tests/fixtures/model.onnx");
  CHECK(same_model_path(abs.string(), abs.string()));
  CHECK(same_model_path("./tests/fixtures/model.onnx", abs.string()));
  CHECK(same_model_path(abs.string(), "./tests/fixtures/model.onnx"));
  CHECK(same_model_path("tests/fixtures/../fixtures/model.onnx", abs.string()));
  CHECK_FALSE(same_model_path("tests/fixtures/model.onnx", "tests/fixtures/model.safetensors"));
  CHECK_FALSE(same_model_path("", abs.string()));
  CHECK_FALSE(same_model_path(abs.string(), ""));
  CHECK_FALSE(same_model_path("", ""));
  // A path that does not exist is still the same as itself (string fallback).
  CHECK(same_model_path("/nonexistent/dir/m.onnx", "/nonexistent/dir/m.onnx"));
  CHECK_FALSE(same_model_path("/nonexistent/dir/m.onnx", "/nonexistent/dir/n.onnx"));
  // A bundle directory with and without a trailing separator.
  const fs::path dir = fs::temp_directory_path();
  CHECK(same_model_path(dir.string() + "/", dir.string()));
}

namespace {

ViewFile model_specific_file() {
  ViewFile f;
  f.model = "/models/a.onnx";
  f.graph = 1;
  f.selected_ir_node = 7;
  f.selected_value = 3;
  f.path_a = 2;
  f.path_b = 9;
  f.pinned = std::vector<uint32_t>{4, 1};
  f.expanded = std::vector<bool>{false, true, false};
  // And some agnostic ones.
  f.zoom = 2.0f;
  f.pan_x = 5.0f;
  f.cost_heatmap = true;
  f.edge_routing = 1;
  f.search_query = "attn";
  return f;
}

ViewSnapshot live_base() {
  ViewSnapshot b;
  b.expanded = {true, true, true};
  b.pan_x = -1.0f;
  return b;
}

}  // namespace

TEST_CASE("overlay: same model applies model-specific and agnostic fields") {
  const ViewOverlayResult r = overlay_view_file(model_specific_file(), live_base(), true, 38, 61);
  CHECK(r.notes.empty());
  CHECK(r.snapshot.selected_ir_node == 7);
  CHECK(r.snapshot.selected_value == 3);
  CHECK(r.snapshot.path_a == 2);
  CHECK(r.snapshot.path_b == 9);
  CHECK(r.snapshot.pinned == std::vector<uint32_t>{4, 1});
  CHECK(r.snapshot.expanded == std::vector<bool>{false, true, false});
  CHECK(r.snapshot.zoom == 2.0f);
  CHECK(r.snapshot.pan_x == 5.0f);
  CHECK(r.snapshot.cost_heatmap);
  CHECK(r.snapshot.edge_routing == 1);
  CHECK(r.snapshot.search_query == "attn");
}

TEST_CASE("overlay: another model keeps its own selection and says why, agnostic still applies") {
  const ViewSnapshot base = live_base();
  const ViewOverlayResult r = overlay_view_file(model_specific_file(), base, false, 38, 61);
  // None of the model-specific fields apply.
  CHECK(r.snapshot.selected_ir_node == base.selected_ir_node);
  CHECK(r.snapshot.selected_value == base.selected_value);
  CHECK(r.snapshot.path_a == base.path_a);
  CHECK(r.snapshot.path_b == base.path_b);
  CHECK(r.snapshot.pinned == base.pinned);
  CHECK(r.snapshot.expanded == base.expanded);
  // The single explanatory note.
  size_t not_applied = 0;
  for (const std::string& n : r.notes)
    if (has(n, "model-specific settings") && has(n, "not applied")) ++not_applied;
  CHECK(not_applied == 1);
  CHECK(any_note_has(r.notes, "/models/a.onnx"));
  CHECK(any_note_has(r.notes, "collapse bitset is model-specific"));
  // Agnostic fields apply in both cases.
  CHECK(r.snapshot.zoom == 2.0f);
  CHECK(r.snapshot.pan_x == 5.0f);
  CHECK(r.snapshot.cost_heatmap);
  CHECK(r.snapshot.edge_routing == 1);
  CHECK(r.snapshot.search_query == "attn");

  // A file that names no model says so.
  ViewFile f = model_specific_file();
  f.model.clear();
  const ViewOverlayResult none = overlay_view_file(f, base, false, 38, 61);
  CHECK(any_note_has(none.notes, "names no model"));
}

TEST_CASE("overlay: out-of-range indices are dropped with a note, never clamped") {
  ViewFile f;
  f.selected_ir_node = 38;  // == node_count
  f.path_b = 999;
  f.path_a = 37;            // in range
  f.selected_value = 61;    // == value_count
  f.pinned = std::vector<uint32_t>{1, 50, 2};
  ViewSnapshot base;
  base.selected_ir_node = 11;
  base.path_b = 12;
  base.selected_value = 13;
  const ViewOverlayResult r = overlay_view_file(f, base, true, 38, 61);
  CHECK(r.snapshot.selected_ir_node == 11);  // base value kept, NOT clamped to 37
  CHECK(r.snapshot.path_b == 12);
  CHECK(r.snapshot.selected_value == 13);
  CHECK(r.snapshot.path_a == 37);
  CHECK(r.snapshot.pinned == std::vector<uint32_t>{1, 2});  // order kept
  CHECK(any_note_has(r.notes, "selected_ir_node 38"));
  CHECK(any_note_has(r.notes, "path_b 999"));
  CHECK(any_note_has(r.notes, "selected_value 61"));
  CHECK(any_note_has(r.notes, "1 pinned node(s) dropped"));
  CHECK(r.notes.size() == 4);

  // -1 ("none") is always valid, even with a zero-node model.
  ViewFile none;
  none.selected_ir_node = -1;
  none.path_a = -1;
  ViewSnapshot sel;
  sel.selected_ir_node = 3;
  sel.path_a = 4;
  const ViewOverlayResult r2 = overlay_view_file(none, sel, true, 0, 0);
  CHECK(r2.snapshot.selected_ir_node == -1);
  CHECK(r2.snapshot.path_a == -1);
  CHECK(r2.notes.empty());
}

TEST_CASE("overlay: the collapse plan") {
  const std::vector<bool> live = {true};

  // A matching bitset wins over the shorthand.
  {
    ViewFile f;
    f.expanded = std::vector<bool>{false};
    f.collapse = CollapseShorthand::None;
    const CollapsePlan p = plan_collapse(f, live, true);
    CHECK(p.expanded == std::vector<bool>{false});
    CHECK(p.notes.empty());
  }
  // A bitset of the wrong size is noted, then the shorthand applies.
  {
    ViewFile f;
    f.expanded = std::vector<bool>{true, true, true, true, true};
    f.collapse = CollapseShorthand::All;
    const CollapsePlan p = plan_collapse(f, live, true);
    CHECK(p.expanded == std::vector<bool>{false});
    REQUIRE(p.notes.size() == 1);
    CHECK(has(p.notes[0], "5 groups"));
    CHECK(has(p.notes[0], "model has 1"));
  }
  // Without a shorthand the live state is kept.
  {
    ViewFile f;
    f.expanded = std::vector<bool>{true, true, true, true, true};
    const CollapsePlan p = plan_collapse(f, live, true);
    CHECK(p.expanded == live);
    CHECK(p.notes.size() == 1);
  }
  // Another model + collapse "none": every live group expanded; the bitset is refused.
  {
    ViewFile f;
    f.collapse = CollapseShorthand::None;
    const CollapsePlan p = plan_collapse(f, std::vector<bool>{false, false, false}, false);
    CHECK(p.expanded == std::vector<bool>{true, true, true});
    CHECK(p.notes.empty());
  }
  // The shorthand needs no model match: "all" collapses any model.
  {
    ViewFile f;
    f.collapse = CollapseShorthand::All;
    const CollapsePlan p = plan_collapse(f, std::vector<bool>{true, true}, false);
    CHECK(p.expanded == std::vector<bool>{false, false});
  }
  // No groups: nothing to do, and the plan stays empty.
  {
    ViewFile f;
    f.collapse = CollapseShorthand::All;
    CHECK(plan_collapse(f, {}, true).expanded.empty());
  }
  // Through the overlay.
  {
    ViewFile f;
    f.collapse = CollapseShorthand::All;
    ViewSnapshot base;
    base.expanded = {true, true};
    const ViewOverlayResult r = overlay_view_file(f, base, false, 10, 10);
    CHECK(r.snapshot.expanded == std::vector<bool>{false, false});
    CHECK(r.notes.empty());
  }
}

TEST_CASE("overlay: a ViewFile built by code with out-of-range enums is not trusted") {
  ViewFile f;
  f.edge_routing = 7;
  f.nav_mode = 9;
  const ViewOverlayResult r = overlay_view_file(f, ViewSnapshot{}, true, 1, 1);
  CHECK(r.snapshot.edge_routing == 0);
  CHECK(r.snapshot.nav_mode == 0);
  CHECK(r.notes.size() == 2);
}

TEST_CASE("view_file_from_snapshot never writes a file this build would refuse") {
  ViewSnapshot s;
  s.search_query = std::string(kMaxViewFileString + 1, 'q');
  for (uint32_t i = 0; i < kMaxViewFilePinned + 10; ++i) s.pinned.push_back(i);
  const ViewFile f = view_file_from_snapshot(s, std::string(kMaxViewFileString + 1, 'm'), 0,
                                             HeatmapMetric::Flops, true);
  CHECK_FALSE(f.search_query);
  CHECK(f.model.empty());
  CHECK(f.pinned->size() == kMaxViewFilePinned);
  const ViewFileLoad l = parse(serialize_view_file(f));
  CHECK(l.ok());
}

TEST_CASE("serialize_view_file: non-UTF-8 model paths do not throw") {
  ViewFile f;
  f.model = std::string("/models/caf\xE9.onnx");  // Latin-1 e-acute: invalid UTF-8
  std::string text;
  CHECK_NOTHROW(text = serialize_view_file(f));
  CHECK(parse(text).ok());
}

TEST_CASE("json_nesting_ok in isolation") {
  CHECK(json_nesting_ok("{}", 1));
  CHECK_FALSE(json_nesting_ok("{{}}", 1));
  CHECK(json_nesting_ok("[[[]]]", 3));
  CHECK_FALSE(json_nesting_ok("[[[[]]]]", 3));
  // An escaped quote must not end the string: the brackets after it are still text.
  CHECK(json_nesting_ok("{\"a\":\"x\\\"[[[[[[[[[[[[\"}", 1));
  // A string that really ends, then real nesting.
  CHECK_FALSE(json_nesting_ok("{\"a\":\"x\",\"b\":[[[[[[[[[[[[]]]]]]]]]]]]}", 3));
  // An unbalanced closer must not drive the depth negative (which would hide later
  // nesting): five stray closers, then four real openers against a limit of three.
  CHECK_FALSE(json_nesting_ok("]]]]][[[[", 3));
  CHECK(json_nesting_ok("]]]]][[[", 3));
  CHECK(json_nesting_ok("", 0));
}

// ---------------------------------------------------------------------------
// Review fixes for #178: filters that fit their boxes, what the user is told after a
// load, relative model paths, the save size limit, comments in the nesting scan.
// ---------------------------------------------------------------------------
TEST_CASE("view file: a filter longer than its box is cut to fit, on a character boundary, with a warning") {
  // Exactly what the box holds is untouched: buffer size minus the NUL.
  const std::string fits(kAttrFilterUiBytes - 1, 'a');
  {
    const ViewFileLoad r = parse(doc("\"attr_filter\":\"" + fits + "\""));
    REQUIRE(r.ok());
    CHECK(r.warnings.empty());
    CHECK(*r.file.attr_filter == fits);
  }
  // One byte more is cut back to what fits.
  {
    const ViewFileLoad r = parse(doc("\"attr_filter\":\"" + fits + "b\""));
    REQUIRE(r.ok());
    CHECK(*r.file.attr_filter == fits);
    REQUIRE(r.warnings.size() == 1);
    CHECK(has(r.warnings[0], "attr_filter"));
    CHECK(has(r.warnings[0], "cut to fit"));
  }
  // The other two boxes have their own size.
  {
    const std::string q(kSearchQueryUiBytes + 40, 'q');
    const std::string t(kTableFilterUiBytes + 40, 't');
    const ViewFileLoad r =
        parse(doc("\"search_query\":\"" + q + "\",\"table_filter\":\"" + t + "\""));
    REQUIRE(r.ok());
    CHECK(r.file.search_query->size() == kSearchQueryUiBytes - 1);
    CHECK(r.file.table_filter->size() == kTableFilterUiBytes - 1);
    CHECK(r.warnings.size() == 2);
  }
  // A multi-byte character that straddles the limit is dropped whole, never split: the
  // result is valid UTF-8 (no stray continuation byte at the end).
  {
    // 62 ASCII bytes, then U+00E9 (2 bytes: bytes 62 and 63; the limit is 63 bytes).
    const std::string s = std::string(kAttrFilterUiBytes - 2, 'a') + "\xC3\xA9" + "z";
    const ViewFileLoad r = parse(doc("\"attr_filter\":\"" + s + "\""));
    REQUIRE(r.ok());
    CHECK(r.file.attr_filter->size() == kAttrFilterUiBytes - 2);
    CHECK(r.file.attr_filter->find('\xC3') == std::string::npos);
  }
  // And what the cut produced is what the overlay will hand to the UI.
  {
    const ViewFileLoad r = parse(doc("\"attr_filter\":\"" + fits + "bbbbbbbb\""));
    REQUIRE(r.ok());
    const ViewOverlayResult o = overlay_view_file(r.file, ViewSnapshot{}, true, 1, 1);
    CHECK(o.snapshot.attr_filter.size() == kAttrFilterUiBytes - 1);
  }
}

TEST_CASE("view_load_toasts: a load that dropped settings does not read as a clean success") {
  using O = ViewApplyOutcome;
  const std::vector<std::string> none;

  SUBCASE("a clean full load is the single plain toast") {
    const auto t = view_load_toasts(O::Full, none);
    REQUIRE(t.size() == 1);
    CHECK(t[0].text == "View loaded");
    CHECK_FALSE(t[0].is_error);
  }
  SUBCASE("one ignored setting is counted and named") {
    const auto t = view_load_toasts(O::Full, {"pinned 999 is out of range"});
    REQUIRE(t.size() == 2);
    CHECK(t[0].text == "View loaded (1 setting ignored)");
    CHECK(t[1].text == "pinned 999 is out of range");
    CHECK_FALSE(t[0].is_error);
    CHECK_FALSE(t[1].is_error);
  }
  SUBCASE("many ignored settings: counted, the first few named, the rest summarised") {
    std::vector<std::string> notes;
    for (size_t i = 0; i < kViewLoadMaxNoteToasts + 2; ++i) notes.push_back("note " + std::to_string(i));
    const auto t = view_load_toasts(O::Full, notes);
    REQUIRE(t.size() == 1 + kViewLoadMaxNoteToasts + 1);
    CHECK(t[0].text == "View loaded (" + std::to_string(notes.size()) + " settings ignored)");
    for (size_t i = 0; i < kViewLoadMaxNoteToasts; ++i) CHECK(t[1 + i].text == notes[i]);
    CHECK(t.back().text == "... and 2 more ignored");
  }
  SUBCASE("exactly the cap needs no summary line") {
    std::vector<std::string> notes(kViewLoadMaxNoteToasts, "n");
    CHECK(view_load_toasts(O::Full, notes).size() == 1 + kViewLoadMaxNoteToasts);
  }
  SUBCASE("another model / no model keep their own explanation, and the notes follow") {
    const auto a = view_load_toasts(O::OtherModel, {"view file was saved for 'x'"});
    REQUIRE(a.size() == 2);
    CHECK(a[0].text == "View loaded (camera only - saved for another model)");
    const auto b = view_load_toasts(O::NoModel, none);
    REQUIRE(b.size() == 1);
    CHECK(b[0].text == "View loaded (open a model to restore selection)");
  }
  SUBCASE("a refused apply is one ERROR toast, whatever the notes say") {
    const auto t = view_load_toasts(O::Failed, {"some note", "another"});
    REQUIRE(t.size() == 1);
    CHECK(t[0].is_error);
    CHECK(has(t[0].text, "could not be applied"));
    CHECK_FALSE(has(t[0].text, "View loaded"));
  }
}

TEST_CASE("same_model_path: a relative model also resolves against the view file's directory") {
  TempDir td("nv170_samemodel_dir");
  const fs::path models = td.path / "models";
  fs::create_directories(models);
  const fs::path live = models / "m.onnx";
  write_file(live, "x");

  // Relative to the view file's directory (what a committed file means) ...
  CHECK(same_model_path("models/m.onnx", live.string(), td.path.string()));
  CHECK(same_model_path("./models/m.onnx", live.string(), td.path.string()));
  CHECK(same_model_path("models/../models/m.onnx", live.string(), td.path.string()));
  // ... but not without that directory, from some other working directory,
  CHECK_FALSE(same_model_path("models/m.onnx", live.string()));
  CHECK_FALSE(same_model_path("models/m.onnx", live.string(), ""));
  // ... and never for a different file.
  CHECK_FALSE(same_model_path("models/n.onnx", live.string(), td.path.string()));
  CHECK_FALSE(same_model_path("m.onnx", live.string(), td.path.string()));
  // The directory only matters for a RELATIVE path: an absolute one stands alone.
  CHECK_FALSE(same_model_path((td.path / "other" / "m.onnx").string(), live.string(),
                              td.path.string()));
  CHECK(same_model_path(live.string(), live.string(), "/somewhere/else"));
  // Empty stays empty.
  CHECK_FALSE(same_model_path("", live.string(), td.path.string()));
}

TEST_CASE("read_view_file records where the file came from, so a relative model resolves") {
  TempDir td("nv170_viewfile_srcdir");
  const fs::path models = td.path / "models";
  fs::create_directories(models);
  write_file(models / "m.onnx", "x");
  const fs::path vf = td.path / "v.netvis-view";
  write_file(vf, doc("\"model\":\"models/m.onnx\",\"selected_ir_node\":2"));

  const ViewFileLoad l = read_view_file(vf.string());
  REQUIRE(l.ok());
  CHECK(l.file.model == "models/m.onnx");
  CHECK_FALSE(l.file.source_dir.empty());
  CHECK(same_model_path(l.file.model, (models / "m.onnx").string(), l.file.source_dir));
  // From bytes there is no directory.
  CHECK(parse(doc("\"model\":\"models/m.onnx\"")).file.source_dir.empty());
  // The directory is not part of the format: it is never written.
  CHECK_FALSE(has(serialize_view_file(l.file), "source_dir"));
}

TEST_CASE("model_path_for_view_file: relative when the model sits inside the view file's directory") {
  TempDir td("nv170_modelpath_rel");
  const fs::path dir = td.path;
  const std::string view = (dir / "v.netvis-view").string();

  CHECK(model_path_for_view_file((dir / "m.onnx").string(), view) == "m.onnx");
  CHECK(model_path_for_view_file((dir / "models" / "m.onnx").string(), view) == "models/m.onnx");
  // A bundle with a trailing separator is the same bundle.
  CHECK(model_path_for_view_file((dir / "b.mlpackage").string() + "/", view) == "b.mlpackage");
  // Outside the directory (a sibling, or the parent): unchanged, absolute as it was.
  const std::string sibling = (dir.parent_path() / "elsewhere" / "m.onnx").string();
  CHECK(model_path_for_view_file(sibling, view) == sibling);
  const std::string parent_model = (dir.parent_path() / "m.onnx").string();
  CHECK(model_path_for_view_file(parent_model, view) == parent_model);
  // The directory itself is not "inside" itself.
  CHECK(model_path_for_view_file(dir.string(), view) == dir.string());
  // Nothing to do for the empty cases.
  CHECK(model_path_for_view_file("", view).empty());
  CHECK(model_path_for_view_file("/a/m.onnx", "") == "/a/m.onnx");
  // And the pair round-trips: what was written resolves back to the same file.
  const std::string rel = model_path_for_view_file((dir / "models" / "m.onnx").string(), view);
  fs::create_directories(dir / "models");
  write_file(dir / "models" / "m.onnx", "x");
  CHECK(same_model_path(rel, (dir / "models" / "m.onnx").string(), dir.string()));
}

TEST_CASE("serialize_view_file: readable when it fits, compact when the readable form would not load back") {
  // A normal file is indented for people.
  {
    ViewFile f;
    f.model = "/m.onnx";
    f.expanded = std::vector<bool>{true, false};
    const std::string text = serialize_view_file(f);
    CHECK(has(text, "\n  \"kind\""));
  }
  // 500k groups: ~11 bytes each indented is 5.5 MB, over the 4 MiB cap the reader
  // enforces (so a save would have written a file this build refuses to load); the
  // compact form (~6 bytes each) fits, and loads.
  {
    ViewFile f;
    f.expanded = std::vector<bool>(500000, false);
    const std::string text = serialize_view_file(f);
    REQUIRE(text.size() <= kMaxViewFileBytes);
    CHECK_FALSE(has(text, "\n"));
    const ViewFileLoad l = parse(text);
    REQUIRE(l.ok());
    REQUIRE(l.file.expanded);
    CHECK(l.file.expanded->size() == 500000);
  }
  // Past the cap even compactly, the text comes back over the limit: documented, so the
  // caller (App::save_view_state) checks and refuses rather than writing it.
  {
    ViewFile f;
    f.expanded = std::vector<bool>(800000, false);
    CHECK(serialize_view_file(f).size() > kMaxViewFileBytes);
  }
}

TEST_CASE("view_file_from_snapshot names everything it had to leave out") {
  ViewSnapshot s;
  s.search_query = std::string(kMaxViewFileString + 1, 'q');
  s.attr_filter = std::string(kMaxViewFileString + 1, 'a');
  s.table_filter = std::string(kMaxViewFileString + 1, 't');
  for (uint32_t i = 0; i < kMaxViewFilePinned + 7; ++i) s.pinned.push_back(i);
  s.expanded.assign(kMaxViewFileGroups + 1, true);

  std::vector<std::string> dropped;
  const ViewFile f = view_file_from_snapshot(s, std::string(kMaxViewFileString + 1, 'm'), 0,
                                             HeatmapMetric::Flops, true, &dropped);
  CHECK(f.model.empty());
  CHECK_FALSE(f.search_query);
  CHECK_FALSE(f.attr_filter);
  CHECK_FALSE(f.table_filter);
  CHECK_FALSE(f.expanded);
  CHECK(f.pinned->size() == kMaxViewFilePinned);
  CHECK(any_note_has(dropped, "model path"));
  CHECK(any_note_has(dropped, "search_query"));
  CHECK(any_note_has(dropped, "attr_filter"));
  CHECK(any_note_has(dropped, "table_filter"));
  CHECK(any_note_has(dropped, "collapse state"));
  CHECK(any_note_has(dropped, "pinned"));
  CHECK(dropped.size() == 6);

  // Nothing over a limit: nothing to report. And a null `dropped` is fine.
  std::vector<std::string> none;
  (void)view_file_from_snapshot(ViewSnapshot{}, "/m.onnx", 0, HeatmapMetric::Flops, true, &none);
  CHECK(none.empty());
  CHECK_NOTHROW((void)view_file_from_snapshot(s, "/m.onnx", 0, HeatmapMetric::Flops, true));
}

TEST_CASE("json_nesting_ok: with comments allowed, a quote in a comment cannot hide the brackets") {
  const std::string deep = std::string(100, '[') + std::string(100, ']');

  // The attack the review found: a comment holding a lone quote flips a naive scan
  // into string mode, so the 100 real brackets after it are never counted.
  const std::string line = "// \"\n" + deep;
  const std::string block = "/* \" */" + deep;
  CHECK_FALSE(json_nesting_ok(line, 64, /*allow_comments=*/true));
  CHECK_FALSE(json_nesting_ok(block, 64, /*allow_comments=*/true));
  // \r also ends a line comment (as in nlohmann's lexer).
  CHECK_FALSE(json_nesting_ok("// \"\r" + deep, 64, true));

  // Brackets INSIDE a comment are not nesting.
  CHECK(json_nesting_ok("// " + deep + "\n[]", 64, true));
  CHECK(json_nesting_ok("/* " + deep + " */ []", 64, true));
  // A comment marker inside a string is just text; the real nesting after it counts.
  CHECK(json_nesting_ok("{\"url\":\"http://example.com/[[[[[[\"}", 1, true));
  CHECK_FALSE(json_nesting_ok("{\"u\":\"//\",\"a\":" + deep + "}", 8, true));
  CHECK_FALSE(json_nesting_ok("{\"u\":\"/*\",\"a\":" + deep + "}", 8, true));
  // An unterminated block comment is a parse error for the real parser: nothing to count.
  CHECK(json_nesting_ok("/* " + deep, 8, true));
  // A lone '/' at the very end, and '/' followed by something else, are harmless.
  CHECK(json_nesting_ok("[/", 8, true));
  CHECK(json_nesting_ok("[/x]", 8, true));

  // The default is unchanged: a comment-rejecting parser (the view-file reader) never
  // gets as far as the brackets, so the scan stays the plain one.
  CHECK(json_nesting_ok("[[[]]]", 3));
  CHECK_FALSE(json_nesting_ok("[[[[]]]]", 3));
  CHECK_FALSE(json_nesting_ok(deep, 64));
}
