// SPDX-License-Identifier: Apache-2.0
// engine/ViewFile.cpp — see ViewFile.h. The `.netvis-view` reader/writer/overlay.
#include "engine/ViewFile.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/JsonNesting.h"

namespace netvis {

namespace {

using json = nlohmann::json;

constexpr int64_t kI32Max = std::numeric_limits<int32_t>::max();
constexpr int64_t kU32Max = std::numeric_limits<uint32_t>::max();

const json* find_key(const json& obj, const char* key) {
  auto it = obj.find(key);
  return it == obj.end() ? nullptr : &*it;
}

// One parse in flight: warnings (ignored keys) and at most one fatal error.
struct Ctx {
  std::vector<std::string> warnings;
  ViewFileErrorKind err = ViewFileErrorKind::None;
  std::string err_msg;
  std::string prefix;  // "cam." while reading the camera object

  bool failed() const { return err != ViewFileErrorKind::None; }
  void fatal(ViewFileErrorKind k, std::string msg) {
    if (!failed()) {
      err = k;
      err_msg = std::move(msg);
    }
  }
  void warn_type(const char* key, const char* expected) {
    warnings.push_back("'" + prefix + key + "' has the wrong type (expected " + expected +
                       "); ignored");
  }
  void warn_range(const char* key, const std::string& got, const std::string& expected) {
    warnings.push_back("'" + prefix + key + "' is out of range (" + got + ", expected " +
                       expected + "); ignored");
  }
};

std::string range_text(int64_t lo, int64_t hi) {
  return std::to_string(lo) + ".." + std::to_string(hi);
}

bool read_bool(Ctx& c, const json& obj, const char* key, std::optional<bool>& out) {
  const json* v = find_key(obj, key);
  if (!v) return false;
  if (!v->is_boolean()) {
    c.warn_type(key, "a boolean");
    return false;
  }
  out = v->get<bool>();
  return true;
}

// An integer in [lo, hi]. Reads through int64/uint64 and range-checks: never the
// truncating get<uint32_t>/get<int32_t> the #56 reader used.
bool read_int(Ctx& c, const json& obj, const char* key, int64_t lo, int64_t hi, int64_t& out) {
  const json* v = find_key(obj, key);
  if (!v) return false;
  if (!v->is_number_integer()) {
    c.warn_type(key, "an integer");
    return false;
  }
  int64_t val = 0;
  if (v->is_number_unsigned()) {
    const uint64_t u = v->get<uint64_t>();
    if (u > static_cast<uint64_t>(hi)) {
      c.warn_range(key, std::to_string(u), range_text(lo, hi));
      return false;
    }
    val = static_cast<int64_t>(u);
  } else {
    val = v->get<int64_t>();
  }
  if (val < lo || val > hi) {
    c.warn_range(key, std::to_string(val), range_text(lo, hi));
    return false;
  }
  out = val;
  return true;
}

// A finite number whose magnitude fits a float.
bool read_float(Ctx& c, const json& obj, const char* key, float& out) {
  const json* v = find_key(obj, key);
  if (!v) return false;
  if (!v->is_number()) {
    c.warn_type(key, "a number");
    return false;
  }
  const double d = v->get<double>();
  if (!std::isfinite(d) || std::fabs(d) > static_cast<double>(FLT_MAX)) {
    c.warn_range(key, std::isfinite(d) ? "too large" : "not finite", "a finite float");
    return false;
  }
  out = static_cast<float>(d);
  return true;
}

bool read_string(Ctx& c, const json& obj, const char* key, std::string& out) {
  const json* v = find_key(obj, key);
  if (!v) return false;
  if (!v->is_string()) {
    c.warn_type(key, "a string");
    return false;
  }
  const std::string& s = v->get_ref<const std::string&>();
  if (s.size() > kMaxViewFileString) {
    c.fatal(ViewFileErrorKind::Invalid, "'" + c.prefix + key + "' is longer than " +
                                            std::to_string(kMaxViewFileString) + " bytes");
    return false;
  }
  out = s;
  return true;
}

void read_opt_bool(Ctx& c, const json& o, const char* key, std::optional<bool>& dst) {
  std::optional<bool> v;
  if (read_bool(c, o, key, v)) dst = v;
}

void read_opt_string(Ctx& c, const json& o, const char* key, std::optional<std::string>& dst) {
  std::string v;
  if (read_string(c, o, key, v)) dst = std::move(v);
}

// A filter string restored into a UI char buffer of `buf_bytes` (NUL included): cut it
// to what the box can hold, at a UTF-8 character boundary, and warn. The box would
// otherwise show a prefix of the string that is actually filtering.
void fit_ui_text(Ctx& c, const char* key, std::optional<std::string>& v, size_t buf_bytes) {
  if (!v || v->size() < buf_bytes) return;
  const size_t was = v->size();
  size_t cut = buf_bytes - 1;
  while (cut > 0 && (static_cast<unsigned char>((*v)[cut]) & 0xC0u) == 0x80u) --cut;
  v->resize(cut);
  c.warnings.push_back("'" + std::string(key) + "' is " + std::to_string(was) +
                       " bytes but the filter box holds " + std::to_string(buf_bytes - 1) +
                       "; cut to fit");
}

template <typename T>
void read_opt_int(Ctx& c, const json& o, const char* key, int64_t lo, int64_t hi,
                  std::optional<T>& dst) {
  int64_t v = 0;
  if (read_int(c, o, key, lo, hi, v)) dst = static_cast<T>(v);
}

void read_camera(Ctx& c, const json& root, ViewFile& f) {
  const json* cam = find_key(root, "cam");
  if (!cam) return;
  if (!cam->is_object()) {
    c.warn_type("cam", "an object");
    return;
  }
  c.prefix = "cam.";
  float v = 0;
  if (read_float(c, *cam, "pan_x", v)) f.pan_x = v;
  if (read_float(c, *cam, "pan_y", v)) f.pan_y = v;
  if (read_float(c, *cam, "zoom", v)) {
    if (v <= 0.0f) {
      c.warn_range("zoom", std::to_string(v), "a number > 0");
    } else {
      f.zoom = std::clamp(v, kViewFileMinZoom, kViewFileMaxZoom);
    }
  }
  c.prefix.clear();
}

void read_expanded(Ctx& c, const json& root, ViewFile& f) {
  const json* v = find_key(root, "expanded");
  if (!v) return;
  if (!v->is_array()) {
    c.warn_type("expanded", "an array of booleans");
    return;
  }
  if (v->size() > kMaxViewFileGroups) {
    c.fatal(ViewFileErrorKind::Invalid,
            "'expanded' has more than " + std::to_string(kMaxViewFileGroups) + " entries");
    return;
  }
  std::vector<bool> bits;
  bits.reserve(v->size());
  for (const json& e : *v) {
    if (!e.is_boolean()) {
      c.warn_type("expanded", "an array of booleans");
      return;
    }
    bits.push_back(e.get<bool>());
  }
  f.expanded = std::move(bits);
}

void read_pinned(Ctx& c, const json& root, ViewFile& f) {
  const json* v = find_key(root, "pinned");
  if (!v) return;
  if (!v->is_array()) {
    c.warn_type("pinned", "an array of node indices");
    return;
  }
  if (v->size() > kMaxViewFilePinned) {
    c.fatal(ViewFileErrorKind::Invalid,
            "'pinned' has more than " + std::to_string(kMaxViewFilePinned) + " entries");
    return;
  }
  std::vector<uint32_t> pins;
  pins.reserve(v->size());
  for (const json& e : *v) {
    if (!e.is_number_unsigned() || e.get<uint64_t>() > static_cast<uint64_t>(kU32Max)) {
      c.warn_type("pinned", "an array of node indices (unsigned 32-bit integers)");
      return;
    }
    pins.push_back(static_cast<uint32_t>(e.get<uint64_t>()));
  }
  f.pinned = std::move(pins);
}

void read_heatmap_metric(Ctx& c, const json& root, ViewFile& f) {
  const json* v = find_key(root, "heatmap_metric");
  if (!v) return;
  if (!v->is_string()) {
    c.warn_type("heatmap_metric", "a string");
    return;
  }
  const std::string& s = v->get_ref<const std::string&>();
  // heatmap_metric_from_name maps every unknown name to FLOPs, so it cannot detect
  // one by itself: accept only a name that round-trips.
  const HeatmapMetric m = heatmap_metric_from_name(s.c_str());
  if (s != heatmap_metric_name(m)) {
    c.warnings.push_back("'heatmap_metric' is unknown ('" + s.substr(0, 64) +
                         "'; expected FLOPs, Params, Act bytes or Arith intensity); ignored");
    return;
  }
  f.heatmap_metric = m;
}

void read_collapse(Ctx& c, const json& root, ViewFile& f) {
  const json* v = find_key(root, "collapse");
  if (!v) return;
  if (!v->is_string()) {
    c.warn_type("collapse", "\"all\" or \"none\"");
    return;
  }
  const std::string& s = v->get_ref<const std::string&>();
  if (s == "all") f.collapse = CollapseShorthand::All;
  else if (s == "none") f.collapse = CollapseShorthand::None;
  else
    c.warnings.push_back("'collapse' is unknown ('" + s.substr(0, 64) +
                         "'; expected \"all\" or \"none\"); ignored");
}

}  // namespace

// ---------------------------------------------------------------------------
// parse
// ---------------------------------------------------------------------------
ViewFileLoad parse_view_file(std::string_view bytes) {
  ViewFileLoad out;
  auto fail = [&out](ViewFileErrorKind k, std::string msg) {
    out.file = ViewFile{};
    out.warnings.clear();
    out.error_kind = k;
    out.error = std::move(msg);
    return out;
  };

  if (bytes.size() > kMaxViewFileBytes)
    return fail(ViewFileErrorKind::TooLarge,
                "view file is " + std::to_string(bytes.size()) + " bytes; the limit is " +
                    std::to_string(kMaxViewFileBytes));
  // nlohmann::json::parse is recursive and has no depth limit: pre-scan.
  if (!json_nesting_ok(bytes, kMaxViewFileDepth))
    return fail(ViewFileErrorKind::TooDeep,
                "view file nests deeper than " + std::to_string(kMaxViewFileDepth) + " levels");

  json root;
  try {
    root = json::parse(bytes.data(), bytes.data() + bytes.size());
  } catch (const json::parse_error& e) {
    std::string what = e.what();
    if (what.size() > 200) what.resize(200);
    return fail(ViewFileErrorKind::NotJson,
                "not valid JSON at byte " + std::to_string(e.byte) + ": " + what);
  } catch (const std::exception& e) {
    return fail(ViewFileErrorKind::NotJson, std::string("could not parse as JSON: ") + e.what());
  }

  if (!root.is_object())
    return fail(ViewFileErrorKind::NotAViewFile,
                "not a NetVis view file (the top level is not an object)");
  const json* kind = find_key(root, "kind");
  if (!kind || !kind->is_string() || kind->get_ref<const std::string&>() != "netvis-view")
    return fail(ViewFileErrorKind::NotAViewFile,
                "not a NetVis view file (missing \"kind\": \"netvis-view\")");

  if (const json* ver = find_key(root, "version")) {
    if (ver->is_number_unsigned()) {
      const uint64_t v = ver->get<uint64_t>();
      if (v == 0) return fail(ViewFileErrorKind::Invalid, "\"version\" must be 1 or greater");
      if (v > 1)
        return fail(ViewFileErrorKind::NewerVersion,
                    "view file is version " + std::to_string(v) +
                        "; this NetVis reads version 1");
    } else {
      return fail(ViewFileErrorKind::Invalid, "\"version\" must be a positive integer");
    }
  }

  Ctx c;
  ViewFile& f = out.file;

  // `model` is a plain string ("" = none recorded), not an optional.
  {
    std::string m;
    if (read_string(c, root, "model", m)) f.model = std::move(m);
  }
  read_opt_int<uint32_t>(c, root, "graph", 0, kU32Max, f.graph);
  read_camera(c, root, f);

  read_opt_bool(c, root, "hide_const_edges", f.hide_const_edges);
  read_opt_bool(c, root, "show_layer_bands", f.show_layer_bands);
  read_opt_bool(c, root, "show_critical_path", f.show_critical_path);
  read_opt_bool(c, root, "cost_heatmap", f.cost_heatmap);
  read_opt_bool(c, root, "show_search_results", f.show_search_results);
  read_opt_bool(c, root, "diff_panel_open", f.diff_panel_open);
  read_opt_bool(c, root, "heatmap_log_scale", f.heatmap_log_scale);
  read_opt_bool(c, root, "follow_preds", f.follow_preds);
  read_opt_bool(c, root, "follow_succs", f.follow_succs);
  read_opt_bool(c, root, "filter_active", f.filter_active);

  read_opt_int<int>(c, root, "edge_routing", 0, 2, f.edge_routing);
  read_heatmap_metric(c, root, f);
  read_opt_int<uint32_t>(c, root, "category_mask", 0, kU32Max, f.category_mask);
  read_opt_int<uint32_t>(c, root, "nav_hops", 0, kU32Max, f.nav_hops);
  read_opt_int<uint8_t>(c, root, "nav_mode", 0, 2, f.nav_mode);

  read_opt_string(c, root, "search_query", f.search_query);
  read_opt_string(c, root, "attr_filter", f.attr_filter);
  read_opt_string(c, root, "table_filter", f.table_filter);
  fit_ui_text(c, "search_query", f.search_query, kSearchQueryUiBytes);
  fit_ui_text(c, "attr_filter", f.attr_filter, kAttrFilterUiBytes);
  fit_ui_text(c, "table_filter", f.table_filter, kTableFilterUiBytes);

  read_collapse(c, root, f);
  read_expanded(c, root, f);

  read_opt_int<int32_t>(c, root, "selected_ir_node", -1, kI32Max, f.selected_ir_node);
  read_opt_int<int32_t>(c, root, "selected_value", -1, kI32Max, f.selected_value);
  read_opt_int<int32_t>(c, root, "path_a", -1, kI32Max, f.path_a);
  read_opt_int<int32_t>(c, root, "path_b", -1, kI32Max, f.path_b);
  read_pinned(c, root, f);

  // Unknown keys are ignored silently: forward compatibility.
  if (c.failed()) return fail(c.err, std::move(c.err_msg));
  out.warnings = std::move(c.warnings);
  return out;
}

ViewFileLoad read_view_file(const std::string& path) {
  namespace fs = std::filesystem;
  ViewFileLoad r;
  auto io_fail = [&r](ViewFileErrorKind k, std::string msg) {
    r.error_kind = k;
    r.error = std::move(msg);
    return r;
  };

  std::error_code ec;
  const uintmax_t size = fs::file_size(fs::path(path), ec);
  if (ec) return io_fail(ViewFileErrorKind::Io, "cannot read view file '" + path + "': " + ec.message());
  // BEFORE any byte is read: a hostile huge file costs nothing.
  if (size > kMaxViewFileBytes)
    return io_fail(ViewFileErrorKind::TooLarge,
                   "view file is " + std::to_string(size) + " bytes; the limit is " +
                       std::to_string(kMaxViewFileBytes));

  std::string buf(static_cast<size_t>(size), '\0');
  std::ifstream f(fs::path(path), std::ios::binary);
  if (!f) return io_fail(ViewFileErrorKind::Io, "cannot open view file '" + path + "'");
  if (size > 0) {
    f.read(buf.data(), static_cast<std::streamsize>(size));
    if (static_cast<uintmax_t>(f.gcount()) != size)
      return io_fail(ViewFileErrorKind::Io, "short read on view file '" + path + "'");
  }
  ViewFileLoad parsed = parse_view_file(buf);
  if (parsed.ok()) {
    // Where a relative "model" is also looked for (see ViewFile::source_dir).
    std::error_code abs_ec;
    const fs::path abs = fs::absolute(fs::path(path), abs_ec);
    if (!abs_ec) parsed.file.source_dir = abs.parent_path().string();
  }
  return parsed;
}

// ---------------------------------------------------------------------------
// serialize / from_snapshot
// ---------------------------------------------------------------------------
std::string serialize_view_file(const ViewFile& f) {
  json j = json::object();
  j["kind"] = "netvis-view";
  j["version"] = 1;
  if (!f.model.empty()) j["model"] = f.model;
  if (f.graph) j["graph"] = *f.graph;

  if (f.has_camera()) {
    json cam = json::object();
    if (f.pan_x) cam["pan_x"] = *f.pan_x;
    if (f.pan_y) cam["pan_y"] = *f.pan_y;
    if (f.zoom) cam["zoom"] = *f.zoom;
    j["cam"] = std::move(cam);
  }

  if (f.hide_const_edges) j["hide_const_edges"] = *f.hide_const_edges;
  if (f.show_layer_bands) j["show_layer_bands"] = *f.show_layer_bands;
  if (f.show_critical_path) j["show_critical_path"] = *f.show_critical_path;
  if (f.cost_heatmap) j["cost_heatmap"] = *f.cost_heatmap;
  if (f.show_search_results) j["show_search_results"] = *f.show_search_results;
  if (f.diff_panel_open) j["diff_panel_open"] = *f.diff_panel_open;
  if (f.heatmap_log_scale) j["heatmap_log_scale"] = *f.heatmap_log_scale;
  if (f.follow_preds) j["follow_preds"] = *f.follow_preds;
  if (f.follow_succs) j["follow_succs"] = *f.follow_succs;
  if (f.filter_active) j["filter_active"] = *f.filter_active;

  if (f.edge_routing) j["edge_routing"] = *f.edge_routing;
  if (f.heatmap_metric) j["heatmap_metric"] = heatmap_metric_name(*f.heatmap_metric);
  if (f.category_mask) j["category_mask"] = *f.category_mask;
  if (f.nav_hops) j["nav_hops"] = *f.nav_hops;
  if (f.nav_mode) j["nav_mode"] = static_cast<unsigned>(*f.nav_mode);

  if (f.search_query) j["search_query"] = *f.search_query;
  if (f.attr_filter) j["attr_filter"] = *f.attr_filter;
  if (f.table_filter) j["table_filter"] = *f.table_filter;

  if (f.collapse) j["collapse"] = (*f.collapse == CollapseShorthand::All) ? "all" : "none";
  if (f.expanded) {
    json arr = json::array();
    for (bool b : *f.expanded) arr.push_back(b);
    j["expanded"] = std::move(arr);
  }

  if (f.selected_ir_node) j["selected_ir_node"] = *f.selected_ir_node;
  if (f.selected_value) j["selected_value"] = *f.selected_value;
  if (f.path_a) j["path_a"] = *f.path_a;
  if (f.path_b) j["path_b"] = *f.path_b;
  if (f.pinned) {
    json arr = json::array();
    for (uint32_t p : *f.pinned) arr.push_back(p);
    j["pinned"] = std::move(arr);
  }

  // `replace`, not the default strict handler: the document carries a model PATH,
  // which is not guaranteed to be valid UTF-8 on Linux, and strict would throw out
  // of a save the user explicitly asked for.
  //
  // Indented, which costs ~11 bytes per `expanded` entry ("    false,\n"): over the
  // reader's 4 MiB cap from ~380k groups. Past the cap use the compact form (~6 bytes
  // per entry) rather than write a file this build then refuses to load.
  std::string pretty = j.dump(2, ' ', false, json::error_handler_t::replace);
  if (pretty.size() <= kMaxViewFileBytes) return pretty;
  return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

ViewFile view_file_from_snapshot(const ViewSnapshot& s, const std::string& model_path,
                                 uint32_t graph, HeatmapMetric heatmap_metric,
                                 bool heatmap_log_scale, std::vector<std::string>* dropped) {
  ViewFile f;
  auto drop = [dropped](std::string what) {
    if (dropped != nullptr) dropped->push_back(std::move(what));
  };
  // Anything the reader would reject as over-limit is left OUT (and named), so a save
  // does not write a file this build then refuses to load. The size of the finished
  // text is the caller's check (serialize_view_file).
  if (model_path.size() <= kMaxViewFileString) f.model = model_path;
  else drop("the model path is longer than " + std::to_string(kMaxViewFileString) + " bytes");
  f.graph = graph;
  f.pan_x = s.pan_x;
  f.pan_y = s.pan_y;
  f.zoom = s.zoom;

  f.hide_const_edges = s.hide_const_edges;
  f.show_layer_bands = s.show_layer_bands;
  f.show_critical_path = s.show_critical_path;
  f.cost_heatmap = s.cost_heatmap;
  f.show_search_results = s.show_search_results;
  f.diff_panel_open = s.diff_panel_open;
  f.heatmap_log_scale = heatmap_log_scale;
  f.follow_preds = s.follow_preds;
  f.follow_succs = s.follow_succs;
  f.filter_active = s.filter_active;

  f.edge_routing = s.edge_routing;
  f.heatmap_metric = heatmap_metric;
  f.category_mask = s.category_mask;
  f.nav_hops = s.nav_hops;
  f.nav_mode = s.nav_mode;

  auto text = [&](const std::string& v, std::optional<std::string>& dst, const char* name) {
    if (v.size() <= kMaxViewFileString) dst = v;
    else drop(std::string("'") + name + "' is longer than " + std::to_string(kMaxViewFileString) + " bytes");
  };
  text(s.search_query, f.search_query, "search_query");
  text(s.attr_filter, f.attr_filter, "attr_filter");
  text(s.table_filter, f.table_filter, "table_filter");

  if (s.expanded.size() <= kMaxViewFileGroups) f.expanded = s.expanded;
  else drop("the collapse state (" + std::to_string(s.expanded.size()) + " groups, the limit is " +
            std::to_string(kMaxViewFileGroups) + ")");
  f.selected_ir_node = s.selected_ir_node;
  f.selected_value = s.selected_value;
  f.path_a = s.path_a;
  f.path_b = s.path_b;
  std::vector<uint32_t> pins = s.pinned;
  if (pins.size() > kMaxViewFilePinned) {
    drop("only the first " + std::to_string(kMaxViewFilePinned) + " of " +
         std::to_string(pins.size()) + " pinned nodes");
    pins.resize(kMaxViewFilePinned);
  }
  f.pinned = std::move(pins);
  return f;
}

std::string model_path_for_view_file(const std::string& model_path,
                                     const std::string& view_file_path) {
  namespace fs = std::filesystem;
  if (model_path.empty() || view_file_path.empty()) return model_path;
  std::error_code e1, e2;
  fs::path m = fs::absolute(fs::path(model_path), e1).lexically_normal();
  const fs::path dir = fs::absolute(fs::path(view_file_path), e2).lexically_normal().parent_path();
  if (e1 || e2 || dir.empty()) return model_path;
  // `bundle.mlpackage/` and `bundle.mlpackage` are the same bundle.
  if (!m.has_filename() && m.has_relative_path()) m = m.parent_path();
  const fs::path rel = m.lexically_relative(dir);
  if (rel.empty() || rel == "." || *rel.begin() == "..") return model_path;
  return rel.generic_string();  // '/' separators on every platform, so the file travels
}

// ---------------------------------------------------------------------------
// same_model_path
// ---------------------------------------------------------------------------
bool same_model_path(const std::string& saved, const std::string& live,
                     const std::string& view_dir) {
  namespace fs = std::filesystem;
  if (saved.empty() || live.empty()) return false;
  if (saved == live) return true;
  std::error_code e;
  // `dir` and `dir/` are the same bundle: drop a trailing separator before comparing.
  auto strip = [](fs::path& p) {
    if (!p.has_filename() && p.has_relative_path()) p = p.parent_path();
  };
  fs::path b = fs::weakly_canonical(fs::path(live), e);
  if (e) return false;  // the plain string comparison above already said no
  strip(b);
  auto names_live = [&](const fs::path& candidate) {
    std::error_code ec;
    fs::path a = fs::weakly_canonical(candidate, ec);
    if (ec) return false;
    strip(a);
    return a == b;
  };
  const fs::path saved_path(saved);
  if (names_live(saved_path)) return true;  // as written: absolute, or relative to the CWD
  // A relative path in a file means "next to the file" to whoever wrote it by hand or
  // committed it with its model.
  if (!view_dir.empty() && saved_path.is_relative() && names_live(fs::path(view_dir) / saved_path))
    return true;
  return false;
}

// ---------------------------------------------------------------------------
// overlay
// ---------------------------------------------------------------------------
CollapsePlan plan_collapse(const ViewFile& f, const std::vector<bool>& live_expanded,
                           bool same_model) {
  CollapsePlan plan;
  plan.expanded = live_expanded;
  if (f.expanded) {
    if (same_model) {
      if (f.expanded->size() == live_expanded.size()) {
        plan.expanded = *f.expanded;
        return plan;  // the exact bitset wins over the shorthand
      }
      plan.notes.push_back("collapse state in view file has " +
                           std::to_string(f.expanded->size()) + " groups, model has " +
                           std::to_string(live_expanded.size()) + "; ignored");
    } else {
      plan.notes.push_back("collapse bitset is model-specific; ignored");
    }
  }
  if (f.collapse) {
    // true = expanded: All collapses every group (false), None expands every group.
    plan.expanded.assign(live_expanded.size(), *f.collapse == CollapseShorthand::None);
  }
  return plan;
}

ViewOverlayResult overlay_view_file(const ViewFile& f, const ViewSnapshot& base,
                                    bool same_model, uint32_t node_count,
                                    uint32_t value_count) {
  ViewOverlayResult r;
  r.snapshot = base;
  ViewSnapshot& s = r.snapshot;

  // --- model-agnostic -----------------------------------------------------------
  if (f.pan_x && std::isfinite(*f.pan_x)) s.pan_x = *f.pan_x;
  if (f.pan_y && std::isfinite(*f.pan_y)) s.pan_y = *f.pan_y;
  if (f.zoom && std::isfinite(*f.zoom) && *f.zoom > 0.0f)
    s.zoom = std::clamp(*f.zoom, kViewFileMinZoom, kViewFileMaxZoom);

  if (f.hide_const_edges) s.hide_const_edges = *f.hide_const_edges;
  if (f.show_layer_bands) s.show_layer_bands = *f.show_layer_bands;
  if (f.show_critical_path) s.show_critical_path = *f.show_critical_path;
  if (f.cost_heatmap) s.cost_heatmap = *f.cost_heatmap;
  if (f.show_search_results) s.show_search_results = *f.show_search_results;
  if (f.diff_panel_open) s.diff_panel_open = *f.diff_panel_open;
  if (f.follow_preds) s.follow_preds = *f.follow_preds;
  if (f.follow_succs) s.follow_succs = *f.follow_succs;
  if (f.filter_active) s.filter_active = *f.filter_active;
  if (f.category_mask) s.category_mask = *f.category_mask;
  if (f.nav_hops) s.nav_hops = *f.nav_hops;
  if (f.search_query) s.search_query = *f.search_query;
  if (f.attr_filter) s.attr_filter = *f.attr_filter;
  if (f.table_filter) s.table_filter = *f.table_filter;

  // A ViewFile built by code (not by parse_view_file) is not guaranteed in range.
  if (f.edge_routing) {
    if (*f.edge_routing >= 0 && *f.edge_routing <= 2) s.edge_routing = *f.edge_routing;
    else r.notes.push_back("edge_routing " + std::to_string(*f.edge_routing) +
                           " is out of range; ignored");
  }
  if (f.nav_mode) {
    if (*f.nav_mode <= 2) s.nav_mode = *f.nav_mode;
    else r.notes.push_back("nav_mode " + std::to_string(*f.nav_mode) + " is out of range; ignored");
  }

  // --- collapse -----------------------------------------------------------------
  CollapsePlan plan = plan_collapse(f, base.expanded, same_model);
  s.expanded = std::move(plan.expanded);
  for (std::string& n : plan.notes) r.notes.push_back(std::move(n));

  // --- model-specific -----------------------------------------------------------
  if (!same_model) {
    if (f.has_model_specific()) {
      const std::string who = f.model.empty() ? "view file names no model"
                                              : "view file was saved for '" + f.model.substr(0, 256) + "'";
      r.notes.push_back(who +
                        "; model-specific settings (graph, selection, collapse bitset, path, "
                        "pins) were not applied");
    }
    return r;
  }

  auto apply_index = [&r](const std::optional<int32_t>& v, int32_t& dst, uint32_t count,
                          const char* name, const char* what) {
    if (!v) return;
    if (*v == -1 || (*v >= 0 && static_cast<uint32_t>(*v) < count)) {
      dst = *v;
    } else {
      // Never clamped to a nearby index: a wrong-but-plausible selection is worse
      // than none.
      r.notes.push_back(std::string(name) + " " + std::to_string(*v) +
                        " is out of range (the model has " + std::to_string(count) + " " +
                        what + "); ignored");
    }
  };
  apply_index(f.selected_ir_node, s.selected_ir_node, node_count, "selected_ir_node", "nodes");
  apply_index(f.path_a, s.path_a, node_count, "path_a", "nodes");
  apply_index(f.path_b, s.path_b, node_count, "path_b", "nodes");
  apply_index(f.selected_value, s.selected_value, value_count, "selected_value", "values");

  if (f.pinned) {
    std::vector<uint32_t> kept;
    kept.reserve(f.pinned->size());
    size_t dropped = 0;
    for (uint32_t p : *f.pinned) {
      if (p < node_count) kept.push_back(p);
      else ++dropped;
    }
    if (dropped > 0)
      r.notes.push_back(std::to_string(dropped) +
                        " pinned node(s) dropped: index out of range for this model");
    s.pinned = std::move(kept);
  }
  return r;
}

const char* view_file_outcome_text(ViewApplyOutcome o) {
  switch (o) {
    case ViewApplyOutcome::Full: return "View loaded";
    case ViewApplyOutcome::OtherModel: return "View loaded (camera only - saved for another model)";
    case ViewApplyOutcome::NoModel: return "View loaded (open a model to restore selection)";
    case ViewApplyOutcome::Failed: return "View could not be applied: the model or graph changed";
  }
  return "View loaded";
}

std::vector<ViewLoadToast> view_load_toasts(ViewApplyOutcome o,
                                            const std::vector<std::string>& notes) {
  std::vector<ViewLoadToast> out;
  if (o == ViewApplyOutcome::Failed) {
    out.push_back({view_file_outcome_text(o), true});
    return out;
  }
  std::string head = view_file_outcome_text(o);
  // The other two outcomes already say what was left out; a full load that dropped
  // settings must not read as a clean success.
  if (o == ViewApplyOutcome::Full && !notes.empty()) {
    head += " (" + std::to_string(notes.size()) +
            (notes.size() == 1 ? " setting ignored)" : " settings ignored)");
  }
  out.push_back({std::move(head), false});
  const size_t shown = std::min(notes.size(), kViewLoadMaxNoteToasts);
  for (size_t i = 0; i < shown; ++i) out.push_back({notes[i], false});
  if (notes.size() > shown)
    out.push_back({"... and " + std::to_string(notes.size() - shown) + " more ignored", false});
  return out;
}

}  // namespace netvis
