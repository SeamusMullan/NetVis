// SPDX-License-Identifier: Apache-2.0
// engine/ViewFile.h — the `.netvis-view` file: a shareable, reproducible view state.
//
// #56 introduced the file (camera, two toggles, category mask, path endpoints,
// selection). #170 (`netvis --screenshot --view`) needs it to carry the WHOLE view
// state, so it grows to exactly the ViewSnapshot field set (#106) plus the heatmap
// metric and scale. The envelope does not change: `"kind": "netvis-view"`,
// `"version": 1`. Every addition is an OPTIONAL key, so:
//   * old readers ignore keys they do not know (forward compatibility);
//   * this reader treats an absent key as "keep the live value" (the #56 contract),
//     so a file written by an older NetVis loads exactly as before.
//
// This is a pure core module (no ImGui, no GL): parsing, serialising, and the
// "which parts of this file apply to this model?" overlay are all unit-tested.
// The GUI-side applier (view/ViewFileApply.h) only moves the result into ViewState.
//
// HOSTILE INPUT. A view file is user-supplied JSON, so the reader follows the same
// rules as every parser here: bounded size (checked before reading), bounded
// nesting (pre-scanned before nlohmann::parse, which is recursive), bounded string
// and array lengths, exact integer ranges (no truncating get<int32_t>), finite
// floats, and no exception crossing the module boundary. Structural problems are
// ERRORS (nothing applies); a single bad VALUE is a WARNING (that key keeps its
// live value, the rest apply), so one stale key never costs a user their file.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "engine/HeatmapGradient.h"
#include "engine/ViewSnapshot.h"

namespace netvis {

// --- Limits (D4) ---------------------------------------------------------------
inline constexpr uint64_t kMaxViewFileBytes = 4ull * 1024 * 1024;  // checked before reading
inline constexpr int kMaxViewFileDepth = 8;                        // brace/bracket nesting
inline constexpr size_t kMaxViewFileString = 4096;                 // model, search_query, ...
inline constexpr size_t kMaxViewFileGroups = 1048576;              // `expanded` entries
inline constexpr size_t kMaxViewFilePinned = 4096;                 // `pinned` entries
// Same bounds as view/Camera.cpp's kMinZoom/kMaxZoom (the camera clamps to these
// on every interaction, so a file may not ask for more).
inline constexpr float kViewFileMinZoom = 0.02f;
inline constexpr float kViewFileMaxZoom = 4.0f;

// A model-independent collapse request: collapse every group / expand every group.
// Works on any model, unlike `expanded`, which is a bitset over THIS model's groups.
enum class CollapseShorthand : uint8_t { All, None };

// Every field is optional: absent means "keep the live value". `model` is the
// exception only because "" already means "no model recorded".
struct ViewFile {
  std::string model;                                   // "" when absent
  std::optional<uint32_t> graph;                       // model-specific (subgraph dive)

  // cam.* — each key independent, as in v1.
  std::optional<float> pan_x, pan_y, zoom;

  // Model-agnostic toggles.
  std::optional<bool> hide_const_edges, show_layer_bands, show_critical_path, cost_heatmap,
      show_search_results, diff_panel_open, heatmap_log_scale, follow_preds, follow_succs,
      filter_active;
  std::optional<int> edge_routing;                     // 0 Bezier, 1 Orthogonal, 2 Straight
  std::optional<HeatmapMetric> heatmap_metric;
  std::optional<uint32_t> category_mask, nav_hops;     // nav_hops UINT32_MAX = unlimited
  std::optional<uint8_t> nav_mode;                     // 0 None, 1 Highlight, 2 Focus
  std::optional<std::string> search_query, attr_filter, table_filter;

  // Collapse: a model-agnostic shorthand and/or a model-specific bitset.
  std::optional<CollapseShorthand> collapse;
  std::optional<std::vector<bool>> expanded;           // one per CollapseTree group; true = expanded

  // Model-specific (IR indices): only meaningful for the model the file was saved for.
  std::optional<int32_t> selected_ir_node, selected_value, path_a, path_b;
  std::optional<std::vector<uint32_t>> pinned;

  bool has_camera() const { return pan_x || pan_y || zoom; }

  // True if any key whose meaning depends on THE model is present.
  bool has_model_specific() const {
    return graph || expanded || selected_ir_node || selected_value || path_a || path_b ||
           pinned;
  }
};

enum class ViewFileErrorKind : uint8_t {
  None = 0,
  Io,            // missing / unreadable / short read
  TooLarge,      // > kMaxViewFileBytes (never read)
  TooDeep,       // nesting > kMaxViewFileDepth
  NotJson,       // syntax error; the message carries the byte offset
  NotAViewFile,  // valid JSON, but not an object with "kind": "netvis-view"
  NewerVersion,  // "version" > 1
  Invalid,       // bad "version", or a string/array over its limit
};

struct ViewFileLoad {
  ViewFile file;
  std::vector<std::string> warnings;     // ignored keys; human readable
  ViewFileErrorKind error_kind = ViewFileErrorKind::None;
  std::string error;                     // set iff error_kind != None
  bool ok() const { return error_kind == ViewFileErrorKind::None; }
};

// Parse a view file's bytes. Never throws. All-or-nothing on structure, key-by-key
// on values (see the file header).
ViewFileLoad parse_view_file(std::string_view bytes);

// Read `path` and parse it. The size is checked with file_size() BEFORE any byte is
// read, so a hostile 40 GB "view file" costs nothing.
ViewFileLoad read_view_file(const std::string& path);

// Deterministic JSON (nlohmann sorts keys) for every PRESENT field. Written with the
// `replace` UTF-8 error handler: the document carries a model PATH, which is not
// guaranteed to be valid UTF-8 on Linux, and a save must not throw.
std::string serialize_view_file(const ViewFile& f);

// The view file that reproduces `s`. Sets every field except `collapse` (a hand-
// written convenience that a snapshot, which has the full bitset, never needs).
ViewFile view_file_from_snapshot(const ViewSnapshot& s, const std::string& model_path,
                                 uint32_t graph, HeatmapMetric heatmap_metric,
                                 bool heatmap_log_scale);

// Is `saved` (the file's "model") the same model as `live`? false if either is empty;
// otherwise compares weakly_canonical forms, so `./m.onnx` and `/abs/m.onnx` match
// (the #56 reader used plain string equality and they did not); falls back to string
// equality if canonicalisation fails.
bool same_model_path(const std::string& saved, const std::string& live);

// --- Overlay: which parts of the file apply to THIS model? ----------------------

// The collapse configuration the overlay decided on, plus why.
struct CollapsePlan {
  std::vector<bool> expanded;           // one per live group; true = expanded
  std::vector<std::string> notes;
};

// Rules, in order:
//   * same model, `expanded` present, size == group count  -> the file's bitset;
//   * same model, `expanded` present, wrong size           -> note, then fall through;
//   * other model, `expanded` present                      -> note (a bitset is
//                                                             model-specific), fall through;
//   * `collapse` present: All -> every group collapsed (false), None -> every group
//     expanded (true), at the live size;
//   * otherwise                                            -> the live state.
CollapsePlan plan_collapse(const ViewFile& f, const std::vector<bool>& live_expanded,
                           bool same_model);

struct ViewOverlayResult {
  ViewSnapshot snapshot;
  std::vector<std::string> notes;
};

// Overlay `f` onto `base` (the capture_view() of the live state; base.expanded.size()
// is the live group count). Pure; nothing is applied to a session here.
//   * Every model-agnostic field present in `f` overwrites the snapshot (zoom clamped).
//   * The collapse plan above.
//   * Model-specific indices apply only when `same_model`, and only when in range
//     (selected_ir_node/path_a/path_b in [-1, node_count), selected_value in
//     [-1, value_count)); an out-of-range index keeps the base value and adds a note.
//     NEVER clamped to a nearby node: a wrong-but-plausible selection is worse than
//     none. `pinned` drops entries >= node_count, keeps order, and says how many.
//   * If !same_model and the file has model-specific keys: one note saying so.
ViewOverlayResult overlay_view_file(const ViewFile& f, const ViewSnapshot& base,
                                    bool same_model, uint32_t node_count,
                                    uint32_t value_count);

}  // namespace netvis
