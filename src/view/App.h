// view/App.h — application shell + shared view state (FROZEN view contract).
//
// DECISION (spec §8): the view is the only layer that touches ImGui. It talks to
// the engine solely through ModelSession. All panels share one mutable
// ViewState (camera, selection, search, inspector) owned by App; each panel is a
// free function draw_*(App&) in its own translation unit so panels are authored
// independently. App owns the GLFW window, the JobSystem, and the ModelSession,
// and drives the once-per-frame session.update() (spec §4).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "imgui.h"

#include "core/JobSystem.h"
#include "engine/DiffLoader.h"
#include "engine/HeatmapGradient.h"
#include "engine/ModelSession.h"
#include "engine/plugin/PluginPrefs.h"   // #11: PluginEnableSet, PluginKind
#include "engine/OpCategory.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"

struct GLFWwindow;

namespace netvis {

// Navigation state lives in its own view header; ViewState holds it by pointer
// so App.h stays light. Forward-declared here, defined in view/GraphNav.h.
struct GraphNavState;

// Static cost/analyzer report (v0.3.0). Forward-declared so App.h needn't include
// engine/CostModel.h; ViewState holds it by unique_ptr, rebuilt by ensure_cost().
struct CostReport;

// World<->screen transform for the graph canvas (spec §8.1). world*zoom+pan.
struct Camera {
  ImVec2 pan{0, 0};
  float zoom = 1.0f;
};

// Async tensor decode backing the weight inspector (spec §8.3). Owned by App;
// filled by a TensorDecodeJob completion on the main thread.
struct PendingDecode {
  ir::TensorRef tensor;
  bool active = false;   // inspector open for this tensor
  bool in_flight = false;
  bool done = false;
  bool ok = false;
  TensorStats stats;
  std::string error;
  uint64_t token = 0;    // guards against stale completions
};

// A toast message (errors / notices, spec §8.7, §6.5 TorchScript notice).
struct Toast {
  std::string text;
  float ttl = 5.0f;      // seconds remaining
  bool is_error = false;
};

// All shared, main-thread-only mutable UI state.
struct ViewState {
  Camera cam;

  // Selection is by display-node index into session.collapse().display_nodes().
  int32_t selected_display = -1;
  int32_t hovered_display = -1;
  int32_t selected_value = -1;   // an edge/value selected via properties jump

  // Search overlay (Ctrl+F).
  bool search_open = false;
  std::string search_query;
  int search_active_result = 0;

  // Camera animation (smooth fly-to on search/jump, spec §8.4).
  bool animating = false;
  ImVec2 anim_from_pan, anim_to_pan;
  float anim_from_zoom = 1, anim_to_zoom = 1;
  float anim_t = 0;

  // Tensor-table mode (spec §8.6).
  std::string table_filter;
  int table_sort_col = 0;        // 0=name,1=dtype,2=shape,3=params,4=bytes,5=offset
  bool table_sort_asc = true;
  int64_t table_selected_row = -1;
  int32_t table_selected_graph = -1;  // which graph/flat the row is in

  bool dark_theme = true;
  bool show_minimap = true;
  bool request_fit = false;      // set to trigger a fit-to-graph next frame
  bool show_plugins = false;     // v0.6.0 #11: the Plugins management window

  // --- v0.2.0 additions (append-only; see CONTRACTS.md: view/ is not frozen) --
  // Layout readability: hide constant/initializer input edges in the canvas and
  // show a "+N" badge on consumers instead. Pure view toggle, no re-layout.
  bool hide_const_edges = false;

  // Graph navigation: highlight/focus/filter state + derived display-space masks.
  // Held by pointer (forward-declared) so App.h needn't include GraphNav.h; the
  // instance is created lazily by ensure_nav(). App.cpp MUST include GraphNav.h
  // so the unique_ptr's deleter sees the complete type at ~App().
  std::unique_ptr<GraphNavState> nav;

  // Model diff: panel-open flag. The comparison model + diff live in DiffLoader
  // (App-owned engine object); this is just the panel's visibility toggle.
  bool diff_panel_open = false;

  // --- v0.3.0 additions (append-only) ----------------------------------------
  // Static cost/analyzer report for the current graph. Held by pointer
  // (forward-declared CostReport) and rebuilt lazily by ensure_cost(); null until
  // a model is loaded. cost_key_* record what `cost` was built for, so
  // ensure_cost() knows when to recompute (generation/graph/collapse changed).
  std::unique_ptr<CostReport> cost;
  uint64_t cost_key_generation = UINT64_MAX;
  uint32_t cost_key_graph = UINT32_MAX;
  uint64_t cost_key_collapse = 0;
  // Shape-enrichment epoch the cached report was built for. ONNX shape inference
  // mutates ValueInfo shapes in place AFTER the model is published (same
  // model/generation), so without this the report is built pre-inference (all
  // FLOPs unknown, peak=0) and served forever. See ModelSession::enrich_generation().
  uint64_t cost_key_enrich = UINT64_MAX;

  // Cost-heatmap overlay toggle: when true, GraphCanvas tints nodes by log(FLOPs)
  // via cost_tint_for_display (mutually exclusive with diff tint at the call site
  // — diff wins if both active).
  bool cost_heatmap = false;

  // --- v0.3.2 additions (append-only) ----------------------------------------
  // Heatmap gradient (preset or custom stops + reverse) and scale. Persisted to
  // view_prefs.json across sessions. Defaults to Viridis / log scale.
  HeatmapGradient heatmap_gradient;
  bool heatmap_log_scale = true;  // false => linear FLOPs normalization

  // Cached heatmap value range for the SELECTED metric, recomputed once per frame
  // in ensure_cost() (not per visible node — cost_tint_for_display and the legend
  // both read this). The valid flag is cleared when there is no cost report or no
  // display node yields a known non-zero metric value.
  // Type is engine/CostPanel's HeatmapRange; forward use only via the fields below
  // to keep App.h free of that include — stored as plain scalars. Held as double
  // (v0.4.0) so arithmetic-intensity (a fractional FLOP/byte) is representable.
  bool heatmap_range_valid = false;
  double heatmap_range_min = 0.0;
  double heatmap_range_max = 0.0;

  // --- v0.4.0 additions (append-only) ----------------------------------------
  // Which scalar the cost heatmap tints by (FLOPs / Params / Act bytes / Arith
  // intensity). Persisted to view_prefs.json under key "heatmap_metric"; a missing
  // key loads as Flops (backward compatible with pre-0.4.0 pref files). Enum lives
  // in engine/HeatmapGradient.h, already included by this header.
  HeatmapMetric heatmap_metric = HeatmapMetric::Flops;

  // --- v0.8.0 additions (append-only) ----------------------------------------
  // #60: Properties-panel attribute-name filter. Per-tab (lives here rather than
  // as a panel-local static) so switching tabs never carries a stale filter onto
  // another model's node. Cleared by draw_attributes when the selected node
  // changes; attr_filter_key records what node the text was typed against.
  std::string attr_filter;
  uint64_t attr_filter_key = UINT64_MAX;

  // --- v0.8.1 additions (append-only) ----------------------------------------
  // #20: draw faint per-Sugiyama-layer bands in the canvas (depth ruler). Pure
  // view toggle over NodeBox::layer; persisted in view_prefs.json.
  bool show_layer_bands = false;
  // #18: show a shape/dtype tooltip when hovering an edge (no click). Persisted.
  bool edge_tooltips = true;
};

// Pre-baked font sizes for LOD text (spec §8.1: switch to no-text LOD rather
// than drawing illegible glyphs).
struct Fonts {
  ImFont* body = nullptr;    // default UI
  ImFont* small = nullptr;   // node subtitle
  ImFont* bold = nullptr;    // node op_type / headers
};

// v0.8.0 (#62): one open model = one Tab. App owns a vector of these and an
// active index; session()/view()/jobs()/decode() redirect to the active tab so
// every panel free-function is UNCHANGED. Each tab owns its OWN JobSystem: the
// generation counter is per-pool, so a slow parse/layout in tab A must never
// cross-cancel tab B's in-flight jobs (the exact reason the diff pipeline has its
// own JobSystem — see App.h/DiffLoader.h). Order matters: jobs_ is declared
// BEFORE session_ so the session (which holds a JobSystem&) is destroyed first;
// ~Tab shuts the pool down before either dies.
struct Tab {
  std::unique_ptr<JobSystem> jobs;
  std::unique_ptr<ModelSession> session;
  ViewState view;
  PendingDecode decode;
  std::string title;   // basename shown on the tab; "(empty)" until a file opens

  Tab();
  ~Tab();
  Tab(const Tab&) = delete;
  Tab& operator=(const Tab&) = delete;
};

class App {
 public:
  App();
  ~App();

  App(const App&) = delete;
  App& operator=(const App&) = delete;

  // Initialize window + GL + ImGui. Returns false on failure. `initial_path` may
  // be empty (opened later via dialog / drop).
  bool init(const std::string& initial_path);

  // Run the main loop until the window closes. Returns process exit code.
  int run();

  // Open a file (from dialog / drop / CLI). Delegates to ModelSession.
  void open_file(const std::string& path);

  // Kick an async decode for the weight inspector (spec §8.3).
  void inspect_tensor(const ir::TensorRef& t);

  void add_toast(const std::string& text, bool is_error);

  // Accessors used by panel free functions — all redirect to the ACTIVE tab
  // (#62), so panels are agnostic to how many models are open.
  ModelSession& session() { return *tabs_[active_tab_]->session; }
  JobSystem& jobs() { return *tabs_[active_tab_]->jobs; }
  // Comparison-model loader for diff mode (v0.2.0). Backed by its OWN JobSystem
  // (diff_jobs_) so its generation counter never cross-cancels the primary
  // session's in-flight parse/layout/shape jobs. Shared across tabs (one diff at
  // a time, always against the active tab's model).
  DiffLoader& diff_loader() { return *diff_loader_; }
  ViewState& view() { return tabs_[active_tab_]->view; }
  PendingDecode& decode() { return tabs_[active_tab_]->decode; }

  // --- v0.8.0 (#62) multi-model tabs -----------------------------------------
  // Number of open tabs (always >= 1: an empty tab exists at startup).
  size_t tab_count() const { return tabs_.size(); }
  size_t active_tab() const { return active_tab_; }
  const std::string& tab_title(size_t i) const { return tabs_[i]->title; }
  // Switch the active tab (no reload — each tab keeps its fully-loaded state).
  void switch_tab(size_t i);
  // Open a brand-new empty tab and make it active.
  void new_tab();
  // Close tab `i`; if it was the last tab, a fresh empty one is created so the
  // app always has at least one. Adjusts active_tab_ to stay in range.
  void close_tab(size_t i);

  // --- v0.8.0 (#59) command palette ------------------------------------------
  // App-global (not per-tab) open flag for the Ctrl+P fuzzy command palette.
  // draw_command_palette() reads/clears it; handle_shortcuts() toggles it.
  bool& command_palette_open() { return command_palette_open_; }
  // Toggle dark/light theme + persist (used by a palette action and the menu).
  void set_theme(bool dark) { view().dark_theme = dark; apply_theme(dark); save_prefs(); }
  // Open the OS file dialog (used by a palette action + the File menu).
  void open_file_dialog();

  const Fonts& fonts() const { return fonts_; }
  std::vector<Toast>& toasts() { return toasts_; }
  GLFWwindow* window() const { return window_; }

  // Fixed op-category palette (dark-first, spec §8.1). Returns header color.
  static ImU32 category_color(OpCategory c, bool dark);

  // Export the current canvas view to PNG at 2x (spec §8.7).
  void export_view_png(const std::string& path);
  // Prompt for a path then export (used by the File menu + command palette).
  void export_view_dialog();

  // Recent files (persisted next to layout cache, spec §8.7).
  const std::vector<std::string>& recent_files() const { return recent_; }

  // Persist view preferences (heatmap gradient/scale, theme, toggles) to
  // view_prefs.json in the cache dir. Called by panels when a pref changes.
  // Best-effort; failure is silent (a missing prefs file just uses defaults).
  void save_prefs();

  // v0.7.0 (#11): per-plugin enable state (persisted in view_prefs "plugins").
  plugin::PluginEnableSet& plugin_enabled() { return plugin_enabled_; }
  // The trust gate consulted per discovered plugin (#11 §0.4).
  bool plugin_gate(std::string_view id, plugin::PluginKind kind) const {
    return plugin_enabled_.effective(id, kind);
  }
  // Reset the Registry to built-ins then re-discover under the current gate (§0.4).
  // Bumps the session generation so an in-flight cost/shape job over the transient
  // table is superseded (its result never published). Called at startup + on toggle.
  void reload_plugins();

 private:
  GLFWwindow* window_ = nullptr;
  // #62: open models. Always non-empty (an empty tab is created at startup and
  // whenever the last tab closes). active_tab_ indexes the currently shown one.
  std::vector<std::unique_ptr<Tab>> tabs_;
  size_t active_tab_ = 0;
  // Set when active_tab_ is moved programmatically (new_tab/close_tab/switch_tab/
  // open_file); tells draw_tab_bar to force ImGui's selection to match for ONE
  // frame, then it is cleared so user clicks own selection thereafter.
  bool want_tab_sync_ = false;
  bool command_palette_open_ = false;  // #59: Ctrl+P palette visibility
  // Second JobSystem dedicated to the comparison-model load/diff pipeline, kept
  // separate from the tabs' pools so the generation counters never interfere.
  std::unique_ptr<JobSystem> diff_jobs_;
  std::unique_ptr<DiffLoader> diff_loader_;
  Fonts fonts_;
  std::vector<Toast> toasts_;
  std::vector<std::string> recent_;
  plugin::PluginEnableSet plugin_enabled_;   // #11: persisted per-plugin enable state

  // Install the font-metric SizeFn on a tab's session (used by layout to measure
  // node label extents). Factored out so new_tab()/init() share one definition.
  void install_size_fn(Tab& tab);

  void frame();                 // one UI frame
  void draw_tab_bar();          // #62: the row of model tabs
  void apply_theme(bool dark);
  void handle_shortcuts();
  void load_recent();
  void save_recent();
  void load_prefs();  // read view_prefs.json into the active tab's view (best-effort)
  void add_recent(const std::string& path);
};

// --- Panel entry points (each implemented in its own .cpp under view/) --------
// All are called once per frame from App::frame(), main thread only.
void draw_graph_canvas(App& app);       // GraphCanvas.cpp (ImDrawList + culling)
void draw_properties_panel(App& app);   // PropertiesPanel.cpp
void draw_weight_inspector(App& app);   // WeightInspector.cpp
void draw_search_bar(App& app);         // SearchBar.cpp
void draw_minimap(App& app);            // Minimap.cpp (drawn inside canvas)
void draw_tensor_table(App& app);       // TensorTable.cpp (has_graph == false)
void draw_status_bar(App& app);         // StatusBar.cpp
void draw_toasts(App& app);             // StatusBar.cpp
void draw_command_palette(App& app);    // CommandPalette.cpp (#59)

// Camera helpers shared by canvas / search / minimap (Camera.cpp).
ImVec2 world_to_screen(const Camera& c, ImVec2 origin, ImVec2 world);
ImVec2 screen_to_world(const Camera& c, ImVec2 origin, ImVec2 screen);
// Animate the camera so `world_pt` centers in a viewport of `view_size`.
void animate_camera_to(ViewState& vs, ImVec2 world_pt, ImVec2 view_size, float target_zoom);
// Fit the whole layout bounds into `view_size` (spec §8.1 'F' key).
void fit_camera(ViewState& vs, ImVec2 bounds_min, ImVec2 bounds_max, ImVec2 view_size);

}  // namespace netvis
