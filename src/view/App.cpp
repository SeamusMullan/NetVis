// view/App.cpp — application shell: window/GL/ImGui lifetime, main loop, menus,
// theme, recent files, async tensor inspect, and PNG export.
//
// DECISION (spec §8): App is the ONLY owner of the GLFW window, the JobSystem,
// and the ModelSession, and it is the sole driver of the once-per-frame
// session.update() (spec §4). Panels are free functions that read App's shared
// ViewState; App itself never draws graph/panel content, it just orchestrates.
#define IMGUI_DEFINE_MATH_OPERATORS
// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h (via
// App.h) but not included there; pre-include it so App.h compiles.
#include "engine/LayoutEngine.h"
#include "view/App.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
// DockBuilder* (used once to seed the default panel layout) lives in the
// internal header. It is a stable-enough API in practice and only touched here.
#include "imgui_internal.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

// Raw GL entry points for the PNG export (glReadPixels/glViewport — GL 1.1 core,
// no extensions needed). Header location is platform-specific:
//  - Apple ships OpenGL under <OpenGL/gl.h>.
//  - Windows' <GL/gl.h> depends on WINGDIAPI/APIENTRY from <windows.h>, so that
//    must be included first.
//  - Other platforms just include <GL/gl.h>.
#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl.h>
#elif defined(_WIN32)
// <GL/gl.h> needs WINGDIAPI/APIENTRY from <windows.h>. Trim it (LEAN_AND_MEAN)
// and suppress the min/max macros so they don't collide with std::min/std::max.
// GLFW's header already defined APIENTRY; undef it so <windows.h> redefining it
// is not a C4005 macro-redefinition (which /WX turns into an error).
#ifdef APIENTRY
#undef APIENTRY
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#else
#include <GL/gl.h>
#endif

// stb_image_write is header-only; its implementation is compiled EXACTLY once,
// here (spec §8.7). Wrap it so its internal style does not trip -Werror. The
// pragmas are compiler-specific: GCC/Clang understand `#pragma GCC diagnostic`;
// MSVC would warn C4068 (unknown pragma) on those under /WX, so it uses
// `#pragma warning` instead (and C4996 sprintf is handled by the CRT define).
#if defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wtype-limits"
// macOS/clang flags stb's use of sprintf(3) as a deprecated-declaration error
// under -Werror; silence it just for this header (we do not call sprintf).
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <nlohmann/json.hpp>

#include "engine/DiffLoader.h"
#include "engine/LayoutCache.h"
#include "engine/TensorDiff.h"  // #50: cross-model same-tensor lookup
#include "engine/plugin/declarative/Manifest.h"  // v0.6.0 #9: plugin discovery
#include "engine/plugin/Registry.h"               // v0.7.0 #11: reset_to_builtins
// GraphNav.h defines GraphNavState so ViewState's unique_ptr<GraphNavState>
// deleter sees the complete type at ~App(); DiffPanel.h for draw_diff_panel.
#include "engine/CostModel.h"  // complete type for ViewState's unique_ptr<CostReport>
#include "view/CostPanel.h"
#include "view/DiffPanel.h"
#include "view/GraphNav.h"
#include "view/PanelHelpers.h"  // #56: display_index_for_node (re-select on load)
#include "view/PluginsPanel.h"
#include "view/Onboarding.h"        // #105: empty state + Help menu
#include "view/PreferencesPanel.h"  // #102: the unified Settings window
#include "view/SessionStore.h"      // #103: tabs + camera persistence
#include "view/ViewHistory.h"       // #106: capture_view / apply_view

// tinyfiledialogs ships only a .c/.h that is NOT on our include path; its two
// entry points are plain C, so we declare them ourselves (spec §8.7). At link
// time the tinyfiledialogs translation unit provides the definitions.
extern "C" {
char* tinyfd_openFileDialog(const char* aTitle, const char* aDefaultPathAndFile,
                            int aNumOfFilterPatterns,
                            const char* const* aFilterPatterns,
                            const char* aSingleFilterDescription,
                            int aAllowMultipleSelects);
char* tinyfd_saveFileDialog(const char* aTitle, const char* aDefaultPathAndFile,
                            int aNumOfFilterPatterns,
                            const char* const* aFilterPatterns,
                            const char* aSingleFilterDescription);
}

namespace netvis {

namespace {

// GLFW drop callback: forwards the first dropped path to App::open_file. The
// App* is stashed via glfwSetWindowUserPointer so this free function can reach it.
void drop_callback(GLFWwindow* window, int count, const char** paths) {
  if (count <= 0 || paths == nullptr) return;
  auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
  if (app != nullptr && paths[0] != nullptr) app->open_file(paths[0]);
}

// Try a short list of platform font files at `size`; return nullptr if none
// load. Keeps init() from hard-failing on machines without our preferred font.
ImFont* try_load_font(ImGuiIO& io, float size) {
  static const char* kCandidates[] = {
      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
      "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
      "/usr/share/fonts/TTF/DejaVuSans.ttf",
      "/System/Library/Fonts/SFNS.ttf",
      "C:\\Windows\\Fonts\\segoeui.ttf",
  };
  // v0.9.4: ImGui's default glyph range is U+0020..U+00FF, so every character
  // above that renders as the missing-glyph box — which this codebase reads as a
  // literal '?' on screen. The UI is full of typographic punctuation (em dashes
  // in DiffPanel, CostPanel, GraphNav and elsewhere), and it has been shipping
  // broken: an xvfb screenshot of the v0.9.4 empty state showed "what is inside
  // them ?" where an em dash belonged.
  //
  // Rather than hunt every string and flatten it to ASCII — which would have to
  // be re-policed on every new panel — extend the atlas once with the handful of
  // punctuation code points the UI actually uses. The array must outlive the
  // call: ImGui stores the POINTER, so a stack-local range would dangle and
  // produce a garbled atlas.
  static const ImWchar kRanges[] = {
      0x0020, 0x00FF,  // Basic Latin + Latin-1 Supplement (ImGui's default)
      0x2013, 0x2014,  // en dash, em dash
      0x2018, 0x201D,  // curly single + double quotes
      0x2026, 0x2026,  // horizontal ellipsis
      0x00D7, 0x00D7,  // multiplication sign (the collapse-group "xN" badge)
      0,
  };
  for (const char* path : kCandidates) {
    std::ifstream probe(path, std::ios::binary);
    if (!probe) continue;
    ImFont* f = io.Fonts->AddFontFromFileTTF(path, size, nullptr, kRanges);
    if (f != nullptr) return f;
  }
  return nullptr;
}

}  // namespace

// --- Tab (#62) -------------------------------------------------------------
// A tab owns its JobSystem, its ModelSession (which takes the pool by ref), its
// ViewState, and its PendingDecode. Construction order: pool first, then session.
Tab::Tab()
    : jobs(std::make_unique<JobSystem>()),
      session(std::make_unique<ModelSession>(*jobs)),
      title("(empty)") {}

Tab::~Tab() {
  // Stop this tab's workers BEFORE session is destroyed: a still-running decode
  // job could touch session->file() after session is gone. shutdown() joins
  // every worker first. (session is declared after jobs, so it destructs first
  // anyway, but a running job could be mid-callback — shutdown closes that.)
  if (jobs) jobs->shutdown();
}

App::App() = default;

App::~App() {
  // DECISION (threading): stop all workers BEFORE any member is destroyed. Each
  // tab shuts its own pool down in ~Tab; do it here explicitly first so nothing
  // races teardown of shared state.
  for (auto& t : tabs_)
    if (t && t->jobs) t->jobs->shutdown();
  // diff_loader_ owns jobs on diff_jobs_; stop those workers before diff_loader_
  // (and its captured shared_ptr<const ir::Model>) is destroyed. Members destruct
  // in reverse declaration order (diff_loader_ then diff_jobs_), so shutting the
  // pool down here first closes the window on a still-running diff job.
  if (diff_jobs_) diff_jobs_->shutdown();
}

// Install the font-metric size function on a tab's session (spec §8.1). Layout
// workers call this to measure each display node's box. It reads only glyph
// advance widths from the pre-baked atlas (immutable after init) plus the
// tab's own model/collapse state (published on the main thread before a layout
// job is queued). Captures the specific ModelSession* so a tab's layout always
// measures against ITS model, never the active tab's.
void App::install_size_fn(Tab& tab) {
  ModelSession* sess = tab.session.get();
  tab.session->set_size_fn([this, sess](const DisplayNode& dn) -> Vec2 {
    const float kPadX = 24.0f;   // horizontal breathing room around the label
    const float kLineH = 18.0f;  // one text line incl. leading
    const float kPadY = 14.0f;   // vertical padding (header strip + margins)

    std::string primary, secondary;
    const ir::Model* m = sess->model();  // a Tab always holds a non-null session
    if (dn.is_group) {
      const auto& groups = sess->collapse().groups();
      if (dn.group_index < groups.size()) {
        const auto& g = groups[dn.group_index];
        primary = g.label;
        secondary = "x" + std::to_string(g.instances);
      }
    } else if (m != nullptr) {
      uint32_t gi = sess->current_graph();
      if (gi < m->graphs.size()) {
        const auto& nodes = m->graphs[gi].nodes;
        if (dn.ir_node < nodes.size()) {
          const auto& n = nodes[dn.ir_node];
          primary = std::string(m->str(n.op_type));
          secondary = std::string(m->str(n.name));
        }
      }
    }
    if (primary.empty()) primary = "node";

    float w = 0.0f;
    if (fonts_.body != nullptr) {
      ImVec2 a = fonts_.bold->CalcTextSizeA(16.0f, FLT_MAX, 0.0f, primary.c_str());
      ImVec2 b = fonts_.small->CalcTextSizeA(12.0f, FLT_MAX, 0.0f,
                                             secondary.c_str());
      w = std::max(a.x, b.x);
    } else {
      w = 8.0f * static_cast<float>(std::max(primary.size(), secondary.size()));
    }
    Vec2 out;
    out.x = w + kPadX;
    out.y = 2.0f * kLineH + kPadY;  // ~two lines: op_type + name/subtitle.
    return out;
  });
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
bool App::init(const std::string& initial_path) {
  if (!glfwInit()) return false;

  // OpenGL 3.3 core + forward-compat (matches ImGui_ImplOpenGL3 "#version 330").
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

  window_ = glfwCreateWindow(1600, 1000, "NetVis", nullptr, nullptr);
  if (window_ == nullptr) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);  // VSync: cap to display refresh, no busy spinning.

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;  // dockable panels (spec §8).

  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  // Pre-bake three font handles for LOD text (spec §8.1). If no TTF is present,
  // fall back to the built-in font and reuse it for all three roles.
  fonts_.body = try_load_font(io, 16.0f);
  fonts_.small = try_load_font(io, 12.0f);
  fonts_.bold = try_load_font(io, 16.0f);
  if (fonts_.body == nullptr) {
    ImFont* def = io.Fonts->AddFontDefault();
    fonts_.body = def;
    fonts_.small = def;
    fonts_.bold = def;
  }
  if (fonts_.small == nullptr) fonts_.small = fonts_.body;
  if (fonts_.bold == nullptr) fonts_.bold = fonts_.body;

  // #62: start with a single empty tab FIRST — session()/view() resolve to the
  // active tab, and load_prefs/reload_plugins/apply_theme below all use them, so
  // the tab must exist before they run. Each tab owns its own JobSystem +
  // ModelSession (see Tab). install_size_fn wires the font-metric measurer.
  tabs_.clear();
  tabs_.push_back(std::make_unique<Tab>());
  active_tab_ = 0;
  install_size_fn(*tabs_.back());

  // Load persisted view prefs BEFORE applying the theme so a saved theme choice
  // and the heatmap gradient take effect on startup. Writes into the active
  // tab's ViewState via view().
  load_prefs();

  // v0.6.0 (#9) / v0.7.0 (#11): discover plugins under the trust gate. Declarative
  // plugins load freely (safe by construction); WASM plugins (#10) register only when
  // explicitly enabled (persisted in view_prefs "plugins", loaded above).
  reload_plugins();

  apply_theme(view().dark_theme);

  // Route OS file drops to open_file() via the window user pointer.
  glfwSetWindowUserPointer(window_, this);
  glfwSetDropCallback(window_, drop_callback);

  // Comparison-model diff pipeline runs on its OWN JobSystem so its generation
  // counter never cross-cancels a tab's in-flight parse/layout/shape jobs
  // (see App.h / DiffLoader.h).
  diff_jobs_ = std::make_unique<JobSystem>();
  diff_loader_ = std::make_unique<DiffLoader>(*diff_jobs_);

  load_recent();

  if (!initial_path.empty()) {
    // An explicit CLI path WINS over the remembered session: the user named a
    // file, and burying it under five restored tabs would ignore what they
    // asked for.
    open_file(initial_path);
  } else if (view().restore_session) {
    // #103: opt-in, so this is reached only when the pref was persisted on.
    const size_t skipped = restore_last_session();
    if (skipped > 0) {
      add_toast(std::to_string(skipped) +
                    " remembered file(s) could not be reopened",
                false);
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
int App::run() {
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
    // Drain EVERY tab's completions, not just the active one (#62): a background
    // tab's parse/layout/shape jobs must still land so switching to it shows a
    // finished model rather than a frozen loading state.
    for (auto& t : tabs_) t->session->update();
    if (diff_loader_) diff_loader_->update();  // drain diff completions after.
    frame();

    // Render the assembled draw data to the default framebuffer.
    ImGui::Render();
    int fb_w = 0, fb_h = 0;
    glfwGetFramebufferSize(window_, &fb_w, &fb_h);
    glViewport(0, 0, fb_w, fb_h);
    const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    glClearColor(bg.x, bg.y, bg.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window_);
  }

  // #103: capture the workspace BEFORE teardown, while the tabs and their
  // cameras still exist. A no-op unless the pref is on.
  save_session_now();

  // Orderly teardown: backends, context, window, GLFW.
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window_);
  window_ = nullptr;
  glfwTerminate();
  return 0;
}

// #62: the row of open-model tabs, drawn just under the menu bar. Uses an ImGui
// tab bar with a close button per tab and a trailing "+" to open a new empty
// tab. Switching is instant (each tab holds its fully-loaded session). Hidden
// entirely when there is a single still-empty tab so the startup UI is clean.
void App::draw_tab_bar() {
  const bool single_empty = tabs_.size() == 1 &&
                            session().stage() == LoadStage::Empty &&
                            session().path().empty();
  if (single_empty) return;

  const ImGuiTabBarFlags flags = ImGuiTabBarFlags_AutoSelectNewTabs |
                                 ImGuiTabBarFlags_Reorderable |
                                 ImGuiTabBarFlags_FittingPolicyScroll |
                                 ImGuiTabBarFlags_TabListPopupButton;
  if (ImGui::BeginTabBar("##model_tabs", flags)) {
    size_t to_close = tabs_.size();  // sentinel: nothing to close
    for (size_t i = 0; i < tabs_.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      bool open = true;
      // Label carries the title + a stable id so identical basenames don't merge.
      std::string label = tabs_[i]->title + "###tab" + std::to_string(i);
      // SetSelected only when a programmatic switch (command palette / new_tab)
      // has moved active_tab_ ahead of ImGui's own selection — force it that ONE
      // frame, then let ImGui own selection so user clicks are not fought.
      ImGuiTabItemFlags item_flags =
          (i == active_tab_ && want_tab_sync_) ? ImGuiTabItemFlags_SetSelected : 0;
      if (ImGui::BeginTabItem(label.c_str(), &open, item_flags)) {
        // A tab becomes active when its item is the selected one this frame.
        active_tab_ = i;
        ImGui::EndTabItem();
      }
      if (!open) to_close = i;  // user clicked the tab's close (x)
      ImGui::PopID();
    }
    // "+" trailing button opens a fresh empty tab.
    if (ImGui::TabItemButton("+", ImGuiTabItemFlags_Trailing |
                                      ImGuiTabItemFlags_NoTooltip)) {
      new_tab();
    }
    ImGui::EndTabBar();
    // Apply a close AFTER the loop so we never mutate tabs_ mid-iteration.
    if (to_close < tabs_.size()) close_tab(to_close);
  }
  // One-frame programmatic-selection sync consumed; user clicks own it now.
  want_tab_sync_ = false;
}

void App::frame() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  // A full-viewport dockspace so every panel is dockable (spec §8). We capture
  // its id so we can seed a sensible default arrangement on first run.
  ImGuiID dock_id = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

  // One-time default layout (spec §8): graph canvas in the center, properties on
  // the right, weight inspector docked below properties (bottom-right), tensor
  // table shares the center for graph-less files. Only built once — afterwards
  // ImGui persists whatever the user rearranges.
  // File-local one-shot (avoids touching the frozen App.h contract for a member).
  static bool dock_layout_built = false;
  if (!dock_layout_built) {
    dock_layout_built = true;
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, ImGui::GetMainViewport()->Size);

    ImGuiID center = dock_id;
    ImGuiID right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.28f,
                                                nullptr, &center);
    ImGuiID right_bottom = ImGui::DockBuilderSplitNode(
        right, ImGuiDir_Down, 0.45f, nullptr, &right);

    // #105: the empty state shares the centre node with the graph/table, so a
    // first-run user meets it where the content will appear.
    ImGui::DockBuilderDockWindow("Welcome", center);
    ImGui::DockBuilderDockWindow("Graph", center);
    ImGui::DockBuilderDockWindow("Tensors", center);
    ImGui::DockBuilderDockWindow("Properties", right);
    ImGui::DockBuilderDockWindow("Weight Inspector", right_bottom);
    ImGui::DockBuilderFinish(dock_id);
  }

  // --- Menu bar -------------------------------------------------------------
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Open...", "Ctrl+O")) open_file_dialog();
      if (ImGui::BeginMenu("Recent", !recent_.empty())) {
        for (const std::string& r : recent_) {
          if (ImGui::MenuItem(r.c_str())) open_file(r);
        }
        ImGui::EndMenu();
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Export View PNG...")) export_view_dialog();
      if (ImGui::MenuItem("Export View SVG...")) export_view_svg_dialog();  // #55
      // #56 (v0.8.2): shareable view-state file (camera + filters + selection).
      if (ImGui::MenuItem("Save View State...")) save_view_state_dialog();
      if (ImGui::MenuItem("Load View State...")) load_view_state_dialog();
      ImGui::Separator();
      if (ImGui::MenuItem("Exit")) glfwSetWindowShouldClose(window_, GLFW_TRUE);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Dark theme", nullptr, view().dark_theme)) {
        view().dark_theme = !view().dark_theme;
        apply_theme(view().dark_theme);
        save_prefs();
      }
      if (ImGui::MenuItem("Light theme", nullptr, !view().dark_theme)) {
        view().dark_theme = false;
        apply_theme(false);
        save_prefs();
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Minimap", nullptr, &view().show_minimap)) save_prefs();
      // Layout-readability toggle (v0.2.0 Feature 2): hide constant/initializer
      // input edges + source boxes; consumers get a "+N" badge instead.
      ImGui::MenuItem("Hide constant edges", nullptr, &view().hide_const_edges);
      // #20 depth ruler + #18 edge tooltips (v0.8.1). Both persisted.
      if (ImGui::MenuItem("Layer bands (depth ruler)", nullptr,
                          &view().show_layer_bands))
        save_prefs();
      if (ImGui::MenuItem("Edge shape tooltips", nullptr, &view().edge_tooltips))
        save_prefs();
      // #22 (v0.9.0): edge routing style.
      if (ImGui::BeginMenu("Edge routing")) {
        int& er = view().edge_routing;
        if (ImGui::MenuItem("Bezier", nullptr, er == 0)) { er = 0; save_prefs(); }
        if (ImGui::MenuItem("Orthogonal", nullptr, er == 1)) { er = 1; save_prefs(); }
        if (ImGui::MenuItem("Straight", nullptr, er == 2)) { er = 2; save_prefs(); }
        ImGui::EndMenu();
      }
      ImGui::Separator();
      // #13/#16 (v0.8.1): op-type legend + saved-view bookmarks panels. Held in
      // the per-tab GraphNavState; ensure it exists so the toggle has a target.
      if (view().nav) {
        ImGui::MenuItem("Op legend", nullptr, &view().nav->show_legend);
        ImGui::MenuItem("Bookmarks", nullptr, &view().nav->show_bookmarks);
      }
      // #53 (v0.8.2): persistent search-results panel (all hits, click to fly).
      ImGui::MenuItem("Search results", nullptr, &view().show_search_results);
      // #14 (v0.9.0): critical-path highlight (heaviest FLOP chain).
      ImGui::MenuItem("Critical path", nullptr, &view().show_critical_path);
      ImGui::Separator();
      // #21 (v0.8.1): global collapse/expand of all repeated-block groups.
      if (ImGui::MenuItem("Collapse all blocks", "C"))
        session().collapse_all();
      if (ImGui::MenuItem("Expand all blocks", "E"))
        session().expand_all();
      ImGui::Separator();
      // Graph navigation controls (v0.2.0): highlight/focus + category filter.
      if (ImGui::BeginMenu("Navigation")) {
        draw_nav_controls(*this);
        ImGui::EndMenu();
      }
      // Model diff panel visibility (v0.2.0).
      ImGui::MenuItem("Model diff panel", nullptr, &view().diff_panel_open);
      // Plugins management panel (v0.6.0 #11).
      ImGui::MenuItem("Plugins", nullptr, &view().show_plugins);
      ImGui::Separator();
      // #102: the Preferences window gathers every persisted setting. The menu
      // items above deliberately STAY — this is additive, and removing them
      // would break the muscle memory of everyone using the app today.
      ImGui::MenuItem("Preferences...", nullptr, &view().show_preferences);
      ImGui::EndMenu();
    }
    // #105: a Help menu, which the app did not have at all.
    if (ImGui::BeginMenu("Help")) {
      draw_help_menu(*this);
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  // --- Model tabs (#62) -----------------------------------------------------
  draw_tab_bar();

  // --- Content --------------------------------------------------------------
  // With a compute graph, the canvas owns the center; otherwise (GGUF / weight-
  // only files) we present the flat tensor table instead (spec §8.6).
  if (session().has_graph()) {
    // Refresh nav adjacency + display-space masks BEFORE the canvas reads them
    // (cheap no-op unless the nav cache key changed).
    ensure_nav(*this);
    // Rebuild the cost report if stale BEFORE the canvas reads cost tints (cheap
    // no-op unless generation/graph/collapse changed).
    ensure_cost(*this);
    draw_graph_canvas(*this);
  } else if (session().model() != nullptr && !session().has_graph()) {
    ensure_cost(*this);  // table-mode report (dtype/quant totals) for Properties
    draw_tensor_table(*this);
  }

  // Panels + overlays are always present (they self-hide when empty).
  draw_diff_panel(*this);
  draw_plugins_panel(*this);
  // #13/#16 (v0.8.1): op legend + bookmarks panels (self-hide via their toggles).
  draw_legend_panel(*this);
  draw_bookmarks_panel(*this);
  draw_properties_panel(*this);
  draw_weight_inspector(*this);
  draw_search_bar(*this);
  draw_search_results_panel(*this);  // #53: persistent results panel (self-hides)
  draw_status_bar(*this);
  draw_toasts(*this);
  // v0.9.4: all three self-guard (Preferences and Shortcuts on their ViewState
  // toggles, the empty state on there being no model), so they are unconditional
  // here exactly like the panels above.
  draw_preferences_panel(*this);   // #102
  draw_empty_state(*this);         // #105
  draw_shortcuts_window(*this);    // #105
  draw_command_palette(*this);   // #59: Ctrl+P fuzzy action palette (drawn on top)

  handle_shortcuts();

  // #106: record AFTER input and panels have run, so this frame's edits are what
  // gets captured. Cheap when nothing changed — ViewHistory drops a snapshot
  // identical to the current entry, which is the common case at 120 fps.
  record_view_history();

  // Age out toasts using the frame delta (spec §8.7).
  const float dt = ImGui::GetIO().DeltaTime;
  for (Toast& t : toasts_) t.ttl -= dt;
  toasts_.erase(std::remove_if(toasts_.begin(), toasts_.end(),
                               [](const Toast& t) { return t.ttl <= 0.0f; }),
                toasts_.end());
}

// ---------------------------------------------------------------------------
// Shortcuts
// ---------------------------------------------------------------------------
void App::handle_shortcuts() {
  ImGuiIO& io = ImGui::GetIO();
  // Never steal keys while the user is typing (e.g. in the search box).
  const bool typing = io.WantTextInput;

  if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
    view().search_open = !view().search_open;
  }
  // #59: Ctrl+P opens (toggles) the command palette. Allowed even while typing so
  // it can be summoned from any focused field; the palette grabs focus itself.
  if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_P, false)) {
    command_palette_open_ = !command_palette_open_;
  }
  // Ctrl+O opens a model (mirrors the File menu; a common muscle-memory chord).
  if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
    open_file_dialog();
  }
  if (!typing && !io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
    view().request_fit = true;  // 'F' fits the whole graph next frame.
  }
  if (!typing && ImGui::IsKeyPressed(ImGuiKey_Home, false)) {
    view().cam = Camera{};      // Home resets pan/zoom to identity.
    view().animating = false;
  }
  // #17 focus history: Alt+Left / Alt+Right step back/forward through visited
  // nodes (browser-style). Guarded so they no-op at the ends of the history.
  if (!typing && io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_LeftArrow, false)) {
    nav_focus_back(*this);
  }
  if (!typing && io.KeyAlt && ImGui::IsKeyPressed(ImGuiKey_RightArrow, false)) {
    nav_focus_forward(*this);
  }
  // #21 collapse/expand all repeated-block groups (no modifier; canvas focus).
  if (!typing && !io.KeyCtrl && !io.KeyAlt &&
      ImGui::IsKeyPressed(ImGuiKey_C, false)) {
    session().collapse_all();
  }
  if (!typing && !io.KeyCtrl && !io.KeyAlt &&
      ImGui::IsKeyPressed(ImGuiKey_E, false)) {
    session().expand_all();
  }
  // #106: undo/redo. Gated on `typing` because Ctrl+Z inside a text field means
  // "undo my typing" to every user alive, and stealing it to rewind the graph
  // would be actively hostile. Ctrl+Y and Ctrl+Shift+Z are both accepted — the
  // former is the Windows convention, the latter the macOS/Linux one, and a user
  // should not have to learn which this app picked.
  if (!typing && io.KeyCtrl && !io.KeyShift &&
      ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
    undo_view();
  }
  if (!typing && io.KeyCtrl &&
      (ImGui::IsKeyPressed(ImGuiKey_Y, false) ||
       (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z, false)))) {
    redo_view();
  }

  if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    view().search_open = false;
    command_palette_open_ = false;
    // Escape also closes the two v0.9.4 windows, matching every other overlay.
    view().show_preferences = false;
    view().show_shortcuts = false;
  }
}

// ---------------------------------------------------------------------------
// File open / recent / inspect
// ---------------------------------------------------------------------------
namespace {
// Basename for a tab title: the file name without directory. Falls back to the
// whole path if there is no separator.
std::string basename_of(const std::string& path) {
  auto slash = path.find_last_of("/\\");
  return slash == std::string::npos ? path : path.substr(slash + 1);
}
}  // namespace

void App::open_file_dialog() {
  const char* filters[] = {"*.onnx", "*.tflite", "*.safetensors",
                           "*.gguf", "*.pt",     "*.pth", "*.bin"};
  char* picked =
      tinyfd_openFileDialog("Open model", "", 7, filters, "Model files", 0);
  if (picked != nullptr) open_file(picked);
}

void App::open_file(const std::string& path) {
  if (path.empty()) return;
  // #62: reuse the active tab if it is still empty (never loaded a file); else
  // open the model in a fresh tab so the current one is not clobbered. This makes
  // "Open" additive once you already have a model up, matching the tabs mental
  // model, while the very first Open (startup empty tab) loads in place.
  if (session().stage() != LoadStage::Empty || !session().path().empty())
    new_tab();
  session().open_async(path);  // non-blocking (spec §4): pipeline kicks off.
  tabs_[active_tab_]->title = basename_of(path);
  add_recent(path);
}

// --- Tabs (#62) ------------------------------------------------------------
void App::switch_tab(size_t i) {
  if (i < tabs_.size()) { active_tab_ = i; want_tab_sync_ = true; }
}

void App::new_tab() {
  tabs_.push_back(std::make_unique<Tab>());
  active_tab_ = tabs_.size() - 1;
  want_tab_sync_ = true;
  install_size_fn(*tabs_.back());
  // A new tab inherits the current theme/toggles so the view is consistent; the
  // persisted prefs already live in the active view we are leaving, and per-tab
  // divergence (e.g. a different heatmap metric) is intentional and harmless.
}

void App::close_tab(size_t i) {
  if (i >= tabs_.size()) return;
  // #62: if a diff was loaded against THIS tab's session, clear it first — the
  // DiffLoader holds that session's address for identity comparison, and erasing
  // the tab frees the session, which would leave primary_session() dangling.
  //
  // #36: this must scan EVERY slot, not just the active one. Pre-N-way there was
  // a single comparison, so checking primary_session() was checking all of them.
  // Now a NON-active slot can be pinned to the closing session, and worse, a
  // non-active slot can be mid-load against it — that load's completion
  // dereferences the captured ModelSession* on the main thread, which is a
  // use-after-free once this tab's session is destroyed. clear() retires every
  // slot's token, so it cancels the in-flight loads as well as dropping the
  // pinned pointers.
  if (diff_loader_) {
    const ModelSession* closing = tabs_[i]->session.get();
    for (size_t k = 0; k < diff_loader_->comparison_count(); ++k) {
      if (diff_loader_->primary_session_of(k) != closing) continue;
      diff_loader_->clear();
      tabs_[i]->view.diff_panel_open = false;
      break;
    }
  }
  // ~Tab shuts the pool down (joins workers) before the ModelSession dies.
  tabs_.erase(tabs_.begin() + static_cast<long>(i));
  want_tab_sync_ = true;
  if (tabs_.empty()) {
    // Always keep at least one tab so session()/view() never dereference an
    // empty vector.
    tabs_.push_back(std::make_unique<Tab>());
    install_size_fn(*tabs_.back());
    active_tab_ = 0;
    return;
  }
  // Keep active_tab_ pointing at a sensible neighbor.
  if (active_tab_ >= tabs_.size()) active_tab_ = tabs_.size() - 1;
  else if (active_tab_ > i) --active_tab_;
}

void App::add_toast(const std::string& text, bool is_error) {
  Toast t;
  t.text = text;
  t.ttl = 5.0f;
  t.is_error = is_error;
  toasts_.push_back(std::move(t));
}

void App::inspect_tensor(const ir::TensorRef& t) {
  // #62: bind the decode to the tab that is active RIGHT NOW. If the user
  // switches tabs mid-decode, the completion must update THAT tab's PendingDecode
  // (and read THAT tab's session/file), never the newly-active one. Capture the
  // Tab* and its ModelSession/JobSystem by pointer; the Tab outlives the job
  // because ~Tab joins its workers before the Tab is destroyed.
  Tab* tab = tabs_[active_tab_].get();
  PendingDecode& decode = tab->decode;
  ModelSession* sess = tab->session.get();
  JobSystem* jobs = tab->jobs.get();

  // Mark the inspector busy for this tensor and bump the token so any older
  // in-flight decode's completion is ignored when it eventually lands.
  decode.tensor = t;
  decode.active = true;
  decode.in_flight = true;
  decode.done = false;
  decode.ok = false;
  decode.error.clear();
  const uint64_t token = ++decode.token;

  // v0.9.1b: the B-side compare (#50) and the quant preview (#49) are ABOUT the
  // tensor being inspected, so a new inspection invalidates both. Bump their
  // tokens (so an in-flight decode of the PREVIOUS tensor drops its result
  // instead of landing under the new tensor's header) and clear their published
  // state. Without this the panel can show one frame of the old tensor's B-side
  // numbers against the new tensor's name — a silent wrong answer, which is
  // worse than an empty panel. Both stay OFF until the user asks again: they are
  // opt-in per contract, and re-arming them here would make the preview fire as
  // a side effect of selection, which #49's scope explicitly forbids.
  ++decode.cmp_token;
  decode.cmp_requested = false;
  decode.cmp_in_flight = false;
  decode.cmp_done = false;
  decode.cmp_ok = false;
  decode.cmp_stats = TensorStats{};
  decode.cmp_error.clear();
  decode.cmp_label.clear();

  ++decode.quant_token;
  decode.quant_requested = false;
  decode.quant_in_flight = false;
  decode.quant_done = false;
  decode.quant = QuantBlockPreview{};
  decode.quant_error.clear();
  decode.quant_block = 0;

  // Copy the tensor by value into the job (a captured reference would dangle).
  // TensorStats is the ONLY payload-reading path (spec §7.5); it runs on a
  // worker so the UI never blocks decoding a multi-GB tensor. The token guards
  // against a stale result overwriting a newer inspection; ~Tab joins workers
  // before sess dies so sess->file() stays valid for the job's life.
  ir::TensorRef tc = t;
  jobs->submit([sess, jobs, &decode, token, tc]() {
    Result<TensorStats> r =
        compute_tensor_stats(tc, sess->file(), sess->model_dir(), sess->model());
    bool ok = r.ok();
    TensorStats stats = ok ? *r : TensorStats{};
    std::string errmsg = ok ? std::string() : r.error().message;
    jobs->post_to_main(
        [&decode, token, ok, stats, errmsg]() {
          if (decode.token != token) return;  // superseded — drop it.
          decode.stats = stats;
          decode.ok = ok;
          decode.error = errmsg;
          decode.done = true;
          decode.in_flight = false;
        });
  });
}

// #50: decode the SAME-NAMED tensor out of the active comparison model, so the
// inspector can show model A and model B side by side.
//
// LIFETIME is the whole difficulty here. The A-side decode above borrows the
// tab's session and mapping and is safe because ~Tab joins its workers first.
// The B side belongs to DiffLoader, which the user can mutate at any time —
// remove_comparison() while this decode is in flight would free the mapping the
// worker is reading. So this job borrows NOTHING from DiffLoader: it takes an
// owning shared_ptr on model B (model_ptr_of) and opens its OWN mapping of the
// comparison path on the worker. An mmap is ~1 ms, so a second mapping of an
// already-resident file is far cheaper than the machinery needed to make a
// borrowed one safe. The job then depends only on values it owns.
void App::inspect_tensor_comparison() {
  Tab* tab = tabs_[active_tab_].get();
  PendingDecode& decode = tab->decode;

  // Bump the token FIRST: every early return below is also a completion, and a
  // stale in-flight decode must not overwrite whatever we settle on here.
  const uint64_t token = ++decode.cmp_token;
  decode.cmp_requested = true;
  decode.cmp_in_flight = false;
  decode.cmp_done = false;
  decode.cmp_ok = false;
  decode.cmp_stats = TensorStats{};
  decode.cmp_error.clear();
  decode.cmp_label.clear();

  // Settle immediately with an honest reason rather than leaving the panel
  // spinning on a decode that will never be submitted.
  auto refuse = [&decode](std::string why) {
    decode.cmp_done = true;
    decode.cmp_error = std::move(why);
  };

  if (!decode.active) return refuse("no tensor is being inspected");

  DiffLoader& dl = *diff_loader_;
  const size_t slot = dl.active_comparison();
  decode.cmp_slot = slot;
  if (dl.comparison_count() == 0)
    return refuse("no comparison model is loaded");
  if (dl.state_of(slot) != DiffLoadState::Ready)
    return refuse("the comparison model is still loading");

  // #62: the comparison belongs to whichever tab loaded it. Comparing this tab's
  // tensor against another tab's comparison model would be a silent
  // wrong-answer, so require the identity match the tint path already requires.
  if (dl.primary_session_of(slot) != tab->session.get())
    return refuse("the comparison was loaded against a different tab");

  std::shared_ptr<const ir::Model> mb = dl.model_ptr_of(slot);
  const ir::Model* ma = tab->session->model();
  if (!mb || !ma) return refuse("no comparison model is loaded");

  const std::string_view name = ma->str(decode.tensor.name);
  if (name.empty()) return refuse("this tensor has no name to match on");

  const TensorLocator loc = find_tensor_by_name(*mb, name);
  const ir::TensorRef* tb = loc.valid() ? resolve_tensor(*mb, loc) : nullptr;
  if (!tb)
    return refuse("the comparison model has no tensor named \"" +
                  std::string(name) + "\"");

  // Label which rung of the ladder this came from, so a 3-way comparison is
  // unambiguous in the panel.
  const std::string& cmp_path = dl.path_of(slot);
  const size_t sep = cmp_path.find_last_of("/\\");
  decode.cmp_label =
      sep == std::string::npos ? cmp_path : cmp_path.substr(sep + 1);

  // Copy everything the worker touches. `tc` by value (a captured reference into
  // model B would dangle on removal even though the model itself is pinned —
  // the TensorRef lives in the model's vector, which is stable, but copying is
  // free at this size and removes the question entirely).
  const ir::TensorRef tc = *tb;
  const std::string dir = dl.model_dir_of(slot);
  decode.cmp_in_flight = true;

  // Runs on the DIFF JobSystem, not the tab's: this is comparison work, and its
  // generation counter must not cross-cancel the tab's parse/layout jobs. ~App
  // shuts diff_jobs_ down before diff_loader_ is destroyed.
  diff_jobs_->submit([this, &decode, token, tc, mb, cmp_path, dir]() {
    std::string errmsg;
    TensorStats stats;
    bool ok = false;

    auto mapped = MappedFile::open(cmp_path);
    if (!mapped) {
      errmsg = mapped.error().message;
    } else {
      const MappedFile file = std::move(*mapped);
      Result<TensorStats> r = compute_tensor_stats(tc, file, dir, mb.get());
      ok = r.ok();
      if (ok) {
        stats = *r;
      } else {
        errmsg = r.error().message;
      }
    }

    diff_jobs_->post_to_main([&decode, token, ok, stats, errmsg]() {
      if (decode.cmp_token != token) return;  // superseded — drop it.
      decode.cmp_stats = stats;
      decode.cmp_ok = ok;
      decode.cmp_error = errmsg;
      decode.cmp_done = true;
      decode.cmp_in_flight = false;
    });
  });
}

// #49: decode ONE quantized block of the inspected tensor for preview.
//
// Opt-in by contract, not merely by convention (see DECISIONS.md, "v0.9.1b —
// dequantization scope"): this is only ever reached from an explicit user action
// in the inspector, never as a side effect of inspect_tensor(). It reads at most
// one block — strictly fewer bytes than the histogram pass the inspector has
// already run over this same tensor.
void App::preview_tensor_quant_block(uint32_t block_index) {
  Tab* tab = tabs_[active_tab_].get();
  PendingDecode& decode = tab->decode;
  if (!decode.active) return;

  ModelSession* sess = tab->session.get();
  JobSystem* jobs = tab->jobs.get();

  decode.quant_requested = true;
  decode.quant_block = block_index;
  decode.quant_in_flight = true;
  decode.quant_done = false;
  decode.quant = QuantBlockPreview{};
  decode.quant_error.clear();
  const uint64_t token = ++decode.quant_token;

  // Same borrowing rules as inspect_tensor: the A-side session and mapping are
  // kept alive by ~Tab joining this pool before the session dies.
  const ir::TensorRef tc = decode.tensor;
  jobs->submit([sess, jobs, &decode, token, tc, block_index]() {
    Result<QuantBlockPreview> r = preview_quant_block(
        tc, sess->file(), block_index, sess->model_dir(), sess->model());
    const bool ok = r.ok();
    QuantBlockPreview p = ok ? *r : QuantBlockPreview{};
    std::string errmsg = ok ? std::string() : r.error().message;

    jobs->post_to_main([&decode, token, p, errmsg]() {
      if (decode.quant_token != token) return;  // superseded — drop it.
      decode.quant = p;
      decode.quant_error = errmsg;
      decode.quant_done = true;
      decode.quant_in_flight = false;
    });
  });
}

// ---------------------------------------------------------------------------
// v0.9.4 — undo/redo, session, category style
// ---------------------------------------------------------------------------

// #106. Called once per frame at the end of frame(), so a change made this frame
// becomes the next undo target.
void App::record_view_history() {
  Tab* tab = tabs_[active_tab_].get();
  ViewState& vs = tab->view;

  // The frame that APPLIES an undo must not record its result: the ring would
  // then hold the restored state as a fresh entry and redo would be unreachable.
  // (ViewHistory's identical-drop already tolerates this, but skipping keeps the
  // camera-coalescing run state clean.)
  if (vs.suppress_history) {
    vs.suppress_history = false;
    return;
  }
  if (tab->session->model() == nullptr) return;

  if (!vs.history) vs.history = std::make_unique<ViewHistory>();

  // A history entry is only applicable within one (generation, graph): IR node
  // indices and CollapseTree group indices are both assigned per build. Rather
  // than let apply_view() refuse entry after entry, drop the ring outright when
  // the model or the graph moves — offering the user undo steps that silently do
  // nothing is worse than offering none.
  const uint64_t gen = tab->session->generation();
  const uint32_t graph = tab->session->current_graph();
  if (vs.history_owner_generation != gen || vs.history_owner_graph != graph) {
    vs.history->clear();
    vs.history_owner_generation = gen;
    vs.history_owner_graph = graph;
  }

  vs.history->push(capture_view(vs, *tab->session));
}

namespace {

// Shared by undo_view/redo_view: apply a stepped-to snapshot, or step back if it
// turns out not to be applicable. `step` returns the entry to try.
template <typename StepFn>
void step_history(App& app, ViewState& vs, ModelSession& session, StepFn step) {
  if (!vs.history) return;
  const ViewSnapshot* s = step();
  if (s == nullptr) return;
  // apply_view refuses a snapshot whose (generation, graph) or collapse-bitset
  // size does not match the live session. That should not happen — the ring is
  // cleared on both changes — but a refusal must not leave the cursor moved onto
  // an entry that was never applied, so report it rather than failing silently.
  if (!apply_view(*s, vs, session)) {
    app.add_toast("Cannot undo across a model or graph change", true);
    vs.history->clear();
    return;
  }
  vs.suppress_history = true;
}

}  // namespace

void App::undo_view() {
  Tab* tab = tabs_[active_tab_].get();
  ViewState& vs = tab->view;
  step_history(*this, vs, *tab->session,
               [&]() { return vs.history ? vs.history->undo() : nullptr; });
}

void App::redo_view() {
  Tab* tab = tabs_[active_tab_].get();
  ViewState& vs = tab->view;
  step_history(*this, vs, *tab->session,
               [&]() { return vs.history ? vs.history->redo() : nullptr; });
}

// #103. Writes nothing when the pref is off, so disabling restore also stops the
// file being maintained (clearing it is clear_session's job, called on the
// toggle itself).
void App::save_session_now() {
  if (!view().restore_session) return;
  SessionState s;
  s.active_tab = active_tab_;
  for (const std::unique_ptr<Tab>& t : tabs_) {
    if (!t || !t->session) continue;
    SessionTab st;
    st.path = t->session->path();
    st.pan_x = t->view.cam.pan.x;
    st.pan_y = t->view.cam.pan.y;
    st.zoom = t->view.cam.zoom;
    s.tabs.push_back(std::move(st));
  }
  save_session(s);
}

size_t App::restore_last_session() {
  const SessionState s = load_session();
  if (s.tabs.empty()) return 0;

  size_t skipped = 0;
  size_t opened = 0;
  for (const SessionTab& st : s.tabs) {
    // A remembered file may have been moved or deleted since. Skipping is the
    // only sane behaviour: a launch must not fail because one of five remembered
    // models is gone. The count is returned so the caller can say so.
    std::error_code ec;
    if (!std::filesystem::exists(st.path, ec) || ec) {
      ++skipped;
      continue;
    }
    open_file(st.path);
    // open_file() may have created a tab; the camera belongs to whichever tab is
    // now active. The model is still loading, but the camera is view state and
    // is applied immediately — the canvas simply uses it once boxes exist.
    view().cam.pan = ImVec2(st.pan_x, st.pan_y);
    view().cam.zoom = st.zoom;
    ++opened;
  }
  if (opened > 0 && s.active_tab < tabs_.size()) switch_tab(s.active_tab);
  return skipped;
}

// #104. The single place the live palette is resolved, so the canvas, legend and
// minimap cannot disagree about what a category looks like.
CategoryStyle App::category_style_for(OpCategory c) const {
  return category_style(c, tabs_[active_tab_]->view.dark_theme,
                        tabs_[active_tab_]->view.accessible_palette);
}

// ---------------------------------------------------------------------------
// Op-category palette (spec §8.1)
// ---------------------------------------------------------------------------
ImU32 App::category_color(OpCategory c, bool dark) {
  // v0.9.4: the palette moved to view/CategoryStyle.cpp, which also owns the
  // accessible variant (#104). This delegates rather than keeping a second copy
  // of the fifteen-entry table — two tables would silently drift, and only one
  // of them has tests pinning its values.
  //
  // Kept as a function because existing call sites use it and it is the
  // colour-only question. Anything that should honour the user's accessibility
  // setting must call App::category_style_for instead, which threads
  // ViewState::accessible_palette through.
  return category_style(c, dark, /*accessible=*/false).color;
}

// ---------------------------------------------------------------------------
// PNG export (spec §8.7): read back the rendered window and write a PNG.
// ---------------------------------------------------------------------------
void App::export_view_dialog() {
  const char* filters[] = {"*.png"};
  char* out =
      tinyfd_saveFileDialog("Export view", "netvis.png", 1, filters, "PNG image");
  if (out != nullptr) export_view_png(out);
}

void App::export_view_svg_dialog() {
  const char* filters[] = {"*.svg"};
  char* out = tinyfd_saveFileDialog("Export SVG", "netvis.svg", 1, filters,
                                    "SVG vector");
  if (out != nullptr) export_view_svg(out);
}

// --- #56 shareable view-state file (.netvis-view JSON) ----------------------
void App::save_view_state_dialog() {
  const char* filters[] = {"*.netvis-view"};
  char* out = tinyfd_saveFileDialog("Save view state", "view.netvis-view", 1,
                                    filters, "NetVis view");
  if (out != nullptr) save_view_state(out);
}

void App::load_view_state_dialog() {
  const char* filters[] = {"*.netvis-view"};
  char* picked = tinyfd_openFileDialog("Load view state", "", 1, filters,
                                       "NetVis view", 0);
  if (picked != nullptr) load_view_state(picked);
}

void App::save_view_state(const std::string& path) {
  ViewState& vs = view();
  ModelSession& s = session();
  nlohmann::json j;
  j["kind"] = "netvis-view";
  j["version"] = 1;
  j["model"] = s.path();          // informational - which model this view was for
  j["graph"] = s.current_graph();
  j["cam"] = {{"pan_x", vs.cam.pan.x}, {"pan_y", vs.cam.pan.y},
              {"zoom", vs.cam.zoom}};
  j["hide_const_edges"] = vs.hide_const_edges;
  j["show_layer_bands"] = vs.show_layer_bands;
  if (vs.nav) {
    j["category_mask"] = vs.nav->category_mask;
    j["path_a"] = vs.nav->path_a;   // stable IR node indices (see GraphNav.h #15)
    j["path_b"] = vs.nav->path_b;
  }
  // Selection stored as a STABLE IR node index (display ids shift on collapse).
  int32_t sel_ir = -1;
  const auto& disp = s.collapse().display_nodes();
  if (vs.selected_display >= 0 &&
      static_cast<size_t>(vs.selected_display) < disp.size() &&
      !disp[static_cast<size_t>(vs.selected_display)].is_group)
    sel_ir = static_cast<int32_t>(disp[static_cast<size_t>(vs.selected_display)].ir_node);
  j["selected_ir_node"] = sel_ir;

  std::ofstream f(path);
  if (f) {
    // `replace` rather than the default strict UTF-8 handler: this document
    // carries the model PATH, and a path is not guaranteed valid UTF-8 on Linux.
    // Strict would throw out of a save the user explicitly asked for. See
    // save_recent for the full reasoning.
    f << j.dump(2, ' ', false, nlohmann::json::error_handler_t::replace);
    add_toast("View saved", false);
  } else {
    add_toast("Could not write view file", true);
  }
}

void App::load_view_state(const std::string& path) {
  std::ifstream f(path);
  if (!f) { add_toast("Could not open view file", true); return; }
  ViewState& vs = view();
  ModelSession& s = session();
  try {
    nlohmann::json j;
    f >> j;
    if (!j.is_object() || j.value("kind", "") != "netvis-view") {
      add_toast("Not a NetVis view file", true);
      return;
    }

    // Model-agnostic view state (camera, display toggles, category filter) always
    // applies. Model-SPECIFIC state (graph dive, selection, path endpoints — all
    // keyed by IR node index) only applies when the file was saved for the model
    // currently loaded; otherwise those indices denote unrelated nodes in a
    // different model (bounds-checked, so no UB, but a confidently-wrong view).
    if (j.contains("cam") && j["cam"].is_object()) {
      const auto& c = j["cam"];
      vs.cam.pan.x = c.value("pan_x", vs.cam.pan.x);
      vs.cam.pan.y = c.value("pan_y", vs.cam.pan.y);
      vs.cam.zoom = c.value("zoom", vs.cam.zoom);
      vs.animating = false;
    }
    if (j.contains("hide_const_edges") && j["hide_const_edges"].is_boolean())
      vs.hide_const_edges = j["hide_const_edges"].get<bool>();
    if (j.contains("show_layer_bands") && j["show_layer_bands"].is_boolean())
      vs.show_layer_bands = j["show_layer_bands"].get<bool>();
    if (!vs.nav) vs.nav = std::make_unique<GraphNavState>();
    if (j.contains("category_mask") && j["category_mask"].is_number_unsigned())
      vs.nav->category_mask = j["category_mask"].get<uint32_t>();

    const ir::Model* m = s.model();
    const std::string saved_model = j.value("model", std::string());
    const bool same_model = m != nullptr && saved_model == s.path();
    if (!same_model) {
      add_toast(m == nullptr ? "View loaded (open a model to restore selection)"
                             : "View loaded (camera only - saved for another model)",
                false);
      return;
    }

    // Same model: restore the graph dive, path endpoints, and selection.
    uint32_t g = s.current_graph();
    if (j.contains("graph") && j["graph"].is_number_unsigned()) {
      uint32_t gg = j["graph"].get<uint32_t>();
      if (gg < m->graphs.size() && gg != s.current_graph()) {
        s.push_graph(gg);
        g = gg;
      }
    }
    if (j.contains("path_a") && j["path_a"].is_number_integer())
      vs.nav->path_a = j["path_a"].get<int32_t>();
    if (j.contains("path_b") && j["path_b"].is_number_integer())
      vs.nav->path_b = j["path_b"].get<int32_t>();
    // Bind nav ownership to the graph we just applied IR-index state for, so
    // ensure_nav's cross-graph guard does not wipe path_a/path_b this frame (it
    // clears IR-index nav collections when owner_graph != current graph — the
    // just-loaded endpoints belong to `g`, so claim ownership).
    vs.nav->owner_generation = s.generation();
    vs.nav->owner_graph = g;
    if (j.contains("selected_ir_node") && j["selected_ir_node"].is_number_integer()) {
      int32_t ir_node = j["selected_ir_node"].get<int32_t>();
      if (ir_node >= 0) {
        int32_t d = panel_detail::display_index_for_node(
            s.collapse(), static_cast<uint32_t>(ir_node));
        if (d >= 0) vs.selected_display = d;
      }
    }
    add_toast("View loaded", false);
  } catch (...) {
    add_toast("Corrupt view file", true);
  }
}

void App::export_view_png(const std::string& path) {
  // DECISION (portability): read back the DEFAULT framebuffer (the just-rendered
  // window) with glReadPixels instead of rendering into an offscreen 2x FBO. The
  // FBO path needs glGenFramebuffers/... which live in GL extensions — available
  // on Linux/macOS core GL but NOT in the Windows SDK's GL 1.1 <GL/gl.h> without
  // a loader library. glReadPixels/glViewport are GL 1.1 core everywhere, so this
  // builds on all three platforms with no loader. Trade-off: capture is at window
  // resolution (no supersample), which is fine for a "screenshot the view" feature.
  //
  // Called from App::frame() immediately AFTER ImGui_ImplOpenGL3_RenderDrawData,
  // so the back buffer already holds the current frame's pixels.
  int fb_w = 0, fb_h = 0;
  glfwGetFramebufferSize(window_, &fb_w, &fb_h);
  const int w = fb_w;
  const int h = fb_h;
  if (w <= 0 || h <= 0) {
    add_toast("Export failed: zero-size viewport", true);
    return;
  }

  std::vector<unsigned char> pixels(static_cast<size_t>(w) *
                                    static_cast<size_t>(h) * 4u);
  glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

  // GL reads bottom-up; flip vertically so the PNG is upright.
  const size_t row = static_cast<size_t>(w) * 4u;
  std::vector<unsigned char> flipped(pixels.size());
  for (int y = 0; y < h; ++y) {
    std::copy(pixels.begin() + static_cast<long>((h - 1 - y) * static_cast<int>(row)),
              pixels.begin() + static_cast<long>((h - y) * static_cast<int>(row)),
              flipped.begin() + static_cast<long>(y * static_cast<int>(row)));
  }

  if (stbi_write_png(path.c_str(), w, h, 4, flipped.data(),
                     static_cast<int>(row)) != 0) {
    add_toast("Exported view to " + path, false);
  } else {
    add_toast("Export failed writing " + path, true);
  }
}

// ---------------------------------------------------------------------------
// Recent files (recent.json next to the layout cache)
// ---------------------------------------------------------------------------
void App::load_recent() {
  recent_.clear();
  const std::string p = layout_cache_dir() + "/recent.json";
  std::ifstream f(p);
  if (!f) return;
  // nlohmann parse can throw; contain it here so a corrupt file is a no-op.
  try {
    nlohmann::json j;
    f >> j;
    if (j.is_array()) {
      for (const auto& e : j)
        if (e.is_string()) recent_.push_back(e.get<std::string>());
    }
  } catch (...) {
    recent_.clear();
  }
}

void App::save_recent() {
  // CRASH GUARD (v0.9.4). nlohmann's dump() defaults to the STRICT UTF-8 error
  // handler and throws type_error.316 on a string that is not valid UTF-8. The
  // strings here are filesystem PATHS, which on Linux are byte sequences with no
  // encoding guarantee at all — so opening a file whose path contains invalid
  // UTF-8 threw out of save_recent, through add_recent, through open_file, and
  // terminated the app. A file viewer must not die because of the name of a file
  // it was asked to view.
  //
  // `replace` substitutes U+FFFD instead of throwing: the recent list is a
  // convenience, and a slightly mangled entry is enormously better than a crash.
  // The catch is still there for anything else dump() might raise.
  try {
    nlohmann::json j = nlohmann::json::array();
    for (const std::string& s : recent_) j.push_back(s);
    const std::string text =
        j.dump(2, ' ', false, nlohmann::json::error_handler_t::replace);
    std::ofstream f(layout_cache_dir() + "/recent.json");
    if (f) f << text;
  } catch (...) {
    // Best-effort, exactly like save_prefs: failing to persist a convenience
    // must never interrupt the user.
  }
}

void App::add_recent(const std::string& path) {
  auto it = std::find(recent_.begin(), recent_.end(), path);
  if (it != recent_.end()) recent_.erase(it);
  recent_.insert(recent_.begin(), path);
  const size_t kMaxRecent = 10;
  if (recent_.size() > kMaxRecent) recent_.resize(kMaxRecent);
  save_recent();
}

// ---------------------------------------------------------------------------
// View preferences (view_prefs.json next to the layout cache) — v0.3.2 QoL.
// Persists the heatmap gradient/scale, theme, and a couple of toggles so they
// survive across sessions. Best-effort: a missing/corrupt file just keeps the
// in-memory defaults, exactly like recent.json.
// ---------------------------------------------------------------------------
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
}  // namespace

void App::save_prefs() {
  const HeatmapGradient& g = view().heatmap_gradient;
  nlohmann::json j;
  j["dark_theme"] = view().dark_theme;
  j["show_minimap"] = view().show_minimap;
  j["show_layer_bands"] = view().show_layer_bands;  // #20 (v0.8.1)
  j["edge_tooltips"] = view().edge_tooltips;        // #18 (v0.8.1)
  j["cost_heatmap"] = view().cost_heatmap;
  j["heatmap_log_scale"] = view().heatmap_log_scale;
  j["heatmap_metric"] = heatmap_metric_name(view().heatmap_metric);
  j["gradient_preset"] = gradient_preset_name(g.preset);
  j["gradient_reverse"] = g.reverse;
  j["gradient_low"] = rgba_to_json(g.low);
  j["gradient_mid"] = rgba_to_json(g.mid);
  j["gradient_high"] = rgba_to_json(g.high);
  // #11: per-plugin enable overrides (empty object if the user changed nothing).
  j["plugins"] = plugin_enabled_.to_json();
  // #4/#30 (v0.8.3): custom roofline ridge + named machine profiles.
  j["edge_routing"] = view().edge_routing;  // #22 (v0.9.0)
  // v0.9.4: the settings a user sets once and expects to survive a restart. The
  // two WINDOW toggles (show_preferences/show_shortcuts) are deliberately NOT
  // here — a settings window that reopens itself every launch is a nuisance, not
  // a restored preference.
  j["accessible_palette"] = view().accessible_palette;  // #104
  j["ui_scale"] = view().ui_scale;                      // #104
  j["restore_session"] = view().restore_session;        // #103
  if (view().custom_ridge > 0.0) j["custom_ridge"] = view().custom_ridge;
  if (!view().machine_profiles.empty()) {
    nlohmann::json profs = nlohmann::json::array();
    for (const auto& [name, ridge] : view().machine_profiles)
      profs.push_back({{"name", name}, {"ridge", ridge}});
    j["machine_profiles"] = profs;
  }
  std::ofstream f(layout_cache_dir() + "/view_prefs.json");
  // `replace` for the same reason as save_recent: machine-profile names are
  // free text the user can paste into, so strict UTF-8 could throw out of a
  // routine preference save.
  if (f) f << j.dump(2, ' ', false, nlohmann::json::error_handler_t::replace);
}

void App::load_prefs() {
  const std::string p = layout_cache_dir() + "/view_prefs.json";
  std::ifstream f(p);
  if (!f) return;
  try {
    nlohmann::json j;
    f >> j;
    if (!j.is_object()) return;
    if (j.contains("dark_theme") && j["dark_theme"].is_boolean())
      view().dark_theme = j["dark_theme"].get<bool>();
    if (j.contains("show_minimap") && j["show_minimap"].is_boolean())
      view().show_minimap = j["show_minimap"].get<bool>();
    if (j.contains("show_layer_bands") && j["show_layer_bands"].is_boolean())
      view().show_layer_bands = j["show_layer_bands"].get<bool>();  // #20
    if (j.contains("edge_tooltips") && j["edge_tooltips"].is_boolean())
      view().edge_tooltips = j["edge_tooltips"].get<bool>();        // #18
    if (j.contains("cost_heatmap") && j["cost_heatmap"].is_boolean())
      view().cost_heatmap = j["cost_heatmap"].get<bool>();
    if (j.contains("heatmap_log_scale") && j["heatmap_log_scale"].is_boolean())
      view().heatmap_log_scale = j["heatmap_log_scale"].get<bool>();
    if (j.contains("heatmap_metric") && j["heatmap_metric"].is_string())
      view().heatmap_metric =
          heatmap_metric_from_name(j["heatmap_metric"].get<std::string>().c_str());

    HeatmapGradient& g = view().heatmap_gradient;
    if (j.contains("gradient_preset") && j["gradient_preset"].is_string()) {
      GradientPreset preset = preset_from_name(j["gradient_preset"].get<std::string>());
      gradient_set_preset(g, preset);  // fills stops for a built-in preset
    }
    if (j.contains("gradient_reverse") && j["gradient_reverse"].is_boolean())
      g.reverse = j["gradient_reverse"].get<bool>();
    // Only a Custom gradient carries its own stops; for a built-in preset the
    // preset's stops (just filled by gradient_set_preset) are authoritative, so a
    // "Viridis" tag always shows Viridis colors and a future change to the preset
    // constants isn't pinned to a stale persisted copy.
    if (g.preset == GradientPreset::Custom) {
      if (j.contains("gradient_low"))
        g.low = rgba_from_json(j["gradient_low"], g.low);
      if (j.contains("gradient_mid"))
        g.mid = rgba_from_json(j["gradient_mid"], g.mid);
      if (j.contains("gradient_high"))
        g.high = rgba_from_json(j["gradient_high"], g.high);
    }
    // #11: per-plugin enable overrides (guarded; never prunes, ignores non-bool).
    if (j.contains("plugins") && j["plugins"].is_object())
      plugin_enabled_.load_json(j["plugins"]);
    // #4/#30: custom ridge + named machine profiles.
    if (j.contains("edge_routing") && j["edge_routing"].is_number_integer()) {
      int er = j["edge_routing"].get<int>();
      if (er >= 0 && er <= 2) view().edge_routing = er;
    }
    if (j.contains("accessible_palette") && j["accessible_palette"].is_boolean())
      view().accessible_palette = j["accessible_palette"].get<bool>();
    if (j.contains("restore_session") && j["restore_session"].is_boolean())
      view().restore_session = j["restore_session"].get<bool>();
    // CLAMPED on load, not merely on edit. A persisted 0, a negative, or a NaN
    // would render an unusable window — and the setting that caused it lives
    // inside that window, so the user could not reach it to undo the damage.
    // Written as !(in range) so NaN is rejected too, where a naive comparison
    // would let it through.
    if (j.contains("ui_scale") && j["ui_scale"].is_number()) {
      const float sc = j["ui_scale"].get<float>();
      view().ui_scale = !(sc >= 0.75f && sc <= 2.0f) ? 1.0f : sc;
    }
    if (j.contains("custom_ridge") && j["custom_ridge"].is_number())
      view().custom_ridge = j["custom_ridge"].get<double>();
    if (j.contains("machine_profiles") && j["machine_profiles"].is_array()) {
      view().machine_profiles.clear();
      for (const auto& e : j["machine_profiles"]) {
        if (e.is_object() && e.contains("name") && e["name"].is_string() &&
            e.contains("ridge") && e["ridge"].is_number())
          view().machine_profiles.emplace_back(e["name"].get<std::string>(),
                                               e["ridge"].get<double>());
      }
    }
  } catch (...) {
    // Corrupt prefs -> keep defaults.
  }
}

void App::reload_plugins() {
  // §0.4/§C.3: reset to built-ins then re-discover under the current gate, so a
  // disabled plugin is structurally absent from the Registry.
  plugin::Registry::instance().reset_to_builtins();
  plugin::discover_and_load_plugins(
      [this](std::string_view id, plugin::PluginKind k) { return plugin_gate(id, k); });
  // The Registry table changed, so the cost report (its FLOP/category handlers route
  // through the registry) is stale. Bump the derived-state epoch so the next
  // ensure_cost rebuilds against the new table. No file reparse.
  // The plugin Registry is a global singleton, so a reset+re-discover changes the
  // FLOP/category routing for EVERY tab's cost report — not just the active one.
  // Invalidate all tabs' derived state (bumps each session's enrich epoch) so each
  // rebuilds against the new table when next drawn. (#62: invalidating only the
  // active tab would leave background tabs serving a cost report built against the
  // old plugin table until some unrelated key change.)
  for (auto& t : tabs_) t->session->invalidate_derived();
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
void App::apply_theme(bool dark) {
  ImGuiStyle& s = ImGui::GetStyle();
  // Rounded, slightly padded frames throughout (spec §8.7).
  s.FrameRounding = 5.0f;
  s.WindowRounding = 6.0f;
  s.PopupRounding = 5.0f;
  s.ChildRounding = 5.0f;
  s.GrabRounding = 4.0f;
  s.TabRounding = 5.0f;
  s.ScrollbarRounding = 5.0f;
  s.FramePadding = ImVec2(8, 5);
  s.ItemSpacing = ImVec2(8, 6);
  s.WindowBorderSize = 1.0f;

  ImVec4* c = s.Colors;
  auto set = [&](ImGuiCol id, float r, float g, float b, float a) {
    c[id] = ImVec4(r, g, b, a);
  };
  if (dark) {
    set(ImGuiCol_Text, 0.90f, 0.91f, 0.93f, 1.00f);
    set(ImGuiCol_TextDisabled, 0.45f, 0.47f, 0.52f, 1.00f);
    set(ImGuiCol_WindowBg, 0.11f, 0.12f, 0.14f, 1.00f);
    set(ImGuiCol_ChildBg, 0.10f, 0.11f, 0.13f, 1.00f);
    set(ImGuiCol_PopupBg, 0.13f, 0.14f, 0.17f, 0.98f);
    set(ImGuiCol_Border, 0.24f, 0.26f, 0.30f, 0.60f);
    set(ImGuiCol_FrameBg, 0.17f, 0.19f, 0.22f, 1.00f);
    set(ImGuiCol_FrameBgHovered, 0.22f, 0.25f, 0.29f, 1.00f);
    set(ImGuiCol_FrameBgActive, 0.26f, 0.30f, 0.35f, 1.00f);
    set(ImGuiCol_TitleBg, 0.10f, 0.11f, 0.13f, 1.00f);
    set(ImGuiCol_TitleBgActive, 0.15f, 0.17f, 0.21f, 1.00f);
    set(ImGuiCol_MenuBarBg, 0.13f, 0.14f, 0.17f, 1.00f);
    set(ImGuiCol_Header, 0.20f, 0.32f, 0.48f, 0.80f);
    set(ImGuiCol_HeaderHovered, 0.26f, 0.40f, 0.58f, 0.90f);
    set(ImGuiCol_HeaderActive, 0.30f, 0.46f, 0.66f, 1.00f);
    set(ImGuiCol_Button, 0.20f, 0.23f, 0.28f, 1.00f);
    set(ImGuiCol_ButtonHovered, 0.28f, 0.42f, 0.62f, 1.00f);
    set(ImGuiCol_ButtonActive, 0.32f, 0.48f, 0.70f, 1.00f);
    set(ImGuiCol_CheckMark, 0.40f, 0.70f, 0.98f, 1.00f);
    set(ImGuiCol_SliderGrab, 0.36f, 0.56f, 0.82f, 1.00f);
    set(ImGuiCol_Tab, 0.14f, 0.16f, 0.19f, 1.00f);
    set(ImGuiCol_TabSelected, 0.22f, 0.34f, 0.50f, 1.00f);
    set(ImGuiCol_TabHovered, 0.28f, 0.42f, 0.62f, 1.00f);
    set(ImGuiCol_Separator, 0.24f, 0.26f, 0.30f, 0.70f);
    set(ImGuiCol_ScrollbarBg, 0.10f, 0.11f, 0.13f, 1.00f);
    set(ImGuiCol_ScrollbarGrab, 0.28f, 0.30f, 0.35f, 1.00f);
  } else {
    set(ImGuiCol_Text, 0.11f, 0.12f, 0.15f, 1.00f);
    set(ImGuiCol_TextDisabled, 0.50f, 0.52f, 0.56f, 1.00f);
    set(ImGuiCol_WindowBg, 0.96f, 0.96f, 0.97f, 1.00f);
    set(ImGuiCol_ChildBg, 0.98f, 0.98f, 0.99f, 1.00f);
    set(ImGuiCol_PopupBg, 0.99f, 0.99f, 1.00f, 0.98f);
    set(ImGuiCol_Border, 0.72f, 0.74f, 0.78f, 0.70f);
    set(ImGuiCol_FrameBg, 0.90f, 0.91f, 0.93f, 1.00f);
    set(ImGuiCol_FrameBgHovered, 0.85f, 0.88f, 0.93f, 1.00f);
    set(ImGuiCol_FrameBgActive, 0.80f, 0.85f, 0.92f, 1.00f);
    set(ImGuiCol_TitleBg, 0.88f, 0.89f, 0.91f, 1.00f);
    set(ImGuiCol_TitleBgActive, 0.78f, 0.83f, 0.90f, 1.00f);
    set(ImGuiCol_MenuBarBg, 0.90f, 0.91f, 0.93f, 1.00f);
    set(ImGuiCol_Header, 0.62f, 0.74f, 0.90f, 0.80f);
    set(ImGuiCol_HeaderHovered, 0.54f, 0.70f, 0.90f, 0.90f);
    set(ImGuiCol_HeaderActive, 0.46f, 0.64f, 0.88f, 1.00f);
    set(ImGuiCol_Button, 0.86f, 0.88f, 0.91f, 1.00f);
    set(ImGuiCol_ButtonHovered, 0.66f, 0.78f, 0.94f, 1.00f);
    set(ImGuiCol_ButtonActive, 0.56f, 0.72f, 0.92f, 1.00f);
    set(ImGuiCol_CheckMark, 0.18f, 0.44f, 0.78f, 1.00f);
    set(ImGuiCol_SliderGrab, 0.44f, 0.62f, 0.86f, 1.00f);
    set(ImGuiCol_Tab, 0.86f, 0.88f, 0.91f, 1.00f);
    set(ImGuiCol_TabSelected, 0.70f, 0.80f, 0.93f, 1.00f);
    set(ImGuiCol_TabHovered, 0.60f, 0.74f, 0.92f, 1.00f);
    set(ImGuiCol_Separator, 0.72f, 0.74f, 0.78f, 0.70f);
    set(ImGuiCol_ScrollbarBg, 0.92f, 0.93f, 0.95f, 1.00f);
    set(ImGuiCol_ScrollbarGrab, 0.72f, 0.74f, 0.78f, 1.00f);
  }
}

}  // namespace netvis
