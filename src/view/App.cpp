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
  for (const char* path : kCandidates) {
    std::ifstream probe(path, std::ios::binary);
    if (!probe) continue;
    ImFont* f = io.Fonts->AddFontFromFileTTF(path, size);
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

  if (!initial_path.empty()) open_file(initial_path);
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
  draw_command_palette(*this);   // #59: Ctrl+P fuzzy action palette (drawn on top)

  handle_shortcuts();

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
  if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    view().search_open = false;
    command_palette_open_ = false;
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
  // #62: if the diff was loaded against THIS tab's session, clear it first — the
  // DiffLoader holds that session's address for identity comparison, and erasing
  // the tab frees the session, which would leave primary_session() dangling.
  if (diff_loader_ &&
      diff_loader_->primary_session() == tabs_[i]->session.get()) {
    diff_loader_->clear();
    tabs_[i]->view.diff_panel_open = false;
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

// ---------------------------------------------------------------------------
// Op-category palette (spec §8.1)
// ---------------------------------------------------------------------------
ImU32 App::category_color(OpCategory c, bool dark) {
  // Dark-first palette: distinct, moderately saturated hues so adjacent node
  // categories read apart at a glance. Indexed by OpCategory order.
  struct RGB { uint8_t r, g, b; };
  static const RGB kPalette[] = {
      {/*Conv*/ 79, 143, 247},    {/*MatMul*/ 138, 110, 246},
      {/*Activation*/ 76, 201, 176}, {/*Norm*/ 232, 168, 56},
      {/*Pool*/ 90, 179, 90},     {/*Elementwise*/ 224, 108, 118},
      {/*Shape*/ 120, 130, 148},  {/*Reduce*/ 210, 120, 200},
      {/*Tensor*/ 150, 160, 90},  {/*ControlFlow*/ 200, 90, 130},
      {/*IO*/ 96, 172, 214},
      // v0.4.0 categories — MUST stay index-aligned with the OpCategory enum,
      // inserted before Other (see OpCategory.h). Distinct hues from the above.
      {/*Attention*/ 216, 100, 208}, {/*Recurrent*/ 96, 190, 150},
      {/*Quantize*/ 214, 178, 72},
      {/*Other*/ 128, 128, 136},
  };
  int idx = static_cast<int>(c);
  if (idx < 0 || idx >= static_cast<int>(sizeof(kPalette) / sizeof(kPalette[0])))
    idx = static_cast<int>(OpCategory::Other);
  RGB p = kPalette[idx];
  if (!dark) {
    // On a light theme, darken the hue so text/edges keep contrast.
    p.r = static_cast<uint8_t>(p.r * 0.72f);
    p.g = static_cast<uint8_t>(p.g * 0.72f);
    p.b = static_cast<uint8_t>(p.b * 0.72f);
  }
  return IM_COL32(p.r, p.g, p.b, 255);
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
  j["model"] = s.path();          // informational — which model this view was for
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
    f << j.dump(2);
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
                             : "View loaded (camera only — saved for another model)",
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
  nlohmann::json j = nlohmann::json::array();
  for (const std::string& s : recent_) j.push_back(s);
  std::ofstream f(layout_cache_dir() + "/recent.json");
  if (f) f << j.dump(2);
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
  if (view().custom_ridge > 0.0) j["custom_ridge"] = view().custom_ridge;
  if (!view().machine_profiles.empty()) {
    nlohmann::json profs = nlohmann::json::array();
    for (const auto& [name, ridge] : view().machine_profiles)
      profs.push_back({{"name", name}, {"ridge", ridge}});
    j["machine_profiles"] = profs;
  }
  std::ofstream f(layout_cache_dir() + "/view_prefs.json");
  if (f) f << j.dump(2);
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
