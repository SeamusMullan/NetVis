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

// Raw GL entry points for the offscreen-FBO PNG export path. Header location is
// platform-specific: Apple ships OpenGL under <OpenGL/>, and its gl3.h already
// declares the FBO core API; elsewhere GL_GLEXT_PROTOTYPES makes <GL/glext.h>
// declare glGenFramebuffers/... so we need no loader library.
#if defined(__APPLE__)
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#else
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#endif

// stb_image_write is header-only; its implementation is compiled EXACTLY once,
// here (spec §8.7). Wrap it so its internal style does not trip -Werror.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wtype-limits"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma GCC diagnostic pop

#include <nlohmann/json.hpp>

#include "engine/LayoutCache.h"

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

App::App() = default;

App::~App() {
  // DECISION (threading): stop all workers BEFORE any member is destroyed.
  // Member destruction order is session_ then jobs_ (reverse declaration), so a
  // still-running decode job could touch session_->file() after session_ is
  // gone. shutdown() joins every worker first, closing that window.
  if (jobs_) jobs_->shutdown();
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

  apply_theme(view_.dark_theme);

  // Route OS file drops to open_file() via the window user pointer.
  glfwSetWindowUserPointer(window_, this);
  glfwSetDropCallback(window_, drop_callback);

  // jobs_ before session_ (session_ takes a JobSystem&; it must outlive nothing
  // it references and must be destroyed before jobs_ — see ~App).
  jobs_ = std::make_unique<JobSystem>();
  session_ = std::make_unique<ModelSession>(*jobs_);

  // Font-metric-based node sizing (spec §8.1): the layout worker calls this to
  // measure each display node's box. It reads only glyph advance widths from the
  // pre-baked atlas (immutable after init), plus the model/collapse state which
  // is published on the main thread before a layout job is queued.
  session_->set_size_fn([this](const DisplayNode& dn) -> Vec2 {
    const float kPadX = 24.0f;   // horizontal breathing room around the label
    const float kLineH = 18.0f;  // one text line incl. leading
    const float kPadY = 14.0f;   // vertical padding (header strip + margins)

    // Resolve the two label lines for this display node.
    std::string primary, secondary;
    const ir::Model* m = session_ ? session_->model() : nullptr;
    if (dn.is_group) {
      const auto& groups = session_->collapse().groups();
      if (dn.group_index < groups.size()) {
        const auto& g = groups[dn.group_index];
        primary = g.label;
        secondary = "x" + std::to_string(g.instances);
      }
    } else if (m != nullptr) {
      uint32_t gi = session_->current_graph();
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

    // Measure with real font metrics when available, else a char-count estimate.
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
    session_->update();  // drain job completions once per frame (spec §4).
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
      if (ImGui::MenuItem("Open...", "Ctrl+O")) {
        const char* filters[] = {"*.onnx", "*.tflite", "*.safetensors",
                                  "*.gguf", "*.pt",     "*.pth",
                                  "*.bin"};
        char* picked = tinyfd_openFileDialog(
            "Open model", "", 7, filters, "Model files", 0);
        if (picked != nullptr) open_file(picked);
      }
      if (ImGui::BeginMenu("Recent", !recent_.empty())) {
        for (const std::string& r : recent_) {
          if (ImGui::MenuItem(r.c_str())) open_file(r);
        }
        ImGui::EndMenu();
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Export View PNG...")) {
        const char* filters[] = {"*.png"};
        char* out = tinyfd_saveFileDialog("Export view", "netvis.png", 1,
                                          filters, "PNG image");
        if (out != nullptr) export_view_png(out);
      }
      ImGui::Separator();
      if (ImGui::MenuItem("Exit")) glfwSetWindowShouldClose(window_, GLFW_TRUE);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View")) {
      if (ImGui::MenuItem("Dark theme", nullptr, view_.dark_theme)) {
        view_.dark_theme = !view_.dark_theme;
        apply_theme(view_.dark_theme);
      }
      if (ImGui::MenuItem("Light theme", nullptr, !view_.dark_theme)) {
        view_.dark_theme = false;
        apply_theme(false);
      }
      ImGui::Separator();
      ImGui::MenuItem("Minimap", nullptr, &view_.show_minimap);
      ImGui::EndMenu();
    }
    ImGui::EndMainMenuBar();
  }

  // --- Content --------------------------------------------------------------
  // With a compute graph, the canvas owns the center; otherwise (GGUF / weight-
  // only files) we present the flat tensor table instead (spec §8.6).
  if (session_->has_graph()) {
    draw_graph_canvas(*this);
  } else if (session_->model() != nullptr && !session_->has_graph()) {
    draw_tensor_table(*this);
  }

  // Panels + overlays are always present (they self-hide when empty).
  draw_properties_panel(*this);
  draw_weight_inspector(*this);
  draw_search_bar(*this);
  draw_status_bar(*this);
  draw_toasts(*this);

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
    view_.search_open = !view_.search_open;
  }
  if (!typing && !io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
    view_.request_fit = true;  // 'F' fits the whole graph next frame.
  }
  if (!typing && ImGui::IsKeyPressed(ImGuiKey_Home, false)) {
    view_.cam = Camera{};      // Home resets pan/zoom to identity.
    view_.animating = false;
  }
  if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    view_.search_open = false;
  }
}

// ---------------------------------------------------------------------------
// File open / recent / inspect
// ---------------------------------------------------------------------------
void App::open_file(const std::string& path) {
  if (path.empty()) return;
  session_->open_async(path);  // non-blocking (spec §4): pipeline kicks off.
  add_recent(path);
}

void App::add_toast(const std::string& text, bool is_error) {
  Toast t;
  t.text = text;
  t.ttl = 5.0f;
  t.is_error = is_error;
  toasts_.push_back(std::move(t));
}

void App::inspect_tensor(const ir::TensorRef& t) {
  // Mark the inspector busy for this tensor and bump the token so any older
  // in-flight decode's completion is ignored when it eventually lands.
  decode_.tensor = t;
  decode_.active = true;
  decode_.in_flight = true;
  decode_.done = false;
  decode_.ok = false;
  decode_.error.clear();
  const uint64_t token = ++decode_.token;

  // Copy the tensor by value into the job (a captured reference would dangle).
  // TensorStats is the ONLY payload-reading path (spec §7.5); it runs on a
  // worker so the UI never blocks decoding a multi-GB tensor. The token guards
  // against a stale result overwriting a newer inspection; ~App joins workers
  // before session_ dies so session_->file() stays valid for the job's life.
  ir::TensorRef tc = t;
  jobs_->submit([this, token, tc]() {
    Result<TensorStats> r =
        compute_tensor_stats(tc, session_->file(), session_->model_dir());
    bool ok = r.ok();
    TensorStats stats = ok ? *r : TensorStats{};
    std::string errmsg = ok ? std::string() : r.error().message;
    jobs_->post_to_main(
        [this, token, ok, stats, errmsg]() {
          if (decode_.token != token) return;  // superseded — drop it.
          decode_.stats = stats;
          decode_.ok = ok;
          decode_.error = errmsg;
          decode_.done = true;
          decode_.in_flight = false;
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
      {/*IO*/ 96, 172, 214},      {/*Other*/ 128, 128, 136},
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
// PNG export (spec §8.7): render current draw data into a 2x offscreen target.
// ---------------------------------------------------------------------------
void App::export_view_png(const std::string& path) {
  ImDrawData* dd = ImGui::GetDrawData();
  if (dd == nullptr || !dd->Valid) {
    add_toast("Export failed: nothing to render yet", true);
    return;
  }

  const ImGuiIO& io = ImGui::GetIO();
  const int scale = 2;  // 2x supersample for a crisp export.
  int w = static_cast<int>(io.DisplaySize.x) * scale;
  int h = static_cast<int>(io.DisplaySize.y) * scale;
  if (w <= 0 || h <= 0) {
    add_toast("Export failed: zero-size viewport", true);
    return;
  }

  // Create an offscreen color texture + framebuffer at 2x.
  GLint prev_fbo = 0;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
  GLuint tex = 0, fbo = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         tex, 0);

  bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  std::vector<unsigned char> pixels;
  if (ok) {
    glViewport(0, 0, w, h);
    const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    glClearColor(bg.x, bg.y, bg.z, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Temporarily rescale the draw data so ImGui rasterizes at 2x into the FBO.
    const ImVec2 saved_scale = dd->FramebufferScale;
    dd->FramebufferScale = ImVec2(static_cast<float>(scale),
                                  static_cast<float>(scale));
    ImGui_ImplOpenGL3_RenderDrawData(dd);
    dd->FramebufferScale = saved_scale;

    pixels.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4u);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
  }

  // Restore the previous framebuffer and release the offscreen resources.
  glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prev_fbo));
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &tex);

  if (!ok) {
    add_toast("Export failed: framebuffer incomplete", true);
    return;
  }

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
