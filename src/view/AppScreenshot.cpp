// SPDX-License-Identifier: Apache-2.0
// view/AppScreenshot.cpp — the `netvis --screenshot` capture driver (#170).
//
// Kept out of App.cpp (already ~1700 lines). The flow:
//
//   run_screenshot
//     init_with(hidden, hermetic)        a hidden GL window; prefs/recent/ini/plugins ignored
//     capture_body
//       OffscreenTarget                  RGBA8 FBO, exactly --size pixels
//       open_async                       the NORMAL open path (not App::open_file: that
//                                        writes recent.json)
//       LOAD      pump until the gate passes     (no frame is rendered while waiting)
//       APPLY     the view file, in phases
//       QUIESCE   pump until the gate passes again
//       RENDER    kScreenshotSettleFrames frames at a fixed DeltaTime, re-fitting each
//                 frame unless a view file's camera is being honoured
//       CAPTURE   glReadPixels -> flip -> RGB -> PNG, written atomically
//     shutdown_window
//
// Every RULE in here (argument limits, the gate, the pixel flip) is core and tested
// (engine/ScreenshotCli.h). What is left is plumbing around ImGui, GLFW and GL, which
// netvis_tests cannot link, so it is covered by the manual verification in the PR.
//
// WHY NOTHING IS RENDERED WHILE WAITING: frame() is never called before the gate
// passes, so ImGui's clock, window first-appearance and docking all start from the
// same point on every run. With the fixed DeltaTime, ImGui::GetTime() at capture is
// exactly 4 x (1/60) s in the normal one-round case, which makes the time-driven
// search-hit pulse (GraphCanvas.cpp) deterministic too.

// LayoutEngine.h defines SizeFn, which ModelSession.h (via App.h) uses but does not
// include; it must come first.
#include "engine/LayoutEngine.h"
#include "view/App.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"  // FindWindowByName: the stray-window check after the last frame

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "engine/ScreenshotCli.h"
#include "engine/ViewFile.h"
#include "view/Screenshot.h"
#include "view/ViewFileApply.h"

namespace netvis {

namespace {

using Clock = std::chrono::steady_clock;

void say(const char* kind, const std::string& msg) {
  std::fprintf(stderr, "netvis --screenshot: %s: %s\n", kind, msg.c_str());
}

int fail(ScreenshotExit code, const std::string& msg) {
  say("error", msg);
  return exit_code(code);
}

const char* stage_text(LoadStage s) {
  switch (s) {
    case LoadStage::Empty: return "starting";
    case LoadStage::Mapping: return "mapping the file";
    case LoadStage::Parsing: return "parsing";
    case LoadStage::Laying: return "laying out the graph";
    case LoadStage::Enriching: return "inferring shapes";
    case LoadStage::Ready: return "finishing up";
    case LoadStage::Failed: return "loading";
  }
  return "loading";
}

}  // namespace

int App::run_screenshot(const ScreenshotOptions& opt, const ViewFile* view_file) {
  capture_.active = true;
  capture_.canvas_only = opt.canvas_only;
  capture_.width = opt.width;
  capture_.height = opt.height;

  InitConfig cfg;
  cfg.hidden = true;
  cfg.hermetic = true;
  if (!init_with(std::string(), cfg)) {
    return fail(ScreenshotExit::Graphics,
                "could not create a hidden OpenGL 3.3 window (no display? on Linux use "
                "xvfb-run)");
  }
  // OffscreenTarget is a local of capture_body, so it is destroyed while the GL
  // context still exists; only then is the window torn down.
  const int code = capture_body(opt, view_file);
  shutdown_window();
  return code;
}

int App::capture_body(const ScreenshotOptions& opt, const ViewFile* view_file) {
  namespace fs = std::filesystem;

  OffscreenTarget target;
  if (!target.init(opt.width, opt.height)) return fail(ScreenshotExit::Graphics, target.error());

  view().dark_theme = opt.dark_theme;
  apply_theme(opt.dark_theme);
  session().set_layout_cache_enabled(opt.use_layout_cache);
  tabs_[active_tab_]->title = screenshot_display_name(opt.model_path);

  const auto deadline = Clock::now() + std::chrono::seconds(opt.timeout_s);

  // A timeout cannot unwind: a worker may be mid-parse, a parse cannot be cancelled,
  // and ~JobSystem joins its workers, so normal teardown could block well past the
  // limit. _Exit skips teardown (the OS reclaims the window and context). No PNG has
  // been written at this point.
  auto timed_out = [this](const std::string& why) -> int {
    session().cancel_layout();
    say("error", why);
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(exit_code(ScreenshotExit::Timeout));
  };
  auto timeout_while_loading = [&]() -> int {
    return timed_out("timed out after " + std::to_string(opt.timeout_s) + " s while " +
                     stage_text(session().stage()) + "; raise --timeout");
  };

  auto gate_now = [&]() {
    ModelSession& s = session();
    CaptureGateInputs in{};
    in.load_failed = s.stage() == LoadStage::Failed;
    in.jobs_idle = jobs().idle();  // sampled AFTER update(), as the gate requires
    in.has_model = s.model() != nullptr;
    in.has_graph = s.has_graph();
    in.has_layout = s.layout() != nullptr;
    in.layout_empty = in.has_layout && s.layout()->boxes.empty();
    in.canvas_only = opt.canvas_only;
    return capture_gate(in);
  };

  // Poll events, drain completions, test the gate; no rendering while waiting, so no
  // CPU or GPU is spent on frames while workers run.
  auto pump_until_gate = [&]() -> std::optional<CaptureGate> {
    for (;;) {
      glfwPollEvents();
      session().update();
      const CaptureGate g = gate_now();
      if (g != CaptureGate::Wait) return g;
      if (Clock::now() >= deadline) return std::nullopt;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  };

  // Map a settled gate to "carry on" (-1) or the exit code to return. It runs after the
  // load and again after a view file is applied, so a note is said once.
  bool noted_empty = false;
  auto settle = [&](CaptureGate g) -> int {
    switch (g) {
      case CaptureGate::Ready: return -1;
      case CaptureGate::Wait: return -1;  // unreachable: pump_until_gate never returns it
      case CaptureGate::LoadFailed: {
        const std::string& m = session().error_message();
        return fail(ScreenshotExit::Load, m.empty() ? std::string("load ended without a layout") : m);
      }
      case CaptureGate::NoCanvas:
        return fail(ScreenshotExit::Load,
                    "'" + opt.model_path +
                        "' has no compute graph (weights-only); --canvas-only has nothing to draw");
      case CaptureGate::EmptyGraph:
        if (opt.canvas_only) return fail(ScreenshotExit::Load, "the graph has no nodes");
        if (!noted_empty) say("note", "the graph has no nodes; capturing the empty canvas");
        noted_empty = true;
        return -1;
    }
    return -1;
  };

  // --- LOAD ---------------------------------------------------------------------
  // The NORMAL open path on a normal tab, so the same parse, layout, layout-cache,
  // search-index and ONNX shape-inference jobs run as in the GUI. Not App::open_file:
  // that records the path in recent.json, and a capture must write no preference.
  session().open_async(opt.model_path);
  {
    const std::optional<CaptureGate> g = pump_until_gate();
    if (!g) return timeout_while_loading();
    if (const int rc = settle(*g); rc >= 0) return rc;
  }

  // --- APPLY the view file ------------------------------------------------------
  if (view_file != nullptr) {
    ViewFileApplier applier = make_view_file_applier(*view_file, {}, session());
    for (;;) {
      glfwPollEvents();
      session().update();
      const ViewStep st = step_view_file(applier, view(), session(), jobs().idle());
      if (st == ViewStep::Done) break;
      if (st == ViewStep::Aborted)
        return fail(ScreenshotExit::View, "the view file could not be applied: the model changed");
      if (Clock::now() >= deadline) return timeout_while_loading();
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    // apply_view refused (the model or graph moved under the load): nothing of the
    // model-specific half applied, and a picture of the wrong state is not evidence.
    if (applier.outcome == ViewApplyOutcome::Failed)
      return fail(ScreenshotExit::View, view_file_outcome_text(applier.outcome));
    for (const std::string& n : applier.notes) say("note", n);
  }

  // --- QUIESCE: applying a view may have queued a re-layout ---------------------
  {
    const std::optional<CaptureGate> g = pump_until_gate();
    if (!g) return timeout_while_loading();
    if (const int rc = settle(*g); rc >= 0) return rc;
  }

  // --- RENDER -------------------------------------------------------------------
  // Fit is the default. A view file's camera wins unless --fit is given. The camera
  // is canvas-relative, so a camera saved from a GUI window with docked panels and
  // replayed --canvas-only shows the same world point at the top-left but more area;
  // --fit is the answer when reusing GUI-saved views at another size.
  const bool fit = opt.fit || view_file == nullptr || !view_file->has_camera();

  bool idle_after_render = false;
  for (int round = 0; round < kScreenshotMaxSettleRounds; ++round) {
    for (int i = 0; i < kScreenshotSettleFrames; ++i) {
      glfwPollEvents();
      session().update();
      // Re-fit every frame, on THAT frame's canvas size: the first frames of a
      // docked layout are not yet at their final size.
      if (fit) view().request_fit = true;
      frame();
      ImGui::Render();
      const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
      target.begin_frame(bg.x, bg.y, bg.z, 1.0f);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
    if (jobs().idle()) {
      idle_after_render = true;
      break;
    }
    // A panel scheduled background work while rendering: wait for it, render again.
    const std::optional<CaptureGate> g = pump_until_gate();
    if (!g) return timeout_while_loading();
    if (const int rc = settle(*g); rc >= 0) return rc;
  }
  if (!idle_after_render) {
    return timed_out("the UI kept scheduling background work after " +
                     std::to_string(kScreenshotMaxSettleRounds * kScreenshotSettleFrames) +
                     " frames");
  }

  // --- VERIFY: the frame really was the requested size at 1x --------------------
  const ImDrawData* dd = ImGui::GetDrawData();
  if (dd == nullptr || dd->DisplaySize.x != static_cast<float>(opt.width) ||
      dd->DisplaySize.y != static_cast<float>(opt.height) || dd->FramebufferScale.x != 1.0f ||
      dd->FramebufferScale.y != 1.0f) {
    return fail(ScreenshotExit::Graphics, "the rendered frame is not the requested size");
  }

  // A widget drawn outside any window lands in ImGui's implicit "Debug##Default"
  // window (400x400 at (60,60) when there is no imgui.ini, as here), on top of the
  // canvas. ImGui hides that window when nothing was written into it, so an active
  // one in the last frame means some panel has a stray widget and the picture has a
  // stray window in it. Say so; the picture is still written.
  if (const ImGuiWindow* dbg = ImGui::FindWindowByName("Debug##Default");
      dbg != nullptr && dbg->Active) {
    say("note",
        "the capture contains ImGui's implicit 'Debug' window (a widget was drawn "
        "outside any window)");
  }

  // --- CAPTURE ------------------------------------------------------------------
  std::vector<uint8_t> rgb;
  if (!target.read_rgb(rgb)) return fail(ScreenshotExit::Graphics, target.error());
  std::string werr;
  if (!write_png_rgb(opt.out_path, rgb, opt.width, opt.height, werr))
    return fail(ScreenshotExit::Write, werr);
  return exit_code(ScreenshotExit::Ok);
}

}  // namespace netvis
