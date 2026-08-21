// view/StatusBar.cpp — bottom status bar + toast stack (spec §8.7).
//
// draw_status_bar: a pinned, no-decoration strip along the bottom of the main
// viewport showing the pipeline stage, per-stage timings once available
// ("mmap Xms . parse Yms . ..."), a progress bar while loading, and the model
// path + node/tensor counts. draw_toasts: transient messages stacked
// bottom-right just above the bar, error toasts tinted red.
//
// NOTE: BeginViewportSideBar lives only in imgui_internal.h (not a stable public
// API), and this build compiles under -Wpedantic without that header, so we use
// a pinned no-decoration Begin positioned at the viewport bottom instead.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h that
// App.h pulls in without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/ModelSession.h"
#include "ir/IR.h"
#include "parsers/Parser.h"  // #45: detect_format(...,reason) + detect_reason_name
#include "view/App.h"

namespace netvis {

namespace {

// Lowercased extension (no dot) of a mapped path, matching ModelSession's own
// detection tiebreaker input. Local copy so the status bar can recompute the
// detected format + reason WITHOUT touching the frozen ModelSession header.
std::string ext_of(const std::string& path) {
  auto slash = path.find_last_of("/\\");
  auto dot = path.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
    return {};
  }
  std::string e = path.substr(dot + 1);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}

// #45: the detected format + WHY, surfaced in the status bar so an
// extension-only / content-default tiebreak is visible (vs. a trustworthy magic
// hit). Recomputed here (not stored on the frozen ModelSession) but cached per
// (session, generation) so detect_format runs once per load, not every frame.
struct DetectTag {
  const ModelSession* session = nullptr;
  uint64_t generation = UINT64_MAX;
  std::string path;  // guards pointer-ABA: a freed session reallocated at the same
                     // address with the same per-tab generation still re-detects.
  bool valid = false;
  Format format = Format::Unknown;
  DetectReason reason = DetectReason::None;
};

const DetectTag& detected_tag(ModelSession& session) {
  static DetectTag cache;
  const uint64_t gen = session.generation();
  if (cache.session == &session && cache.generation == gen &&
      cache.path == session.path() && cache.valid) {
    return cache;
  }
  cache = DetectTag{};
  cache.session = &session;
  cache.generation = gen;
  cache.path = session.path();
  // Recompute over the exact inputs the parse pipeline used: the mapped inner
  // file and its extension (map_path_ == file().path()).
  const MappedFile& f = session.file();
  if (f.valid()) {
    DetectReason r = DetectReason::None;
    cache.format = detect_format(f, ext_of(f.path()), r);
    cache.reason = r;
    cache.valid = true;
  }
  return cache;
}

// Append "label Xms . " for a stage that actually ran (>0), spec §8.7 omits 0.
void append_timing(std::string& out, const char* label, double ms) {
  if (ms <= 0.0) return;
  if (!out.empty()) out += "  .  ";
  char buf[48];
  std::snprintf(buf, sizeof(buf), "%s %.0fms", label, ms);
  out += buf;
}

}  // namespace

// Height reserved for the status bar. App::frame() subtracts this from the main
// viewport's work area BEFORE building the dockspace, so the strip below is ours
// alone (see draw_status_bar for why it cannot simply overlay the dockspace).
float status_bar_height() { return ImGui::GetFrameHeight(); }

// Draw the bottom status bar (spec §8.7). Called once per frame.
void draw_status_bar(App& app) {
  ImGuiViewport* vp = ImGui::GetMainViewport();
  const float h = status_bar_height();

  // The work area has ALREADY been shrunk by h (App::frame), so the strip we own
  // starts exactly at its bottom edge. Overlaying the dockspace instead does not
  // work: DockSpaceOverViewport claims the whole work area, and the dock node's
  // windows paint over a bar that carries NoBringToFrontOnFocus.
  ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x, vp->WorkPos.y + vp->WorkSize.y),
                          ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(vp->WorkSize.x, h), ImGuiCond_Always);
  ImGui::SetNextWindowViewport(vp->ID);

  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
      ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav |
      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;

  // WindowMinSize defaults to 32x32 and is applied even to a fixed-size
  // NoDecoration window, which silently inflated this 23px bar to 32px and hung
  // 9px of it off the bottom of the screen. Relax it for this window only.
  ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 2.0f));
  if (!ImGui::Begin("##status_bar", nullptr, flags)) {
    ImGui::End();
    ImGui::PopStyleVar(2);
    return;
  }

  // Leftmost, before any model-dependent content, so it is the one thing in this
  // bar that is there in every state (no file, loading, failed, loaded) — the
  // right-hand side is a right-aligned string that a long path can already push
  // to the window edge, which is no place for a credit that must always be read.
  // Borderless so the bar still reads as a status strip rather than a toolbar.
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                        ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
  if (ImGui::SmallButton("Created By Seamus Mullan")) app.view().show_about = true;
  ImGui::PopStyleColor(2);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("About NetVis");
  ImGui::SameLine();
  ImGui::TextDisabled("|");
  ImGui::SameLine();

  ModelSession& session = app.session();
  const LoadStage stage = session.stage();

  // Left: stage label + timings (once available).
  const char* stage_name = "Empty";
  switch (stage) {
    case LoadStage::Empty: stage_name = "Ready"; break;
    case LoadStage::Mapping: stage_name = "Mapping"; break;
    case LoadStage::Parsing: stage_name = "Parsing"; break;
    case LoadStage::Laying: stage_name = "Layout"; break;
    case LoadStage::Enriching: stage_name = "Enriching"; break;
    case LoadStage::Ready: stage_name = "Ready"; break;
    case LoadStage::Failed: stage_name = "Failed"; break;
  }
  ImGui::TextUnformatted(stage_name);

  const bool loading = (stage == LoadStage::Mapping || stage == LoadStage::Parsing ||
                        stage == LoadStage::Laying || stage == LoadStage::Enriching);

  if (loading) {
    // Progress bar + which stage the worker is on right now.
    ImGui::SameLine();
    std::string ps = session.progress_stage();
    if (!ps.empty()) {
      ImGui::TextDisabled("%s", ps.c_str());
      ImGui::SameLine();
    }
    ImGui::SetNextItemWidth(180.0f);
    ImGui::ProgressBar(session.progress(), ImVec2(180.0f, h - 6.0f));
    // #61: layout is the long cancellable stage; offer a Cancel button only
    // during Laying (mapping/parsing/enriching are not user-cancellable).
    if (stage == LoadStage::Laying) {
      ImGui::SameLine();
      if (ImGui::SmallButton("Cancel")) session.cancel_layout();
    }
  } else {
    // Timings string (omit stages that never ran, spec §8.7).
    const StageTimings& t = session.timings();
    std::string timings;
    append_timing(timings, "mmap", t.mmap_ms);
    append_timing(timings, "parse", t.parse_ms);
    append_timing(timings, "layout", t.layout_ms);
    append_timing(timings, "shapes", t.shapes_ms);
    append_timing(timings, "search", t.search_ms);
    if (!timings.empty()) {
      ImGui::SameLine();
      ImGui::TextDisabled("%s", timings.c_str());
    }
  }

  // Right: format tag + model path + counts. Compute a right-aligned string.
  const ir::Model* model = session.model();
  std::string right;
  if (model) {
    // #45: format name + a short reason tag, e.g. "ONNX (structure)" or
    // "PyTorch (content default)". Only shown once a model is loaded.
    const DetectTag& tag = detected_tag(session);
    if (tag.valid && tag.format != Format::Unknown) {
      right += format_name(tag.format);
      right += " (";
      right += detect_reason_name(tag.reason);
      right += ")  |  ";
    }
    uint64_t nodes = 0;
    uint64_t tensors = model->flat_tensors.size();
    for (const ir::Graph& g : model->graphs) {
      nodes += g.nodes.size();
      tensors += g.initializers.size();
    }
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%llu nodes  .  %llu tensors  |  ",
                  static_cast<unsigned long long>(nodes),
                  static_cast<unsigned long long>(tensors));
    right += buf;
  }
  right += session.path().empty() ? "(no file)" : session.path();

  float rw = ImGui::CalcTextSize(right.c_str()).x;
  float avail = ImGui::GetWindowWidth();
  ImGui::SameLine();
  float x = avail - rw - 12.0f;
  if (x > ImGui::GetCursorPosX()) ImGui::SetCursorPosX(x);
  ImGui::TextDisabled("%s", right.c_str());

  ImGui::End();
  ImGui::PopStyleVar(2);
}

// Draw the toast stack (spec §8.7): translucent boxes bottom-right, above the
// status bar, error toasts tinted red. Only toasts with ttl>0 are drawn (App is
// responsible for decrementing ttl / removing dead ones each frame).
void draw_toasts(App& app) {
  std::vector<Toast>& toasts = app.toasts();
  if (toasts.empty()) return;

  ImGuiViewport* vp = ImGui::GetMainViewport();
  const float margin = 12.0f;

  // Stack upward from just above the status bar. The work area already excludes
  // the bar (App::frame reserves it), so its bottom edge IS the top of the bar —
  // subtracting status_bar_height() again would leave a bar-sized dead gap.
  float y = vp->WorkPos.y + vp->WorkSize.y - margin;

  int idx = 0;
  for (auto it = toasts.rbegin(); it != toasts.rend(); ++it) {
    const Toast& t = *it;
    if (t.ttl <= 0.0f) continue;

    // Fade out over the last second of life.
    float alpha = t.ttl < 1.0f ? t.ttl : 1.0f;
    ImVec2 pos(vp->WorkPos.x + vp->WorkSize.x - margin, y);

    ImGui::SetNextWindowPos(pos, ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.85f * alpha);
    ImGui::SetNextWindowViewport(vp->ID);

    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize;

    char win_id[32];
    std::snprintf(win_id, sizeof(win_id), "##toast_%d", idx);

    if (t.is_error) {
      ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.35f, 0.10f, 0.10f, 1.0f));
    }
    if (ImGui::Begin(win_id, nullptr, flags)) {
      ImVec4 col = t.is_error ? ImVec4(1.0f, 0.75f, 0.75f, alpha)
                              : ImVec4(1.0f, 1.0f, 1.0f, alpha);
      ImGui::PushTextWrapPos(360.0f);
      ImGui::TextColored(col, "%s", t.text.c_str());
      ImGui::PopTextWrapPos();
      // Advance the stack cursor by this toast's height + spacing.
      y -= ImGui::GetWindowHeight() + 6.0f;
    }
    ImGui::End();
    if (t.is_error) ImGui::PopStyleColor();
    ++idx;
  }
}

}  // namespace netvis
