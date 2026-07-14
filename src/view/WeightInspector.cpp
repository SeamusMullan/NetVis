// view/WeightInspector.cpp — the "Weight Inspector" panel (spec §8.3).
//
// The one panel that shows decoded tensor values. It never reads bytes itself:
// App::inspect_tensor() kicks a background TensorDecodeJob (the sole payload
// reader, spec §2.1) and stores the async state in App::decode(). This panel
// only renders that PendingDecode: an idle hint, a spinner while in-flight,
// stats + a 64-bucket histogram when done, or the error otherwise.
//
// Export buttons call export_npy / export_raw (which stream from the mmap) after
// a native save dialog, then toast the outcome.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h that
// App.h pulls in without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"
#include "view/App.h"
#include "view/PanelHelpers.h"

// tinyfiledialogs (C, spec §8.7). Declared here rather than pulling the C header
// so this TU has no extra include dependency; the symbol links from the
// tinyfiledialogs static lib.
extern "C" char const* tinyfd_saveFileDialog(char const* aTitle,
                                             char const* aDefaultPathAndFile,
                                             int aNumOfFilterPatterns,
                                             char const* const* aFilterPatterns,
                                             char const* aSingleFilterDescription);

namespace netvis {

namespace {

using panel_detail::human_bytes;
using panel_detail::shape_string;

// A tiny rotating spinner drawn into the current window's draw list. ImGui has
// no built-in spinner; a few arc segments is enough and costs nothing.
void draw_spinner(float radius, float thickness, ImU32 color) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  ImVec2 p = ImGui::GetCursorScreenPos();
  ImVec2 center(p.x + radius, p.y + radius);
  const int segments = 30;
  const float t = static_cast<float>(ImGui::GetTime());
  const float start = t * 6.0f;
  const float arc = 3.14159265f * 1.4f;
  dl->PathClear();
  for (int i = 0; i <= segments; ++i) {
    float a = start + (static_cast<float>(i) / segments) * arc;
    dl->PathLineTo(ImVec2(center.x + std::cos(a) * radius,
                          center.y + std::sin(a) * radius));
  }
  dl->PathStroke(color, 0, thickness);
  // Reserve layout space so following widgets don't overlap the spinner.
  ImGui::Dummy(ImVec2(radius * 2, radius * 2));
}

// Draw the 64-bucket histogram using AddRectFilled bars scaled to the max bucket
// (spec §8.3). Pure drawing of already-computed counts — no payload access here.
void draw_histogram(const TensorStats& s) {
  uint64_t maxc = 0;
  for (uint64_t c : s.histogram) maxc = c > maxc ? c : maxc;

  ImVec2 avail = ImGui::GetContentRegionAvail();
  float h = 120.0f;
  float w = avail.x > 16.0f ? avail.x : 256.0f;
  ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList* dl = ImGui::GetWindowDrawList();

  const ImU32 bg = ImGui::GetColorU32(ImGuiCol_FrameBg);
  const ImU32 bar = ImGui::GetColorU32(ImGuiCol_PlotHistogram);
  dl->AddRectFilled(origin, ImVec2(origin.x + w, origin.y + h), bg);

  const int n = kHistogramBuckets;
  const float bw = w / static_cast<float>(n);
  for (int i = 0; i < n; ++i) {
    float frac = maxc ? static_cast<float>(s.histogram[i]) /
                            static_cast<float>(maxc)
                      : 0.0f;
    float bh = frac * (h - 2.0f);
    ImVec2 a(origin.x + i * bw + 0.5f, origin.y + h - bh);
    ImVec2 b(origin.x + (i + 1) * bw - 0.5f, origin.y + h);
    if (bh > 0.0f) dl->AddRectFilled(a, b, bar);
  }
  ImGui::Dummy(ImVec2(w, h));

  // Range labels beneath the plot.
  ImGui::Text("min %g", s.hist_min);
  ImGui::SameLine(w - 120.0f);
  ImGui::Text("max %g", s.hist_max);
}

// Run a save dialog + export, toasting the result. `raw` selects raw bin vs npy.
void do_export(App& app, const ir::TensorRef& t, bool raw) {
  const char* pat_npy[] = {"*.npy"};
  const char* pat_bin[] = {"*.bin"};
  const char* def = raw ? "tensor.bin" : "tensor.npy";
  const char* path =
      tinyfd_saveFileDialog("Export tensor", def, 1, raw ? pat_bin : pat_npy,
                            raw ? "raw binary" : "NumPy array");
  if (!path) return;  // user cancelled

  const std::string model_dir = app.session().model_dir();
  Result<bool> r = raw ? export_raw(t, app.session().file(), model_dir, path)
                       : export_npy(t, app.session().file(), model_dir, path);
  if (r.ok() && *r) {
    app.add_toast(std::string("Exported ") + path, false);
  } else {
    std::string msg = "Export failed";
    if (!r.ok()) msg += ": " + r.error().message;
    app.add_toast(msg, true);
  }
}

}  // namespace

// Draw the Weight Inspector panel (spec §8.3). Called once per frame.
void draw_weight_inspector(App& app) {
  if (!ImGui::Begin("Weight Inspector")) {
    ImGui::End();
    return;
  }

  PendingDecode& d = app.decode();

  if (!d.active) {
    ImGui::TextDisabled("Select a tensor and click Inspect.");
    ImGui::End();
    return;
  }

  const ir::TensorRef& t = d.tensor;
  const ir::Model* model = app.session().model();
  std::string_view name = model ? model->str(t.name) : std::string_view{};

  // Header: identity is known even before the decode finishes.
  ImGui::TextUnformatted(name.empty() ? "(unnamed tensor)"
                                      : std::string(name).c_str());
  ImGui::Text("dtype: %s   shape: %s", ir::dtype_name(t.dtype),
              shape_string(t.shape).c_str());
  ImGui::Text("size:  %s", human_bytes(t.byte_len).c_str());
  ImGui::Separator();

  // In-flight: show a spinner while the worker streams the payload.
  if (d.in_flight || !d.done) {
    draw_spinner(12.0f, 3.0f, ImGui::GetColorU32(ImGuiCol_Text));
    ImGui::SameLine();
    ImGui::TextDisabled("Decoding...");
    ImGui::End();
    return;
  }

  // Done + failed: surface the decode error.
  if (!d.ok) {
    ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f), "Decode failed");
    ImGui::TextWrapped("%s", d.error.c_str());
    ImGui::End();
    return;
  }

  const TensorStats& s = d.stats;

  // GGUF quantized blocks are labels only in v1 (spec §7.5, §12).
  if (s.quantized_unsupported) {
    ImGui::TextWrapped(
        "Dequantization is not supported in v1. This tensor uses a quantized "
        "block format (e.g. GGUF Q4/Q8); only its metadata is shown.");
    ImGui::End();
    return;
  }

  // Stats table.
  ImGui::SeparatorText("Statistics");
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
  if (ImGui::BeginTable("stats", 2, flags)) {
    ImGui::TableSetupColumn("stat", ImGuiTableColumnFlags_WidthFixed, 130.0f);
    ImGui::TableSetupColumn("value");
    auto row = [](const char* k, const std::string& v) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::TextUnformatted(k);
      ImGui::TableSetColumnIndex(1);
      ImGui::TextUnformatted(v.c_str());
    };
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%g", s.min);
    row("min", buf);
    std::snprintf(buf, sizeof(buf), "%g", s.max);
    row("max", buf);
    std::snprintf(buf, sizeof(buf), "%g", s.mean);
    row("mean", buf);
    std::snprintf(buf, sizeof(buf), "%g", s.std);
    row("std", buf);
    row("zeros", std::to_string(s.zero_count));
    row("nan/inf", std::to_string(s.nan_inf_count));
    row("count", std::to_string(s.count));
    ImGui::EndTable();
  }

  ImGui::SeparatorText("Histogram");
  draw_histogram(s);

  ImGui::SeparatorText("Export");
  if (ImGui::Button("Export .npy")) do_export(app, t, /*raw=*/false);
  ImGui::SameLine();
  if (ImGui::Button("Export raw .bin")) do_export(app, t, /*raw=*/true);

  ImGui::End();
}

}  // namespace netvis
