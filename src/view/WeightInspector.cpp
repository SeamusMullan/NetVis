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
//
// v0.9.1b adds two OPT-IN sections that stay inert until asked for (neither
// fires as a side effect of selecting a tensor):
//   #50 cross-model compare -- decodes the SAME-NAMED tensor out of the active
//       DiffLoader comparison slot via App::inspect_tensor_comparison() and
//       renders it beside the A-side stats/histogram with a B-A delta.
//   #49 single-block dequant preview -- for a quantized_unsupported tensor,
//       decodes exactly ONE block via App::preview_tensor_quant_block() and
//       renders the raw floats. View-only; never exported (DECISIONS.md
//       "v0.9.1b -- dequantization scope").
// Both kick their async request only on a real state change (tensor/slot/
// block), never unconditionally every frame -- see the staleness comments on
// draw_quant_preview / draw_comparison_section below.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h that
// App.h pulls in without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
// #50: DiffLoader owns the comparison-model slots the cross-model compare
// section reads (comparison_count/active_comparison/state_of/error_of).
// Reachable transitively via view/App.h already, but named explicitly here
// (matching DiffPanel.cpp) so this TU documents its own dependency.
#include "engine/DiffLoader.h"
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

// #51: unflatten a row-major flat index to a multi-dim coordinate string like
// "[12, 0, 3]" via the tensor shape: coord[k] = (idx / prod(shape[k+1:])) %
// shape[k]. Returns "[]" for a scalar/empty shape. A dynamic (-1) OR zero dim
// can't be unflattened meaningfully (and a 0 dim would make the `% shape[k]`
// divide-by-zero on an inconsistent/hostile model whose payload has bytes but a
// 0-valued declared dim), so we fall back to the raw flat index.
std::string coord_string(const ir::TensorRef& t, uint64_t flat) {
  if (t.shape.empty()) return "[]";
  for (int64_t d : t.shape)
    if (d <= 0) return "flat #" + std::to_string(flat);
  std::string out = "[";
  for (size_t k = 0; k < t.shape.size(); ++k) {
    uint64_t inner = 1;  // prod(shape[k+1:])
    for (size_t j = k + 1; j < t.shape.size(); ++j)
      inner *= static_cast<uint64_t>(t.shape[j]);
    uint64_t c = inner ? (flat / inner) % static_cast<uint64_t>(t.shape[k]) : 0;
    if (k) out += ", ";
    out += std::to_string(c);
  }
  out += "]";
  return out;
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
  Result<bool> r = raw ? export_raw(t, app.session().file(), model_dir, path,
                                    app.session().model())
                       : export_npy(t, app.session().file(), model_dir, path,
                                    app.session().model());
  if (r.ok() && *r) {
    app.add_toast(std::string("Exported ") + path, false);
  } else {
    std::string msg = "Export failed";
    if (!r.ok()) msg += ": " + r.error().message;
    app.add_toast(msg, true);
  }
}

// #49: the block selector + kick-on-change + render for the single-block
// dequant preview. Called only while ViewState::inspector_quant_preview is on
// (the caller gates it), so this function's only job is deciding WHEN to
// (re)request a decode and how to draw whatever PendingDecode::quant currently
// holds.
void draw_quant_preview(App& app, PendingDecode& d) {
  // Stale-request tracking is keyed on the PendingDecode's OWN ADDRESS plus its
  // A-side token, not the bare token: PendingDecode lives one-per-tab and each
  // tab's token sequence starts at 0 independently, so two tabs can carry the
  // same token value at the same moment. Keying on pointer identity is the same
  // discipline DiffPanel's CacheKey and the #62 tint guard already use, for the
  // same reason (see MEMORY: "process-global static caches must key on model/
  // report POINTER identity").
  static const PendingDecode* s_owner = nullptr;
  static uint64_t s_owner_token = UINT64_MAX;
  static uint32_t s_owner_block = UINT32_MAX;

  const bool new_tensor = s_owner != &d || s_owner_token != d.token;
  // Don't inherit a stale index from the PREVIOUS tensor's block ladder.
  if (new_tensor) d.quant_block = 0;

  // Block selector: prev/next steppers + a direct index box. `total` is 0
  // until the first successful preview reports the tensor's real block count,
  // so the next-button/clamp only engage once it is known -- clamping to an
  // unknown bound would itself be a guess, and preview_quant_block() already
  // reports an out-of-range index honestly (available=false) rather than
  // faulting, so an unclamped OOB request here is still safe.
  const uint64_t total = d.quant.total_blocks;
  ImGui::BeginDisabled(d.quant_block == 0);
  if (ImGui::ArrowButton("##quant_prev", ImGuiDir_Left)) --d.quant_block;
  ImGui::EndDisabled();
  ImGui::SameLine();
  int block_i = static_cast<int>(d.quant_block);
  ImGui::SetNextItemWidth(90.0f);
  if (ImGui::InputInt("block", &block_i, 1, 10))
    d.quant_block = block_i < 0 ? 0u : static_cast<uint32_t>(block_i);
  ImGui::SameLine();
  ImGui::BeginDisabled(total > 0 &&
                       static_cast<uint64_t>(d.quant_block) + 1 >= total);
  if (ImGui::ArrowButton("##quant_next", ImGuiDir_Right)) ++d.quant_block;
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (total > 0)
    ImGui::TextDisabled("of %llu", static_cast<unsigned long long>(total));
  else
    ImGui::TextDisabled("(block count unknown until previewed)");

  // Re-kick ONLY when what's on screen no longer matches (tensor, block). An
  // unconditional per-frame call would bump quant_token every frame, so the
  // job would never win the race to publish before being superseded --
  // exactly the "spinner forever" failure mode the spec warns against.
  if (new_tensor || s_owner_block != d.quant_block) {
    app.preview_tensor_quant_block(d.quant_block);
    s_owner = &d;
    s_owner_token = d.token;
    s_owner_block = d.quant_block;
  }

  if (d.quant_in_flight) {
    draw_spinner(10.0f, 2.5f, ImGui::GetColorU32(ImGuiCol_Text));
    ImGui::SameLine();
    ImGui::TextDisabled("Decoding block...");
    return;
  }
  if (!d.quant_done) return;  // request just posted; lands in a frame or two

  if (!d.quant_error.empty()) {
    ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f), "Preview failed");
    ImGui::TextWrapped("%s", d.quant_error.c_str());
    return;
  }

  const QuantBlockPreview& q = d.quant;
  if (!q.available) {
    // Honest per spec: never render a partial/approximated value here -- show
    // the engine's exact, fixed reason (e.g. a K-quant super-block) verbatim.
    ImGui::TextWrapped("%s", q.unavailable_reason.c_str());
    return;
  }

  ImGui::Text("type: %s   block %u of %llu", q.type_name.c_str(), q.block_index,
             static_cast<unsigned long long>(q.total_blocks));
  ImGui::Text("elements [%llu, %llu)",
             static_cast<unsigned long long>(q.first_elem),
             static_cast<unsigned long long>(q.first_elem + q.elem_count));

  // A small grid of the decoded floats. No export button here, EVER: exporting
  // dequantized payload is an explicit non-goal (DECISIONS.md "v0.9.1b --
  // dequantization scope"). This preview is view-only by contract -- if a
  // future change wants to export these values, that is the wrong side of the
  // non-goal and belongs in a design discussion, not a quick button here.
  const ImGuiTableFlags gflags =
      ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit;
  if (ImGui::BeginTable("quant_values", 8, gflags)) {
    for (uint32_t i = 0; i < q.elem_count; ++i) {
      if (i % 8 == 0) ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(static_cast<int>(i % 8));
      ImGui::Text("%g", static_cast<double>(q.values[i]));
    }
    ImGui::EndTable();
  }
}

// #50: the cross-model same-tensor compare toggle + B-side render. `s` is the
// already-computed A-side stats (the caller only reaches here once s is valid,
// i.e. never for a quantized_unsupported A-side tensor -- its min/max/mean/std
// are meaningless placeholders, so a delta against them would be too).
void draw_comparison_section(App& app, PendingDecode& d, const TensorStats& s) {
  ImGui::SeparatorText("Comparison");
  // Sticky across tensor selections: the flag lives in ViewState (not here) so
  // a user walking a quant ladder doesn't have to re-enable it per tensor.
  ImGui::Checkbox("Compare with comparison model", &app.view().inspector_compare);
  if (!app.view().inspector_compare) return;

  DiffLoader& diff = app.diff_loader();
  if (diff.comparison_count() == 0) {
    ImGui::TextDisabled("No comparison model loaded (Diff panel opens one).");
    return;
  }

  const size_t slot = diff.active_comparison();
  const DiffLoadState state = diff.state_of(slot);
  if (state == DiffLoadState::Loading || state == DiffLoadState::Empty) {
    draw_spinner(10.0f, 2.5f, ImGui::GetColorU32(ImGuiCol_Text));
    ImGui::SameLine();
    ImGui::TextDisabled("Comparison model is loading...");
    return;
  }
  if (state == DiffLoadState::Failed) {
    ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f),
                       "Comparison model failed to load");
    ImGui::TextWrapped("%s", diff.error_of(slot).c_str());
    return;
  }

  // Same pointer+token staleness discipline as draw_quant_preview, PLUS the
  // active slot: a quant-ladder user can switch which comparison is "active"
  // (#36) without touching the inspected tensor, and the B side must follow.
  static const PendingDecode* s_owner = nullptr;
  static uint64_t s_owner_token = UINT64_MAX;
  static size_t s_owner_slot = SIZE_MAX;
  if (s_owner != &d || s_owner_token != d.token || s_owner_slot != slot) {
    app.inspect_tensor_comparison();
    s_owner = &d;
    s_owner_token = d.token;
    s_owner_slot = slot;
  }

  if (d.cmp_in_flight) {
    draw_spinner(10.0f, 2.5f, ImGui::GetColorU32(ImGuiCol_Text));
    ImGui::SameLine();
    ImGui::TextDisabled("Decoding comparison tensor...");
    return;
  }
  if (!d.cmp_done) return;  // request just posted

  if (!d.cmp_ok) {
    ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.4f, 1.0f), "Comparison unavailable");
    if (!d.cmp_label.empty()) ImGui::TextDisabled("(%s)", d.cmp_label.c_str());
    ImGui::TextWrapped("%s", d.cmp_error.c_str());
    return;
  }
  if (d.cmp_stats.quantized_unsupported) {
    ImGui::TextWrapped(
        "%s's copy of this tensor uses a quantized block format; only "
        "metadata is available there (use the block preview above on THIS "
        "side; there is no B-side equivalent here), so no value comparison "
        "is shown.",
        d.cmp_label.empty() ? "The comparison model" : d.cmp_label.c_str());
    return;
  }

  const TensorStats& cs = d.cmp_stats;
  ImGui::Text("B = %s", d.cmp_label.c_str());

  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg;
  if (ImGui::BeginTable("cmp_stats", 4, flags)) {
    ImGui::TableSetupColumn("stat", ImGuiTableColumnFlags_WidthFixed, 90.0f);
    ImGui::TableSetupColumn("A");
    ImGui::TableSetupColumn("B");
    ImGui::TableSetupColumn("B - A");
    ImGui::TableHeadersRow();
    // Deltas are B - A (matching the sign convention engine/TensorDiff.h and
    // the #31 cost-delta panel already use, so a user reading both agrees on
    // which side a positive number favors).
    auto frow = [](const char* k, double a, double b) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::TextUnformatted(k);
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%g", a);
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%g", b);
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%+g", b - a);
    };
    auto urow = [](const char* k, uint64_t a, uint64_t b) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::TextUnformatted(k);
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%llu", static_cast<unsigned long long>(a));
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%llu", static_cast<unsigned long long>(b));
      ImGui::TableSetColumnIndex(3);
      // Signed delta -- an unsigned subtraction on a shrink would wrap to
      // ~1.8e19 instead of reading negative (the same bug class engine/
      // TensorDiff.h's d_zero_count()/d_nan_inf_count() guard against).
      ImGui::Text("%+lld", static_cast<long long>(b) - static_cast<long long>(a));
    };
    frow("min", s.min, cs.min);
    frow("max", s.max, cs.max);
    frow("mean", s.mean, cs.mean);
    frow("std", s.std, cs.std);
    urow("zeros", s.zero_count, cs.zero_count);
    urow("nan/inf", s.nan_inf_count, cs.nan_inf_count);
    urow("count", s.count, cs.count);
    ImGui::EndTable();
  }

  // Histograms side by side, each scaled to its OWN [hist_min, hist_max] --
  // NOT forced onto a shared range. compute_tensor_stats bucketizes during its
  // one streaming pass and only the 64 bucket COUNTS survive; the raw values
  // that produced them are gone by design (that's the whole point of the
  // zero-payload streaming decode). Rebinning onto a common range from here
  // would need either the discarded raw values or a second, wider decode of
  // whichever side has the narrower range -- not worth another payload read
  // for a preview panel. So both are drawn self-scaled (draw_histogram already
  // labels each with its own min/max underneath) and the mismatch is called
  // out explicitly rather than left implicit: two histograms with different
  // x-axes and no label would be exactly the misleading chart the honesty
  // rules forbid.
  ImGui::TextDisabled(
      "Each histogram is scaled to its OWN range (see min/max below it) -- "
      "bucket i in A and bucket i in B are not necessarily the same values.");
  ImGui::Columns(2, "cmp_hist_cols", false);
  ImGui::TextUnformatted("A");
  draw_histogram(s);
  ImGui::NextColumn();
  ImGui::TextUnformatted(d.cmp_label.c_str());
  draw_histogram(cs);
  ImGui::Columns(1);
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
  // Prefer the exact dtype label when the type has no ir::DType (#85: OpenVINO
  // i4/u4/nf4/u1, CoreML MIL sub-byte/fp8), so a quant tensor reads honestly
  // instead of "?".
  std::string dtype_disp = ir::dtype_name(t.dtype);
  if (t.dtype == ir::DType::Unknown && model && t.dtype_label.valid()) {
    std::string_view lbl = model->str(t.dtype_label);
    if (!lbl.empty()) dtype_disp = std::string(lbl);
  }
  ImGui::Text("dtype: %s   shape: %s", dtype_disp.c_str(),
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

  // #49 (DECISIONS.md "v0.9.1b -- dequantization scope"): GGUF quantized
  // blocks are metadata-only in the normal decode path above -- TensorStats
  // never materializes per-element floats for a block format, so min/max/
  // mean/std/histogram would all read as a meaningless 0 here (spec §7.5,
  // §12). NetVis still does NOT dequantize this tensor as a transform: it
  // never converts the whole payload to float, never writes a dequantized
  // file, and never hands dequantized data to a plugin. What IS supported is
  // a bounded, opt-in, view-only preview of a single decoded block, so a user
  // can see real numbers behind the metadata without NetVis crossing into
  // "dequantizer" territory.
  if (s.quantized_unsupported) {
    ImGui::TextWrapped(
        "This tensor uses a quantized block format (e.g. GGUF Q4/Q8). NetVis "
        "does not dequantize it as a whole -- below is an optional, "
        "read-only preview of ONE decoded block (up to %u values) for the "
        "legacy GGUF layouts (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0); K-quants and IQ* "
        "formats report why they can't be previewed instead of guessing.",
        kQuantPreviewMaxElems);

    // #49: opt-in BY CONTRACT, not merely by convention -- default off, and
    // nothing below fires until the user explicitly flips this checkbox.
    ImGui::Checkbox("Preview one block", &app.view().inspector_quant_preview);
    if (app.view().inspector_quant_preview) draw_quant_preview(app, d);

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

  // #51: argmin/argmax with their multi-dim coordinates ("jump to"). Skip an
  // extreme whose index is UINT64_MAX (no finite element was scanned).
  if (s.min_index != UINT64_MAX || s.max_index != UINT64_MAX) {
    ImGui::SeparatorText("Extremes");
    if (s.min_index != UINT64_MAX)
      ImGui::Text("min %g at %s", s.min, coord_string(t, s.min_index).c_str());
    if (s.max_index != UINT64_MAX)
      ImGui::Text("max %g at %s", s.max, coord_string(t, s.max_index).c_str());
  }

  // #48: whole-tensor + per-channel outlier warnings (red for the alarming ones).
  const ImVec4 warn(0.95f, 0.4f, 0.4f, 1.0f);
  if (s.has_nan_inf())
    ImGui::TextColored(warn, "%llu NaN/Inf values",
                       static_cast<unsigned long long>(s.nan_inf_count));
  if (s.all_zero()) ImGui::TextColored(warn, "tensor is all zero");
  if (!s.per_channel.empty()) {
    uint64_t dead = 0, nan_ch = 0;
    for (const ChannelStat& c : s.per_channel) {
      if (c.all_zero()) ++dead;
      if (c.has_nan_inf()) ++nan_ch;
    }
    if (dead || nan_ch)
      ImGui::TextColored(warn, "%llu dead channels, %llu channels with NaN/Inf",
                         static_cast<unsigned long long>(dead),
                         static_cast<unsigned long long>(nan_ch));
  }

  // #46: per-output-channel stats. Collapsible + virtualized (up to kMaxChannels
  // rows). Capped tensors show a note instead of an unbounded table.
  if (s.per_channel_capped) {
    ImGui::SeparatorText("Per-channel");
    ImGui::TextDisabled("too many channels (>%u) - per-channel stats omitted",
                        kMaxChannels);
  } else if (!s.per_channel.empty()) {
    char hdr[48];
    std::snprintf(hdr, sizeof(hdr), "Per-channel (%zu)", s.per_channel.size());
    if (ImGui::CollapsingHeader(hdr)) {
      const ImGuiTableFlags cflags = ImGuiTableFlags_Borders |
                                     ImGuiTableFlags_RowBg |
                                     ImGuiTableFlags_ScrollY;
      ImVec2 tsize(0.0f, 220.0f);
      if (ImGui::BeginTable("per_channel", 5, cflags, tsize)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("ch");
        ImGui::TableSetupColumn("min");
        ImGui::TableSetupColumn("max");
        ImGui::TableSetupColumn("mean");
        ImGui::TableSetupColumn("flag");
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(s.per_channel.size()));
        while (clipper.Step()) {
          for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
            const ChannelStat& c = s.per_channel[static_cast<size_t>(r)];
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%d", r);
            ImGui::TableSetColumnIndex(1);
            ImGui::Text("%g", c.min);
            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%g", c.max);
            ImGui::TableSetColumnIndex(3);
            ImGui::Text("%g", c.mean);
            ImGui::TableSetColumnIndex(4);
            if (c.has_nan_inf())
              ImGui::TextColored(warn, "nan");
            else if (c.all_zero())
              ImGui::TextColored(warn, "dead");
          }
        }
        ImGui::EndTable();
      }
    }
  }

  ImGui::SeparatorText("Export");
  if (ImGui::Button("Export .npy")) do_export(app, t, /*raw=*/false);
  ImGui::SameLine();
  if (ImGui::Button("Export raw .bin")) do_export(app, t, /*raw=*/true);

  // #50: appended after everything above so the existing single-tensor
  // sections (stats/histogram/extremes/warnings/per-channel/export) render
  // exactly as before whether or not the user ever opens this.
  draw_comparison_section(app, d, s);

  ImGui::End();
}

}  // namespace netvis
