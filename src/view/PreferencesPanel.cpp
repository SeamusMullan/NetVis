// view/PreferencesPanel.cpp — the one Settings window (#102).
//
// Every widget here writes THROUGH to the live ViewState the rest of the app
// already renders from, so there is no apply step and no shadow copy of the
// settings to keep in sync (PreferencesPanel.h). The only bookkeeping is `dirty`,
// which decides whether view_prefs.json is rewritten at the end of the frame — a
// checkbox that is clicked is a preference that changed; one that is merely drawn
// is not.
#include "view/PreferencesPanel.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h (pulled in by
// App.h) references without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/HeatmapGradient.h"
// ViewState holds nav / cost / history by unique_ptr behind forward declarations.
// pristine_defaults() below default-CONSTRUCTS a ViewState, so ~ViewState is
// instantiated in this TU and each deleter needs its complete type.
#include "engine/CostModel.h"
#include "view/App.h"
#include "view/CostPanel.h"   // rgba8_to_imu32 (the gradient preview strip)
#include "view/GraphNav.h"
#include "view/ViewHistory.h"

namespace netvis {

namespace {

// UI-scale bounds (#104). The lower bound is not a taste judgement: ViewState's
// comment spells out the trap, and it is a one-way door. A scale of 0 (or a
// negative) renders every window at zero size, INCLUDING this one, so the control
// that caused the problem becomes unreachable and the only cure is hand-editing
// view_prefs.json. The widget below is therefore built so it cannot emit a value
// outside this range at all, rather than emitting one and correcting it later.
constexpr float kUiScaleMin = 0.75f;
constexpr float kUiScaleMax = 2.0f;

// A pristine ViewState, used as the source of every section's reset values.
//
// The alternative — a second list of literals next to the reset buttons — drifts:
// when someone changes a field's initializer in App.h, "Reset" silently keeps
// restoring the OLD default, which is worse than having no reset at all because
// it looks like it worked. Reading the defaults off a default-constructed
// ViewState makes that drift impossible by construction. Cheap: a handful of
// empty strings/vectors, and the three unique_ptr members stay null.
const ViewState& pristine_defaults() {
  static const ViewState kDefaults;
  return kDefaults;
}

// Round a 0..1 colour component back to the 8-bit stop representation.
uint8_t to_byte(float v) {
  return static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// Edit one gradient stop. The colour is applied LIVE so the preview strip tracks
// a drag, but persistence is deferred to IsItemDeactivatedAfterEdit: a drag is
// one preference change, not the four hundred view_prefs.json rewrites a
// per-frame save would produce. Returns true on the frame the edit is committed.
bool edit_gradient_stop(const char* label, HeatmapGradient& grad, Rgba8& stop) {
  float col[3] = {stop.r / 255.0f, stop.g / 255.0f, stop.b / 255.0f};
  if (ImGui::ColorEdit3(label, col,
                        ImGuiColorEditFlags_NoInputs |
                            ImGuiColorEditFlags_NoLabel)) {
    stop.r = to_byte(col[0]);
    stop.g = to_byte(col[1]);
    stop.b = to_byte(col[2]);
    stop.a = 255;
    // Moving a stop makes the gradient Custom by definition: the preset tag is a
    // claim about what the stops ARE, and it stops being true the instant one
    // moves. Leaving it on "Viridis" would also lose the edit on next launch —
    // load_prefs only reads persisted stops for a Custom gradient (App.cpp).
    grad.preset = GradientPreset::Custom;
  }
  const bool commit = ImGui::IsItemDeactivatedAfterEdit();
  ImGui::SameLine();
  ImGui::TextUnformatted(label);
  return commit;
}

// Compact preview of the active gradient. Editing three stops with no sight of
// the ramp they produce is not an editor, so this is the minimum that makes the
// controls above usable; the analyzer panel keeps the full-size version.
void draw_gradient_preview(const HeatmapGradient& grad) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  constexpr float kBarW = 180.0f, kBarH = 12.0f;
  constexpr int kSegs = 40;
  for (int i = 0; i < kSegs; ++i) {
    const float t0 = static_cast<float>(i) / kSegs;
    const float t1 = static_cast<float>(i + 1) / kSegs;
    const ImU32 c0 = rgba8_to_imu32(gradient_sample(grad, t0));
    const ImU32 c1 = rgba8_to_imu32(gradient_sample(grad, t1));
    dl->AddRectFilledMultiColor(ImVec2(p0.x + kBarW * t0, p0.y),
                                ImVec2(p0.x + kBarW * t1, p0.y + kBarH), c0, c1,
                                c1, c0);
  }
  dl->AddRect(p0, ImVec2(p0.x + kBarW, p0.y + kBarH), IM_COL32(90, 90, 90, 255));
  ImGui::Dummy(ImVec2(kBarW, kBarH));
}

// A section's reset control. Deliberately at the FOOT of its section and named
// after it: reset is per-section by contract (PreferencesPanel.h), and a button
// labelled only "Reset" sitting under the title of a window called Preferences
// reads as "reset everything", which is exactly the accident that design exists
// to prevent. Small, unemphasised, and last.
bool reset_button(const char* label) {
  const bool hit = ImGui::SmallButton(label);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("Restores only the settings in this section.");
  return hit;
}

}  // namespace

void draw_preferences_panel(App& app) {
  ViewState& vs = app.view();
  if (!vs.show_preferences) return;

  const ViewState& def = pristine_defaults();

  ImGui::SetNextWindowSize(ImVec2(470.0f, 640.0f), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Preferences", &vs.show_preferences)) {
    ImGui::End();
    return;
  }

  ImGui::TextDisabled(
      "Changes apply immediately and are remembered across launches.");
  ImGui::Spacing();

  // Set by any control that changed a PERSISTED field; drives the single
  // save_prefs() at the end. Theme changes are the exception — App::set_theme
  // persists on its own because it also has to re-apply the ImGui style.
  bool dirty = false;

  // --- Appearance -----------------------------------------------------------
  ImGui::SeparatorText("Appearance");
  // Routed through set_theme rather than assigning dark_theme directly: the
  // colours only change when apply_theme() re-runs, so a raw assignment would
  // persist a preference this window appears not to have applied.
  if (ImGui::RadioButton("Dark", vs.dark_theme)) app.set_theme(true);
  ImGui::SameLine();
  if (ImGui::RadioButton("Light", !vs.dark_theme)) app.set_theme(false);
  if (reset_button("Reset appearance")) app.set_theme(def.dark_theme);

  // --- Graph ----------------------------------------------------------------
  ImGui::SeparatorText("Graph");
  if (ImGui::Checkbox("Minimap", &vs.show_minimap)) dirty = true;
  if (ImGui::Checkbox("Layer bands (depth ruler)", &vs.show_layer_bands))
    dirty = true;
  if (ImGui::Checkbox("Edge shape tooltips", &vs.edge_tooltips)) dirty = true;
  // Combo over the frozen int encoding (0=Bezier, 1=Orthogonal, 2=Straight).
  // Combo cannot select outside its item list, so the persisted value stays in
  // the range load_prefs validates.
  ImGui::SetNextItemWidth(160.0f);
  if (ImGui::Combo("Edge routing", &vs.edge_routing,
                   "Bezier\0Orthogonal\0Straight\0"))
    dirty = true;
  if (reset_button("Reset graph")) {
    vs.show_minimap = def.show_minimap;
    vs.show_layer_bands = def.show_layer_bands;
    vs.edge_tooltips = def.edge_tooltips;
    vs.edge_routing = def.edge_routing;
    dirty = true;
  }

  // --- Analyzer -------------------------------------------------------------
  ImGui::SeparatorText("Analyzer");
  if (ImGui::Checkbox("Cost heatmap overlay", &vs.cost_heatmap)) dirty = true;

  ImGui::SetNextItemWidth(160.0f);
  if (ImGui::BeginCombo("Heatmap metric",
                        heatmap_metric_name(vs.heatmap_metric))) {
    for (int i = 0; i < kHeatmapMetricCount; ++i) {
      const auto m = static_cast<HeatmapMetric>(i);
      const bool is_sel = (vs.heatmap_metric == m);
      if (ImGui::Selectable(heatmap_metric_name(m), is_sel)) {
        vs.heatmap_metric = m;
        dirty = true;
      }
      if (is_sel) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  ImGui::TextUnformatted("Scale:");
  ImGui::SameLine();
  if (ImGui::RadioButton("log", vs.heatmap_log_scale)) {
    vs.heatmap_log_scale = true;
    dirty = true;
  }
  ImGui::SameLine();
  if (ImGui::RadioButton("linear", !vs.heatmap_log_scale)) {
    vs.heatmap_log_scale = false;
    dirty = true;
  }

  HeatmapGradient& grad = vs.heatmap_gradient;
  ImGui::SetNextItemWidth(160.0f);
  if (ImGui::BeginCombo("Gradient", gradient_preset_name(grad.preset))) {
    for (int i = 0; i < kGradientPresetCount; ++i) {
      const auto p = static_cast<GradientPreset>(i);
      const bool is_sel = (grad.preset == p);
      if (ImGui::Selectable(gradient_preset_name(p), is_sel)) {
        gradient_set_preset(grad, p);
        dirty = true;
      }
      if (is_sel) ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
  draw_gradient_preview(grad);
  if (edit_gradient_stop("low", grad, grad.low)) dirty = true;
  if (edit_gradient_stop("mid", grad, grad.mid)) dirty = true;
  if (edit_gradient_stop("high", grad, grad.high)) dirty = true;
  if (ImGui::Checkbox("Reverse gradient", &grad.reverse)) dirty = true;

  // Roofline machine balance. custom_ridge > 0 overrides whichever preset the
  // analyzer panel has selected; 0 means "use the preset", which is why the edit
  // below floors at 0 instead of rejecting it — 0 is the meaningful "no override"
  // value, not an error.
  double ridge = vs.custom_ridge;
  ImGui::SetNextItemWidth(120.0f);
  if (ImGui::InputDouble("Custom ridge (FLOP/byte)", &ridge, 0.0, 0.0, "%.1f",
                         ImGuiInputTextFlags_EnterReturnsTrue)) {
    vs.custom_ridge = ridge > 0.0 ? ridge : 0.0;
    dirty = true;
  }
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("0 = use the analyzer's selected machine-balance preset.");

  if (vs.machine_profiles.empty()) {
    ImGui::TextDisabled(
        "No saved machine profiles. Name one from the analyzer's Efficiency\n"
        "section, where the ridge you are naming is on screen next to it.");
  } else {
    // Deferred delete: erasing mid-loop would invalidate the very vector the
    // remaining rows are read from.
    int to_delete = -1;
    for (size_t i = 0; i < vs.machine_profiles.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      if (ImGui::SmallButton("use")) {
        vs.custom_ridge = vs.machine_profiles[i].second;
        dirty = true;
      }
      ImGui::SameLine();
      if (ImGui::SmallButton("x")) to_delete = static_cast<int>(i);
      ImGui::SameLine();
      ImGui::Text("%s (%.1f FLOP/byte)", vs.machine_profiles[i].first.c_str(),
                  vs.machine_profiles[i].second);
      ImGui::PopID();
    }
    if (to_delete >= 0) {
      vs.machine_profiles.erase(vs.machine_profiles.begin() + to_delete);
      dirty = true;
    }
  }

  if (reset_button("Reset analyzer")) {
    vs.cost_heatmap = def.cost_heatmap;
    vs.heatmap_metric = def.heatmap_metric;
    vs.heatmap_log_scale = def.heatmap_log_scale;
    vs.heatmap_gradient = def.heatmap_gradient;
    vs.custom_ridge = def.custom_ridge;
    // machine_profiles is deliberately NOT reset. It is the one thing in this
    // section the user AUTHORED — names they typed for their own hardware — not a
    // setting with a default to return to, and destroying it is unrecoverable
    // (nothing else remembers those numbers). Same principle as the plugin
    // carve-out below, and it costs the user nothing: the per-row "x" above is a
    // deliberate, targeted way to remove one.
    dirty = true;
  }

  // --- Accessibility --------------------------------------------------------
  ImGui::SeparatorText("Accessibility");
  if (ImGui::Checkbox("Accessible op-category palette", &vs.accessible_palette))
    dirty = true;
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip(
        "Swaps the category colours AND adds a border-style cue.\n"
        "Fifteen categories cannot be separated by hue alone, so the\n"
        "non-colour cue is part of the same encoding, not a second option.");

  // Clamped in two independent ways, because this is the setting that can lock a
  // user out of the window that fixes it: AlwaysClamp stops the ctrl+click text
  // entry from leaving the range, and the std::clamp means even a widget that
  // somehow returned something else never reaches ViewState. The edit lands in a
  // local first for exactly that reason.
  float scale = vs.ui_scale;
  ImGui::SetNextItemWidth(160.0f);
  if (ImGui::SliderFloat("UI scale", &scale, kUiScaleMin, kUiScaleMax, "%.2fx",
                         ImGuiSliderFlags_AlwaysClamp)) {
    vs.ui_scale = std::clamp(scale, kUiScaleMin, kUiScaleMax);
    dirty = true;
  }
  if (reset_button("Reset accessibility")) {
    vs.accessible_palette = def.accessible_palette;
    vs.ui_scale = std::clamp(def.ui_scale, kUiScaleMin, kUiScaleMax);
    dirty = true;
  }

  // --- Session --------------------------------------------------------------
  ImGui::SeparatorText("Session");
  if (ImGui::Checkbox("Reopen last session's tabs on launch",
                      &vs.restore_session))
    dirty = true;
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip(
        "Off by default: silently reopening several multi-gigabyte models\n"
        "turns a deliberate action into an unasked-for startup cost.");
  if (reset_button("Reset session")) {
    vs.restore_session = def.restore_session;
    dirty = true;
  }

  // --- Plugins --------------------------------------------------------------
  ImGui::SeparatorText("Plugins");
  ImGui::TextWrapped(
      "Per-plugin enablement lives in the Plugins panel, which shows each "
      "plugin's trust tier, what it registers, and why it was rejected.");
  if (ImGui::Button("Open Plugins panel")) vs.show_plugins = true;
  // Stated in the UI, not only in the header, because a user who has just pressed
  // four reset buttons deserves to know which of their decisions survived.
  ImGui::TextDisabled("Plugin enablement is never changed by any reset above.");
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip(
        "Enabling a WASM plugin is a trust decision made through a\n"
        "confirmation dialog. A preferences reset must not silently\n"
        "undo a security choice.");

  ImGui::End();

  if (dirty) app.save_prefs();
}

}  // namespace netvis
