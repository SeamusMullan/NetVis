// engine/HeatmapGradient.h — the cost-heatmap color gradient (v0.3.2 QoL).
//
// DECISION (v0.3.2): the analyzer heatmap colors nodes by a normalized magnitude
// t in [0,1] (log or linear FLOPs). The color mapping is pure logic — a set of
// stops sampled at t — so it lives in the engine, free of ImGui, and is unit-
// tested headless (mirrors OpCategory / CostModel). The view converts the
// returned bytes to an ImU32 and owns the picker/preset UI.
//
// The gradient is a 3-stop ramp (low -> mid -> high) plus a reverse flag. Presets
// are perceptually-monotonic, colorblind-reasonable sequential ramps (a heatmap is
// a SEQUENTIAL encoding: one increasing magnitude, so lightness must increase
// monotonically along the ramp — never a rainbow). "Custom" keeps whatever stops
// the user set. Determinism: gradient_sample is a pure function of (gradient, t).
#pragma once

#include <cstdint>

namespace netvis {

// A straight 8-bit RGBA color, engine-side (no ImGui dependency). The view maps
// this to an ImU32 via IM_COL32(r, g, b, a).
struct Rgba8 {
  uint8_t r = 0, g = 0, b = 0, a = 255;
};

inline bool operator==(const Rgba8& x, const Rgba8& y) {
  return x.r == y.r && x.g == y.g && x.b == y.b && x.a == y.a;
}

// Built-in gradient presets. Custom means "use the stops stored in the struct".
enum class GradientPreset : uint8_t {
  Viridis = 0,   // dark violet -> teal -> yellow  (perceptually uniform)
  Magma,         // near-black -> magenta -> pale   (perceptually uniform)
  CoolHot,       // blue -> amber -> red            (the v0.3.0/0.3.1 default)
  Grayscale,     // dark -> mid -> light            (prints/forced-colors safe)
  Custom,        // use HeatmapGradient::low/mid/high verbatim
};

// Number of built-in presets (excludes Custom's own slot ordering; Custom is the
// last enumerator). Used to drive the UI dropdown.
constexpr int kGradientPresetCount = 5;

const char* gradient_preset_name(GradientPreset p);

// A 3-stop heatmap gradient. When preset != Custom, low/mid/high are kept in sync
// with the preset (set_preset fills them) so the UI can show/edit the stops and a
// preset selection is self-describing when persisted.
struct HeatmapGradient {
  GradientPreset preset = GradientPreset::Viridis;
  Rgba8 low;    // color at t = 0 (cheapest)
  Rgba8 mid;    // color at t = 0.5
  Rgba8 high;   // color at t = 1 (most expensive)
  bool reverse = false;  // flip the ramp (t -> 1 - t)

  // Default-constructs to the Viridis preset stops (not three black stops).
  HeatmapGradient();
};

// Overwrite low/mid/high with a built-in preset's stops and record the preset.
// GradientPreset::Custom is a no-op on the stops (keeps whatever is there) and
// just records the preset tag.
void gradient_set_preset(HeatmapGradient& gradient, GradientPreset preset);

// Fill low/mid/high for a preset without touching a gradient (UI preview / init).
// Custom returns the CoolHot stops as a neutral starting point for editing.
void gradient_preset_stops(GradientPreset preset, Rgba8& low, Rgba8& mid,
                           Rgba8& high);

// Sample the gradient at t. t is clamped to [0,1]; reverse flips it. Piecewise-
// linear: [0,0.5] interpolates low->mid, [0.5,1] interpolates mid->high. Pure and
// deterministic. Alpha is interpolated too (stops are normally opaque).
Rgba8 gradient_sample(const HeatmapGradient& gradient, float t);

}  // namespace netvis
