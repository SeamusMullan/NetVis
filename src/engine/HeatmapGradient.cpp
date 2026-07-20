// engine/HeatmapGradient.cpp — heatmap gradient presets + sampling.
#include "engine/HeatmapGradient.h"

#include <algorithm>

namespace netvis {

namespace {

// Preset stop tables. Values are 3-point approximations of the named ramps —
// enough for a per-node tint (the tint is a coarse categorical-ish cue, not a
// scientific colormap). Viridis/Magma keep monotonically increasing lightness so
// the sequential-encoding rule holds; CoolHot preserves the original default;
// Grayscale is the print/forced-colors fallback.
constexpr Rgba8 kViridisLow{68, 1, 84, 255};      // dark violet
constexpr Rgba8 kViridisMid{33, 145, 140, 255};   // teal
constexpr Rgba8 kViridisHigh{253, 231, 37, 255};  // yellow

constexpr Rgba8 kMagmaLow{20, 14, 54, 255};       // near-black indigo
constexpr Rgba8 kMagmaMid{183, 55, 121, 255};     // magenta
constexpr Rgba8 kMagmaHigh{252, 253, 191, 255};   // pale cream

constexpr Rgba8 kCoolHotLow{100, 150, 240, 255};  // blue (v0.3.0 default)
constexpr Rgba8 kCoolHotMid{240, 180, 60, 255};   // amber
constexpr Rgba8 kCoolHotHigh{240, 80, 80, 255};   // red

constexpr Rgba8 kGrayLow{60, 60, 60, 255};
constexpr Rgba8 kGrayMid{140, 140, 140, 255};
constexpr Rgba8 kGrayHigh{230, 230, 230, 255};

uint8_t lerp_u8(uint8_t a, uint8_t b, float t) {
  float v = static_cast<float>(a) + (static_cast<float>(b) - static_cast<float>(a)) * t;
  if (v <= 0.0f) return 0;
  if (v >= 255.0f) return 255;
  return static_cast<uint8_t>(v + 0.5f);  // round to nearest
}

Rgba8 lerp_rgba(const Rgba8& a, const Rgba8& b, float t) {
  return Rgba8{lerp_u8(a.r, b.r, t), lerp_u8(a.g, b.g, t),
               lerp_u8(a.b, b.b, t), lerp_u8(a.a, b.a, t)};
}

}  // namespace

const char* gradient_preset_name(GradientPreset p) {
  switch (p) {
    case GradientPreset::Viridis: return "Viridis";
    case GradientPreset::Magma: return "Magma";
    case GradientPreset::CoolHot: return "Cool→Hot";
    case GradientPreset::Grayscale: return "Grayscale";
    case GradientPreset::Custom: return "Custom";
  }
  return "Custom";
}

void gradient_preset_stops(GradientPreset preset, Rgba8& low, Rgba8& mid,
                           Rgba8& high) {
  switch (preset) {
    case GradientPreset::Viridis:
      low = kViridisLow; mid = kViridisMid; high = kViridisHigh; return;
    case GradientPreset::Magma:
      low = kMagmaLow; mid = kMagmaMid; high = kMagmaHigh; return;
    case GradientPreset::CoolHot:
      low = kCoolHotLow; mid = kCoolHotMid; high = kCoolHotHigh; return;
    case GradientPreset::Grayscale:
      low = kGrayLow; mid = kGrayMid; high = kGrayHigh; return;
    case GradientPreset::Custom:
      // No canonical stops; hand back CoolHot as a neutral editing seed.
      low = kCoolHotLow; mid = kCoolHotMid; high = kCoolHotHigh; return;
  }
}

HeatmapGradient::HeatmapGradient() {
  preset = GradientPreset::Viridis;
  gradient_preset_stops(preset, low, mid, high);
  reverse = false;
}

void gradient_set_preset(HeatmapGradient& gradient, GradientPreset preset) {
  gradient.preset = preset;
  if (preset != GradientPreset::Custom) {
    gradient_preset_stops(preset, gradient.low, gradient.mid, gradient.high);
  }
}

Rgba8 gradient_sample(const HeatmapGradient& gradient, float t) {
  if (!(t == t)) t = 0.0f;  // NaN -> 0 (std::clamp would pass NaN through)
  t = std::clamp(t, 0.0f, 1.0f);
  if (gradient.reverse) t = 1.0f - t;
  if (t < 0.5f) {
    return lerp_rgba(gradient.low, gradient.mid, t * 2.0f);
  }
  return lerp_rgba(gradient.mid, gradient.high, (t - 0.5f) * 2.0f);
}

}  // namespace netvis
