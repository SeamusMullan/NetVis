// tests/test_gradient.cpp — heatmap gradient presets + sampling (v0.3.2 QoL).
//
// The gradient is a pure engine function (no ImGui), so its endpoints, midpoint,
// monotonicity, reverse, clamping, and preset round-trip are all unit-testable
// headless. This is the piece the cost heatmap's correctness rests on.
#include <doctest/doctest.h>

#include <initializer_list>

#include "engine/HeatmapGradient.h"

using namespace netvis;

namespace {

// Perceived lightness proxy (Rec. 601 luma). Monotonic-lightness ramps should
// have this increase from low to high.
int luma(const Rgba8& c) {
  return static_cast<int>(0.299 * c.r + 0.587 * c.g + 0.114 * c.b);
}

}  // namespace

TEST_CASE("Gradient: endpoints and midpoint hit the stops exactly") {
  HeatmapGradient g;
  gradient_set_preset(g, GradientPreset::CoolHot);
  CHECK(gradient_sample(g, 0.0f) == g.low);
  CHECK(gradient_sample(g, 1.0f) == g.high);
  CHECK(gradient_sample(g, 0.5f) == g.mid);
}

TEST_CASE("Gradient: t is clamped to [0,1]") {
  HeatmapGradient g;
  gradient_set_preset(g, GradientPreset::Viridis);
  // Below 0 and above 1 clamp to the endpoints (no out-of-range color / no crash).
  CHECK(gradient_sample(g, -5.0f) == g.low);
  CHECK(gradient_sample(g, 5.0f) == g.high);
}

TEST_CASE("Gradient: reverse flips the ramp") {
  HeatmapGradient g;
  gradient_set_preset(g, GradientPreset::Viridis);
  Rgba8 fwd_lo = gradient_sample(g, 0.0f);
  Rgba8 fwd_hi = gradient_sample(g, 1.0f);
  g.reverse = true;
  CHECK(gradient_sample(g, 0.0f) == fwd_hi);
  CHECK(gradient_sample(g, 1.0f) == fwd_lo);
  CHECK(gradient_sample(g, 0.5f) == g.mid);  // midpoint is invariant under reverse
}

TEST_CASE("Gradient: Viridis/Magma/Grayscale are monotonic in lightness") {
  for (GradientPreset p : {GradientPreset::Viridis, GradientPreset::Magma,
                           GradientPreset::Grayscale}) {
    HeatmapGradient g;
    gradient_set_preset(g, p);
    int prev = -1;
    for (int i = 0; i <= 10; ++i) {
      Rgba8 c = gradient_sample(g, i / 10.0f);
      int l = luma(c);
      CHECK(l >= prev);  // non-decreasing lightness along the ramp
      prev = l;
    }
  }
}

TEST_CASE("Gradient: preset round-trips through name") {
  for (int i = 0; i < kGradientPresetCount; ++i) {
    auto p = static_cast<GradientPreset>(i);
    const char* name = gradient_preset_name(p);
    // Names are distinct and non-empty.
    CHECK(name != nullptr);
    CHECK(name[0] != '\0');
  }
}

TEST_CASE("Gradient: default construction is Viridis with filled stops") {
  HeatmapGradient g;
  CHECK(g.preset == GradientPreset::Viridis);
  // Not three black stops — the preset filled them.
  Rgba8 lo, mid, hi;
  gradient_preset_stops(GradientPreset::Viridis, lo, mid, hi);
  CHECK(g.low == lo);
  CHECK(g.mid == mid);
  CHECK(g.high == hi);
}

TEST_CASE("Gradient: set_preset(Custom) keeps existing stops") {
  HeatmapGradient g;
  gradient_set_preset(g, GradientPreset::Grayscale);
  Rgba8 lo = g.low, mid = g.mid, hi = g.high;
  gradient_set_preset(g, GradientPreset::Custom);
  CHECK(g.preset == GradientPreset::Custom);
  CHECK(g.low == lo);   // unchanged
  CHECK(g.mid == mid);
  CHECK(g.high == hi);
}
