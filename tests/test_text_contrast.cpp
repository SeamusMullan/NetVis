// SPDX-License-Identifier: Apache-2.0
// tests/test_text_contrast.cpp — node label contrast (#150).
//
// WHY THIS FILE CAN RUN AT ALL: netvis_tests links netvis_core and nothing else,
// so a test of GUI code normally cannot link (see the long note at the top of
// tests/test_category_style.cpp). src/view/TextContrast.cpp is therefore written
// GUI-free and listed explicitly in netvis_core's sources, exactly as
// CategoryStyle.cpp is — the maths is in a pure translation unit precisely so
// these ratios are asserted rather than eyeballed.
//
// WHAT IS NOT COVERED: the draw call itself. GraphCanvas.cpp needs ImGui and
// cannot be linked here, so what this file pins is the maths plus the exact
// colour arithmetic the canvas performs (dim multiplier, body-over-canvas
// compositing). The canvas constants are duplicated below rather than shared,
// because sharing them would mean dragging ImVec4/ImU32 into core; each copy
// names its source so a drift shows up as a failing test rather than silently.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>

#include "engine/OpCategory.h"
#include "view/CategoryStyle.h"
#include "view/TextContrast.h"

using namespace netvis;

namespace {

// WCAG AA. 4.5:1 is the normal-text threshold; node labels are 12-16px at
// zoom 1.0, which is NOT "large text" (18pt / 14pt bold), so 3:1 does not apply.
constexpr double kAA = 4.5;

constexpr uint32_t pack(int r, int g, int b) {
  return 0xFF000000u | (static_cast<uint32_t>(b) << 16) |
         (static_cast<uint32_t>(g) << 8) | static_cast<uint32_t>(r);
}

// --- The canvas's own numbers, duplicated (see the header note) --------------
// ImGuiCol_ChildBg from App::apply_theme(): 0.10,0.11,0.13 dark / 0.98,0.98,0.99
// light, through ImGui's float->u32 conversion ((int)(f*255+0.5f)).
constexpr uint32_t kCanvasDark = pack(26, 28, 33);
constexpr uint32_t kCanvasLight = pack(250, 250, 252);
// Node body rects, from GraphCanvas.cpp's node loop.
constexpr uint32_t kBodyDark = pack(34, 38, 44);
constexpr uint32_t kBodyDarkHover = pack(48, 54, 62);
constexpr uint32_t kBodyLight = pack(244, 246, 249);
constexpr uint32_t kBodyLightHover = pack(232, 238, 246);
// kDimAlpha (0.18f) applied by with_alpha_mul(): ImGui round-trips the alpha as
// (int)(1.0f * 0.18f * 255.0f + 0.5f) == 46.
constexpr uint32_t kDimA = 46;

uint32_t dimmed(uint32_t col) {
  return (col & 0x00FFFFFFu) | (kDimA << 24);
}

// The exact expression GraphCanvas.cpp evaluates per node, so a change to one
// without the other fails here.
double canvas_label_contrast(uint32_t header, uint32_t body, uint32_t canvas) {
  return readable_label_contrast(header, composite_over(body, canvas));
}

constexpr int kLastCategory = static_cast<int>(OpCategory::Other);

}  // namespace

TEST_CASE("relative luminance matches the WCAG reference points") {
  CHECK(relative_luminance(pack(0, 0, 0)) == doctest::Approx(0.0));
  CHECK(relative_luminance(pack(255, 255, 255)) == doctest::Approx(1.0));
  // Mid grey is the classic sanity value: sRGB 128 is NOT luminance 0.5, and a
  // linear (gamma-ignoring) implementation would report 0.502 here instead.
  CHECK(relative_luminance(pack(128, 128, 128)) == doctest::Approx(0.2158).epsilon(0.001));
  // Below the 0.03928 knee the curve is the linear segment, not the power one.
  CHECK(relative_luminance(pack(5, 5, 5)) == doctest::Approx(0.001518).epsilon(0.01));
  // Alpha is ignored, never premultiplied — callers composite first.
  CHECK(relative_luminance(0x00FFFFFFu) == doctest::Approx(1.0));
}

TEST_CASE("contrast ratio is symmetric and bounded by 21:1") {
  const uint32_t black = pack(0, 0, 0), white = pack(255, 255, 255);
  CHECK(contrast_ratio(black, white) == doctest::Approx(21.0));
  CHECK(contrast_ratio(white, black) == doctest::Approx(21.0));
  CHECK(contrast_ratio(white, white) == doctest::Approx(1.0));
  const uint32_t c = pack(0, 114, 178);
  CHECK(contrast_ratio(c, white) == doctest::Approx(contrast_ratio(white, c)));
}

TEST_CASE("composite_over blends in sRGB and always returns an opaque colour") {
  const uint32_t fg = pack(200, 100, 50), bg = pack(0, 0, 0);
  CHECK(composite_over(fg, bg) == fg);                       // alpha 255 -> fg
  CHECK(composite_over(fg & 0x00FFFFFFu, bg) == bg);         // alpha 0   -> bg
  // Half alpha lands on the midpoint of the 8-bit values, rounded to nearest —
  // truncating here biased every dimmed node's reported contrast (see the
  // comment on the mix lambda).
  const uint32_t half = (fg & 0x00FFFFFFu) | (128u << 24);
  CHECK(composite_over(half, pack(0, 0, 0)) == pack(100, 50, 25));
  CHECK((composite_over(half, bg) >> 24) == 0xFFu);
}

TEST_CASE("black or white clears AA against every colour that can exist") {
  // The two curves max((L+.05)/.05, 1.05/(L+.05)) cross at L=0.1791 with value
  // 4.5826:1, so the guarantee is real but the headroom over 4.5 is only 0.08.
  // Grey 117 sits essentially on that crossover — it is the hardest colour there
  // is, and it still passes.
  CHECK(readable_label_contrast(pack(117, 117, 117), pack(0, 0, 0)) >= kAA);

  // Sweep the cube coarsely; every sample must clear AA with no palette help.
  double worst = 21.0;
  for (int r = 0; r < 256; r += 8)
    for (int g = 0; g < 256; g += 8)
      for (int b = 0; b < 256; b += 8) {
        const double c = readable_label_contrast(pack(r, g, b), pack(0, 0, 0));
        if (c < worst) worst = c;
      }
  CHECK(worst >= kAA);
  // Pinned tight on purpose: the floor really is ~4.58, not comfortably high, so
  // any future softening of kLabelBlack/kLabelWhite drops it straight under AA.
  CHECK(worst < 4.7);

  // And the choice is the max, not a coin flip.
  CHECK(readable_label_color(pack(240, 228, 66), kCanvasDark) == kLabelBlack);
  CHECK(readable_label_color(pack(0, 114, 178), kCanvasDark) == kLabelWhite);
}

TEST_CASE("the theme text colours #150 replaced could NOT clear AA") {
  // This is the bug, pinned. col_text was IM_COL32(235,238,242) on dark and
  // IM_COL32(24,28,36) on light, regardless of the header colour underneath.
  const uint32_t old_dark_text = pack(235, 238, 242);
  const uint32_t old_light_text = pack(24, 28, 36);
  // Okabe-Ito Yellow is the accessible palette's Pool header.
  CHECK(contrast_ratio(old_dark_text, pack(240, 228, 66)) < 1.5);
  // Okabe-Ito Blue (light variant) is the accessible palette's Conv header.
  CHECK(contrast_ratio(old_light_text, pack(0, 86, 138)) < kAA);
  // Softened text is not merely uglier than pure black/white, it is unsafe: the
  // guaranteed floor drops from 4.5826 to 3.81, which is BELOW AA. This is why
  // TextContrast.h insists on the extremes.
  CHECK(contrast_ratio(old_dark_text, pack(117, 117, 117)) < kAA);
  CHECK(contrast_ratio(old_light_text, pack(117, 117, 117)) < kAA);
}

TEST_CASE("every category label clears AA in both themes and both palettes") {
  for (int acc = 0; acc < 2; ++acc) {
    for (int d = 0; d < 2; ++d) {
      const bool dark = d != 0;
      const uint32_t canvas = dark ? kCanvasDark : kCanvasLight;
      const uint32_t body = dark ? kBodyDark : kBodyLight;
      const uint32_t body_hover = dark ? kBodyDarkHover : kBodyLightHover;
      for (int c = 0; c <= kLastCategory; ++c) {
        const OpCategory cat = static_cast<OpCategory>(c);
        const uint32_t fill = category_style(cat, dark, acc != 0).color;
        const std::string what = std::string(category_name(cat)) +
                                 (dark ? " dark" : " light") +
                                 (acc ? " accessible" : " default");
        CAPTURE(what);
        // Normal, hovered (the body brightens, the header does not) and
        // selected (an outline only) all draw the same header fill.
        CHECK(canvas_label_contrast(fill, body, canvas) >= kAA);
        CHECK(canvas_label_contrast(fill, body_hover, canvas) >= kAA);
        // Nav-dimmed: BOTH the header and the body get the alpha multiplier, so
        // the label lands on header-over-(body-over-canvas). Checking the raw
        // header here would pass while the on-screen pixels failed.
        CHECK(canvas_label_contrast(dimmed(fill), dimmed(body), canvas) >= kAA);
        CHECK(canvas_label_contrast(dimmed(fill), dimmed(body_hover), canvas) >= kAA);
      }
    }
  }
}

TEST_CASE("the diff and cost overlays are covered by the same guarantee") {
  // These override the header with colours that are NOT in either palette, so
  // the fix has to be a property of the maths rather than a palette audit. A
  // sample of saturated overlay-ish colours, dimmed and not.
  const uint32_t samples[] = {pack(220, 80, 80),   pack(80, 200, 120),
                              pack(255, 215, 0),   pack(30, 30, 30),
                              pack(250, 250, 250), pack(117, 117, 117)};
  for (const uint32_t s : samples) {
    CHECK(canvas_label_contrast(s, kBodyDark, kCanvasDark) >= kAA);
    CHECK(canvas_label_contrast(s, kBodyLight, kCanvasLight) >= kAA);
    CHECK(canvas_label_contrast(dimmed(s), dimmed(kBodyDark), kCanvasDark) >= kAA);
    CHECK(canvas_label_contrast(dimmed(s), dimmed(kBodyLight), kCanvasLight) >= kAA);
  }
}

TEST_CASE("the second label line still clears AA on the node body") {
  // The secondary line is drawn BELOW the header with col_text_muted, so #150
  // deliberately left it alone — but "left alone" has to be a measurement, not
  // an assumption, especially once dimming composites the body over the canvas.
  const uint32_t muted_dark = pack(160, 168, 180);
  const uint32_t muted_light = pack(90, 100, 116);
  CHECK(contrast_ratio(muted_dark, composite_over(kBodyDark, kCanvasDark)) >= kAA);
  CHECK(contrast_ratio(muted_dark, composite_over(kBodyDarkHover, kCanvasDark)) >= kAA);
  CHECK(contrast_ratio(muted_dark,
                       composite_over(dimmed(kBodyDark), kCanvasDark)) >= kAA);
  CHECK(contrast_ratio(muted_light, composite_over(kBodyLight, kCanvasLight)) >= kAA);
  CHECK(contrast_ratio(muted_light,
                       composite_over(kBodyLightHover, kCanvasLight)) >= kAA);
  CHECK(contrast_ratio(muted_light,
                       composite_over(dimmed(kBodyLight), kCanvasLight)) >= kAA);
}
