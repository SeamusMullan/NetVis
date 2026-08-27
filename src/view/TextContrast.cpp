// view/TextContrast.cpp — WCAG contrast maths (#150). Rationale is in the header.
#include "view/TextContrast.h"

#include <cmath>

namespace netvis {
namespace {

// IM_COL32 byte order is 0xAABBGGRR, so red is the LOW byte. Unpacked by hand
// rather than via ImGui's helpers to keep this translation unit GUI-free and
// therefore testable.
constexpr uint8_t chan(uint32_t c, unsigned shift) {
  return static_cast<uint8_t>((c >> shift) & 0xFFu);
}

// The sRGB electro-optical transfer function from WCAG 2.x (relative luminance).
// The 0.03928 knee and the 2.4 exponent are the published constants; they are
// written out rather than approximated because a "close enough" curve shifts
// ratios near the 4.5:1 boundary by more than the margin we have to spend.
double linearize(uint8_t v) {
  const double s = static_cast<double>(v) / 255.0;
  return s <= 0.03928 ? s / 12.92 : std::pow((s + 0.055) / 1.055, 2.4);
}

}  // namespace

double relative_luminance(uint32_t rgba) {
  return 0.2126 * linearize(chan(rgba, 0)) +
         0.7152 * linearize(chan(rgba, 8)) +
         0.0722 * linearize(chan(rgba, 16));
}

double contrast_ratio(uint32_t a, uint32_t b) {
  const double la = relative_luminance(a);
  const double lb = relative_luminance(b);
  const double hi = la > lb ? la : lb;
  const double lo = la > lb ? lb : la;
  return (hi + 0.05) / (lo + 0.05);
}

uint32_t composite_over(uint32_t fg, uint32_t bg) {
  const double af = static_cast<double>(chan(fg, 24)) / 255.0;
  // Round to nearest rather than truncating: truncation biases every channel
  // down, which quietly darkens the composite and reports a contrast that is off
  // by a whole step for the low-alpha dimmed case this function exists to serve.
  auto mix = [af](uint8_t f, uint8_t b) -> uint32_t {
    const double v =
        static_cast<double>(f) * af + static_cast<double>(b) * (1.0 - af);
    const double r = v < 0.0 ? 0.0 : (v > 255.0 ? 255.0 : v);
    return static_cast<uint32_t>(r + 0.5);
  };
  return 0xFF000000u | (mix(chan(fg, 16), chan(bg, 16)) << 16) |
         (mix(chan(fg, 8), chan(bg, 8)) << 8) | mix(chan(fg, 0), chan(bg, 0));
}

uint32_t readable_label_color(uint32_t fill, uint32_t background) {
  const uint32_t eff = composite_over(fill, background);
  return contrast_ratio(kLabelBlack, eff) >= contrast_ratio(kLabelWhite, eff)
             ? kLabelBlack
             : kLabelWhite;
}

double readable_label_contrast(uint32_t fill, uint32_t background) {
  const uint32_t eff = composite_over(fill, background);
  return contrast_ratio(readable_label_color(fill, background), eff);
}

}  // namespace netvis
