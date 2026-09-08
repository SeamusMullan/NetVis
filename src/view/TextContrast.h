// SPDX-License-Identifier: Apache-2.0
// view/TextContrast.h — WCAG contrast maths for text drawn on top of a fill (#150).
//
// THE FAILURE (#150): GraphCanvas draws a node's op_type INSIDE the coloured
// header strip, but picked the glyph colour from the THEME — near-white on dark,
// near-black on light — regardless of what colour that strip actually is. The
// category palette spans the whole luminance range (Okabe-Ito Yellow #F0E442 sits
// at L=0.72; Blue #0072B2 at L=0.13), so exactly one end of the palette read
// correctly in each theme and the other end was near-invisible: white on yellow
// is 1.4:1, black on blue is 3.6:1. The label colour has to come from the fill it
// lands on, not from the theme.
//
// WHY BLACK OR WHITE AND NOTHING ELSE. For a fill of relative luminance L the
// best achievable ratio is max((L+0.05)/0.05, 1.05/(L+0.05)), and the two curves
// cross at L=0.1791 with a value of 4.5826:1. So pure black/white text clears the
// WCAG AA threshold of 4.5:1 against EVERY colour that can exist — the palette
// never has to change to satisfy AA, which is what keeps the accessibility
// argument of #104 intact. That headroom is only 0.08 wide, though: substituting
// the theme's softened text colours (235,238,242) and (24,28,36) drops the
// guarantee to 3.81:1 and puts mid-luminance fills back under AA. The extremes
// are load-bearing, not a stylistic preference.
//
// Lives in netvis_core (see the list above add_library in CMakeLists.txt) and
// carries no ImGui dependency, so netvis_tests can link it and actually assert
// the ratios. Colours are plain uint32_t in IM_COL32 byte order (0xAABBGGRR),
// interchangeable with ImU32 at every call site, for the same reason
// CategoryStyle::color is.
#pragma once

#include <cstdint>

namespace netvis {

// Relative luminance per WCAG 2.x, from a packed colour's RGB. Alpha is IGNORED
// rather than premultiplied: a translucent colour has no luminance of its own,
// only the composite does, so callers must run composite_over() first. Range
// [0,1]; sRGB in, linear out.
double relative_luminance(uint32_t rgba);

// WCAG 2.x contrast ratio between two OPAQUE colours, in [1.0, 21.0]. Symmetric.
double contrast_ratio(uint32_t a, uint32_t b);

// Source-over blend of `fg` (using its own alpha) onto `bg`, returning an opaque
// colour. Done on the 8-bit sRGB values with no linearization because that is
// what the GPU actually does with these draw commands — computing the "correct"
// linear blend here would report a contrast the user never sees.
uint32_t composite_over(uint32_t fg, uint32_t bg);

// The only two label colours. See the header comment for why nothing softer.
constexpr uint32_t kLabelBlack = 0xFF000000u;  // IM_COL32(0,0,0,255)
constexpr uint32_t kLabelWhite = 0xFFFFFFFFu;  // IM_COL32(255,255,255,255)

// The label colour for text drawn on `fill`, which may be translucent (a
// nav-dimmed node header) and is therefore composited over `background` first —
// the dim multiplier changes the effective fill, so a colour chosen against the
// unblended header can be the wrong one on a dimmed node.
uint32_t readable_label_color(uint32_t fill, uint32_t background);

// The ratio readable_label_color() achieves for the same arguments. Exists so
// tests can assert the number rather than re-deriving it, and so a future caller
// can decide to fall back (e.g. draw a plate behind the text) when the fill is
// one AA cannot reach — nothing needs that today, because pure black/white
// always reaches it.
double readable_label_contrast(uint32_t fill, uint32_t background);

}  // namespace netvis
