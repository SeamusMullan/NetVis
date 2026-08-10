// view/CategoryStyle.h — accessible op-category styling (#104).
//
// DECISION (v0.9.4): NetVis has FIFTEEN op categories, and colour alone cannot
// carry fifteen distinguishable classes for a colourblind viewer. Published
// colourblind-safe categorical palettes top out around eight to twelve entries
// for exactly this reason — beyond that, pairs collapse under deuteranopia or
// protanopia no matter how they are chosen. Shipping a "colourblind-safe
// 15-colour palette" would therefore be a claim the palette cannot honour, which
// is the kind of thing the honesty rules exist to prevent everywhere else in this
// codebase.
//
// So accessible mode does two things instead of one:
//   * a colourblind-safe palette for the categories a user actually meets most
//     (Conv, MatMul, Activation, Norm, Pool, Elementwise, Attention, Other), and
//   * a NON-COLOUR cue — a border style — that disambiguates the remainder.
// A viewer who cannot separate two hues can still separate a solid border from a
// dashed one, so the encoding degrades to something that still works rather than
// to something that merely looks like it does.
//
// The default palette is unchanged: this is opt-in, and the existing colours are
// tuned for the dark theme. Enabling accessible mode swaps both the colours and
// the border cue together — they are one encoding, not two independent toggles.
#pragma once

#include <cstdint>

#include "engine/OpCategory.h"
#include "imgui.h"

namespace netvis {

// The non-colour channel. Drawn as the node's border treatment, which is already
// a per-node stroke in GraphCanvas, so this adds no new draw call — it changes
// the parameters of one that already happens.
enum class CategoryPattern : uint8_t {
  Solid = 0,   // default; no extra cue
  Dashed,      // long dashes
  Dotted,      // short dots
  DoubleLine,  // a second inset stroke
};

struct CategoryStyle {
  ImU32 color = 0;
  CategoryPattern pattern = CategoryPattern::Solid;
};

// Style for `c`. `dark` selects the theme variant (the existing palette darkens
// by 0.72 for light mode; the accessible palette carries its own light values
// rather than being scaled, because scaling a safe palette does not preserve the
// contrast ratios that made it safe).
//
// `accessible` false reproduces the pre-#104 colours EXACTLY and always returns
// Solid — the default view must not change appearance.
CategoryStyle category_style(OpCategory c, bool dark, bool accessible);

// Human name for the pattern, for the legend. The legend must show the cue as
// well as the swatch, or the cue is undiscoverable.
const char* pattern_name(CategoryPattern p);

}  // namespace netvis
