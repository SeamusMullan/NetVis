// view/CategoryStyle.cpp — op-category colour + non-colour cue (#104).
//
// The design argument lives in CategoryStyle.h; this file is the data plus the
// two lookups. Read the header first — the "fifteen categories cannot be carried
// by colour alone" reasoning is what shapes every table below.
//
// PALETTE PROVENANCE (#104). The accessible hues are the Okabe-Ito Color
// Universal Design set (Okabe & Ito, 2008, "Color Universal Design — how to make
// figures and presentations that are friendly to colorblind people",
// https://jfly.uni-koeln.de/color/). It is eight colours, chosen so that every
// pair stays separable under protanopia, deuteranopia and tritanopia. We use
// seven of its eight plus its neutral grey; CUD black is dropped because a black
// node border is invisible against NetVis' dark theme, which is the theme the
// palette is selected in most often.
//
// Eight safe hues cannot cover fifteen categories, so the eight go to the
// categories a user meets most (Conv, MatMul, Activation, Norm, Pool,
// Elementwise, Attention, Other) as SOLID borders, and the remaining seven reuse
// a hue with a different border pattern. That pairing is the whole feature: the
// (hue, pattern) product is what is unique, never the hue on its own. A
// static_assert at the bottom of the anonymous namespace enforces it at compile
// time, so the invariant holds even where the unit tests cannot be linked (see
// tests/test_category_style.cpp for why that caveat exists).
#include "view/CategoryStyle.h"

#include <cstddef>

namespace netvis {
namespace {

struct Rgb {
  uint8_t r, g, b;
};

// --- Legacy palette (pre-#104) ----------------------------------------------
// Copied VERBATIM from App::category_color, including the comments that explain
// the index alignment, because requirement 1 of #104 is that the default view
// does not change by a single byte. Any drift here is a regression, not a
// refresh — the accessible palette is where new colour decisions belong.
//
// Dark-first palette: distinct, moderately saturated hues so adjacent node
// categories read apart at a glance. Indexed by OpCategory order.
constexpr Rgb kLegacy[] = {
    {/*Conv*/ 79, 143, 247},       {/*MatMul*/ 138, 110, 246},
    {/*Activation*/ 76, 201, 176}, {/*Norm*/ 232, 168, 56},
    {/*Pool*/ 90, 179, 90},        {/*Elementwise*/ 224, 108, 118},
    {/*Shape*/ 120, 130, 148},     {/*Reduce*/ 210, 120, 200},
    {/*Tensor*/ 150, 160, 90},     {/*ControlFlow*/ 200, 90, 130},
    {/*IO*/ 96, 172, 214},
    // v0.4.0 categories — MUST stay index-aligned with the OpCategory enum,
    // inserted before Other (see OpCategory.h). Distinct hues from the above.
    {/*Attention*/ 216, 100, 208}, {/*Recurrent*/ 96, 190, 150},
    {/*Quantize*/ 214, 178, 72},
    {/*Other*/ 128, 128, 136},
};

// --- Accessible palette ------------------------------------------------------
// A hue SLOT rather than a raw colour, so the light and dark tables can hold
// different RGB while a category keeps its identity across themes. Swapping the
// theme must not renumber the encoding — a user who learned "Conv is the blue
// one" keeps that in both themes.
enum class Hue : uint8_t {
  Blue = 0,
  Sky,
  Green,
  Yellow,
  Orange,
  Vermillion,
  Purple,
  Grey,
  Count,
};

// Okabe-Ito as published, minus black, plus the CUD grey. These are the values
// the safety claim rests on, so they are transcribed exactly and never computed.
constexpr Rgb kAccessibleDark[] = {
    {/*Blue       #0072B2*/ 0, 114, 178},
    {/*Sky        #56B4E9*/ 86, 180, 233},
    {/*Green      #009E73*/ 0, 158, 115},
    {/*Yellow     #F0E442*/ 240, 228, 66},
    {/*Orange     #E69F00*/ 230, 159, 0},
    {/*Vermillion #D55E00*/ 213, 94, 0},
    {/*Purple     #CC79A7*/ 204, 121, 167},
    {/*Grey       #999999*/ 153, 153, 153},
};

// Explicit light values — NOT the dark ones scaled (requirement 4 of #104).
// Scaling every channel by a constant is what the legacy palette does, and it is
// wrong here for two separate reasons: it drops the whole set's luminance
// together, so a colour that was readable on dark is merely dimmer rather than
// darker-against-light; and Okabe-Ito separates same-family pairs (Blue vs Sky)
// by LUMINANCE, not hue, so a uniform scale preserves neither the contrast
// against the background nor the ratio between the pair that made them
// separable. These values keep each slot's hue and keep the set's internal
// luminance ordering (Sky stays clearly lighter than Blue; Yellow stays clearly
// lighter than Orange) while pulling the pale entries — Yellow, Sky, Grey — down
// far enough to hold their own against a light canvas.
constexpr Rgb kAccessibleLight[] = {
    {/*Blue*/ 0, 86, 138},     {/*Sky*/ 42, 138, 190},
    {/*Green*/ 0, 115, 84},    {/*Yellow*/ 176, 155, 30},
    {/*Orange*/ 176, 110, 0},  {/*Vermillion*/ 163, 66, 0},
    {/*Purple*/ 158, 73, 122}, {/*Grey*/ 95, 95, 102},
};

struct Accessible {
  Hue hue;
  CategoryPattern pattern;
};

// Index-aligned with OpCategory, exactly like kLegacy.
//
// The eight SOLID rows are the eight distinct hues; every other row reuses one
// of them and is told apart by its border pattern. Hue reuse is deliberate and
// semantic — a category shares a hue with the one it is most conceptually
// adjacent to, so a viewer who cannot resolve the pattern at low zoom still
// lands in the right family rather than on an unrelated op:
//
//   Blue       Conv (solid)
//   Sky        Attention (solid)   | IO (dotted)          — both "edges of the
//                                                            graph": entry
//                                                            points and the
//                                                            block that
//                                                            consumes them
//   Green      Activation (solid)  | Recurrent (dashed)    — per-step nonlinear
//   Yellow     Pool (solid)
//   Orange     Norm (solid)        | Quantize (doubleline) — both rescale a
//                                                            tensor in place
//   Vermillion Elementwise (solid) | ControlFlow (dashed)
//   Purple     MatMul (solid)      | Reduce (dotted)       — both contract an
//                                                            axis away
//   Grey       Other (solid)       | Shape (dashed)
//                                  | Tensor (doubleline)   — plumbing: no
//                                                            arithmetic, no
//                                                            hue worth spending
//
// Pattern choice inside a hue is picked for separability, not for tidiness:
// where a hue carries three rows (Grey) the pair most likely to sit side by side
// in a graph gets Dashed vs DoubleLine rather than Dashed vs Dotted, because
// dash-versus-dot is the pattern pair that degrades first as the node shrinks.
constexpr Accessible kAccessible[] = {
    {/*Conv*/ Hue::Blue, CategoryPattern::Solid},
    {/*MatMul*/ Hue::Purple, CategoryPattern::Solid},
    {/*Activation*/ Hue::Green, CategoryPattern::Solid},
    {/*Norm*/ Hue::Orange, CategoryPattern::Solid},
    {/*Pool*/ Hue::Yellow, CategoryPattern::Solid},
    {/*Elementwise*/ Hue::Vermillion, CategoryPattern::Solid},
    {/*Shape*/ Hue::Grey, CategoryPattern::Dashed},
    {/*Reduce*/ Hue::Purple, CategoryPattern::Dotted},
    {/*Tensor*/ Hue::Grey, CategoryPattern::DoubleLine},
    {/*ControlFlow*/ Hue::Vermillion, CategoryPattern::Dashed},
    {/*IO*/ Hue::Sky, CategoryPattern::Dotted},
    {/*Attention*/ Hue::Sky, CategoryPattern::Solid},
    {/*Recurrent*/ Hue::Green, CategoryPattern::Dashed},
    {/*Quantize*/ Hue::Orange, CategoryPattern::DoubleLine},
    {/*Other*/ Hue::Grey, CategoryPattern::Solid},
};

constexpr size_t kCategoryCount = sizeof(kLegacy) / sizeof(kLegacy[0]);
constexpr size_t kHueCount = static_cast<size_t>(Hue::Count);

// Every table is indexed by the raw enum value, so a category added to
// OpCategory without a palette row would silently read the wrong colour (or, for
// the last row, clamp to Other and look like a classification bug). Catch it at
// compile time instead — the same reason OpCategory.h insists Other stays last.
static_assert(kCategoryCount == sizeof(kAccessible) / sizeof(kAccessible[0]),
              "accessible palette must have one row per OpCategory");
static_assert(kCategoryCount == static_cast<size_t>(OpCategory::Other) + 1,
              "palette tables must cover every OpCategory enumerator");
static_assert(kHueCount == sizeof(kAccessibleDark) / sizeof(kAccessibleDark[0]),
              "dark hue table must cover every Hue");
static_assert(kHueCount ==
                  sizeof(kAccessibleLight) / sizeof(kAccessibleLight[0]),
              "light hue table must cover every Hue");

// --- The invariant the feature rests on --------------------------------------
// Two categories may share a hue OR a pattern, never both. Proving it here as
// well as in the unit tests is not belt-and-braces: tests/test_category_style.cpp
// cannot currently link (this translation unit is not in netvis_core), so this
// static_assert is the only gate that runs on every build today. It is cheap and
// exhaustive, so it stays even once the tests do link.
constexpr bool hues_distinct(const Rgb* t) {
  for (size_t i = 0; i < kHueCount; ++i)
    for (size_t j = i + 1; j < kHueCount; ++j)
      if (t[i].r == t[j].r && t[i].g == t[j].g && t[i].b == t[j].b)
        return false;
  return true;
}

constexpr bool encoding_distinct() {
  for (size_t i = 0; i < kCategoryCount; ++i)
    for (size_t j = i + 1; j < kCategoryCount; ++j)
      if (kAccessible[i].hue == kAccessible[j].hue &&
          kAccessible[i].pattern == kAccessible[j].pattern)
        return false;
  return true;
}

// Distinct hue SLOTS only buy distinct colours if the slot->RGB maps are
// injective, so both halves are asserted; together they are the (colour,
// pattern) uniqueness the legend promises, in both themes.
static_assert(hues_distinct(kAccessibleDark), "dark hues must be distinct");
static_assert(hues_distinct(kAccessibleLight), "light hues must be distinct");
static_assert(encoding_distinct(),
              "two categories share BOTH hue and pattern — the accessible "
              "encoding cannot disambiguate them (#104)");

// Out-of-range clamps to Other, matching the pre-#104 App::category_color
// behaviour exactly: a category from a plugin manifest or a corrupt cast must
// render as "unclassified", never index off the end of the table.
size_t palette_index(OpCategory c) {
  const size_t idx = static_cast<size_t>(c);
  return idx < kCategoryCount ? idx : static_cast<size_t>(OpCategory::Other);
}

}  // namespace

CategoryStyle category_style(OpCategory c, bool dark, bool accessible) {
  const size_t idx = palette_index(c);
  CategoryStyle out;

  if (accessible) {
    const Accessible& a = kAccessible[idx];
    const Rgb p = (dark ? kAccessibleDark : kAccessibleLight)
                      [static_cast<size_t>(a.hue)];
    out.color = IM_COL32(p.r, p.g, p.b, 255);
    out.pattern = a.pattern;
    return out;
  }

  Rgb p = kLegacy[idx];
  if (!dark) {
    // On a light theme, darken the hue so text/edges keep contrast. Kept as the
    // original float multiply-and-truncate so every legacy colour round-trips
    // bit-identically; a "cleaner" rounding would shift half the palette by one.
    p.r = static_cast<uint8_t>(p.r * 0.72f);
    p.g = static_cast<uint8_t>(p.g * 0.72f);
    p.b = static_cast<uint8_t>(p.b * 0.72f);
  }
  out.color = IM_COL32(p.r, p.g, p.b, 255);
  out.pattern = CategoryPattern::Solid;  // default view gains no new cue.
  return out;
}

const char* pattern_name(CategoryPattern p) {
  switch (p) {
    case CategoryPattern::Solid: return "solid";
    case CategoryPattern::Dashed: return "dashed";
    case CategoryPattern::Dotted: return "dotted";
    case CategoryPattern::DoubleLine: return "double";
  }
  // Unreachable for a valid enumerator, but a legend that renders an empty
  // string is worse than one that renders the default cue's name, and the
  // contract says never null and never empty.
  return "solid";
}

}  // namespace netvis
