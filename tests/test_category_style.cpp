// tests/test_category_style.cpp — the accessible category palette (#104).
//
// BUILD SITUATION — read this before assuming the file is dead weight.
//
// `netvis_tests` links `netvis_core` + doctest and nothing else, and
// `netvis_core` globs src/core, src/ir, src/parsers, src/engine — NOT src/view.
// `src/view/CategoryStyle.cpp` is therefore compiled into the GUI target only, so
// a test that calls category_style() does not link. Worse, it does not even
// COMPILE: view/CategoryStyle.h includes "imgui.h", and imgui's include
// directory reaches only the `imgui` target, which the test executable does not
// link. Neither wall can be climbed from inside tests/ — the unreachable include
// lives in a frozen header, not here — so the tests below are gated on
// __has_include("imgui.h"). That probe is the honest one to use because the
// single CMake edit that makes imgui.h reachable to netvis_tests is the same edit
// that must add CategoryStyle.cpp to its sources; if only half of it lands, the
// build fails loudly at link rather than quietly passing.
//
// The fix belongs to whoever owns CMakeLists.txt (see the report on #104): give
// netvis_tests the imgui *headers* and compile src/view/CategoryStyle.cpp
// directly into the test executable. Deliberately NOT into netvis_core — core is
// GUI-free by design (spec §9) and must stay that way; CategoryStyle.cpp needs
// only the ImU32 typedef and the IM_COL32 macro, both header-only, so nothing is
// linked and no layering rule is bent.
//
// Until then the compile-time static_asserts in src/view/CategoryStyle.cpp carry
// the disambiguation invariant on their own. They are exhaustive over the same
// pairs the test below walks, so the property is guarded on every build; what is
// lost while this file is inert is the pinning of concrete colour values.

#include <doctest/doctest.h>

#if __has_include("imgui.h")

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "engine/OpCategory.h"
#include "view/CategoryStyle.h"

using namespace netvis;

namespace {

// Every enumerator, in order. OpCategory.h guarantees Other is last, and
// CategoryStyle.cpp's tables are indexed by the raw value, so this loop bound is
// the same one the implementation trusts — if a category is ever appended after
// Other, both break together rather than one silently drifting.
constexpr int kLast = static_cast<int>(OpCategory::Other);

std::vector<OpCategory> all_categories() {
  std::vector<OpCategory> v;
  for (int i = 0; i <= kLast; ++i) v.push_back(static_cast<OpCategory>(i));
  return v;
}

// The legacy light rule, reproduced here so the "explicit light table" test can
// state what it is NOT. Uses the same float multiply-and-truncate as the
// pre-#104 code, because a rounding difference would make the comparison
// meaningless.
ImU32 scaled_by_072(ImU32 c) {
  const unsigned r = (c >> IM_COL32_R_SHIFT) & 0xFFu;
  const unsigned g = (c >> IM_COL32_G_SHIFT) & 0xFFu;
  const unsigned b = (c >> IM_COL32_B_SHIFT) & 0xFFu;
  return IM_COL32(static_cast<unsigned>(r * 0.72f),
                  static_cast<unsigned>(g * 0.72f),
                  static_cast<unsigned>(b * 0.72f), 255);
}

}  // namespace

// --- The property the whole feature rests on ---------------------------------
TEST_CASE("accessible mode: no two categories share both colour and pattern") {
  // Exhaustive over the enum, not a sample: the encoding claims that (colour,
  // pattern) identifies a category, and a claim that holds for "most" pairs is
  // exactly the kind of approximate-honesty this codebase refuses elsewhere.
  // Checked in BOTH themes because the light table is hand-written and could
  // collapse two hues that the dark table keeps apart.
  for (bool dark : {true, false}) {
    for (OpCategory a : all_categories()) {
      for (OpCategory b : all_categories()) {
        if (a == b) continue;
        const CategoryStyle sa = category_style(a, dark, true);
        const CategoryStyle sb = category_style(b, dark, true);
        const bool same = sa.color == sb.color && sa.pattern == sb.pattern;
        INFO("dark=" << dark << " a=" << category_name(a)
                     << " b=" << category_name(b));
        CHECK_FALSE(same);
      }
    }
  }
}

TEST_CASE("accessible mode: hue reuse is real, and bounded to the safe set") {
  // The design says eight colourblind-safe hues cover fifteen categories, with
  // the overflow separated by pattern. Assert both halves of that sentence: the
  // colour count is exactly the size of the published palette (so nobody
  // "fixed" the crowding by inventing a ninth hue), and hues genuinely repeat
  // (so the pattern channel is load-bearing rather than decorative).
  std::set<ImU32> colors;
  for (OpCategory c : all_categories())
    colors.insert(category_style(c, true, true).color);
  CHECK(colors.size() == 8);
  CHECK(colors.size() < all_categories().size());

  std::set<ImU32> light;
  for (OpCategory c : all_categories())
    light.insert(category_style(c, false, true).color);
  CHECK(light.size() == 8);
}

TEST_CASE("accessible mode: the pattern cue does not change with the theme") {
  // The legend prints one cue name per category. If the pattern flipped between
  // themes the legend would be wrong in one of them, so pin it.
  for (OpCategory c : all_categories())
    CHECK(category_style(c, true, true).pattern ==
          category_style(c, false, true).pattern);
}

// --- Requirement 1: the default view must not change -------------------------
TEST_CASE("default mode reproduces the pre-#104 colours exactly") {
  // Packed literals, not IM_COL32(r,g,b,255) round-trips: spelling the expected
  // value the same way the implementation does would pass even if the palette
  // table were rewritten. These are the bytes App::category_color emitted at
  // HEAD, in ImGui's default RGBA packing (R at bit 0 .. A at bit 24).
  CHECK(category_style(OpCategory::Conv, true, false).color == 0xFFF78F4Fu);
  CHECK(category_style(OpCategory::MatMul, true, false).color == 0xFFF66E8Au);
  CHECK(category_style(OpCategory::Attention, true, false).color ==
        0xFFD064D8u);
  CHECK(category_style(OpCategory::Other, true, false).color == 0xFF888080u);

  // Light = the legacy 0.72 truncating scale, byte for byte.
  //   Conv (79,143,247) * 0.72 -> (56,102,177)
  CHECK(category_style(OpCategory::Conv, false, false).color == 0xFFB16638u);
  CHECK(category_style(OpCategory::Other, false, false).color == 0xFF615C5Cu);

  // And every category, both themes, follows the same rule — a spot check on
  // four rows would not catch a single transposed row further down the table.
  for (OpCategory c : all_categories()) {
    INFO("category=" << category_name(c));
    CHECK(category_style(c, false, false).color ==
          scaled_by_072(category_style(c, true, false).color));
  }
}

TEST_CASE("default mode never emits a pattern cue") {
  // Opt-in means opt-in: a user who never enables accessible mode must see the
  // exact borders they saw before #104.
  for (OpCategory c : all_categories()) {
    INFO("category=" << category_name(c));
    CHECK(category_style(c, true, false).pattern == CategoryPattern::Solid);
    CHECK(category_style(c, false, false).pattern == CategoryPattern::Solid);
  }
}

// --- Requirement 4: light accessible values are explicit, not scaled ----------
TEST_CASE("accessible light table is explicit, not the dark table * 0.72") {
  // Pinned so the assertion cannot be satisfied by a table that happens to be
  // "different somewhere". Conv is Okabe-Ito Blue #0072B2 on dark; the 0.72
  // scale of that is (0,82,128), and the shipped light value is (0,86,138) —
  // chosen to hold contrast against a light canvas rather than to be dimmer.
  const ImU32 conv_dark = category_style(OpCategory::Conv, true, true).color;
  const ImU32 conv_light = category_style(OpCategory::Conv, false, true).color;
  CHECK(conv_dark == 0xFFB27200u);   // (0,114,178)
  CHECK(conv_light == 0xFF8A5600u);  // (0,86,138)
  CHECK(conv_light != scaled_by_072(conv_dark));

  // Pool is Okabe-Ito Yellow #F0E442, the entry where a uniform scale fails
  // hardest: (172,164,47) is still a pale wash on white, whereas the explicit
  // (176,155,30) is a readable gold that keeps Yellow above Orange in luminance.
  const ImU32 pool_dark = category_style(OpCategory::Pool, true, true).color;
  const ImU32 pool_light = category_style(OpCategory::Pool, false, true).color;
  CHECK(pool_dark == 0xFF42E4F0u);   // (240,228,66)
  CHECK(pool_light == 0xFF1E9BB0u);  // (176,155,30)
  CHECK(pool_light != scaled_by_072(pool_dark));

  // No accessible row may coincide with the scaled value, or the explicit table
  // is not actually being consulted for that row.
  for (OpCategory c : all_categories()) {
    INFO("category=" << category_name(c));
    CHECK(category_style(c, false, true).color !=
          scaled_by_072(category_style(c, true, true).color));
  }
}

TEST_CASE("themes differ for every category, in both modes") {
  for (OpCategory c : all_categories()) {
    INFO("category=" << category_name(c));
    CHECK(category_style(c, true, false).color !=
          category_style(c, false, false).color);
    CHECK(category_style(c, true, true).color !=
          category_style(c, false, true).color);
  }
}

TEST_CASE("accessible mode is a different encoding, not a re-tint") {
  // If accessible mode returned a legacy colour for a category, a user toggling
  // it would be told the palette changed when it had not. Every row must move.
  for (OpCategory c : all_categories()) {
    INFO("category=" << category_name(c));
    CHECK(category_style(c, true, true).color !=
          category_style(c, true, false).color);
  }
}

// --- Requirement 6: out-of-range clamps to Other ------------------------------
TEST_CASE("out-of-range OpCategory clamps to Other") {
  // A plugin manifest, a corrupt cast, or a category added upstream without a
  // palette row must render as "unclassified" rather than index off the table.
  // 200 and 255 both sit past the end of a uint8_t-backed enum's valid range.
  for (uint8_t raw : {static_cast<uint8_t>(kLast + 1), uint8_t{200},
                      uint8_t{255}}) {
    const OpCategory bad = static_cast<OpCategory>(raw);
    for (bool dark : {true, false}) {
      for (bool acc : {true, false}) {
        INFO("raw=" << int(raw) << " dark=" << dark << " accessible=" << acc);
        const CategoryStyle got = category_style(bad, dark, acc);
        const CategoryStyle ref =
            category_style(OpCategory::Other, dark, acc);
        CHECK(got.color == ref.color);
        CHECK(got.pattern == ref.pattern);
      }
    }
  }
}

// --- Requirement 5: the legend always has something to print ------------------
TEST_CASE("pattern_name is never null and never empty") {
  for (CategoryPattern p :
       {CategoryPattern::Solid, CategoryPattern::Dashed,
        CategoryPattern::Dotted, CategoryPattern::DoubleLine}) {
    const char* n = pattern_name(p);
    REQUIRE(n != nullptr);
    CHECK(std::string(n).size() > 0);
  }

  // Distinct labels, or the legend cannot tell two cues apart in words either —
  // which is the fallback for a viewer who also cannot resolve the stroke.
  std::set<std::string> names;
  for (CategoryPattern p :
       {CategoryPattern::Solid, CategoryPattern::Dashed,
        CategoryPattern::Dotted, CategoryPattern::DoubleLine})
    names.insert(pattern_name(p));
  CHECK(names.size() == 4);

  // Every pattern the palette actually emits has a label, and an out-of-range
  // value still yields a printable string rather than a crash or an empty cell.
  for (OpCategory c : all_categories()) {
    const char* n = pattern_name(category_style(c, true, true).pattern);
    REQUIRE(n != nullptr);
    CHECK(std::string(n).size() > 0);
  }
  const char* fallback = pattern_name(static_cast<CategoryPattern>(200));
  REQUIRE(fallback != nullptr);
  CHECK(std::string(fallback).size() > 0);
}

#else  // !__has_include("imgui.h")

// Reported by doctest as SKIPPED, not passed. That distinction is the point: a
// green run must not imply #104 was verified when the translation unit under
// test was never linked in. See the header comment for the one-edit fix.
TEST_CASE(
    "#104 category palette NOT VERIFIED — netvis_tests cannot reach imgui.h or "
    "src/view/CategoryStyle.cpp (see file header for the CMake fix)" *
    doctest::skip()) {
  FAIL("unreachable: this case is decorated skip()");
}

#endif  // __has_include("imgui.h")
