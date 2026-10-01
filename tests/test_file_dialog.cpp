// SPDX-License-Identifier: Apache-2.0
// tests/test_file_dialog.cpp — the pure helpers behind the native chooser: path
// handling, and the shared list of extensions the Open dialogs offer.
//
// BUILD SITUATION: src/view/FileDialog.cpp is GUI/platform code (fork/exec on
// Linux, tinyfiledialogs elsewhere) compiled into the `netvis` target only, and
// netvis_tests links netvis_core alone. The helpers below are therefore
// `inline` in view/FileDialog.h, which pulls in nothing but the standard library
// and the parser header (both available to netvis_core) — so this file includes
// the header and links nothing from the GUI.

#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"
#include "parsers/Parser.h"
#include "view/FileDialog.h"

using netvis::detail::apply_default_extension;
using netvis::detail::basename_of;
using netvis::detail::case_folded_glob;
using netvis::detail::first_line;
using netvis::detail::helper_globs;
using netvis::detail::kdialog_filter;
using netvis::detail::strip_trailing_separators;
using netvis::detail::suffix_of;
using netvis::kDialogOnlyExtensions;
using netvis::kExtensionFormats;
using netvis::openable_extensions;
using netvis::openable_patterns;

TEST_CASE("first_line keeps a single clean path unchanged") {
  CHECK(first_line("/home/u/model.onnx") == "/home/u/model.onnx");
  CHECK(first_line("") == "");
}

TEST_CASE("first_line drops everything after the first newline") {
  // A chooser prints one path per line; we never ask for multi-select, but a
  // helper that ignores --separate-output would still hand back several.
  CHECK(first_line("/a/one.onnx\n/a/two.onnx\n") == "/a/one.onnx");
  CHECK(first_line("/a/one.onnx\n") == "/a/one.onnx");
}

TEST_CASE("first_line strips trailing CR and spaces") {
  CHECK(first_line("/a/m.onnx\r\n") == "/a/m.onnx");
  CHECK(first_line("/a/m.onnx  ") == "/a/m.onnx");
  CHECK(first_line("/a/m.onnx \r") == "/a/m.onnx");
  // Interior spaces are part of the name and must survive.
  CHECK(first_line("/a/my model.onnx") == "/a/my model.onnx");
}

TEST_CASE("suffix_of accepts only simple suffix globs") {
  CHECK(suffix_of("*.png") == ".png");
  CHECK(suffix_of("*.netvis-view") == ".netvis-view");
  CHECK(suffix_of("*") == "");
  CHECK(suffix_of("*.") == "");
  CHECK(suffix_of("model.png") == "");
  // A second wildcard past the dot means the pattern is not a plain extension.
  CHECK(suffix_of("*.t*t") == "");
  CHECK(suffix_of("*.[ch]") == "");
}

TEST_CASE("apply_default_extension fills in a bare name") {
  CHECK(apply_default_extension("diff", {"*.md"}) == "diff.md");
  CHECK(apply_default_extension("/home/u/diff", {"*.tsv"}) == "/home/u/diff.tsv");
}

TEST_CASE("apply_default_extension leaves an existing extension alone") {
  CHECK(apply_default_extension("diff.md", {"*.md"}) == "diff.md");
  // Any extension counts, even a mismatched one — the user typed it on purpose.
  CHECK(apply_default_extension("diff.txt", {"*.md"}) == "diff.txt");
}

TEST_CASE("apply_default_extension only inspects the file name") {
  // A dot in a parent directory must not read as "already has an extension".
  CHECK(apply_default_extension("/home/u/v1.2/diff", {"*.md"}) ==
        "/home/u/v1.2/diff.md");
  CHECK(apply_default_extension("/home/u/v1.2/diff.md", {"*.md"}) ==
        "/home/u/v1.2/diff.md");
}

TEST_CASE("apply_default_extension is a no-op without a usable pattern") {
  CHECK(apply_default_extension("diff", {}) == "diff");
  CHECK(apply_default_extension("", {"*.md"}) == "");
  // Not a plain suffix glob: append nothing rather than a nonsense extension.
  CHECK(apply_default_extension("diff", {"*"}) == "diff");
}

TEST_CASE("apply_default_extension uses the first pattern") {
  CHECK(apply_default_extension("out", {"*.npy", "*.bin"}) == "out.npy");
}


// --- picked-path normalisation (#135 review) ----------------------------------
// macOS hands a package back as "/a/M.mlpackage/". Left alone, the tab title and
// the diff ladder's file column (both "text after the last separator") came out
// empty, and the Recent list kept the slashed and unslashed spellings apart.

TEST_CASE("strip_trailing_separators drops the slash a macOS package comes back with") {
  CHECK(strip_trailing_separators("/a/M.mlpackage/") == "/a/M.mlpackage");
  CHECK(strip_trailing_separators("/a/M.mlpackage//") == "/a/M.mlpackage");
  CHECK(strip_trailing_separators("C:\\m\\M.mlpackage\\") == "C:\\m\\M.mlpackage");
}

TEST_CASE("strip_trailing_separators leaves a clean path and a bare root alone") {
  CHECK(strip_trailing_separators("/a/model.onnx") == "/a/model.onnx");
  CHECK(strip_trailing_separators("model.onnx") == "model.onnx");
  CHECK(strip_trailing_separators("") == "");
  CHECK(strip_trailing_separators("/") == "/");
  CHECK(strip_trailing_separators("///") == "/");
  CHECK(strip_trailing_separators("C:\\") == "C:\\");
  CHECK(strip_trailing_separators("C:/") == "C:/");
}

TEST_CASE("basename_of names a package even with a trailing separator") {
  // This is the exact input that produced a blank tab title and "(none)".
  CHECK(basename_of("/a/M.mlpackage/") == "M.mlpackage");
  CHECK(basename_of("/a/M.mlpackage") == "M.mlpackage");
  CHECK(basename_of("C:\\m\\M.mlpackage\\") == "M.mlpackage");
  CHECK(basename_of("/a/b/model.onnx") == "model.onnx");
  CHECK(basename_of("model.onnx") == "model.onnx");
  CHECK(basename_of("model.onnx/") == "model.onnx");
}

TEST_CASE("basename_of of nothing is empty, never out of range") {
  CHECK(basename_of("").empty());
  CHECK(basename_of("/").empty());
  CHECK(basename_of("///").empty());
}

TEST_CASE("a stripped path and a drag-and-drop path name the same file") {
  const std::string dialog = strip_trailing_separators("/a/M.mlpackage/");
  const std::string dropped = "/a/M.mlpackage";
  CHECK(dialog == dropped);  // one Recent entry, not two
  CHECK(basename_of(dialog) == basename_of(dropped));
}

// --- the shared "open a model" filter list (#135 section 4) -------------------
// App::open_file_dialog and the DiffPanel comparison picker both build their
// filter from openable_extensions(), which is kExtensionFormats (the detector's
// own table) plus kDialogOnlyExtensions. There is no second hand-kept list, so
// these tests pin the composition and the shape of what is emitted.

namespace {
bool openable(std::string_view ext) {
  for (const std::string_view e : openable_extensions()) {
    if (e == ext) return true;
  }
  return false;
}

bool detector_extension(std::string_view ext) {
  for (const auto& e : kExtensionFormats) {
    if (e.ext == ext) return true;
  }
  return false;
}
}  // namespace

TEST_CASE("every extension the format detector recognises is openable") {
  for (const auto& e : kExtensionFormats) {
    INFO("detector extension: " << e.ext);
    CHECK(openable(e.ext));
  }
}

TEST_CASE("the openable list is exactly the detector's extensions plus the dialog-only ones") {
  // Equality, not just "detector is a subset": a misspelled, stale or
  // hand-added entry must fail here rather than offer files that resolve to
  // Format::Unknown.
  std::set<std::string_view> expected;
  for (const auto& e : kExtensionFormats) expected.insert(e.ext);
  for (const std::string_view ext : kDialogOnlyExtensions) expected.insert(ext);

  const std::vector<std::string_view> got = openable_extensions();
  const std::set<std::string_view> got_set(got.begin(), got.end());
  CHECK(got_set == expected);
  CHECK(got.size() == expected.size());  // and no extension is listed twice
}

TEST_CASE("dialog-only extensions are not ones the detector already acts on") {
  // If the detector learns one of these, it belongs in kExtensionFormats alone;
  // keeping it here too would be a second list creeping back in.
  for (const std::string_view ext : kDialogOnlyExtensions) {
    INFO("dialog-only extension: " << ext);
    CHECK_FALSE(detector_extension(ext));
  }
  // mlpackage is the one that exists today: a directory bundle, not a file the
  // detector sees (engine/ModelPath resolves it to the inner model).
  bool has_mlpackage = false;
  for (const std::string_view ext : kDialogOnlyExtensions) {
    has_mlpackage = has_mlpackage || ext == "mlpackage";
  }
  CHECK(has_mlpackage);
}

TEST_CASE("formats that used to be missing from the dialogs are listed") {
  // The original dialogs offered only onnx/tflite/safetensors/gguf/pt/pth/bin/pb.
  for (const char* ext : {"xml", "npz", "keras", "h5", "hdf5", "mlmodel",
                          "mlpackage", "pkl", "pickle"}) {
    INFO("extension: " << ext);
    CHECK(openable(ext));
  }
  // ...and nothing that used to be offered was dropped.
  for (const char* ext : {"onnx", "tflite", "safetensors", "gguf", "pt", "pth",
                          "bin", "pb"}) {
    INFO("extension: " << ext);
    CHECK(openable(ext));
  }
}

TEST_CASE("openable extensions are well-formed and unique") {
  std::set<std::string_view> seen;
  for (const std::string_view e : openable_extensions()) {
    INFO("extension: " << e);
    CHECK_FALSE(e.empty());
    // Lowercase, no leading dot, no glob metacharacters: detection and the
    // chooser patterns both compare against the lowercased bare extension.
    CHECK(e.find_first_of(".*?[ ") == std::string_view::npos);
    for (const char c : e) CHECK_FALSE((c >= 'A' && c <= 'Z'));
    CHECK(seen.insert(e).second);  // a duplicate is a hand-merge mistake
  }
}

TEST_CASE("openable_patterns yields one plain suffix glob per extension") {
  const std::vector<std::string_view> exts = openable_extensions();
  const std::vector<std::string> patterns = openable_patterns();
  REQUIRE(patterns.size() == exts.size());
  for (size_t i = 0; i < patterns.size(); ++i) {
    INFO("pattern: " << patterns[i]);
    // Plain, NOT case-folded: tinyfd's macOS path slices `pattern + 2`.
    CHECK(patterns[i] == "*." + std::string(exts[i]));
    // The Save-side helper must read each one back as a plain suffix.
    CHECK(suffix_of(patterns[i]) == "." + std::string(exts[i]));
  }
  CHECK(patterns.front() == "*.onnx");
}

TEST_CASE("the open filter is labelled as the all-models group") {
  CHECK(std::string(netvis::kOpenFilterDescription) == "All supported models");
}

// --- what the Linux helper choosers are handed (#135 review) ------------------

TEST_CASE("case_folded_glob matches either case of every letter") {
  CHECK(case_folded_glob("*.onnx") == "*.[oO][nN][nN][xX]");
  CHECK(case_folded_glob("*.ONNX") == "*.[oO][nN][nN][xX]");  // idempotent shape
  CHECK(case_folded_glob("*.h5") == "*.[hH]5");                // digits stay
  CHECK(case_folded_glob("*.mlpackage") ==
        "*.[mM][lL][pP][aA][cC][kK][aA][gG][eE]");
}

TEST_CASE("case_folded_glob leaves anything but a simple suffix glob alone") {
  CHECK(case_folded_glob("*") == "*");
  CHECK(case_folded_glob("model.*") == "model.*");
  CHECK(case_folded_glob("*.[ab]") == "*.[ab]");
  CHECK(case_folded_glob("") == "");
}

TEST_CASE("helper_globs folds every open-filter glob and joins with spaces") {
  CHECK(helper_globs({}) == "");
  CHECK(helper_globs({"*.onnx", "*.h5"}) == "*.[oO][nN][nN][xX] *.[hH]5");
  // Every extension the dialog offers is reachable in upper case too.
  const std::string all = helper_globs(openable_patterns());
  CHECK(all.find("*.[oO][nN][nN][xX]") != std::string::npos);
  CHECK(all.find("*.[mM][lL][pP][aA][cC][kK][aA][gG][eE]") != std::string::npos);
  CHECK(all.find("*.onnx") == std::string::npos);  // no case-sensitive leftovers
}

TEST_CASE("the kdialog filter offers All files after the model filter") {
  // kdialog (KDE's first choice) has no second-filter flag: one argument holds
  // newline-separated "globs|Description" entries. Without the second entry a
  // plugin-only format could not be picked at all.
  const std::string f = kdialog_filter("*.a *.b", "All supported models");
  CHECK(f == "*.a *.b|All supported models\n*|All files");
  const size_t nl = f.find('\n');
  REQUIRE(nl != std::string::npos);
  CHECK(f.find('\n', nl + 1) == std::string::npos);  // exactly two entries
  CHECK(f.substr(0, nl) == "*.a *.b|All supported models");
  CHECK(f.substr(nl + 1) == "*|All files");
}
