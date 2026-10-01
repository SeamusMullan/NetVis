// SPDX-License-Identifier: Apache-2.0
// tests/test_file_dialog.cpp — the pure helpers behind the native chooser: path
// handling, and the shared list of extensions the Open dialogs offer.
//
// BUILD SITUATION: src/view/FileDialog.cpp is GUI/platform code (fork/exec on
// Linux, tinyfiledialogs elsewhere) compiled into the `netvis` target only, and
// netvis_tests links netvis_core alone. The helpers below are therefore
// `inline` in view/FileDialog.h, which pulls in nothing but the standard library
// — so this file includes the header and links nothing from the GUI.

#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"
#include "parsers/Parser.h"
#include "view/FileDialog.h"

using netvis::detail::apply_default_extension;
using netvis::detail::first_line;
using netvis::detail::suffix_of;
using netvis::kExtensionFormats;
using netvis::kOpenableExtensions;
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

// --- the shared "open a model" filter list (#135 section 4) -------------------
// App::open_file_dialog and the DiffPanel comparison picker both build their
// filter from kOpenableExtensions. These tests are the regression guard: the
// detector's extension table (parsers/Parser.h) must be a subset of it.

namespace {
bool openable(std::string_view ext) {
  for (const std::string_view e : kOpenableExtensions) {
    if (e == ext) return true;
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
  for (const std::string_view e : kOpenableExtensions) {
    INFO("extension: " << e);
    CHECK_FALSE(e.empty());
    // Lowercase, no leading dot, no glob metacharacters: detection and the
    // chooser patterns both compare against the lowercased bare extension.
    CHECK(e.find_first_of(".*?[ ") == std::string_view::npos);
    for (const char c : e) CHECK_FALSE((c >= 'A' && c <= 'Z'));
    CHECK(seen.insert(e).second);  // a duplicate is a hand-merge mistake
  }
}

TEST_CASE("openable_patterns yields one simple suffix glob per extension") {
  const std::vector<std::string> patterns = openable_patterns();
  REQUIRE(patterns.size() == std::size(kOpenableExtensions));
  for (size_t i = 0; i < patterns.size(); ++i) {
    INFO("pattern: " << patterns[i]);
    CHECK(patterns[i] == "*." + std::string(kOpenableExtensions[i]));
    // The Save-side helper must read each one back as a plain suffix.
    CHECK(suffix_of(patterns[i]) == "." + std::string(kOpenableExtensions[i]));
  }
  CHECK(patterns.front() == "*.onnx");
}

TEST_CASE("the open filter is labelled as the all-models group") {
  CHECK(std::string(netvis::kOpenFilterDescription) == "All supported models");
}
