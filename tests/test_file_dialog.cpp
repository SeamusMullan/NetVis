// SPDX-License-Identifier: Apache-2.0
// tests/test_file_dialog.cpp — the pure path helpers behind the native chooser.
//
// BUILD SITUATION: src/view/FileDialog.cpp is GUI/platform code (fork/exec on
// Linux, tinyfiledialogs elsewhere) compiled into the `netvis` target only, and
// netvis_tests links netvis_core alone. The two helpers below are therefore
// `inline` in view/FileDialog.h, which pulls in nothing but <memory>, <string>
// and <vector> — so this file includes the header and links nothing.

#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "view/FileDialog.h"

using netvis::detail::apply_default_extension;
using netvis::detail::first_line;
using netvis::detail::suffix_of;

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
