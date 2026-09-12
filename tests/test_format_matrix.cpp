// SPDX-License-Identifier: Apache-2.0
// tests/test_format_matrix.cpp — the format-support matrix, checked against reality (#114).
//
// docs/format-support.md tells a user what NetVis does with each format: whether a
// file of that kind yields a compute graph, whether its tensors come back with
// shapes, and whether their weights are addressable on disk. A support table that
// is written by hand and never re-checked is exactly the kind of claim that goes
// quietly wrong - and it did: #107 inserted a TensorFlow structural sniff ahead of
// the `.mlmodel` extension guard, so every ordinary CoreML file was routed to the
// TensorFlow parser and failed to open, while the doc kept saying CoreML worked.
//
// So the table in that document IS the test input. Every row names a shipped
// fixture; this file opens it exactly the way the app does (resolve_model_path ->
// MappedFile -> parse_model) and asserts the row it was promised. A row that stops
// being true fails here, not in a user's hands.
//
// Set NETVIS_EMIT_FORMAT_MATRIX=1 to print the rows this run derived, in the
// document's own format, ready to paste back after an intentional change.
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "engine/ModelPath.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {

// --- The document's table ----------------------------------------------------

struct MatrixRow {
  std::string format;       // expected ir::Model format label
  std::string fixture;      // repo-relative path, as opened
  std::string graph;        // yes | no
  std::string tensors;      // yes | no
  std::string shapes;       // all | some | none
  std::string offsets;      // all | some | none
  int line = 0;             // for the failure message
};

std::string trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t");
  if (b == std::string::npos) return {};
  size_t e = s.find_last_not_of(" \t");
  return s.substr(b, e - b + 1);
}

// Strip the markdown emphasis the table uses for readability (`code`, **bold**).
std::string plain(const std::string& s) {
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '`') continue;
    if (s[i] == '*' && i + 1 < s.size() && s[i + 1] == '*') { ++i; continue; }
    out += s[i];
  }
  return trim(out);
}

std::vector<std::string> split_cells(const std::string& row) {
  std::vector<std::string> cells;
  std::string cur;
  // Leading and trailing '|' produce empty edge cells; drop them below.
  for (char c : row) {
    if (c == '|') { cells.push_back(plain(cur)); cur.clear(); }
    else cur += c;
  }
  cells.push_back(plain(cur));
  if (!cells.empty() && cells.front().empty()) cells.erase(cells.begin());
  if (!cells.empty() && cells.back().empty()) cells.pop_back();
  return cells;
}

// Rows between the BEGIN/END markers, skipping the header and its separator.
std::vector<MatrixRow> read_matrix(const std::string& path, std::string& err) {
  std::ifstream f(path);
  if (!f) { err = "cannot open " + path; return {}; }

  std::vector<MatrixRow> rows;
  std::string line;
  bool inside = false;
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    if (line.find("<!-- BEGIN MATRIX") != std::string::npos) { inside = true; continue; }
    if (line.find("<!-- END MATRIX") != std::string::npos) { inside = false; continue; }
    if (!inside) continue;
    const std::string t = trim(line);
    if (t.empty() || t[0] != '|') continue;
    if (t.find("---") != std::string::npos) continue;   // the separator row

    std::vector<std::string> c = split_cells(t);
    if (c.size() < 6) { err = path + ":" + std::to_string(lineno) + ": too few columns"; return {}; }
    if (c[0] == "Format") continue;                     // the header row
    rows.push_back({c[0], c[1], c[2], c[3], c[4], c[5], lineno});
  }
  if (!inside && rows.empty()) err = "no matrix rows found in " + path;
  return rows;
}

// --- What a fixture actually does -------------------------------------------

struct Observed {
  bool parsed = false;
  std::string error;
  std::string format;
  bool has_graph = false;
  size_t tensors = 0;
  size_t shaped = 0;      // tensors with a non-empty shape
  size_t addressable = 0; // tensors with a real file offset, or external data
  uint64_t payload_reads = 0;
};

// "all" when every tensor qualifies, "none" when none does, "some" in between.
// A model with no tensors at all reports "none", which the table then has to say
// out loud rather than leaving the column ambiguous.
const char* fraction(size_t part, size_t whole) {
  if (whole == 0 || part == 0) return "none";
  return part == whole ? "all" : "some";
}

std::string ext_of(const std::string& path) {
  std::string e = std::filesystem::path(path).extension().string();
  if (!e.empty() && e[0] == '.') e.erase(0, 1);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}

Observed observe(const std::string& fixture) {
  Observed o;
  // Exactly the app's open path: a .mlpackage or SavedModel directory resolves to
  // the one file to map, and its directory stays the place sibling weights are
  // looked up from.
  ResolvedModelPath rp = resolve_model_path(fixture);
  auto mf = MappedFile::open(rp.map_path);
  if (!mf) { o.error = "cannot map " + rp.map_path; return o; }

  ByteReader::payload_read_counter() = 0;
  ProgressSink progress;
  Result<ir::Model> r = parse_model(*mf, ext_of(rp.map_path), progress);
  o.payload_reads = ByteReader::payload_read_counter();
  if (!r) { o.error = r.error().message; return o; }

  const ir::Model& m = *r;
  o.parsed = true;
  o.format = std::string(m.str(m.format_name));
  o.has_graph = m.has_graph;

  auto count = [&](const ir::TensorRef& t) {
    ++o.tensors;
    if (!t.shape.empty()) ++o.shaped;
    // A weight is addressable when it can be located without guessing: a real
    // mmap offset, or an external-data path the loader resolves at inspect time.
    if (t.file_offset != UINT64_MAX || m.str(t.external_path).size() > 0) ++o.addressable;
  };
  for (const ir::TensorRef& t : m.flat_tensors) count(t);
  for (const ir::Graph& g : m.graphs)
    for (const ir::TensorRef& t : g.initializers) count(t);
  return o;
}

std::string emit_row(const MatrixRow& row, const Observed& o) {
  std::ostringstream ss;
  ss << "| " << o.format << " | `" << row.fixture << "` | "
     << (o.has_graph ? "yes" : "no") << " | " << (o.tensors ? "yes" : "no") << " | "
     << fraction(o.shaped, o.tensors) << " | " << fraction(o.addressable, o.tensors)
     << " |";
  return ss.str();
}

}  // namespace

TEST_CASE("format matrix: every documented row matches what the parser produces") {
  std::string err;
  std::vector<MatrixRow> rows = read_matrix("docs/format-support.md", err);
  REQUIRE_MESSAGE(err.empty(), err);
  REQUIRE_MESSAGE(!rows.empty(), "docs/format-support.md has no matrix rows");

  const bool emit = std::getenv("NETVIS_EMIT_FORMAT_MATRIX") != nullptr;

  for (const MatrixRow& row : rows) {
    CAPTURE(row.fixture);
    CAPTURE(row.line);
    Observed o = observe(row.fixture);
    // Emit mode prints the whole table and asserts nothing: a failing row would
    // otherwise abort the loop at exactly the moment you wanted the rest of it.
    if (emit) {
      std::cout << (o.parsed ? emit_row(row, o)
                             : "| ?? | `" + row.fixture + "` | FAILED: " + o.error + " |")
                << "\n";
      continue;
    }

    REQUIRE_MESSAGE(o.parsed,
                    "docs/format-support.md claims support for " << row.fixture
                        << " but opening it failed: " << o.error);
    CHECK_MESSAGE(o.format == row.format,
                  "format label: documented " << row.format << ", got " << o.format);
    CHECK_MESSAGE((o.has_graph ? "yes" : "no") == row.graph,
                  "graph: documented " << row.graph << ", got "
                                       << (o.has_graph ? "yes" : "no"));
    CHECK_MESSAGE((o.tensors ? "yes" : "no") == row.tensors,
                  "tensors: documented " << row.tensors << ", got "
                                         << (o.tensors ? "yes" : "no")
                                         << " (" << o.tensors << " recorded)");
    CHECK_MESSAGE(fraction(o.shaped, o.tensors) == row.shapes,
                  "shapes: documented " << row.shapes << ", got "
                                        << fraction(o.shaped, o.tensors) << " ("
                                        << o.shaped << "/" << o.tensors << ")");
    CHECK_MESSAGE(fraction(o.addressable, o.tensors) == row.offsets,
                  "weight offsets: documented " << row.offsets << ", got "
                                                << fraction(o.addressable, o.tensors)
                                                << " (" << o.addressable << "/"
                                                << o.tensors << ")");

    // The zero-payload thesis, asserted for every format in one place. Individual
    // parser tests check this for the formats they cover; this catches a format
    // whose own test forgot to, and any future parser added to the table.
    CHECK_MESSAGE(o.payload_reads == 0,
                  "structural parse read " << o.payload_reads
                      << " payload byte range(s) - the open must record "
                         "offset+len only");
  }
}

TEST_CASE("format matrix: an unresolved dimension is -1, never a guess") {
  // The honest-unknown rule for shapes: a parser that does not know a dimension
  // writes -1. A 0 would claim an empty tensor and silently zero every FLOP and
  // byte count downstream, which is a fabricated answer wearing a real one's
  // clothes. (A genuinely zero-length dim is not something any fixture format
  // expresses, so treat 0 as the fabrication marker it would be.)
  std::string err;
  std::vector<MatrixRow> rows = read_matrix("docs/format-support.md", err);
  REQUIRE_MESSAGE(err.empty(), err);

  for (const MatrixRow& row : rows) {
    CAPTURE(row.fixture);
    ResolvedModelPath rp = resolve_model_path(row.fixture);
    auto mf = MappedFile::open(rp.map_path);
    REQUIRE(mf);
    ProgressSink progress;
    Result<ir::Model> r = parse_model(*mf, ext_of(rp.map_path), progress);
    REQUIRE(r);
    const ir::Model& m = *r;

    auto check_shape = [&](const netvis::SmallVec<int64_t, 6>& shape, const char* what) {
      for (int64_t d : shape) {
        CHECK_MESSAGE(d != 0, what << " carries a 0 dimension");
        CHECK_MESSAGE(d >= -1, what << " carries dimension " << d
                                    << " (only -1 means unresolved)");
      }
    };
    for (const ir::TensorRef& t : m.flat_tensors) check_shape(t.shape, "a flat tensor");
    for (const ir::Graph& g : m.graphs) {
      for (const ir::TensorRef& t : g.initializers) check_shape(t.shape, "an initializer");
      for (const ir::ValueInfo& v : g.values) check_shape(v.shape, "a value");
    }
  }
}
