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
// What is checked is the TABLE between the BEGIN/END markers - and nothing else in
// the document. The prose around it (the per-format "known gaps") is hand-written
// and is NOT verified by this file; each gap there is pinned only to the extent a
// table row shows it (e.g. the typed-data ONNX and compressed npz rows).
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
  size_t addressable = 0; // tensors with an in-file byte range, or external data
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

  const uint64_t file_size = mf->size();
  auto count = [&](const ir::TensorRef& t) {
    ++o.tensors;
    // "Shaped" is "has a non-empty shape". ir::TensorRef has no shape-known flag,
    // so a genuine rank-0 scalar (empty shape) is indistinguishable from a shape
    // the parser could not determine and reads as unshaped here. That errs toward
    // claiming LESS, and a row that flips to "some" because of a scalar constant
    // is a prompt to look, not a silent over-claim.
    if (!t.shape.empty()) ++o.shaped;
    // A weight is addressable when it can be located without guessing: a byte
    // range that lies inside the mapped file (overflow-safe), or an external-data
    // path the loader resolves at inspect time.
    const bool external = m.str(t.external_path).size() > 0;
    const bool in_file = t.file_offset != UINT64_MAX && t.file_offset <= file_size &&
                         t.byte_len <= file_size - t.file_offset;
    if (external || in_file) ++o.addressable;
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

    // The zero-payload rule, as far as it can be observed from here.
    // ByteReader::payload_read_counter() is bumped only by code that explicitly
    // calls mark_payload_read() - the weight inspector's decoders (TensorStats)
    // and the WASM host's file reader. No built-in parser calls it, and plain
    // ByteReader::bytes()/raw() reads are not counted at all; the counter is also
    // thread_local, so a read on a worker thread would not register. So this
    // asserts that a structural open never routes through the payload decoders.
    // It is NOT proof that no parser touches a weight page: that holds because
    // parsers record offset+len by construction, which this test cannot see.
    CHECK_MESSAGE(o.payload_reads == 0,
                  "structural parse routed through " << o.payload_reads
                      << " payload decode(s) - the open must record "
                         "offset+len only");
  }
}

TEST_CASE("format matrix: no fixture carries a dimension below -1") {
  // The honest-unknown rule for shapes: -1 is the one marker for "unresolved", so
  // nothing below it may appear. A parser that wrote some other negative number
  // would be inventing a second, unreadable spelling of "unknown". (0 is NOT
  // policed here: a zero-length dimension is a real, expressible shape - ONNX
  // dim_value, a .npy shape and a SafeTensors shape can all hold one - and
  // forcing it to -1 would turn a known fact into a fabricated unknown. The
  // test below pins that 0 survives and -1 means unresolved.)
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

namespace {

// Minimal protobuf encoders for hand-building an ONNX ModelProto in memory.
void pb_put_varint(std::vector<uint8_t>& out, uint64_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<uint8_t>(v | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<uint8_t>(v));
}
std::vector<uint8_t> pb_varint_field(uint32_t field, uint64_t v) {
  std::vector<uint8_t> out;
  pb_put_varint(out, (static_cast<uint64_t>(field) << 3) | 0);
  pb_put_varint(out, v);
  return out;
}
std::vector<uint8_t> pb_len_field(uint32_t field, const std::vector<uint8_t>& body) {
  std::vector<uint8_t> out;
  pb_put_varint(out, (static_cast<uint64_t>(field) << 3) | 2);
  pb_put_varint(out, body.size());
  out.insert(out.end(), body.begin(), body.end());
  return out;
}
std::vector<uint8_t> pb_str_field(uint32_t field, const std::string& s) {
  return pb_len_field(field, std::vector<uint8_t>(s.begin(), s.end()));
}
void append(std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  a.insert(a.end(), b.begin(), b.end());
}

}  // namespace

TEST_CASE("format matrix: -1 means unresolved, and a real 0 stays 0") {
  // The rule, pinned directly rather than inferred from the fixtures (none of
  // which has a symbolic dimension). One ONNX graph input with four dimensions:
  //   dim_param "N"   -> symbolic        -> -1 (unresolved)
  //   dim_value 3     -> known           ->  3
  //   dim_value 0     -> a real 0-length ->  0 (known, NOT turned into -1)
  //   (neither field) -> unspecified     -> -1 (unresolved)
  std::vector<uint8_t> shape;
  append(shape, pb_len_field(1, pb_str_field(2, "N")));      // dim_param
  append(shape, pb_len_field(1, pb_varint_field(1, 3)));     // dim_value = 3
  append(shape, pb_len_field(1, pb_varint_field(1, 0)));     // dim_value = 0
  append(shape, pb_len_field(1, {}));                        // empty Dimension

  std::vector<uint8_t> tensor_type;
  append(tensor_type, pb_varint_field(1, 1));                // elem_type = FLOAT
  append(tensor_type, pb_len_field(2, shape));               // shape
  std::vector<uint8_t> value_info;
  append(value_info, pb_str_field(1, "x"));                  // name
  append(value_info, pb_len_field(2, pb_len_field(1, tensor_type)));  // type

  std::vector<uint8_t> graph;
  append(graph, pb_str_field(2, "g"));
  append(graph, pb_len_field(11, value_info));               // input
  std::vector<uint8_t> modelproto;
  append(modelproto, pb_varint_field(1, 8));                 // ir_version
  append(modelproto, pb_len_field(7, graph));

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "nv_matrix_dims.onnx";
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(modelproto.data()),
              static_cast<std::streamsize>(modelproto.size()));
  }
  {
    // Scoped so the mapping is released before the file is removed (Windows
    // refuses to delete a file that is still mapped).
    auto mf = MappedFile::open(path.string());
    REQUIRE(mf);
    ProgressSink progress;
    Result<ir::Model> r = parse_model(*mf, "onnx", progress);
    REQUIRE_MESSAGE(r, "hand-built ONNX model failed to parse");
    const ir::Model& m = *r;
    REQUIRE(m.graphs.size() == 1);
    const ir::Graph& g = m.graphs[0];
    REQUIRE(g.graph_inputs.size() == 1);
    const ir::ValueInfo& v = g.values[g.graph_inputs[0]];
    REQUIRE(v.shape.size() == 4);
    CHECK(v.shape[0] == -1);
    CHECK(v.shape[1] == 3);
    CHECK(v.shape[2] == 0);
    CHECK(v.shape[3] == -1);
  }
  std::error_code ec;
  std::filesystem::remove(path, ec);
}
