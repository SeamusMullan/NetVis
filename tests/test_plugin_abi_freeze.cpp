// SPDX-License-Identifier: Apache-2.0
// tests/test_plugin_abi_freeze.cpp — the plugin ABI v1 freeze + version negotiation (#113).
//
// tests/test_sdk_abi.cpp already static_asserts that the SHARED VALUES in
// plugins/sdk/netvis_plugin.h match the host C++ contracts. That catches drift in
// what both sides already agree on; it cannot catch the two ways a frozen ABI
// actually breaks:
//
//   1. The SURFACE changes. A renamed import, a changed signature, a dropped macro,
//      a retired enumerator, a reordered struct field, a renamed guest export - all
//      compile fine on the host, and all break a plugin binary that was compiled
//      against the old header. So the surface is written down in
//      plugins/sdk/abi-v1-surface.txt, and the first cases below re-derive it from
//      the header and from the host's own link tables / call sites and demand they
//      agree. Changing the header now requires editing the freeze file too, which is
//      the reviewable moment where someone asks "does this need an ABI bump?". The
//      extractors are themselves tested (a mutated header must be noticed, a
//      reworded comment must not), because a guard that cannot fail guards nothing.
//
//   2. VERSION NEGOTIATION does not actually reject. The promise in
//      docs/plugin-abi.md is that a plugin the host cannot honour is refused
//      CLEANLY - no half-registration, no partial answers, and the built-in result
//      still stands. The remaining cases prove that over the real paths: the
//      Registry's registration check, the WASM adapters' own gates, the manifest
//      gate, and the whole loader (load_wasm_op_plugin -> resolve_op -> compute_cost,
//      and discover_and_load_plugins), for a plugin from the future, one that
//      declares no version, one the host cannot link, and one with a bad manifest.
//      Each refusal case sits next to a matching-plugin control, so a gate that
//      refuses everything cannot pass. The same goes for the ABI probe's own bound
//      (a module whose start section never ends is refused after a small step
//      budget, with that reason) and for the details of the pass gate (a refused
//      pass yields no metrics; a broken ABI export is refused, not taken for v1).
#include <doctest/doctest.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/MappedFile.h"
#include "engine/CostModel.h"
#include "engine/OpCategory.h"
#include "engine/plugin/OpHandler.h"
#include "engine/plugin/ParserPlugin.h"
#include "engine/plugin/PassPlugin.h"
#include "engine/plugin/Registry.h"
#include "engine/plugin/declarative/Manifest.h"
#include "engine/plugin/wasm/WasmHost.h"
#include "engine/plugin/wasm/WasmOpHandler.h"
#include "engine/plugin/wasm/WasmParser.h"
#include "engine/plugin/wasm/WasmRuntime.h"
#include "ir/IR.h"

using namespace netvis;
using namespace netvis::plugin;

namespace {

namespace fs = std::filesystem;

// --- Text utilities ----------------------------------------------------------

std::string read_file_or_empty(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<std::string> split_lines(const std::string& text) {
  std::vector<std::string> out;
  std::string line;
  std::istringstream ss(text);
  while (std::getline(ss, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    out.push_back(line);
  }
  return out;
}

std::string trim(const std::string& s) {
  const size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  const size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// Collapse every run of whitespace to one space (a declaration split over lines
// reads the same as one on a single line).
std::string squeeze(const std::string& s) {
  std::string out;
  bool space = false;
  for (char c : s) {
    if (std::isspace(static_cast<unsigned char>(c))) { space = true; continue; }
    if (space && !out.empty()) out += ' ';
    space = false;
    out += c;
  }
  return out;
}

std::string join(const std::vector<std::string>& v, const char* sep = ", ") {
  std::string s;
  for (const std::string& e : v) { if (!s.empty()) s += sep; s += e; }
  return s;
}

std::vector<std::string> difference(const std::set<std::string>& a,
                                    const std::set<std::string>& b) {
  std::vector<std::string> out;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

bool is_ident_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// Position of `word` as a whole identifier at or after `from`, else npos.
size_t find_word(const std::string& text, const std::string& word, size_t from) {
  for (size_t pos = text.find(word, from); pos != std::string::npos;
       pos = text.find(word, pos + 1)) {
    const bool left_ok = pos == 0 || !is_ident_char(text[pos - 1]);
    const bool right_ok =
        pos + word.size() >= text.size() || !is_ident_char(text[pos + word.size()]);
    if (left_ok && right_ok) return pos;
  }
  return std::string::npos;
}

size_t count_word(const std::string& text, const std::string& word) {
  size_t n = 0;
  for (size_t pos = find_word(text, word, 0); pos != std::string::npos;
       pos = find_word(text, word, pos + 1))
    ++n;
  return n;
}

// Replace every comment with whitespace and keep every newline, so line numbers and
// line structure survive. Comments are NOT part of the ABI: rewording one must never
// fail the freeze, and text inside one (a commented-out `L("op_set_color", ...)`, an
// example `NV_MAX_RANK = 9`) must never count as code. Handles `//` and `/* */`
// across lines, leaves string and character literals alone, and does not mistake a
// C++14 digit separator (2'000'000) for the start of a character literal.
std::string strip_comments(const std::string& src) {
  enum class St { Code, Line, Block, Str, Chr };
  St st = St::Code;
  std::string out;
  out.reserve(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    const char c = src[i];
    const char n = i + 1 < src.size() ? src[i + 1] : '\0';
    switch (st) {
      case St::Code:
        if (c == '/' && n == '/') { st = St::Line; out += ' '; ++i; }
        else if (c == '/' && n == '*') { st = St::Block; out += ' '; ++i; }
        else if (c == '"') { st = St::Str; out += c; }
        else if (c == '\'' && !(!out.empty() && is_ident_char(out.back()))) {
          st = St::Chr; out += c;
        }
        else out += c;
        break;
      case St::Line:
        if (c == '\n') { st = St::Code; out += c; }
        break;
      case St::Block:
        if (c == '*' && n == '/') { st = St::Code; ++i; }
        else if (c == '\n') out += c;
        break;
      case St::Str:
      case St::Chr:
        out += c;
        if (c == '\\' && i + 1 < src.size()) out += src[++i];
        else if (c == (st == St::Str ? '"' : '\'')) st = St::Code;
        else if (c == '\n') st = St::Code;   // unterminated: recover at end of line
        break;
    }
  }
  return out;
}

// --- Signature notation ------------------------------------------------------
// Both sides are rendered as `(i32,i64)->void`, the notation wasm itself uses, so a
// C prototype and a wasm3 link string can be compared for equality.

std::string canon_c_type(const std::string& type) {
  static const std::map<std::string, std::string> kMap = {
      {"int32_t", "i32"}, {"int64_t", "i64"}, {"double", "f64"},
      {"float", "f32"},   {"void", "void"}};
  const std::string t = squeeze(type);
  const auto it = kMap.find(t);
  return it == kMap.end() ? "?" + t : it->second;   // "?" never matches a frozen line
}

// "int32_t slot, int64_t def" -> {"i32","i64"}; "void" or "" -> {}.
std::vector<std::string> c_param_types(const std::string& params) {
  std::vector<std::string> out;
  const std::string p = trim(params);
  if (p.empty() || p == "void") return out;
  std::istringstream ss(p);
  std::string part;
  while (std::getline(ss, part, ',')) {
    part = trim(part);
    const size_t sp = part.find_last_of(" \t");   // the trailing parameter name
    out.push_back(canon_c_type(sp == std::string::npos ? part : part.substr(0, sp)));
  }
  return out;
}

std::string render_sig(const std::vector<std::string>& params, const std::string& ret) {
  return "(" + join(params, ",") + ")->" + ret;
}

// wasm3's m3_LinkRawFunctionEx notation: `i(i*i)` = (i32,i32,i32)->i32. i=i32, I=i64,
// f=f32, F=f64, *=i32 (a guest memory offset), v=void (return only).
std::string canon_host_sig(const std::string& sig) {
  if (sig.size() < 3 || sig[1] != '(' || sig.back() != ')') return "?" + sig;
  auto one = [](char c, bool is_ret) -> std::string {
    switch (c) {
      case 'i': case '*': return "i32";
      case 'I': return "i64";
      case 'f': return "f32";
      case 'F': return "f64";
      case 'v': if (is_ret) return "void"; break;
      default: break;
    }
    return std::string("?") + c;
  };
  std::vector<std::string> params;
  for (size_t i = 2; i + 1 < sig.size(); ++i) params.push_back(one(sig[i], false));
  return render_sig(params, one(sig[0], true));
}

// --- Header surface ----------------------------------------------------------
// Deliberately line- and statement-oriented and dumb: it must be obvious that this
// reads the same thing a plugin author reads, and it must not need a C parser to
// stay honest. Anything it cannot read becomes an `unfrozen ...`/`unparsed ...`
// entry that no freeze file contains, so it fails loudly instead of being skipped.

// Walks the typedef statements of the (comment-free) header text.
void extract_typedefs(const std::string& text, std::set<std::string>& out) {
  size_t enum_blocks = 0, struct_blocks = 0;
  size_t pos = find_word(text, "typedef", 0);
  while (pos != std::string::npos) {
    const size_t after = pos + std::string("typedef").size();
    const size_t t = text.find_first_not_of(" \t\r\n", after);
    if (t == std::string::npos) { out.insert("unparsed typedef at end of header"); break; }
    size_t semi = std::string::npos;
    const bool is_enum = text.compare(t, 4, "enum") == 0 && !is_ident_char(text[t + 4]);
    const bool is_struct = text.compare(t, 6, "struct") == 0 && !is_ident_char(text[t + 6]);
    if (is_enum || is_struct) {
      const size_t lb = text.find('{', t);
      const size_t rb = lb == std::string::npos ? lb : text.find('}', lb);
      semi = rb == std::string::npos ? rb : text.find(';', rb);
      if (semi == std::string::npos) { out.insert("unparsed typedef block"); break; }
      const std::string body = text.substr(lb + 1, rb - lb - 1);
      const std::string name = trim(text.substr(rb + 1, semi - rb - 1));
      if (is_enum) {
        ++enum_blocks;
        out.insert("type " + name + " = enum");
        static const std::regex re_item(R"RX(^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?[0-9]+)$)RX");
        std::istringstream ss(body);
        std::string item;
        while (std::getline(ss, item, ',')) {
          item = squeeze(item);
          if (item.empty()) continue;   // trailing comma
          std::smatch m;
          if (std::regex_match(item, m, re_item))
            out.insert("enum " + m[1].str() + " = " + m[2].str());
          else
            // No explicit decimal value (implicit, or hex): the value is not on the
            // page where a reviewer reads it, so it cannot be frozen.
            out.insert("unfrozen enumerator (needs an explicit decimal value): " + item);
        }
      } else {
        ++struct_blocks;
        out.insert("type " + name + " = struct");
        static const std::regex re_field(R"RX(^(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)$)RX");
        std::istringstream ss(body);
        std::string item;
        size_t index = 0;
        while (std::getline(ss, item, ';')) {
          item = squeeze(item);
          if (item.empty()) continue;
          std::smatch m;
          if (std::regex_match(item, m, re_field))
            out.insert("field " + name + " " + std::to_string(index++) + " " +
                       m[1].str() + " " + m[2].str());
          else
            out.insert("unfrozen struct member: " + item);
        }
      }
    } else {
      semi = text.find(';', after);
      if (semi == std::string::npos) { out.insert("unparsed typedef"); break; }
      // `typedef <type...> <name>;` - the last token is the name.
      const std::string decl = squeeze(text.substr(after, semi - after));
      const size_t sp = decl.find_last_of(' ');
      if (sp == std::string::npos) out.insert("unparsed typedef: " + decl);
      else out.insert("type " + decl.substr(sp + 1) + " = " + decl.substr(0, sp));
    }
    pos = find_word(text, "typedef", semi);
  }
  // A declaration this walker did not read is a surface it did not freeze.
  if (count_word(text, "enum") != enum_blocks)
    out.insert("unparsed enum declaration (only `typedef enum {...} name;` is frozen)");
  if (count_word(text, "struct") != struct_blocks)
    out.insert("unparsed struct declaration (only `typedef struct {...} name;` is frozen)");
  if (count_word(text, "union") != 0)
    out.insert("unparsed union declaration");
}

// The public surface of the SDK header, as unique `<kind> ...` entries. The format
// is documented at the top of abi-v1-surface.txt.
std::set<std::string> extract_sdk_surface(const std::string& header_text) {
  // A function-like macro is frozen by NAME only: its body expands to
  // toolchain attributes (import_module / export_name) that a non-clang guest
  // toolchain is expected to spell differently. The `(` must touch the name,
  // otherwise `#define X (1)` would be misread as function-like.
  static const std::regex re_define(
      R"RX(^#\s*define\s+((?:NV_|NETVIS_)[A-Za-z0-9_]*)(\([^)]*\))?(.*)$)RX");
  static const std::regex re_import(
      R"RX(NV_IMPORT\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)\s*([A-Za-z_][A-Za-z0-9_]*)\s+)RX"
      R"RX(([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*;)RX");
  static const std::regex re_facet(R"RX(NV_FACET:\s*([a-z]+))RX");
  static const std::regex re_helper(
      R"RX(^static\s+inline\s+[A-Za-z_][A-Za-z0-9_]*\s*\**\s*(nv_[A-Za-z0-9_]+)\s*\()RX");

  std::set<std::string> surface;
  const std::vector<std::string> raw = split_lines(header_text);
  const std::string stripped_text = strip_comments(header_text);
  const std::vector<std::string> stripped = split_lines(stripped_text);

  std::string facet = "none";
  for (size_t i = 0; i < stripped.size(); ++i) {
    // The facet marker lives in a comment (so it is not part of the surface), which
    // is why it is read from the raw line.
    std::smatch fm;
    if (i < raw.size() && std::regex_search(raw[i], fm, re_facet)) facet = fm[1].str();

    const std::string line = trim(stripped[i]);
    if (line.empty()) continue;

    std::smatch m;
    if (std::regex_search(line, m, re_define)) {
      const std::string name = m[1].str();
      if (m[2].matched && m[2].length() > 0) {
        surface.insert("define " + name + "()");
      } else {
        const std::string value = trim(m[3].str());
        surface.insert(value.empty() ? "define " + name
                                     : "define " + name + " = " + value);
      }
      continue;
    }
    if (line.find("NV_IMPORT(\"") != std::string::npos) {
      if (std::regex_search(line, m, re_import)) {
        const std::string sig =
            render_sig(c_param_types(m[5].str()), canon_c_type(m[3].str()));
        surface.insert("import " + facet + " " + m[1].str() + " " + m[2].str() + " " +
                       sig + " as " + m[4].str());
      } else {
        surface.insert("unparsed import declaration: " + squeeze(line));
      }
      continue;
    }
    if (std::regex_search(line, m, re_helper)) surface.insert("helper " + m[1].str());
  }
  extract_typedefs(stripped_text, surface);
  return surface;
}

// The committed freeze, minus its `#` comment header and blank lines.
std::set<std::string> read_frozen_surface(const std::string& text) {
  std::set<std::string> out;
  for (const std::string& raw : split_lines(text)) {
    const std::string line = trim(raw);
    if (line.empty() || line[0] == '#') continue;
    out.insert(line);
  }
  return out;
}

// --- Host side: link tables and entry-point call sites -----------------------
// The three WASM adapters. A facet is the part of the ABI one adapter implements:
// the module name alone cannot tell the parser's "netvis" imports from the pass's.
struct HostSite {
  const char* path;
  const char* facet;
};
const HostSite kHostSites[] = {
    {"src/engine/plugin/wasm/WasmOpHandler.cpp", "op"},
    {"src/engine/plugin/wasm/WasmParser.cpp", "parser"},
    {"src/engine/plugin/wasm/WasmHost.cpp", "pass"},
};

// Every `import <facet> <module> <name> <sig>` the host links into a guest, read off
// its link tables. `ns` is assigned once per link function and every entry below it
// uses it, so tracking the last assignment is enough. Comments are stripped first: a
// commented-out link is not a link.
std::set<std::string> extract_host_imports_from(const std::string& source,
                                                const std::string& facet) {
  static const std::regex re_ns(R"RX(const char\*\s*ns\s*=\s*"([^"]+)")RX");
  static const std::regex re_short(R"RX(\bL\(\s*"([^"]+)"\s*,\s*"([^"]+)")RX");
  static const std::regex re_long(
      R"RX(m3_LinkRawFunctionEx\(\s*mod\s*,\s*ns\s*,\s*"([^"]+)"\s*,\s*"([^"]+)")RX");
  std::set<std::string> out;
  std::string ns;
  for (const std::string& line : split_lines(strip_comments(source))) {
    std::smatch m;
    if (std::regex_search(line, m, re_ns)) { ns = m[1].str(); continue; }
    if (ns.empty()) continue;
    if (std::regex_search(line, m, re_short) || std::regex_search(line, m, re_long)) {
      out.insert("import " + facet + " " + ns + " " + m[1].str() + " " +
                 canon_host_sig(m[2].str()));
    }
  }
  return out;
}

// Every guest export the host looks up by name, read off its call sites. All entry
// points are `() -> i32` (WasmModule::call_i32 refuses anything else), so that is the
// frozen signature.
std::set<std::string> extract_host_exports_from(const std::string& source,
                                                const std::string& facet) {
  static const std::regex re_site(
      R"RX(\b(?:call_i32|invoke_facet|read_abi_declaration)\([^"]*"([^"]+)")RX");
  std::set<std::string> out;
  for (const std::string& line : split_lines(strip_comments(source))) {
    for (auto it = std::sregex_iterator(line.begin(), line.end(), re_site);
         it != std::sregex_iterator(); ++it) {
      out.insert("export " + facet + " " + (*it)[1].str() + " ()->i32");
    }
  }
  return out;
}

std::set<std::string> collect_host(bool imports) {
  std::set<std::string> out;
  for (const HostSite& s : kHostSites) {
    const std::string src = read_file_or_empty(s.path);
    REQUIRE_MESSAGE(!src.empty(), s.path << " not readable");
    const std::set<std::string> part = imports ? extract_host_imports_from(src, s.facet)
                                               : extract_host_exports_from(src, s.facet);
    REQUIRE_MESSAGE(!part.empty(), "found no " << (imports ? "imports" : "exports")
                                               << " in " << s.path
                                               << " -- the link tables or call sites "
                                                  "were reshaped, so this guard no "
                                                  "longer reads them");
    out.insert(part.begin(), part.end());
  }
  return out;
}

// What the frozen file must contain: the header's surface plus the host's exports.
std::set<std::string> actual_surface(const std::string& header_text) {
  std::set<std::string> s = extract_sdk_surface(header_text);
  const std::set<std::string> exports = collect_host(/*imports=*/false);
  s.insert(exports.begin(), exports.end());
  return s;
}

// `import <facet> <module> <name> <sig> as <cname>` -> without the ` as <cname>`.
std::set<std::string> header_imports(const std::set<std::string>& surface) {
  std::set<std::string> out;
  for (const std::string& e : surface) {
    if (e.rfind("import ", 0) != 0) continue;
    const size_t as = e.rfind(" as ");
    out.insert(as == std::string::npos ? e : e.substr(0, as));
  }
  return out;
}

// --- Stub plugins ------------------------------------------------------------
// Declares `delta` away from the host's ABI version (0 = a plugin that matches).

class StubOpHandler final : public OpHandler {
 public:
  explicit StubOpHandler(uint32_t delta) : delta_(delta) {}
  OpCategory category(const OpContext&) const override { return OpCategory::Conv; }
  uint32_t api_version() const override { return kOpHandlerAbiVersion + delta_; }
 private:
  uint32_t delta_;
};

class StubParser final : public ParserPlugin {
 public:
  explicit StubParser(uint32_t delta) : delta_(delta) {}
  Format format() const override { return Format::Unknown; }
  std::string_view display_name() const override { return "abi-freeze-stub-parser"; }
  bool can_parse(const MappedFile&, const std::string&) const override { return true; }
  int priority() const override { return 1000; }
  Result<ir::Model> parse(const MappedFile&, ProgressSink&) const override {
    return err("a stub parser is never asked to parse");
  }
  uint32_t api_version() const override { return kParserPluginAbiVersion + delta_; }
 private:
  uint32_t delta_;
};

class StubPass final : public PassPlugin {
 public:
  StubPass(std::string name, uint32_t delta) : name_(std::move(name)), delta_(delta) {}
  std::string_view display_name() const override { return name_; }
  PassResult run(const ir::Model&, uint32_t, const CostReport&) const override { return {}; }
  uint32_t api_version() const override { return kPassPluginAbiVersion + delta_; }
 private:
  std::string name_;
  uint32_t delta_;
};

// --- WASM helpers ------------------------------------------------------------

// A skipped WASM case must be VISIBLE, not a silent pass with zero assertions.
bool wasm_available() {
  if (wasm::WasmEngine::instance().enabled()) return true;
  WARN_MESSAGE(false, "built without NETVIS_ENABLE_WASM; WASM ABI-gate cases skipped");
  return false;
}

std::vector<uint8_t> read_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

std::shared_ptr<const std::vector<uint8_t>> load_image(const std::string& fixture) {
  std::vector<uint8_t> bytes = read_bytes("tests/fixtures/" + fixture);
  if (bytes.empty()) return nullptr;
  return std::make_shared<std::vector<uint8_t>>(std::move(bytes));
}

// A model with one node: `op_name`(A[2,2]) -> Y[2,2].
ir::Model make_one_node(const std::string& op_name) {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  auto add_val = [&](const char* nm) {
    uint32_t vi = static_cast<uint32_t>(g.values.size());
    ir::ValueInfo v;
    v.name = m.intern(nm);
    v.dtype = ir::DType::F32;
    v.shape.push_back(2);
    v.shape.push_back(2);
    g.values.push_back(std::move(v));
    return vi;
  };
  uint32_t a = add_val("A"), y = add_val("Y");
  ir::Node n;
  n.op_type = m.intern(op_name);
  n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(a);
  n.inputs.count = 1;
  n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(y);
  n.outputs.count = 1;
  g.values[y].producer = 0;
  g.nodes.push_back(std::move(n));
  return m;
}

// MatMul A[2,3] x B[3,4] -> Y[2,4]: the built-in answer is a known 2*8*3 = 48 FLOPs,
// which is what an override must not silently turn into "unknown".
ir::Model make_matmul() {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  auto add_val = [&](const char* nm, int64_t d0, int64_t d1) {
    uint32_t vi = static_cast<uint32_t>(g.values.size());
    ir::ValueInfo v;
    v.name = m.intern(nm);
    v.dtype = ir::DType::F32;
    v.shape.push_back(d0);
    v.shape.push_back(d1);
    g.values.push_back(std::move(v));
    return vi;
  };
  const uint32_t a = add_val("A", 2, 3), b = add_val("B", 3, 4), y = add_val("Y", 2, 4);
  ir::Node n;
  n.op_type = m.intern("MatMul");
  n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(a);
  g.edge_refs.push_back(b);
  n.inputs.count = 2;
  n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(y);
  n.outputs.count = 1;
  g.values[y].producer = 0;
  g.nodes.push_back(std::move(n));
  return m;
}

// A scratch plugin directory holding one WASM plugin: <dir>/plugin.json + plugin.wasm.
struct PluginDir {
  fs::path dir;
  std::string manifest;   // path of plugin.json

  PluginDir(const std::string& stem, const std::string& manifest_json,
            const std::string& wasm_fixture) {
    dir = fs::temp_directory_path() / ("nv_abi_freeze_" + stem);
    fs::remove_all(dir);
    fs::create_directories(dir);
    manifest = (dir / "plugin.json").string();
    std::ofstream(manifest) << manifest_json;
    if (wasm_fixture.empty()) return;   // a declarative plugin has no module
    const std::vector<uint8_t> bytes = read_bytes("tests/fixtures/" + wasm_fixture);
    std::ofstream wf((dir / "plugin.wasm").string(), std::ios::binary);
    wf.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  }
  ~PluginDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

// plugin.json for a WASM op plugin that overrides MatMul. `version` is raw JSON text
// (so the gate can be handed a string, a float, a wrapping integer, or nothing).
std::string op_manifest(const std::string& version_member) {
  return "{" + version_member +
         R"("name": "abi-freeze-op", "wasm": "plugin.wasm",
            "ops": [{"name": "matmul", "override": true}]})";
}
std::string parser_manifest(const std::string& version_member) {
  return "{" + version_member +
         R"("name": "abi-freeze-parser", "parser_wasm": "plugin.wasm"})";
}

}  // namespace

// ===========================================================================
// 1. The SDK header surface is frozen - signatures, exports, layout and all.
// ===========================================================================
TEST_CASE("ABI v1: the SDK header surface matches the committed freeze") {
  const std::string header = read_file_or_empty("plugins/sdk/netvis_plugin.h");
  const std::string frozen_text = read_file_or_empty("plugins/sdk/abi-v1-surface.txt");
  REQUIRE_MESSAGE(!header.empty(), "plugins/sdk/netvis_plugin.h not readable");
  REQUIRE_MESSAGE(!frozen_text.empty(), "plugins/sdk/abi-v1-surface.txt not readable");

  const std::set<std::string> actual = actual_surface(header);
  const std::set<std::string> frozen = read_frozen_surface(frozen_text);

  const std::vector<std::string> added = difference(actual, frozen);
  const std::vector<std::string> removed = difference(frozen, actual);

  // Additions and removals are reported separately because they mean different
  // things: an addition may be a compatible extension, a removal or a changed
  // value never is. See docs/plugin-abi.md ("Compatibility promise").
  CHECK_MESSAGE(added.empty(),
                "SDK surface entries NOT in plugins/sdk/abi-v1-surface.txt:\n  "
                    << join(added, "\n  ")
                    << "\n-- add the lines to the freeze file, and classify the "
                       "change against the table in docs/plugin-abi.md "
                       "(\"Compatibility promise\"): some surface changes keep "
                       "ABI v1, others force a version bump.");
  CHECK_MESSAGE(removed.empty(),
                "frozen ABI v1 surface entries MISSING from the header / host:\n  "
                    << join(removed, "\n  ")
                    << "\n-- removing a name, renumbering an enumerator, changing a "
                       "signature or tightening a cap breaks every plugin already "
                       "compiled against ABI v1.");
}

// ===========================================================================
// 2. The host links exactly the imports the header promises - signatures included.
// ===========================================================================
TEST_CASE("ABI v1: host link tables and the SDK header declare the same imports") {
  const std::string header = read_file_or_empty("plugins/sdk/netvis_plugin.h");
  REQUIRE_MESSAGE(!header.empty(), "plugins/sdk/netvis_plugin.h not readable");

  const std::set<std::string> from_header = header_imports(extract_sdk_surface(header));
  const std::set<std::string> from_host = collect_host(/*imports=*/true);
  REQUIRE(!from_header.empty());

  // An import is (facet, module, name, signature). The signature is compared too: a
  // name that is linked with a different signature is NOT linked at all - wasm3
  // refuses it with "function signature mismatch" and every guest that imports it
  // fails. (That is exactly how op_input_const_ints was broken: header i32 x3, host
  // linked i32 x2.) Host link strings are mapped into the header's notation first.
  CHECK_MESSAGE(difference(from_header, from_host).empty(),
                "declared by the SDK header but NOT linked by the host (same facet, "
                "module, name AND signature):\n  "
                    << join(difference(from_header, from_host), "\n  ")
                    << "\n-- a guest importing one of these fails to link.");
  CHECK_MESSAGE(difference(from_host, from_header).empty(),
                "linked by the host but NOT declared by the SDK header (same facet, "
                "module, name AND signature):\n  "
                    << join(difference(from_host, from_header), "\n  ")
                    << "\n-- plugin authors have no supported way to reach it, so "
                       "either publish it in the header or stop linking it.");
}

// ===========================================================================
// 3. The guards themselves can fail: mutated sources are noticed, rewording is not.
// ===========================================================================
namespace {
// `text` with the first occurrence of `from` replaced by `to`; fails the test if
// `from` is not there (a mutation that changed nothing would prove nothing).
std::string mutate(const std::string& text, const std::string& from, const std::string& to) {
  const size_t at = text.find(from);
  REQUIRE_MESSAGE(at != std::string::npos, "mutation anchor not found: " << from);
  std::string out = text;
  out.replace(at, from.size(), to);
  return out;
}
}  // namespace

TEST_CASE("ABI v1 guard: a change to the header surface is noticed") {
  const std::string header = read_file_or_empty("plugins/sdk/netvis_plugin.h");
  REQUIRE(!header.empty());
  const std::set<std::string> base = extract_sdk_surface(header);

  auto noticed = [&](const std::string& mutated) {
    return extract_sdk_surface(mutated) != base;
  };

  SUBCASE("an import's parameter type") {
    CHECK(noticed(mutate(header, "nv_op_input_const_ints(int32_t slot, int32_t dst",
                         "nv_op_input_const_ints(int32_t slot, int64_t dst")));
  }
  SUBCASE("an import's return type") {
    CHECK(noticed(mutate(header, "int64_t nv_host_file_len(void)",
                         "int32_t nv_host_file_len(void)")));
  }
  SUBCASE("an import's value width") {
    CHECK(noticed(mutate(header, "nv_op_set_flops(int64_t value", "nv_op_set_flops(int32_t value")));
  }
  SUBCASE("an import's wasm name") {
    CHECK(noticed(mutate(header, "\"op_input_count\"", "\"op_inputs_count\"")));
  }
  SUBCASE("a C prototype's name (source compatibility)") {
    CHECK(noticed(mutate(header, "nv_op_input_count(void)", "nv_op_inputs_count(void)")));
  }
  SUBCASE("an import moved to another facet") {
    CHECK(noticed(mutate(header, "/* NV_FACET: pass */", "/* NV_FACET: parser */")));
  }
  SUBCASE("a guest helper") {
    CHECK(noticed(mutate(header, "nv_arena_reset(void)", "nv_arena_clear(void)")));
    CHECK(noticed(mutate(header, "nv_alloc(uint32_t n)", "nv_malloc(uint32_t n)")));
  }
  SUBCASE("a plain typedef's name and its underlying type") {
    CHECK(noticed(mutate(header, "typedef uint32_t nv_strid_t;", "typedef uint32_t nv_stringid_t;")));
    CHECK(noticed(mutate(header, "typedef uint32_t nv_strid_t;", "typedef uint64_t nv_strid_t;")));
  }
  SUBCASE("a wire struct's field order") {
    // Same size, same field types: only the order differs, so sizeof cannot see it.
    std::string swapped = mutate(header, "int32_t dtype;", "int32_t @@;");
    swapped = mutate(swapped, "int32_t rank;", "int32_t dtype;");
    swapped = mutate(swapped, "int32_t @@;", "int32_t rank;");
    CHECK(noticed(swapped));
  }
  SUBCASE("a wire struct field's type") {
    CHECK(noticed(mutate(header, "int64_t off;", "int32_t off;")));
  }
  SUBCASE("an enumerator's value") {
    CHECK(noticed(mutate(header, "NV_DT_UNKNOWN = 15", "NV_DT_UNKNOWN = 16")));
  }
  SUBCASE("an enumerator with no explicit value") {
    CHECK(noticed(mutate(header, "NV_DT_U8 = 8,", "NV_DT_U8,")));
  }
  SUBCASE("an enumerator written in hex (value not on the page)") {
    CHECK(noticed(mutate(header, "NV_CAT_OTHER = 14", "NV_CAT_OTHER = 0x0E")));
  }
  SUBCASE("a cap, a version macro, a dropped macro") {
    CHECK(noticed(mutate(header, "#define NV_MAX_RANK         8", "#define NV_MAX_RANK         4")));
    CHECK(noticed(mutate(header, "#define NETVIS_OP_ABI_VERSION     1u",
                         "#define NETVIS_OP_ABI_VERSION     2u")));
    CHECK(noticed(mutate(header, "#define NV_STATUS_ABSTAIN 1", "")));
  }
  SUBCASE("a new declaration the walker cannot read becomes a failure, not a skip") {
    CHECK(noticed(header + "\ntypedef enum { NV_NEW_THING } nv_new_t;\n"));
    CHECK(noticed(header + "\nenum { NV_LOOSE = 1 };\n"));
  }
}

TEST_CASE("ABI v1 guard: rewording a comment is not a change, and comments are not code") {
  const std::string header = read_file_or_empty("plugins/sdk/netvis_plugin.h");
  REQUIRE(!header.empty());
  const std::set<std::string> base = extract_sdk_surface(header);

  SUBCASE("a continuation line of a block comment that looks like code") {
    // The old extractor only cut at the opening of a comment, so these inner lines
    // were parsed as code and produced `enum NV_MAX_RANK = 9`.
    const std::string noisy = mutate(
        header, "/* ---- ABI versions",
        "/* a doc note:\n *   - e.g. NV_MAX_RANK = 9 would break plugins\n"
        " *   #define NV_FAKE_MACRO 7\n"
        " *   NV_IMPORT(\"netvis_op\",\"op_fake\") int32_t nv_op_fake(void);\n"
        " * typedef uint32_t nv_fake_t;\n */\n/* ---- ABI versions");
    CHECK(extract_sdk_surface(noisy) == base);
  }
  SUBCASE("reworded comment text") {
    CHECK(extract_sdk_surface(mutate(header, "interned string id (StringId::id)",
                                     "any interned string handle")) == base);
  }
  SUBCASE("a commented-out declaration") {
    CHECK(extract_sdk_surface(header + "\n// #define NV_OLD_CAP 1\n"
                                       "/* NV_IMPORT(\"netvis\",\"gone\") void nv_gone(void); */\n") == base);
  }
}

TEST_CASE("ABI v1 guard: the host-side extractors read code, not comments") {
  const std::string op_src = read_file_or_empty("src/engine/plugin/wasm/WasmOpHandler.cpp");
  REQUIRE(!op_src.empty());
  const std::set<std::string> base = extract_host_imports_from(op_src, "op");
  REQUIRE(base.count("import op netvis_op op_set_color (i32)->void") == 1);

  SUBCASE("a link commented out with // is gone") {
    const std::set<std::string> got = extract_host_imports_from(
        mutate(op_src, "L(\"op_set_color\"", "// L(\"op_set_color\""), "op");
    CHECK(got.count("import op netvis_op op_set_color (i32)->void") == 0);
    CHECK(got.size() + 1 == base.size());
  }
  SUBCASE("a link commented out with /* */ is gone") {
    const std::set<std::string> got = extract_host_imports_from(
        mutate(mutate(op_src, "L(\"op_set_color\"", "/* L(\"op_set_color\""),
               "&op_set_color);", "&op_set_color); */"), "op");
    CHECK(got.count("import op netvis_op op_set_color (i32)->void") == 0);
  }
  SUBCASE("a changed link signature is a different import") {
    const std::set<std::string> got = extract_host_imports_from(
        mutate(op_src, "L(\"op_set_color\", \"v(i)\"", "L(\"op_set_color\", \"v(I)\""), "op");
    CHECK(got != base);
  }
  SUBCASE("a renamed entry point is a different export") {
    const std::set<std::string> exp = extract_host_exports_from(op_src, "op");
    REQUIRE(exp.count("export op netvis_op_category ()->i32") == 1);
    const std::set<std::string> got = extract_host_exports_from(
        mutate(op_src, "\"netvis_op_category\"", "\"netvis_op_cat\""), "op");
    CHECK(got != exp);
  }
}

TEST_CASE("ABI v1 guard: notation helpers agree on every wasm3 signature character") {
  CHECK(canon_host_sig("i(i*i)") == "(i32,i32,i32)->i32");
  CHECK(canon_host_sig("v(Ii)") == "(i64,i32)->void");
  CHECK(canon_host_sig("F(*i*)") == "(i32,i32,i32)->f64");
  CHECK(canon_host_sig("I()") == "()->i64");
  CHECK(canon_host_sig("v(*iFi)") == "(i32,i32,f64,i32)->void");
  CHECK(canon_host_sig("x(i)").find('?') != std::string::npos);
  CHECK(render_sig(c_param_types("int32_t slot, int64_t def"), canon_c_type("double")) ==
        "(i32,i64)->f64");
  CHECK(c_param_types("void").empty());
}

// ===========================================================================
// 4. Registration refuses a wrong ABI for each plugin kind - and admits a right one.
// ===========================================================================
TEST_CASE("ABI v1: the Registry refuses a plugin declaring a future ABI") {
  Registry& reg = Registry::instance();
  reg.reset_to_builtins();

  // Op keys no built-in recognizes, so nothing here depends on the shadow/override
  // rules - only on the version check.
  const std::string future_key = "netvis_abi_freeze_future_probe";
  const std::string current_key = "netvis_abi_freeze_current_probe";
  const size_t parsers_before = reg.snapshot()->parsers.size();

  reg.register_op_handler(future_key, "", std::make_unique<StubOpHandler>(1), Origin::Wasm,
                          /*override_flag=*/true, "abi-freeze-future-op");
  reg.register_parser(std::make_unique<StubParser>(1));
  reg.register_pass(std::make_unique<StubPass>("abi-freeze-future-pass", 1));

  // The controls: byte-identical stubs that declare the host's version. They prove
  // the check discriminates, rather than the Registry refusing everything.
  reg.register_op_handler(current_key, "", std::make_unique<StubOpHandler>(0), Origin::Wasm,
                          /*override_flag=*/true, "abi-freeze-current-op");
  reg.register_pass(std::make_unique<StubPass>("abi-freeze-current-pass", 0));
  const size_t parsers_after_future = reg.snapshot()->parsers.size();
  reg.register_parser(std::make_unique<StubParser>(0));

  auto tbl = reg.snapshot();

  SUBCASE("the op handler never enters the table") {
    CHECK(tbl->op_by_key.count(future_key) == 0);
    // And resolution still lands on the built-in catch-all, so the app's answer
    // for that op is the honest built-in one, not a hole.
    OpResolution r = reg.resolve_op(future_key);
    CHECK(r.handler != nullptr);
    CHECK(r.origin == Origin::Builtin);
  }
  SUBCASE("the parser never enters the table") {
    CHECK(parsers_after_future == parsers_before);
  }
  SUBCASE("the pass never enters the table") {
    CHECK(tbl->passes.count("abi-freeze-future-pass") == 0);
  }
  SUBCASE("control: a plugin declaring the host's ABI is admitted") {
    CHECK(tbl->op_by_key.count(current_key) == 1);
    CHECK(reg.resolve_op(current_key).origin == Origin::Wasm);
    CHECK(tbl->parsers.size() == parsers_before + 1);
    CHECK(tbl->passes.count("abi-freeze-current-pass") == 1);
  }

  reg.reset_to_builtins();
}

// ===========================================================================
// 5. The WASM op adapter: it reports the module's declared ABI and gates on it.
// ===========================================================================
TEST_CASE("ABI v1: a WASM op plugin from the future is refused, answer stays built-in") {
  if (!wasm_available()) return;

  // Relu's built-in category is Activation. Every fixture below answers Conv through
  // op_set_category, so a Conv here can only come from the module and an Activation
  // can only be the default - the category itself tells the cases apart.
  ir::Model model = make_one_node("Relu");
  Registry& reg = Registry::instance();
  reg.reset_to_builtins();
  OpContext ctx = reg.make_context(model, model.graphs[0], model.graphs[0].nodes[0],
                                   normalize_op_key("Relu"));
  REQUIRE(ctx.default_category() == OpCategory::Activation);

  SUBCASE("netvis_op_abi_version returns a version the host was not built for") {
    auto image = load_image("plugin_ophandler_future_abi.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("future-abi", image);

    CHECK(h.api_version() == 2);                          // reports what the MODULE declared
    CHECK(h.category(ctx) == OpCategory::Activation);     // the default, not the module's Conv
    CHECK(h.diag().loaded);            // the module itself is fine ...
    CHECK(h.diag().abi_mismatch);      // ... it is the declared version that is not
    CHECK(h.flops(ctx).known == false);
    CHECK(h.infer_shape(ctx).outputs.empty());
    CHECK(h.color(ctx).overridden == false);
  }

  SUBCASE("netvis_op_abi_version is not exported at all") {
    auto image = load_image("plugin_ophandler_no_abi.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("no-abi", image);
    CHECK(h.api_version() == 0);                          // declares nothing
    CHECK(h.category(ctx) == OpCategory::Activation);
    CHECK(h.diag().abi_mismatch);
    CHECK(h.flops(ctx).known == false);
  }

  SUBCASE("control: the matching fixture still answers, so the gate is not refusing everything") {
    auto image = load_image("plugin_ophandler.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("ok-abi", image);
    CHECK(h.api_version() == kOpHandlerAbiVersion);
    CHECK(h.category(ctx) == OpCategory::Conv);           // the module's answer
    CHECK(h.diag().abi_mismatch == false);
    CHECK(h.flops(ctx).known);
    CHECK(h.flops(ctx).flops == 1024);
  }

  SUBCASE("through register_op_handler: the Registry gate is real for a WASM handler") {
    // The same check every loader relies on. Before the adapters reported the
    // module's version this admitted a wrong-ABI module, because api_version()
    // echoed the host's own constant.
    auto future = load_image("plugin_ophandler_future_abi.wasm");
    auto none = load_image("plugin_ophandler_no_abi.wasm");
    auto good = load_image("plugin_ophandler.wasm");
    REQUIRE((future && none && good));
    reg.register_op_handler("netvis_abi_freeze_wasm_future", "",
                            std::make_unique<wasm::WasmOpHandler>("f", future),
                            Origin::Wasm, true, "f");
    reg.register_op_handler("netvis_abi_freeze_wasm_none", "",
                            std::make_unique<wasm::WasmOpHandler>("n", none),
                            Origin::Wasm, true, "n");
    reg.register_op_handler("netvis_abi_freeze_wasm_good", "",
                            std::make_unique<wasm::WasmOpHandler>("g", good),
                            Origin::Wasm, true, "g");
    CHECK(reg.resolve_op("netvis_abi_freeze_wasm_future").origin == Origin::Builtin);
    CHECK(reg.resolve_op("netvis_abi_freeze_wasm_none").origin == Origin::Builtin);
    CHECK(reg.resolve_op("netvis_abi_freeze_wasm_good").origin == Origin::Wasm);
  }

  reg.reset_to_builtins();
}

// ===========================================================================
// 6. Linking: an import the host does not answer makes the module unusable.
// ===========================================================================
TEST_CASE("ABI v1: a module whose imports the host cannot bind is refused") {
  if (!wasm_available()) return;

  ir::Model model = make_one_node("Relu");
  Registry& reg = Registry::instance();
  reg.reset_to_builtins();
  OpContext ctx = reg.make_context(model, model.graphs[0], model.graphs[0].nodes[0],
                                   normalize_op_key("Relu"));

  SUBCASE("control: op_input_const_ints imported with the SDK header's signature binds") {
    // The host linked this as a two-argument function while the header declared
    // three, so wasm3 rejected the header's own signature and no SDK-built plugin
    // could use it. The fixture's flops is const_ints(0,64,4) + 8; with no mmap base
    // the host answers -1, so a known 7 proves the import linked AND returned.
    auto image = load_image("plugin_ophandler_const_ints.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("const-ints", image);
    CHECK(h.api_version() == kOpHandlerAbiVersion);
    CHECK(h.diag().unresolved_import == false);
    const FlopResult fr = h.flops(ctx);
    CHECK(fr.known);
    CHECK(fr.flops == 7);
  }

  SUBCASE("a real import name with the wrong signature") {
    auto image = load_image("plugin_ophandler_bad_sig_import.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("bad-sig", image);
    CHECK(h.api_version() == 0);   // declares v1 but cannot run here: not speakable
    CHECK(h.diag().loaded);
    CHECK(h.diag().unresolved_import);
    CHECK(h.diag().message.find("op_set_flops") != std::string::npos);
    CHECK(h.flops(ctx).known == false);
    // Refused at registration too.
    reg.register_op_handler("netvis_abi_freeze_bad_sig", "",
                            std::make_unique<wasm::WasmOpHandler>("b", image),
                            Origin::Wasm, true, "b");
    CHECK(reg.resolve_op("netvis_abi_freeze_bad_sig").origin == Origin::Builtin);
  }

  SUBCASE("an import this host has no function for (a newer v1 header, an older v1 host)") {
    auto image = load_image("plugin_ophandler_unknown_import.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("unknown-import", image);
    CHECK(h.api_version() == 0);
    CHECK(h.diag().unresolved_import);
    CHECK(h.diag().message.find("op_from_a_newer_sdk") != std::string::npos);
    // The module never calls the unknown import, and its flops would answer 1024 -
    // but a module that needs host functions this NetVis lacks is not honoured.
    CHECK(h.flops(ctx).known == false);
  }

  reg.reset_to_builtins();
}

// ===========================================================================
// 7. Shape honesty: a rank above NV_MAX_RANK stays unknown, never truncated.
// ===========================================================================
TEST_CASE("ABI v1: op_set_output_shape leaves a rank above NV_MAX_RANK unknown instead of truncating") {
  if (!wasm_available()) return;

  ir::Model model = make_one_node("MyShapeOp");
  OpContext ctx = Registry::instance().make_context(
      model, model.graphs[0], model.graphs[0].nodes[0], normalize_op_key("MyShapeOp"));

  SUBCASE("control: rank 8 == NV_MAX_RANK is answered in full") {
    auto image = load_image("plugin_ophandler_shape_r8.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("shape-r8", image);
    REQUIRE(h.api_version() == kOpHandlerAbiVersion);
    const ShapeResult sr = h.infer_shape(ctx);
    REQUIRE(sr.outputs.size() == 1);
    REQUIRE(sr.outputs[0].shape.size() == 8);
    CHECK(sr.outputs[0].shape[0] == 3);
    CHECK(sr.outputs[0].shape[7] == 10);
    CHECK(sr.outputs[0].dtype == ir::DType::F32);
  }

  SUBCASE("rank 9, one over the cap: the output keeps its dtype and no shape (unknown)") {
    // A plugin built against a later v1 header with a raised NV_MAX_RANK reaches this
    // host. The old host kept the first eight dims and reported a known rank-8 shape:
    // a fabricated one. It must be honest-unknown instead - and an empty shape is how
    // ShapeResult spells that ("don't set a shape for this slot"), while the dtype,
    // which the overflow does not touch, is still carried.
    auto image = load_image("plugin_ophandler_shape_r9.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("shape-r9", image);
    REQUIRE(h.api_version() == kOpHandlerAbiVersion);
    const ShapeResult sr = h.infer_shape(ctx);
    REQUIRE(sr.outputs.size() == 1);
    CHECK(sr.outputs[0].slot == 0);
    CHECK(sr.outputs[0].shape.empty());              // not truncated to eight dims
    CHECK(sr.outputs[0].dtype == ir::DType::F32);    // the declared dtype survives
  }
}

// ===========================================================================
// 8. The parser adapter.
// ===========================================================================
TEST_CASE("ABI v1: a WASM parser plugin that is not ABI v1 never claims a file") {
  if (!wasm_available()) return;

  // A file no built-in format claims, so reaching a plugin parser is the only way
  // it could ever be parsed.
  const std::string path =
      (fs::temp_directory_path() / "nv_abi_freeze_input.bin").string();
  { std::ofstream f(path, std::ios::binary); f << std::string(64, '\0'); }
  auto mf = MappedFile::open(path);
  REQUIRE(mf);

  Registry& reg = Registry::instance();
  reg.reset_to_builtins();
  const size_t parsers_before = reg.snapshot()->parsers.size();

  SUBCASE("netvis_parser_abi_version returns a version the host was not built for") {
    auto image = load_image("plugin_toyparser_future_abi.wasm");
    REQUIRE(image != nullptr);
    std::unique_ptr<ParserPlugin> parser = wasm::make_wasm_parser("future-abi", image);
    REQUIRE(parser != nullptr);
    CHECK(parser->api_version() == 2);

    // The matching fixture claims this exact file, so a refusal here is the ABI gate
    // and not the sniff failing.
    CHECK(parser->can_parse(*mf, "bin") == false);

    // And parse() refuses too, rather than trusting that nobody calls it after
    // can_parse said no.
    ProgressSink prog;
    CHECK(!parser->parse(*mf, prog));

    reg.register_parser(std::move(parser));
    CHECK(reg.snapshot()->parsers.size() == parsers_before);
  }

  SUBCASE("netvis_parser_abi_version is not exported at all") {
    auto image = load_image("plugin_toyparser_no_abi.wasm");
    REQUIRE(image != nullptr);
    std::unique_ptr<ParserPlugin> parser = wasm::make_wasm_parser("no-abi", image);
    REQUIRE(parser != nullptr);
    CHECK(parser->api_version() == 0);
    CHECK(parser->can_parse(*mf, "bin") == false);
    ProgressSink prog;
    CHECK(!parser->parse(*mf, prog));
    reg.register_parser(std::move(parser));
    CHECK(reg.snapshot()->parsers.size() == parsers_before);
  }

  SUBCASE("control: the matching fixture still claims it, and is admitted") {
    auto image = load_image("plugin_toyparser.wasm");
    REQUIRE(image != nullptr);
    std::unique_ptr<ParserPlugin> parser = wasm::make_wasm_parser("ok-abi", image);
    REQUIRE(parser != nullptr);
    CHECK(parser->api_version() == kParserPluginAbiVersion);
    CHECK(parser->can_parse(*mf, "bin") == true);
    reg.register_parser(std::move(parser));
    CHECK(reg.snapshot()->parsers.size() == parsers_before + 1);
  }

  reg.reset_to_builtins();
  fs::remove(path);
}

// ===========================================================================
// 9. The pass adapter: optional export, so shipped v0.6.0 passes keep running.
// ===========================================================================
TEST_CASE("ABI v1: a WASM pass plugin is gated on the ABI it declares") {
  if (!wasm_available()) return;

  CostReport report;
  report.total_flops = 21;
  ir::Model m;
  m.graphs.emplace_back();

  Registry& reg = Registry::instance();
  reg.reset_to_builtins();

  SUBCASE("netvis_pass_abi_version returns a version the host was not built for") {
    wasm::WasmPassPlugin pass("future-pass", read_bytes("tests/fixtures/plugin_pass_future_abi.wasm"));
    CHECK(pass.api_version() == 2);
    CHECK(pass.run(m, 0, report).metrics.empty());   // no metrics shown under another ABI
    reg.register_pass(std::make_unique<wasm::WasmPassPlugin>(
        "future-pass", read_bytes("tests/fixtures/plugin_pass_future_abi.wasm")));
    CHECK(reg.snapshot()->passes.count("future-pass") == 0);
  }

  SUBCASE("control: a pass declaring ABI v1 runs and is admitted") {
    wasm::WasmPassPlugin pass("ok-pass", read_bytes("tests/fixtures/plugin_pass_abi1.wasm"));
    CHECK(pass.api_version() == kPassPluginAbiVersion);
    const PassResult res = pass.run(m, 0, report);
    REQUIRE(res.metrics.size() == 1);
    CHECK(res.metrics[0].value == doctest::Approx(42.0));
    reg.register_pass(std::make_unique<wasm::WasmPassPlugin>(
        "ok-pass", read_bytes("tests/fixtures/plugin_pass_abi1.wasm")));
    CHECK(reg.snapshot()->passes.count("ok-pass") == 1);
  }

  SUBCASE("a refused pass yields no metrics even when its ABI export emitted one") {
    // The ABI export is guest code, and it can call host_emit_metric. The refusal path
    // used to return the metrics collected so far, so a module declaring a future ABI
    // that emitted one while being judged still showed it - the opposite of the
    // documented "run() yields no metrics".
    wasm::WasmPassPlugin pass("emitting-future-pass",
                              read_bytes("tests/fixtures/plugin_pass_abi_emits_future.wasm"));
    CHECK(pass.api_version() == 2);
    CHECK(pass.run(m, 0, report).metrics.empty());
  }

  SUBCASE("control: a pass declaring v1 reports run()'s metric, and only that one") {
    // Same module shape as above but declaring the host's ABI: it runs, so a refusal
    // that dropped everything could not pass. The metric its ABI export emitted while
    // being judged (7) is not a metric; run()'s (2 * 21) is.
    wasm::WasmPassPlugin pass("emitting-v1-pass",
                              read_bytes("tests/fixtures/plugin_pass_abi_emits_v1.wasm"));
    CHECK(pass.api_version() == kPassPluginAbiVersion);
    const PassResult res = pass.run(m, 0, report);
    REQUIRE(res.metrics.size() == 1);
    CHECK(res.metrics[0].name == "double_flops");
    CHECK(res.metrics[0].value == doctest::Approx(42.0));
  }

  SUBCASE("a pass whose ABI export is present but broken is refused, not taken for v1") {
    // netvis_pass_abi_version exists but its body calls function index 99, which the
    // module does not have. wasm3 reports that as "function lookup failed" - the very
    // error it gives for a name nothing answers to - so the export used to be read as
    // absent and the module run as a pre-negotiation v1 pass. If it were, run() below
    // would report double_flops = 42.
    wasm::WasmPassPlugin pass("broken-abi-pass",
                              read_bytes("tests/fixtures/plugin_pass_abi_broken.wasm"));
    CHECK(pass.probe().loaded);
    CHECK_FALSE(pass.probe().abi.declared);
    CHECK_FALSE(pass.probe().abi.export_missing);   // present, not absent
    CHECK(pass.api_version() == 0);
    const std::string why =
        pass.probe().refusal(kPassPluginAbiVersion, "netvis_pass_abi_version");
    CHECK_MESSAGE(why.find("present but failed") != std::string::npos, "refusal said: " << why);
    CHECK(pass.run(m, 0, report).metrics.empty());
    reg.register_pass(std::make_unique<wasm::WasmPassPlugin>(
        "broken-abi-pass", read_bytes("tests/fixtures/plugin_pass_abi_broken.wasm")));
    CHECK(reg.snapshot()->passes.count("broken-abi-pass") == 0);
  }

  SUBCASE("a pass that predates the export (the v0.6.0 shape) is ABI v1 and keeps running") {
    // The pass facet shipped before netvis_pass_abi_version existed. Refusing a
    // module for lacking it would break every shipped pass, which the ABI v1
    // promise forbids - so for this facet, and only this facet, "absent" means v1.
    wasm::WasmPassPlugin pass("legacy-pass", read_bytes("tests/fixtures/plugin_pass.wasm"));
    CHECK(pass.api_version() == kPassPluginAbiVersion);
    const PassResult res = pass.run(m, 0, report);
    REQUIRE(res.metrics.size() == 1);
    CHECK(res.metrics[0].name == "double_flops");
    CHECK(res.metrics[0].value == doctest::Approx(42.0));
    reg.register_pass(std::make_unique<wasm::WasmPassPlugin>(
        "legacy-pass", read_bytes("tests/fixtures/plugin_pass.wasm")));
    CHECK(reg.snapshot()->passes.count("legacy-pass") == 1);
  }

  reg.reset_to_builtins();
}

// ===========================================================================
// 9b. "Absent" is decided by the module's export names, not by an error string.
// ===========================================================================
TEST_CASE("ABI v1: an export that is present but fails is not reported as absent") {
  if (!wasm_available()) return;

  wasm::WasmEngine& eng = wasm::WasmEngine::instance();
  std::lock_guard<std::mutex> guard(eng.lock());
  auto load = [&](const char* fixture) {
    wasm::RunResult lerr;
    wasm::WasmModule mod = eng.load(read_bytes(std::string("tests/fixtures/") + fixture),
                                    wasm::SandboxLimits{}, nullptr, &lerr);
    REQUIRE_MESSAGE(mod.loaded(), fixture << ": " << lerr.message);
    return mod;
  };

  SUBCASE("a pass with no ABI export: absent") {
    wasm::WasmModule mod = load("plugin_pass.wasm");
    CHECK(mod.has_export("run"));
    CHECK_FALSE(mod.has_export("netvis_pass_abi_version"));
    int32_t v = -1;
    const wasm::RunResult r = mod.call_i32("netvis_pass_abi_version", &v);
    CHECK(r.status == wasm::RunStatus::LoadError);
    CHECK(r.export_missing);
  }

  SUBCASE("a pass whose ABI export calls a function that does not exist: present") {
    wasm::WasmModule mod = load("plugin_pass_abi_broken.wasm");
    CHECK(mod.has_export("netvis_pass_abi_version"));
    int32_t v = -1;
    const wasm::RunResult r = mod.call_i32("netvis_pass_abi_version", &v);
    CHECK(r.status == wasm::RunStatus::LoadError);   // it fails ...
    CHECK_FALSE(r.export_missing);                   // ... but it is there
    // The same call for a name that really is absent, to show the two differ.
    CHECK(mod.call_i32("netvis_no_such_export", &v).export_missing);
  }

  SUBCASE("control: a working ABI export is present and returns") {
    wasm::WasmModule mod = load("plugin_pass_abi1.wasm");
    CHECK(mod.has_export("netvis_pass_abi_version"));
    int32_t v = -1;
    CHECK(mod.call_i32("netvis_pass_abi_version", &v).status == wasm::RunStatus::Ok);
    CHECK(v == 1);
  }
}

// ===========================================================================
// 9c. The ABI probe is bounded: loading a plugin never runs guest code for long.
// ===========================================================================
// The probe runs wherever plugins are loaded - the UI thread, at start-up and on
// every Plugins-panel toggle - and it executes guest code: wasm3 runs the module's
// start section inside the first function lookup, then the ABI export runs. It used
// to be bounded only by the budget of the work the module is later trusted with
// (500,000 steps for the parser sniff, 2,000,000 for an op handler, 200,000,000 for a
// pass), so a hostile start section froze start-up for as long as that lasted. The
// fixtures below never finish, or need far more steps than a version read can
// justify; each must be refused with the budget named, and its control (a short start
// section) must still be admitted so the budget is not simply refusing everything.
// No timing is asserted: a probe that ignored the budget would not return at all.
TEST_CASE("ABI v1: a module that cannot answer the ABI question within the probe budget is refused") {
  if (!wasm_available()) return;

  Registry& reg = Registry::instance();
  reg.reset_to_builtins();
  const std::string current = R"("api_version": 1,)";
  const std::string reason = "ABI probe budget";

  // The three ways to blow the budget, per facet. start_long finishes on its own
  // (1,000,000 iterations) and is far below every run budget: it is refused because
  // of the probe's budget, not because it is infinite.
  const char* const kVariants[] = {"start_loop", "export_loop", "start_long"};

  // The budget is small, shared, and far below what any facet runs under (the
  // static_asserts in WasmRuntime.h / the adapters hold the margin; this holds the
  // number the docs promise).
  CHECK(wasm::kAbiProbeStepBudget == 50'000);

  SUBCASE("op handler: load_wasm_op_plugin refuses it and the built-in answer stands") {
    for (const char* v : kVariants) {
      const std::string fixture = std::string("plugin_probe_op_") + v + ".wasm";
      CAPTURE(fixture);
      PluginDir pd("probe_op", op_manifest(current), fixture);
      const std::string why = wasm::load_wasm_op_plugin(pd.manifest);
      CHECK_MESSAGE(why.find(reason) != std::string::npos, "loader said: " << why);
      CHECK(reg.resolve_op("matmul").origin == Origin::Builtin);

      auto image = load_image(fixture);
      REQUIRE(image != nullptr);
      const wasm::WasmAbiProbe probe = wasm::probe_op_module(*image);
      CHECK(probe.loaded);
      CHECK(probe.abi.fuel_exhausted);
      CHECK_FALSE(probe.compatible(kOpHandlerAbiVersion));
    }
  }

  SUBCASE("op handler: the reason reaches the Plugins panel through discovery") {
    const fs::path root = fs::temp_directory_path() / "nv_abi_freeze_probe_discovery";
    fs::remove_all(root);
    fs::create_directories(root / "hostile");
    std::ofstream(root / "hostile" / "plugin.json") << op_manifest(current);
    const std::vector<uint8_t> bytes =
        read_bytes("tests/fixtures/plugin_probe_op_start_loop.wasm");
    {
      std::ofstream wf((root / "hostile" / "plugin.wasm").string(), std::ios::binary);
      wf.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    }
    const auto manifests = discover_and_load_plugins(
        [](std::string_view, PluginKind) { return true; }, root.string());
    REQUIRE(manifests.size() == 1);
    CHECK(manifests[0].enabled);
    CHECK_FALSE(manifests[0].registered);
    CHECK_MESSAGE(manifests[0].error.find(reason) != std::string::npos,
                  "panel error: " << manifests[0].error);
    CHECK(reg.resolve_op("matmul").origin == Origin::Builtin);
    fs::remove_all(root);
  }

  SUBCASE("op handler control: a short start section is admitted") {
    PluginDir pd("probe_op_ok", op_manifest(current), "plugin_probe_op_start_short.wasm");
    CHECK(wasm::load_wasm_op_plugin(pd.manifest).empty());
    CHECK(reg.resolve_op("matmul").origin == Origin::Wasm);
  }

  SUBCASE("parser: load_wasm_parser_plugin refuses it and registers nothing") {
    const size_t before = reg.snapshot()->parsers.size();
    for (const char* v : kVariants) {
      const std::string fixture = std::string("plugin_probe_parser_") + v + ".wasm";
      CAPTURE(fixture);
      PluginDir pd("probe_parser", parser_manifest(current), fixture);
      const std::string why = wasm::load_wasm_parser_plugin(pd.manifest);
      CHECK_MESSAGE(why.find(reason) != std::string::npos, "loader said: " << why);
      CHECK(reg.snapshot()->parsers.size() == before);
    }
  }

  SUBCASE("parser control: a short start section is admitted") {
    const size_t before = reg.snapshot()->parsers.size();
    PluginDir pd("probe_parser_ok", parser_manifest(current),
                 "plugin_probe_parser_start_short.wasm");
    CHECK(wasm::load_wasm_parser_plugin(pd.manifest).empty());
    CHECK(reg.snapshot()->parsers.size() == before + 1);
  }

  SUBCASE("pass: the adapter's probe refuses it, so the Registry never admits it") {
    // Not run(): a refused pass is never registered, and run() re-checks the module
    // under the pass's own (full) budget by design.
    for (const char* v : kVariants) {
      const std::string fixture = std::string("plugin_probe_pass_") + v + ".wasm";
      CAPTURE(fixture);
      wasm::WasmPassPlugin pass("probe-pass", read_bytes("tests/fixtures/" + fixture));
      CHECK(pass.probe().loaded);
      CHECK(pass.probe().abi.fuel_exhausted);
      CHECK(pass.api_version() == 0);
      const std::string why =
          pass.probe().refusal(kPassPluginAbiVersion, "netvis_pass_abi_version");
      CHECK_MESSAGE(why.find(reason) != std::string::npos, "refusal said: " << why);
      reg.register_pass(std::make_unique<wasm::WasmPassPlugin>(
          "probe-pass", read_bytes("tests/fixtures/" + fixture)));
      CHECK(reg.snapshot()->passes.count("probe-pass") == 0);
    }
  }

  SUBCASE("pass control: a short start section is admitted") {
    wasm::WasmPassPlugin pass("probe-pass-ok",
                              read_bytes("tests/fixtures/plugin_probe_pass_start_short.wasm"));
    CHECK(pass.api_version() == kPassPluginAbiVersion);
    CHECK(pass.probe().abi.declared);
    CHECK(pass.probe().refusal(kPassPluginAbiVersion, "netvis_pass_abi_version").empty());
    reg.register_pass(std::make_unique<wasm::WasmPassPlugin>(
        "probe-pass-ok", read_bytes("tests/fixtures/plugin_probe_pass_start_short.wasm")));
    CHECK(reg.snapshot()->passes.count("probe-pass-ok") == 1);
  }

  reg.reset_to_builtins();
}

// ===========================================================================
// 10. The manifest gate: strict, full-width, required - for every loader.
// ===========================================================================
TEST_CASE("ABI v1: the manifest api_version gate is strict for every loader") {
  if (!wasm_available()) return;

  Registry& reg = Registry::instance();
  reg.reset_to_builtins();

  struct Bad { const char* label; const char* member; };
  const Bad kBad[] = {
      {"a wrong version", R"("api_version": 2,)"},
      {"zero", R"("api_version": 0,)"},
      {"missing", ""},
      {"a string", R"("api_version": "1",)"},
      {"a float", R"("api_version": 1.0,)"},
      {"negative", R"("api_version": -1,)"},
      {"a boolean", R"("api_version": true,)"},
      // 2^32 + 1: narrowing this to 32 bits yields 1, the host's version.
      {"an unsigned value that wraps to the host version", R"("api_version": 4294967297,)"},
  };

  for (const Bad& b : kBad) {
    CAPTURE(b.label);

    PluginDir op("gate_op", op_manifest(b.member), "plugin_ophandler.wasm");
    const std::string op_err = wasm::load_wasm_op_plugin(op.manifest);
    CHECK_MESSAGE(op_err.find("api_version") != std::string::npos,
                  "op loader error: " << op_err);
    CHECK(reg.resolve_op("matmul").origin == Origin::Builtin);

    PluginDir parser("gate_parser", parser_manifest(b.member), "plugin_toyparser.wasm");
    const size_t before = reg.snapshot()->parsers.size();
    const std::string parser_err = wasm::load_wasm_parser_plugin(parser.manifest);
    CHECK_MESSAGE(parser_err.find("api_version") != std::string::npos,
                  "parser loader error: " << parser_err);
    CHECK(reg.snapshot()->parsers.size() == before);

    // The declarative loader reads the same key through the same gate.
    PluginDir decl("gate_decl",
                   std::string("{") + b.member +
                       R"("name": "abi-freeze-decl", "ops": [{"name": "MyFreezeOp", "category": "Other"}]})",
                   "");
    const LoadedManifest lm = load_manifest_file(decl.manifest, /*register_into=*/true);
    CHECK_FALSE(lm.ok);
    CHECK(lm.error.find("api_version") != std::string::npos);
    CHECK(reg.resolve_op("myfreezeop").origin == Origin::Builtin);
  }

  reg.reset_to_builtins();
}

TEST_CASE("ABI v1: the manifest gate admits the right version through each loader") {
  if (!wasm_available()) return;

  Registry& reg = Registry::instance();
  reg.reset_to_builtins();

  PluginDir op("gate_ok_op", op_manifest(R"("api_version": 1,)"), "plugin_ophandler.wasm");
  CHECK(wasm::load_wasm_op_plugin(op.manifest).empty());
  CHECK(reg.resolve_op("matmul").origin == Origin::Wasm);
  reg.reset_to_builtins();

  PluginDir parser("gate_ok_parser", parser_manifest(R"("api_version": 1,)"),
                   "plugin_toyparser.wasm");
  const size_t before = reg.snapshot()->parsers.size();
  CHECK(wasm::load_wasm_parser_plugin(parser.manifest).empty());
  CHECK(reg.snapshot()->parsers.size() == before + 1);

  PluginDir decl("gate_ok_decl",
                 R"({"api_version": 1, "name": "abi-freeze-decl",
                     "ops": [{"name": "MyFreezeOp", "category": "Other"}]})",
                 "");
  const LoadedManifest lm = load_manifest_file(decl.manifest, /*register_into=*/true);
  CHECK(lm.ok);
  CHECK(reg.resolve_op("myfreezeop").origin == Origin::Declarative);

  reg.reset_to_builtins();
}

// ===========================================================================
// 11. End to end: a refused plugin leaves the built-in result standing.
// ===========================================================================
TEST_CASE("ABI v1: a wrong-ABI WASM op plugin does not replace the built-in cost") {
  if (!wasm_available()) return;

  Registry& reg = Registry::instance();
  reg.reset_to_builtins();
  const ir::Model model = make_matmul();

  const CostReport builtin = compute_cost(model, 0);
  REQUIRE(builtin.per_node.size() == 1);
  REQUIRE(builtin.per_node[0].flops_known);   // a real, known number to protect
  REQUIRE(builtin.per_node[0].flops == 48);   // 2 * |[2,4]| * K(3)

  const std::string current = R"("api_version": 1,)";

  SUBCASE("control: the matching plugin, loaded the same way, DOES take over MatMul") {
    PluginDir pd("e2e_ok", op_manifest(current), "plugin_ophandler.wasm");
    REQUIRE(wasm::load_wasm_op_plugin(pd.manifest).empty());
    CHECK(reg.resolve_op("matmul").origin == Origin::Wasm);
    const CostReport cr = compute_cost(model, 0);
    CHECK(cr.per_node[0].flops_known);
    CHECK(cr.per_node[0].flops == 1024);        // the plugin's number, not the built-in's
  }

  SUBCASE("a module declaring ABI v2 behind a manifest that says 1") {
    PluginDir pd("e2e_future", op_manifest(current), "plugin_ophandler_future_abi.wasm");
    const std::string why = wasm::load_wasm_op_plugin(pd.manifest);
    CHECK_MESSAGE(why.find("ABI v2") != std::string::npos, "loader said: " << why);
    CHECK(reg.resolve_op("matmul").origin == Origin::Builtin);
    const CostReport cr = compute_cost(model, 0);
    CHECK(cr.per_node[0].flops_known);          // previously flipped to unknown
    CHECK(cr.per_node[0].flops == builtin.per_node[0].flops);
  }

  SUBCASE("a module that declares no ABI version") {
    PluginDir pd("e2e_none", op_manifest(current), "plugin_ophandler_no_abi.wasm");
    const std::string why = wasm::load_wasm_op_plugin(pd.manifest);
    CHECK_MESSAGE(why.find("does not declare an ABI version") != std::string::npos,
                  "loader said: " << why);
    CHECK(reg.resolve_op("matmul").origin == Origin::Builtin);
    CHECK(compute_cost(model, 0).per_node[0].flops == 48);
  }

  SUBCASE("a module importing a host function this NetVis does not provide") {
    PluginDir pd("e2e_unknown", op_manifest(current), "plugin_ophandler_unknown_import.wasm");
    const std::string why = wasm::load_wasm_op_plugin(pd.manifest);
    CHECK_MESSAGE(why.find("op_from_a_newer_sdk") != std::string::npos, "loader said: " << why);
    CHECK(reg.resolve_op("matmul").origin == Origin::Builtin);
    CHECK(compute_cost(model, 0).per_node[0].flops == 48);
  }

  SUBCASE("a module importing a host function with the wrong signature") {
    PluginDir pd("e2e_badsig", op_manifest(current), "plugin_ophandler_bad_sig_import.wasm");
    const std::string why = wasm::load_wasm_op_plugin(pd.manifest);
    CHECK_MESSAGE(why.find("op_set_flops") != std::string::npos, "loader said: " << why);
    CHECK(reg.resolve_op("matmul").origin == Origin::Builtin);
    CHECK(compute_cost(model, 0).per_node[0].flops == 48);
  }

  SUBCASE("control: an op using op_input_const_ints loads and answers through the loader") {
    PluginDir pd("e2e_const", op_manifest(current), "plugin_ophandler_const_ints.wasm");
    REQUIRE(wasm::load_wasm_op_plugin(pd.manifest).empty());
    const CostReport cr = compute_cost(model, 0);
    CHECK(cr.per_node[0].flops_known);
    CHECK(cr.per_node[0].flops == 7);
  }

  reg.reset_to_builtins();
}

TEST_CASE("ABI v1: discovery reports a wrong-ABI WASM plugin as refused, not loaded") {
  if (!wasm_available()) return;

  // The Plugins panel reads LoadedManifest. A refused module must show up there as
  // not registered, with the reason - not as a loaded plugin that silently abstains.
  const fs::path root = fs::temp_directory_path() / "nv_abi_freeze_discovery";
  fs::remove_all(root);
  auto place = [&](const std::string& id, const char* fixture) {
    fs::create_directories(root / id);
    std::ofstream(root / id / "plugin.json") << op_manifest(R"("api_version": 1,)");
    const std::vector<uint8_t> bytes = read_bytes(std::string("tests/fixtures/") + fixture);
    std::ofstream wf((root / id / "plugin.wasm").string(), std::ios::binary);
    wf.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  };

  Registry& reg = Registry::instance();

  SUBCASE("a module declaring a future ABI") {
    place("future", "plugin_ophandler_future_abi.wasm");
    reg.reset_to_builtins();
    const auto manifests = discover_and_load_plugins(
        [](std::string_view, PluginKind) { return true; }, root.string());
    REQUIRE(manifests.size() == 1);
    CHECK(manifests[0].enabled);
    CHECK_FALSE(manifests[0].registered);
    CHECK(manifests[0].error.find("ABI v2") != std::string::npos);
    CHECK(reg.resolve_op("matmul").origin == Origin::Builtin);
  }

  SUBCASE("control: a matching module is registered") {
    place("ok", "plugin_ophandler.wasm");
    reg.reset_to_builtins();
    const auto manifests = discover_and_load_plugins(
        [](std::string_view, PluginKind) { return true; }, root.string());
    REQUIRE(manifests.size() == 1);
    CHECK(manifests[0].registered);
    CHECK(manifests[0].error.empty());
    CHECK(reg.resolve_op("matmul").origin == Origin::Wasm);
  }

  reg.reset_to_builtins();
  fs::remove_all(root);
}
