// SPDX-License-Identifier: Apache-2.0
// tests/test_plugin_abi_freeze.cpp — the plugin ABI v1 freeze + version negotiation (#113).
//
// tests/test_sdk_abi.cpp already static_asserts that the SHARED VALUES in
// plugins/sdk/netvis_plugin.h match the host C++ contracts. That catches drift in
// what both sides already agree on; it cannot catch the two ways a frozen ABI
// actually breaks:
//
//   1. The SURFACE changes. A renamed import, a dropped macro, a retired
//      enumerator - all compile fine on the host, and all break a plugin binary
//      that was compiled against the old header. So the surface is written down in
//      plugins/sdk/abi-v1-surface.txt, and the first two cases below re-derive it
//      from the header and from the host's own link tables and demand all three
//      agree. Changing the header now requires editing the freeze file too, which
//      is the reviewable moment where someone asks "does this need an ABI bump?".
//
//   2. VERSION NEGOTIATION does not actually reject. The promise in
//      docs/plugin-abi.md is that a plugin declaring an ABI the host was not built
//      for is refused CLEANLY - no half-registration, no partial answers, and the
//      built-in result still stands. The remaining cases prove that for all three
//      plugin kinds, over both the C++ registration path and the real WASM ABI
//      gate: a plugin from the future, and a plugin so old it declares no version
//      at all.
#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "engine/OpCategory.h"
#include "engine/plugin/OpHandler.h"
#include "engine/plugin/ParserPlugin.h"
#include "engine/plugin/PassPlugin.h"
#include "engine/plugin/Registry.h"
#include "engine/plugin/wasm/WasmOpHandler.h"
#include "engine/plugin/wasm/WasmParser.h"
#include "engine/plugin/wasm/WasmRuntime.h"
#include "ir/IR.h"

using namespace netvis;
using namespace netvis::plugin;

namespace {

// --- Surface extraction ------------------------------------------------------
// Deliberately line-oriented and dumb: it must be obvious that this reads the same
// thing a plugin author reads, and it must not need a C parser to stay honest.

std::string read_file_or_empty(const char* path) {
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
  size_t b = s.find_first_not_of(" \t");
  if (b == std::string::npos) return {};
  size_t e = s.find_last_not_of(" \t");
  return s.substr(b, e - b + 1);
}

// Comments are NOT part of the ABI: strip them first so rewording a doc comment
// never fails this test, while renaming a macro always does.
std::string strip_comment(const std::string& line) {
  size_t block = line.find("/*");
  size_t slash = line.find("//");
  size_t cut = std::min(block == std::string::npos ? line.size() : block,
                        slash == std::string::npos ? line.size() : slash);
  return trim(line.substr(0, cut));
}

// The public surface of the SDK header, as sorted unique `<kind> <name>[ = <value>]`
// entries. Mirrors the format documented at the top of abi-v1-surface.txt.
std::set<std::string> extract_sdk_surface(const std::string& header_text) {
  // A function-like macro is frozen by NAME only: its body expands to
  // toolchain attributes (import_module / export_name) that a non-clang guest
  // toolchain is expected to spell differently.
  static const std::regex re_define(
      R"RX(^#\s*define\s+((?:NV_|NETVIS_)[A-Za-z0-9_]*)\s*(\([^)]*\))?(.*)$)RX");
  static const std::regex re_import(R"RX(NV_IMPORT\("([^"]+)"\s*,\s*"([^"]+)"\))RX");
  static const std::regex re_enum(R"RX(\b(NV_[A-Z0-9_]+)\s*=\s*(-?[0-9]+))RX");
  static const std::regex re_type(R"RX(^\}\s*(nv_[A-Za-z0-9_]+_t)\s*;)RX");

  std::set<std::string> surface;
  for (const std::string& raw : split_lines(header_text)) {
    const std::string line = strip_comment(raw);
    if (line.empty()) continue;

    std::smatch m;
    if (std::regex_search(line, m, re_define)) {
      const std::string name = m[1].str();
      if (m[2].matched) {
        surface.insert("define " + name + "()");
      } else {
        const std::string value = trim(m[3].str());
        surface.insert(value.empty() ? "define " + name
                                     : "define " + name + " = " + value);
      }
    }
    for (auto it = std::sregex_iterator(line.begin(), line.end(), re_import);
         it != std::sregex_iterator(); ++it) {
      surface.insert("import " + (*it)[1].str() + " " + (*it)[2].str());
    }
    for (auto it = std::sregex_iterator(line.begin(), line.end(), re_enum);
         it != std::sregex_iterator(); ++it) {
      surface.insert("enum " + (*it)[1].str() + " = " + (*it)[2].str());
    }
    if (std::regex_search(line, m, re_type)) surface.insert("type " + m[1].str());
  }
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

std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (const std::string& e : v) { if (!s.empty()) s += ", "; s += e; }
  return s;
}

std::vector<std::string> difference(const std::set<std::string>& a,
                                    const std::set<std::string>& b) {
  std::vector<std::string> out;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

// Every (module, name) pair the host actually links into a guest module, read off
// the link tables themselves. `ns` is assigned once per link function and every
// entry below it uses it, so tracking the last assignment is enough.
std::set<std::string> extract_host_imports() {
  static const char* kLinkSites[] = {
      "src/engine/plugin/wasm/WasmOpHandler.cpp",  // module "netvis_op"
      "src/engine/plugin/wasm/WasmParser.cpp",     // module "netvis" (parser facet)
      "src/engine/plugin/wasm/WasmHost.cpp",       // module "netvis" (pass facet)
  };
  static const std::regex re_ns(R"RX(const char\*\s*ns\s*=\s*"([^"]+)")RX");
  static const std::regex re_short(R"RX(\bL\(\s*"([^"]+)")RX");
  static const std::regex re_long(
      R"RX(m3_LinkRawFunctionEx\(\s*mod\s*,\s*ns\s*,\s*"([^"]+)")RX");

  std::set<std::string> out;
  for (const char* path : kLinkSites) {
    std::string ns;
    for (const std::string& raw : split_lines(read_file_or_empty(path))) {
      std::smatch m;
      if (std::regex_search(raw, m, re_ns)) { ns = m[1].str(); continue; }
      if (ns.empty()) continue;
      if (std::regex_search(raw, m, re_short) ||
          std::regex_search(raw, m, re_long)) {
        out.insert("import " + ns + " " + m[1].str());
      }
    }
  }
  return out;
}

// --- Stub plugins that declare an ABI the host was not built for -------------

class FutureOpHandler final : public OpHandler {
 public:
  OpCategory category(const OpContext&) const override { return OpCategory::Conv; }
  uint32_t api_version() const override { return kOpHandlerAbiVersion + 1; }
};

class FutureParser final : public ParserPlugin {
 public:
  Format format() const override { return Format::Unknown; }
  std::string_view display_name() const override { return "abi-freeze-future-parser"; }
  bool can_parse(const MappedFile&, const std::string&) const override { return true; }
  int priority() const override { return 1000; }
  Result<ir::Model> parse(const MappedFile&, ProgressSink&) const override {
    return err("a refused parser must never be asked to parse");
  }
  uint32_t api_version() const override { return kParserPluginAbiVersion + 1; }
};

class FuturePass final : public PassPlugin {
 public:
  std::string_view display_name() const override { return "abi-freeze-future-pass"; }
  PassResult run(const ir::Model&, uint32_t, const CostReport&) const override {
    return {};
  }
  uint32_t api_version() const override { return kPassPluginAbiVersion + 1; }
};

std::shared_ptr<const std::vector<uint8_t>> load_image(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return nullptr;
  return std::make_shared<std::vector<uint8_t>>(
      (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// A 1-node model whose op is `op_name`, enough to build an OpContext.
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

}  // namespace

// ---------------------------------------------------------------------------
// 1. The SDK header surface is frozen.
// ---------------------------------------------------------------------------
TEST_CASE("ABI v1: the SDK header surface matches the committed freeze") {
  const std::string header = read_file_or_empty("plugins/sdk/netvis_plugin.h");
  const std::string frozen_text = read_file_or_empty("plugins/sdk/abi-v1-surface.txt");
  REQUIRE_MESSAGE(!header.empty(), "plugins/sdk/netvis_plugin.h not readable");
  REQUIRE_MESSAGE(!frozen_text.empty(), "plugins/sdk/abi-v1-surface.txt not readable");

  const std::set<std::string> actual = extract_sdk_surface(header);
  const std::set<std::string> frozen = read_frozen_surface(frozen_text);

  const std::vector<std::string> added = difference(actual, frozen);
  const std::vector<std::string> removed = difference(frozen, actual);

  // Additions and removals are reported separately because they mean different
  // things: an addition may be a compatible extension, a removal or a changed
  // value never is. See docs/plugin-abi.md ("Compatibility promise").
  CHECK_MESSAGE(added.empty(),
                "SDK surface entries NOT in plugins/sdk/abi-v1-surface.txt: "
                    << join(added)
                    << " -- add the line to the freeze file, and classify the "
                       "change against the table in docs/plugin-abi.md "
                       "(\"Compatibility promise\"): some surface changes keep "
                       "ABI v1, others force a version bump.");
  CHECK_MESSAGE(removed.empty(),
                "frozen ABI v1 surface entries MISSING from the header: "
                    << join(removed)
                    << " -- removing a name, renumbering an enumerator or "
                       "tightening a cap breaks every plugin already compiled "
                       "against ABI v1.");
}

// ---------------------------------------------------------------------------
// 2. The host links exactly the import set the header promises.
// ---------------------------------------------------------------------------
TEST_CASE("ABI v1: host link tables and the SDK header declare the same imports") {
  const std::string header = read_file_or_empty("plugins/sdk/netvis_plugin.h");
  REQUIRE_MESSAGE(!header.empty(), "plugins/sdk/netvis_plugin.h not readable");

  std::set<std::string> header_imports;
  for (const std::string& e : extract_sdk_surface(header)) {
    if (e.rfind("import ", 0) == 0) header_imports.insert(e);
  }
  const std::set<std::string> host_imports = extract_host_imports();

  REQUIRE_MESSAGE(!host_imports.empty(),
                  "no host imports found -- the link tables moved or were reshaped, "
                  "so this guard is no longer reading them");

  // Only the NAMES are compared, not the wasm3 signature strings: those are a
  // different notation from the C declarations and are audited against them in the
  // header's own comment. A name mismatch is the drift that actually strands a
  // plugin at load time with an unresolved import.
  CHECK_MESSAGE(difference(header_imports, host_imports).empty(),
                "declared in the SDK header but NOT linked by the host: "
                    << join(difference(header_imports, host_imports))
                    << " -- a guest importing one of these fails to instantiate.");
  CHECK_MESSAGE(difference(host_imports, header_imports).empty(),
                "linked by the host but NOT declared in the SDK header: "
                    << join(difference(host_imports, header_imports))
                    << " -- plugin authors have no supported way to reach it, so "
                       "either publish it in the header or stop linking it.");
}

// ---------------------------------------------------------------------------
// 3. Registration refuses a future ABI, for each of the three plugin kinds.
// ---------------------------------------------------------------------------
TEST_CASE("ABI v1: the Registry refuses a plugin declaring a future ABI") {
  Registry& reg = Registry::instance();
  reg.reset_to_builtins();

  // An op key no built-in recognizes, so nothing here depends on the
  // shadow/override rules - only on the version check.
  const std::string key = "netvis_abi_freeze_probe";
  const size_t parsers_before = reg.snapshot()->parsers.size();
  const size_t passes_before = reg.snapshot()->passes.size();

  reg.register_op_handler(key, "", std::make_unique<FutureOpHandler>(), Origin::Wasm,
                          /*override_flag=*/true, "abi-freeze-future-op");
  reg.register_parser(std::make_unique<FutureParser>());
  reg.register_pass(std::make_unique<FuturePass>());

  auto tbl = reg.snapshot();

  SUBCASE("the op handler never enters the table") {
    CHECK(tbl->op_by_key.count(key) == 0);
    // And resolution still lands on the built-in catch-all, so the app's answer
    // for that op is the honest built-in one, not a hole.
    OpResolution r = reg.resolve_op(key);
    CHECK(r.handler != nullptr);
    CHECK(r.origin == Origin::Builtin);
  }
  SUBCASE("the parser never enters the table") {
    CHECK(tbl->parsers.size() == parsers_before);
  }
  SUBCASE("the pass never enters the table") {
    CHECK(tbl->passes.size() == passes_before);
    CHECK(tbl->passes.count("abi-freeze-future-pass") == 0);
  }

  reg.reset_to_builtins();
}

// ---------------------------------------------------------------------------
// 4. The WASM ABI gate rejects a future version and a missing export.
// ---------------------------------------------------------------------------
TEST_CASE("ABI v1: a WASM op plugin from the future is refused, answer stays built-in") {
  if (!wasm::WasmEngine::instance().enabled()) return;  // no-WASM build: nothing to gate

  ir::Model model = make_one_node("Conv");
  Registry& reg = Registry::instance();
  OpContext ctx = reg.make_context(model, model.graphs[0], model.graphs[0].nodes[0],
                                   normalize_op_key("Conv"));
  const OpCategory expected = ctx.default_category();

  SUBCASE("netvis_op_abi_version returns a version the host was not built for") {
    auto image = load_image("tests/fixtures/plugin_ophandler_future_abi.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("future-abi", image);

    // The well-formed fixture answers Conv through op_set_category; this one is
    // byte-identical apart from the version it reports, so an unchanged answer
    // here is specifically the ABI gate firing and not the module failing to run.
    CHECK(h.category(ctx) == expected);
    CHECK(h.diag().loaded);            // the module itself is fine ...
    CHECK(h.diag().abi_mismatch);      // ... it is the declared version that is not
    CHECK(h.flops(ctx).known == false);
  }

  SUBCASE("netvis_op_abi_version is not exported at all") {
    auto image = load_image("tests/fixtures/plugin_ophandler_no_abi.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("no-abi", image);
    CHECK(h.category(ctx) == expected);
    CHECK(h.diag().abi_mismatch);
    CHECK(h.flops(ctx).known == false);
  }

  SUBCASE("the matching fixture still answers, so the gate is not refusing everything") {
    auto image = load_image("tests/fixtures/plugin_ophandler.wasm");
    REQUIRE(image != nullptr);
    wasm::WasmOpHandler h("ok-abi", image);
    CHECK(h.category(ctx) == OpCategory::Conv);
    CHECK(h.diag().abi_mismatch == false);
    CHECK(h.flops(ctx).known);
  }
}

TEST_CASE("ABI v1: a WASM parser plugin from the future never claims a file") {
  if (!wasm::WasmEngine::instance().enabled()) return;  // no-WASM build

  // A file no built-in format claims, so reaching a plugin parser is the only way
  // it could ever be parsed.
  const std::string path =
      (std::filesystem::temp_directory_path() / "nv_abi_freeze_input.bin").string();
  {
    std::ofstream f(path, std::ios::binary);
    f << std::string(64, '\0');
  }
  auto mf = MappedFile::open(path);
  REQUIRE(mf);

  SUBCASE("netvis_parser_abi_version returns a version the host was not built for") {
    auto image = load_image("tests/fixtures/plugin_toyparser_future_abi.wasm");
    REQUIRE(image != nullptr);
    std::unique_ptr<ParserPlugin> parser = wasm::make_wasm_parser("future-abi", image);
    REQUIRE(parser != nullptr);

    // The matching fixture claims this exact file (see test_wasm_parser.cpp), so a
    // refusal here is the ABI gate and not the sniff failing.
    CHECK(parser->can_parse(*mf, "bin") == false);

    // And parse() refuses too, rather than trusting that nobody calls it after
    // can_parse said no.
    ProgressSink prog;
    auto res = parser->parse(*mf, prog);
    CHECK(!res);
  }

  SUBCASE("the matching fixture still claims it, so the gate is not refusing everything") {
    auto image = load_image("tests/fixtures/plugin_toyparser.wasm");
    REQUIRE(image != nullptr);
    std::unique_ptr<ParserPlugin> parser = wasm::make_wasm_parser("ok-abi", image);
    REQUIRE(parser != nullptr);
    CHECK(parser->can_parse(*mf, "bin") == true);
  }

  std::filesystem::remove(path);
}
