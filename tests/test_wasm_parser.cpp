// tests/test_wasm_parser.cpp — WASM ParserPlugin + host parser API (#10, Increment B).
//
// Loads plugin_toyparser.wasm through the real WasmEngine and asserts: can_parse
// claims, parse builds a model via the append-only command path (a graph + a
// recorded tensor), and the structural parse reads no weight payload. Registry
// routing (try_unknown_parsers) is covered by test_registry's fake parser; here we
// test the WASM adapter itself. Guarded by WasmEngine::enabled().
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "engine/plugin/ParserPlugin.h"
#include "engine/plugin/wasm/WasmParser.h"
#include "engine/plugin/wasm/WasmRuntime.h"
#include "ir/IR.h"

using namespace netvis;
using namespace netvis::plugin;
using namespace netvis::plugin::wasm;

namespace {
std::shared_ptr<const std::vector<uint8_t>> load_image(const char* path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return nullptr;
  return std::make_shared<std::vector<uint8_t>>(
      (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
// A tiny throwaway file for the parser to be handed (content unused by the toy).
std::string write_tmp(const std::string& stem, const std::string& bytes) {
  std::filesystem::path p = std::filesystem::temp_directory_path() / stem;
  std::ofstream o(p, std::ios::binary | std::ios::trunc);
  o.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return p.string();
}
}  // namespace

TEST_CASE("WASM parser: can_parse + parse builds a model, zero payload reads") {
  if (!WasmEngine::instance().enabled()) {
    WARN_MESSAGE(false, "built without NETVIS_ENABLE_WASM; skipping");
    return;
  }
  auto image = load_image("tests/fixtures/plugin_toyparser.wasm");
  if (!image) { WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py"); return; }

  std::string path = write_tmp("nv_toy_input.bin", std::string(64, '\0'));
  auto mf = MappedFile::open(path);
  REQUIRE(mf);

  std::unique_ptr<ParserPlugin> parser = make_wasm_parser("toy", image);
  REQUIRE(parser != nullptr);

  // can_parse: the toy always claims.
  CHECK(parser->can_parse(*mf, "bin") == true);

  // parse: builds a model via the append-only host commands.
  ByteReader::payload_read_counter() = 0;
  ProgressSink prog;
  auto res = parser->parse(*mf, prog);
  REQUIRE_MESSAGE(res, "wasm parser returned an error");
  const ir::Model& m = *res;
  // The toy created one graph and recorded one tensor.
  REQUIRE(m.graphs.size() >= 1);
  CHECK(m.graphs[0].initializers.size() == 1);
  // The structural parse reads NO weight payload (the toy records offset+len only;
  // host_read_range is never called, so the counter stays 0).
  CHECK(ByteReader::payload_read_counter() == 0);

  std::filesystem::remove(path);
}

TEST_CASE("WASM parser: priority is below built-ins, format self-labels Unknown") {
  if (!WasmEngine::instance().enabled()) { WARN_MESSAGE(false, "no WASM"); return; }
  auto image = load_image("tests/fixtures/plugin_toyparser.wasm");
  if (!image) { WARN_MESSAGE(false, "fixture missing"); return; }
  std::unique_ptr<ParserPlugin> parser = make_wasm_parser("toy", image);
  REQUIRE(parser != nullptr);
  CHECK(parser->format() == Format::Unknown);   // never hijacks a known format
  CHECK(parser->priority() > 0);                // below built-ins (which are <= 0-ish)
  CHECK(parser->api_version() == kParserPluginAbiVersion);
}
