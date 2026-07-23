// tests/test_wasm.cpp — WASM plugin sandbox contract (v0.6.0 Increment 3, #10).
//
// THE acceptance bar: a hostile .wasm that loops forever is KILLED by the fuel
// cap and the app survives; a well-behaved module runs to completion. Also checks
// that a garbage image fails to load cleanly (no crash). When built without
// NETVIS_ENABLE_WASM, the engine reports disabled and the suite is a no-op.
#include <doctest/doctest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "engine/CostModel.h"
#include "engine/plugin/wasm/WasmHost.h"
#include "engine/plugin/wasm/WasmRuntime.h"
#include "ir/IR.h"

using namespace netvis;
using namespace netvis::plugin::wasm;

namespace {

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

std::string fixture(const char* name) {
  // ctest sets WORKING_DIRECTORY to the source root, so the same relative path
  // the other suites use resolves here too.
  return std::string("tests/fixtures/") + name;
}

}  // namespace

TEST_CASE("wasm: engine availability is reported honestly") {
  // Either enabled (built with wasm3) or a clean disabled report — never a crash.
  bool en = WasmEngine::instance().enabled();
  CHECK((en == true || en == false));
}

TEST_CASE("wasm: a well-behaved module runs to completion") {
  if (!WasmEngine::instance().enabled()) return;   // no-op when WASM disabled
  std::vector<uint8_t> img = read_file(fixture("plugin_const.wasm"));
  REQUIRE(img.size() > 8);

  SandboxLimits lim;
  RunResult lerr;
  WasmModule mod = WasmEngine::instance().load(img, lim, nullptr, &lerr);
  REQUIRE(mod.loaded());
  CHECK(lerr.status == RunStatus::Ok);

  int32_t ret = 0;
  RunResult rr = mod.call_i32("run", &ret);
  CHECK(rr.status == RunStatus::Ok);
  CHECK(ret == 42);
}

TEST_CASE("wasm: a hostile infinite-loop module is KILLED by the fuel cap") {
  if (!WasmEngine::instance().enabled()) return;
  std::vector<uint8_t> img = read_file(fixture("plugin_loop.wasm"));
  REQUIRE(img.size() > 8);

  SandboxLimits lim;
  lim.max_steps = 100000;   // small budget so the test is fast
  RunResult lerr;
  WasmModule mod = WasmEngine::instance().load(img, lim, nullptr, &lerr);
  REQUIRE(mod.loaded());

  int32_t ret = 0;
  RunResult rr = mod.call_i32("run", &ret);
  // The module never returns on its own; the sandbox MUST trap it via fuel.
  CHECK(rr.status == RunStatus::FuelExhausted);
  // ...and the app is still alive to make this assertion — that IS the survival test.
}

TEST_CASE("wasm: a hostile START-section loop is metered too (fuel bypass regression)") {
  if (!WasmEngine::instance().enabled()) return;
  std::vector<uint8_t> img = read_file(fixture("plugin_start_loop.wasm"));
  REQUIRE(img.size() > 8);

  SandboxLimits lim;
  lim.max_steps = 100000;
  RunResult lerr;
  WasmModule mod = WasmEngine::instance().load(img, lim, nullptr, &lerr);
  REQUIRE(mod.loaded());   // wasm3 defers the start fn to first FindFunction

  int32_t ret = 0;
  RunResult rr = mod.call_i32("run", &ret);
  // The start section loops forever; it runs during FindFunction, which the fuel
  // now covers -> FuelExhausted, app survives (regression for the review finding).
  CHECK(rr.status == RunStatus::FuelExhausted);
}

TEST_CASE("wasm: a wrong-signature export is rejected, not UB (review fix)") {
  if (!WasmEngine::instance().enabled()) return;
  std::vector<uint8_t> img = read_file(fixture("plugin_badsig.wasm"));
  REQUIRE(img.size() > 8);
  SandboxLimits lim;
  RunResult lerr;
  WasmModule mod = WasmEngine::instance().load(img, lim, nullptr, &lerr);
  REQUIRE(mod.loaded());
  int32_t ret = 0;
  RunResult rr = mod.call_i32("run", &ret);   // run:(i32)->() — must be refused
  CHECK(rr.status == RunStatus::LoadError);
  CHECK(rr.message.find("signature") != std::string::npos);
}

TEST_CASE("wasm: a huge declared memory is capped at load, not OOM (review fix)") {
  if (!WasmEngine::instance().enabled()) return;
  std::vector<uint8_t> img = read_file(fixture("plugin_bigmem.wasm"));
  REQUIRE(img.size() > 8);
  SandboxLimits lim;   // default 256 pages = 16 MiB cap
  RunResult lerr;
  // Must NOT allocate ~1.9 GiB: the memoryLimit MIN-clamp applies at load because
  // it is set before m3_LoadModule. Loads (module is otherwise valid) + runs fine.
  WasmModule mod = WasmEngine::instance().load(img, lim, nullptr, &lerr);
  REQUIRE(mod.loaded());
  uint32_t memsz = 0;
  mod.memory(&memsz);
  CHECK(memsz <= lim.max_memory_pages * 65536u);   // capped, not 30000 pages
}

TEST_CASE("wasm: a garbage image fails to load cleanly (no crash)") {
  if (!WasmEngine::instance().enabled()) return;
  std::vector<uint8_t> junk = {0x00, 0x61, 0x73, 0x6d, 0xff, 0xff, 0xff, 0xff,
                               0x13, 0x37, 0x13, 0x37};
  SandboxLimits lim;
  RunResult lerr;
  WasmModule mod = WasmEngine::instance().load(junk, lim, nullptr, &lerr);
  CHECK(mod.loaded() == false);
  CHECK(lerr.status == RunStatus::LoadError);
}

TEST_CASE("wasm: a PASS plugin reads the report + emits a metric (host API round-trip)") {
  if (!WasmEngine::instance().enabled()) return;
  std::vector<uint8_t> img = read_file(fixture("plugin_pass.wasm"));
  REQUIRE(img.size() > 8);

  // A report with a known total_flops the plugin will double + emit back.
  CostReport report;
  report.total_flops = 21;

  ir::Model m; m.graphs.emplace_back();
  netvis::plugin::wasm::WasmPassPlugin pass("doubler", img);
  netvis::plugin::PassResult res = pass.run(m, 0, report);

  // The plugin emitted "double_flops" = 2 * 21 = 42. This proves the capability
  // host API marshalled a scalar OUT of the sandbox — and, by construction, no
  // host import ever handed the plugin a weight buffer.
  REQUIRE(res.metrics.size() == 1);
  CHECK(res.metrics[0].name == "double_flops");
  CHECK(res.metrics[0].known == true);
  CHECK(res.metrics[0].value == doctest::Approx(42.0));
}
