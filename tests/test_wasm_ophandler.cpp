// tests/test_wasm_ophandler.cpp — WASM OpHandler adapter (#10, Increment A).
//
// Loads the hand-encoded plugin_ophandler.wasm fixtures through the real WasmEngine
// and asserts: FLOPs flow through the handler, THE CLAMP turns a hostile category
// into Other (never OOB), and a structural op resolve leaves the payload counter at
// 0. All guarded by WasmEngine::enabled() so a no-WASM build stays green.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "engine/OpCategory.h"
#include "engine/plugin/OpHandler.h"
#include "engine/plugin/Registry.h"
#include "engine/plugin/wasm/WasmOpHandler.h"
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

// A 1-node model whose single op is `op_name` with a 2x2 input + output.
ir::Model make_one_node(const std::string& op_name) {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  auto add_val = [&](const char* nm) {
    uint32_t vi = static_cast<uint32_t>(g.values.size());
    ir::ValueInfo v; v.name = m.intern(nm); v.dtype = ir::DType::F32;
    v.shape.push_back(2); v.shape.push_back(2);
    g.values.push_back(std::move(v));
    return vi;
  };
  uint32_t a = add_val("A"), y = add_val("Y");
  ir::Node n; n.op_type = m.intern(op_name);
  n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(a); n.inputs.count = 1;
  n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(y); n.outputs.count = 1;
  g.values[y].producer = 0;
  g.nodes.push_back(std::move(n));
  return m;
}
}  // namespace

TEST_CASE("WASM op handler: flops flow through, category clamps, zero payload") {
  if (!WasmEngine::instance().enabled()) {
    WARN_MESSAGE(false, "built without NETVIS_ENABLE_WASM; skipping");
    return;
  }
  auto image = load_image("tests/fixtures/plugin_ophandler.wasm");
  if (!image) { WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py"); return; }

  ByteReader::payload_read_counter() = 0;

  ir::Model m = make_one_node("MyCustomConv");
  const ir::Graph& g = m.graphs[0];
  const ir::Node& node = g.nodes[0];

  WasmOpHandler h("test-op", image);
  Registry& reg = Registry::instance();
  OpContext ctx = reg.make_context(m, g, node, {});

  // FLOPs: the fixture emits a known 1024.
  FlopResult fr = h.flops(ctx);
  CHECK(fr.known == true);
  CHECK(fr.flops == 1024);

  // Category: the well-behaved fixture sets Conv (0).
  CHECK(h.category(ctx) == OpCategory::Conv);

  // A structural op resolve reads NO payload.
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("WASM op handler: hostile category is clamped to Other, no OOB") {
  if (!WasmEngine::instance().enabled()) { WARN_MESSAGE(false, "no WASM"); return; }
  auto image = load_image("tests/fixtures/plugin_ophandler_hostile.wasm");
  if (!image) { WARN_MESSAGE(false, "fixture missing"); return; }

  ir::Model m = make_one_node("EvilOp");
  const ir::Graph& g = m.graphs[0];
  WasmOpHandler h("test-hostile", image);
  OpContext ctx = Registry::instance().make_context(m, g, g.nodes[0], {});

  // op_set_category(9999) must be clamped -> Other (a valid, in-range enum; the
  // view does 1u<<cat + palette index, so an unclamped 9999 would be UB/OOB).
  OpCategory c = h.category(ctx);
  CHECK(c == OpCategory::Other);
  CHECK(static_cast<int>(c) <= static_cast<int>(OpCategory::Other));
}

TEST_CASE("WASM op handler: clamp_category unit bounds") {
  CHECK(clamp_category(-1) == OpCategory::Other);
  CHECK(clamp_category(9999) == OpCategory::Other);
  CHECK(clamp_category(0) == OpCategory::Conv);
  CHECK(clamp_category(static_cast<int32_t>(OpCategory::Other)) == OpCategory::Other);
}
