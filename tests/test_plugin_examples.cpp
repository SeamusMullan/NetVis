// tests/test_plugin_examples.cpp — the shipped declarative plugin examples must
// stay valid (v0.6.2). CI loads every plugins/examples/<dir>/plugin.json through
// the REAL host loader and asserts it parses + every op compiles, so a broken
// example (bad DSL, unknown category, missing override) fails the build, not a
// user's first plugin experience. ctest runs from the source root.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "engine/CostModel.h"
#include "engine/plugin/OpHandler.h"
#include "engine/plugin/Registry.h"
#include "engine/plugin/declarative/Manifest.h"
#include "ir/IR.h"

using namespace netvis;
using namespace netvis::plugin;

TEST_CASE("plugin examples: every shipped manifest loads and all ops compile") {
  const std::vector<std::string> manifests = {
      "plugins/examples/my-ops/plugin.json",
      "plugins/examples/linear-layer/plugin.json",
      "plugins/examples/custom-norm/plugin.json",
      "plugins/examples/group-conv/plugin.json",
      "plugins/examples/color-theme/plugin.json",
  };
  for (const std::string& path : manifests) {
    CAPTURE(path);
    LoadedManifest lm = load_manifest_file(path, /*register_into=*/false);
    CHECK(lm.ok == true);
    CHECK(lm.error.empty());
    CHECK(lm.ops.size() > 0);
    for (const LoadedOp& op : lm.ops) {
      CAPTURE(op.op_name);
      CHECK(op.ok == true);        // every op parsed + its DSL compiled
      CHECK(op.error.empty());
    }
  }
}

TEST_CASE("plugin examples: linear-layer registers as Declarative and FLOP-counts") {
  // Register the example, then resolve its op and evaluate the DSL FLOP formula on a
  // concrete node — proving the example works end-to-end, not just parses.
  LoadedManifest lm =
      load_manifest_file("plugins/examples/linear-layer/plugin.json", true);
  REQUIRE(lm.ok);

  ir::Model m; m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  auto add = [&](const char* n, std::vector<int64_t> s) {
    ir::ValueInfo v; v.name = m.intern(n); v.dtype = ir::DType::F32;
    for (int64_t d : s) v.shape.push_back(d);
    g.values.push_back(v);
    return static_cast<uint32_t>(g.values.size() - 1);
  };
  uint32_t x = add("X", {8, 512}), w = add("W", {512, 1024}), y = add("Y", {8, 1024});
  ir::Node n; n.op_type = m.intern("FullyConnected");
  n.inputs.begin = 0; n.inputs.count = 2; g.edge_refs.push_back(x); g.edge_refs.push_back(w);
  n.outputs.begin = 2; n.outputs.count = 1; g.edge_refs.push_back(y);
  g.nodes.push_back(n);

  Registry& r = Registry::instance();
  OpResolution res = r.resolve_op("fullyconnected");
  CHECK(res.origin == Origin::Declarative);   // the example actually registered
  REQUIRE(res.handler != nullptr);
  OpContext ctx = r.make_context(m, g, g.nodes[0], {});
  CHECK(res.handler->category(ctx) == OpCategory::MatMul);
  FlopResult fr = res.handler->flops(ctx);
  CHECK(fr.known == true);
  // flops = 2 * O * K = 2 * (8*1024) * 512
  CHECK(fr.flops == static_cast<uint64_t>(2) * 8 * 1024 * 512);
}
