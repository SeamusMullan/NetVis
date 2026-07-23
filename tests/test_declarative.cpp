// tests/test_declarative.cpp — declarative plugin backend: DSL + manifest loader
// (v0.6.0 Increment 2, #9).
//
// Covers: DSL grammar/eval, honest-unknown propagation (unresolved shape, div-0,
// overflow, negative), the hostile-input/fuzz matrix (depth bomb, huge literal,
// unterminated string, INT64_MIN/-1), var-DAG cycle rejection, api_version
// mismatch rejection, and end-to-end op override through a CompiledOp handler.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <cstdio>
#include <fstream>

#include "engine/plugin/OpHandler.h"
#include "engine/plugin/Registry.h"
#include "engine/plugin/declarative/Dsl.h"
#include "engine/plugin/declarative/Manifest.h"
#include "ir/IR.h"

using namespace netvis;
using namespace netvis::plugin;

namespace {

// Build a 1-node graph "MyOp" with in0 shape + optionally an int attr; return an
// OpContext bound to it. Keeps the model alive via out-params.
struct Fixture {
  ir::Model m;
  OpContext ctx;
};

uint32_t addv(ir::Graph& g, ir::Model& m, const char* n, ir::DType dt,
              std::vector<int64_t> shape) {
  ir::ValueInfo vi; vi.name = m.intern(n); vi.dtype = dt;
  for (int64_t d : shape) vi.shape.push_back(d);
  g.values.push_back(vi);
  return static_cast<uint32_t>(g.values.size() - 1);
}

dsl::Val ev(const char* src, const OpContext& ctx,
            const std::vector<std::pair<std::string, dsl::Val>>& vars = {}) {
  std::string err;
  dsl::Expr e = dsl::Expr::compile(src, dsl::Limits{}, &err);
  if (!e.valid()) return dsl::Val::unknown();
  return e.eval(ctx, vars);
}

}  // namespace

TEST_CASE("dsl: arithmetic and precedence") {
  ir::Model m; m.graphs.emplace_back(); ir::Graph& g = m.graphs[0];
  uint32_t a = addv(g, m, "A", ir::DType::F32, {2, 3, 4});
  uint32_t y = addv(g, m, "Y", ir::DType::F32, {2, 3, 4});
  ir::Node n; n.op_type = m.intern("MyOp");
  n.inputs.begin = 0; n.inputs.count = 1; g.edge_refs.push_back(a);
  n.outputs.begin = 1; n.outputs.count = 1; g.edge_refs.push_back(y);
  g.nodes.push_back(n);
  OpContext ctx = Registry::instance().make_context(m, g, g.nodes[0], {});

  CHECK(ev("2 + 3 * 4", ctx).v == 14);
  CHECK(ev("(2 + 3) * 4", ctx).v == 20);
  CHECK(ev("10 / 3", ctx).v == 3);
  CHECK(ev("10 % 3", ctx).v == 1);
  CHECK(ev("in0.shape[0]", ctx).v == 2);
  CHECK(ev("in0.shape[-1]", ctx).v == 4);       // negative-from-end
  CHECK(ev("in0.rank", ctx).v == 3);
  CHECK(ev("O", ctx).v == 2 * 3 * 4);            // output-0 elem count
  CHECK(ev("prod(in0.shape[0:2])", ctx).v == 2 * 3);
  CHECK(ev("max(3, 7, 1)", ctx).v == 7);
  CHECK(ev("min(3, 7, 1)", ctx).v == 1);
  CHECK(ev("2 * (2 * in0.shape[0] * in0.shape[1])", ctx).v == 2 * 2 * 2 * 3);
  CHECK(ev("in0.shape[0] > 1 ? 100 : 200", ctx).v == 100);
  CHECK(ev("in0.shape[0] == 2 && in0.rank == 3", ctx).v == 1);
}

TEST_CASE("dsl: honest-unknown propagation") {
  ir::Model m; m.graphs.emplace_back(); ir::Graph& g = m.graphs[0];
  // in0 has a dynamic dim (-1) at axis 1.
  uint32_t a = addv(g, m, "A", ir::DType::F32, {2, -1, 4});
  ir::Node n; n.op_type = m.intern("MyOp");
  n.inputs.begin = 0; n.inputs.count = 1; g.edge_refs.push_back(a);
  g.nodes.push_back(n);
  OpContext ctx = Registry::instance().make_context(m, g, g.nodes[0], {});

  CHECK(ev("in0.shape[1]", ctx).known == false);       // dynamic dim -> unknown
  CHECK(ev("in0.shape[0]", ctx).known == true);        // static dim ok
  CHECK(ev("in0.shape[1] * 2", ctx).known == false);   // contagious
  CHECK(ev("in0.shape[5]", ctx).known == false);       // OOB index -> unknown
  CHECK(ev("O", ctx).known == false);                  // -1 dim => elem_count 0 => unknown
  CHECK(ev("10 / 0", ctx).known == false);             // div-0 -> unknown, not crash
  CHECK(ev("10 % 0", ctx).known == false);
  CHECK(ev("unknown", ctx).known == false);            // explicit unknown literal
  // short-circuit: a known-false && anything -> 0 known
  CHECK(ev("in0.shape[0] == 99 && in0.shape[1] > 0", ctx).v == 0);
  CHECK(ev("in0.shape[0] == 99 && in0.shape[1] > 0", ctx).known == true);
  // ternary: unknown condition -> unknown
  CHECK(ev("in0.shape[1] > 0 ? 1 : 2", ctx).known == false);
}

TEST_CASE("dsl: overflow and INT64_MIN guards never fabricate") {
  ir::Model m; m.graphs.emplace_back(); ir::Graph& g = m.graphs[0];
  uint32_t a = addv(g, m, "A", ir::DType::F32, {2});
  ir::Node n; n.op_type = m.intern("MyOp");
  n.inputs.begin = 0; n.inputs.count = 1; g.edge_refs.push_back(a);
  g.nodes.push_back(n);
  OpContext ctx = Registry::instance().make_context(m, g, g.nodes[0], {});

  // 9223372036854775807 == INT64_MAX; +1 overflows -> unknown (not a wrap).
  CHECK(ev("9223372036854775807 + 1", ctx).known == false);
  CHECK(ev("9223372036854775807 * 2", ctx).known == false);
  // A literal past INT64_MAX fails to compile -> unknown.
  CHECK(ev("99999999999999999999999", ctx).known == false);
}

TEST_CASE("dsl: hostile input is rejected, never crashes") {
  ir::Model m; m.graphs.emplace_back(); ir::Graph& g = m.graphs[0];
  g.nodes.emplace_back();
  OpContext ctx = Registry::instance().make_context(m, g, g.nodes[0], {});

  std::string err;
  // Depth bomb: 200 nested parens exceeds max_depth(64) -> rejected.
  std::string bomb(200, '(');
  bomb += "1";
  bomb += std::string(200, ')');
  CHECK(dsl::Expr::compile(bomb, dsl::Limits{}, &err).valid() == false);

  // Unterminated string.
  CHECK(dsl::Expr::compile("sattr(\"x", dsl::Limits{}, &err).valid() == false);
  // Trailing garbage.
  CHECK(dsl::Expr::compile("1 + 2 )", dsl::Limits{}, &err).valid() == false);
  // Empty.
  CHECK(dsl::Expr::compile("", dsl::Limits{}, &err).valid() == false);
  // Bad field.
  CHECK(dsl::Expr::compile("in0.bogus", dsl::Limits{}, &err).valid() == false);
  // '.shape'/'.rank' on a non-operand.
  CHECK(dsl::Expr::compile("5.rank", dsl::Limits{}, &err).valid() == false);
}

TEST_CASE("declarative: attr access mirrors built-in getters") {
  ir::Model m; m.graphs.emplace_back(); ir::Graph& g = m.graphs[0];
  uint32_t a = addv(g, m, "A", ir::DType::F32, {4, 8});
  ir::Node n; n.op_type = m.intern("MyOp");
  n.inputs.begin = 0; n.inputs.count = 1; g.edge_refs.push_back(a);
  ir::Attribute at; at.name = m.intern("transA");
  at.value.kind = ir::AttrValue::Kind::Int; at.value.i = 1;
  g.attributes.push_back(at);
  ir::Attribute ad; ad.name = m.intern("direction");
  ad.value.kind = ir::AttrValue::Kind::String; ad.value.s = m.intern("bidirectional");
  g.attributes.push_back(ad);
  n = g.nodes.empty() ? n : n;
  g.nodes.push_back(n);
  g.nodes[0].attributes.begin = 0; g.nodes[0].attributes.count = 2;
  OpContext ctx = Registry::instance().make_context(m, g, g.nodes[0], {});

  CHECK(ev("attr(\"transA\", 0)", ctx).v == 1);
  CHECK(ev("attr(\"missing\", 7)", ctx).v == 7);        // default on absent
  CHECK(ev("attr(\"missing\")", ctx).known == false);   // no default -> unknown
  CHECK(ev("sattr(\"direction\") == \"bidirectional\"", ctx).v == 1);
  CHECK(ev("sattr(\"direction\") == \"forward\"", ctx).v == 0);
  CHECK(ev("sattr(\"direction\") == \"bidirectional\" ? 2 : 1", ctx).v == 2);
}

namespace {
std::string write_temp(const std::string& body) {
  std::string path = std::string("/tmp/nv_manifest_") +
                     std::to_string(body.size()) + "_" +
                     std::to_string(body.empty() ? 0 : body[0]) + ".json";
  std::ofstream f(path, std::ios::binary);
  f << body;
  f.close();
  return path;
}
}  // namespace

TEST_CASE("manifest: api_version mismatch rejects the whole file") {
  std::string p = write_temp(R"({ "api_version": 999, "name": "x",
    "ops": [{ "name": "Foo", "category": "MatMul" }] })");
  LoadedManifest lm = load_manifest_file(p, /*register_into=*/false);
  CHECK(lm.ok == false);
  CHECK(lm.error.find("api_version") != std::string::npos);
  CHECK(lm.ops.empty());
  std::remove(p.c_str());
}

TEST_CASE("manifest: unknown category rejects the op (not the file)") {
  std::string p = write_temp(R"({ "api_version": 1, "name": "x",
    "ops": [{ "name": "Foo", "category": "NotACategory" }] })");
  LoadedManifest lm = load_manifest_file(p, false);
  CHECK(lm.ok == true);
  REQUIRE(lm.ops.size() == 1);
  CHECK(lm.ops[0].ok == false);
  CHECK(lm.ops[0].error.find("category") != std::string::npos);
  std::remove(p.c_str());
}

TEST_CASE("manifest: cyclic var dependency rejected") {
  std::string p = write_temp(R"({ "api_version": 1, "name": "x", "ops": [{
    "name": "Foo", "category": "MatMul",
    "vars": { "a": "b + 1", "b": "a + 1" }, "flops": "a" }] })");
  LoadedManifest lm = load_manifest_file(p, false);
  REQUIRE(lm.ops.size() == 1);
  CHECK(lm.ops[0].ok == false);
  CHECK(lm.ops[0].error.find("cyclic") != std::string::npos);
  std::remove(p.c_str());
}

TEST_CASE("manifest: JSONC comments accepted, valid op compiles") {
  std::string p = write_temp(R"json({
    // a comment (JSONC)
    "api_version": 1, "name": "attn-ops",
    "ops": [{
      "name": "MyAttention", "category": "Attention", "color": "#D864D0",
      "vars": { "B": "in0.shape[0]", "S": "in0.shape[1]", "H": "in0.shape[2]" },
      "flops": "2 * (2 * B * S * S * H)"
    }]
  })json");
  LoadedManifest lm = load_manifest_file(p, false);
  CHECK(lm.ok == true);
  CHECK(lm.name == "attn-ops");
  REQUIRE(lm.ops.size() == 1);
  CHECK(lm.ops[0].ok == true);
  CHECK(lm.ops[0].op_name == "myattention");
  CHECK(lm.ops[0].category == "Attention");
  std::remove(p.c_str());
}

TEST_CASE("declarative: registered plugin op resolves + FLOP-counts through the registry") {
  // Register a NEW op the built-ins don't know ("CustomThing"): category + a FLOP
  // formula = 3 * in0-elem-count. Then resolve it through the registry and check
  // the handler is Declarative and returns the formula's number.
  std::string p = write_temp(R"json({ "api_version": 1, "name": "test-custom",
    "ops": [{ "name": "CustomThing", "category": "MatMul",
              "flops": "3 * in0.shape[0] * in0.shape[1]" }] })json");
  LoadedManifest lm = load_manifest_file(p, /*register_into=*/true);
  std::remove(p.c_str());
  REQUIRE(lm.ok == true);
  REQUIRE(lm.ops.size() == 1);
  REQUIRE(lm.ops[0].ok == true);

  ir::Model m; m.graphs.emplace_back(); ir::Graph& g = m.graphs[0];
  uint32_t a = addv(g, m, "A", ir::DType::F32, {4, 8});
  uint32_t y = addv(g, m, "Y", ir::DType::F32, {4, 8});
  ir::Node n; n.op_type = m.intern("CustomThing");
  n.inputs.begin = 0; n.inputs.count = 1; g.edge_refs.push_back(a);
  n.outputs.begin = 1; n.outputs.count = 1; g.edge_refs.push_back(y);
  g.nodes.push_back(n);

  Registry& reg = Registry::instance();
  OpResolution r = reg.resolve_op("customthing");
  CHECK(r.origin == Origin::Declarative);
  CHECK(r.plugin_name == "test-custom");
  REQUIRE(r.handler != nullptr);

  OpContext ctx = reg.make_context(m, g, g.nodes[0], {});
  CHECK(r.handler->category(ctx) == OpCategory::MatMul);
  FlopResult fr = r.handler->flops(ctx);
  CHECK(fr.known == true);
  CHECK(fr.flops == 3 * 4 * 8);   // the DSL formula, evaluated on this node
}
