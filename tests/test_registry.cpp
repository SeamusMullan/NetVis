// tests/test_registry.cpp — plugin registry spine contract (v0.6.0 Increment 1, #8).
//
// The spine is a ZERO-behavior-change refactor: with no user plugins loaded, the
// registry resolves every op to the built-in catch-all, whose flops()/category()
// delegate to the UNCHANGED v0.4.0 formula table. These tests assert:
//   1. resolve_op returns the built-in catch-all (origin==Builtin, overrides=false)
//      for both a known op and an unknown op.
//   2. normalize_op_key strips domain + lowercases exactly like CostModel::norm_op.
//   3. The OpContext accessors mirror the CostModel helpers (attrs, shapes,
//      initializer membership).
//   4. THE ACCEPTANCE INVARIANT: BuiltinOpHandler.flops() through an OpContext is
//      byte-identical to compute_cost's per-node FLOPs (they are the same code, so
//      any divergence is a wiring bug) — checked over a hand-built multi-op graph.
//   5. resolve_category(...) == categorize_op(...) with no plugins.
//   6. The zero-payload invariant still holds (payload_read_counter()==0).

#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "engine/CostModel.h"
#include "engine/OpCategory.h"
#include "engine/plugin/OpHandler.h"
#include "engine/plugin/Registry.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// Append a value (name/dtype/shape); return its index.
uint32_t add_value(ir::Graph& g, ir::Model& m, const char* name, ir::DType dt,
                   std::vector<int64_t> shape) {
  ir::ValueInfo vi;
  vi.name = m.intern(name);
  vi.dtype = dt;
  for (int64_t d : shape) vi.shape.push_back(d);
  g.values.push_back(vi);
  return static_cast<uint32_t>(g.values.size() - 1);
}

// Append a node with the given op + input/output value indices.
uint32_t add_node(ir::Graph& g, ir::Model& m, const char* op,
                  std::vector<uint32_t> ins, std::vector<uint32_t> outs) {
  ir::Node n;
  n.op_type = m.intern(op);
  n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  n.inputs.count = static_cast<uint32_t>(ins.size());
  for (uint32_t v : ins) g.edge_refs.push_back(v);
  n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  n.outputs.count = static_cast<uint32_t>(outs.size());
  for (uint32_t v : outs) g.edge_refs.push_back(v);
  g.nodes.push_back(n);
  return static_cast<uint32_t>(g.nodes.size() - 1);
}

}  // namespace

TEST_CASE("registry: normalize_op_key strips domain and lowercases") {
  CHECK(plugin::normalize_op_key("Conv") == "conv");
  CHECK(plugin::normalize_op_key("com.microsoft.QLinearConv") == "qlinearconv");
  CHECK(plugin::normalize_op_key("MatMul") == "matmul");
  CHECK(plugin::normalize_op_key("") == "");
}

TEST_CASE("registry: resolve_op returns the built-in catch-all with no plugins") {
  plugin::Registry& reg = plugin::Registry::instance();

  // A known op and a totally unknown op both resolve to the built-in.
  for (const char* op : {"conv", "matmul", "totally_unknown_op_xyz"}) {
    plugin::OpResolution r = reg.resolve_op(op);
    CHECK(r.handler != nullptr);
    CHECK(r.origin == plugin::Origin::Builtin);
    CHECK(r.overrides_builtin == false);
    CHECK(r.plugin_name.empty());
    CHECK(r.handler->api_version() == plugin::kOpHandlerAbiVersion);
  }
}

TEST_CASE("registry: OpContext accessors mirror the IR structure") {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(g, m, "A", ir::DType::F32, {4, 8});
  uint32_t w = add_value(g, m, "W", ir::DType::F32, {8, 16});
  uint32_t y = add_value(g, m, "Y", ir::DType::F32, {4, 16});
  uint32_t ni = add_node(g, m, "Gemm", {a, w}, {y});

  // W is an initializer (a weight).
  ir::TensorRef init;
  init.name = m.intern("W");
  init.dtype = ir::DType::F32;
  init.shape.push_back(8);
  init.shape.push_back(16);
  init.byte_len = 8 * 16 * 4;
  g.initializers.push_back(init);

  // A "transA" int attr.
  ir::Attribute attr;
  attr.name = m.intern("transA");
  attr.value.kind = ir::AttrValue::Kind::Int;
  attr.value.i = 1;
  g.attributes.push_back(attr);
  g.nodes[ni].attributes.begin = 0;
  g.nodes[ni].attributes.count = 1;

  plugin::Registry& reg = plugin::Registry::instance();
  plugin::OpContext ctx = reg.make_context(m, g, g.nodes[ni], {});

  CHECK(ctx.op_raw() == "Gemm");
  CHECK(ctx.op_type() == "gemm");
  CHECK(ctx.input_count() == 2);
  CHECK(ctx.output_count() == 1);

  REQUIRE(ctx.input_shape(0) != nullptr);
  CHECK(ctx.input_shape(0)->size() == 2);
  CHECK((*ctx.input_shape(0))[1] == 8);
  REQUIRE(ctx.output_shape(0) != nullptr);
  CHECK((*ctx.output_shape(0))[1] == 16);
  CHECK(ctx.input_dtype(0) == ir::DType::F32);

  CHECK(ctx.input_is_initializer(1) == true);   // W
  CHECK(ctx.input_is_initializer(0) == false);   // A
  auto ir_rec = ctx.input_initializer(1);
  REQUIRE(ir_rec.has_value());
  CHECK(ir_rec->byte_len == 8 * 16 * 4);
  CHECK(ir_rec->elem_count == 8 * 16);

  CHECK(ctx.has_attr("transA") == true);
  CHECK(ctx.attr_int("transA", 0) == 1);
  CHECK(ctx.attr_int("missing", 42) == 42);
  CHECK(ctx.attr_string("nope", "def") == "def");

  // mmap_base is null (cost driver) -> input_const_ints deterministically nullptr.
  CHECK(ctx.input_const_ints(0) == nullptr);
}

TEST_CASE("registry: built-in handler FLOPs == compute_cost per-node (byte-identical)") {
  ByteReader::payload_read_counter() = 0;

  // Build a small graph exercising several formula families.
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  // MatMul: A[4,8] x B[8,16] -> Y[4,16]; K=8; macs=|O|*K=4*16*8; flops=2*macs.
  uint32_t a = add_value(g, m, "A", ir::DType::F32, {4, 8});
  uint32_t b = add_value(g, m, "B", ir::DType::F32, {8, 16});
  uint32_t y = add_value(g, m, "Y", ir::DType::F32, {4, 16});
  add_node(g, m, "MatMul", {a, b}, {y});

  // Relu (elementwise): |O| flops.
  uint32_t r = add_value(g, m, "R", ir::DType::F32, {4, 16});
  add_node(g, m, "Relu", {y}, {r});

  // Unknown op: flops_known=false.
  uint32_t z = add_value(g, m, "Z", ir::DType::F32, {4, 16});
  add_node(g, m, "SomeExoticOp", {r}, {z});

  CostReport report = compute_cost(m, 0);
  REQUIRE(report.per_node.size() == 3);

  // Independently resolve each node through the registry and compare — this is
  // the same code path compute_cost now uses, so equality is the wiring proof.
  plugin::Registry& reg = plugin::Registry::instance();
  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    plugin::OpContext ctx = reg.make_context(m, g, g.nodes[i], {});
    const plugin::OpHandler* h =
        reg.resolve_op(plugin::normalize_op_key(m.str(g.nodes[i].op_type))).handler;
    REQUIRE(h != nullptr);
    plugin::FlopResult fr = h->flops(ctx);
    CHECK(fr.flops == report.per_node[i].flops);
    CHECK(fr.known == report.per_node[i].flops_known);
  }

  // MatMul: 2*4*16*8 = 1024; Relu: 4*16 = 64; exotic: unknown.
  CHECK(report.per_node[0].flops == 1024);
  CHECK(report.per_node[0].flops_known == true);
  CHECK(report.per_node[1].flops == 64);
  CHECK(report.per_node[1].flops_known == true);
  CHECK(report.per_node[2].flops_known == false);

  // The whole thing read zero payload bytes.
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("registry: resolve_category == categorize_op with no plugins") {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  uint32_t a = add_value(g, m, "A", ir::DType::F32, {4, 8});
  uint32_t y = add_value(g, m, "Y", ir::DType::F32, {4, 8});

  for (const char* op : {"Conv", "MatMul", "Relu", "Add", "Reshape",
                         "com.microsoft.QLinearConv", "WeirdUnknownOp"}) {
    uint32_t ni = add_node(g, m, op, {a}, {y});
    OpCategory via_registry = plugin::resolve_category(m, g, g.nodes[ni]);
    OpCategory via_direct = categorize_op(op);
    CHECK(via_registry == via_direct);
  }
}
