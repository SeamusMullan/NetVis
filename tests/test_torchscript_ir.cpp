// SPDX-License-Identifier: Apache-2.0
// tests/test_torchscript_ir.cpp — the shared TorchScript -> IR layer (#137 §7.5).
//
// parsers/pytorch/TorchScriptIR.h is the layer #136 (the code/*.py front-end)
// reuses unchanged, so its contracts are pinned here independently of the mobile
// bytecode decoder: constant folding/repr, GraphBuilder layout, initializer
// dedup, module attribute binding and CONSTANTS.c<i> naming. #136 extends this
// file.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "ir/IR.h"
#include "parsers/pytorch/PickleVM.h"
#include "parsers/pytorch/TorchScriptIR.h"

using namespace netvis;
using netvis::pytorch::Value;
using netvis::pytorch::ValuePtr;
namespace ts = netvis::pytorch::ts;

namespace {

ValuePtr mk(Value::Kind k) {
  auto v = std::make_shared<Value>();
  v->kind = k;
  return v;
}
ValuePtr I(int64_t i) { auto v = mk(Value::Kind::Int); v->i = i; return v; }
ValuePtr D(double d) { auto v = mk(Value::Kind::Double); v->d = d; return v; }
ValuePtr B(bool b) { auto v = mk(Value::Kind::Bool); v->b = b; return v; }
ValuePtr S(const std::string& s) { auto v = mk(Value::Kind::Str); v->s = s; return v; }
ValuePtr Y(const std::string& s) { auto v = mk(Value::Kind::Bytes); v->s = s; return v; }
ValuePtr N() { return mk(Value::Kind::None); }
ValuePtr L(std::vector<ValuePtr> items) {
  auto v = mk(Value::Kind::List);
  v->items = std::move(items);
  return v;
}
ValuePtr T(std::vector<ValuePtr> items) {
  auto v = mk(Value::Kind::Tuple);
  v->items = std::move(items);
  return v;
}
ValuePtr Dict(std::vector<std::pair<ValuePtr, ValuePtr>> pairs) {
  auto v = mk(Value::Kind::Dict);
  v->pairs = std::move(pairs);
  return v;
}
ValuePtr Tensor(uint64_t off, uint64_t len, std::vector<int64_t> shape,
                ir::DType dt = ir::DType::F32) {
  auto v = mk(Value::Kind::Tensor);
  v->tensor.dtype = dt;
  for (int64_t d : shape) v->tensor.shape.push_back(d);
  v->tensor.file_offset = off;
  v->tensor.byte_len = len;
  return v;
}
// A TorchScript object __torch__.<cls> with state attrs in slot order.
ValuePtr Obj(const std::string& module, const std::string& name,
             std::vector<std::pair<std::string, ValuePtr>> attrs) {
  auto v = mk(Value::Kind::Opaque);
  v->module = module;
  v->name = name;
  std::vector<std::pair<ValuePtr, ValuePtr>> pairs;
  for (auto& a : attrs) pairs.emplace_back(S(a.first), a.second);
  v->inner = Dict(std::move(pairs));
  return v;
}

std::string attr_str(const ir::Model& m, const ir::AttrValue& a) {
  return std::string(m.str(a.s));
}

// constant_to_attr expecting a String result.
std::string as_string_attr(const ValuePtr& v, ir::Model& m,
                           const ts::ConstLimits& lim = {}) {
  ir::AttrValue a;
  REQUIRE(ts::constant_to_attr(*v, m, a, lim));
  REQUIRE(a.kind == ir::AttrValue::Kind::String);
  return attr_str(m, a);
}

}  // namespace

TEST_CASE("TorchScriptIR: constant_to_attr covers every pickled kind") {
  ir::Model m;
  using K = ir::AttrValue::Kind;
  ir::AttrValue a;

  REQUIRE(ts::constant_to_attr(*I(7), m, a));
  CHECK(a.kind == K::Int);
  CHECK(a.i == 7);

  REQUIRE(ts::constant_to_attr(*D(0.25), m, a));
  CHECK(a.kind == K::Float);
  CHECK(a.f == doctest::Approx(0.25));

  CHECK(as_string_attr(B(true), m) == "True");
  CHECK(as_string_attr(B(false), m) == "False");
  CHECK(as_string_attr(N(), m) == "None");
  CHECK(as_string_attr(S("cpu"), m) == "'cpu'");
  CHECK(as_string_attr(Y("ab"), m) == "b'ab'");

  REQUIRE(ts::constant_to_attr(*L({I(1), I(2)}), m, a));
  CHECK(a.kind == K::Ints);
  CHECK(a.ints == std::vector<int64_t>{1, 2});
  REQUIRE(ts::constant_to_attr(*T({I(3)}), m, a));
  CHECK(a.kind == K::Ints);

  REQUIRE(ts::constant_to_attr(*L({D(0.5), D(1.0)}), m, a));
  CHECK(a.kind == K::Floats);
  CHECK(a.floats.size() == 2);

  CHECK(as_string_attr(L({B(true), B(false)}), m) == "[True, False]");
  CHECK(as_string_attr(T({S("a"), I(1)}), m) == "('a', 1)");
  CHECK(as_string_attr(T({I(1), S("x")}), m) == "(1, 'x')");
  CHECK(as_string_attr(T({S("a")}), m) == "('a',)");
  CHECK(as_string_attr(Dict({{S("k"), I(2)}}), m) == "{'k': 2}");
  CHECK(as_string_attr(L({I(1), D(1.5)}), m) == "[1, 1.5]");
  // Mixed with None so it is a repr, not Floats: Python float repr rules.
  CHECK(as_string_attr(L({D(100.0), D(1e-5), D(1e16), D(-0.5), N()}), m) ==
        "[100.0, 1e-05, 1e+16, -0.5, None]");
  CHECK(as_string_attr(L({Tensor(0, 32, {2, 4})}), m) == "[<tensor f32[2,4]>]");

  // A bare tensor is not a constant attribute: the caller makes an initializer.
  ir::AttrValue untouched;
  untouched.kind = K::Int;
  untouched.i = 99;
  CHECK_FALSE(ts::constant_to_attr(*Tensor(0, 8, {2}), m, untouched));
  CHECK(untouched.kind == K::Int);
  CHECK(untouched.i == 99);
}

TEST_CASE("TorchScriptIR: string repr escaping, empty and long lists") {
  ir::Model m;
  CHECK(as_string_attr(S("a'b"), m) == "'a\\'b'");
  CHECK(as_string_attr(S(std::string("\x01", 1)), m) == "'\\x01'");
  CHECK(as_string_attr(S("back\\slash"), m) == "'back\\\\slash'");
  CHECK(as_string_attr(L({}), m) == "[]");
  CHECK(as_string_attr(T({}), m) == "()");

  std::vector<ValuePtr> many;
  for (int i = 0; i < 5000; ++i) many.push_back(I(i));
  CHECK(as_string_attr(L(many), m) == "[\xE2\x80\xA6 5000 items]");

  // Over-long reprs are truncated to max_repr bytes plus an ellipsis.
  std::string s = as_string_attr(S(std::string(1000, 'x')), m);
  CHECK(s.size() == 256 + 3);
  CHECK(s.substr(256) == "\xE2\x80\xA6");
}

TEST_CASE("TorchScriptIR: cyclic and deeply nested constants terminate") {
  ir::Model m;
  // A list that contains itself (a memo cycle).
  ValuePtr cyc = L({I(1)});
  cyc->items.push_back(cyc);
  std::string r = as_string_attr(cyc, m);
  CHECK(r == "[1, ...]");
  cyc->items.clear();  // break the cycle so the test does not leak

  // 17 nested single-element lists: deeper than max_depth (16) is cut.
  ValuePtr deep = I(42);
  for (int i = 0; i < 17; ++i) deep = L({deep});
  std::string d = as_string_attr(deep, m);
  CHECK(d.find("...") != std::string::npos);
  CHECK(d.find("42") == std::string::npos);

  // 16 levels still render fully.
  ValuePtr ok = I(42);
  for (int i = 0; i < 16; ++i) ok = L({ok});
  CHECK(as_string_attr(ok, m).find("42") != std::string::npos);
}

TEST_CASE("TorchScriptIR: Opaque with retained args renders as a call") {
  ir::Model m;
  ValuePtr dev = mk(Value::Kind::Opaque);
  dev->module = "torch";
  dev->name = "device";
  dev->items = {S("cpu")};
  CHECK(as_string_attr(dev, m) == "torch.device('cpu')");
  ValuePtr bare = mk(Value::Kind::Opaque);
  bare->module = "foo";
  bare->name = "Bar";
  CHECK(as_string_attr(bare, m) == "<foo.Bar>");
}

TEST_CASE("TorchScriptIR: GraphBuilder::finish lays out contiguous ranges") {
  ir::Model m;
  ts::NameCounter names;
  ts::GraphBuilder gb(m, names, "g", 0, nullptr);
  uint32_t x = gb.add_input("x");
  uint32_t y = gb.add_input("y");
  uint32_t n0 = gb.node("aten::add");
  gb.add_edge(n0, x);
  gb.add_edge(n0, y);
  uint32_t a = gb.add_output(n0);  // %1
  uint32_t n1 = gb.node("aten::relu");
  gb.add_edge(n1, a);
  ir::Attribute at;
  at.name = m.intern("arg1");
  at.value.kind = ir::AttrValue::Kind::Int;
  at.value.i = 3;
  gb.add_attr(n1, at);
  uint32_t b = gb.add_output(n1);  // %2
  gb.add_output_value(b);
  uint32_t self = gb.add_input("self");
  gb.prepend_input(self);
  CHECK(gb.value_name(a) == "%1");
  CHECK(gb.value_name(b) == "%2");

  ir::Graph g = gb.finish();
  CHECK(m.str(g.name) == "g");
  REQUIRE(g.nodes.size() == 2);
  CHECK(m.str(g.nodes[0].op_type) == "aten::add");
  CHECK(g.nodes[0].inputs.begin == 0);
  CHECK(g.nodes[0].inputs.count == 2);
  CHECK(g.nodes[0].outputs.begin == 2);
  CHECK(g.nodes[0].outputs.count == 1);
  CHECK(g.nodes[1].inputs.begin == 3);
  CHECK(g.nodes[1].outputs.begin == 4);
  CHECK(g.edge_refs == std::vector<uint32_t>{x, y, a, a, b});
  CHECK(g.nodes[1].attributes.begin == 0);
  CHECK(g.nodes[1].attributes.count == 1);
  CHECK(g.nodes[0].attributes.count == 0);
  CHECK(g.values[a].producer == 0);
  CHECK(g.values[b].producer == 1);
  CHECK(g.values[x].producer == -1);
  CHECK(g.graph_inputs == std::vector<uint32_t>{self, x, y});
  CHECK(g.graph_outputs == std::vector<uint32_t>{b});
  for (const ir::Node& n : g.nodes) CHECK(m.str(n.name).empty());
}

TEST_CASE("TorchScriptIR: initializer dedup and #2 suffixes") {
  ir::Model m;
  ts::NameCounter names;
  ts::GraphBuilder gb(m, names, "g", 0, nullptr);
  ir::TensorRef t;
  t.dtype = ir::DType::F32;
  t.shape.push_back(2);
  t.file_offset = 100;
  t.byte_len = 8;
  uint32_t v1 = gb.initializer("w", t);
  uint32_t v2 = gb.initializer("w", t);
  CHECK(v1 == v2);
  ir::TensorRef t2 = t;
  t2.file_offset = 200;
  uint32_t v3 = gb.initializer("w", t2);
  CHECK(v3 != v1);
  CHECK(gb.value_name(v3) == "w#2");
  CHECK(gb.initializer("w", t2) == v3);  // reuses the suffixed one
  // A non-initializer value with the name forces a suffix too.
  uint32_t in = gb.add_input("b");
  uint32_t v4 = gb.initializer("b", t);
  CHECK(v4 != in);
  CHECK(gb.value_name(v4) == "b#2");
  // Duplicate input names are unique-ified as well.
  CHECK(gb.value_name(gb.add_input("b")) == "b#3");

  ir::Graph g = gb.finish();
  REQUIRE(g.initializers.size() == 3);
  CHECK(m.str(g.initializers[0].name) == "w");
  CHECK(m.str(g.initializers[1].name) == "w#2");
  CHECK(g.values[v1].dtype == ir::DType::F32);
  CHECK(g.values[v1].shape.size() == 1);
}

TEST_CASE("TorchScriptIR: captures are cached per parent value") {
  ir::Model m;
  ts::NameCounter names;
  ts::GraphBuilder parent(m, names, "p", 0, nullptr);
  uint32_t x = parent.add_input("x");
  ts::GraphBuilder child(m, names, "p/if0.then", 1, &parent);
  uint32_t c1 = child.capture(x, parent.value_name(x));
  uint32_t c2 = child.capture(x, parent.value_name(x));
  CHECK(c1 == c2);
  CHECK(child.value_name(c1) == "x");
  CHECK(child.captured_parent_values() == std::vector<uint32_t>{x});
  CHECK(child.parent() == &parent);
  ir::Graph g = child.finish();
  CHECK(g.graph_inputs == std::vector<uint32_t>{c1});
}

TEST_CASE("TorchScriptIR: ModuleEnv by_name agrees with by_slot") {
  ValuePtr w = Tensor(10, 32, {2, 4});
  ValuePtr b = Tensor(42, 8, {2});
  ValuePtr dev = mk(Value::Kind::Opaque);
  dev->module = "torch";
  dev->name = "device";
  dev->items = {S("cpu")};
  ValuePtr packed = mk(Value::Kind::Opaque);  // object without dict state
  packed->module = "__torch__.torch.classes.quantized";
  packed->name = "LinearPackedParamsBase";
  packed->inner = T({I(1)});
  ValuePtr fc = Obj("__torch__.torch.nn.modules.linear", "Linear",
                    {{"training", B(true)}, {"weight", w}, {"bias", b}});
  ValuePtr root = Obj("__torch__", "Model",
                      {{"training", B(true)}, {"fc", fc}, {"dev", dev},
                       {"packed", packed}, {"tied", w}});
  ts::ModuleEnv env(root);
  CHECK(env.loaded());
  REQUIRE(env.root() == root.get());
  CHECK(env.root_class() == "__torch__.Model");

  using K = ts::ModuleEnv::Attr::Kind;
  const K expect_root[] = {K::Constant, K::Object, K::Constant, K::Opaque,
                           K::Tensor};
  for (int64_t s = 0; s < 5; ++s) {
    auto a = env.by_slot(root.get(), s);
    CHECK(a.kind == expect_root[s]);
    auto n = env.by_name(root.get(), a.name);
    CHECK(n.kind == a.kind);
    CHECK(n.value == a.value);
    CHECK(n.name == a.name);
  }
  for (int64_t s = 0; s < 3; ++s) {
    auto a = env.by_slot(fc.get(), s);
    auto n = env.by_name(fc.get(), a.name);
    CHECK(n.kind == a.kind);
    CHECK(n.value == a.value);
  }
  CHECK(env.by_slot(root.get(), 5).kind == K::Unknown);
  CHECK(env.by_slot(root.get(), -1).kind == K::Unknown);
  CHECK(env.by_name(root.get(), "nope").kind == K::Unknown);
  CHECK(env.by_slot(w.get(), 0).kind == K::Unknown);  // not an object

  // Tied weight: one canonical (first-traversal) name.
  CHECK(env.tensor_path(w.get(), "x") == "fc.weight");
  CHECK(env.tensor_path(b.get(), "x") == "fc.bias");
  CHECK(env.tensor_path(Tensor(0, 1, {1}).get(), "fallback") == "fallback");
  CHECK(env.module_tensor_count() == 2);

  ts::ModuleEnv empty;
  CHECK_FALSE(empty.loaded());
  CHECK(empty.root() == nullptr);
  CHECK(empty.root_class().empty());
  ts::ModuleEnv not_obj(Dict({}));
  CHECK(not_obj.loaded());
  CHECK(not_obj.root() == nullptr);
}

TEST_CASE("TorchScriptIR: TensorConstNames matches only exact records") {
  ts::TensorConstNames names;
  ValuePtr c0 = Tensor(100, 8, {2});
  ValuePtr c1 = Tensor(200, 16, {4});
  names.add_from_constants_pkl(T({I(5), c0, c1}));
  CHECK(names.name_for(c0->tensor, "fb") == "CONSTANTS.c1");
  CHECK(names.name_for(c1->tensor, "fb") == "CONSTANTS.c2");

  ir::TensorRef t = c0->tensor;
  t.byte_len = 4;
  CHECK(names.name_for(t, "fb") == "fb");
  t = c0->tensor;
  t.dtype = ir::DType::I32;
  CHECK(names.name_for(t, "fb") == "fb");
  t = c0->tensor;
  t.shape.clear();
  t.shape.push_back(1);
  t.shape.push_back(2);
  CHECK(names.name_for(t, "fb") == "fb");
  t = c0->tensor;
  t.file_offset = 101;
  CHECK(names.name_for(t, "fb") == "fb");

  // Unlocated tensors never match; a non-tuple adds nothing.
  ts::TensorConstNames n2;
  n2.add_from_constants_pkl(T({Tensor(UINT64_MAX, 8, {2})}));
  CHECK(n2.name_for(Tensor(UINT64_MAX, 8, {2})->tensor, "fb") == "fb");
  ts::TensorConstNames n3;
  n3.add_from_constants_pkl(L({c0}));
  CHECK(n3.name_for(c0->tensor, "fb") == "fb");
}

TEST_CASE("TorchScriptIR: is_torch_object and for_each_tensor") {
  ValuePtr w = Tensor(0, 4, {1});
  ValuePtr obj = Obj("__torch__", "M", {{"w", w}, {"l", L({w})}});
  CHECK(ts::is_torch_object(*obj));
  ValuePtr other = Obj("mymod", "M", {{"w", w}});
  CHECK_FALSE(ts::is_torch_object(*other));
  std::vector<std::string> paths;
  ts::for_each_tensor(obj, [&](const std::string& p, const Value&) {
    paths.push_back(p);
  });
  CHECK(paths == std::vector<std::string>{"w", "l.0"});
  paths.clear();
  ts::for_each_tensor(other, [&](const std::string& p, const Value&) {
    paths.push_back(p);
  });
  CHECK(paths.empty());
  CHECK(ts::arg_attr_name(2) == "arg2");
}
