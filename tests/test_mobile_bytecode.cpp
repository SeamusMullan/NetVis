// SPDX-License-Identifier: Apache-2.0
// tests/test_mobile_bytecode.cpp — PyTorch Mobile bytecode.pkl (#137).
//
// Three layers:
//   T-U*  decode_bytecode on Value trees built in C++ (tables, versions, caps)
//   T-B*  build_method_graph on BcModules built in C++ (the symbolic executor:
//         pending outputs, registers, If/Loop structuring, GET_ATTR binding and
//         every honest-failure path)
//   T-M*  the hand-built .ptl fixtures end to end through pytorch::parse_zip,
//         plus per-byte truncation and corruption sweeps of bytecode.pkl
// Every fixture-level test asserts zero payload reads.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "engine/CostModel.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/pytorch/MobileBytecode.h"
#include "parsers/pytorch/PickleVM.h"
#include "parsers/pytorch/TorchScriptIR.h"

#include "miniz.h"
#include "temp_file_guard.h"

using namespace netvis;
using netvis::pytorch::Value;
using netvis::pytorch::ValuePtr;
namespace ts = netvis::pytorch::ts;
namespace mb = netvis::pytorch::mobile;
using mb::BcOp;

namespace {

// ---- Value-tree helpers (T-U) ------------------------------------------------
ValuePtr mk(Value::Kind k) {
  auto v = std::make_shared<Value>();
  v->kind = k;
  return v;
}
ValuePtr I(int64_t i) { auto v = mk(Value::Kind::Int); v->i = i; return v; }
ValuePtr S(const std::string& s) { auto v = mk(Value::Kind::Str); v->s = s; return v; }
ValuePtr B(bool b) { auto v = mk(Value::Kind::Bool); v->b = b; return v; }
ValuePtr N() { return mk(Value::Kind::None); }
ValuePtr T(std::vector<ValuePtr> items) {
  auto v = mk(Value::Kind::Tuple);
  v->items = std::move(items);
  return v;
}
ValuePtr L(std::vector<ValuePtr> items) {
  auto v = mk(Value::Kind::List);
  v->items = std::move(items);
  return v;
}
ValuePtr row(const std::string& name, ValuePtr v) { return T({S(name), std::move(v)}); }
ValuePtr ins(const std::string& op, int64_t x, int64_t n) { return T({S(op), I(x), I(n)}); }
ValuePtr arg(const std::string& name, const std::string& type) {
  return T({row("name", S(name)), row("type", S(type)), row("default_value", N())});
}
ValuePtr fn(const std::string& name, std::vector<ValuePtr> instrs,
            std::vector<ValuePtr> ops, std::vector<ValuePtr> consts,
            std::vector<ValuePtr> types, int64_t regs, std::vector<ValuePtr> args,
            std::vector<ValuePtr> rets, bool schema = true) {
  ValuePtr code = T({row("instructions", T(std::move(instrs))),
                     row("operators", T(std::move(ops))),
                     row("constants", T(std::move(consts))),
                     row("types", T(std::move(types))),
                     row("register_size", I(regs))});
  if (!schema) return T({S(name), code});
  ValuePtr sch = T({row("arguments", T(std::move(args))), row("returns", T(std::move(rets)))});
  return T({S(name), code, sch});
}
// A minimal valid method: forward(self, x) -> Tensor { return relu(x) }.
ValuePtr simple_fn(const std::string& name = "__torch__.M.forward", bool schema = true) {
  return fn(name,
            {ins("STOREN", 1, 2), ins("MOVE", 2, 0), ins("OP", 0, 0), ins("RET", 0, 0)},
            {T({S("aten::relu"), S(""), I(1)})}, {}, {S("Tensor")}, 2,
            {arg("self", "__torch__.M"), arg("x", "Tensor")}, {arg("", "Tensor")},
            schema);
}

std::string err_of(const Result<mb::BcModule>& r) {
  return r ? std::string() : r.error().message;
}

// ---- BcModule helpers (T-B) --------------------------------------------------
mb::BcInstruction In(BcOp op, int32_t x = 0, int32_t n = 0) {
  mb::BcInstruction i;
  i.op = op;
  i.x = x;
  i.n = n;
  return i;
}
mb::BcOperator Op(const std::string& name, const std::string& ov, int64_t nargs) {
  mb::BcOperator o;
  o.name = name;
  o.overload = ov;
  o.num_args = nargs;
  o.has_num_args = true;
  return o;
}
mb::BcArg Arg(const std::string& name, const std::string& type) {
  return mb::BcArg{name, type, nullptr};
}
mb::BcFunction make_fn(std::vector<mb::BcInstruction> code,
                       std::vector<mb::BcOperator> ops, std::vector<ValuePtr> consts,
                       int64_t regs, std::vector<mb::BcArg> args,
                       std::vector<mb::BcArg> rets,
                       std::vector<std::string> types = {}) {
  mb::BcFunction f;
  f.qualified_name = "__torch__.M.forward";
  for (size_t i = 0; i < code.size(); ++i) code[i].pos = 1000 + i;
  f.instructions = std::move(code);
  f.operators = std::move(ops);
  f.constants = std::move(consts);
  for (auto& t : types) {
    f.types.push_back(t);
    f.type_ok.push_back(1);
  }
  f.register_size = regs;
  f.has_schema = true;
  f.arguments = std::move(args);
  f.returns = std::move(rets);
  f.pos = 999;
  return f;
}
mb::BcModule mod(mb::BcFunction f, int64_t version = 8) {
  mb::BcModule m;
  m.version = version;
  m.functions.push_back(std::move(f));
  return m;
}

struct Run {
  ir::Model model;
  std::optional<Result<mb::BuildResult>> res;
  bool ok() const { return res && static_cast<bool>(*res); }
  std::string error() const {
    return (res && !*res) ? res->error().message : std::string();
  }
};
const ts::ModuleEnv& empty_env() {
  static const ts::ModuleEnv e;
  return e;
}
const ts::TensorConstNames& no_names() {
  static const ts::TensorConstNames n;
  return n;
}
void build_into(Run& r, const mb::BcModule& m, const ts::ModuleEnv& env = empty_env(),
                const ts::TensorConstNames& cn = no_names(),
                mb::BytecodeLimits lim = {}) {
  mb::BuildContext ctx{r.model, env, cn, lim};
  r.res.emplace(mb::build_method_graph(m, 0, ctx));
}

// ---- IR inspection -------------------------------------------------------------
std::string str(const ir::Model& m, StringId id) { return std::string(m.str(id)); }
std::vector<std::string> ops_of(const ir::Model& m, const ir::Graph& g) {
  std::vector<std::string> out;
  for (const auto& n : g.nodes) out.push_back(str(m, n.op_type));
  return out;
}
std::vector<std::string> names_of(const ir::Model& m, const ir::Graph& g,
                                  const std::vector<uint32_t>& vals) {
  std::vector<std::string> out;
  for (uint32_t v : vals) out.push_back(v < g.values.size() ? str(m, g.values[v].name) : "?");
  return out;
}
std::vector<std::string> ins_of(const ir::Model& m, const ir::Graph& g, size_t n) {
  const ir::Node& nd = g.nodes[n];
  std::vector<uint32_t> v(g.edge_refs.begin() + nd.inputs.begin,
                          g.edge_refs.begin() + nd.inputs.begin + nd.inputs.count);
  return names_of(m, g, v);
}
std::vector<std::string> outs_of(const ir::Model& m, const ir::Graph& g, size_t n) {
  const ir::Node& nd = g.nodes[n];
  std::vector<uint32_t> v(g.edge_refs.begin() + nd.outputs.begin,
                          g.edge_refs.begin() + nd.outputs.begin + nd.outputs.count);
  return names_of(m, g, v);
}
const ir::AttrValue* attr_of(const ir::Model& m, const ir::Graph& g, size_t n,
                             const std::string& name) {
  const ir::Node& nd = g.nodes[n];
  for (uint32_t i = 0; i < nd.attributes.count; ++i) {
    const ir::Attribute& a = g.attributes[nd.attributes.begin + i];
    if (m.str(a.name) == name) return &a.value;
  }
  return nullptr;
}
std::vector<std::string> attr_names(const ir::Model& m, const ir::Graph& g, size_t n) {
  std::vector<std::string> out;
  const ir::Node& nd = g.nodes[n];
  for (uint32_t i = 0; i < nd.attributes.count; ++i)
    out.push_back(str(m, g.attributes[nd.attributes.begin + i].name));
  return out;
}
std::string attr_s(const ir::Model& m, const ir::Graph& g, size_t n, const std::string& name) {
  const ir::AttrValue* a = attr_of(m, g, n, name);
  REQUIRE(a);
  REQUIRE(a->kind == ir::AttrValue::Kind::String);
  return str(m, a->s);
}
int64_t attr_i(const ir::Model& m, const ir::Graph& g, size_t n, const std::string& name) {
  const ir::AttrValue* a = attr_of(m, g, n, name);
  REQUIRE(a);
  REQUIRE(a->kind == ir::AttrValue::Kind::Int);
  return a->i;
}
int32_t attr_g(const ir::Model& m, const ir::Graph& g, size_t n, const std::string& name) {
  const ir::AttrValue* a = attr_of(m, g, n, name);
  REQUIRE(a);
  REQUIRE(a->kind == ir::AttrValue::Kind::Graph);
  return a->graph;
}
std::vector<std::string> init_names(const ir::Model& m, const ir::Graph& g) {
  std::vector<std::string> out;
  for (const auto& t : g.initializers) out.push_back(str(m, t.name));
  return out;
}

using VS = std::vector<std::string>;

}  // namespace

// ============================================================================
// Opcode table
// ============================================================================
TEST_CASE("mobile bytecode: opcode table round-trips with upstream spellings") {
  for (int i = 0; i < static_cast<int>(BcOp::Unknown); ++i) {
    auto op = static_cast<BcOp>(i);
    CHECK(mb::opcode_from_string(mb::opcode_name(op)) == op);
  }
  CHECK(std::string(mb::opcode_name(BcOp::IS)) == "__IS__");
  CHECK(std::string(mb::opcode_name(BcOp::NOT)) == "__NOT__");
  CHECK(std::string(mb::opcode_name(BcOp::AWAITABLE)) == "AWAITABLE");
  CHECK(std::string(mb::opcode_name(BcOp::Unknown)) == "?");
  CHECK(mb::opcode_from_string("op") == BcOp::Unknown);  // case-sensitive
  CHECK(mb::opcode_from_string("FROB") == BcOp::Unknown);
  CHECK(mb::opcode_valid_in_mobile(BcOp::OP));
  CHECK(mb::opcode_valid_in_mobile(BcOp::INTERFACE_CALL));
  CHECK_FALSE(mb::opcode_valid_in_mobile(BcOp::CALL));
  CHECK_FALSE(mb::opcode_valid_in_mobile(BcOp::FORK));
  CHECK_FALSE(mb::opcode_valid_in_mobile(BcOp::WAIT));
  CHECK_FALSE(mb::opcode_valid_in_mobile(BcOp::Unknown));
}

// ============================================================================
// T-U: decode
// ============================================================================
TEST_CASE("T-U1 v8 tables decode") {
  ValuePtr f = fn("__torch__.M.forward",
                  {ins("STOREN", 1, 2), ins("MOVE", 2, 0), ins("OP", 0, 0),
                   ins("JMP", -3, 0), ins("RET", 0, 0)},
                  {T({S("aten::add"), S("Tensor"), I(3)}), T({S("aten::relu"), S("")})},
                  {I(2), S("cpu")}, {S("Tensor"), S("List[int]")}, 7,
                  {arg("self", "__torch__.M"), arg("x", "Tensor")}, {arg("", "Tensor")});
  auto r = mb::decode_bytecode(T({I(8), f}));
  REQUIRE_MESSAGE(r, err_of(r));
  CHECK(r->version == 8);
  CHECK_FALSE(r->version_implicit);
  REQUIRE(r->functions.size() == 1);
  const mb::BcFunction& g = r->functions[0];
  CHECK(g.qualified_name == "__torch__.M.forward");
  REQUIRE(g.instructions.size() == 5);
  CHECK(g.instructions[0].op == BcOp::STOREN);
  CHECK(g.instructions[0].x == 1);
  CHECK(g.instructions[0].n == 2);
  CHECK(g.instructions[3].op == BcOp::JMP);
  CHECK(g.instructions[3].x == -3);
  CHECK(g.instructions[4].op == BcOp::RET);
  REQUIRE(g.operators.size() == 2);
  CHECK(g.operators[0].name == "aten::add");
  CHECK(g.operators[0].overload == "Tensor");
  CHECK(g.operators[0].has_num_args);
  CHECK(g.operators[0].num_args == 3);
  CHECK(g.operators[1].well_formed);
  CHECK_FALSE(g.operators[1].has_num_args);
  REQUIRE(g.constants.size() == 2);
  CHECK(g.constants[1]->s == "cpu");
  CHECK(g.types == VS{"Tensor", "List[int]"});
  CHECK(g.type_ok == std::vector<uint8_t>{1, 1});
  CHECK(g.register_size == 7);
  CHECK(g.has_schema);
  REQUIRE(g.arguments.size() == 2);
  CHECK(g.arguments[0].name == "self");
  CHECK(g.arguments[0].type == "__torch__.M");
  CHECK(g.arguments[1].name == "x");
  REQUIRE(g.returns.size() == 1);
  CHECK(g.returns[0].type == "Tensor");
  CHECK(mb::method_signature(g) == "forward(self: __torch__.M, x: Tensor) -> Tensor");
}

TEST_CASE("T-U2 implicit v3: no version int, no schema") {
  auto r = mb::decode_bytecode(T({simple_fn("__torch__.M.forward", false)}));
  REQUIRE_MESSAGE(r, err_of(r));
  CHECK(r->version == 3);
  CHECK(r->version_implicit);
  REQUIRE(r->functions.size() == 1);
  CHECK_FALSE(r->functions[0].has_schema);
  CHECK(mb::method_signature(r->functions[0]).empty());
}

TEST_CASE("T-U3 v4 schema is optional") {
  auto two = mb::decode_bytecode(T({I(4), simple_fn("a.forward", false)}));
  REQUIRE(two);
  CHECK_FALSE(two->functions[0].has_schema);
  auto three = mb::decode_bytecode(T({I(4), simple_fn("a.forward", true)}));
  REQUIRE(three);
  CHECK(three->functions[0].has_schema);
}

TEST_CASE("T-U4 v5+ requires a schema") {
  auto r = mb::decode_bytecode(T({I(5), simple_fn("a.forward", false)}));
  REQUIRE_FALSE(r);
  CHECK(err_of(r).find("missing schema") != std::string::npos);
}

TEST_CASE("T-U5 unsupported versions and layouts") {
  for (int64_t v : {10, 2, -1}) {
    auto r = mb::decode_bytecode(T({I(v), simple_fn()}));
    REQUIRE_FALSE(r);
    CHECK(err_of(r).find("unsupported bytecode version") != std::string::npos);
  }
  auto s = mb::decode_bytecode(T({S("8"), simple_fn()}));
  REQUIRE_FALSE(s);
  CHECK(err_of(s).find("unrecognised top-level layout") != std::string::npos);
  CHECK_FALSE(mb::decode_bytecode(T({})));
  CHECK_FALSE(mb::decode_bytecode(L({I(8)})));
  CHECK_FALSE(mb::decode_bytecode(nullptr));
  // v9 pickles exactly like v8.
  auto v9 = mb::decode_bytecode(T({I(9), simple_fn()}));
  REQUIRE(v9);
  CHECK(v9->version == 9);
}

TEST_CASE("T-U6 code-table rows are checked by name at a fixed index") {
  ValuePtr code = T({row("instruction", T({})), row("operators", T({})),
                     row("constants", T({})), row("types", T({})),
                     row("register_size", I(0))});
  ValuePtr sch = T({row("arguments", T({})), row("returns", T({}))});
  auto r = mb::decode_bytecode(T({I(8), T({S("a.forward"), code, sch})}));
  REQUIRE_FALSE(r);
  CHECK(err_of(r).find("'instructions'") != std::string::npos);
  CHECK(err_of(r).find("index 0") != std::string::npos);
}

TEST_CASE("T-U7 instruction shape and operand ranges") {
  auto mkf = [](ValuePtr instr) {
    return mb::decode_bytecode(T({I(8), fn("a.forward", {instr}, {}, {}, {}, 0, {}, {})}));
  };
  auto r1 = mkf(T({S("RET"), I(0)}));
  REQUIRE_FALSE(r1);
  CHECK(err_of(r1).find("is not (str, int, int)") != std::string::npos);
  auto r2 = mkf(ins("JF", int64_t{1} << 33, 0));
  REQUIRE_FALSE(r2);
  CHECK(err_of(r2).find("X out of range") != std::string::npos);
  auto r3 = mkf(ins("OPN", 0, 70000));
  REQUIRE_FALSE(r3);
  CHECK(err_of(r3).find("N out of range") != std::string::npos);
  CHECK(mkf(ins("JF", INT32_MIN, 65535)));
}

TEST_CASE("T-U8 unknown opcodes are kept, never mapped to OP") {
  std::string longop(100, 'Z');
  auto r = mb::decode_bytecode(T({I(8), fn("a.forward",
                                           {ins("FROB", 1, 0), ins(longop, 0, 0),
                                            ins("FROB", 2, 0)},
                                           {}, {}, {}, 0, {}, {})}));
  REQUIRE_MESSAGE(r, err_of(r));
  const mb::BcFunction& f = r->functions[0];
  CHECK(f.instructions[0].op == BcOp::Unknown);
  REQUIRE(f.instructions[0].unknown_name < f.unknown_opcodes.size());
  CHECK(f.unknown_opcodes[f.instructions[0].unknown_name] == "FROB");
  CHECK(f.unknown_opcodes[f.instructions[1].unknown_name] ==
        std::string(32, 'Z') + "\xE2\x80\xA6");
  CHECK(f.instructions[2].unknown_name == f.instructions[0].unknown_name);
  CHECK(f.unknown_opcodes.size() == 2);
}

TEST_CASE("T-U9 malformed operator entries decode as not well-formed") {
  auto r = mb::decode_bytecode(T({I(8), fn("a.forward", {}, {T({I(1), S("")}), I(3),
                                                             T({S("a::b"), S(""), S("x")})},
                                           {}, {}, 0, {}, {})}));
  REQUIRE_MESSAGE(r, err_of(r));
  const auto& ops = r->functions[0].operators;
  REQUIRE(ops.size() == 3);
  CHECK_FALSE(ops[0].well_formed);
  CHECK_FALSE(ops[1].well_formed);
  CHECK(ops[2].well_formed);
  CHECK_FALSE(ops[2].has_num_args);  // present but not an Int
}

TEST_CASE("T-U10 caps are enforced before iteration") {
  {
    mb::BytecodeLimits lim;
    lim.max_instructions = 4;
    std::vector<ValuePtr> five(5, ins("RET", 0, 0));
    auto r = mb::decode_bytecode(T({I(8), fn("a.forward", five, {}, {}, {}, 0, {}, {})}), lim);
    REQUIRE_FALSE(r);
    CHECK(err_of(r).find("exceed the cap") != std::string::npos);
  }
  {
    mb::BytecodeLimits lim;
    lim.max_functions = 1;
    auto r = mb::decode_bytecode(T({I(8), simple_fn("a.forward"), simple_fn("a.g")}), lim);
    REQUIRE_FALSE(r);
    CHECK(err_of(r).find("functions exceed") != std::string::npos);
  }
  {
    auto r = mb::decode_bytecode(T({I(8), simple_fn(std::string(5000, 'n'))}));
    REQUIRE_FALSE(r);
    CHECK(err_of(r).find("method name") != std::string::npos);
  }
  {
    auto r = mb::decode_bytecode(T({I(8), fn("a.forward", {}, {}, {}, {}, -1, {}, {})}));
    REQUIRE_FALSE(r);
    CHECK(err_of(r).find("register_size") != std::string::npos);
  }
  {
    // A non-string type entry is kept as "?" (type_ok == 0), not an error.
    auto r = mb::decode_bytecode(T({I(8), fn("a.forward", {}, {}, {}, {I(5)}, 0, {}, {})}));
    REQUIRE(r);
    CHECK(r->functions[0].types == VS{"?"});
    CHECK(r->functions[0].type_ok == std::vector<uint8_t>{0});
  }
}

TEST_CASE("T-U11 select_main_function") {
  auto names = [](std::vector<std::string> qs) {
    mb::BcModule m;
    for (auto& q : qs) {
      mb::BcFunction f;
      f.qualified_name = q;
      m.functions.push_back(f);
    }
    return m;
  };
  CHECK(mb::select_main_function(names({"__torch__.Sub.forward", "__torch__.M.forward"}),
                                 "__torch__.M") == 1);
  CHECK(mb::select_main_function(names({"__torch__.M.helper", "__torch__.Sub.forward"}),
                                 "__torch__.M") == 1);
  CHECK(mb::select_main_function(names({"__torch__.Sub.forward"}), "") == 0);
  CHECK(mb::select_main_function(names({"__torch__.M.__setstate__", "__torch__.M.helper"}),
                                 "__torch__.M") == 1);
  CHECK(mb::select_main_function(names({"__torch__.M.__getstate__"}), "__torch__.M") == -1);
  CHECK(mb::select_main_function(names({"__torch__.X.helper"}), "") == -1);
  CHECK(mb::select_main_function(names({}), "__torch__.M") == -1);
}

TEST_CASE("T-U12 select_bytecode_entry") {
  CHECK(mb::select_bytecode_entry({"a/data.pkl", "b/bytecode.pkl", "a/bytecode.pkl"},
                                  "a/data.pkl") == 2);
  CHECK(mb::select_bytecode_entry({"a/data.pkl", "b/bytecode.pkl", "a/bytecode.pkl"}, "") == 1);
  CHECK(mb::select_bytecode_entry({"x/extra/bytecode.pkl", "x/bytecode.pkl"}, "") == 1);
  CHECK(mb::select_bytecode_entry({"a/data.pkl", "a/extra/bytecode.pkl"}, "a/data.pkl") == -1);
  CHECK(mb::select_bytecode_entry({"bytecode.pkl.bak", "xbytecode.pkl"}, "") == -1);
  CHECK(mb::select_bytecode_entry({"data.pkl", "bytecode.pkl"}, "data.pkl") == 1);
  CHECK(mb::select_bytecode_entry({}, "") == -1);
}

// ============================================================================
// T-B: build
// ============================================================================
TEST_CASE("T-B1 inline expression tree with one pending output") {
  Run r;
  build_into(r, mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::OP, 0),
                             In(BcOp::OP, 1), In(BcOp::RET)},
                            {Op("aten::neg", "", 1), Op("aten::relu", "", 1)}, {}, 1,
                            {Arg("x", "Tensor")}, {Arg("", "Tensor")})));
  REQUIRE_MESSAGE(r.ok(), r.error());
  REQUIRE(r.model.graphs.size() == 1);
  const ir::Graph& g = r.model.graphs[0];
  CHECK(str(r.model, g.name) == "__torch__.M.forward");
  CHECK(ops_of(r.model, g) == VS{"aten::neg", "aten::relu"});
  CHECK(ins_of(r.model, g, 0) == VS{"x"});
  CHECK(outs_of(r.model, g, 0) == VS{"%1"});
  CHECK(ins_of(r.model, g, 1) == VS{"%1"});
  CHECK(names_of(r.model, g, g.graph_outputs) == VS{"%2"});
  CHECK(names_of(r.model, g, g.graph_inputs) == VS{"x"});
  CHECK((*r.res)->main_graph == 0);
}

TEST_CASE("T-B2 a zero-output op is retracted at the next checkpoint") {
  Run r;
  build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::LOAD, 1), In(BcOp::LOAD, 2),
                             In(BcOp::LOADC, 0), In(BcOp::OP, 0), In(BcOp::MOVE, 2),
                             In(BcOp::OP, 1), In(BcOp::STORE, 3), In(BcOp::DROPR, 1),
                             In(BcOp::MOVE, 3), In(BcOp::RET)},
                            {Op("aten::_set_item", "str", 3), Op("aten::relu", "", 1)},
                            {S("k")}, 3, {Arg("d", "Dict(str, Tensor)"), Arg("x", "Tensor")},
                            {Arg("", "Tensor")})));
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ops_of(r.model, g) == VS{"aten::_set_item", "aten::relu"});
  CHECK(ins_of(r.model, g, 0) == VS{"d", "x"});
  CHECK(outs_of(r.model, g, 0).empty());
  CHECK(attr_s(r.model, g, 0, "overload") == "str");
  CHECK(attr_s(r.model, g, 0, "arg2") == "'k'");
  CHECK(outs_of(r.model, g, 1) == VS{"%1"});
  CHECK(names_of(r.model, g, g.graph_outputs) == VS{"%1"});
}

TEST_CASE("T-B3 STOREN after an op gives it N outputs") {
  Run r;
  build_into(r, mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::LOADC, 0),
                             In(BcOp::OP, 0), In(BcOp::STOREN, 2, 2), In(BcOp::MOVE, 2),
                             In(BcOp::MOVE, 3), In(BcOp::OP, 1), In(BcOp::RET)},
                            {Op("aten::max", "dim", 2), Op("aten::add", "Tensor", 2)},
                            {I(1)}, 3, {Arg("x", "Tensor")}, {Arg("", "Tensor")})));
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ops_of(r.model, g) == VS{"aten::max", "aten::add"});
  CHECK(outs_of(r.model, g, 0) == VS{"%1", "%2"});
  CHECK(attr_i(r.model, g, 0, "arg1") == 1);
  CHECK(ins_of(r.model, g, 1) == VS{"%1", "%2"});
  CHECK(names_of(r.model, g, g.graph_outputs) == VS{"%3"});
}

TEST_CASE("T-B4 OPN takes exactly N inputs") {
  Run r;
  build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 4), In(BcOp::MOVE, 1), In(BcOp::MOVE, 2),
                             In(BcOp::MOVE, 3), In(BcOp::MOVE, 4), In(BcOp::OPN, 0, 3),
                             In(BcOp::OP, 1), In(BcOp::RET)},
                            {Op("aten::stack", "", -1), Op("aten::add", "Tensor", 2)}, {},
                            4,
                            {Arg("a", "Tensor"), Arg("b", "Tensor"), Arg("c", "Tensor"),
                             Arg("d", "Tensor")},
                            {Arg("", "Tensor")})));
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ins_of(r.model, g, 0) == VS{"b", "c", "d"});
  CHECK(outs_of(r.model, g, 0) == VS{"%1"});  // inline: `a` is still waiting below
  CHECK(ins_of(r.model, g, 1) == VS{"a", "%1"});
}

TEST_CASE("T-B5 constants fold, tensors become initializers") {
  ValuePtr tc = mk(Value::Kind::Tensor);
  tc->tensor.dtype = ir::DType::F32;
  tc->tensor.shape.push_back(2);
  tc->tensor.file_offset = 100;
  tc->tensor.byte_len = 8;
  Run r;
  build_into(r, mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::LOADC, 0),
                             In(BcOp::LOADC, 1), In(BcOp::LOADC, 2), In(BcOp::LOADC, 3),
                             In(BcOp::OP, 0), In(BcOp::LOADC, 4), In(BcOp::RET)},
                            {Op("custom::f", "", 5)},
                            {I(3), L({I(1), I(2)}), N(), tc, I(7)}, 1, {Arg("x", "Tensor")},
                            {Arg("", "Tensor"), Arg("", "int")})));
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ops_of(r.model, g) == VS{"custom::f", "prim::Constant"});
  CHECK(ins_of(r.model, g, 0) == VS{"x", "forward.c3"});
  CHECK(attr_i(r.model, g, 0, "arg1") == 3);
  const ir::AttrValue* l = attr_of(r.model, g, 0, "arg2");
  REQUIRE(l);
  CHECK(l->kind == ir::AttrValue::Kind::Ints);
  CHECK(l->ints == std::vector<int64_t>{1, 2});
  CHECK(attr_s(r.model, g, 0, "arg3") == "None");
  CHECK(attr_i(r.model, g, 1, "value") == 7);
  CHECK(names_of(r.model, g, g.graph_outputs) == VS{"%1", "%2"});
  CHECK(init_names(r.model, g) == VS{"forward.c3"});
  REQUIRE(g.initializers.size() == 1);
  CHECK(g.initializers[0].file_offset == 100);
  CHECK(g.initializers[0].byte_len == 8);
}

TEST_CASE("T-B6 If: stored, inline, dropped, raising, nested, two outputs") {
  const std::vector<mb::BcArg> xc = {Arg("x", "Tensor"), Arg("c", "bool")};
  const std::vector<mb::BcArg> ret1 = {Arg("", "Tensor")};

  SUBCASE("stored (k = 1)") {
    Run r;
    build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::JF, 4),
                               In(BcOp::LOAD, 1), In(BcOp::OP, 0), In(BcOp::JMP, 3),
                               In(BcOp::LOAD, 1), In(BcOp::OP, 1), In(BcOp::STORE, 3),
                               In(BcOp::DROPR, 1), In(BcOp::MOVE, 3), In(BcOp::RET)},
                              {Op("aten::relu", "", 1), Op("aten::neg", "", 1)}, {}, 3, xc,
                              ret1)));
    REQUIRE_MESSAGE(r.ok(), r.error());
    REQUIRE(r.model.graphs.size() == 3);
    const ir::Graph& g = r.model.graphs[0];
    CHECK(ops_of(r.model, g) == VS{"prim::If"});
    CHECK(ins_of(r.model, g, 0) == VS{"c", "x"});
    CHECK(outs_of(r.model, g, 0) == VS{"%3"});
    CHECK(g.nodes[0].subgraph == 1);
    CHECK(attr_g(r.model, g, 0, "then_branch") == 1);
    CHECK(attr_g(r.model, g, 0, "else_branch") == 2);
    CHECK(attr_i(r.model, g, 0, "captures") == 1);
    CHECK(attr_names(r.model, g, 0) == VS{"then_branch", "else_branch", "captures"});
    const ir::Graph& t = r.model.graphs[1];
    CHECK(str(r.model, t.name) == "__torch__.M.forward/if0.then");
    CHECK(str(r.model, r.model.graphs[2].name) == "__torch__.M.forward/if0.else");
    CHECK(names_of(r.model, t, t.graph_inputs) == VS{"x"});
    CHECK(names_of(r.model, t, t.graph_outputs) == VS{"%1"});
    const ir::Graph& e = r.model.graphs[2];
    CHECK(ops_of(r.model, e) == VS{"aten::neg"});
    CHECK(names_of(r.model, e, e.graph_outputs) == VS{"%2"});
  }

  SUBCASE("inline into a consumer (constant branches)") {
    Run r;
    build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 1), In(BcOp::MOVE, 2),
                               In(BcOp::JF, 3), In(BcOp::LOADC, 0), In(BcOp::JMP, 2),
                               In(BcOp::LOADC, 1), In(BcOp::OP, 0), In(BcOp::RET)},
                              {Op("aten::mul", "Scalar", 2)}, {I(1), I(2)}, 2, xc, ret1)));
    REQUIRE_MESSAGE(r.ok(), r.error());
    const ir::Graph& g = r.model.graphs[0];
    CHECK(ops_of(r.model, g) == VS{"prim::If", "aten::mul"});
    CHECK(ins_of(r.model, g, 0) == VS{"c"});
    CHECK(attr_i(r.model, g, 0, "captures") == 0);
    CHECK(outs_of(r.model, g, 0) == VS{"%3"});
    CHECK(ins_of(r.model, g, 1) == VS{"x", "%3"});
    const ir::Graph& t = r.model.graphs[1];
    CHECK(ops_of(r.model, t) == VS{"prim::Constant"});
    CHECK(attr_i(r.model, t, 0, "value") == 1);
    CHECK(names_of(r.model, t, t.graph_outputs) == VS{"%1"});
    CHECK(names_of(r.model, g, g.graph_outputs) == VS{"%4"});
  }

  SUBCASE("dropped: the pending If is confirmed by DROP") {
    Run r;
    build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::JF, 4),
                               In(BcOp::LOAD, 1), In(BcOp::OP, 0), In(BcOp::JMP, 3),
                               In(BcOp::LOAD, 1), In(BcOp::OP, 1), In(BcOp::DROP),
                               In(BcOp::MOVE, 1), In(BcOp::RET)},
                              {Op("aten::relu", "", 1), Op("aten::neg", "", 1)}, {}, 2, xc,
                              ret1)));
    REQUIRE_MESSAGE(r.ok(), r.error());
    const ir::Graph& g = r.model.graphs[0];
    CHECK(outs_of(r.model, g, 0) == VS{"%3"});
    CHECK(outs_of(r.model, r.model.graphs[1], 0) == VS{"%1"});
    CHECK(outs_of(r.model, r.model.graphs[2], 0) == VS{"%2"});
    CHECK(names_of(r.model, g, g.graph_outputs) == VS{"x"});
  }

  SUBCASE("zero outputs: the then-branch raises") {
    Run r;
    build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::JF, 4),
                               In(BcOp::LOADC, 0), In(BcOp::OP, 0), In(BcOp::JMP, 1),
                               In(BcOp::MOVE, 1), In(BcOp::RET)},
                              {Op("prim::RaiseException", "", 1)}, {S("boom")}, 2, xc,
                              ret1)));
    REQUIRE_MESSAGE(r.ok(), r.error());
    const ir::Graph& g = r.model.graphs[0];
    CHECK(ops_of(r.model, g) == VS{"prim::If"});
    CHECK(outs_of(r.model, g, 0).empty());
    const ir::Graph& t = r.model.graphs[1];
    CHECK(ops_of(r.model, t) == VS{"prim::RaiseException"});
    CHECK(outs_of(r.model, t, 0).empty());
    CHECK(attr_s(r.model, t, 0, "arg0") == "'boom'");
    CHECK(t.graph_outputs.empty());
    CHECK(r.model.graphs[2].nodes.empty());
  }

  SUBCASE("nested: captures propagate two levels") {
    Run r;
    build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 3), In(BcOp::LOAD, 2), In(BcOp::JF, 9),
                               In(BcOp::LOAD, 3), In(BcOp::JF, 4), In(BcOp::LOAD, 1),
                               In(BcOp::OP, 0), In(BcOp::JMP, 3), In(BcOp::LOAD, 1),
                               In(BcOp::OP, 1), In(BcOp::JMP, 3), In(BcOp::LOAD, 1),
                               In(BcOp::OP, 2), In(BcOp::STORE, 4), In(BcOp::MOVE, 4),
                               In(BcOp::RET)},
                              {Op("aten::relu", "", 1), Op("aten::neg", "", 1),
                               Op("aten::abs", "", 1)},
                              {}, 4, {Arg("x", "Tensor"), Arg("c", "bool"), Arg("d", "bool")},
                              ret1)));
    REQUIRE_MESSAGE(r.ok(), r.error());
    REQUIRE(r.model.graphs.size() == 5);
    const ir::Graph& g = r.model.graphs[0];
    CHECK(ins_of(r.model, g, 0) == VS{"c", "d", "x"});
    CHECK(attr_i(r.model, g, 0, "captures") == 2);
    CHECK(attr_g(r.model, g, 0, "then_branch") == 1);
    const ir::Graph& ot = r.model.graphs[1];
    CHECK(names_of(r.model, ot, ot.graph_inputs) == VS{"d", "x"});
    CHECK(ops_of(r.model, ot) == VS{"prim::If"});
    CHECK(ins_of(r.model, ot, 0) == VS{"d", "x"});
    CHECK(attr_g(r.model, ot, 0, "then_branch") == 3);
    CHECK(attr_g(r.model, ot, 0, "else_branch") == 4);
    CHECK(str(r.model, r.model.graphs[3].name) ==
          "__torch__.M.forward/if0.then/if1.then");
    const ir::Graph& it = r.model.graphs[3];
    CHECK(names_of(r.model, it, it.graph_inputs) == VS{"x"});
    CHECK(outs_of(r.model, it, 0) == VS{"%1"});
    CHECK(outs_of(r.model, ot, 0) == VS{"%3"});
    CHECK(outs_of(r.model, r.model.graphs[2], 0) == VS{"%4"});
    CHECK(outs_of(r.model, g, 0) == VS{"%5"});
  }

  SUBCASE("two outputs: STOREN after the join is not credited to the else's last op") {
    Run r;
    build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::LOADC, 0), In(BcOp::JF, 4),
                               In(BcOp::LOAD, 1), In(BcOp::LOAD, 2), In(BcOp::JMP, 3),
                               In(BcOp::OP, 0), In(BcOp::OP, 1), In(BcOp::STOREN, 3, 2),
                               In(BcOp::MOVE, 3), In(BcOp::MOVE, 4), In(BcOp::OP, 2),
                               In(BcOp::RET)},
                              {Op("custom::f", "", 0), Op("custom::g", "", 0),
                               Op("aten::add", "Tensor", 2)},
                              {B(true)}, 4, {Arg("x", "Tensor"), Arg("y", "Tensor")}, ret1)));
    REQUIRE_MESSAGE(r.ok(), r.error());
    const ir::Graph& g = r.model.graphs[0];
    CHECK(ops_of(r.model, g) == VS{"prim::Constant", "prim::If", "aten::add"});
    CHECK(attr_s(r.model, g, 0, "value") == "True");
    CHECK(outs_of(r.model, g, 1).size() == 2);
    const ir::Graph& e = r.model.graphs[2];
    CHECK(ops_of(r.model, e) == VS{"custom::f", "custom::g"});
    CHECK(outs_of(r.model, e, 0).size() == 1);
    CHECK(outs_of(r.model, e, 1).size() == 1);
    CHECK(e.graph_outputs.size() == 2);
    const ir::Graph& t = r.model.graphs[1];
    CHECK(names_of(r.model, t, t.graph_outputs) == VS{"x", "y"});
  }
}

TEST_CASE("T-B7 If branches that disagree on arity are an error") {
  Run r;
  build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::JF, 3),
                             In(BcOp::LOAD, 1), In(BcOp::JMP, 1), In(BcOp::STORE, 3),
                             In(BcOp::MOVE, 3), In(BcOp::RET)},
                            {}, {}, 3, {Arg("x", "Tensor"), Arg("c", "bool")},
                            {Arg("", "Tensor")})));
  REQUIRE_FALSE(r.ok());
  CHECK(r.error().find("disagree on output count") != std::string::npos);
  CHECK(r.model.graphs.empty());
  CHECK((*r.res).error().offset == 1002);  // the JF instruction
}

TEST_CASE("T-B8 Loop with a carried value") {
  Run r;
  build_into(r, mod(make_fn({In(BcOp::STORE, 1), In(BcOp::LOADC, 0), In(BcOp::LOADC, 1),
                             In(BcOp::LOADC, 2), In(BcOp::LOAD, 1), In(BcOp::LOOP, 6, 3),
                             In(BcOp::STOREN, 2, 2), In(BcOp::LOADC, 2), In(BcOp::MOVE, 3),
                             In(BcOp::OP, 0), In(BcOp::JMP, -5), In(BcOp::STORE, 4),
                             In(BcOp::DROPR, 1), In(BcOp::MOVE, 4), In(BcOp::RET)},
                            {Op("aten::relu", "", 1)}, {I(0), I(10), B(true)}, 4,
                            {Arg("x", "Tensor")}, {Arg("", "Tensor")})));
  REQUIRE_MESSAGE(r.ok(), r.error());
  REQUIRE(r.model.graphs.size() == 2);
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ops_of(r.model, g) == VS{"prim::Constant", "prim::Constant", "prim::Loop"});
  CHECK(attr_i(r.model, g, 0, "value") == 10);
  CHECK(attr_s(r.model, g, 1, "value") == "True");
  CHECK(ins_of(r.model, g, 2) == VS{"%1", "%2", "x"});
  CHECK(outs_of(r.model, g, 2) == VS{"%7"});
  CHECK(g.nodes[2].subgraph == 1);
  CHECK(attr_g(r.model, g, 2, "body") == 1);
  CHECK(attr_i(r.model, g, 2, "captures") == 0);
  const ir::Graph& body = r.model.graphs[1];
  CHECK(str(r.model, body.name) == "__torch__.M.forward/loop0.body");
  CHECK(names_of(r.model, body, body.graph_inputs) == VS{"%3", "%4"});
  CHECK(ops_of(r.model, body) == VS{"aten::relu", "prim::Constant"});
  CHECK(ins_of(r.model, body, 0) == VS{"%4"});
  CHECK(names_of(r.model, body, body.graph_outputs) == VS{"%6", "%5"});
  CHECK(names_of(r.model, g, g.graph_outputs) == VS{"%7"});
}

namespace {
// `depth` nested `if c:` statements with empty else-blocks.
mb::BcModule nested_ifs(int depth) {
  std::vector<mb::BcInstruction> code;
  code.push_back(In(BcOp::STORE, 1));
  std::function<void(int)> emit = [&](int level) {
    if (level == depth) return;
    size_t a = code.size();
    code.push_back(In(BcOp::LOAD, 1));
    code.push_back(In(BcOp::JF, 0));
    emit(level + 1);
    size_t j = code.size();
    code[a + 1].x = static_cast<int32_t>(j - a);
    code.push_back(In(BcOp::JMP, 1));
  };
  emit(0);
  code.push_back(In(BcOp::LOADC, 0));
  code.push_back(In(BcOp::RET));
  return mod(make_fn(code, {}, {N()}, 1, {Arg("c", "bool")}, {Arg("", "NoneType")}));
}
}  // namespace

TEST_CASE("T-B9 control-flow depth cap") {
  Run ok;
  build_into(ok, nested_ifs(64));
  REQUIRE_MESSAGE(ok.ok(), ok.error());
  CHECK(ok.model.graphs.size() == 1 + 2 * 64);
  Run deep;
  build_into(deep, nested_ifs(65));
  REQUIRE_FALSE(deep.ok());
  CHECK(deep.error().find("nested deeper than 64") != std::string::npos);
  CHECK(deep.model.graphs.empty());
  // A subgraph cap below what the method needs fails the same honest way.
  Run capped;
  mb::BytecodeLimits lim;
  lim.max_subgraphs = 10;
  build_into(capped, nested_ifs(6), empty_env(), no_names(), lim);
  REQUIRE_FALSE(capped.ok());
  CHECK(capped.error().find("subgraphs") != std::string::npos);
}

namespace {
ValuePtr tensor_v(uint64_t off, uint64_t len, std::vector<int64_t> shape) {
  ValuePtr t = mk(Value::Kind::Tensor);
  t->tensor.dtype = ir::DType::F32;
  for (int64_t d : shape) t->tensor.shape.push_back(d);
  t->tensor.file_offset = off;
  t->tensor.byte_len = len;
  return t;
}
ValuePtr obj(const std::string& module, const std::string& name,
             std::vector<std::pair<std::string, ValuePtr>> attrs) {
  ValuePtr v = mk(Value::Kind::Opaque);
  v->module = module;
  v->name = name;
  ValuePtr st = mk(Value::Kind::Dict);
  for (auto& a : attrs) st->pairs.emplace_back(S(a.first), a.second);
  v->inner = st;
  return v;
}
// forward(self, x) = custom::f(x, self.a.b.weight, self.scale, self.packed,
//                              self.w, self.bias)
mb::BcModule binding_module() {
  return mod(make_fn(
      {In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::LOAD, 1), In(BcOp::GET_ATTR, 1),
       In(BcOp::GET_ATTR, 0), In(BcOp::GET_ATTR, 0), In(BcOp::LOAD, 1), In(BcOp::GET_ATTR, 2),
       In(BcOp::LOAD, 1), In(BcOp::GET_ATTR, 3), In(BcOp::LOAD, 1), In(BcOp::GET_ATTR, 4),
       In(BcOp::MOVE, 1), In(BcOp::GET_ATTR, 5), In(BcOp::OP, 0), In(BcOp::RET)},
      {Op("custom::f", "", 6)}, {}, 2, {Arg("self", "__torch__.M"), Arg("x", "Tensor")},
      {Arg("", "Tensor")}));
}
}  // namespace

TEST_CASE("T-B10 GET_ATTR binds module attributes") {
  ValuePtr W = tensor_v(64, 32, {2, 4});
  ValuePtr Bt = tensor_v(96, 8, {2});
  ValuePtr packed = mk(Value::Kind::Opaque);
  packed->module = "__torch__.torch.classes.quantized";
  packed->name = "LinearPackedParamsBase";
  packed->inner = T({I(1)});
  ValuePtr b = obj("__torch__", "Bmod", {{"weight", W}});
  ValuePtr a = obj("__torch__", "Amod", {{"b", b}});
  ValuePtr root = obj("__torch__", "M",
                      {{"training", B(true)}, {"a", a}, {"scale", I(3)}, {"packed", packed},
                       {"w", W}, {"bias", Bt}});
  ts::ModuleEnv env(root);
  Run r;
  build_into(r, binding_module(), env);
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(names_of(r.model, g, g.graph_inputs) == VS{"self", "x"});
  CHECK(ops_of(r.model, g) == VS{"prim::GetAttr", "custom::f"});
  CHECK(ins_of(r.model, g, 0) == VS{"self"});
  CHECK(attr_s(r.model, g, 0, "name") == "packed");
  CHECK(attr_i(r.model, g, 0, "slot") == 3);
  CHECK(outs_of(r.model, g, 0) == VS{"packed"});
  // The tied weight self.w resolves to its canonical name a.b.weight.
  CHECK(ins_of(r.model, g, 1) == VS{"x", "a.b.weight", "packed", "a.b.weight", "bias"});
  CHECK(attr_i(r.model, g, 1, "arg2") == 3);
  CHECK(init_names(r.model, g) == VS{"a.b.weight", "bias"});
  CHECK(g.initializers[0].file_offset == 64);
  CHECK(g.values[g.graph_inputs[0]].dtype == ir::DType::Unknown);
  const auto& bound = (*r.res)->bound_module_tensors;
  REQUIRE(bound.size() == 2);
  CHECK(bound[0] == W.get());
  CHECK(bound[1] == Bt.get());
  CHECK((*r.res)->self_note.empty());
}

TEST_CASE("T-B11 unbound self becomes the first graph input") {
  Run r;
  build_into(r, binding_module());
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(names_of(r.model, g, g.graph_inputs) == VS{"self", "x"});
  VS ops = ops_of(r.model, g);
  REQUIRE(ops.size() == 8);  // 7 GET_ATTRs + custom::f
  for (size_t i = 0; i < 7; ++i) {
    CHECK(ops[i] == "prim::GetAttr");
    CHECK(attr_of(r.model, g, i, "name") == nullptr);
  }
  CHECK(ops[7] == "custom::f");
  CHECK(attr_i(r.model, g, 0, "slot") == 1);
  CHECK(ins_of(r.model, g, 0) == VS{"self"});
  CHECK(ins_of(r.model, g, 1) == outs_of(r.model, g, 0));
  CHECK(g.initializers.empty());
  CHECK((*r.res)->self_note.find("no data.pkl") != std::string::npos);
  CHECK((*r.res)->bound_module_tensors.empty());

  // A schema type that does not match the data.pkl root is also unbound.
  ts::ModuleEnv other(obj("__torch__", "Other", {}));
  Run r2;
  build_into(r2, binding_module(), other);
  REQUIRE_MESSAGE(r2.ok(), r2.error());
  CHECK((*r2.res)->self_note ==
        "self not bound: schema type __torch__.M != data.pkl root __torch__.Other");
}

TEST_CASE("T-B12 SET_ATTR disables binding of that slot") {
  ts::ModuleEnv env(obj("__torch__", "M", {{"training", B(true)}, {"count", I(0)}}));
  Run r;
  build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::LOAD, 1), In(BcOp::LOADC, 0),
                             In(BcOp::SET_ATTR, 1), In(BcOp::MOVE, 1), In(BcOp::GET_ATTR, 1),
                             In(BcOp::MOVE, 2), In(BcOp::OP, 0), In(BcOp::RET)},
                            {Op("aten::add", "Tensor", 2)}, {I(5)}, 2,
                            {Arg("self", "__torch__.M"), Arg("x", "Tensor")},
                            {Arg("", "Tensor")})),
             env);
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ops_of(r.model, g) == VS{"prim::SetAttr", "prim::GetAttr", "aten::add"});
  CHECK(ins_of(r.model, g, 0) == VS{"self"});
  CHECK(attr_s(r.model, g, 0, "name") == "count");
  CHECK(attr_i(r.model, g, 0, "slot") == 1);
  CHECK(attr_i(r.model, g, 0, "arg1") == 5);
  CHECK(outs_of(r.model, g, 0).empty());
  CHECK(attr_of(r.model, g, 1, "name") == nullptr);
  CHECK(attr_i(r.model, g, 1, "slot") == 1);
  CHECK(ins_of(r.model, g, 2) == VS{"%1", "x"});
}

TEST_CASE("T-B13 every malformed stream is an honest error") {
  const std::vector<mb::BcArg> xa = {Arg("x", "Tensor")};
  const std::vector<mb::BcArg> xc = {Arg("x", "Tensor"), Arg("c", "bool")};
  const std::vector<mb::BcArg> r1 = {Arg("", "Tensor")};
  auto expect = [](const mb::BcModule& m, const std::string& needle) {
    Run r;
    build_into(r, m);
    INFO("expected: " << needle << " got: " << r.error());
    REQUIRE_FALSE(r.ok());
    CHECK(r.error().find(needle) != std::string::npos);
    CHECK(r.model.graphs.empty());
  };
  const std::vector<mb::BcOperator> relu = {Op("aten::relu", "", 1)};

  expect(mod(make_fn({In(BcOp::MOVE, 1), In(BcOp::RET)}, {}, {}, 1, xa, r1)),
         "read of undefined register 1");
  expect(mod(make_fn({In(BcOp::STORE, 5), In(BcOp::MOVE, 5), In(BcOp::RET)}, {}, {}, 2, xa, r1)),
         "exceeds register_size");
  expect(mod(make_fn({In(BcOp::STOREN, INT32_MAX, 2), In(BcOp::RET)}, {}, {}, 4, xc, r1)),
         "exceeds register_size");
  expect(mod(make_fn({In(BcOp::STORE, 0), In(BcOp::RET)}, {}, {}, 2, xa, r1)),
         "invalid register 0");
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::LOADC, 3), In(BcOp::RET)}, {}, {I(1)}, 1, xa,
                     r1)),
         "constant index 3 out of range");
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::OP, 4), In(BcOp::RET)},
                     relu, {}, 1, xa, r1)),
         "operator index 4 out of range");
  {
    mb::BcOperator noarity;
    noarity.name = "aten::relu";
    expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::OP, 0), In(BcOp::RET)},
                       {noarity}, {}, 1, xa, r1)),
           "no recorded arity");
  }
  // A branch may not pop a value that belongs to the enclosing block.
  expect(mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 1), In(BcOp::MOVE, 2),
                      In(BcOp::JF, 3), In(BcOp::OP, 0), In(BcOp::JMP, 1), In(BcOp::RET)},
                     relu, {}, 2, xc, r1)),
         "underflow below block base");
  expect(mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::JF, 100),
                      In(BcOp::MOVE, 1), In(BcOp::RET)},
                     {}, {}, 2, xc, r1)),
         "malformed if");
  expect(mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::JF, 2),
                      In(BcOp::MOVE, 1), In(BcOp::RET)},
                     {}, {}, 2, xc, r1)),
         "malformed if");
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::JMP, 1), In(BcOp::MOVE, 1), In(BcOp::RET)},
                     {}, {}, 1, xa, r1)),
         "unstructured JMP");
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::LOADC, 0), In(BcOp::LOADC, 0),
                      In(BcOp::LOADC, 0), In(BcOp::LOAD, 1), In(BcOp::LOOP, 3, 3),
                      In(BcOp::DROP), In(BcOp::JMP, -1), In(BcOp::MOVE, 1), In(BcOp::RET)},
                     {}, {I(0)}, 1, xa, r1)),
         "malformed loop");
  expect(mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::JF, 3),
                      In(BcOp::MOVE, 1), In(BcOp::RET), In(BcOp::JMP, 1)},
                     {}, {}, 2, xc, r1)),
         "malformed if");
  expect(mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::JF, 4),
                      In(BcOp::MOVE, 1), In(BcOp::RET), In(BcOp::JMP, 1), In(BcOp::RET)},
                     {}, {}, 2, xc, r1)),
         "RET inside a branch");
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1)}, {}, {}, 1, xa, r1)),
         "missing RET");
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::RET), In(BcOp::DROP)},
                     {}, {}, 1, xa, r1)),
         "trailing instructions after RET");
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::FORK), In(BcOp::RET)}, {}, {}, 1, xa, r1)),
         "instruction FORK is not valid in mobile bytecode");
  {
    mb::BcFunction f = make_fn({In(BcOp::STORE, 1), In(BcOp::Unknown), In(BcOp::RET)}, {}, {},
                               1, xa, r1);
    f.unknown_opcodes = {"FROB"};
    f.instructions[1].unknown_name = 0;
    expect(mod(f), "pc 1: unknown instruction 'FROB'");
  }
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::LOAD, 1), In(BcOp::LOAD, 1),
                      In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::RET)},
                     {}, {}, 1, xa, r1)),
         "stack not at block base after STORE (1 residual values)");
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::LOAD, 1), In(BcOp::MOVE, 1), In(BcOp::RET)},
                     {}, {}, 1, xa, r1)),
         "values left on stack at RET");
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::RET)}, {}, {}, 1, xa, r1),
             5),
         "operator arity");
  {
    mb::BcFunction f = make_fn({In(BcOp::RET)}, {}, {}, 0, {}, {});
    f.has_schema = false;
    expect(mod(f), "no schema");
  }
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::LIST_CONSTRUCT, 3, 0), In(BcOp::RET)}, {},
                     {}, 1, xa, r1, {"Tensor"})),
         "types index 3 out of range");
  // A hostile output count hits the symbolic stack cap before any allocation.
  expect(mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::LIST_UNPACK, INT32_MAX),
                      In(BcOp::RET)},
                     {}, {}, 1, xa, r1)),
         "symbolic stack deeper than");
}

TEST_CASE("T-B13b error offsets point at the failing instruction") {
  Run r;
  build_into(r, mod(make_fn({In(BcOp::STORE, 1), In(BcOp::MOVE, 1), In(BcOp::FORK),
                             In(BcOp::RET)},
                            {}, {}, 1, {Arg("x", "Tensor")}, {Arg("", "Tensor")})));
  REQUIRE_FALSE(r.ok());
  CHECK((*r.res).error().offset == 1002);
  // A caller's non-empty graph list is never touched.
  Run pre;
  pre.model.graphs.emplace_back();
  build_into(pre, mod(make_fn({In(BcOp::RET)}, {}, {}, 0, {}, {})));
  REQUIRE_FALSE(pre.ok());
  CHECK(pre.model.graphs.size() == 1);
}

TEST_CASE("T-B14 a malformed operator entry with an arity is graphed as '?'") {
  auto d = mb::decode_bytecode(T({I(8), fn("__torch__.M.forward",
                                           {ins("STORE", 1, 0), ins("MOVE", 1, 0),
                                            ins("OP", 0, 0), ins("RET", 0, 0)},
                                           {T({I(1), S(""), I(1)})}, {}, {}, 1,
                                           {arg("x", "Tensor")}, {arg("", "Tensor")})}));
  REQUIRE_MESSAGE(d, err_of(d));
  CHECK_FALSE(d->functions[0].operators[0].well_formed);
  CHECK(d->functions[0].operators[0].has_num_args);
  Run r;
  build_into(r, *d);
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ops_of(r.model, g) == VS{"?"});
  CHECK(attr_s(r.model, g, 0, "unresolved") == "operators[0] malformed");
  CHECK(attr_of(r.model, g, 0, "overload") == nullptr);
}

TEST_CASE("T-B15 a non-string type entry is labelled, not guessed") {
  auto d = mb::decode_bytecode(T({I(8), fn("__torch__.M.forward",
                                           {ins("STORE", 1, 0), ins("LOAD", 1, 0),
                                            ins("MOVE", 1, 0), ins("LIST_CONSTRUCT", 0, 2),
                                            ins("RET", 0, 0)},
                                           {}, {}, {I(5)}, 1, {arg("x", "Tensor")},
                                           {arg("", "List[Tensor]")})}));
  REQUIRE_MESSAGE(d, err_of(d));
  Run r;
  build_into(r, *d);
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ops_of(r.model, g) == VS{"prim::ListConstruct"});
  CHECK(attr_s(r.model, g, 0, "type") == "?");
  CHECK(attr_s(r.model, g, 0, "unresolved") == "types[0] is not a string");
  CHECK(ins_of(r.model, g, 0) == VS{"x", "x"});
}

namespace {
void check_same_graphs(const ir::Model& a, const ir::Model& b) {
  REQUIRE(a.graphs.size() == b.graphs.size());
  for (size_t gi = 0; gi < a.graphs.size(); ++gi) {
    const ir::Graph& ga = a.graphs[gi];
    const ir::Graph& gb = b.graphs[gi];
    CHECK(str(a, ga.name) == str(b, gb.name));
    CHECK(ops_of(a, ga) == ops_of(b, gb));
    CHECK(ga.edge_refs == gb.edge_refs);
    REQUIRE(ga.values.size() == gb.values.size());
    for (size_t v = 0; v < ga.values.size(); ++v)
      CHECK(str(a, ga.values[v].name) == str(b, gb.values[v].name));
    REQUIRE(ga.nodes.size() == gb.nodes.size());
    for (size_t n = 0; n < ga.nodes.size(); ++n) {
      CHECK(ga.nodes[n].subgraph == gb.nodes[n].subgraph);
      CHECK(attr_names(a, ga, n) == attr_names(b, gb, n));
    }
    REQUIRE(ga.attributes.size() == gb.attributes.size());
    for (size_t i = 0; i < ga.attributes.size(); ++i) {
      const ir::AttrValue& x = ga.attributes[i].value;
      const ir::AttrValue& y = gb.attributes[i].value;
      CHECK(x.kind == y.kind);
      CHECK(x.i == y.i);
      CHECK(str(a, x.s) == str(b, y.s));
      CHECK(x.ints == y.ints);
      CHECK(x.graph == y.graph);
    }
    REQUIRE(ga.initializers.size() == gb.initializers.size());
    for (size_t i = 0; i < ga.initializers.size(); ++i) {
      CHECK(str(a, ga.initializers[i].name) == str(b, gb.initializers[i].name));
      CHECK(ga.initializers[i].file_offset == gb.initializers[i].file_offset);
    }
    CHECK(ga.graph_inputs == gb.graph_inputs);
    CHECK(ga.graph_outputs == gb.graph_outputs);
  }
}
}  // namespace

TEST_CASE("T-B16 builds are deterministic") {
  mb::BcModule m = nested_ifs(5);
  Run a, b;
  build_into(a, m);
  build_into(b, m);
  REQUIRE(a.ok());
  REQUIRE(b.ok());
  check_same_graphs(a.model, b.model);
  Run c, d;
  build_into(c, binding_module());
  build_into(d, binding_module());
  REQUIRE(c.ok());
  REQUIRE(d.ok());
  check_same_graphs(c.model, d.model);
}

// ============================================================================
// T-M: the .ptl fixtures through pytorch::parse_zip
// ============================================================================
namespace {

const char* kMobile = "tests/fixtures/model_mobile.ptl";
const char* kMobileV4 = "tests/fixtures/model_mobile_v4.ptl";
const char* kMobileBadOp = "tests/fixtures/model_mobile_badop.ptl";
const char* kMobileNoData = "tests/fixtures/model_mobile_nodata.ptl";

bool have(const char* path) {
  if (std::filesystem::exists(path)) return true;
  WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
  return false;
}

std::optional<ir::Model> parse_ptl(const char* path) {
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  auto res = pytorch::parse_zip(*mf, progress);
  REQUIRE_MESSAGE(res, (res ? std::string() : res.error().message));
  return res.take();
}

std::optional<std::string> meta(const ir::Model& m, const std::string& key) {
  for (const auto& kv : m.metadata)
    if (m.str(kv.first) == key) return std::string(m.str(kv.second));
  return std::nullopt;
}

std::vector<uint8_t> read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
}

size_t find_bytes(const std::vector<uint8_t>& hay, const void* needle, size_t n,
                  size_t* count) {
  size_t first = SIZE_MAX;
  *count = 0;
  for (size_t i = 0; i + n <= hay.size(); ++i) {
    if (std::memcmp(hay.data() + i, needle, n) == 0) {
      if (first == SIZE_MAX) first = i;
      ++*count;
    }
  }
  return first;
}

const VS kFourOps = {"aten::linear", "aten::relu", "aten::add.Tensor", "aten::mean.dim"};
std::string joined(const VS& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + v[i];
  return s;
}

void check_parameter_table(const ir::Model& m) {
  REQUIRE(m.flat_tensors.size() == 2);
  CHECK(str(m, m.flat_tensors[0].name) == "fc.weight");
  CHECK(str(m, m.flat_tensors[1].name) == "fc.bias");
  CHECK(m.flat_tensors[0].file_offset != UINT64_MAX);
  CHECK(m.flat_tensors[1].file_offset != UINT64_MAX);
}

}  // namespace

TEST_CASE("T-M1 mobile .ptl v8: forward decodes to the expected graph") {
  if (!have(kMobile)) return;
  ByteReader::payload_read_counter() = 0;
  auto pm = parse_ptl(kMobile);
  const ir::Model& m = *pm;
  CHECK(m.has_graph);
  CHECK(m.flat_tensors.empty());
  REQUIRE(m.graphs.size() == 3);
  CHECK(str(m, m.format_name) == "PyTorch");
  CHECK(str(m, m.version_info) == "bytecode v8");

  const ir::Graph& g = m.graphs[0];
  CHECK(str(m, g.name) == "__torch__.Model.forward");
  CHECK(str(m, m.graphs[1].name) == "__torch__.Model.forward/if0.then");
  CHECK(str(m, m.graphs[2].name) == "__torch__.Model.forward/if0.else");
  CHECK(names_of(m, g, g.graph_inputs) == VS{"x", "flag"});
  REQUIRE(g.initializers.size() == 2);
  CHECK(init_names(m, g) == VS{"fc.weight", "fc.bias"});
  CHECK(g.initializers[0].dtype == ir::DType::F32);
  CHECK(g.initializers[0].shape.size() == 2);
  CHECK(g.initializers[0].shape[0] == 2);
  CHECK(g.initializers[0].shape[1] == 4);
  CHECK(g.initializers[0].byte_len == 32);
  CHECK(g.initializers[1].dtype == ir::DType::F32);
  REQUIRE(g.initializers[1].shape.size() == 1);
  CHECK(g.initializers[1].shape[0] == 2);
  CHECK(g.initializers[1].byte_len == 8);
  CHECK(g.initializers[0].file_offset != UINT64_MAX);
  CHECK(g.initializers[1].file_offset != UINT64_MAX);
  CHECK(g.initializers[0].file_offset != g.initializers[1].file_offset);

  CHECK(ops_of(m, g) == VS{"aten::linear", "aten::relu", "prim::If"});
  CHECK(ins_of(m, g, 0) == VS{"x", "fc.weight", "fc.bias"});
  CHECK(outs_of(m, g, 0) == VS{"%1"});
  CHECK(ins_of(m, g, 1) == outs_of(m, g, 0));
  CHECK(outs_of(m, g, 1) == VS{"%2"});
  CHECK(ins_of(m, g, 2) == VS{"flag", "%2"});
  CHECK(g.nodes[2].subgraph == 1);
  CHECK(attr_g(m, g, 2, "then_branch") == 1);
  CHECK(attr_g(m, g, 2, "else_branch") == 2);
  CHECK(attr_i(m, g, 2, "captures") == 1);
  CHECK(outs_of(m, g, 2) == VS{"%5"});
  CHECK(names_of(m, g, g.graph_outputs) == VS{"%5"});

  const ir::Graph& t = m.graphs[1];
  CHECK(names_of(m, t, t.graph_inputs) == VS{"%2"});
  CHECK(init_names(m, t) == VS{"CONSTANTS.c0"});
  REQUIRE(t.initializers.size() == 1);
  CHECK(t.initializers[0].dtype == ir::DType::F32);
  REQUIRE(t.initializers[0].shape.size() == 1);
  CHECK(t.initializers[0].shape[0] == 2);
  CHECK(ops_of(m, t) == VS{"aten::add"});
  CHECK(ins_of(m, t, 0) == VS{"%2", "CONSTANTS.c0"});
  CHECK(attr_s(m, t, 0, "overload") == "Tensor");
  CHECK(attr_i(m, t, 0, "arg2") == 2);
  CHECK(attr_names(m, t, 0) == VS{"overload", "arg2"});
  CHECK(outs_of(m, t, 0) == VS{"%3"});
  CHECK(names_of(m, t, t.graph_outputs) == VS{"%3"});

  const ir::Graph& e = m.graphs[2];
  CHECK(names_of(m, e, e.graph_inputs) == VS{"%2"});
  CHECK(e.initializers.empty());
  CHECK(ops_of(m, e) == VS{"aten::mean"});
  CHECK(ins_of(m, e, 0) == VS{"%2"});
  CHECK(attr_s(m, e, 0, "overload") == "dim");
  const ir::AttrValue* a1 = attr_of(m, e, 0, "arg1");
  REQUIRE(a1);
  CHECK(a1->kind == ir::AttrValue::Kind::Ints);
  CHECK(a1->ints == std::vector<int64_t>{1});
  CHECK(attr_s(m, e, 0, "arg2") == "True");
  CHECK(outs_of(m, e, 0) == VS{"%4"});
  CHECK(names_of(m, e, e.graph_outputs) == VS{"%4"});

  for (const ir::Graph& gr : m.graphs) {
    std::vector<bool> is_init(gr.values.size(), false);
    for (const auto& init : gr.initializers)
      for (size_t v = 0; v < gr.values.size(); ++v)
        if (gr.values[v].name == init.name) is_init[v] = true;
    for (size_t v = 0; v < gr.values.size(); ++v) {
      if (is_init[v]) continue;
      CHECK(gr.values[v].dtype == ir::DType::Unknown);
      CHECK(gr.values[v].shape.empty());
    }
    for (const ir::Node& n : gr.nodes) CHECK(m.str(n.name).empty());
  }

  CHECK(meta(m, "torchscript.bytecode_version") == std::string("8"));
  CHECK(meta(m, "torchscript.ops") == joined(kFourOps));
  CHECK(meta(m, "torchscript.methods") == std::string("__torch__.Model.forward"));
  CHECK(meta(m, "torchscript.signature") ==
        std::string("forward(self: __torch__.Model, x: Tensor, flag: bool) -> Tensor"));
  CHECK(meta(m, "torchscript") ==
        std::string("mobile lite-interpreter bytecode v8: __torch__.Model.forward "
                    "decoded from bytecode.pkl"));
  CHECK(meta(m, "tensors") == std::string("3"));
  CHECK_FALSE(meta(m, "torchscript.unreferenced_tensors"));
  CHECK_FALSE(meta(m, "torchscript.self"));
  CHECK_FALSE(meta(m, "torchscript.bytecode"));
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("T-M2 mobile .ptl: parse is deterministic") {
  if (!have(kMobile)) return;
  ByteReader::payload_read_counter() = 0;
  auto a = parse_ptl(kMobile);
  auto b = parse_ptl(kMobile);
  check_same_graphs(*a, *b);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("T-M3 mobile .ptl: CONSTANTS.c0 shares the constants.pkl record") {
  if (!have(kMobile)) return;
  ByteReader::payload_read_counter() = 0;
  auto pm = parse_ptl(kMobile);
  std::vector<uint8_t> bytes = read_file(kMobile);
  float pat[2] = {7.5f, -3.25f};
  size_t count = 0;
  size_t at = find_bytes(bytes, pat, sizeof(pat), &count);
  REQUIRE(count == 1);
  const ir::Graph& t = pm->graphs[1];
  REQUIRE(t.initializers.size() == 1);
  CHECK(t.initializers[0].file_offset == at);
  CHECK(t.initializers[0].byte_len == 8);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("T-M4 mobile .ptl v4: exact inventory, parameters listed, no graph") {
  if (!have(kMobileV4)) return;
  ByteReader::payload_read_counter() = 0;
  auto pm = parse_ptl(kMobileV4);
  const ir::Model& m = *pm;
  CHECK_FALSE(m.has_graph);
  CHECK(m.graphs.empty());
  check_parameter_table(m);
  CHECK(str(m, m.version_info) == "bytecode v4");
  CHECK(meta(m, "torchscript.bytecode_version") == std::string("4"));
  auto ops = meta(m, "torchscript.ops");
  REQUIRE(ops);
  CHECK(*ops == joined(kFourOps));
  CHECK(ops->find("torch.relu") == std::string::npos);  // heuristic scan skipped
  auto note = meta(m, "torchscript");
  REQUIRE(note);
  CHECK(note->find("graph not built") != std::string::npos);
  CHECK(note->find("v6") != std::string::npos);
  CHECK(meta(m, "torchscript.signature") ==
        std::string("forward(self: __torch__.Model, x: Tensor, flag: bool) -> Tensor"));
  CHECK(meta(m, "tensors") == std::string("2"));
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("T-M5 mobile .ptl unknown opcode: honest fallback with location") {
  if (!have(kMobileBadOp)) return;
  ByteReader::payload_read_counter() = 0;
  auto pm = parse_ptl(kMobileBadOp);
  const ir::Model& m = *pm;
  CHECK_FALSE(m.has_graph);
  CHECK(m.graphs.empty());
  check_parameter_table(m);
  auto note = meta(m, "torchscript");
  REQUIRE(note);
  INFO(*note);
  CHECK(note->find("unknown instruction 'FROB'") != std::string::npos);
  CHECK(note->find("pc 10") != std::string::npos);
  size_t at = note->find("(byte ");
  REQUIRE(at != std::string::npos);
  uint64_t n = std::stoull(note->substr(at + 6));
  std::vector<uint8_t> bytes = read_file(kMobileBadOp);
  size_t count = 0;
  size_t q = find_bytes(bytes, "FROB", 4, &count);
  REQUIRE(count == 1);
  // "FROB", BINPUT id, BININT1 1, BININT1 0, then the instruction's TUPLE3.
  CHECK((n - q == 10 || n - q == 13));
  CHECK(bytes[n] == 0x87);
  CHECK(meta(m, "torchscript.ops") == joined(kFourOps));
  CHECK(ByteReader::payload_read_counter() == 0);
}

namespace {
// model/bytecode.pkl of the v8 fixture, extracted with miniz.
std::vector<uint8_t> fixture_bytecode_pkl() {
  std::vector<uint8_t> file = read_file(kMobile);
  mz_zip_archive zip;
  std::memset(&zip, 0, sizeof(zip));
  REQUIRE(mz_zip_reader_init_mem(&zip, file.data(), file.size(), 0));
  int idx = mz_zip_reader_locate_file(&zip, "model/bytecode.pkl", nullptr, 0);
  REQUIRE(idx >= 0);
  size_t n = 0;
  void* p = mz_zip_reader_extract_to_heap(&zip, static_cast<mz_uint>(idx), &n, 0);
  REQUIRE(p);
  std::vector<uint8_t> out(static_cast<uint8_t*>(p), static_cast<uint8_t*>(p) + n);
  mz_free(p);
  mz_zip_reader_end(&zip);
  return out;
}

// Decode `n` bytes and, if the tables decode, try to build the main method
// with no data.pkl. Either may fail; neither may crash or read a payload.
void decode_and_build(const uint8_t* data, size_t n) {
  pytorch::StorageResolver none;
  none.resolve = [](const std::string&, uint64_t&, uint64_t&) { return false; };
  auto d = mb::decode_bytecode_pickle(data, n, none);
  if (!d) {
    CHECK(d.error().offset <= n);
    return;
  }
  int32_t fi = mb::select_main_function(*d, "");
  if (fi < 0) return;
  ir::Model model;
  mb::BuildContext ctx{model, empty_env(), no_names(), {}};
  auto r = mb::build_method_graph(*d, static_cast<size_t>(fi), ctx);
  if (!r) CHECK(model.graphs.empty());
}
}  // namespace

TEST_CASE("T-M6 bytecode.pkl: every prefix decodes or errors, never crashes") {
  if (!have(kMobile)) return;
  std::vector<uint8_t> bc = fixture_bytecode_pkl();
  REQUIRE(bc.size() > 100);
  ByteReader::payload_read_counter() = 0;
  for (size_t n = 0; n < bc.size(); ++n) decode_and_build(bc.data(), n);
  // The full stream is the fixture: it must decode and graph.
  auto full = mb::decode_bytecode_pickle(bc.data(), bc.size(), pytorch::StorageResolver{});
  REQUIRE(full);
  CHECK(full->version == 8);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("T-M7 bytecode.pkl: every single-byte corruption decodes or errors") {
  if (!have(kMobile)) return;
  std::vector<uint8_t> bc = fixture_bytecode_pkl();
  ByteReader::payload_read_counter() = 0;
  for (size_t i = 0; i < bc.size(); ++i) {
    std::vector<uint8_t> c = bc;
    c[i] ^= 0xFF;
    decode_and_build(c.data(), c.size());
  }
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("T-M8 archive without data.pkl still graphs, self unbound") {
  if (!have(kMobileNoData)) return;
  ByteReader::payload_read_counter() = 0;
  {
    // The bytecode.pkl signal alone (no data.pkl, no extension) decides it.
    auto mf = MappedFile::open(kMobileNoData);
    REQUIRE(mf);
    DetectReason reason = DetectReason::None;
    CHECK(detect_format(*mf, "", reason) == Format::PyTorchZip);
    CHECK(reason == DetectReason::Magic);
  }
  auto pm = parse_ptl(kMobileNoData);
  const ir::Model& m = *pm;
  CHECK(m.has_graph);
  REQUIRE(m.graphs.size() == 3);
  const ir::Graph& g = m.graphs[0];
  CHECK(names_of(m, g, g.graph_inputs) == VS{"self", "x", "flag"});
  VS ops = ops_of(m, g);
  REQUIRE(ops.size() == 6);
  CHECK(VS(ops.begin(), ops.begin() + 3) ==
        VS{"prim::GetAttr", "prim::GetAttr", "prim::GetAttr"});
  CHECK(attr_i(m, g, 0, "slot") == 1);
  CHECK(attr_i(m, g, 1, "slot") == 1);
  CHECK(attr_i(m, g, 2, "slot") == 2);
  for (size_t i = 0; i < 3; ++i) CHECK(attr_of(m, g, i, "name") == nullptr);
  CHECK(ins_of(m, g, 0) == VS{"self"});
  CHECK(ins_of(m, g, 1) == outs_of(m, g, 0));  // fc-slot -> weight
  CHECK(ins_of(m, g, 2) == outs_of(m, g, 0));  // fc-slot -> bias
  CHECK(ops[3] == "aten::linear");
  CHECK(ins_of(m, g, 3) ==
        VS{"x", outs_of(m, g, 1)[0], outs_of(m, g, 2)[0]});
  CHECK(g.initializers.empty());
  CHECK(init_names(m, m.graphs[1]) == VS{"CONSTANTS.c0"});
  auto self = meta(m, "torchscript.self");
  REQUIRE(self);
  CHECK(self->find("no data.pkl") != std::string::npos);
  CHECK(m.flat_tensors.empty());
  CHECK(ByteReader::payload_read_counter() == 0);
}

// ============================================================================
// T-R: review hardening (#179) — whole-module and whole-method budgets,
// constant folding cost, packed-object weights, initializer identity, loop
// layout, container-held tensors, nested control flow, parse_zip routing
// ============================================================================
namespace {
std::string decode_err(const ValuePtr& root, const mb::BytecodeLimits& lim) {
  auto r = mb::decode_bytecode(root, lim);
  return r ? std::string() : r.error().message;
}
}  // namespace

TEST_CASE("T-R1 decode: memoized tables are charged against a whole-module budget") {
  // One method tuple referenced twice (BINGET): each reference decodes a full
  // copy, so the second is charged again.
  std::vector<ValuePtr> code(600, ins("LOAD", 1, 0));  // one shared instruction
  ValuePtr m = fn("__torch__.M.forward", code, {}, {}, {}, 1, {arg("x", "Tensor")},
                  {arg("", "Tensor")});
  mb::BytecodeLimits lim;
  CHECK(decode_err(T({I(8), m, m}), lim).empty());
  lim.max_total_entries = 1000;
  CHECK(decode_err(T({I(8), m}), lim).empty());
  CHECK(decode_err(T({I(8), m, m}), lim).find("total budget of 1000 entries") !=
        std::string::npos);

  // One 4000-byte operator name referenced by every operator entry.
  ValuePtr op = T({S(std::string(4000, 'o')), S(""), I(1)});
  ValuePtr m2 = fn("__torch__.M.forward", {}, std::vector<ValuePtr>(100, op), {}, {}, 0, {},
                   {});
  mb::BytecodeLimits lim2;
  CHECK(decode_err(T({I(8), m2}), lim2).empty());
  lim2.max_total_string_bytes = 100000;
  CHECK(decode_err(T({I(8), m2}), lim2).find("decoded strings exceed the total budget") !=
        std::string::npos);
  // The defaults do bound the product of the per-function caps.
  const mb::BytecodeLimits d;
  CHECK(d.max_total_entries < static_cast<uint64_t>(d.max_functions) * d.max_instructions);
}

TEST_CASE("T-R2 executor: fan-out instructions are charged against a work budget") {
  const std::vector<mb::BcArg> xa = {Arg("x", "Tensor")};
  const std::vector<mb::BcArg> r1 = {Arg("", "Tensor")};
  // ISINSTANCE with 65535 types per node, default limits.
  std::vector<std::string> types(65535, "Tensor");
  auto isinstance_fn = [&](int reps) {
    std::vector<mb::BcInstruction> code = {In(BcOp::STORE, 1)};
    for (int i = 0; i < reps; ++i) {
      code.push_back(In(BcOp::LOAD, 1));
      code.push_back(In(BcOp::ISINSTANCE, 0, 65535));
      code.push_back(In(BcOp::DROP));
    }
    code.push_back(In(BcOp::MOVE, 1));
    code.push_back(In(BcOp::RET));
    return mod(make_fn(code, {}, {}, 1, xa, r1, types));
  };
  {
    Run ok;
    build_into(ok, isinstance_fn(4));
    REQUIRE_MESSAGE(ok.ok(), ok.error());
    CHECK(ok.model.graphs[0].nodes.size() == 4);
  }
  {
    Run r;
    build_into(r, isinstance_fn(1000));
    REQUIRE_FALSE(r.ok());
    CHECK(r.error().find("method too large") != std::string::npos);
    CHECK(r.error().find("work budget") != std::string::npos);
    CHECK(r.model.graphs.empty());
  }
  // LIST_UNPACK 65535 then STOREN 65535: values plus register writes.
  {
    mb::BytecodeLimits lim;
    lim.max_exec_work = 100000;
    Run r;
    build_into(r,
               mod(make_fn({In(BcOp::STORE, 1), In(BcOp::LOAD, 1), In(BcOp::LIST_UNPACK, 65535),
                            In(BcOp::STOREN, 2, 65535), In(BcOp::MOVE, 1), In(BcOp::RET)},
                           {}, {}, 65536, xa, r1)),
               empty_env(), no_names(), lim);
    REQUIRE_FALSE(r.ok());
    CHECK(r.error().find("work budget of 100000") != std::string::npos);
    CHECK(r.model.graphs.empty());
  }
  // A fan-out larger than what is left is refused before any output exists.
  {
    mb::BytecodeLimits lim;
    lim.max_exec_work = 1000;
    Run r;
    build_into(r,
               mod(make_fn({In(BcOp::STORE, 1), In(BcOp::LOAD, 1), In(BcOp::LIST_UNPACK, 5000),
                            In(BcOp::RET)},
                           {}, {}, 1, xa, r1)),
               empty_env(), no_names(), lim);
    REQUIRE_FALSE(r.ok());
    CHECK(r.error().find("pc 2: method too large") != std::string::npos);
  }
}

TEST_CASE("T-R3 constant folding: caps apply before O(size) work, folds are memoized") {
  // An 8 MB string and a 1M-element list, each folded into 2000 nodes. Without
  // the caps and the memo this is ~16 GB of copying plus 2e9 element scans.
  ValuePtr big_s = S(std::string(8u << 20, 'a'));
  ValuePtr big_l = L(std::vector<ValuePtr>(1u << 20, I(1)));
  std::vector<mb::BcInstruction> code = {In(BcOp::STORE, 1)};
  for (int i = 0; i < 2000; ++i) {
    code.push_back(In(BcOp::LOADC, i % 2));
    code.push_back(In(BcOp::OP, 0));
    code.push_back(In(BcOp::DROP));
  }
  code.push_back(In(BcOp::MOVE, 1));
  code.push_back(In(BcOp::RET));
  Run r;
  build_into(r, mod(make_fn(code, {Op("custom::g", "", 1)}, {big_s, big_l}, 1,
                            {Arg("x", "Tensor")}, {Arg("", "Tensor")})));
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  REQUIRE(g.nodes.size() == 2000);
  const std::string s0 = attr_s(r.model, g, 0, "arg0");
  CHECK(s0.size() <= ts::ConstLimits{}.max_repr + 3);
  CHECK(s0.rfind("'aaa", 0) == 0);
  CHECK(attr_s(r.model, g, 1, "arg0") == "[\xE2\x80\xA6 1048576 items]");
  CHECK(attr_s(r.model, g, 1999, "arg0") == attr_s(r.model, g, 1, "arg0"));

  // constant_repr alone stops copying at the cap.
  ValuePtr huge = S(std::string(50u << 20, '\''));
  CHECK(ts::constant_repr(*huge).size() <= ts::ConstLimits{}.max_repr + 3);

  // INTERFACE_CALL's method name is capped like every other decoded name.
  Run ic;
  build_into(ic, mod(make_fn({In(BcOp::STORE, 1), In(BcOp::LOAD, 1),
                              In(BcOp::INTERFACE_CALL, 0, 1), In(BcOp::MOVE, 1), In(BcOp::RET)},
                             {}, {S(std::string(5000, 'm'))}, 1, {Arg("x", "Tensor")},
                             {Arg("", "Tensor")})));
  REQUIRE_FALSE(ic.ok());
  CHECK(ic.error().find("INTERFACE_CALL method name longer than 4096 bytes") !=
        std::string::npos);
}

TEST_CASE("T-R4 packed objects: weights in their state are located and counted") {
  ValuePtr W = tensor_v(64, 32, {2, 4});
  ValuePtr Bt = tensor_v(96, 8, {2});
  ValuePtr ctxv = mk(Value::Kind::Opaque);
  ctxv->module = "__torch__.torch.classes.xnnpack";
  ctxv->name = "Conv2dOpContext";
  ctxv->inner = T({W, Bt, L({I(1), I(1)}), N()});
  ValuePtr root = obj("__torch__", "M",
                      {{"training", B(false)}, {"conv", obj("__torch__", "Conv", {{"packed", ctxv}})}});
  ts::ModuleEnv env(root);
  CHECK(env.packed().objects == 1);
  CHECK(env.packed().tensors == 2);
  CHECK(env.packed().unlocated == 0);
  CHECK(env.module_tensor_count() == 2);
  CHECK(env.tensor_path(W.get(), "") == "conv.packed.0");
  // The tensor table walk sees them too.
  VS paths;
  ts::for_each_tensor(root, [&](const std::string& p, const Value&) { paths.push_back(p); });
  CHECK(paths == VS{"conv.packed.0", "conv.packed.1"});

  // forward(self, x) = prepacked::conv2d_clamp_run(x, self.conv.packed)
  Run r;
  build_into(r,
             mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::MOVE, 1),
                          In(BcOp::GET_ATTR, 1), In(BcOp::GET_ATTR, 0), In(BcOp::OP, 0),
                          In(BcOp::RET)},
                         {Op("prepacked::conv2d_clamp_run", "", 2)}, {}, 2,
                         {Arg("self", "__torch__.M"), Arg("x", "Tensor")}, {Arg("", "Tensor")})),
             env);
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ops_of(r.model, g) ==
        VS{"prim::GetAttr", "prim::GetAttr", "prepacked::conv2d_clamp_run"});
  CHECK(attr_s(r.model, g, 1, "name") == "conv.packed");
  CHECK(ins_of(r.model, g, 1) == VS{"conv", "conv.packed.0", "conv.packed.1"});
  CHECK(init_names(r.model, g) == VS{"conv.packed.0", "conv.packed.1"});
  CHECK(g.initializers[0].file_offset == 64);
  const auto& bound = (*r.res)->bound_module_tensors;
  REQUIRE(bound.size() == 2);
  CHECK(bound[0] == W.get());
  CHECK(bound[1] == Bt.get());

  // A quantized packed context: its tensor cannot be decoded, so it is counted
  // as unlocated (never guessed) and the parameter totals become a lower bound.
  ValuePtr q = mk(Value::Kind::Opaque);
  q->module = "torch._utils";
  q->name = "_rebuild_qtensor";
  ValuePtr qp = mk(Value::Kind::Opaque);
  qp->module = "__torch__.torch.classes.quantized";
  qp->name = "LinearPackedParamsBase";
  qp->inner = T({S("3"), L({q, N()}), L({N()})});
  ts::ModuleEnv qenv(
      obj("__torch__", "M", {{"fc", obj("__torch__", "L", {{"_packed_params", qp}})}}));
  CHECK(qenv.packed().objects == 1);
  CHECK(qenv.packed().tensors == 0);
  CHECK(qenv.packed().unlocated == 1);

  ir::Model table;
  table.has_graph = false;
  table.unlocated_weights = 1;
  CHECK(compute_cost(table, 0).unlocated_weights == 1);
  CHECK(compute_cost(r.model, 0).unlocated_weights == 0);
  r.model.unlocated_weights = 2;
  CHECK(compute_cost(r.model, 0).unlocated_weights == 2);
}

TEST_CASE("T-R5 initializers: identity decides, unlocated refs never merge by name") {
  ir::Model m;
  ts::NameCounter names;
  ts::GraphBuilder gb(m, names, "g", 0, nullptr);
  ir::TensorRef t;
  t.dtype = ir::DType::F32;
  t.file_offset = UINT64_MAX;  // storage key did not resolve
  t.byte_len = 8;
  int a = 0, b = 0;
  const uint32_t va = gb.initializer("w", t, &a);
  const uint32_t vb = gb.initializer("w", t, &b);
  CHECK(va != vb);
  CHECK(gb.value_name(vb) == "w#2");
  CHECK(gb.initializer("w", t, &a) == va);
  CHECK(gb.initializer("w", t) != va);  // no identity, unlocated: a new value
  // Suffixes continue from the last one instead of re-probing from #2.
  uint32_t last = 0;
  for (int i = 0; i < 20000; ++i) last = gb.initializer("x", t);
  CHECK(gb.value_name(last) == "x#20000");
  ir::Graph g = gb.finish();
  CHECK(g.initializers.size() == 3 + 20000);
}

TEST_CASE("T-R6 LOOP: the iteration-counter slot must be the emitter's constant 0") {
  const std::vector<mb::BcOperator> relu = {Op("aten::relu", "", 1)};
  auto loop_fn = [&](std::vector<mb::BcInstruction> prefix, std::vector<ValuePtr> consts) {
    std::vector<mb::BcInstruction> code = std::move(prefix);
    for (const auto& in :
         {In(BcOp::LOADC, 1), In(BcOp::LOADC, 2), In(BcOp::LOAD, 1), In(BcOp::LOOP, 6, 3),
          In(BcOp::STOREN, 2, 2), In(BcOp::LOADC, 2), In(BcOp::MOVE, 3), In(BcOp::OP, 0),
          In(BcOp::JMP, -5), In(BcOp::STORE, 4), In(BcOp::DROPR, 1), In(BcOp::MOVE, 4),
          In(BcOp::RET)})
      code.push_back(in);
    return mod(make_fn(code, relu, std::move(consts), 4, {Arg("x", "Tensor")},
                       {Arg("", "Tensor")}));
  };
  {
    Run ok;
    build_into(ok, loop_fn({In(BcOp::STORE, 1), In(BcOp::LOADC, 0)}, {I(0), I(10), B(true)}));
    REQUIRE_MESSAGE(ok.ok(), ok.error());
  }
  {
    Run r;
    build_into(r, loop_fn({In(BcOp::STORE, 1), In(BcOp::LOADC, 0)}, {I(1), I(10), B(true)}));
    REQUIRE_FALSE(r.ok());
    CHECK(r.error().find("pc 5: malformed loop (iteration counter") != std::string::npos);
  }
  {
    // A pending op where iter0 belongs: an error, not a phantom output.
    Run r;
    build_into(r, loop_fn({In(BcOp::STORE, 1), In(BcOp::LOAD, 1), In(BcOp::OP, 0)},
                          {I(0), I(10), B(true)}));
    REQUIRE_FALSE(r.ok());
    CHECK(r.error().find("malformed loop (iteration counter") != std::string::npos);
    CHECK(r.model.graphs.empty());
  }
}

TEST_CASE("T-R7 tensors inside a folded constant attribute count as referenced") {
  ValuePtr W = tensor_v(64, 32, {2, 4});
  ValuePtr Bt = tensor_v(96, 8, {2});
  ts::ModuleEnv env(obj("__torch__", "M", {{"training", B(false)}, {"bufs", L({W, Bt})}}));
  CHECK(env.module_tensor_count() == 2);
  Run r;
  build_into(r,
             mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::MOVE, 2), In(BcOp::MOVE, 1),
                          In(BcOp::GET_ATTR, 1), In(BcOp::OP, 0), In(BcOp::RET)},
                         {Op("custom::h", "", 2)}, {}, 2,
                         {Arg("self", "__torch__.M"), Arg("x", "Tensor")}, {Arg("", "Tensor")})),
             env);
  REQUIRE_MESSAGE(r.ok(), r.error());
  const ir::Graph& g = r.model.graphs[0];
  CHECK(ops_of(r.model, g) == VS{"custom::h"});
  CHECK(attr_s(r.model, g, 0, "arg1") == "[<tensor f32[2,4]>, <tensor f32[2]>]");
  const auto& bound = (*r.res)->bound_module_tensors;
  REQUIRE(bound.size() == 2);
  CHECK(bound[0] == W.get());
  CHECK(bound[1] == Bt.get());
}

TEST_CASE("T-R8 nested control flow: If in Loop, Loop in If, pending If chains") {
  const std::vector<mb::BcArg> xc = {Arg("x", "Tensor"), Arg("c", "bool")};
  const std::vector<mb::BcArg> r1 = {Arg("", "Tensor")};
  SUBCASE("If inside a loop body captures through the body") {
    Run r;
    build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::LOADC, 0), In(BcOp::LOADC, 1),
                               In(BcOp::LOADC, 2), In(BcOp::LOAD, 1), In(BcOp::LOOP, 12, 3),
                               In(BcOp::STOREN, 3, 2), In(BcOp::LOAD, 2), In(BcOp::JF, 4),
                               In(BcOp::LOAD, 4), In(BcOp::OP, 0), In(BcOp::JMP, 2),
                               In(BcOp::LOAD, 4), In(BcOp::STORE, 5), In(BcOp::LOADC, 2),
                               In(BcOp::MOVE, 5), In(BcOp::JMP, -11), In(BcOp::STORE, 6),
                               In(BcOp::MOVE, 6), In(BcOp::RET)},
                              {Op("aten::relu", "", 1)}, {I(0), I(10), B(true)}, 6, xc, r1)));
    REQUIRE_MESSAGE(r.ok(), r.error());
    const ir::Model& m = r.model;
    REQUIRE(m.graphs.size() == 4);
    CHECK(str(m, m.graphs[1].name) == "__torch__.M.forward/loop0.body");
    CHECK(str(m, m.graphs[2].name) == "__torch__.M.forward/loop0.body/if1.then");
    const ir::Graph& g = m.graphs[0];
    CHECK(ops_of(m, g) == VS{"prim::Constant", "prim::Constant", "prim::Loop"});
    CHECK(attr_i(m, g, 2, "captures") == 1);
    CHECK(ins_of(m, g, 2).back() == "c");
    const ir::Graph& body = m.graphs[1];
    CHECK(ops_of(m, body) == VS{"prim::If", "prim::Constant"});
    CHECK(attr_i(m, body, 0, "captures") == 1);
    CHECK(outs_of(m, body, 0).size() == 1);
    CHECK(ops_of(m, m.graphs[2]) == VS{"aten::relu"});
    CHECK(m.graphs[2].graph_outputs.size() == 1);
    CHECK(m.graphs[3].nodes.empty());
    CHECK(m.graphs[3].graph_outputs.size() == 1);
    CHECK(names_of(m, g, g.graph_outputs) == outs_of(m, g, 2));
  }
  SUBCASE("Loop inside a then-branch") {
    Run r;
    build_into(r, mod(make_fn({In(BcOp::STOREN, 1, 2), In(BcOp::LOAD, 2), In(BcOp::JF, 12),
                               In(BcOp::LOADC, 0), In(BcOp::LOADC, 1), In(BcOp::LOADC, 2),
                               In(BcOp::LOAD, 1), In(BcOp::LOOP, 6, 3), In(BcOp::STOREN, 3, 2),
                               In(BcOp::LOADC, 2), In(BcOp::MOVE, 4), In(BcOp::OP, 0),
                               In(BcOp::JMP, -5), In(BcOp::JMP, 2), In(BcOp::LOAD, 1),
                               In(BcOp::STORE, 5), In(BcOp::MOVE, 5), In(BcOp::RET)},
                              {Op("aten::relu", "", 1)}, {I(0), I(10), B(true)}, 5, xc, r1)));
    REQUIRE_MESSAGE(r.ok(), r.error());
    const ir::Model& m = r.model;
    REQUIRE(m.graphs.size() == 4);
    CHECK(str(m, m.graphs[3].name) == "__torch__.M.forward/if0.then/loop1.body");
    const ir::Graph& g = m.graphs[0];
    CHECK(ops_of(m, g) == VS{"prim::If"});
    CHECK(attr_i(m, g, 0, "captures") == 1);
    CHECK(outs_of(m, g, 0).size() == 1);
    const ir::Graph& t = m.graphs[1];
    CHECK(names_of(m, t, t.graph_inputs) == VS{"x"});
    CHECK(ops_of(m, t) == VS{"prim::Constant", "prim::Constant", "prim::Loop"});
    CHECK(names_of(m, t, t.graph_outputs) == outs_of(m, t, 2));
    CHECK(ops_of(m, m.graphs[3]) == VS{"aten::relu", "prim::Constant"});
    CHECK(m.graphs[2].nodes.empty());
    CHECK(names_of(m, m.graphs[2], m.graphs[2].graph_inputs) == VS{"x"});
  }
  // `if c: (if c: f(x) else: f(x)) else: f(x)` as a bare statement: both Ifs
  // stay pending until something decides their output count.
  auto chain = [&](std::vector<mb::BcInstruction> tail) {
    std::vector<mb::BcInstruction> code = {
        In(BcOp::STOREN, 1, 2), In(BcOp::LOAD, 2), In(BcOp::JF, 9),  In(BcOp::LOAD, 2),
        In(BcOp::JF, 4),        In(BcOp::LOAD, 1), In(BcOp::OP, 0),  In(BcOp::JMP, 3),
        In(BcOp::LOAD, 1),      In(BcOp::OP, 0),   In(BcOp::JMP, 3), In(BcOp::LOAD, 1),
        In(BcOp::OP, 0)};
    for (const auto& in : tail) code.push_back(in);
    return mod(make_fn(code, {Op("custom::f", "", 1)}, {}, 2, xc, r1));
  };
  auto check_outputs = [](const ir::Model& m, size_t want) {
    REQUIRE(m.graphs.size() == 5);
    CHECK(ops_of(m, m.graphs[0]) == VS{"prim::If"});
    CHECK(outs_of(m, m.graphs[0], 0).size() == want);
    CHECK(ops_of(m, m.graphs[1]) == VS{"prim::If"});
    CHECK(outs_of(m, m.graphs[1], 0).size() == want);
    for (size_t gi : {size_t{2}, size_t{3}, size_t{4}}) {
      INFO("graph " << gi);
      CHECK(ops_of(m, m.graphs[gi]) == VS{"custom::f"});
      CHECK(outs_of(m, m.graphs[gi], 0).size() == want);
      CHECK(m.graphs[gi].graph_outputs.size() == want);
    }
  };
  SUBCASE("a pending If inside a pending If is retracted as a chain") {
    Run r;
    build_into(r, chain({In(BcOp::LOAD, 1), In(BcOp::RET)}));
    REQUIRE_MESSAGE(r.ok(), r.error());
    check_outputs(r.model, 0);
  }
  SUBCASE("a pending If inside a pending If is confirmed as a chain") {
    Run r;
    build_into(r, chain({In(BcOp::DROP), In(BcOp::LOAD, 1), In(BcOp::RET)}));
    REQUIRE_MESSAGE(r.ok(), r.error());
    check_outputs(r.model, 1);
  }
}

// ---- parse_zip error routing on rewritten copies of the v8 fixture ------------
namespace {
struct ZipEntry {
  std::string name;
  std::vector<uint8_t> data;
  bool deflate = false;
};
std::vector<ZipEntry> fixture_entries(const char* path) {
  std::vector<uint8_t> file = read_file(path);
  mz_zip_archive zip;
  std::memset(&zip, 0, sizeof(zip));
  REQUIRE(mz_zip_reader_init_mem(&zip, file.data(), file.size(), 0));
  std::vector<ZipEntry> out;
  for (mz_uint i = 0; i < mz_zip_reader_get_num_files(&zip); ++i) {
    mz_zip_archive_file_stat st;
    REQUIRE(mz_zip_reader_file_stat(&zip, i, &st));
    size_t n = 0;
    void* p = mz_zip_reader_extract_to_heap(&zip, i, &n, 0);
    ZipEntry e;
    e.name = st.m_filename;
    if (p) e.data.assign(static_cast<uint8_t*>(p), static_cast<uint8_t*>(p) + n);
    mz_free(p);
    out.push_back(std::move(e));
  }
  mz_zip_reader_end(&zip);
  return out;
}
std::vector<uint8_t> write_zip(const std::vector<ZipEntry>& es) {
  mz_zip_archive zw;
  std::memset(&zw, 0, sizeof(zw));
  REQUIRE(mz_zip_writer_init_heap(&zw, 0, 0));
  for (const ZipEntry& e : es)
    REQUIRE(mz_zip_writer_add_mem(&zw, e.name.c_str(), e.data.data(), e.data.size(),
                                  e.deflate ? MZ_BEST_SPEED : MZ_NO_COMPRESSION));
  void* buf = nullptr;
  size_t n = 0;
  REQUIRE(mz_zip_writer_finalize_heap_archive(&zw, &buf, &n));
  std::vector<uint8_t> out(static_cast<uint8_t*>(buf), static_cast<uint8_t*>(buf) + n);
  mz_free(buf);
  mz_zip_writer_end(&zw);
  return out;
}
// Overwrite the central-directory uncompressed size of entry `name`.
void patch_uncomp_size(std::vector<uint8_t>& z, const std::string& name, uint32_t v) {
  for (size_t i = 0; i + 46 <= z.size(); ++i) {
    if (z[i] != 0x50 || z[i + 1] != 0x4b || z[i + 2] != 0x01 || z[i + 3] != 0x02) continue;
    const size_t nlen = z[i + 28] | (z[i + 29] << 8);
    if (i + 46 + nlen > z.size() ||
        std::string(reinterpret_cast<const char*>(&z[i + 46]), nlen) != name)
      continue;
    for (int k = 0; k < 4; ++k) z[i + 24 + k] = static_cast<uint8_t>(v >> (8 * k));
    return;
  }
  FAIL("central directory entry not found: " << name);
}
Result<ir::Model> parse_bytes(const std::string& stem, const std::vector<uint8_t>& bytes) {
  std::string path =
      (std::filesystem::temp_directory_path() / ("netvis_r9_" + stem + ".ptl")).string();
  netvis_test::TempFileGuard cleanup(path);  // before mf: unmap first, then delete
  {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  auto mf = MappedFile::open(path);
  REQUIRE(mf);
  ProgressSink progress;
  return pytorch::parse_zip(*mf, progress);
}
// The fixture with bytecode.pkl cut in half, optionally deflated, optionally
// without data.pkl.
std::vector<uint8_t> broken_bytecode(bool deflate, bool keep_data) {
  std::vector<ZipEntry> es;
  for (ZipEntry& e : fixture_entries(kMobile)) {
    if (!keep_data && e.name == "model/data.pkl") continue;
    if (e.name == "model/bytecode.pkl") {
      e.data.resize(e.data.size() / 2);
      e.deflate = deflate;
    }
    es.push_back(std::move(e));
  }
  return write_zip(es);
}
}  // namespace

TEST_CASE("T-R9 parse_zip: bytecode.pkl failures are routed and located honestly") {
  if (!have(kMobile)) return;
  ByteReader::payload_read_counter() = 0;
  SUBCASE("stored, with data.pkl: parameter table plus a located note") {
    auto res = parse_bytes("stored_data", broken_bytecode(false, true));
    REQUIRE_MESSAGE(res, (res ? std::string() : res.error().message));
    const ir::Model& m = *res;
    CHECK_FALSE(m.has_graph);
    check_parameter_table(m);
    auto note = meta(m, "torchscript.bytecode");
    REQUIRE(note);
    CHECK(note->rfind("not decoded: ", 0) == 0);
    CHECK(note->find("(byte ") != std::string::npos);
    CHECK(meta(m, "tensors") == std::string("2"));
  }
  SUBCASE("stored, without data.pkl: an error at an absolute offset") {
    std::vector<uint8_t> z = broken_bytecode(false, false);
    auto res = parse_bytes("stored_nodata", z);
    REQUIRE_FALSE(res);
    CHECK_FALSE(res.error().message.empty());
    CHECK(res.error().offset < z.size());
    CHECK(res.error().message.find("compressed") == std::string::npos);
  }
  SUBCASE("deflated, with data.pkl: the note gives the in-entry offset") {
    auto res = parse_bytes("deflate_data", broken_bytecode(true, true));
    REQUIRE_MESSAGE(res, (res ? std::string() : res.error().message));
    auto note = meta(*res, "torchscript.bytecode");
    REQUIRE(note);
    CHECK(note->find("(in compressed bytecode.pkl at +") != std::string::npos);
  }
  SUBCASE("deflated, without data.pkl: the error says where, no file offset") {
    auto res = parse_bytes("deflate_nodata", broken_bytecode(true, false));
    REQUIRE_FALSE(res);
    CHECK(res.error().message.find("(in compressed bytecode.pkl at +") != std::string::npos);
    CHECK(res.error().offset == UINT64_MAX);
  }
  SUBCASE("a declared bytecode.pkl size over the cap is refused before extraction") {
    std::vector<ZipEntry> es = fixture_entries(kMobile);
    for (ZipEntry& e : es)
      if (e.name == "model/bytecode.pkl") e.deflate = true;
    std::vector<uint8_t> z = write_zip(es);
    patch_uncomp_size(z, "model/bytecode.pkl", 0x7FFFFFFFu);
    auto res = parse_bytes("bc_cap", z);
    REQUIRE_MESSAGE(res, (res ? std::string() : res.error().message));
    auto note = meta(*res, "torchscript.bytecode");
    REQUIRE(note);
    CHECK(note->find("bytecode.pkl too large") != std::string::npos);
    check_parameter_table(*res);
  }
  SUBCASE("a broken or oversized constants.pkl falls back to <method>.c<k> names") {
    for (bool oversize : {false, true}) {
      INFO("oversize: " << oversize);
      std::vector<ZipEntry> es = fixture_entries(kMobile);
      for (ZipEntry& e : es) {
        if (e.name != "model/constants.pkl") continue;
        if (oversize)
          e.deflate = true;
        else
          e.data.resize(3);
      }
      std::vector<uint8_t> z = write_zip(es);
      if (oversize) patch_uncomp_size(z, "model/constants.pkl", 0x7FFFFFFFu);
      auto res = parse_bytes(oversize ? "const_cap" : "const_broken", z);
      REQUIRE_MESSAGE(res, (res ? std::string() : res.error().message));
      const ir::Model& m = *res;
      REQUIRE(m.has_graph);
      REQUIRE(m.graphs.size() == 3);
      const VS inits = init_names(m, m.graphs[1]);
      REQUIRE(inits.size() == 1);
      CHECK(inits[0].rfind("forward.c", 0) == 0);
    }
  }
  CHECK(ByteReader::payload_read_counter() == 0);
}
