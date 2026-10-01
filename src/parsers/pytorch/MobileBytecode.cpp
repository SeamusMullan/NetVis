// SPDX-License-Identifier: Apache-2.0
// parsers/pytorch/MobileBytecode.cpp — bytecode.pkl decode + symbolic executor.
//
// See MobileBytecode.h. The authority for every rule here is upstream PyTorch
// (torch/csrc/jit/serialization/export_module.cpp, mobile/parse_bytecode.cpp,
// runtime/interpreter/code_impl.h, can_emit_inline.h, preprocess_graph.cpp,
// mobile/interpreter.cpp); spec #137 §2 cites the exact functions.
//
// The executor recovers a dataflow graph WITHOUT operator schemas from the JIT
// emitter's invariants (spec §2.5): block-level statements return the stack to
// the block base (I1, I5); only single-output nodes are emitted inline (I2);
// v6+ operators record their exact input count (I4); If/Loop have a fixed
// JF/JMP/LOOP layout (I6, I7). The output count of OP/OPN/INTERFACE_CALL is not
// recorded, so a node that might have 0 or 1 outputs is pushed as a PENDING
// entry that a later pop confirms (1 output) or the next checkpoint retracts (0
// outputs). Any contradiction is a method error, never a guessed graph.
#include "parsers/pytorch/MobileBytecode.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace netvis::pytorch::mobile {

// ============================================================================
// Opcodes
// ============================================================================
namespace {

constexpr const char* kOpNames[] = {
    "OP", "OPN", "LOAD", "MOVE", "STOREN", "STORE", "DROP", "DROPR", "LOADC",
    "JF", "JMP", "LOOP", "RET", "WAIT", "CALL", "GUARD", "TYPECHECK",
    "FAIL_GUARD", "PROFILE_OP", "TAIL_CALL", "INTERFACE_CALL", "GET_ATTR",
    "SET_ATTR", "LIST_UNPACK", "TUPLE_CONSTRUCT", "NAMED_TUPLE_CONSTRUCT",
    "LIST_CONSTRUCT", "DICT_CONSTRUCT", "CREATE_OBJECT", "ISINSTANCE",
    "TUPLE_SLICE", "TUPLE_INDEX", "RAISE_EXCEPTION", "DICT_INDEX",
    "UNCHECKED_CAST", "__IS__", "UN_INITIALIZED", "__ISNOT__", "FORMAT", "DEVICE",
    "DTYPE", "DIM", "__NOT__", "TO_LIST", "NUM_TO_TENSOR", "IS_CUDA", "FORK",
    "WARN", "ENTER", "EXIT", "AWAITABLE"};
constexpr size_t kNumOps = sizeof(kOpNames) / sizeof(kOpNames[0]);
static_assert(kNumOps == static_cast<size_t>(BcOp::Unknown),
              "opcode string table out of sync with BcOp");

}  // namespace

BcOp opcode_from_string(std::string_view s) {
  for (size_t i = 0; i < kNumOps; ++i)
    if (s == kOpNames[i]) return static_cast<BcOp>(i);
  return BcOp::Unknown;
}

const char* opcode_name(BcOp op) {
  auto i = static_cast<size_t>(op);
  return i < kNumOps ? kOpNames[i] : "?";
}

bool opcode_valid_in_mobile(BcOp op) {
  switch (op) {
    case BcOp::WAIT:
    case BcOp::CALL:  // only produced at load time by upgraders
    case BcOp::GUARD:
    case BcOp::TYPECHECK:
    case BcOp::FAIL_GUARD:
    case BcOp::PROFILE_OP:
    case BcOp::TAIL_CALL:
    case BcOp::FORK:
    case BcOp::ENTER:
    case BcOp::EXIT:
    case BcOp::AWAITABLE:
    case BcOp::Unknown:
      return false;
    default:
      return true;
  }
}

// ============================================================================
// Decode
// ============================================================================
namespace {

uint64_t pos_of(const ValuePtr& v) { return v ? v->src_pos : UINT64_MAX; }
bool is(const ValuePtr& v, Value::Kind k) { return v && v->kind == k; }

// Row `i` of an export_module.cpp Table: a 2-tuple ("<field>", value). Returns
// the value, or an error naming the field and index (expect_field semantics).
Result<ValuePtr> table_row(const ValuePtr& table, size_t i, const char* field,
                           const std::string& method) {
  auto bad = [&](const ValuePtr& at) -> Error {
    return err("method " + method + ": expected '" + field + "' at index " +
                   std::to_string(i),
               pos_of(at ? at : table));
  };
  if (!is(table, Value::Kind::Tuple) || i >= table->items.size()) return bad(table);
  const ValuePtr& row = table->items[i];
  if (!is(row, Value::Kind::Tuple) || row->items.size() != 2 ||
      !is(row->items[0], Value::Kind::Str) || row->items[0]->s != field)
    return bad(row);
  if (!row->items[1]) return bad(row);
  return row->items[1];
}

Result<const std::vector<ValuePtr>*> tuple_items(const ValuePtr& v, const char* what,
                                                 const std::string& method,
                                                 uint64_t cap) {
  if (!is(v, Value::Kind::Tuple))
    return err("method " + method + ": '" + what + "' is not a tuple", pos_of(v));
  if (v->items.size() > cap)
    return err("method " + method + ": " + std::to_string(v->items.size()) + " " +
                   what + " exceed the cap of " + std::to_string(cap),
               pos_of(v));
  return &v->items;
}

Result<bool> decode_args(const ValuePtr& list, const char* what,
                         const std::string& method, const BytecodeLimits& lim,
                         std::vector<BcArg>& out) {
  auto items = tuple_items(list, what, method, lim.max_schema_args);
  if (!items) return items.error();
  out.reserve((*items)->size());
  for (const ValuePtr& a : **items) {
    auto name = table_row(a, 0, "name", method);
    if (!name) return name.error();
    auto type = table_row(a, 1, "type", method);
    if (!type) return type.error();
    auto dflt = table_row(a, 2, "default_value", method);
    if (!dflt) return dflt.error();
    if (!is(*name, Value::Kind::Str) || !is(*type, Value::Kind::Str) ||
        (*name)->s.size() > lim.max_string || (*type)->s.size() > lim.max_string)
      return err("method " + method + ": malformed schema argument", pos_of(a));
    out.push_back(BcArg{(*name)->s, (*type)->s, *dflt});
  }
  return true;
}

Result<BcFunction> decode_function(const ValuePtr& mv, int64_t version,
                                   const BytecodeLimits& lim) {
  if (!is(mv, Value::Kind::Tuple) || mv->items.size() < 2)
    return err("bytecode.pkl: method entry is not a (name, code[, schema]) tuple",
               pos_of(mv));
  const ValuePtr& nv = mv->items[0];
  if (!is(nv, Value::Kind::Str) || nv->s.empty() || nv->s.size() > lim.max_string)
    return err("bytecode.pkl: method name is not a string of 1-" +
                   std::to_string(lim.max_string) + " bytes",
               pos_of(nv ? nv : mv));
  BcFunction f;
  f.qualified_name = nv->s;
  f.pos = mv->src_pos;
  const std::string& name = f.qualified_name;
  f.has_schema = version > 4 || (version == 4 && mv->items.size() >= 3);
  if (version >= 5 && mv->items.size() < 3)
    return err("method " + name + ": missing schema (required for v5+)", mv->src_pos);

  const ValuePtr& code = mv->items[1];
  static const char* kFields[] = {"instructions", "operators", "constants", "types",
                                  "register_size"};
  ValuePtr rows[5];
  for (size_t i = 0; i < 5; ++i) {
    auto r = table_row(code, i, kFields[i], name);
    if (!r) return r.error();
    rows[i] = *r;
  }

  // --- instructions ---
  {
    auto items = tuple_items(rows[0], "instructions", name, lim.max_instructions);
    if (!items) return items.error();
    f.instructions.reserve((*items)->size());
    std::unordered_map<std::string, uint32_t> unknown_ids;
    for (size_t i = 0; i < (*items)->size(); ++i) {
      const ValuePtr& t = (**items)[i];
      if (!is(t, Value::Kind::Tuple) || t->items.size() != 3 ||
          !is(t->items[0], Value::Kind::Str) || !is(t->items[1], Value::Kind::Int) ||
          !is(t->items[2], Value::Kind::Int))
        return err("method " + name + ": instruction " + std::to_string(i) +
                       " is not (str, int, int)",
                   pos_of(t ? t : rows[0]));
      int64_t x = t->items[1]->i, n = t->items[2]->i;
      if (x < INT32_MIN || x > INT32_MAX)
        return err("method " + name + ": instruction " + std::to_string(i) +
                       " operand X out of range",
                   t->src_pos);
      if (n < 0 || n > 65535)
        return err("method " + name + ": instruction " + std::to_string(i) +
                       " operand N out of range",
                   t->src_pos);
      BcInstruction in;
      in.x = static_cast<int32_t>(x);
      in.n = static_cast<int32_t>(n);
      in.pos = t->src_pos;
      const std::string& s = t->items[0]->s;
      std::string raw;
      if (s.size() > lim.max_opcode_string) {
        in.op = BcOp::Unknown;
        raw = s.substr(0, lim.max_opcode_string) + "\xE2\x80\xA6";
      } else {
        in.op = opcode_from_string(s);
        if (in.op == BcOp::Unknown) raw = s;
      }
      if (in.op == BcOp::Unknown) {
        auto it = unknown_ids.find(raw);
        if (it == unknown_ids.end()) {
          it = unknown_ids.emplace(raw, static_cast<uint32_t>(f.unknown_opcodes.size()))
                   .first;
          f.unknown_opcodes.push_back(raw);
        }
        in.unknown_name = it->second;
      }
      f.instructions.push_back(in);
    }
  }

  // --- operators ---
  {
    auto items = tuple_items(rows[1], "operators", name, lim.max_operators);
    if (!items) return items.error();
    f.operators.reserve((*items)->size());
    for (const ValuePtr& o : **items) {
      BcOperator op;
      op.pos = pos_of(o);
      bool tup = is(o, Value::Kind::Tuple);
      if (tup && o->items.size() >= 2 && is(o->items[0], Value::Kind::Str) &&
          is(o->items[1], Value::Kind::Str) && o->items[0]->s.size() <= lim.max_string &&
          o->items[1]->s.size() <= lim.max_string) {
        op.name = o->items[0]->s;
        op.overload = o->items[1]->s;
      } else {
        op.well_formed = false;
      }
      if (tup && o->items.size() >= 3 && is(o->items[2], Value::Kind::Int)) {
        op.has_num_args = true;
        op.num_args = o->items[2]->i;
      }
      f.operators.push_back(std::move(op));
    }
  }

  // --- constants ---
  {
    auto items = tuple_items(rows[2], "constants", name, lim.max_constants);
    if (!items) return items.error();
    f.constants = **items;
  }

  // --- types ---
  {
    auto items = tuple_items(rows[3], "types", name, lim.max_types);
    if (!items) return items.error();
    f.types.reserve((*items)->size());
    f.type_ok.reserve((*items)->size());
    for (const ValuePtr& t : **items) {
      if (is(t, Value::Kind::Str) && t->s.size() <= lim.max_string) {
        f.types.push_back(t->s);
        f.type_ok.push_back(1);
      } else {
        f.types.push_back("?");
        f.type_ok.push_back(0);
      }
    }
  }

  // --- register_size ---
  if (!is(rows[4], Value::Kind::Int) || rows[4]->i < 0 ||
      rows[4]->i > lim.max_register_size)
    return err("method " + name + ": register_size is not an int in [0, " +
                   std::to_string(lim.max_register_size) + "]",
               pos_of(rows[4]));
  f.register_size = rows[4]->i;

  // --- schema ---
  if (f.has_schema) {
    const ValuePtr& schema = mv->items[2];
    auto args = table_row(schema, 0, "arguments", name);
    if (!args) return args.error();
    auto rets = table_row(schema, 1, "returns", name);
    if (!rets) return rets.error();
    auto a = decode_args(*args, "arguments", name, lim, f.arguments);
    if (!a) return a.error();
    auto r = decode_args(*rets, "returns", name, lim, f.returns);
    if (!r) return r.error();
  }
  return f;
}

}  // namespace

Result<BcModule> decode_bytecode(const ValuePtr& root, const BytecodeLimits& lim) {
  if (!is(root, Value::Kind::Tuple) || root->items.empty())
    return err("bytecode.pkl: root is not a non-empty tuple", pos_of(root));
  BcModule m;
  size_t first = 0;
  const ValuePtr& e0 = root->items[0];
  if (is(e0, Value::Kind::Int)) {
    m.version = e0->i;
    first = 1;
  } else if (is(e0, Value::Kind::Tuple)) {
    m.version = 3;  // the pre-v4 layout has no version int
    m.version_implicit = true;
  } else {
    return err("bytecode.pkl: unrecognised top-level layout", pos_of(e0 ? e0 : root));
  }
  if (m.version < 3 || m.version > 9)
    return err("unsupported bytecode version " + std::to_string(m.version) +
                   " (NetVis reads 3-9)",
               pos_of(e0));
  size_t nfn = root->items.size() - first;
  if (nfn > lim.max_functions)
    return err("bytecode.pkl: " + std::to_string(nfn) +
                   " functions exceed the cap of " + std::to_string(lim.max_functions),
               root->src_pos);
  m.functions.reserve(nfn);
  for (size_t i = first; i < root->items.size(); ++i) {
    auto f = decode_function(root->items[i], m.version, lim);
    if (!f) return f.error();
    m.functions.push_back(f.take());
  }
  return m;
}

Result<BcModule> decode_bytecode_pickle(const uint8_t* data, uint64_t size,
                                        const StorageResolver& resolver,
                                        const BytecodeLimits& lim) {
  if (size > lim.max_pickle_bytes)
    return err("bytecode.pkl too large: " + std::to_string(size) + " bytes", 0);
  PickleLimits pl;
  pl.max_values = lim.max_pickle_values;
  PickleVM vm(data, size, resolver, pl);
  auto root = vm.run();
  if (!root) return root.error();
  return decode_bytecode(*root, lim);
}

namespace {
std::string basename_of(std::string_view p) {
  auto sl = p.find_last_of('/');
  return std::string(sl == std::string_view::npos ? p : p.substr(sl + 1));
}
std::string dir_of(std::string_view p) {
  auto sl = p.find_last_of('/');
  return sl == std::string_view::npos ? std::string() : std::string(p.substr(0, sl + 1));
}
}  // namespace

int32_t select_bytecode_entry(const std::vector<std::string>& names,
                              std::string_view data_pkl_path) {
  const size_t n = std::min(names.size(), static_cast<size_t>(INT32_MAX));
  if (!data_pkl_path.empty()) {
    const std::string want = dir_of(data_pkl_path) + "bytecode.pkl";
    for (size_t i = 0; i < n; ++i)
      if (names[i] == want) return static_cast<int32_t>(i);
    return -1;
  }
  int32_t best = -1;
  size_t best_slashes = SIZE_MAX;
  for (size_t i = 0; i < n; ++i) {
    if (basename_of(names[i]) != "bytecode.pkl") continue;
    size_t slashes = static_cast<size_t>(std::count(names[i].begin(), names[i].end(), '/'));
    if (slashes < best_slashes) {
      best_slashes = slashes;
      best = static_cast<int32_t>(i);
    }
  }
  return best;
}

int32_t select_main_function(const BcModule& m, std::string_view root_class) {
  auto method_of = [](const std::string& q) {
    auto d = q.rfind('.');
    return d == std::string::npos ? q : q.substr(d + 1);
  };
  const auto& fs = m.functions;
  if (!root_class.empty()) {
    const std::string want = std::string(root_class) + ".forward";
    for (size_t i = 0; i < fs.size(); ++i)
      if (fs[i].qualified_name == want) return static_cast<int32_t>(i);
  }
  for (size_t i = 0; i < fs.size(); ++i)
    if (method_of(fs[i].qualified_name) == "forward") return static_cast<int32_t>(i);
  if (!root_class.empty()) {
    const std::string pre = std::string(root_class) + ".";
    for (size_t i = 0; i < fs.size(); ++i) {
      const std::string& q = fs[i].qualified_name;
      if (q.rfind(pre, 0) != 0) continue;
      std::string meth = method_of(q);
      if (meth == "__getstate__" || meth == "__setstate__") continue;
      return static_cast<int32_t>(i);
    }
  }
  return -1;
}

std::string method_signature(const BcFunction& f) {
  if (!f.has_schema) return "";
  auto d = f.qualified_name.rfind('.');
  std::string s = d == std::string::npos ? f.qualified_name
                                         : f.qualified_name.substr(d + 1);
  auto arg = [](const BcArg& a) {
    return a.name.empty() ? a.type : a.name + ": " + a.type;
  };
  s += "(";
  for (size_t i = 0; i < f.arguments.size(); ++i) {
    if (i) s += ", ";
    s += arg(f.arguments[i]);
  }
  s += ") -> ";
  if (f.returns.size() == 1) {
    s += f.returns[0].type;
  } else {
    s += "(";
    for (size_t i = 0; i < f.returns.size(); ++i) {
      if (i) s += ", ";
      s += f.returns[i].type;
    }
    s += ")";
  }
  return s;
}

// ============================================================================
// Symbolic executor
// ============================================================================
namespace {

using ts::GraphBuilder;

// A stack entry or register value (spec §4.3).
struct Sym {
  enum class Kind : uint8_t { Empty, Edge, Pending, Const, TensorConst, Module };
  Kind kind = Kind::Empty;
  uint32_t scope = 0;          // Edge/Pending: id of the GraphBuilder it lives in
  uint32_t idx = 0;            // Edge: value index there; Pending: pending record
  const Value* v = nullptr;    // Const/TensorConst: the pickled value; Module: object
  uint32_t path = UINT32_MAX;  // Module: path string; TensorConst: initializer name

  bool operator==(const Sym& o) const {
    return kind == o.kind && scope == o.scope && idx == o.idx && v == o.v &&
           path == o.path;
  }
  bool operator!=(const Sym& o) const { return !(*this == o); }
};

Sym edge_sym(uint32_t scope, uint32_t value) {
  Sym s;
  s.kind = Sym::Kind::Edge;
  s.scope = scope;
  s.idx = value;
  return s;
}

ir::Attribute make_attr(ir::Model& m, std::string_view name, ir::AttrValue v) {
  ir::Attribute a;
  a.name = m.intern(name);
  a.value = std::move(v);
  return a;
}
ir::Attribute attr_str(ir::Model& m, std::string_view name, std::string_view s) {
  ir::AttrValue v;
  v.kind = ir::AttrValue::Kind::String;
  v.s = m.intern(s);
  return make_attr(m, name, std::move(v));
}
ir::Attribute attr_int(ir::Model& m, std::string_view name, int64_t i) {
  ir::AttrValue v;
  v.kind = ir::AttrValue::Kind::Int;
  v.i = i;
  return make_attr(m, name, std::move(v));
}
ir::Attribute attr_graph(ir::Model& m, std::string_view name, int32_t g) {
  ir::AttrValue v;
  v.kind = ir::AttrValue::Kind::Graph;
  v.graph = g;
  return make_attr(m, name, std::move(v));
}

class Executor {
 public:
  Executor(const BcModule& m, const BcFunction& f, BuildContext& ctx)
      : m_(m), f_(f), ctx_(ctx), model_(ctx.model), lim_(ctx.lim),
        ins_(f.instructions) {}

  Result<BuildResult> run();

 private:
  // A pending output (spec §4.3.2): an op node or a whole If node whose output
  // count (0 or 1) is decided by whether a later instruction pops it.
  struct PendRec {
    uint32_t scope = 0;
    uint32_t node = 0;
    int32_t if_rec = -1;  // >= 0: a pending prim::If (index into ifs_)
    bool done = false;
  };
  // An If whose branches have executed but whose output count is not yet
  // final; finalize_if() closes both branches with k outputs.
  struct IfRec {
    uint32_t scope = 0, node = 0, then_id = 0, else_id = 0, cond = 0;
    std::vector<Sym> s_then, s_else;
    bool finalized = false;
  };
  struct JournalEnt {
    uint32_t reg;
    Sym old;
  };

  // --- errors ---
  uint64_t ins_pos(int64_t pc) const {
    if (pc >= 0 && static_cast<uint64_t>(pc) < ins_.size())
      return ins_[static_cast<size_t>(pc)].pos;
    return f_.pos;
  }
  Error fail(int64_t pc, const std::string& msg) const {
    return err("pc " + std::to_string(pc) + ": " + msg, ins_pos(pc));
  }
  Error internal(const std::string& msg) const {
    return err("internal: " + msg, f_.pos);
  }

  GraphBuilder& B(uint32_t id) { return *builders_[id]; }

  uint32_t intern_str(const std::string& s) {
    auto it = str_ids_.find(s);
    if (it != str_ids_.end()) return it->second;
    auto id = static_cast<uint32_t>(strs_.size());
    strs_.push_back(s);
    str_ids_.emplace(s, id);
    return id;
  }

  // --- graph slots ---
  Result<uint32_t> reserve_graph(int64_t pc, const std::string& name,
                                 GraphBuilder* parent) {
    if (model_.graphs.size() - 1 >= lim_.max_subgraphs)
      return fail(pc, "more than " + std::to_string(lim_.max_subgraphs) +
                          " subgraphs in one method");
    auto id = static_cast<uint32_t>(model_.graphs.size());
    model_.graphs.emplace_back();
    builders_.push_back(
        std::make_unique<GraphBuilder>(model_, names_, name, id, parent));
    return id;
  }

  // --- stack ---
  size_t pendings_above(size_t base) const {
    if (st_.size() <= base) return 0;
    return pend_prefix_.back() - (base ? pend_prefix_[base - 1] : 0);
  }
  bool at_base(size_t base) const {
    return pendings_above(base) == st_.size() - base;
  }
  Result<bool> push(const Sym& s, int64_t pc) {
    if (st_.size() >= lim_.max_stack)
      return fail(pc, "symbolic stack deeper than " + std::to_string(lim_.max_stack));
    uint32_t prev = pend_prefix_.empty() ? 0 : pend_prefix_.back();
    st_.push_back(s);
    pend_prefix_.push_back(prev + (s.kind == Sym::Kind::Pending ? 1u : 0u));
    return true;
  }
  std::vector<Sym> take_segment(size_t base) {
    std::vector<Sym> seg(st_.begin() + static_cast<std::ptrdiff_t>(base), st_.end());
    st_.resize(base);
    pend_prefix_.resize(base);
    return seg;
  }
  // Pop n entries (bottom->top order), confirming any pending among them.
  Result<std::vector<Sym>> pop_n(int64_t n, size_t base, int64_t pc) {
    if (n < 0 || static_cast<uint64_t>(n) > st_.size() - base)
      return fail(pc, "stack underflow below block base");
    std::vector<Sym> out = take_segment(st_.size() - static_cast<size_t>(n));
    for (Sym& s : out) {
      auto c = confirm(s);
      if (!c) return c.error();
      s = *c;
    }
    return out;
  }

  // --- pending resolution ---
  Result<Sym> confirm(const Sym& s);
  Result<bool> retract(const Sym& s);
  // Retract seg[0, cut). By the prefix invariant (pendings above a block base
  // are always a contiguous prefix) every one of them is Pending; a confirmed
  // value there would be silently lost, so it is an internal error instead.
  Result<bool> retract_prefix(const std::vector<Sym>& seg, size_t cut) {
    for (size_t i = 0; i < cut; ++i) {
      if (seg[i].kind != Sym::Kind::Pending)
        return internal("pending-prefix invariant violated");
      auto r = retract(seg[i]);
      if (!r) return r.error();
    }
    return true;
  }
  Result<std::vector<uint32_t>> finalize_if(size_t rec, int64_t k);
  Result<bool> checkpoint(size_t base, int64_t pc, BcOp op);

  // --- registers (undo journal) ---
  void set_reg(uint32_t r, const Sym& s) {
    journal_.push_back({r, regs_[r]});
    regs_[r] = s;
  }
  std::vector<std::pair<uint32_t, Sym>> collect_writes(size_t mark) {
    std::vector<std::pair<uint32_t, Sym>> out;
    ++epoch_;
    for (size_t i = mark; i < journal_.size(); ++i) {
      uint32_t r = journal_[i].reg;
      if (stamp_[r] == epoch_) continue;
      stamp_[r] = epoch_;
      out.emplace_back(r, regs_[r]);
    }
    return out;
  }
  void rollback(size_t mark) {
    for (size_t i = journal_.size(); i > mark; --i)
      regs_[journal_[i - 1].reg] = journal_[i - 1].old;
    journal_.resize(mark);
  }
  void merge_if(const std::vector<std::pair<uint32_t, Sym>>& wt,
                const std::vector<std::pair<uint32_t, Sym>>& we);

  // --- edges ---
  Result<uint32_t> resolve(GraphBuilder& b, uint32_t scope, uint32_t idx);
  Result<uint32_t> use_edge(GraphBuilder& b, const Sym& s);
  Result<uint32_t> materialize_module(GraphBuilder& b, const Sym& s);
  uint32_t tensor_init(GraphBuilder& b, const Sym& s);
  Result<uint32_t> make_node(GraphBuilder& b, std::string_view op_type,
                             const std::vector<Sym>& inputs,
                             std::vector<ir::Attribute> pre);
  Result<bool> push_outputs(GraphBuilder& b, uint32_t node, int64_t count,
                            int64_t pc);
  Result<bool> push_sn(GraphBuilder& b, uint32_t node, int64_t pc, int64_t hi,
                       size_t base);
  Result<bool> type_attr(int64_t x, int64_t pc, std::vector<ir::Attribute>& pre);
  uint32_t const_name(size_t x);

  // --- execution ---
  Result<bool> exec_block(GraphBuilder& b, int64_t lo, int64_t hi, size_t base);
  Result<int64_t> exec_if(GraphBuilder& b, int64_t p, int64_t hi, size_t base);
  Result<int64_t> exec_loop(GraphBuilder& b, int64_t s, int64_t hi, size_t base);
  Result<bool> exec_ret(GraphBuilder& b, int64_t pc, size_t base);
  Result<bool> exec_get_attr(GraphBuilder& b, const BcInstruction& in, int64_t pc,
                             size_t base);
  Result<bool> prescan();

  const BcModule& m_;
  const BcFunction& f_;
  BuildContext& ctx_;
  ir::Model& model_;
  const BytecodeLimits& lim_;
  const std::vector<BcInstruction>& ins_;

  ts::NameCounter names_;
  std::vector<std::unique_ptr<GraphBuilder>> builders_;  // index == graph slot
  std::vector<Sym> st_;
  std::vector<uint32_t> pend_prefix_;  // #Pending in st_[0..i]
  std::vector<Sym> regs_;
  std::vector<JournalEnt> journal_;
  std::vector<uint32_t> stamp_;
  uint32_t epoch_ = 0;
  std::vector<PendRec> pend_;
  std::vector<IfRec> ifs_;
  std::vector<std::string> strs_;
  std::unordered_map<std::string, uint32_t> str_ids_;
  std::unordered_map<uint64_t, uint32_t> module_cache_;  // (graph, path) -> value
  std::unordered_map<size_t, uint32_t> const_names_;     // constant idx -> name
  std::unordered_set<int64_t> setattr_slots_;
  std::unordered_set<const Value*> bound_seen_;
  BuildResult result_;
  uint32_t root_path_ = 0;
  uint32_t self_value_ = UINT32_MAX;
  uint32_t cf_depth_ = 0;
  uint32_t control_ord_ = 0;
  bool returned_ = false;
};

// ---- pending resolution -----------------------------------------------------
Result<Sym> Executor::confirm(const Sym& s) {
  if (s.kind != Sym::Kind::Pending) return s;
  if (s.idx >= pend_.size() || pend_[s.idx].done)
    return internal("pending output resolved twice");
  pend_[s.idx].done = true;
  const PendRec r = pend_[s.idx];
  if (r.if_rec < 0) return edge_sym(r.scope, B(r.scope).add_output(r.node));
  auto outs = finalize_if(static_cast<size_t>(r.if_rec), 1);
  if (!outs) return outs.error();
  return edge_sym(r.scope, (*outs)[0]);
}

Result<bool> Executor::retract(const Sym& s) {
  if (s.kind != Sym::Kind::Pending) return true;
  if (s.idx >= pend_.size() || pend_[s.idx].done)
    return internal("pending output resolved twice");
  pend_[s.idx].done = true;
  const PendRec r = pend_[s.idx];
  if (r.if_rec >= 0) {
    auto outs = finalize_if(static_cast<size_t>(r.if_rec), 0);
    if (!outs) return outs.error();
  }
  return true;  // an op node simply keeps zero outputs
}

Result<std::vector<uint32_t>> Executor::finalize_if(size_t rec, int64_t k) {
  if (rec >= ifs_.size() || ifs_[rec].finalized) return internal("if finalized twice");
  ifs_[rec].finalized = true;
  // Copy out: nested finalizations below must not hold references into ifs_.
  const uint32_t scope = ifs_[rec].scope, node = ifs_[rec].node,
                 cond = ifs_[rec].cond;
  const uint32_t ids[2] = {ifs_[rec].then_id, ifs_[rec].else_id};
  std::vector<Sym> segs[2] = {std::move(ifs_[rec].s_then), std::move(ifs_[rec].s_else)};
  // Then-branch outputs, then else-branch outputs, then the If's own outputs:
  // that order fixes the value-name order (spec §4.5.1 step 8).
  for (int br = 0; br < 2; ++br) {
    std::vector<Sym>& seg = segs[br];
    if (k < 0 || static_cast<uint64_t>(k) > seg.size())
      return internal("if arity outside its branch");
    size_t cut = seg.size() - static_cast<size_t>(k);
    auto rp = retract_prefix(seg, cut);
    if (!rp) return rp.error();
    for (size_t i = cut; i < seg.size(); ++i) {
      auto v = use_edge(B(ids[br]), seg[i]);
      if (!v) return v.error();
      B(ids[br]).add_output_value(*v);
    }
  }
  GraphBuilder& b = B(scope);
  b.add_edge(node, cond);
  // Captures: then-branch first-use order, then else-only ones.
  std::vector<uint32_t> caps = B(ids[0]).captured_parent_values();
  std::unordered_set<uint32_t> seen(caps.begin(), caps.end());
  for (uint32_t p : B(ids[1]).captured_parent_values())
    if (seen.insert(p).second) caps.push_back(p);
  for (uint32_t c : caps) b.add_edge(node, c);
  std::vector<uint32_t> outs;
  for (int64_t i = 0; i < k; ++i) outs.push_back(b.add_output(node));
  b.set_subgraph(node, static_cast<int32_t>(ids[0]));
  b.add_attr(node, attr_graph(model_, ts::attr::kThen, static_cast<int32_t>(ids[0])));
  b.add_attr(node, attr_graph(model_, ts::attr::kElse, static_cast<int32_t>(ids[1])));
  b.add_attr(node, attr_int(model_, ts::attr::kCaptures, static_cast<int64_t>(caps.size())));
  return outs;
}

Result<bool> Executor::checkpoint(size_t base, int64_t pc, BcOp op) {
  size_t above = st_.size() - base;
  size_t pend = pendings_above(base);
  if (pend != above)
    return fail(pc, std::string("stack not at block base after ") + opcode_name(op) +
                        " (" + std::to_string(above - pend) + " residual values)");
  std::vector<Sym> seg = take_segment(base);
  for (const Sym& s : seg) {
    auto r = retract(s);
    if (!r) return r.error();
  }
  return true;
}

void Executor::merge_if(const std::vector<std::pair<uint32_t, Sym>>& wt,
                        const std::vector<std::pair<uint32_t, Sym>>& we) {
  // A register keeps its value only if both branches leave the same sym (an
  // unwritten register's "final" value is its pre-branch value).
  std::unordered_map<uint32_t, Sym> tm(wt.begin(), wt.end()), em(we.begin(), we.end());
  std::vector<uint32_t> order;
  for (const auto& w : wt) order.push_back(w.first);
  for (const auto& w : we)
    if (tm.find(w.first) == tm.end()) order.push_back(w.first);
  for (uint32_t r : order) {
    auto ti = tm.find(r);
    auto ei = em.find(r);
    Sym t = ti != tm.end() ? ti->second : regs_[r];
    Sym e = ei != em.end() ? ei->second : regs_[r];
    set_reg(r, t == e ? t : Sym{});
  }
}

// ---- edges -------------------------------------------------------------------
Result<uint32_t> Executor::resolve(GraphBuilder& b, uint32_t scope, uint32_t idx) {
  if (b.id() == scope) return idx;
  GraphBuilder* p = b.parent();
  if (!p) return internal("value used outside its scope");
  auto pv = resolve(*p, scope, idx);
  if (!pv) return pv.error();
  return b.capture(*pv, p->value_name(*pv));
}

Result<uint32_t> Executor::use_edge(GraphBuilder& b, const Sym& s0) {
  auto c = confirm(s0);
  if (!c) return c.error();
  const Sym& s = *c;
  switch (s.kind) {
    case Sym::Kind::Edge:
      return resolve(b, s.scope, s.idx);
    case Sym::Kind::Const: {
      ir::AttrValue av;
      ts::constant_to_attr(*s.v, model_, av);
      uint32_t n = b.node(ts::kind::kConstant);
      b.add_attr(n, make_attr(model_, ts::attr::kValue, std::move(av)));
      return b.add_output(n);
    }
    case Sym::Kind::TensorConst:
      return tensor_init(b, s);
    case Sym::Kind::Module:
      return materialize_module(b, s);
    default:
      return internal("use of an empty value");
  }
}

uint32_t Executor::tensor_init(GraphBuilder& b, const Sym& s) {
  ir::TensorRef t = s.v->tensor;
  if (!s.v->dtype_label.empty()) t.dtype_label = model_.intern(s.v->dtype_label);
  return b.initializer(strs_[s.path], t);
}

Result<uint32_t> Executor::materialize_module(GraphBuilder& b, const Sym& s) {
  if (s.path == root_path_) {
    if (self_value_ == UINT32_MAX) {
      self_value_ = B(0).add_input("self");
      B(0).prepend_input(self_value_);
    }
    return resolve(b, 0, self_value_);
  }
  const uint64_t key = (static_cast<uint64_t>(b.id()) << 32) | s.path;
  auto it = module_cache_.find(key);
  if (it != module_cache_.end()) return it->second;
  Sym root;
  root.kind = Sym::Kind::Module;
  root.v = ctx_.env.root();
  root.path = root_path_;
  auto rv = materialize_module(b, root);
  if (!rv) return rv.error();
  uint32_t n = b.node(ts::kind::kGetAttr);
  b.add_edge(n, *rv);
  b.add_attr(n, attr_str(model_, ts::attr::kName, strs_[s.path]));
  uint32_t out = b.add_output(n, strs_[s.path]);
  module_cache_.emplace(key, out);
  return out;
}

// Inputs of an op-like consumer: a Const operand folds into attribute arg<k>
// (k = its position among the popped operands); everything else is an edge.
// Edges are resolved BEFORE the node is staged so any prim::Constant /
// prim::GetAttr they need precede it in node order.
Result<uint32_t> Executor::make_node(GraphBuilder& b, std::string_view op_type,
                                     const std::vector<Sym>& inputs,
                                     std::vector<ir::Attribute> pre) {
  std::vector<uint32_t> edges;
  std::vector<ir::Attribute> args;
  for (size_t k = 0; k < inputs.size(); ++k) {
    const Sym& s = inputs[k];
    if (s.kind == Sym::Kind::Const) {
      ir::AttrValue av;
      ts::constant_to_attr(*s.v, model_, av);
      args.push_back(make_attr(model_, ts::arg_attr_name(static_cast<uint32_t>(k)),
                               std::move(av)));
      continue;
    }
    auto e = use_edge(b, s);
    if (!e) return e.error();
    edges.push_back(*e);
  }
  uint32_t n = b.node(op_type);
  for (ir::Attribute& a : pre) b.add_attr(n, std::move(a));
  for (ir::Attribute& a : args) b.add_attr(n, std::move(a));
  for (uint32_t e : edges) b.add_edge(n, e);
  return n;
}

Result<bool> Executor::push_outputs(GraphBuilder& b, uint32_t node, int64_t count,
                                    int64_t pc) {
  if (count < 0 || static_cast<uint64_t>(count) > lim_.max_stack - st_.size())
    return fail(pc, "symbolic stack deeper than " + std::to_string(lim_.max_stack));
  for (int64_t i = 0; i < count; ++i) {
    auto p = push(edge_sym(b.id(), b.add_output(node)), pc);
    if (!p) return p.error();
  }
  return true;
}

// Output count of OP / OPN / INTERFACE_CALL (spec §4.3.2).
Result<bool> Executor::push_sn(GraphBuilder& b, uint32_t node, int64_t pc, int64_t hi,
                               size_t base) {
  // 1. Followed (inside this block) by STOREN N: a multi-output node (I2).
  if (pc + 1 < hi && ins_[static_cast<size_t>(pc + 1)].op == BcOp::STOREN)
    return push_outputs(b, node, ins_[static_cast<size_t>(pc + 1)].n, pc);
  // 2. Effectively at the block base: 0 or 1 outputs, decided later.
  if (at_base(base)) {
    PendRec r;
    r.scope = b.id();
    r.node = node;
    pend_.push_back(r);
    Sym s;
    s.kind = Sym::Kind::Pending;
    s.scope = b.id();
    s.idx = static_cast<uint32_t>(pend_.size() - 1);
    return push(s, pc);
  }
  // 3. Something below still waits for a parent: an inline node, 1 output.
  return push_outputs(b, node, 1, pc);
}

Result<bool> Executor::type_attr(int64_t x, int64_t pc, std::vector<ir::Attribute>& pre) {
  if (x < 0 || static_cast<uint64_t>(x) >= f_.types.size())
    return fail(pc, "types index " + std::to_string(x) + " out of range");
  auto i = static_cast<size_t>(x);
  if (f_.type_ok[i]) {
    pre.push_back(attr_str(model_, ts::attr::kType, f_.types[i]));
  } else {
    pre.push_back(attr_str(model_, ts::attr::kType, "?"));
    pre.push_back(attr_str(model_, ts::attr::kUnresolved,
                           "types[" + std::to_string(x) + "] is not a string"));
  }
  return true;
}

uint32_t Executor::const_name(size_t x) {
  auto it = const_names_.find(x);
  if (it != const_names_.end()) return it->second;
  auto d = f_.qualified_name.rfind('.');
  std::string short_name = d == std::string::npos ? f_.qualified_name
                                                   : f_.qualified_name.substr(d + 1);
  std::string fallback = short_name + ".c" + std::to_string(x);
  uint32_t id = intern_str(ctx_.const_names.name_for(f_.constants[x]->tensor, fallback));
  const_names_.emplace(x, id);
  return id;
}

// ---- GET_ATTR (spec §4.6.1) ---------------------------------------------------
Result<bool> Executor::exec_get_attr(GraphBuilder& b, const BcInstruction& in,
                                     int64_t pc, size_t base) {
  auto popped = pop_n(1, base, pc);
  if (!popped) return popped.error();
  const Sym r = (*popped)[0];
  const int64_t X = in.x;
  using AK = ts::ModuleEnv::Attr::Kind;
  ts::ModuleEnv::Attr a;  // Unknown
  std::string child;
  bool bound_receiver =
      r.kind == Sym::Kind::Module && setattr_slots_.find(X) == setattr_slots_.end();
  if (bound_receiver) {
    a = ctx_.env.by_slot(r.v, X);
    if (a.kind != AK::Unknown) {
      const std::string& rp = strs_[r.path];
      child = rp.empty() ? a.name : rp + "." + a.name;
      if (child.size() > lim_.max_attr_path) {
        a = ts::ModuleEnv::Attr{};  // path too long: not recorded
        child.clear();
      }
    }
    switch (a.kind) {
      case AK::Tensor: {
        if (bound_seen_.insert(a.value).second)
          result_.bound_module_tensors.push_back(a.value);
        Sym s;
        s.kind = Sym::Kind::TensorConst;
        s.v = a.value;
        s.path = intern_str(ctx_.env.tensor_path(a.value, child));
        return push(s, pc);
      }
      case AK::Object: {
        Sym s;
        s.kind = Sym::Kind::Module;
        s.v = a.value;
        s.path = intern_str(child);
        return push(s, pc);
      }
      case AK::Constant: {
        Sym s;
        s.kind = Sym::Kind::Const;
        s.v = a.value;
        return push(s, pc);
      }
      default:
        break;  // Opaque / Unknown: a prim::GetAttr node below
    }
  }
  // prim::GetAttr on an unbindable receiver or attribute.
  auto recv = use_edge(b, r);
  if (!recv) return recv.error();
  uint32_t n = b.node(ts::kind::kGetAttr);
  b.add_edge(n, *recv);
  const bool named = bound_receiver && a.kind == AK::Opaque;
  if (named) b.add_attr(n, attr_str(model_, ts::attr::kName, child));
  b.add_attr(n, attr_int(model_, ts::attr::kSlot, X));
  return push(edge_sym(b.id(), b.add_output(n, named ? child : std::string())), pc);
}

// ---- If (spec §4.5.1) ---------------------------------------------------------
Result<int64_t> Executor::exec_if(GraphBuilder& b, int64_t p, int64_t hi, size_t base) {
  auto cp = pop_n(1, base, p);
  if (!cp) return cp.error();
  const int64_t X = ins_[static_cast<size_t>(p)].x;
  if (X < 2) return fail(p, "malformed if (JF/JMP layout)");
  const int64_t j = p + X - 1;
  if (j >= hi || ins_[static_cast<size_t>(j)].op != BcOp::JMP ||
      ins_[static_cast<size_t>(j)].x < 1)
    return fail(p, "malformed if (JF/JMP layout)");
  const int64_t e = j + ins_[static_cast<size_t>(j)].x;
  if (e > hi) return fail(p, "malformed if (JF/JMP layout)");
  if (cf_depth_ + 1 > lim_.max_cf_depth)
    return fail(p, "control flow nested deeper than " + std::to_string(lim_.max_cf_depth));
  if (model_.graphs.size() - 1 + 2 > lim_.max_subgraphs)
    return fail(p, "more than " + std::to_string(lim_.max_subgraphs) +
                       " subgraphs in one method");

  auto cond = use_edge(b, (*cp)[0]);
  if (!cond) return cond.error();
  const std::string k = std::to_string(control_ord_++);
  auto t = reserve_graph(p, b.name() + "/if" + k + ".then", &b);
  if (!t) return t.error();
  auto s = reserve_graph(p, b.name() + "/if" + k + ".else", &b);
  if (!s) return s.error();

  const size_t d = st_.size();
  ++cf_depth_;
  const size_t mark = journal_.size();
  auto rt = exec_block(B(*t), p + 1, j, d);
  if (!rt) return rt.error();
  std::vector<Sym> seg_t = take_segment(d);
  auto wt = collect_writes(mark);
  rollback(mark);
  auto re = exec_block(B(*s), j + 1, e, d);
  if (!re) return re.error();
  std::vector<Sym> seg_e = take_segment(d);
  auto we = collect_writes(mark);
  rollback(mark);
  merge_if(wt, we);
  --cf_depth_;

  // Each segment is [pendings][confirmed] (the prefix invariant).
  auto confirmed = [](const std::vector<Sym>& seg) {
    int64_t c = 0;
    for (const Sym& x : seg) c += x.kind == Sym::Kind::Pending ? 0 : 1;
    return c;
  };
  const int64_t lo_k = std::max(confirmed(seg_t), confirmed(seg_e));
  const int64_t hi_k = static_cast<int64_t>(std::min(seg_t.size(), seg_e.size()));

  IfRec rec;
  rec.scope = b.id();
  rec.node = b.node(ts::kind::kIf);
  rec.then_id = *t;
  rec.else_id = *s;
  rec.cond = *cond;
  rec.s_then = std::move(seg_t);
  rec.s_else = std::move(seg_e);
  ifs_.push_back(std::move(rec));
  const size_t if_idx = ifs_.size() - 1;

  const std::string disagree = "if-branches disagree on output count";
  int64_t kout = -1;
  bool pending = false;
  // The If's own STORE/STOREN must be inside THIS block: an If that closes an
  // else-block would otherwise be credited the enclosing If's store.
  const BcOp after = e < hi ? ins_[static_cast<size_t>(e)].op : BcOp::Unknown;
  if (after == BcOp::STORE || after == BcOp::STOREN) {
    kout = after == BcOp::STORE ? 1 : ins_[static_cast<size_t>(e)].n;
    if (kout < lo_k || kout > hi_k) return fail(p, disagree);
  } else {
    const bool c0 = lo_k <= 0 && 0 <= hi_k;
    const bool c1 = lo_k <= 1 && 1 <= hi_k;
    if (!c0 && !c1) return fail(p, disagree);
    if (c0 && c1) {
      if (at_base(base))
        pending = true;
      else
        kout = 1;
    } else {
      kout = c0 ? 0 : 1;
    }
  }
  if (pending) {
    PendRec r;
    r.scope = b.id();
    r.node = ifs_[if_idx].node;
    r.if_rec = static_cast<int32_t>(if_idx);
    pend_.push_back(r);
    Sym ps;
    ps.kind = Sym::Kind::Pending;
    ps.scope = b.id();
    ps.idx = static_cast<uint32_t>(pend_.size() - 1);
    auto pu = push(ps, p);
    if (!pu) return pu.error();
  } else {
    auto outs = finalize_if(if_idx, kout);
    if (!outs) return outs.error();
    for (uint32_t v : *outs) {
      auto pu = push(edge_sym(b.id(), v), p);
      if (!pu) return pu.error();
    }
  }
  return e;
}

// ---- Loop (spec §4.5.2) -------------------------------------------------------
Result<int64_t> Executor::exec_loop(GraphBuilder& b, int64_t s, int64_t hi, size_t base) {
  const int64_t X = ins_[static_cast<size_t>(s)].x;
  const int64_t N = ins_[static_cast<size_t>(s)].n;
  if (N < 2 || X < 2) return fail(s, "malformed loop");
  const int64_t j = s + X - 1;
  if (j >= hi || ins_[static_cast<size_t>(j)].op != BcOp::JMP ||
      static_cast<int64_t>(ins_[static_cast<size_t>(j)].x) != -(X - 1))
    return fail(s, "malformed loop");
  if (cf_depth_ + 1 > lim_.max_cf_depth)
    return fail(s, "control flow nested deeper than " + std::to_string(lim_.max_cf_depth));

  // [iter0, max_trip, cond, carried...]; iter0 (the emitter's LOADC 0) is dropped.
  auto popped = pop_n(N + 1, base, s);
  if (!popped) return popped.error();
  std::vector<uint32_t> operands;
  for (size_t i = 1; i < popped->size(); ++i) {
    auto v = use_edge(b, (*popped)[i]);
    if (!v) return v.error();
    operands.push_back(*v);
  }
  const std::string k = std::to_string(control_ord_++);
  auto body = reserve_graph(s, b.name() + "/loop" + k + ".body", &b);
  if (!body) return body.error();
  GraphBuilder& bb = B(*body);

  const size_t d = st_.size();
  for (int64_t i = 0; i < N - 1; ++i) {  // [trip, carried params...]
    auto pu = push(edge_sym(bb.id(), bb.add_input("")), s);
    if (!pu) return pu.error();
  }
  ++cf_depth_;
  const size_t mark = journal_.size();
  auto rb = exec_block(bb, s + 1, j, d);
  if (!rb) return rb.error();
  std::vector<Sym> seg = take_segment(d);
  auto writes = collect_writes(mark);
  rollback(mark);
  for (const auto& w : writes)  // body-scoped values never escape via registers
    if (w.second != regs_[w.first]) set_reg(w.first, Sym{});
  --cf_depth_;

  int64_t c = 0;
  for (const Sym& x : seg) c += x.kind == Sym::Kind::Pending ? 0 : 1;
  const auto want = static_cast<size_t>(N - 1);
  if (c > N - 1 || seg.size() < want)
    return fail(s, "malformed loop body (" + std::to_string(N - 1) +
                       " values expected at the back-edge, " +
                       std::to_string(seg.size()) + " found)");
  const size_t cut = seg.size() - want;
  auto rp = retract_prefix(seg, cut);
  if (!rp) return rp.error();
  for (size_t i = cut; i < seg.size(); ++i) {  // [cond_next, carried_next...]
    auto v = use_edge(bb, seg[i]);
    if (!v) return v.error();
    bb.add_output_value(*v);
  }

  uint32_t n = b.node(ts::kind::kLoop);
  for (uint32_t v : operands) b.add_edge(n, v);
  const std::vector<uint32_t> caps = bb.captured_parent_values();
  for (uint32_t cpv : caps) b.add_edge(n, cpv);
  b.set_subgraph(n, static_cast<int32_t>(*body));
  b.add_attr(n, attr_graph(model_, ts::attr::kBody, static_cast<int32_t>(*body)));
  b.add_attr(n, attr_int(model_, ts::attr::kCaptures, static_cast<int64_t>(caps.size())));
  auto po = push_outputs(b, n, N - 2, s);
  if (!po) return po.error();
  return s + X;
}

// ---- RET ----------------------------------------------------------------------
Result<bool> Executor::exec_ret(GraphBuilder& b, int64_t pc, size_t base) {
  if (cf_depth_ > 0) return fail(pc, "RET inside a branch or loop body");
  if (static_cast<uint64_t>(pc) + 1 != ins_.size())
    return fail(pc, "trailing instructions after RET");
  const size_t n = f_.returns.size();
  const size_t above = st_.size() - base;
  if (above < n)
    return fail(pc, "stack underflow at RET (" + std::to_string(above) + " values for " +
                        std::to_string(n) + " returns)");
  const size_t below = above - n;
  const size_t pend_below =
      below == 0 ? 0
                 : pend_prefix_[base + below - 1] - (base ? pend_prefix_[base - 1] : 0);
  if (pend_below != below) return fail(pc, "values left on stack at RET");
  std::vector<Sym> seg = take_segment(base);
  auto rp = retract_prefix(seg, below);
  if (!rp) return rp.error();
  for (size_t i = below; i < seg.size(); ++i) {
    auto v = use_edge(b, seg[i]);
    if (!v) return v.error();
    b.add_output_value(*v);
  }
  returned_ = true;
  return true;
}

// ---- the instruction loop -----------------------------------------------------
// Returns true when the block executed RET (top level only).
Result<bool> Executor::exec_block(GraphBuilder& b, int64_t lo, int64_t hi, size_t base) {
  using namespace ts::kind;
  int64_t pc = lo;
  // Fixed-arity node: pop `nin`, stage `op_type`, push `nout` confirmed outputs.
  auto fixed = [&](const char* op_type, int64_t nin, int64_t nout,
                   std::vector<ir::Attribute> pre = {}) -> Result<uint32_t> {
    auto in = pop_n(nin, base, pc);
    if (!in) return in.error();
    auto n = make_node(b, op_type, *in, std::move(pre));
    if (!n) return n.error();
    auto po = push_outputs(b, *n, nout, pc);
    if (!po) return po.error();
    return *n;
  };
  while (pc < hi) {
    const BcInstruction& in = ins_[static_cast<size_t>(pc)];
    const int64_t X = in.x, N = in.n;
    Result<bool> ok = true;
    switch (in.op) {
      case BcOp::OP:
      case BcOp::OPN: {
        if (X < 0 || static_cast<uint64_t>(X) >= f_.operators.size())
          return fail(pc, "operator index " + std::to_string(X) + " out of range");
        const BcOperator& o = f_.operators[static_cast<size_t>(X)];
        int64_t nin = N;
        if (in.op == BcOp::OP) {
          if (!o.has_num_args || o.num_args < 0)
            return fail(pc, "operator '" + (o.well_formed ? o.name : std::string("?")) +
                                "' has no recorded arity");
          nin = o.num_args;
        }
        auto popped = pop_n(nin, base, pc);
        if (!popped) return popped.error();
        std::vector<ir::Attribute> pre;
        std::string op_type;
        if (o.well_formed) {
          op_type = o.name;
          if (!o.overload.empty())
            pre.push_back(attr_str(model_, ts::attr::kOverload, o.overload));
        } else {
          op_type = kUnresolved;
          pre.push_back(attr_str(model_, ts::attr::kUnresolved,
                                 "operators[" + std::to_string(X) + "] malformed"));
        }
        auto n = make_node(b, op_type, *popped, std::move(pre));
        if (!n) return n.error();
        ok = push_sn(b, *n, pc, hi, base);
        break;
      }
      case BcOp::LOAD:
      case BcOp::MOVE: {
        const auto r = static_cast<uint32_t>(X);  // range-checked by prescan()
        if (regs_[r].kind == Sym::Kind::Empty)
          return fail(pc, "read of undefined register " + std::to_string(X));
        ok = push(regs_[r], pc);
        if (in.op == BcOp::MOVE) set_reg(r, Sym{});
        break;
      }
      case BcOp::STORE: {
        auto v = pop_n(1, base, pc);
        if (!v) return v.error();
        set_reg(static_cast<uint32_t>(X), (*v)[0]);
        ok = checkpoint(base, pc, in.op);
        break;
      }
      case BcOp::STOREN: {
        if (N < 1) return fail(pc, "STOREN with N < 1");
        auto v = pop_n(N, base, pc);
        if (!v) return v.error();
        for (int64_t i = 0; i < N; ++i)
          set_reg(static_cast<uint32_t>(X + i), (*v)[static_cast<size_t>(i)]);
        ok = checkpoint(base, pc, in.op);
        break;
      }
      case BcOp::DROP: {
        auto v = pop_n(1, base, pc);  // confirms a pending: the value exists, unused
        if (!v) return v.error();
        ok = checkpoint(base, pc, in.op);
        break;
      }
      case BcOp::DROPR:
        set_reg(static_cast<uint32_t>(X), Sym{});
        ok = checkpoint(base, pc, in.op);
        break;
      case BcOp::LOADC: {
        if (X < 0 || static_cast<uint64_t>(X) >= f_.constants.size())
          return fail(pc, "constant index " + std::to_string(X) + " out of range");
        const auto x = static_cast<size_t>(X);
        const Value* v = f_.constants[x].get();
        if (!v) return fail(pc, "constant " + std::to_string(X) + " is malformed");
        Sym s;
        s.v = v;
        if (v->kind == Value::Kind::Tensor) {
          s.kind = Sym::Kind::TensorConst;
          s.path = const_name(x);
        } else {
          s.kind = Sym::Kind::Const;
        }
        ok = push(s, pc);
        break;
      }
      case BcOp::JF: {
        auto next = exec_if(b, pc, hi, base);
        if (!next) return next.error();
        pc = *next;
        continue;
      }
      case BcOp::JMP:
        return fail(pc, "unstructured JMP");
      case BcOp::LOOP: {
        auto next = exec_loop(b, pc, hi, base);
        if (!next) return next.error();
        pc = *next;
        continue;
      }
      case BcOp::RET: {
        auto r = exec_ret(b, pc, base);
        if (!r) return r.error();
        return true;
      }
      case BcOp::GET_ATTR:
        ok = exec_get_attr(b, in, pc, base);
        break;
      case BcOp::SET_ATTR: {
        auto v = pop_n(2, base, pc);  // [receiver, value]
        if (!v) return v.error();
        std::vector<ir::Attribute> pre;
        const Sym& recv = (*v)[0];
        if (recv.kind == Sym::Kind::Module) {
          auto a = ctx_.env.by_slot(recv.v, X);
          if (a.kind != ts::ModuleEnv::Attr::Kind::Unknown) {
            const std::string& rp = strs_[recv.path];
            pre.push_back(attr_str(model_, ts::attr::kName,
                                   rp.empty() ? a.name : rp + "." + a.name));
          }
        }
        pre.push_back(attr_int(model_, ts::attr::kSlot, X));
        auto n = make_node(b, kSetAttr, *v, std::move(pre));
        if (!n) return n.error();
        ok = checkpoint(base, pc, in.op);
        break;
      }
      case BcOp::LIST_CONSTRUCT:
      case BcOp::NAMED_TUPLE_CONSTRUCT:
      case BcOp::DICT_CONSTRUCT: {
        if (in.op == BcOp::DICT_CONSTRUCT && N % 2 != 0)
          return fail(pc, "DICT_CONSTRUCT with an odd operand count");
        std::vector<ir::Attribute> pre;
        auto ta = type_attr(X, pc, pre);
        if (!ta) return ta.error();
        const char* kind = in.op == BcOp::LIST_CONSTRUCT  ? kListConstruct
                           : in.op == BcOp::DICT_CONSTRUCT ? kDictConstruct
                                                          : kTupleConstruct;
        auto n = fixed(kind, N, 1, std::move(pre));
        if (!n) return n.error();
        break;
      }
      case BcOp::TUPLE_CONSTRUCT: {
        if (X < 0) return fail(pc, "TUPLE_CONSTRUCT with a negative size");
        auto n = fixed(kTupleConstruct, X, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::LIST_UNPACK: {
        if (X < 0) return fail(pc, "LIST_UNPACK with a negative size");
        auto n = fixed(kListUnpack, 1, X);
        if (!n) return n.error();
        break;
      }
      case BcOp::TUPLE_SLICE: {
        if (X < 0) return fail(pc, "TUPLE_SLICE with a negative begin");
        std::vector<ir::Attribute> pre;
        pre.push_back(attr_int(model_, ts::attr::kBeg, X));
        pre.push_back(attr_int(model_, ts::attr::kEnd, X + N));
        auto n = fixed(kTupleSlice, 1, 1, std::move(pre));
        if (!n) return n.error();
        break;
      }
      case BcOp::TUPLE_INDEX: {
        auto n = fixed(kTupleIndex, 2, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::DICT_INDEX: {
        auto n = fixed(kGetItem, 2, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::CREATE_OBJECT: {
        std::vector<ir::Attribute> pre;
        auto ta = type_attr(X, pc, pre);
        if (!ta) return ta.error();
        auto n = fixed(kCreateObject, 0, 1, std::move(pre));
        if (!n) return n.error();
        break;
      }
      case BcOp::ISINSTANCE: {
        if (X < 0 || X + N > static_cast<int64_t>(f_.types.size()))
          return fail(pc, "ISINSTANCE types range out of bounds");
        ir::AttrValue tv;
        tv.kind = ir::AttrValue::Kind::Strings;
        bool bad = false;
        for (int64_t i = X; i < X + N; ++i) {
          tv.strings.push_back(model_.intern(f_.types[static_cast<size_t>(i)]));
          bad = bad || !f_.type_ok[static_cast<size_t>(i)];
        }
        std::vector<ir::Attribute> pre;
        pre.push_back(make_attr(model_, ts::attr::kTypes, std::move(tv)));
        if (bad)
          pre.push_back(attr_str(model_, ts::attr::kUnresolved,
                                 "a types[] entry is not a string"));
        auto n = fixed(kIsInstance, 1, 1, std::move(pre));
        if (!n) return n.error();
        break;
      }
      case BcOp::INTERFACE_CALL: {
        if (N < 1) return fail(pc, "INTERFACE_CALL without a receiver");
        if (X < 0 || static_cast<uint64_t>(X) >= f_.constants.size() ||
            !is(f_.constants[static_cast<size_t>(X)], Value::Kind::Str))
          return fail(pc, "INTERFACE_CALL method name is not a string constant");
        auto popped = pop_n(N, base, pc);
        if (!popped) return popped.error();
        std::vector<ir::Attribute> pre;
        pre.push_back(attr_str(model_, ts::attr::kName,
                               f_.constants[static_cast<size_t>(X)]->s));
        auto n = make_node(b, kCallMethod, *popped, std::move(pre));
        if (!n) return n.error();
        ok = push_sn(b, *n, pc, hi, base);
        break;
      }
      case BcOp::RAISE_EXCEPTION:
      case BcOp::WARN: {
        auto n = fixed(in.op == BcOp::WARN ? kWarn : kRaiseException, 2, 0);
        if (!n) return n.error();
        ok = checkpoint(base, pc, in.op);
        break;
      }
      case BcOp::UNCHECKED_CAST: {
        auto n = fixed(kUncheckedCast, 1, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::IS:
      case BcOp::ISNOT: {
        auto n = fixed(in.op == BcOp::IS ? kIs : kIsNot, 2, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::NOT: {
        auto n = fixed(kNot, 1, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::UN_INITIALIZED: {
        auto n = fixed(kUninitialized, 0, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::FORMAT: {
        if (X < 0) return fail(pc, "FORMAT with a negative operand count");
        auto n = fixed(kFormat, X, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::DEVICE:
      case BcOp::DTYPE:
      case BcOp::DIM:
      case BcOp::IS_CUDA: {
        const char* kind = in.op == BcOp::DEVICE  ? kDevice
                           : in.op == BcOp::DTYPE ? kDtype
                           : in.op == BcOp::DIM   ? kDim
                                                  : kIsCuda;
        auto n = fixed(kind, 1, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::TO_LIST: {
        auto n = fixed(kToList, 3, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::NUM_TO_TENSOR: {
        auto n = fixed(kNumToTensor, 1, 1);
        if (!n) return n.error();
        break;
      }
      case BcOp::Unknown: {
        const std::string raw = in.unknown_name < f_.unknown_opcodes.size()
                                    ? f_.unknown_opcodes[in.unknown_name]
                                    : std::string("?");
        return fail(pc, "unknown instruction '" + raw + "'");
      }
      default:  // WAIT CALL GUARD TYPECHECK FAIL_GUARD PROFILE_OP TAIL_CALL FORK ...
        return fail(pc, std::string("instruction ") + opcode_name(in.op) +
                            " is not valid in mobile bytecode");
    }
    if (!ok) return ok.error();
    ++pc;
  }
  return false;
}

// Register operands are 1-based; registers are allocated to the largest one
// referenced, never to register_size blindly.
Result<bool> Executor::prescan() {
  int64_t max_ref = 0, max_pc = 0;
  for (size_t i = 0; i < ins_.size(); ++i) {
    const BcInstruction& in = ins_[i];
    int64_t last = 0;
    switch (in.op) {
      case BcOp::LOAD:
      case BcOp::MOVE:
      case BcOp::STORE:
      case BcOp::DROPR:
        last = in.x;
        break;
      case BcOp::STOREN:
        last = static_cast<int64_t>(in.x) + std::max<int64_t>(in.n, 1) - 1;
        break;
      case BcOp::SET_ATTR:
        setattr_slots_.insert(in.x);
        continue;
      default:
        continue;
    }
    if (in.x < 1)
      return fail(static_cast<int64_t>(i), "invalid register " + std::to_string(in.x));
    if (last > max_ref) {
      max_ref = last;
      max_pc = static_cast<int64_t>(i);
    }
  }
  if (max_ref > f_.register_size)
    return fail(max_pc, "register " + std::to_string(max_ref) +
                            " exceeds register_size " + std::to_string(f_.register_size));
  regs_.assign(static_cast<size_t>(max_ref) + 1, Sym{});
  stamp_.assign(static_cast<size_t>(max_ref) + 1, 0);
  return true;
}

Result<BuildResult> Executor::run() {
  if (m_.version < 6)
    return err("bytecode v" + std::to_string(m_.version) +
                   " does not record operator arity (added in v6)",
               f_.pos);
  if (!f_.has_schema)
    return err("method " + f_.qualified_name + " has no schema", f_.pos);
  auto ps = prescan();
  if (!ps) return ps.error();

  root_path_ = intern_str("");
  model_.graphs.emplace_back();
  builders_.push_back(
      std::make_unique<GraphBuilder>(model_, names_, f_.qualified_name, 0, nullptr));
  GraphBuilder& main = B(0);

  // Initial stack: one entry per schema argument (the prologue STOREN pops them).
  const ts::ModuleEnv& env = ctx_.env;
  for (size_t i = 0; i < f_.arguments.size(); ++i) {
    const BcArg& a = f_.arguments[i];
    if (i == 0 && a.name == "self") {
      if (env.root() && a.type == env.root_class()) {
        Sym s;
        s.kind = Sym::Kind::Module;
        s.v = env.root();
        s.path = root_path_;
        auto pu = push(s, 0);
        if (!pu) return pu.error();
        continue;
      }
      if (!env.loaded())
        result_.self_note = "self not bound: no data.pkl";
      else if (!env.root())
        result_.self_note = "self not bound: data.pkl root is not a TorchScript object";
      else
        result_.self_note = "self not bound: schema type " + a.type +
                            " != data.pkl root " + env.root_class();
    }
    auto pu = push(edge_sym(0, main.add_input(a.name)), 0);
    if (!pu) return pu.error();
  }

  auto r = exec_block(main, 0, static_cast<int64_t>(ins_.size()), 0);
  if (!r) return r.error();
  if (!returned_) return err("method " + f_.qualified_name + ": missing RET", f_.pos);
  for (const PendRec& p : pend_)
    if (!p.done) return internal("unresolved pending output");

  for (size_t id = 0; id < builders_.size(); ++id)
    model_.graphs[id] = builders_[id]->finish();
  result_.main_graph = 0;
  return std::move(result_);
}

}  // namespace

Result<BuildResult> build_method_graph(const BcModule& m, size_t fn, BuildContext& ctx) {
  if (fn >= m.functions.size()) return err("method index out of range", UINT64_MAX);
  // The main graph must land at index 0; never touch a caller's existing graphs.
  if (!ctx.model.graphs.empty())
    return err("internal: model.graphs must be empty", UINT64_MAX);
  Result<BuildResult> r = [&]() -> Result<BuildResult> {
    Executor ex(m, m.functions[fn], ctx);
    return ex.run();
  }();
  if (!r) ctx.model.graphs.clear();  // nothing partial is ever published
  return r;
}

}  // namespace netvis::pytorch::mobile
