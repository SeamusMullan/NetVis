// engine/plugin/declarative/Dsl.cpp — recursive-descent parser + checked evaluator
// for the declarative expression DSL (v0.6.0 #9). See Dsl.h for the grammar.
#include "engine/plugin/declarative/Dsl.h"

#include <cctype>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/SafeMath.h"
#include "engine/plugin/ShapeMath.h"

namespace netvis::plugin::dsl {

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------
enum class Op : uint8_t {
  Int, Ident, Unknown,
  Neg, Not,
  Add, Sub, Mul, Div, Mod,
  Eq, Ne, Lt, Le, Gt, Ge,
  And, Or,
  Ternary,
  Min, Max, Prod,        // funcs
  ProdSlice,             // prod(inK.shape[a:b])
  ShapeIdx,              // operand.shape[expr]
  Rank,                  // operand.rank
  OperandRef,            // in0/out0/... resolved to (is_input, slot)
  IntrinsicO, IntrinsicNin, IntrinsicNout,
  AttrInt,               // attr("name") or attr("name", default)
  SAttrEq, SAttrNe,      // sattr("name") == "literal"
};

struct Expr::Node {
  Op op;
  int64_t ival = 0;                       // Int
  std::string sval;                       // Ident / attr name / string literal
  std::string sval2;                      // second string (SAttr literal)
  bool is_input = true;                   // OperandRef / ShapeIdx / Rank / ProdSlice
  uint32_t slot = 0;
  bool has_default = false;               // AttrInt with default
  std::vector<std::shared_ptr<Node>> kids;
};

namespace {

// Parser builds MUTABLE nodes; kids/root store them as const (mutable->const is a
// valid shared_ptr conversion). Evaluator only ever reads.
using NodePtr = std::shared_ptr<Expr::Node>;

// ---- Tokenizer ----
enum class Tk : uint8_t {
  End, Int, Ident, Str,
  Plus, Minus, Star, Slash, Pct,
  Eq, Ne, Lt, Le, Gt, Ge,
  AndAnd, OrOr, Bang,
  Question, Colon, Comma, Dot, LBrack, RBrack, LParen, RParen,
};

struct Token {
  Tk kind = Tk::End;
  int64_t ival = 0;
  std::string text;
  Token() = default;
  Token(Tk k) : kind(k) {}
  Token(Tk k, int64_t v, std::string t) : kind(k), ival(v), text(std::move(t)) {}
};

struct Lexer {
  std::string_view s;
  size_t i = 0;
  std::string* err;
  bool ok = true;

  void fail(const char* m) { if (ok && err) *err = m; ok = false; }

  void skip_ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  }

  Token next() {
    skip_ws();
    if (i >= s.size()) return {Tk::End};
    char c = s[i];
    // number
    if (std::isdigit(static_cast<unsigned char>(c))) {
      int64_t v = 0;
      bool overflow = false;
      while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        int d = s[i] - '0';
        if (!checked_mul_i64(v, 10, &v) || !checked_add_i64(v, d, &v)) overflow = true;
        ++i;
      }
      if (overflow) { fail("integer literal too large"); }
      return {Tk::Int, v, {}};
    }
    // identifier / keyword
    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
      size_t start = i;
      while (i < s.size() &&
             (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_'))
        ++i;
      return {Tk::Ident, 0, std::string(s.substr(start, i - start))};
    }
    // string literal "..."
    if (c == '"') {
      ++i;
      std::string out;
      while (i < s.size() && s[i] != '"') {
        if (out.size() > 256) { fail("string literal too long"); break; }
        out.push_back(s[i++]);
      }
      if (i >= s.size()) { fail("unterminated string"); return {Tk::End}; }
      ++i;  // closing quote
      return {Tk::Str, 0, out};
    }
    // operators
    auto two = [&](char a, char b) { return c == a && i + 1 < s.size() && s[i + 1] == b; };
    if (two('&', '&')) { i += 2; return {Tk::AndAnd}; }
    if (two('|', '|')) { i += 2; return {Tk::OrOr}; }
    if (two('=', '=')) { i += 2; return {Tk::Eq}; }
    if (two('!', '=')) { i += 2; return {Tk::Ne}; }
    if (two('<', '=')) { i += 2; return {Tk::Le}; }
    if (two('>', '=')) { i += 2; return {Tk::Ge}; }
    ++i;
    switch (c) {
      case '+': return {Tk::Plus};
      case '-': return {Tk::Minus};
      case '*': return {Tk::Star};
      case '/': return {Tk::Slash};
      case '%': return {Tk::Pct};
      case '<': return {Tk::Lt};
      case '>': return {Tk::Gt};
      case '!': return {Tk::Bang};
      case '?': return {Tk::Question};
      case ':': return {Tk::Colon};
      case ',': return {Tk::Comma};
      case '.': return {Tk::Dot};
      case '[': return {Tk::LBrack};
      case ']': return {Tk::RBrack};
      case '(': return {Tk::LParen};
      case ')': return {Tk::RParen};
    }
    fail("unexpected character");
    return {Tk::End};
  }
};

// ---- Parser (recursive descent) ----
struct Parser {
  std::vector<Token> toks;
  size_t p = 0;
  int depth = 0;
  const Limits& lim;
  std::string* err;
  bool ok = true;

  Parser(const Limits& l, std::string* e) : lim(l), err(e) {}

  void fail(const char* m) { if (ok && err) *err = m; ok = false; }
  const Token& peek() const { return toks[p]; }
  Tk kind() const { return toks[p].kind; }
  const Token& advance() { return toks[p < toks.size() - 1 ? p++ : p]; }
  bool accept(Tk k) { if (kind() == k) { advance(); return true; } return false; }
  void expect(Tk k, const char* m) { if (!accept(k)) fail(m); }

  struct Depth {
    Parser& pr; bool ok;
    Depth(Parser& p) : pr(p) { ok = (++pr.depth <= pr.lim.max_depth); if (!ok) pr.fail("expression too deeply nested"); }
    ~Depth() { --pr.depth; }
  };

  NodePtr mk(Op op) { auto n = std::make_shared<Expr::Node>(); n->op = op; return n; }

  NodePtr parse_expr() {
    Depth d(*this); if (!d.ok || !ok) return nullptr;
    NodePtr cond = parse_logic();
    if (!ok) return nullptr;
    if (accept(Tk::Question)) {
      auto n = std::make_shared<Expr::Node>();
      n->op = Op::Ternary;
      NodePtr a = parse_expr();
      expect(Tk::Colon, "expected ':' in ternary");
      NodePtr b = parse_expr();
      n->kids = {cond, a, b};
      return n;
    }
    return cond;
  }

  NodePtr parse_logic() {
    NodePtr lhs = parse_cmp();
    while (ok && (kind() == Tk::AndAnd || kind() == Tk::OrOr)) {
      Op op = kind() == Tk::AndAnd ? Op::And : Op::Or;
      advance();
      NodePtr rhs = parse_cmp();
      auto n = std::make_shared<Expr::Node>(); n->op = op; n->kids = {lhs, rhs};
      lhs = n;
    }
    return lhs;
  }

  NodePtr parse_cmp() {
    NodePtr lhs = parse_add();
    while (ok) {
      Op op;
      switch (kind()) {
        case Tk::Eq: op = Op::Eq; break;
        case Tk::Ne: op = Op::Ne; break;
        case Tk::Lt: op = Op::Lt; break;
        case Tk::Le: op = Op::Le; break;
        case Tk::Gt: op = Op::Gt; break;
        case Tk::Ge: op = Op::Ge; break;
        default: return lhs;
      }
      advance();
      NodePtr rhs = parse_add();
      auto n = std::make_shared<Expr::Node>(); n->op = op; n->kids = {lhs, rhs};
      lhs = n;
    }
    return lhs;
  }

  NodePtr parse_add() {
    NodePtr lhs = parse_mul();
    while (ok && (kind() == Tk::Plus || kind() == Tk::Minus)) {
      Op op = kind() == Tk::Plus ? Op::Add : Op::Sub;
      advance();
      NodePtr rhs = parse_mul();
      auto n = std::make_shared<Expr::Node>(); n->op = op; n->kids = {lhs, rhs};
      lhs = n;
    }
    return lhs;
  }

  NodePtr parse_mul() {
    NodePtr lhs = parse_unary();
    while (ok && (kind() == Tk::Star || kind() == Tk::Slash || kind() == Tk::Pct)) {
      Op op = kind() == Tk::Star ? Op::Mul : (kind() == Tk::Slash ? Op::Div : Op::Mod);
      advance();
      NodePtr rhs = parse_unary();
      auto n = std::make_shared<Expr::Node>(); n->op = op; n->kids = {lhs, rhs};
      lhs = n;
    }
    return lhs;
  }

  NodePtr parse_unary() {
    // Depth-guard the unary chain: `-`/`!` recurse into parse_unary directly, so
    // without this a token stream like `!!!!...`(x4096) would recurse past the
    // native stack, bypassing the max_depth bound the other productions enforce.
    Depth d(*this); if (!d.ok || !ok) return nullptr;
    if (accept(Tk::Minus)) {
      auto n = mk(Op::Neg); n->kids = {parse_unary()}; return n;
    }
    if (accept(Tk::Bang)) {
      auto n = mk(Op::Not); n->kids = {parse_unary()}; return n;
    }
    return parse_postfix();
  }

  // Parse the digit tail of an operand token (e.g. "42" in "in42") into a slot.
  // Bounded + NON-throwing (std::stoul throws out_of_range on a huge literal and
  // wraps to uint32 — a hostile "in4294967296" would crash the app or alias slot 0).
  // Every char must be a digit; the value must fit under kMaxSlot; else NOT an
  // operand (returns false → the token falls through to a plain variable name).
  static bool parse_slot(std::string_view digits, uint32_t& out) {
    static constexpr uint32_t kMaxSlot = 4096;   // far beyond any real op arity
    if (digits.empty()) return false;
    uint32_t v = 0;
    for (char c : digits) {
      if (c < '0' || c > '9') return false;
      v = v * 10 + static_cast<uint32_t>(c - '0');
      if (v > kMaxSlot) return false;            // overflow-proof: reject early
    }
    out = v;
    return true;
  }

  // operand token: inK / outK -> (is_input, slot). Returns false if not an operand.
  bool operand_of(const std::string& id, bool& is_input, uint32_t& slot) {
    if (id.size() >= 3 && id.compare(0, 2, "in") == 0 &&
        std::isdigit(static_cast<unsigned char>(id[2]))) {
      if (!parse_slot(std::string_view(id).substr(2), slot)) return false;
      is_input = true; return true;
    }
    if (id.size() >= 4 && id.compare(0, 3, "out") == 0 &&
        std::isdigit(static_cast<unsigned char>(id[3]))) {
      if (!parse_slot(std::string_view(id).substr(3), slot)) return false;
      is_input = false; return true;
    }
    return false;
  }

  NodePtr parse_postfix() {
    Depth d(*this); if (!d.ok || !ok) return nullptr;
    NodePtr base = parse_primary();
    if (!ok) return base;
    // .shape[expr] / .shape[a:b] / .rank apply to an operand
    while (ok && kind() == Tk::Dot) {
      advance();
      if (kind() != Tk::Ident) { fail("expected 'shape' or 'rank' after '.'"); break; }
      std::string field = advance().text;
      if (base->op != Op::OperandRef) { fail("'.shape'/'.rank' only apply to inK/outK"); break; }
      if (field == "rank") {
        auto n = mk(Op::Rank); n->is_input = base->is_input; n->slot = base->slot;
        base = n;
      } else if (field == "shape") {
        expect(Tk::LBrack, "expected '[' after .shape");
        // slice a:b?  peek for a ':' — support only literal-or-expr : expr in prod().
        NodePtr idx = parse_expr();
        if (accept(Tk::Colon)) {
          NodePtr hi = parse_expr();
          expect(Tk::RBrack, "expected ']'");
          auto n = mk(Op::ProdSlice);   // a bare slice is only meaningful inside prod();
          n->is_input = base->is_input; n->slot = base->slot; n->kids = {idx, hi};
          base = n;
        } else {
          expect(Tk::RBrack, "expected ']'");
          auto n = mk(Op::ShapeIdx);
          n->is_input = base->is_input; n->slot = base->slot; n->kids = {idx};
          base = n;
        }
      } else {
        fail("unknown field after '.'"); break;
      }
    }
    return base;
  }

  NodePtr parse_primary() {
    Depth d(*this); if (!d.ok || !ok) return nullptr;
    switch (kind()) {
      case Tk::Int: { auto n = mk(Op::Int); n->ival = advance().ival; return n; }
      case Tk::LParen: {
        advance();
        NodePtr e = parse_expr();
        expect(Tk::RParen, "expected ')'");
        return e;
      }
      case Tk::Ident: {
        std::string id = advance().text;
        // function call?
        if (kind() == Tk::LParen && (id == "min" || id == "max" || id == "prod")) {
          advance();
          std::vector<NodePtr> args;
          if (kind() != Tk::RParen) {
            args.push_back(parse_expr());
            while (accept(Tk::Comma)) args.push_back(parse_expr());
          }
          expect(Tk::RParen, "expected ')' after function args");
          Op op = id == "min" ? Op::Min : (id == "max" ? Op::Max : Op::Prod);
          // prod(inK.shape[a:b]) collapses to ProdSlice
          if (op == Op::Prod && args.size() == 1 && args[0] && args[0]->op == Op::ProdSlice)
            return args[0];
          if ((op == Op::Min || op == Op::Max) && args.size() < 2)
            fail("min/max need >= 2 args");
          if (args.size() > 256) fail("too many function args");
          auto n = mk(op); n->kids = std::move(args); return n;
        }
        // attr("name") / attr("name", default)
        if (kind() == Tk::LParen && id == "attr") {
          advance();
          if (kind() != Tk::Str) { fail("attr() needs a string name"); return nullptr; }
          auto n = mk(Op::AttrInt); n->sval = advance().text;
          if (accept(Tk::Comma)) { n->has_default = true; n->kids = {parse_expr()}; }
          expect(Tk::RParen, "expected ')'");
          return n;
        }
        // sattr("name") == "lit" | != "lit"
        if (kind() == Tk::LParen && id == "sattr") {
          advance();
          if (kind() != Tk::Str) { fail("sattr() needs a string name"); return nullptr; }
          std::string aname = advance().text;
          expect(Tk::RParen, "expected ')'");
          Op op;
          if (accept(Tk::Eq)) op = Op::SAttrEq;
          else if (accept(Tk::Ne)) op = Op::SAttrNe;
          else { fail("sattr(...) must be compared with == or != to a string"); return nullptr; }
          if (kind() != Tk::Str) { fail("sattr comparison needs a string literal"); return nullptr; }
          auto n = mk(op); n->sval = aname; n->sval2 = advance().text; return n;
        }
        // intrinsics / operands / vars
        if (id == "unknown") return mk(Op::Unknown);
        if (id == "O") return mk(Op::IntrinsicO);
        if (id == "nin") return mk(Op::IntrinsicNin);
        if (id == "nout") return mk(Op::IntrinsicNout);
        bool is_input; uint32_t slot;
        if (operand_of(id, is_input, slot)) {
          auto n = mk(Op::OperandRef); n->is_input = is_input; n->slot = slot; return n;
        }
        // a bound variable
        auto n = mk(Op::Ident); n->sval = id; return n;
      }
      default:
        fail("unexpected token");
        return nullptr;
    }
  }
};

// ---- Evaluator ----
struct Evaluator {
  const OpContext& ctx;
  const std::vector<std::pair<std::string, Val>>& vars;

  const Shape* operand_shape(bool is_input, uint32_t slot) const {
    return is_input ? ctx.input_shape(slot) : ctx.output_shape(slot);
  }

  Val lookup_var(const std::string& name) const {
    for (const auto& kv : vars) if (kv.first == name) return kv.second;
    return Val::unknown();
  }

  // Narrow a uint64 elem count to int64 with saturating clamp (never wraps).
  static Val from_u64(uint64_t x) {
    if (x == 0) return Val::of(0);
    if (x > static_cast<uint64_t>(INT64_MAX)) return Val::of(INT64_MAX);
    return Val::of(static_cast<int64_t>(x));
  }

  Val ev(const Expr::Node* n) const {
    if (!n) return Val::unknown();
    switch (n->op) {
      case Op::Int: return Val::of(n->ival);
      case Op::Unknown: return Val::unknown();
      case Op::Ident: return lookup_var(n->sval);
      case Op::IntrinsicNin: return Val::of(static_cast<int64_t>(ctx.input_count()));
      case Op::IntrinsicNout: return Val::of(static_cast<int64_t>(ctx.output_count()));
      case Op::IntrinsicO: {
        const Shape* s = ctx.output_shape(0);
        if (!s) return Val::unknown();
        uint64_t e = elem_count(*s);          // 0 if any dim < 1
        if (e == 0) return Val::unknown();
        return from_u64(e);
      }
      case Op::OperandRef: {
        // A bare operand ref outside .shape/.rank is meaningless; treat as unknown.
        return Val::unknown();
      }
      case Op::Rank: {
        const Shape* s = operand_shape(n->is_input, n->slot);
        if (!s || s->empty()) return Val::unknown();
        return Val::of(static_cast<int64_t>(s->size()));
      }
      case Op::ShapeIdx: {
        const Shape* s = operand_shape(n->is_input, n->slot);
        if (!s || s->empty()) return Val::unknown();
        Val idx = ev(n->kids[0].get());
        if (!idx.known) return Val::unknown();
        int64_t i = idx.v;
        int64_t rank = static_cast<int64_t>(s->size());
        if (i < 0) i += rank;             // negative-from-end
        if (i < 0 || i >= rank) return Val::unknown();
        int64_t d = (*s)[static_cast<size_t>(i)];
        if (d < 1) return Val::unknown(); // dynamic/-1 dim -> honest-unknown
        return Val::of(d);
      }
      case Op::ProdSlice: {
        const Shape* s = operand_shape(n->is_input, n->slot);
        if (!s) return Val::unknown();
        Val a = ev(n->kids[0].get()), b = ev(n->kids[1].get());
        if (!a.known || !b.known) return Val::unknown();
        int64_t rank = static_cast<int64_t>(s->size());
        int64_t lo = a.v < 0 ? a.v + rank : a.v;
        int64_t hi = b.v < 0 ? b.v + rank : b.v;
        if (lo < 0) lo = 0;
        if (hi > rank) hi = rank;
        if (lo >= hi) return Val::unknown();
        uint64_t p = 1;
        for (int64_t i = lo; i < hi; ++i) {
          int64_t d = (*s)[static_cast<size_t>(i)];
          if (d < 1) return Val::unknown();
          p = safe_mul(p, static_cast<uint64_t>(d));
        }
        return from_u64(p);
      }
      case Op::AttrInt: {
        if (n->has_default) {
          Val dv = ev(n->kids[0].get());
          int64_t def = dv.known ? dv.v : 0;
          return Val::of(ctx.attr_int(n->sval, def));
        }
        if (!ctx.has_attr(n->sval)) return Val::unknown();
        // present but wrong-kind -> attr_int returns default; treat as 0-known only
        // when it's actually an int. has_attr is true for any kind; guard on int.
        int64_t sentinel = ctx.attr_int(n->sval, INT64_MIN);
        int64_t alt = ctx.attr_int(n->sval, INT64_MAX);
        if (sentinel != alt) return Val::unknown();  // not an Int attr
        return Val::of(sentinel);
      }
      case Op::SAttrEq:
        return Val::of(ctx.attr_string(n->sval, "\x01__absent__") == n->sval2 ? 1 : 0);
      case Op::SAttrNe:
        return Val::of(ctx.attr_string(n->sval, "\x01__absent__") != n->sval2 ? 1 : 0);
      case Op::Neg: {
        Val a = ev(n->kids[0].get());
        if (!a.known) return Val::unknown();
        int64_t r;
        if (!checked_sub_i64(0, a.v, &r)) return Val::unknown();
        return Val::of(r);
      }
      case Op::Not: {
        Val a = ev(n->kids[0].get());
        if (!a.known) return Val::unknown();
        return Val::of(a.v == 0 ? 1 : 0);
      }
      case Op::And: {
        Val a = ev(n->kids[0].get());
        if (a.known && a.v == 0) return Val::of(0);   // short-circuit
        Val b = ev(n->kids[1].get());
        if (!a.known || !b.known) return Val::unknown();
        return Val::of((a.v != 0 && b.v != 0) ? 1 : 0);
      }
      case Op::Or: {
        Val a = ev(n->kids[0].get());
        if (a.known && a.v != 0) return Val::of(1);   // short-circuit
        Val b = ev(n->kids[1].get());
        if (!a.known || !b.known) return Val::unknown();
        return Val::of((a.v != 0 || b.v != 0) ? 1 : 0);
      }
      case Op::Ternary: {
        Val c = ev(n->kids[0].get());
        if (!c.known) return Val::unknown();
        return ev(n->kids[c.v != 0 ? 1 : 2].get());
      }
      case Op::Min: case Op::Max: {
        Val acc = ev(n->kids[0].get());
        if (!acc.known) return Val::unknown();
        for (size_t k = 1; k < n->kids.size(); ++k) {
          Val x = ev(n->kids[k].get());
          if (!x.known) return Val::unknown();
          if (n->op == Op::Min) acc.v = x.v < acc.v ? x.v : acc.v;
          else acc.v = x.v > acc.v ? x.v : acc.v;
        }
        return acc;
      }
      case Op::Prod: {
        int64_t acc = 1;
        for (const auto& k : n->kids) {
          Val x = ev(k.get());
          if (!x.known) return Val::unknown();
          if (!checked_mul_i64(acc, x.v, &acc)) return Val::unknown();
        }
        return Val::of(acc);
      }
      default: break;   // binary arithmetic + comparisons below
    }
    // binary arithmetic / comparison
    Val a = ev(n->kids[0].get());
    Val b = ev(n->kids[1].get());
    if (!a.known || !b.known) return Val::unknown();
    int64_t r = 0;
    switch (n->op) {
      case Op::Add: return checked_add_i64(a.v, b.v, &r) ? Val::of(r) : Val::unknown();
      case Op::Sub: return checked_sub_i64(a.v, b.v, &r) ? Val::of(r) : Val::unknown();
      case Op::Mul: return checked_mul_i64(a.v, b.v, &r) ? Val::of(r) : Val::unknown();
      case Op::Div: return checked_div_i64(a.v, b.v, &r) ? Val::of(r) : Val::unknown();
      case Op::Mod: return checked_mod_i64(a.v, b.v, &r) ? Val::of(r) : Val::unknown();
      case Op::Eq: return Val::of(a.v == b.v ? 1 : 0);
      case Op::Ne: return Val::of(a.v != b.v ? 1 : 0);
      case Op::Lt: return Val::of(a.v < b.v ? 1 : 0);
      case Op::Le: return Val::of(a.v <= b.v ? 1 : 0);
      case Op::Gt: return Val::of(a.v > b.v ? 1 : 0);
      case Op::Ge: return Val::of(a.v >= b.v ? 1 : 0);
      default: return Val::unknown();
    }
  }
};

void collect_vars(const Expr::Node* n, std::unordered_set<std::string>& out) {
  if (!n) return;
  if (n->op == Op::Ident) out.insert(n->sval);
  for (const auto& k : n->kids) collect_vars(k.get(), out);
}

}  // namespace

Expr Expr::compile(std::string_view src, const Limits& lim, std::string* err) {
  Expr e;
  if (src.size() > lim.max_source) { if (err) *err = "expression source too long"; return e; }
  Lexer lx{src, 0, err, true};
  std::vector<Token> toks;
  for (;;) {
    Token t = lx.next();
    if (!lx.ok) return e;
    if (static_cast<int>(toks.size()) > lim.max_tokens) { if (err) *err = "too many tokens"; return e; }
    bool end = (t.kind == Tk::End);
    toks.push_back(std::move(t));
    if (end) break;
  }
  Parser pr(lim, err);
  pr.toks = std::move(toks);
  NodePtr root = pr.parse_expr();
  if (!pr.ok) return e;
  if (pr.kind() != Tk::End) { if (err) *err = "trailing tokens after expression"; return e; }
  e.root_ = root;
  return e;
}

Val Expr::eval(const OpContext& ctx,
               const std::vector<std::pair<std::string, Val>>& vars) const {
  if (!root_) return Val::unknown();
  Evaluator ev{ctx, vars};
  return ev.ev(root_.get());
}

std::vector<std::string> Expr::referenced_vars() const {
  std::unordered_set<std::string> set;
  collect_vars(root_.get(), set);
  return std::vector<std::string>(set.begin(), set.end());
}

}  // namespace netvis::plugin::dsl
