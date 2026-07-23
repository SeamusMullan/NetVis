// engine/plugin/declarative/Dsl.h — the declarative-plugin expression DSL
// (v0.6.0 Increment 2, issue #9).
//
// A pure, total, integer expression language over a node's resolved shapes + attrs.
// It compiles a source string to an AST ONCE at load; evaluation per node walks the
// AST with a variable environment. SAFETY BY CONSTRUCTION:
//   - NO file/memory/loop/recursion/byte-access primitive -> a malicious manifest
//     can at worst fail to parse or yield an honest-unknown; zero-payload upheld by
//     the vocabulary (there is no term that reads a tensor).
//   - All arithmetic is CHECKED signed int64 (core/SafeMath.h): overflow / div-0 /
//     any negative feeding a FLOP or element count -> Val{known=false}. A formula
//     can never fabricate a wrong small number or crash.
//   - Bounded parse depth + token/length caps -> pathological nesting is rejected,
//     never a stack overflow.
//
// Grammar (EBNF), lowest-to-highest precedence:
//   expr    := ternary
//   ternary := logic ( "?" expr ":" expr )?
//   logic   := cmp ( ("&&"|"||") cmp )*
//   cmp     := add ( ("=="|"!="|"<"|"<="|">"|">=") add )*
//   add     := mul ( ("+"|"-") mul )*
//   mul     := unary ( ("*"|"/"|"%") unary )*
//   unary   := ("-"|"!")? postfix
//   postfix := primary ( ".shape" "[" expr "]" | ".rank" )?
//   primary := INT | IDENT | "unknown" | func "(" args ")" | inK | outK | "O"
//            | "nin" | "nout" | "(" expr ")"
//            | "attr" "(" STRING ("," expr)? ")"
//            | "sattr" "(" STRING ")" ("=="|"!=") STRING     (string predicate)
//   func    := "min" | "max" | "prod"   (prod supports a slice: prod(inK.shape[a:b]))
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "engine/plugin/OpHandler.h"   // OpContext

namespace netvis::plugin::dsl {

// A DSL value: an int64 that may be "unknown" (unresolved shape/attr, overflow,
// div-0, or a negative feeding a nonnegative-required position).
struct Val {
  int64_t v = 0;
  bool known = false;
  static Val of(int64_t x) { return {x, true}; }
  static Val unknown() { return {}; }
};

// Compile limits (fatal-to-op, never a crash). Mirrors design Q2.
struct Limits {
  int max_depth = 64;
  int max_tokens = 4096;
  size_t max_source = 16 * 1024;
};

// A compiled expression. Cheap to copy (shared AST). Evaluate against an OpContext
// (for shape/attr reads) plus a variable environment (the [op.vars] bindings, and
// intrinsics O/nin/nout). Never throws.
class Expr {
 public:
  // Parse `src`. On any syntax error / limit breach, returns a null Expr and sets
  // *err (if non-null) to a human message. A null Expr always evaluates to unknown.
  static Expr compile(std::string_view src, const Limits& lim, std::string* err);

  bool valid() const { return root_ != nullptr; }

  // Evaluate. `vars` maps a variable name to its already-evaluated Val (the loader
  // evaluates [op.vars] in dependency order and passes them in). Unknown-contagious
  // except through short-circuited ?: / && / ||.
  Val eval(const OpContext& ctx,
           const std::vector<std::pair<std::string, Val>>& vars) const;

  // The set of identifiers this expression references (for the loader's var-DAG
  // cycle check + evaluation ordering). Deduplicated.
  std::vector<std::string> referenced_vars() const;

  // AST node — public so the (translation-unit-local) parser/evaluator can build
  // and walk it; opaque to every other TU (they only ever hold an Expr).
  struct Node;

 private:
  std::shared_ptr<const Node> root_;
};

}  // namespace netvis::plugin::dsl
