// core/SafeMath.h — saturating/checked integer arithmetic, single source of truth.
//
// DECISION (v0.6.0 §2.0): the built-in FLOP path clamps at UINT64_MAX (never
// overflow to a wrong small number); the declarative-plugin DSL evaluates in
// SIGNED checked int64 (it has subtraction; the built-ins never do). Both the
// built-in handlers and the plugin backends MUST route through THESE primitives
// so their arithmetic is bit-identical. Previously these lived as TU-local
// statics in CostModel.cpp; promoted here so plugins can share them. Pure leaf:
// <cstdint> only, no other NetVis header.
#pragma once

#include <cstdint>

namespace netvis {

// --- unsigned saturating (the built-in FLOP path; clamp at UINT64_MAX) --------
inline uint64_t safe_add(uint64_t a, uint64_t b) {
  if (UINT64_MAX - a < b) return UINT64_MAX;
  return a + b;
}
inline uint64_t safe_mul(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0) return 0;
  if (UINT64_MAX / a < b) return UINT64_MAX;
  return a * b;
}

// --- signed checked int64 (the DSL domain) ------------------------------------
// Return false on overflow/UB; the DSL treats any false here as honest-unknown.
// Guards INT64_MIN/-1 (div/rem overflow UB) and add/sub/mul overflow. Defined in
// core/SafeMath.cpp (portable impl; no compiler builtins baked into the header,
// so MSVC is not forced onto __builtin_*_overflow — invariant 2).
bool checked_add_i64(int64_t a, int64_t b, int64_t* out);
bool checked_sub_i64(int64_t a, int64_t b, int64_t* out);
bool checked_mul_i64(int64_t a, int64_t b, int64_t* out);
bool checked_div_i64(int64_t a, int64_t b, int64_t* out);   // false if b==0 || (a==INT64_MIN && b==-1)
bool checked_mod_i64(int64_t a, int64_t b, int64_t* out);   // same guard as div

}  // namespace netvis
