// core/SafeMath.cpp — portable checked signed-int64 arithmetic (v0.6.0).
// Header-declared; defined here so no compiler builtin is baked into the header
// (MSVC has no __builtin_*_overflow). The DSL treats any `false` as honest-unknown.
#include "core/SafeMath.h"

#include <cstdint>

namespace netvis {

bool checked_add_i64(int64_t a, int64_t b, int64_t* out) {
  // Overflow iff signs equal and the sum's sign flips. Compute in uint64 (defined
  // wraparound) then check the sign bits — no UB, no compiler builtin needed.
  uint64_t ua = static_cast<uint64_t>(a);
  uint64_t ub = static_cast<uint64_t>(b);
  uint64_t us = ua + ub;
  int64_t s = static_cast<int64_t>(us);
  if (((a ^ s) & (b ^ s)) < 0) return false;  // both operands differ in sign from sum
  *out = s;
  return true;
}

bool checked_sub_i64(int64_t a, int64_t b, int64_t* out) {
  uint64_t ua = static_cast<uint64_t>(a);
  uint64_t ub = static_cast<uint64_t>(b);
  uint64_t ud = ua - ub;
  int64_t d = static_cast<int64_t>(ud);
  if (((a ^ b) & (a ^ d)) < 0) return false;  // operands differ in sign AND result sign flips from a
  *out = d;
  return true;
}

bool checked_mul_i64(int64_t a, int64_t b, int64_t* out) {
  if (a == 0 || b == 0) { *out = 0; return true; }
  int64_t r = static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
  // Verify by dividing back; guard the INT64_MIN/-1 UB case explicitly.
  if (a == -1 && b == INT64_MIN) return false;
  if (b == -1 && a == INT64_MIN) return false;
  if (r / a != b) return false;
  *out = r;
  return true;
}

bool checked_div_i64(int64_t a, int64_t b, int64_t* out) {
  if (b == 0 || (a == INT64_MIN && b == -1)) return false;
  *out = a / b;
  return true;
}

bool checked_mod_i64(int64_t a, int64_t b, int64_t* out) {
  if (b == 0 || (a == INT64_MIN && b == -1)) return false;
  *out = a % b;
  return true;
}

}  // namespace netvis
