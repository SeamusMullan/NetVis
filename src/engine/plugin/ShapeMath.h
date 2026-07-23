// engine/plugin/ShapeMath.h — pure shape arithmetic shared by built-in handlers
// and plugin backends (v0.6.0 §2.0). Mirrors CostModel.cpp's elem_count_from_shape
// / partial_shape_product so plugin FLOPs are bit-identical to the built-ins, and
// hardens the partial-product bound against uint32 wrap. Pure leaf: SmallVec +
// SafeMath only.
#pragma once

#include <cstdint>

#include "core/SafeMath.h"
#include "core/SmallVec.h"

namespace netvis::plugin {

using Shape = SmallVec<int64_t, 6>;

// Product of dims; 0 if any dim < 1 (== CostModel.cpp elem_count_from_shape:
// a dynamic/unset dim makes the element count honestly unknown).
inline uint64_t elem_count(const Shape& s) {
  if (s.empty()) return 0;
  uint64_t n = 1;
  for (int64_t d : s) {
    if (d < 1) return 0;
    n = safe_mul(n, static_cast<uint64_t>(d));
  }
  return n;
}

// Product of dims [off, off+count); 0 if any dim < 1. Bound is wrap-proof: NO
// addition that can overflow uint32 (the CostModel.cpp form `off+count > size`
// wraps when count is computed as size-2 on a <2-length shape).
inline uint64_t partial_product(const Shape& s, uint32_t off, uint32_t count) {
  const uint32_t n = static_cast<uint32_t>(s.size());
  if (count > n || off > n - count) return 0;   // no wrapping add
  uint64_t p = 1;
  for (uint32_t i = 0; i < count; ++i) {
    int64_t d = s[off + i];
    if (d < 1) return 0;
    p = safe_mul(p, static_cast<uint64_t>(d));
  }
  return p;
}

}  // namespace netvis::plugin
