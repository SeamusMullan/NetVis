// core/Half.h — IEEE half (F16) and bfloat16 bit-pattern -> float.
//
// Lives in core/ because two layers need it and neither may include the other:
// engine/TensorStats.cpp decodes F16/BF16 tensor payloads, and
// parsers/gguf/GgufBlocks.cpp decodes the F16 scale at the head of every legacy
// ggml quant block (#49). Parsers sit BELOW engine, so the engine copy was not
// reachable — hoisting is the fix, rather than a second hand-rolled decoder that
// can drift from this one.
//
// Pure, constexpr-friendly bit manipulation: no lookup table, no FP hardware
// assumption, no locale, no allocation. Handles subnormals, inf and NaN.
#pragma once

#include <cstdint>
#include <cstring>

namespace netvis {

// F16 (IEEE half) bit pattern -> float.
inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +/- zero
    } else {
      // subnormal: normalize
      exp = 1;
      while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
      mant &= 0x3FF;
      bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf/nan
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// BF16 (upper 16 bits of a float) -> float.
inline float bf16_to_f32(uint16_t b) {
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

}  // namespace netvis
