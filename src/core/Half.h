// SPDX-License-Identifier: Apache-2.0
// core/Half.h — IEEE half (F16), bfloat16 and OCP microscaling (MX) minifloat
// bit-pattern -> float.
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

#include <cmath>
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

// FP4 E2M1 (OCP MX spec, the element type of both MXFP4 and NVFP4) -> float.
// Only the low 4 bits of `nibble` are used: 1 sign, 2 exponent (bias 1), 1
// mantissa. No inf/NaN encodings exist, so all 16 codes are finite:
//   0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (0x8 is -0).
inline float fp4_e2m1_to_f32(uint8_t nibble) {
  static constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = kMag[nibble & 0x7];
  return (nibble & 0x8) ? -m : m;
}

// E8M0 (OCP MX shared block scale, used by MXFP4) -> float. An unsigned,
// exponent-only power of two: 2^(e - 127). 0xFF is the one NaN encoding.
// e == 0 is 2^-127, which is subnormal in float and built by hand.
inline float e8m0_to_f32(uint8_t e) {
  if (e == 0xFF) return std::nanf("");
  const uint32_t bits = (e == 0) ? 0x00400000u : (static_cast<uint32_t>(e) << 23);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// FP8 E4M3FN (OCP / NVIDIA "fn" variant) -> float: 1 sign, 4 exponent (bias 7),
// 3 mantissa, no infinities, and S.1111.111 is the only NaN. NVFP4 stores its
// per-16-element block scales in this format (sign bit expected clear). Max
// finite magnitude is 448.
inline float fp8_e4m3fn_to_f32(uint8_t b) {
  const bool neg = (b & 0x80) != 0;
  const int exp = (b >> 3) & 0xF;
  const int man = b & 0x7;
  if (exp == 0xF && man == 0x7) return std::nanf("");
  const float mag = (exp == 0)
      ? std::ldexp(static_cast<float>(man), -9)               // subnormal: m/8 * 2^-6
      : std::ldexp(1.0f + static_cast<float>(man) / 8.0f, exp - 7);
  return neg ? -mag : mag;
}

}  // namespace netvis
