// parsers/gguf/GgufBlocks.cpp — ggml type table + bounded single-block dequant.
//
// See GgufBlocks.h for the scope decision. Everything here is pure: a caller
// supplies the bytes and the output buffer, so this file allocates nothing,
// reads no files, and holds no state.
//
// The five supported layouts are the legacy ggml quants, all QK=32:
//
//   Q4_0  18B  { f16 d;                    u8 qs[16] }  v = (nibble - 8) * d
//   Q4_1  20B  { f16 d; f16 m;             u8 qs[16] }  v = nibble * d + m
//   Q5_0  22B  { f16 d;          u8 qh[4]; u8 qs[16] }  v = (nibble|bit5 - 16) * d
//   Q5_1  24B  { f16 d; f16 m;   u8 qh[4]; u8 qs[16] }  v = (nibble|bit5) * d + m
//   Q8_0  34B  { f16 d;                    i8 qs[32] }  v = q * d
//
// In every 4-bit layout the low nibble of qs[j] is element j and the high nibble
// is element j+16 — the halves are interleaved, NOT sequential. Getting that
// backwards produces plausible-looking but wrong numbers, which is exactly the
// failure mode the honesty rules exist to prevent, so it is asserted in tests.
#include "parsers/gguf/GgufBlocks.h"

#include <cstring>

#include "core/Half.h"

namespace netvis::gguf {

namespace {

constexpr uint32_t kQK = 32;      // elements per block, all supported layouts
constexpr uint32_t kQK_K = 256;   // elements per K-quant / IQ super-block

// Read a little-endian u16 without alignment assumptions.
uint16_t rd_u16(const uint8_t* p) {
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

// Read a little-endian u32 without alignment assumptions.
uint32_t rd_u32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

}  // namespace

std::string_view ggml_type_name(uint32_t type_id) {
  switch (static_cast<GgmlType>(type_id)) {
    case GgmlType::F32:     return "F32";
    case GgmlType::F16:     return "F16";
    case GgmlType::Q4_0:    return "Q4_0";
    case GgmlType::Q4_1:    return "Q4_1";
    case GgmlType::Q5_0:    return "Q5_0";
    case GgmlType::Q5_1:    return "Q5_1";
    case GgmlType::Q8_0:    return "Q8_0";
    case GgmlType::Q8_1:    return "Q8_1";
    case GgmlType::Q2_K:    return "Q2_K";
    case GgmlType::Q3_K:    return "Q3_K";
    case GgmlType::Q4_K:    return "Q4_K";
    case GgmlType::Q5_K:    return "Q5_K";
    case GgmlType::Q6_K:    return "Q6_K";
    case GgmlType::Q8_K:    return "Q8_K";
    case GgmlType::IQ2_XXS: return "IQ2_XXS";
    case GgmlType::IQ2_XS:  return "IQ2_XS";
    case GgmlType::IQ3_XXS: return "IQ3_XXS";
    case GgmlType::IQ1_S:   return "IQ1_S";
    case GgmlType::IQ4_NL:  return "IQ4_NL";
    case GgmlType::IQ3_S:   return "IQ3_S";
    case GgmlType::IQ2_S:   return "IQ2_S";
    case GgmlType::IQ4_XS:  return "IQ4_XS";
    case GgmlType::I8:      return "I8";
    case GgmlType::I16:     return "I16";
    case GgmlType::I32:     return "I32";
    case GgmlType::I64:     return "I64";
    case GgmlType::F64:     return "F64";
    case GgmlType::IQ1_M:   return "IQ1_M";
    case GgmlType::BF16:    return "BF16";
  }
  return {};  // an id this build does not know
}

GgmlBlockLayout ggml_block_layout(uint32_t type_id) {
  GgmlBlockLayout L;
  switch (static_cast<GgmlType>(type_id)) {
    // --- decodable legacy quants ---------------------------------------------
    case GgmlType::Q4_0: L = {kQK, 18, true, true}; break;
    case GgmlType::Q4_1: L = {kQK, 20, true, true}; break;
    case GgmlType::Q5_0: L = {kQK, 22, true, true}; break;
    case GgmlType::Q5_1: L = {kQK, 24, true, true}; break;
    case GgmlType::Q8_0: L = {kQK, 34, true, true}; break;

    // --- known quants we deliberately do NOT decode ---------------------------
    // Geometry is still reported so the inspector can say "block 3 of 1024" and
    // name the type honestly; only the decode is refused.
    case GgmlType::Q8_1: L = {kQK, 36, true, false}; break;
    case GgmlType::Q2_K: L = {kQK_K, 84, true, false}; break;
    case GgmlType::Q3_K: L = {kQK_K, 110, true, false}; break;
    case GgmlType::Q4_K: L = {kQK_K, 144, true, false}; break;
    case GgmlType::Q5_K: L = {kQK_K, 176, true, false}; break;
    case GgmlType::Q6_K: L = {kQK_K, 210, true, false}; break;
    case GgmlType::Q8_K: L = {kQK_K, 292, true, false}; break;
    case GgmlType::IQ2_XXS: L = {kQK_K, 66, true, false}; break;
    case GgmlType::IQ2_XS:  L = {kQK_K, 74, true, false}; break;
    case GgmlType::IQ2_S:   L = {kQK_K, 82, true, false}; break;
    case GgmlType::IQ3_XXS: L = {kQK_K, 98, true, false}; break;
    case GgmlType::IQ3_S:   L = {kQK_K, 110, true, false}; break;
    case GgmlType::IQ1_S:   L = {kQK_K, 50, true, false}; break;
    case GgmlType::IQ1_M:   L = {kQK_K, 56, true, false}; break;
    case GgmlType::IQ4_NL:  L = {kQK, 18, true, false}; break;
    case GgmlType::IQ4_XS:  L = {kQK_K, 136, true, false}; break;

    // --- plain scalar element types ------------------------------------------
    case GgmlType::F32:  L = {1, 4, false, false}; break;
    case GgmlType::F16:  L = {1, 2, false, false}; break;
    case GgmlType::BF16: L = {1, 2, false, false}; break;
    case GgmlType::F64:  L = {1, 8, false, false}; break;
    case GgmlType::I8:   L = {1, 1, false, false}; break;
    case GgmlType::I16:  L = {1, 2, false, false}; break;
    case GgmlType::I32:  L = {1, 4, false, false}; break;
    case GgmlType::I64:  L = {1, 8, false, false}; break;
  }
  return L;
}

bool ggml_is_quantized(uint32_t type_id) {
  return ggml_block_layout(type_id).quantized;
}

std::string_view dequant_status_message(DequantStatus s) {
  switch (s) {
    case DequantStatus::Ok:
      return "ok";
    case DequantStatus::UnknownType:
      return "unrecognized ggml tensor type — no block layout is known for it";
    case DequantStatus::NotQuantized:
      return "tensor is not block-quantized; values are read directly";
    case DequantStatus::UnsupportedLayout:
      return "K-quant / IQ super-block layout — preview supports only the "
             "legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 blocks";
    case DequantStatus::ShortBuffer:
      return "truncated: fewer bytes remain than this block needs";
  }
  return "unavailable";
}

DequantStatus dequant_block(uint32_t type_id, const uint8_t* block, size_t avail,
                            float* out, uint32_t* out_count) {
  if (out_count) *out_count = 0;
  if (!block || !out || !out_count) return DequantStatus::UnknownType;

  const GgmlBlockLayout L = ggml_block_layout(type_id);
  if (L.block_bytes == 0) return DequantStatus::UnknownType;
  if (!L.quantized) return DequantStatus::NotQuantized;
  if (!L.dequant_supported) return DequantStatus::UnsupportedLayout;
  if (avail < L.block_bytes) return DequantStatus::ShortBuffer;

  // Every supported layout is QK=32, which the header's kMaxDequantBlockElems
  // pins; this guards the caller's fixed-size buffer against a table edit that
  // adds a wider layout without widening the bound.
  static_assert(kQK <= kMaxDequantBlockElems,
                "kMaxDequantBlockElems must cover every dequant_supported layout");

  switch (static_cast<GgmlType>(type_id)) {
    case GgmlType::Q4_0: {
      const float d = f16_to_f32(rd_u16(block));
      const uint8_t* qs = block + 2;
      for (uint32_t j = 0; j < 16; ++j) {
        const int32_t lo = static_cast<int32_t>(qs[j] & 0x0F) - 8;
        const int32_t hi = static_cast<int32_t>(qs[j] >> 4) - 8;
        out[j] = static_cast<float>(lo) * d;
        out[j + 16] = static_cast<float>(hi) * d;
      }
      break;
    }
    case GgmlType::Q4_1: {
      const float d = f16_to_f32(rd_u16(block));
      const float m = f16_to_f32(rd_u16(block + 2));
      const uint8_t* qs = block + 4;
      for (uint32_t j = 0; j < 16; ++j) {
        out[j] = static_cast<float>(qs[j] & 0x0F) * d + m;
        out[j + 16] = static_cast<float>(qs[j] >> 4) * d + m;
      }
      break;
    }
    case GgmlType::Q5_0: {
      const float d = f16_to_f32(rd_u16(block));
      const uint32_t qh = rd_u32(block + 2);
      const uint8_t* qs = block + 6;
      for (uint32_t j = 0; j < 16; ++j) {
        // The 5th bit of element j lives in qh bit j; of element j+16 in bit
        // j+16. Shift it up to the 0x10 position and OR it onto the nibble.
        const uint32_t xh_lo = ((qh >> j) << 4) & 0x10u;
        const uint32_t xh_hi = (qh >> (j + 12)) & 0x10u;
        const int32_t lo = static_cast<int32_t>((qs[j] & 0x0F) | xh_lo) - 16;
        const int32_t hi = static_cast<int32_t>((qs[j] >> 4) | xh_hi) - 16;
        out[j] = static_cast<float>(lo) * d;
        out[j + 16] = static_cast<float>(hi) * d;
      }
      break;
    }
    case GgmlType::Q5_1: {
      const float d = f16_to_f32(rd_u16(block));
      const float m = f16_to_f32(rd_u16(block + 2));
      const uint32_t qh = rd_u32(block + 4);
      const uint8_t* qs = block + 8;
      for (uint32_t j = 0; j < 16; ++j) {
        const uint32_t xh_lo = ((qh >> j) << 4) & 0x10u;
        const uint32_t xh_hi = (qh >> (j + 12)) & 0x10u;
        out[j] = static_cast<float>((qs[j] & 0x0F) | xh_lo) * d + m;
        out[j + 16] = static_cast<float>((qs[j] >> 4) | xh_hi) * d + m;
      }
      break;
    }
    case GgmlType::Q8_0: {
      const float d = f16_to_f32(rd_u16(block));
      const uint8_t* qs = block + 2;
      for (uint32_t j = 0; j < kQK; ++j) {
        const int8_t q = static_cast<int8_t>(qs[j]);
        out[j] = static_cast<float>(q) * d;
      }
      break;
    }
    default:
      // dequant_supported was true but no case handled it — a table/switch drift
      // bug. Refuse rather than emit an uninitialized buffer.
      return DequantStatus::UnsupportedLayout;
  }

  *out_count = kQK;
  return DequantStatus::Ok;
}

}  // namespace netvis::gguf
