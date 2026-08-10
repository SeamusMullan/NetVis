// tests/test_gguf_blocks.cpp — ggml quant type table + single-block dequant
// value tests (#49).
//
// GgufBlocks.cpp is pure math over a fully specified wire format (see the doc
// comments in GgufBlocks.h/.cpp), so every expected value below is HAND-
// COMPUTED from that spec, not copied from the implementation: f16 scales use
// bit patterns with an exact binary representation (1.0=0x3C00, 0.5=0x3800,
// 2.0=0x4000) so comparisons below are EXACT `==`, never epsilon-fuzzy Approx.
// Every 4/5-bit block reuses one asymmetric nibble ramp — low nibble of qs[j]
// = j, high nibble = 15-j — so a low/high-nibble swap ("the single most likely
// implementation bug in that file", per the .cpp header comment) changes the
// decoded numbers rather than silently producing a differently-wrong-but-
// plausible result.
#include <doctest/doctest.h>

#include <array>
#include <cstdint>

#include "parsers/gguf/GgufBlocks.h"

using namespace netvis;

namespace {

// Every supported layout is QK=32; the header pins kMaxDequantBlockElems to it.
static_assert(gguf::kMaxDequantBlockElems == 32,
             "test assumes the frozen v0.9.1b bound");

// Shared shape of every dequant_block FAILURE path: no elements written, and a
// non-empty human explanation. Factored out so each refusal test below states
// only what differs (the status, the type id, and the buffer).
void check_refusal(gguf::DequantStatus expected, uint32_t type_id,
                    const uint8_t* block, size_t avail) {
  std::array<float, gguf::kMaxDequantBlockElems> out{};
  uint32_t count = 123;  // poison value; must become 0 on failure
  gguf::DequantStatus st =
      gguf::dequant_block(type_id, block, avail, out.data(), &count);
  CHECK(st == expected);
  CHECK(count == 0);
  CHECK_FALSE(gguf::dequant_status_message(st).empty());
}

}  // namespace

// --- ggml_type_name -----------------------------------------------------------

TEST_CASE("#49 ggml_type_name: decodable, K-quant, scalar, and unknown ids") {
  CHECK(gguf::ggml_type_name(static_cast<uint32_t>(gguf::GgmlType::Q4_0)) == "Q4_0");
  CHECK(gguf::ggml_type_name(static_cast<uint32_t>(gguf::GgmlType::Q4_K)) == "Q4_K");
  CHECK(gguf::ggml_type_name(static_cast<uint32_t>(gguf::GgmlType::I16)) == "I16");
  CHECK(gguf::ggml_type_name(9999).empty());  // id this build does not know
}

// --- ggml_block_layout geometry ------------------------------------------------

TEST_CASE("#49 ggml_block_layout: geometry for every decodable legacy layout") {
  struct Expect { gguf::GgmlType type; uint32_t block_bytes; };
  const Expect table[] = {
      {gguf::GgmlType::Q4_0, 18}, {gguf::GgmlType::Q4_1, 20},
      {gguf::GgmlType::Q5_0, 22}, {gguf::GgmlType::Q5_1, 24},
      {gguf::GgmlType::Q8_0, 34},
  };
  for (const Expect& e : table) {
    gguf::GgmlBlockLayout L =
        gguf::ggml_block_layout(static_cast<uint32_t>(e.type));
    CHECK(L.elems_per_block == 32);
    CHECK(L.block_bytes == e.block_bytes);
    CHECK(L.quantized);
    CHECK(L.dequant_supported);
  }
}

TEST_CASE("#49 ggml_block_layout: K-quant and IQ4_NL are known but NOT decodable") {
  gguf::GgmlBlockLayout k =
      gguf::ggml_block_layout(static_cast<uint32_t>(gguf::GgmlType::Q4_K));
  CHECK(k.elems_per_block == 256);
  CHECK(k.block_bytes == 144);
  CHECK(k.quantized);
  CHECK_FALSE(k.dequant_supported);

  // IQ4_NL is QK=32 (the SAME block size as Q4_0!) but a codebook lookup, not
  // a linear scale, so geometry is known while decode is refused — distinct
  // from the K-quant's QK_K=256 super-block refusal above, and worth its own
  // case so a future implementation can't conflate "refused" with "256-wide".
  gguf::GgmlBlockLayout nl =
      gguf::ggml_block_layout(static_cast<uint32_t>(gguf::GgmlType::IQ4_NL));
  CHECK(nl.elems_per_block == 32);
  CHECK(nl.block_bytes == 18);
  CHECK(nl.quantized);
  CHECK_FALSE(nl.dequant_supported);
}

TEST_CASE("#49 ggml_block_layout: plain scalar types are not quantized") {
  gguf::GgmlBlockLayout f32 =
      gguf::ggml_block_layout(static_cast<uint32_t>(gguf::GgmlType::F32));
  CHECK_FALSE(f32.quantized);
  CHECK_FALSE(f32.dequant_supported);
  gguf::GgmlBlockLayout f16 =
      gguf::ggml_block_layout(static_cast<uint32_t>(gguf::GgmlType::F16));
  CHECK_FALSE(f16.quantized);
  CHECK_FALSE(f16.dequant_supported);
}

// --- dequant_block value correctness -------------------------------------------

TEST_CASE("#49 dequant_block Q4_0: scale + asymmetric nibble interleave") {
  // 18B = f16 d(2) + u8 qs[16]. d=1.0 -> 0x3C00 LE.
  // qs[j] packs element j in the LOW nibble (= j, so value j-8) and element
  // j+16 in the HIGH nibble (= 15-j, so value 7-j): out[0] must be -8 and
  // out[16] must be 7 — swapping the halves would flip that pair to 7/-8.
  std::array<uint8_t, 18> block = {
      0x00, 0x3C,                                     // d = 1.0
      0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,  // qs[0..7]
      0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F,  // qs[8..15]
  };
  std::array<float, gguf::kMaxDequantBlockElems> out{};
  uint32_t count = 0;
  auto st = gguf::dequant_block(static_cast<uint32_t>(gguf::GgmlType::Q4_0),
                                block.data(), block.size(), out.data(), &count);
  REQUIRE(st == gguf::DequantStatus::Ok);
  REQUIRE(count == 32);
  for (uint32_t j = 0; j < 16; ++j) {
    CHECK(out[j] == static_cast<float>(j) - 8.0f);            // low nibble: v=(j-8)*d
    CHECK(out[j + 16] == 7.0f - static_cast<float>(j));       // high nibble: v=(15-j-8)*d
  }
}

TEST_CASE("#49 dequant_block Q4_1: scale+min, no offset (nibble range 0..15)") {
  // 20B = f16 d(2) + f16 m(2) + u8 qs[16]. d=0.5 (0x3800), m=1.0 (0x3C00).
  // Same asymmetric qs ramp as the Q4_0 case above.
  std::array<uint8_t, 20> block = {
      0x00, 0x38,                                      // d = 0.5
      0x00, 0x3C,                                      // m = 1.0
      0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
      0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F,
  };
  std::array<float, gguf::kMaxDequantBlockElems> out{};
  uint32_t count = 0;
  auto st = gguf::dequant_block(static_cast<uint32_t>(gguf::GgmlType::Q4_1),
                                block.data(), block.size(), out.data(), &count);
  REQUIRE(st == gguf::DequantStatus::Ok);
  REQUIRE(count == 32);
  for (uint32_t j = 0; j < 16; ++j) {
    CHECK(out[j] == static_cast<float>(j) * 0.5f + 1.0f);              // v=nibble*d+m
    CHECK(out[j + 16] == static_cast<float>(15 - j) * 0.5f + 1.0f);
  }
}

TEST_CASE("#49 dequant_block Q5_0: 5th bit read from DIFFERENT qh bit positions") {
  // 22B = f16 d(2) + u8 qh[4] + u8 qs[16]. d=1.0. qh sets ONLY bit0 and
  // bit16, so exactly one low-half element (j=0, output index 0) and one
  // high-half element (j=0, output index 16) gain the +16 bit — proving the
  // low half reads qh bit j while the high half reads qh bit (j+16), not the
  // same bit applied twice.
  std::array<uint8_t, 22> block = {
      0x00, 0x3C,                                     // d = 1.0
      0x01, 0x00, 0x01, 0x00,                          // qh = bit0 | bit16
      0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
      0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F,
  };
  std::array<float, gguf::kMaxDequantBlockElems> out{};
  uint32_t count = 0;
  auto st = gguf::dequant_block(static_cast<uint32_t>(gguf::GgmlType::Q5_0),
                                block.data(), block.size(), out.data(), &count);
  REQUIRE(st == gguf::DequantStatus::Ok);
  REQUIRE(count == 32);
  // j=0 in each half picks up the extra bit; j=1..15 do not.
  CHECK(out[0] == 0.0f);    // (nibble 0 | bit5 0x10) - 16 = 0
  CHECK(out[16] == 15.0f);  // (nibble 15 | bit5 0x10) - 16 = 15
  for (uint32_t j = 1; j < 16; ++j) {
    CHECK(out[j] == static_cast<float>(j) - 16.0f);             // no bit5: nibble=j
    CHECK(out[j + 16] == static_cast<float>(15 - j) - 16.0f);   // no bit5: nibble=15-j
  }
}

TEST_CASE("#49 dequant_block Q5_1: 5th bit + scale/min, no offset") {
  // 24B = f16 d(2) + f16 m(2) + u8 qh[4] + u8 qs[16]. d=0.5, m=1.0; same qh
  // (bit0|bit16) and qs ramp as the Q5_0 case above.
  std::array<uint8_t, 24> block = {
      0x00, 0x38,                                     // d = 0.5
      0x00, 0x3C,                                     // m = 1.0
      0x01, 0x00, 0x01, 0x00,                          // qh = bit0 | bit16
      0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
      0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F,
  };
  std::array<float, gguf::kMaxDequantBlockElems> out{};
  uint32_t count = 0;
  auto st = gguf::dequant_block(static_cast<uint32_t>(gguf::GgmlType::Q5_1),
                                block.data(), block.size(), out.data(), &count);
  REQUIRE(st == gguf::DequantStatus::Ok);
  REQUIRE(count == 32);
  CHECK(out[0] == 9.0f);    // (0|0x10)*0.5+1.0 = 9.0
  CHECK(out[16] == 16.5f);  // (15|0x10)*0.5+1.0 = 16.5
  for (uint32_t j = 1; j < 16; ++j) {
    CHECK(out[j] == static_cast<float>(j) * 0.5f + 1.0f);
    CHECK(out[j + 16] == static_cast<float>(15 - j) * 0.5f + 1.0f);
  }
}

TEST_CASE("#49 dequant_block Q8_0: signed int8 * scale across the full range") {
  // 34B = f16 d(2) + i8 qs[32]. d=2.0 (0x4000). qs[j] = j-16 as two's
  // complement (0xF0..0xFF then 0x00..0x0F), covering the negative and
  // non-negative halves of the int8 range in one sweep.
  std::array<uint8_t, 34> block = {
      0x00, 0x40,                                      // d = 2.0
      0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,
      0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF,
      0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
      0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
  };
  std::array<float, gguf::kMaxDequantBlockElems> out{};
  uint32_t count = 0;
  auto st = gguf::dequant_block(static_cast<uint32_t>(gguf::GgmlType::Q8_0),
                                block.data(), block.size(), out.data(), &count);
  REQUIRE(st == gguf::DequantStatus::Ok);
  REQUIRE(count == 32);
  for (uint32_t j = 0; j < 32; ++j) {
    CHECK(out[j] == (static_cast<float>(j) - 16.0f) * 2.0f);
  }
}

TEST_CASE("#49 dequant_block Q8_0: int8 sign extension at the extremes") {
  // Isolate INT8_MIN/INT8_MAX so a naive `uint8_t -> float` cast (which would
  // turn -128 into +128 and leave 127 unchanged) is caught independently of
  // the ramp test above.
  std::array<uint8_t, 34> block{};
  block[0] = 0x00; block[1] = 0x3C;  // d = 1.0
  block[2] = 0x80;                   // qs[0] = INT8_MIN = -128
  block[3] = 0x7F;                   // qs[1] = INT8_MAX = 127
  // qs[2..31] left at 0.
  std::array<float, gguf::kMaxDequantBlockElems> out{};
  uint32_t count = 0;
  auto st = gguf::dequant_block(static_cast<uint32_t>(gguf::GgmlType::Q8_0),
                                block.data(), block.size(), out.data(), &count);
  REQUIRE(st == gguf::DequantStatus::Ok);
  CHECK(out[0] == -128.0f);
  CHECK(out[1] == 127.0f);
  for (uint32_t j = 2; j < 32; ++j) CHECK(out[j] == 0.0f);
}

// --- refusal paths --------------------------------------------------------------

TEST_CASE("#49 dequant_block: UnknownType for an id this build does not recognize") {
  const std::array<uint8_t, 18> buf{};
  check_refusal(gguf::DequantStatus::UnknownType, 9999, buf.data(), buf.size());
}

TEST_CASE("#49 dequant_block: NotQuantized for plain scalar data (F32)") {
  const std::array<uint8_t, 18> buf{};
  check_refusal(gguf::DequantStatus::NotQuantized,
               static_cast<uint32_t>(gguf::GgmlType::F32), buf.data(), 4);
}

TEST_CASE("#49 dequant_block: UnsupportedLayout for a known K-quant (Q4_K)") {
  // dequant_supported is checked BEFORE the avail bound (GgufBlocks.cpp), so
  // even a way-too-short buffer still reports UnsupportedLayout, never
  // ShortBuffer — the scope refusal takes priority over the size refusal.
  const std::array<uint8_t, 18> buf{};
  check_refusal(gguf::DequantStatus::UnsupportedLayout,
               static_cast<uint32_t>(gguf::GgmlType::Q4_K), buf.data(), 1);
}

TEST_CASE("#49 dequant_block: ShortBuffer when avail is one byte short") {
  const std::array<uint8_t, 18> buf{};  // Q4_0 needs exactly 18 bytes
  check_refusal(gguf::DequantStatus::ShortBuffer,
               static_cast<uint32_t>(gguf::GgmlType::Q4_0), buf.data(), 17);
}

TEST_CASE("#49 dequant_status_message: every status has a non-empty message") {
  using gguf::DequantStatus;
  CHECK_FALSE(gguf::dequant_status_message(DequantStatus::Ok).empty());
  CHECK_FALSE(gguf::dequant_status_message(DequantStatus::UnknownType).empty());
  CHECK_FALSE(gguf::dequant_status_message(DequantStatus::NotQuantized).empty());
  CHECK_FALSE(gguf::dequant_status_message(DequantStatus::UnsupportedLayout).empty());
  CHECK_FALSE(gguf::dequant_status_message(DequantStatus::ShortBuffer).empty());
}
