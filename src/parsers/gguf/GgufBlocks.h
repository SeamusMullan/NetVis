// parsers/gguf/GgufBlocks.h — exact ggml quantization types + bounded, view-only
// single-block dequantization (#49).
//
// DECISION (v0.9.1b, see DECISIONS.md "dequantization scope"): dequantization is
// a non-goal as a TRANSFORM — NetVis never dequantizes a model, never writes or
// exports dequantized payload, and never exposes dequantization to plugins. What
// IS in scope is a bounded, opt-in, view-only preview of ONE block, so a user
// staring at a Q4_K tensor can see the actual numbers behind the histogram they
// are already allowed to see.
//
// This header holds the FORMAT knowledge (which ggml type ids exist, what their
// blocks look like). It reads nothing on its own: every entry point takes a
// caller-supplied byte span and a caller-supplied output buffer, so it cannot
// allocate, cannot stream, and cannot be pointed at more than one block. The
// payload read itself stays where it has always been — engine/TensorStats.cpp,
// the single decode path (spec §2.1).
//
// SCOPE OF DEQUANT SUPPORT — deliberately narrow. The five legacy layouts below
// (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) are simple, stable, and fully specified: a scale (+
// optional min) followed by packed integers, 32 elements per block. The K-quants
// (Q2_K..Q8_K) and the IQ* family use 256-element super-blocks with hierarchical
// scales and, for IQ*, codebook lookups; they are NOT supported here and report
// an honest reason rather than a guess. That is the honesty rule the whole
// analyzer already follows — an unsupported layout is reported unsupported, never
// approximated.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace netvis::gguf {

// Raw ggml tensor type ids, exactly as written in the GGUF tensor-info table.
// Values are the wire ids and MUST NOT be renumbered. Not exhaustive of every
// ggml build — unknown ids are handled by value, not by enumerator.
enum class GgmlType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
  BF16 = 30,
};

// The exact human name for a ggml type id ("Q4_K", "IQ2_XXS", "F16", ...).
// Returns an empty view for an id this build does not know. Used for the honest
// dtype label in the inspector as well as by the preview.
std::string_view ggml_type_name(uint32_t type_id);

// True for the block-quantized types (everything that is not a plain scalar
// element type). A false answer means the tensor is ordinary strided data and
// the normal decode path handles it.
bool ggml_is_quantized(uint32_t type_id);

// Block geometry for a ggml type.
//
// elems_per_block / block_bytes are 0 when the type is unknown to this build or
// is not block-based. dequant_supported is true only for the five legacy layouts
// this file can actually decode; for every other quantized type the geometry may
// still be reported (so the caller can say "block 3 of 1024") while decoding is
// refused.
struct GgmlBlockLayout {
  uint32_t elems_per_block = 0;
  uint32_t block_bytes = 0;
  bool quantized = false;
  bool dequant_supported = false;
};

GgmlBlockLayout ggml_block_layout(uint32_t type_id);

// The largest elems_per_block across every type with dequant_supported == true.
// Callers size their output buffer with this, so no allocation is ever needed.
// A static_assert in the .cpp pins the table to this bound.
constexpr uint32_t kMaxDequantBlockElems = 32;

// Why a block could not be decoded. Ordered so the caller can branch, but each
// value also maps to a fixed human string via dequant_status_message().
enum class DequantStatus : uint8_t {
  Ok,
  UnknownType,       // this build does not know the ggml type id at all
  NotQuantized,      // plain F32/F16/I8/... — the caller should use normal decode
  UnsupportedLayout, // known quantized type, but a K-quant / IQ* super-block
  ShortBuffer,       // fewer bytes available than the block needs (truncated file)
};

// A fixed, honest one-line explanation for a non-Ok status. Never empty.
std::string_view dequant_status_message(DequantStatus s);

// Dequantize EXACTLY ONE block.
//
//   block  — pointer to the first byte of the block. The caller is responsible
//            for having bounds-checked it against the mapping.
//   avail  — bytes readable from `block`. Must be >= layout.block_bytes or the
//            call fails with ShortBuffer; this function never reads past `avail`.
//   out    — caller-provided buffer of at least kMaxDequantBlockElems floats.
//   out_count — set to the number of floats written (0 on failure).
//
// Pure: no allocation, no I/O, no global state, and it touches at most
// layout.block_bytes input bytes. Deterministic for a given input.
DequantStatus dequant_block(uint32_t type_id, const uint8_t* block, size_t avail,
                            float* out, uint32_t* out_count);

}  // namespace netvis::gguf
