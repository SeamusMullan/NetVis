// parsers/gguf/GgufParser.cpp — GGUF (v2/v3) reader for llama.cpp-style models.
//
// File layout (all little-endian):
//   u32  magic  "GGUF" (0x46554747)
//   u32  version (2 or 3)
//   u64  tensor_count
//   u64  metadata_kv_count
//   metadata_kv_count x { gguf_string key; u32 value_type; value }
//   tensor_count      x { gguf_string name; u32 n_dims; u64 dims[n_dims];
//                         u32 ggml_type; u64 offset }
//   <pad to general.alignment> then the data section (raw tensor payloads).
//   gguf_string := u64 len; byte[len]
//
// DECISION (spec §2.1, the product thesis): we read the header, the full
// metadata KV table, and the tensor-info table — all tiny — and record only
// file_offset+byte_len for each tensor. The data section (the multi-GB bulk) is
// NEVER read here; it stays on disk (mmap) until the weight inspector touches a
// specific tensor. That is what makes cold-open near-instant.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

namespace netvis::gguf {
namespace {

// GGUF metadata value type tags (spec §6.4).
enum GgufType : uint32_t {
  GT_UINT8 = 0, GT_INT8 = 1, GT_UINT16 = 2, GT_INT16 = 3,
  GT_UINT32 = 4, GT_INT32 = 5, GT_FLOAT32 = 6, GT_BOOL = 7,
  GT_STRING = 8, GT_ARRAY = 9, GT_UINT64 = 10, GT_INT64 = 11,
  GT_FLOAT64 = 12
};

// Map a ggml tensor type id to the IR DType. Quantized blocks collapse to the
// two coarse labels Q4/Q8 (spec §6.4): dequantization is out of scope for v1,
// so these are display labels only. Best-effort bucketing — 4/5-bit and the
// sub-4-bit IQ/K-quants group under Q4; 6/8-bit group under Q8.
ir::DType map_ggml_type(uint32_t t) {
  switch (t) {
    case 0:  return ir::DType::F32;   // GGML_TYPE_F32
    case 1:  return ir::DType::F16;   // GGML_TYPE_F16
    case 2:  return ir::DType::Q4;    // Q4_0
    case 3:  return ir::DType::Q4;    // Q4_1
    case 6:  return ir::DType::Q4;    // Q5_0 (5-bit -> Q4 bucket)
    case 7:  return ir::DType::Q4;    // Q5_1
    case 8:  return ir::DType::Q8;    // Q8_0
    case 9:  return ir::DType::Q8;    // Q8_1
    case 10: return ir::DType::Q4;    // Q2_K
    case 11: return ir::DType::Q4;    // Q3_K
    case 12: return ir::DType::Q4;    // Q4_K
    case 13: return ir::DType::Q4;    // Q5_K
    case 14: return ir::DType::Q8;    // Q6_K (6-bit -> Q8 bucket)
    case 15: return ir::DType::Q8;    // Q8_K
    case 16: return ir::DType::Q4;    // IQ2_XXS
    case 17: return ir::DType::Q4;    // IQ2_XS
    case 18: return ir::DType::Q4;    // IQ3_XXS
    case 19: return ir::DType::Q4;    // IQ1_S
    case 20: return ir::DType::Q4;    // IQ4_NL
    case 21: return ir::DType::Q4;    // IQ3_S
    case 22: return ir::DType::Q4;    // IQ2_S
    case 23: return ir::DType::Q4;    // IQ4_XS
    case 24: return ir::DType::I8;    // GGML_TYPE_I8
    case 25: return ir::DType::I16;   // GGML_TYPE_I16
    case 26: return ir::DType::I32;   // GGML_TYPE_I32
    case 27: return ir::DType::I64;   // GGML_TYPE_I64
    case 28: return ir::DType::F64;   // GGML_TYPE_F64
    case 29: return ir::DType::Q4;    // IQ1_M
    case 30: return ir::DType::BF16;  // GGML_TYPE_BF16
    default: return ir::DType::Unknown;
  }
}

// Read a gguf_string (u64 length + bytes). Bounds-checked; a truncated string
// returns an error at the current byte offset.
Result<std::string> read_gguf_string(ByteReader& r) {
  auto len = r.u64le();
  if (!len) return len.error();
  return r.bytes(*len);  // metadata bytes only — never a tensor payload
}

// Read one scalar metadata value of `type`, always advancing the cursor. If
// `out` is non-null, append the value's textual form. Nested ARRAY inside an
// array element is not part of the GGUF spec and is rejected.
Result<bool> read_scalar(ByteReader& r, uint32_t type, std::string* out) {
  switch (type) {
    case GT_UINT8: {
      auto v = r.u8(); if (!v) return v.error();
      if (out) *out += std::to_string(static_cast<unsigned>(*v));
      return true;
    }
    case GT_INT8: {
      auto v = r.u8(); if (!v) return v.error();
      if (out) *out += std::to_string(static_cast<int>(static_cast<int8_t>(*v)));
      return true;
    }
    case GT_UINT16: {
      auto v = r.u16le(); if (!v) return v.error();
      if (out) *out += std::to_string(*v);
      return true;
    }
    case GT_INT16: {
      auto v = r.u16le(); if (!v) return v.error();
      if (out) *out += std::to_string(static_cast<int>(static_cast<int16_t>(*v)));
      return true;
    }
    case GT_UINT32: {
      auto v = r.u32le(); if (!v) return v.error();
      if (out) *out += std::to_string(*v);
      return true;
    }
    case GT_INT32: {
      auto v = r.i32le(); if (!v) return v.error();
      if (out) *out += std::to_string(*v);
      return true;
    }
    case GT_FLOAT32: {
      auto v = r.f32le(); if (!v) return v.error();
      if (out) *out += std::to_string(*v);
      return true;
    }
    case GT_BOOL: {
      auto v = r.u8(); if (!v) return v.error();
      if (out) *out += (*v ? "true" : "false");
      return true;
    }
    case GT_STRING: {
      auto s = read_gguf_string(r);
      if (!s) return s.error();
      if (out) *out += *s;
      return true;
    }
    case GT_UINT64: {
      auto v = r.u64le(); if (!v) return v.error();
      if (out) *out += std::to_string(*v);
      return true;
    }
    case GT_INT64: {
      auto v = r.i64le(); if (!v) return v.error();
      if (out) *out += std::to_string(*v);
      return true;
    }
    case GT_FLOAT64: {
      auto v = r.f64le(); if (!v) return v.error();
      if (out) *out += std::to_string(*v);
      return true;
    }
    default:
      return err("unknown/nested gguf value type", r.pos());
  }
}

// Read an ARRAY value: u32 elem_type + u64 count + count elements. All elements
// are consumed to keep the cursor aligned, but only the first few are rendered
// into the summary string so a 100k-token vocab array doesn't blow up metadata.
Result<bool> read_array(ByteReader& r, std::string& out) {
  auto et = r.u32le(); if (!et) return et.error();
  auto cnt = r.u64le(); if (!cnt) return cnt.error();
  const uint32_t elem_type = *et;
  const uint64_t count = *cnt;

  constexpr uint64_t kMaxShow = 8;  // cap rendered elements; huge arrays truncate
  out += "[";
  for (uint64_t i = 0; i < count; ++i) {
    const bool show = (i < kMaxShow);
    if (show && i > 0) out += ", ";
    // Pass nullptr past the cap so we still advance the cursor without building
    // a giant string (perf: metadata stringification stays O(kMaxShow)).
    auto ok = read_scalar(r, elem_type, show ? &out : nullptr);
    if (!ok) return ok.error();
  }
  if (count > kMaxShow) out += ", ...";
  out += "] (" + std::to_string(count) + " items)";
  return true;
}

// Align `x` up to a multiple of `a` (a>=1). Works for any alignment, not just
// powers of two; overflow-guarded against the file size by the caller.
uint64_t align_up(uint64_t x, uint64_t a) {
  if (a == 0) a = 1;
  return ((x + a - 1) / a) * a;
}

}  // namespace

// Parse a GGUF file into tensor-table mode (has_graph == false). Every KV goes
// into Model::metadata (arrays summarized); every tensor becomes a TensorRef
// with offset+len only. Malformed/truncated input returns a Result error with
// the offending byte offset — never a crash or out-of-bounds read.
Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "GGUF header");

  const uint64_t file_size = file.size();
  ByteReader reader(file.data(), file_size);

  // --- Fixed header ----------------------------------------------------------
  auto magic = reader.u32le();
  if (!magic) return magic.error();
  if (*magic != 0x46554747u) {  // "GGUF" little-endian
    return err("not a GGUF file (bad magic)", 0);
  }
  auto version = reader.u32le();
  if (!version) return version.error();
  if (*version != 2 && *version != 3) {
    return err("unsupported GGUF version (expected 2 or 3)", 4);
  }
  auto tcount = reader.u64le();
  if (!tcount) return tcount.error();
  auto kvcount = reader.u64le();
  if (!kvcount) return kvcount.error();
  const uint64_t tensor_count = *tcount;
  const uint64_t kv_count = *kvcount;

  ir::Model model;
  model.has_graph = false;  // tensor-table mode: no compute graph (spec §8.6)
  model.format_name = model.intern("GGUF");
  model.version_info = model.intern(*version == 3 ? "GGUF v3" : "GGUF v2");

  // Defensive reserve: `kv_count`/`tensor_count` are attacker-controlled u64s, so
  // cap the reservation by what could physically fit (a KV/tensor info is at
  // least ~12/~24 bytes). Prevents a huge count from throwing bad_alloc.
  model.metadata.reserve(static_cast<size_t>(std::min<uint64_t>(kv_count, file_size / 12 + 1)));

  // --- Metadata KV table -----------------------------------------------------
  progress.set(0.2f, "GGUF metadata");
  uint64_t alignment = 32;  // spec default; overridden by general.alignment KV
  for (uint64_t i = 0; i < kv_count; ++i) {
    auto key = read_gguf_string(reader);
    if (!key) return key.error();
    auto vtype = reader.u32le();
    if (!vtype) return vtype.error();

    std::string value_str;
    if (*vtype == GT_ARRAY) {
      auto ok = read_array(reader, value_str);
      if (!ok) return ok.error();
    } else {
      auto ok = read_scalar(reader, *vtype, &value_str);
      if (!ok) return ok.error();
      // Capture the data-section alignment override (u32 scalar).
      if (*key == "general.alignment" && *vtype == GT_UINT32) {
        const uint64_t a = std::strtoull(value_str.c_str(), nullptr, 10);
        if (a >= 1) alignment = a;
      }
    }
    model.metadata.emplace_back(model.intern(*key), model.intern(value_str));
  }

  // --- Tensor info table -----------------------------------------------------
  progress.set(0.6f, "GGUF tensors");
  // Parsed tensor descriptors; `offset` is relative to the data section base.
  struct RawTensor {
    ir::TensorRef ref;
    uint64_t offset = 0;
    bool quantized = false;
  };
  std::vector<RawTensor> raws;
  raws.reserve(static_cast<size_t>(std::min<uint64_t>(tensor_count, file_size / 24 + 1)));

  for (uint64_t i = 0; i < tensor_count; ++i) {
    auto name = read_gguf_string(reader);
    if (!name) return name.error();
    auto ndims = reader.u32le();
    if (!ndims) return ndims.error();

    ir::TensorRef t;
    t.name = model.intern(*name);
    // Reading n_dims u64 dims: bounded by ByteReader — a bogus huge n_dims hits
    // EOF and errors out rather than looping forever or reading OOB.
    for (uint32_t d = 0; d < *ndims; ++d) {
      auto dim = reader.u64le();
      if (!dim) return dim.error();
      t.shape.push_back(static_cast<int64_t>(*dim));
    }
    auto gtype = reader.u32le();
    if (!gtype) return gtype.error();
    auto off = reader.u64le();
    if (!off) return off.error();

    t.dtype = map_ggml_type(*gtype);

    RawTensor rt;
    rt.ref = std::move(t);
    rt.offset = *off;
    // dtype_size == 0 marks a block-based (quantized/unknown) type whose byte
    // length we cannot derive from shape; we size it from the offset gap below.
    rt.quantized = (ir::dtype_size(rt.ref.dtype) == 0);
    raws.push_back(std::move(rt));
  }

  // --- Data section base -----------------------------------------------------
  // The payload region starts at the current cursor, padded UP to `alignment`.
  const uint64_t after_table = reader.pos();
  const uint64_t data_base = align_up(after_table, alignment);
  if (data_base > file_size) {
    return err("GGUF data section base exceeds file size", after_table);
  }
  const uint64_t data_span = file_size - data_base;  // bytes available to payloads

  // For quantized tensors we size by the gap to the next tensor's offset (or to
  // end-of-data for the last). Build a sorted, unique offset list once so each
  // lookup is an O(log n) upper_bound rather than an O(n) scan.
  std::vector<uint64_t> sorted_off;
  sorted_off.reserve(raws.size());
  for (const RawTensor& rt : raws) sorted_off.push_back(rt.offset);
  std::sort(sorted_off.begin(), sorted_off.end());
  sorted_off.erase(std::unique(sorted_off.begin(), sorted_off.end()), sorted_off.end());

  model.flat_tensors.reserve(raws.size());
  for (RawTensor& rt : raws) {
    // Validate the tensor's start lies within the data section.
    if (rt.offset > data_span) {
      return err("GGUF tensor offset exceeds data section", data_base + rt.offset);
    }
    rt.ref.file_offset = data_base + rt.offset;  // absolute mmap offset

    uint64_t byte_len;
    if (!rt.quantized) {
      // Non-quantized: exact size = element count * element byte size.
      const int64_t elems = rt.ref.elem_count();
      const uint64_t esz = ir::dtype_size(rt.ref.dtype);
      byte_len = (elems > 0) ? static_cast<uint64_t>(elems) * esz : 0;
    } else {
      // Quantized/block-based: derive from the gap to the next tensor offset,
      // or to end-of-data for the final tensor (offsets sorted above).
      auto it = std::upper_bound(sorted_off.begin(), sorted_off.end(), rt.offset);
      const uint64_t next = (it != sorted_off.end()) ? *it : data_span;
      byte_len = (next > rt.offset) ? (next - rt.offset) : 0;
    }
    // Clamp so a bad size can never let the inspector read past EOF.
    if (byte_len > data_span - rt.offset) byte_len = data_span - rt.offset;
    rt.ref.byte_len = byte_len;

    // Payload bytes are never read here — only offset+len recorded (spec §2.1).
    model.flat_tensors.push_back(std::move(rt.ref));
  }

  progress.set(1.0f, "GGUF done");
  return model;
}

}  // namespace netvis::gguf
