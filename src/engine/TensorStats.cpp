// engine/TensorStats.cpp — the ONE place a tensor payload is read.
//
// DECISION (spec §7.5, §2.1): structural parsing never touches payload bytes.
// This file is the sole payload-reader; it calls ByteReader::mark_payload_read()
// exactly once per decode so the "counting ByteReader" tests can assert the
// parser left the counter at 0. Stats stream in chunks with NO converted copy
// of the whole tensor — inspecting a 500 MB weight never allocates 500 MB.
//
// THREADING: runs as a TensorDecodeJob on a worker; `base` is an immutable mmap
// (safe to read concurrently), external files are mapped locally for the call.
#include "engine/TensorStats.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "core/ByteReader.h"

namespace netvis {

namespace {

using ir::DType;

// Resolved payload view: a pointer into some mmap + length. Holds an optional
// owned MappedFile when the data is external (kept alive for the read).
struct Payload {
  const uint8_t* ptr = nullptr;
  uint64_t len = 0;
  MappedFile external;  // holds the mapping alive if external_path was used
};

// CoreML MIL "blob v2" storage (weight.bin). BlobFileValue.offset points at this
// 64-byte metadata header, NOT the raw data; the data lives at header.data_offset
// and is header.size_bytes long. Verified against apple/coremltools
// MILBlob/Blob/StorageFormat.hpp. We read only the four fields we trust (reserved
// fields are garbage pre-iOS18) via memcpy (offset may be unaligned).
constexpr uint32_t kBlobSentinel = 0xDEADBEEFu;
constexpr uint64_t kBlobMetadataSize = 64;

// Follow a blob_metadata header located at [file_base + hdr_off] within a file of
// `file_size` bytes. Fills *data_off / *data_len with the raw payload location.
Result<bool> follow_blob_header(const uint8_t* file_base, uint64_t file_size,
                                uint64_t hdr_off, uint64_t* data_off,
                                uint64_t* data_len) {
  if (hdr_off > file_size || kBlobMetadataSize > file_size - hdr_off)
    return err("MIL blob metadata header out of range", hdr_off);
  const uint8_t* h = file_base + hdr_off;
  uint32_t sentinel;
  std::memcpy(&sentinel, h + 0, 4);
  if (sentinel != kBlobSentinel)
    return err("MIL blob metadata sentinel mismatch", hdr_off);
  uint64_t size_bytes, off;
  std::memcpy(&size_bytes, h + 8, 8);   // sizeInBytes
  std::memcpy(&off, h + 16, 8);         // data_offset
  if (off > file_size || size_bytes > file_size - off)
    return err("MIL blob payload out of range", off);
  *data_off = off;
  *data_len = size_bytes;
  return true;
}

// Resolve the payload for a tensor. external_path nonempty -> open+mmap that
// file relative to model_dir; else use base.data()+file_offset with a bounds
// check against base.size(). When t.blob_indirect, file_offset points at a MIL
// blob_metadata header that we follow to the real data (still one payload read).
Result<Payload> resolve_payload(const ir::TensorRef& t, const MappedFile& base,
                                const std::string& model_dir,
                                const ir::Model* model) {
  Payload out;

  std::string ext;
  if (t.external_path.valid() && model) ext = std::string(model->str(t.external_path));

  // Select the backing file (external sibling vs the model's own mmap) and the
  // starting offset; the blob-indirect follow + bounds are shared below.
  const uint8_t* file_base = nullptr;
  uint64_t file_size = 0;

  if (!ext.empty()) {
    std::string path = ext;
    // Resolve relative to model_dir unless already absolute.
    if (!path.empty() && path[0] != '/' && !model_dir.empty()) {
      path = model_dir;
      if (path.back() != '/') path.push_back('/');
      path += ext;
    }
    auto mf = MappedFile::open(path);
    if (!mf) return mf.error();
    out.external = std::move(*mf);
    file_base = out.external.data();
    file_size = out.external.size();
  } else {
    file_base = base.data();
    file_size = base.size();
  }

  if (t.file_offset == UINT64_MAX && !ext.empty()) {
    // External payload with no explicit offset starts at 0 (legacy behavior).
    out.ptr = file_base;
    out.len = t.byte_len;
    if (t.byte_len > file_size) return err("external tensor payload out of range", 0);
    return out;
  }
  if (t.file_offset == UINT64_MAX)
    return err("tensor has no payload offset", UINT64_MAX);

  if (t.blob_indirect) {
    uint64_t data_off = 0, data_len = 0;
    auto ok = follow_blob_header(file_base, file_size, t.file_offset, &data_off, &data_len);
    if (!ok) return ok.error();
    out.ptr = file_base + data_off;
    out.len = data_len;
    return out;
  }

  if (t.file_offset > file_size || t.byte_len > file_size - t.file_offset)
    return err("tensor payload out of range", t.file_offset);
  out.ptr = file_base + t.file_offset;
  out.len = t.byte_len;
  return out;
}

// F16 (IEEE half) bit pattern -> float.
float f16_to_f32(uint16_t h) {
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
float bf16_to_f32(uint16_t b) {
  uint32_t bits = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Number of elements from byte_len given dtype (0 for quantized/unknown).
uint64_t elem_count_from_bytes(const ir::TensorRef& t, Payload& p) {
  uint32_t es = ir::dtype_size(t.dtype);
  if (es == 0) return 0;
  // Prefer the declared shape's product; fall back to byte_len/es.
  // SECURITY: compare against the byte budget via DIVISION, never multiplication.
  // A hostile shape (e.g. [2^61]) times es overflows uint64 and could wrap to a
  // small value that spuriously passes a `need <= p.len` check, then drive an
  // out-of-bounds streaming read. `p.len / es` cannot overflow, so this clamps
  // the element count to what the payload can actually hold.
  int64_t ec = t.elem_count();
  uint64_t max_elems = p.len / es;
  if (ec > 0 && static_cast<uint64_t>(ec) <= max_elems)
    return static_cast<uint64_t>(ec);
  return max_elems;
}

// Read element `i` (0-based) of a payload as double, per dtype. Assumes bounds
// already validated by the caller's count computation.
double read_elem(const uint8_t* base, DType dt, uint64_t i) {
  switch (dt) {
    case DType::F32: { float v; std::memcpy(&v, base + i * 4, 4); return v; }
    case DType::F64: { double v; std::memcpy(&v, base + i * 8, 8); return v; }
    case DType::F16: { uint16_t v; std::memcpy(&v, base + i * 2, 2); return f16_to_f32(v); }
    case DType::BF16: { uint16_t v; std::memcpy(&v, base + i * 2, 2); return bf16_to_f32(v); }
    case DType::I8: { int8_t v; std::memcpy(&v, base + i, 1); return v; }
    case DType::I16: { int16_t v; std::memcpy(&v, base + i * 2, 2); return v; }
    case DType::I32: { int32_t v; std::memcpy(&v, base + i * 4, 4); return v; }
    case DType::I64: { int64_t v; std::memcpy(&v, base + i * 8, 8); return static_cast<double>(v); }
    case DType::U8: { uint8_t v; std::memcpy(&v, base + i, 1); return v; }
    case DType::U16: { uint16_t v; std::memcpy(&v, base + i * 2, 2); return v; }
    case DType::U32: { uint32_t v; std::memcpy(&v, base + i * 4, 4); return v; }
    case DType::U64: { uint64_t v; std::memcpy(&v, base + i * 8, 8); return static_cast<double>(v); }
    case DType::Bool: { uint8_t v; std::memcpy(&v, base + i, 1); return v ? 1.0 : 0.0; }
    default: return 0.0;
  }
}

bool is_quantized(DType d) { return d == DType::Q4 || d == DType::Q8; }

// NumPy dtype descr string for a dtype (little-endian). bf16 exports as <f4.
const char* npy_descr(DType d) {
  switch (d) {
    case DType::F32: case DType::BF16: return "<f4";
    case DType::F16: return "<f2";
    case DType::F64: return "<f8";
    case DType::I8: return "|i1";
    case DType::I16: return "<i2";
    case DType::I32: return "<i4";
    case DType::I64: return "<i8";
    case DType::U8: return "|u1";
    case DType::U16: return "<u2";
    case DType::U32: return "<u4";
    case DType::U64: return "<u8";
    case DType::Bool: return "|b1";
    default: return nullptr;
  }
}

}  // namespace

Result<TensorStats> compute_tensor_stats(const ir::TensorRef& t,
                                         const MappedFile& base,
                                         const std::string& model_dir,
                                         const ir::Model* model) {
  // We need the model to resolve an external_path StringId.
  auto pr = resolve_payload(t, base, model_dir, model);
  if (!pr) return pr.error();
  Payload p = std::move(*pr);

  // PAYLOAD READ: this is the single accounted payload access for this decode.
  ByteReader::mark_payload_read();

  TensorStats stats;

  // Quantized blocks: v1 does not dequantize. Report metadata only.
  if (is_quantized(t.dtype)) {
    stats.quantized_unsupported = true;
    // count in "elements" is not well-defined for block quant; expose byte len
    // via count so the UI can show something meaningful.
    stats.count = 0;
    return stats;  // note: dequantization not supported in v1
  }

  uint32_t es = ir::dtype_size(t.dtype);
  if (es == 0) {
    // Unknown dtype: nothing to compute.
    stats.count = 0;
    return stats;
  }

  uint64_t n = elem_count_from_bytes(t, p);
  stats.count = n;
  if (n == 0) return stats;

  // --- Pass 1: min/max/mean/std/counts, streaming in chunks ----------------
  constexpr uint64_t kChunk = 65536;  // ~64K elems per chunk (no full copy)
  double vmin = std::numeric_limits<double>::infinity();
  double vmax = -std::numeric_limits<double>::infinity();
  double sum = 0.0, sumsq = 0.0;
  uint64_t zero_count = 0, naninf = 0;
  uint64_t finite_count = 0;

  for (uint64_t start = 0; start < n; start += kChunk) {
    uint64_t end = std::min(start + kChunk, n);
    for (uint64_t i = start; i < end; ++i) {
      double v = read_elem(p.ptr, t.dtype, i);
      if (v == 0.0) ++zero_count;
      if (std::isnan(v) || std::isinf(v)) { ++naninf; continue; }
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
      sum += v;
      sumsq += v * v;
      ++finite_count;
    }
  }

  stats.zero_count = zero_count;
  stats.nan_inf_count = naninf;
  if (finite_count > 0) {
    double mean = sum / static_cast<double>(finite_count);
    stats.mean = mean;
    double var = sumsq / static_cast<double>(finite_count) - mean * mean;
    if (var < 0) var = 0;  // guard tiny negatives from rounding
    stats.std = std::sqrt(var);
    stats.min = vmin;
    stats.max = vmax;
  }

  // --- Pass 2: 64-bucket histogram (re-read from mmap, no full copy) --------
  double hmin = stats.min, hmax = stats.max;
  stats.hist_min = hmin;
  stats.hist_max = hmax;
  double range = hmax - hmin;
  if (finite_count > 0 && range > 0) {
    const double inv = static_cast<double>(kHistogramBuckets) / range;
    for (uint64_t start = 0; start < n; start += kChunk) {
      uint64_t end = std::min(start + kChunk, n);
      for (uint64_t i = start; i < end; ++i) {
        double v = read_elem(p.ptr, t.dtype, i);
        if (std::isnan(v) || std::isinf(v)) continue;
        int b = static_cast<int>((v - hmin) * inv);
        if (b < 0) b = 0;
        if (b >= kHistogramBuckets) b = kHistogramBuckets - 1;
        ++stats.histogram[static_cast<size_t>(b)];
      }
    }
  } else if (finite_count > 0) {
    // All values equal (or single value): dump into the first bucket.
    stats.histogram[0] = finite_count;
  }

  return stats;
}

Result<bool> export_npy(const ir::TensorRef& t, const MappedFile& base,
                        const std::string& model_dir,
                        const std::string& out_path,
                        const ir::Model* model) {
  auto pr = resolve_payload(t, base, model_dir, model);
  if (!pr) return pr.error();
  Payload p = std::move(*pr);

  // PAYLOAD READ: single accounted access for this export.
  ByteReader::mark_payload_read();

  if (is_quantized(t.dtype))
    return err("cannot export quantized tensor as .npy (v1)", UINT64_MAX);
  const char* descr = npy_descr(t.dtype);
  if (!descr) return err("unsupported dtype for .npy export", UINT64_MAX);

  // Build the header dict. NumPy tuple convention: "(2, 3)" for rank>=2, a
  // trailing comma ONLY for a 1-element tuple "(3,)", and "()" for a scalar.
  std::string shape_str = "(";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (t.shape[i] < 0) return err("cannot export tensor with dynamic dim", UINT64_MAX);
    if (i) shape_str += ", ";
    shape_str += std::to_string(t.shape[i]);
  }
  if (t.shape.size() == 1) shape_str += ",";
  shape_str += ")";

  std::string dict = "{'descr': '";
  dict += descr;
  dict += "', 'fortran_order': False, 'shape': ";
  dict += shape_str;
  dict += ", }";

  // Header: 6-byte magic + 2 version bytes + u16 header length. Total up to and
  // including the dict (+ trailing '\n') must be a multiple of 64.
  const size_t preamble = 10;  // magic(6)+version(2)+len(2)
  size_t unpadded = preamble + dict.size() + 1;  // +1 for '\n'
  size_t total = (unpadded + 63) / 64 * 64;
  size_t pad = total - unpadded;
  dict.append(pad, ' ');
  dict.push_back('\n');
  uint16_t header_len = static_cast<uint16_t>(dict.size());

  FILE* f = std::fopen(out_path.c_str(), "wb");
  if (!f) return err("cannot open output file for .npy", UINT64_MAX);

  auto fail = [&](const char* msg) -> Result<bool> {
    std::fclose(f);
    return err(msg, UINT64_MAX);
  };

  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  if (std::fwrite(magic, 1, 8, f) != 8) return fail("write error (.npy magic)");
  unsigned char hl[2] = {static_cast<unsigned char>(header_len & 0xFF),
                         static_cast<unsigned char>((header_len >> 8) & 0xFF)};
  if (std::fwrite(hl, 1, 2, f) != 2) return fail("write error (.npy hdr len)");
  if (std::fwrite(dict.data(), 1, dict.size(), f) != dict.size())
    return fail("write error (.npy header)");

  // Data. bf16 -> convert each element to f4; everything else is a raw copy.
  if (t.dtype == DType::BF16) {
    uint64_t n = p.len / 2;
    int64_t ec = t.elem_count();
    if (ec > 0 && static_cast<uint64_t>(ec) <= n) n = static_cast<uint64_t>(ec);
    // Stream conversion in chunks; no full converted buffer for the tensor.
    constexpr uint64_t kChunk = 65536;
    std::vector<float> buf;
    buf.reserve(kChunk);
    for (uint64_t start = 0; start < n; start += kChunk) {
      uint64_t end = std::min(start + kChunk, n);
      buf.clear();
      for (uint64_t i = start; i < end; ++i) {
        uint16_t h; std::memcpy(&h, p.ptr + i * 2, 2);
        buf.push_back(bf16_to_f32(h));
      }
      size_t bytes = buf.size() * sizeof(float);
      if (std::fwrite(buf.data(), 1, bytes, f) != bytes)
        return fail("write error (.npy bf16 data)");
    }
  } else {
    // Raw payload copy, chunked (avoids one giant write, keeps memory flat).
    constexpr uint64_t kChunk = 1u << 20;  // 1 MiB
    for (uint64_t off = 0; off < p.len; off += kChunk) {
      uint64_t w = std::min(kChunk, p.len - off);
      if (std::fwrite(p.ptr + off, 1, w, f) != w)
        return fail("write error (.npy data)");
    }
  }

  if (std::fclose(f) != 0) return err("close error (.npy)", UINT64_MAX);
  return true;
}

Result<bool> export_raw(const ir::TensorRef& t, const MappedFile& base,
                        const std::string& model_dir,
                        const std::string& out_path,
                        const ir::Model* model) {
  auto pr = resolve_payload(t, base, model_dir, model);
  if (!pr) return pr.error();
  Payload p = std::move(*pr);

  // PAYLOAD READ: single accounted access for this export.
  ByteReader::mark_payload_read();

  FILE* f = std::fopen(out_path.c_str(), "wb");
  if (!f) return err("cannot open output file for .bin", UINT64_MAX);

  constexpr uint64_t kChunk = 1u << 20;  // 1 MiB
  for (uint64_t off = 0; off < p.len; off += kChunk) {
    uint64_t w = std::min(kChunk, p.len - off);
    if (std::fwrite(p.ptr + off, 1, w, f) != w) {
      std::fclose(f);
      return err("write error (.bin data)", off);
    }
  }
  if (std::fclose(f) != 0) return err("close error (.bin)", UINT64_MAX);
  return true;
}

}  // namespace netvis
