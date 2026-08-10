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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "core/Half.h"
#include "parsers/gguf/GgufBlocks.h"

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

// True if `child`, lexically normalized, stays within `root` (component-wise, no
// `..`/absolute escape). Mirrors ModelPath.cpp::within_root — an external tensor
// path (ONNX external_data.location / CoreML MIL blobFileValue.fileName) is fully
// attacker-controlled, so a malicious model must not read arbitrary local files.
bool path_within_root(const std::filesystem::path& root,
                      const std::filesystem::path& child) {
  const std::filesystem::path nroot = root.lexically_normal();
  const std::filesystem::path nchild = child.lexically_normal();
  auto ri = nroot.begin();
  auto ci = nchild.begin();
  for (; ri != nroot.end(); ++ri, ++ci) {
    // Skip a trailing empty component from a trailing slash on root.
    if (ri->empty()) break;
    if (ci == nchild.end() || *ci != *ri) return false;
  }
  return true;
}

// Resolve the payload for a tensor. external_path nonempty -> open+mmap that
// file relative to model_dir; else use base.data()+file_offset with a bounds
// check against base.size(). When t.blob_indirect, file_offset points at a MIL
// blob_metadata header that we follow to the real data (still one payload read).
//
// SECURITY: external_path comes verbatim from the model file (attacker-
// controlled). When a model_dir context exists, the resolved external file is
// confined to model_dir — a `../` or absolute path that escapes is rejected, so a
// hostile model cannot turn the weight inspector into an arbitrary file reader.
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
    // Resolve + confine relative to model_dir. When model_dir is empty the caller
    // passed an already-trusted absolute path (headless/tests): use it verbatim.
    if (!model_dir.empty()) {
      std::filesystem::path root(model_dir);
      std::filesystem::path resolved = root / ext;  // absolute ext replaces root
      if (!path_within_root(root, resolved))
        return err("external tensor path escapes model directory", 0);
      path = resolved.string();
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

// f16_to_f32 / bf16_to_f32 moved to core/Half.h in v0.9.1b — GgufBlocks.cpp
// (parsers layer, below engine) needs the same F16 decode for quant-block scales.

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

  // #46: per-output-channel accumulation setup. Channel = index along dim 0
  // (weights are [Cout, ...]). GUARD (spec §7.5 + kMaxChannels): only accumulate
  // when there are >=2 channels that divide the scanned count cleanly AND the
  // channel count stays within kMaxChannels — so a vocab-sized embedding
  // ([50257, 768]) never allocates an unbounded per-channel vector or blows the
  // decode budget. `elems_per_channel` uses the ACTUAL scanned element count `n`
  // (already clamped to the payload by elem_count_from_bytes), never the declared
  // shape product, so a hostile shape can't drive `i / elems_per_channel` OOB.
  //   - channels > kMaxChannels          -> per_channel_capped = true, empty
  //   - channels < 2 or n % channels != 0 -> skip cleanly (empty, NOT capped)
  const int64_t channels_i = t.shape.empty() ? 0 : t.shape[0];
  uint64_t channels = 0, elems_per_channel = 0;
  bool per_channel_on = false;
  if (channels_i > static_cast<int64_t>(kMaxChannels)) {
    stats.per_channel_capped = true;  // too many channels; per_channel stays empty
  } else if (channels_i >= 2 &&
             n % static_cast<uint64_t>(channels_i) == 0) {
    channels = static_cast<uint64_t>(channels_i);
    elems_per_channel = n / channels;
    stats.per_channel.assign(channels, ChannelStat{});
    per_channel_on = true;
  }
  // Scratch sums parallel to per_channel; ChannelStat has no sum field. min/max
  // start at +/-inf so the first finite value wins (default 0 would be wrong).
  std::vector<double> ch_sum;
  if (per_channel_on) {
    ch_sum.assign(channels, 0.0);
    for (auto& c : stats.per_channel) {
      c.min = std::numeric_limits<double>::infinity();
      c.max = -std::numeric_limits<double>::infinity();
    }
  }

  // --- Pass 1: min/max/mean/std/counts, streaming in chunks ----------------
  // Also records argmin/argmax flat indices (#51) and per-channel stats (#46) in
  // this SAME pass — no extra full re-read of the payload.
  constexpr uint64_t kChunk = 65536;  // ~64K elems per chunk (no full copy)
  double vmin = std::numeric_limits<double>::infinity();
  double vmax = -std::numeric_limits<double>::infinity();
  double sum = 0.0, sumsq = 0.0;
  uint64_t zero_count = 0, naninf = 0;
  uint64_t finite_count = 0;

  // #46: the channel index is MONOTONIC in i (row-major: dim0 is the outermost
  // axis), so advance it incrementally at each channel boundary rather than doing
  // a 64-bit divide per element on the streaming path (a real DIV since
  // elems_per_channel is a runtime value; ~131M of them on a 500 MB weight).
  uint64_t ci = 0;
  uint64_t ch_boundary = per_channel_on ? elems_per_channel : n;  // next channel start
  for (uint64_t start = 0; start < n; start += kChunk) {
    uint64_t end = std::min(start + kChunk, n);
    for (uint64_t i = start; i < end; ++i) {
      double v = read_elem(p.ptr, t.dtype, i);
      const bool is_zero = (v == 0.0);
      const bool bad = std::isnan(v) || std::isinf(v);
      if (is_zero) ++zero_count;

      ChannelStat* c = nullptr;
      if (per_channel_on) {
        if (i >= ch_boundary) { ++ci; ch_boundary += elems_per_channel; }
        c = &stats.per_channel[ci];
        if (is_zero) ++c->zero_count;
      }

      if (bad) {
        ++naninf;
        if (c) ++c->nan_inf_count;
        continue;
      }
      // #51: track the flat index of the running min/max over finite values.
      if (v < vmin) { vmin = v; stats.min_index = i; }
      if (v > vmax) { vmax = v; stats.max_index = i; }
      sum += v;
      sumsq += v * v;
      ++finite_count;
      if (c) {
        if (v < c->min) c->min = v;
        if (v > c->max) c->max = v;
        ch_sum[ci] += v;
      }
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

  // #46: finalize per-channel stats. count is the same constant for every channel
  // (guaranteed by the n % channels == 0 gate), so set it once here instead of
  // per-element in the hot loop. Then compute the mean over finite elements and
  // reset the min/max sentinels for any all-NaN/Inf channel back to 0.
  if (per_channel_on) {
    for (uint64_t k = 0; k < channels; ++k) {
      ChannelStat& c = stats.per_channel[k];
      c.count = elems_per_channel;
      const uint64_t finite = c.count - c.nan_inf_count;  // zeros are finite
      if (finite > 0) {
        c.mean = ch_sum[k] / static_cast<double>(finite);
      } else {
        c.min = 0;
        c.max = 0;
      }
    }
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

Result<TensorThumbnail> compute_tensor_thumbnail(const ir::TensorRef& t,
                                                 const MappedFile& base,
                                                 const std::string& model_dir,
                                                 const ir::Model* model) {
  auto pr = resolve_payload(t, base, model_dir, model);
  if (!pr) return pr.error();
  Payload p = std::move(*pr);

  // PAYLOAD READ: single accounted access for this decode (like the others).
  ByteReader::mark_payload_read();

  TensorThumbnail thumb;  // available=false, width=height=0 by default.

  // Unavailable (NOT an error): quantized/unknown dtype, rank<2, or a dynamic
  // (-1) / empty dim in the image plane (the last two axes).
  if (is_quantized(t.dtype)) return thumb;
  const uint32_t es = ir::dtype_size(t.dtype);
  if (es == 0) return thumb;                    // unknown dtype
  if (t.shape.size() < 2) return thumb;         // rank < 2
  const int64_t rows_i = t.shape[t.shape.size() - 2];
  const int64_t cols_i = t.shape[t.shape.size() - 1];
  if (rows_i <= 0 || cols_i <= 0) return thumb; // dynamic (-1) / empty plane

  const uint64_t rows = static_cast<uint64_t>(rows_i);
  const uint64_t cols = static_cast<uint64_t>(cols_i);

  // Output dimensions capped per side; block-average when the plane is larger.
  const uint64_t height = std::min<uint64_t>(rows, kThumbMax);
  const uint64_t width = std::min<uint64_t>(cols, kThumbMax);
  const uint64_t block_r = (rows + height - 1) / height;  // ceil(rows/height)
  const uint64_t block_c = (cols + width - 1) / width;    // ceil(cols/width)

  // Downsampled accumulators (bounded by kThumbMax^2 — never rows*cols).
  const size_t cells = static_cast<size_t>(width * height);
  std::vector<double> sum(cells, 0.0);
  std::vector<uint64_t> cnt(cells, 0);

  // Elements we may read: the first 2D slice [0, rows*cols), clamped to what the
  // payload actually holds. elem_count_from_bytes divides (never multiplies) so a
  // hostile shape can't drive an OOB streaming read; higher dims are fixed at 0.
  const uint64_t n = elem_count_from_bytes(t, p);

  // --- Single streaming pass over the slice: accumulate block sums -----------
  // The flat slice index `i` equals r*cols + c (row-major). Track the output
  // (row,col) incrementally so there is no 64-bit divide per element.
  uint64_t i = 0;
  for (uint64_t r = 0; r < rows && i < n; ++r) {
    uint64_t out_row = r / block_r;
    if (out_row >= height) out_row = height - 1;  // defensive clamp
    const size_t row_base = static_cast<size_t>(out_row * width);
    uint64_t oc = 0, cc = 0;  // output col + element count within the block
    for (uint64_t c = 0; c < cols && i < n; ++c, ++i) {
      const double v = read_elem(p.ptr, t.dtype, i);
      if (!std::isnan(v) && !std::isinf(v)) {
        const size_t idx = row_base + static_cast<size_t>(oc);
        sum[idx] += v;
        ++cnt[idx];
      }
      if (++cc == block_c) { cc = 0; if (oc + 1 < width) ++oc; }
    }
  }

  // --- Min/max over the DOWNSAMPLED grid's finite cells (display contrast) ----
  double gmin = std::numeric_limits<double>::infinity();
  double gmax = -std::numeric_limits<double>::infinity();
  for (size_t k = 0; k < cells; ++k) {
    if (cnt[k] == 0) continue;
    const double avg = sum[k] / static_cast<double>(cnt[k]);
    if (avg < gmin) gmin = avg;
    if (avg > gmax) gmax = avg;
  }
  if (gmin > gmax) { gmin = 0.0; gmax = 0.0; }  // no finite cell anywhere
  thumb.slice_min = gmin;
  thumb.slice_max = gmax;
  const double range = gmax - gmin;

  // --- Normalize + pack RGBA8 via a simple blue->red ramp --------------------
  auto to_u8 = [](double x) -> uint8_t {
    long q = std::lround(x * 255.0);
    if (q < 0) q = 0;
    if (q > 255) q = 255;
    return static_cast<uint8_t>(q);
  };
  thumb.rgba.assign(cells * 4, static_cast<uint8_t>(0));
  for (size_t k = 0; k < cells; ++k) {
    uint8_t r8, g8, b8;
    if (cnt[k] == 0) {
      r8 = g8 = b8 = 128;  // block with no finite values -> mid-gray
    } else {
      const double avg = sum[k] / static_cast<double>(cnt[k]);
      const double t01 = (range > 0.0) ? (avg - gmin) / range : 0.5;  // guard
      r8 = to_u8(t01);                                // R rises 0->1
      g8 = to_u8(1.0 - std::fabs(2.0 * t01 - 1.0));   // G peaks at mid
      b8 = to_u8(1.0 - t01);                          // B falls 1->0
    }
    thumb.rgba[k * 4 + 0] = r8;
    thumb.rgba[k * 4 + 1] = g8;
    thumb.rgba[k * 4 + 2] = b8;
    thumb.rgba[k * 4 + 3] = 255;
  }

  thumb.width = static_cast<uint32_t>(width);
  thumb.height = static_cast<uint32_t>(height);
  thumb.available = true;
  return thumb;
}

// The preview buffer (QuantBlockPreview::values) is sized for exactly one
// dequant_block() call; pin it to GgufBlocks' own bound so a future widened
// layout can't silently overflow it (mirrors the static_assert in
// GgufBlocks.cpp that pins kQK to the same constant).
static_assert(kQuantPreviewMaxElems == gguf::kMaxDequantBlockElems,
             "QuantBlockPreview::values must match GgufBlocks' output bound");

Result<QuantBlockPreview> preview_quant_block(const ir::TensorRef& t,
                                              const MappedFile& base,
                                              uint32_t block_index,
                                              const std::string& model_dir,
                                              const ir::Model* model) {
  auto pr = resolve_payload(t, base, model_dir, model);
  if (!pr) return pr.error();
  Payload p = std::move(*pr);

  // PAYLOAD READ: single accounted access for this preview, same convention as
  // the other entry points (spec §2.1) — resolving the payload is what the
  // counting test observes; the actual bytes TOUCHED below are bounded to one
  // block regardless (that bound is the entire point of #49, see the header).
  ByteReader::mark_payload_read();

  QuantBlockPreview preview;  // available=false, empty reason, by default.
  preview.block_index = block_index;

  // #49 scope: quant_type_id is parser-specific (ir::TensorRef doc comment) —
  // for a format other than GGUF it means nothing, and GgufBlocks only speaks
  // ggml ids. Both checks below must pass before the id is trusted.
  if (t.quant_type_id == UINT32_MAX) {
    preview.unavailable_reason =
        "no exact quantization type was recorded for this tensor";
    return preview;
  }
  if (!model || model->str(model->format_name) != "GGUF") {
    preview.unavailable_reason =
        "single-block preview supports only GGUF's ggml quantization types";
    return preview;
  }

  const gguf::GgmlBlockLayout layout = gguf::ggml_block_layout(t.quant_type_id);
  // Same enum drives both lookups, so an id this build doesn't know yields an
  // empty name here exactly when it yields block_bytes==0 below — consistent,
  // not a coincidence.
  preview.type_name = std::string(gguf::ggml_type_name(t.quant_type_id));

  if (layout.block_bytes == 0) {
    preview.unavailable_reason = std::string(
        gguf::dequant_status_message(gguf::DequantStatus::UnknownType));
    return preview;
  }
  if (!layout.quantized) {
    preview.unavailable_reason = std::string(
        gguf::dequant_status_message(gguf::DequantStatus::NotQuantized));
    return preview;
  }

  // Blocks available from the RESOLVED payload length, via DIVISION (never
  // multiplication) — house rule (spec hostile-input discipline): a
  // short/truncated payload naturally floors to fewer blocks, so a stale or
  // hostile block_index fails the bound check just below instead of forming
  // an out-of-range pointer.
  preview.total_blocks = p.len / layout.block_bytes;

  if (!layout.dequant_supported) {
    // Known geometry (K-quant / IQ* super-block) but decode is deliberately
    // out of scope (#49) — still return the honest type_name + total_blocks
    // so the panel can say "Q4_K, 1024 blocks, preview unsupported".
    preview.unavailable_reason = std::string(
        gguf::dequant_status_message(gguf::DequantStatus::UnsupportedLayout));
    return preview;
  }

  if (block_index >= preview.total_blocks) {
    preview.unavailable_reason = "block index out of range for this tensor";
    return preview;
  }

  // SAFE: block_index < total_blocks == p.len / block_bytes (floor division),
  // so block_index * block_bytes < p.len for any block_bytes > 0 — the product
  // cannot overflow uint64_t (block_bytes is a handful of bytes; p.len already
  // fits in uint64_t) and the resulting pointer + avail stay inside the
  // resolved payload without needing a separate saturating-multiply guard.
  const uint64_t byte_off = static_cast<uint64_t>(block_index) *
                            static_cast<uint64_t>(layout.block_bytes);
  const uint8_t* block_ptr = p.ptr + byte_off;
  const uint64_t avail = p.len - byte_off;

  uint32_t elem_count = 0;
  const gguf::DequantStatus st = gguf::dequant_block(
      t.quant_type_id, block_ptr, avail, preview.values.data(), &elem_count);
  if (st != gguf::DequantStatus::Ok) {
    // Should not happen given the checks above (avail >= block_bytes is
    // guaranteed by the total_blocks/block_index bound), but dequant_block is
    // the authority on its own preconditions — surface whatever it says
    // rather than assume success.
    preview.unavailable_reason = std::string(gguf::dequant_status_message(st));
    return preview;
  }

  preview.available = true;
  preview.elem_count = elem_count;
  preview.first_elem =
      static_cast<uint64_t>(block_index) * layout.elems_per_block;
  return preview;
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
