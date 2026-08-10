// engine/TensorStats.h — lazy weight decode -> streaming stats + NPY export.
//
// DECISION (spec §7.5, §2.1): the ONLY place a tensor payload is read. Stats are
// computed streaming (one pass, no converted copy materialized) so inspecting a
// 500 MB tensor never allocates 500 MB or blocks the UI. Runs as TensorDecodeJob
// on a worker; the inspector shows a progress state until done.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"

namespace netvis {

constexpr int kHistogramBuckets = 64;

// #46: per-output-channel accumulation is capped so a weight with a huge dim-0
// (e.g. a vocab-sized embedding) never allocates an unbounded per-channel vector
// or blows the decode budget. Beyond this, per_channel is left empty and
// per_channel_capped is set (the inspector shows whole-tensor stats only).
constexpr uint32_t kMaxChannels = 4096;

// #46: per-output-channel summary. Channel = index along dim 0 (weights are
// [Cout, ...] by convention). Streaming min/max/mean + a dead/NaN flag per
// channel for quant debugging.
struct ChannelStat {
  double min = 0, max = 0, mean = 0;
  uint64_t zero_count = 0;      // zeros in this channel
  uint64_t nan_inf_count = 0;   // NaN/Inf in this channel
  uint64_t count = 0;           // elements in this channel (== elems_per_channel)
  bool all_zero() const { return count > 0 && zero_count == count; }  // #48 dead
  bool has_nan_inf() const { return nan_inf_count > 0; }              // #48
};

// One-pass summary of a tensor's values (spec §7.5).
struct TensorStats {
  double min = 0, max = 0, mean = 0, std = 0;
  uint64_t zero_count = 0;
  uint64_t nan_inf_count = 0;
  uint64_t count = 0;                       // elements scanned
  std::array<uint64_t, kHistogramBuckets> histogram{};
  double hist_min = 0, hist_max = 0;        // histogram range
  bool quantized_unsupported = false;       // GGUF Q* -> metadata only (spec §7.5)

  // --- v0.8.3 additions (append-only) ---------------------------------------
  // #51: flat index of the first minimum / maximum element (argmin/argmax over
  // finite values), recorded during the streaming pass. UINT64_MAX if no finite
  // element was scanned (empty / all-NaN). The inspector maps a flat index back
  // to a multi-dim coordinate via the tensor shape for "jump to".
  uint64_t min_index = UINT64_MAX;
  uint64_t max_index = UINT64_MAX;

  // #48 whole-tensor outlier flags (derived from the counts above; convenience so
  // the inspector doesn't re-derive them). all_zero: every element is 0. has_*: at
  // least one such element.
  bool all_zero() const { return count > 0 && zero_count == count; }
  bool has_nan_inf() const { return nan_inf_count > 0; }

  // #46: per-output-channel stats (dim-0 slices), or EMPTY when the tensor is
  // rank<1, has one channel, or exceeds kMaxChannels (see per_channel_capped).
  std::vector<ChannelStat> per_channel;
  bool per_channel_capped = false;  // true => too many channels; per_channel empty
};

// Decode a tensor's payload and compute stats. `base` is the model's mmap;
// `model_dir` resolves ONNX external_data; `model` resolves StringId in
// external_path (pass nullptr if external_path is already a path string).
// This calls ByteReader::mark_payload_read() exactly once (tests assert
// structural parse left the counter at 0). Streams in chunks; never
// materializes a copy.
Result<TensorStats> compute_tensor_stats(const ir::TensorRef& t,
                                         const MappedFile& base,
                                         const std::string& model_dir,
                                         const ir::Model* model = nullptr);

// #47: a 2D heatmap thumbnail of a tensor slice, produced by streaming decode.
// The last two dims are taken as the [rows, cols] image plane; higher dims are
// fixed at index 0 (the first 2D slice). Values are normalized to [0,1] over the
// slice's own [min,max] and packed as RGBA8 via a simple blue->red ramp so the
// view can upload it straight to a texture without a second pass. Dimensions are
// capped (kThumbMax per side) — a larger slice is BLOCK-averaged down so a huge
// weight never allocates a huge texture. Never materializes the whole tensor.
constexpr uint32_t kThumbMax = 128;

struct TensorThumbnail {
  uint32_t width = 0, height = 0;      // <= kThumbMax each; 0 => unavailable
  std::vector<uint8_t> rgba;           // width*height*4, row-major, or empty
  double slice_min = 0, slice_max = 0; // the normalization range used
  bool available = false;              // false => quant/unknown/rank<2/dynamic
};

// Compute a 2D heatmap thumbnail for `t`. `base`/`model_dir`/`model` resolve the
// payload exactly like compute_tensor_stats. Returns available=false (not an
// error) for quantized/unknown dtypes, rank<2, or dynamic/oversized-unresolvable
// dims. Calls ByteReader::mark_payload_read() once. Streams; no full copy.
Result<TensorThumbnail> compute_tensor_thumbnail(const ir::TensorRef& t,
                                                 const MappedFile& base,
                                                 const std::string& model_dir,
                                                 const ir::Model* model = nullptr);

// --- #49: opt-in, view-only single-block dequant preview ---------------------
//
// SCOPE (frozen — see DECISIONS.md "v0.9.1b — dequantization scope"): this is the
// ONLY dequantization entry point in NetVis, it decodes exactly ONE block, it
// returns the values by value in a fixed-size buffer, and nothing may persist or
// export what it returns. It exists so a user inspecting a quantized tensor can
// see real numbers behind the metadata, not so anything downstream can consume a
// dequantized tensor. It reads at most one block's bytes — strictly fewer than
// the whole-tensor histogram pass the inspector already runs.
//
// Only the five legacy GGUF layouts (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) decode; K-quants
// and the IQ* family report available=false with an honest reason rather than an
// approximation. See parsers/gguf/GgufBlocks.h for why.
constexpr uint32_t kQuantPreviewMaxElems = 32;

struct QuantBlockPreview {
  bool available = false;
  std::string type_name;    // exact source quant type, e.g. "Q4_0"; "" if unknown
  uint32_t block_index = 0; // which block was decoded
  uint64_t total_blocks = 0;// blocks in the tensor (0 if the geometry is unknown)
  uint32_t elem_count = 0;  // floats written to `values`
  std::array<float, kQuantPreviewMaxElems> values{};
  // The flat element index `values[0]` corresponds to, so the inspector can show
  // where in the tensor the preview came from. block_index * elems_per_block.
  uint64_t first_elem = 0;
  // Non-empty whenever available == false: a fixed, honest explanation
  // ("Q4_K uses 256-element super-blocks; preview not supported", ...). Never a
  // guess and never a partially-decoded result.
  std::string unavailable_reason;
};

// Decode one block of `t` for preview. `base`/`model_dir`/`model` resolve the
// payload exactly like compute_tensor_stats (including ONNX external data and
// the CoreML blob_indirect header). `block_index` is clamped-checked against the
// tensor's block count; an out-of-range index yields available=false, not an
// error. Calls ByteReader::mark_payload_read() once. Returns an error Result only
// for an unreadable mapping — an unsupported LAYOUT is a successful call with
// available=false, so the caller can render the reason.
Result<QuantBlockPreview> preview_quant_block(const ir::TensorRef& t,
                                              const MappedFile& base,
                                              uint32_t block_index,
                                              const std::string& model_dir,
                                              const ir::Model* model = nullptr);

// Export a tensor to NumPy .npy (v1.0 header, spec §7.5) or raw .bin.
// Reads the payload from the mmap/external file; writes to `out_path`.
// Pass `model` to resolve StringId in external_path (nullptr if not needed).
Result<bool> export_npy(const ir::TensorRef& t, const MappedFile& base,
                        const std::string& model_dir, const std::string& out_path,
                        const ir::Model* model = nullptr);
Result<bool> export_raw(const ir::TensorRef& t, const MappedFile& base,
                        const std::string& model_dir, const std::string& out_path,
                        const ir::Model* model = nullptr);

}  // namespace netvis
