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
