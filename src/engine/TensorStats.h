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

#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"

namespace netvis {

constexpr int kHistogramBuckets = 64;

// One-pass summary of a tensor's values (spec §7.5).
struct TensorStats {
  double min = 0, max = 0, mean = 0, std = 0;
  uint64_t zero_count = 0;
  uint64_t nan_inf_count = 0;
  uint64_t count = 0;                       // elements scanned
  std::array<uint64_t, kHistogramBuckets> histogram{};
  double hist_min = 0, hist_max = 0;        // histogram range
  bool quantized_unsupported = false;       // GGUF Q* -> metadata only (spec §7.5)
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
