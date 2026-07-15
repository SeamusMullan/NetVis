// engine/ShapeInferenceExt.h — mmap-base-aware shape inference (v0.2.0).
//
// DECISION: the frozen 3-arg infer_shapes (ShapeInference.h) cannot read
// initializer payloads, so ops whose OUTPUT shape depends on a constant INPUT
// (Reshape/Slice/Gather/Expand/Tile/Resize with a shape/indices tensor) can't be
// resolved. This overload takes the mmap base+size so those handlers can read a
// shape initializer's raw bytes (offset/len already recorded by the parser) and
// resolve the output. It stays additive: the frozen 3-arg form remains and
// delegates here with base=nullptr. Same MUTATES-in-place, worker-thread,
// best-effort contract as the frozen version.
//
// LIMITATION: only initializers stored as raw_data (a contiguous mmap range) are
// readable. ONNX shape tensors packed into int64_data/int32_data protobuf fields
// have no recorded offset (parser sets file_offset=UINT64_MAX) and remain
// Unknown — best-effort, never a crash.
#pragma once

#include <cstddef>
#include <cstdint>

#include "core/JobSystem.h"
#include "ir/IR.h"

namespace netvis {

// Same as infer_shapes, but `mmap_base`/`mmap_size` (the model file mapping,
// e.g. session.file().data()/.size()) let constant-driven ops resolve. Pass
// base=nullptr for the pure structural subset (identical to the 3-arg form).
uint32_t infer_shapes_ext(ir::Model& model, uint32_t graph_index,
                          const uint8_t* mmap_base, size_t mmap_size,
                          ProgressSink* progress = nullptr);

}  // namespace netvis
