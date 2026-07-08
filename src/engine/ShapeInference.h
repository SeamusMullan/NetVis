// engine/ShapeInference.h — best-effort ONNX shape/dtype propagation.
//
// DECISION (spec §7.3): many ONNX files omit intermediate value_info. We
// propagate shapes for the common ops so edge labels fill in. Best-effort:
// unknown ops leave outputs Unknown and never fail the job. Runs on a worker
// thread; the view refreshes edge labels when the ShapeInferJob completes.
//
// This MUTATES the ValueInfo.shape/dtype fields of a graph in place. Ownership:
// the job operates on a snapshot the main thread is not reading; the enriched
// model is republished on completion (generation-checked).
#pragma once

#include <cstdint>

#include "core/JobSystem.h"
#include "ir/IR.h"

namespace netvis {

// Propagate shapes/dtypes over graphs[graph_index] in place. Returns the number
// of values whose shape became known. Only meaningful for ONNX models.
uint32_t infer_shapes(ir::Model& model, uint32_t graph_index,
                      ProgressSink* progress = nullptr);

}  // namespace netvis
