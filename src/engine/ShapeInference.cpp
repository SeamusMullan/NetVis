// engine/ShapeInference.cpp — frozen 3-arg entry point (spec §7.3).
//
// The full best-effort ONNX shape/dtype propagation lives in
// ShapeInferenceExt.cpp (mmap-base-aware). This frozen 3-arg form keeps its
// signature and simply delegates with base=nullptr, i.e. the pure structural
// subset that cannot read constant shape/index initializers. Callers that hold
// the model's mmap (ModelSession) call infer_shapes_ext directly to resolve the
// constant-driven ops (Reshape/Slice/Gather/...).
#include "engine/ShapeInference.h"

#include "engine/ShapeInferenceExt.h"

namespace netvis {

uint32_t infer_shapes(ir::Model& model, uint32_t graph_index,
                      ProgressSink* progress) {
  return infer_shapes_ext(model, graph_index, nullptr, 0, progress);
}

}  // namespace netvis
