// parsers/coreml/CoreMlParser.cpp — CoreML .mlmodel (protobuf) -> ir::Model.
//
// STUB (v0.5.0 spine): entry point wired so the tree links and detection is
// testable ahead of the parser implementation (issue #38). Replaced by the real
// protobuf walk (reusing onnx::WireReader), recording weight offset+len only.
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

namespace netvis::coreml {

Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  (void)file;
  progress.set(0.0f, "CoreML");
  return err("CoreML .mlmodel parser not yet implemented (#38)", UINT64_MAX);
}

}  // namespace netvis::coreml
