// parsers/openvino/OpenVinoParser.cpp — OpenVINO IR (.xml + .bin) -> ir::Model.
//
// STUB (v0.5.0 spine): the spine wires Detect/dispatch to this entry point so
// the tree links and detection is testable ahead of the parser implementation
// (issue #39). Replaced by the real hand-rolled XML-subset parser.
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

namespace netvis::openvino {

Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  (void)file;
  progress.set(0.0f, "OpenVINO");
  return err("OpenVINO IR parser not yet implemented (#39)", UINT64_MAX);
}

}  // namespace netvis::openvino
