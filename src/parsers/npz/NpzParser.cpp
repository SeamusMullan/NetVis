// parsers/npz/NpzParser.cpp — NumPy .npz (zip of .npy arrays) -> ir::Model.
//
// STUB (v0.5.0 spine): entry point wired so the tree links and detection is
// testable ahead of the parser implementation (issue #41). Replaced by the real
// zip + .npy-header parser (tensor-table mode, offset+len only).
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

namespace netvis::npz {

Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  (void)file;
  progress.set(0.0f, "npz");
  return err("NumPy .npz parser not yet implemented (#41)", UINT64_MAX);
}

}  // namespace netvis::npz
