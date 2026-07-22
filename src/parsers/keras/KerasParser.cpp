// parsers/keras/KerasParser.cpp — Keras .h5 / .keras -> ir::Model.
//
// STUB (v0.5.0 spine): entry point wired so the tree links and detection is
// testable ahead of the parser implementation (issue #42). Replaced by the real
// minimal-HDF5 / keras-v3-zip reader (tensor-table mode, offset+len only).
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

namespace netvis::keras {

Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  (void)file;
  progress.set(0.0f, "Keras");
  return err("Keras .h5/.keras parser not yet implemented (#42)", UINT64_MAX);
}

}  // namespace netvis::keras
