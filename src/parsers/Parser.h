// parsers/Parser.h — common parser interface + content-based format detection.
//
// DECISION (spec §3, §5): the view never touches parsers directly; it goes
// through ModelSession. Every parser exposes the same signature
//   Result<ir::Model> parse(const MappedFile&, ProgressSink&)
// and detection is by CONTENT (magic bytes / structure), with the file
// extension used only as a tiebreaker.
#pragma once

#include <cstdint>
#include <string>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"

namespace netvis {

enum class Format : uint8_t {
  Unknown,
  ONNX,
  TFLite,
  SafeTensors,
  GGUF,
  PyTorchZip,     // modern zip-based .pt/.pth/.bin
  PyTorchLegacy,  // standalone pickle
};

const char* format_name(Format f);

// Detect the format from file content. `ext_hint` is a lowercased extension
// (without dot), used only to break ties (spec §5). Returns Format::Unknown if
// nothing matches.
Format detect_format(const MappedFile& file, const std::string& ext_hint);

// Parse dispatch: detect then route to the right parser. Runs on a worker
// thread (never the UI thread). `progress` receives stage updates.
Result<ir::Model> parse_model(const MappedFile& file, const std::string& ext_hint,
                              ProgressSink& progress);

// --- Individual parser entry points (one per format module) -----------------
// Each lives in its own translation unit under parsers/<fmt>/.
namespace onnx { Result<ir::Model> parse(const MappedFile&, ProgressSink&); }
namespace tflite { Result<ir::Model> parse(const MappedFile&, ProgressSink&); }
namespace safetensors { Result<ir::Model> parse(const MappedFile&, ProgressSink&); }
namespace gguf { Result<ir::Model> parse(const MappedFile&, ProgressSink&); }
namespace pytorch {
Result<ir::Model> parse_zip(const MappedFile&, ProgressSink&);
Result<ir::Model> parse_legacy(const MappedFile&, ProgressSink&);
}  // namespace pytorch

}  // namespace netvis
