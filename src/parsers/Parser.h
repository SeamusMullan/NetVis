// SPDX-License-Identifier: Apache-2.0
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
#include <string_view>

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
  // v0.5.0 format-breadth additions. APPEND ONLY — detection & any persisted
  // state key off the format NAME, but never renumber the existing values.
  OpenVINO,       // .xml topology + sibling .bin weight blob
  Npz,            // NumPy .npz — zip of .npy arrays
  Keras,          // .h5 / .keras (HDF5, or keras-v3 zip)
  CoreML,         // .mlmodel — CoreML Model protobuf
  // v0.9.5 Pillar 5 (#107).
  TensorFlow,     // frozen .pb (GraphDef) + SavedModel (saved_model.pb)
};

const char* format_name(Format f);

// Every file extension (lowercased, no dot) detect_format() acts on, and the
// Format it breaks an otherwise-ambiguous file toward (spec §5). This is the ONE
// place the detector's extension knowledge lives: the extension tiebreak at the
// end of detect_format() walks this table, and the file chooser's openable list
// (view/FileDialog.h) is built from it, so a format that parses can never be
// unpickable from the GUI. First match wins.
//
// detect_format() consults an extension in three places, all routed through
// names declared here so none can hide a literal:
//   1. kCoreMLExtension - one of the two signals of the early CoreML guard
//      (content via looks_like_coreml, OR this extension; a bare CoreML protobuf
//      looks like ONNX, so the guard decides before the ONNX sniff);
//   2. kZipExtensionFormats - the tiebreak for a zip no content signal claimed;
//   3. kExtensionFormats - the final tiebreak for everything else.
// Every extension used by 1 and 2 must also appear in this table, with the same
// Format (tests/test_detect.cpp enforces it), and tests/test_detect.cpp checks
// that each one routes the way it says.
struct ExtensionFormat {
  std::string_view ext;
  Format format;
};

// The CoreML `.mlmodel` extension; see (1) above.
inline constexpr std::string_view kCoreMLExtension = "mlmodel";

inline constexpr ExtensionFormat kExtensionFormats[] = {
    {"onnx", Format::ONNX},
    {"tflite", Format::TFLite},
    {"safetensors", Format::SafeTensors},
    {"gguf", Format::GGUF},
    {"xml", Format::OpenVINO},
    {"npz", Format::Npz},
    {"keras", Format::Keras},
    {"h5", Format::Keras},
    {"hdf5", Format::Keras},
    {kCoreMLExtension, Format::CoreML},
    {"pb", Format::TensorFlow},
    {"pt", Format::PyTorchZip},
    {"pth", Format::PyTorchZip},
    {"bin", Format::PyTorchZip},
    {"pkl", Format::PyTorchLegacy},
    {"pickle", Format::PyTorchLegacy},
};

// The subset of extensions that break a tie for a ZIP archive whose central
// directory carries no content signal (see (2) above). It is deliberately
// narrower than kExtensionFormats: an `.h5` or `.onnx` name on a zip is not
// evidence of that format, so such a zip falls through to the content default
// (PyTorch zip) instead. A new zip-based format (say, TorchScript-Lite) adds its
// extension HERE and in kExtensionFormats; the test fails if an extension is in
// this table but missing from kExtensionFormats (the reverse is allowed: the main
// table is the wider one).
inline constexpr ExtensionFormat kZipExtensionFormats[] = {
    {"npz", Format::Npz},
    {"keras", Format::Keras},
    {"pt", Format::PyTorchZip},
    {"pth", Format::PyTorchZip},
    {"bin", Format::PyTorchZip},
};

// Detect the format from file content. `ext_hint` is a lowercased extension
// (without dot), used only to break ties (spec §5). Returns Format::Unknown if
// nothing matches.
Format detect_format(const MappedFile& file, const std::string& ext_hint);

// #45: how a format was decided, surfaced in the status bar so a user can see
// WHY NetVis picked a parser (a magic-byte match is trustworthy; an
// extension-only tiebreak on ambiguous content is worth flagging).
enum class DetectReason : uint8_t {
  None,          // Unknown / nothing matched
  Magic,         // a magic-byte / file-identifier match (high confidence)
  Structure,     // a bounded structural sniff matched (e.g. protobuf/XML shape)
  Extension,     // content ambiguous; the file extension broke the tie
  ContentDefault // content matched a family (e.g. a zip) but the specific format
                 // fell back to a default (lower confidence)
};
const char* detect_reason_name(DetectReason r);

// Same detection, additionally reporting WHY (#45). `reason` is set to the signal
// that decided the returned Format. The plain overload above delegates to this.
Format detect_format(const MappedFile& file, const std::string& ext_hint,
                     DetectReason& reason);

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
// v0.5.0 format-breadth parsers (one TU each under parsers/<fmt>/).
namespace openvino { Result<ir::Model> parse(const MappedFile&, ProgressSink&); }
namespace npz { Result<ir::Model> parse(const MappedFile&, ProgressSink&); }
namespace keras { Result<ir::Model> parse(const MappedFile&, ProgressSink&); }
namespace coreml { Result<ir::Model> parse(const MappedFile&, ProgressSink&); }
// v0.9.5 Pillar 5 parsers (#107). One entry point covers both TF containers: a
// frozen GraphDef and a SavedModel (whose meta_graphs[0].graph_def it unwraps).
namespace tensorflow { Result<ir::Model> parse(const MappedFile&, ProgressSink&); }

}  // namespace netvis
