// engine/plugin/ParserPlugin.h — FROZEN parser-extension ABI (v0.6.0, issue #7).
//
// A ParserPlugin turns one whole file into an ir::Model, run ONCE on a worker
// thread. Built-in parsers (onnx/tflite/.../coreml) register as ParserPlugins on
// the SAME path a WASM parser would (dogfood). The WASM backend supplies an
// ADAPTER implementing this interface and marshalling across wasm3; no C++ object
// crosses the sandbox. Signature MIRRORS the real, precedent-frozen parser form
// `Result<ir::Model> parse(const MappedFile&, ProgressSink&)` — there is no
// HostFileView/ModelBuilder/ParseResult in the codebase; inventing them would fork
// every existing parser. [plan-correction]
//
// ZERO-PAYLOAD OBLIGATION (frozen contract, tested per plugin): the returned Model
// carries only TensorRef{offset,len,...} — the IR has NO decoded-payload field, so
// a plugin structurally cannot inject weight buffers into the view. A WASM parser's
// host_read_range is confined to declared-structure ranges and REJECTED (or marks
// the payload counter) on any read overlapping a recorded tensor range; the
// payload_read_counter()==0 test remains meaningful only because the HOST, not the
// plugin, owns that marking. See docs/v0.6.0-design.md Part 5.2.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/JobSystem.h"    // ProgressSink
#include "core/MappedFile.h"   // MappedFile
#include "core/Result.h"       // Result<T>
#include "ir/IR.h"             // ir::Model
#include "parsers/Parser.h"    // Format

namespace netvis::plugin {

inline constexpr uint32_t kParserPluginAbiVersion = 1;

class ParserPlugin {
 public:
  virtual ~ParserPlugin() = default;

  // The Format this plugin emits. A third-party format not in the APPEND-ONLY
  // Format enum returns Format::Unknown and self-labels via ir::Model::format_name
  // (a StringId); the enum is NEVER renumbered/extended per plugin. Built-ins
  // return their real Format.
  virtual Format format() const = 0;
  virtual std::string_view display_name() const = 0;

  // CONTENT sniff over the WHOLE mapped file (real sniffers need it: EOCD at the
  // file end, HDF5 probes to offset 8192, SafeTensors u64-len+JSON). NOT a bounded
  // head buffer. Pure, bounded, no payload read.
  virtual bool can_parse(const MappedFile& file,
                         const std::string& ext_hint) const = 0;

  // Priority for the content-sniff phase (lower = sniffed earlier). Reproduces
  // detect_format's fixed order (design Part 4.4). User plugins default BELOW
  // built-ins so a plugin cannot hijack a known format.
  virtual int priority() const = 0;

  // Parse whole file -> ir::Model. IDENTICAL signature to every existing parser.
  // Records offset+len only (inv 3); returns an error Result on malformed input
  // (inv 5); never throws.
  virtual Result<ir::Model> parse(const MappedFile& file,
                                  ProgressSink& progress) const = 0;

  virtual uint32_t api_version() const = 0;
};

}  // namespace netvis::plugin
