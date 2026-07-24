// engine/plugin/wasm/WasmParser.h — WASM ParserPlugin host side (Increment B, #10).
//
// A WASM parser turns a whole file into an ir::Model via append-only COMMANDS over
// the frozen "netvis" parser import set (plugins/sdk/netvis_plugin.h). The HOST owns
// the Model; the guest never holds a host pointer.
//
// ZERO-PAYLOAD MECHANISM (design §0.1/§B.1 — the load-bearing redesign):
//   - host_read_range is confined to a small up-front sniff window (head+tail); a
//     read outside the window, or overlapping a recorded tensor range, returns -1.
//   - The host reads through the file's ByteReader and marks each returned byte, so
//     payload_read_counter() reflects ONLY the bounded structural window — never a
//     weight. The witness is "structural reads <= declared window", NOT counter==0.
//   - Names/metadata are supplied as (file_offset,len) ranges the HOST reads through
//     the same marked, window-bounded path (host_intern_range) — never from arbitrary
//     guest memory. At parse-end the host re-validates that no rendered string's
//     source range overlaps any recorded tensor range and rejects the Model if so.
//   - Tensors are declared by (offset,byte_len) only (host_record_tensor); zero bytes copied.
//
// LEAD MUST WIRE (§B.5 — kept out of this TU to stay disjoint from the op agent):
//   - Detect.cpp Format::Unknown branch -> Registry::try_unknown_parsers(...).
//   - Registry.{h,cpp}: try_unknown_parsers iterates registered parsers (priority
//     sorted), first can_parse() claimer parses. register_parser(make_wasm_parser(...)).
//   - Manifest.cpp: a "parser" sidecar section registers a WasmParserPlugin (gated, C).
// Compiles to safe stubs without NETVIS_ENABLE_WASM.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/plugin/ParserPlugin.h"

namespace netvis::plugin::wasm {

// Host-side parse limits (design §0.6). Every guest-supplied count is capped here
// AND validated before any memory access.
struct ParseLimits {
  uint32_t max_graphs = 64;
  uint32_t max_nodes = 5'000'000;
  uint32_t max_values = 10'000'000;
  uint32_t max_tensors = 5'000'000;
  uint32_t max_metadata = 4096;
  uint32_t max_intern_calls = 2'000'000;
};

// A WASM parser plugin backed by an immutable .wasm image exporting netvis_can_parse
// and netvis_parse (+ netvis_parser_abi_version). Self-labels its format via
// host_set_model_info, so format()==Format::Unknown and priority() sits below all
// built-ins (a plugin never hijacks a known format).
std::unique_ptr<ParserPlugin> make_wasm_parser(
    std::string plugin_name, std::shared_ptr<const std::vector<uint8_t>> image);

// Load a WASM parser sidecar (plugin.json + .wasm) and register it into the Registry
// (only invoked for ENABLED plugins — the gate is Increment C). Empty string on
// success, else a human-readable diagnostic. Defined in the .cpp; LEAD calls it.
std::string load_wasm_parser_plugin(const std::string& plugin_json_path);

}  // namespace netvis::plugin::wasm
