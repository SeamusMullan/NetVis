// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeSniff.h — content sniffs for the two Caffe files (#138).
//
// Used by BOTH parsers/Detect.cpp (format detection) and CaffeParser.cpp (which
// front-end to run), so detection and parsing can never disagree. All three are
// pure, bounded and allocation-free on the accept path: the text sniff reads at
// most 4 KiB + 64 KiB / 4096 tokens through TextLexer, the binary sniff at most
// 64 top-level tags + one 32-tag peek through onnx::WireReader, and neither ever
// touches a length-delimited payload's bytes.
#pragma once

#include <cstdint>

namespace netvis::caffe {

// True if the first min(n, 4096) bytes contain no byte in 0x00-0x08, 0x0E-0x1F
// or 0x7F (TAB, LF, VT, FF, CR and UTF-8 bytes >= 0x80 are fine). Every binary
// NetParameter with a layer contains such a byte (the `layer` / `layers` tags).
bool is_text_prefix(const uint8_t* d, uint64_t n);

// Text NetParameter: only NetParameter field names at the top level, and the
// first `layer` / `layers` block carries a plausible `type` (a modern type
// string, a V1 LayerType identifier / number, or a V0 `layer { }`).
bool looks_like_prototxt(const uint8_t* d, uint64_t n);

// Binary NetParameter: only NetParameter field numbers with their schema wire
// types at the top level (a field-1 VARINT — ONNX ir_version, SavedModel
// schema_version, CoreML specificationVersion — is rejected), and the first
// layer peek finds a plausible type.
bool looks_like_caffemodel(const uint8_t* d, uint64_t n);

}  // namespace netvis::caffe
