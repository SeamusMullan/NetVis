// parsers/Detect.cpp — content-based format detection and parse dispatch.
//
// DECISION (spec §5): detection is by CONTENT (magic bytes / structure), with
// the file extension used only as a tiebreaker. Every check reads through a
// small local view of the mapped bytes; nothing here touches tensor payloads.
#include "parsers/Parser.h"

#include <cstdint>
#include <cstring>
#include <string>

namespace netvis {

const char* format_name(Format f) {
  switch (f) {
    case Format::ONNX:          return "ONNX";
    case Format::TFLite:        return "TFLite";
    case Format::SafeTensors:   return "SafeTensors";
    case Format::GGUF:          return "GGUF";
    case Format::PyTorchZip:    return "PyTorch";
    case Format::PyTorchLegacy: return "PyTorch";
    case Format::Unknown:       return "Unknown";
  }
  return "Unknown";
}

namespace {

// Little-endian load helpers over a bounds-checked span. All detection reads go
// through these so a short/empty file can never over-run.
bool load_u64(const uint8_t* d, uint64_t size, uint64_t off, uint64_t& out) {
  if (off + 8 > size) return false;
  std::memcpy(&out, d + off, 8);
  return true;
}

// Try to read a base-128 varint at [off]. Returns false on overrun or if the
// varint is unterminated / longer than 10 bytes (max for 64-bit).
bool read_varint(const uint8_t* d, uint64_t size, uint64_t& off, uint64_t& out) {
  uint64_t v = 0;
  int shift = 0;
  uint64_t p = off;
  for (int i = 0; i < 10; ++i) {
    if (p >= size) return false;
    uint8_t b = d[p++];
    v |= (static_cast<uint64_t>(b & 0x7F) << shift);
    if ((b & 0x80) == 0) {
      off = p;
      out = v;
      return true;
    }
    shift += 7;
  }
  return false;
}

// Heuristic: does the buffer look like a top-level protobuf ModelProto?
// We scan a few top-level fields and require valid wire types with no overrun.
// ONNX ModelProto has field 1 (ir_version, varint) or field 7 (graph,
// length-delimited GraphProto). We accept if we see one of those plausibly.
bool looks_like_onnx_proto(const uint8_t* d, uint64_t size) {
  if (size < 2) return false;
  uint64_t off = 0;
  bool saw_signal = false;
  // Scan up to a handful of top-level fields.
  for (int i = 0; i < 12 && off < size; ++i) {
    uint64_t tag = 0;
    if (!read_varint(d, size, off, tag)) return false;
    uint32_t field = static_cast<uint32_t>(tag >> 3);
    uint32_t wire = static_cast<uint32_t>(tag & 0x7);
    if (field == 0) return false;  // field number 0 is invalid
    switch (wire) {
      case 0: {  // varint
        uint64_t v;
        if (!read_varint(d, size, off, v)) return false;
        if (field == 1) saw_signal = true;  // ir_version
        break;
      }
      case 1: {  // 64-bit
        if (off + 8 > size) return false;
        off += 8;
        break;
      }
      case 2: {  // length-delimited
        uint64_t len;
        if (!read_varint(d, size, off, len)) return false;
        if (off + len > size) return false;
        // graph (7), opset_import (8), metadata_props (14), producer (2/3),
        // etc. are all length-delimited in ModelProto.
        if (field == 7 || field == 8) saw_signal = true;
        off += len;
        break;
      }
      case 5: {  // 32-bit
        if (off + 4 > size) return false;
        off += 4;
        break;
      }
      default:
        return false;  // deprecated/invalid wire types -> not protobuf
    }
    if (saw_signal) return true;
  }
  return saw_signal;
}

}  // namespace

Format detect_format(const MappedFile& file, const std::string& ext_hint) {
  const uint8_t* d = file.data();
  const uint64_t size = file.size();
  if (d == nullptr || size == 0) return Format::Unknown;

  // GGUF: magic "GGUF" at byte 0.
  if (size >= 4 && std::memcmp(d, "GGUF", 4) == 0) return Format::GGUF;

  // TFLite: flatbuffer file identifier "TFL3" at bytes [4..8).
  if (size >= 8 && std::memcmp(d + 4, "TFL3", 4) == 0) return Format::TFLite;

  // SafeTensors: u64 LE header length N at 0, then 8+N <= filesize and the JSON
  // header (after optional whitespace) begins with '{'.
  {
    uint64_t n;
    if (load_u64(d, size, 0, n) && n > 0 && 8 + n <= size) {
      uint64_t p = 8;
      while (p < 8 + n && (d[p] == ' ' || d[p] == '\t' ||
                           d[p] == '\r' || d[p] == '\n')) {
        ++p;
      }
      if (p < 8 + n && d[p] == '{') return Format::SafeTensors;
    }
  }

  // PyTorch zip-based (.pt/.pth/.bin): local file header magic "PK\x03\x04".
  if (size >= 4 && std::memcmp(d, "PK\x03\x04", 4) == 0) return Format::PyTorchZip;

  // Legacy pickle: opcode PROTO (0x80) followed by protocol byte 2..5.
  if (size >= 2 && d[0] == 0x80 && d[1] >= 0x02 && d[1] <= 0x05) {
    return Format::PyTorchLegacy;
  }

  // ONNX: plausible top-level protobuf ModelProto.
  if (looks_like_onnx_proto(d, size)) return Format::ONNX;

  // Extension tiebreaker for ambiguous content.
  if (!ext_hint.empty()) {
    if (ext_hint == "onnx") return Format::ONNX;
    if (ext_hint == "tflite") return Format::TFLite;
    if (ext_hint == "safetensors") return Format::SafeTensors;
    if (ext_hint == "gguf") return Format::GGUF;
    if (ext_hint == "pt" || ext_hint == "pth" || ext_hint == "bin") {
      return Format::PyTorchZip;
    }
    if (ext_hint == "pkl" || ext_hint == "pickle") return Format::PyTorchLegacy;
  }

  return Format::Unknown;
}

Result<ir::Model> parse_model(const MappedFile& file, const std::string& ext_hint,
                              ProgressSink& progress) {
  Format f = detect_format(file, ext_hint);
  switch (f) {
    case Format::ONNX:          return onnx::parse(file, progress);
    case Format::TFLite:        return tflite::parse(file, progress);
    case Format::SafeTensors:   return safetensors::parse(file, progress);
    case Format::GGUF:          return gguf::parse(file, progress);
    case Format::PyTorchZip:    return pytorch::parse_zip(file, progress);
    case Format::PyTorchLegacy: return pytorch::parse_legacy(file, progress);
    case Format::Unknown:
      return err("unrecognized model file format", UINT64_MAX);
  }
  return err("unrecognized model file format", UINT64_MAX);
}

}  // namespace netvis
