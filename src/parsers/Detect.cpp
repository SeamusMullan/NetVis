// parsers/Detect.cpp — content-based format detection and parse dispatch.
//
// DECISION (spec §5): detection is by CONTENT (magic bytes / structure), with
// the file extension used only as a tiebreaker. Every check reads through a
// small local view of the mapped bytes; nothing here touches tensor payloads.
#include "parsers/Parser.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#include "core/ByteReader.h"
#include "engine/plugin/Registry.h"   // v0.7.0 #10: WASM parser fallback (Unknown branch)

namespace netvis {

const char* format_name(Format f) {
  switch (f) {
    case Format::ONNX:          return "ONNX";
    case Format::TFLite:        return "TFLite";
    case Format::SafeTensors:   return "SafeTensors";
    case Format::GGUF:          return "GGUF";
    case Format::PyTorchZip:    return "PyTorch";
    case Format::PyTorchLegacy: return "PyTorch";
    case Format::OpenVINO:      return "OpenVINO";
    case Format::Npz:           return "NumPy npz";
    case Format::Keras:         return "Keras";
    case Format::CoreML:        return "CoreML";
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

// ---- ZIP central-directory peek (shared by .npz / .keras / PyTorch) --------
// All three begin with the local-file-header magic "PK\x03\x04", so magic alone
// cannot tell them apart. We locate the End Of Central Directory record and scan
// the central directory filenames, testing each against a predicate. Everything
// is bounds-checked; a truncated/absurd zip simply yields "no match" (never a
// crash, never an over-read). This is a bounded structural scan, not a full zip
// parse — enough to classify the archive.
struct ZipFlags {
  bool any_npy = false;         // an entry ends in ".npy"        -> npz
  bool keras_config = false;    // has "config.json"             -> keras v3
  bool keras_weights = false;   // has "model.weights.h5"        -> keras v3
  bool pytorch_pkl = false;     // has "*/data.pkl" or "data.pkl" -> PyTorch
};

bool ends_with(std::string_view s, std::string_view suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::string_view basename_view(std::string_view s) {
  auto sl = s.find_last_of('/');
  return sl == std::string_view::npos ? s : s.substr(sl + 1);
}

// Scan the ZIP central directory for the filename signals above. Reads through
// ByteReader so any malformed offset/length is caught, not fatal.
ZipFlags scan_zip_names(const uint8_t* d, uint64_t size) {
  ZipFlags fl;
  constexpr uint32_t kEocdSig = 0x06054b50;   // "PK\x05\x06"
  constexpr uint32_t kCenSig  = 0x02014b50;   // "PK\x01\x02"
  // EOCD is within the last 64KiB+22 bytes; search backwards for its signature.
  if (size < 22) return fl;
  uint64_t max_back = size < (0x10000ULL + 22) ? size : (0x10000ULL + 22);
  uint64_t start = size - max_back;
  uint64_t eocd = UINT64_MAX;
  for (uint64_t p = size - 22; p + 22 <= size && p + 1 >= start + 1; --p) {
    uint32_t sig;
    std::memcpy(&sig, d + p, 4);
    if (sig == kEocdSig) { eocd = p; break; }
    if (p == start) break;
  }
  if (eocd == UINT64_MAX) return fl;

  ByteReader r(d, size);
  if (!r.seek(eocd + 10)) return fl;
  auto n_entries = r.u16le();
  if (!n_entries) return fl;
  if (!r.seek(eocd + 16)) return fl;
  auto cd_off = r.u32le();
  if (!cd_off) return fl;

  uint64_t off = *cd_off;
  uint32_t count = *n_entries;
  // Cap the scan so a bogus entry count can't spin.
  for (uint32_t i = 0; i < count && i < 4096; ++i) {
    if (!r.seek(off)) break;
    auto sig = r.u32le();
    if (!sig || *sig != kCenSig) break;
    if (!r.seek(off + 28)) break;
    auto fn_len = r.u16le();
    auto extra_len = r.u16le();
    auto comment_len = r.u16le();
    if (!fn_len || !extra_len || !comment_len) break;
    if (!r.seek(off + 46)) break;
    auto name = r.bytes(*fn_len);        // filename bytes are structural
    if (!name) break;
    std::string_view nm(*name);
    std::string_view bn = basename_view(nm);
    if (ends_with(nm, ".npy")) fl.any_npy = true;
    if (bn == "config.json") fl.keras_config = true;
    if (bn == "model.weights.h5") fl.keras_weights = true;
    if (bn == "data.pkl") fl.pytorch_pkl = true;
    off += 46ULL + *fn_len + *extra_len + *comment_len;
  }
  return fl;
}

// HDF5 superblock magic "\x89HDF\r\n\x1a\n". The superblock may sit at offset 0
// or an aligned 2^k*512 boundary; we check a bounded set of aligned offsets.
bool looks_like_hdf5(const uint8_t* d, uint64_t size) {
  static const uint8_t kMagic[8] = {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
  const uint64_t offs[] = {0, 512, 1024, 2048, 4096, 8192};
  for (uint64_t o : offs) {
    if (o + 8 <= size && std::memcmp(d + o, kMagic, 8) == 0) return true;
  }
  return false;
}

// OpenVINO IR sniff: XML text whose root element is <net ...> with a
// version="10|11" attribute. Scans a bounded prefix; tolerant of a leading
// <?xml?> declaration and whitespace/comments. Never a false positive on a bare
// .xml that isn't an IR (the "<net" + version gate).
bool looks_like_openvino_xml(const uint8_t* d, uint64_t size) {
  uint64_t scan = size < 2048 ? size : 2048;
  std::string_view sv(reinterpret_cast<const char*>(d), scan);
  auto net = sv.find("<net");
  if (net == std::string_view::npos) return false;
  // Require a version attribute of 10 or 11 somewhere in the root tag region.
  auto ver = sv.find("version=", net);
  if (ver == std::string_view::npos) return false;
  std::string_view rest = sv.substr(ver + 8);
  // Skip an opening quote.
  if (!rest.empty() && (rest[0] == '"' || rest[0] == '\'')) rest = rest.substr(1);
  return rest.rfind("10", 0) == 0 || rest.rfind("11", 0) == 0;
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

  // ZIP-based formats all start with the local-file-header magic "PK\x03\x04":
  // NumPy .npz, Keras v3 .keras, and PyTorch .pt/.pth. Disambiguate by scanning
  // the central-directory filenames (bounded, bounds-checked).
  if (size >= 4 && std::memcmp(d, "PK\x03\x04", 4) == 0) {
    ZipFlags fl = scan_zip_names(d, size);
    // PyTorch is the most specific signal (a data.pkl object graph); prefer it.
    if (fl.pytorch_pkl) return Format::PyTorchZip;
    if (fl.keras_config && fl.keras_weights) return Format::Keras;
    if (fl.any_npy) return Format::Npz;
    // Ambiguous zip: fall through to the extension tiebreaker below.
    if (ext_hint == "npz") return Format::Npz;
    if (ext_hint == "keras") return Format::Keras;
    if (ext_hint == "pt" || ext_hint == "pth" || ext_hint == "bin")
      return Format::PyTorchZip;
    // Unknown zip contents: default to PyTorch zip (its parser errors cleanly
    // if there is no data.pkl) rather than mis-claiming a tensor format.
    return Format::PyTorchZip;
  }

  // Keras legacy / raw HDF5 (.h5): superblock magic at an aligned offset.
  if (looks_like_hdf5(d, size)) return Format::Keras;

  // OpenVINO IR: XML text with a <net version="10|11"> root element.
  if (looks_like_openvino_xml(d, size)) return Format::OpenVINO;

  // Legacy pickle: opcode PROTO (0x80) followed by protocol byte 2..5.
  if (size >= 2 && d[0] == 0x80 && d[1] >= 0x02 && d[1] <= 0x05) {
    return Format::PyTorchLegacy;
  }

  // CoreML .mlmodel is a bare `Model` protobuf whose first field
  // (specificationVersion, a field-1 varint) structurally mimics ONNX's
  // ir_version, so a bare .mlmodel also satisfies looks_like_onnx_proto. The
  // two are otherwise ambiguous, so the `.mlmodel` extension is the decisive
  // tiebreaker (spec §5): a file carrying it routes to CoreML before the ONNX
  // structural sniff below could claim it.
  if (ext_hint == "mlmodel") return Format::CoreML;

  // ONNX: plausible top-level protobuf ModelProto. Runs after the .mlmodel
  // guard because a bare CoreML protobuf would otherwise be misread as ONNX;
  // ONNX carries the ir_version/graph structural signal for extension-less
  // protobufs.
  if (looks_like_onnx_proto(d, size)) return Format::ONNX;

  // Extension tiebreaker for ambiguous content.
  if (!ext_hint.empty()) {
    if (ext_hint == "onnx") return Format::ONNX;
    if (ext_hint == "tflite") return Format::TFLite;
    if (ext_hint == "safetensors") return Format::SafeTensors;
    if (ext_hint == "gguf") return Format::GGUF;
    if (ext_hint == "xml") return Format::OpenVINO;
    if (ext_hint == "npz") return Format::Npz;
    if (ext_hint == "keras" || ext_hint == "h5" || ext_hint == "hdf5")
      return Format::Keras;
    if (ext_hint == "mlmodel") return Format::CoreML;
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
    case Format::OpenVINO:      return openvino::parse(file, progress);
    case Format::Npz:           return npz::parse(file, progress);
    case Format::Keras:         return keras::parse(file, progress);
    case Format::CoreML:        return coreml::parse(file, progress);
    case Format::Unknown:
      // v0.7.0 (#10): a file no built-in format claimed may still be handled by an
      // enabled WASM parser plugin (structurally absent when disabled, §0.4). Only
      // reached AFTER every built-in sniff failed, so a plugin never hijacks a known
      // format.
      if (auto r = plugin::Registry::instance().try_unknown_parsers(file, ext_hint, progress))
        return std::move(*r);
      return err("unrecognized model file format", UINT64_MAX);
  }
  return err("unrecognized model file format", UINT64_MAX);
}

}  // namespace netvis
