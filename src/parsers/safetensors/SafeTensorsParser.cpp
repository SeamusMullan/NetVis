// parsers/safetensors/SafeTensorsParser.cpp — SafeTensors (.safetensors) reader.
//
// File layout (all little-endian):
//   [0,8)        u64 N               header length
//   [8, 8+N)     JSON object         one entry per tensor + optional __metadata__
//   [8+N, EOF)   data section        raw tensor payloads, referenced by offset
//
// DECISION (spec §2.1, the product thesis): we parse ONLY the JSON header and
// record file_offset+byte_len for every tensor. The multi-GB data section is
// never touched here — it stays on disk (mmap) until the weight inspector reads
// it. Cold-opening a 5 GB model is therefore header-parse time, not I/O time.
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

#include "core/ByteReader.h"
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

namespace netvis::safetensors {
namespace {

using json = nlohmann::json;

// Map a SafeTensors dtype string to the IR DType. Unrecognized strings become
// DType::Unknown (still a valid tensor, just an unlabeled element type).
ir::DType map_dtype(const std::string& s) {
  if (s == "F32") return ir::DType::F32;
  if (s == "F16") return ir::DType::F16;
  if (s == "BF16") return ir::DType::BF16;
  if (s == "F64") return ir::DType::F64;
  if (s == "I8") return ir::DType::I8;
  if (s == "I16") return ir::DType::I16;
  if (s == "I32") return ir::DType::I32;
  if (s == "I64") return ir::DType::I64;
  if (s == "U8") return ir::DType::U8;
  if (s == "U16") return ir::DType::U16;
  if (s == "U32") return ir::DType::U32;
  if (s == "U64") return ir::DType::U64;
  if (s == "BOOL") return ir::DType::Bool;
  return ir::DType::Unknown;
}

// Extract a non-negative integer from a JSON number without throwing. Returns
// false if the value is not an integral non-negative number.
bool as_u64(const json& v, uint64_t& out) {
  if (v.is_number_unsigned()) {
    out = v.get<uint64_t>();
    return true;
  }
  if (v.is_number_integer()) {
    int64_t i = v.get<int64_t>();
    if (i < 0) return false;
    out = static_cast<uint64_t>(i);
    return true;
  }
  return false;
}

}  // namespace

// Parse a SafeTensors file into tensor-table mode (has_graph == false). Malformed
// JSON, a truncated header, or an out-of-range data offset all return a Result
// error carrying the byte offset where the problem was detected.
Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "SafeTensors header");

  const uint64_t file_size = file.size();
  ByteReader reader(file.data(), file_size);

  // 8-byte little-endian header length. Bounds-checked by ByteReader.
  auto n_res = reader.u64le();
  if (!n_res) return n_res.error();
  const uint64_t header_len = *n_res;

  // The JSON header must fit entirely within the file: [8, 8+N) <= file_size.
  // Guard against overflow of 8+N before comparing.
  if (header_len > file_size || 8 + header_len > file_size) {
    return err("safetensors header length exceeds file size", 0);
  }

  // The data section begins immediately after the header.
  const uint64_t data_base = 8 + header_len;

  // Read the header bytes (this is metadata, NOT tensor payload — bounded by N,
  // which is a tiny fraction of a large model). reader.bytes copies N bytes.
  auto json_res = reader.bytes(header_len);
  if (!json_res) return json_res.error();
  const std::string& json_text = *json_res;

  // Non-throwing parse: on malformed JSON we get a discarded value, never an
  // exception crossing the module boundary (spec §13).
  json doc = json::parse(json_text, /*cb=*/nullptr, /*allow_exceptions=*/false);
  if (doc.is_discarded() || !doc.is_object()) {
    return err("safetensors header is not valid JSON object", 8);
  }

  progress.set(0.4f, "SafeTensors tensors");

  ir::Model model;
  model.has_graph = false;  // tensor-table mode: no compute graph (spec §8.6)
  model.format_name = model.intern("SafeTensors");

  model.flat_tensors.reserve(doc.size());

  for (auto it = doc.begin(); it != doc.end(); ++it) {
    const std::string& key = it.key();
    const json& val = it.value();

    // "__metadata__" is a free-form string->string map surfaced into the model
    // metadata table, not a tensor.
    if (key == "__metadata__") {
      if (val.is_object()) {
        for (auto m = val.begin(); m != val.end(); ++m) {
          // Values are strings by spec; anything else is serialized compactly so
          // no information is silently dropped.
          const std::string mv =
              m.value().is_string() ? m.value().get<std::string>() : m.value().dump();
          model.metadata.emplace_back(model.intern(m.key()), model.intern(mv));
        }
      }
      continue;
    }

    // Every other key is a tensor descriptor object.
    if (!val.is_object()) {
      return err("safetensors tensor entry is not an object: " + key, 8);
    }
    if (!val.contains("dtype") || !val["dtype"].is_string()) {
      return err("safetensors tensor missing string dtype: " + key, 8);
    }
    if (!val.contains("shape") || !val["shape"].is_array()) {
      return err("safetensors tensor missing shape array: " + key, 8);
    }
    if (!val.contains("data_offsets") || !val["data_offsets"].is_array() ||
        val["data_offsets"].size() != 2) {
      return err("safetensors tensor missing [begin,end] data_offsets: " + key, 8);
    }

    uint64_t begin = 0, end = 0;
    if (!as_u64(val["data_offsets"][0], begin) ||
        !as_u64(val["data_offsets"][1], end)) {
      return err("safetensors data_offsets are not integers: " + key, 8);
    }
    if (begin > end) {
      return err("safetensors data_offsets begin > end: " + key, 8);
    }
    // Offsets are RELATIVE to the data section; validate the absolute range lies
    // within the file so the weight inspector can never read out of bounds.
    // SECURITY: `end` is an unbounded u64 from JSON, so `data_base + end` can
    // wrap. Bound `end` against the remaining bytes BEFORE adding (subtraction
    // can't overflow since data_base <= file_size), avoiding the wrap.
    if (end > file_size - data_base) {
      return err("safetensors tensor data range exceeds file size: " + key,
                 data_base);
    }

    ir::TensorRef t;
    t.name = model.intern(key);
    t.dtype = map_dtype(val["dtype"].get<std::string>());
    for (const json& d : val["shape"]) {
      // Dimensions are non-negative ints in practice; store integrally and let
      // elem_count() treat anything <0 as dynamic (0 elems).
      if (d.is_number_integer() || d.is_number_unsigned()) {
        t.shape.push_back(d.get<int64_t>());
      } else {
        return err("safetensors shape dimension is not an integer: " + key, 8);
      }
    }
    // Record offset+length only — payload is never read here (spec §2.1, §6).
    t.file_offset = data_base + begin;
    t.byte_len = end - begin;
    // external_path stays empty: SafeTensors payloads are always in-file.

    model.flat_tensors.push_back(std::move(t));
  }

  progress.set(1.0f, "SafeTensors done");
  return model;
}

}  // namespace netvis::safetensors
