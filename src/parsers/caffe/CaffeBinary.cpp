// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeBinary.cpp — binary NetParameter (.caffemodel) -> NetRaw.
//
// DECISION (#138): a .caffemodel is protobuf wire data, so this walks it with the
// hand-rolled, bounds-checked onnx::WireReader (the same reuse as the TensorFlow
// and CoreML parsers) — every tag, varint and length is checked and truncation
// is a byte-offset error. Groups (wire 3/4) are rejected by the reader.
//
// ZERO PAYLOAD: a BlobProto's packed `data` / `double_data` chunk is skipped with
// read_len_delim() — a pointer advance that never touches the bytes — and only
// its absolute offset + length are recorded. `diff` is skipped the same way. An
// UNPACKED `data` (one tag per float, which Caffe never writes) is not walked:
// the BlobProto is abandoned at its first element, because walking every tag
// would page the whole payload in (§3.6).
//
// Recursion is schema-bounded (Net -> Layer -> *_param -> Filler/BlobShape),
// never data-driven, so depth stays <= 4.
#include <cassert>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "parsers/caffe/CaffeNet.h"
#include "parsers/caffe/CaffeSchema.h"
#include "parsers/onnx/WireReader.h"

namespace netvis::caffe {
namespace {

using onnx::SubRange;
using onnx::WireReader;
using onnx::WireType;

struct Ctx {
  ir::Model* model;      // nullptr in BlobsOnly mode: nothing is interned
  ReadMode mode;
  NetRaw& net;
};

Error bin_error(const std::string& what, uint64_t offset) {
  return Error("Caffe caffemodel: " + what, offset);
}

// A length-delimited string, capped BEFORE it is copied (read_string() would
// copy any length).
Result<std::string> read_capped_string(WireReader& r, uint64_t cap, const char* what) {
  const uint64_t at = r.abs_pos();
  auto sr = r.read_len_delim();
  if (!sr) return sr.error();
  if (sr->len > cap) return bin_error(std::string(what) + " is too long", at);
  return std::string(reinterpret_cast<const char*>(sr->ptr), static_cast<size_t>(sr->len));
}

Result<int64_t> as_int32(uint64_t raw, const char* field, uint64_t at) {
  // int32 is a sign-extended 10-byte varint when negative.
  const int64_t v = static_cast<int64_t>(raw);
  if (v < INT32_MIN || v > INT32_MAX)
    return bin_error(std::string("int32 field '") + field + "' out of range", at);
  return v;
}

// Decode a varint-carried scalar of `kind` into an int64 for the attribute.
Result<int64_t> varint_value(const FieldSpec& f, uint64_t raw, uint64_t at) {
  switch (f.kind) {
    case FieldKind::Int32:
    case FieldKind::Enum:
      return as_int32(raw, f.name, at);
    case FieldKind::Int64:
      return static_cast<int64_t>(raw);
    case FieldKind::UInt32:
      return static_cast<int64_t>(raw & 0xFFFFFFFFull);
    case FieldKind::UInt64:
      if (raw > static_cast<uint64_t>(INT64_MAX)) return INT64_MAX;
      return static_cast<int64_t>(raw);
    case FieldKind::Bool:
      return raw != 0 ? 1 : 0;
    default:
      return bin_error(std::string("field '") + f.name + "' has an unexpected encoding", at);
  }
}

Result<bool> emit_varint(const FieldSpec& f, uint64_t raw, uint64_t at,
                         const std::string& path, AttrSink& sink) {
  auto v = varint_value(f, raw, at);
  if (!v) return v.error();
  bool ok = false;
  if (f.kind == FieldKind::Enum) {
    const char* nm = f.enum_spec ? enum_name(*f.enum_spec, *v) : nullptr;
    ok = sink.add_string(path, nm ? std::string(nm) : std::to_string(*v), f.repeated);
  } else {
    ok = sink.add_int(path, *v, f.repeated);
  }
  if (!ok) return bin_error(std::string("too many values in repeated field '") + f.name + "'", at);
  return true;
}

Result<bool> decode_message(const Ctx& c, const SubRange& sr, const MessageSpec& spec,
                            const std::string& prefix, AttrSink& sink, int depth);

// Decode one field whose schema entry is `f` (the tag is already consumed).
Result<bool> decode_field(const Ctx& c, WireReader& r, const FieldSpec& f, WireType wt,
                          uint64_t at, const std::string& path, AttrSink& sink,
                          int depth) {
  const bool numeric_varint = f.kind == FieldKind::Int32 || f.kind == FieldKind::Int64 ||
                              f.kind == FieldKind::UInt32 || f.kind == FieldKind::UInt64 ||
                              f.kind == FieldKind::Bool || f.kind == FieldKind::Enum;
  if (f.kind == FieldKind::Message) {
    if (wt != WireType::LenDelim) {
      auto sk = r.skip_field(wt);
      if (!sk) return sk.error();
      ++c.net.undecoded_fields;
      return true;
    }
    auto sub = r.read_len_delim();
    if (!sub) return sub.error();
    if (f.message == nullptr) {
      if (!sink.add_string(f.name, "(not decoded)", false))
        return bin_error("too many attributes", at);
      return true;
    }
    return decode_message(c, *sub, *f.message, path + ".", sink, depth + 1);
  }
  if (f.kind == FieldKind::String) {
    if (wt != WireType::LenDelim) {
      auto sk = r.skip_field(wt);
      if (!sk) return sk.error();
      ++c.net.undecoded_fields;
      return true;
    }
    auto s = read_capped_string(r, kMaxBinaryStringBytes, f.name);
    if (!s) return s.error();
    if (!sink.add_string(path, *s, f.repeated))
      return bin_error(std::string("too many values in repeated field '") + f.name + "'", at);
    return true;
  }
  if (numeric_varint) {
    if (wt == WireType::Varint) {
      auto v = r.read_varint();
      if (!v) return v.error();
      return emit_varint(f, *v, at, path, sink);
    }
    if (wt == WireType::LenDelim && f.repeated) {   // packed
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      WireReader pr = WireReader::sub(*sub);
      while (!pr.at_end()) {
        const uint64_t el = pr.abs_pos();
        auto v = pr.read_varint();
        if (!v) return v.error();
        auto e = emit_varint(f, *v, el, path, sink);
        if (!e) return e.error();
      }
      return true;
    }
  } else if (f.kind == FieldKind::Float || f.kind == FieldKind::Double) {
    const bool dbl = f.kind == FieldKind::Double;
    const WireType want = dbl ? WireType::Fixed64 : WireType::Fixed32;
    auto one = [&](WireReader& rr) -> Result<double> {
      if (dbl) {
        auto u = rr.read_fixed64();
        if (!u) return u.error();
        double x;
        const uint64_t bits = *u;
        std::memcpy(&x, &bits, sizeof x);
        return x;
      }
      auto fl = rr.read_float();
      if (!fl) return fl.error();
      return static_cast<double>(*fl);
    };
    if (wt == want) {
      auto x = one(r);
      if (!x) return x.error();
      if (!sink.add_float(path, *x, f.repeated))
        return bin_error(std::string("too many values in repeated field '") + f.name + "'", at);
      return true;
    }
    if (wt == WireType::LenDelim && f.repeated) {   // packed
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      const uint64_t width = dbl ? 8 : 4;
      if (sub->len % width != 0)
        return bin_error(std::string("packed field '") + f.name + "' has a ragged length",
                         sub->offset);
      WireReader pr = WireReader::sub(*sub);
      while (!pr.at_end()) {
        auto x = one(pr);
        if (!x) return x.error();
        if (!sink.add_float(path, *x, true))
          return bin_error(std::string("too many values in repeated field '") + f.name + "'", at);
      }
      return true;
    }
  }
  // Wire type does not match the schema: protobuf treats it as unknown.
  auto sk = r.skip_field(wt);
  if (!sk) return sk.error();
  ++c.net.undecoded_fields;
  return true;
}

// Per-message counters for repeated-message indices ("[i]").
struct RepCounter {
  SmallVec<std::pair<uint32_t, uint32_t>, 4> seen;
  uint32_t next(uint32_t field) {
    for (auto& s : seen)
      if (s.first == field) return s.second++;
    seen.push_back({field, 1});
    return 0;
  }
};

std::string field_path(const std::string& prefix, const FieldSpec& f, RepCounter& rc) {
  std::string p = prefix + f.name;
  if (f.kind == FieldKind::Message && f.repeated)
    p += "[" + std::to_string(rc.next(f.number)) + "]";
  return p;
}

Result<bool> decode_message(const Ctx& c, const SubRange& sr, const MessageSpec& spec,
                            const std::string& prefix, AttrSink& sink, int depth) {
  assert(depth <= 8);
  WireReader r = WireReader::sub(sr);
  RepCounter rc;
  while (!r.at_end()) {
    const uint64_t at = r.abs_pos();
    auto h = r.read_tag();
    if (!h) return h.error();
    const FieldSpec* f = find_field(spec, h->field_number);
    if (!f) {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
      ++c.net.undecoded_fields;
      continue;
    }
    auto d = decode_field(c, r, *f, h->wire_type, at, field_path(prefix, *f, rc), sink, depth);
    if (!d) return d.error();
  }
  return true;
}

// BlobShape -> dims (packed or unpacked int64), <= kMaxBlobAxes.
Result<SmallVec<int64_t, 6>> read_blob_shape(const SubRange& sr) {
  SmallVec<int64_t, 6> dims;
  WireReader r = WireReader::sub(sr);
  auto push = [&](uint64_t raw, uint64_t at) -> Result<bool> {
    if (dims.size() >= kMaxBlobAxes) return bin_error("shape has more than 32 axes", at);
    dims.push_back(static_cast<int64_t>(raw));
    return true;
  };
  while (!r.at_end()) {
    const uint64_t at = r.abs_pos();
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::Varint) {
      auto v = r.read_varint();
      if (!v) return v.error();
      auto p = push(*v, at);
      if (!p) return p.error();
    } else if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      WireReader pr = WireReader::sub(*sub);
      while (!pr.at_end()) {
        const uint64_t el = pr.abs_pos();
        auto v = pr.read_varint();
        if (!v) return v.error();
        auto p = push(*v, el);
        if (!p) return p.error();
      }
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return dims;
}

// One payload stream (data or double_data) of a BlobProto.
struct Chunk {
  uint64_t offset = UINT64_MAX;
  uint64_t len = 0;
  uint32_t chunks = 0;
};

// BlobProto: num 1, channels 2, height 3, width 4, data 5 (packed float), diff 6,
// shape 7, double_data 8, double_diff 9. Blob::ToProto writes data BEFORE shape.
Result<BlobRaw> read_blob(const Ctx& c, const SubRange& sr) {
  BlobRaw b;
  b.proto_offset = sr.offset;
  int64_t legacy[4] = {0, 0, 0, 0};
  bool legacy_seen = false, shape_seen = false;
  Chunk f32, f64;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    const uint64_t at = r.abs_pos();
    auto h = r.read_tag();
    if (!h) return h.error();
    const uint32_t fn = h->field_number;
    const WireType wt = h->wire_type;
    if (fn >= 1 && fn <= 4 && wt == WireType::Varint) {
      auto v = r.read_varint();
      if (!v) return v.error();
      static const char* const kNames[4] = {"num", "channels", "height", "width"};
      auto x = as_int32(*v, kNames[fn - 1], at);
      if (!x) return x.error();
      legacy[fn - 1] = *x;
      legacy_seen = true;
    } else if ((fn == 5 || fn == 8) && wt == WireType::LenDelim) {
      auto chunk = r.read_len_delim();   // advances; never reads the payload bytes
      if (!chunk) return chunk.error();
      const uint64_t width = fn == 5 ? 4 : 8;
      if (chunk->len % width != 0)
        return Error(fn == 5 ? "Caffe: packed float blob length not a multiple of 4"
                             : "Caffe: packed double blob length not a multiple of 8",
                     chunk->offset);
      Chunk& ch = fn == 5 ? f32 : f64;
      if (ch.chunks == 0) {
        ch.offset = chunk->offset;
        ch.len = chunk->len;
      } else {
        ch.len += chunk->len;   // legal concatenation, but no longer one range
      }
      ++ch.chunks;
    } else if ((fn == 5 && wt == WireType::Fixed32) || (fn == 8 && wt == WireType::Fixed64)) {
      // Unpacked payload: abandon this BlobProto here (the parent reader is
      // already past its end); later fields — usually `shape` — are not read.
      b.unpacked = true;
      b.dtype = fn == 5 ? ir::DType::F32 : ir::DType::F64;
      ++c.net.unpacked_blobs;
      break;
    } else if (fn == 7 && wt == WireType::LenDelim) {
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      auto s = read_blob_shape(*sub);
      if (!s) return s.error();
      b.shape = std::move(*s);
      shape_seen = true;
    } else {
      auto sk = r.skip_field(wt);   // diff / double_diff / fork fields, by length
      if (!sk) return sk.error();
    }
  }

  if (!b.unpacked) {
    // Blob::FromProto uses double_data when it is non-empty.
    const Chunk* use = nullptr;
    if (f64.chunks > 0 && f64.len > 0) {
      use = &f64;
      b.dtype = ir::DType::F64;
    } else if (f32.chunks > 0) {
      use = &f32;
      b.dtype = ir::DType::F32;
    } else if (f64.chunks > 0) {
      use = &f64;
      b.dtype = ir::DType::F64;
    }
    if (use) {
      b.offset = use->offset;
      b.byte_len = use->len;
      b.value_count = use->len / (b.dtype == ir::DType::F64 ? 8 : 4);
      b.addressable = use->chunks == 1;
      if (!b.addressable) {
        b.offset = UINT64_MAX;
        b.byte_len = 0;
      }
    }
  }

  if (!shape_seen && legacy_seen) {
    for (int64_t v : legacy) b.shape.push_back(v);
    b.legacy_dims = true;
    shape_seen = true;
  }
  if (shape_seen) {
    if (sanitize_shape(b.shape)) {
      b.shape_known = true;
    } else {
      b.shape.clear();
      ++c.net.oversized_shapes;
    }
  }
  if (b.shape_known && b.addressable) {
    int64_t elems = 1;
    bool all_known = true;
    for (int64_t d : b.shape) {
      if (d < 0) { all_known = false; break; }
      elems *= d;   // bounded by sanitize_shape (<= INT32_MAX)
    }
    const uint64_t width = b.dtype == ir::DType::F64 ? 8 : 4;
    if (all_known && static_cast<uint64_t>(elems) * width != b.byte_len)
      ++c.net.blob_size_mismatch;
  }
  return b;
}

Result<RuleRaw> read_rule(const SubRange& sr) {
  RuleRaw rule;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    const uint64_t at = r.abs_pos();
    auto h = r.read_tag();
    if (!h) return h.error();
    const uint32_t fn = h->field_number;
    if ((fn == 1 || fn == 2 || fn == 3) && h->wire_type == WireType::Varint) {
      auto v = r.read_varint();
      if (!v) return v.error();
      auto x = as_int32(*v, fn == 1 ? "phase" : fn == 2 ? "min_level" : "max_level", at);
      if (!x) return x.error();
      if (fn == 1) {
        rule.phase = *x < 0 ? -2 : static_cast<int32_t>(*x);
      } else if (fn == 2) {
        rule.has_min_level = true;
        rule.min_level = static_cast<int32_t>(*x);
      } else {
        rule.has_max_level = true;
        rule.max_level = static_cast<int32_t>(*x);
      }
    } else if ((fn == 4 || fn == 5) && h->wire_type == WireType::LenDelim) {
      auto s = read_capped_string(r, kMaxNameBytes, "stage");
      if (!s) return s.error();
      if (rule.stage.size() + rule.not_stage.size() >= kMaxRepeatedScalars)
        return bin_error("too many stages in a rule", at);
      (fn == 4 ? rule.stage : rule.not_stage).push_back(std::move(*s));
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return rule;
}

Result<bool> push_name(WireReader& r, std::vector<std::string>& out, const char* what,
                       uint64_t at) {
  auto s = read_capped_string(r, kMaxNameBytes, what);
  if (!s) return s.error();
  if (out.size() >= kMaxBottomsOrTops)
    return bin_error(std::string("too many ") + what + " entries in one layer", at);
  out.push_back(std::move(*s));
  return true;
}

// V0LayerParameter: only name (1) and type (2) are read.
Result<bool> read_v0(const SubRange& sr, LayerRaw& L) {
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if ((h->field_number == 1 || h->field_number == 2) &&
        h->wire_type == WireType::LenDelim) {
      auto s = read_capped_string(r, kMaxNameBytes, h->field_number == 1 ? "name" : "type");
      if (!s) return s.error();
      if (h->field_number == 1) {
        if (L.name.empty()) L.name = std::move(*s);
      } else {
        L.type = std::move(*s);
        L.has_type = true;
      }
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return true;
}

// LayerParameter (v1 == false) or V1LayerParameter (v1 == true).
Result<bool> read_layer(const Ctx& c, const SubRange& sr, bool v1) {
  if (c.net.layers.size() >= kMaxLayers) return bin_error("too many layers", sr.offset);
  LayerRaw L;
  L.v1 = v1;
  L.offset = sr.offset;
  const bool full = c.mode == ReadMode::Full && c.model != nullptr;
  const MessageSpec& spec = v1 ? v1_layer_parameter_spec() : layer_parameter_spec();
  // Field numbers of the structural fields for this generation.
  const uint32_t kName = v1 ? 4 : 1, kType = v1 ? 5 : 2, kBottom = v1 ? 2 : 3,
                 kTop = v1 ? 3 : 4, kBlobs = v1 ? 6 : 7, kInclude = v1 ? 32 : 8,
                 kExclude = v1 ? 33 : 9;
  // The attribute builder writes into L.attrs and exists only in Full mode
  // (BlobsOnly interns nothing).
  std::optional<AttrSink> sink;
  if (full) sink.emplace(*c.model, L.attrs);
  RepCounter rc;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    const uint64_t at = r.abs_pos();
    auto h = r.read_tag();
    if (!h) return h.error();
    const uint32_t fn = h->field_number;
    const WireType wt = h->wire_type;
    const bool ld = wt == WireType::LenDelim;
    if (fn == kName && ld) {
      auto s = read_capped_string(r, kMaxNameBytes, "layer name");
      if (!s) return s.error();
      L.name = std::move(*s);
    } else if (fn == kType && !v1 && ld) {
      auto s = read_capped_string(r, kMaxNameBytes, "layer type");
      if (!s) return s.error();
      L.type = std::move(*s);
      L.has_type = true;
    } else if (fn == kType && v1 && wt == WireType::Varint) {
      auto v = r.read_varint();
      if (!v) return v.error();
      auto x = as_int32(*v, "type", at);
      if (!x) return x.error();
      const char* nm = enum_name(v1_layer_type_enum(), *x);
      L.has_type = true;
      L.v1_type_ident = nm ? std::string(nm) : std::to_string(*x);
      L.type = nm ? v1_modern_type(*x) : "";
    } else if ((fn == kBottom || fn == kTop) && ld) {
      if (!full) {
        auto sk = r.skip_field(wt);
        if (!sk) return sk.error();
        continue;
      }
      auto p = push_name(r, fn == kBottom ? L.bottom : L.top,
                         fn == kBottom ? "bottom" : "top", at);
      if (!p) return p.error();
    } else if (fn == kBlobs && ld) {
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      if (L.blobs.size() >= kMaxBlobsPerLayer)
        return bin_error("too many blobs in one layer", at);
      auto b = read_blob(c, *sub);
      if (!b) return b.error();
      L.blobs.push_back(std::move(*b));
    } else if (v1 && fn == 1 && ld) {   // V0 layer
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      L.v0 = true;
      auto v0 = read_v0(*sub, L);
      if (!v0) return v0.error();
    } else if (!full) {
      // BlobsOnly (sibling pairing): name, type and blobs are all it needs.
      auto sk = r.skip_field(wt);
      if (!sk) return sk.error();
    } else if ((fn == kInclude || fn == kExclude) && ld) {
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      auto rule = read_rule(*sub);
      if (!rule) return rule.error();
      (fn == kInclude ? L.include : L.exclude).push_back(std::move(*rule));
      const FieldSpec* f = find_field(spec, fn);
      auto d = decode_message(c, *sub, *f->message, field_path("", *f, rc) + ".", *sink, 1);
      if (!d) return d.error();
    } else if (!v1 && fn == 143 && ld) {   // input_param: attributes + declared shapes
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      L.has_input_param = true;
      WireReader ipr = WireReader::sub(*sub);
      while (!ipr.at_end()) {
        auto ih = ipr.read_tag();
        if (!ih) return ih.error();
        if (ih->field_number == 1 && ih->wire_type == WireType::LenDelim) {
          auto ss = ipr.read_len_delim();
          if (!ss) return ss.error();
          auto dims = read_blob_shape(*ss);
          if (!dims) return dims.error();
          const bool ok = sanitize_shape(*dims);
          if (!ok) {
            dims->clear();
            ++c.net.oversized_shapes;
          }
          L.input_param_shapes.push_back(std::move(*dims));
          L.input_param_shape_ok.push_back(ok);
        } else {
          auto sk = ipr.skip_field(ih->wire_type);
          if (!sk) return sk.error();
        }
      }
      const FieldSpec* f = find_field(spec, fn);
      auto d = decode_message(c, *sub, *f->message, "input_param.", *sink, 1);
      if (!d) return d.error();
    } else if (const FieldSpec* f = find_field(spec, fn)) {
      auto d = decode_field(c, r, *f, wt, at, field_path("", *f, rc), *sink, 1);
      if (!d) return d.error();
    } else {
      auto sk = r.skip_field(wt);
      if (!sk) return sk.error();
      ++c.net.undecoded_fields;
    }
  }
  if (L.v0) {
    L.v1 = false;
    L.v1_type_ident.clear();
    ++c.net.v0_layers;
  }
  c.net.layers.push_back(std::move(L));
  return true;
}

Result<bool> read_state(const Ctx& c, const SubRange& sr) {
  c.net.has_state = true;
  c.net.state_phase = 1;   // NetState.phase defaults to TEST
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    const uint64_t at = r.abs_pos();
    auto h = r.read_tag();
    if (!h) return h.error();
    const uint32_t fn = h->field_number;
    if ((fn == 1 || fn == 2) && h->wire_type == WireType::Varint) {
      auto v = r.read_varint();
      if (!v) return v.error();
      auto x = as_int32(*v, fn == 1 ? "phase" : "level", at);
      if (!x) return x.error();
      if (fn == 1) c.net.state_phase = *x < 0 ? -2 : static_cast<int32_t>(*x);
      else c.net.state_level = static_cast<int32_t>(*x);
    } else if (fn == 3 && h->wire_type == WireType::LenDelim) {
      auto s = read_capped_string(r, kMaxNameBytes, "stage");
      if (!s) return s.error();
      if (c.net.state_stages.size() >= kMaxRepeatedScalars)
        return bin_error("too many stages", at);
      c.net.state_stages.push_back(std::move(*s));
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return true;
}

}  // namespace

Result<NetRaw> read_caffemodel(const uint8_t* d, uint64_t n, ir::Model* model,
                               ReadMode mode, ProgressSink* p) {
  NetRaw net;
  if (d == nullptr || n == 0) return net;
  const Ctx c{model, mode, net};
  WireReader r(d, n, 0);
  uint64_t next_progress = 1ull << 20;
  while (!r.at_end()) {
    if (p && r.abs_pos() >= next_progress) {
      next_progress = r.abs_pos() + (1ull << 20);
      const double frac = static_cast<double>(r.abs_pos()) / static_cast<double>(n);
      p->set(0.2f + 0.3f * static_cast<float>(frac), "Reading caffemodel");
    }
    const uint64_t at = r.abs_pos();
    auto h = r.read_tag();
    if (!h) return h.error();
    const uint32_t fn = h->field_number;
    const WireType wt = h->wire_type;
    const bool ld = wt == WireType::LenDelim;
    if (fn == 1 && ld) {
      auto s = read_capped_string(r, kMaxNameBytes, "net name");
      if (!s) return s.error();
      net.name = std::move(*s);
    } else if ((fn == 2 || fn == 100) && ld) {
      const bool v1 = fn == 2;
      note_generation(net, v1 ? Generation::V1 : Generation::Layer, at);
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      auto L = read_layer(c, *sub, v1);
      if (!L) return L.error();
    } else if (fn == 3 && ld) {
      auto pn = push_name(r, net.inputs, "input", at);
      if (!pn) return pn.error();
    } else if (fn == 4 && (wt == WireType::Varint || ld)) {
      auto add = [&](uint64_t raw, uint64_t where) -> Result<bool> {
        auto x = as_int32(raw, "input_dim", where);
        if (!x) return x.error();
        if (net.input_dims.size() >= kMaxBottomsOrTops * 4)
          return bin_error("too many input_dim entries", where);
        net.input_dims.push_back(*x);
        return true;
      };
      if (wt == WireType::Varint) {
        auto v = r.read_varint();
        if (!v) return v.error();
        auto a = add(*v, at);
        if (!a) return a.error();
      } else {
        auto sub = r.read_len_delim();
        if (!sub) return sub.error();
        WireReader pr = WireReader::sub(*sub);
        while (!pr.at_end()) {
          const uint64_t el = pr.abs_pos();
          auto v = pr.read_varint();
          if (!v) return v.error();
          auto a = add(*v, el);
          if (!a) return a.error();
        }
      }
    } else if (fn == 6 && ld) {
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      auto s = read_state(c, *sub);
      if (!s) return s.error();
    } else if (fn == 8 && ld) {
      auto sub = r.read_len_delim();
      if (!sub) return sub.error();
      auto dims = read_blob_shape(*sub);
      if (!dims) return dims.error();
      if (net.input_shapes.size() >= kMaxBottomsOrTops)
        return bin_error("too many input_shape entries", at);
      const bool ok = sanitize_shape(*dims);
      if (!ok) {
        dims->clear();
        ++net.oversized_shapes;
      }
      net.input_shapes.push_back(std::move(*dims));
      net.input_shape_ok.push_back(ok);
    } else if ((fn == 5 || fn == 7) && wt == WireType::Varint) {
      auto v = r.read_varint();   // force_backward / debug_info: training-only flags
      if (!v) return v.error();
    } else {
      auto sk = r.skip_field(wt);
      if (!sk) return sk.error();
      if (fn == 1 || fn == 2 || fn == 3 || fn == 4 || fn == 5 || fn == 6 || fn == 7 ||
          fn == 8 || fn == 100)
        ++net.undecoded_fields;   // a known field with the wrong wire type
      else
        ++net.ignored_top_level;
    }
  }
  return net;
}

}  // namespace netvis::caffe
