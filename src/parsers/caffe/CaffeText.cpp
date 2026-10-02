// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeText.cpp — text NetParameter (.prototxt) -> NetRaw.
//
// The document comes from the bounded TextProto reader (no recursion there,
// every cap enforced). This pass is schema-guided: a field the caffe.proto
// subset in CaffeSchema knows is converted by its declared kind (a value of the
// wrong kind is an error at that value's offset, as protobuf would refuse it);
// a field it does not know — a fork's layer parameters, an unknown message —
// is flattened generically so a .prototxt always shows every field.
//
// Inline weights (`blobs { data: ... }`) are elided by the reader: counted,
// never converted or stored, and never addressable (§3.6).
#include <cfloat>
#include <cmath>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "parsers/caffe/CaffeNet.h"
#include "parsers/caffe/CaffeSchema.h"
#include "parsers/caffe/TextProto.h"

namespace netvis::caffe {
namespace {

// Elision rule indices, in the order of the options built in read_prototxt.
constexpr uint8_t kElideData = 0;
constexpr uint8_t kElideDoubleData = 2;

struct Ctx {
  const uint8_t* d;
  uint64_t n;
  const TextDocument& doc;
  ir::Model& model;
  NetRaw& net;
};

Error text_error(const Ctx& c, const std::string& what, uint64_t offset) {
  auto lc = line_col(c.d, c.n, offset);
  return Error("Caffe prototxt: " + what + " (line " + std::to_string(lc.first) +
                   ", column " + std::to_string(lc.second) + ")",
               offset);
}

std::string shown(const TextNode& v) {
  if (v.kind == TextKind::Message) return "a message";
  std::string s = v.value.size() > 64 ? v.value.substr(0, 64) + "..." : v.value;
  return "'" + s + "'";
}

Error expects(const Ctx& c, const TextNode& v, const char* what) {
  return text_error(c, "field '" + v.name + "' expects " + what + ", got " + shown(v),
                    v.kind == TextKind::Message ? v.offset : v.value_offset);
}

// A float field: round through float so a value read from text equals the
// 32-bit value the binary writer would have stored (text/binary agree exactly).
double as_float32(double v) {
  if (std::isnan(v) || std::isinf(v)) return v;
  if (std::fabs(v) > static_cast<double>(FLT_MAX)) return std::copysign(HUGE_VAL, v);
  return static_cast<double>(static_cast<float>(v));
}

Result<std::string> want_string(const Ctx& c, const TextNode& v, uint64_t cap) {
  if (v.kind != TextKind::String) return expects(c, v, "a string");
  if (v.value.size() > cap)
    return text_error(c, "field '" + v.name + "' is too long", v.value_offset);
  return v.value;
}

Result<int64_t> want_int(const Ctx& c, const TextNode& v, int64_t lo, int64_t hi,
                         const char* what) {
  auto x = text_to_int64(v);
  if (!x || *x < lo || *x > hi) return expects(c, v, what);
  return *x;
}

// Phase-like enum for NetState / NetStateRule: -1 absent, -2 present but not a
// valid Phase (such a rule can never match), else the number.
Result<int32_t> want_phase(const Ctx& c, const TextNode& v) {
  if (v.kind == TextKind::Identifier) {
    auto p = enum_number(phase_enum(), v.value);
    return p ? *p : -2;
  }
  if (v.kind == TextKind::Number) {
    auto x = want_int(c, v, INT32_MIN, INT32_MAX, "a phase");
    if (!x) return x.error();
    return *x < 0 ? -2 : static_cast<int32_t>(*x);
  }
  return expects(c, v, "a phase");
}

// ---- generic + schema-guided flattening (§3.9) ------------------------------------

// Convert one scalar node through its schema field.
Result<bool> emit_scalar(const Ctx& c, const FieldSpec& f, const TextNode& v,
                         const std::string& path, AttrSink& sink) {
  if (v.kind == TextKind::Message) return expects(c, v, "a value");
  bool ok = true;
  switch (f.kind) {
    case FieldKind::Int32: {
      auto x = want_int(c, v, INT32_MIN, INT32_MAX, "a 32-bit integer");
      if (!x) return x.error();
      ok = sink.add_int(path, *x, f.repeated);
      break;
    }
    case FieldKind::Int64: {
      auto x = want_int(c, v, INT64_MIN, INT64_MAX, "an integer");
      if (!x) return x.error();
      ok = sink.add_int(path, *x, f.repeated);
      break;
    }
    case FieldKind::UInt32:
    case FieldKind::UInt64: {
      auto x = text_to_uint64(v);
      const uint64_t hi = f.kind == FieldKind::UInt32 ? static_cast<uint64_t>(UINT32_MAX)
                                                      : static_cast<uint64_t>(INT64_MAX);
      if (!x || *x > hi) return expects(c, v, "an unsigned integer");
      ok = sink.add_int(path, static_cast<int64_t>(*x), f.repeated);
      break;
    }
    case FieldKind::Bool: {
      auto x = text_to_bool(v);
      if (!x) return expects(c, v, "a boolean");
      ok = sink.add_int(path, *x ? 1 : 0, f.repeated);
      break;
    }
    case FieldKind::Float:
    case FieldKind::Double: {
      auto x = text_to_double(v);
      if (!x) return expects(c, v, "a number");
      ok = sink.add_float(path, f.kind == FieldKind::Float ? as_float32(*x) : *x,
                          f.repeated);
      break;
    }
    case FieldKind::Enum: {
      if (v.kind == TextKind::Identifier) {
        // A known identifier is stored as itself; an unknown one (a fork's enum
        // value, a typo) verbatim — never mapped to a guess.
        ok = sink.add_string(path, v.value, f.repeated);
      } else if (v.kind == TextKind::Number) {
        auto x = want_int(c, v, INT32_MIN, INT32_MAX, "an enum value");
        if (!x) return x.error();
        const char* nm = f.enum_spec ? enum_name(*f.enum_spec, *x) : nullptr;
        ok = sink.add_string(path, nm ? std::string(nm) : std::to_string(*x), f.repeated);
      } else {
        return expects(c, v, "an enum identifier");
      }
      break;
    }
    case FieldKind::String: {
      auto s = want_string(c, v, kMaxBinaryStringBytes);
      if (!s) return s.error();
      ok = sink.add_string(path, *s, f.repeated);
      break;
    }
    case FieldKind::Message:
      return expects(c, v, "a message");
  }
  if (!ok)
    return text_error(c, "too many values in repeated field '" + v.name + "'", v.offset);
  return true;
}

enum class Generic : uint8_t { Ints, Floats, Strings };

// Emit a group of same-named unknown scalars as one attribute (a list when the
// name repeats in its message): Ints if all parse as int64, Floats if all are
// numeric, else Strings (numbers keep their raw text).
Result<bool> emit_generic_group(const Ctx& c, const std::vector<const TextNode*>& group,
                                const std::string& path, AttrSink& sink) {
  Generic g = Generic::Ints;
  for (const TextNode* v : group) {
    if (v->kind != TextKind::Number) {
      g = Generic::Strings;
      break;
    }
    if (!text_to_int64(*v)) {
      if (text_to_double(*v)) {
        g = Generic::Floats;
      } else {
        g = Generic::Strings;
        break;
      }
    }
  }
  const bool list = group.size() > 1;
  for (const TextNode* v : group) {
    bool ok = true;
    if (g == Generic::Ints) ok = sink.add_int(path, *text_to_int64(*v), list);
    else if (g == Generic::Floats) ok = sink.add_float(path, *text_to_double(*v), list);
    else ok = sink.add_string(path, v->value, list);
    if (!ok)
      return text_error(c, "too many values in repeated field '" + v->name + "'", v->offset);
  }
  return true;
}

bool in_list(const std::vector<std::string_view>& skip, std::string_view nm) {
  for (std::string_view s : skip)
    if (s == nm) return true;
  return false;
}

// Flatten `msg`'s children into `sink` under `prefix`. Known fields follow the
// schema; unknown ones are grouped by name (the first occurrence fixes the
// attribute's position). Recursion depth is bounded by the document's own depth
// cap (kMaxTextDepth), which the reader already enforced.
Result<bool> flatten(const Ctx& c, const TextNode& msg, const MessageSpec* spec,
                     const std::string& prefix, AttrSink& sink,
                     const std::vector<std::string_view>& skip) {
  struct Group {
    std::vector<const TextNode*> scalars;
    size_t messages = 0;
    size_t msg_seen = 0;
    bool emitted = false;
  };
  // Lookup-only maps (never iterated for output; output order is document order).
  std::unordered_map<std::string_view, Group> unknown;
  std::unordered_map<std::string_view, size_t> repeated_seen;

  for (uint32_t ci : msg.children) {
    const TextNode& ch = c.doc.node(ci);
    if (in_list(skip, ch.name)) continue;
    if (spec && find_field(*spec, ch.name)) continue;
    Group& g = unknown[ch.name];
    if (ch.kind == TextKind::Message) ++g.messages;
    else g.scalars.push_back(&ch);
  }

  for (uint32_t ci : msg.children) {
    const TextNode& ch = c.doc.node(ci);
    if (in_list(skip, ch.name)) continue;
    const FieldSpec* f = spec ? find_field(*spec, ch.name) : nullptr;
    if (f) {
      if (f->kind == FieldKind::Message) {
        if (ch.kind != TextKind::Message) return expects(c, ch, "a message");
        std::string path = prefix + ch.name;
        if (f->repeated) path += "[" + std::to_string(repeated_seen[ch.name]++) + "]";
        auto r = flatten(c, ch, f->message, path + ".", sink, {});
        if (!r) return r.error();
      } else {
        auto r = emit_scalar(c, *f, ch, prefix + ch.name, sink);
        if (!r) return r.error();
      }
      continue;
    }
    Group& g = unknown[ch.name];
    if (ch.kind == TextKind::Message) {
      std::string path = prefix + ch.name;
      if (g.messages > 1) path += "[" + std::to_string(g.msg_seen++) + "]";
      auto r = flatten(c, ch, nullptr, path + ".", sink, {});
      if (!r) return r.error();
    } else if (!g.emitted) {
      g.emitted = true;
      auto r = emit_generic_group(c, g.scalars, prefix + ch.name, sink);
      if (!r) return r.error();
    }
  }
  return true;
}

// ---- structural readers -------------------------------------------------------------

// BlobShape { dim: ... } -> dims (<= kMaxBlobAxes).
Result<SmallVec<int64_t, 6>> read_shape(const Ctx& c, const TextNode& m) {
  if (m.kind != TextKind::Message) return expects(c, m, "a message");
  SmallVec<int64_t, 6> dims;
  for (uint32_t ci : m.children) {
    const TextNode& d = c.doc.node(ci);
    if (d.name != "dim") continue;
    auto x = want_int(c, d, INT64_MIN, INT64_MAX, "an integer");
    if (!x) return x.error();
    if (dims.size() >= kMaxBlobAxes)
      return text_error(c, "shape has more than 32 axes", d.offset);
    dims.push_back(*x);
  }
  return dims;
}

Result<RuleRaw> read_rule(const Ctx& c, const TextNode& m) {
  if (m.kind != TextKind::Message) return expects(c, m, "a message");
  RuleRaw r;
  for (uint32_t ci : m.children) {
    const TextNode& f = c.doc.node(ci);
    if (f.name == "phase") {
      auto p = want_phase(c, f);
      if (!p) return p.error();
      r.phase = *p;
    } else if (f.name == "min_level" || f.name == "max_level") {
      auto x = want_int(c, f, INT32_MIN, INT32_MAX, "a 32-bit integer");
      if (!x) return x.error();
      if (f.name == "min_level") {
        r.has_min_level = true;
        r.min_level = static_cast<int32_t>(*x);
      } else {
        r.has_max_level = true;
        r.max_level = static_cast<int32_t>(*x);
      }
    } else if (f.name == "stage" || f.name == "not_stage") {
      auto s = want_string(c, f, kMaxNameBytes);
      if (!s) return s.error();
      if (r.stage.size() + r.not_stage.size() >= kMaxRepeatedScalars)
        return text_error(c, "too many stages in a rule", f.offset);
      (f.name == "stage" ? r.stage : r.not_stage).push_back(std::move(*s));
    }
  }
  return r;
}

// An inline text-format BlobProto: shape and dtype are recorded, the values were
// elided by the reader (counted only) and the blob is never addressable.
Result<BlobRaw> read_text_blob(const Ctx& c, const TextNode& m) {
  BlobRaw b;
  b.text = true;
  b.proto_offset = m.offset;
  int64_t legacy[4] = {0, 0, 0, 0};
  bool legacy_seen = false;
  bool shape_seen = false;
  for (uint32_t ci : m.children) {
    const TextNode& f = c.doc.node(ci);
    if (f.name == "shape") {
      auto s = read_shape(c, f);
      if (!s) return s.error();
      b.shape = std::move(*s);
      shape_seen = true;
    } else if (f.name == "num" || f.name == "channels" || f.name == "height" ||
               f.name == "width") {
      auto x = want_int(c, f, INT32_MIN, INT32_MAX, "a 32-bit integer");
      if (!x) return x.error();
      const int idx = f.name == "num" ? 0 : f.name == "channels" ? 1 : f.name == "height" ? 2 : 3;
      legacy[idx] = *x;
      legacy_seen = true;
    }
  }
  uint64_t n_f32 = 0, n_f64 = 0;
  for (const auto& e : m.elided) {
    if (e.first == kElideData) n_f32 += e.second;
    if (e.first == kElideDoubleData) n_f64 += e.second;
  }
  // Blob::FromProto prefers double_data when it is non-empty.
  if (n_f64 > 0) {
    b.dtype = ir::DType::F64;
    b.value_count = n_f64;
  } else if (n_f32 > 0) {
    b.dtype = ir::DType::F32;
    b.value_count = n_f32;
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
  ++c.net.text_blobs;
  return b;
}

// Names (layer, blob, type): String, capped.
Result<std::string> want_name(const Ctx& c, const TextNode& v) {
  return want_string(c, v, kMaxNameBytes);
}

Result<bool> push_name(const Ctx& c, const TextNode& v, std::vector<std::string>& out,
                       const char* what) {
  auto s = want_name(c, v);
  if (!s) return s.error();
  if (out.size() >= kMaxBottomsOrTops)
    return text_error(c, std::string("too many ") + what + " entries in one layer", v.offset);
  out.push_back(std::move(*s));
  return true;
}

// V1 `type`: an enum identifier or number; unknown values stay unresolved.
Result<bool> read_v1_type(const Ctx& c, const TextNode& v, LayerRaw& L) {
  L.has_type = true;
  if (v.kind == TextKind::Identifier) {
    auto num = enum_number(v1_layer_type_enum(), v.value);
    L.v1_type_ident = v.value;
    L.type = num ? v1_modern_type(*num) : "";
    return true;
  }
  if (v.kind == TextKind::Number) {
    auto x = want_int(c, v, INT32_MIN, INT32_MAX, "a layer type");
    if (!x) return x.error();
    const char* nm = enum_name(v1_layer_type_enum(), *x);
    L.v1_type_ident = nm ? std::string(nm) : v.value;
    L.type = nm ? v1_modern_type(*x) : "";
    return true;
  }
  return expects(c, v, "a layer type enum");
}

Result<bool> read_layer(const Ctx& c, const TextNode& m, bool v1) {
  if (m.kind != TextKind::Message) return expects(c, m, "a message");
  if (c.net.layers.size() >= kMaxLayers)
    return text_error(c, "too many layers", m.offset);
  LayerRaw L;
  L.v1 = v1;
  L.offset = m.offset;

  for (uint32_t ci : m.children) {
    const TextNode& f = c.doc.node(ci);
    if (f.name == "name") {
      auto s = want_name(c, f);
      if (!s) return s.error();
      L.name = std::move(*s);
    } else if (f.name == "type") {
      if (v1) {
        auto r = read_v1_type(c, f, L);
        if (!r) return r.error();
      } else {
        auto s = want_name(c, f);
        if (!s) return s.error();
        L.type = std::move(*s);
        L.has_type = true;
      }
    } else if (f.name == "bottom") {
      auto r = push_name(c, f, L.bottom, "bottom");
      if (!r) return r.error();
    } else if (f.name == "top") {
      auto r = push_name(c, f, L.top, "top");
      if (!r) return r.error();
    } else if (f.name == "blobs") {
      if (f.kind != TextKind::Message) return expects(c, f, "a message");
      if (L.blobs.size() >= kMaxBlobsPerLayer)
        return text_error(c, "too many blobs in one layer", f.offset);
      auto b = read_text_blob(c, f);
      if (!b) return b.error();
      L.blobs.push_back(std::move(*b));
    } else if (f.name == "include" || f.name == "exclude") {
      auto r = read_rule(c, f);
      if (!r) return r.error();
      (f.name == "include" ? L.include : L.exclude).push_back(std::move(*r));
    } else if (f.name == "input_param" && !v1) {
      if (f.kind != TextKind::Message) return expects(c, f, "a message");
      L.has_input_param = true;
      for (uint32_t si : f.children) {
        const TextNode& sh = c.doc.node(si);
        if (sh.name != "shape") continue;
        auto s = read_shape(c, sh);
        if (!s) return s.error();
        const bool ok = sanitize_shape(*s);
        if (!ok) {
          s->clear();
          ++c.net.oversized_shapes;
        }
        L.input_param_shapes.push_back(std::move(*s));
        L.input_param_shape_ok.push_back(ok);
      }
    } else if (f.name == "layer" && v1) {
      // V0 (pre-2014) `layers { layer { ... } }`: name and type verbatim, the
      // rest of its parameters (and V0 blobs) are not decoded.
      if (f.kind != TextKind::Message) return expects(c, f, "a message");
      L.v0 = true;
      for (uint32_t vi : f.children) {
        const TextNode& v = c.doc.node(vi);
        if (v.name == "name" && L.name.empty()) {
          auto s = want_name(c, v);
          if (!s) return s.error();
          L.name = std::move(*s);
        } else if (v.name == "type") {
          auto s = want_name(c, v);
          if (!s) return s.error();
          L.type = std::move(*s);
          L.has_type = true;
        }
      }
    }
  }
  if (L.v0) {
    L.v1 = false;
    L.v1_type_ident.clear();
    ++c.net.v0_layers;
  }

  // Native attributes, schema-guided, in document order.
  AttrSink sink(c.model, L.attrs);
  static const std::vector<std::string_view> kStructural = {"name", "type", "bottom",
                                                            "top", "blobs"};
  static const std::vector<std::string_view> kStructuralV1 = {"name", "type", "bottom",
                                                              "top", "blobs", "layer"};
  const MessageSpec& spec = v1 ? v1_layer_parameter_spec() : layer_parameter_spec();
  auto fr = flatten(c, m, &spec, "", sink, v1 ? kStructuralV1 : kStructural);
  if (!fr) return fr.error();

  c.net.layers.push_back(std::move(L));
  return true;
}

Result<bool> read_state(const Ctx& c, const TextNode& m) {
  if (m.kind != TextKind::Message) return expects(c, m, "a message");
  c.net.has_state = true;
  c.net.state_phase = 1;   // NetState.phase defaults to TEST
  for (uint32_t ci : m.children) {
    const TextNode& f = c.doc.node(ci);
    if (f.name == "phase") {
      auto p = want_phase(c, f);
      if (!p) return p.error();
      c.net.state_phase = *p;
    } else if (f.name == "level") {
      auto x = want_int(c, f, INT32_MIN, INT32_MAX, "a 32-bit integer");
      if (!x) return x.error();
      c.net.state_level = static_cast<int32_t>(*x);
    } else if (f.name == "stage") {
      auto s = want_string(c, f, kMaxNameBytes);
      if (!s) return s.error();
      if (c.net.state_stages.size() >= kMaxRepeatedScalars)
        return text_error(c, "too many stages", f.offset);
      c.net.state_stages.push_back(std::move(*s));
    }
  }
  return true;
}

}  // namespace

Result<NetRaw> read_prototxt(const uint8_t* d, uint64_t n, ir::Model& model,
                             ProgressSink* p) {
  TextParseOptions opts;
  // Index order matters: kElideData = 0, kElideDoubleData = 2.
  opts.elide = {{"blobs", "data"}, {"blobs", "diff"}, {"blobs", "double_data"},
                {"blobs", "double_diff"}};
  opts.progress = p;
  opts.progress_from = 0.2f;
  opts.progress_to = 0.5f;
  opts.progress_stage = "Reading prototxt";
  auto doc = TextDocument::parse(d, n, 0, opts);
  if (!doc) return Error("Caffe prototxt: " + doc.error().message, doc.error().offset);

  NetRaw net;
  const Ctx c{d, n, *doc, model, net};
  for (uint32_t ci : doc->root().children) {
    const TextNode& f = doc->node(ci);
    if (f.name == "name") {
      auto s = want_name(c, f);
      if (!s) return s.error();
      net.name = std::move(*s);
    } else if (f.name == "input") {
      auto r = push_name(c, f, net.inputs, "input");
      if (!r) return r.error();
    } else if (f.name == "input_dim") {
      auto x = want_int(c, f, INT32_MIN, INT32_MAX, "a 32-bit integer");
      if (!x) return x.error();
      if (net.input_dims.size() >= kMaxBottomsOrTops * 4)
        return text_error(c, "too many input_dim entries", f.offset);
      net.input_dims.push_back(*x);
    } else if (f.name == "input_shape") {
      auto s = read_shape(c, f);
      if (!s) return s.error();
      if (net.input_shapes.size() >= kMaxBottomsOrTops)
        return text_error(c, "too many input_shape entries", f.offset);
      const bool ok = sanitize_shape(*s);
      if (!ok) {
        s->clear();
        ++net.oversized_shapes;
      }
      net.input_shapes.push_back(std::move(*s));
      net.input_shape_ok.push_back(ok);
    } else if (f.name == "state") {
      auto r = read_state(c, f);
      if (!r) return r.error();
    } else if (f.name == "layer") {
      note_generation(net, Generation::Layer, f.offset);
      auto r = read_layer(c, f, false);
      if (!r) return r.error();
    } else if (f.name == "layers") {
      note_generation(net, Generation::V1, f.offset);
      auto r = read_layer(c, f, true);
      if (!r) return r.error();
    } else if (f.name == "force_backward" || f.name == "debug_info") {
      // Training-only flags: known, not shown.
    } else {
      ++net.ignored_top_level;
    }
  }
  return net;
}

}  // namespace netvis::caffe
