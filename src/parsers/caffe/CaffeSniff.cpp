// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeSniff.cpp — bounded content sniffs for .prototxt and
// .caffemodel (#138). See CaffeSniff.h; every read is bounds-checked (TextLexer
// with a byte budget for text, onnx::WireReader for binary) and every
// length-delimited value is skipped by pointer arithmetic, never read.
#include "parsers/caffe/CaffeSniff.h"

#include <string_view>

#include "parsers/caffe/CaffeSchema.h"
#include "parsers/caffe/TextProto.h"
#include "parsers/onnx/WireReader.h"

namespace netvis::caffe {
namespace {

constexpr uint64_t kTextPrefixBytes = 4096;
constexpr uint64_t kTextSniffBytes = 64 * 1024;
constexpr int kTextSniffTokens = 4096;
constexpr int kBinaryTopFields = 64;
constexpr int kBinaryPeekFields = 32;

bool is_type_char(uint8_t c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
         c == '_';
}

bool plausible_type(const uint8_t* p, uint64_t len) {
  if (len == 0 || len > 64) return false;
  for (uint64_t i = 0; i < len; ++i)
    if (!is_type_char(p[i])) return false;
  return true;
}

// ---- text -----------------------------------------------------------------------

// Token stream with one-token lookahead and a hard token budget.
class Tokens {
 public:
  Tokens(const uint8_t* d, uint64_t n) : lex_(d, n, 0, kTextSniffBytes) { lex_.set_quiet(true); }

  bool next(Token& t) {
    if (have_) {
      have_ = false;
      t = pending_;
      return true;
    }
    if (budget_-- <= 0) return false;
    auto r = lex_.next();
    if (!r) return false;
    t = *r;
    return true;
  }
  void unget(const Token& t) {
    pending_ = t;
    have_ = true;
  }

 private:
  TextLexer lex_;
  int budget_ = kTextSniffTokens;
  Token pending_;
  bool have_ = false;
};

bool is_punct(const Token& t, char c) { return t.kind == TokKind::Punct && t.punct == c; }
bool is_open(const Token& t) { return is_punct(t, '{') || is_punct(t, '<'); }
bool is_close(const Token& t) { return is_punct(t, '}') || is_punct(t, '>'); }

// Skip to the end of a message / list whose opener was just consumed.
bool skip_nested(Tokens& ts) {
  int depth = 1;
  Token t;
  while (depth > 0) {
    if (!ts.next(t) || t.kind == TokKind::End) return false;
    if (is_open(t) || is_punct(t, '[')) ++depth;
    else if (is_close(t) || is_punct(t, ']')) --depth;
  }
  return true;
}

// Skip one value whose first token `v` was just consumed.
bool skip_value(Tokens& ts, const Token& v) {
  if (is_open(v) || is_punct(v, '[')) return skip_nested(ts);
  if (v.kind == TokKind::String) {
    Token t;
    while (ts.next(t)) {
      if (t.kind != TokKind::String) {
        ts.unget(t);
        return true;
      }
    }
    return false;
  }
  if (is_punct(v, '-')) {
    Token t;
    return ts.next(t) && (t.kind == TokKind::Number || t.kind == TokKind::Name);
  }
  return v.kind == TokKind::Number || v.kind == TokKind::Name;
}

// Read the value token after a field name (optional ':').
bool value_token(Tokens& ts, Token& v) {
  if (!ts.next(v)) return false;
  if (is_punct(v, ':') && !ts.next(v)) return false;
  return v.kind != TokKind::End;
}

void skip_separator(Tokens& ts) {
  Token t;
  if (ts.next(t) && !(is_punct(t, ',') || is_punct(t, ';'))) ts.unget(t);
}

// Scan the first layer / layers block (opener consumed) for its type.
bool scan_layer_block(Tokens& ts, bool v1) {
  Token t;
  for (;;) {
    if (!ts.next(t) || t.kind == TokKind::End) return false;
    if (is_close(t)) return false;   // closed without a valid type
    if (is_punct(t, ',') || is_punct(t, ';')) continue;
    if (t.kind != TokKind::Name) return false;
    const std::string_view name = t.raw;
    Token v;
    if (!value_token(ts, v)) return false;
    if (name == "type") {
      if (!v1) {
        return v.kind == TokKind::String && v.raw.size() >= 2 &&
               plausible_type(reinterpret_cast<const uint8_t*>(v.raw.data()) + 1,
                              v.raw.size() - 2);
      }
      if (v.kind == TokKind::Name) return enum_number(v1_layer_type_enum(), v.raw).has_value();
      if (v.kind == TokKind::Number) {
        auto x = text_to_int64(v.raw);
        return x && *x >= 0 && *x <= kMaxV1LayerType;
      }
      return false;
    }
    if (v1 && name == "layer" && is_open(v)) return true;   // V0 layer
    if (!skip_value(ts, v)) return false;
  }
}

bool top_level_name(std::string_view n) {
  return n == "name" || n == "input" || n == "input_dim" || n == "input_shape" ||
         n == "force_backward" || n == "state" || n == "debug_info" || n == "layer" ||
         n == "layers";
}

// ---- binary -----------------------------------------------------------------------

using onnx::WireReader;
using onnx::WireType;

bool valid_wire(WireType w) {
  return w == WireType::Varint || w == WireType::Fixed64 || w == WireType::LenDelim ||
         w == WireType::Fixed32;
}

bool peek_layer(const onnx::SubRange& sr) {
  WireReader r = WireReader::sub(sr);
  for (int i = 0; i < kBinaryPeekFields && !r.at_end(); ++i) {
    auto h = r.read_tag();
    if (!h) return false;
    const uint32_t fn = h->field_number;
    const WireType w = h->wire_type;
    const bool ld = w == WireType::LenDelim;
    if (fn == 2) {
      if (!ld) return false;
      auto s = r.read_len_delim();
      return s && plausible_type(s->ptr, s->len);
    }
    bool ok = false;
    if (fn == 1 || fn == 3 || fn == 4 || (fn >= 6 && fn <= 9)) ok = ld;
    else if (fn == 5) ok = w == WireType::Fixed32 || ld;
    else if (fn == 10) ok = w == WireType::Varint;
    else if (fn == 11) ok = w == WireType::Varint || ld;
    else if (fn >= 100 && fn <= 148) ok = ld;
    else ok = valid_wire(w);   // a fork's field
    if (!ok) return false;
    if (!r.skip_field(w)) return false;
  }
  return false;
}

bool v1_param_slot(uint32_t fn) {
  return (fn >= 9 && fn <= 27) || (fn >= 29 && fn <= 31) || fn == 34 || (fn >= 36 && fn <= 42);
}

bool peek_v1_layer(const onnx::SubRange& sr) {
  WireReader r = WireReader::sub(sr);
  for (int i = 0; i < kBinaryPeekFields && !r.at_end(); ++i) {
    auto h = r.read_tag();
    if (!h) return false;
    const uint32_t fn = h->field_number;
    const WireType w = h->wire_type;
    const bool ld = w == WireType::LenDelim;
    if (fn == 5) {
      if (w != WireType::Varint) return false;
      auto v = r.read_varint();
      return v && *v <= static_cast<uint64_t>(kMaxV1LayerType);
    }
    if (fn == 1) {   // V0 layer: its first field must be name (1) or type (2)
      if (!ld) return false;
      auto s = r.read_len_delim();
      if (!s) return false;
      WireReader vr = WireReader::sub(*s);
      auto vh = vr.read_tag();
      return vh && (vh->field_number == 1 || vh->field_number == 2) &&
             vh->wire_type == WireType::LenDelim;
    }
    bool ok = false;
    if (fn == 2 || fn == 3 || fn == 4 || fn == 6) ok = ld;
    else if (fn == 7 || fn == 8 || fn == 35) ok = w == WireType::Fixed32 || ld;
    else if (fn == 32 || fn == 33) ok = ld;
    else if (v1_param_slot(fn)) ok = ld;
    else if (fn == 1001) ok = ld;
    else if (fn == 1002) ok = w == WireType::Varint || ld;
    else ok = false;
    if (!ok) return false;
    if (!r.skip_field(w)) return false;
  }
  return false;
}

}  // namespace

bool is_text_prefix(const uint8_t* d, uint64_t n) {
  if (d == nullptr || n == 0) return false;
  const uint64_t m = n < kTextPrefixBytes ? n : kTextPrefixBytes;
  for (uint64_t i = 0; i < m; ++i) {
    const uint8_t c = d[i];
    if (c <= 0x08 || (c >= 0x0E && c <= 0x1F) || c == 0x7F) return false;
  }
  return true;
}

bool looks_like_prototxt(const uint8_t* d, uint64_t n) {
  if (!is_text_prefix(d, n)) return false;
  Tokens ts(d, n);
  Token t;
  for (;;) {
    if (!ts.next(t) || t.kind != TokKind::Name) return false;   // End, junk, lex error
    const std::string_view name = t.raw;
    if (!top_level_name(name)) return false;
    Token v;
    if (!value_token(ts, v)) return false;
    if (name == "layer" || name == "layers") {
      if (!is_open(v)) return false;
      return scan_layer_block(ts, name == "layers");
    }
    if (!skip_value(ts, v)) return false;
    skip_separator(ts);
  }
}

bool looks_like_caffemodel(const uint8_t* d, uint64_t n) {
  if (d == nullptr || n == 0) return false;
  WireReader r(d, n, 0);
  for (int i = 0; i < kBinaryTopFields && !r.at_end(); ++i) {
    auto h = r.read_tag();
    if (!h) return false;
    const uint32_t fn = h->field_number;
    const WireType w = h->wire_type;
    const bool ld = w == WireType::LenDelim;
    switch (fn) {
      case 2:
      case 100: {
        if (!ld) return false;
        auto s = r.read_len_delim();
        if (!s) return false;
        return fn == 2 ? peek_v1_layer(*s) : peek_layer(*s);
      }
      case 1: case 3: case 6: case 8:
        if (!ld) return false;
        break;
      case 4:
        if (!(ld || w == WireType::Varint)) return false;
        break;
      case 5: case 7:
        if (w != WireType::Varint) return false;
        break;
      default:
        return false;
    }
    if (!r.skip_field(w)) return false;
  }
  return false;
}

}  // namespace netvis::caffe
