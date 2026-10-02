// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/TextProto.cpp — the bounded protobuf text-format reader.
//
// The tokenizer reads an exact-size buffer through index checks against
// `limit_` (never a trailing NUL); every next() consumes >= 1 byte or returns
// End/error, so it always terminates. The document builder is a flat loop over
// an explicit open-message stack plus a two-state list mode — no recursion at
// any depth. Line/column numbers are computed only on the error path.
#include "parsers/caffe/TextProto.h"

#include <cassert>
#include <charconv>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <system_error>

namespace netvis::caffe {

// ---- helpers ----------------------------------------------------------------

namespace {

inline bool is_ws(uint8_t c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}
inline bool is_alpha_(uint8_t c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}
inline bool is_digit(uint8_t c) { return c >= '0' && c <= '9'; }
inline bool is_alnum_(uint8_t c) { return is_alpha_(c) || is_digit(c); }
inline bool is_hex(uint8_t c) {
  return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}
inline uint32_t hex_val(uint8_t c) {
  if (is_digit(c)) return static_cast<uint32_t>(c - '0');
  if (c >= 'a' && c <= 'f') return static_cast<uint32_t>(c - 'a' + 10);
  return static_cast<uint32_t>(c - 'A' + 10);
}

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::string with_line_col(const char* what, const uint8_t* d, uint64_t size,
                          uint64_t rel) {
  auto lc = line_col(d, size, rel);
  std::string m(what);
  m += " (line ";
  m += std::to_string(lc.first);
  m += ", column ";
  m += std::to_string(lc.second);
  m += ")";
  return m;
}

}  // namespace

std::pair<uint32_t, uint32_t> line_col(const uint8_t* d, uint64_t size,
                                       uint64_t rel_offset) {
  if (d == nullptr) return {1, 1};
  if (rel_offset > size) rel_offset = size;
  uint64_t line = 1;
  uint64_t line_start = 0;
  for (uint64_t i = 0; i < rel_offset; ++i) {
    if (d[i] == '\n') {
      ++line;
      line_start = i + 1;
    }
  }
  const uint64_t col = rel_offset - line_start + 1;
  const uint64_t cap = UINT32_MAX;
  return {static_cast<uint32_t>(line < cap ? line : cap),
          static_cast<uint32_t>(col < cap ? col : cap)};
}

// ---- TextLexer -----------------------------------------------------------------

TextLexer::TextLexer(const uint8_t* d, uint64_t size, uint64_t base_offset,
                     uint64_t byte_budget)
    : d_(d), size_(d ? size : 0), limit_(0), base_(base_offset) {
  limit_ = size_ < byte_budget ? size_ : byte_budget;
  // UTF-8 BOM at offset 0.
  if (limit_ >= 3 && d_[0] == 0xEF && d_[1] == 0xBB && d_[2] == 0xBF) pos_ = 3;
}

Error TextLexer::make_error(const char* what, uint64_t rel_offset) const {
  if (quiet_) return Error("text proto: lex error", base_ + rel_offset);
  return Error(with_line_col(what, d_, size_, rel_offset), base_ + rel_offset);
}

Result<Token> TextLexer::next() {
  // Skip whitespace and comments.
  while (pos_ < limit_) {
    const uint8_t c = d_[pos_];
    if (is_ws(c)) {
      ++pos_;
    } else if (c == '#') {
      while (pos_ < limit_ && d_[pos_] != '\n') ++pos_;
    } else {
      break;
    }
  }
  Token t;
  if (pos_ >= limit_) {
    t.kind = TokKind::End;
    t.offset = base_ + pos_;
    return t;
  }
  const uint64_t start = pos_;
  const uint8_t c = d_[pos_];
  t.offset = base_ + start;

  if (is_alpha_(c)) {
    uint64_t p = pos_ + 1;
    while (p < limit_ && is_alnum_(d_[p])) ++p;
    if (p - start > kMaxTokenBytes) return make_error("token too long", start);
    // A budget cut mid-buffer cannot tell whether the token goes on: refuse it
    // rather than hand a bounded caller a possibly-partial name.
    if (p >= limit_ && limit_ < size_) return make_error("token crosses the scan budget", start);
    t.kind = TokKind::Name;
    t.raw = std::string_view(reinterpret_cast<const char*>(d_ + start),
                             static_cast<size_t>(p - start));
    pos_ = p;
    return t;
  }

  const bool dot_digit = c == '.' && pos_ + 1 < limit_ && is_digit(d_[pos_ + 1]);
  if (is_digit(c) || dot_digit) {
    const bool hex = c == '0' && pos_ + 1 < limit_ &&
                     (d_[pos_ + 1] == 'x' || d_[pos_ + 1] == 'X');
    uint64_t p = pos_ + 1;
    while (p < limit_) {
      const uint8_t ch = d_[p];
      if (is_alnum_(ch) || ch == '.') {
        ++p;
      } else if ((ch == '+' || ch == '-') && !hex &&
                 (d_[p - 1] == 'e' || d_[p - 1] == 'E')) {
        ++p;
      } else {
        break;
      }
    }
    if (p - start > kMaxTokenBytes) return make_error("token too long", start);
    if (p >= limit_ && limit_ < size_) return make_error("token crosses the scan budget", start);
    t.kind = TokKind::Number;
    t.raw = std::string_view(reinterpret_cast<const char*>(d_ + start),
                             static_cast<size_t>(p - start));
    pos_ = p;
    return t;
  }

  if (c == '"' || c == '\'') {
    uint64_t p = pos_ + 1;
    for (;;) {
      if (p >= limit_ || d_[p] == '\n')
        return make_error("unterminated string", start);
      const uint8_t ch = d_[p];
      if (ch == c) break;
      if (ch == '\\') {
        // The escaped byte must exist and must not be a raw newline.
        if (p + 1 >= limit_ || d_[p + 1] == '\n')
          return make_error("unterminated string", start);
        p += 2;
        continue;
      }
      ++p;
    }
    t.kind = TokKind::String;
    t.raw = std::string_view(reinterpret_cast<const char*>(d_ + start),
                             static_cast<size_t>(p + 1 - start));
    pos_ = p + 1;
    return t;
  }

  switch (c) {
    case '{': case '}': case '<': case '>': case '[': case ']':
    case ':': case ',': case ';': case '-':
      t.kind = TokKind::Punct;
      t.punct = static_cast<char>(c);
      t.raw = std::string_view(reinterpret_cast<const char*>(d_ + start), 1);
      pos_ = start + 1;
      return t;
    default:
      break;
  }
  return make_error("unexpected character", start);
}

Result<std::string> TextLexer::decode_string(const Token& t) const {
  std::string out;
  if (t.kind != TokKind::String || t.raw.size() < 2)
    return Error("text proto: not a string token", t.offset);
  const uint64_t rel0 = t.offset - base_;   // relative offset of the opening quote
  const std::string_view body = t.raw.substr(1, t.raw.size() - 2);
  out.reserve(body.size());
  const size_t n = body.size();
  size_t i = 0;
  while (i < n) {
    const uint8_t c = static_cast<uint8_t>(body[i]);
    if (c != '\\') {
      out.push_back(static_cast<char>(c));
      ++i;
      continue;
    }
    const uint64_t esc_rel = rel0 + 1 + i;   // offset of the backslash
    if (i + 1 >= n) return make_error("invalid escape", esc_rel);
    const uint8_t e = static_cast<uint8_t>(body[i + 1]);
    i += 2;
    switch (e) {
      case 'a': out.push_back('\a'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      case 'v': out.push_back('\v'); break;
      case '\\': out.push_back('\\'); break;
      case '\'': out.push_back('\''); break;
      case '"': out.push_back('"'); break;
      case '?': out.push_back('?'); break;
      case '0': case '1': case '2': case '3':
      case '4': case '5': case '6': case '7': {
        uint32_t v = static_cast<uint32_t>(e - '0');
        for (int k = 0; k < 2 && i < n; ++k) {
          const uint8_t o = static_cast<uint8_t>(body[i]);
          if (o < '0' || o > '7') break;
          v = v * 8 + static_cast<uint32_t>(o - '0');
          ++i;
        }
        if (v > 0xFF) return make_error("octal escape out of range", esc_rel);
        out.push_back(static_cast<char>(static_cast<uint8_t>(v)));
        break;
      }
      case 'x': case 'X': {
        uint32_t v = 0;
        int digits = 0;
        while (digits < 2 && i < n && is_hex(static_cast<uint8_t>(body[i]))) {
          v = v * 16 + hex_val(static_cast<uint8_t>(body[i]));
          ++i;
          ++digits;
        }
        if (digits == 0) return make_error("invalid \\x escape (no hex digits)", esc_rel);
        out.push_back(static_cast<char>(static_cast<uint8_t>(v)));
        break;
      }
      case 'u': case 'U': {
        const int want = e == 'u' ? 4 : 8;
        uint32_t v = 0;
        for (int k = 0; k < want; ++k) {
          if (i >= n || !is_hex(static_cast<uint8_t>(body[i])))
            return make_error("invalid unicode escape", esc_rel);
          v = v * 16 + hex_val(static_cast<uint8_t>(body[i]));
          ++i;
        }
        if (v > 0x10FFFF || (v >= 0xD800 && v <= 0xDFFF))
          return make_error("invalid unicode code point", esc_rel);
        append_utf8(out, v);
        break;
      }
      default:
        return make_error("invalid escape", esc_rel);
    }
  }
  return out;
}

// ---- TextDocument ----------------------------------------------------------------

const TextNode* TextDocument::child(const TextNode& parent,
                                    std::string_view name) const {
  for (uint32_t ci : parent.children) {
    const TextNode& c = nodes_[ci];
    if (c.name == name) return &c;
  }
  return nullptr;
}

namespace {

// One open message on the explicit stack.
struct Open {
  uint32_t node = 0;
  char closer = 0;
  uint64_t name_rel = 0;     // relative offset of the field name (unclosed errors)
  bool in_list = false;      // opened as an element of a list
  std::string_view list_name;   // that list's field name (restored on close)
  uint64_t list_name_off = 0;   // absolute offset of that list's field name
};

}  // namespace

Result<TextDocument> TextDocument::parse(const uint8_t* data, uint64_t size,
                                         uint64_t base_offset,
                                         const TextParseOptions& opts) {
  assert(opts.elide.size() <= kMaxElideRules);
  TextDocument doc;
  doc.nodes_.emplace_back();   // root
  doc.nodes_[0].offset = base_offset;
  if (data == nullptr) {
    if (size == 0) return doc;
    return err("text proto: null buffer", base_offset);
  }

  TextLexer lex(data, size, base_offset);
  auto fail = [&](const char* what, uint64_t abs_off) -> Error {
    const uint64_t rel = abs_off >= base_offset ? abs_off - base_offset : 0;
    return Error(with_line_col(what, data, size, rel), abs_off);
  };
  auto fail_s = [&](const std::string& what, uint64_t abs_off) -> Error {
    return fail(what.c_str(), abs_off);
  };

  // One-token lookahead.
  bool have_pending = false;
  Token pending;
  auto get = [&]() -> Result<Token> {
    if (have_pending) {
      have_pending = false;
      return pending;
    }
    return lex.next();
  };
  auto unget = [&](const Token& t) {
    pending = t;
    have_pending = true;
  };

  std::vector<Open> stack;
  stack.push_back(Open{0, 0, 0, false, {}, 0});

  const uint64_t stored_cap = opts.max_nodes;
  uint64_t next_progress = 1ull << 20;

  // Create a stored node under `parent`; enforces the node cap.
  auto add_node = [&](uint32_t parent, std::string_view name, TextKind kind,
                      uint64_t name_off, uint64_t value_off) -> Result<uint32_t> {
    if (doc.nodes_.size() - 1 >= stored_cap)
      return fail("too many fields (node cap reached)", name_off);
    const uint32_t idx = static_cast<uint32_t>(doc.nodes_.size());
    TextNode n;
    n.name.assign(name.data(), name.size());
    n.kind = kind;
    n.offset = name_off;
    n.value_offset = value_off;
    doc.nodes_.push_back(std::move(n));
    doc.nodes_[parent].children.push_back(idx);
    return idx;
  };

  // Index of the elision rule matching (parent, field), or -1.
  auto elide_rule = [&](uint32_t parent, std::string_view field) -> int {
    if (opts.elide.empty()) return -1;
    const std::string& pname = doc.nodes_[parent].name;
    for (size_t r = 0; r < opts.elide.size(); ++r) {
      if (opts.elide[r].field == field && opts.elide[r].parent == pname)
        return static_cast<int>(r);
    }
    return -1;
  };

  // Read one scalar starting at `first` (already consumed). Stores it as a node
  // named `name` under `parent`, or counts it under an elision rule.
  auto scalar = [&](const Token& first, uint32_t parent, std::string_view name,
                    uint64_t name_off) -> Result<bool> {
    const int rule = elide_rule(parent, name);
    const bool store = rule < 0;
    TextKind kind = TextKind::Number;
    std::string value;
    const uint64_t value_off = first.offset;
    if (first.kind == TokKind::String) {
      kind = TextKind::String;
      Token cur = first;
      for (;;) {
        if (store) {
          auto s = lex.decode_string(cur);
          if (!s) return s.error();
          if (value.size() + s->size() > kMaxStringBytes)
            return fail("string value too long", cur.offset);
          value += *s;
        }
        auto nx = get();
        if (!nx) return nx.error();
        if (nx->kind != TokKind::String) {
          unget(*nx);
          break;
        }
        cur = *nx;
      }
    } else if (first.kind == TokKind::Number || first.kind == TokKind::Name) {
      kind = first.kind == TokKind::Number ? TextKind::Number : TextKind::Identifier;
      if (store) value.assign(first.raw.data(), first.raw.size());
    } else if (first.kind == TokKind::Punct && first.punct == '-') {
      auto nx = get();
      if (!nx) return nx.error();
      if (nx->kind != TokKind::Number && nx->kind != TokKind::Name)
        return fail("expected a number or identifier after '-'", nx->offset);
      kind = nx->kind == TokKind::Number ? TextKind::Number : TextKind::Identifier;
      if (store) {
        value.push_back('-');
        value.append(nx->raw.data(), nx->raw.size());
      }
    } else {
      return fail_s("missing value for field '" + std::string(name) + "'", first.offset);
    }
    if (!store) {
      TextNode& p = doc.nodes_[parent];
      for (auto& e : p.elided) {
        if (e.first == static_cast<uint8_t>(rule)) {
          ++e.second;
          return true;
        }
      }
      p.elided.push_back({static_cast<uint8_t>(rule), 1});
      return true;
    }
    auto idx = add_node(parent, name, kind, name_off, value_off);
    if (!idx) return idx.error();
    doc.nodes_[*idx].value = std::move(value);
    return true;
  };

  // Open a message node `name` under `parent`; checks the depth cap first.
  auto open_message = [&](uint32_t parent, std::string_view name, uint64_t name_off,
                          const Token& opener, bool in_list,
                          std::string_view list_name,
                          uint64_t list_name_off) -> Result<bool> {
    if (stack.size() - 1 >= static_cast<size_t>(kMaxTextDepth))
      return fail("nesting too deep", in_list ? opener.offset : name_off);
    auto idx = add_node(parent, name, TextKind::Message, name_off, opener.offset);
    if (!idx) return idx.error();
    Open o;
    o.node = *idx;
    o.closer = opener.punct == '{' ? '}' : '>';
    o.name_rel = name_off - base_offset;
    o.in_list = in_list;
    o.list_name = list_name;
    o.list_name_off = list_name_off;
    stack.push_back(o);
    return true;
  };

  // Consume an optional ',' / ';' after a completed field.
  auto separator = [&]() -> Result<bool> {
    auto nx = get();
    if (!nx) return nx.error();
    if (!(nx->kind == TokKind::Punct && (nx->punct == ',' || nx->punct == ';')))
      unget(*nx);
    return true;
  };

  enum class Mode { Field, ListElem, ListSep };
  Mode mode = Mode::Field;
  bool list_first = false;
  std::string_view list_name;
  uint64_t list_name_off = 0;

  for (;;) {
    if (opts.progress && lex.pos() >= next_progress) {
      next_progress = lex.pos() + (1ull << 20);
      const float frac = size ? static_cast<float>(static_cast<double>(lex.pos()) /
                                                   static_cast<double>(size))
                              : 1.0f;
      opts.progress->set(opts.progress_from + (opts.progress_to - opts.progress_from) * frac,
                         opts.progress_stage);
    }
    auto tr = get();
    if (!tr) return tr.error();
    const Token tok = *tr;
    const uint32_t parent = stack.back().node;

    if (mode == Mode::ListElem) {
      if (tok.kind == TokKind::Punct && tok.punct == ']' && list_first) {
        mode = Mode::Field;
        auto s = separator();
        if (!s) return s.error();
        continue;
      }
      if (tok.kind == TokKind::Punct && tok.punct == '[')
        return fail("nested lists are not supported", tok.offset);
      if (tok.kind == TokKind::Punct && (tok.punct == '{' || tok.punct == '<')) {
        auto o = open_message(parent, list_name, list_name_off, tok, true, list_name,
                              list_name_off);
        if (!o) return o.error();
        mode = Mode::Field;
        continue;
      }
      if (tok.kind == TokKind::End)
        return fail("unterminated list", list_name_off);
      auto s = scalar(tok, parent, list_name, list_name_off);
      if (!s) return s.error();
      mode = Mode::ListSep;
      list_first = false;
      continue;
    }

    if (mode == Mode::ListSep) {
      if (tok.kind == TokKind::Punct && tok.punct == ',') {
        mode = Mode::ListElem;
        list_first = false;
        continue;
      }
      if (tok.kind == TokKind::Punct && tok.punct == ']') {
        mode = Mode::Field;
        auto s = separator();
        if (!s) return s.error();
        continue;
      }
      if (tok.kind == TokKind::End) return fail("unterminated list", list_name_off);
      return fail("expected ',' or ']' in list", tok.offset);
    }

    // Mode::Field
    if (tok.kind == TokKind::End) {
      if (stack.size() > 1) {
        const Open& o = stack.back();
        const std::string& nm = doc.nodes_[o.node].name;
        return fail_s("unclosed message '" + nm + "' at end of input",
                      base_offset + o.name_rel);
      }
      break;
    }
    if (tok.kind == TokKind::Punct && (tok.punct == '}' || tok.punct == '>')) {
      if (stack.size() <= 1) return fail("unbalanced closing bracket", tok.offset);
      const Open o = stack.back();
      if (tok.punct != o.closer)
        return fail(o.closer == '}' ? "mismatched closer: expected '}'"
                                    : "mismatched closer: expected '>'",
                    tok.offset);
      stack.pop_back();
      if (o.in_list) {
        list_name = o.list_name;
        list_name_off = o.list_name_off;
        list_first = false;
        mode = Mode::ListSep;
      } else {
        auto s = separator();
        if (!s) return s.error();
      }
      continue;
    }
    if (tok.kind == TokKind::Punct && tok.punct == '[')
      return fail("extension and Any fields are not supported", tok.offset);
    if (tok.kind != TokKind::Name) return fail("expected a field name", tok.offset);

    const std::string_view name = tok.raw;
    const uint64_t name_off = tok.offset;
    auto t2r = get();
    if (!t2r) return t2r.error();
    const Token t2 = *t2r;
    if (t2.kind == TokKind::Punct && t2.punct == ':') {
      auto t3r = get();
      if (!t3r) return t3r.error();
      const Token t3 = *t3r;
      if (t3.kind == TokKind::Punct && (t3.punct == '{' || t3.punct == '<')) {
        auto o = open_message(parent, name, name_off, t3, false, {}, 0);
        if (!o) return o.error();
        continue;
      }
      if (t3.kind == TokKind::Punct && t3.punct == '[') {
        mode = Mode::ListElem;
        list_first = true;
        list_name = name;
        list_name_off = name_off;
        continue;
      }
      auto s = scalar(t3, parent, name, name_off);
      if (!s) return s.error();
      auto sp = separator();
      if (!sp) return sp.error();
      continue;
    }
    if (t2.kind == TokKind::Punct && (t2.punct == '{' || t2.punct == '<')) {
      auto o = open_message(parent, name, name_off, t2, false, {}, 0);
      if (!o) return o.error();
      continue;
    }
    if (t2.kind == TokKind::End)
      return fail_s("missing value for field '" + std::string(name) + "'", t2.offset);
    return fail_s("missing ':' after field '" + std::string(name) + "'", t2.offset);
  }
  return doc;
}

// ---- converters -------------------------------------------------------------------

namespace {

// Parse an unsigned magnitude in decimal / 0x hex / leading-0 octal. The whole
// view must be consumed.
std::optional<uint64_t> parse_magnitude(std::string_view s) {
  if (s.empty()) return std::nullopt;
  int base = 10;
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
    base = 16;
    s.remove_prefix(2);
  } else if (s.size() > 1 && s[0] == '0') {
    base = 8;
    s.remove_prefix(1);
  }
  if (s.empty()) return std::nullopt;
  // from_chars accepts no sign or prefix here; reject anything else up front.
  for (char c : s) {
    const uint8_t u = static_cast<uint8_t>(c);
    if (base == 16 ? !is_hex(u) : !(is_digit(u) && (base == 10 || u <= '7')))
      return std::nullopt;
  }
  uint64_t v = 0;
  auto r = std::from_chars(s.data(), s.data() + s.size(), v, base);
  if (r.ec != std::errc() || r.ptr != s.data() + s.size()) return std::nullopt;
  return v;
}

bool iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    char x = a[i], y = b[i];
    if (x >= 'A' && x <= 'Z') x = static_cast<char>(x - 'A' + 'a');
    if (y >= 'A' && y <= 'Z') y = static_cast<char>(y - 'A' + 'a');
    if (x != y) return false;
  }
  return true;
}

}  // namespace

std::optional<int64_t> text_to_int64(std::string_view raw) {
  bool neg = false;
  if (!raw.empty() && raw[0] == '-') {
    neg = true;
    raw.remove_prefix(1);
  }
  auto m = parse_magnitude(raw);
  if (!m) return std::nullopt;
  if (neg) {
    if (*m > static_cast<uint64_t>(INT64_MAX) + 1) return std::nullopt;
    if (*m == static_cast<uint64_t>(INT64_MAX) + 1) return INT64_MIN;
    return -static_cast<int64_t>(*m);
  }
  if (*m > static_cast<uint64_t>(INT64_MAX)) return std::nullopt;
  return static_cast<int64_t>(*m);
}

std::optional<uint64_t> text_to_uint64(std::string_view raw) {
  if (!raw.empty() && raw[0] == '-') return std::nullopt;
  return parse_magnitude(raw);
}

std::optional<double> text_to_double(std::string_view raw) {
  if (raw.empty()) return std::nullopt;
  bool neg = false;
  std::string_view body = raw;
  if (body[0] == '-') {
    neg = true;
    body.remove_prefix(1);
  }
  if (body.empty()) return std::nullopt;
  if (iequals(body, "inf") || iequals(body, "infinity"))
    return neg ? -std::numeric_limits<double>::infinity()
               : std::numeric_limits<double>::infinity();
  if (iequals(body, "nan")) return std::numeric_limits<double>::quiet_NaN();
  if (auto i = text_to_int64(raw)) return static_cast<double>(*i);
  if (auto u = text_to_uint64(raw)) return static_cast<double>(*u);
  // Decimal float, optional [fF] suffix (not for hex, where 'f' is a digit).
  if (body.size() > 1 && (body.back() == 'f' || body.back() == 'F'))
    body.remove_suffix(1);
  for (char c : body) {
    const uint8_t u = static_cast<uint8_t>(c);
    if (!(is_digit(u) || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-'))
      return std::nullopt;
  }
  std::istringstream is{std::string(body)};
  is.imbue(std::locale::classic());
  double v = 0.0;
  is >> v;
  if (is.fail()) return std::nullopt;
  if (is.peek() != std::char_traits<char>::eof()) return std::nullopt;
  if (!std::isfinite(v)) return std::nullopt;
  return neg ? -v : v;
}

std::optional<int64_t> text_to_int64(const TextNode& n) {
  if (n.kind != TextKind::Number) return std::nullopt;
  return text_to_int64(std::string_view(n.value));
}

std::optional<uint64_t> text_to_uint64(const TextNode& n) {
  if (n.kind != TextKind::Number) return std::nullopt;
  return text_to_uint64(std::string_view(n.value));
}

std::optional<double> text_to_double(const TextNode& n) {
  if (n.kind != TextKind::Number && n.kind != TextKind::Identifier) return std::nullopt;
  return text_to_double(std::string_view(n.value));
}

std::optional<bool> text_to_bool(const TextNode& n) {
  const std::string& v = n.value;
  if (n.kind == TextKind::Identifier) {
    if (v == "true" || v == "True" || v == "t") return true;
    if (v == "false" || v == "False" || v == "f") return false;
    return std::nullopt;
  }
  if (n.kind == TextKind::Number) {
    if (v == "1") return true;
    if (v == "0") return false;
  }
  return std::nullopt;
}

}  // namespace netvis::caffe
