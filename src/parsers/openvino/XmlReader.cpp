// parsers/openvino/XmlReader.cpp — implementation of the bounded XML-subset
// reader. Tokenizes ITERATIVELY over a bounds-checked ByteReader with an explicit
// open-element stack, so parsing a pathological document never recurses (no stack
// overflow) — the depth cap (kMaxXmlDepth) is a clean policy error, not a crash.
//
// Every scan advances by >=1 byte and is bounds-checked, so the tokenizer always
// terminates; truncation / unterminated construct -> Result error with a byte
// offset. NO entity expansion beyond the five XML built-ins (single-pass literal
// replacement), NO DTD processing, NO external entities -> immune to
// billion-laughs / XXE by construction.
#include "parsers/openvino/XmlReader.h"

#include <cstdlib>
#include <cstring>

#include "core/ByteReader.h"

namespace netvis::openvino {

// ---- XmlNode accessors ------------------------------------------------------

const std::string* XmlNode::attr(std::string_view key) const {
  for (const XmlAttr& a : attrs)
    if (a.name == key) return &a.value;
  return nullptr;
}

std::optional<int64_t> XmlNode::attr_int(std::string_view key) const {
  const std::string* v = attr(key);
  if (!v || v->empty()) return std::nullopt;
  errno = 0;
  char* end = nullptr;
  long long parsed = std::strtoll(v->c_str(), &end, 10);
  if (errno != 0 || end == v->c_str()) return std::nullopt;
  // Allow only trailing whitespace after the number (reject "3x", "1 2").
  while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
  if (*end != '\0') return std::nullopt;
  return static_cast<int64_t>(parsed);
}

const XmlNode* XmlDocument::child(const XmlNode& parent,
                                  std::string_view name) const {
  for (uint32_t ci : parent.children) {
    const XmlNode& c = nodes_[ci];
    if (c.name == name) return &c;
  }
  return nullptr;
}

namespace {

inline bool is_space(uint8_t c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

// A name char terminates on whitespace or any of '/', '>', '='. This matches the
// tag-name and attribute-name lexical rules of the subset we accept.
inline bool is_name_end(uint8_t c) {
  return is_space(c) || c == '/' || c == '>' || c == '=';
}

// Decode the five XML built-in entities (single pass, literal). Any other "&..."
// is copied verbatim (NO entity expansion — the whole XXE/billion-laughs class is
// avoided by construction). `n` is bounded by the source buffer, so the output is
// bounded too; there is no recursion and no re-scan of decoded output.
std::string decode_entities(const uint8_t* p, uint64_t n) {
  std::string out;
  out.reserve(static_cast<size_t>(n));
  uint64_t i = 0;
  while (i < n) {
    if (p[i] == '&') {
      uint64_t rem = n - i;
      if (rem >= 4 && std::memcmp(p + i, "&lt;", 4) == 0) { out.push_back('<'); i += 4; continue; }
      if (rem >= 4 && std::memcmp(p + i, "&gt;", 4) == 0) { out.push_back('>'); i += 4; continue; }
      if (rem >= 5 && std::memcmp(p + i, "&amp;", 5) == 0) { out.push_back('&'); i += 5; continue; }
      if (rem >= 6 && std::memcmp(p + i, "&apos;", 6) == 0) { out.push_back('\''); i += 6; continue; }
      if (rem >= 6 && std::memcmp(p + i, "&quot;", 6) == 0) { out.push_back('"'); i += 6; continue; }
      // Unknown entity: copy the '&' literally and continue (no expansion).
    }
    out.push_back(static_cast<char>(p[i]));
    ++i;
  }
  return out;
}

// Trim ASCII whitespace from both ends of a std::string in place.
void trim(std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && is_space(static_cast<uint8_t>(s[b]))) ++b;
  while (e > b && is_space(static_cast<uint8_t>(s[e - 1]))) --e;
  if (b != 0 || e != s.size()) s = s.substr(b, e - b);
}

}  // namespace

Result<XmlDocument> XmlDocument::parse(const uint8_t* data, uint64_t size,
                                       uint64_t base_offset) {
  XmlDocument doc;
  if (data == nullptr || size == 0)
    return err("empty XML input", base_offset);

  ByteReader br(data, size);
  std::vector<uint32_t> stack;  // indices of currently-open elements

  // Peek at data[pos + ahead] without consuming; -1 if out of range.
  auto peek = [&](uint64_t ahead) -> int {
    uint64_t p = br.pos() + ahead;
    if (p >= size) return -1;
    return data[p];
  };
  // True if the buffer at the cursor begins with `lit`.
  auto starts_with = [&](std::string_view lit) -> bool {
    if (br.remaining() < lit.size()) return false;
    return std::memcmp(data + br.pos(), lit.data(), lit.size()) == 0;
  };
  // Advance past the next occurrence of `lit` (inclusive). Errors if absent.
  auto skip_past = [&](std::string_view lit, const char* what,
                       uint64_t start) -> Result<bool> {
    while (br.remaining() >= lit.size()) {
      if (std::memcmp(data + br.pos(), lit.data(), lit.size()) == 0) {
        br.skip(lit.size());
        return true;
      }
      br.skip(1);
    }
    return err(std::string("unterminated ") + what, base_offset + start);
  };

  while (!br.at_end()) {
    int c0 = peek(0);
    if (c0 < 0) break;

    // ---- text content between tags -----------------------------------------
    if (c0 != '<') {
      uint64_t text_start = br.pos();
      // Scan to the next '<' (or EOF).
      const uint8_t* base_p = data + br.pos();
      const void* lt = std::memchr(base_p, '<', static_cast<size_t>(br.remaining()));
      uint64_t text_len = (lt == nullptr)
                              ? br.remaining()
                              : static_cast<uint64_t>(
                                    static_cast<const uint8_t*>(lt) - base_p);
      std::string txt = decode_entities(data + text_start, text_len);
      trim(txt);
      if (!txt.empty() && !stack.empty()) {
        std::string& dst = doc.nodes_[stack.back()].text;
        if (!dst.empty()) dst.push_back(' ');
        dst += txt;
      }
      br.skip(text_len);
      continue;
    }

    uint64_t tag_start = br.pos();

    // ---- <?xml ...?> declaration / processing instruction ------------------
    if (starts_with("<?")) {
      br.skip(2);
      auto ok = skip_past("?>", "XML declaration/PI", tag_start);
      if (!ok) return ok.error();
      continue;
    }

    // ---- comment <!-- ... --> ----------------------------------------------
    if (starts_with("<!--")) {
      br.skip(4);
      auto ok = skip_past("-->", "comment", tag_start);
      if (!ok) return ok.error();
      continue;
    }

    // ---- CDATA <![CDATA[ ... ]]> -> text content ---------------------------
    if (starts_with("<![CDATA[")) {
      br.skip(9);
      uint64_t cd_start = br.pos();
      // Locate the closing "]]>".
      uint64_t p = cd_start;
      bool found = false;
      while (p + 3 <= size) {
        if (data[p] == ']' && data[p + 1] == ']' && data[p + 2] == '>') {
          found = true;
          break;
        }
        ++p;
      }
      if (!found) return err("unterminated CDATA", base_offset + tag_start);
      uint64_t cd_len = p - cd_start;
      if (!stack.empty()) {
        std::string chunk(reinterpret_cast<const char*>(data + cd_start),
                          static_cast<size_t>(cd_len));
        trim(chunk);
        if (!chunk.empty()) {
          std::string& dst = doc.nodes_[stack.back()].text;
          if (!dst.empty()) dst.push_back(' ');
          dst += chunk;
        }
      }
      auto sk = br.seek(p + 3);
      if (!sk) return sk.error();
      continue;
    }

    // ---- DOCTYPE / other <! ... > declaration ------------------------------
    // Skipped structurally: track the internal-subset '[' ... ']' so a '>' inside
    // the subset does not end the declaration early. Bounded, no DTD processing.
    if (starts_with("<!")) {
      br.skip(2);
      int bracket = 0;
      bool closed = false;
      while (!br.at_end()) {
        auto b = br.u8();
        if (!b) return b.error();
        uint8_t ch = *b;
        if (ch == '[') ++bracket;
        else if (ch == ']') { if (bracket > 0) --bracket; }
        else if (ch == '>' && bracket == 0) { closed = true; break; }
      }
      if (!closed) return err("unterminated declaration", base_offset + tag_start);
      continue;
    }

    // ---- closing tag </name> -----------------------------------------------
    if (starts_with("</")) {
      br.skip(2);
      uint64_t name_start = br.pos();
      while (!br.at_end() && !is_name_end(static_cast<uint8_t>(peek(0))))
        br.skip(1);
      std::string cname(reinterpret_cast<const char*>(data + name_start),
                        static_cast<size_t>(br.pos() - name_start));
      // Skip to and consume '>'.
      auto ok = skip_past(">", "closing tag", tag_start);
      if (!ok) return ok.error();
      if (stack.empty())
        return err("unbalanced closing tag", base_offset + tag_start);
      const std::string& open_name = doc.nodes_[stack.back()].name;
      if (open_name != cname)
        return err("mismatched closing tag </" + cname + "> for <" +
                       open_name + ">",
                   base_offset + tag_start);
      stack.pop_back();
      continue;
    }

    // ---- opening (or self-closing) tag <name attr...> ----------------------
    br.skip(1);  // consume '<'
    uint64_t name_start = br.pos();
    while (!br.at_end() && !is_name_end(static_cast<uint8_t>(peek(0))))
      br.skip(1);
    uint64_t name_len = br.pos() - name_start;
    if (name_len == 0)
      return err("empty element name", base_offset + tag_start);
    std::string ename(reinterpret_cast<const char*>(data + name_start),
                      static_cast<size_t>(name_len));

    XmlNode node;
    node.name = std::move(ename);
    node.offset = base_offset + tag_start;

    bool self_closing = false;
    // Attribute loop.
    for (;;) {
      // Skip whitespace between attributes.
      while (!br.at_end() && is_space(static_cast<uint8_t>(peek(0))))
        br.skip(1);
      int c = peek(0);
      if (c < 0) return err("unterminated tag", base_offset + tag_start);
      if (c == '>') { br.skip(1); break; }
      if (c == '/') {
        if (peek(1) == '>') { br.skip(2); self_closing = true; break; }
        return err("malformed self-closing tag", base_offset + br.pos());
      }
      // Read attribute name.
      uint64_t an_start = br.pos();
      while (!br.at_end() && !is_name_end(static_cast<uint8_t>(peek(0))))
        br.skip(1);
      uint64_t an_len = br.pos() - an_start;
      if (an_len == 0)
        return err("malformed attribute", base_offset + br.pos());
      std::string aname(reinterpret_cast<const char*>(data + an_start),
                        static_cast<size_t>(an_len));
      // Expect '=' (with optional surrounding whitespace).
      while (!br.at_end() && is_space(static_cast<uint8_t>(peek(0))))
        br.skip(1);
      if (peek(0) != '=')
        return err("attribute missing '=' value", base_offset + br.pos());
      br.skip(1);  // '='
      while (!br.at_end() && is_space(static_cast<uint8_t>(peek(0))))
        br.skip(1);
      int q = peek(0);
      if (q != '"' && q != '\'')
        return err("attribute value not quoted", base_offset + br.pos());
      br.skip(1);  // opening quote
      uint64_t val_start = br.pos();
      const uint8_t* vp = data + val_start;
      const void* qpos = std::memchr(vp, q, static_cast<size_t>(br.remaining()));
      if (qpos == nullptr)
        return err("unterminated attribute value", base_offset + tag_start);
      uint64_t val_len =
          static_cast<uint64_t>(static_cast<const uint8_t*>(qpos) - vp);
      XmlAttr attr;
      attr.name = std::move(aname);
      attr.value = decode_entities(data + val_start, val_len);
      node.attrs.push_back(std::move(attr));
      auto sk = br.seek(val_start + val_len + 1);  // past closing quote
      if (!sk) return sk.error();
    }

    uint32_t idx = static_cast<uint32_t>(doc.nodes_.size());
    doc.nodes_.push_back(std::move(node));

    if (!stack.empty()) {
      doc.nodes_[stack.back()].children.push_back(idx);
    } else {
      if (doc.root_ >= 0)
        return err("multiple root elements", base_offset + tag_start);
      doc.root_ = static_cast<int32_t>(idx);
    }

    if (!self_closing) {
      // Depth cap: stack.size() is the current nesting depth. Enforce BEFORE
      // pushing so adversarial nesting errors cleanly.
      if (static_cast<int>(stack.size()) >= kMaxXmlDepth)
        return err("XML nesting too deep", base_offset + tag_start);
      stack.push_back(idx);
    }
  }

  if (!stack.empty())
    return err("unclosed element <" + doc.nodes_[stack.back()].name + ">",
               base_offset + doc.nodes_[stack.back()].offset);

  return doc;
}

}  // namespace netvis::openvino
