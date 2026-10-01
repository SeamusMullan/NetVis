// SPDX-License-Identifier: Apache-2.0
// tests/test_caffe_textproto.cpp — the bounded protobuf text-format reader (#138).
//
// No Caffe knowledge here: grammar coverage (both message brackets, optional
// ':', separators, comments, CRLF, BOM, lists, adjacent string literals,
// escapes), the value converters, every cap (depth, nodes, token, string), the
// error contract (absolute offset + "line L"), elision, and the lexer's byte
// budget. Every buffer is an exact-size std::string / vector with no trailing
// NUL relied upon, so AddressSanitizer proves the reader never over-reads.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "parsers/caffe/TextProto.h"

using namespace netvis;
using namespace netvis::caffe;

namespace {

// Parse from an exact-size heap copy (no NUL terminator past the end).
Result<TextDocument> parse_text(const std::string& s, const TextParseOptions& o = {}) {
  std::vector<uint8_t> buf(s.begin(), s.end());
  return TextDocument::parse(buf.empty() ? nullptr : buf.data(), buf.size(), 0, o);
}

std::vector<const TextNode*> kids(const TextDocument& d, const TextNode& n) {
  std::vector<const TextNode*> out;
  for (uint32_t c : n.children) out.push_back(&d.node(c));
  return out;
}

const TextNode* field(const TextDocument& d, std::string_view name) {
  return d.child(d.root(), name);
}

}  // namespace

TEST_CASE("TextProto: messages, brackets, separators, comments, CRLF, BOM") {
  const std::string src =
      "\xEF\xBB\xBF# header comment\r\n"
      "name: \"net\"\r\n"
      "layer {\r\n"
      "  name: 'a', type: \"Conv\";\r\n"
      "  param < lr_mult: 1 >\r\n"
      "  sub: { x: 2 }  # trailing\r\n"
      "}\r\n"
      "layer: < name: \"b\" >\r\n";
  auto r = parse_text(src);
  REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
  const TextDocument& d = *r;
  auto top = kids(d, d.root());
  REQUIRE(top.size() == 3);
  CHECK(top[0]->name == "name");
  CHECK(top[0]->kind == TextKind::String);
  CHECK(top[0]->value == "net");
  CHECK(top[1]->name == "layer");
  CHECK(top[1]->kind == TextKind::Message);
  CHECK(top[2]->name == "layer");
  CHECK(top[2]->kind == TextKind::Message);

  auto l0 = kids(d, *top[1]);
  REQUIRE(l0.size() == 4);
  CHECK(l0[0]->name == "name");
  CHECK(l0[0]->value == "a");
  CHECK(l0[1]->name == "type");
  CHECK(l0[1]->value == "Conv");
  CHECK(l0[2]->name == "param");
  CHECK(l0[2]->kind == TextKind::Message);
  REQUIRE(l0[2]->children.size() == 1);
  CHECK(d.node(l0[2]->children[0]).name == "lr_mult");
  CHECK(d.node(l0[2]->children[0]).kind == TextKind::Number);
  CHECK(d.node(l0[2]->children[0]).value == "1");
  CHECK(l0[3]->name == "sub");
  REQUIRE(l0[3]->children.size() == 1);
  CHECK(d.node(l0[3]->children[0]).value == "2");

  auto l1 = kids(d, *top[2]);
  REQUIRE(l1.size() == 1);
  CHECK(l1[0]->value == "b");

  // Offsets are absolute byte positions of the field name / value.
  CHECK(top[0]->offset == src.find("name: \"net\""));
  CHECK(top[0]->value_offset == src.find("\"net\""));

  // An empty document is a valid empty message.
  auto e = parse_text("");
  REQUIRE(e);
  CHECK(e->root().children.empty());
}

TEST_CASE("TextProto: scalar converters") {
  const std::string src =
      "a: 1 b: -7 c: 0x1F d: 017 e: 1e-4 f: .5 g: 1. h: 2.5f i: -inf j: nan\n"
      "k: true l: t m: False n: IDENT o: 9223372036854775808 p: -1 q: 0.75\n"
      "r: -9223372036854775808 s: 0 u: Infinity\n";
  auto r = parse_text(src);
  REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
  const TextDocument& d = *r;
  auto at = [&](const char* n) -> const TextNode& {
    const TextNode* p = field(d, n);
    REQUIRE(p != nullptr);
    return *p;
  };
  CHECK(text_to_int64(at("a")) == 1);
  CHECK(text_to_int64(at("b")) == -7);
  CHECK(at("b").kind == TextKind::Number);
  CHECK(at("b").value == "-7");
  CHECK(text_to_int64(at("c")) == 31);
  CHECK(text_to_int64(at("d")) == 15);
  auto e = text_to_double(at("e"));
  REQUIRE(e.has_value());
  CHECK(*e == doctest::Approx(1e-4));
  CHECK_FALSE(text_to_int64(at("e")).has_value());
  CHECK(text_to_double(at("f")) == 0.5);
  CHECK(text_to_double(at("g")) == 1.0);
  CHECK(text_to_double(at("h")) == 2.5);
  auto inf = text_to_double(at("i"));
  REQUIRE(inf.has_value());
  CHECK(std::isinf(*inf));
  CHECK(*inf < 0);
  CHECK(at("i").kind == TextKind::Identifier);
  auto nan = text_to_double(at("j"));
  REQUIRE(nan.has_value());
  CHECK(std::isnan(*nan));
  CHECK(text_to_bool(at("k")) == true);
  CHECK(text_to_bool(at("l")) == true);
  CHECK(text_to_bool(at("m")) == false);
  CHECK(text_to_bool(at("s")) == false);
  CHECK(text_to_bool(at("a")) == true);
  CHECK_FALSE(text_to_bool(at("n")).has_value());
  CHECK(at("n").kind == TextKind::Identifier);
  CHECK(at("n").value == "IDENT");
  CHECK_FALSE(text_to_int64(at("o")).has_value());
  CHECK(text_to_uint64(at("o")) == 9223372036854775808ull);
  CHECK_FALSE(text_to_uint64(at("p")).has_value());
  CHECK(text_to_int64(at("p")) == -1);
  CHECK(text_to_double(at("q")) == 0.75);   // exact
  CHECK(text_to_int64(at("r")) == INT64_MIN);
  auto pinf = text_to_double(at("u"));
  REQUIRE(pinf.has_value());
  CHECK(std::isinf(*pinf));
  CHECK(*pinf > 0);
  // Identifiers and strings are not numbers.
  CHECK_FALSE(text_to_int64(at("n")).has_value());
  CHECK_FALSE(text_to_double(at("n")).has_value());
  // Raw-view converters agree; junk is rejected rather than guessed.
  CHECK_FALSE(text_to_int64(std::string_view("08")).has_value());
  CHECK_FALSE(text_to_int64(std::string_view("1x")).has_value());
  CHECK_FALSE(text_to_double(std::string_view("1e")).has_value());
  CHECK_FALSE(text_to_double(std::string_view("1e999")).has_value());
}

TEST_CASE("TextProto: strings, escapes, adjacent literals") {
  const std::string src = std::string("s1: \"a\\n\\t\\\\\\\"'\"\n") +
                          "s2: 'it\\'s'\n" +
                          "s3: \"\\101\\x42\xC3\xA9\"\n" +
                          "s4: \"a\" 'b' \"c\"\n" +
                          "s5: \"\\u00e9\\U0001F600\\?\"\n";
  auto r = parse_text(src);
  REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
  const TextDocument& d = *r;
  REQUIRE(field(d, "s1") != nullptr);
  CHECK(field(d, "s1")->value == std::string("a\n\t\\\"'"));
  CHECK(field(d, "s2")->value == "it's");
  CHECK(field(d, "s3")->value == std::string("AB") + "\xC3\xA9");
  CHECK(field(d, "s4")->value == "abc");
  CHECK(field(d, "s5")->value == std::string("\xC3\xA9") + "\xF0\x9F\x98\x80" + "?");
  CHECK(field(d, "s4")->kind == TextKind::String);
}

TEST_CASE("TextProto: lists") {
  auto r = parse_text("dim: [1, 2, 3]\ns: [{a: 1}, {a: 2}]\nnone: []\nm: [{ in: [4, 5] }]\n");
  REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
  const TextDocument& d = *r;
  int dims = 0, ss = 0, nones = 0;
  for (const TextNode* k : kids(d, d.root())) {
    if (k->name == "dim") {
      ++dims;
      CHECK(k->kind == TextKind::Number);
    }
    if (k->name == "s") {
      ++ss;
      CHECK(k->kind == TextKind::Message);
      REQUIRE(k->children.size() == 1);
      CHECK(d.node(k->children[0]).name == "a");
    }
    if (k->name == "none") ++nones;
  }
  CHECK(dims == 3);
  CHECK(ss == 2);
  CHECK(nones == 0);
  const TextNode* m = field(d, "m");
  REQUIRE(m != nullptr);
  CHECK(m->children.size() == 2);   // in: 4, in: 5

  auto nested = parse_text("x: [[1]]");
  REQUIRE_FALSE(nested);
  CHECK(nested.error().message.find("nested") != std::string::npos);
  CHECK_FALSE(parse_text("x: [1,]"));
  CHECK_FALSE(parse_text("x: [1 2]"));
  CHECK_FALSE(parse_text("x: [1, 2"));
}

TEST_CASE("TextProto: depth cap is 64, error at the 65th opener") {
  std::string ok, bad;
  for (int i = 0; i < 64; ++i) ok += "a { ";
  for (int i = 0; i < 64; ++i) ok += "} ";
  CHECK(parse_text(ok));

  for (int i = 0; i < 65; ++i) bad += "a { ";
  for (int i = 0; i < 65; ++i) bad += "} ";
  auto r = parse_text(bad);
  REQUIRE_FALSE(r);
  CHECK(r.error().offset == 64u * 4u);   // the 65th "a"
  CHECK(r.error().message.find("deep") != std::string::npos);
}

TEST_CASE("TextProto: node cap") {
  TextParseOptions o;
  o.max_nodes = 10;
  std::string ten, eleven;
  for (int i = 0; i < 10; ++i) ten += "f: 1\n";
  eleven = ten + "f: 1\n";
  CHECK(parse_text(ten, o));
  auto r = parse_text(eleven, o);
  REQUIRE_FALSE(r);
  CHECK(r.error().offset == 50u);
}

TEST_CASE("TextProto: token and string caps") {
  const std::string id4096(4096, 'a');
  CHECK(parse_text("v: " + id4096));
  CHECK(parse_text(id4096 + ": 1"));
  auto r = parse_text("v: " + id4096 + "a");
  REQUIRE_FALSE(r);
  CHECK(r.error().offset == 3u);

  const std::string s1m(1u << 20, 'x');
  CHECK(parse_text("s: \"" + s1m + "\""));
  CHECK_FALSE(parse_text("s: \"" + s1m + "x\""));
  // The cap applies after adjacent-literal concatenation.
  CHECK_FALSE(parse_text("s: \"" + s1m + "\" \"y\""));
}

TEST_CASE("TextProto: errors carry an absolute offset and line/column") {
  const std::string pre = "x: 1\ny: 2\n";   // 10 bytes; errors land on line 3
  struct Case {
    std::string tail;
    uint64_t offset;
    const char* needle;
  };
  const std::vector<Case> cases = {
      {"z: \"abc", 13, "unterminated"},
      {"z: \"ab\ncd\"", 13, "unterminated"},
      {"layer {\n  a: 1\n", 10, "unclosed"},
      {"}", 10, "unbalanced"},
      {"m { a: 1 >", 19, "mismatched"},
      {"z: \"a\\qb\"", 15, "escape"},
      {"z: \"\\xg\"", 14, "\\x"},
      {"z: \"\\uD800\"", 14, "code point"},
      {"@", 10, "unexpected"},
      {"\xC3\xA9", 10, "unexpected"},
      {"[ext.field]: 1", 10, "extension"},
      {"a 1", 12, "missing ':'"},
      {"a: }", 13, "missing value"},
  };
  for (const Case& c : cases) {
    auto r = parse_text(pre + c.tail);
    REQUIRE_FALSE_MESSAGE(r, c.tail);
    CHECK_MESSAGE(r.error().offset == c.offset, c.tail);
    CHECK_MESSAGE(r.error().message.find("line 3") != std::string::npos,
                  r.error().message);
    CHECK_MESSAGE(r.error().message.find(c.needle) != std::string::npos,
                  r.error().message);
  }
  // Column counts from 1 at the start of the line.
  auto r = parse_text(pre + "  @");
  REQUIRE_FALSE(r);
  CHECK(r.error().message.find("column 3") != std::string::npos);
}

TEST_CASE("TextProto: elided values are counted, never stored") {
  TextParseOptions o;
  o.elide = {{"blobs", "data"}, {"blobs", "diff"}};
  auto r = parse_text(
      "blobs { data: 1 data: 2 data: [3, 4] diff: 5 shape { dim: 4 } }\n"
      "other { data: 9 }\n",
      o);
  REQUIRE_MESSAGE(r, (r ? "" : r.error().message));
  const TextDocument& d = *r;
  const TextNode* b = field(d, "blobs");
  REQUIRE(b != nullptr);
  REQUIRE(b->children.size() == 1);
  CHECK(d.node(b->children[0]).name == "shape");
  REQUIRE(b->elided.size() == 2);
  CHECK(b->elided[0].first == 0);
  CHECK(b->elided[0].second == 4u);
  CHECK(b->elided[1].first == 1);
  CHECK(b->elided[1].second == 1u);
  // The rule is scoped to its parent message name.
  const TextNode* other = field(d, "other");
  REQUIRE(other != nullptr);
  CHECK(other->children.size() == 1);
  CHECK(other->elided.empty());
}

TEST_CASE("TextProto: the lexer never reads past its byte budget") {
  const std::string src =
      "name: \"abcdefghij\" layer { type: 12345 } tail_identifier_long";
  std::vector<uint8_t> buf(src.begin(), src.end());
  for (uint64_t budget = 0; budget <= buf.size(); ++budget) {
    TextLexer lex(buf.data(), buf.size(), 0, budget);
    for (int guard = 0; guard < 64; ++guard) {
      auto t = lex.next();
      if (!t) break;
      if (t->kind == TokKind::End) break;
      CHECK(t->offset + t->raw.size() <= budget);
    }
  }
  // A whole buffer lexes to the expected token kinds.
  TextLexer lex(buf.data(), buf.size(), 100);
  auto t = lex.next();
  REQUIRE(t);
  CHECK(t->kind == TokKind::Name);
  CHECK(t->offset == 100u);   // base offset applied
  t = lex.next();
  REQUIRE(t);
  CHECK(t->kind == TokKind::Punct);
  CHECK(t->punct == ':');
  t = lex.next();
  REQUIRE(t);
  CHECK(t->kind == TokKind::String);
  auto s = lex.decode_string(*t);
  REQUIRE(s);
  CHECK(*s == "abcdefghij");
}
