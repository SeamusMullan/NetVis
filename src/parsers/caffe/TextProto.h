// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/TextProto.h — bounded, hostile-input-safe reader for the
// protobuf TEXT format (the `.prototxt` syntax).
//
// DECISION (#138): NetVis adds no protobuf dependency, so a text-format reader is
// built in the discipline of parsers/openvino/XmlReader.h: the document is built
// ITERATIVELY over an explicit open-message stack (no recursion, so adversarial
// nesting is a clean policy error, never a stack overflow), every scan advances
// by >= 1 byte and is bounds-checked against an exact-size buffer (no reliance on
// a trailing NUL), and every malformed input becomes a Result error carrying an
// absolute byte offset plus "(line L, column C)" — never a throw, never a crash,
// never an over-read.
//
// GRAMMAR ACCEPTED (the subset Caffe, and protobuf's own TextFormat printer, use):
//   document   := field* EOF
//   field      := NAME ( ':' value | ':'? message ) [',' | ';']
//   message    := '{' field* '}'  |  '<' field* '>'      (closer must match opener)
//   value      := scalar | message | '[' [ elem (',' elem)* ] ']'
//   elem       := scalar | message                         (no nested lists)
//   scalar     := STRING+ | ['-'] NUMBER | ['-'] IDENT     (adjacent STRINGs concatenate)
//   comment    := '#' to end of line; a UTF-8 BOM is skipped at offset 0
// Extension / Any fields (`[pkg.ext]: ...`) are rejected with a clear error.
//
// HARDENING: depth cap (kMaxTextDepth, checked BEFORE pushing), stored-node cap
// (kMaxTextNodes), unquoted-token cap (kMaxTokenBytes), decoded-string cap
// (kMaxStringBytes, after adjacent-literal concatenation). Allocation is
// O(stored nodes) <= O(file). ELISION: a caller can name (parent, field) pairs
// whose scalar values are lexed and COUNTED but never converted or stored —
// Caffe's inline `blobs { data: ... }` weights stay O(1) memory.
//
// It lives under parsers/caffe/ (not parsers/common/) until a second consumer
// (#135, TensorFlow `.pbtxt`) justifies promotion — the same YAGNI note as
// XmlReader.h. It has no Caffe knowledge.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/JobSystem.h"
#include "core/Result.h"
#include "core/SmallVec.h"

namespace netvis::caffe {

// Nesting cap, matching openvino::kMaxXmlDepth and ONNX's kMaxGraphDepth.
constexpr int kMaxTextDepth = 64;
// Default cap on stored (non-root) nodes; overridable per parse.
constexpr uint32_t kMaxTextNodes = 1u << 20;
// Longest unquoted token (field name, identifier, number).
constexpr uint64_t kMaxTokenBytes = 4096;
// Longest decoded string value, after adjacent-literal concatenation.
constexpr uint64_t kMaxStringBytes = 1ull << 20;
// Most elision rules one parse may carry (a programmer error beyond this).
constexpr size_t kMaxElideRules = 8;

enum class TextKind : uint8_t { Message, Identifier, Number, String };

// One stored field. children[] index the owning TextDocument's flat node table,
// in document order (the XmlDocument design: no pointers, nothing dangles).
struct TextNode {
  std::string name;                 // "" for the synthetic root (index 0)
  TextKind kind = TextKind::Message;
  std::string value;                // Identifier/Number: raw token incl. '-'; String: decoded bytes
  std::vector<uint32_t> children;   // Message only, document order
  uint64_t offset = 0;              // absolute offset of the field-name token
  uint64_t value_offset = 0;        // absolute offset of the value token
  SmallVec<std::pair<uint8_t, uint64_t>, 2> elided;   // (rule index, value count)
};

// Scalars of `field` inside a message named `parent` are counted, not stored.
struct ElideRule {
  std::string_view parent;
  std::string_view field;
};

struct TextParseOptions {
  std::vector<ElideRule> elide;     // <= kMaxElideRules
  uint32_t max_nodes = kMaxTextNodes;
  ProgressSink* progress = nullptr; // set(...) at most once per MiB
  // Fraction range the reader reports into, so a caller's pipeline stages do not
  // jump backwards (e.g. 0.2 .. 0.5 for the Caffe parser's "reading" stage).
  float progress_from = 0.0f;
  float progress_to = 1.0f;
  const char* progress_stage = "Reading text";
};

class TextDocument {
 public:
  // Parse [data, data+size). `base_offset` is the absolute file offset of byte 0
  // (errors and node offsets are absolute; line/column count from byte 0).
  static Result<TextDocument> parse(const uint8_t* data, uint64_t size,
                                    uint64_t base_offset,
                                    const TextParseOptions& opts = {});

  const TextNode& root() const { return nodes_[0]; }
  const TextNode& node(uint32_t i) const { return nodes_[i]; }
  size_t node_count() const { return nodes_.size(); }

  // First direct child of `parent` named `name`, or nullptr.
  const TextNode* child(const TextNode& parent, std::string_view name) const;

 private:
  TextDocument() = default;
  std::vector<TextNode> nodes_;
};

// ---- Lexer (shared with CaffeSniff) -----------------------------------------

enum class TokKind : uint8_t { End, Name, Number, String, Punct };

struct Token {
  TokKind kind = TokKind::End;
  std::string_view raw;   // exact source bytes (a String keeps its quotes)
  uint64_t offset = 0;    // absolute offset of the first byte
  char punct = 0;         // for Punct: one of { } < > [ ] : , ; -
};

// Tokenizer over an exact-size buffer. Never reads past min(size, byte_budget):
// a token that would cross the budget is an error, so a bounded sniff can never
// see a partial token as valid. Every next() consumes >= 1 byte or returns End /
// an error, so a caller's loop always terminates.
class TextLexer {
 public:
  TextLexer(const uint8_t* d, uint64_t size, uint64_t base_offset,
            uint64_t byte_budget = UINT64_MAX);

  Result<Token> next();

  // Decode a String token's escapes (\a \b \f \n \r \t \v \\ \' \" \? \ooo
  // \xH{1,2} \uHHHH \UHHHHHHHH). Errors carry the escape's absolute offset.
  Result<std::string> decode_string(const Token& t) const;

  // Quiet mode (the detection sniff): errors carry a fixed short message and skip
  // the line/column scan, so a rejecting sniff stays cheap.
  void set_quiet(bool q) { quiet_ = q; }

  // Relative position of the cursor (bytes consumed so far).
  uint64_t pos() const { return pos_; }

 private:
  Error make_error(const char* what, uint64_t rel_offset) const;

  const uint8_t* d_;
  uint64_t size_;     // full buffer size (line/column counting)
  uint64_t limit_;    // min(size, byte_budget)
  uint64_t base_;
  uint64_t pos_ = 0;
  bool quiet_ = false;
};

// ---- Value converters ---------------------------------------------------------
// All return nullopt rather than guess. Integers accept decimal, 0x hex and
// leading-0 octal with an optional '-'; std::from_chars (locale-free).
std::optional<int64_t> text_to_int64(const TextNode& n);
std::optional<uint64_t> text_to_uint64(const TextNode& n);   // rejects '-'
// Integers, decimals, exponents, an [fF] suffix, inf/infinity/nan (any case, with
// '-'). Parsed through a classic-locale istringstream (locale-independent).
std::optional<double> text_to_double(const TextNode& n);
// true/false/True/False/t/f and the numbers 1/0.
std::optional<bool> text_to_bool(const TextNode& n);

// The same converters over a raw token (used by the generic flattening path).
std::optional<int64_t> text_to_int64(std::string_view raw);
std::optional<uint64_t> text_to_uint64(std::string_view raw);
std::optional<double> text_to_double(std::string_view raw);

// 1-based (line, column) of `rel_offset` in [d, d+size): counts '\n' in
// [0, rel_offset). Only ever called on an error path.
std::pair<uint32_t, uint32_t> line_col(const uint8_t* d, uint64_t size,
                                       uint64_t rel_offset);

}  // namespace netvis::caffe
