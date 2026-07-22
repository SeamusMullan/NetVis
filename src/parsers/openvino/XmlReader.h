// parsers/openvino/XmlReader.h — minimal, bounded, hostile-input-safe reader for
// the XML subset OpenVINO IR (.xml topology) uses.
//
// DECISION (v0.5.0 plan §"The one genuinely new piece"): NetVis adds NO XML
// dependency. A general XML parser is a hostile-input minefield (entity
// expansion / billion-laughs / XXE / DTD) and a single-binary liability. We
// build a tiny reader in the same spirit as onnx/WireReader.h: it walks STRUCTURE
// only, over the bounds-checked ByteReader, and can never over-run. Malformed or
// truncated input becomes a Result error carrying a byte offset — never a throw,
// never a crash, never an unbounded loop or allocation.
//
// SUPPORTED SUBSET (exactly what OpenVINO IR needs):
//   - elements, nesting, self-closing tags (<port .../>);
//   - attributes name="value" AND name='value';
//   - element text content (needed: <dim>3</dim> shape dims are text);
//   - skips the <?xml ...?> declaration, <!-- comments -->, <!DOCTYPE ...>;
//   - reads <![CDATA[...]]> content structurally.
//
// HARDENING (non-negotiable — mirrors WireReader discipline):
//   - bounded nesting depth (kMaxXmlDepth) -> adversarial nesting errors, never a
//     stack overflow (the document is built ITERATIVELY, so construction itself
//     never recurses; the cap is a policy limit that errors cleanly);
//   - NO entity expansion beyond the five XML built-ins (&lt; &gt; &amp; &apos;
//     &quot;) as literal single-pass replacement; every other &...; is copied
//     verbatim. No DTD, no external entities, ever -> immune to billion-laughs /
//     XXE by construction;
//   - every read is bounds-checked; truncation / unterminated tag / unterminated
//     attribute -> Result error with a byte offset, not a loop;
//   - total allocation is O(file size): the mmap bounds the element/attribute
//     count, and each token consumes >=1 byte so the tokenizer always terminates.
//
// It lives under parsers/openvino/ (not core/) until a second consumer justifies
// promotion (YAGNI) — see the plan's "where the XML reader ultimately lives".
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/Result.h"

namespace netvis::openvino {

// Guard against adversarial nesting, matching ONNX's kMaxGraphDepth = 64.
constexpr int kMaxXmlDepth = 64;

// One attribute: name and its (entity-decoded) value, both owned copies.
struct XmlAttr {
  std::string name;
  std::string value;
};

// One element in the parsed tree. children[] are indices into the owning
// XmlDocument's flat node table (not pointers), so the tree is a handful of
// contiguous vectors and never dangles across document moves.
struct XmlNode {
  std::string name;                // element tag name, e.g. "layer"
  std::vector<XmlAttr> attrs;      // in document order
  std::vector<uint32_t> children;  // -> XmlDocument node indices, in doc order
  std::string text;                // trimmed concatenated direct text content
  uint64_t offset = 0;             // absolute byte offset of this element's '<'

  // Attribute value by name, or nullptr if absent. O(#attrs); attr counts per
  // element are tiny in IR, so a linear scan is the right tradeoff.
  const std::string* attr(std::string_view key) const;

  // Attribute parsed as a base-10 int64, or nullopt if absent / unparseable.
  std::optional<int64_t> attr_int(std::string_view key) const;
};

// A fully-parsed XML document: a flat, index-linked node table with one root.
// Move-only-friendly (plain members). Construct via XmlDocument::parse.
class XmlDocument {
 public:
  // Parse the whole [data, data+size) range. `base_offset` is the absolute file
  // offset of byte 0 (for error reporting; 0 for a whole-file parse). Returns a
  // Result error (with a byte offset) on any malformed / truncated / too-deep
  // input; never throws, never over-reads.
  static Result<XmlDocument> parse(const uint8_t* data, uint64_t size,
                                   uint64_t base_offset = 0);

  bool has_root() const { return root_ >= 0; }
  const XmlNode& root() const { return nodes_[static_cast<size_t>(root_)]; }
  const XmlNode& node(uint32_t i) const { return nodes_[i]; }
  size_t node_count() const { return nodes_.size(); }

  // First direct child of `parent` whose tag name == `name`, or nullptr.
  const XmlNode* child(const XmlNode& parent, std::string_view name) const;

 private:
  XmlDocument() = default;

  std::vector<XmlNode> nodes_;
  int32_t root_ = -1;
};

}  // namespace netvis::openvino
