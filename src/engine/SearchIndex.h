// engine/SearchIndex.h — background-built fuzzy search over model names.
//
// DECISION (spec §7.4): the index is a flat array of precomputed lowercase
// strings + a tag of what each entry is (node / op / value / tensor). Query is a
// linear memchr-style scan — at model sizes (<=~1M entries) this returns in
// <5ms and needs no fancier structure. Built on a worker thread; the UI reads it
// once built (ownership transfer, no shared mutation while building).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ir/IR.h"

namespace netvis {

enum class SearchKind : uint8_t { Node, OpType, Value, Tensor };

// One searchable entity. `ref` is interpreted per (graph, kind): node index,
// value index, or flat_tensors index.
struct SearchEntry {
  std::string lower;   // precomputed lowercased text
  std::string display; // original text for the results list
  SearchKind kind = SearchKind::Node;
  uint32_t graph = 0;
  uint32_t ref = 0;
};

struct SearchHit {
  uint32_t entry = 0;  // index into entries()
  int score = 0;       // higher = better (prefix > word-boundary > sub > subseq)
};

class SearchIndex {
 public:
  // Build from a fully-parsed model. Deterministic order.
  void build(const ir::Model& model);

  // Case-insensitive substring + subsequence fuzzy match, ranked. `limit` caps
  // results. Returns quickly on empty query (no hits).
  std::vector<SearchHit> query(const std::string& q, size_t limit = 100) const;

  const std::vector<SearchEntry>& entries() const { return entries_; }
  size_t size() const { return entries_.size(); }

 private:
  std::vector<SearchEntry> entries_;
};

// Score `query` (already lowercased) against `text` (already lowercased).
// Returns a positive score on match, or -1 if no match. Exposed for testing.
int fuzzy_score(std::string_view query, std::string_view text);

}  // namespace netvis
