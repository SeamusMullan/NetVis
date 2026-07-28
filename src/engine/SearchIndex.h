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
#include <string_view>
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

// #52/#54 structured query: a bare query is fuzzy-matched on names (legacy path).
// A query containing field tokens ("op:", "name:", "dtype:", "shape:", "params:")
// switches to FIELD mode: every token is a predicate ANDed together. Bare words in
// a field query fuzzy-match the name. Field values support a trailing/leading '*'
// glob (prefix/suffix/contains) — NOT full regex, so a linear per-entry scan over
// ~1M entries stays within the <5ms budget with no ReDoS surface. Numeric/dtype/
// shape predicates need per-entry type info that shape inference fills in AFTER the
// index is built, so they are resolved against the LIVE model at query time (via a
// hit's graph+ref+kind) rather than baked into SearchEntry.
struct QueryField {
  enum class Key : uint8_t { Name, Op, Dtype, Shape, Params } key = Key::Name;
  enum class Cmp : uint8_t { Match, Lt, Le, Gt, Ge, Eq } cmp = Cmp::Match;
  std::string text;   // lowercased match text (Name/Op/Dtype/Shape)
  int64_t num = 0;    // parsed threshold for Params (K/M/G suffix expanded)
};
struct ParsedQuery {
  std::vector<QueryField> fields;  // ANDed
  bool is_field_query = false;     // true if any explicit field: token present
  bool valid = true;               // false on a malformed token (no hits)
};
// Parse a raw query string into predicates. Pure + testable. A query with no
// "field:" token yields a single Name/Match field carrying the whole (lowercased)
// string — i.e. the legacy fuzzy path.
ParsedQuery parse_query(std::string_view q);

class SearchIndex {
 public:
  // Build from a fully-parsed model. Deterministic order.
  void build(const ir::Model& model);

  // Case-insensitive substring + subsequence fuzzy match, ranked. `limit` caps
  // results. Returns quickly on empty query (no hits).
  std::vector<SearchHit> query(const std::string& q, size_t limit = 100) const;

  // #52/#54 field-scoped query. `model` supplies dtype/shape/params for the
  // dtype:/shape:/params: predicates (resolved live, post shape-inference). A
  // query with no field token behaves exactly like query() (fuzzy name match).
  // Ranked + limited identically. Returns no hits on a malformed query.
  //
  // `types_ready` MUST be false until shape inference has published (stage ==
  // Ready): ShapeInferJob mutates ValueInfo.dtype/.shape IN PLACE on a worker
  // thread until then, so reading those fields from the main thread mid-flight is
  // a data race. When false, dtype:/shape:/params: predicates match nothing (the
  // types they'd test are not yet known anyway). op:/name: read only immutable
  // parse-time data and are always safe.
  std::vector<SearchHit> query(const std::string& q, const ir::Model& model,
                               bool types_ready, size_t limit = 100) const;

  // True if entry `e` satisfies every predicate in `pq`, given the live model for
  // type/shape/param lookups. Exposed for testing. `name_score` receives the
  // fuzzy score of the Name/Op predicate (for ranking) or 0 when none applies.
  // `types_ready` gates the dtype/shape/params predicates (see query() above).
  bool matches(const SearchEntry& e, const ParsedQuery& pq, const ir::Model& model,
               bool types_ready, int& name_score) const;

  const std::vector<SearchEntry>& entries() const { return entries_; }
  size_t size() const { return entries_.size(); }

 private:
  std::vector<SearchEntry> entries_;
};

// Score `query` (already lowercased) against `text` (already lowercased).
// Returns a positive score on match, or -1 if no match. Exposed for testing.
int fuzzy_score(std::string_view query, std::string_view text);

// Lowercase an ASCII string (non-ASCII bytes pass through unchanged). The one
// canonical fold used to prepare inputs for fuzzy_score; shared so callers
// (search bar, command palette) don't each re-implement the loop.
std::string to_lower(std::string_view s);

}  // namespace netvis
