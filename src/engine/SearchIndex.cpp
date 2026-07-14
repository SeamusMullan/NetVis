// engine/SearchIndex.cpp — build + query the fuzzy search index.
//
// DECISION (spec §7.4): the index is a flat, precomputed array. build() runs
// once on a worker thread (ownership transfer to the UI when done); query() is a
// single linear scan with a tight scoring inner loop and no per-entry heap
// allocation. At <=~1M entries this stays under the 5ms budget.
#include "engine/SearchIndex.h"

#include <algorithm>
#include <cctype>
#include <unordered_set>

namespace netvis {

namespace {

// Lowercase an ASCII/UTF-8 string. We only fold ASCII (model identifiers are
// effectively ASCII); non-ASCII bytes pass through unchanged so bytes still
// compare consistently between query and entry.
std::string to_lower(std::string_view s) {
  std::string out;
  out.resize(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    out[i] = static_cast<char>(std::tolower(c));
  }
  return out;
}

// True if `c` is a word-boundary separator: a match starting right after one of
// these (or at position 0) is a "word start".
bool is_boundary(char c) {
  return c == '/' || c == '.' || c == '_' || c == ' ' || c == '-' || c == ':';
}

// Append one entry, precomputing its lowercase form.
void add_entry(std::vector<SearchEntry>& out, std::string_view display,
               SearchKind kind, uint32_t graph, uint32_t ref) {
  if (display.empty()) return;
  SearchEntry e;
  e.display.assign(display.begin(), display.end());
  e.lower = to_lower(display);
  e.kind = kind;
  e.graph = graph;
  e.ref = ref;
  out.push_back(std::move(e));
}

}  // namespace

int fuzzy_score(std::string_view q, std::string_view text) {
  if (q.empty()) return -1;
  if (text.empty()) return -1;

  // Exact match — best possible.
  if (q == text) return 1000;

  const size_t qn = q.size();
  const size_t tn = text.size();

  // Prefix: ~800 minus how much longer the text is than the query.
  if (tn >= qn && text.compare(0, qn, q) == 0) {
    int diff = static_cast<int>(tn - qn);
    int score = 800 - diff;
    return score < 601 ? 601 : score;  // keep strictly above word-boundary tier
  }

  // Substring search: find first occurrence, note if it is at a word boundary.
  size_t found = text.find(q);
  if (found != std::string_view::npos) {
    bool boundary = (found == 0) || is_boundary(text[found - 1]);
    if (boundary) {
      // Word-boundary substring ~600 (minus position so earlier is better).
      int score = 600 - static_cast<int>(found);
      return score < 401 ? 401 : score;  // stay above plain-substring tier
    }
    // Plain substring ~400 minus position.
    int score = 400 - static_cast<int>(found);
    return score < 201 ? 201 : score;  // stay above subsequence tier
  }

  // Subsequence: all query chars appear in order. ~200 minus gap count.
  size_t ti = 0;
  size_t gaps = 0;
  bool matching = false;
  for (size_t qi = 0; qi < qn; ++qi) {
    char qc = q[qi];
    bool hit = false;
    while (ti < tn) {
      if (text[ti] == qc) {
        if (matching && ti != 0) {
          // gap since previous matched char handled below
        }
        ++ti;
        hit = true;
        break;
      }
      ++ti;
      if (matching) ++gaps;  // count skipped chars between matches
    }
    if (!hit) return -1;  // ran out of text before matching a query char
    matching = true;
  }
  int score = 200 - static_cast<int>(gaps);
  return score < 1 ? 1 : score;  // any subsequence match beats no match
}

void SearchIndex::build(const ir::Model& model) {
  entries_.clear();

  // Rough reserve to avoid reallocations on large models.
  size_t est = 0;
  for (const auto& g : model.graphs)
    est += g.nodes.size() * 2 + g.values.size();
  if (!model.has_graph) est += model.flat_tensors.size();
  entries_.reserve(est);

  for (uint32_t gi = 0; gi < model.graphs.size(); ++gi) {
    const ir::Graph& g = model.graphs[gi];

    // Nodes: display is the node name, or the op_type when unnamed. Also emit
    // one OpType entry per distinct op_type in this graph (deterministic order:
    // first appearance).
    std::unordered_set<uint32_t> seen_ops;
    for (uint32_t ni = 0; ni < g.nodes.size(); ++ni) {
      const ir::Node& n = g.nodes[ni];
      std::string_view name = model.str(n.name);
      std::string_view op = model.str(n.op_type);
      std::string_view display = !name.empty() ? name : op;
      add_entry(entries_, display, SearchKind::Node, gi, ni);

      if (n.op_type.valid() && seen_ops.insert(n.op_type.id).second) {
        add_entry(entries_, op, SearchKind::OpType, gi, ni);
      }
    }

    // Named values (graph edges).
    for (uint32_t vi = 0; vi < g.values.size(); ++vi) {
      std::string_view vname = model.str(g.values[vi].name);
      add_entry(entries_, vname, SearchKind::Value, gi, vi);
    }
  }

  // Tensor-table mode: flat tensors are searchable directly.
  if (!model.has_graph) {
    for (uint32_t ti = 0; ti < model.flat_tensors.size(); ++ti) {
      std::string_view tname = model.str(model.flat_tensors[ti].name);
      add_entry(entries_, tname, SearchKind::Tensor, 0, ti);
    }
  }
}

std::vector<SearchHit> SearchIndex::query(const std::string& q,
                                          size_t limit) const {
  std::vector<SearchHit> hits;
  if (q.empty() || limit == 0) return hits;

  const std::string ql = to_lower(q);

  // PERF: single linear pass, no allocation inside the loop. fuzzy_score works
  // on the precomputed .lower views; hits are collected into one vector.
  hits.reserve(entries_.size() < 256 ? entries_.size() : 256);
  for (uint32_t i = 0; i < entries_.size(); ++i) {
    int s = fuzzy_score(ql, entries_[i].lower);
    if (s >= 0) hits.push_back(SearchHit{i, s});
  }

  // Rank by score desc, stable by entry index asc (deterministic). partial_sort
  // only up to `limit` since callers show a bounded list.
  size_t keep = std::min(limit, hits.size());
  auto cmp = [](const SearchHit& a, const SearchHit& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.entry < b.entry;
  };
  if (keep < hits.size()) {
    std::partial_sort(hits.begin(), hits.begin() + keep, hits.end(), cmp);
    hits.resize(keep);
  } else {
    std::sort(hits.begin(), hits.end(), cmp);
  }
  return hits;
}

}  // namespace netvis
