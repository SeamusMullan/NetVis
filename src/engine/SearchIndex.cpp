// engine/SearchIndex.cpp — build + query the fuzzy search index.
//
// DECISION (spec §7.4): the index is a flat, precomputed array. build() runs
// once on a worker thread (ownership transfer to the UI when done); query() is a
// single linear scan with a tight scoring inner loop and no per-entry heap
// allocation. At <=~1M entries this stays under the 5ms budget.
#include "engine/SearchIndex.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace netvis {

// Lowercase an ASCII/UTF-8 string. We only fold ASCII (model identifiers are
// effectively ASCII); non-ASCII bytes pass through unchanged so bytes still
// compare consistently between query and entry. Declared in SearchIndex.h so
// the command palette and other callers share this one fold.
std::string to_lower(std::string_view s) {
  std::string out;
  out.resize(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    out[i] = static_cast<char>(std::tolower(c));
  }
  return out;
}

namespace {

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

// --- #52/#54 structured (field-scoped) query --------------------------------

namespace {

// Leading/trailing '*' glob (defined below; forward-declared for contains_or_glob).
bool glob_match(std::string_view pat, std::string_view text);

// Format a shape into `buf` (no heap alloc): "[a, b, c]", "?" for a dynamic dim,
// "[]". Shapes are tiny; a stack buffer suffices. Returns a view into `buf`. Only
// the digits/brackets/commas appear, all case-insensitive, so no lowercasing is
// ever needed on the result.
std::string_view shape_to_buf(const SmallVec<int64_t, 6>& shape, char* buf,
                              size_t cap) {
  int n = 0;
  auto put = [&](const char* s) {
    while (*s && static_cast<size_t>(n) + 1 < cap) buf[n++] = *s++;
  };
  put("[");
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) put(", ");
    if (shape[i] < 0) {
      put("?");
    } else {
      char num[24];
      std::snprintf(num, sizeof(num), "%lld", static_cast<long long>(shape[i]));
      put(num);
    }
  }
  put("]");
  buf[n] = '\0';
  return std::string_view(buf, static_cast<size_t>(n));
}

// Product of non-negative dims; 0 if any dim is dynamic/unset (mirrors
// TensorRef::elem_count for a ValueInfo shape).
int64_t shape_params(const SmallVec<int64_t, 6>& shape) {
  if (shape.empty()) return 0;
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) return 0;
    n *= d;
  }
  return n;
}

// "contains, or glob if the pattern has a '*'". Both args already lowercased.
bool contains_or_glob(std::string_view pat, std::string_view text) {
  if (pat.find('*') != std::string_view::npos) return glob_match(pat, text);
  return text.find(pat) != std::string_view::npos;
}

// Lowercase `s` into `buf` (no heap alloc); returns a view. Truncates at cap.
// For short interned tokens (op types, dtype labels) matched per entry per frame.
std::string_view lower_to_buf(std::string_view s, char* buf, size_t cap) {
  size_t n = std::min(s.size(), cap - 1);
  for (size_t i = 0; i < n; ++i)
    buf[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
  buf[n] = '\0';
  return std::string_view(buf, n);
}

// The shape an entry refers to (Value or flat Tensor), or nullptr if the entry
// kind carries no single shape or the index is out of range.
const SmallVec<int64_t, 6>* entry_shape(const SearchEntry& e, const ir::Graph* g,
                                        const ir::Model& model) {
  if (e.kind == SearchKind::Value) {
    if (g == nullptr || e.ref >= g->values.size()) return nullptr;
    return &g->values[e.ref].shape;
  }
  if (e.kind == SearchKind::Tensor) {
    if (e.ref >= model.flat_tensors.size()) return nullptr;
    return &model.flat_tensors[e.ref].shape;
  }
  return nullptr;
}

// Leading/trailing '*' glob only (prefix/suffix/contains); no '*' => whole-string
// containment is decided by the caller. Both args already lowercased. A bare "*"
// matches anything.
bool glob_match(std::string_view pat, std::string_view text) {
  if (pat.empty()) return true;
  if (pat == "*") return true;
  bool star_lead = pat.front() == '*';
  bool star_trail = pat.back() == '*';
  std::string_view core = pat;
  if (star_lead) core.remove_prefix(1);
  if (star_trail && !core.empty()) core.remove_suffix(1);
  if (core.empty()) return true;  // "*" / "**"
  if (star_lead && star_trail) return text.find(core) != std::string_view::npos;
  if (star_lead) return text.size() >= core.size() &&
                        text.compare(text.size() - core.size(), core.size(), core) == 0;
  if (star_trail) return text.size() >= core.size() &&
                         text.compare(0, core.size(), core) == 0;
  return text == core;  // no wildcard: exact (used for dtype/shape equality)
}

// Parse a params: threshold value like ">1M", ">=1000", "2.5G", with K/M/G/B
// suffix. Returns false if unparseable. Sets cmp + num.
bool parse_params_value(std::string_view v, QueryField::Cmp& cmp, int64_t& num) {
  cmp = QueryField::Cmp::Eq;
  size_t i = 0;
  if (i < v.size() && (v[i] == '>' || v[i] == '<')) {
    bool gt = v[i] == '>';
    ++i;
    bool eq = i < v.size() && v[i] == '=';
    if (eq) ++i;
    cmp = gt ? (eq ? QueryField::Cmp::Ge : QueryField::Cmp::Gt)
             : (eq ? QueryField::Cmp::Le : QueryField::Cmp::Lt);
  } else if (i < v.size() && v[i] == '=') {
    ++i;
    cmp = QueryField::Cmp::Eq;
  }
  if (i >= v.size()) return false;
  // Parse a decimal number (allow a fractional part for suffix scaling) WITHOUT
  // exceptions or overflow UB: accumulate as double, saturate to INT64 at the end
  // (std::stod would throw std::out_of_range on a huge magnitude, and there is no
  // try/catch on the per-frame query path — that would reach std::terminate).
  double val = 0.0, frac_div = 0.0;
  bool seen_digit = false, seen_dot = false;
  for (; i < v.size(); ++i) {
    char c = v[i];
    if (c >= '0' && c <= '9') {
      seen_digit = true;
      if (seen_dot) { frac_div *= 10.0; val += (c - '0') / frac_div; }
      else { val = val * 10.0 + (c - '0'); }
    } else if (c == '.' && !seen_dot) {
      seen_dot = true;
      frac_div = 1.0;
    } else {
      break;
    }
  }
  if (!seen_digit) return false;
  // Optional K/M/G/B suffix.
  double mult = 1.0;
  if (i < v.size()) {
    char c = static_cast<char>(std::tolower(static_cast<unsigned char>(v[i])));
    if (c == 'k') mult = 1e3;
    else if (c == 'm') mult = 1e6;
    else if (c == 'g' || c == 'b') mult = 1e9;
    else return false;  // trailing garbage
    ++i;
    if (i != v.size()) return false;  // more after the suffix
  }
  double scaled = val * mult;
  // Saturate: double->int64 is UB when out of range.
  const double kMax = 9.2e18;  // just under INT64_MAX
  num = scaled >= kMax ? INT64_MAX : static_cast<int64_t>(scaled);
  return true;
}

// Recognize a field key token; returns true + sets key if `k` names a field.
bool field_key_from(std::string_view k, QueryField::Key& key) {
  std::string kl = to_lower(k);
  if (kl == "op") { key = QueryField::Key::Op; return true; }
  if (kl == "name") { key = QueryField::Key::Name; return true; }
  if (kl == "dtype") { key = QueryField::Key::Dtype; return true; }
  if (kl == "shape") { key = QueryField::Key::Shape; return true; }
  if (kl == "params") { key = QueryField::Key::Params; return true; }
  return false;
}

}  // namespace

ParsedQuery parse_query(std::string_view q) {
  ParsedQuery pq;
  // Tokenize on whitespace.
  std::vector<std::string_view> tokens;
  size_t i = 0;
  while (i < q.size()) {
    while (i < q.size() && q[i] == ' ') ++i;
    size_t start = i;
    while (i < q.size() && q[i] != ' ') ++i;
    if (i > start) tokens.push_back(q.substr(start, i - start));
  }
  if (tokens.empty()) { pq.valid = false; return pq; }

  // First pass: detect any field token.
  bool any_field = false;
  for (std::string_view t : tokens) {
    size_t colon = t.find(':');
    if (colon != std::string_view::npos && colon > 0) {
      QueryField::Key k;
      if (field_key_from(t.substr(0, colon), k)) { any_field = true; break; }
    }
  }

  if (!any_field) {
    // Legacy fuzzy path: the whole query is one Name/Match predicate.
    QueryField f;
    f.key = QueryField::Key::Name;
    f.cmp = QueryField::Cmp::Match;
    f.text = to_lower(q);
    // Trim to the non-space span so a padded query still matches.
    pq.fields.push_back(std::move(f));
    pq.is_field_query = false;
    return pq;
  }

  pq.is_field_query = true;
  for (std::string_view t : tokens) {
    size_t colon = t.find(':');
    QueryField::Key k;
    if (colon != std::string_view::npos && colon > 0 &&
        field_key_from(t.substr(0, colon), k)) {
      std::string_view val = t.substr(colon + 1);
      if (val.empty()) { pq.valid = false; return pq; }
      QueryField f;
      f.key = k;
      if (k == QueryField::Key::Params) {
        if (!parse_params_value(val, f.cmp, f.num)) { pq.valid = false; return pq; }
      } else {
        f.cmp = QueryField::Cmp::Match;
        f.text = to_lower(val);
      }
      pq.fields.push_back(std::move(f));
    } else {
      // Bare word inside a field query -> a Name/Match predicate.
      QueryField f;
      f.key = QueryField::Key::Name;
      f.cmp = QueryField::Cmp::Match;
      f.text = to_lower(t);
      pq.fields.push_back(std::move(f));
    }
  }
  return pq;
}

// Match a name/op text predicate: a wildcard pattern uses glob; a plain pattern
// uses fuzzy_score (so ranking works). Returns the score (>=0) or -1 on no match.
namespace {
int name_match_score(std::string_view pat, std::string_view text) {
  if (pat.find('*') != std::string_view::npos)
    return glob_match(pat, text) ? 500 : -1;  // wildcard: flat score
  return fuzzy_score(pat, text);              // plain: ranked fuzzy
}
}  // namespace

bool SearchIndex::matches(const SearchEntry& e, const ParsedQuery& pq,
                          const ir::Model& model, bool types_ready,
                          int& name_score) const {
  name_score = 0;
  bool name_applied = false;

  // Resolve type info lazily (only when a predicate needs it).
  const ir::Graph* g =
      e.graph < model.graphs.size() ? &model.graphs[e.graph] : nullptr;

  for (const QueryField& f : pq.fields) {
    switch (f.key) {
      case QueryField::Key::Name: {
        int s = name_match_score(f.text, e.lower);
        if (s < 0) return false;
        if (!name_applied) { name_score = s; name_applied = true; }
        break;
      }
      case QueryField::Key::Op: {
        // Only Node / OpType entries have an op. op_type is immutable parse-time
        // data (safe regardless of types_ready). Lower into a stack buffer — no
        // per-entry heap alloc on this hot (per-entry, per-frame) path.
        if (g == nullptr) return false;
        if (e.kind != SearchKind::Node && e.kind != SearchKind::OpType) return false;
        if (e.ref >= g->nodes.size()) return false;
        char buf[64];
        std::string_view op = lower_to_buf(model.str(g->nodes[e.ref].op_type), buf,
                                           sizeof(buf));
        int s = name_match_score(f.text, op);
        if (s < 0) return false;
        if (!name_applied) { name_score = s; name_applied = true; }
        break;
      }
      case QueryField::Key::Dtype: {
        // dtype is filled by shape inference on a worker thread; reading it before
        // Ready is a data race. Gate it (an un-inferred dtype is Unknown anyway).
        if (!types_ready) return false;
        ir::DType dt = ir::DType::Unknown;
        std::string_view dname;
        char lbuf[32];
        if (e.kind == SearchKind::Value) {
          if (g == nullptr || e.ref >= g->values.size()) return false;
          dt = g->values[e.ref].dtype;
          dname = ir::dtype_name(dt);  // already lowercase literal — no alloc
        } else if (e.kind == SearchKind::Tensor) {
          if (e.ref >= model.flat_tensors.size()) return false;
          const ir::TensorRef& t = model.flat_tensors[e.ref];
          dt = t.dtype;
          if (dt == ir::DType::Unknown && t.dtype_label.valid())
            dname = lower_to_buf(model.str(t.dtype_label), lbuf, sizeof(lbuf));
          else
            dname = ir::dtype_name(dt);
        } else {
          return false;  // Node/OpType have no single dtype
        }
        if (!contains_or_glob(f.text, dname)) return false;
        break;
      }
      case QueryField::Key::Shape: {
        if (!types_ready) return false;  // shape filled by inference (see Dtype)
        const SmallVec<int64_t, 6>* shp = entry_shape(e, g, model);
        if (shp == nullptr) return false;
        char sbuf[96];
        std::string_view ss = shape_to_buf(*shp, sbuf, sizeof(sbuf));  // no alloc
        if (!contains_or_glob(f.text, ss)) return false;
        break;
      }
      case QueryField::Key::Params: {
        if (!types_ready) return false;  // shape filled by inference (see Dtype)
        const SmallVec<int64_t, 6>* shp = entry_shape(e, g, model);
        if (shp == nullptr) return false;
        int64_t p = shape_params(*shp);
        bool ok = false;
        switch (f.cmp) {
          case QueryField::Cmp::Lt: ok = p < f.num; break;
          case QueryField::Cmp::Le: ok = p <= f.num; break;
          case QueryField::Cmp::Gt: ok = p > f.num; break;
          case QueryField::Cmp::Ge: ok = p >= f.num; break;
          case QueryField::Cmp::Eq: ok = p == f.num; break;
          case QueryField::Cmp::Match: ok = true; break;  // n/a for params
        }
        if (!ok) return false;
        break;
      }
    }
  }
  return true;
}

std::vector<SearchHit> SearchIndex::query(const std::string& q,
                                          const ir::Model& model,
                                          bool types_ready, size_t limit) const {
  ParsedQuery pq = parse_query(q);
  std::vector<SearchHit> hits;
  if (!pq.valid || limit == 0) return hits;
  // A bare fuzzy query: delegate to the legacy path for byte-identical behavior.
  if (!pq.is_field_query) return query(q, limit);

  hits.reserve(entries_.size() < 256 ? entries_.size() : 256);
  for (uint32_t i = 0; i < entries_.size(); ++i) {
    int name_score = 0;
    if (matches(entries_[i], pq, model, types_ready, name_score)) {
      // Field-only hits (no name predicate contributed) get a flat score so they
      // sort deterministically by entry index; name/op predicates rank by fuzzy.
      hits.push_back(SearchHit{i, name_score > 0 ? name_score : 500});
    }
  }

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
