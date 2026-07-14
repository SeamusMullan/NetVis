// tests/test_search.cpp — fuzzy search ranking + index (spec §7.4).
//
// Asserts the fuzzy_score ordering contract directly (prefix > substring >
// subsequence > no-match) and that SearchIndex::build + query return hits for a
// small in-code model. Score is the frozen ranking primitive the UI relies on.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "engine/SearchIndex.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// Small model: three nodes with distinct names/op_types so the index has
// several searchable entries of different kinds.
ir::Model make_named_model() {
  ir::Model m;
  m.has_graph = true;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  g.name = m.intern("g");

  auto add = [&](const char* op, const char* name) {
    ir::Node n;
    n.op_type = m.intern(op);
    n.name = m.intern(name);
    g.nodes.push_back(n);
    ir::ValueInfo v;
    v.name = m.intern(std::string(name) + "_out");
    g.values.push_back(v);
  };
  add("Conv", "conv_layer_0");
  add("Relu", "relu_activation");
  add("MatMul", "final_matmul");
  return m;
}

}  // namespace

TEST_CASE("fuzzy_score: prefix beats substring beats subsequence") {
  // All inputs are already-lowercased, per the contract.
  int prefix = fuzzy_score("conv", "conv_layer");        // starts-with
  int substring = fuzzy_score("layer", "conv_layer");    // contiguous, not at 0
  int subseq = fuzzy_score("cvl", "conv_layer");         // scattered subsequence
  int nomatch = fuzzy_score("xyz", "conv_layer");        // no match

  CHECK(prefix > 0);
  CHECK(substring > 0);
  CHECK(subseq > 0);
  CHECK(nomatch == -1);

  // The frozen ranking order (spec §7.4): prefix > substring > subsequence.
  CHECK(prefix > substring);
  CHECK(substring > subseq);
}

TEST_CASE("fuzzy_score: empty query and exact-length behaviors") {
  // An exact full match should score at least as high as a prefix of a longer
  // string (both are prefix matches; exact is never worse).
  int exact = fuzzy_score("relu", "relu");
  int longer_prefix = fuzzy_score("relu", "relu_activation");
  CHECK(exact > 0);
  CHECK(longer_prefix > 0);
  CHECK(exact >= longer_prefix);
}

TEST_CASE("SearchIndex builds and returns ranked hits") {
  ir::Model model = make_named_model();
  SearchIndex index;
  index.build(model);

  CHECK(index.size() > 0);

  auto hits = index.query("conv", 50);
  REQUIRE_MESSAGE(!hits.empty(), "expected at least one hit for 'conv'");

  // Hits must be sorted by descending score (best first).
  for (size_t i = 1; i < hits.size(); ++i) {
    CHECK(hits[i - 1].score >= hits[i].score);
  }

  // The top hit's entry index must be in range and its text must contain the
  // query (case-insensitively) — a sanity check on the entry mapping.
  const auto& entries = index.entries();
  REQUIRE(hits[0].entry < entries.size());
  const std::string& lower = entries[hits[0].entry].lower;
  CHECK(lower.find("conv") != std::string::npos);

  // Empty query returns no hits quickly (contract).
  CHECK(index.query("", 50).empty());
}
