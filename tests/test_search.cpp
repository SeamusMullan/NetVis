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

// --- #52/#54 structured (field-scoped) query --------------------------------

namespace {
// A model whose values carry dtypes + shapes so dtype:/shape:/params: predicates
// have something to resolve against (shape inference would fill these in the real
// pipeline; we set them directly since build() only reads names).
ir::Model make_typed_model() {
  ir::Model m;
  m.has_graph = true;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];
  auto add = [&](const char* op, const char* name, ir::DType dt,
                 std::vector<int64_t> shape) {
    ir::Node n;
    n.op_type = m.intern(op);
    n.name = m.intern(name);
    g.nodes.push_back(n);
    ir::ValueInfo v;
    v.name = m.intern(std::string(name) + "_out");
    v.dtype = dt;
    for (int64_t d : shape) v.shape.push_back(d);
    g.values.push_back(v);
  };
  add("Conv", "layer0_conv", ir::DType::F16, {1, 64, 224, 224});   // big f16
  add("Conv", "layer1_conv", ir::DType::F32, {1, 3, 8, 8});         // small f32
  add("Relu", "layer0_relu", ir::DType::F16, {1, 64, 224, 224});
  add("MatMul", "head_matmul", ir::DType::F32, {1, 1000});
  return m;
}
}  // namespace

TEST_CASE("parse_query: field detection and comparators") {
  CHECK_FALSE(parse_query("conv").is_field_query);           // bare fuzzy
  CHECK(parse_query("op:Conv").is_field_query);
  CHECK(parse_query("op:Conv name:layer0").fields.size() == 2);

  ParsedQuery p = parse_query("params:>1M");
  REQUIRE(p.valid);
  REQUIRE(p.fields.size() == 1);
  CHECK(p.fields[0].key == QueryField::Key::Params);
  CHECK(p.fields[0].cmp == QueryField::Cmp::Gt);
  CHECK(p.fields[0].num == 1000000);

  CHECK(parse_query("params:>=1000").fields[0].cmp == QueryField::Cmp::Ge);
  CHECK(parse_query("params:<8k").fields[0].num == 8000);
  CHECK(parse_query("dtype:fp16").fields[0].key == QueryField::Key::Dtype);

  // Malformed values.
  CHECK_FALSE(parse_query("params:abc").valid);
  CHECK_FALSE(parse_query("op:").valid);
}

TEST_CASE("field query: op / dtype / params / combined") {
  ir::Model m = make_typed_model();
  SearchIndex index;
  index.build(m);

  auto op_names = [&](const std::vector<SearchHit>& hits) {
    std::vector<std::string> out;
    for (const auto& h : hits) out.push_back(index.entries()[h.entry].lower);
    return out;
  };

  // op:Conv -> only the two Conv nodes (+ the OpType entry), no Relu/MatMul.
  auto conv = index.query("op:conv", m, true, 50);
  REQUIRE(!conv.empty());
  for (const auto& h : conv) {
    const SearchEntry& e = index.entries()[h.entry];
    // Every hit must be a node/optype whose op is Conv.
    REQUIRE(e.graph < m.graphs.size());
    if (e.kind == SearchKind::Node || e.kind == SearchKind::OpType)
      CHECK(std::string(m.str(m.graphs[0].nodes[e.ref].op_type)) == "Conv");
  }

  // dtype:f32 -> only the f32 values (layer1_conv_out, head_matmul_out).
  auto f32 = index.query("dtype:f32", m, true, 50);
  REQUIRE(!f32.empty());
  for (const auto& h : f32) {
    const SearchEntry& e = index.entries()[h.entry];
    if (e.kind == SearchKind::Value)
      CHECK(m.graphs[0].values[e.ref].dtype == ir::DType::F32);
  }

  // params:>1M -> only the big [1,64,224,224] values (~3.2M each).
  auto big = index.query("params:>1M", m, true, 50);
  REQUIRE(!big.empty());
  for (const auto& h : big) {
    const SearchEntry& e = index.entries()[h.entry];
    REQUIRE(e.kind == SearchKind::Value);
    CHECK(m.graphs[0].values[e.ref].shape.size() == 4);  // the 224x224 tensors
  }
  // The small f32 value must NOT appear.
  for (const std::string& nm : op_names(big))
    CHECK(nm.find("layer1") == std::string::npos);

  // Combined AND: op:Conv name:layer0 -> only layer0_conv.
  auto combined = index.query("op:conv name:layer0", m, true, 50);
  REQUIRE(!combined.empty());
  for (const auto& h : combined) {
    const SearchEntry& e = index.entries()[h.entry];
    if (e.kind == SearchKind::Node)
      CHECK(std::string(m.str(m.graphs[0].nodes[e.ref].name)).find("layer0") !=
            std::string::npos);
  }

  // A field query with no matches is empty; a malformed query is empty.
  CHECK(index.query("op:DoesNotExist", m, true, 50).empty());
  CHECK(index.query("params:xyz", m, true, 50).empty());
}

TEST_CASE("field query: glob + legacy-fuzzy parity") {
  ir::Model m = make_typed_model();
  SearchIndex index;
  index.build(m);

  // Glob on name: prefix / contains.
  CHECK(!index.query("name:layer0*", m, true, 50).empty());
  CHECK(!index.query("name:*conv*", m, true, 50).empty());

  // A bare (non-field) query through the model overload must match the fuzzy path.
  auto a = index.query("conv", m, true, 50);
  auto b = index.query("conv", 50);
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].entry == b[i].entry);
    CHECK(a[i].score == b[i].score);
  }
}

TEST_CASE("field query: types_ready gate (dtype/shape/params race guard)") {
  ir::Model m = make_typed_model();
  SearchIndex index;
  index.build(m);

  // Before shape inference publishes (types_ready=false), the type-dependent
  // predicates must match NOTHING (their fields are worker-mutated — unsafe to
  // read). op:/name: read only immutable parse-time data, so they still work.
  CHECK(index.query("dtype:f32", m, false, 50).empty());
  CHECK(index.query("shape:*224*", m, false, 50).empty());
  CHECK(index.query("params:>1M", m, false, 50).empty());
  CHECK(!index.query("op:conv", m, false, 50).empty());
  CHECK(!index.query("name:layer0*", m, false, 50).empty());

  // Once ready, the same type predicates return hits.
  CHECK(!index.query("dtype:f32", m, true, 50).empty());
  CHECK(!index.query("params:>1M", m, true, 50).empty());
}
