// tests/test_modeldiff.cpp — structural diff between two models (v0.2.0).
//
// Builds model A and a model B derived from A: same nodes, but one node removed,
// one node added, and one node's op_type changed. Asserts the counts and the
// per-node Same/Added/Removed/Changed classifications, that matching works by
// string CONTENT across INDEPENDENT StringArenas (B interns its own strings),
// and that diff_models is deterministic + OOB-safe.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "engine/ModelDiff.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// Build a linear chain of named nodes with the given (op, name) pairs. Each node
// i consumes node i-1's output value and produces its own. This gives a
// deterministic topology so fingerprint+position fallback is exercised too.
ir::Model make_chain(const std::vector<std::pair<std::string, std::string>>& ops) {
  ir::Model m;
  m.has_graph = true;
  m.format_name = m.intern("TEST");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  for (size_t i = 0; i < ops.size(); ++i) {
    ir::ValueInfo v;
    v.name = m.intern("val" + std::to_string(i));
    v.producer = static_cast<int32_t>(i);
    g.values.push_back(v);
  }
  for (size_t i = 0; i < ops.size(); ++i) {
    ir::Node n;
    n.op_type = m.intern(ops[i].first);
    n.name = m.intern(ops[i].second);
    n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    if (i > 0) {
      g.edge_refs.push_back(static_cast<uint32_t>(i - 1));  // prev output value
      n.inputs.count = 1;
    } else {
      n.inputs.count = 0;
    }
    n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    g.edge_refs.push_back(static_cast<uint32_t>(i));
    n.outputs.count = 1;
    g.nodes.push_back(n);
  }
  return m;
}

}  // namespace

TEST_CASE("diff: add / remove / change classified by name-content matching") {
  // A: relu -> conv -> gemm -> softmax
  ir::Model a = make_chain({{"Relu", "relu"},
                            {"Conv", "conv"},
                            {"Gemm", "gemm"},
                            {"Softmax", "softmax"}});
  // B: (drop "conv"), keep "relu", "gemm", add "sigmoid", and CHANGE "softmax"
  //    op_type Softmax -> LogSoftmax (same name -> matched-but-changed).
  ir::Model b = make_chain({{"Relu", "relu"},
                            {"Gemm", "gemm"},
                            {"LogSoftmax", "softmax"},
                            {"Sigmoid", "sigmoid"}});

  ModelDiffResult d = diff_models(a, 0, b, 0);
  REQUIRE(d.valid);
  REQUIRE(d.a_status.size() == 4);
  REQUIRE(d.b_status.size() == 4);

  // A nodes: relu(0)=Same, conv(1)=Removed, gemm(2)=Same, softmax(3)=Changed.
  CHECK(d.a_status[0] == DiffStatus::Same);      // relu
  CHECK(d.a_status[1] == DiffStatus::Removed);   // conv (not in B)
  CHECK(d.a_status[2] == DiffStatus::Same);      // gemm
  CHECK(d.a_status[3] == DiffStatus::Changed);   // softmax -> logsoftmax

  // B nodes: relu(0)=Same, gemm(1)=Same, softmax(2)=Changed, sigmoid(3)=Added.
  CHECK(d.b_status[0] == DiffStatus::Same);      // relu
  CHECK(d.b_status[1] == DiffStatus::Same);      // gemm
  CHECK(d.b_status[2] == DiffStatus::Changed);   // logsoftmax
  CHECK(d.b_status[3] == DiffStatus::Added);     // sigmoid

  CHECK(d.same == 2);
  CHECK(d.removed == 1);
  CHECK(d.changed == 1);
  CHECK(d.added == 1);

  // Cross-references are consistent.
  CHECK(d.a_to_b[0] == 0);   // relu -> relu
  CHECK(d.a_to_b[1] == -1);  // conv unmatched
  CHECK(d.a_to_b[2] == 1);   // gemm -> gemm
  CHECK(d.a_to_b[3] == 2);   // softmax -> logsoftmax
  CHECK(d.b_to_a[3] == -1);  // sigmoid unmatched
  CHECK(d.b_to_a[0] == 0);
  CHECK(d.b_to_a[2] == 3);
}

TEST_CASE("diff: matching is by string content across independent arenas") {
  // Two identical graphs built from SEPARATE Models (separate StringArenas), so
  // raw StringIds would collide/be meaningless — matching must use str content.
  ir::Model a = make_chain({{"Add", "a"}, {"Mul", "b"}});
  ir::Model b = make_chain({{"Add", "a"}, {"Mul", "b"}});
  ModelDiffResult d = diff_models(a, 0, b, 0);
  REQUIRE(d.valid);
  CHECK(d.same == 2);
  CHECK(d.added == 0);
  CHECK(d.removed == 0);
  CHECK(d.changed == 0);
  CHECK(d.a_status[0] == DiffStatus::Same);
  CHECK(d.a_status[1] == DiffStatus::Same);
}

TEST_CASE("diff: unnamed nodes match by fingerprint + topological position") {
  // No node names -> the name pass matches nothing; the fingerprint/position
  // fallback must still classify a same-structure chain as all Same.
  ir::Model a = make_chain({{"Relu", ""}, {"Conv", ""}, {"Gemm", ""}});
  ir::Model b = make_chain({{"Relu", ""}, {"Conv", ""}, {"Gemm", ""}});
  ModelDiffResult d = diff_models(a, 0, b, 0);
  REQUIRE(d.valid);
  CHECK(d.same == 3);
  CHECK(d.added == 0);
  CHECK(d.removed == 0);
  CHECK(d.changed == 0);

  // Now B drops the middle op -> one Removed in A, rest Same.
  ir::Model b2 = make_chain({{"Relu", ""}, {"Gemm", ""}});
  ModelDiffResult d2 = diff_models(a, 0, b2, 0);
  REQUIRE(d2.valid);
  CHECK(d2.same == 2);
  CHECK(d2.removed == 1);
  CHECK(d2.added == 0);
  CHECK(d2.changed == 0);
}

TEST_CASE("diff: deterministic and OOB-safe") {
  ir::Model a = make_chain({{"Relu", "r"}, {"Conv", "c"}});
  ir::Model b = make_chain({{"Relu", "r"}, {"Gemm", "g"}});

  ModelDiffResult d1 = diff_models(a, 0, b, 0);
  ModelDiffResult d2 = diff_models(a, 0, b, 0);
  CHECK(d1.a_status == d2.a_status);
  CHECK(d1.b_status == d2.b_status);
  CHECK(d1.a_to_b == d2.a_to_b);
  CHECK(d1.b_to_a == d2.b_to_a);

  // Out-of-range graph index => valid == false.
  CHECK_FALSE(diff_models(a, 5, b, 0).valid);
  CHECK_FALSE(diff_models(a, 0, b, 5).valid);
}
