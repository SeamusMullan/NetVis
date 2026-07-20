// tests/test_cost.cpp — compute_cost contract (v0.3.0 analyzer mode).
//
// DECISION: compute_cost is a pure structural function that NEVER reads tensor
// payloads. It computes FLOPs, params, weight_bytes, activation_bytes, and peak
// liveness from ir::ValueInfo shapes/dtypes and TensorRef metadata (elem_count,
// byte_len, dtype). These tests build minimal graphs by hand, compute the
// expected values manually (formulas in comments), and assert compute_cost
// matches. The critical zero-payload-read invariant is asserted in every test.
//
// Covers: MatMul, Conv, elementwise, unsupported ops, peak liveness, mixed
// dtypes (including Q4/Q8 with byte_len-fallback), table mode, hostile input.

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/ByteReader.h"
#include "engine/CostModel.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// Helper: append a value with the given name/dtype/shape; returns its index.
uint32_t add_value(ir::Model& m, ir::Graph& g, const std::string& name,
                   ir::DType dt, const std::vector<int64_t>& shape) {
  ir::ValueInfo v;
  v.name = m.intern(name);
  v.dtype = dt;
  for (int64_t d : shape) v.shape.push_back(d);
  v.producer = -1;
  uint32_t idx = static_cast<uint32_t>(g.values.size());
  g.values.push_back(std::move(v));
  return idx;
}

// Helper: wire a node with the given input/output value indices. Sets producer
// of each output to this node index.
uint32_t add_node(ir::Model& m, ir::Graph& g, const std::string& op,
                  const std::vector<uint32_t>& ins,
                  const std::vector<uint32_t>& outs) {
  ir::Node n;
  n.op_type = m.intern(op);
  n.name = m.intern(op + "0");
  uint32_t node_idx = static_cast<uint32_t>(g.nodes.size());
  n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  for (uint32_t vi : ins) g.edge_refs.push_back(vi);
  n.inputs.count = static_cast<uint32_t>(ins.size());
  n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  for (uint32_t vi : outs) {
    g.edge_refs.push_back(vi);
    g.values[vi].producer = static_cast<int32_t>(node_idx);
  }
  n.outputs.count = static_cast<uint32_t>(outs.size());
  g.nodes.push_back(std::move(n));
  return node_idx;
}

// Helper: add an attribute to the last node.
void add_attr_int(ir::Model& m, ir::Graph& g, const std::string& name,
                  int64_t val) {
  ir::Attribute a;
  a.name = m.intern(name);
  a.value.kind = ir::AttrValue::Kind::Int;
  a.value.i = val;
  uint32_t idx = static_cast<uint32_t>(g.attributes.size());
  g.attributes.push_back(std::move(a));
  ir::Node& node = g.nodes.back();
  if (node.attributes.count == 0) {
    node.attributes.begin = idx;
  }
  node.attributes.count++;
}

void add_attr_ints(ir::Model& m, ir::Graph& g, const std::string& name,
                   const std::vector<int64_t>& vals) {
  ir::Attribute a;
  a.name = m.intern(name);
  a.value.kind = ir::AttrValue::Kind::Ints;
  a.value.ints = vals;
  uint32_t idx = static_cast<uint32_t>(g.attributes.size());
  g.attributes.push_back(std::move(a));
  ir::Node& node = g.nodes.back();
  if (node.attributes.count == 0) {
    node.attributes.begin = idx;
  }
  node.attributes.count++;
}

// Helper: add an initializer (weight) TensorRef to the graph.
void add_initializer(ir::Model& m, ir::Graph& g, const std::string& name,
                     ir::DType dtype, const std::vector<int64_t>& shape,
                     uint64_t byte_len = 0) {
  ir::TensorRef tr;
  tr.name = m.intern(name);
  tr.dtype = dtype;
  for (int64_t d : shape) tr.shape.push_back(d);
  tr.file_offset = 0;  // arbitrary; not dereferenced
  tr.byte_len = byte_len;
  g.initializers.push_back(std::move(tr));
}

// Helper: product of shape dimensions (>=1); 0 if any dim < 1.
int64_t shape_prod(const std::vector<int64_t>& shape) {
  int64_t prod = 1;
  for (int64_t d : shape) {
    if (d < 1) return 0;
    prod *= d;
  }
  return prod;
}

}  // namespace

TEST_CASE("Cost: MatMul [M,K]x[K,N]->[M,N] flops = 2*M*N*K") {
  // MatMul: macs = |O| * K, where K is last dim of input[0].
  // |O| = M*N, so macs = M*N*K, flops = 2*M*N*K.
  // Test: M=8, K=16, N=32 -> flops = 2*8*32*16 = 8192.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {8, 16});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {16, 32});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {8, 32});
  add_node(m, g, "MatMul", {a, b}, {out});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 2 * 8 * 32 * 16);  // 8192
  CHECK(r.total_flops == 2 * 8 * 32 * 16);
  CHECK(r.nodes_flops_known == 1);
  CHECK(r.nodes_total == 1);
  // Output activation bytes: 8*32 elems * 4 bytes/elem = 1024.
  CHECK(r.per_node[0].act_bytes == 8 * 32 * 4);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Conv with weight initializer, known output shape") {
  // Conv: macs = |O| * (Cin/g) * kh*kw. Weight shape [Cout, Cin/g, kh, kw].
  // Test: out [1,64,28,28], weight [64,32,3,3], group=1.
  // |O| = 1*64*28*28 = 50176, kernel = 3*3 = 9, Cin/g = 32.
  // macs = 50176 * 32 * 9 = 14450688, flops = 2*macs = 28901376.
  // params = 64*32*3*3 = 18432, weight_bytes = 18432*4 = 73728 (fp32).
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t inp = add_value(m, g, "inp", ir::DType::F32, {1, 32, 30, 30});
  uint32_t w = add_value(m, g, "w", ir::DType::F32, {64, 32, 3, 3});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {1, 64, 28, 28});
  add_initializer(m, g, "w", ir::DType::F32, {64, 32, 3, 3});
  add_node(m, g, "Conv", {inp, w}, {out});
  add_attr_int(m, g, "group", 1);

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  int64_t O = 1 * 64 * 28 * 28;       // 50176
  int64_t macs = O * 32 * 9;          // 14450688
  CHECK(r.per_node[0].flops == 2 * macs);  // 28901376
  CHECK(r.per_node[0].params == 64 * 32 * 3 * 3);  // 18432
  CHECK(r.per_node[0].weight_bytes == 18432 * 4);  // 73728
  CHECK(r.total_params == 18432);
  CHECK(r.total_weight_bytes == 73728);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Conv with byte_len set (quantized fallback path)") {
  // Same Conv, but dtype Q4 (dtype_size == 0). byte_len must be used.
  // byte_len = 36864 (half of fp32 for test). params still 18432.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t inp = add_value(m, g, "inp", ir::DType::F32, {1, 32, 30, 30});
  uint32_t w = add_value(m, g, "w", ir::DType::Q4, {64, 32, 3, 3});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {1, 64, 28, 28});
  add_initializer(m, g, "w", ir::DType::Q4, {64, 32, 3, 3}, 36864);
  add_node(m, g, "Conv", {inp, w}, {out});
  add_attr_int(m, g, "group", 1);

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.per_node[0].params == 64 * 32 * 3 * 3);
  CHECK(r.per_node[0].weight_bytes == 36864);
  CHECK(r.total_weight_bytes == 36864);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Elementwise Add [n]->[n] flops = n") {
  // Elementwise: flops = |O|. Test: [10] -> flops = 10.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {10});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {10});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {10});
  add_node(m, g, "Add", {a, b}, {out});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 10);
  CHECK(r.total_flops == 10);
  CHECK(r.nodes_flops_known == 1);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Unsupported op (Reshape) flops_known=false") {
  // Reshape is structural (Shape category), no arithmetic -> flops_known=false.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t data = add_value(m, g, "data", ir::DType::F32, {4, 6});
  uint32_t shape = add_value(m, g, "shape", ir::DType::I64, {2});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {2, 12});
  add_node(m, g, "Reshape", {data, shape}, {out});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK_FALSE(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 0);
  CHECK(r.total_flops == 0);
  CHECK(r.nodes_flops_known == 0);
  CHECK(r.nodes_total == 1);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Peak activation liveness 2-node chain") {
  // Graph: a[100] -> Relu -> b[100] -> Sqrt -> c[100] (graph output).
  // b is freed after Sqrt consumes it; c is retained (graph output).
  // Liveness: initially 100 (a), +100 after Relu (peak=200), -100 (a freed),
  // +100 after Sqrt (c, peak stays 200 since we're at 100 before adding c).
  // Actually: start with a=100. After Relu: +b=100, live=200, peak=200.
  // a is consumed by Relu, but freed after Relu finishes. Then Sqrt runs:
  // b consumed, +c=100. But b is freed after c is produced if c is last use of b.
  // Proper topological accounting (ONNX order):
  //   Node 0 (Relu): live starts at 100 (a). Produce b=100 -> live=200, peak=200.
  //   After node 0, a's last use is 0 -> free a -> live=100.
  //   Node 1 (Sqrt): produce c=100 -> live=200, peak=200.
  //   After node 1, b's last use is 1 -> free b (but c is graph_output, not freed).
  //   Final live=100 (c). Peak=200.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {100});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {100});
  uint32_t c = add_value(m, g, "c", ir::DType::F32, {100});
  g.graph_inputs.push_back(a);
  g.graph_outputs.push_back(c);
  add_node(m, g, "Relu", {a}, {b});
  add_node(m, g, "Sqrt", {b}, {c});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  // Each value is 100 elems * 4 bytes = 400.
  CHECK(r.peak_activation_bytes == 2 * 400);  // 800
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Peak activation with multiple consumers") {
  // Graph: a[100] -> Relu -> b[100] used by both Add and Mul -> ...
  // b has last_use = max(Add node idx, Mul node idx). Check b not freed early.
  // Graph: a -> Relu -> b; c[100] input. b+c -> d[100], b*c -> e[100].
  // d and e are graph outputs. Both b and c are consumed by Add AND Mul, so
  // last_use(b) = last_use(c) = 2 (Mul node) -> neither freed until node 2.
  // Each value = 100 elems * 4 bytes = 400.
  // Liveness (bytes): start = a(400) + c(400) = 800.
  // Node 0 (Relu): +b=400 -> live=1200, peak=1200. a freed (last_use=0) -> 800.
  // Node 1 (Add):  +d=400 -> live=1200, peak=1200. nothing freed (b,c live to 2).
  // Node 2 (Mul):  +e=400 -> live=1600, peak=1600. b,c freed (last_use=2) -> 800.
  // d,e are outputs, not freed. Final live=800. Peak=1600 = 4*400.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {100});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {100});
  uint32_t c = add_value(m, g, "c", ir::DType::F32, {100});
  uint32_t d = add_value(m, g, "d", ir::DType::F32, {100});
  uint32_t e = add_value(m, g, "e", ir::DType::F32, {100});
  g.graph_inputs.push_back(a);
  g.graph_inputs.push_back(c);
  g.graph_outputs.push_back(d);
  g.graph_outputs.push_back(e);
  add_node(m, g, "Relu", {a}, {b});    // node 0
  add_node(m, g, "Add", {b, c}, {d});  // node 1
  add_node(m, g, "Mul", {b, c}, {e});  // node 2

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  // Each value 400 bytes. Peak at node 2: a-freed(800) + d + e... = 4*400 = 1600.
  CHECK(r.peak_activation_bytes == 4 * 400);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: dtype_usage mixed dtypes sorted by bytes desc") {
  // 3 initializers: fp32 [10] (40 bytes), fp16 [20] (40 bytes), Q4 [100] with
  // byte_len=50. Sort by bytes desc, ties by dtype enum order.
  // fp32=40, fp16=40, Q4=50 -> Q4 (50), fp32 (40), fp16 (40).
  // But ties: fp32 < fp16 in enum -> fp32, fp16.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  add_initializer(m, g, "w1", ir::DType::F32, {10});
  add_initializer(m, g, "w2", ir::DType::F16, {20});
  add_initializer(m, g, "w3", ir::DType::Q4, {100}, 50);

  // Add a dummy node so from_graph=true.
  uint32_t a = add_value(m, g, "a", ir::DType::F32, {1});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {1});
  add_node(m, g, "Add", {a}, {b});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.total_params == 10 + 20 + 100);  // 130
  CHECK(r.total_weight_bytes == 40 + 40 + 50);  // 130
  REQUIRE(r.dtype_usage.size() == 3);
  // Sorted by bytes desc (50, 40, 40), ties by dtype enum order (F32 < F16).
  CHECK(r.dtype_usage[0].dtype == ir::DType::Q4);
  CHECK(r.dtype_usage[0].bytes == 50);
  CHECK(r.dtype_usage[0].params == 100);
  CHECK(r.dtype_usage[1].dtype == ir::DType::F32);
  CHECK(r.dtype_usage[1].bytes == 40);
  CHECK(r.dtype_usage[1].params == 10);
  CHECK(r.dtype_usage[2].dtype == ir::DType::F16);
  CHECK(r.dtype_usage[2].bytes == 40);
  CHECK(r.dtype_usage[2].params == 20);

  // Check derived metrics.
  double bits_per_param = r.effective_bits_per_param();
  CHECK(bits_per_param == doctest::Approx(130.0 * 8.0 / 130.0));  // 8.0
  double vs_fp32 = r.size_vs_fp32();
  CHECK(vs_fp32 == doctest::Approx(130.0 / (130.0 * 4.0)));  // 0.25
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Table mode (has_graph=false) builds from flat_tensors") {
  // Model with has_graph=false and flat_tensors populated.
  ir::Model m;
  m.format_name = m.intern("GGUF");
  m.has_graph = false;

  ir::TensorRef t1;
  t1.name = m.intern("t1");
  t1.dtype = ir::DType::F32;
  t1.shape.push_back(100);
  t1.byte_len = 400;
  m.flat_tensors.push_back(std::move(t1));

  ir::TensorRef t2;
  t2.name = m.intern("t2");
  t2.dtype = ir::DType::Q8;
  t2.shape.push_back(200);
  t2.byte_len = 200;
  m.flat_tensors.push_back(std::move(t2));

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK_FALSE(r.from_graph);
  CHECK(r.per_node.empty());
  CHECK(r.nodes_total == 0);
  CHECK(r.total_params == 100 + 200);  // 300
  CHECK(r.total_weight_bytes == 400 + 200);  // 600
  REQUIRE(r.dtype_usage.size() == 2);
  // Sorted by bytes desc: F32 (400), Q8 (200).
  CHECK(r.dtype_usage[0].dtype == ir::DType::F32);
  CHECK(r.dtype_usage[0].bytes == 400);
  CHECK(r.dtype_usage[1].dtype == ir::DType::Q8);
  CHECK(r.dtype_usage[1].bytes == 200);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Table mode (out-of-range graph_index) falls back") {
  // Model with has_graph=true but graph_index out of range -> table mode.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.has_graph = true;
  m.graphs.emplace_back();  // 1 graph, index 0
  ir::TensorRef t;
  t.name = m.intern("t");
  t.dtype = ir::DType::F16;
  t.shape.push_back(50);
  t.byte_len = 100;
  m.flat_tensors.push_back(std::move(t));

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 999);  // out of range

  CHECK_FALSE(r.from_graph);
  CHECK(r.per_node.empty());
  CHECK(r.total_params == 50);
  CHECK(r.total_weight_bytes == 100);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Hostile input — edge_ref out of range, no crash") {
  // A node input edge_ref that points beyond g.values.size(). compute_cost
  // must bounds-check and treat the node as flops_known=false or skip it.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {10});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {10});
  // Manually construct a node with a bogus edge_ref.
  ir::Node n;
  n.op_type = m.intern("Add");
  n.name = m.intern("BadAdd");
  n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(a);
  g.edge_refs.push_back(9999);  // out of range
  n.inputs.count = 2;
  n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(b);
  n.outputs.count = 1;
  g.values[b].producer = 0;
  g.nodes.push_back(std::move(n));

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);  // must not crash

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  // The node should have flops_known=false due to the bogus input.
  CHECK_FALSE(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 0);
  CHECK(r.nodes_flops_known == 0);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Dynamic dim in shape makes flops_known=false") {
  // MatMul with a dynamic dim (-1) in the output shape -> flops_known=false.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {8, 16});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {16, 32});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {-1, 32});  // dynamic
  add_node(m, g, "MatMul", {a, b}, {out});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK_FALSE(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 0);
  CHECK(r.nodes_flops_known == 0);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Pool with kernel_shape attribute") {
  // Pool: flops = |O| * prod(kernel_shape). Test: MaxPool, out [1,64,14,14],
  // kernel [2,2] -> flops = 1*64*14*14*2*2 = 50176.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t inp = add_value(m, g, "inp", ir::DType::F32, {1, 64, 28, 28});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {1, 64, 14, 14});
  add_node(m, g, "MaxPool", {inp}, {out});
  add_attr_ints(m, g, "kernel_shape", {2, 2});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  int64_t O = 1 * 64 * 14 * 14;  // 12544
  CHECK(r.per_node[0].flops == O * 2 * 2);  // 50176
  CHECK(r.total_flops == O * 2 * 2);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Norm (LayerNorm) flops = |O|") {
  // Norm ops (LayerNorm, BatchNorm, etc.) count as flops = |O|.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t inp = add_value(m, g, "inp", ir::DType::F32, {8, 512});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {8, 512});
  add_node(m, g, "LayerNorm", {inp}, {out});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 8 * 512);  // 4096
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Gemm 2D [M,K]x[K,N]->[M,N] flops = 2*M*N*K") {
  // Gemm: macs = M*N*K, flops = 2*M*N*K. Test: M=4, K=8, N=16.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {4, 8});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {8, 16});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {4, 16});
  add_node(m, g, "Gemm", {a, b}, {out});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 2 * 4 * 16 * 8);  // 1024
  CHECK(r.total_flops == 2 * 4 * 16 * 8);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Reduce (ReduceSum) flops = |input|") {
  // Reduce: flops = |input[0]| (elements reduced). Test: [4,8] -> [] (scalar).
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t inp = add_value(m, g, "inp", ir::DType::F32, {4, 8});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {});
  add_node(m, g, "ReduceSum", {inp}, {out});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 4 * 8);  // 32
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Zero-param derived metrics when total_params=0") {
  // Empty model: effective_bits_per_param and size_vs_fp32 should return 0.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.has_graph = false;

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.total_params == 0);
  CHECK(r.effective_bits_per_param() == 0.0);
  CHECK(r.size_vs_fp32() == 0.0);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Conv with group > 1 (depthwise)") {
  // Conv with group=2: Cin/g = 32/2 = 16. macs = |O| * 16 * kh*kw.
  // Test: out [1,32,14,14], weight [32,16,3,3], group=2.
  // |O| = 1*32*14*14 = 6272, kernel=9, Cin/g=16.
  // macs = 6272 * 16 * 9 = 903168, flops = 2*macs = 1806336.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t inp = add_value(m, g, "inp", ir::DType::F32, {1, 32, 16, 16});
  uint32_t w = add_value(m, g, "w", ir::DType::F32, {32, 16, 3, 3});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {1, 32, 14, 14});
  add_initializer(m, g, "w", ir::DType::F32, {32, 16, 3, 3});
  add_node(m, g, "Conv", {inp, w}, {out});
  add_attr_int(m, g, "group", 2);

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  int64_t O = 1 * 32 * 14 * 14;  // 6272
  int64_t macs = O * 16 * 9;     // 903168
  CHECK(r.per_node[0].flops == 2 * macs);  // 1806336
  CHECK(r.per_node[0].params == 32 * 16 * 3 * 3);  // 4608
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Activation (Softmax) flops = |O|") {
  // Activation ops count as flops = |O|. Test: Softmax [10,20] -> flops = 200.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t inp = add_value(m, g, "inp", ir::DType::F32, {10, 20});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {10, 20});
  add_node(m, g, "Softmax", {inp}, {out});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  CHECK(r.from_graph);
  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 10 * 20);  // 200
  CHECK(ByteReader::payload_read_counter() == 0);
}

// --- v0.3.0 sweep regressions (bugs found by the Opus adversarial sweep) -------

TEST_CASE("Cost: Gemm transA=1 uses the correct contraction dim K") {
  // Regression: the Gemm branch took K = in0->shape.back() unconditionally.
  // With transA=1 operand A is stored [K, M] (shape inference does NOT rewrite
  // the input operand shape), so back() is M, giving M*N*M instead of M*N*K.
  // A = [K,M] = [16,8], B = [K,N] = [16,32], out = [M,N] = [8,32].
  // macs = |O|*K = (8*32)*16 = 4096; flops = 8192. (Not (8*32)*8 = 2048*... .)
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {16, 8});   // [K, M]
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {16, 32});  // [K, N]
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {8, 32});
  add_node(m, g, "Gemm", {a, b}, {out});
  add_attr_int(m, g, "transA", 1);

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 2 * (8 * 32) * 16);  // 8192, K=16 not M=8
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: ConvTranspose uses transposed weight layout + input base") {
  // Regression: ConvTranspose went through the Conv formula (base=|O|, chan from
  // weight.shape[1] read as Cin/g). For ConvTranspose weight is [Cin, Cout/g,
  // k...] and MACs are over the INPUT: macs = |input| * (Cout/g) * prod(kernel).
  // input [1,32,14,14], weight [32,64,3,3] (Cin=32, Cout/g=64), out [1,64,28,28].
  // |input| = 1*32*14*14 = 6272; macs = 6272 * 64 * 9 = 3612672; flops = 7225344.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t inp = add_value(m, g, "inp", ir::DType::F32, {1, 32, 14, 14});
  uint32_t w = add_value(m, g, "w", ir::DType::F32, {32, 64, 3, 3});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {1, 64, 28, 28});
  add_initializer(m, g, "w", ir::DType::F32, {32, 64, 3, 3});
  add_node(m, g, "ConvTranspose", {inp, w}, {out});
  add_attr_int(m, g, "group", 1);

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  REQUIRE(r.per_node.size() == 1);
  CHECK(r.per_node[0].flops_known);
  int64_t in_elems = 1 * 32 * 14 * 14;   // 6272
  int64_t macs = in_elems * 64 * 9;      // 3612672 (Cout/g=64 = weight.shape[1])
  CHECK(r.per_node[0].flops == 2 * macs);  // 7225344
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: Einsum is NOT given fabricated MatMul FLOPs") {
  // Regression: Einsum shares the MatMul category and was routed through the
  // |O|*K formula, fabricating a number. Its FLOPs depend on the equation, so it
  // must stay flops_known=false (honest unknown).
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {8, 16});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {16, 32});
  uint32_t out = add_value(m, g, "out", ir::DType::F32, {8, 32});
  add_node(m, g, "Einsum", {a, b}, {out});

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  REQUIRE(r.per_node.size() == 1);
  CHECK_FALSE(r.per_node[0].flops_known);  // not fabricated
  CHECK(r.per_node[0].flops == 0);
  CHECK(r.total_flops == 0);
  CHECK(r.nodes_flops_known == 0);
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: peak activation counts a value once per production, not per slot") {
  // Regression: the add-side of the liveness pass added output bytes per output
  // SLOT. A malformed node whose two output slots map to the SAME value index
  // then added its bytes twice but freed once, inflating the peak. Build exactly
  // that (an edge_ref list with a duplicated output vidx) and assert the value is
  // counted once. Single node "Op": input a[100], output b[100] listed TWICE.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {100});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {100});
  g.graph_inputs.push_back(a);
  g.graph_outputs.push_back(b);

  ir::Node n;
  n.op_type = m.intern("Op");
  n.name = m.intern("dup");
  n.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(a);
  n.inputs.count = 1;
  n.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  g.edge_refs.push_back(b);
  g.edge_refs.push_back(b);  // duplicate output slot -> same vidx
  n.outputs.count = 2;
  g.values[b].producer = 0;
  g.nodes.push_back(std::move(n));

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);  // must not crash

  // Peak = a(400) live, then +b once (400) => 800, NOT 1200 (b double-added).
  CHECK(r.peak_activation_bytes == 2 * 400);  // 800
  CHECK(ByteReader::payload_read_counter() == 0);
}

TEST_CASE("Cost: unknown op leaves flops UNTOUCHED (sentinel guard)") {
  // The unknown-op test elsewhere checks flops==0, but 0 is also the default, so
  // it can't distinguish "computed unknown" from "never written". Pre-seed a
  // sentinel is impossible (per_node is built internally), so instead assert the
  // stronger observable: an unknown op does NOT increment nodes_flops_known and
  // contributes nothing to total_flops even when a KNOWN op is present too.
  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {10, 20});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {10, 20});
  uint32_t c = add_value(m, g, "c", ir::DType::F32, {10, 20});
  add_node(m, g, "Relu", {a}, {b});      // known: flops = 200
  add_node(m, g, "NonExistentOp", {b}, {c});  // unknown

  ByteReader::payload_read_counter() = 0;
  CostReport r = compute_cost(m, 0);

  REQUIRE(r.per_node.size() == 2);
  CHECK(r.per_node[0].flops_known);
  CHECK(r.per_node[0].flops == 200);
  CHECK_FALSE(r.per_node[1].flops_known);
  CHECK(r.nodes_total == 2);
  CHECK(r.nodes_flops_known == 1);   // only Relu
  CHECK(r.total_flops == 200);       // unknown contributes nothing
  CHECK(ByteReader::payload_read_counter() == 0);
}
