// tests/test_shape_ext.cpp — mmap-base-aware shape inference (spec §7.3).
//
// Builds ONNX-like ir::Models by hand and exercises infer_shapes_ext:
//   * a raw_data Reshape shape initializer resolves the output shape (the whole
//     point of the ext overload — reading a constant shape tensor via mmap);
//   * an int64_data-stored shape (file_offset==UINT64_MAX) stays Unknown WITHOUT
//     crashing (the documented limitation);
//   * a few pure-structural ops (Concat/Transpose/Gather) resolve with base=null.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "engine/ShapeInference.h"
#include "engine/ShapeInferenceExt.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

// Append a value with a given name; returns its index.
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

// Wire a node with the given input/output value indices. Sets producer of each
// output to this node.
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

}  // namespace

TEST_CASE("ShapeExt: Reshape with raw_data shape initializer resolves") {
  // A synthetic mmap holding the target shape [2, 12] as two little-endian
  // int64 values. infer_shapes_ext must read these through the base pointer.
  int64_t shape_vals[2] = {2, 12};
  std::vector<uint8_t> mmap(sizeof(shape_vals));
  std::memcpy(mmap.data(), shape_vals, sizeof(shape_vals));

  ir::Model m;
  m.format_name = m.intern("ONNX");
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t data = add_value(m, g, "data", ir::DType::F32, {4, 6});
  uint32_t shape = add_value(m, g, "shape", ir::DType::I64, {2});
  uint32_t out = add_value(m, g, "out", ir::DType::Unknown, {});
  add_node(m, g, "Reshape", {data, shape}, {out});

  // Record the shape tensor as a raw_data initializer at offset 0.
  ir::TensorRef tr;
  tr.name = g.values[shape].name;
  tr.dtype = ir::DType::I64;
  tr.shape.push_back(2);
  tr.file_offset = 0;
  tr.byte_len = sizeof(shape_vals);
  g.initializers.push_back(std::move(tr));

  uint32_t resolved =
      infer_shapes_ext(m, 0, mmap.data(), mmap.size(), nullptr);
  CHECK(resolved >= 1);

  const ir::ValueInfo& ov = g.values[out];
  REQUIRE(ov.shape.size() == 2);
  CHECK(ov.shape[0] == 2);
  CHECK(ov.shape[1] == 12);
  CHECK(ov.dtype == ir::DType::F32);  // dtype propagated from input
}

TEST_CASE("ShapeExt: int64_data shape (UINT64_MAX offset) stays Unknown, no crash") {
  // Documented limitation: a shape tensor packed into ONNX int64_data has no
  // recorded mmap offset (parser sets file_offset==UINT64_MAX). Ext inference
  // must leave the Reshape output Unknown and MUST NOT crash / read OOB.
  int64_t dummy[2] = {2, 12};
  std::vector<uint8_t> mmap(sizeof(dummy));
  std::memcpy(mmap.data(), dummy, sizeof(dummy));

  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t data = add_value(m, g, "data", ir::DType::F32, {4, 6});
  uint32_t shape = add_value(m, g, "shape", ir::DType::I64, {2});
  uint32_t out = add_value(m, g, "out", ir::DType::Unknown, {});
  add_node(m, g, "Reshape", {data, shape}, {out});

  ir::TensorRef tr;
  tr.name = g.values[shape].name;
  tr.dtype = ir::DType::I64;
  tr.shape.push_back(2);
  tr.file_offset = UINT64_MAX;  // int64_data: no readable offset
  tr.byte_len = 0;
  g.initializers.push_back(std::move(tr));

  // Should not crash even with a valid base present.
  infer_shapes_ext(m, 0, mmap.data(), mmap.size(), nullptr);

  const ir::ValueInfo& ov = g.values[out];
  CHECK(ov.shape.empty());              // unresolved shape
  CHECK(ov.dtype == ir::DType::F32);    // but dtype still propagates
}

TEST_CASE("ShapeExt: Concat sums along axis (structural, base=null)") {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::F32, {2, 3});
  uint32_t b = add_value(m, g, "b", ir::DType::F32, {2, 5});
  uint32_t out = add_value(m, g, "out", ir::DType::Unknown, {});
  ir::Node n;
  n.op_type = m.intern("Concat");
  n.name = m.intern("concat0");
  n.inputs.begin = 0;
  g.edge_refs.push_back(a);
  g.edge_refs.push_back(b);
  n.inputs.count = 2;
  n.outputs.begin = 2;
  g.edge_refs.push_back(out);
  n.outputs.count = 1;
  g.values[out].producer = 0;
  // axis = 1 attribute.
  ir::Attribute attr;
  attr.name = m.intern("axis");
  attr.value.kind = ir::AttrValue::Kind::Int;
  attr.value.i = 1;
  n.attributes.begin = 0;
  n.attributes.count = 1;
  g.attributes.push_back(std::move(attr));
  g.nodes.push_back(std::move(n));

  infer_shapes(m, 0, nullptr);  // 3-arg delegate

  const ir::ValueInfo& ov = g.values[out];
  REQUIRE(ov.shape.size() == 2);
  CHECK(ov.shape[0] == 2);
  CHECK(ov.shape[1] == 8);  // 3 + 5
}

TEST_CASE("ShapeExt: Transpose applies perm") {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t x = add_value(m, g, "x", ir::DType::F32, {2, 3, 4});
  uint32_t out = add_value(m, g, "out", ir::DType::Unknown, {});
  uint32_t node = add_node(m, g, "Transpose", {x}, {out});
  (void)node;

  ir::Attribute attr;
  attr.name = m.intern("perm");
  attr.value.kind = ir::AttrValue::Kind::Ints;
  attr.value.ints = {2, 0, 1};
  g.nodes[0].attributes.begin = static_cast<uint32_t>(g.attributes.size());
  g.nodes[0].attributes.count = 1;
  g.attributes.push_back(std::move(attr));

  infer_shapes(m, 0, nullptr);

  const ir::ValueInfo& ov = g.values[out];
  REQUIRE(ov.shape.size() == 3);
  CHECK(ov.shape[0] == 4);
  CHECK(ov.shape[1] == 2);
  CHECK(ov.shape[2] == 3);
}

// Helper: add an int attribute to the last node.
static void add_attr_int(ir::Model& m, ir::Graph& g, const std::string& name,
                         int64_t val) {
  ir::Attribute a;
  a.name = m.intern(name);
  a.value.kind = ir::AttrValue::Kind::Int;
  a.value.i = val;
  uint32_t idx = static_cast<uint32_t>(g.attributes.size());
  g.attributes.push_back(std::move(a));
  ir::Node& node = g.nodes.back();
  if (node.attributes.count == 0) node.attributes.begin = idx;
  node.attributes.count++;
}

static void add_attr_str(ir::Model& m, ir::Graph& g, const std::string& name,
                         const std::string& val) {
  ir::Attribute a;
  a.name = m.intern(name);
  a.value.kind = ir::AttrValue::Kind::String;
  a.value.s = m.intern(val);
  uint32_t idx = static_cast<uint32_t>(g.attributes.size());
  g.attributes.push_back(std::move(a));
  ir::Node& node = g.nodes.back();
  if (node.attributes.count == 0) node.attributes.begin = idx;
  node.attributes.count++;
}

TEST_CASE("ShapeExt: Gather inserts indices shape at axis") {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  // data [5, 8], indices [3] (1-D), axis 0 -> out [3, 8].
  uint32_t data = add_value(m, g, "data", ir::DType::F32, {5, 8});
  uint32_t idx = add_value(m, g, "idx", ir::DType::I64, {3});
  uint32_t out = add_value(m, g, "out", ir::DType::Unknown, {});
  add_node(m, g, "Gather", {data, idx}, {out});
  // axis default 0, no attribute needed.

  infer_shapes(m, 0, nullptr);

  const ir::ValueInfo& ov = g.values[out];
  REQUIRE(ov.shape.size() == 2);
  CHECK(ov.shape[0] == 3);
  CHECK(ov.shape[1] == 8);
  CHECK(ov.dtype == ir::DType::F32);
}

// =============================================================================
//  v0.4.0 — new shape handlers (Attention / Recurrent / Quantized / QDQ / gap-fill)
// =============================================================================

TEST_CASE("ShapeExt: LSTM resolves Y / Y_h / Y_c") {
  // X=[seq,batch,input]=[5,2,8]; hidden_size=4; direction=forward -> dirs=1.
  // Y=[seq,dirs,batch,hidden]=[5,1,2,4]; Y_h=Y_c=[dirs,batch,hidden]=[1,2,4].
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t x = add_value(m, g, "x", ir::DType::F32, {5, 2, 8});
  uint32_t w = add_value(m, g, "w", ir::DType::F32, {1, 16, 8});
  uint32_t rr = add_value(m, g, "r", ir::DType::F32, {1, 16, 4});
  uint32_t y = add_value(m, g, "y", ir::DType::Unknown, {});
  uint32_t yh = add_value(m, g, "yh", ir::DType::Unknown, {});
  uint32_t yc = add_value(m, g, "yc", ir::DType::Unknown, {});
  add_node(m, g, "LSTM", {x, w, rr}, {y, yh, yc});
  add_attr_int(m, g, "hidden_size", 4);
  add_attr_str(m, g, "direction", "forward");

  infer_shapes(m, 0, nullptr);

  const ir::ValueInfo& vy = g.values[y];
  REQUIRE(vy.shape.size() == 4);
  CHECK(vy.shape[0] == 5);
  CHECK(vy.shape[1] == 1);
  CHECK(vy.shape[2] == 2);
  CHECK(vy.shape[3] == 4);
  const ir::ValueInfo& vyh = g.values[yh];
  REQUIRE(vyh.shape.size() == 3);
  CHECK(vyh.shape[0] == 1);
  CHECK(vyh.shape[1] == 2);
  CHECK(vyh.shape[2] == 4);
  const ir::ValueInfo& vyc = g.values[yc];
  REQUIRE(vyc.shape.size() == 3);  // gate G=4 emits Y_c
  CHECK(vyc.shape[2] == 4);
}

TEST_CASE("ShapeExt: bidirectional LSTM has dirs=2 in Y") {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t x = add_value(m, g, "x", ir::DType::F32, {5, 2, 8});
  uint32_t w = add_value(m, g, "w", ir::DType::F32, {2, 16, 8});
  uint32_t rr = add_value(m, g, "r", ir::DType::F32, {2, 16, 4});
  uint32_t y = add_value(m, g, "y", ir::DType::Unknown, {});
  add_node(m, g, "LSTM", {x, w, rr}, {y});
  add_attr_int(m, g, "hidden_size", 4);
  add_attr_str(m, g, "direction", "bidirectional");

  infer_shapes(m, 0, nullptr);

  const ir::ValueInfo& vy = g.values[y];
  REQUIRE(vy.shape.size() == 4);
  CHECK(vy.shape[1] == 2);  // num_directions
}

TEST_CASE("ShapeExt: LSTM with dynamic seq stays honestly partial/unknown") {
  // Dynamic seq (-1) — the handler may leave a partial (-1) dim or dtype-only, but
  // must not fabricate a concrete seq extent and must not crash.
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t x = add_value(m, g, "x", ir::DType::F32, {-1, 2, 8});
  uint32_t w = add_value(m, g, "w", ir::DType::F32, {1, 16, 8});
  uint32_t rr = add_value(m, g, "r", ir::DType::F32, {1, 16, 4});
  uint32_t y = add_value(m, g, "y", ir::DType::Unknown, {});
  add_node(m, g, "LSTM", {x, w, rr}, {y});
  add_attr_int(m, g, "hidden_size", 4);
  add_attr_str(m, g, "direction", "forward");

  infer_shapes(m, 0, nullptr);  // must not crash

  const ir::ValueInfo& vy = g.values[y];
  // If a shape was emitted, the seq dim must NOT be a fabricated positive extent.
  if (vy.shape.size() == 4) {
    CHECK(vy.shape[0] <= 0);   // partial (-1), not a made-up seq length
    CHECK(vy.shape[3] == 4);   // hidden still resolvable
  }
}

TEST_CASE("ShapeExt: MultiHeadAttention output resolves") {
  // Pre-projected: query=in[0]=[B,S_q,hidden]=[1,4,8]; out=[B,S_q,v_hidden].
  // value rank3 -> v_hidden = value.back(). value=[1,4,8] -> out=[1,4,8].
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t q = add_value(m, g, "q", ir::DType::F32, {1, 4, 8});
  uint32_t k = add_value(m, g, "k", ir::DType::F32, {1, 4, 8});
  uint32_t v = add_value(m, g, "v", ir::DType::F32, {1, 4, 8});
  uint32_t out = add_value(m, g, "out", ir::DType::Unknown, {});
  add_node(m, g, "MultiHeadAttention", {q, k, v}, {out});

  infer_shapes(m, 0, nullptr);

  const ir::ValueInfo& ov = g.values[out];
  REQUIRE(ov.shape.size() == 3);
  CHECK(ov.shape[0] == 1);
  CHECK(ov.shape[1] == 4);
  CHECK(ov.shape[2] == 8);  // v_hidden
  CHECK(ov.dtype == ir::DType::F32);
}

TEST_CASE("ShapeExt: QLinearConv output shape + dtype") {
  // Conv math on slots 0 (x) / 3 (w). x=[1,32,30,30], w=[64,32,3,3], stride 1,
  // no pad -> out spatial 28x28 -> out=[1,64,28,28]. Out dtype = y_zp (slot7) U8.
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t x = add_value(m, g, "x", ir::DType::U8, {1, 32, 30, 30});
  uint32_t xsc = add_value(m, g, "x_scale", ir::DType::F32, {1});
  uint32_t xzp = add_value(m, g, "x_zp", ir::DType::U8, {1});
  uint32_t w = add_value(m, g, "w", ir::DType::U8, {64, 32, 3, 3});
  uint32_t wsc = add_value(m, g, "w_scale", ir::DType::F32, {1});
  uint32_t wzp = add_value(m, g, "w_zp", ir::DType::U8, {1});
  uint32_t ysc = add_value(m, g, "y_scale", ir::DType::F32, {1});
  uint32_t yzp = add_value(m, g, "y_zp", ir::DType::U8, {1});
  uint32_t out = add_value(m, g, "out", ir::DType::Unknown, {});
  add_node(m, g, "QLinearConv", {x, xsc, xzp, w, wsc, wzp, ysc, yzp}, {out});

  infer_shapes(m, 0, nullptr);

  const ir::ValueInfo& ov = g.values[out];
  REQUIRE(ov.shape.size() == 4);
  CHECK(ov.shape[0] == 1);
  CHECK(ov.shape[1] == 64);
  CHECK(ov.shape[2] == 28);
  CHECK(ov.shape[3] == 28);
  CHECK(ov.dtype == ir::DType::U8);  // from y_zp slot
}

TEST_CASE("ShapeExt: MatMulInteger output shape + I32 dtype") {
  // A=[8,16], B=[16,32] -> out=[8,32], dtype I32 (integer accumulation).
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t a = add_value(m, g, "a", ir::DType::U8, {8, 16});
  uint32_t b = add_value(m, g, "b", ir::DType::I8, {16, 32});
  uint32_t out = add_value(m, g, "out", ir::DType::Unknown, {});
  add_node(m, g, "MatMulInteger", {a, b}, {out});

  infer_shapes(m, 0, nullptr);

  const ir::ValueInfo& ov = g.values[out];
  REQUIRE(ov.shape.size() == 2);
  CHECK(ov.shape[0] == 8);
  CHECK(ov.shape[1] == 32);
  CHECK(ov.dtype == ir::DType::I32);
}

TEST_CASE("ShapeExt: QuantizeLinear flips dtype, DequantizeLinear flips back") {
  // QuantizeLinear: shape-preserving, output dtype from y_zp (slot2) or U8.
  ir::Model mq;
  mq.graphs.emplace_back();
  ir::Graph& gq = mq.graphs[0];
  uint32_t xf = add_value(mq, gq, "xf", ir::DType::F32, {2, 3});
  uint32_t qsc = add_value(mq, gq, "q_scale", ir::DType::F32, {1});
  uint32_t qzp = add_value(mq, gq, "q_zp", ir::DType::U8, {1});
  uint32_t qout = add_value(mq, gq, "qout", ir::DType::Unknown, {});
  add_node(mq, gq, "QuantizeLinear", {xf, qsc, qzp}, {qout});
  infer_shapes(mq, 0, nullptr);
  const ir::ValueInfo& qv = gq.values[qout];
  REQUIRE(qv.shape.size() == 2);
  CHECK(qv.shape[0] == 2);
  CHECK(qv.shape[1] == 3);
  CHECK(qv.dtype == ir::DType::U8);  // quantized integer

  // DequantizeLinear: shape-preserving, output dtype from scale (slot1) -> F32.
  ir::Model md;
  md.graphs.emplace_back();
  ir::Graph& gd = md.graphs[0];
  uint32_t xq = add_value(md, gd, "xq", ir::DType::U8, {2, 3});
  uint32_t dsc = add_value(md, gd, "d_scale", ir::DType::F32, {1});
  uint32_t dzp = add_value(md, gd, "d_zp", ir::DType::U8, {1});
  uint32_t dout = add_value(md, gd, "dout", ir::DType::Unknown, {});
  add_node(md, gd, "DequantizeLinear", {xq, dsc, dzp}, {dout});
  infer_shapes(md, 0, nullptr);
  const ir::ValueInfo& dv = gd.values[dout];
  REQUIRE(dv.shape.size() == 2);
  CHECK(dv.dtype == ir::DType::F32);  // dequantized back to float
}

TEST_CASE("ShapeExt: gap-fill unary Mish is shape-preserving") {
  ir::Model m;
  m.graphs.emplace_back();
  ir::Graph& g = m.graphs[0];

  uint32_t x = add_value(m, g, "x", ir::DType::F32, {3, 5, 7});
  uint32_t out = add_value(m, g, "out", ir::DType::Unknown, {});
  add_node(m, g, "Mish", {x}, {out});

  infer_shapes(m, 0, nullptr);

  const ir::ValueInfo& ov = g.values[out];
  REQUIRE(ov.shape.size() == 3);
  CHECK(ov.shape[0] == 3);
  CHECK(ov.shape[1] == 5);
  CHECK(ov.shape[2] == 7);
  CHECK(ov.dtype == ir::DType::F32);
}
