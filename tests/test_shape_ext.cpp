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
