// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeOps.h — Caffe layer type -> IR op mapping (#138/#109).
//
// DECISION (honesty rule): a Caffe layer is renamed to an ONNX op ONLY when the
// ONNX op has IDENTICAL shape semantics in ShapeInferenceExt and CostModel under
// the attributes we derive ("vetted"). Then colouring, shape inference and the
// cost analyzer apply for free. Everything else keeps its Caffe name — prefixed
// with "Caffe" whenever the bare name would collide with an engine handler (a
// Caffe Split COPIES its input; an ONNX Split PARTITIONS it) — so no unvetted
// layer can inherit ONNX semantics it does not have. Derived attributes carry
// Caffe's defaults made explicit and are emitted only for vetted ops.
//
// A separate TU so the collision-guard test can drive the table directly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "ir/IR.h"
#include "parsers/caffe/CaffeNet.h"

namespace netvis::caffe {

// Bound on kernel / stride / pad / dilation / group / num_output. Together with
// sanitize_shape's INT32_MAX element rule this keeps infer_shapes_ext's int64
// arithmetic far from overflow (UBSan runs the tests).
constexpr int64_t kMaxGeometry = 1 << 20;

struct DerivedAttr {
  std::string name;
  ir::AttrValue::Kind kind = ir::AttrValue::Kind::Int;
  int64_t i = 0;
  double f = 0.0;
  std::vector<int64_t> ints;
};

struct MappedOp {
  std::string op_type;
  bool vetted = false;
  std::vector<DerivedAttr> derived;
};

// One row per BVLC layer type: its fallback (unvetted) op name, or nullptr when
// the type is always vetted (Input). test_bottoms / test_tops is an arity no
// vetted mapping of that type accepts, so the collision-guard test can build an
// unvetted instance without any parameters.
struct BvlcLayerType {
  const char* type;
  const char* fallback;
  uint8_t test_bottoms;
  uint8_t test_tops;
};
extern const BvlcLayerType kBvlcLayerTypes[];
extern const size_t kBvlcLayerTypeCount;

// Map a layer. `weight_rank` is the rank of its first blob when that blob's shape
// is known, else -1. Reads the layer's native attributes (interned in `model`).
MappedOp map_layer(const LayerRaw& layer, const ir::Model& model, int weight_rank);

// Display name for a type NetVis does not know (a fork's layer): verbatim, unless
// it names an op an engine handler would claim, in which case "Caffe" + type.
std::string safe_display_name(std::string_view type);

// Role of blob `index` of `layer` ("weight", "bias", "mean", ...; "blob<i>").
std::string blob_role(const LayerRaw& layer, size_t index);

// Conv/Deconv geometry (§3.8), shared by the mapper and pairing verification.
struct ConvGeometry {
  bool determined = false;   // spatial rank R known
  bool ok = true;            // every list resolvable and within kMaxGeometry
  int64_t rank = -1;
  std::vector<int64_t> kernel;      // empty => no kernel declared (engine reads w[2+d])
  std::vector<int64_t> strides, pads /* 2R */, dilations;
  int64_t group = 1;
  bool has_num_output = false;
  int64_t num_output = 0;
  bool bias_term = true;
};
ConvGeometry resolve_conv_geometry(const LayerRaw& layer, const ir::Model& model,
                                   int weight_rank);

// Native attribute lookup on a raw layer (nullptr when absent).
const ir::AttrValue* layer_attr(const LayerRaw& layer, const ir::Model& model,
                                std::string_view name);

// Expected learnable-blob count for pairing verification, or -1 when any count
// is accepted.
int expected_blob_count(const LayerRaw& layer, const ir::Model& model);

}  // namespace netvis::caffe
