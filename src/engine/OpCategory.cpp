// engine/OpCategory.cpp — op_type string -> coloring category (spec §8.1).
//
// Pure logic, no GUI. The classification is case-insensitive on the last dot
// segment of the op name so framework/domain prefixes ("com.microsoft.Gelu",
// "ai.onnx.Conv") are tolerated. A single static lookup table keeps this O(1)
// per call and puts the whole palette-driving decision in one place.
#include "engine/OpCategory.h"

#include <cctype>
#include <string>
#include <unordered_map>

namespace netvis {

namespace {

// Lowercase copy of the last dot-segment of `op_type`. Domain prefixes such as
// "com.microsoft." are stripped by keeping only the final segment.
std::string normalize(std::string_view op_type) {
  // Keep the substring after the last '.' (the bare op name).
  size_t dot = op_type.rfind('.');
  std::string_view last = (dot == std::string_view::npos)
                              ? op_type
                              : op_type.substr(dot + 1);
  std::string out;
  out.reserve(last.size());
  for (char c : last)
    out.push_back(static_cast<char>(
        std::tolower(static_cast<unsigned char>(c))));
  return out;
}

// Build once (function-local static): keys are lowercase op names. string_view
// keys point at string literals with static storage, so they never dangle.
const std::unordered_map<std::string_view, OpCategory>& table() {
  static const std::unordered_map<std::string_view, OpCategory> t = {
      // Conv
      {"conv", OpCategory::Conv},
      {"convtranspose", OpCategory::Conv},
      // MatMul
      {"matmul", OpCategory::MatMul},
      {"gemm", OpCategory::MatMul},
      {"linear", OpCategory::MatMul},
      {"einsum", OpCategory::MatMul},
      // Activation
      {"relu", OpCategory::Activation},
      {"leakyrelu", OpCategory::Activation},
      {"gelu", OpCategory::Activation},
      {"sigmoid", OpCategory::Activation},
      {"tanh", OpCategory::Activation},
      {"softmax", OpCategory::Activation},
      {"elu", OpCategory::Activation},
      {"selu", OpCategory::Activation},
      {"clip", OpCategory::Activation},
      {"prelu", OpCategory::Activation},
      {"hardsigmoid", OpCategory::Activation},
      {"swish", OpCategory::Activation},
      {"silu", OpCategory::Activation},
      // Norm — both the ONNX "*Normalization" op names and the bare short forms
      // common in PyTorch/exported/custom graphs (e.g. "LayerNorm", "BatchNorm").
      {"batchnormalization", OpCategory::Norm},
      {"batchnorm", OpCategory::Norm},
      {"layernormalization", OpCategory::Norm},
      {"layernorm", OpCategory::Norm},
      {"groupnormalization", OpCategory::Norm},
      {"groupnorm", OpCategory::Norm},
      {"instancenormalization", OpCategory::Norm},
      {"instancenorm", OpCategory::Norm},
      {"rmsnorm", OpCategory::Norm},
      {"rmsnormalization", OpCategory::Norm},
      {"lpnormalization", OpCategory::Norm},
      // Pool
      {"maxpool", OpCategory::Pool},
      {"averagepool", OpCategory::Pool},
      {"globalaveragepool", OpCategory::Pool},
      {"globalmaxpool", OpCategory::Pool},
      // Elementwise
      {"add", OpCategory::Elementwise},
      {"sub", OpCategory::Elementwise},
      {"mul", OpCategory::Elementwise},
      {"div", OpCategory::Elementwise},
      {"pow", OpCategory::Elementwise},
      {"sqrt", OpCategory::Elementwise},
      {"exp", OpCategory::Elementwise},
      {"log", OpCategory::Elementwise},
      {"abs", OpCategory::Elementwise},
      {"neg", OpCategory::Elementwise},
      {"min", OpCategory::Elementwise},
      {"max", OpCategory::Elementwise},
      {"where", OpCategory::Elementwise},
      {"equal", OpCategory::Elementwise},
      {"greater", OpCategory::Elementwise},
      {"less", OpCategory::Elementwise},
      {"and", OpCategory::Elementwise},
      {"or", OpCategory::Elementwise},
      // Tensor
      {"constant", OpCategory::Tensor},
      {"cast", OpCategory::Tensor},
      // Shape
      {"reshape", OpCategory::Shape},
      {"transpose", OpCategory::Shape},
      {"concat", OpCategory::Shape},
      {"slice", OpCategory::Shape},
      {"split", OpCategory::Shape},
      {"squeeze", OpCategory::Shape},
      {"unsqueeze", OpCategory::Shape},
      {"gather", OpCategory::Shape},
      {"flatten", OpCategory::Shape},
      {"expand", OpCategory::Shape},
      {"pad", OpCategory::Shape},
      {"tile", OpCategory::Shape},
      {"shape", OpCategory::Shape},
      // Reduce
      {"reducesum", OpCategory::Reduce},
      {"reducemean", OpCategory::Reduce},
      {"reducemax", OpCategory::Reduce},
      {"reducemin", OpCategory::Reduce},
      {"reduceprod", OpCategory::Reduce},
      // ControlFlow
      {"if", OpCategory::ControlFlow},
      {"loop", OpCategory::ControlFlow},
      {"scan", OpCategory::ControlFlow},
  };
  return t;
}

}  // namespace

// Classify an op_type string; unknown ops fall through to Other.
OpCategory categorize_op(std::string_view op_type) {
  if (op_type.empty()) return OpCategory::Other;
  const std::string key = normalize(op_type);
  const auto& t = table();
  auto it = t.find(std::string_view{key});
  return it == t.end() ? OpCategory::Other : it->second;
}

// Stable label per category (used by tests/legends; not the palette).
const char* category_name(OpCategory c) {
  switch (c) {
    case OpCategory::Conv: return "Conv";
    case OpCategory::MatMul: return "MatMul";
    case OpCategory::Activation: return "Activation";
    case OpCategory::Norm: return "Norm";
    case OpCategory::Pool: return "Pool";
    case OpCategory::Elementwise: return "Elementwise";
    case OpCategory::Shape: return "Shape";
    case OpCategory::Reduce: return "Reduce";
    case OpCategory::Tensor: return "Tensor";
    case OpCategory::ControlFlow: return "ControlFlow";
    case OpCategory::IO: return "IO";
    case OpCategory::Other: return "Other";
  }
  return "Other";
}

}  // namespace netvis
