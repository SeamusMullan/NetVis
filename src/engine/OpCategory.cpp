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
      {"softplus", OpCategory::Activation},
      {"softsign", OpCategory::Activation},
      {"hardswish", OpCategory::Activation},
      {"mish", OpCategory::Activation},
      {"logsoftmax", OpCategory::Activation},
      {"celu", OpCategory::Activation},
      {"thresholdedrelu", OpCategory::Activation},
      {"shrink", OpCategory::Activation},
      {"quickgelu", OpCategory::Activation},
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
      {"sin", OpCategory::Elementwise},
      {"cos", OpCategory::Elementwise},
      {"tan", OpCategory::Elementwise},
      {"asin", OpCategory::Elementwise},
      {"acos", OpCategory::Elementwise},
      {"atan", OpCategory::Elementwise},
      {"sinh", OpCategory::Elementwise},
      {"cosh", OpCategory::Elementwise},
      {"asinh", OpCategory::Elementwise},
      {"acosh", OpCategory::Elementwise},
      {"atanh", OpCategory::Elementwise},
      {"erf", OpCategory::Elementwise},
      {"reciprocal", OpCategory::Elementwise},
      {"floor", OpCategory::Elementwise},
      {"ceil", OpCategory::Elementwise},
      {"round", OpCategory::Elementwise},
      {"sign", OpCategory::Elementwise},
      {"mod", OpCategory::Elementwise},
      {"not", OpCategory::Elementwise},
      {"xor", OpCategory::Elementwise},
      {"bitwiseand", OpCategory::Elementwise},
      {"bitwiseor", OpCategory::Elementwise},
      {"bitwisexor", OpCategory::Elementwise},
      {"bitwisenot", OpCategory::Elementwise},
      {"bitshift", OpCategory::Elementwise},
      {"isnan", OpCategory::Elementwise},
      {"isinf", OpCategory::Elementwise},
      {"sum", OpCategory::Elementwise},
      {"mean", OpCategory::Elementwise},
      {"greaterorequal", OpCategory::Elementwise},
      {"lessorequal", OpCategory::Elementwise},
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
      {"reducel1", OpCategory::Reduce},
      {"reducel2", OpCategory::Reduce},
      {"reducelogsum", OpCategory::Reduce},
      {"reducelogsumexp", OpCategory::Reduce},
      {"reducesumsquare", OpCategory::Reduce},
      {"argmax", OpCategory::Reduce},
      {"argmin", OpCategory::Reduce},
      {"cumsum", OpCategory::Reduce},
      // ControlFlow
      {"if", OpCategory::ControlFlow},
      {"loop", OpCategory::ControlFlow},
      {"scan", OpCategory::ControlFlow},
      // Attention
      {"attention", OpCategory::Attention},
      {"multiheadattention", OpCategory::Attention},
      // Recurrent
      {"lstm", OpCategory::Recurrent},
      {"gru", OpCategory::Recurrent},
      {"rnn", OpCategory::Recurrent},
      // Quantize — QDQ marker ops only. Quant COMPUTE ops are colored like their
      // float sibling (mapped to Conv/MatMul/Elementwise/Pool below).
      {"quantizelinear", OpCategory::Quantize},
      {"dequantizelinear", OpCategory::Quantize},
      {"dynamicquantizelinear", OpCategory::Quantize},
      // Quantized compute ops -> float-sibling category (spec §8.1 nuance).
      {"qlinearconv", OpCategory::Conv},
      {"convinteger", OpCategory::Conv},
      {"qlinearmatmul", OpCategory::MatMul},
      {"matmulinteger", OpCategory::MatMul},
      {"qgemm", OpCategory::MatMul},
      {"qlinearadd", OpCategory::Elementwise},
      {"qlinearmul", OpCategory::Elementwise},
      {"qlinearaveragepool", OpCategory::Pool},
      {"qlinearglobalaveragepool", OpCategory::Pool},
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
    case OpCategory::Attention: return "Attention";
    case OpCategory::Recurrent: return "Recurrent";
    case OpCategory::Quantize: return "Quantize";
    case OpCategory::Other: return "Other";
  }
  return "Other";
}

}  // namespace netvis
