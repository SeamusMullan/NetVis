// SPDX-License-Identifier: Apache-2.0
// engine/OpCategory.cpp — op_type string -> coloring category (spec §8.1).
//
// Pure logic, no GUI. The classification is case-insensitive on the bare op
// name: for dotted op types the last dot segment, so framework/domain prefixes
// ("com.microsoft.Gelu", "ai.onnx.Conv") are tolerated; for TorchScript
// "ns::name[.overload]" op types (#137: "aten::relu_", "aten::add.Tensor",
// "prim::If") the name after the last "::", without its overload and without
// the trailing '_' of an in-place variant. A single static lookup table keeps
// this O(1) per call and puts the whole palette-driving decision in one place.
#include "engine/OpCategory.h"

#include <cctype>
#include <string>
#include <unordered_map>

namespace netvis {

namespace {

// Lowercase bare op name of `op_type` (see the file comment).
std::string normalize(std::string_view op_type) {
  std::string_view last;
  size_t ns = op_type.rfind("::");
  if (ns != std::string_view::npos) {
    // TorchScript: "aten::add.Tensor" -> "add", "aten::relu_" -> "relu".
    last = op_type.substr(ns + 2);
    size_t dot = last.find('.');
    if (dot != std::string_view::npos) last = last.substr(0, dot);
    // In-place variants end in exactly one '_'; dunder names ("__is__") keep it.
    if (last.size() >= 2 && last.substr(0, 2) != "__" && last.back() == '_' &&
        last[last.size() - 2] != '_')
      last.remove_suffix(1);
  } else {
    // Keep the substring after the last '.' (the bare op name).
    size_t dot = op_type.rfind('.');
    last = (dot == std::string_view::npos) ? op_type : op_type.substr(dot + 1);
  }
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
      {"matmulnbits", OpCategory::MatMul},  // #1 com.microsoft weight-quant MatMul
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
      {"greaterequal", OpCategory::Elementwise},
      {"lessequal", OpCategory::Elementwise},
      {"select", OpCategory::Elementwise},
      // Tensor
      {"constant", OpCategory::Tensor},
      {"cast", OpCategory::Tensor},
      {"convert", OpCategory::Tensor},
      {"range", OpCategory::Tensor},
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
      {"shapeof", OpCategory::Shape},
      {"broadcast", OpCategory::Shape},
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
      {"topk", OpCategory::Reduce},
      // ControlFlow
      {"if", OpCategory::ControlFlow},
      {"loop", OpCategory::ControlFlow},
      {"scan", OpCategory::ControlFlow},
      // Attention
      {"attention", OpCategory::Attention},
      {"multiheadattention", OpCategory::Attention},
      {"scaleddotproductattention", OpCategory::Attention},
      // IO
      {"parameter", OpCategory::IO},
      {"result", OpCategory::IO},
      {"readvalue", OpCategory::IO},
      {"assign", OpCategory::IO},
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
      // TorchScript aten:: names (#137), reached through the "::" rule above.
      // Deliberately NOT mapped: addmm/baddbmm (CostModel's MatMul formula would
      // read K from the bias input) and conv_transpose* (its Conv formula only
      // special-cases ONNX ConvTranspose). They stay Other until an aten shape
      // pass exists and the formulas are audited.
      {"conv1d", OpCategory::Conv},
      {"conv2d", OpCategory::Conv},
      {"conv3d", OpCategory::Conv},
      {"convolution", OpCategory::Conv},
      {"_convolution", OpCategory::Conv},
      {"mm", OpCategory::MatMul},
      {"bmm", OpCategory::MatMul},
      {"batch_norm", OpCategory::Norm},
      {"layer_norm", OpCategory::Norm},
      {"group_norm", OpCategory::Norm},
      {"instance_norm", OpCategory::Norm},
      {"relu6", OpCategory::Activation},
      {"hardtanh", OpCategory::Activation},
      {"leaky_relu", OpCategory::Activation},
      {"log_softmax", OpCategory::Activation},
      {"max_pool1d", OpCategory::Pool},
      {"max_pool2d", OpCategory::Pool},
      {"max_pool3d", OpCategory::Pool},
      {"avg_pool1d", OpCategory::Pool},
      {"avg_pool2d", OpCategory::Pool},
      {"avg_pool3d", OpCategory::Pool},
      {"adaptive_avg_pool1d", OpCategory::Pool},
      {"adaptive_avg_pool2d", OpCategory::Pool},
      {"adaptive_avg_pool3d", OpCategory::Pool},
      {"adaptive_max_pool1d", OpCategory::Pool},
      {"adaptive_max_pool2d", OpCategory::Pool},
      {"adaptive_max_pool3d", OpCategory::Pool},
      {"cat", OpCategory::Shape},
      {"stack", OpCategory::Shape},
      {"view", OpCategory::Shape},
      {"permute", OpCategory::Shape},
      {"contiguous", OpCategory::Shape},
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

std::optional<OpCategory> category_from_name(std::string_view s) {
  for (int i = 0; i <= static_cast<int>(OpCategory::Other); ++i) {
    OpCategory c = static_cast<OpCategory>(i);
    if (s == category_name(c)) return c;
  }
  return std::nullopt;
}

}  // namespace netvis
