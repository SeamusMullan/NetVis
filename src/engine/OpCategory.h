// engine/OpCategory.h — map an op_type string to a coloring category.
//
// DECISION (spec §8.1): node header strips are colored by op category. The
// classification is pure logic (op string -> enum), so it lives in the engine
// and the view only maps category -> color. Keeps the palette decision in one
// place and testable without a GUI.
#pragma once

#include <cstdint>
#include <string_view>

namespace netvis {

enum class OpCategory : uint8_t {
  Conv,        // Conv, ConvTranspose, ...
  MatMul,      // MatMul, Gemm, Linear, Einsum
  Activation,  // Relu, Gelu, Sigmoid, Tanh, Softmax, ...
  Norm,        // BatchNorm, LayerNorm, GroupNorm, RMSNorm, ...
  Pool,        // MaxPool, AveragePool, GlobalAveragePool, ...
  Elementwise, // Add, Mul, Sub, Div, Pow, Sqrt, ...
  Shape,       // Reshape, Transpose, Concat, Slice, Gather, Squeeze, ...
  Reduce,      // ReduceSum, ReduceMean, ...
  Tensor,      // Constant, Cast, initializer-ish
  ControlFlow, // If, Loop, Scan
  IO,          // graph input/output markers
  Other,
};

// Classify an op_type (case-insensitive on the leading token). Framework
// prefixes (e.g. "com.microsoft.") are tolerated.
OpCategory categorize_op(std::string_view op_type);

const char* category_name(OpCategory c);

}  // namespace netvis
