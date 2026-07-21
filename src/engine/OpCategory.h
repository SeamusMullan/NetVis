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

// NOTE: Other MUST remain the LAST enumerator — GraphNav's legend/filter loop
// iterates `c <= static_cast<int>(OpCategory::Other)` and App::category_color
// indexes a palette array by enum value with an Other-clamp. New categories are
// inserted BEFORE Other (v0.4.0 added Attention/Recurrent/Quantize). No code
// serializes OpCategory (not in LayoutCache/prefs), so Other's numeric shift is
// safe; category_mask is uint32 so up to 32 categories fit.
enum class OpCategory : uint8_t {
  Conv,        // Conv, ConvTranspose, QLinearConv, ConvInteger
  MatMul,      // MatMul, Gemm, Linear, Einsum, QLinearMatMul, MatMulInteger, QGemm
  Activation,  // Relu, Gelu, Sigmoid, Tanh, Softmax, ...
  Norm,        // BatchNorm, LayerNorm, GroupNorm, RMSNorm, ...
  Pool,        // MaxPool, AveragePool, GlobalAveragePool, QLinear*Pool
  Elementwise, // Add, Mul, Sub, Div, Pow, Sqrt, QLinearAdd/Mul, ...
  Shape,       // Reshape, Transpose, Concat, Slice, Gather, Squeeze, ...
  Reduce,      // ReduceSum, ReduceMean, ReduceL1/L2, ArgMax/ArgMin, CumSum, ...
  Tensor,      // Constant, Cast, initializer-ish
  ControlFlow, // If, Loop, Scan
  IO,          // graph input/output markers
  Attention,   // (v0.4.0) Attention, MultiHeadAttention
  Recurrent,   // (v0.4.0) LSTM, GRU, RNN
  Quantize,    // (v0.4.0) QuantizeLinear/DequantizeLinear/DynamicQuantizeLinear markers
  Other,
};

// Classify an op_type (case-insensitive on the leading token). Framework
// prefixes (e.g. "com.microsoft.") are tolerated.
OpCategory categorize_op(std::string_view op_type);

const char* category_name(OpCategory c);

}  // namespace netvis
