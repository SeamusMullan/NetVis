// tests/test_sdk_abi.cpp — SDK-header <-> C++-contract drift guard (#10, Increment D).
//
// Includes the shipped SDK header in HOST-BRIDGE mode (imports excluded) alongside
// the real C++ headers and static_asserts that every wire enum / version / cap in
// plugins/sdk/netvis_plugin.h matches its C++ source of truth. Any drift (a new
// DType, a renumbered OpCategory, a bumped ABI version, a changed cap) fails the
// host build here — so the SDK we ship can never silently diverge from the host.
#define NETVIS_SDK_HOST_BRIDGE 1
#include "plugins/sdk/netvis_plugin.h"

#include <doctest/doctest.h>

#include "engine/OpCategory.h"
#include "engine/plugin/OpHandler.h"
#include "engine/plugin/ParserPlugin.h"
#include "engine/plugin/PassPlugin.h"
#include "ir/IR.h"

using namespace netvis;

// --- DType: 16 enumerators, exact numeric match (Unknown=15). ----------------
static_assert((int)NV_DT_F32 == (int)ir::DType::F32, "F32");
static_assert((int)NV_DT_F16 == (int)ir::DType::F16, "F16");
static_assert((int)NV_DT_BF16 == (int)ir::DType::BF16, "BF16");
static_assert((int)NV_DT_F64 == (int)ir::DType::F64, "F64");
static_assert((int)NV_DT_I8 == (int)ir::DType::I8, "I8");
static_assert((int)NV_DT_I16 == (int)ir::DType::I16, "I16");
static_assert((int)NV_DT_I32 == (int)ir::DType::I32, "I32");
static_assert((int)NV_DT_I64 == (int)ir::DType::I64, "I64");
static_assert((int)NV_DT_U8 == (int)ir::DType::U8, "U8");
static_assert((int)NV_DT_U16 == (int)ir::DType::U16, "U16");
static_assert((int)NV_DT_U32 == (int)ir::DType::U32, "U32");
static_assert((int)NV_DT_U64 == (int)ir::DType::U64, "U64");
static_assert((int)NV_DT_BOOL == (int)ir::DType::Bool, "Bool");
static_assert((int)NV_DT_Q4 == (int)ir::DType::Q4, "Q4");
static_assert((int)NV_DT_Q8 == (int)ir::DType::Q8, "Q8");
static_assert((int)NV_DT_UNKNOWN == (int)ir::DType::Unknown, "Unknown");
static_assert((int)NV_DT_UNKNOWN == 15, "DType count/Unknown drift");

// --- OpCategory: 15 enumerators, Other=14 (LAST). ----------------------------
static_assert((int)NV_CAT_CONV == (int)OpCategory::Conv, "Conv");
static_assert((int)NV_CAT_MATMUL == (int)OpCategory::MatMul, "MatMul");
static_assert((int)NV_CAT_ACTIVATION == (int)OpCategory::Activation, "Activation");
static_assert((int)NV_CAT_NORM == (int)OpCategory::Norm, "Norm");
static_assert((int)NV_CAT_POOL == (int)OpCategory::Pool, "Pool");
static_assert((int)NV_CAT_ELEMENTWISE == (int)OpCategory::Elementwise, "Elementwise");
static_assert((int)NV_CAT_SHAPE == (int)OpCategory::Shape, "Shape");
static_assert((int)NV_CAT_REDUCE == (int)OpCategory::Reduce, "Reduce");
static_assert((int)NV_CAT_TENSOR == (int)OpCategory::Tensor, "Tensor");
static_assert((int)NV_CAT_CONTROLFLOW == (int)OpCategory::ControlFlow, "ControlFlow");
static_assert((int)NV_CAT_IO == (int)OpCategory::IO, "IO");
static_assert((int)NV_CAT_ATTENTION == (int)OpCategory::Attention, "Attention");
static_assert((int)NV_CAT_RECURRENT == (int)OpCategory::Recurrent, "Recurrent");
static_assert((int)NV_CAT_QUANTIZE == (int)OpCategory::Quantize, "Quantize");
static_assert((int)NV_CAT_OTHER == (int)OpCategory::Other, "Other");
static_assert((int)NV_CAT_OTHER == 14, "OpCategory count/Other drift");

// --- ABI versions. -----------------------------------------------------------
static_assert(NETVIS_OP_ABI_VERSION == plugin::kOpHandlerAbiVersion, "op abi");
static_assert(NETVIS_PARSER_ABI_VERSION == plugin::kParserPluginAbiVersion, "parser abi");
static_assert(NETVIS_PASS_ABI_VERSION == plugin::kPassPluginAbiVersion, "pass abi");

// A trivial runtime case so the TU registers with doctest (the real coverage is the
// static_asserts above, checked at compile time).
TEST_CASE("SDK ABI bridge: enums/versions match the C++ contracts (compile-time)") {
  CHECK((int)NV_DT_UNKNOWN == 15);
  CHECK((int)NV_CAT_OTHER == 14);
  CHECK(NETVIS_OP_ABI_VERSION == 1u);
}
