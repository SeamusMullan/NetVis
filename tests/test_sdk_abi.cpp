// SPDX-License-Identifier: Apache-2.0
// tests/test_sdk_abi.cpp — SDK-header <-> C++-contract drift guard (#10, Increment D).
//
// Includes the shipped SDK header in HOST-BRIDGE mode (imports excluded) alongside
// the real C++ headers and static_asserts that every wire enum / version in
// plugins/sdk/netvis_plugin.h matches its C++ source of truth, and that the wire
// struct's layout (size, alignment, every member offset) is the one the header
// documents. Any drift (a new DType, a renumbered OpCategory, a bumped ABI version, a
// reshuffled struct) fails the host build here — so the SDK we ship can never
// silently diverge from the host.
//
// The marshalling/sandbox CAPS are not copied into a second place to be compared:
// the host takes them straight from the header (src/engine/plugin/wasm/SdkCaps.h)
// and static_asserts its per-facet budgets against the header's ceilings, so a cap
// edited in the header changes the host in the same commit. What each import's
// SIGNATURE and each guest export's NAME must be is held by
// tests/test_plugin_abi_freeze.cpp against plugins/sdk/abi-v1-surface.txt.
#define NETVIS_SDK_HOST_BRIDGE 1
#include "plugins/sdk/netvis_plugin.h"

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>

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

// --- nv_tensor_hdr_t: layout. The header asserts only the total size; two int32_t
// members swapped keep the size, so the members are pinned individually here. The
// header says the struct is documentation-only in v1 (no import takes it), but its
// layout is frozen so a later import that does pass it inherits one that never moved.
static_assert(sizeof(nv_tensor_hdr_t) == 24, "nv_tensor_hdr_t size");
static_assert(alignof(nv_tensor_hdr_t) == 8, "nv_tensor_hdr_t alignment");
static_assert(offsetof(nv_tensor_hdr_t, off) == 0, "nv_tensor_hdr_t::off");
static_assert(offsetof(nv_tensor_hdr_t, len) == 8, "nv_tensor_hdr_t::len");
static_assert(offsetof(nv_tensor_hdr_t, dtype) == 16, "nv_tensor_hdr_t::dtype");
static_assert(offsetof(nv_tensor_hdr_t, rank) == 20, "nv_tensor_hdr_t::rank");
static_assert(sizeof(nv_strid_t) == 4, "nv_strid_t is a 32-bit interned-string id");

// A trivial runtime case so the TU registers with doctest (the real coverage is the
// static_asserts above, checked at compile time).
TEST_CASE("SDK ABI bridge: enums/versions match the C++ contracts (compile-time)") {
  CHECK((int)NV_DT_UNKNOWN == 15);
  CHECK((int)NV_CAT_OTHER == 14);
  CHECK(NETVIS_OP_ABI_VERSION == 1u);
}
