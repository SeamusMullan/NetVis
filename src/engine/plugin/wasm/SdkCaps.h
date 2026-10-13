// SPDX-License-Identifier: Apache-2.0
// engine/plugin/wasm/SdkCaps.h — the host's marshalling and sandbox caps, taken from
// the shipped SDK header rather than copied from it.
//
// plugins/sdk/netvis_plugin.h is the one place a cap is written down (NV_MAX_RANK,
// NV_MAX_NODE_IO, NV_MAX_MEMORY_PAGES, ...). Before this header the host kept its
// own hand-copied constants next to each import, so lowering one of them (or the
// header's copy) changed what plugins were refused without any test noticing.
// Now every host cap below IS the header's macro, and the budgets that are
// deliberately smaller than the header's ceiling are static_assert'ed against it:
// editing a cap in the header changes the host in the same commit, and a host
// budget that would exceed the ceiling a plugin was promised does not compile.
//
// Included only by the WASM adapter TUs. NETVIS_SDK_HOST_BRIDGE keeps the guest-only
// part of the header (the import declarations, the bump allocator) out of C++.
#pragma once

#include <cstdint>

#ifndef NETVIS_SDK_HOST_BRIDGE
#define NETVIS_SDK_HOST_BRIDGE 1
#endif
#include "plugins/sdk/netvis_plugin.h"

#include "engine/plugin/wasm/WasmRuntime.h"

namespace netvis::plugin::wasm::caps {

// Parser facet marshalling caps (every guest-supplied count is bounded by these).
inline constexpr int64_t kSniffHead = NV_SNIFF_HEAD;
inline constexpr int64_t kSniffTail = NV_SNIFF_TAIL;
inline constexpr int32_t kReadChunkCap = NV_READ_CHUNK_CAP;
inline constexpr int32_t kMaxRank = NV_MAX_RANK;
inline constexpr int32_t kMaxNodeIO = NV_MAX_NODE_IO;
inline constexpr int32_t kMaxAttrLen = NV_MAX_ATTR_LEN;
inline constexpr int32_t kMaxInternLen = NV_MAX_INTERN_LEN;
inline constexpr int32_t kErrMsgMax = NV_ERR_MSG_MAX;

// The ceiling on a sandbox budget: SandboxLimits' defaults ARE the SDK's promise.
static_assert(SandboxLimits{}.max_memory_pages == NV_MAX_MEMORY_PAGES,
              "SandboxLimits::max_memory_pages drifted from NV_MAX_MEMORY_PAGES");
static_assert(SandboxLimits{}.max_steps == NV_MAX_STEPS,
              "SandboxLimits::max_steps drifted from NV_MAX_STEPS");

// A facet may be granted LESS than the ceiling, never more.
inline constexpr uint32_t kCeilingPages = NV_MAX_MEMORY_PAGES;
inline constexpr uint64_t kCeilingSteps = NV_MAX_STEPS;

}  // namespace netvis::plugin::wasm::caps
