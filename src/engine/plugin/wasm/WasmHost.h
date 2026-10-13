// SPDX-License-Identifier: Apache-2.0
// engine/plugin/wasm/WasmHost.h — capability-scoped host API + WasmPassPlugin
// (v0.6.0 #10). This is the THESIS-ENFORCEMENT surface: the set of imports a WASM
// pass plugin may call is finite and contains NO function that returns a decoded
// tensor buffer — so a plugin structurally cannot exfiltrate weight bytes to the
// view. A pass sees IR STRUCTURE (node/op counts, the CostReport scalars) and can
// only EMIT named metrics back. Coarse entry point: run(whole model) once.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/CostModel.h"
#include "engine/plugin/PassPlugin.h"
#include "engine/plugin/wasm/WasmRuntime.h"
#include "ir/IR.h"

namespace netvis::plugin::wasm {

// A WASM analysis pass: a .wasm image exporting `run() -> i32` (0 = ok). While it
// runs, it may call the host imports (see WasmHost.cpp) to read the model/report
// and emit metrics. Implements the frozen PassPlugin ABI.
//
// ABI NEGOTIATION. A pass module may export `netvis_pass_abi_version` ( () -> i32 ).
// When it does, it must return kPassPluginAbiVersion: a module declaring any other
// version is refused (run() yields no metrics, and api_version() reports the
// declared version so the Registry drops it). When it does NOT, it is taken to be an
// ABI v1 pass: the pass facet shipped in v0.6.0 before this export existed, and the
// compatibility promise is that a plugin built against v1 keeps loading. That is the
// one place the pass facet differs from the op and parser facets, which require the
// export (they have no pre-negotiation plugins to protect). "Does not" means the
// module has no such export: one that is present but fails (traps, will not compile,
// runs out of the ABI probe's step budget) is refused, never taken for a v1 pass.
class WasmPassPlugin final : public PassPlugin {
 public:
  WasmPassPlugin(std::string name, std::vector<uint8_t> image);
  std::string_view display_name() const override { return name_; }
  PassResult run(const ir::Model& model, uint32_t graph_index,
                 const CostReport& report) const override;
  uint32_t api_version() const override {
    return probe_.reported_version(kPassPluginAbiVersion);
  }
  const WasmAbiProbe& probe() const { return probe_; }

 private:
  std::string name_;
  std::vector<uint8_t> image_;
  WasmAbiProbe probe_;
};

}  // namespace netvis::plugin::wasm
