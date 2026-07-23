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
#include "ir/IR.h"

namespace netvis::plugin::wasm {

// A WASM analysis pass: a .wasm image exporting `run() -> i32` (0 = ok). While it
// runs, it may call the host imports (see WasmHost.cpp) to read the model/report
// and emit metrics. Implements the frozen PassPlugin ABI.
class WasmPassPlugin final : public PassPlugin {
 public:
  WasmPassPlugin(std::string name, std::vector<uint8_t> image);
  std::string_view display_name() const override { return name_; }
  PassResult run(const ir::Model& model, uint32_t graph_index,
                 const CostReport& report) const override;
  uint32_t api_version() const override { return kPassPluginAbiVersion; }

 private:
  std::string name_;
  std::vector<uint8_t> image_;
};

}  // namespace netvis::plugin::wasm
