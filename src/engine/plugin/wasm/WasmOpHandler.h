// SPDX-License-Identifier: Apache-2.0
// engine/plugin/wasm/WasmOpHandler.h — WASM OpHandler adapter (Increment A, #10).
//
// The fourth OpHandler implementor (alongside Builtin/Declarative). It answers
// category/color/flops/infer_shape over a read-only OpContext by running a
// sandboxed .wasm module exporting the "netvis_op_*" entry points and reading the
// OpContext through the frozen "netvis_op" host imports (plugins/sdk/netvis_plugin.h).
//
// THREAD MODEL (design §A.1, critique-fix): a WASM handler is NEVER entered from
// resolve_category on the render thread. All sandbox work happens on the worker
// cost/shape pass and is memoized per op-type by the LEAD (see "LEAD MUST WIRE"
// below). Every WasmEngine load/call is serialized by WasmEngine::lock() (§0.3).
//
// LEAD MUST WIRE (§A.6 — deliberately NOT done here to keep TUs disjoint):
//   - CostModel.cpp / ShapeInferenceExt.cpp: on the worker pass, resolve each
//     Origin::Wasm handler ONCE per op-type and cache category/color/flops into the
//     per-op-type style/cost state the UI reads (call category()/flops()/... here).
//   - Registry.cpp resolve_category: for Origin::Wasm return the CACHED scalar,
//     never enter the sandbox from the render thread.
//   - App.cpp: call load_wasm_op_plugin() behind the Increment-C enable gate.
//   - PluginsPanel.cpp: surface WasmOpDiag (missing-export / budget-tripped / abstain).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/OpCategory.h"
#include "engine/plugin/OpHandler.h"
#include "engine/plugin/wasm/WasmRuntime.h"

namespace netvis::plugin::wasm {

// Clamp a guest-supplied raw category int to a valid OpCategory (§A.3 THE CLAMP).
// <0 or >Other -> Other, so no invalid enum ever reaches `1u<<cat` / kPalette[].
OpCategory clamp_category(int32_t raw);

// Per-handler load/abstain diagnostics for the Plugins panel (#11 / §A.2).
struct WasmOpDiag {
  bool   loaded = false;         // module loaded (and linked)
  bool   abi_mismatch = false;   // declares a different ABI version, or none
  bool   unresolved_import = false;  // imports a host function this host lacks
  bool   missing_category = false;  // required export absent
  std::string message;           // human-readable load/abi diagnostic
};

// A WASM op handler backed by an immutable .wasm image. Holds the image bytes and
// constructs a fresh WasmModule per invocation under WasmEngine::lock() (no live
// module shared across threads). category() is required; color/flops/infer_shape
// default to honest-unknown on abstain/trap/fuel/load-error.
//
// ABI NEGOTIATION. Building a handler loads the module once and asks it which ABI
// it was built for (WasmAbiProbe), so api_version() reports what the MODULE
// declared rather than echoing the host's constant. That makes the Registry's
// version check on register_op_handler a real gate for a WASM handler: a module
// that declares another version, declares none, or imports host functions this host
// does not provide is dropped at registration instead of replacing a built-in
// result with honest-unknown. Every invocation re-checks the same things, so a
// handler that somehow reached the table by another route still cannot answer.
class WasmOpHandler final : public OpHandler {
 public:
  WasmOpHandler(std::string plugin_name, std::shared_ptr<const std::vector<uint8_t>> image);
  // For a caller that already holds the probe (the loader, which builds one handler
  // per declared op from one image), so the module is probed once, not once per op.
  WasmOpHandler(std::string plugin_name, std::shared_ptr<const std::vector<uint8_t>> image,
                const WasmAbiProbe& probe);

  OpCategory  category(const OpContext& ctx) const override;
  ColorResult color(const OpContext& ctx) const override;
  FlopResult  flops(const OpContext& ctx) const override;
  ShapeResult infer_shape(const OpContext& ctx) const override;
  uint32_t    api_version() const override { return abi_version_; }

  const WasmOpDiag& diag() const { return diag_; }

 private:
  void adopt_probe(const WasmAbiProbe& probe);

  std::string plugin_name_;
  std::shared_ptr<const std::vector<uint8_t>> image_;
  uint32_t abi_version_ = 0;    // what the module declared; see api_version()
  mutable WasmOpDiag diag_;
};

// Load `image` once and report whether this host can honour it as an op handler
// (loads, every import is bound, declares kOpHandlerAbiVersion). Takes the engine
// lock. With WASM disabled, returns a probe that is not compatible.
WasmAbiProbe probe_op_module(const std::vector<uint8_t>& image);

// A discovered op-plugin op entry (from the sidecar plugin.json "ops" list).
struct WasmOpDecl {
  std::string op_key;     // normalized op key this handler answers for
  std::string domain;     // "" or e.g. "com.example"
  bool override_builtin = false;
};

// Load a WASM op-plugin sidecar (plugin.json + its .wasm) and, for each declared
// op, register a WasmOpHandler into the Registry (Origin::Wasm). Returns a
// diagnostic string per the panel. Registration is the caller's gate concern
// (Increment C): this is only invoked for ENABLED plugins. Defined in the .cpp;
// LEAD calls it from App.cpp behind the enable gate.
std::string load_wasm_op_plugin(const std::string& plugin_json_path);

}  // namespace netvis::plugin::wasm
