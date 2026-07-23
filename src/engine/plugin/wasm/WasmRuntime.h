// engine/plugin/wasm/WasmRuntime.h — runtime-agnostic WASM sandbox wrapper
// (v0.6.0 Increment 3, #10).
//
// Wraps the wasm3 interpreter behind a thin, runtime-AGNOSTIC surface so wasmtime
// is a later drop-in (design §Q4). Enforces the sandbox: a linear-memory cap and a
// STEP/DEADLINE cap (via a strong m3_Yield override + the loop-backedge patch) so a
// hostile or buggy module that loops or over-allocates is trapped and the plugin is
// disabled — the app always survives. NO host import returns a decoded weight
// buffer (the zero-payload thesis is a property of the exposed import SET, enforced
// in WasmHost.cpp): a parser plugin can only DECLARE a tensor by offset+len.
//
// Compiles to safe no-ops when NETVIS_ENABLE_WASM is undefined.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace netvis::plugin::wasm {

// Sandbox limits. Defaults are deliberately small — a plugin is metadata-scale.
struct SandboxLimits {
  uint32_t max_memory_pages = 256;    // 256 * 64KiB = 16 MiB linear-memory cap
  uint64_t max_steps = 200'000'000;   // fuel: yield-check decrements; 0 => trap
};

// Outcome of a sandboxed run.
enum class RunStatus : uint8_t {
  Ok,           // ran to completion
  Trap,         // wasm trap (OOB, unreachable, div-0, ...) — plugin survives-safe
  FuelExhausted,// step cap hit (runaway loop/recursion killed)
  LoadError,    // module failed to parse/load/link
  Disabled,     // built without NETVIS_ENABLE_WASM
};

struct RunResult {
  RunStatus status = RunStatus::Disabled;
  std::string message;   // human-readable (trap text / link error)
};

// A loaded, sandboxed WASM module instance. One instance per run (fresh state, no
// cross-call leakage). Move-only. Construction loads+links; call() runs an export.
class WasmModule {
 public:
  ~WasmModule();
  WasmModule(WasmModule&&) noexcept;
  WasmModule& operator=(WasmModule&&) noexcept;
  WasmModule(const WasmModule&) = delete;
  WasmModule& operator=(const WasmModule&) = delete;

  bool loaded() const { return impl_ != nullptr; }

  // Call an exported function taking no args and returning an i32 status (0=ok).
  // Host imports the module may call are bound at load (see WasmHost). Enforces the
  // fuel/deadline cap; a runaway module returns FuelExhausted, never hangs.
  RunResult call_i32(const char* export_name, int32_t* out_ret);

  // Access the instance's linear memory (for marshalling); nullptr/0 if absent.
  uint8_t* memory(uint32_t* out_size);

  // Opaque native handles for the host-linking TU (WasmHost.cpp). Return nullptr
  // when built without NETVIS_ENABLE_WASM. Typed void* so this header stays free
  // of wasm3 types (runtime-agnostic surface).
  void* raw_module() const;
  void* raw_runtime() const;

 private:
  friend class WasmEngine;
  WasmModule() = default;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Process-wide engine: owns the wasm3 environment. Thread-compat: build one module
// per parse/pass call on the worker thread; do not share a WasmModule across threads.
class WasmEngine {
 public:
  static WasmEngine& instance();

  // Load a .wasm image (bytes copied). Returns a module with loaded()==false + a
  // message on parse/validate failure. `host_ctx` is threaded to host imports.
  WasmModule load(const std::vector<uint8_t>& wasm, const SandboxLimits& lim,
                  void* host_ctx, RunResult* out_err);

  bool enabled() const;   // false if built without NETVIS_ENABLE_WASM

 private:
  WasmEngine();
  ~WasmEngine();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace netvis::plugin::wasm
