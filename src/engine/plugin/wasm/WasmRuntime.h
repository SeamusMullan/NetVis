// SPDX-License-Identifier: Apache-2.0
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
#include <mutex>
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
  // LoadError only: the named export does not exist at all (as opposed to existing
  // with the wrong signature, or failing to compile). Lets a caller treat "absent"
  // differently from "present but broken" - the ABI negotiation does (see below).
  bool export_missing = false;
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

  // Function imports the module declares that NO host function answers, as
  // "module.name" strings. Call after the facet's link step. m3_LinkRawFunctionEx
  // leaves an import unbound in two cases that the link call itself hides from the
  // caller: the host has no function of that name (a plugin built against a newer
  // SDK, or a typo), and the host has one but the guest declared a different
  // signature (wasm3 reports "function signature mismatch" and links nothing). In
  // both a call to the import fails later, and only for the entry point that
  // reaches it - so the module looks fine until it silently abstains. Naming the
  // unbound imports up front lets a loader refuse the module visibly instead.
  // Empty when everything the module imports is bound (or built without WASM).
  std::vector<std::string> unresolved_imports() const;

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

// What a module says about the ABI it was built against: the result of calling its
// `netvis_<facet>_abi_version` export. Pure data; the version policy lives with
// each facet (op/parser: a missing export is a refusal; pass: a missing export is
// a pre-negotiation v1 plugin, see WasmHost.cpp).
struct AbiDeclaration {
  bool declared = false;         // the export exists and returned
  bool export_missing = false;   // the export is absent (vs. present but failing)
  uint32_t version = 0;          // the declared version; 0 when !declared or negative
  std::string message;           // why it was not read, when !declared
};

inline AbiDeclaration read_abi_declaration(WasmModule& mod, const char* export_name) {
  AbiDeclaration d;
  int32_t v = -1;
  RunResult r = mod.call_i32(export_name, &v);
  if (r.status == RunStatus::Ok) {
    d.declared = true;
    d.version = v < 0 ? 0u : static_cast<uint32_t>(v);
  } else {
    d.export_missing = r.export_missing;
    d.message = r.message;
  }
  return d;
}

// The outcome of loading a plugin module once and asking whether THIS host can
// honour it: it loads, every import it declares is bound, and it declares the ABI
// version the host speaks. One of these is taken when an adapter is built, which is
// what lets the adapters' api_version() report what the module declared (so the
// Registry's version check is a real gate for WASM plugins, not a constant) and
// lets a loader refuse a module before anything is registered.
struct WasmAbiProbe {
  bool loaded = false;                         // parsed, instantiated and linked
  AbiDeclaration abi;                          // what it declared
  std::vector<std::string> unresolved_imports; // imports no host function answers
  std::string message;                         // load failure text, when !loaded

  bool compatible(uint32_t host_version) const {
    return loaded && abi.declared && abi.version == host_version &&
           unresolved_imports.empty();
  }
  // The version an adapter reports through api_version(): what the module declared
  // when that differs from the host's, and 0 ("not speakable") when it declared the
  // host's version but cannot run here (did not load, or imports something this host
  // does not provide).
  uint32_t reported_version(uint32_t host_version) const {
    if (compatible(host_version)) return host_version;
    if (loaded && abi.declared && abi.version != host_version) return abi.version;
    return 0;
  }
  // Why a loader refuses the module; empty when compatible.
  std::string refusal(uint32_t host_version, const char* abi_export) const {
    if (compatible(host_version)) return {};
    if (!loaded) return "module failed to load: " + message;
    if (!abi.declared) {
      return std::string("module does not declare an ABI version (no usable `") +
             abi_export + "` export); this NetVis speaks ABI v" +
             std::to_string(host_version);
    }
    if (abi.version != host_version) {
      return "module declares ABI v" + std::to_string(abi.version) +
             "; this NetVis speaks ABI v" + std::to_string(host_version);
    }
    std::string list;
    for (const std::string& u : unresolved_imports) {
      if (!list.empty()) list += ", ";
      list += u;
    }
    return "module imports host functions this NetVis does not provide (or declares "
           "them with a different signature): " + list;
  }
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

  // Process-wide lock serializing ALL load/link/call access (design §0.3): wasm3's
  // IM3Environment is shared mutable state (m3_ParseModule mutates env->funcTypes)
  // and fuel is a thread-local, so a WASM op-handler resolved on a worker thread and
  // a parser sniff on another must not race. Callers hold this across the whole
  // load()+link+call_i32() sequence. Plain std::mutex — portable (inv 2, no
  // std::atomic<shared_ptr>). Recursive NOT needed: no nested sandbox entry.
  std::mutex& lock() { return call_mutex_; }

 private:
  WasmEngine();
  ~WasmEngine();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::mutex call_mutex_;   // see lock()
};

}  // namespace netvis::plugin::wasm
