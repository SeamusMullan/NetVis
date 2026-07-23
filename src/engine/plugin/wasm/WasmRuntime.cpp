// engine/plugin/wasm/WasmRuntime.cpp — wasm3-backed sandbox (v0.6.0 #10).
// See WasmRuntime.h. Compiles to no-ops without NETVIS_ENABLE_WASM.
#include "engine/plugin/wasm/WasmRuntime.h"

#if defined(NETVIS_ENABLE_WASM)

#include <cstring>

// wasm3's public + internal headers use C constructs (flexible array members)
// that trip our -Wpedantic -Werror when pulled into this C++ TU. Silence just for
// these includes; our own code below stays under the strict flags.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
extern "C" {
#include "wasm3.h"
#include "m3_env.h"
}
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace netvis::plugin::wasm {

// ---------------------------------------------------------------------------
// Fuel: a thread-local step budget. wasm3 calls the weak m3_Yield() on every
// function Call, and (via tools/patch_wasm3.py) on every loop back-edge, so this
// strong override traps ANY runaway module — recursion or tight loop. Returning a
// non-null M3Result from m3_Yield propagates as a trap that unwinds the interpreter
// cleanly; the host frees the module and disables the plugin.
// ---------------------------------------------------------------------------
namespace {
thread_local uint64_t g_fuel = 0;
thread_local bool g_fuel_armed = false;
const char* const kFuelExhausted = "[trap] netvis: fuel exhausted (runaway plugin)";
}  // namespace

}  // namespace netvis::plugin::wasm

// Strong override of wasm3's weak m3_Yield. MUST be extern "C" + global to shadow
// the library's weak symbol at link time.
extern "C" M3Result m3_Yield(void) {
  using namespace netvis::plugin::wasm;
  if (!g_fuel_armed) return m3Err_none;
  if (g_fuel == 0) return kFuelExhausted;
  --g_fuel;
  return m3Err_none;
}

namespace netvis::plugin::wasm {

// ---------------------------------------------------------------------------
struct WasmModule::Impl {
  IM3Runtime runtime = nullptr;
  IM3Module module = nullptr;   // owned by runtime once loaded
  std::vector<uint8_t> image;   // keep the bytes alive for the module's lifetime
  SandboxLimits lim;
};

WasmModule::~WasmModule() {
  if (impl_ && impl_->runtime) m3_FreeRuntime(impl_->runtime);
}
WasmModule::WasmModule(WasmModule&&) noexcept = default;
WasmModule& WasmModule::operator=(WasmModule&&) noexcept = default;

uint8_t* WasmModule::memory(uint32_t* out_size) {
  if (!impl_ || !impl_->runtime) { if (out_size) *out_size = 0; return nullptr; }
  uint32_t sz = 0;
  uint8_t* mem = m3_GetMemory(impl_->runtime, &sz, 0);
  if (out_size) *out_size = sz;
  return mem;
}

void* WasmModule::raw_module() const { return impl_ ? impl_->module : nullptr; }
void* WasmModule::raw_runtime() const { return impl_ ? impl_->runtime : nullptr; }

RunResult WasmModule::call_i32(const char* export_name, int32_t* out_ret) {
  RunResult r;
  if (!impl_ || !impl_->runtime) { r.status = RunStatus::LoadError; r.message = "no module"; return r; }

  IM3Function fn = nullptr;
  M3Result err = m3_FindFunction(&fn, impl_->runtime, export_name);
  if (err || !fn) { r.status = RunStatus::LoadError; r.message = err ? err : "export not found"; return r; }

  // Arm fuel for this call; disarm after so host-side code isn't metered.
  g_fuel = impl_->lim.max_steps;
  g_fuel_armed = true;
  err = m3_CallV(fn);
  g_fuel_armed = false;

  if (err) {
    if (std::strcmp(err, kFuelExhausted) == 0) {
      r.status = RunStatus::FuelExhausted; r.message = err; return r;
    }
    r.status = RunStatus::Trap; r.message = err; return r;
  }
  if (out_ret) {
    // Best-effort: read the i32 result if present.
    uint64_t ret = 0;
    m3_GetResultsV(fn, &ret);
    *out_ret = static_cast<int32_t>(ret);
  }
  r.status = RunStatus::Ok;
  return r;
}

// ---------------------------------------------------------------------------
struct WasmEngine::Impl {
  IM3Environment env = nullptr;
};

WasmEngine::WasmEngine() : impl_(std::make_unique<Impl>()) {
  impl_->env = m3_NewEnvironment();
}
WasmEngine::~WasmEngine() {
  if (impl_ && impl_->env) m3_FreeEnvironment(impl_->env);
}
WasmEngine& WasmEngine::instance() {
  static WasmEngine inst;
  return inst;
}
bool WasmEngine::enabled() const { return impl_ && impl_->env; }

WasmModule WasmEngine::load(const std::vector<uint8_t>& wasm, const SandboxLimits& lim,
                            void* host_ctx, RunResult* out_err) {
  WasmModule mod;
  RunResult err;
  if (!impl_ || !impl_->env) { err.status = RunStatus::LoadError; err.message = "no env"; if (out_err) *out_err = err; return mod; }
  if (wasm.size() < 8) { err.status = RunStatus::LoadError; err.message = "image too small"; if (out_err) *out_err = err; return mod; }

  auto impl = std::make_unique<WasmModule::Impl>();
  impl->lim = lim;
  impl->image = wasm;

  // Stack size is a native-stack budget for the interpreter; keep it bounded.
  impl->runtime = m3_NewRuntime(impl_->env, /*stackSizeInBytes=*/64 * 1024, host_ctx);
  if (!impl->runtime) { err.status = RunStatus::LoadError; err.message = "runtime alloc failed"; if (out_err) *out_err = err; return mod; }

  IM3Module m = nullptr;
  M3Result r = m3_ParseModule(impl_->env, &m, impl->image.data(),
                              static_cast<uint32_t>(impl->image.size()));
  if (r) { err.status = RunStatus::LoadError; err.message = r; m3_FreeRuntime(impl->runtime); if (out_err) *out_err = err; return mod; }

  r = m3_LoadModule(impl->runtime, m);
  if (r) { err.status = RunStatus::LoadError; err.message = r; m3_FreeRuntime(impl->runtime); if (out_err) *out_err = err; return mod; }
  impl->module = m;

  // Cap linear memory: a module that grows past this many bytes traps on grow.
  impl->runtime->memoryLimit =
      static_cast<uint32_t>(lim.max_memory_pages) * 65536u;

  mod.impl_ = std::move(impl);
  if (out_err) { err.status = RunStatus::Ok; *out_err = err; }
  return mod;
}

}  // namespace netvis::plugin::wasm

#else  // !NETVIS_ENABLE_WASM — safe no-op stubs

namespace netvis::plugin::wasm {
WasmModule::~WasmModule() = default;
WasmModule::WasmModule(WasmModule&&) noexcept = default;
WasmModule& WasmModule::operator=(WasmModule&&) noexcept = default;
uint8_t* WasmModule::memory(uint32_t* out_size) { if (out_size) *out_size = 0; return nullptr; }
void* WasmModule::raw_module() const { return nullptr; }
void* WasmModule::raw_runtime() const { return nullptr; }
RunResult WasmModule::call_i32(const char*, int32_t*) { return {RunStatus::Disabled, "WASM disabled"}; }

struct WasmEngine::Impl {};
WasmEngine::WasmEngine() = default;
WasmEngine::~WasmEngine() = default;
WasmEngine& WasmEngine::instance() { static WasmEngine e; return e; }
bool WasmEngine::enabled() const { return false; }
WasmModule WasmEngine::load(const std::vector<uint8_t>&, const SandboxLimits&, void*, RunResult* out_err) {
  if (out_err) *out_err = {RunStatus::Disabled, "WASM disabled"};
  return WasmModule{};
}
}  // namespace netvis::plugin::wasm

#endif
