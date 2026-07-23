// engine/plugin/wasm/WasmHost.cpp — capability host API + WasmPassPlugin run path
// (v0.6.0 #10). The imports bound here are the ENTIRE vocabulary a WASM pass gets.
// THESIS: none of them returns a tensor/weight buffer — a pass reads structure
// (counts + CostReport scalars) and emits named metrics. That is enforced by the
// SET, not a convention.
#include "engine/plugin/wasm/WasmHost.h"

#include "engine/plugin/wasm/WasmRuntime.h"

#if defined(NETVIS_ENABLE_WASM)

#include <cstring>
#include <string>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4200)  // nonstandard: zero-sized array in struct/union
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
extern "C" {
#include "wasm3.h"
#include "m3_env.h"
}
#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

// The m3ApiRawFunction macro expands to a fixed (runtime, _ctx, _sp, _mem)
// signature; host fns that don't touch memory leave `runtime`/`_mem` unused. That
// is intentional (the ABI is fixed), so silence MSVC C4100 for this TU — the
// GCC/clang build already passes via -Wno-error=unused-parameter.
#if defined(_MSC_VER)
#pragma warning(disable : 4100)  // unreferenced formal parameter
#endif

namespace netvis::plugin::wasm {

namespace {

// The host context threaded to every import via IM3ImportContext::userdata. Borrows
// the model/report for the duration of one run(); collects emitted metrics.
struct PassHostCtx {
  const ir::Model* model = nullptr;
  uint32_t graph_index = 0;
  const CostReport* report = nullptr;
  PassResult* out = nullptr;
};

PassHostCtx* ctx_of(IM3ImportContext c) {
  return static_cast<PassHostCtx*>(c->userdata);
}

// --- capability imports (module "netvis") ---------------------------------
// i64 host_node_count()
m3ApiRawFunction(host_node_count) {
  m3ApiReturnType(int64_t);
  PassHostCtx* h = ctx_of(_ctx);
  int64_t n = 0;
  if (h && h->model && h->graph_index < h->model->graphs.size())
    n = static_cast<int64_t>(h->model->graphs[h->graph_index].nodes.size());
  m3ApiReturn(n);
}

// f64 host_total_flops()  — from the CostReport scalar (already payload-free)
m3ApiRawFunction(host_total_flops) {
  m3ApiReturnType(double);
  PassHostCtx* h = ctx_of(_ctx);
  double v = (h && h->report) ? static_cast<double>(h->report->total_flops) : 0.0;
  m3ApiReturn(v);
}

// f64 host_total_params()
m3ApiRawFunction(host_total_params) {
  m3ApiReturnType(double);
  PassHostCtx* h = ctx_of(_ctx);
  double v = (h && h->report) ? static_cast<double>(h->report->total_params) : 0.0;
  m3ApiReturn(v);
}

// void host_emit_metric(const char* name_ptr, i32 name_len, f64 value, i32 known)
// The one OUTPUT path: a plugin declares a named scalar. It CANNOT hand back bytes.
m3ApiRawFunction(host_emit_metric) {
  m3ApiGetArgMem(const char*, name);
  m3ApiGetArg(int32_t, name_len);
  m3ApiGetArg(double, value);
  m3ApiGetArg(int32_t, known);
  PassHostCtx* h = ctx_of(_ctx);
  if (h && h->out && name && name_len > 0 && name_len < 256) {
    m3ApiCheckMem(name, static_cast<size_t>(name_len));
    Metric m;
    m.name.assign(name, static_cast<size_t>(name_len));
    m.value = value;
    m.known = (known != 0);
    m.unit = "";
    h->out->metrics.push_back(std::move(m));
  }
  m3ApiSuccess();
}

// Link the capability set into a freshly loaded module. Missing-import links are
// tolerated (a plugin that doesn't use one still loads).
void link_capabilities(IM3Module mod, PassHostCtx* ctx) {
  const char* ns = "netvis";
  m3_LinkRawFunctionEx(mod, ns, "host_node_count",  "I()",     &host_node_count,  ctx);
  m3_LinkRawFunctionEx(mod, ns, "host_total_flops", "F()",     &host_total_flops, ctx);
  m3_LinkRawFunctionEx(mod, ns, "host_total_params","F()",     &host_total_params,ctx);
  m3_LinkRawFunctionEx(mod, ns, "host_emit_metric", "v(*iFi)", &host_emit_metric, ctx);
}

}  // namespace

WasmPassPlugin::WasmPassPlugin(std::string name, std::vector<uint8_t> image)
    : name_(std::move(name)), image_(std::move(image)) {}

PassResult WasmPassPlugin::run(const ir::Model& model, uint32_t graph_index,
                               const CostReport& report) const {
  PassResult out;
  if (!WasmEngine::instance().enabled()) return out;

  PassHostCtx hc;
  hc.model = &model;
  hc.graph_index = graph_index;
  hc.report = &report;
  hc.out = &out;

  SandboxLimits lim;
  RunResult lerr;
  WasmModule mod = WasmEngine::instance().load(image_, lim, &hc, &lerr);
  if (!mod.loaded()) return out;   // honest: a bad plugin yields no metrics

  // Link the capability imports before the entry call.
  link_capabilities(static_cast<IM3Module>(mod.raw_module()), &hc);

  int32_t ret = 0;
  RunResult rr = mod.call_i32("run", &ret);
  (void)rr;  // trap/fuel-exhaustion simply yields whatever metrics were emitted.
  return out;
}

}  // namespace netvis::plugin::wasm

#else  // !NETVIS_ENABLE_WASM

namespace netvis::plugin::wasm {
WasmPassPlugin::WasmPassPlugin(std::string name, std::vector<uint8_t> image)
    : name_(std::move(name)), image_(std::move(image)) {}
PassResult WasmPassPlugin::run(const ir::Model&, uint32_t, const CostReport&) const {
  return {};
}
}  // namespace netvis::plugin::wasm

#endif
