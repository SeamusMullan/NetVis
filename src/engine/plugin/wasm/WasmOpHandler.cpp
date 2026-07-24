// engine/plugin/wasm/WasmOpHandler.cpp — WASM OpHandler adapter host side (#10).
// See WasmOpHandler.h. The "netvis_op" import set lets a sandboxed module READ the
// current OpContext (counts/shapes/dtypes/attrs/initializer records) and PUSH a
// verdict (category/flops/shape/color) via out-imports; NO import returns a weight
// buffer (inv 3). Compiles to safe stubs without NETVIS_ENABLE_WASM.
#include "engine/plugin/wasm/WasmOpHandler.h"

#include "engine/plugin/wasm/WasmRuntime.h"

namespace netvis::plugin::wasm {

OpCategory clamp_category(int32_t raw) {
  if (raw < 0 || raw > static_cast<int32_t>(OpCategory::Other)) return OpCategory::Other;
  return static_cast<OpCategory>(raw);
}

}  // namespace netvis::plugin::wasm

#if defined(NETVIS_ENABLE_WASM)

#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/plugin/Registry.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4200)
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

// m3ApiRawFunction's fixed (runtime,_ctx,_sp,_mem) signature leaves some params
// unused per-fn; the ABI is fixed, so silence MSVC C4100 for this TU (as WasmHost).
#if defined(_MSC_VER)
#pragma warning(disable : 4100)
#endif

namespace netvis::plugin::wasm {

namespace {

// Op-handler sandbox limits: ~100x below the pass default (§A.4). A per-node
// invocation is a footgun; small caps make trap/fuel-exhaustion cheap + safe.
constexpr uint32_t kOpMaxPages = 64;         // 4 MiB
constexpr uint64_t kOpMaxSteps = 2'000'000;

// Threaded to every op import via IM3ImportContext::userdata. Borrows the ctx for
// one invocation; captures the guest's pushed results.
struct OpHostCtx {
  const OpContext* ctx = nullptr;
  // captured verdict (host reads after the entry call returns):
  bool     have_category = false; int32_t raw_category = 0;
  bool     have_flops = false;    uint64_t flops = 0; bool flops_known = false;
  bool     have_color = false;    uint32_t rgba = 0;
  ShapeResult shape;
};

OpHostCtx* octx(IM3ImportContext c) { return static_cast<OpHostCtx*>(c->userdata); }

// Copy up to `cap` bytes of `src` into guest memory at `dst`; return bytes written
// (or -1 on a bounds failure). Bounds-checked via m3ApiCheckMem on the CLAMPED len.
// (Helper body is inlined per-import because m3ApiCheckMem needs the macro locals.)

// --- READ imports -----------------------------------------------------------
// i32 op_input_count()
m3ApiRawFunction(op_input_count) {
  m3ApiReturnType(int32_t);
  OpHostCtx* h = octx(_ctx);
  int32_t n = (h && h->ctx) ? static_cast<int32_t>(h->ctx->input_count()) : 0;
  m3ApiReturn(n);
}
// i32 op_output_count()
m3ApiRawFunction(op_output_count) {
  m3ApiReturnType(int32_t);
  OpHostCtx* h = octx(_ctx);
  int32_t n = (h && h->ctx) ? static_cast<int32_t>(h->ctx->output_count()) : 0;
  m3ApiReturn(n);
}
// i32 op_default_category()
m3ApiRawFunction(op_default_category) {
  m3ApiReturnType(int32_t);
  OpHostCtx* h = octx(_ctx);
  int32_t c = (h && h->ctx) ? static_cast<int32_t>(h->ctx->default_category())
                            : static_cast<int32_t>(OpCategory::Other);
  m3ApiReturn(c);
}
// i32 op_type(i32 dst, i32 cap) -> bytes written of the normalized op_type
m3ApiRawFunction(op_type_name) {
  m3ApiReturnType(int32_t);
  m3ApiGetArgMem(char*, dst);
  m3ApiGetArg(int32_t, cap);
  OpHostCtx* h = octx(_ctx);
  int32_t written = 0;
  if (h && h->ctx && cap > 0) {
    std::string_view t = h->ctx->op_type();
    int32_t n = static_cast<int32_t>(t.size());
    if (n > cap) n = cap;
    if (n > 0) {
      m3ApiCheckMem(dst, static_cast<size_t>(n));
      std::memcpy(dst, t.data(), static_cast<size_t>(n));
    }
    written = n;
  }
  m3ApiReturn(written);
}
// i32 op_input_rank(i32 slot) / op_output_rank(i32 slot) -> rank, -1 if unresolved
m3ApiRawFunction(op_input_rank) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, slot);
  OpHostCtx* h = octx(_ctx);
  int32_t r = -1;
  if (h && h->ctx && slot >= 0) {
    const Shape* s = h->ctx->input_shape(static_cast<uint32_t>(slot));
    if (s) r = static_cast<int32_t>(s->size());
  }
  m3ApiReturn(r);
}
m3ApiRawFunction(op_output_rank) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, slot);
  OpHostCtx* h = octx(_ctx);
  int32_t r = -1;
  if (h && h->ctx && slot >= 0) {
    const Shape* s = h->ctx->output_shape(static_cast<uint32_t>(slot));
    if (s) r = static_cast<int32_t>(s->size());
  }
  m3ApiReturn(r);
}
// Shared dims writer: writes i64 dims of `s` into guest [dst,cap); count written.
static int32_t write_dims(const Shape* s, int64_t* dst, int32_t cap,
                          IM3Runtime runtime, void* _mem) {
  if (!s) return -1;
  int32_t n = static_cast<int32_t>(s->size());
  if (n > cap) n = cap;
  if (n > 0) {
    // Bounds-check via the macro (needs runtime/_mem in scope).
    if (m3ApiIsNullPtr(dst) ||
        ((uint64_t)(uintptr_t)(dst) + static_cast<uint64_t>(n) * 8u) >
            ((uint64_t)(uintptr_t)(_mem) + m3_GetMemorySize(runtime)))
      return -1;
    for (int32_t i = 0; i < n; ++i) dst[i] = (*s)[static_cast<uint32_t>(i)];
  }
  return n;
}
// i32 op_input_dims(i32 slot, i32 dst, i32 cap)
m3ApiRawFunction(op_input_dims) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, slot);
  m3ApiGetArgMem(int64_t*, dst);
  m3ApiGetArg(int32_t, cap);
  OpHostCtx* h = octx(_ctx);
  int32_t w = (h && h->ctx && slot >= 0 && cap > 0)
      ? write_dims(h->ctx->input_shape(static_cast<uint32_t>(slot)), dst, cap, runtime, _mem)
      : -1;
  m3ApiReturn(w);
}
m3ApiRawFunction(op_output_dims) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, slot);
  m3ApiGetArgMem(int64_t*, dst);
  m3ApiGetArg(int32_t, cap);
  OpHostCtx* h = octx(_ctx);
  int32_t w = (h && h->ctx && slot >= 0 && cap > 0)
      ? write_dims(h->ctx->output_shape(static_cast<uint32_t>(slot)), dst, cap, runtime, _mem)
      : -1;
  m3ApiReturn(w);
}
// i32 op_input_dtype / op_output_dtype -> ir::DType (Unknown=15)
m3ApiRawFunction(op_input_dtype) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, slot);
  OpHostCtx* h = octx(_ctx);
  int32_t d = static_cast<int32_t>(ir::DType::Unknown);
  if (h && h->ctx && slot >= 0) d = static_cast<int32_t>(h->ctx->input_dtype(static_cast<uint32_t>(slot)));
  m3ApiReturn(d);
}
m3ApiRawFunction(op_output_dtype) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, slot);
  OpHostCtx* h = octx(_ctx);
  int32_t d = static_cast<int32_t>(ir::DType::Unknown);
  if (h && h->ctx && slot >= 0) d = static_cast<int32_t>(h->ctx->output_dtype(static_cast<uint32_t>(slot)));
  m3ApiReturn(d);
}
// i32 op_input_is_initializer(i32 slot)
m3ApiRawFunction(op_input_is_initializer) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, slot);
  OpHostCtx* h = octx(_ctx);
  int32_t b = (h && h->ctx && slot >= 0 &&
               h->ctx->input_is_initializer(static_cast<uint32_t>(slot))) ? 1 : 0;
  m3ApiReturn(b);
}
// I64 op_input_init_elem_count(i32 slot)
m3ApiRawFunction(op_input_init_elem_count) {
  m3ApiReturnType(int64_t);
  m3ApiGetArg(int32_t, slot);
  OpHostCtx* h = octx(_ctx);
  int64_t v = 0;
  if (h && h->ctx && slot >= 0) {
    auto ir = h->ctx->input_initializer(static_cast<uint32_t>(slot));
    if (ir) v = ir->elem_count;
  }
  m3ApiReturn(v);
}
// I64 op_input_init_byte_len(i32 slot)  (Q4/Q8 truth)
m3ApiRawFunction(op_input_init_byte_len) {
  m3ApiReturnType(int64_t);
  m3ApiGetArg(int32_t, slot);
  OpHostCtx* h = octx(_ctx);
  int64_t v = 0;
  if (h && h->ctx && slot >= 0) {
    auto ir = h->ctx->input_initializer(static_cast<uint32_t>(slot));
    if (ir) v = static_cast<int64_t>(ir->byte_len);
  }
  m3ApiReturn(v);
}
// i32 op_input_init_dtype(i32 slot)
m3ApiRawFunction(op_input_init_dtype) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, slot);
  OpHostCtx* h = octx(_ctx);
  int32_t d = static_cast<int32_t>(ir::DType::Unknown);
  if (h && h->ctx && slot >= 0) {
    auto ir = h->ctx->input_initializer(static_cast<uint32_t>(slot));
    if (ir) d = static_cast<int32_t>(ir->dtype);
  }
  m3ApiReturn(d);
}
// i32 op_has_attr(i32 name_ptr, i32 name_len)
m3ApiRawFunction(op_has_attr) {
  m3ApiReturnType(int32_t);
  m3ApiGetArgMem(const char*, name);
  m3ApiGetArg(int32_t, name_len);
  OpHostCtx* h = octx(_ctx);
  int32_t b = 0;
  if (h && h->ctx && name && name_len > 0 && name_len < 256) {
    m3ApiCheckMem(name, static_cast<size_t>(name_len));
    b = h->ctx->has_attr(std::string_view(name, static_cast<size_t>(name_len))) ? 1 : 0;
  }
  m3ApiReturn(b);
}
// I64 op_attr_int(i32 name_ptr, i32 name_len, I64 def)
m3ApiRawFunction(op_attr_int) {
  m3ApiReturnType(int64_t);
  m3ApiGetArgMem(const char*, name);
  m3ApiGetArg(int32_t, name_len);
  m3ApiGetArg(int64_t, def);
  OpHostCtx* h = octx(_ctx);
  int64_t v = def;
  if (h && h->ctx && name && name_len > 0 && name_len < 256) {
    m3ApiCheckMem(name, static_cast<size_t>(name_len));
    v = h->ctx->attr_int(std::string_view(name, static_cast<size_t>(name_len)), def);
  }
  m3ApiReturn(v);
}
// f64 op_attr_float(i32 name_ptr, i32 name_len, i32 known_ptr)
m3ApiRawFunction(op_attr_float) {
  m3ApiReturnType(double);
  m3ApiGetArgMem(const char*, name);
  m3ApiGetArg(int32_t, name_len);
  m3ApiGetArgMem(int32_t*, known);
  OpHostCtx* h = octx(_ctx);
  double v = 0.0;
  int32_t k = 0;
  if (h && h->ctx && name && name_len > 0 && name_len < 256) {
    m3ApiCheckMem(name, static_cast<size_t>(name_len));
    auto f = h->ctx->attr_float(std::string_view(name, static_cast<size_t>(name_len)));
    if (f) { v = *f; k = 1; }
  }
  if (known) { m3ApiCheckMem(known, sizeof(int32_t)); *known = k; }
  m3ApiReturn(v);
}
// i32 op_attr_string(i32 name_ptr, i32 name_len, i32 dst, i32 cap)
m3ApiRawFunction(op_attr_string) {
  m3ApiReturnType(int32_t);
  m3ApiGetArgMem(const char*, name);
  m3ApiGetArg(int32_t, name_len);
  m3ApiGetArgMem(char*, dst);
  m3ApiGetArg(int32_t, cap);
  OpHostCtx* h = octx(_ctx);
  int32_t written = 0;
  if (h && h->ctx && name && name_len > 0 && name_len < 256 && cap > 0) {
    m3ApiCheckMem(name, static_cast<size_t>(name_len));
    std::string_view v = h->ctx->attr_string(
        std::string_view(name, static_cast<size_t>(name_len)), std::string_view{});
    int32_t n = static_cast<int32_t>(v.size());
    if (n > cap) n = cap;
    if (n > 0) { m3ApiCheckMem(dst, static_cast<size_t>(n)); std::memcpy(dst, v.data(), static_cast<size_t>(n)); }
    written = n;
  }
  m3ApiReturn(written);
}
// i32 op_attr_ints(i32 name_ptr, i32 name_len, i32 dst, i32 cap) -> i64[] count
m3ApiRawFunction(op_attr_ints) {
  m3ApiReturnType(int32_t);
  m3ApiGetArgMem(const char*, name);
  m3ApiGetArg(int32_t, name_len);
  m3ApiGetArgMem(int64_t*, dst);
  m3ApiGetArg(int32_t, cap);
  OpHostCtx* h = octx(_ctx);
  int32_t written = -1;
  if (h && h->ctx && name && name_len > 0 && name_len < 256 && cap > 0) {
    m3ApiCheckMem(name, static_cast<size_t>(name_len));
    const std::vector<int64_t>* v =
        h->ctx->attr_ints(std::string_view(name, static_cast<size_t>(name_len)));
    if (v) {
      int32_t n = static_cast<int32_t>(v->size());
      if (n > cap) n = cap;
      if (n > 0) {
        m3ApiCheckMem(dst, static_cast<size_t>(n) * sizeof(int64_t));
        std::memcpy(dst, v->data(), static_cast<size_t>(n) * sizeof(int64_t));
      }
      written = n;
    }
  }
  m3ApiReturn(written);
}
// i32 op_input_const_ints(i32 slot, i32 dst, i32 cap) -> the ONE guarded shape-
// tensor accessor; -1 when mmap_base_ null (cost driver always gets -1).
m3ApiRawFunction(op_input_const_ints) {
  m3ApiReturnType(int32_t);
  m3ApiGetArg(int32_t, slot);
  m3ApiGetArgMem(int64_t*, dst);
  m3ApiGetArg(int32_t, cap);
  OpHostCtx* h = octx(_ctx);
  int32_t written = -1;
  if (h && h->ctx && slot >= 0 && cap > 0) {
    const std::vector<int64_t>* v = h->ctx->input_const_ints(static_cast<uint32_t>(slot));
    if (v) {
      int32_t n = static_cast<int32_t>(v->size());
      if (n > cap) n = cap;
      if (n > 0) {
        m3ApiCheckMem(dst, static_cast<size_t>(n) * sizeof(int64_t));
        std::memcpy(dst, v->data(), static_cast<size_t>(n) * sizeof(int64_t));
      }
      written = n;
    }
  }
  m3ApiReturn(written);
}

// --- RESULT out-imports (guest pushes verdict; host records into OpHostCtx) ---
// void op_set_category(i32 cat)  — recorded RAW; clamped host-side after the call
m3ApiRawFunction(op_set_category) {
  m3ApiGetArg(int32_t, cat);
  OpHostCtx* h = octx(_ctx);
  if (h) { h->have_category = true; h->raw_category = cat; }
  m3ApiSuccess();
}
// void op_set_flops(I64 value, i32 known)
m3ApiRawFunction(op_set_flops) {
  m3ApiGetArg(int64_t, value);
  m3ApiGetArg(int32_t, known);
  OpHostCtx* h = octx(_ctx);
  if (h) {
    h->have_flops = true;
    h->flops = value < 0 ? 0 : static_cast<uint64_t>(value);
    h->flops_known = (known != 0 && value >= 0);
  }
  m3ApiSuccess();
}
// void op_set_output_shape(i32 slot, i32 dims_ptr, i32 rank, i32 dtype)
m3ApiRawFunction(op_set_output_shape) {
  m3ApiGetArg(int32_t, slot);
  m3ApiGetArgMem(int64_t*, dims);
  m3ApiGetArg(int32_t, rank);
  m3ApiGetArg(int32_t, dtype);
  OpHostCtx* h = octx(_ctx);
  if (h && slot >= 0 && rank >= 0) {
    // Clamp rank to NV_MAX_RANK(8); bounds-check the CLAMPED byte count (§A.2).
    int32_t r = rank > 8 ? 8 : rank;
    ShapeResult::Out out;
    out.slot = static_cast<uint32_t>(slot);
    if (dtype >= 0 && dtype <= static_cast<int32_t>(ir::DType::Unknown))
      out.dtype = static_cast<ir::DType>(dtype);
    if (r > 0 && dims) {
      if (!m3ApiIsNullPtr(dims) &&
          ((uint64_t)(uintptr_t)(dims) + static_cast<uint64_t>(r) * 8u) <=
              ((uint64_t)(uintptr_t)(_mem) + m3_GetMemorySize(runtime))) {
        for (int32_t i = 0; i < r; ++i) {
          int64_t d = dims[i];
          out.shape.push_back((d < 1 && d != -1) ? -1 : d);  // <1 && !=-1 -> dynamic
        }
      }
    }
    h->shape.outputs.push_back(std::move(out));
  }
  m3ApiSuccess();
}
// void op_set_color(i32 rgba)
m3ApiRawFunction(op_set_color) {
  m3ApiGetArg(int32_t, rgba);
  OpHostCtx* h = octx(_ctx);
  if (h) { h->have_color = true; h->rgba = static_cast<uint32_t>(rgba); }
  m3ApiSuccess();
}

void link_op_capabilities(IM3Module mod, OpHostCtx* ctx) {
  const char* ns = "netvis_op";
  auto L = [&](const char* nm, const char* sig, M3RawCall fn) {
    m3_LinkRawFunctionEx(mod, ns, nm, sig, fn, ctx);
  };
  L("op_input_count", "i()", &op_input_count);
  L("op_output_count", "i()", &op_output_count);
  L("op_default_category", "i()", &op_default_category);
  L("op_type", "i(*i)", &op_type_name);
  L("op_input_rank", "i(i)", &op_input_rank);
  L("op_output_rank", "i(i)", &op_output_rank);
  L("op_input_dims", "i(i*i)", &op_input_dims);
  L("op_output_dims", "i(i*i)", &op_output_dims);
  L("op_input_dtype", "i(i)", &op_input_dtype);
  L("op_output_dtype", "i(i)", &op_output_dtype);
  L("op_input_is_initializer", "i(i)", &op_input_is_initializer);
  L("op_input_init_elem_count", "I(i)", &op_input_init_elem_count);
  L("op_input_init_byte_len", "I(i)", &op_input_init_byte_len);
  L("op_input_init_dtype", "i(i)", &op_input_init_dtype);
  L("op_has_attr", "i(*i)", &op_has_attr);
  L("op_attr_int", "I(*iI)", &op_attr_int);
  L("op_attr_float", "F(*i*)", &op_attr_float);
  L("op_attr_string", "i(*i*i)", &op_attr_string);
  L("op_attr_ints", "i(*i*i)", &op_attr_ints);
  L("op_input_const_ints", "i(*i)", &op_input_const_ints);
  L("op_set_category", "v(i)", &op_set_category);
  L("op_set_flops", "v(Ii)", &op_set_flops);
  L("op_set_output_shape", "v(i*ii)", &op_set_output_shape);
  L("op_set_color", "v(i)", &op_set_color);
}

// Run one facet export (netvis_op_<facet>) with `hc` bound. Returns true if the
// module loaded, linked, and the export returned status 0 (answered); false on
// abstain / trap / fuel / load / abi failure (caller -> honest-unknown).
bool invoke_facet(const std::vector<uint8_t>& image, const char* facet_export,
                  OpHostCtx* hc, WasmOpDiag* diag) {
  WasmEngine& eng = WasmEngine::instance();
  if (!eng.enabled()) return false;

  // §0.3: serialize ALL engine load/link/call — shared IM3Environment + fuel TLS.
  std::lock_guard<std::mutex> guard(eng.lock());

  SandboxLimits lim{kOpMaxPages, kOpMaxSteps};
  RunResult lerr;
  WasmModule mod = eng.load(image, lim, hc, &lerr);
  if (!mod.loaded()) {
    if (diag) { diag->loaded = false; diag->message = lerr.message; }
    return false;
  }
  link_op_capabilities(static_cast<IM3Module>(mod.raw_module()), hc);

  // ABI gate: refuse a module whose netvis_op_abi_version != kOpHandlerAbiVersion.
  int32_t abi = -1;
  RunResult ar = mod.call_i32("netvis_op_abi_version", &abi);
  if (ar.status != RunStatus::Ok ||
      abi != static_cast<int32_t>(kOpHandlerAbiVersion)) {
    if (diag) {
      diag->loaded = true; diag->abi_mismatch = true;
      diag->message = "netvis_op_abi_version mismatch or missing";
    }
    return false;
  }
  if (diag) diag->loaded = true;

  int32_t status = 0;
  RunResult rr = mod.call_i32(facet_export, &status);
  if (rr.status != RunStatus::Ok) return false;  // trap/fuel/missing -> honest-unknown
  return status == 0;                            // nonzero => abstain
}

}  // namespace

WasmOpHandler::WasmOpHandler(std::string plugin_name,
                             std::shared_ptr<const std::vector<uint8_t>> image)
    : plugin_name_(std::move(plugin_name)), image_(std::move(image)) {}

OpCategory WasmOpHandler::category(const OpContext& ctx) const {
  if (!image_) return ctx.default_category();
  OpHostCtx hc; hc.ctx = &ctx;
  if (!invoke_facet(*image_, "netvis_op_category", &hc, &diag_) || !hc.have_category) {
    if (diag_.loaded && !hc.have_category) diag_.missing_category = true;
    return ctx.default_category();
  }
  return clamp_category(hc.raw_category);   // THE CLAMP (§A.3)
}

ColorResult WasmOpHandler::color(const OpContext& ctx) const {
  if (!image_) return ColorResult::use_category();
  OpHostCtx hc; hc.ctx = &ctx;
  if (!invoke_facet(*image_, "netvis_op_color", &hc, nullptr) || !hc.have_color)
    return ColorResult::use_category();
  Rgba8 c;
  c.r = static_cast<uint8_t>((hc.rgba >> 24) & 0xFF);
  c.g = static_cast<uint8_t>((hc.rgba >> 16) & 0xFF);
  c.b = static_cast<uint8_t>((hc.rgba >> 8) & 0xFF);
  c.a = 255;   // force opaque (§A.3)
  return ColorResult::explicit_color(c);
}

FlopResult WasmOpHandler::flops(const OpContext& ctx) const {
  if (!image_) return FlopResult::unknown();
  OpHostCtx hc; hc.ctx = &ctx;
  if (!invoke_facet(*image_, "netvis_op_flops", &hc, nullptr) ||
      !hc.have_flops || !hc.flops_known)
    return FlopResult::unknown();
  return FlopResult::of(hc.flops);
}

ShapeResult WasmOpHandler::infer_shape(const OpContext& ctx) const {
  if (!image_) return ShapeResult::none();
  OpHostCtx hc; hc.ctx = &ctx;
  if (!invoke_facet(*image_, "netvis_op_infer_shape", &hc, nullptr))
    return ShapeResult::none();
  return hc.shape;   // driver applies write-if-empty / carry-dtype-if-Unknown
}

// --- loader -----------------------------------------------------------------
std::string load_wasm_op_plugin(const std::string& plugin_json_path) {
  std::ifstream f(plugin_json_path);
  if (!f) return "cannot open " + plugin_json_path;

  std::string dir;
  {
    auto slash = plugin_json_path.find_last_of("/\\");
    dir = (slash == std::string::npos) ? std::string() : plugin_json_path.substr(0, slash + 1);
  }

  std::string wasm_rel, plugin_name;
  std::vector<WasmOpDecl> ops;
  try {
    nlohmann::json j; f >> j;
    if (!j.is_object()) return "manifest is not an object";
    if (auto v = j.find("api_version"); v != j.end() && v->is_number_integer()) {
      if (v->get<int>() != static_cast<int>(kOpHandlerAbiVersion))
        return "api_version mismatch";
    }
    if (auto n = j.find("name"); n != j.end() && n->is_string()) plugin_name = n->get<std::string>();
    if (auto w = j.find("wasm"); w != j.end() && w->is_string()) wasm_rel = w->get<std::string>();
    if (wasm_rel.empty()) return "manifest missing \"wasm\"";
    if (auto oa = j.find("ops"); oa != j.end() && oa->is_array()) {
      for (const auto& e : *oa) {
        if (!e.is_object()) continue;
        WasmOpDecl d;
        if (auto nm = e.find("name"); nm != e.end() && nm->is_string()) d.op_key = nm->get<std::string>();
        if (auto dm = e.find("domain"); dm != e.end() && dm->is_string()) d.domain = dm->get<std::string>();
        if (auto ov = e.find("override"); ov != e.end() && ov->is_boolean()) d.override_builtin = ov->get<bool>();
        if (!d.op_key.empty()) ops.push_back(std::move(d));
      }
    }
  } catch (...) {
    return "manifest parse error";
  }
  if (ops.empty()) return "manifest declares no ops";

  // Load the .wasm bytes into a shared, immutable image.
  std::ifstream wf(dir + wasm_rel, std::ios::binary);
  if (!wf) return "cannot open wasm: " + wasm_rel;
  auto image = std::make_shared<std::vector<uint8_t>>(
      (std::istreambuf_iterator<char>(wf)), std::istreambuf_iterator<char>());
  if (image->size() < 8) return "wasm image too small";

  for (const WasmOpDecl& d : ops) {
    Registry::instance().register_op_handler(
        d.op_key, d.domain,
        std::make_unique<WasmOpHandler>(plugin_name, image),
        Origin::Wasm, d.override_builtin, plugin_name);
  }
  return {};  // empty => success
}

}  // namespace netvis::plugin::wasm

#else  // !NETVIS_ENABLE_WASM — safe stubs

namespace netvis::plugin::wasm {

WasmOpHandler::WasmOpHandler(std::string plugin_name,
                             std::shared_ptr<const std::vector<uint8_t>> image)
    : plugin_name_(std::move(plugin_name)), image_(std::move(image)) {}
OpCategory  WasmOpHandler::category(const OpContext& ctx) const { return ctx.default_category(); }
ColorResult WasmOpHandler::color(const OpContext&) const { return ColorResult::use_category(); }
FlopResult  WasmOpHandler::flops(const OpContext&) const { return FlopResult::unknown(); }
ShapeResult WasmOpHandler::infer_shape(const OpContext&) const { return ShapeResult::none(); }
std::string load_wasm_op_plugin(const std::string&) { return "WASM disabled"; }

}  // namespace netvis::plugin::wasm

#endif
