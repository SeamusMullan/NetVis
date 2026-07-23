// engine/plugin/OpHandler.h — FROZEN op-level plugin ABI (v0.6.0, issue #7).
//
// An OpHandler answers "what is this op": color category, optional explicit color,
// FLOP estimate, and output-shape propagation — from STRUCTURE ONLY. OpContext has
// NO byte-access term, so the zero-payload thesis (invariant 3) is a property of
// the TYPE, not a convention: a handler literally cannot read a weight. The one
// bytes-adjacent accessor (input_const_ints) is HOST-mediated, bounded to <=64
// I32/I64 metadata elems, and never bumps the payload counter — see below.
//
// This mirrors compute_flops() in CostModel.cpp, the reference implementation
// every built-in handler ports from. PURITY: includes ir/IR.h + engine leaf
// headers only — identical include set to CostModel.cpp. No view/ImGui/parser dep.
//
// See docs/v0.6.0-design.md Part 2.1 + the Part 3 coverage-proof table.
#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "core/SmallVec.h"
#include "engine/HeatmapGradient.h"   // Rgba8 (leaf: <cstdint> only)
#include "engine/OpCategory.h"        // OpCategory, categorize_op
#include "engine/plugin/ShapeMath.h"  // Shape, elem_count, partial_product
#include "ir/IR.h"                    // ir::DType, ir::Model, ir::Graph, ir::Node

namespace netvis::plugin {

// Payload-format ABI version for DSL manifests + WASM modules. NOTE: this is NOT
// a C++ binary-vtable version — invariant 1 means every handler (built-in, DSL
// closure, WASM adapter) is compiled INTO the single binary as one unit, so there
// is no dlopen of an external C++ .so and vtable layout is never crossed. This
// constant gates (a) declarative manifest `api_version` and (b) the WASM module's
// exported netvis_abi_version(); a mismatch => that plugin is refused, never
// half-registered.
inline constexpr uint32_t kOpHandlerAbiVersion = 1;

// Initializer record for an input slot, when that slot names a Graph::initializers
// entry. byte_len is the ONLY correct byte source for Q4/Q8 (dtype_size()==0), so
// params/weight-byte handlers MUST read it, not shape*dtype_size.
struct InitRef {
  int64_t   elem_count = 0;     // TensorRef::elem_count() (>0 when known)
  uint64_t  byte_len   = 0;     // recorded byte_len (truth for quantized)
  ir::DType dtype      = ir::DType::Unknown;
};

// ---------------------------------------------------------------------------
// OpContext — READ-ONLY node view. Every accessor is bounds-checked and returns a
// "not available" sentinel (nullptr / nullopt / Unknown / default) instead of
// throwing, so a handler over hostile/partial IR never crashes (inv 4/5). A NULL/
// absent/empty shape means "unresolved" -> honest-unknown (inv 6), exactly as
// compute_flops returns early when get_input_value()/get_output_value() are null
// or elem_count_from_shape()==0. Borrows (model,graph,node); copies nothing.
// ---------------------------------------------------------------------------
class OpContext {
 public:
  // --- Identity -----------------------------------------------------------
  std::string_view op_type() const;          // NORMALIZED: lowercased last dot-
                                             // segment (== CostModel.cpp norm_op).
                                             // This is the exact-tier registry key.
  std::string_view op_raw() const;           // original ("com.microsoft.QLinearConv")
  std::string_view domain() const;           // "com.microsoft" or "" if none
  OpCategory default_category() const;       // categorize_op(op_raw) — the color
                                             // tier key + the generic-formula branch
                                             // selector (Norm/Activation/Elementwise
                                             // -> |O|, Reduce -> |in0|, etc.).

  // --- Slot counts / shapes / dtypes -------------------------------------
  uint32_t input_count() const;
  uint32_t output_count() const;
  const Shape* input_shape(uint32_t slot) const;   // nullptr if slot/edge/value OOR
  const Shape* output_shape(uint32_t slot) const;  // (== get_input/output_value)
  ir::DType input_dtype(uint32_t slot) const;      // Unknown if slot absent
  ir::DType output_dtype(uint32_t slot) const;

  // --- Initializer (weight) membership + record --------------------------
  bool input_is_initializer(uint32_t slot) const;                 // build_initializer_index
  std::optional<InitRef> input_initializer(uint32_t slot) const;  // elem_count/byte_len/dtype

  // --- Attributes (mirror CostModel.cpp's three helpers EXACTLY) ----------
  bool has_attr(std::string_view name) const;      // present AND of a known kind
  int64_t attr_int(std::string_view name, int64_t default_val) const;  // == get_int_attr
  std::string_view attr_string(std::string_view name,
                               std::string_view default_val) const;    // == get_string_attr
  const std::vector<int64_t>* attr_ints(std::string_view name) const;  // nullptr if absent/wrong-kind
  std::optional<double> attr_float(std::string_view name) const;

  // --- Constant integer INPUT (shape/index/axes tensors) — the ONE bytes-
  // adjacent accessor, reconciling with ShapeInferenceExt's constant-driven
  // rules (Reshape/Slice/Gather/Expand/Tile/Resize). Host reads it under EXACTLY
  // ShapeInferenceExt::read_shape_initializer discipline:
  //   raw_data initializer only (file_offset!=UINT64_MAX, not external);
  //   dtype I32/I64 only; elem_count in (0,64] (can never span a weight);
  //   does NOT call ByteReader::mark_payload_read (<=64-elem shape tensor is
  //   METADATA). Returns nullptr when ANY condition fails, OR when mmap_base_ is
  //   null (the COST driver leaves it null, so flops() handlers deterministically
  //   get honest-unknown; only the SHAPE driver, called with an mmap, resolves
  //   const inputs).
  const std::vector<int64_t>* input_const_ints(uint32_t slot) const;

  // --- Raw handles for BuiltinOpHandler's delegation to compute_flops -------
  const ir::Model& model() const { return *model_; }
  const ir::Graph& graph() const { return *graph_; }
  const ir::Node&  node()  const { return *node_; }

 private:
  const ir::Model* model_ = nullptr;
  const ir::Graph* graph_ = nullptr;
  const ir::Node*  node_  = nullptr;
  const uint8_t*   mmap_base_ = nullptr;   // set only by the shape-inference driver
  uint64_t         mmap_size_ = 0;
  // input_const_ints resolution needs a scratch buffer owned by the context.
  mutable std::vector<int64_t> const_scratch_;
  friend class Registry;
};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------
struct FlopResult {                 // honest-unknown by default (==NodeCost{0,false})
  uint64_t flops = 0;
  bool     known = false;
  static FlopResult unknown() { return {}; }
  static FlopResult of(uint64_t f) { return {f, true}; }   // MAC=2 is caller's job
};

struct ColorResult {                // default = use category color (view palette)
  bool  overridden = false;
  Rgba8 rgba;
  static ColorResult use_category() { return {}; }
  static ColorResult explicit_color(Rgba8 c) { return {true, c}; }
};

// Per-output-slot propagation, reconciled with infer_shapes_ext's MUTATE-IN-PLACE,
// best-effort, honest-unknown contract. The registry ADAPTER applies each entry
// with the SAME set_shape/carry_dtype semantics: write shape ONLY if the value's
// shape is currently empty; carry dtype ONLY if currently Unknown -> a handler can
// never clobber an already-resolved value, two handlers can't fight. A dim of -1
// is a legal dynamic dim, propagated as-is.
struct ShapeResult {
  struct Out {
    uint32_t  slot  = 0;
    Shape     shape;                       // empty => don't set shape for this slot
    ir::DType dtype = ir::DType::Unknown;  // Unknown => don't carry dtype
  };
  std::vector<Out> outputs;                // empty => handler resolved nothing
  static ShapeResult none() { return {}; }
};

// ---------------------------------------------------------------------------
// OpHandler — implemented by BuiltinOpHandler, DeclarativeOpHandler, WasmOpHandler.
// category() is required; color/flops/infer_shape default to
// use-category/unknown/none so ADDING one of them to a handler is never forced and
// a new defaulted virtual is a non-breaking (non-version-bumping) ABI change.
// ---------------------------------------------------------------------------
class OpHandler {
 public:
  virtual ~OpHandler() = default;
  virtual OpCategory  category(const OpContext& ctx) const = 0;      // required
  virtual ColorResult color(const OpContext&) const { return ColorResult::use_category(); }
  virtual FlopResult  flops(const OpContext&) const { return FlopResult::unknown(); }
  virtual ShapeResult infer_shape(const OpContext&) const { return ShapeResult::none(); }
  virtual uint32_t    api_version() const = 0;   // == kOpHandlerAbiVersion or refused
};

}  // namespace netvis::plugin
