// engine/plugin/Registry.cpp — registry spine + OpContext accessors +
// BuiltinOpHandler (v0.6.0 Increment 1, issue #8).
//
// ZERO BEHAVIOR CHANGE: with no user plugins loaded, resolve_op() always returns
// the built-in catch-all, whose flops()/category() delegate to the UNCHANGED
// v0.4.0 formula table (detail::builtin_compute_flops + categorize_op). The
// registry adds an indirection, never a reimplementation — the per-node FLOP /
// category output is byte-identical to a pre-registry build by construction.
//
// The OpContext accessors here mirror CostModel.cpp's TU-local helpers
// (get_int_attr / get_string_attr / get_input_value / build_initializer_index)
// and ShapeInferenceExt.cpp's read_shape_initializer discipline EXACTLY, so a
// plugin handler sees the same structural view the built-ins do.
#include "engine/plugin/Registry.h"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <string>

#include "core/ByteReader.h"
#include "engine/CostModel.h"     // NodeCost, detail::builtin_compute_flops
#include "engine/OpCategory.h"    // categorize_op
#include "ir/IR.h"

namespace netvis::plugin {

// Normalize an op string EXACTLY as CostModel.cpp::norm_op: strip domain (keep the
// last dot-segment) and lowercase. "com.microsoft.QLinearConv" -> "qlinearconv".
std::string normalize_op_key(std::string_view op) {
  std::size_t dot = op.find_last_of('.');
  std::string_view tail = (dot == std::string_view::npos) ? op : op.substr(dot + 1);
  std::string out(tail);
  for (char& c : out)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

namespace {

// Domain prefix of an op string ("com.microsoft.QLinearConv" -> "com.microsoft"),
// or "" if none.
std::string_view op_domain(std::string_view op) {
  std::size_t dot = op.find_last_of('.');
  return (dot == std::string_view::npos) ? std::string_view{} : op.substr(0, dot);
}

// ---------------------------------------------------------------------------
// BuiltinOpHandler — the catch-all. Delegates to the v0.4.0 formula table so the
// built-in path and the plugin path are one path (dogfood).
// ---------------------------------------------------------------------------
class BuiltinOpHandler final : public OpHandler {
 public:
  OpCategory category(const OpContext& ctx) const override {
    return categorize_op(ctx.op_raw());
  }
  FlopResult flops(const OpContext& ctx) const override {
    NodeCost nc;
    detail::builtin_compute_flops(ctx.model(), ctx.graph(), ctx.node(), nc);
    return {nc.flops, nc.flops_known};
  }
  // Shape inference for built-ins stays in ShapeInferenceExt's monolith (the hook
  // is skipped when origin==Builtin), so the built-in handler resolves nothing here.
  ShapeResult infer_shape(const OpContext&) const override { return ShapeResult::none(); }
  ColorResult color(const OpContext&) const override { return ColorResult::use_category(); }
  uint32_t api_version() const override { return kOpHandlerAbiVersion; }
};

}  // namespace

// ===========================================================================
// OpContext accessors
// ===========================================================================

std::string_view OpContext::op_raw() const {
  return model_ ? model_->str(node_->op_type) : std::string_view{};
}

std::string_view OpContext::op_type() const {
  // The normalized key is precomputed by the registry into a member? No — recompute
  // is cheap and this is called at most once per op-type (hoisted). Kept pure.
  static thread_local std::string scratch;
  scratch = normalize_op_key(op_raw());
  return scratch;
}

std::string_view OpContext::domain() const { return op_domain(op_raw()); }

OpCategory OpContext::default_category() const { return categorize_op(op_raw()); }

uint32_t OpContext::input_count() const { return node_ ? node_->inputs.count : 0; }
uint32_t OpContext::output_count() const { return node_ ? node_->outputs.count : 0; }

const Shape* OpContext::input_shape(uint32_t slot) const {
  if (!graph_ || !node_ || slot >= node_->inputs.count) return nullptr;
  uint32_t er = node_->inputs.begin + slot;
  if (er >= graph_->edge_refs.size()) return nullptr;
  uint32_t vidx = graph_->edge_refs[er];
  if (vidx >= graph_->values.size()) return nullptr;
  return &graph_->values[vidx].shape;
}

const Shape* OpContext::output_shape(uint32_t slot) const {
  if (!graph_ || !node_ || slot >= node_->outputs.count) return nullptr;
  uint32_t er = node_->outputs.begin + slot;
  if (er >= graph_->edge_refs.size()) return nullptr;
  uint32_t vidx = graph_->edge_refs[er];
  if (vidx >= graph_->values.size()) return nullptr;
  return &graph_->values[vidx].shape;
}

ir::DType OpContext::input_dtype(uint32_t slot) const {
  if (!graph_ || !node_ || slot >= node_->inputs.count) return ir::DType::Unknown;
  uint32_t er = node_->inputs.begin + slot;
  if (er >= graph_->edge_refs.size()) return ir::DType::Unknown;
  uint32_t vidx = graph_->edge_refs[er];
  if (vidx >= graph_->values.size()) return ir::DType::Unknown;
  return graph_->values[vidx].dtype;
}

ir::DType OpContext::output_dtype(uint32_t slot) const {
  if (!graph_ || !node_ || slot >= node_->outputs.count) return ir::DType::Unknown;
  uint32_t er = node_->outputs.begin + slot;
  if (er >= graph_->edge_refs.size()) return ir::DType::Unknown;
  uint32_t vidx = graph_->edge_refs[er];
  if (vidx >= graph_->values.size()) return ir::DType::Unknown;
  return graph_->values[vidx].dtype;
}

// Find the initializer index for input slot's value name, or -1.
static int64_t init_index_for_slot(const ir::Model& /*m*/, const ir::Graph& g,
                                    const ir::Node& node, uint32_t slot) {
  if (slot >= node.inputs.count) return -1;
  uint32_t er = node.inputs.begin + slot;
  if (er >= g.edge_refs.size()) return -1;
  uint32_t vidx = g.edge_refs[er];
  if (vidx >= g.values.size()) return -1;
  StringId want = g.values[vidx].name;
  if (!want.valid()) return -1;
  for (uint32_t i = 0; i < g.initializers.size(); ++i) {
    if (g.initializers[i].name == want) return static_cast<int64_t>(i);
  }
  return -1;
}

bool OpContext::input_is_initializer(uint32_t slot) const {
  if (!model_ || !graph_ || !node_) return false;
  return init_index_for_slot(*model_, *graph_, *node_, slot) >= 0;
}

std::optional<InitRef> OpContext::input_initializer(uint32_t slot) const {
  if (!model_ || !graph_ || !node_) return std::nullopt;
  int64_t idx = init_index_for_slot(*model_, *graph_, *node_, slot);
  if (idx < 0) return std::nullopt;
  const ir::TensorRef& t = graph_->initializers[static_cast<size_t>(idx)];
  InitRef r;
  r.elem_count = t.elem_count();
  r.byte_len = t.byte_len;
  r.dtype = t.dtype;
  return r;
}

// --- attributes (mirror CostModel.cpp helpers EXACTLY) ---------------------
static const ir::Attribute* OpContext_find_attr(const ir::Model& m, const ir::Graph& g,
                                                const ir::Node& node, std::string_view name) {
  for (uint32_t i = 0; i < node.attributes.count; ++i) {
    uint32_t aidx = node.attributes.begin + i;
    if (aidx >= g.attributes.size()) continue;
    const ir::Attribute& attr = g.attributes[aidx];
    if (m.str(attr.name) == name) return &attr;
  }
  return nullptr;
}

bool OpContext::has_attr(std::string_view name) const {
  if (!model_ || !graph_ || !node_) return false;
  const ir::Attribute* a = OpContext_find_attr(*model_, *graph_, *node_, name);
  return a && a->value.kind != ir::AttrValue::Kind::None;
}

int64_t OpContext::attr_int(std::string_view name, int64_t default_val) const {
  if (!model_ || !graph_ || !node_) return default_val;
  const ir::Attribute* a = OpContext_find_attr(*model_, *graph_, *node_, name);
  if (a && a->value.kind == ir::AttrValue::Kind::Int) return a->value.i;
  return default_val;
}

std::string_view OpContext::attr_string(std::string_view name,
                                        std::string_view default_val) const {
  if (!model_ || !graph_ || !node_) return default_val;
  const ir::Attribute* a = OpContext_find_attr(*model_, *graph_, *node_, name);
  if (a && a->value.kind == ir::AttrValue::Kind::String) return model_->str(a->value.s);
  return default_val;
}

const std::vector<int64_t>* OpContext::attr_ints(std::string_view name) const {
  if (!model_ || !graph_ || !node_) return nullptr;
  const ir::Attribute* a = OpContext_find_attr(*model_, *graph_, *node_, name);
  if (a && a->value.kind == ir::AttrValue::Kind::Ints) return &a->value.ints;
  return nullptr;
}

std::optional<double> OpContext::attr_float(std::string_view name) const {
  if (!model_ || !graph_ || !node_) return std::nullopt;
  const ir::Attribute* a = OpContext_find_attr(*model_, *graph_, *node_, name);
  if (a && a->value.kind == ir::AttrValue::Kind::Float) return a->value.f;
  return std::nullopt;
}

// Constant int input — mirrors ShapeInferenceExt::read_shape_initializer discipline:
// raw_data initializer only, dtype I32/I64, elem_count in (0,64], no payload-counter
// bump, and only when mmap_base_ is non-null (shape driver). nullptr otherwise.
const std::vector<int64_t>* OpContext::input_const_ints(uint32_t slot) const {
  const_scratch_.clear();
  if (!model_ || !graph_ || !node_ || mmap_base_ == nullptr) return nullptr;
  int64_t idx = init_index_for_slot(*model_, *graph_, *node_, slot);
  if (idx < 0) return nullptr;
  const ir::TensorRef& t = graph_->initializers[static_cast<size_t>(idx)];
  if (t.external_path.valid()) return nullptr;
  if (t.file_offset == UINT64_MAX) return nullptr;
  int64_t n = t.elem_count();
  if (n <= 0 || n > 64) return nullptr;
  ByteReader r(mmap_base_, mmap_size_);
  if (!r.seek(t.file_offset)) return nullptr;
  const_scratch_.reserve(static_cast<size_t>(n));
  if (t.dtype == ir::DType::I64) {
    for (int64_t i = 0; i < n; ++i) {
      auto v = r.i64le();
      if (!v) { const_scratch_.clear(); return nullptr; }
      const_scratch_.push_back(*v);
    }
  } else if (t.dtype == ir::DType::I32) {
    for (int64_t i = 0; i < n; ++i) {
      auto v = r.i32le();
      if (!v) { const_scratch_.clear(); return nullptr; }
      const_scratch_.push_back(*v);
    }
  } else {
    return nullptr;
  }
  return &const_scratch_;
}

// ===========================================================================
// Registry
// ===========================================================================

Registry& Registry::instance() {
  static Registry inst;
  return inst;
}

namespace {
// The one catch-all built-in handler. Static lifetime; the RegistryTable points
// at it. Stateless, so a single shared instance is safe across threads.
BuiltinOpHandler& builtin_handler() {
  static BuiltinOpHandler h;
  return h;
}

// Build the default table (built-ins only). Called lazily on first snapshot.
std::shared_ptr<const RegistryTable> make_default_table() {
  auto t = std::make_shared<RegistryTable>();
  t->builtin_op = &builtin_handler();
  return t;
}
}  // namespace

std::shared_ptr<const RegistryTable> Registry::snapshot() const {
  // Lazy-init on first use. Reads copy the shared_ptr under a short lock (a refcount
  // bump); the rare load/reload swap takes the same lock. Portable across libstdc++
  // + libc++ (no atomic<shared_ptr>). resolve_op is hoisted once per op-type, so the
  // lock cost is off the per-node hot path.
  std::lock_guard<std::mutex> lk(table_mutex_);
  if (!table_) table_ = make_default_table();
  return table_;
}

OpResolution Registry::resolve_op(std::string_view op_norm) const {
  auto tbl = snapshot();
  auto it = tbl->op_by_key.find(std::string(op_norm));
  if (it != tbl->op_by_key.end()) return it->second;
  return OpResolution{tbl->builtin_op, Origin::Builtin, false, std::string_view{}};
}

OpContext Registry::make_context(const ir::Model& model, const ir::Graph& g,
                                 const ir::Node& node, std::string_view /*op_norm*/,
                                 const uint8_t* mmap_base, uint64_t mmap_size) const {
  OpContext ctx;
  ctx.model_ = &model;
  ctx.graph_ = &g;
  ctx.node_ = &node;
  ctx.mmap_base_ = mmap_base;
  ctx.mmap_size_ = mmap_size;
  return ctx;
}

void Registry::reload(std::shared_ptr<const RegistryTable> next) {
  std::lock_guard<std::mutex> lk(table_mutex_);
  table_ = std::move(next);
}

// Registration entry points. RegistryTable owns non-copyable unique_ptr storage, so
// registration mutates the single owned table IN PLACE under table_mutex_ (the same
// lock readers take). Registration is a load-time event; the per-node read path
// (resolve_op) is hoisted once per op-type so the shared lock is off the hot path.
// NOTE: hold table_mutex_ directly and init inline — do NOT call snapshot() here
// (non-recursive mutex would deadlock).
void Registry::register_op_handler(std::string norm_key, std::string /*domain*/,
                                   std::unique_ptr<OpHandler> h, Origin origin,
                                   bool override_flag, std::string plugin_name) {
  if (!h || h->api_version() != kOpHandlerAbiVersion) return;  // refuse mismatch
  std::lock_guard<std::mutex> lk(table_mutex_);
  if (!table_) table_ = make_default_table();
  auto& mut = const_cast<RegistryTable&>(*table_);
  // A key shadows a BUILT-IN when categorize_op recognizes it as a known op family
  // (the built-in catch-all would otherwise answer it), or shadows an already-
  // registered USER handler. Either way, shadowing REQUIRES an explicit
  // override:true (design §0.5/§Q3 — a plugin must not silently replace a built-in
  // FLOP/category formula). Refuse otherwise; never half-register.
  bool shadows_builtin = categorize_op(norm_key) != OpCategory::Other;
  bool shadows_user = mut.op_by_key.count(norm_key) > 0;
  if ((shadows_builtin || shadows_user) && !override_flag) return;
  OpHandler* raw = h.get();
  mut.op_storage.push_back(std::move(h));
  // Own the plugin name in stable (deque) storage so the string_view never dangles.
  mut.plugin_names.push_back(std::move(plugin_name));
  std::string_view name_view = mut.plugin_names.back();
  mut.op_by_key[norm_key] =
      OpResolution{raw, origin, shadows_builtin || shadows_user, name_view};
}

void Registry::register_parser(std::unique_ptr<ParserPlugin> p) {
  if (!p || p->api_version() != kParserPluginAbiVersion) return;
  std::lock_guard<std::mutex> lk(table_mutex_);
  if (!table_) table_ = make_default_table();
  auto& mut = const_cast<RegistryTable&>(*table_);
  mut.parsers.push_back(p.get());
  mut.parser_storage.push_back(std::move(p));
  std::sort(mut.parsers.begin(), mut.parsers.end(),
            [](const ParserPlugin* a, const ParserPlugin* b) {
              return a->priority() < b->priority();
            });
}

void Registry::register_pass(std::unique_ptr<PassPlugin> p) {
  if (!p || p->api_version() != kPassPluginAbiVersion) return;
  std::lock_guard<std::mutex> lk(table_mutex_);
  if (!table_) table_ = make_default_table();
  auto tbl = table_;
  auto& mut = const_cast<RegistryTable&>(*tbl);
  mut.passes[std::string(p->display_name())] = p.get();
  mut.pass_storage.push_back(std::move(p));
}

OpCategory resolve_category(const ir::Model& model, const ir::Graph& g,
                            const ir::Node& node) {
  Registry& reg = Registry::instance();
  OpResolution r = reg.resolve_op(normalize_op_key(model.str(node.op_type)));
  OpContext ctx = reg.make_context(model, g, node, {});
  return r.handler ? r.handler->category(ctx) : categorize_op(model.str(node.op_type));
}

Format Registry::detect_format(const MappedFile& file, const std::string& ext_hint) const {
  // Increment 1: no registered parser plugins claim formats, so this defers to the
  // built-in detect_format (which #8's Detect refactor wraps). Kept as a thin
  // pass-through; the built-in sniff order is authoritative.
  return netvis::detect_format(file, ext_hint);
}

const ParserPlugin* Registry::parser_for_format(Format f) const {
  auto tbl = snapshot();
  for (const ParserPlugin* p : tbl->parsers)
    if (p->format() == f) return p;
  return nullptr;
}

}  // namespace netvis::plugin
