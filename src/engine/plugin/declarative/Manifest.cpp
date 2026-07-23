// engine/plugin/declarative/Manifest.cpp — JSONC manifest loader + DeclarativeOpHandler
// (v0.6.0 #9). See Manifest.h.
#include "engine/plugin/declarative/Manifest.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/LayoutCache.h"        // layout_cache_dir (for the sibling config dir)
#include "engine/OpCategory.h"
#include "engine/plugin/OpHandler.h"
#include "engine/plugin/Registry.h"
#include "engine/plugin/declarative/Dsl.h"

namespace netvis::plugin {

using json = nlohmann::json;

namespace {

// A compiled variable binding: name + expression + resolved dependency order.
struct VarBinding {
  std::string name;
  dsl::Expr expr;
};

// A fully compiled declarative op.
struct CompiledOp {
  std::string op_name;                 // normalized
  std::string domain;                  // optional
  OpCategory category = OpCategory::Other;
  bool has_color = false;
  Rgba8 color;
  dsl::Expr flops;                     // may be invalid => flops unknown
  std::vector<VarBinding> vars;        // in dependency order
  // Per-output shape: slot -> list of dim expressions; dtype-source operand slot.
  struct ShapeRule { uint32_t slot; std::vector<dsl::Expr> dims; };
  std::vector<ShapeRule> shape_rules;
};

// Evaluate the var DAG in order, then any expression, against a node.
std::vector<std::pair<std::string, dsl::Val>> eval_vars(const CompiledOp& op,
                                                        const OpContext& ctx) {
  std::vector<std::pair<std::string, dsl::Val>> env;
  env.reserve(op.vars.size());
  for (const VarBinding& vb : op.vars) {
    dsl::Val v = vb.expr.eval(ctx, env);
    env.emplace_back(vb.name, v);
  }
  return env;
}

// ---------------------------------------------------------------------------
// DeclarativeOpHandler — implements the frozen OpHandler ABI over a CompiledOp.
// ---------------------------------------------------------------------------
class DeclarativeOpHandler final : public OpHandler {
 public:
  explicit DeclarativeOpHandler(std::shared_ptr<const CompiledOp> op)
      : op_(std::move(op)) {}

  OpCategory category(const OpContext&) const override { return op_->category; }

  ColorResult color(const OpContext&) const override {
    return op_->has_color ? ColorResult::explicit_color(op_->color)
                          : ColorResult::use_category();
  }

  FlopResult flops(const OpContext& ctx) const override {
    if (!op_->flops.valid()) return FlopResult::unknown();
    auto env = eval_vars(*op_, ctx);
    dsl::Val v = op_->flops.eval(ctx, env);
    if (!v.known || v.v < 0) return FlopResult::unknown();  // honest-unknown / negative
    return FlopResult::of(static_cast<uint64_t>(v.v));
  }

  ShapeResult infer_shape(const OpContext& ctx) const override {
    if (op_->shape_rules.empty()) return ShapeResult::none();
    auto env = eval_vars(*op_, ctx);
    ShapeResult out;
    for (const auto& rule : op_->shape_rules) {
      ShapeResult::Out o;
      o.slot = rule.slot;
      bool any = false;
      for (const dsl::Expr& de : rule.dims) {
        dsl::Val d = de.eval(ctx, env);
        // Unknown dim -> -1 (dynamic); a resolved dim must be >=1 else -1.
        int64_t dim = (d.known && d.v >= 1) ? d.v : -1;
        if (dim >= 1) any = true;
        o.shape.push_back(dim);
      }
      // Emit only if at least one dim resolved (else leave the value unknown).
      if (any) out.outputs.push_back(std::move(o));
    }
    return out;
  }

  uint32_t api_version() const override { return kOpHandlerAbiVersion; }

 private:
  std::shared_ptr<const CompiledOp> op_;
};

// Parse "#RRGGBB" -> Rgba8; returns false if malformed.
bool parse_color(const std::string& s, Rgba8& out) {
  if (s.size() != 7 || s[0] != '#') return false;
  auto hex = [](char c, int& v) {
    if (c >= '0' && c <= '9') { v = c - '0'; return true; }
    if (c >= 'a' && c <= 'f') { v = c - 'a' + 10; return true; }
    if (c >= 'A' && c <= 'F') { v = c - 'A' + 10; return true; }
    return false;
  };
  int hi, lo;
  uint8_t rgb[3];
  for (int k = 0; k < 3; ++k) {
    if (!hex(s[1 + k * 2], hi) || !hex(s[2 + k * 2], lo)) return false;
    rgb[k] = static_cast<uint8_t>(hi * 16 + lo);
  }
  out.r = rgb[0]; out.g = rgb[1]; out.b = rgb[2]; out.a = 255;
  return true;
}

// Pre-scan brace/bracket nesting depth BEFORE nlohmann::parse (it has no built-in
// limit; a deep-nesting file could stack-overflow the recursive parser). [critique-fix]
bool nesting_ok(const std::string& text, int max_depth) {
  int depth = 0;
  bool in_str = false, esc = false;
  for (char c : text) {
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') in_str = true;
    else if (c == '{' || c == '[') { if (++depth > max_depth) return false; }
    else if (c == '}' || c == ']') { if (depth > 0) --depth; }
  }
  return true;
}

// Compile one [[op]] JSON block. Returns nullptr + fills err on rejection.
std::shared_ptr<CompiledOp> compile_op(const json& jop, LoadedOp& diag) {
  auto op = std::make_shared<CompiledOp>();
  const dsl::Limits lim;

  if (!jop.contains("name") || !jop["name"].is_string()) {
    diag.error = "op missing string 'name'"; return nullptr;
  }
  op->op_name = normalize_op_key(jop["name"].get<std::string>());
  diag.op_name = op->op_name;
  if (jop.contains("domain") && jop["domain"].is_string())
    op->domain = jop["domain"].get<std::string>();

  // category (required, must be known)
  if (!jop.contains("category") || !jop["category"].is_string()) {
    diag.error = "op missing string 'category'"; return nullptr;
  }
  std::string cat = jop["category"].get<std::string>();
  auto c = category_from_name(cat);
  if (!c) { diag.error = "unknown category '" + cat + "'"; return nullptr; }
  op->category = *c;
  diag.category = cat;

  // color (optional)
  if (jop.contains("color") && jop["color"].is_string()) {
    if (parse_color(jop["color"].get<std::string>(), op->color)) op->has_color = true;
    // malformed color is non-fatal: fall back to category color.
  }

  // vars (optional): compile each, then cycle-check the DAG.
  if (jop.contains("vars") && jop["vars"].is_object()) {
    if (jop["vars"].size() > 256) { diag.error = "too many vars (>256)"; return nullptr; }
    // First pass: compile.
    std::vector<VarBinding> raw;
    for (auto& [k, v] : jop["vars"].items()) {
      if (!v.is_string()) { diag.error = "var '" + k + "' is not a string expr"; return nullptr; }
      std::string e;
      dsl::Expr expr = dsl::Expr::compile(v.get<std::string>(), lim, &e);
      if (!expr.valid()) { diag.error = "var '" + k + "': " + e; return nullptr; }
      raw.push_back({k, std::move(expr)});
    }
    // Topological order over var->var references (cycle => reject).
    std::unordered_set<std::string> names;
    for (auto& vb : raw) names.insert(vb.name);
    std::unordered_set<std::string> done;
    std::vector<VarBinding> ordered;
    // Kahn-ish: repeatedly emit any var whose var-deps are all done. If a full pass
    // emits nothing but some remain, there is a cycle.
    bool progress = true;
    std::vector<bool> used(raw.size(), false);
    while (ordered.size() < raw.size() && progress) {
      progress = false;
      for (size_t i = 0; i < raw.size(); ++i) {
        if (used[i]) continue;
        bool ready = true;
        for (const std::string& ref : raw[i].expr.referenced_vars()) {
          if (names.count(ref) && !done.count(ref)) { ready = false; break; }
        }
        if (ready) {
          used[i] = true; done.insert(raw[i].name);
          ordered.push_back(std::move(raw[i])); progress = true;
        }
      }
    }
    if (ordered.size() != raw.size()) { diag.error = "cyclic var dependency"; return nullptr; }
    op->vars = std::move(ordered);
  }

  // flops (optional): empty/absent => unknown (honest).
  if (jop.contains("flops") && jop["flops"].is_string()) {
    std::string e;
    op->flops = dsl::Expr::compile(jop["flops"].get<std::string>(), lim, &e);
    if (!op->flops.valid() && !jop["flops"].get<std::string>().empty()) {
      diag.error = "flops: " + e; return nullptr;
    }
  }

  // shape (optional): { "out0": ["expr", "expr", ...], ... }
  if (jop.contains("shape") && jop["shape"].is_object()) {
    for (auto& [k, v] : jop["shape"].items()) {
      // key form "outN"
      if (k.size() < 4 || k.compare(0, 3, "out") != 0) continue;
      uint32_t slot = 0;
      try { slot = static_cast<uint32_t>(std::stoul(k.substr(3))); }
      catch (...) { continue; }
      if (!v.is_array()) { diag.error = "shape '" + k + "' is not an array"; return nullptr; }
      CompiledOp::ShapeRule rule; rule.slot = slot;
      for (const auto& de : v) {
        if (!de.is_string()) { diag.error = "shape dim is not a string expr"; return nullptr; }
        std::string e;
        dsl::Expr expr = dsl::Expr::compile(de.get<std::string>(), lim, &e);
        if (!expr.valid()) { diag.error = "shape dim: " + e; return nullptr; }
        rule.dims.push_back(std::move(expr));
      }
      op->shape_rules.push_back(std::move(rule));
    }
  }

  diag.ok = true;
  return op;
}

}  // namespace

std::string plugin_dir() {
  // Config-tree sibling of the cache dir (NOT the wipeable cache). We derive it by
  // replacing the trailing cache component; simplest robust route is a parallel
  // platform switch mirroring layout_cache_dir but under config.
  std::filesystem::path dir;
#if defined(_WIN32)
  if (const char* la = std::getenv("LOCALAPPDATA"))
    dir = std::filesystem::path(la) / "NetVis" / "plugins";
  else
    dir = std::filesystem::path(".") / "NetVis" / "plugins";
#elif defined(__APPLE__)
  if (const char* home = std::getenv("HOME"))
    dir = std::filesystem::path(home) / "Library" / "Application Support" / "NetVis" / "plugins";
  else
    dir = std::filesystem::path(".") / "NetVis" / "plugins";
#else
  if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg && xdg[0] != '\0')
    dir = std::filesystem::path(xdg) / "netvis" / "plugins";
  else if (const char* home = std::getenv("HOME"))
    dir = std::filesystem::path(home) / ".config" / "netvis" / "plugins";
  else
    dir = std::filesystem::path(".") / "netvis" / "plugins";
#endif
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);  // best-effort
  return dir.string();
}

LoadedManifest load_manifest_file(const std::string& path, bool register_into) {
  LoadedManifest lm;
  lm.path = path;

  std::ifstream f(path, std::ios::binary);
  if (!f) { lm.error = "cannot open file"; return lm; }
  std::stringstream ss; ss << f.rdbuf();
  std::string text = ss.str();
  if (text.size() > 4u * 1024 * 1024) { lm.error = "manifest too large (>4MiB)"; return lm; }
  if (!nesting_ok(text, 64)) { lm.error = "manifest nesting too deep"; return lm; }

  json j = json::parse(text, nullptr, /*allow_exceptions=*/false, /*ignore_comments=*/true);
  if (j.is_discarded()) { lm.error = "JSON parse error"; return lm; }
  if (!j.is_object()) { lm.error = "manifest root is not an object"; return lm; }

  if (j.contains("api_version") && j["api_version"].is_number_unsigned())
    lm.api_version = j["api_version"].get<uint32_t>();
  if (lm.api_version != kOpHandlerAbiVersion) {
    lm.error = "api_version " + std::to_string(lm.api_version) + " != host " +
               std::to_string(kOpHandlerAbiVersion);
    return lm;  // whole file rejected, nothing registered
  }
  if (j.contains("name") && j["name"].is_string()) lm.name = j["name"].get<std::string>();
  if (j.contains("author") && j["author"].is_string()) lm.author = j["author"].get<std::string>();

  if (!j.contains("ops") || !j["ops"].is_array()) { lm.error = "manifest has no 'ops' array"; return lm; }
  if (j["ops"].size() > 4096) { lm.error = "too many ops (>4096)"; return lm; }

  lm.ok = true;
  for (const auto& jop : j["ops"]) {
    LoadedOp diag;
    bool override_flag = jop.contains("override") && jop["override"].is_boolean() &&
                         jop["override"].get<bool>();
    diag.overrides_builtin = override_flag;
    std::shared_ptr<CompiledOp> op = compile_op(jop, diag);
    if (op && register_into) {
      Registry::instance().register_op_handler(
          op->op_name, op->domain,
          std::make_unique<DeclarativeOpHandler>(op),
          Origin::Declarative, override_flag, lm.name);
    }
    lm.ops.push_back(std::move(diag));
  }
  return lm;
}

std::vector<LoadedManifest> discover_and_load_plugins() {
  std::vector<LoadedManifest> out;
  std::string root = plugin_dir();
  std::error_code ec;
  std::vector<std::filesystem::path> manifests;
  for (auto it = std::filesystem::directory_iterator(root, ec);
       !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
    if (!it->is_directory(ec)) continue;
    std::filesystem::path pj = it->path() / "plugin.json";
    if (std::filesystem::exists(pj, ec)) manifests.push_back(pj);
  }
  std::sort(manifests.begin(), manifests.end());  // deterministic path order
  for (const auto& p : manifests) out.push_back(load_manifest_file(p.string(), true));
  return out;
}

}  // namespace netvis::plugin
