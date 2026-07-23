// engine/plugin/Registry.h — FROZEN registry spine (v0.6.0, issue #7 / Increment 1).
//
// Immutable-snapshot, lock-free reads. Reads come from BOTH the worker thread
// (CostModel + infer_shapes_ext run as jobs) and the UI thread (per-frame color)
// concurrently, so the active table is a shared_ptr<const RegistryTable> swapped
// atomically on the rare user-driven reload (plugin enable/disable). Handlers are
// owned by unique_ptr and looked up ONCE PER OP-TYPE (hoisted), never copied per
// node — no std::function-copy cost on the 3095-node hot path.
//
// See docs/v0.6.0-design.md Part 2.4 + Part 3.3 (two-tier resolution + hoist).
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/MappedFile.h"
#include "engine/plugin/OpHandler.h"
#include "engine/plugin/ParserPlugin.h"
#include "engine/plugin/PassPlugin.h"
#include "parsers/Parser.h"          // Format

namespace netvis::plugin {

// Normalize an op string to the exact-tier registry key: strip domain (keep the
// last dot-segment) + lowercase ("com.microsoft.QLinearConv" -> "qlinearconv").
// Byte-identical to CostModel.cpp::norm_op — the single normalization both the
// built-in dispatch and the registry key use.
std::string normalize_op_key(std::string_view op);

// Resolve the color CATEGORY for a node through the registry (v0.6.0 #8): the
// winning handler's category(). With no user plugins this is exactly
// categorize_op(op) — the view sites route through here so a declarative/WASM
// plugin can recolor an op (#9), while layout-structural predicates keep calling
// categorize_op directly so a plugin category can never shift the layout.
OpCategory resolve_category(const ir::Model& model, const ir::Graph& g,
                            const ir::Node& node);

// Backend tier = PRIMARY key of the total override order (design §0.5). Higher wins.
enum class Origin : uint8_t { Builtin = 0, Declarative = 1, Wasm = 2 };

struct OpResolution {
  const OpHandler* handler = nullptr;   // never null for a resolved op (builtin is catch-all)
  Origin           origin  = Origin::Builtin;
  bool             overrides_builtin = false;  // a user handler shadows a builtin key
  std::string_view plugin_name;                // "" for builtin
};

// IMMUTABLE once built.
struct RegistryTable {
  // Exact-op tier: normalized-op key -> the winning user handler (WASM/Declarative
  // after tie-resolution). Built-in is the monolithic catch-all below.
  std::unordered_map<std::string, OpResolution> op_by_key;   // owns key strings
  const OpHandler* builtin_op = nullptr;   // answers EVERY op; honest-unknown lives here
  std::vector<const ParserPlugin*> parsers;                  // sorted by priority()
  std::unordered_map<std::string, const PassPlugin*> passes; // display_name -> pass

  // Backing storage (owns everything the const-pointer views above point at).
  std::vector<std::unique_ptr<OpHandler>>     op_storage;
  std::vector<std::unique_ptr<ParserPlugin>>  parser_storage;
  std::vector<std::unique_ptr<PassPlugin>>    pass_storage;
};

class Registry {
 public:
  static Registry& instance();                              // Meyers singleton
  std::shared_ptr<const RegistryTable> snapshot() const;    // lock-free load

  // Two-tier op resolution (design Part 3.3): exact normalized-op key first, else
  // the builtin catch-all (which itself category-dispatches + honest-unknowns).
  OpResolution resolve_op(std::string_view op_norm) const;

  // Build an OpContext for a node. `mmap_base` non-null ONLY from the shape driver.
  OpContext make_context(const ir::Model&, const ir::Graph&, const ir::Node&,
                         std::string_view op_norm,
                         const uint8_t* mmap_base = nullptr,
                         uint64_t mmap_size = 0) const;

  // Content-sniff phase then a SEPARATE ext-hint fallback (design Part 4.4).
  Format detect_format(const MappedFile&, const std::string& ext_hint) const;
  const ParserPlugin* parser_for_format(Format) const;

  // Registration (used by built-in self-registration + plugin loaders). Refuses on
  // api_version mismatch: drops the handler, logs, never half-registers.
  void register_op_handler(std::string norm_key, std::string domain,
                           std::unique_ptr<OpHandler>, Origin, bool override_flag,
                           std::string plugin_name);
  void register_parser(std::unique_ptr<ParserPlugin>);
  void register_pass(std::unique_ptr<PassPlugin>);

  void reload(std::shared_ptr<const RegistryTable> next);   // atomic swap

 private:
  // Lock-free reads (UI + worker threads); rare atomic swap on load/reload.
  std::atomic<std::shared_ptr<const RegistryTable>> table_;
};

}  // namespace netvis::plugin
