// engine/plugin/PluginPrefs.h — per-plugin enable/disable state (#11, Increment C).
//
// Persisted in view_prefs.json under a "plugins" object keyed by the discovery
// SUBDIRECTORY name (stable/unique; the manifest "name" may be empty or duplicated,
// so it is NOT the key). Trust model: declarative plugins default ENABLED (safe by
// construction — a bounded expression DSL); WASM plugins default DISABLED (arbitrary
// sandboxed code — explicit per-plugin opt-in behind a confirm dialog).
//
// Lives in CORE (not view) so it is headless-testable: App is view-only and not
// linked by the test target. Uses <nlohmann/json_fwd.hpp> so this header stays light.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

#include <nlohmann/json_fwd.hpp>

namespace netvis::plugin {

enum class PluginKind : uint8_t { Declarative = 0, Wasm = 1 };

// The safe default for a kind absent from the persisted set.
bool plugin_default_enabled(PluginKind k);   // Declarative => true, Wasm => false

// A set of explicit per-plugin enable overrides. Absence of a key => the kind's
// safe default. NEVER prunes entries for plugins not present this session (a plugin
// removed then re-added keeps its prior choice). std::map => deterministic JSON order.
class PluginEnableSet {
 public:
  // Effective state: an explicit override if present, else the kind default.
  bool effective(std::string_view id, PluginKind k) const;
  void set(std::string id, bool on);
  bool has_explicit(std::string_view id) const;

  // Load per-key booleans from a JSON object; non-bool / non-object entries are
  // ignored (never throws). Merges into (does NOT clear) the existing set.
  void load_json(const nlohmann::json& o);
  // Serialize the explicit overrides (deterministic key order).
  nlohmann::json to_json() const;

  bool empty() const { return explicit_.empty(); }

 private:
  std::map<std::string, bool> explicit_;
};

}  // namespace netvis::plugin
