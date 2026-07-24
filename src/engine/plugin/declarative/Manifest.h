// engine/plugin/declarative/Manifest.h — declarative plugin loader (v0.6.0 #9).
//
// Loads a JSONC plugin manifest (one `plugin.json` per plugin dir), compiles each
// [[op]] block's flops / vars / shape expressions ONCE, and registers a
// DeclarativeOpHandler per op into the plugin Registry. Uses the already-linked
// nlohmann/json (no new dependency, design §Q1). A malformed manifest is rejected
// with a legible message; a bad op is skipped; never throws, never crashes.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "engine/plugin/PluginPrefs.h"   // PluginKind

namespace netvis::plugin {

// Result of loading a manifest file (for the Plugins panel #11 + diagnostics).
struct LoadedOp {
  std::string op_name;        // normalized key
  std::string category;       // resolved category name (or "" if defaulted)
  bool overrides_builtin = false;
  bool ok = false;            // false => this op was rejected (see error)
  std::string error;
};
struct LoadedManifest {
  std::string path;
  std::string name;           // manifest "name"
  std::string author;
  uint32_t api_version = 0;
  bool ok = false;            // false => whole file rejected (api mismatch/parse)
  std::string error;
  std::vector<LoadedOp> ops;
  // v0.7.0 (#11, Increment C): discovery/trust metadata for the Plugins panel.
  std::string id;             // discovery subdir name — the STABLE enable-key
  PluginKind kind = PluginKind::Declarative;   // Declarative | Wasm (trust tier)
  bool enabled = true;        // effective enable state (gate result this session)
  bool registered = false;    // did it actually reach the Registry?
};

// Gate consulted ONCE per discovered plugin (#11 §0.4): given the stable id + kind,
// return whether it should be REGISTERED. A disabled plugin is still discovered (its
// diagnostics populate the panel) but never reaches the Registry — so a disabled
// WASM parser's can_parse (which executes code) never runs. Passed by const& (no
// per-node std::function copy).
using PluginGate = std::function<bool(std::string_view id, PluginKind kind)>;

// Parse + compile + REGISTER every op in a manifest file into the global Registry.
// Returns a diagnostic record (never throws). If `register_into` is false, only
// parses/compiles for validation (dry run) and registers nothing.
LoadedManifest load_manifest_file(const std::string& path, bool register_into);

// The directory declarative plugins are discovered from: a CONFIG-tree sibling of
// layout_cache_dir() (NOT the wipeable cache tree). Created best-effort.
std::string plugin_dir();

// Discover + load every `<plugin_dir>/*/plugin.json`, deterministic path order.
// Returns one LoadedManifest per file, and caches them for the panel. The gate
// decides per-plugin registration (§0.4): a plugin failing the gate is discovered
// (diagnostics filled) but NOT registered — structurally absent from the Registry.
// A .wasm sidecar (op or parser plugin) is discovered here too; it is registered
// only when the gate allows AND the WASM engine is enabled.
//
// Overloads: the no-arg form registers everything (legacy/back-compat); the gated
// form applies the trust gate; the (gate, root) form also overrides the discovery
// root (testable without touching the user config dir).
std::vector<LoadedManifest> discover_and_load_plugins();
std::vector<LoadedManifest> discover_and_load_plugins(const PluginGate& gate);
std::vector<LoadedManifest> discover_and_load_plugins(const PluginGate& gate,
                                                      const std::string& root);

// The manifests loaded by the most recent discover_and_load_plugins() call — the
// Plugins panel (#11) reads this to list plugins, their ops, overrides, and any
// rejection reason. Empty until discovery runs.
const std::vector<LoadedManifest>& loaded_manifests();

}  // namespace netvis::plugin
