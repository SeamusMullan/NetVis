// engine/plugin/declarative/Manifest.h — declarative plugin loader (v0.6.0 #9).
//
// Loads a JSONC plugin manifest (one `plugin.json` per plugin dir), compiles each
// [[op]] block's flops / vars / shape expressions ONCE, and registers a
// DeclarativeOpHandler per op into the plugin Registry. Uses the already-linked
// nlohmann/json (no new dependency, design §Q1). A malformed manifest is rejected
// with a legible message; a bad op is skipped; never throws, never crashes.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

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
};

// Parse + compile + REGISTER every op in a manifest file into the global Registry.
// Returns a diagnostic record (never throws). If `register_into` is false, only
// parses/compiles for validation (dry run) and registers nothing.
LoadedManifest load_manifest_file(const std::string& path, bool register_into);

// The directory declarative plugins are discovered from: a CONFIG-tree sibling of
// layout_cache_dir() (NOT the wipeable cache tree). Created best-effort.
std::string plugin_dir();

// Discover + load every `<plugin_dir>/*/plugin.json`, deterministic path order.
// Returns one LoadedManifest per file. Declarative plugins load freely (safe by
// construction); enable/disable filtering (#11) is applied by the caller. The
// results are also cached process-wide for the Plugins panel (loaded_manifests()).
std::vector<LoadedManifest> discover_and_load_plugins();

// The manifests loaded by the most recent discover_and_load_plugins() call — the
// Plugins panel (#11) reads this to list plugins, their ops, overrides, and any
// rejection reason. Empty until discovery runs.
const std::vector<LoadedManifest>& loaded_manifests();

}  // namespace netvis::plugin
