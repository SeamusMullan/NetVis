// engine/plugin/PluginPrefs.cpp — see PluginPrefs.h.
#include "engine/plugin/PluginPrefs.h"

#include <nlohmann/json.hpp>

namespace netvis::plugin {

bool plugin_default_enabled(PluginKind k) {
  return k == PluginKind::Declarative;   // declarative on, WASM off
}

bool PluginEnableSet::effective(std::string_view id, PluginKind k) const {
  auto it = explicit_.find(std::string(id));
  if (it != explicit_.end()) return it->second;
  return plugin_default_enabled(k);
}

void PluginEnableSet::set(std::string id, bool on) {
  explicit_[std::move(id)] = on;
}

bool PluginEnableSet::has_explicit(std::string_view id) const {
  return explicit_.find(std::string(id)) != explicit_.end();
}

void PluginEnableSet::load_json(const nlohmann::json& o) {
  if (!o.is_object()) return;   // ignore garbage, never throw
  for (auto it = o.begin(); it != o.end(); ++it) {
    if (it.value().is_boolean()) explicit_[it.key()] = it.value().get<bool>();
    // non-bool entries silently ignored (forward-compatible)
  }
}

nlohmann::json PluginEnableSet::to_json() const {
  nlohmann::json o = nlohmann::json::object();
  for (const auto& [id, on] : explicit_) o[id] = on;   // std::map => sorted keys
  return o;
}

}  // namespace netvis::plugin
