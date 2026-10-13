// SPDX-License-Identifier: Apache-2.0
// engine/plugin/AbiGate.h — the manifest `api_version` gate, shared by every loader.
//
// plugin.json declares the ABI version the plugin was written for, and the loader
// must refuse the whole file before constructing anything when that is not the
// version this host speaks (docs/plugin-abi.md, "How a mismatch is refused").
// Three loaders read that key - the declarative one and the two WASM ones - and
// they used to each do it their own way: the WASM loaders only compared when the
// key was present and an integer, so a missing key, a string, a float, or an
// unsigned value that wraps when narrowed (4294967297 -> 1) all passed. One
// strict reader, so the three cannot drift apart again.
#pragma once

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace netvis::plugin {

struct ApiVersionCheck {
  bool ok = false;
  bool present = false;        // the key exists with an unsigned-integer value
  uint64_t declared = 0;       // the value, full width (never narrowed)
  std::string error;           // diagnostic when !ok
};

// The key must be present, an unsigned JSON integer, and EXACTLY `host_version`,
// compared at full 64-bit width.
inline ApiVersionCheck check_manifest_api_version(const nlohmann::json& manifest,
                                                  uint32_t host_version) {
  ApiVersionCheck c;
  const auto it = manifest.find("api_version");
  if (it == manifest.end()) {
    c.error = "api_version missing (this NetVis speaks plugin ABI v" +
              std::to_string(host_version) + ")";
    return c;
  }
  if (!it->is_number_unsigned()) {
    c.error = "api_version must be an unsigned integer (this NetVis speaks plugin ABI v" +
              std::to_string(host_version) + ")";
    return c;
  }
  c.present = true;
  c.declared = it->get<uint64_t>();
  if (c.declared != static_cast<uint64_t>(host_version)) {
    c.error = "api_version " + std::to_string(c.declared) + " != host " +
              std::to_string(host_version);
    return c;
  }
  c.ok = true;
  return c;
}

}  // namespace netvis::plugin
