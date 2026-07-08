// engine/LayoutCache.h — persistent per-file layout cache.
//
// DECISION (spec §2.7, §7.2.7): same file -> same layout, so we serialize node
// positions keyed by (structure hash, collapse hash) into a local cache dir.
// Reopening a file skips layout entirely. The key hashes STRUCTURE only, never
// weights, so a re-export with identical topology still hits the cache.
#pragma once

#include <cstdint>
#include <string>

#include "core/Result.h"
#include "engine/Layout.h"

namespace netvis {

// Returns the platform cache directory: %LOCALAPPDATA%/NetVis/cache on Windows,
// $XDG_CACHE_HOME/netvis or ~/.cache/netvis on Linux, ~/Library/Caches/NetVis on
// macOS. Created on first use.
std::string layout_cache_dir();

// Load a cached layout for (structure_hash, collapse_hash). Returns an error if
// absent or the cache file is stale/corrupt (caller then recomputes).
Result<LayoutResult> load_cached_layout(uint64_t structure_hash,
                                        uint64_t collapse_hash);

// Persist a layout. Best-effort: a write failure is non-fatal (returns error but
// the app keeps working with the in-memory layout).
Result<bool> store_cached_layout(const LayoutResult& layout);

}  // namespace netvis
