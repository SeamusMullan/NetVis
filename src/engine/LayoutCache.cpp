// engine/LayoutCache.cpp — persistent per-file layout cache.
//
// DECISION (spec §2.7, §7.2.7): a layout is a pure function of (structure hash,
// collapse hash). We serialize the flat POD arrays (NodeBox/EdgeCurve + bounds +
// hashes) to <sh>_<ch>.nvl in a platform cache dir so reopening a file skips
// layout entirely. Structure-only key => a re-export with identical topology
// still hits. Corrupt/stale files are treated as a miss (recompute), never a
// crash: every field is bounds-checked against the declared counts.
#include "engine/LayoutCache.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace netvis {

namespace {

// File format + algorithm version. BUMP kVersion on ANY change to the layout
// struct OR the layout ALGORITHM — otherwise a stale cached layout is loaded and
// the new algorithm never runs (a changed layout would appear "identical").
//   v1: initial layered layout.
//   v2: constants/initializers pulled down next to their consumers.
constexpr uint32_t kMagic = 0x4C56454Eu;  // "NEVL" little-endian
constexpr uint32_t kVersion = 2;

// On-disk header. POD, written/read verbatim. All fields little-endian on the
// platforms we target (x86-64 / arm64); the cache is machine-local so we do not
// byte-swap — a mismatched machine simply misses and recomputes.
struct Header {
  uint32_t magic;
  uint32_t version;
  uint64_t structure_hash;
  uint64_t collapse_hash;
  uint64_t node_count;
  uint64_t edge_count;
  float bounds_min_x, bounds_min_y;
  float bounds_max_x, bounds_max_y;
};

std::string home_dir() {
  if (const char* h = std::getenv("HOME")) return h;
  return ".";
}

}  // namespace

// ---------------------------------------------------------------------------
// layout_cache_dir
// ---------------------------------------------------------------------------
std::string layout_cache_dir() {
  std::filesystem::path dir;
#if defined(_WIN32)
  if (const char* la = std::getenv("LOCALAPPDATA"))
    dir = std::filesystem::path(la) / "NetVis" / "cache";
  else
    dir = std::filesystem::path(".") / "NetVis" / "cache";
#elif defined(__APPLE__)
  dir = std::filesystem::path(home_dir()) / "Library" / "Caches" / "NetVis";
#else
  if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0] != '\0')
    dir = std::filesystem::path(xdg) / "netvis";
  else
    dir = std::filesystem::path(home_dir()) / ".cache" / "netvis";
#endif
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);  // best-effort
  return dir.string();
}

// ---------------------------------------------------------------------------
// load_cached_layout
// ---------------------------------------------------------------------------
Result<LayoutResult> load_cached_layout(uint64_t structure_hash,
                                        uint64_t collapse_hash) {
  std::filesystem::path path =
      std::filesystem::path(layout_cache_dir()) /
      (std::to_string(structure_hash) + "_" + std::to_string(collapse_hash) +
       ".nvl");

  std::ifstream f(path, std::ios::binary);
  if (!f) return err("layout cache miss: " + path.string());

  // Actual file size — the ONLY trustworthy bound on the array counts below.
  f.seekg(0, std::ios::end);
  const std::streamoff file_bytes = f.tellg();
  f.seekg(0, std::ios::beg);

  Header h{};
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!f || f.gcount() != static_cast<std::streamsize>(sizeof(h)))
    return err("layout cache: truncated header", 0);
  if (h.magic != kMagic) return err("layout cache: bad magic", 0);
  if (h.version != kVersion) return err("layout cache: version mismatch", 4);
  if (h.structure_hash != structure_hash || h.collapse_hash != collapse_hash)
    return err("layout cache: hash mismatch", 8);

  // SECURITY: a corrupt count field must not drive a multi-GB value-initialized
  // resize() (each NodeBox/EdgeCurve has default member initializers, so resize
  // commits real RSS before the truncated-read check ever fires). Bound the
  // counts by the bytes ACTUALLY present in the file: a genuine cache holds
  // node_count*sizeof(NodeBox)+edge_count*sizeof(EdgeCurve) payload bytes after
  // the header, so anything larger is corrupt. This makes the allocation
  // proportional to the file, not to an attacker-chosen integer.
  const uint64_t payload_bytes =
      (file_bytes > static_cast<std::streamoff>(sizeof(Header)))
          ? static_cast<uint64_t>(file_bytes) - sizeof(Header)
          : 0;
  if (h.node_count > payload_bytes / sizeof(NodeBox) ||
      h.edge_count > payload_bytes / sizeof(EdgeCurve) ||
      h.node_count * sizeof(NodeBox) + h.edge_count * sizeof(EdgeCurve) >
          payload_bytes)
    return err("layout cache: counts exceed file payload", 24);

  LayoutResult out;
  out.structure_hash = h.structure_hash;
  out.collapse_hash = h.collapse_hash;
  out.bounds_min = Vec2{h.bounds_min_x, h.bounds_min_y};
  out.bounds_max = Vec2{h.bounds_max_x, h.bounds_max_y};

  out.boxes.resize(static_cast<size_t>(h.node_count));
  if (h.node_count) {
    std::streamsize bytes =
        static_cast<std::streamsize>(h.node_count * sizeof(NodeBox));
    f.read(reinterpret_cast<char*>(out.boxes.data()), bytes);
    if (!f || f.gcount() != bytes)
      return err("layout cache: truncated node array", sizeof(Header));
  }

  out.edges.resize(static_cast<size_t>(h.edge_count));
  if (h.edge_count) {
    std::streamsize bytes =
        static_cast<std::streamsize>(h.edge_count * sizeof(EdgeCurve));
    f.read(reinterpret_cast<char*>(out.edges.data()), bytes);
    if (!f || f.gcount() != bytes)
      return err("layout cache: truncated edge array", 0);
  }

  out.from_cache = true;  // hit
  return out;
}

// ---------------------------------------------------------------------------
// store_cached_layout
// ---------------------------------------------------------------------------
Result<bool> store_cached_layout(const LayoutResult& layout) {
  std::filesystem::path dir(layout_cache_dir());
  std::filesystem::path path =
      dir / (std::to_string(layout.structure_hash) + "_" +
             std::to_string(layout.collapse_hash) + ".nvl");

  // Write to a temp file then atomically rename, so a crash mid-write never
  // leaves a half-written cache entry that would later fail validation.
  std::filesystem::path tmp = path;
  tmp += ".tmp";

  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return err("layout cache: cannot open for write: " + tmp.string());

    Header h{};
    h.magic = kMagic;
    h.version = kVersion;
    h.structure_hash = layout.structure_hash;
    h.collapse_hash = layout.collapse_hash;
    h.node_count = layout.boxes.size();
    h.edge_count = layout.edges.size();
    h.bounds_min_x = layout.bounds_min.x;
    h.bounds_min_y = layout.bounds_min.y;
    h.bounds_max_x = layout.bounds_max.x;
    h.bounds_max_y = layout.bounds_max.y;

    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    if (!layout.boxes.empty())
      f.write(reinterpret_cast<const char*>(layout.boxes.data()),
              static_cast<std::streamsize>(layout.boxes.size() *
                                           sizeof(NodeBox)));
    if (!layout.edges.empty())
      f.write(reinterpret_cast<const char*>(layout.edges.data()),
              static_cast<std::streamsize>(layout.edges.size() *
                                           sizeof(EdgeCurve)));
    if (!f) {
      f.close();
      std::error_code ec;
      std::filesystem::remove(tmp, ec);
      return err("layout cache: write failed");  // non-fatal to caller
    }
  }

  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(tmp, ec);
    return err("layout cache: rename failed: " + ec.message());
  }
  return true;
}

}  // namespace netvis
