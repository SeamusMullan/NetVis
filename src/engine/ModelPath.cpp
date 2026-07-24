// engine/ModelPath.cpp — CoreML .mlpackage bundle -> inner model file resolution.
//
// See ModelPath.h. This is the ONLY place that turns a directory into a file to
// map; parsers never see a directory. Manifest parsing is best-effort and fully
// contained (a corrupt manifest falls back to the coremltools convention), and
// the resolved path is confined to the bundle root (no `..` escape).
#include "engine/ModelPath.h"

#include <filesystem>
#include <fstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace netvis {

namespace {

namespace fs = std::filesystem;

// coremltools conventions (verified against apple/coremltools):
//   Data/com.apple.CoreML/model.mlmodel   — inner spec
//   Data/com.apple.CoreML/weights/weight.bin — MIL blob storage
constexpr const char* kDataDir = "Data";
constexpr const char* kConventionalInner = "Data/com.apple.CoreML/model.mlmodel";

// True if `child`, lexically normalized, stays within `root` (no `..` escape).
// Both are treated as already-absolute-ish lexical paths; we compare normalized
// prefixes so a crafted Manifest path can never point outside the bundle.
bool within_root(const fs::path& root, const fs::path& child) {
  const fs::path nroot = root.lexically_normal();
  const fs::path nchild = child.lexically_normal();
  auto ri = nroot.begin();
  auto ci = nchild.begin();
  for (; ri != nroot.end(); ++ri, ++ci) {
    // A trailing slash on root yields a trailing empty component; stop there so a
    // legitimate inner path is not falsely rejected on a trailing-slash invocation.
    if (ri->empty()) break;
    if (ci == nchild.end() || *ci != *ri) return false;
  }
  return true;
}

// Resolve the inner model file for a .mlpackage bundle from its Manifest.json.
// Returns an empty string if the manifest cannot direct us to an existing,
// in-root file (the caller then tries the convention / gives up).
std::string inner_from_manifest(const fs::path& root) {
  std::error_code ec;
  const fs::path manifest = root / "Manifest.json";
  if (!fs::is_regular_file(manifest, ec)) return {};

  std::ifstream f(manifest);
  if (!f) return {};

  // Contained: a corrupt/huge/malformed manifest must never throw out of here.
  try {
    nlohmann::json j;
    f >> j;
    if (!j.is_object()) return {};

    const auto items = j.find("itemInfoEntries");
    if (items == j.end() || !items->is_object()) return {};

    // Prefer the declared root model; else accept the sole/first entry that
    // resolves to an existing in-root file whose name looks like a model spec.
    std::string root_id;
    if (auto rid = j.find("rootModelIdentifier");
        rid != j.end() && rid->is_string()) {
      root_id = rid->get<std::string>();
    }

    auto try_entry = [&](const nlohmann::json& entry) -> std::string {
      if (!entry.is_object()) return {};
      auto p = entry.find("path");
      if (p == entry.end() || !p->is_string()) return {};
      // The stored path is <author>/<name> (no Data/ prefix); prepend Data/.
      fs::path candidate = root / kDataDir / p->get<std::string>();
      if (!within_root(root, candidate)) return {};  // traversal guard
      std::error_code ec2;
      if (!fs::is_regular_file(candidate, ec2)) return {};
      return candidate.string();
    };

    if (!root_id.empty()) {
      if (auto e = items->find(root_id); e != items->end()) {
        std::string r = try_entry(*e);
        if (!r.empty()) return r;
      }
    }
    // Fallback: first entry that resolves to an existing .mlmodel file in-root.
    for (auto it = items->begin(); it != items->end(); ++it) {
      std::string r = try_entry(*it);
      if (!r.empty()) {
        fs::path rp(r);
        if (rp.extension() == ".mlmodel") return r;
      }
    }
  } catch (...) {
    return {};
  }
  return {};
}

}  // namespace

ResolvedModelPath resolve_model_path(const std::string& path) {
  ResolvedModelPath out{path, path};

  std::error_code ec;
  const fs::path p(path);
  if (!fs::is_directory(p, ec)) return out;  // plain file: pass through

  // Only treat a directory as a bundle when it has a Data/ dir (a .mlpackage
  // shape) — avoids grabbing an arbitrary directory the user dropped in.
  const fs::path root = p;
  if (!fs::is_directory(root / kDataDir, ec)) return out;

  // 1) Authoritative: Manifest.json -> rootModelIdentifier -> path.
  std::string inner = inner_from_manifest(root);
  // 2) Convention fallback.
  if (inner.empty()) {
    fs::path conv = root / kConventionalInner;
    if (within_root(root, conv) && fs::is_regular_file(conv, ec)) {
      inner = conv.string();
    }
  }

  if (!inner.empty()) out.map_path = inner;
  // If nothing resolved, map_path stays the directory; MappedFile::open will fail
  // with a clean error surfaced to the user (never a crash).
  return out;
}

}  // namespace netvis
