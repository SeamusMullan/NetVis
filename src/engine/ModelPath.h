// engine/ModelPath.h — resolve a user-opened path to the single file to mmap.
//
// DECISION (#85): a CoreML `.mlpackage` is a DIRECTORY bundle, but the whole
// pipeline maps one file (MappedFile) and every parser takes `const MappedFile&`.
// Rather than teach parsers about directories, we resolve the bundle to its inner
// model file HERE, at the open layer. The inner file's directory then becomes the
// session's model_dir(), so the existing external-data plumbing (resolve_payload)
// finds weights/weight.bin with no parser changes. Non-bundle paths pass through.
#pragma once

#include <string>

namespace netvis {

struct ResolvedModelPath {
  std::string display_path;  // what the user opened (the .mlpackage dir, or file)
  std::string map_path;      // the single file to mmap (inner model for a bundle)
};

// If `path` is a `.mlpackage` directory: read Manifest.json, resolve
// rootModelIdentifier -> itemInfoEntries[uuid].path -> <root>/Data/<path>. Falls
// back to the conventional Data/com.apple.CoreML/model.mlmodel when the manifest
// is missing/unreadable/malformed. Path-traversal guarded: a resolved map_path
// that escapes the bundle root is rejected (falls back / returns the dir itself
// so the CoreML parser can emit an honest error). Any other path (a plain file,
// or a directory that is not a resolvable bundle) yields map_path == display_path.
ResolvedModelPath resolve_model_path(const std::string& path);

}  // namespace netvis
