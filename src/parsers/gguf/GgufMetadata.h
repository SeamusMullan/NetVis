// parsers/gguf/GgufMetadata.h — #44: well-known GGUF KV grouping (view-facing).
//
// The GGUF parser already drops EVERY key/value pair into Model::metadata
// (GgufParser.cpp; arrays summarized). The Properties panel renders that table
// verbatim. This helper is the #44 value-add: it pulls the important keys
// (arch / quant / rope / attention / context length / block count, …) out of
// that flat table into a small set of LABELLED, GROUPED summary rows so a reader
// sees the model's shape at a glance, with the full KV table still below.
//
// PURE + read-only over Model::metadata (StringId -> model.str()). Lives in the
// parser module (compiled into netvis_core) rather than src/view/ so it is unit
// testable — the headless test target links netvis_core, not the GUI. The GGUF
// architecture prefix (e.g. "llama.", "qwen2.") is dynamic, so classification is
// pattern-based (".attention.", ".rope.", the "general." namespace) rather than
// a hard-coded key list; unknown architectures still group correctly.
#pragma once

#include <string>
#include <vector>

#include "ir/IR.h"

namespace netvis::gguf {

// One summary row: a short label (the GGUF namespace/arch prefix stripped) and
// the resolved value string exactly as the parser stored it.
struct GgufSummaryItem {
  std::string label;
  std::string value;
};

// A titled bucket of related rows (e.g. "General", "Attention", "RoPE").
struct GgufSummaryGroup {
  std::string title;
  std::vector<GgufSummaryItem> items;
};

// Group the well-known GGUF metadata keys of `model` into labelled summary rows.
// Returns an empty vector when the model is not GGUF (format_name != "GGUF") or
// carries none of the recognised keys — so a caller can simply skip the section
// when the result is empty. Groups are emitted in a stable, readable order and
// empty groups are omitted; a key absent from the file just yields no row.
std::vector<GgufSummaryGroup> summarize_gguf_metadata(const ir::Model& model);

}  // namespace netvis::gguf
