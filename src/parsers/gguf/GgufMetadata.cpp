// parsers/gguf/GgufMetadata.cpp — #44 well-known GGUF KV grouping (see header).
#include "parsers/gguf/GgufMetadata.h"

#include <string_view>

namespace netvis::gguf {
namespace {

// Scan the flat KV table for an exact key; append a row to `g` if present.
// `label` is what the reader sees (the namespace prefix already stripped by the
// caller). Absent keys simply add nothing — graceful fallback per #44.
void add_if_present(const ir::Model& model, GgufSummaryGroup& g,
                    std::string_view key, std::string_view label) {
  for (const auto& [k, v] : model.metadata) {
    if (model.str(k) == key) {
      g.items.push_back({std::string(label), std::string(model.str(v))});
      return;  // keys are unique in a GGUF file; first match wins
    }
  }
}

// Scan the KV table for the (single) key ending in ".<suffix>" — the arch prefix
// (llama./qwen2./gemma2./…) is dynamic, so we match by suffix, not prefix. The
// label is the suffix itself. Adds nothing if no arch key carries that suffix.
void add_arch_suffix(const ir::Model& model, GgufSummaryGroup& g,
                     std::string_view suffix) {
  const std::string dotted = "." + std::string(suffix);
  for (const auto& [k, v] : model.metadata) {
    std::string_view key = model.str(k);
    if (key.size() > dotted.size() && key.ends_with(dotted)) {
      g.items.push_back({std::string(suffix), std::string(model.str(v))});
      return;
    }
  }
}

}  // namespace

std::vector<GgufSummaryGroup> summarize_gguf_metadata(const ir::Model& model) {
  std::vector<GgufSummaryGroup> out;
  if (model.str(model.format_name) != "GGUF") return out;

  // --- General: curated general.* keys in a readable order. -----------------
  GgufSummaryGroup general{"General", {}};
  {
    // {full key, display label}. Ordered most-identifying first.
    static const std::pair<const char*, const char*> kGeneral[] = {
        {"general.name", "name"},
        {"general.architecture", "architecture"},
        {"general.quantization_version", "quantization_version"},
        {"general.file_type", "file_type"},
        {"general.size_label", "size_label"},
        {"general.basename", "basename"},
        {"general.finetune", "finetune"},
        {"general.author", "author"},
        {"general.organization", "organization"},
        {"general.license", "license"},
        {"general.version", "version"},
        {"general.description", "description"},
    };
    for (const auto& [key, label] : kGeneral)
      add_if_present(model, general, key, label);
  }

  // --- Architecture: prefix-agnostic hyperparameters (suffix match). --------
  GgufSummaryGroup arch{"Architecture", {}};
  {
    static const char* kArchSuffixes[] = {
        "context_length",      "block_count",  "embedding_length",
        "feed_forward_length", "vocab_size",   "expert_count",
        "expert_used_count",
    };
    for (const char* s : kArchSuffixes) add_arch_suffix(model, arch, s);
  }

  // --- Attention + RoPE: any key carrying the ".attention." / ".rope."
  // segment, in file order (GGUF already emits them contiguously). Label is the
  // path after the segment (e.g. "head_count", "scaling.type"). ------------
  GgufSummaryGroup attn{"Attention", {}};
  GgufSummaryGroup rope{"RoPE", {}};
  {
    constexpr std::string_view kAttn = ".attention.";
    constexpr std::string_view kRope = ".rope.";
    for (const auto& [k, v] : model.metadata) {
      std::string_view key = model.str(k);
      if (auto p = key.find(kAttn); p != std::string_view::npos) {
        std::string_view label = key.substr(p + kAttn.size());
        attn.items.push_back({std::string(label), std::string(model.str(v))});
      } else if (auto q = key.find(kRope); q != std::string_view::npos) {
        std::string_view label = key.substr(q + kRope.size());
        rope.items.push_back({std::string(label), std::string(model.str(v))});
      }
    }
  }

  // --- Tokenizer: curated scalar keys only. The token/merge/score arrays are
  // huge and already summarized in the full KV table; they are not a legible
  // "summary", so we surface just the model type + special-token ids. --------
  GgufSummaryGroup tok{"Tokenizer", {}};
  {
    static const std::pair<const char*, const char*> kTokenizer[] = {
        {"tokenizer.ggml.model", "model"},
        {"tokenizer.ggml.pre", "pre"},
        {"tokenizer.ggml.bos_token_id", "bos_token_id"},
        {"tokenizer.ggml.eos_token_id", "eos_token_id"},
        {"tokenizer.ggml.unknown_token_id", "unknown_token_id"},
        {"tokenizer.ggml.padding_token_id", "padding_token_id"},
        {"tokenizer.ggml.add_bos_token", "add_bos_token"},
        {"tokenizer.ggml.add_eos_token", "add_eos_token"},
    };
    for (const auto& [key, label] : kTokenizer)
      add_if_present(model, tok, key, label);
  }

  // Emit non-empty groups in a stable, readable order.
  for (GgufSummaryGroup* g : {&general, &arch, &attn, &rope, &tok})
    if (!g->items.empty()) out.push_back(std::move(*g));
  return out;
}

}  // namespace netvis::gguf
