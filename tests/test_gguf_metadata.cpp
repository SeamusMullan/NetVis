// tests/test_gguf_metadata.cpp — #44 GGUF well-known KV grouping.
//
// summarize_gguf_metadata is PURE over an ir::Model (StringId -> str()), so
// these tests hand-build a tiny GGUF-style model in code (no file dependency)
// and assert the grouping: general.* curated, arch hyperparameters matched by
// suffix (prefix-agnostic), ".attention." / ".rope." keys bucketed, graceful
// fallback when keys are absent, and an empty result for non-GGUF models.
#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "ir/IR.h"
#include "parsers/gguf/GgufMetadata.h"

using namespace netvis;

namespace {

// Helper: append a (key,value) pair to a model's flat metadata table.
void put(ir::Model& m, const char* k, const char* v) {
  m.metadata.emplace_back(m.intern(k), m.intern(v));
}

// Helper: find a group by title, or nullptr.
const gguf::GgufSummaryGroup* find_group(
    const std::vector<gguf::GgufSummaryGroup>& groups, const char* title) {
  for (const auto& g : groups)
    if (g.title == title) return &g;
  return nullptr;
}

// Helper: value for a label within a group, or empty string.
std::string value_of(const gguf::GgufSummaryGroup& g, const char* label) {
  for (const auto& it : g.items)
    if (it.label == label) return it.value;
  return {};
}

}  // namespace

TEST_CASE("GGUF summary: non-GGUF model yields nothing") {
  ir::Model m;
  m.format_name = m.intern("ONNX");
  put(m, "general.architecture", "resnet");  // even a lookalike key
  CHECK(gguf::summarize_gguf_metadata(m).empty());
}

TEST_CASE("GGUF summary: empty when no recognised keys present") {
  ir::Model m;
  m.format_name = m.intern("GGUF");
  put(m, "some.random.key", "x");
  put(m, "another_key", "y");
  CHECK(gguf::summarize_gguf_metadata(m).empty());
}

TEST_CASE("GGUF summary: general + arch (suffix) + attention + rope grouped") {
  ir::Model m;
  m.format_name = m.intern("GGUF");
  // General namespace (curated, exact keys).
  put(m, "general.name", "TinyLlama");
  put(m, "general.architecture", "llama");
  put(m, "general.quantization_version", "2");
  put(m, "general.file_type", "15");
  // Architecture hyperparameters — arch prefix is dynamic ("llama."); matched
  // by suffix, so the label is just the suffix.
  put(m, "llama.context_length", "4096");
  put(m, "llama.block_count", "22");
  put(m, "llama.embedding_length", "2048");
  // Attention + RoPE segments.
  put(m, "llama.attention.head_count", "32");
  put(m, "llama.attention.head_count_kv", "4");
  put(m, "llama.attention.layer_norm_rms_epsilon", "0.00001");
  put(m, "llama.rope.dimension_count", "64");
  put(m, "llama.rope.freq_base", "10000.0");
  // Tokenizer curated scalars (+ a huge array that must NOT appear as a row).
  put(m, "tokenizer.ggml.model", "llama");
  put(m, "tokenizer.ggml.bos_token_id", "1");
  put(m, "tokenizer.ggml.tokens", "[a, b, c, ...] (32000 items)");

  auto groups = gguf::summarize_gguf_metadata(m);
  REQUIRE(!groups.empty());

  const auto* general = find_group(groups, "General");
  REQUIRE(general != nullptr);
  CHECK(value_of(*general, "name") == "TinyLlama");
  CHECK(value_of(*general, "architecture") == "llama");
  CHECK(value_of(*general, "quantization_version") == "2");

  const auto* arch = find_group(groups, "Architecture");
  REQUIRE(arch != nullptr);
  // Label is the SUFFIX, not the prefixed key.
  CHECK(value_of(*arch, "context_length") == "4096");
  CHECK(value_of(*arch, "block_count") == "22");
  CHECK(value_of(*arch, "embedding_length") == "2048");

  const auto* attn = find_group(groups, "Attention");
  REQUIRE(attn != nullptr);
  CHECK(value_of(*attn, "head_count") == "32");
  CHECK(value_of(*attn, "head_count_kv") == "4");
  CHECK(attn->items.size() == 3);

  const auto* rope = find_group(groups, "RoPE");
  REQUIRE(rope != nullptr);
  CHECK(value_of(*rope, "dimension_count") == "64");
  CHECK(value_of(*rope, "freq_base") == "10000.0");

  const auto* tok = find_group(groups, "Tokenizer");
  REQUIRE(tok != nullptr);
  CHECK(value_of(*tok, "model") == "llama");
  CHECK(value_of(*tok, "bos_token_id") == "1");
  // The vocab array must be excluded from the curated summary.
  CHECK(value_of(*tok, "tokens").empty());
}

TEST_CASE("GGUF summary: prefix-agnostic — non-llama arch still groups") {
  ir::Model m;
  m.format_name = m.intern("GGUF");
  put(m, "general.architecture", "qwen2");
  put(m, "qwen2.context_length", "32768");
  put(m, "qwen2.attention.head_count", "28");
  put(m, "qwen2.rope.freq_base", "1000000.0");

  auto groups = gguf::summarize_gguf_metadata(m);
  const auto* arch = find_group(groups, "Architecture");
  REQUIRE(arch != nullptr);
  CHECK(value_of(*arch, "context_length") == "32768");
  const auto* attn = find_group(groups, "Attention");
  REQUIRE(attn != nullptr);
  CHECK(value_of(*attn, "head_count") == "28");
  const auto* rope = find_group(groups, "RoPE");
  REQUIRE(rope != nullptr);
  CHECK(value_of(*rope, "freq_base") == "1000000.0");
}

TEST_CASE("GGUF summary: absent keys are skipped, empty groups omitted") {
  ir::Model m;
  m.format_name = m.intern("GGUF");
  // Only a couple general keys; no arch/attention/rope/tokenizer at all.
  put(m, "general.name", "solo");
  put(m, "general.architecture", "phi3");

  auto groups = gguf::summarize_gguf_metadata(m);
  // Exactly one group ("General"); the others are empty and omitted.
  REQUIRE(groups.size() == 1);
  CHECK(groups[0].title == "General");
  CHECK(groups[0].items.size() == 2);
  CHECK(find_group(groups, "Attention") == nullptr);
  CHECK(find_group(groups, "RoPE") == nullptr);
  CHECK(find_group(groups, "Architecture") == nullptr);
  CHECK(find_group(groups, "Tokenizer") == nullptr);
}
