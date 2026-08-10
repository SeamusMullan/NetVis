// tests/test_quant_preview.cpp — engine::preview_quant_block, the #49
// bounded single-block dequant preview (engine/TensorStats.h).
//
// Uses tests/fixtures/model_quant.gguf (tools/gen_fixtures.py:build_gguf_quant),
// a real PARSED GGUF model rather than a hand-built ir::Model, because
// preview_quant_block's first gate is `model->format_name == "GGUF"` — the
// only honest way to exercise that is through the real parser. The fixture
// carries three tensors: weight_f32 (not quantized), weight_q4k (a K-quant,
// refused by scope), weight_q4_0 (two hand-encoded blocks with KNOWN values,
// block 1's scale doubled so a block_index mixup — always decoding block 0 —
// would also be caught).
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>

#include "core/ByteReader.h"
#include "core/MappedFile.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {
const char* kFixture = "tests/fixtures/model_quant.gguf";

const ir::TensorRef* find_flat(const ir::Model& m, std::string_view name) {
  for (const ir::TensorRef& t : m.flat_tensors) {
    if (m.str(t.name) == name) return &t;
  }
  return nullptr;
}

// Open + parse the fixture, mirroring the map_f32()/REQUIRE idiom in
// test_tensor_stats.cpp. Returns false (having already WARN'd) if the fixture
// hasn't been generated, the same skip convention test_gguf.cpp uses.
bool load_fixture(MappedFile& mf, ir::Model& model) {
  if (!std::filesystem::exists(kFixture)) {
    WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
    return false;
  }
  auto opened = MappedFile::open(kFixture);
  REQUIRE(opened);
  mf = std::move(*opened);
  ProgressSink progress;
  auto res = gguf::parse(mf, progress);
  REQUIRE_MESSAGE(res, "gguf::parse returned an error");
  model = std::move(*res);
  return true;
}

}  // namespace

TEST_CASE("#49 GGUF quant fixture: structural parse reads zero payload") {
  MappedFile mf;
  ir::Model model;
  ByteReader::payload_read_counter() = 0;
  if (!load_fixture(mf, model)) return;
  // The #49 additions to GgufParser.cpp (recording quant_type_id/dtype_label
  // per tensor) read only the already-parsed tensor-info table, never the
  // data section — so structural parse must still leave the counter at 0
  // (spec §2.1), exactly like the plain model.gguf fixture in test_gguf.cpp.
  CHECK(ByteReader::payload_read_counter() == 0);

  CHECK_FALSE(model.has_graph);
  REQUIRE(model.flat_tensors.size() == 3);
  CHECK(find_flat(model, "weight_f32") != nullptr);
  CHECK(find_flat(model, "weight_q4k") != nullptr);
  CHECK(find_flat(model, "weight_q4_0") != nullptr);
}

TEST_CASE("#49 preview_quant_block: Q4_0 block 0 matches the fixture's hand-encoded ramp") {
  MappedFile mf;
  ir::Model model;
  if (!load_fixture(mf, model)) return;
  const ir::TensorRef* q4_0 = find_flat(model, "weight_q4_0");
  REQUIRE(q4_0 != nullptr);

  // The counting ByteReader only tracks the NUMBER of mark_payload_read()
  // calls, not bytes touched, so this is the strongest zero-payload-style
  // assertion available here: a single preview call must mark exactly ONE
  // payload read — the same convention compute_tensor_stats uses — never a
  // second, which would signal the preview fell back to scanning the tensor
  // instead of touching just the one requested block.
  ByteReader::payload_read_counter() = 0;
  auto pr = preview_quant_block(*q4_0, mf, 0, "", &model);
  REQUIRE_MESSAGE(pr, "preview_quant_block returned an error");
  CHECK(ByteReader::payload_read_counter() == 1);

  const QuantBlockPreview& p = *pr;
  CHECK(p.available);
  CHECK(p.type_name == "Q4_0");
  CHECK(p.block_index == 0);
  CHECK(p.total_blocks == 2);
  CHECK(p.elem_count == 32);
  CHECK(p.first_elem == 0);
  CHECK(p.unavailable_reason.empty());
  // Fixture encoding (_q4_0_block in gen_fixtures.py): low nibble of qs[j] = j
  // (element j = j-8), high nibble = 15-j (element j+16 = 7-j), scale d=1.0.
  for (uint32_t j = 0; j < 16; ++j) {
    CHECK(p.values[j] == static_cast<float>(j) - 8.0f);
    CHECK(p.values[j + 16] == 7.0f - static_cast<float>(j));
  }
}

TEST_CASE("#49 preview_quant_block: Q4_0 block 1 (the LAST valid block) is selected correctly") {
  MappedFile mf;
  ir::Model model;
  if (!load_fixture(mf, model)) return;
  const ir::TensorRef* q4_0 = find_flat(model, "weight_q4_0");
  REQUIRE(q4_0 != nullptr);

  auto pr = preview_quant_block(*q4_0, mf, 1, "", &model);
  REQUIRE_MESSAGE(pr, "preview_quant_block returned an error");
  const QuantBlockPreview& p = *pr;
  CHECK(p.available);
  CHECK(p.block_index == 1);
  CHECK(p.total_blocks == 2);
  CHECK(p.elem_count == 32);
  CHECK(p.first_elem == 32);  // block_index * elems_per_block
  // Same qs ramp as block 0, but scale d=2.0 in the fixture: a block_index
  // bug that always decodes block 0 would produce block 0's UNSCALED values
  // here instead of these doubled ones.
  for (uint32_t j = 0; j < 16; ++j) {
    CHECK(p.values[j] == (static_cast<float>(j) - 8.0f) * 2.0f);
    CHECK(p.values[j + 16] == (7.0f - static_cast<float>(j)) * 2.0f);
  }
}

TEST_CASE("#49 preview_quant_block: one past the end is unavailable, NOT an error") {
  MappedFile mf;
  ir::Model model;
  if (!load_fixture(mf, model)) return;
  const ir::TensorRef* q4_0 = find_flat(model, "weight_q4_0");
  REQUIRE(q4_0 != nullptr);

  // weight_q4_0 has exactly 2 blocks (checked above); index 2 is one past.
  auto pr = preview_quant_block(*q4_0, mf, 2, "", &model);
  REQUIRE_MESSAGE(pr, "an out-of-range block_index must be a successful "
                      "Result with available=false, never an error");
  const QuantBlockPreview& p = *pr;
  CHECK_FALSE(p.available);
  CHECK_FALSE(p.unavailable_reason.empty());
}

TEST_CASE("#49 preview_quant_block: a K-quant (Q4_K) is refused by scope, but still names itself") {
  MappedFile mf;
  ir::Model model;
  if (!load_fixture(mf, model)) return;
  const ir::TensorRef* q4k = find_flat(model, "weight_q4k");
  REQUIRE(q4k != nullptr);

  auto pr = preview_quant_block(*q4k, mf, 0, "", &model);
  REQUIRE_MESSAGE(pr, "preview_quant_block returned an error");
  const QuantBlockPreview& p = *pr;
  CHECK_FALSE(p.available);
  CHECK(p.type_name == "Q4_K");  // honest label even though decode is refused
  CHECK_FALSE(p.unavailable_reason.empty());
}

TEST_CASE("#49 preview_quant_block: a non-quantized tensor (F32) is refused as not-quantized") {
  MappedFile mf;
  ir::Model model;
  if (!load_fixture(mf, model)) return;
  const ir::TensorRef* f32 = find_flat(model, "weight_f32");
  REQUIRE(f32 != nullptr);

  auto pr = preview_quant_block(*f32, mf, 0, "", &model);
  REQUIRE_MESSAGE(pr, "preview_quant_block returned an error");
  const QuantBlockPreview& p = *pr;
  CHECK_FALSE(p.available);
  CHECK_FALSE(p.unavailable_reason.empty());
}
