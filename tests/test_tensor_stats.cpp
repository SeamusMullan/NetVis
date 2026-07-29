// tests/test_tensor_stats.cpp — compute_tensor_stats value tests (#46/#48/#51).
//
// Mirrors the test_npy.cpp harness: write a raw F32 payload to a temp file, map
// it, build a TensorRef over offset 0, and call compute_tensor_stats (the sole
// payload reader, spec §2.1). Covers argmin/argmax flat indices, per-output-
// channel stats, dead/NaN channel flags, the kMaxChannels cap, and whole-tensor
// all-zero. compute_tensor_stats calls mark_payload_read() exactly once; these
// tests add no extra reads of their own.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "core/MappedFile.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"

using namespace netvis;

namespace {

std::string temp_path(const std::string& stem) {
  return (std::filesystem::temp_directory_path() / ("nv_stats_" + stem)).string();
}

// Write F32 values to a temp file and map it (the "model" mmap).
MappedFile map_f32(const std::string& stem, const std::vector<float>& values) {
  std::string src = temp_path(stem);
  {
    std::ofstream out(src, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
  }
  auto mf = MappedFile::open(src);
  REQUIRE(mf);
  return std::move(*mf);
}

ir::TensorRef ref_f32(const std::vector<int64_t>& shape, uint64_t byte_len) {
  ir::TensorRef t;
  t.dtype = ir::DType::F32;
  for (int64_t d : shape) t.shape.push_back(d);
  t.file_offset = 0;
  t.byte_len = byte_len;
  return t;
}

}  // namespace

TEST_CASE("#51 argmin/argmax record the flat index of min and max") {
  // Unique min (-3.5 at index 2) and unique max (9.0 at index 5).
  const std::vector<float> v = {1.0f, 2.0f, -3.5f, 0.0f, 4.0f, 9.0f};
  MappedFile mf = map_f32("argmm", v);
  ir::TensorRef t = ref_f32({6}, v.size() * sizeof(float));

  auto stats = compute_tensor_stats(t, mf, "");
  REQUIRE_MESSAGE(stats, "compute_tensor_stats failed");
  CHECK(stats->min_index == 2);
  CHECK(stats->max_index == 5);
  CHECK(stats->min == doctest::Approx(-3.5));
  CHECK(stats->max == doctest::Approx(9.0));
}

TEST_CASE("#46 per-channel stats for a [3,4] tensor") {
  // 3 channels (rows) of 4 elements each, distinct per-row values.
  const std::vector<float> v = {
      1.0f, 2.0f, 3.0f, 4.0f,      // ch0: min 1 max 4 mean 2.5
      10.0f, 20.0f, 30.0f, 40.0f,  // ch1: min 10 max 40 mean 25
      -1.0f, -2.0f, -3.0f, -4.0f,  // ch2: min -4 max -1 mean -2.5
  };
  MappedFile mf = map_f32("perch", v);
  ir::TensorRef t = ref_f32({3, 4}, v.size() * sizeof(float));

  auto stats = compute_tensor_stats(t, mf, "");
  REQUIRE(stats);
  REQUIRE(stats->per_channel.size() == 3);
  CHECK_FALSE(stats->per_channel_capped);

  CHECK(stats->per_channel[0].min == doctest::Approx(1.0));
  CHECK(stats->per_channel[0].max == doctest::Approx(4.0));
  CHECK(stats->per_channel[0].mean == doctest::Approx(2.5));
  CHECK(stats->per_channel[0].count == 4);

  CHECK(stats->per_channel[1].min == doctest::Approx(10.0));
  CHECK(stats->per_channel[1].max == doctest::Approx(40.0));
  CHECK(stats->per_channel[1].mean == doctest::Approx(25.0));

  CHECK(stats->per_channel[2].min == doctest::Approx(-4.0));
  CHECK(stats->per_channel[2].max == doctest::Approx(-1.0));
  CHECK(stats->per_channel[2].mean == doctest::Approx(-2.5));
}

TEST_CASE("#48 dead channel: an all-zero row is flagged, a nonzero row is not") {
  // [2,4]: row 0 nonzero, row 1 all zeros.
  const std::vector<float> v = {
      1.0f, 2.0f, 3.0f, 4.0f,
      0.0f, 0.0f, 0.0f, 0.0f,
  };
  MappedFile mf = map_f32("dead", v);
  ir::TensorRef t = ref_f32({2, 4}, v.size() * sizeof(float));

  auto stats = compute_tensor_stats(t, mf, "");
  REQUIRE(stats);
  REQUIRE(stats->per_channel.size() == 2);
  CHECK_FALSE(stats->per_channel[0].all_zero());
  CHECK(stats->per_channel[1].all_zero());
  CHECK(stats->per_channel[1].zero_count == 4);
  CHECK_FALSE(stats->all_zero());  // whole tensor is not all zero
}

TEST_CASE("#48 NaN channel: flagged per-channel and whole-tensor") {
  // [2,3]: row 1 contains a NaN.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> v = {
      1.0f, 2.0f, 3.0f,
      4.0f, nan, 6.0f,
  };
  MappedFile mf = map_f32("nan", v);
  ir::TensorRef t = ref_f32({2, 3}, v.size() * sizeof(float));

  auto stats = compute_tensor_stats(t, mf, "");
  REQUIRE(stats);
  REQUIRE(stats->per_channel.size() == 2);
  CHECK_FALSE(stats->per_channel[0].has_nan_inf());
  CHECK(stats->per_channel[1].has_nan_inf());
  CHECK(stats->per_channel[1].nan_inf_count == 1);
  CHECK(stats->has_nan_inf());
  CHECK(stats->nan_inf_count == 1);
  // Channel 1's finite min/max ignore the NaN.
  CHECK(stats->per_channel[1].min == doctest::Approx(4.0));
  CHECK(stats->per_channel[1].max == doctest::Approx(6.0));
}

TEST_CASE("#46 cap: dim0 > kMaxChannels leaves per_channel empty + capped") {
  // [kMaxChannels+1, 1] — 4097 single-element channels. Buffer stays tiny.
  const uint32_t rows = kMaxChannels + 1;
  std::vector<float> v(rows, 1.0f);
  MappedFile mf = map_f32("cap", v);
  ir::TensorRef t = ref_f32({static_cast<int64_t>(rows), 1},
                            v.size() * sizeof(float));

  auto stats = compute_tensor_stats(t, mf, "");
  REQUIRE(stats);
  CHECK(stats->per_channel.empty());
  CHECK(stats->per_channel_capped);
  // Whole-tensor stats still computed.
  CHECK(stats->count == rows);
  CHECK(stats->min == doctest::Approx(1.0));
}

TEST_CASE("#48 all_zero whole tensor") {
  const std::vector<float> v(8, 0.0f);
  MappedFile mf = map_f32("allzero", v);
  ir::TensorRef t = ref_f32({8}, v.size() * sizeof(float));

  auto stats = compute_tensor_stats(t, mf, "");
  REQUIRE(stats);
  CHECK(stats->all_zero());
  CHECK(stats->zero_count == 8);
  CHECK(stats->count == 8);
  // 1-D with a single "channel" of dim0=8 is >=2 channels of 1 elem each: each
  // channel is a single zero, so every channel is dead.
  REQUIRE(stats->per_channel.size() == 8);
  for (const ChannelStat& c : stats->per_channel) CHECK(c.all_zero());
}

// --- #47 2D weight heatmap thumbnail --------------------------------------

TEST_CASE("#47 thumbnail: [4,4] gradient -> 4x4, blue->red corners") {
  // Row-major values 0..15: min 0 at pixel (0,0), max 15 at pixel (3,3).
  std::vector<float> v(16);
  for (int i = 0; i < 16; ++i) v[static_cast<size_t>(i)] = static_cast<float>(i);
  MappedFile mf = map_f32("thumb44", v);
  ir::TensorRef t = ref_f32({4, 4}, v.size() * sizeof(float));

  auto th = compute_tensor_thumbnail(t, mf, "");
  REQUIRE(th);
  CHECK(th->available);
  CHECK(th->width == 4);
  CHECK(th->height == 4);
  CHECK(th->rgba.size() == 4u * 4u * 4u);
  CHECK(th->slice_min == doctest::Approx(0.0));
  CHECK(th->slice_max == doctest::Approx(15.0));

  // Top-left pixel is the slice min (t=0) -> blue (0,0,255,255).
  CHECK(th->rgba[0] == 0);
  CHECK(th->rgba[1] == 0);
  CHECK(th->rgba[2] == 255);
  CHECK(th->rgba[3] == 255);
  // Bottom-right pixel (index 15) is the slice max (t=1) -> red (255,0,0,255).
  const size_t last = (3u * 4u + 3u) * 4u;  // == 60
  CHECK(th->rgba[last + 0] == 255);
  CHECK(th->rgba[last + 1] == 0);
  CHECK(th->rgba[last + 2] == 0);
  CHECK(th->rgba[last + 3] == 255);
}

TEST_CASE("#47 thumbnail: [1,3,8,8] uses the last two dims (first 2D slice)") {
  // 192 elements; only the first 8*8=64 (the first [8,8] slice) are imaged.
  std::vector<float> v(192);
  for (int i = 0; i < 192; ++i) v[static_cast<size_t>(i)] = static_cast<float>(i);
  MappedFile mf = map_f32("thumb1388", v);
  ir::TensorRef t = ref_f32({1, 3, 8, 8}, v.size() * sizeof(float));

  auto th = compute_tensor_thumbnail(t, mf, "");
  REQUIRE(th);
  CHECK(th->available);
  CHECK(th->width == 8);
  CHECK(th->height == 8);
  CHECK(th->rgba.size() == 8u * 8u * 4u);
  // Slice is elements [0,64): min 0, max 63 (not the whole-tensor max 191).
  CHECK(th->slice_min == doctest::Approx(0.0));
  CHECK(th->slice_max == doctest::Approx(63.0));
}

TEST_CASE("#47 thumbnail: oversized [256,256] block-averages to 128x128") {
  std::vector<float> v(256u * 256u);
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i);
  MappedFile mf = map_f32("thumb256", v);
  ir::TensorRef t = ref_f32({256, 256}, v.size() * sizeof(float));

  auto th = compute_tensor_thumbnail(t, mf, "");
  REQUIRE(th);
  CHECK(th->available);
  CHECK(th->width == kThumbMax);   // 128
  CHECK(th->height == kThumbMax);  // 128
  CHECK(th->rgba.size() == static_cast<size_t>(kThumbMax) * kThumbMax * 4u);
}

TEST_CASE("#47 thumbnail: rank<2 is unavailable (not an error)") {
  const std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
  MappedFile mf = map_f32("thumb1d", v);
  ir::TensorRef t = ref_f32({16}, v.size() * sizeof(float));

  auto th = compute_tensor_thumbnail(t, mf, "");
  REQUIRE(th);              // ok Result, not an error
  CHECK_FALSE(th->available);
  CHECK(th->width == 0);
  CHECK(th->height == 0);
  CHECK(th->rgba.empty());
}

TEST_CASE("#47 thumbnail: quantized / unknown dtype is unavailable") {
  const std::vector<float> v(16, 1.0f);
  MappedFile mf = map_f32("thumbq", v);

  // Quantized dtype -> unavailable.
  {
    ir::TensorRef t = ref_f32({4, 4}, v.size() * sizeof(float));
    t.dtype = ir::DType::Q8;
    auto th = compute_tensor_thumbnail(t, mf, "");
    REQUIRE(th);
    CHECK_FALSE(th->available);
  }
  // Unknown dtype (dtype_size == 0) -> unavailable.
  {
    ir::TensorRef t = ref_f32({4, 4}, v.size() * sizeof(float));
    t.dtype = ir::DType::Unknown;
    auto th = compute_tensor_thumbnail(t, mf, "");
    REQUIRE(th);
    CHECK_FALSE(th->available);
  }
}

TEST_CASE("#47 thumbnail: dynamic dim in the image plane is unavailable") {
  const std::vector<float> v(16, 1.0f);
  MappedFile mf = map_f32("thumbdyn", v);
  ir::TensorRef t = ref_f32({-1, 4}, v.size() * sizeof(float));

  auto th = compute_tensor_thumbnail(t, mf, "");
  REQUIRE(th);
  CHECK_FALSE(th->available);
}

TEST_CASE("#47 thumbnail: a block with only NaN/Inf becomes mid-gray") {
  // [2,2] where one whole cell is NaN. No downsampling (2<=128), so each cell is
  // a single element; the NaN cell has no finite value -> (128,128,128,255).
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<float> v = {1.0f, 2.0f, 3.0f, nan};
  MappedFile mf = map_f32("thumbnan", v);
  ir::TensorRef t = ref_f32({2, 2}, v.size() * sizeof(float));

  auto th = compute_tensor_thumbnail(t, mf, "");
  REQUIRE(th);
  CHECK(th->available);
  CHECK(th->width == 2);
  CHECK(th->height == 2);
  // Finite range is [1,3]; the NaN cell (pixel index 3) is mid-gray.
  const size_t nan_px = 3u * 4u;
  CHECK(th->rgba[nan_px + 0] == 128);
  CHECK(th->rgba[nan_px + 1] == 128);
  CHECK(th->rgba[nan_px + 2] == 128);
  CHECK(th->rgba[nan_px + 3] == 255);
}
