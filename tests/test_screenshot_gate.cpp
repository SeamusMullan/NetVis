// SPDX-License-Identifier: Apache-2.0
// tests/test_screenshot_gate.cpp — the `--screenshot` capture gate, driven against
// the REAL ModelSession + JobSystem on real fixtures (#170).
//
// The capture driver (view/AppScreenshot.cpp) loops
//     session.update(); gate = capture_gate({..., jobs.idle(), ...});
// until the gate stops saying Wait. `drive()` below is that loop, so these tests
// exercise the same decision the GUI binary makes, without a window.
//
// Layout cache: every session here switches it OFF. The tests must never read or
// write the user's cache directory, and a cache hit would hide the compute path.
//
// Declaration order matters: `JobSystem jobs;` first, `ModelSession s(jobs);`
// second, so the session is destroyed before the pool (and the pool is idle by
// then, because drive() only returns once it is).

// LayoutEngine.h defines SizeFn, which ModelSession.h uses but does not include.
#include "engine/LayoutEngine.h"

#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <set>
#include <thread>

#include "core/ByteReader.h"
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "engine/CollapseTree.h"
#include "engine/ModelSession.h"
#include "engine/ScreenshotCli.h"
#include "parsers/Parser.h"

using namespace netvis;

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr const char* kBlocks = "tests/fixtures/model_blocks.onnx";
constexpr const char* kGguf = "tests/fixtures/model.gguf";

bool fixture_ok(const char* path) {
  if (fs::exists(path)) return true;
  WARN_MESSAGE(false, "fixture missing; run tools/gen_fixtures.py");
  return false;
}

CaptureGateInputs gate_inputs(const ModelSession& s, const JobSystem& j, bool canvas_only) {
  CaptureGateInputs in{};
  in.load_failed = s.stage() == LoadStage::Failed;
  in.jobs_idle = j.idle();  // sampled AFTER update(), as the driver does
  in.has_model = s.model() != nullptr;
  in.has_graph = s.has_graph();
  in.has_layout = s.layout() != nullptr;
  in.layout_empty = in.has_layout && s.layout()->boxes.empty();
  in.canvas_only = canvas_only;
  return in;
}

// Number of distinct display nodes the layout placed. boxes.size() can be larger:
// the layout adds clone boxes that share their source's display_id (see
// LayoutEngine.cpp), so the display-node count is what must be compared.
size_t display_nodes_placed(const LayoutResult& l) {
  std::set<uint32_t> ids;
  for (const NodeBox& b : l.boxes) ids.insert(b.display_id);
  return ids.size();
}

// Pump until the gate decides. A deadline FAIL keeps CI from hanging if it never does.
CaptureGate drive(ModelSession& s, JobSystem& j, bool canvas_only, double timeout_s = 20) {
  const auto deadline = Clock::now() + std::chrono::duration<double>(timeout_s);
  for (;;) {
    s.update();
    const CaptureGate g = capture_gate(gate_inputs(s, j, canvas_only));
    if (g != CaptureGate::Wait) return g;
    if (Clock::now() > deadline) FAIL("capture gate still Wait after the deadline");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

}  // namespace

TEST_CASE("screenshot gate: model_blocks.onnx has the shape the tests assume") {
  if (!fixture_ok(kBlocks)) return;

  // Direct parse on THIS thread: payload_read_counter() is thread-local, so a
  // session (whose parse runs on a worker) could not be checked this way.
  ByteReader::payload_read_counter() = 0;
  auto mf = MappedFile::open(kBlocks);
  REQUIRE(mf);
  ProgressSink progress;
  auto parsed = onnx::parse(*mf, progress);
  REQUIRE(parsed);
  const ir::Model& model = *parsed;
  REQUIRE(model.has_graph);
  REQUIRE(model.graphs.size() == 1);
  CHECK(model.graphs[0].nodes.size() == 38);
  // The PARSE reads zero tensor payload bytes. (A session's later shape inference does
  // read shape-constant initializers of at most 64 elements, as in the GUI; this
  // fixture has none, so this check says nothing about that step.)
  CHECK(ByteReader::payload_read_counter() == 0);

  CollapseTree tree;
  tree.build(model, 0);
  REQUIRE(tree.groups().size() == 1);               // "/model/layers.#"
  CHECK(tree.groups()[0].instances == 12);
  CHECK(tree.groups()[0].member_nodes.size() == 36);
  CHECK(tree.display_nodes().size() == 38);         // default: expanded
  REQUIRE(tree.collapse_all());
  // stem + one collapsed group + head. (CollapseTree::rebuild_display emits a
  // collapsed group as ONE display node.)
  CHECK(tree.display_nodes().size() == 3);
}

TEST_CASE("screenshot gate: an ONNX fixture loads to Ready with the cache off") {
  if (!fixture_ok(kBlocks)) return;
  JobSystem jobs;
  ModelSession s(jobs);
  s.set_layout_cache_enabled(false);
  CHECK_FALSE(s.layout_cache_enabled());

  s.open_async(kBlocks);
  REQUIRE(drive(s, jobs, /*canvas_only=*/true) == CaptureGate::Ready);
  REQUIRE(s.layout() != nullptr);
  CHECK_FALSE(s.layout()->from_cache);
  CHECK_FALSE(s.layout()->boxes.empty());
  CHECK(s.stage() == LoadStage::Ready);
  CHECK(display_nodes_placed(*s.layout()) == 38);
  CHECK(jobs.idle());
}

TEST_CASE("screenshot gate: collapse_all / expand_all make the pool busy, then settle") {
  if (!fixture_ok(kBlocks)) return;
  JobSystem jobs;
  ModelSession s(jobs);
  s.set_layout_cache_enabled(false);
  s.open_async(kBlocks);
  REQUIRE(drive(s, jobs, true) == CaptureGate::Ready);
  REQUIRE(display_nodes_placed(*s.layout()) == 38);

  s.collapse_all();
  CHECK_FALSE(jobs.idle());  // a re-layout is queued: the gate must say Wait
  CHECK(capture_gate(gate_inputs(s, jobs, true)) == CaptureGate::Wait);
  REQUIRE(drive(s, jobs, true) == CaptureGate::Ready);
  CHECK(display_nodes_placed(*s.layout()) == 3);  // stem + one "layers x12" block + head
  CHECK(s.stage() == LoadStage::Ready);

  s.expand_all();
  CHECK_FALSE(jobs.idle());
  REQUIRE(drive(s, jobs, true) == CaptureGate::Ready);
  CHECK(display_nodes_placed(*s.layout()) == 38);
}

TEST_CASE("screenshot gate: a missing file is LoadFailed with a message") {
  JobSystem jobs;
  ModelSession s(jobs);
  s.set_layout_cache_enabled(false);
  s.open_async("/nonexistent/x.onnx");
  CHECK(drive(s, jobs, false) == CaptureGate::LoadFailed);
  CHECK(s.stage() == LoadStage::Failed);
  CHECK_FALSE(s.error_message().empty());
}

TEST_CASE("screenshot gate: a weights-only model has no canvas") {
  if (!fixture_ok(kGguf)) return;
  {
    JobSystem jobs;
    ModelSession s(jobs);
    s.set_layout_cache_enabled(false);
    s.open_async(kGguf);
    CHECK(drive(s, jobs, /*canvas_only=*/true) == CaptureGate::NoCanvas);
  }
  {
    JobSystem jobs;
    ModelSession s(jobs);
    s.set_layout_cache_enabled(false);
    s.open_async(kGguf);
    // Full window: the tensor table is the picture.
    CHECK(drive(s, jobs, /*canvas_only=*/false) == CaptureGate::Ready);
  }
}

TEST_CASE("ModelSession: an ONNX load always ends Ready once the pool is idle") {
  if (!fixture_ok(kBlocks)) return;
  // Regression for the Enriching race (#170 R1): when ONNX shape inference landed
  // before the first layout, the shape completion saw no layout and left the stage
  // at Enriching, and the layout completion then declined to advance it, so the
  // load stayed "Enriching" forever. Which completion lands first is a scheduling
  // race, so a pre-fix build only fails SOME iterations; hence the loop. The
  // assertion itself is deterministic once the bug is fixed: shapes_pending_ gates
  // both completions, so every ordering ends Ready.
  JobSystem jobs;
  ModelSession s(jobs);
  s.set_layout_cache_enabled(false);
  for (int i = 0; i < 25; ++i) {
    INFO("iteration " << i);
    const uint64_t enriched_before = s.enrich_generation();
    s.open_async(kBlocks);
    REQUIRE(drive(s, jobs, true) == CaptureGate::Ready);
    CHECK(s.stage() == LoadStage::Ready);
    CHECK(s.enrich_generation() > enriched_before);  // shape inference ran and published
  }
}

TEST_CASE("ModelSession: with the layout cache off a layout is never served from it") {
  if (!fixture_ok(kBlocks)) return;
  JobSystem jobs;
  ModelSession s(jobs);
  s.set_layout_cache_enabled(false);
  for (int i = 0; i < 2; ++i) {
    s.open_async(kBlocks);
    REQUIRE(drive(s, jobs, true) == CaptureGate::Ready);
    REQUIRE(s.layout() != nullptr);
    CHECK_FALSE(s.layout()->from_cache);
  }
}
