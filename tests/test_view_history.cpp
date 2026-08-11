// tests/test_view_history.cpp — the #106 undo/redo ring, against the REAL type.
//
// This file used to test a hand-copied MIRROR of the ring, because
// view/ViewHistory.h included "imgui.h" for ImVec2 and netvis_tests links
// netvis_core only. A mirror test certifies the transcription, not the code, and
// silently rots the moment the two diverge — so instead the ring itself moved.
//
// engine/ViewSnapshot.h now holds ViewSnapshot + ViewHistory with plain floats
// for the camera and no ImGui dependency at all, and lives in netvis_core. Only
// capture_view/apply_view — which need ViewState, and therefore ImGui — stayed
// behind in view/ViewHistory.h. So everything below exercises the shipped code.
//
// STILL NOT COVERED (needs the app, which no headless test can link):
// capture_view/apply_view, i.e. the display-index <-> IR-index translation, the
// (generation, graph) refusal, and the collapse-bitset size check.
//
// `selected_value` stands in for "some non-camera field" throughout: camera_only_diff
// is an AND over every non-camera field, so one differing scalar drives it
// exactly as any other would.
#include <doctest/doctest.h>

#include <cstddef>
#include <vector>

#include "engine/ViewSnapshot.h"

using netvis::kMaxHistoryEntries;
using netvis::ViewHistory;
using netvis::ViewSnapshot;

namespace {

// A snapshot that differs only in a NON-camera field — i.e. a real edit, which
// closes any open camera-coalescing run.
ViewSnapshot edit(int tag) {
  ViewSnapshot s;
  s.selected_value = tag;
  return s;
}

// The same edit, moved. Consecutive calls with one `tag` and differing `x` are
// exactly the camera-only run coalescing must collapse.
ViewSnapshot pan(int tag, float x) {
  ViewSnapshot s = edit(tag);
  s.pan_x = x;
  return s;
}

}  // namespace

TEST_CASE("ViewHistory: a run of camera-only pushes is one undo step") {
  ViewHistory h;
  h.push(edit(1));
  // One drag delivered as 40 frames of pan. Each frame moves the camera and
  // nothing else, so all 40 collapse into the single entry that opened the run.
  for (int i = 1; i <= 40; ++i) h.push(pan(1, static_cast<float>(i)));
  CHECK(h.size() == 2);
  // The run's slot tracks the NEWEST camera, so undo returns to the pre-drag
  // pose and redo returns to where the drag ended (not to its first frame).
  CHECK(h.at(1)->pan_x == 40.0f);
  const ViewSnapshot* u = h.undo();
  REQUIRE(u != nullptr);
  CHECK(u->pan_x == 0.0f);
  const ViewSnapshot* r = h.redo();
  REQUIRE(r != nullptr);
  CHECK(r->pan_x == 40.0f);
}

TEST_CASE("ViewHistory: a non-camera change closes the run, the next pan opens a new one") {
  ViewHistory h;
  h.push(edit(1));
  h.push(pan(1, 10.0f));
  h.push(pan(1, 20.0f));
  CHECK(h.size() == 2);          // both pans in one slot
  h.push(edit(2));               // closes the run
  CHECK(h.size() == 3);
  h.push(pan(2, 30.0f));         // opens a NEW run rather than joining the old
  CHECK(h.size() == 4);
  h.push(pan(2, 40.0f));
  CHECK(h.size() == 4);
  CHECK(h.at(3)->pan_x == 40.0f);
}

TEST_CASE("ViewHistory: identical pushes are dropped") {
  ViewHistory h;
  // record_view_history() fires every frame; a second of idling at 120 fps must
  // not fill the ring with copies of the same state.
  for (int i = 0; i < 120; ++i) h.push(edit(7));
  CHECK(h.size() == 1);
  CHECK(!h.can_undo());
  CHECK(!h.can_redo());
}

TEST_CASE("ViewHistory: pushing after an undo truncates the redo tail") {
  ViewHistory h;
  for (int i = 0; i < 5; ++i) h.push(edit(i));
  CHECK(h.size() == 5);
  h.undo();
  h.undo();
  CHECK(h.cursor() == 2);
  CHECK(h.can_redo());
  h.push(edit(99));
  CHECK(h.size() == 4);          // 0,1,2 + the new branch
  CHECK(h.cursor() == 3);
  CHECK(!h.can_redo());
  CHECK(h.at(3)->selected_value == 99);
}

TEST_CASE("ViewHistory: the frame after an undo must not truncate the redo tail") {
  // The regression this ordering exists for: record_view_history() runs again
  // the frame after a restore and pushes a snapshot IDENTICAL to the entry the
  // cursor just landed on. The identical-drop has to be tested before the tail
  // is truncated, or redo becomes unreachable after every single undo.
  ViewHistory h;
  h.push(edit(1));
  h.push(edit(2));
  h.push(edit(3));
  const ViewSnapshot* u = h.undo();
  REQUIRE(u != nullptr);
  ViewSnapshot restored = *u;
  for (int i = 0; i < 10; ++i) h.push(restored);
  CHECK(h.size() == 3);
  CHECK(h.can_redo());
  const ViewSnapshot* r = h.redo();
  REQUIRE(r != nullptr);
  CHECK(r->selected_value == 3);
}

TEST_CASE("ViewHistory: a camera run does not clobber the entry an undo landed on") {
  // undo() closes the open run. Without that, the run's in-place overwrite would
  // rewrite entries_[cursor_] — the state the user just stepped back to.
  ViewHistory h;
  h.push(edit(1));
  h.push(pan(1, 5.0f));          // opens a run
  CHECK(h.size() == 2);
  const ViewSnapshot* u = h.undo();
  REQUIRE(u != nullptr);
  CHECK(u->pan_x == 0.0f);
  h.push(pan(1, 9.0f));          // must APPEND (truncating the tail), not overwrite
  CHECK(h.at(0)->pan_x == 0.0f);  // the undone-to entry is intact
  CHECK(h.size() == 2);
  CHECK(h.at(1)->pan_x == 9.0f);
}

TEST_CASE("ViewHistory: the ring is bounded and drops the oldest") {
  ViewHistory h;
  const int total = static_cast<int>(kMaxHistoryEntries) + 10;
  for (int i = 0; i < total; ++i) h.push(edit(i));
  CHECK(h.size() == kMaxHistoryEntries);
  // The first 10 are gone; the window is [10, kMaxHistoryEntries+10).
  CHECK(h.at(0)->selected_value == 10);
  CHECK(h.at(kMaxHistoryEntries - 1)->selected_value == total - 1);
  // Dropping from the front shifted every index, so the cursor must have moved
  // with them — it still points at the newest entry.
  CHECK(h.cursor() == kMaxHistoryEntries - 1);
  CHECK(!h.can_redo());
  // And the cursor is still walkable all the way to the surviving front.
  size_t steps = 0;
  while (h.can_undo()) {
    h.undo();
    ++steps;
  }
  CHECK(steps == kMaxHistoryEntries - 1);
  CHECK(h.cursor() == 0);
}

TEST_CASE("ViewHistory: the cursor stops at both ends") {
  ViewHistory h;
  CHECK(h.undo() == nullptr);    // empty history
  CHECK(h.redo() == nullptr);
  h.push(edit(1));
  h.push(edit(2));
  CHECK(h.redo() == nullptr);    // already at the newest
  REQUIRE(h.undo() != nullptr);
  CHECK(h.undo() == nullptr);    // already at the oldest
  CHECK(h.cursor() == 0);
  REQUIRE(h.redo() != nullptr);
  CHECK(h.redo() == nullptr);
  CHECK(h.cursor() == 1);
}

TEST_CASE("ViewHistory: clear() empties the ring and the cursor") {
  ViewHistory h;
  for (int i = 0; i < 5; ++i) h.push(edit(i));
  h.undo();
  h.clear();
  CHECK(h.size() == 0);
  CHECK(h.cursor() == 0);
  CHECK(!h.can_undo());
  CHECK(!h.can_redo());
  CHECK(h.undo() == nullptr);
  CHECK(h.redo() == nullptr);
  // Usable again afterwards: the first push seeds entry 0, not a stale slot.
  h.push(edit(42));
  CHECK(h.size() == 1);
  CHECK(h.cursor() == 0);
  CHECK(h.at(0)->selected_value == 42);
}

TEST_CASE("ViewHistory: camera_only_diff ignores the camera and nothing else") {
  ViewSnapshot a = edit(1);
  ViewSnapshot b = pan(1, 100.0f);
  b.zoom = 4.0f;
  CHECK(a.camera_only_diff(b));      // camera moved, everything else equal
  CHECK(a.camera_only_diff(a));      // identical also satisfies it (see the .cpp)
  CHECK(!a.camera_only_diff(edit(2)));
  CHECK(!b.camera_only_diff(pan(2, 100.0f)));
}
