// tests/test_view_history.cpp — the #106 undo/redo ring's algorithm, headless.
//
// WHAT THIS FILE CAN AND CANNOT TEST — read before adding a case.
//
// netvis_tests links `netvis_core` and `doctest` ONLY (see CMakeLists.txt), and
// its include path is <repo>/src plus <repo> — it has no ImGui include dir and
// no view/ object files. view/ViewHistory.h includes "imgui.h" for ImVec2, so
// this TU cannot even INCLUDE the real header, let alone link ViewHistory.cpp.
// That is the correct layering (spec §9: the core library has no GUI
// dependency), not an oversight to work around.
//
// So the real ViewHistory/ViewSnapshot are NOT exercised here. What IS exercised
// is the ring's ALGORITHM, transcribed below onto a minimal stand-in whose
// comparison semantics mirror ViewSnapshot's: a camera triple that coalescing
// ignores, and one scalar standing in for the twenty-odd non-camera fields that
// camera_only_diff() compares. The behaviours that make the ring correct —
// camera-run coalescing, duplicate suppression, redo-tail truncation, the
// bound's effect on the cursor, and cursor walking at both ends — are pure
// index arithmetic over that predicate, so they are fully covered.
//
// NOT covered here (needs the app, which no headless test can link):
// capture_view/apply_view, i.e. the display-index <-> IR-index translation, the
// (generation, graph) refusal, and the collapse-bitset size check.
//
// MIRROR DISCIPLINE: MiniHistory::push/undo/redo below is a line-for-line
// transcription of ViewHistory's. A change to one is a change to both, or this
// file silently starts certifying an algorithm the app no longer runs.
#include <doctest/doctest.h>

#include <cstddef>
#include <vector>

namespace {

// Must equal netvis::kMaxHistoryEntries (view/ViewHistory.h). Duplicated rather
// than included for the reason in the file header; if that constant moves, this
// one moves with it.
constexpr size_t kMirrorMax = 64;

// Stand-in for ViewSnapshot. `tag` collapses every non-camera field into one
// value: camera_only_diff() is an AND over field equalities, so one differing
// scalar reproduces its result exactly for the purposes of the ring.
struct MiniSnapshot {
  float pan_x = 0.0f, pan_y = 0.0f, zoom = 1.0f;  // the camera
  int tag = 0;                                    // everything else

  bool camera_only_diff(const MiniSnapshot& o) const { return tag == o.tag; }
};

bool same_camera(const MiniSnapshot& a, const MiniSnapshot& b) {
  return a.pan_x == b.pan_x && a.pan_y == b.pan_y && a.zoom == b.zoom;
}

class MiniHistory {
 public:
  void push(const MiniSnapshot& s) {
    if (entries_.empty()) {
      entries_.push_back(s);
      cursor_ = 0;
      coalescing_ = false;
      return;
    }
    const MiniSnapshot& cur = entries_[cursor_];
    const bool camera_only = s.camera_only_diff(cur);
    if (camera_only && same_camera(s, cur)) return;
    if (camera_only && coalescing_) {
      entries_[cursor_] = s;
      return;
    }
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(cursor_) + 1,
                   entries_.end());
    entries_.push_back(s);
    coalescing_ = camera_only;
    cursor_ = entries_.size() - 1;
    if (entries_.size() > kMirrorMax) {
      const size_t drop = entries_.size() - kMirrorMax;
      entries_.erase(entries_.begin(),
                     entries_.begin() + static_cast<std::ptrdiff_t>(drop));
      cursor_ = (cursor_ >= drop) ? cursor_ - drop : 0;
    }
  }

  bool can_undo() const { return !entries_.empty() && cursor_ > 0; }
  bool can_redo() const { return !entries_.empty() && cursor_ + 1 < entries_.size(); }

  const MiniSnapshot* undo() {
    if (!can_undo()) return nullptr;
    --cursor_;
    coalescing_ = false;
    return &entries_[cursor_];
  }

  const MiniSnapshot* redo() {
    if (!can_redo()) return nullptr;
    ++cursor_;
    coalescing_ = false;
    return &entries_[cursor_];
  }

  void clear() {
    entries_.clear();
    cursor_ = 0;
    coalescing_ = false;
  }

  size_t size() const { return entries_.size(); }

  // Test-only introspection; the real class needs neither.
  size_t cursor() const { return cursor_; }
  const MiniSnapshot& at(size_t i) const { return entries_[i]; }

 private:
  std::vector<MiniSnapshot> entries_;
  size_t cursor_ = 0;
  bool coalescing_ = false;
};

// A snapshot that differs from its predecessor in a NON-camera field.
MiniSnapshot edit(int tag) {
  MiniSnapshot s;
  s.tag = tag;
  return s;
}

// A snapshot with the same non-camera state but a moved camera.
MiniSnapshot pan(int tag, float x) {
  MiniSnapshot s;
  s.tag = tag;
  s.pan_x = x;
  return s;
}

}  // namespace

TEST_CASE("ViewHistory: a run of camera-only pushes is one undo step") {
  MiniHistory h;
  h.push(edit(1));
  // One drag delivered as 40 frames of pan. Each frame moves the camera and
  // nothing else, so all 40 collapse into the single entry that opened the run.
  for (int i = 1; i <= 40; ++i) h.push(pan(1, static_cast<float>(i)));
  CHECK(h.size() == 2);
  // The run's slot tracks the NEWEST camera, so undo returns to the pre-drag
  // pose and redo returns to where the drag ended (not to its first frame).
  CHECK(h.at(1).pan_x == 40.0f);
  const MiniSnapshot* u = h.undo();
  REQUIRE(u != nullptr);
  CHECK(u->pan_x == 0.0f);
  const MiniSnapshot* r = h.redo();
  REQUIRE(r != nullptr);
  CHECK(r->pan_x == 40.0f);
}

TEST_CASE("ViewHistory: a non-camera change closes the run, the next pan opens a new one") {
  MiniHistory h;
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
  CHECK(h.at(3).pan_x == 40.0f);
}

TEST_CASE("ViewHistory: identical pushes are dropped") {
  MiniHistory h;
  // record_view_history() fires every frame; a second of idling at 120 fps must
  // not fill the ring with copies of the same state.
  for (int i = 0; i < 120; ++i) h.push(edit(7));
  CHECK(h.size() == 1);
  CHECK(!h.can_undo());
  CHECK(!h.can_redo());
}

TEST_CASE("ViewHistory: pushing after an undo truncates the redo tail") {
  MiniHistory h;
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
  CHECK(h.at(3).tag == 99);
}

TEST_CASE("ViewHistory: the frame after an undo must not truncate the redo tail") {
  // The regression this ordering exists for: record_view_history() runs again
  // the frame after a restore and pushes a snapshot IDENTICAL to the entry the
  // cursor just landed on. The identical-drop has to be tested before the tail
  // is truncated, or redo becomes unreachable after every single undo.
  MiniHistory h;
  h.push(edit(1));
  h.push(edit(2));
  h.push(edit(3));
  const MiniSnapshot* u = h.undo();
  REQUIRE(u != nullptr);
  MiniSnapshot restored = *u;
  for (int i = 0; i < 10; ++i) h.push(restored);
  CHECK(h.size() == 3);
  CHECK(h.can_redo());
  const MiniSnapshot* r = h.redo();
  REQUIRE(r != nullptr);
  CHECK(r->tag == 3);
}

TEST_CASE("ViewHistory: a camera run does not clobber the entry an undo landed on") {
  // undo() closes the open run. Without that, the run's in-place overwrite would
  // rewrite entries_[cursor_] — the state the user just stepped back to.
  MiniHistory h;
  h.push(edit(1));
  h.push(pan(1, 5.0f));          // opens a run
  CHECK(h.size() == 2);
  const MiniSnapshot* u = h.undo();
  REQUIRE(u != nullptr);
  CHECK(u->pan_x == 0.0f);
  h.push(pan(1, 9.0f));          // must APPEND (truncating the tail), not overwrite
  CHECK(h.at(0).pan_x == 0.0f);  // the undone-to entry is intact
  CHECK(h.size() == 2);
  CHECK(h.at(1).pan_x == 9.0f);
}

TEST_CASE("ViewHistory: the ring is bounded and drops the oldest") {
  MiniHistory h;
  const int total = static_cast<int>(kMirrorMax) + 10;
  for (int i = 0; i < total; ++i) h.push(edit(i));
  CHECK(h.size() == kMirrorMax);
  // The first 10 are gone; the window is [10, kMirrorMax+10).
  CHECK(h.at(0).tag == 10);
  CHECK(h.at(kMirrorMax - 1).tag == total - 1);
  // Dropping from the front shifted every index, so the cursor must have moved
  // with them — it still points at the newest entry.
  CHECK(h.cursor() == kMirrorMax - 1);
  CHECK(!h.can_redo());
  // And the cursor is still walkable all the way to the surviving front.
  size_t steps = 0;
  while (h.can_undo()) {
    h.undo();
    ++steps;
  }
  CHECK(steps == kMirrorMax - 1);
  CHECK(h.cursor() == 0);
}

TEST_CASE("ViewHistory: the cursor stops at both ends") {
  MiniHistory h;
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
  MiniHistory h;
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
  CHECK(h.at(0).tag == 42);
}

TEST_CASE("ViewHistory: camera_only_diff ignores the camera and nothing else") {
  MiniSnapshot a = edit(1);
  MiniSnapshot b = pan(1, 100.0f);
  b.zoom = 4.0f;
  CHECK(a.camera_only_diff(b));      // camera moved, everything else equal
  CHECK(a.camera_only_diff(a));      // identical also satisfies it (see the .cpp)
  CHECK(!a.camera_only_diff(edit(2)));
  CHECK(!b.camera_only_diff(pan(2, 100.0f)));
}
