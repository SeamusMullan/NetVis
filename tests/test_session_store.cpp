// tests/test_session_store.cpp — #103 session/workspace persistence.
//
// LINKAGE NOTE (read before trusting these results): SessionStore.h is
// deliberately ImGui-free, but SessionStore.cpp lives under src/view/, and
// CMakeLists.txt's netvis_view sources are only compiled into the `netvis`
// GUI target — NOT into netvis_core, which is the only library netvis_tests
// links (see CMakeLists.txt: `add_executable(netvis_tests ...)` /
// `target_link_libraries(netvis_tests PRIVATE netvis_core doctest::doctest)`).
// So as things stand, this translation unit compiles but the executable will
// fail to LINK (undefined reference to netvis::save_session/load_session/
// clear_session) until the lead does ONE of:
//   (a) add an explicit `target_sources(netvis_core PRIVATE
//       src/view/SessionStore.cpp)` in CMakeLists.txt, or
//   (b) relocate SessionStore.cpp under src/engine/ (the GLOB_RECURSE that
//       builds netvis_core already includes src/engine/*.cpp).
// Recommendation: (a). SessionStore.cpp has no ImGui/GLFW dependency (it only
// touches <fstream>, nlohmann/json, and engine/LayoutCache.h's
// layout_cache_dir(), itself already part of netvis_core) — same as
// engine/ModelPath.cpp and engine/ReportJson.cpp, which already use nlohmann
// from inside netvis_core. Adding one source line keeps SessionStore.cpp next
// to the header it implements (matching every other view/*.h + view/*.cpp
// pair in the tree) instead of splitting the pair across two directories.
// This is a CMakeLists.txt change, which is outside this file's owned set —
// flagged for the lead rather than made here.
//
// The tests below are written against the REAL contract in SessionStore.h and
// will pass once linked; they are not stubs and do not reimplement any of
// SessionStore.cpp's logic to fake a pass.
#include <doctest/doctest.h>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "view/SessionStore.h"

using namespace netvis;

namespace {

// session_file_path() always resolves to a real path under the user's cache
// dir (view_prefs.json's neighbour) — there is no override hook in the frozen
// contract, so these tests exercise that real file. Back up/restore whatever
// was there (a developer's real session, if any) around every test rather
// than clobbering it, and leave nothing behind for a test that starts clean.
struct SessionFileBackup {
  std::string path = session_file_path();
  bool existed = false;
  std::string content;

  SessionFileBackup() {
    std::ifstream f(path, std::ios::binary);
    if (f) {
      existed = true;
      content.assign(std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>());
    }
  }
  ~SessionFileBackup() {
    if (existed) {
      std::ofstream f(path, std::ios::binary | std::ios::trunc);
      f << content;
    } else {
      std::remove(path.c_str());
    }
  }
};

// Writes exactly `text` to the session file, bypassing save_session() — used
// to inject malformed/hostile content that a well-behaved writer never would.
void write_raw_session_file(const std::string& text) {
  std::ofstream f(session_file_path(), std::ios::binary | std::ios::trunc);
  f << text;
}

}  // namespace

TEST_CASE("SessionStore: save then load round-trips tabs and active_tab") {
  SessionFileBackup backup;

  SessionState in;
  in.tabs.push_back(SessionTab{"/models/a.onnx", 10.0f, -5.5f, 1.25f});
  in.tabs.push_back(SessionTab{"/models/b.gguf", -2.0f, 3.0f, 0.5f});
  in.active_tab = 1;

  save_session(in);
  SessionState out = load_session();

  REQUIRE(out.tabs.size() == 2);
  CHECK(out.tabs[0].path == "/models/a.onnx");
  CHECK(out.tabs[0].pan_x == doctest::Approx(10.0f));
  CHECK(out.tabs[0].pan_y == doctest::Approx(-5.5f));
  CHECK(out.tabs[0].zoom == doctest::Approx(1.25f));
  CHECK(out.tabs[1].path == "/models/b.gguf");
  CHECK(out.tabs[1].zoom == doctest::Approx(0.5f));
  CHECK(out.active_tab == 1);
}

TEST_CASE("SessionStore: save truncates to kMaxSessionTabs") {
  SessionFileBackup backup;

  SessionState in;
  const size_t total = kMaxSessionTabs + 5;
  for (size_t i = 0; i < total; ++i)
    in.tabs.push_back(SessionTab{"/models/tab_" + std::to_string(i) + ".onnx",
                                 0.0f, 0.0f, 1.0f});
  in.active_tab = 0;

  save_session(in);
  SessionState out = load_session();

  REQUIRE(out.tabs.size() == kMaxSessionTabs);
  // Order is preserved for the entries that survive the cap.
  CHECK(out.tabs.front().path == "/models/tab_0.onnx");
  CHECK(out.tabs.back().path ==
       "/models/tab_" + std::to_string(kMaxSessionTabs - 1) + ".onnx");
}

TEST_CASE("SessionStore: empty-path tabs are skipped, active_tab remaps") {
  SessionFileBackup backup;

  SessionState in;
  in.tabs.push_back(SessionTab{"", 0.0f, 0.0f, 1.0f});          // 0: fresh tab, skipped
  in.tabs.push_back(SessionTab{"/models/a.onnx", 1.0f, 1.0f, 1.2f});  // 1: kept -> new index 0
  in.tabs.push_back(SessionTab{"", 0.0f, 0.0f, 1.0f});          // 2: fresh tab, skipped
  in.tabs.push_back(SessionTab{"/models/b.onnx", 2.0f, 2.0f, 0.5f});  // 3: kept -> new index 1
  in.active_tab = 3;  // originally "/models/b.onnx"

  save_session(in);
  SessionState out = load_session();

  REQUIRE(out.tabs.size() == 2);
  CHECK(out.tabs[0].path == "/models/a.onnx");
  CHECK(out.tabs[1].path == "/models/b.onnx");
  // active_tab must follow "/models/b.onnx" to its NEW index (1), not stay at
  // the stale original index (3), and not silently point at the wrong tab.
  CHECK(out.active_tab == 1);
}

TEST_CASE("SessionStore: a tab with only an empty path never round-trips") {
  SessionFileBackup backup;

  SessionState in;
  in.tabs.push_back(SessionTab{"", 5.0f, 5.0f, 2.0f});
  in.active_tab = 0;

  save_session(in);
  SessionState out = load_session();

  CHECK(out.tabs.empty());
  CHECK(out.active_tab == 0);
}

TEST_CASE("SessionStore: load_session on a missing file returns empty state") {
  SessionFileBackup backup;
  std::remove(session_file_path().c_str());

  SessionState out = load_session();
  CHECK(out.tabs.empty());
  CHECK(out.active_tab == 0);
}

TEST_CASE("SessionStore: malformed JSON degrades to empty state, never crashes") {
  SessionFileBackup backup;

  SUBCASE("not JSON at all") {
    write_raw_session_file("this is not { json at all");
  }
  SUBCASE("valid JSON but wrong top-level shape (array, not object)") {
    write_raw_session_file("[1, 2, 3]");
  }
  SUBCASE("valid JSON object missing the tabs key") {
    write_raw_session_file(R"({"active_tab": 0})");
  }
  SUBCASE("tabs is not an array") {
    write_raw_session_file(R"({"tabs": "nope", "active_tab": 0})");
  }
  SUBCASE("a numerically overflowed field forces the WHOLE parse to fail") {
    // JSON's own grammar has no NaN/Infinity literal (RFC 8259), so a corrupt
    // zoom can only reach the parser as a syntactically valid but enormous
    // number. nlohmann rejects a float token that overflows to +-inf during
    // lexing itself (json.hpp: `if (!std::isfinite(res)) return parse_error`),
    // so this never even reaches SessionStore's own per-entry zoom check —
    // it fails a full parse step earlier, landing here as "malformed file".
    write_raw_session_file(
        R"({"tabs": [{"path": "/x", "zoom": 1e400}], "active_tab": 0})");
  }

  SessionState out = load_session();
  CHECK(out.tabs.empty());
  CHECK(out.active_tab == 0);
}

TEST_CASE("SessionStore: an out-of-range zoom drops only that entry") {
  SessionFileBackup backup;

  // "/bad" has a finite but out-of-clamp-range zoom (camera zoom is clamped to
  // [0.02, 4.0] — see GraphCanvas.cpp's kMinZoom/kMaxZoom); "/good" is valid.
  // A corrupt/hostile entry must not take its siblings down with it.
  write_raw_session_file(R"({
    "tabs": [
      {"path": "/bad_high", "zoom": 999.0},
      {"path": "/bad_low", "zoom": -1.0},
      {"path": "/good", "zoom": 1.5}
    ],
    "active_tab": 2
  })");

  SessionState out = load_session();

  REQUIRE(out.tabs.size() == 1);
  CHECK(out.tabs[0].path == "/good");
  CHECK(out.tabs[0].zoom == doctest::Approx(1.5f));
  // active_tab (2) pointed at "/good" before the drop; after the drop "/good"
  // is the only (and therefore last) surviving tab, so the clamp lands on it.
  CHECK(out.active_tab == 0);
}

TEST_CASE("SessionStore: active_tab is clamped into the post-validation range") {
  SessionFileBackup backup;

  SUBCASE("far past the end") {
    write_raw_session_file(R"({
      "tabs": [{"path": "/a", "zoom": 1.0}, {"path": "/b", "zoom": 1.0}],
      "active_tab": 100
    })");
    SessionState out = load_session();
    REQUIRE(out.tabs.size() == 2);
    CHECK(out.active_tab == 1);  // clamped to the last valid index
  }

  SUBCASE("negative") {
    write_raw_session_file(R"({
      "tabs": [{"path": "/a", "zoom": 1.0}, {"path": "/b", "zoom": 1.0}],
      "active_tab": -5
    })");
    SessionState out = load_session();
    REQUIRE(out.tabs.size() == 2);
    CHECK(out.active_tab == 0);
  }

  SUBCASE("missing entirely") {
    write_raw_session_file(R"({"tabs": [{"path": "/a", "zoom": 1.0}]})");
    SessionState out = load_session();
    REQUIRE(out.tabs.size() == 1);
    CHECK(out.active_tab == 0);
  }
}

TEST_CASE("SessionStore: a hostile-length path is dropped, not truncated in") {
  SessionFileBackup backup;

  const std::string huge_path(8192, 'x');  // past kMaxPathLen (4096)
  write_raw_session_file(
      R"({"tabs": [{"path": ")" + huge_path + R"(", "zoom": 1.0},
                   {"path": "/fine", "zoom": 1.0}], "active_tab": 0})");

  SessionState out = load_session();
  REQUIRE(out.tabs.size() == 1);
  CHECK(out.tabs[0].path == "/fine");
}

TEST_CASE("SessionStore: clear_session on a missing file is harmless") {
  SessionFileBackup backup;
  std::remove(session_file_path().c_str());  // make sure nothing is there

  clear_session();  // must not throw / crash on an absent file

  CHECK(load_session().tabs.empty());
}

TEST_CASE("SessionStore: clear_session removes an existing file") {
  SessionFileBackup backup;

  SessionState in;
  in.tabs.push_back(SessionTab{"/models/a.onnx", 0.0f, 0.0f, 1.0f});
  save_session(in);
  REQUIRE_FALSE(load_session().tabs.empty());

  clear_session();
  CHECK(load_session().tabs.empty());
}
