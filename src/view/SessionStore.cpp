// view/SessionStore.cpp — persistence for #103. See SessionStore.h for the
// full contract (what is stored, why collapse state is excluded, why restore
// is opt-in); this file only implements what that header already promises.
//
// Follows the house shape for a prefs-adjacent file — App::save_prefs/
// load_prefs and App::save_recent/load_recent (view/App.cpp) — exactly:
// nlohmann/json, std::ofstream on write, and a `try { ... } catch (...) { }`
// around the whole read so ANY parse/type failure degrades to "start empty"
// rather than throwing across the view/engine boundary NetVis otherwise keeps
// exception-free (spec §9, Result<T> everywhere).
#include "view/SessionStore.h"

#include <cmath>
#include <cstdio>
#include <fstream>

#include <nlohmann/json.hpp>

#include "engine/LayoutCache.h"

namespace netvis {

namespace {

// Mirrors GraphCanvas.cpp's kMinZoom/kMaxZoom. Those constants are file-local
// to the canvas (no header has exported them, since nothing outside the
// canvas needed the clamp before #103) — duplicated here with the SAME values
// rather than pulled in via view/GraphCanvas.h, which would drag ImGui into a
// file this header deliberately keeps free of it (so it can plausibly live in
// netvis_core; see the test file for the fuller story).
constexpr float kMinZoom = 0.02f;
constexpr float kMaxZoom = 4.0f;

// A path this long could not be a real filesystem path on any platform NetVis
// ships for (Linux/macOS PATH_MAX is 4096; Windows' legacy MAX_PATH is far
// smaller, and even an opt-in long path stays well under this). Session.json
// is user-writable, so a string past this bound is treated as hostile input —
// the tab is dropped like any other malformed entry — rather than round-
// tripped as-is into App::open_file.
constexpr size_t kMaxPathLen = 4096;

// A NaN/Inf pan would silently poison canvas math the moment it is added to a
// world coordinate (world*zoom+pan, per Camera.cpp). Unlike path/zoom, pan is
// not one of the two fields the header's contract calls out for outright
// entry-rejection ("non-empty path, finite zoom"), so an invalid pan is
// sanitized to the origin instead of discarding an otherwise-good tab over a
// cosmetic field.
float sanitize_pan(float v) { return std::isfinite(v) ? v : 0.0f; }

// Numeric field reader shared by pan_x/pan_y/zoom. nlohmann stores a JSON
// number as whichever of {number_integer, number_unsigned, number_float}
// matches how it was written, so `is_number()` (true for all three) is the
// right guard here — `is_number_float()` alone would reject a plain `1`.
float read_float(const nlohmann::json& e, const char* key, float fallback) {
  if (!e.contains(key) || !e[key].is_number()) return fallback;
  return e[key].get<float>();
}

}  // namespace

std::string session_file_path() { return layout_cache_dir() + "/session.json"; }

void save_session(const SessionState& s) {
  // Best-effort, exactly like save_prefs/save_recent: failing to persist this
  // convenience feature must never interrupt the user, so there is nothing to
  // report back to the caller — only an attempt.
  nlohmann::json tabs_json = nlohmann::json::array();
  size_t new_active = 0;
  bool active_found = false;

  for (size_t i = 0; i < s.tabs.size() && tabs_json.size() < kMaxSessionTabs; ++i) {
    const SessionTab& t = s.tabs[i];
    // A fresh, never-opened tab has an empty path and nothing to reopen.
    if (t.path.empty()) continue;
    if (i == s.active_tab) {
      new_active = tabs_json.size();
      active_found = true;
    }
    nlohmann::json tj;
    tj["path"] = t.path;
    tj["pan_x"] = t.pan_x;
    tj["pan_y"] = t.pan_y;
    tj["zoom"] = t.zoom;
    tabs_json.push_back(std::move(tj));
  }
  // The tab that was active had an empty path, was past the cap, or the input
  // active_tab index was out of range to begin with — none of the tabs we
  // actually wrote correspond to it. Fall back to the first written tab
  // (index 0), which load_session's own clamp would also land on.
  if (!active_found) new_active = 0;

  nlohmann::json j;
  j["tabs"] = std::move(tabs_json);
  j["active_tab"] = new_active;

  // j.dump()'s default error_handler (strict) THROWS type_error.316 if any
  // string is not valid UTF-8 — and unlike most of this app's other persisted
  // strings, a raw filesystem path is exactly the kind of value that is not
  // guaranteed to be: Linux paths are arbitrary bytes, not validated text.
  // save_prefs/save_recent skip this guard (their strings are UI-chosen, not
  // attacker/filesystem-chosen), but this file's contract is explicit —
  // "silently best-effort... must never interrupt the user" — so the write is
  // wrapped rather than letting a single odd path turn a QoL feature into a
  // crash.
  try {
    std::ofstream f(session_file_path());
    if (f) f << j.dump(2);
  } catch (...) {
    // Best-effort: nothing to do but give up on persisting this time.
  }
}

SessionState load_session() {
  SessionState out;  // empty by default: the contract for missing/corrupt/malformed
  std::ifstream f(session_file_path());
  if (!f) return out;

  // nlohmann's parse (via operator>>) and every .get<T>() below can throw on a
  // malformed document or an unexpected type; contain the ENTIRE read so a
  // corrupt file degrades to the empty SessionState rather than propagating an
  // exception out of this function.
  try {
    nlohmann::json j;
    f >> j;
    if (!j.is_object() || !j.contains("tabs") || !j["tabs"].is_array())
      return SessionState{};

    for (const auto& e : j["tabs"]) {
      // Hostile-file cap: never replay more than kMaxSessionTabs regardless of
      // how many entries the file claims to hold (a huge array must not be
      // replayed — it would spawn a JobSystem per tab at launch).
      if (out.tabs.size() >= kMaxSessionTabs) break;
      if (!e.is_object()) continue;
      if (!e.contains("path") || !e["path"].is_string()) continue;

      std::string path = e["path"].get<std::string>();
      if (path.empty() || path.size() > kMaxPathLen) continue;

      const float zoom = read_float(e, "zoom", 1.0f);
      // Written as !(z >= min && z <= max) rather than the "naive" negation
      // (z < min || z > max) ON PURPOSE: every comparison against NaN is
      // false, so the naive form evaluates false||false == "in range" for
      // NaN, while this form evaluates !(false && ...) == true == "reject" —
      // the one place a corrupt/hostile float must not slip through. The same
      // range also rejects +-Inf, since neither satisfies <= kMaxZoom.
      if (!(zoom >= kMinZoom && zoom <= kMaxZoom)) continue;

      SessionTab t;
      t.path = std::move(path);
      t.pan_x = sanitize_pan(read_float(e, "pan_x", 0.0f));
      t.pan_y = sanitize_pan(read_float(e, "pan_y", 0.0f));
      t.zoom = zoom;
      out.tabs.push_back(std::move(t));
    }

    // active_tab may be absent (older/hand-edited file), negative, or huge;
    // is_number_integer()/is_number_unsigned() covers however nlohmann typed
    // it, and the final clamp handles a value past the (post-validation) tab
    // count — including when every tab in the file was dropped as invalid.
    size_t active = 0;
    if (j.contains("active_tab")) {
      const nlohmann::json& av = j["active_tab"];
      if (av.is_number_integer() || av.is_number_unsigned()) {
        const int64_t v = av.get<int64_t>();
        if (v > 0) active = static_cast<size_t>(v);
      }
    }
    out.active_tab =
        out.tabs.empty() ? 0
                          : (active >= out.tabs.size() ? out.tabs.size() - 1 : active);
    return out;
  } catch (...) {
    return SessionState{};
  }
}

void clear_session() {
  // A missing file is not an error, so the result of remove() goes unchecked:
  // "the file is gone" and "the file was never there" are the same outcome
  // for a caller that only wants the session forgotten.
  std::remove(session_file_path().c_str());
}

}  // namespace netvis
