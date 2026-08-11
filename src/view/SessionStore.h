// view/SessionStore.h — reopen the last session's tabs + camera (#103).
//
// DECISION (v0.9.4): OPT-IN, and off by default. Silently reopening several
// multi-GB models on launch would turn a deliberate action into a startup cost
// the user did not ask for, and NetVis's whole pitch is that opening is instant.
// The pref is honoured strictly: with restore off, the session file is neither
// read nor written.
//
// WHAT IS STORED: per tab, the path the user opened and the camera pose, plus
// which tab was active. Nothing derived. Notably NOT the collapse state — group
// indices are assigned by CollapseTree during detection against a specific model
// build, so a bitset saved against one file and replayed after that file changed
// on disk would collapse arbitrary unrelated groups. A restored tab therefore
// opens in the default view, which is honest, rather than in a view that claims
// to be the user's but is not.
//
// A path that no longer resolves is SKIPPED, not an error: files move, and a
// launch must not fail because one of five remembered models was deleted. The
// caller is told how many were skipped so it can say so rather than silently
// opening fewer tabs than the user left behind.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace netvis {

struct SessionTab {
  std::string path;   // exactly the string passed to App::open_file
  float pan_x = 0.0f;
  float pan_y = 0.0f;
  float zoom = 1.0f;
};

struct SessionState {
  std::vector<SessionTab> tabs;
  size_t active_tab = 0;
};

// Cap. A session with hundreds of tabs is a corrupt or hostile file, and
// replaying it would spawn a JobSystem per tab at launch.
inline constexpr size_t kMaxSessionTabs = 16;

// Path of the session file, next to view_prefs.json in layout_cache_dir().
std::string session_file_path();

// Write. Silently best-effort, exactly like save_prefs/save_recent — failing to
// persist a convenience feature must never interrupt the user. Truncates to
// kMaxSessionTabs. Tabs with an empty path (a fresh, never-opened tab) are
// skipped, since there is nothing to reopen.
void save_session(const SessionState& s);

// Read. Returns an empty SessionState when the file is missing, unreadable, or
// malformed — a corrupt session file must degrade to "start empty", never to a
// crash or a partially-applied session. Entries are validated (non-empty path,
// finite zoom within the camera's clamp range) and invalid ones dropped.
SessionState load_session();

// Delete the session file. Called when the pref is turned off, so disabling
// restore also forgets what was remembered rather than leaving it on disk.
void clear_session();

}  // namespace netvis
