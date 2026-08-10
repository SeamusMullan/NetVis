// view/ViewHistory.h — capture/apply of a ViewSnapshot against the live view.
//
// The SNAPSHOT and the bounded history ring live in engine/ViewSnapshot.h, in
// netvis_core, so netvis_tests can link and exercise them for real. Only these
// two functions need ViewState and ModelSession, and ViewState drags in ImGui —
// so only these two live here.
#pragma once

#include "engine/ViewSnapshot.h"

namespace netvis {

struct ViewState;    // view/App.h
class ModelSession;  // engine/ModelSession.h

// Capture / apply. Both live here rather than in App so the rules above are in
// one place and cannot drift between the two directions.
ViewSnapshot capture_view(const ViewState& vs, const ModelSession& session);

// Applies `s` to `vs`/`session`. Returns false and changes NOTHING when the
// snapshot's (generation, graph) does not match the live session, or when its
// collapse bitset does not match the current group count — a partial apply would
// leave the view in a state the user never chose.
bool apply_view(const ViewSnapshot& s, ViewState& vs, ModelSession& session);

}  // namespace netvis
