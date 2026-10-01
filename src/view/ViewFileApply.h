// SPDX-License-Identifier: Apache-2.0
// view/ViewFileApply.h — apply a parsed .netvis-view file to a live tab (#56, #170).
//
// The FORMAT, the parse policy and the "which parts apply to this model?" overlay
// are core (engine/ViewFile.h) and unit-tested. This file is the thin, ImGui-side
// half that moves the result into ViewState and ModelSession. It is shared by
// File > Load View State and by `netvis --screenshot --view`, so the two cannot
// disagree about what a view file means.
//
// WHY PHASES. Some steps change the session (push_graph into a subgraph, a
// collapse change) and queue a layout. A layout worker reads the collapse state
// while it runs, so mutating it under a running layout is a data race, and two
// in-flight layouts in one generation publish in completion order. Every
// session-mutating step therefore runs only when the tab's pool is IDLE, which is
// what makes the screenshot path free of both hazards (and interactive Load View
// State safer). The model-independent half applies at once; the model-specific
// half waits for the pool. In the normal interactive case the pool is already
// idle and the whole load completes in the one call, exactly as the old
// synchronous loader did.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/ViewFile.h"

namespace netvis {

struct ViewState;    // view/App.h
class ModelSession;  // engine/ModelSession.h

// How much of the file applied (drives the toast text).
enum class ViewApplyOutcome : uint8_t {
  Full,        // same model: everything that was in range applied
  OtherModel,  // another model: the model-independent settings applied
  NoModel,     // nothing loaded yet: the model-independent settings applied
};

enum class ViewStep : uint8_t {
  Pending,  // waiting for the pool to go idle; call again next frame
  Done,
  Aborted,  // the tab's model changed underneath; nothing further was applied
};

struct ViewFileApplier {
  enum class Phase : uint8_t { Agnostic, Dive, Specific, Done };

  ViewFile file;
  std::vector<std::string> notes;   // what was ignored and why (parse warnings first)
  uint64_t generation = 0;          // the session generation this load was started against
  bool has_model = false;           // session.model() != nullptr at creation
  bool same_model = false;          // the file's "model" is the live model
  Phase phase = Phase::Agnostic;
  ViewApplyOutcome outcome = ViewApplyOutcome::NoModel;
};

// `notes` seeds ViewFileApplier::notes (typically the parse warnings).
ViewFileApplier make_view_file_applier(const ViewFile& f, std::vector<std::string> notes,
                                       const ModelSession& s);

// Advance the load as far as it can go. `jobs_idle` is the tab's JobSystem::idle(),
// sampled by the caller AFTER session.update(). Main thread only.
ViewStep step_view_file(ViewFileApplier& a, ViewState& vs, ModelSession& s, bool jobs_idle);

// The model-independent half: camera, toggles, edge routing, heatmap metric/scale,
// the three filters and the navigation intent. Touches no session state, so it can
// run at any time.
void apply_view_file_agnostic(const ViewFile& f, ViewState& vs);

// The toast text for a finished load (the three strings #56 introduced).
const char* view_file_outcome_text(ViewApplyOutcome o);

}  // namespace netvis
