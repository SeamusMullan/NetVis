// view/PreferencesPanel.h — one Settings window (#102).
//
// DECISION (v0.9.4): preferences are currently seventeen scattered menu items
// across File and View plus a checkbox in the Plugins panel, and `view_prefs.json`
// persists a subset of them chosen by whoever added each feature. A user looking
// for "how do I turn off edge tooltips" has to already know it lives under View.
// This panel gathers every persisted preference into one place, grouped by what
// it affects, with a reset.
//
// THE MENU ITEMS STAY. This is deliberately additive: removing them would break
// the muscle memory of anyone using the app today, and a preferences window is a
// place to DISCOVER a setting, not the only place to reach one. Both surfaces
// mutate the same ViewState fields and call the same save_prefs(), so they cannot
// disagree.
//
// RESET IS PER-SECTION, NOT GLOBAL ONLY. A single "reset everything" button is
// the one most likely to be pressed by accident and the least likely to be what
// was wanted — a user who dislikes their heatmap gradient does not want their
// plugin trust decisions cleared too. Plugin enablement is deliberately excluded
// from reset entirely: re-enabling a WASM plugin is a TRUST decision that went
// through a confirmation dialog, and a preferences reset must never silently
// undo a security choice.
#pragma once

namespace netvis {

class App;

// Draw the Preferences window when ViewState::show_preferences is set. Writes
// changes straight through to the live ViewState and calls App::save_prefs() on
// commit, so the window is not modal and has no separate apply step — the app
// already updates live from these fields everywhere else.
void draw_preferences_panel(App& app);

}  // namespace netvis
