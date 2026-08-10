// view/Onboarding.h — first-run empty state and the Help menu (#105).
//
// DECISION (v0.9.4): with no model open, NetVis currently draws NOTHING — the
// graph canvas is not even reached, because App::frame() only enters it when
// session().has_graph() is true. A first-time user therefore sees an empty grey
// dockspace and has to find File > Open on their own. This panel fills that
// space with the three things a new user needs: how to open something, what this
// application is for, and something to open right now.
//
// SAMPLE MODELS ARE LINKS, NOT DOWNLOADS. Netron ships sample models; NetVis
// will not fetch anything over the network. A binary that reaches out on its own
// is a different security proposition from one that reads a file you gave it,
// and "single static binary, no runtime deps" is a product commitment. The
// onboarding surface therefore offers URLs the user can choose to visit, opened
// in their browser by their own action, plus any fixture already on disk.
#pragma once

namespace netvis {

class App;

// Draw the empty-state surface. Called from App::frame() when no model is open
// in the active tab; draws nothing otherwise. Covers the dock's central node so
// it reads as the application's front door rather than as another panel.
void draw_empty_state(App& app);

// Draw the Help menu's contents (an About entry, a keyboard-shortcut reference,
// and links to the docs). Called from inside the main menu bar's BeginMenu.
void draw_help_menu(App& app);

// Draw the shortcut-reference window when it is open. Kept separate from the
// menu so the same reference can be reached from the empty state and from the
// command palette, not only from a menu a keyboard-first user may never open.
void draw_shortcuts_window(App& app);

}  // namespace netvis
