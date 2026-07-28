// view/CommandPalette.h — Ctrl+P fuzzy command palette (#59).
//
// A single entry point drawn each frame from App::frame(). When
// App::command_palette_open() is set (toggled by Ctrl+P in handle_shortcuts), it
// pops a centered overlay with a text field and a fuzzy-ranked list of every
// registered action; Enter (or click) runs the highlighted one and closes.
//
// DESIGN: actions are rebuilt each frame from the live App state (so
// context-dependent items like "Switch to tab: <name>" and toggles reflecting
// current values are always current), then ranked by the SAME fuzzy_score used
// by the model search bar (engine/SearchIndex.h) — one matcher, consistent feel.
// The palette owns no persistent state beyond ImGui's input buffer + a selected
// index kept as a function-local static (single palette, main-thread ImGui).
#pragma once

namespace netvis {
class App;
void draw_command_palette(App& app);
}  // namespace netvis
