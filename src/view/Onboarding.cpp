// view/Onboarding.cpp — first-run empty state, Help menu, shortcuts (#105).
//
// The three surfaces here are the only places in the app that speak to someone
// who has not opened a model yet, so all three are held to the same rule: say
// what is true. The shortcut table is generated from one list that mirrors
// App::handle_shortcuts, the About block refuses to invent a version string it
// cannot see, and the sample-model section offers links WITHOUT pretending
// NetVis will follow them (Onboarding.h: a binary that reaches out on its own is
// a different security proposition from one that reads a file you gave it).
#include "view/Onboarding.h"

#include <cstddef>
#include <string>
#include <vector>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h (pulled in by
// App.h) references without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/ModelSession.h"
#include "view/App.h"

namespace netvis {

namespace {

// One keyboard binding, exactly as App::handle_shortcuts implements it. Kept as
// data so the reference table cannot fall out of step with itself row by row —
// a shortcut reference that lies is worse than none, because it costs the reader
// a failed attempt before they stop believing it.
struct Binding {
  const char* keys;
  const char* action;
};

constexpr Binding kFileBindings[] = {
    {"Ctrl+O", "Open a model"},
};

constexpr Binding kNavBindings[] = {
    {"F", "Fit the whole graph to the window"},
    {"Home", "Reset pan and zoom"},
    {"C", "Collapse every repeated block"},
    {"E", "Expand every repeated block"},
    {"Alt+Left", "Back through the nodes you focused"},
    {"Alt+Right", "Forward through the nodes you focused"},
    // #106, new in this release.
    {"Ctrl+Z", "Undo the last view change"},
    {"Ctrl+Y", "Redo the change you just undid"},
};

constexpr Binding kOverlayBindings[] = {
    {"Ctrl+F", "Search (toggles the search overlay)"},
    {"Ctrl+P", "Command palette (toggles it)"},
    {"Escape", "Close the search overlay and the command palette"},
};

// Where a sample model can be found. Never fetched — see the section body.
struct SampleSource {
  const char* label;
  const char* url;
};

constexpr SampleSource kSampleSources[] = {
    {"ONNX Model Zoo (.onnx)", "https://github.com/onnx/models"},
    {"Hugging Face, GGUF models (.gguf)",
     "https://huggingface.co/models?library=gguf"},
    {"Hugging Face, SafeTensors models (.safetensors)",
     "https://huggingface.co/models?library=safetensors"},
};

constexpr const char* kRepoUrl = "https://github.com/SeamusMullan/NetVis";
constexpr const char* kPluginDocUrl =
    "https://github.com/SeamusMullan/NetVis/blob/master/docs/plugin-abi.md";

// Render one binding group as a two-column table.
void draw_binding_table(const char* id, const Binding* rows, size_t count) {
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders |
                                ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_SizingStretchProp;
  if (!ImGui::BeginTable(id, 2, flags)) return;
  ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthFixed, 110.0f);
  ImGui::TableSetupColumn("does");
  for (size_t i = 0; i < count; ++i) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(rows[i].keys);
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted(rows[i].action);
  }
  ImGui::EndTable();
}

// A link the user can take somewhere themselves. Clicking COPIES; it does not
// navigate, and the tooltip says so. Rendering these as real hyperlinks would be
// the more familiar affordance and the dishonest one — a link that does not
// follow when clicked teaches the user to distrust the rest of the window, and
// making it follow would mean handing a URL to a platform shell, which is a
// capability this binary deliberately does not have.
void draw_copy_link(App& app, const char* url) {
  ImGui::PushID(url);
  if (ImGui::Selectable(url)) {
    ImGui::SetClipboardText(url);
    app.add_toast("Link copied to the clipboard", false);
  }
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip(
        "Click to copy. NetVis never opens a network connection —\n"
        "paste this into your browser, then open the file you download.");
  ImGui::PopID();
}

}  // namespace

void draw_empty_state(App& app) {
  ModelSession& session = app.session();

  // Nothing parsed yet. Two cases are NOT the empty state and must fall through:
  // a model that is loaded (obviously), and a model that is on its way in — a
  // panel headed "Open a model" drawn over an in-flight parse would read as if
  // the open had failed, when the status bar is at that moment showing progress
  // for it. A FAILED load does belong here: the error is reported by its own
  // toast and status line, and what the user needs next is the front door again.
  const LoadStage stage = session.stage();
  if (session.model() != nullptr) return;
  if (stage != LoadStage::Empty && stage != LoadStage::Failed) return;

  // Centred on the viewport until the user (or the dock layout) puts it
  // somewhere; dockable, so it can share the central node with Graph/Tensors.
  const ImGuiViewport* vp = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(
      ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f,
             vp->WorkPos.y + vp->WorkSize.y * 0.5f),
      ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(560.0f, 460.0f), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Welcome")) {
    ImGui::End();
    return;
  }

  // DEFERRED OPEN (mirrors PluginsPanel's §C.4 pattern). Both actions below run
  // App::open_file, which appends to App::recent_ and can switch the active tab —
  // so it would invalidate the `recent` reference this function iterates and
  // change which ViewState app.view() returns, mid-draw. Nothing is applied until
  // after ImGui::End(), when no reference into either is live.
  bool want_dialog = false;
  std::string want_path;

  ImGui::TextWrapped(
      "NetVis reads neural-network model files and shows you what is inside "
      "them — the compute graph, the tensors, and what they cost.");
  ImGui::Spacing();

  // --- How to open something ------------------------------------------------
  ImGui::SeparatorText("Open a model");
  if (ImGui::Button("Open model...", ImVec2(150.0f, 0.0f))) want_dialog = true;
  ImGui::SameLine();
  ImGui::TextDisabled("Ctrl+O, or File > Open");
  ImGui::BulletText("Drag a model file onto this window.");
  ImGui::BulletText(
      "ONNX, TFLite, SafeTensors, GGUF, PyTorch, OpenVINO, CoreML, Keras,"
      " NumPy.");

  // --- Somewhere to pick up where they left off -----------------------------
  const std::vector<std::string>& recent = app.recent_files();
  if (!recent.empty()) {
    ImGui::SeparatorText("Recent");
    for (size_t i = 0; i < recent.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      if (ImGui::Selectable(recent[i].c_str())) want_path = recent[i];
      ImGui::PopID();
    }
  }

  // --- Something to open right now ------------------------------------------
  ImGui::SeparatorText("Where to find models");
  ImGui::TextWrapped(
      "These are links, not downloads. NetVis does not fetch anything; open one "
      "in your browser if you want to, then open the file you get.");
  for (const SampleSource& s : kSampleSources) {
    ImGui::BulletText("%s", s.label);
    ImGui::Indent();
    draw_copy_link(app, s.url);
    ImGui::Unindent();
  }

  // --- The way back to the rest of the app ----------------------------------
  ImGui::SeparatorText("Getting around");
  if (ImGui::Button("Keyboard shortcuts")) app.view().show_shortcuts = true;
  ImGui::SameLine();
  if (ImGui::Button("Preferences")) app.view().show_preferences = true;
  ImGui::SameLine();
  ImGui::TextDisabled("Ctrl+P opens the command palette.");

  ImGui::End();

  // Deferred actions, applied with nothing live (see the note above).
  if (want_dialog) {
    app.open_file_dialog();
  } else if (!want_path.empty()) {
    app.open_file(want_path);
  }
}

void draw_help_menu(App& app) {
  // About, rendered inline rather than as its own window. A separate About window
  // needs a visibility flag, and ViewState is frozen for this release with none —
  // inventing a file-static one to hold three lines of text would be more state
  // than the content earns.
#ifdef NETVIS_VERSION_STRING
  ImGui::TextDisabled("NetVis %s", NETVIS_VERSION_STRING);
#else
  // The version exists only in CMake (`NETVIS_VERSION` feeds project(VERSION ...)
  // and from there CPack/the bundle plist); it is never compiled into the binary,
  // so there is no honest number to print here. Saying that is better than
  // hardcoding a string that goes stale on the next release and lies quietly.
  // Defining NETVIS_VERSION_STRING on the `netvis` target is all it needs.
  ImGui::TextDisabled("NetVis (version not compiled in)");
#endif
  ImGui::TextDisabled("Dear ImGui %s", ImGui::GetVersion());
  ImGui::Separator();

  if (ImGui::MenuItem("Keyboard shortcuts...")) app.view().show_shortcuts = true;
  ImGui::Separator();

  // Same click-to-copy contract as the empty state, for the same reason: a menu
  // entry that looked like it opened a browser and did not would be worse than
  // one that plainly says what it does.
  if (ImGui::BeginMenu("Documentation")) {
    if (ImGui::MenuItem("Copy project link")) {
      ImGui::SetClipboardText(kRepoUrl);
      app.add_toast("Link copied to the clipboard", false);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kRepoUrl);
    if (ImGui::MenuItem("Copy plugin ABI link")) {
      ImGui::SetClipboardText(kPluginDocUrl);
      app.add_toast("Link copied to the clipboard", false);
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", kPluginDocUrl);
    ImGui::EndMenu();
  }
}

void draw_shortcuts_window(App& app) {
  ViewState& vs = app.view();
  if (!vs.show_shortcuts) return;

  ImGui::SetNextWindowSize(ImVec2(430.0f, 480.0f), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("Keyboard shortcuts", &vs.show_shortcuts)) {
    ImGui::End();
    return;
  }

  ImGui::SeparatorText("Files");
  draw_binding_table("sc_file", kFileBindings, IM_ARRAYSIZE(kFileBindings));
  ImGui::SeparatorText("Graph");
  draw_binding_table("sc_nav", kNavBindings, IM_ARRAYSIZE(kNavBindings));
  ImGui::SeparatorText("Overlays");
  draw_binding_table("sc_overlay", kOverlayBindings,
                     IM_ARRAYSIZE(kOverlayBindings));

  ImGui::Spacing();
  // The one caveat that would otherwise look like a broken binding: the
  // unmodified keys and the Alt chords are suppressed while a text field has
  // focus, so typing a node name into search never collapses the graph behind it.
  ImGui::TextWrapped(
      "Single keys and the Alt chords are ignored while you are typing in a "
      "text field, so a search query never triggers them.");
  // Ctrl on every platform, including macOS: the handlers test ImGui's KeyCtrl,
  // which is the physical Control key there, not Command.
  ImGui::TextDisabled("Ctrl is the Control key on macOS too, not Command.");

  ImGui::End();
}

}  // namespace netvis
