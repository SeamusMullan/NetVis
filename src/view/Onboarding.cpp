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
constexpr const char* kAuthor = "Seamus Mullan";
constexpr const char* kAuthorUrl = "https://github.com/SeamusMullan";
constexpr const char* kAuthorEmail = "seamusmullan2023@gmail.com";

// The build stamp. Every one of these comes from CMake (see the netvis target's
// target_compile_definitions) because none of it can be recovered at runtime.
// Each falls back to a string that says it is missing rather than to a plausible
// guess — an About window that invents its own provenance is worse than one that
// admits a gap, because a bug report quoting it would send the reader hunting a
// toolchain that never built the binary.
#ifdef NETVIS_VERSION_STRING
constexpr const char* kVersion = NETVIS_VERSION_STRING;
#else
constexpr const char* kVersion = "(not compiled in)";
#endif
#ifdef NETVIS_BUILD_CXX_STANDARD
constexpr const char* kCxxStandard = "C++" NETVIS_BUILD_CXX_STANDARD;
#else
constexpr const char* kCxxStandard = "(unknown)";
#endif
#ifdef NETVIS_BUILD_COMPILER
constexpr const char* kCompiler = NETVIS_BUILD_COMPILER;
#else
constexpr const char* kCompiler = "(unknown)";
#endif
#ifdef NETVIS_BUILD_CMAKE_VERSION
constexpr const char* kCMakeVersion = NETVIS_BUILD_CMAKE_VERSION;
#else
constexpr const char* kCMakeVersion = "(unknown)";
#endif
#ifdef NETVIS_BUILD_SYSTEM
constexpr const char* kBuildSystem = NETVIS_BUILD_SYSTEM;
#else
constexpr const char* kBuildSystem = "(unknown)";
#endif
#ifdef NETVIS_BUILD_CONFIG
constexpr const char* kBuildConfig = NETVIS_BUILD_CONFIG;
#else
constexpr const char* kBuildConfig = "(unknown)";
#endif

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
void draw_copy_link(App& app, const char* url, const char* tip = nullptr) {
  ImGui::PushID(url);
  if (ImGui::Selectable(url)) {
    ImGui::SetClipboardText(url);
    app.add_toast("Link copied to the clipboard", false);
  }
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("%s", tip != nullptr
                                ? tip
                                : "Click to copy. NetVis never opens a network "
                                  "connection \u2014\npaste this into your "
                                  "browser, then open the file you download.");
  ImGui::PopID();
}

// Percent-encode everything outside RFC 3986's unreserved set, for the mailto:
// URL below. Substituting a couple of characters by hand would be shorter and
// would also turn "C++20" into "C  20" in the subject a recipient sees.
void append_encoded(std::string& out, const std::string& s) {
  static const char kHex[] = "0123456789ABCDEF";
  for (char ch : s) {
    const unsigned char c = static_cast<unsigned char>(ch);
    const bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                            (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                            c == '.' || c == '~';
    if (unreserved) {
      out += static_cast<char>(c);
    } else {
      out += '%';
      out += kHex[c >> 4];
      out += kHex[c & 0x0F];
    }
  }
}

// A mailto: URL whose subject carries the "[NetVis <version>]" prefix the author
// filters on, and whose body already quotes the build stamp \u2014 the three things
// every bug report needs (which version, which toolchain, which platform)
// answered before it is sent, rather than asked for in a reply.
std::string author_mailto() {
  std::string url = "mailto:";
  url += kAuthorEmail;
  url += "?subject=";
  append_encoded(url, "[NetVis " + std::string(kVersion) + "] ");
  url += "&body=";
  std::string body = "\n\n--\nNetVis ";
  body += kVersion;
  body += "\n";
  body += kCxxStandard;
  body += "  |  ";
  body += kCompiler;
  body += "  |  CMake ";
  body += kCMakeVersion;
  body += "\n";
  body += kBuildSystem;
  body += "  |  ";
  body += kBuildConfig;
  body += "\n";
  append_encoded(url, body);
  return url;
}

// The logo, drawn from the same primitives as assets/netvis.svg (five nodes,
// five edges, rounded backdrop) instead of decoding assets/netvis.png. That
// keeps the single-static-binary invariant intact \u2014 no image decoder, no GL
// texture, no runtime asset path to resolve differently in the build tree, the
// .app bundle and the installed tree \u2014 and it stays sharp at any ui_scale.
void draw_logo(float side) {
  const ImVec2 o = ImGui::GetCursorScreenPos();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const float s = side / 512.0f;  // the SVG's viewBox is 512x512
  auto at = [&](float x, float y) { return ImVec2(o.x + x * s, o.y + y * s); };

  struct Node {
    float x, y, r;
    ImU32 col;
  };
  static const Node kNodes[] = {
      {136.0f, 150.0f, 40.0f, IM_COL32(0x5a, 0xa9, 0xf0, 0xff)},
      {136.0f, 362.0f, 40.0f, IM_COL32(0x60, 0xbe, 0x96, 0xff)},
      {256.0f, 256.0f, 48.0f, IM_COL32(0xd8, 0x64, 0xd0, 0xff)},
      {376.0f, 150.0f, 40.0f, IM_COL32(0xd6, 0xb2, 0x48, 0xff)},
      {376.0f, 362.0f, 40.0f, IM_COL32(0x5a, 0xa9, 0xf0, 0xff)},
  };
  static const int kEdges[][2] = {{0, 2}, {1, 2}, {2, 3}, {2, 4}, {0, 1}};

  // Flat mid-tone instead of the SVG's gradient: AddRectFilledMultiColor has no
  // rounding parameter, and a square corner would read as a bug at this size.
  dl->AddRectFilled(o, ImVec2(o.x + side, o.y + side),
                    IM_COL32(0x15, 0x1d, 0x27, 0xff), 96.0f * s);
  for (const auto& e : kEdges) {
    dl->AddLine(at(kNodes[e[0]].x, kNodes[e[0]].y),
                at(kNodes[e[1]].x, kNodes[e[1]].y),
                IM_COL32(0x3a, 0x46, 0x58, 0xff), 10.0f * s);
  }
  for (const Node& n : kNodes) {
    dl->AddCircleFilled(at(n.x, n.y), n.r * s, n.col, 32);
  }
  ImGui::Dummy(ImVec2(side, side));  // claim the space we just drew into
}

// One label/value row of the build table.
void build_row(const char* label, const char* value) {
  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  ImGui::TextDisabled("%s", label);
  ImGui::TableSetColumnIndex(1);
  ImGui::TextUnformatted(value);
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
      "them \u2014 the compute graph, the tensors, and what they cost.");
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
  // The version at a glance, with the rest of the provenance a click away in the
  // About window (which the status-bar credit also opens).
  ImGui::TextDisabled("NetVis %s", kVersion);
  ImGui::TextDisabled("Dear ImGui %s", ImGui::GetVersion());
  ImGui::Separator();

  if (ImGui::MenuItem("About NetVis...")) app.view().show_about = true;
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

void draw_about_window(App& app) {
  ViewState& vs = app.view();
  if (!vs.show_about) return;

  ImGui::SetNextWindowSize(ImVec2(520.0f, 460.0f), ImGuiCond_FirstUseEver);
  if (!ImGui::Begin("About NetVis", &vs.show_about)) {
    ImGui::End();
    return;
  }

  // --- Identity -------------------------------------------------------------
  const float logo = 88.0f;
  draw_logo(logo);
  ImGui::SameLine();
  ImGui::BeginGroup();
  ImGui::TextUnformatted("NetVis");
  ImGui::TextDisabled("Version %s", kVersion);
  ImGui::Spacing();
  ImGui::TextUnformatted("Created By Seamus Mullan");
  ImGui::TextDisabled("Neural-network model viewer.");
  ImGui::EndGroup();

  // --- Author ---------------------------------------------------------------
  // Same click-to-copy contract as the empty state and the Help menu: this
  // binary has no shell-launch capability and gains none for an About window.
  // A row that looked like it opened a browser or a mail client and did not
  // would teach the reader to distrust the version string above it.
  ImGui::SeparatorText("Author");
  ImGui::TextDisabled("Click any line below to copy it. Nothing is opened for you.");
  ImGui::BulletText("GitHub");
  ImGui::Indent();
  draw_copy_link(app, kAuthorUrl, "Click to copy this URL to the clipboard.");
  ImGui::Unindent();
  ImGui::BulletText("Email");
  ImGui::Indent();
  draw_copy_link(app, kAuthorEmail, "Click to copy this address to the clipboard.");
  if (ImGui::Button("Copy mailto link (with subject + build info)")) {
    ImGui::SetClipboardText(author_mailto().c_str());
    app.add_toast("mailto link copied to the clipboard", false);
  }
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip(
        "Copies a mailto: URL. Paste it into your browser or mail client:\n"
        "the subject is prefixed \"[NetVis %s]\" and the body already quotes\n"
        "the build details below.",
        kVersion);
  }
  ImGui::Unindent();

  // --- Build ----------------------------------------------------------------
  ImGui::SeparatorText("This build");
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_SizingStretchProp;
  if (ImGui::BeginTable("about_build", 2, flags)) {
    ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthFixed, 130.0f);
    ImGui::TableSetupColumn("value");
    build_row("Version", kVersion);
    build_row("Configuration", kBuildConfig);
    build_row("Language", kCxxStandard);
    build_row("Compiler", kCompiler);
    build_row("CMake", kCMakeVersion);
    build_row("Built for", kBuildSystem);
    build_row("Dear ImGui", ImGui::GetVersion());
    ImGui::EndTable();
  }
  ImGui::TextDisabled("Rendering backend: GLFW + OpenGL 3");

  ImGui::SeparatorText("Project");
  draw_copy_link(app, kRepoUrl, "Click to copy this URL to the clipboard.");

  ImGui::End();
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
