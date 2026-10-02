// SPDX-License-Identifier: Apache-2.0
#pragma once
// Native file chooser (spec §8.7), non-blocking.
//
// The old path called tinyfiledialogs straight from the frame callback. On Linux
// that popen()s a shell running `zenity ...` and blocks until the user answers —
// so GLFW stops pumping events, the window never repaints and the WM greys it
// out as "not responding". This wrapper instead spawns the desktop's own chooser
// (zenity / kdialog / qarma / matedialog / yad — whichever the session has) in a
// child process on a worker thread, and the caller polls once per frame.
//
// macOS and Windows keep tinyfiledialogs: their native choosers are main-thread
// only and already modal-but-responsive, so there the answer is ready on the
// first poll().

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "parsers/Parser.h"

namespace netvis {

// Pure path/filter helpers, in the header rather than the .cpp so netvis_tests
// can exercise them: FileDialog.cpp is GUI/platform code compiled into the
// `netvis` target only, and the test executable links netvis_core alone.
namespace detail {

inline bool is_path_separator(char c) { return c == '/' || c == '\\'; }

// Drop trailing path separators from a picked path. A macOS package (a
// `.mlpackage` bundle) comes back from the system chooser as "/a/M.mlpackage/",
// and callers that show the file name or compare paths as strings (tab title,
// diff ladder, Recent list) want the same string a drag-and-drop gives. A bare
// root ("/" or "C:\") is kept as is.
inline std::string strip_trailing_separators(std::string s) {
  while (s.size() > 1 && is_path_separator(s.back())) {
    if (s.size() == 3 && s[1] == ':') break;  // "C:\" is a root; "C:" is not
    s.pop_back();
  }
  return s;
}

// Last path component, ignoring trailing separators ("/a/M.mlpackage/" ->
// "M.mlpackage"); the whole path when it has no separator. The view aliases
// `path`, so the caller's string must outlive it.
inline std::string_view basename_of(std::string_view path) {
  size_t end = path.size();
  while (end > 0 && is_path_separator(path[end - 1])) --end;
  path = path.substr(0, end);
  const size_t slash = path.find_last_of("/\\");
  return slash == std::string_view::npos ? path : path.substr(slash + 1);
}

// Strip everything after the first newline (a chooser prints one path per line;
// we never ask for multi-select) plus any trailing CR/whitespace.
inline std::string first_line(std::string s) {
  const size_t nl = s.find('\n');
  if (nl != std::string::npos) s.resize(nl);
  while (!s.empty() && (s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s;
}

// "*.png" -> ".png". Empty when the pattern is not a simple suffix glob.
inline std::string suffix_of(const std::string& pattern) {
  if (pattern.size() < 3 || pattern.compare(0, 2, "*.") != 0) return {};
  if (pattern.find_first_of("*?[", 2) != std::string::npos) return {};
  return pattern.substr(1);
}

// A Save chooser hands back exactly what the user typed. If that has no
// extension, apply the first filter's so "diff" becomes "diff.md" — the export
// writers key off the caller's format, not the name, so a bare name would
// otherwise produce an extension-less file.
inline std::string apply_default_extension(
    std::string path, const std::vector<std::string>& patterns) {
  if (path.empty() || patterns.empty()) return path;
  const size_t slash = path.find_last_of('/');
  const size_t name = slash == std::string::npos ? 0 : slash + 1;
  if (path.find('.', name) != std::string::npos) return path;
  return path + suffix_of(patterns.front());
}

// "*.onnx" -> "*.[oO][nN][nN][xX]". The GTK and Qt choosers match a glob
// case-sensitively on Linux, yet detection lowercases the extension, so
// `Model.ONNX` opens by drag-and-drop but would be hidden by a plain "*.onnx"
// filter. Only the Linux helper choosers get this, and only for Open (see
// helper_globs); tinyfd's macOS path slices the plain "*.ext" itself, so
// openable_patterns() stays plain. Anything that is not a simple suffix glob is
// returned unchanged.
inline std::string case_folded_glob(const std::string& pattern) {
  if (suffix_of(pattern).empty()) return pattern;
  std::string out = "*.";
  for (size_t i = 2; i < pattern.size(); ++i) {
    const char c = pattern[i];
    const bool upper = c >= 'A' && c <= 'Z';
    const bool lower = c >= 'a' && c <= 'z';
    if (upper || lower) {
      const char lo = upper ? static_cast<char>(c - 'A' + 'a') : c;
      out += '[';
      out += lo;
      out += static_cast<char>(lo - 'a' + 'A');
      out += ']';
    } else {
      out += c;
    }
  }
  return out;
}

// The space-separated glob list the Linux helper choosers take.
//
// Open (`open` true): case-folded, so `Model.ONNX` is listed under "All supported
// models" the way detection treats it.
//
// Save (`open` false): the plain globs, never folded. A Save chooser uses its
// filter to pick the extension it appends to a bare name, and KDE's KFileWidget
// (what kdialog and qarma show on Plasma) deliberately skips any pattern holding
// '[' or ']' when it works that out. A folded "*.[nN][pP][yY]" therefore stops it
// appending ".npy" AND makes its overwrite check run on the bare name; NetVis
// adds the suffix afterwards (apply_default_extension), so an existing "w.npy"
// would be replaced without a prompt. Plain "*.npy" keeps both working.
inline std::string helper_globs(const std::vector<std::string>& patterns,
                                bool open) {
  std::string out;
  for (const std::string& p : patterns) {
    if (!out.empty()) out += ' ';
    out += open ? case_folded_glob(p) : p;
  }
  return out;
}

// kdialog takes ONE argument holding newline-separated filters.
//
// Open: Qt-style "Description (globs)" followed by "All files (*)", so a format
// with no listed extension (a WASM-plugin format, or a file detection would
// accept under another name) can still be picked. The Qt style is deliberate:
// kdialog hands the argument to QFileDialog::setNameFilter after turning every
// '|' into a newline, so the KDE-style "globs|Description" would become two
// filters, the second one a glob list made of the words of the label (it would
// match files literally named "All" or "files"). A filter without '|' reaches Qt
// unchanged, and the Plasma platform theme reads the parenthesised globs.
//
// Save: the single "globs|Description" filter, as before #135. No "All files"
// entry (a Save chooser has no use for one) and no change to what KDE's save
// dialog does with the filter.
inline std::string kdialog_filter(const std::string& globs,
                                  const std::string& description, bool open) {
  if (!open) return globs + "|" + description;
  return description + " (" + globs + ")\nAll files (*)";
}

// The repeated --file-filter arguments of the zenity family. Open gets the model
// filter followed by an "All files" entry; Save gets the model filter only.
inline std::vector<std::string> zenity_filters(const std::string& globs,
                                               const std::string& description,
                                               bool open) {
  std::vector<std::string> out;
  if (globs.empty()) return out;
  out.push_back("--file-filter=" + description + " | " + globs);
  if (open) out.emplace_back("--file-filter=All files | *");
  return out;
}

}  // namespace detail


// "mlpackage" is the one extension the Open choosers offer that detect_format()
// does not act on: a CoreML bundle is a directory that engine/ModelPath resolves
// to its inner model, and macOS choosers treat a package as a single file, so
// listing it lets the bundle be picked there. The GTK/Qt/Win32 choosers cannot
// select a directory; drag-and-drop or the CLI argument still open it. Anything
// else the detector acts on is NOT listed here - it comes from kExtensionFormats.
inline constexpr std::string_view kDialogOnlyExtensions[] = {"mlpackage"};

// Every extension the "open a model" choosers offer (App::open_file_dialog and
// the DiffPanel comparison picker both build their filter from this): the
// detector's own table (kExtensionFormats, parsers/Parser.h) in table order, then
// kDialogOnlyExtensions. There is no second hand-kept list, so a format that
// parses cannot be unpickable and a stale or misspelled entry cannot exist.
// Lowercase, no dot.
inline std::vector<std::string_view> openable_extensions() {
  std::vector<std::string_view> out;
  out.reserve(std::size(kExtensionFormats) + std::size(kDialogOnlyExtensions));
  for (const ExtensionFormat& e : kExtensionFormats) out.push_back(e.ext);
  for (const std::string_view ext : kDialogOnlyExtensions) {
    if (std::find(out.begin(), out.end(), ext) == out.end()) out.push_back(ext);
  }
  return out;
}

// First filter in the chooser. The Linux helpers (zenity family, and kdialog)
// and Windows follow it with an "All files" entry; macOS's chooser takes a list
// of types only, so it shows neither this label nor an "All files" fallback.
inline constexpr const char* kOpenFilterDescription = "All supported models";

// openable_extensions() as chooser globs ("*.onnx", ...), in the same order.
inline std::vector<std::string> openable_patterns() {
  std::vector<std::string> out;
  for (const std::string_view ext : openable_extensions()) {
    out.push_back("*." + std::string(ext));
  }
  return out;
}

class FileDialog {
 public:
  enum class Mode { Open, Save };

  FileDialog() = default;
  // Starts the chooser immediately. `patterns` are globs ("*.onnx"); on Save the
  // first pattern's extension is appended when the user typed a bare name.
  // `default_path` seeds the file name (Save) or the start folder (Open).
  FileDialog(Mode mode, const std::string& title, const std::string& default_path,
             const std::vector<std::string>& patterns,
             const std::string& filter_description);

  ~FileDialog();
  FileDialog(FileDialog&&) noexcept = default;
  FileDialog& operator=(FileDialog&&) noexcept = default;
  FileDialog(const FileDialog&) = delete;
  FileDialog& operator=(const FileDialog&) = delete;

  bool in_flight() const { return state_ != nullptr; }

  // Non-blocking; call once per frame. Returns true exactly once — on the frame
  // the chooser closes. *out is the picked path, or empty when the user
  // cancelled.
  bool poll(std::string* out);

 private:
  // Shared with the (detached) worker thread and defined in the .cpp. Held by
  // shared_ptr so destroying the FileDialog never has to wait for a chooser the
  // user has not answered: the destructor asks the helper to quit and walks
  // away, and the worker tidies up against a state block that outlives it.
  struct State;
  std::shared_ptr<State> state_;
};

// False only on Linux with no chooser binary installed. Call sites turn it into
// a toast instead of opening a dialog that can never appear.
bool file_dialog_available();

// Panel-friendly wrapper: one in-flight chooser plus a caller-chosen tag, so a
// draw function can start a dialog on click and finish the work a few frames
// later. Hold one as a function-local static.
class DeferredFileDialog {
 public:
  // No-op while a chooser is already up (a second one would race the first).
  void start(FileDialog::Mode mode, const std::string& title,
             const std::string& default_path,
             const std::vector<std::string>& patterns,
             const std::string& filter_description, int tag = 0);

  bool busy() const { return active_; }

  // True once, on the frame the chooser closes with a pick. Cancellation clears
  // the request and returns false.
  bool ready(std::string* out, int* tag = nullptr);

 private:
  FileDialog dialog_;
  bool active_ = false;
  int tag_ = 0;
};

}  // namespace netvis
