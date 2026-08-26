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

#include <memory>
#include <string>
#include <vector>

namespace netvis {

// Pure path helpers, in the header rather than the .cpp so netvis_tests can
// exercise them: FileDialog.cpp is GUI/platform code compiled into the `netvis`
// target only, and the test executable links netvis_core alone.
namespace detail {

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

}  // namespace detail


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
