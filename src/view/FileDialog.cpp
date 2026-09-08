// SPDX-License-Identifier: Apache-2.0
#include "view/FileDialog.h"

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

#if defined(__linux__) || (defined(__unix__) && !defined(__APPLE__))
#define NETVIS_FD_HELPER_PROCESS 1
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#else
#define NETVIS_FD_HELPER_PROCESS 0
// tinyfiledialogs (C) — declared here rather than pulling its header, which is
// not on our include path. The definitions link from the tinyfiledialogs TU.
extern "C" {
char* tinyfd_openFileDialog(const char* aTitle, const char* aDefaultPathAndFile,
                            int aNumOfFilterPatterns,
                            const char* const* aFilterPatterns,
                            const char* aSingleFilterDescription,
                            int aAllowMultipleSelects);
char* tinyfd_saveFileDialog(const char* aTitle, const char* aDefaultPathAndFile,
                            int aNumOfFilterPatterns,
                            const char* const* aFilterPatterns,
                            const char* aSingleFilterDescription);
}
#endif

namespace netvis {

namespace {

// first_line / suffix_of / apply_default_extension now live in FileDialog.h so
// netvis_tests can reach them (this TU is GUI-only and never linked into
// netvis_core). Pulled in unqualified here to keep the call sites below intact.
using detail::apply_default_extension;
using detail::first_line;

#if NETVIS_FD_HELPER_PROCESS

// PATH lookup without a shell (popen would drag in the /bin/sh dependency this
// whole file exists to avoid).
bool executable_in_path(const char* exe) {
  const char* path = std::getenv("PATH");
  if (path == nullptr) return false;
  const std::string p(path);
  size_t start = 0;
  while (start <= p.size()) {
    size_t colon = p.find(':', start);
    if (colon == std::string::npos) colon = p.size();
    std::string dir = p.substr(start, colon - start);
    if (dir.empty()) dir = ".";
    if (::access((dir + "/" + exe).c_str(), X_OK) == 0) return true;
    start = colon + 1;
  }
  return false;
}

// The desktop's own chooser, preferring the toolkit the session runs. zenity 4
// and kdialog both route through xdg-desktop-portal when one is running, so this
// lands on the real system dialog rather than a look-alike.
const char* detect_helper() {
  static const char* const kQtFirst[] = {"kdialog", "qarma", "zenity",
                                         "matedialog", "yad"};
  static const char* const kGtkFirst[] = {"zenity", "matedialog", "yad",
                                          "qarma", "kdialog"};
  const char* desktop = std::getenv("XDG_CURRENT_DESKTOP");
  const bool qt = desktop != nullptr && (std::strstr(desktop, "KDE") != nullptr ||
                                         std::strstr(desktop, "LXQt") != nullptr);
  const char* const* order = qt ? kQtFirst : kGtkFirst;
  for (int i = 0; i < 5; ++i) {
    if (executable_in_path(order[i])) return order[i];
  }
  return nullptr;
}

// Cached: the answer cannot change while the app runs, and every menu open would
// otherwise re-walk PATH five times.
const char* helper_binary() {
  static const char* const kHelper = detect_helper();
  return kHelper;
}

// kdialog wants one "glob glob|Description" string; the zenity family wants a
// repeated --file-filter=Description | glob glob.
std::vector<std::string> helper_argv(const char* helper, FileDialog::Mode mode,
                                     const std::string& title,
                                     const std::string& default_path,
                                     const std::vector<std::string>& patterns,
                                     const std::string& description) {
  std::string globs;
  for (const std::string& p : patterns) {
    if (!globs.empty()) globs += ' ';
    globs += p;
  }

  std::vector<std::string> argv{helper};
  if (std::strcmp(helper, "kdialog") == 0) {
    argv.emplace_back(mode == FileDialog::Mode::Save ? "--getsavefilename"
                                                     : "--getopenfilename");
    argv.emplace_back(default_path.empty() ? "." : default_path);
    if (!globs.empty()) argv.emplace_back(globs + "|" + description);
    if (!title.empty()) {
      argv.emplace_back("--title");
      argv.emplace_back(title);
    }
    return argv;
  }

  // zenity / matedialog / qarma / yad.
  if (std::strcmp(helper, "yad") == 0) {
    argv.emplace_back("--file");
  } else {
    argv.emplace_back("--file-selection");
  }
  if (mode == FileDialog::Mode::Save) {
    argv.emplace_back("--save");
    argv.emplace_back("--confirm-overwrite");
  }
  if (!title.empty()) argv.emplace_back("--title=" + title);
  if (!default_path.empty()) argv.emplace_back("--filename=" + default_path);
  if (!globs.empty()) {
    argv.emplace_back("--file-filter=" + description + " | " + globs);
    argv.emplace_back("--file-filter=All files | *");
  }
  return argv;
}

// Fork/exec the chooser and read the picked path off its stdout. Runs on a
// worker thread; the caller polls the shared state so the render loop keeps drawing.
std::string run_helper(const std::vector<std::string>& argv,
                       const std::atomic<int>& cancelled,
                       std::atomic<int>& child_pid) {
  // Build the char* vector BEFORE forking: between fork() and exec() in a
  // multi-threaded process only async-signal-safe calls are legal, and that
  // rules out allocating.
  std::vector<char*> raw;
  raw.reserve(argv.size() + 1);
  for (const std::string& a : argv) raw.push_back(const_cast<char*>(a.c_str()));
  raw.push_back(nullptr);

  int fds[2];
  if (::pipe(fds) != 0) return {};

  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    return {};
  }
  if (pid == 0) {
    ::dup2(fds[1], STDOUT_FILENO);
    ::close(fds[0]);
    ::close(fds[1]);
    ::execvp(raw[0], raw.data());
    ::_exit(127);
  }

  child_pid.store(static_cast<int>(pid));
  // The owner may have been destroyed between the checks above and the fork, in
  // which case its SIGTERM found no pid to aim at. Re-check and close the
  // chooser ourselves so no orphan dialog is left on screen.
  if (cancelled.load()) ::kill(pid, SIGTERM);
  ::close(fds[1]);
  std::string out;
  char buf[512];
  for (;;) {
    const ssize_t n = ::read(fds[0], buf, sizeof(buf));
    if (n > 0) {
      out.append(buf, static_cast<size_t>(n));
    } else if (n < 0 && errno == EINTR) {
      continue;
    } else {
      break;
    }
  }
  ::close(fds[0]);

  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  child_pid.store(0);

  // Non-zero exit means cancelled (1) or the helper failed to run (127).
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return {};
  return first_line(std::move(out));
}

#endif  // NETVIS_FD_HELPER_PROCESS

}  // namespace

bool file_dialog_available() {
#if NETVIS_FD_HELPER_PROCESS
  return helper_binary() != nullptr;
#else
  return true;
#endif
}

// Written by the worker thread, read by the UI thread through poll().
struct FileDialog::State {
  std::atomic<bool> done{false};
  std::atomic<int> cancelled{0};   // owner went away; close the chooser
  std::atomic<int> child_pid{0};   // 0 when no helper process is running
  std::mutex mu;
  std::string result;              // guarded by mu; valid once done is set
};

FileDialog::FileDialog(Mode mode, const std::string& title,
                       const std::string& default_path,
                       const std::vector<std::string>& patterns,
                       const std::string& filter_description) {
#if NETVIS_FD_HELPER_PROCESS
  const char* helper = helper_binary();
  if (helper == nullptr) return;  // in_flight() stays false; caller toasts.
  std::vector<std::string> argv =
      helper_argv(helper, mode, title, default_path, patterns, filter_description);
  state_ = std::make_shared<State>();
  // DETACHED, not a std::async future: ~future() joins, which at shutdown would
  // mean blocking the app until the user answers a chooser they can no longer
  // see. The state block is shared, so the worker stays valid either way.
  std::thread([state = state_, argv = std::move(argv), mode, patterns] {
    std::string picked =
        run_helper(argv, state->cancelled, state->child_pid);
    if (mode == Mode::Save)
      picked = apply_default_extension(std::move(picked), patterns);
    {
      std::lock_guard<std::mutex> lock(state->mu);
      state->result = std::move(picked);
    }
    state->done.store(true);
  }).detach();
#else
  // macOS/Windows: the native chooser must run on the thread that owns the
  // window, and it pumps the host event loop itself, so it does not freeze us.
  std::vector<const char*> raw;
  raw.reserve(patterns.size());
  for (const std::string& p : patterns) raw.push_back(p.c_str());
  const char* picked =
      mode == Mode::Save
          ? tinyfd_saveFileDialog(title.c_str(), default_path.c_str(),
                                  static_cast<int>(raw.size()), raw.data(),
                                  filter_description.c_str())
          : tinyfd_openFileDialog(title.c_str(), default_path.c_str(),
                                  static_cast<int>(raw.size()), raw.data(),
                                  filter_description.c_str(), 0);
  state_ = std::make_shared<State>();
  std::string result = picked != nullptr ? std::string(picked) : std::string();
  if (mode == Mode::Save)
    result = apply_default_extension(std::move(result), patterns);
  state_->result = std::move(result);
  state_->done.store(true);
#endif
}

FileDialog::~FileDialog() {
#if NETVIS_FD_HELPER_PROCESS
  if (!state_ || state_->done.load()) return;
  // Order matters: publish the cancel FIRST, so a worker that has not forked yet
  // closes its own child (it re-reads the flag right after fork).
  state_->cancelled.store(1);
  const int pid = state_->child_pid.load();
  if (pid > 0) ::kill(static_cast<pid_t>(pid), SIGTERM);
#endif
}

bool FileDialog::poll(std::string* out) {
  if (!state_) return false;
  if (!state_->done.load()) return false;
  std::string result;
  {
    std::lock_guard<std::mutex> lock(state_->mu);
    result = std::move(state_->result);
  }
  state_.reset();  // in_flight() goes false; poll() answers exactly once
  if (out != nullptr) *out = std::move(result);
  return true;
}

void DeferredFileDialog::start(FileDialog::Mode mode, const std::string& title,
                               const std::string& default_path,
                               const std::vector<std::string>& patterns,
                               const std::string& filter_description, int tag) {
  if (active_) return;
  dialog_ = FileDialog(mode, title, default_path, patterns, filter_description);
  if (!dialog_.in_flight()) return;
  active_ = true;
  tag_ = tag;
}

bool DeferredFileDialog::ready(std::string* out, int* tag) {
  if (!active_) return false;
  std::string picked;
  if (!dialog_.poll(&picked)) return false;
  active_ = false;
  if (picked.empty()) return false;  // cancelled
  if (out != nullptr) *out = std::move(picked);
  if (tag != nullptr) *tag = tag_;
  return true;
}

}  // namespace netvis
