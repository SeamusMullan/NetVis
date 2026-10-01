// SPDX-License-Identifier: Apache-2.0
// tests/temp_file_guard.h — delete a temp file when a test scope ends.
//
// Why a guard instead of a trailing `std::filesystem::remove(path)`: Windows
// refuses to delete a file that still has an open handle or a mapped view, and
// MappedFile holds both until it is destroyed (it also opens the file with
// FILE_SHARE_READ only). A test that maps a temp file and then removes it by
// path while the mapping is still alive passes on POSIX, where unlink-while-
// mapped is fine, and throws filesystem_error on Windows.
//
// Declare the guard BEFORE the MappedFile (or ModelCache) that reads the file:
// locals are destroyed in reverse order, so the mapping is released first and
// the delete runs second. It also cleans up when a REQUIRE aborts the test
// early, which the trailing-remove pattern never did.
//
// The delete is best-effort and never throws. A failed cleanup must not fail a
// test whose assertions already passed, and a destructor must not throw anyway.
#pragma once

#include <filesystem>
#include <string>
#include <system_error>
#include <utility>

namespace netvis_test {

class TempFileGuard {
 public:
  explicit TempFileGuard(std::string path) : path_(std::move(path)) {}
  ~TempFileGuard() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  TempFileGuard(const TempFileGuard&) = delete;
  TempFileGuard& operator=(const TempFileGuard&) = delete;

 private:
  std::string path_;
};

}  // namespace netvis_test
