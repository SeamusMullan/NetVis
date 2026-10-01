// SPDX-License-Identifier: Apache-2.0
// tests/test_temp_file_guard.cpp — the temp-file cleanup helper the other tests
// rely on to stay portable (see temp_file_guard.h).
//
// The contract under test is the destruction ORDER: a guard declared before a
// MappedFile on the same path deletes the file only after the mapping is gone.
// On POSIX that is merely tidy; on Windows deleting a still-mapped file fails,
// so these tests are what would turn red there if the helper (or MappedFile's
// handle lifetime) ever stopped honouring it.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/MappedFile.h"
#include "temp_file_guard.h"

using namespace netvis;

namespace {
std::string probe_path(const char* leaf) {
  return (std::filesystem::temp_directory_path() / leaf).string();
}
}  // namespace

TEST_CASE("TempFileGuard: deletes the file once the mapping that reads it is gone") {
  const std::string path = probe_path("nv_guard_mapped.bin");
  {
    netvis_test::TempFileGuard cleanup(path);  // declared first => destroyed last
    {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      out << "NETVIS";
    }
    auto mf = MappedFile::open(path);
    REQUIRE(mf);
    CHECK(mf->size() == 6);
    CHECK(std::filesystem::exists(path));  // still there while it is mapped
  }
  CHECK_FALSE(std::filesystem::exists(path));
}

TEST_CASE("TempFileGuard: cleans up when an exception leaves the scope") {
  // doctest's REQUIRE aborts a test by throwing; the trailing-remove pattern the
  // guard replaces would have leaked the file in exactly that case.
  const std::string path = probe_path("nv_guard_aborted.bin");
  auto leave_by_exception = [&] {
    netvis_test::TempFileGuard cleanup(path);
    {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      out << "x";
    }
    throw std::runtime_error("simulated REQUIRE failure");
  };
  CHECK_THROWS_AS(leave_by_exception(), std::runtime_error);
  CHECK_FALSE(std::filesystem::exists(path));
}

TEST_CASE("TempFileGuard: ownership moves with the guard") {
  // A helper that creates the guard first and returns it with the mapping relies
  // on this: the moved-from guard must not delete the file early (it is still
  // being used), and the destination must delete it exactly once at the end.
  const std::string path = probe_path("nv_guard_moved.bin");
  {
    auto handed_back = [&] {
      netvis_test::TempFileGuard local(path);
      {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << "x";
      }
      return netvis_test::TempFileGuard(std::move(local));
    }();  // `local` (moved-from) is destroyed here
    CHECK(std::filesystem::exists(path));
  }
  CHECK_FALSE(std::filesystem::exists(path));
}

TEST_CASE("TempFileGuard: a file that never existed is not an error") {
  // A destructor must not throw; remove() with an error_code reports ENOENT
  // quietly, so guarding a path the test never got to create is harmless.
  {
    netvis_test::TempFileGuard cleanup(probe_path("nv_guard_never_created.bin"));
  }
  CHECK_FALSE(std::filesystem::exists(probe_path("nv_guard_never_created.bin")));
}
