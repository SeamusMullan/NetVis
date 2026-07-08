// core/MappedFile.h — memory-mapped read-only file.
//
// DECISION (spec §2.1, the whole product thesis): weights are NEVER eagerly
// loaded. A model file is mapped into the address space with mmap/MapViewOfFile;
// the OS pages in only the bytes we actually touch. Parsers read structure
// (which is a tiny fraction of a multi-GB file) and record offset+length for
// tensor payloads, which stay on disk until the weight inspector reads them.
// This is what makes cold-opening a 5 GB model sub-second.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "core/Result.h"

namespace netvis {

// Read-only view of a file's bytes. Owns the mapping; RAII-closed on destruction.
// Thread-safety: after a successful map, the byte range is immutable, so many
// threads may read concurrently without synchronization.
class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();

  MappedFile(MappedFile&&) noexcept;
  MappedFile& operator=(MappedFile&&) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  // Map `path` read-only. Returns an error if the file can't be opened/mapped.
  static Result<MappedFile> open(const std::string& path);

  const uint8_t* data() const { return data_; }
  uint64_t size() const { return size_; }
  const std::string& path() const { return path_; }
  bool valid() const { return data_ != nullptr; }

 private:
  const uint8_t* data_ = nullptr;
  uint64_t size_ = 0;
  std::string path_;

  // Platform handles, type-erased to keep the header clean.
  void* platform_handle_ = nullptr;  // fd (as intptr) on POSIX; HANDLEs on Win
  void* platform_handle2_ = nullptr;

  void close();
};

}  // namespace netvis
