// core/MappedFile.cpp — POSIX mmap / Windows MapViewOfFile implementation.
#include "core/MappedFile.h"

#include <utility>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace netvis {

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& o) noexcept
    : data_(o.data_),
      size_(o.size_),
      path_(std::move(o.path_)),
      platform_handle_(o.platform_handle_),
      platform_handle2_(o.platform_handle2_) {
  o.data_ = nullptr;
  o.size_ = 0;
  o.platform_handle_ = nullptr;
  o.platform_handle2_ = nullptr;
}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
  if (this != &o) {
    close();
    data_ = o.data_;
    size_ = o.size_;
    path_ = std::move(o.path_);
    platform_handle_ = o.platform_handle_;
    platform_handle2_ = o.platform_handle2_;
    o.data_ = nullptr;
    o.size_ = 0;
    o.platform_handle_ = nullptr;
    o.platform_handle2_ = nullptr;
  }
  return *this;
}

#if defined(_WIN32)

Result<MappedFile> MappedFile::open(const std::string& path) {
  HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    return err("cannot open file: " + path);

  LARGE_INTEGER sz;
  if (!GetFileSizeEx(file, &sz)) {
    CloseHandle(file);
    return err("cannot stat file: " + path);
  }
  if (sz.QuadPart == 0) {
    CloseHandle(file);
    return err("empty file: " + path);
  }

  HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping) {
    CloseHandle(file);
    return err("cannot create file mapping: " + path);
  }
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (!view) {
    CloseHandle(mapping);
    CloseHandle(file);
    return err("cannot map view: " + path);
  }

  MappedFile mf;
  mf.data_ = static_cast<const uint8_t*>(view);
  mf.size_ = static_cast<uint64_t>(sz.QuadPart);
  mf.path_ = path;
  mf.platform_handle_ = file;
  mf.platform_handle2_ = mapping;
  return mf;
}

void MappedFile::close() {
  if (data_) UnmapViewOfFile(const_cast<uint8_t*>(data_));
  if (platform_handle2_) CloseHandle(static_cast<HANDLE>(platform_handle2_));
  if (platform_handle_) CloseHandle(static_cast<HANDLE>(platform_handle_));
  data_ = nullptr;
  size_ = 0;
  platform_handle_ = nullptr;
  platform_handle2_ = nullptr;
}

#else  // POSIX

Result<MappedFile> MappedFile::open(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) return err("cannot open file: " + path);

  struct stat st;
  if (fstat(fd, &st) != 0) {
    ::close(fd);
    return err("cannot stat file: " + path);
  }
  if (st.st_size == 0) {
    ::close(fd);
    return err("empty file: " + path);
  }

  void* addr = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ,
                    MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    ::close(fd);
    return err("mmap failed: " + path);
  }

  // Hint the kernel: we read structure roughly front-to-back but touch weights
  // randomly and rarely. MADV_RANDOM keeps readahead from wasting bandwidth on
  // weight pages we will never fault in during a normal browse session.
  madvise(addr, static_cast<size_t>(st.st_size), MADV_RANDOM);

  MappedFile mf;
  mf.data_ = static_cast<const uint8_t*>(addr);
  mf.size_ = static_cast<uint64_t>(st.st_size);
  mf.path_ = path;
  mf.platform_handle_ = reinterpret_cast<void*>(static_cast<intptr_t>(fd));
  return mf;
}

void MappedFile::close() {
  if (data_) munmap(const_cast<uint8_t*>(data_), static_cast<size_t>(size_));
  if (platform_handle_) {
    int fd = static_cast<int>(reinterpret_cast<intptr_t>(platform_handle_));
    ::close(fd);
  }
  data_ = nullptr;
  size_ = 0;
  platform_handle_ = nullptr;
}

#endif

}  // namespace netvis
