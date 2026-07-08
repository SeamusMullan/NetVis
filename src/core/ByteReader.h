// core/ByteReader.h — bounds-checked cursor over a byte range.
//
// DECISION (spec §6): every parser reads through ByteReader so malformed input
// can never read out of bounds — an over-run returns an error with the byte
// offset instead of a segfault. It also carries a `payload_reads` counter used
// by tests to ASSERT that structural parsing never touches tensor payloads
// (spec §10, the "counting ByteReader"): structural reads use the checked
// accessors; a payload access is deliberately routed through mark_payload_read.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include "core/Result.h"

namespace netvis {

class ByteReader {
 public:
  ByteReader(const uint8_t* data, uint64_t size) : data_(data), size_(size) {}

  uint64_t pos() const { return pos_; }
  uint64_t size() const { return size_; }
  uint64_t remaining() const { return size_ - pos_; }
  bool at_end() const { return pos_ >= size_; }
  const uint8_t* data() const { return data_; }

  // Absolute-position seek; clamps into [0,size].
  Result<bool> seek(uint64_t p) {
    if (p > size_) return err("seek out of range", p);
    pos_ = p;
    return true;
  }
  void skip(uint64_t n) { pos_ = (n > remaining()) ? size_ : pos_ + n; }

  bool has(uint64_t n) const { return n <= remaining(); }

  Result<uint8_t> u8() {
    if (remaining() < 1) return err("eof reading u8", pos_);
    return data_[pos_++];
  }

  Result<uint16_t> u16le() {
    if (remaining() < 2) return err("eof reading u16", pos_);
    uint16_t v;
    std::memcpy(&v, data_ + pos_, 2);
    pos_ += 2;
    return le16(v);
  }

  Result<uint32_t> u32le() {
    if (remaining() < 4) return err("eof reading u32", pos_);
    uint32_t v;
    std::memcpy(&v, data_ + pos_, 4);
    pos_ += 4;
    return le32(v);
  }

  Result<uint64_t> u64le() {
    if (remaining() < 8) return err("eof reading u64", pos_);
    uint64_t v;
    std::memcpy(&v, data_ + pos_, 8);
    pos_ += 8;
    return le64(v);
  }

  Result<int32_t> i32le() {
    auto r = u32le();
    if (!r) return r.error();
    return static_cast<int32_t>(*r);
  }
  Result<int64_t> i64le() {
    auto r = u64le();
    if (!r) return r.error();
    return static_cast<int64_t>(*r);
  }
  Result<float> f32le() {
    auto r = u32le();
    if (!r) return r.error();
    float f;
    uint32_t v = *r;
    std::memcpy(&f, &v, 4);
    return f;
  }
  Result<double> f64le() {
    auto r = u64le();
    if (!r) return r.error();
    double d;
    uint64_t v = *r;
    std::memcpy(&d, &v, 8);
    return d;
  }

  // Read `n` raw bytes into a std::string (used for names/headers, never for
  // tensor payloads).
  Result<std::string> bytes(uint64_t n) {
    if (remaining() < n) return err("eof reading bytes", pos_);
    std::string s(reinterpret_cast<const char*>(data_ + pos_), n);
    pos_ += n;
    return s;
  }

  // Return a raw pointer to `n` bytes at the current position and advance,
  // without copying. Used to snapshot a sub-range for a nested parse.
  Result<const uint8_t*> raw(uint64_t n) {
    if (remaining() < n) return err("eof reading raw", pos_);
    const uint8_t* p = data_ + pos_;
    pos_ += n;
    return p;
  }

  // --- Payload-read accounting (tests) ---------------------------------------
  // Structural parsing must never call this. The weight inspector does, exactly
  // once per tensor it decodes. Tests assert count()==0 after a parse.
  static uint64_t& payload_read_counter() {
    static thread_local uint64_t counter = 0;
    return counter;
  }
  static void mark_payload_read() { ++payload_read_counter(); }

 private:
  const uint8_t* data_;
  uint64_t size_;
  uint64_t pos_ = 0;

  static uint16_t le16(uint16_t v) {
#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    return __builtin_bswap16(v);
#else
    return v;
#endif
  }
  static uint32_t le32(uint32_t v) {
#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    return __builtin_bswap32(v);
#else
    return v;
#endif
  }
  static uint64_t le64(uint64_t v) {
#if defined(__BIG_ENDIAN__) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    return __builtin_bswap64(v);
#else
    return v;
#endif
  }
};

}  // namespace netvis
