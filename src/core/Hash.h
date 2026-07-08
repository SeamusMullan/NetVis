// core/Hash.h — FNV-1a hashing for structure fingerprints and cache keys.
//
// DECISION (spec §2.7, §7.1): the layout cache is keyed by a hash of the graph
// *structure* (never weights), and collapse-group detection fingerprints nodes.
// FNV-1a is chosen for being tiny, dependency-free, order-sensitive, and fast
// enough that hashing an entire graph's structure is negligible next to layout.
// Not cryptographic — collision risk is acceptable for a local cache key, and a
// stale/mismatched cache simply triggers a recompute.
#pragma once

#include <cstdint>
#include <string_view>

namespace netvis {

// Streaming 64-bit FNV-1a. Combine values in order to build a structural digest.
class Hasher {
 public:
  Hasher() = default;

  Hasher& bytes(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) {
      state_ ^= b[i];
      state_ *= kPrime;
    }
    return *this;
  }

  Hasher& u64(uint64_t v) { return bytes(&v, sizeof(v)); }
  Hasher& u32(uint32_t v) { return bytes(&v, sizeof(v)); }
  Hasher& i64(int64_t v) { return bytes(&v, sizeof(v)); }
  Hasher& str(std::string_view s) { return bytes(s.data(), s.size()); }

  uint64_t value() const { return state_; }

 private:
  static constexpr uint64_t kOffset = 1469598103934665603ULL;
  static constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t state_ = kOffset;
};

inline uint64_t fnv1a(std::string_view s) {
  return Hasher().str(s).value();
}

}  // namespace netvis
