// core/StringArena.h — interned strings with 32-bit handles.
//
// DECISION (spec §3.2): all IR strings live in one arena and are referenced by
// StringId (a 32-bit index), not std::string. Model files have massive string
// redundancy — the same op_type ("Conv"), the same shape names, repeated across
// millions of nodes. Interning collapses those to one copy and shrinks every
// Node/ValueInfo to fixed-size POD, which keeps the IR cache-friendly and cheap
// to hash for the layout cache. StringId(0) is the canonical empty string.
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <unordered_map>

namespace netvis {

// Opaque handle into a StringArena. Trivially copyable, comparable.
struct StringId {
  uint32_t id = 0;  // 0 == empty string
  bool operator==(const StringId& o) const { return id == o.id; }
  bool operator!=(const StringId& o) const { return id != o.id; }
  bool valid() const { return id != 0; }
};

class StringArena {
 public:
  StringArena() {
    // Reserve id 0 for the empty string so a default-constructed StringId is
    // always dereferenceable.
    storage_.emplace_back();
    map_.emplace(std::string_view{}, StringId{0});
  }

  // Intern `s`, returning a stable handle. Same bytes -> same id.
  StringId intern(std::string_view s) {
    if (s.empty()) return StringId{0};
    auto it = map_.find(s);
    if (it != map_.end()) return it->second;
    StringId id{static_cast<uint32_t>(storage_.size())};
    storage_.emplace_back(s);
    // Key the map on the stored copy's view so it stays valid.
    map_.emplace(std::string_view{storage_.back()}, id);
    return id;
  }

  // Resolve a handle back to its bytes. Out-of-range -> empty.
  std::string_view get(StringId id) const {
    if (id.id >= storage_.size()) return {};
    return storage_[id.id];
  }

  size_t count() const { return storage_.size(); }

  // Thread-safety: NOT internally synchronized. Each parser owns its arena on a
  // worker thread; the arena is published to the main thread only after the
  // ParseJob completes (ownership transfer, no shared mutation).
 private:
  // std::deque (not vector): element addresses are stable across growth, so the
  // string_view keys in map_ (which point into stored strings) never dangle.
  std::deque<std::string> storage_;
  std::unordered_map<std::string_view, StringId> map_;
};

}  // namespace netvis
