// core/SmallVec.h — small-buffer-optimized vector.
//
// DECISION: Tensor shapes are almost always <=6 dims and ValueInfo/TensorRef are
// hot, allocated in the millions for large models. A heap allocation per shape
// would dominate parse time and thrash caches. SmallVec<T,N> stores up to N
// elements inline and only spills to the heap beyond that, so the common case is
// allocation-free (spec §3.2 uses SmallVec<int64_t,6> for shapes).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <utility>

namespace netvis {

template <typename T, size_t N>
class SmallVec {
 public:
  SmallVec() = default;

  SmallVec(std::initializer_list<T> init) {
    reserve(init.size());
    for (const T& v : init) push_back(v);
  }

  SmallVec(const SmallVec& other) {
    reserve(other.size_);
    for (size_t i = 0; i < other.size_; ++i) push_back(other[i]);
  }

  SmallVec(SmallVec&& other) noexcept { move_from(std::move(other)); }

  SmallVec& operator=(const SmallVec& other) {
    if (this != &other) {
      clear();
      reserve(other.size_);
      for (size_t i = 0; i < other.size_; ++i) push_back(other[i]);
    }
    return *this;
  }

  SmallVec& operator=(SmallVec&& other) noexcept {
    if (this != &other) {
      destroy();
      move_from(std::move(other));
    }
    return *this;
  }

  ~SmallVec() { destroy(); }

  void push_back(const T& v) {
    if (size_ == cap_) grow(cap_ ? cap_ * 2 : N);
    new (data_ + size_) T(v);
    ++size_;
  }
  void push_back(T&& v) {
    if (size_ == cap_) grow(cap_ ? cap_ * 2 : N);
    new (data_ + size_) T(std::move(v));
    ++size_;
  }

  void reserve(size_t n) {
    if (n > cap_) grow(n);
  }

  void resize(size_t n, const T& fill = T{}) {
    if (n > cap_) grow(n);
    while (size_ < n) push_back(fill);
    while (size_ > n) { data_[--size_].~T(); }
  }

  void clear() {
    for (size_t i = 0; i < size_; ++i) data_[i].~T();
    size_ = 0;
  }

  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }

  T* begin() { return data_; }
  T* end() { return data_ + size_; }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  T* data() { return data_; }
  const T* data() const { return data_; }

  T& back() { return data_[size_ - 1]; }
  const T& back() const { return data_[size_ - 1]; }

  bool operator==(const SmallVec& o) const {
    if (size_ != o.size_) return false;
    for (size_t i = 0; i < size_; ++i)
      if (!(data_[i] == o.data_[i])) return false;
    return true;
  }

 private:
  alignas(T) unsigned char inline_[sizeof(T) * N];
  T* data_ = reinterpret_cast<T*>(inline_);
  size_t size_ = 0;
  size_t cap_ = N;

  bool on_heap() const { return data_ != reinterpret_cast<const T*>(inline_); }

  void grow(size_t new_cap) {
    new_cap = std::max(new_cap, N + 1);
    T* nd = static_cast<T*>(::operator new(sizeof(T) * new_cap));
    for (size_t i = 0; i < size_; ++i) {
      new (nd + i) T(std::move(data_[i]));
      data_[i].~T();
    }
    if (on_heap()) ::operator delete(data_);
    data_ = nd;
    cap_ = new_cap;
  }

  void destroy() {
    for (size_t i = 0; i < size_; ++i) data_[i].~T();
    if (on_heap()) ::operator delete(data_);
    data_ = reinterpret_cast<T*>(inline_);
    size_ = 0;
    cap_ = N;
  }

  void move_from(SmallVec&& other) {
    if (other.on_heap()) {
      // Steal the heap buffer wholesale.
      data_ = other.data_;
      cap_ = other.cap_;
      size_ = other.size_;
      other.data_ = reinterpret_cast<T*>(other.inline_);
      other.size_ = 0;
      other.cap_ = N;
    } else {
      // Inline: element-wise move into our own inline buffer.
      data_ = reinterpret_cast<T*>(inline_);
      cap_ = N;
      size_ = 0;
      for (size_t i = 0; i < other.size_; ++i) push_back(std::move(other.data_[i]));
      other.clear();
    }
  }
};

}  // namespace netvis
