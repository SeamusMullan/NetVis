// core/Result.h — Result<T, Error>-style error handling.
//
// DECISION: No exceptions cross module boundaries (spec §13). Parsers and jobs
// return Result<T>; malformed input is a value, not a throw. This keeps error
// paths explicit and lets us carry a byte offset for diagnostics (spec §6).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace netvis {

// A parse/engine error. `offset` is the byte position in the source file when
// meaningful (UINT64_MAX when not applicable), so error toasts can point at the
// exact spot a malformed file went wrong.
struct Error {
  std::string message;
  uint64_t offset = UINT64_MAX;

  Error() = default;
  explicit Error(std::string msg, uint64_t off = UINT64_MAX)
      : message(std::move(msg)), offset(off) {}
};

// Result<T>: either a value or an Error. Move-only-friendly, no exceptions.
// Usage:
//   Result<ir::Model> r = parse(...);
//   if (!r) return r.error();
//   use(*r);
template <typename T>
class Result {
 public:
  Result(T value) : data_(std::move(value)) {}          // NOLINT: implicit ok
  Result(Error error) : data_(std::move(error)) {}      // NOLINT: implicit err

  bool ok() const { return std::holds_alternative<T>(data_); }
  explicit operator bool() const { return ok(); }

  T& value() { return std::get<T>(data_); }
  const T& value() const { return std::get<T>(data_); }
  T& operator*() { return std::get<T>(data_); }
  const T& operator*() const { return std::get<T>(data_); }
  T* operator->() { return &std::get<T>(data_); }
  const T* operator->() const { return &std::get<T>(data_); }

  const Error& error() const { return std::get<Error>(data_); }

  // Consume the value (moves it out). Caller must have checked ok().
  T take() { return std::move(std::get<T>(data_)); }

 private:
  std::variant<T, Error> data_;
};

// Convenience: build an error Result at a byte offset.
inline Error err(std::string msg, uint64_t offset = UINT64_MAX) {
  return Error(std::move(msg), offset);
}

}  // namespace netvis
