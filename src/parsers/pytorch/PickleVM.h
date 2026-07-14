// parsers/pytorch/PickleVM.h — restricted (non-executing) pickle virtual machine.
//
// SECURITY (spec, non-negotiable): pickle is a stack VM whose GLOBAL/REDUCE
// opcodes normally *import and call arbitrary Python*. We never execute
// anything. Only an explicit ALLOWLIST of torch/collections symbols is
// interpreted (rebuild_tensor, OrderedDict, Size, the *Storage types); every
// other global/reduce target degrades to an inert Opaque(module,name) value
// that is recorded but never invoked. This lets us reconstruct a state_dict's
// structure and tensor metadata without running untrusted code.
//
// The VM reads ONLY through a bounds-checked cursor; a malformed stream yields
// a Result error with a byte offset, never an out-of-bounds read or crash. It
// records tensor payload location (file_offset + byte_len) but NEVER reads the
// payload bytes.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/Result.h"
#include "ir/IR.h"

namespace netvis::pytorch {

// A pickle value. Reference-counted so tuples/lists/dicts can share subvalues
// cheaply and so BINGET/memo can alias without copying large containers.
struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
  enum class Kind : uint8_t {
    None,
    Bool,
    Int,
    Double,
    Str,
    Bytes,
    Tuple,
    List,
    Dict,        // ordered list of (key,value) pairs
    Global,      // allowlisted symbol (module,name) that we interpret
    Opaque,      // NON-allowlisted symbol — recorded, never executed
    Persistent,  // BINPERSID payload (the pid value, usually a tuple)
    Tensor,      // a reconstructed ir::TensorRef
  };

  Kind kind = Kind::None;

  bool b = false;
  int64_t i = 0;
  double d = 0.0;
  std::string s;          // Str text or Bytes bytes (Kind::Str / Kind::Bytes)
  std::string module;     // Global/Opaque module
  std::string name;       // Global/Opaque qualified name
  std::vector<ValuePtr> items;                       // Tuple / List
  std::vector<std::pair<ValuePtr, ValuePtr>> pairs;  // Dict (ordered)
  ValuePtr inner;         // Persistent -> the pid value
  ir::TensorRef tensor;   // Kind::Tensor

  static ValuePtr make_none() { auto v = std::make_shared<Value>(); v->kind = Kind::None; return v; }
};

// Resolves a storage key ("0", "1", ...) to the absolute payload offset+length
// in the mmap. Provided by the zip/legacy driver. Returns false if unresolvable
// (legacy files without a resolvable payload map): the VM then emits a
// TensorRef with file_offset == UINT64_MAX.
struct StorageResolver {
  // key -> (payload_file_offset, payload_byte_len). byte_len is the full
  // storage byte length (numel * elem_size) as stored in the archive.
  std::function<bool(const std::string& key, uint64_t& out_offset,
                     uint64_t& out_len)>
      resolve;
};

class PickleVM {
 public:
  // `data`/`size` bound the pickle stream. `resolver` maps storage keys to
  // payload locations. The VM does not own the mmap; the caller guarantees it
  // outlives the run and any TensorRefs produced.
  PickleVM(const uint8_t* data, uint64_t size, const StorageResolver& resolver)
      : data_(data), size_(size), resolver_(resolver) {}

  // Execute until STOP. Returns the top-of-stack value (the unpickled object).
  Result<ValuePtr> run();

 private:
  const uint8_t* data_;
  uint64_t size_;
  const StorageResolver& resolver_;

  uint64_t pos_ = 0;
  std::vector<ValuePtr> stack_;
  std::vector<size_t> marks_;                 // MARK positions into stack_
  std::map<int64_t, ValuePtr> memo_;
  int64_t memo_seq_ = 0;                      // next MEMOIZE index

  // Safety cap: pickle streams are structural (tiny vs. payload), so a huge
  // opcode count means a malformed/adversarial file — bail instead of looping.
  uint64_t op_count_ = 0;
  static constexpr uint64_t kMaxOps = 200'000'000ULL;

  // --- bounds-checked stream readers (byte offsets on error) ---
  Result<uint8_t> rd_u8();
  Result<uint16_t> rd_u16();
  Result<uint32_t> rd_u32();
  Result<uint64_t> rd_u64();
  Result<double> rd_f64be();
  Result<std::string> rd_bytes(uint64_t n);
  Result<std::string> rd_line();  // until '\n' (for GLOBAL, text ints)
  Result<int64_t> rd_long(uint64_t n);  // little-endian signed, n bytes

  // --- stack helpers ---
  Error underflow() const { return err("pickle stack underflow", pos_); }
  Result<ValuePtr> pop();
  void push(ValuePtr v) { stack_.push_back(std::move(v)); }
  Result<std::vector<ValuePtr>> pop_to_mark();

  // Opcode handlers that build structure.
  Result<ValuePtr> do_reduce(const ValuePtr& callable,
                             std::vector<ValuePtr> args);
  Result<ValuePtr> do_newobj(const ValuePtr& cls, const ValuePtr& args);
  Result<bool> do_build(const ValuePtr& obj, const ValuePtr& state);
  Result<ValuePtr> resolve_global(const std::string& module,
                                  const std::string& name);
  Result<ValuePtr> do_persid(const ValuePtr& pid);
};

// Maps a torch storage global name (e.g. "FloatStorage") to an ir::DType and
// element byte size. Returns false for unknown storage types.
bool torch_storage_dtype(const std::string& storage_name, ir::DType& out_dtype,
                         uint32_t& out_elem_size);

}  // namespace netvis::pytorch
