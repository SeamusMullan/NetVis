// parsers/pytorch/PickleVM.cpp — restricted pickle VM implementation.
//
// See PickleVM.h for the security rationale. This file interprets pickle
// protocols 2-5 opcodes needed to reconstruct a torch state_dict. It NEVER
// executes user code and NEVER reads tensor payload bytes — only records
// offset+length via the StorageResolver.
#include "parsers/pytorch/PickleVM.h"

#include <cstring>

namespace netvis::pytorch {

// ---- storage dtype table -----------------------------------------------------
bool torch_storage_dtype(const std::string& storage_name, ir::DType& out_dtype,
                         uint32_t& out_elem_size) {
  struct Ent { const char* name; ir::DType dt; uint32_t sz; };
  static const Ent kTable[] = {
      {"FloatStorage", ir::DType::F32, 4},
      {"HalfStorage", ir::DType::F16, 2},
      {"BFloat16Storage", ir::DType::BF16, 2},
      {"DoubleStorage", ir::DType::F64, 8},
      {"LongStorage", ir::DType::I64, 8},
      {"IntStorage", ir::DType::I32, 4},
      {"ShortStorage", ir::DType::I16, 2},
      {"CharStorage", ir::DType::I8, 1},
      {"ByteStorage", ir::DType::U8, 1},
      {"BoolStorage", ir::DType::Bool, 1},
  };
  for (const auto& e : kTable) {
    if (storage_name == e.name) {
      out_dtype = e.dt;
      out_elem_size = e.sz;
      return true;
    }
  }
  return false;
}

// ---- stream readers ----------------------------------------------------------
Result<uint8_t> PickleVM::rd_u8() {
  if (pos_ + 1 > size_) return err("eof reading u8", pos_);
  return data_[pos_++];
}
Result<uint16_t> PickleVM::rd_u16() {
  if (pos_ + 2 > size_) return err("eof reading u16", pos_);
  uint16_t v;
  std::memcpy(&v, data_ + pos_, 2);
  pos_ += 2;
  return v;  // torch runs on LE hosts; pickle little-endian
}
Result<uint32_t> PickleVM::rd_u32() {
  if (pos_ + 4 > size_) return err("eof reading u32", pos_);
  uint32_t v;
  std::memcpy(&v, data_ + pos_, 4);
  pos_ += 4;
  return v;
}
Result<uint64_t> PickleVM::rd_u64() {
  if (pos_ + 8 > size_) return err("eof reading u64", pos_);
  uint64_t v;
  std::memcpy(&v, data_ + pos_, 8);
  pos_ += 8;
  return v;
}
Result<double> PickleVM::rd_f64be() {
  if (pos_ + 8 > size_) return err("eof reading f64", pos_);
  // BINFLOAT stores a big-endian IEEE-754 double.
  uint8_t tmp[8];
  for (int i = 0; i < 8; ++i) tmp[i] = data_[pos_ + 7 - i];
  double d;
  std::memcpy(&d, tmp, 8);
  pos_ += 8;
  return d;
}
Result<std::string> PickleVM::rd_bytes(uint64_t n) {
  if (pos_ + n > size_ || pos_ + n < pos_) return err("eof reading bytes", pos_);
  std::string s(reinterpret_cast<const char*>(data_ + pos_), n);
  pos_ += n;
  return s;
}
Result<std::string> PickleVM::rd_line() {
  uint64_t start = pos_;
  while (pos_ < size_ && data_[pos_] != '\n') ++pos_;
  if (pos_ >= size_) return err("eof reading line", start);
  std::string s(reinterpret_cast<const char*>(data_ + start), pos_ - start);
  ++pos_;  // consume newline
  return s;
}
Result<int64_t> PickleVM::rd_long(uint64_t n) {
  if (n == 0) return int64_t{0};
  if (n > 8) return err("LONG too wide (unsupported >64-bit)", pos_);
  if (pos_ + n > size_) return err("eof reading long", pos_);
  // Little-endian two's-complement, sign-extended to 64 bits.
  uint64_t v = 0;
  for (uint64_t i = 0; i < n; ++i) v |= static_cast<uint64_t>(data_[pos_ + i]) << (8 * i);
  pos_ += n;
  if (data_[pos_ - 1] & 0x80) {  // negative -> sign-extend
    if (n < 8) v |= ~uint64_t{0} << (8 * n);
  }
  return static_cast<int64_t>(v);
}

// ---- stack helpers -----------------------------------------------------------
Result<ValuePtr> PickleVM::pop() {
  if (stack_.empty()) return underflow();
  ValuePtr v = std::move(stack_.back());
  stack_.pop_back();
  return v;
}
Result<std::vector<ValuePtr>> PickleVM::pop_to_mark() {
  if (marks_.empty()) return err("no MARK on stack", pos_);
  size_t m = marks_.back();
  marks_.pop_back();
  if (m > stack_.size()) return err("corrupt MARK", pos_);
  std::vector<ValuePtr> out(stack_.begin() + static_cast<std::ptrdiff_t>(m),
                            stack_.end());
  stack_.resize(m);
  return out;
}

// ---- global resolution (ALLOWLIST) ------------------------------------------
Result<ValuePtr> PickleVM::resolve_global(const std::string& module,
                                          const std::string& name) {
  auto v = std::make_shared<Value>();
  // Allowlisted torch/collections symbols become interpretable Globals; the
  // REDUCE/BUILD handlers know how to act on them. Everything else is inert
  // Opaque and is NEVER executed.
  ir::DType dt;
  uint32_t esz;
  bool allow = false;
  if (module == "torch._utils" &&
      (name == "_rebuild_tensor_v2" || name == "_rebuild_tensor" ||
       name == "_rebuild_parameter")) {
    allow = true;
  } else if (module == "collections" && name == "OrderedDict") {
    allow = true;
  } else if (module == "torch" && name == "Size") {
    allow = true;
  } else if ((module == "torch" || module == "torch.storage") &&
             torch_storage_dtype(name, dt, esz)) {
    allow = true;
  }
  if (allow) {
    v->kind = Value::Kind::Global;
    v->module = module;
    v->name = name;
  } else {
    // Recorded but inert. Non-negotiable: no import, no call.
    v->kind = Value::Kind::Opaque;
    v->module = module;
    v->name = name;
  }
  return v;
}

// ---- REDUCE: apply an allowlisted callable to args --------------------------
Result<ValuePtr> PickleVM::do_reduce(const ValuePtr& callable,
                                     std::vector<ValuePtr> args) {
  if (!callable) return err("REDUCE on null callable", pos_);

  // Non-allowlisted callable: return an Opaque result, never execute.
  if (callable->kind != Value::Kind::Global) {
    auto v = std::make_shared<Value>();
    v->kind = Value::Kind::Opaque;
    v->module = callable->module;
    v->name = callable->name;
    return v;
  }

  const std::string& mod = callable->module;
  const std::string& nm = callable->name;

  if (mod == "collections" && nm == "OrderedDict") {
    // OrderedDict(...) -> empty ordered dict; state applied later via BUILD.
    auto v = std::make_shared<Value>();
    v->kind = Value::Kind::Dict;
    return v;
  }

  if (mod == "torch" && nm == "Size") {
    // torch.Size(iterable) -> tuple of ints.
    auto v = std::make_shared<Value>();
    v->kind = Value::Kind::Tuple;
    if (!args.empty() && args[0] &&
        (args[0]->kind == Value::Kind::Tuple ||
         args[0]->kind == Value::Kind::List)) {
      v->items = args[0]->items;
    }
    return v;
  }

  if (mod == "torch._utils" &&
      (nm == "_rebuild_tensor_v2" || nm == "_rebuild_tensor")) {
    // args: (storage, storage_offset, size, stride, [requires_grad,
    //        backward_hooks], ...). storage is a Persistent value whose pid is
    //        ("storage", <StorageType global>, key, device, numel).
    auto tref = std::make_shared<Value>();
    tref->kind = Value::Kind::Tensor;
    ir::TensorRef& t = tref->tensor;

    if (args.empty()) return tref;
    ValuePtr storage = args[0];
    ir::DType dtype = ir::DType::Unknown;
    uint32_t elem_size = 0;
    uint64_t storage_offset_bytes = 0;  // storage payload absolute offset
    uint64_t storage_len = 0;
    bool have_storage = false;

    if (storage && storage->kind == Value::Kind::Persistent && storage->inner &&
        storage->inner->kind == Value::Kind::Tuple) {
      const auto& pid = storage->inner->items;
      // pid[1] = storage type global, pid[2] = key, pid[4] = numel
      if (pid.size() >= 3 && pid[1] &&
          pid[1]->kind == Value::Kind::Global) {
        torch_storage_dtype(pid[1]->name, dtype, elem_size);
      }
      if (pid.size() >= 3 && pid[2] && pid[2]->kind == Value::Kind::Str) {
        uint64_t off = 0, len = 0;
        if (resolver_.resolve && resolver_.resolve(pid[2]->s, off, len)) {
          storage_offset_bytes = off;
          storage_len = len;
          have_storage = true;
        }
      }
    }
    t.dtype = dtype;

    // storage_offset (in elements) -> byte offset within the storage payload.
    int64_t elem_offset = 0;
    if (args.size() >= 2 && args[1] && args[1]->kind == Value::Kind::Int)
      elem_offset = args[1]->i;

    // size tuple -> shape.
    if (args.size() >= 3 && args[2] &&
        (args[2]->kind == Value::Kind::Tuple ||
         args[2]->kind == Value::Kind::List)) {
      for (const auto& d : args[2]->items) {
        if (d && d->kind == Value::Kind::Int) t.shape.push_back(d->i);
      }
    }

    int64_t n = t.elem_count();
    uint64_t byte_len = (elem_size > 0 && n > 0)
                            ? static_cast<uint64_t>(n) * elem_size
                            : 0;
    if (have_storage) {
      uint64_t off_bytes = storage_offset_bytes +
                           static_cast<uint64_t>(elem_offset < 0 ? 0 : elem_offset) *
                               elem_size;
      t.file_offset = off_bytes;
      // Prefer the exact tensor extent; fall back to full storage length.
      t.byte_len = byte_len ? byte_len : storage_len;
    } else {
      // Unresolvable storage (legacy). Emit shape/dtype; mark offset absent.
      t.file_offset = UINT64_MAX;
      t.byte_len = byte_len;
    }
    return tref;
  }

  if (mod == "torch._utils" && nm == "_rebuild_parameter") {
    // _rebuild_parameter(tensor, requires_grad, backward_hooks) -> the tensor.
    if (!args.empty() && args[0]) return args[0];
    return Value::make_none();
  }

  // Allowlisted but unhandled -> inert Opaque.
  auto v = std::make_shared<Value>();
  v->kind = Value::Kind::Opaque;
  v->module = mod;
  v->name = nm;
  return v;
}

Result<ValuePtr> PickleVM::do_newobj(const ValuePtr& cls, const ValuePtr& args) {
  // NEWOBJ(cls, argtuple). For our allowlist, cls of interest is OrderedDict.
  if (cls && cls->kind == Value::Kind::Global && cls->module == "collections" &&
      cls->name == "OrderedDict") {
    auto v = std::make_shared<Value>();
    v->kind = Value::Kind::Dict;
    return v;
  }
  if (cls && cls->kind == Value::Kind::Global && cls->module == "torch" &&
      cls->name == "Size") {
    auto v = std::make_shared<Value>();
    v->kind = Value::Kind::Tuple;
    if (args && (args->kind == Value::Kind::Tuple ||
                 args->kind == Value::Kind::List)) {
      v->items = args->items;
    }
    return v;
  }
  // Anything else -> inert Opaque object.
  auto v = std::make_shared<Value>();
  v->kind = Value::Kind::Opaque;
  if (cls) {
    v->module = cls->module;
    v->name = cls->name;
  }
  return v;
}

Result<bool> PickleVM::do_build(const ValuePtr& obj, const ValuePtr& state) {
  if (!obj || !state) return true;
  // For a dict-like object, BUILD applies a state dict of members. torch's
  // OrderedDict __setstate__ / __reduce__ typically carries items as a dict.
  if (obj->kind == Value::Kind::Dict && state->kind == Value::Kind::Dict) {
    for (auto& kv : state->pairs) obj->pairs.push_back(kv);
  }
  // Other objects: state ignored (inert). Never executes __setstate__.
  return true;
}

// ---- BINPERSID ---------------------------------------------------------------
Result<ValuePtr> PickleVM::do_persid(const ValuePtr& pid) {
  auto v = std::make_shared<Value>();
  v->kind = Value::Kind::Persistent;
  v->inner = pid;
  return v;
}

// ---- main dispatch -----------------------------------------------------------
Result<ValuePtr> PickleVM::run() {
  for (;;) {
    if (++op_count_ > kMaxOps) return err("pickle opcode cap exceeded", pos_);
    auto op_r = rd_u8();
    if (!op_r) return op_r.error();
    uint8_t op = *op_r;

    switch (op) {
      case 0x80: {  // PROTO
        auto r = rd_u8();
        if (!r) return r.error();
        break;
      }
      case 0x95: {  // FRAME (8-byte length; we stream, so just skip the field)
        auto r = rd_u64();
        if (!r) return r.error();
        break;
      }
      case '.': {  // STOP
        if (stack_.empty()) return err("STOP with empty stack", pos_);
        return stack_.back();
      }
      case '(': {  // MARK
        marks_.push_back(stack_.size());
        break;
      }
      case '0': {  // POP
        auto r = pop();
        if (!r) return r.error();
        break;
      }
      case '1': {  // POP_MARK
        auto r = pop_to_mark();
        if (!r) return r.error();
        break;
      }
      case '2': {  // DUP
        if (stack_.empty()) return underflow();
        stack_.push_back(stack_.back());
        break;
      }
      case 0x94: {  // MEMOIZE
        if (stack_.empty()) return underflow();
        memo_[memo_seq_++] = stack_.back();
        break;
      }
      case 'q': {  // BINPUT
        auto r = rd_u8();
        if (!r) return r.error();
        if (stack_.empty()) return underflow();
        memo_[*r] = stack_.back();
        break;
      }
      case 'r': {  // LONG_BINPUT
        auto r = rd_u32();
        if (!r) return r.error();
        if (stack_.empty()) return underflow();
        memo_[*r] = stack_.back();
        break;
      }
      case 'h': {  // BINGET
        auto r = rd_u8();
        if (!r) return r.error();
        auto it = memo_.find(*r);
        if (it == memo_.end()) return err("BINGET missing memo", pos_);
        push(it->second);
        break;
      }
      case 'j': {  // LONG_BINGET
        auto r = rd_u32();
        if (!r) return r.error();
        auto it = memo_.find(*r);
        if (it == memo_.end()) return err("LONG_BINGET missing memo", pos_);
        push(it->second);
        break;
      }
      case 'J': {  // BININT (signed 4-byte)
        auto r = rd_u32();
        if (!r) return r.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Int;
        v->i = static_cast<int32_t>(*r);
        push(v);
        break;
      }
      case 'K': {  // BININT1 (unsigned 1-byte)
        auto r = rd_u8();
        if (!r) return r.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Int;
        v->i = *r;
        push(v);
        break;
      }
      case 'M': {  // BININT2 (unsigned 2-byte)
        auto r = rd_u16();
        if (!r) return r.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Int;
        v->i = *r;
        push(v);
        break;
      }
      case 0x8a: {  // LONG1
        auto n = rd_u8();
        if (!n) return n.error();
        auto r = rd_long(*n);
        if (!r) return r.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Int;
        v->i = *r;
        push(v);
        break;
      }
      case 0x8b: {  // LONG4
        auto n = rd_u32();
        if (!n) return n.error();
        auto r = rd_long(*n);
        if (!r) return r.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Int;
        v->i = *r;
        push(v);
        break;
      }
      case 0x88: {  // NEWTRUE
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Bool;
        v->b = true;
        push(v);
        break;
      }
      case 0x89: {  // NEWFALSE
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Bool;
        v->b = false;
        push(v);
        break;
      }
      case 'N': {  // NONE
        push(Value::make_none());
        break;
      }
      case 'X': {  // BINUNICODE (LE u32 len)
        auto n = rd_u32();
        if (!n) return n.error();
        auto s = rd_bytes(*n);
        if (!s) return s.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Str;
        v->s = std::move(*s);
        push(v);
        break;
      }
      case 0x8c: {  // SHORT_BINUNICODE (u8 len)
        auto n = rd_u8();
        if (!n) return n.error();
        auto s = rd_bytes(*n);
        if (!s) return s.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Str;
        v->s = std::move(*s);
        push(v);
        break;
      }
      case 0x8d: {  // BINUNICODE8 (u64 len)
        auto n = rd_u64();
        if (!n) return n.error();
        auto s = rd_bytes(*n);
        if (!s) return s.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Str;
        v->s = std::move(*s);
        push(v);
        break;
      }
      case 'B': {  // BINBYTES (LE u32 len)
        auto n = rd_u32();
        if (!n) return n.error();
        auto s = rd_bytes(*n);
        if (!s) return s.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Bytes;
        v->s = std::move(*s);
        push(v);
        break;
      }
      case 'C': {  // SHORT_BINBYTES (u8 len)
        auto n = rd_u8();
        if (!n) return n.error();
        auto s = rd_bytes(*n);
        if (!s) return s.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Bytes;
        v->s = std::move(*s);
        push(v);
        break;
      }
      case 0x8e: {  // BINBYTES8 (u64 len)
        auto n = rd_u64();
        if (!n) return n.error();
        auto s = rd_bytes(*n);
        if (!s) return s.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Bytes;
        v->s = std::move(*s);
        push(v);
        break;
      }
      case '}': {  // EMPTY_DICT
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Dict;
        push(v);
        break;
      }
      case ']': {  // EMPTY_LIST
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::List;
        push(v);
        break;
      }
      case ')': {  // EMPTY_TUPLE
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Tuple;
        push(v);
        break;
      }
      case 't': {  // TUPLE
        auto items = pop_to_mark();
        if (!items) return items.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Tuple;
        v->items = std::move(*items);
        push(v);
        break;
      }
      case 0x85: {  // TUPLE1
        auto a = pop();
        if (!a) return a.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Tuple;
        v->items = {*a};
        push(v);
        break;
      }
      case 0x86: {  // TUPLE2
        auto b = pop();
        if (!b) return b.error();
        auto a = pop();
        if (!a) return a.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Tuple;
        v->items = {*a, *b};
        push(v);
        break;
      }
      case 0x87: {  // TUPLE3
        auto c = pop();
        if (!c) return c.error();
        auto b = pop();
        if (!b) return b.error();
        auto a = pop();
        if (!a) return a.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Tuple;
        v->items = {*a, *b, *c};
        push(v);
        break;
      }
      case 'a': {  // APPEND
        auto x = pop();
        if (!x) return x.error();
        if (stack_.empty()) return underflow();
        ValuePtr lst = stack_.back();
        if (lst && lst->kind == Value::Kind::List) lst->items.push_back(*x);
        break;
      }
      case 'e': {  // APPENDS
        auto items = pop_to_mark();
        if (!items) return items.error();
        if (stack_.empty()) return underflow();
        ValuePtr lst = stack_.back();
        if (lst && lst->kind == Value::Kind::List)
          for (auto& it : *items) lst->items.push_back(it);
        break;
      }
      case 's': {  // SETITEM
        auto val = pop();
        if (!val) return val.error();
        auto key = pop();
        if (!key) return key.error();
        if (stack_.empty()) return underflow();
        ValuePtr d = stack_.back();
        if (d && d->kind == Value::Kind::Dict) d->pairs.emplace_back(*key, *val);
        break;
      }
      case 'u': {  // SETITEMS
        auto items = pop_to_mark();
        if (!items) return items.error();
        if (items->size() % 2 != 0) return err("SETITEMS odd count", pos_);
        if (stack_.empty()) return underflow();
        ValuePtr d = stack_.back();
        if (d && d->kind == Value::Kind::Dict)
          for (size_t k = 0; k + 1 < items->size(); k += 2)
            d->pairs.emplace_back((*items)[k], (*items)[k + 1]);
        break;
      }
      case 'G': {  // BINFLOAT (big-endian double)
        auto r = rd_f64be();
        if (!r) return r.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Double;
        v->d = *r;
        push(v);
        break;
      }
      case 'R': {  // REDUCE
        auto args = pop();
        if (!args) return args.error();
        auto callable = pop();
        if (!callable) return callable.error();
        std::vector<ValuePtr> arg_vec;
        if (*args && (*args)->kind == Value::Kind::Tuple)
          arg_vec = (*args)->items;
        auto res = do_reduce(*callable, std::move(arg_vec));
        if (!res) return res.error();
        push(*res);
        break;
      }
      case 'b': {  // BUILD
        auto state = pop();
        if (!state) return state.error();
        if (stack_.empty()) return underflow();
        ValuePtr obj = stack_.back();
        auto r = do_build(obj, *state);
        if (!r) return r.error();
        break;
      }
      case 0x81: {  // NEWOBJ
        auto args = pop();
        if (!args) return args.error();
        auto cls = pop();
        if (!cls) return cls.error();
        auto res = do_newobj(*cls, *args);
        if (!res) return res.error();
        push(*res);
        break;
      }
      case 'c': {  // GLOBAL (module\nname\n)
        auto mod = rd_line();
        if (!mod) return mod.error();
        auto nm = rd_line();
        if (!nm) return nm.error();
        auto res = resolve_global(*mod, *nm);
        if (!res) return res.error();
        push(*res);
        break;
      }
      case 0x93: {  // STACK_GLOBAL
        auto nm = pop();
        if (!nm) return nm.error();
        auto mod = pop();
        if (!mod) return mod.error();
        std::string ms = (*mod && (*mod)->kind == Value::Kind::Str) ? (*mod)->s : "";
        std::string ns = (*nm && (*nm)->kind == Value::Kind::Str) ? (*nm)->s : "";
        auto res = resolve_global(ms, ns);
        if (!res) return res.error();
        push(*res);
        break;
      }
      case 'Q': {  // BINPERSID
        auto pid = pop();
        if (!pid) return pid.error();
        auto res = do_persid(*pid);
        if (!res) return res.error();
        push(*res);
        break;
      }
      case 'P': {  // PERSID (text pid, newline-terminated)
        auto line = rd_line();
        if (!line) return line.error();
        auto pidv = std::make_shared<Value>();
        pidv->kind = Value::Kind::Str;
        pidv->s = *line;
        auto res = do_persid(pidv);
        if (!res) return res.error();
        push(*res);
        break;
      }
      case 'l': {  // LIST (from mark)
        auto items = pop_to_mark();
        if (!items) return items.error();
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::List;
        v->items = std::move(*items);
        push(v);
        break;
      }
      case 'd': {  // DICT (from mark)
        auto items = pop_to_mark();
        if (!items) return items.error();
        if (items->size() % 2 != 0) return err("DICT odd count", pos_);
        auto v = std::make_shared<Value>();
        v->kind = Value::Kind::Dict;
        for (size_t k = 0; k + 1 < items->size(); k += 2)
          v->pairs.emplace_back((*items)[k], (*items)[k + 1]);
        push(v);
        break;
      }
      default:
        return err("unsupported pickle opcode " + std::to_string(op), pos_ - 1);
    }
  }
}

}  // namespace netvis::pytorch
