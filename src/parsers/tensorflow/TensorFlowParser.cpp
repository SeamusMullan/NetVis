// parsers/tensorflow/TensorFlowParser.cpp — TensorFlow GraphDef / SavedModel -> ir::Model.
//
// DECISION (#107, spec §2.1): TensorFlow files are protobuf, exactly like ONNX,
// so this reuses the hand-rolled bounds-checked onnx::WireReader (it is
// protobuf-generic in substance; CoreMlParser.cpp reuses it the same way) rather
// than pulling in libprotobuf. Two containers land here:
//
//   frozen .pb  -> a bare GraphDef      (field 1 = node, len-delimited)
//   saved_model.pb -> a SavedModel      (field 1 = schema_version, varint)
//                     -> meta_graphs[0].graph_def
//
// A TF2 SavedModel's top-level graph_def is mostly PartitionedCall stubs; the
// real bodies live in GraphDef.library.function[]. Each FunctionDef becomes its
// own ir::Graph and a calling node's `f` attribute links to it via Node::subgraph,
// so the UI can drill into the actual compute.
//
// ZERO PAYLOAD: a Const node's TensorProto.tensor_content is recorded as an
// absolute mmap offset+length and never read; every other *_val payload field is
// skipped without touching its bytes. HONEST-UNKNOWN: an unresolved dimension is
// -1, never guessed, and a SavedModel's checkpoint weights (variables/) are NOT
// decoded — the variables note in metadata says so rather than inventing offsets.
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/onnx/WireReader.h"

namespace netvis::tensorflow {
namespace {

using onnx::SubRange;
using onnx::WireReader;
using onnx::WireType;

// A node output slot beyond this is treated as a typo/hostile value: we still
// wire the edge, but we do not materialize the intervening slots.
constexpr int64_t kMaxOutputSlot = 1024;

// tensorflow/core/framework/types.proto — DataType. A "_REF" type is the base
// value + 100 (DT_FLOAT_REF == 101) and carries the same element type.
enum TfDType : int32_t {
  DT_INVALID = 0, DT_FLOAT = 1, DT_DOUBLE = 2, DT_INT32 = 3, DT_UINT8 = 4,
  DT_INT16 = 5, DT_INT8 = 6, DT_STRING = 7, DT_COMPLEX64 = 8, DT_INT64 = 9,
  DT_BOOL = 10, DT_QINT8 = 11, DT_QUINT8 = 12, DT_QINT32 = 13, DT_BFLOAT16 = 14,
  DT_QINT16 = 15, DT_QUINT16 = 16, DT_UINT16 = 17, DT_COMPLEX128 = 18,
  DT_HALF = 19, DT_RESOURCE = 20, DT_VARIANT = 21, DT_UINT32 = 22,
  DT_UINT64 = 23,
};

int64_t deref_dtype(int64_t t) { return t > 100 ? t - 100 : t; }

ir::DType map_dtype(int64_t t) {
  switch (deref_dtype(t)) {
    case DT_FLOAT:    return ir::DType::F32;
    case DT_DOUBLE:   return ir::DType::F64;
    case DT_INT32:    return ir::DType::I32;
    case DT_UINT8:    return ir::DType::U8;
    case DT_INT16:    return ir::DType::I16;
    case DT_INT8:     return ir::DType::I8;
    case DT_INT64:    return ir::DType::I64;
    case DT_BOOL:     return ir::DType::Bool;
    case DT_BFLOAT16: return ir::DType::BF16;
    case DT_HALF:     return ir::DType::F16;
    case DT_UINT16:   return ir::DType::U16;
    case DT_UINT32:   return ir::DType::U32;
    case DT_UINT64:   return ir::DType::U64;
    // Quantized types ARE these widths; the label below keeps the exact TF name
    // so a panel never claims a plain int8 tensor is unquantized.
    case DT_QINT8:    return ir::DType::I8;
    case DT_QUINT8:   return ir::DType::U8;
    case DT_QINT16:   return ir::DType::I16;
    case DT_QUINT16:  return ir::DType::U16;
    case DT_QINT32:   return ir::DType::I32;
    default:          return ir::DType::Unknown;
  }
}

// Exact TF type name for the types ir::DType cannot express on its own (the
// frozen 16-enumerator rule — see TensorRef::dtype_label). "" => dtype_name() is
// already honest.
const char* dtype_label(int64_t t) {
  switch (deref_dtype(t)) {
    case DT_STRING:     return "string";
    case DT_COMPLEX64:  return "complex64";
    case DT_COMPLEX128: return "complex128";
    case DT_RESOURCE:   return "resource";
    case DT_VARIANT:    return "variant";
    case DT_QINT8:      return "qint8";
    case DT_QUINT8:     return "quint8";
    case DT_QINT16:     return "qint16";
    case DT_QUINT16:    return "quint16";
    case DT_QINT32:     return "qint32";
    default:            return "";
  }
}

// TensorShapeProto: 2 dim (repeated Dim{1 size}), 3 unknown_rank (bool).
// A negative size means "unknown" and is normalized to -1 (honest-unknown).
struct ShapeInfo {
  bool present = false;
  bool unknown_rank = false;
  SmallVec<int64_t, 6> dims;
};

Result<ShapeInfo> parse_tensor_shape(const SubRange& sr) {
  ShapeInfo out;
  out.present = true;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 2 && h->wire_type == WireType::LenDelim) {
      auto d = r.read_len_delim();
      if (!d) return d.error();
      WireReader dr = WireReader::sub(*d);
      int64_t size = -1;
      while (!dr.at_end()) {
        auto dh = dr.read_tag();
        if (!dh) return dh.error();
        if (dh->field_number == 1 && dh->wire_type == WireType::Varint) {
          auto v = dr.read_varint();
          if (!v) return v.error();
          size = static_cast<int64_t>(*v);
        } else {
          auto sk = dr.skip_field(dh->wire_type);
          if (!sk) return sk.error();
        }
      }
      out.dims.push_back(size < 0 ? -1 : size);
    } else if (h->field_number == 3 && h->wire_type == WireType::Varint) {
      auto v = r.read_varint();
      if (!v) return v.error();
      out.unknown_rank = (*v != 0);
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return out;
}

// TensorProto: 1 dtype, 2 tensor_shape, 4 tensor_content (bytes). The content is
// the PAYLOAD: we record its absolute offset+length and never read the bytes.
// The typed *_val fields (5..17) are skipped the same way — skip_field on a
// length-delimited field only advances the cursor.
Result<ir::TensorRef> parse_tensor_proto(const SubRange& sr, ir::Model& model) {
  ir::TensorRef tr;
  bool has_content = false;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::Varint) {
      auto v = r.read_varint();
      if (!v) return v.error();
      const int64_t dt = static_cast<int64_t>(*v);
      tr.dtype = map_dtype(dt);
      const char* lbl = dtype_label(dt);
      if (*lbl != '\0') tr.dtype_label = model.intern(lbl);
    } else if (h->field_number == 2 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_len_delim();
      if (!s) return s.error();
      auto si = parse_tensor_shape(*s);
      if (!si) return si.error();
      tr.shape = si->dims;
    } else if (h->field_number == 4 && h->wire_type == WireType::LenDelim) {
      auto sub = r.read_len_delim();  // advances WITHOUT reading the contents
      if (!sub) return sub.error();
      tr.file_offset = sub->offset;
      tr.byte_len = sub->len;
      has_content = true;
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  if (!has_content) {  // inline typed *_val payload, or empty: no mmap range
    tr.file_offset = UINT64_MAX;
    tr.byte_len = 0;
  }
  return tr;
}

// AttrValue (attr_value.proto), plus the few fields the graph builder needs to
// read structurally (a Const's tensor, a Placeholder's shape/dtype, the
// _output_shapes hint, and a function reference).
struct AttrOut {
  ir::AttrValue value;
  bool has_type = false;
  int64_t type = DT_INVALID;
  bool has_shape = false;
  ShapeInfo shape;
  std::vector<ShapeInfo> shapes;  // list.shape — used for _output_shapes
  bool has_tensor = false;
  std::string func;               // NameAttrList.name
};

// AttrValue.ListValue (field 1): 2 s, 3 i, 4 f, 5 b, 6 type, 7 shape.
Result<bool> parse_list_value(const SubRange& sr, ir::Model& model, AttrOut* out) {
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 2: {  // s (repeated bytes)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          out->value.strings.push_back(model.intern(*s));
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 3:    // i (repeated int64)
      case 5:    // b (repeated bool)
      case 6: {  // type (repeated DataType)
        if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          out->value.ints.push_back(static_cast<int64_t>(*v));
        } else if (h->wire_type == WireType::LenDelim) {  // packed
          auto pk = r.read_len_delim();
          if (!pk) return pk.error();
          WireReader sub = WireReader::sub(*pk);
          while (!sub.at_end()) {
            auto v = sub.read_varint();
            if (!v) return v.error();
            out->value.ints.push_back(static_cast<int64_t>(*v));
          }
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 4: {  // f (repeated float)
        if (h->wire_type == WireType::Fixed32) {
          auto v = r.read_float();
          if (!v) return v.error();
          out->value.floats.push_back(static_cast<double>(*v));
        } else if (h->wire_type == WireType::LenDelim) {  // packed
          auto pk = r.read_len_delim();
          if (!pk) return pk.error();
          WireReader sub = WireReader::sub(*pk);
          while (!sub.at_end()) {
            auto v = sub.read_float();
            if (!v) return v.error();
            out->value.floats.push_back(static_cast<double>(*v));
          }
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 7: {  // shape (repeated TensorShapeProto) — the _output_shapes hint
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          auto si = parse_tensor_shape(*s);
          if (!si) return si.error();
          out->shapes.push_back(si.take());
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      default: {
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
        break;
      }
    }
  }
  return true;
}

// AttrValue: 1 list, 2 s, 3 i, 4 f, 5 b, 6 type, 7 shape, 8 tensor, 9 func,
// 10 placeholder. Exactly one is set in a well-formed file.
Result<AttrOut> parse_attr_value(const SubRange& sr, ir::Model& model) {
  AttrOut out;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 1: {  // list
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          auto ok = parse_list_value(*s, model, &out);
          if (!ok) return ok.error();
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 2: {  // s (bytes)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          out.value.s = model.intern(*s);
          out.value.kind = ir::AttrValue::Kind::String;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 3:    // i (int64)
      case 5: {  // b (bool) — IR has no Bool kind; 0/1 as Int is lossless
        if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          out.value.i = static_cast<int64_t>(*v);
          out.value.kind = ir::AttrValue::Kind::Int;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 4: {  // f (float)
        if (h->wire_type == WireType::Fixed32) {
          auto v = r.read_float();
          if (!v) return v.error();
          out.value.f = static_cast<double>(*v);
          out.value.kind = ir::AttrValue::Kind::Float;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 6: {  // type (DataType) — surfaced as the readable dtype name
        if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          out.type = static_cast<int64_t>(*v);
          out.has_type = true;
          const char* lbl = dtype_label(out.type);
          out.value.s = model.intern(*lbl != '\0' ? lbl
                                                  : ir::dtype_name(map_dtype(out.type)));
          out.value.kind = ir::AttrValue::Kind::String;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 7: {  // shape (TensorShapeProto)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          auto si = parse_tensor_shape(*s);
          if (!si) return si.error();
          out.shape = si.take();
          out.has_shape = true;
          for (int64_t d : out.shape.dims) out.value.ints.push_back(d);
          out.value.kind = ir::AttrValue::Kind::Ints;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 8: {  // tensor (TensorProto) — a Const's payload
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          auto tr = parse_tensor_proto(*s, model);
          if (!tr) return tr.error();
          out.value.tensor = tr.take();
          out.value.kind = ir::AttrValue::Kind::Tensor;
          out.has_tensor = true;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 9: {  // func (NameAttrList: 1 name, 2 attr) — a library reference
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          WireReader fr = WireReader::sub(*s);
          while (!fr.at_end()) {
            auto fh = fr.read_tag();
            if (!fh) return fh.error();
            if (fh->field_number == 1 && fh->wire_type == WireType::LenDelim) {
              auto n = fr.read_string();
              if (!n) return n.error();
              out.func = std::move(*n);
            } else {
              auto sk = fr.skip_field(fh->wire_type);
              if (!sk) return sk.error();
            }
          }
          out.value.s = model.intern(out.func);
          out.value.kind = ir::AttrValue::Kind::String;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 10: {  // placeholder (string)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          out.value.s = model.intern(*s);
          out.value.kind = ir::AttrValue::Kind::String;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      default: {
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
        break;
      }
    }
  }
  // A list attribute has no scalar kind of its own; pick the populated one.
  if (out.value.kind == ir::AttrValue::Kind::None) {
    if (!out.value.ints.empty())         out.value.kind = ir::AttrValue::Kind::Ints;
    else if (!out.value.floats.empty())  out.value.kind = ir::AttrValue::Kind::Floats;
    else if (!out.value.strings.empty()) out.value.kind = ir::AttrValue::Kind::Strings;
  }
  return out;
}

// NodeDef.attr is a protobuf map: each entry is a message { 1 key, 2 value }.
// A default-valued AttrValue serializes to zero bytes, so field 2 may be absent —
// `val` starts as a valid empty range rather than a null one.
Result<bool> parse_attr_entry(const SubRange& sr, std::string* key, SubRange* val) {
  *val = SubRange{sr.ptr, sr.offset, 0};
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_string();
      if (!s) return s.error();
      *key = std::move(*s);
    } else if (h->field_number == 2 && h->wire_type == WireType::LenDelim) {
      auto v = r.read_len_delim();
      if (!v) return v.error();
      *val = *v;
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return true;
}

// NodeDef: 1 name, 2 op, 3 input (repeated string), 4 device, 5 attr (map).
struct RawNode {
  std::string name;
  StringId op;
  std::vector<std::string> inputs;
  std::vector<std::pair<std::string, SubRange>> attrs;
};

Result<RawNode> parse_node_def(const SubRange& sr, ir::Model& model) {
  RawNode n;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 1: {
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          n.name = std::move(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 2: {
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          n.op = model.intern(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 3: {
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          n.inputs.push_back(std::move(*s));
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 5: {
        if (h->wire_type == WireType::LenDelim) {
          auto e = r.read_len_delim();
          if (!e) return e.error();
          std::string key;
          SubRange val;
          auto ok = parse_attr_entry(*e, &key, &val);
          if (!ok) return ok.error();
          n.attrs.emplace_back(std::move(key), val);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      default: {
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
        break;
      }
    }
  }
  return n;
}

// A TF input reference is one of:
//   "name"            slot 0
//   "name:2"          slot 2 (top-level GraphDef)
//   "name:out_arg:2"  slot 2 (inside a FunctionDef)
//   "^name"           control dependency on `name`
// In every form the PRODUCING NODE is the text before the first ':', and the
// slot is the trailing ":N" when that last component is all digits. One rule
// covers both containers.
struct InputRef {
  std::string_view node;
  int64_t slot = 0;
  bool control = false;
};

InputRef split_input(std::string_view s) {
  InputRef ref;
  if (!s.empty() && s.front() == '^') {
    ref.control = true;
    s.remove_prefix(1);
  }
  const auto first = s.find(':');
  ref.node = first == std::string_view::npos ? s : s.substr(0, first);
  if (first != std::string_view::npos) {
    const std::string_view tail = s.substr(s.find_last_of(':') + 1);
    int64_t v = 0;
    bool digits = !tail.empty();
    for (char c : tail) {
      if (c < '0' || c > '9' || v > kMaxOutputSlot) { digits = false; break; }
      v = v * 10 + (c - '0');
    }
    if (digits) ref.slot = v;
  }
  return ref;
}

// Canonical edge name: slot 0 is the bare node name (so "foo" and "foo:0" are
// the same value), higher slots get the ":N" suffix.
std::string value_name(std::string_view node, int64_t slot) {
  std::string s(node);
  if (slot != 0) {
    s += ':';
    s += std::to_string(slot);
  }
  return s;
}

// A node attribute that names a library function, resolved to a graph index once
// every FunctionDef has been parsed.
struct PendingFunc {
  int32_t graph = 0;
  uint32_t node = 0;
  std::string func;
};

// Build one ir::Graph from a flat list of NodeDefs. `arg_names` are FunctionDef
// signature inputs (they have no producing node) and `ret_refs` are its declared
// outputs; both are empty for a top-level GraphDef.
Result<int32_t> build_graph(const std::vector<SubRange>& node_subs, ir::Model& model,
                            StringId graph_name,
                            const std::vector<std::string>& arg_names,
                            const std::vector<std::string>& ret_refs,
                            std::vector<PendingFunc>* pending) {
  const int32_t my_idx = static_cast<int32_t>(model.graphs.size());
  model.graphs.emplace_back();  // reserve the slot; never hold a reference

  ir::Graph g;
  g.name = graph_name;
  std::unordered_map<std::string, uint32_t> vmap;

  auto get_value = [&](const std::string& nm) -> uint32_t {
    auto it = vmap.find(nm);
    if (it != vmap.end()) return it->second;
    const uint32_t idx = static_cast<uint32_t>(g.values.size());
    ir::ValueInfo v;
    v.name = model.intern(nm);
    g.values.push_back(std::move(v));
    vmap.emplace(nm, idx);
    return idx;
  };

  std::vector<RawNode> raw;
  raw.reserve(node_subs.size());
  for (const SubRange& sr : node_subs) {
    auto rn = parse_node_def(sr, model);
    if (!rn) return rn.error();
    raw.push_back(rn.take());
  }

  // Pass A: a NodeDef never declares its output arity, so derive it from the
  // highest slot any consumer references.
  std::unordered_map<std::string, int64_t> max_slot;
  for (const RawNode& rn : raw) max_slot.emplace(rn.name, 0);
  for (const RawNode& rn : raw) {
    for (const std::string& in : rn.inputs) {
      const InputRef ref = split_input(in);
      if (ref.control || ref.slot > kMaxOutputSlot) continue;
      auto it = max_slot.find(std::string(ref.node));
      if (it != max_slot.end() && ref.slot > it->second) it->second = ref.slot;
    }
  }

  // Function arguments exist as values before any node consumes them.
  for (const std::string& a : arg_names) get_value(a);

  std::vector<uint32_t> consumed;

  // Pass B: nodes in file order, so node indices match the on-disk order.
  for (const RawNode& rn : raw) {
    const int32_t node_idx = static_cast<int32_t>(g.nodes.size());
    ir::Node node;
    node.op_type = rn.op;
    node.name = model.intern(rn.name);

    node.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    for (const std::string& in : rn.inputs) {
      const InputRef ref = split_input(in);
      // A control dependency IS an edge from the named node to this one; wiring
      // it to that node's slot 0 keeps the dependency visible in the graph.
      const uint32_t vi = get_value(value_name(ref.node, ref.control ? 0 : ref.slot));
      g.edge_refs.push_back(vi);
      consumed.push_back(vi);
    }
    node.inputs.count = static_cast<uint32_t>(rn.inputs.size());

    const int64_t slots = max_slot[rn.name];
    std::vector<uint32_t> out_vis;
    out_vis.reserve(static_cast<size_t>(slots) + 1);
    node.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    for (int64_t s = 0; s <= slots; ++s) {
      const uint32_t vi = get_value(value_name(rn.name, s));
      g.values[vi].producer = node_idx;
      g.edge_refs.push_back(vi);
      out_vis.push_back(vi);
    }
    node.outputs.count = static_cast<uint32_t>(out_vis.size());

    node.attributes.begin = static_cast<uint32_t>(g.attributes.size());
    for (const auto& kv : rn.attrs) {
      auto a = parse_attr_value(kv.second, model);
      if (!a) return a.error();

      if (kv.first == "value" && a->has_tensor &&
          (model.str(rn.op) == "Const" || model.str(rn.op) == "HostConst")) {
        // The Const's payload: offset+len already recorded, bytes untouched.
        ir::TensorRef tr = a->value.tensor;
        tr.name = node.name;
        g.values[out_vis[0]].dtype = tr.dtype;
        g.values[out_vis[0]].shape = tr.shape;
        g.initializers.push_back(std::move(tr));
      } else if (kv.first == "dtype" && a->has_type &&
                 g.values[out_vis[0]].dtype == ir::DType::Unknown) {
        g.values[out_vis[0]].dtype = map_dtype(a->type);
      } else if (kv.first == "shape" && a->has_shape && !a->shape.unknown_rank &&
                 g.values[out_vis[0]].shape.empty()) {
        g.values[out_vis[0]].shape = a->shape.dims;
      } else if (kv.first == "_output_shapes") {
        for (size_t i = 0; i < a->shapes.size() && i < out_vis.size(); ++i) {
          if (a->shapes[i].unknown_rank) continue;  // honest-unknown: leave empty
          if (g.values[out_vis[i]].shape.empty())
            g.values[out_vis[i]].shape = a->shapes[i].dims;
        }
      }
      if (!a->func.empty() && pending != nullptr) {
        pending->push_back({my_idx, static_cast<uint32_t>(node_idx), a->func});
      }

      ir::Attribute attr;
      attr.name = model.intern(kv.first);
      attr.value = std::move(a->value);
      g.attributes.push_back(std::move(attr));
    }
    node.attributes.count =
        static_cast<uint32_t>(g.attributes.size()) - node.attributes.begin;

    if (model.str(rn.op) == "Placeholder" ||
        model.str(rn.op) == "PlaceholderWithDefault") {
      g.graph_inputs.push_back(out_vis[0]);
    }
    g.nodes.push_back(std::move(node));
  }

  // Anything referenced but produced by no node in this graph is an input too
  // (function arguments, and dangling refs in a partial graph).
  {
    std::vector<uint8_t> is_input(g.values.size(), 0);
    for (uint32_t vi : g.graph_inputs) is_input[vi] = 1;
    for (uint32_t vi = 0; vi < g.values.size(); ++vi) {
      if (g.values[vi].producer < 0 && !is_input[vi]) g.graph_inputs.push_back(vi);
    }
  }

  if (!ret_refs.empty()) {
    for (const std::string& r : ret_refs) {
      const InputRef ref = split_input(r);
      g.graph_outputs.push_back(get_value(value_name(ref.node, ref.slot)));
    }
  } else {
    // A GraphDef declares no outputs; a produced value nothing consumes is one.
    std::vector<uint8_t> used(g.values.size(), 0);
    for (uint32_t vi : consumed) used[vi] = 1;
    for (uint32_t vi = 0; vi < g.values.size(); ++vi) {
      if (!used[vi] && g.values[vi].producer >= 0) g.graph_outputs.push_back(vi);
    }
  }

  model.graphs[my_idx] = std::move(g);
  return my_idx;
}

// FunctionDef: 1 signature (OpDef), 3 node_def (repeated NodeDef), 4 ret (map).
// OpDef: 1 name, 2 input_arg (repeated ArgDef{1 name}).
struct FuncParts {
  std::string name;
  std::vector<std::string> args;
  std::vector<std::string> rets;
  std::vector<SubRange> nodes;
};

Result<bool> parse_op_def(const SubRange& sr, FuncParts* out) {
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_string();
      if (!s) return s.error();
      out->name = std::move(*s);
    } else if (h->field_number == 2 && h->wire_type == WireType::LenDelim) {
      auto a = r.read_len_delim();
      if (!a) return a.error();
      WireReader ar = WireReader::sub(*a);
      while (!ar.at_end()) {
        auto ah = ar.read_tag();
        if (!ah) return ah.error();
        if (ah->field_number == 1 && ah->wire_type == WireType::LenDelim) {
          auto n = ar.read_string();
          if (!n) return n.error();
          out->args.push_back(std::move(*n));
        } else {
          auto sk = ar.skip_field(ah->wire_type);
          if (!sk) return sk.error();
        }
      }
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return true;
}

Result<FuncParts> parse_function_def(const SubRange& sr) {
  FuncParts out;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 1: {  // signature
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          auto ok = parse_op_def(*s, &out);
          if (!ok) return ok.error();
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 3: {  // node_def
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          out.nodes.push_back(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 4: {  // ret: map<output_arg_name, node_ref>
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          WireReader er = WireReader::sub(*s);
          while (!er.at_end()) {
            auto eh = er.read_tag();
            if (!eh) return eh.error();
            if (eh->field_number == 2 && eh->wire_type == WireType::LenDelim) {
              auto v = er.read_string();
              if (!v) return v.error();
              out.rets.push_back(std::move(*v));
            } else {
              auto sk = er.skip_field(eh->wire_type);
              if (!sk) return sk.error();
            }
          }
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      default: {
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
        break;
      }
    }
  }
  return out;
}

// GraphDef: 1 node, 2 versions (VersionDef{1 producer}), 4 library
// (FunctionDefLibrary{1 function}).
Result<bool> build_from_graph_def(const SubRange& sr, ir::Model& model,
                                  int64_t* producer_version) {
  std::vector<SubRange> node_subs, func_subs;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 1: {
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          node_subs.push_back(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 2: {  // versions
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          WireReader vr = WireReader::sub(*s);
          while (!vr.at_end()) {
            auto vh = vr.read_tag();
            if (!vh) return vh.error();
            if (vh->field_number == 1 && vh->wire_type == WireType::Varint) {
              auto v = vr.read_varint();
              if (!v) return v.error();
              *producer_version = static_cast<int64_t>(*v);
            } else {
              auto sk = vr.skip_field(vh->wire_type);
              if (!sk) return sk.error();
            }
          }
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 4: {  // library (FunctionDefLibrary): 1 function (repeated FunctionDef)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          WireReader lr = WireReader::sub(*s);
          while (!lr.at_end()) {
            auto lh = lr.read_tag();
            if (!lh) return lh.error();
            if (lh->field_number == 1 && lh->wire_type == WireType::LenDelim) {
              auto f = lr.read_len_delim();
              if (!f) return f.error();
              func_subs.push_back(*f);
            } else {
              auto sk = lr.skip_field(lh->wire_type);
              if (!sk) return sk.error();
            }
          }
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      default: {
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
        break;
      }
    }
  }

  std::vector<PendingFunc> pending;
  auto main = build_graph(node_subs, model, model.intern("graph"), {}, {}, &pending);
  if (!main) return main.error();

  std::unordered_map<std::string, int32_t> func_index;
  for (const SubRange& f : func_subs) {
    auto fp = parse_function_def(f);
    if (!fp) return fp.error();
    auto gi = build_graph(fp->nodes, model, model.intern(fp->name), fp->args,
                          fp->rets, &pending);
    if (!gi) return gi.error();
    func_index.emplace(fp->name, *gi);
  }

  // Link every `f`/`body`/`then_branch`/... attribute to the graph it names, so
  // a PartitionedCall in a TF2 SavedModel can be drilled into.
  for (const PendingFunc& p : pending) {
    auto it = func_index.find(p.func);
    if (it == func_index.end()) continue;
    ir::Node& n = model.graphs[static_cast<size_t>(p.graph)].nodes[p.node];
    if (n.subgraph < 0) n.subgraph = it->second;
  }

  if (!func_subs.empty()) {
    model.metadata.emplace_back(model.intern("functions"),
                                model.intern(std::to_string(func_subs.size())));
  }
  return true;
}

// MetaGraphDef: 1 meta_info_def, 2 graph_def. MetaInfoDef: 4 tags (repeated
// string), 5 tensorflow_version, 6 tensorflow_git_version.
Result<bool> build_from_meta_graph(const SubRange& sr, ir::Model& model,
                                   int64_t* producer_version) {
  std::string tags, tf_version;
  bool built = false;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_len_delim();
      if (!s) return s.error();
      WireReader mr = WireReader::sub(*s);
      while (!mr.at_end()) {
        auto mh = mr.read_tag();
        if (!mh) return mh.error();
        if (mh->field_number == 4 && mh->wire_type == WireType::LenDelim) {
          auto t = mr.read_string();
          if (!t) return t.error();
          if (!tags.empty()) tags += ", ";
          tags += *t;
        } else if (mh->field_number == 5 && mh->wire_type == WireType::LenDelim) {
          auto t = mr.read_string();
          if (!t) return t.error();
          tf_version = std::move(*t);
        } else {
          auto sk = mr.skip_field(mh->wire_type);
          if (!sk) return sk.error();
        }
      }
    } else if (h->field_number == 2 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_len_delim();
      if (!s) return s.error();
      auto ok = build_from_graph_def(*s, model, producer_version);
      if (!ok) return ok.error();
      built = true;
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  if (!tags.empty())
    model.metadata.emplace_back(model.intern("tags"), model.intern(tags));
  if (!tf_version.empty()) {
    model.metadata.emplace_back(model.intern("tensorflow_version"),
                                model.intern(tf_version));
    model.producer = model.intern("TensorFlow " + tf_version);
  }
  if (!built) return err("SavedModel meta_graph carries no graph_def", sr.offset);
  return true;
}

// SavedModel weights live in a checkpoint under variables/, whose index is a
// compressed sstable. NetVis does not decode it — say so rather than inventing
// offsets for tensors it cannot locate (honest-unknown).
void note_variables(const MappedFile& file, ir::Model& model) {
  std::error_code ec;
  const std::filesystem::path dir =
      std::filesystem::path(file.path()).parent_path() / "variables";
  if (!std::filesystem::is_regular_file(dir / "variables.index", ec)) return;
  model.metadata.emplace_back(
      model.intern("variables"),
      model.intern("checkpoint present (payloads not decoded)"));
}

}  // namespace

// TensorFlow entry point (declared in parsers/Parser.h). Reads structure only.
Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "Parsing TensorFlow");

  ir::Model model;
  model.format_name = model.intern("TensorFlow");
  model.has_graph = true;

  if (!file.valid() || file.data() == nullptr)
    return err("empty or unmapped file", 0);

  // SavedModel's first field is saved_model_schema_version (1, varint); a bare
  // GraphDef's is node (1, length-delimited). That single tag separates them.
  bool saved_model = false;
  {
    WireReader probe(file.data(), file.size(), 0);
    auto h = probe.read_tag();
    if (h && h->field_number == 1 && h->wire_type == WireType::Varint)
      saved_model = true;
  }

  int64_t producer_version = -1;
  int64_t schema_version = -1;
  size_t meta_graphs = 0;

  if (saved_model) {
    progress.set(0.2f, "Reading SavedModel");
    WireReader r(file.data(), file.size(), 0);
    while (!r.at_end()) {
      auto h = r.read_tag();
      if (!h) return h.error();
      if (h->field_number == 1 && h->wire_type == WireType::Varint) {
        auto v = r.read_varint();
        if (!v) return v.error();
        schema_version = static_cast<int64_t>(*v);
      } else if (h->field_number == 2 && h->wire_type == WireType::LenDelim) {
        auto mg = r.read_len_delim();
        if (!mg) return mg.error();
        ++meta_graphs;
        if (meta_graphs == 1) {
          progress.set(0.3f, "Building graph");
          auto ok = build_from_meta_graph(*mg, model, &producer_version);
          if (!ok) return ok.error();
        }
      } else {
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
      }
    }
    if (schema_version >= 0) {
      model.metadata.emplace_back(model.intern("saved_model_schema_version"),
                                  model.intern(std::to_string(schema_version)));
    }
    // Only the first meta_graph is rendered; saying how many exist beats
    // silently dropping the rest.
    if (meta_graphs > 1) {
      model.metadata.emplace_back(model.intern("meta_graphs"),
                                  model.intern(std::to_string(meta_graphs)));
    }
    note_variables(file, model);
  } else {
    progress.set(0.3f, "Building graph");
    SubRange whole{file.data(), 0, file.size()};
    auto ok = build_from_graph_def(whole, model, &producer_version);
    if (!ok) return ok.error();
  }

  if (model.graphs.empty()) model.graphs.emplace_back();

  std::string version_info = saved_model ? "SavedModel" : "frozen GraphDef";
  if (producer_version >= 0)
    version_info += "; GraphDef producer " + std::to_string(producer_version);
  model.version_info = model.intern(version_info);

  progress.set(1.0f, "Done");
  return model;
}

}  // namespace netvis::tensorflow
