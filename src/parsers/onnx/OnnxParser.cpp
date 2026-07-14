// parsers/onnx/OnnxParser.cpp — ONNX (protobuf) -> ir::Model.
//
// DECISION (spec §2.1): we walk the protobuf structure with a hand-rolled wire
// reader (no libprotobuf) and NEVER decode tensor payloads. For every weight we
// record only its absolute mmap offset + byte length into a TensorRef; the raw
// bytes stay on disk until the weight inspector reads them. All reads go through
// the bounds-checked WireReader/ByteReader so a malformed/truncated file yields
// a Result error carrying a byte offset instead of crashing.
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/onnx/WireReader.h"

namespace netvis::onnx {
namespace {

// ---- ONNX enums (from onnx.proto3) ----------------------------------------
// TensorProto.DataType
enum OnnxDType : int32_t {
  UNDEFINED = 0, FLOAT = 1, UINT8 = 2, INT8 = 3, UINT16 = 4, INT16 = 5,
  INT32 = 6, INT64 = 7, STRING = 8, BOOL = 9, FLOAT16 = 10, DOUBLE = 11,
  UINT32 = 12, UINT64 = 13, COMPLEX64 = 14, COMPLEX128 = 15, BFLOAT16 = 16,
};

// AttributeProto.AttributeType
enum OnnxAttrType : int32_t {
  AT_UNDEFINED = 0, AT_FLOAT = 1, AT_INT = 2, AT_STRING = 3, AT_TENSOR = 4,
  AT_GRAPH = 5, AT_FLOATS = 6, AT_INTS = 7, AT_STRINGS = 8, AT_GRAPHS = 9,
};

constexpr int kMaxGraphDepth = 64;  // guard against adversarial nesting

ir::DType map_dtype(int64_t t) {
  switch (t) {
    case FLOAT:    return ir::DType::F32;
    case UINT8:    return ir::DType::U8;
    case INT8:     return ir::DType::I8;
    case UINT16:   return ir::DType::U16;
    case INT16:    return ir::DType::I16;
    case INT32:    return ir::DType::I32;
    case INT64:    return ir::DType::I64;
    case BOOL:     return ir::DType::Bool;
    case FLOAT16:  return ir::DType::F16;
    case DOUBLE:   return ir::DType::F64;
    case UINT32:   return ir::DType::U32;
    case UINT64:   return ir::DType::U64;
    case BFLOAT16: return ir::DType::BF16;
    default:       return ir::DType::Unknown;
  }
}

// Read all varints packed into a length-delimited sub-range as int64.
Result<bool> read_packed_int64(const SubRange& sr, SmallVec<int64_t, 6>* out) {
  WireReader sub = WireReader::sub(sr);
  while (!sub.at_end()) {
    auto v = sub.read_varint();
    if (!v) return v.error();
    out->push_back(static_cast<int64_t>(*v));
  }
  return true;
}

// Parsed TypeProto info for a ValueInfo.
struct TypeInfo {
  bool present = false;
  ir::DType dtype = ir::DType::Unknown;
  SmallVec<int64_t, 6> shape;
};

// TensorShapeProto: field 1 = dim (repeated Dimension).
// Dimension: field 1 dim_value(varint) OR field 2 dim_param(str) -> -1 dynamic.
Result<bool> parse_tensor_shape(const SubRange& sr, SmallVec<int64_t, 6>* shape) {
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
      auto dim = r.read_len_delim();
      if (!dim) return dim.error();
      // Parse one Dimension message.
      WireReader d = WireReader::sub(*dim);
      int64_t dim_value = -1;  // default dynamic
      bool got = false;
      while (!d.at_end()) {
        auto dh = d.read_tag();
        if (!dh) return dh.error();
        if (dh->field_number == 1 && dh->wire_type == WireType::Varint) {
          auto dv = d.read_varint();
          if (!dv) return dv.error();
          dim_value = static_cast<int64_t>(*dv);
          got = true;
        } else if (dh->field_number == 2 && dh->wire_type == WireType::LenDelim) {
          auto dp = d.read_len_delim();  // dim_param: symbolic -> dynamic (-1)
          if (!dp) return dp.error();
          dim_value = -1;
          got = true;
        } else {
          auto sk = d.skip_field(dh->wire_type);
          if (!sk) return sk.error();
        }
      }
      (void)got;
      shape->push_back(dim_value);
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return true;
}

// TypeProto: field 1 = tensor_type (Tensor).
// Tensor: field 1 elem_type(varint), field 2 shape(TensorShapeProto).
Result<TypeInfo> parse_type_proto(const SubRange& sr) {
  TypeInfo info;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
      auto tt = r.read_len_delim();
      if (!tt) return tt.error();
      info.present = true;
      WireReader t = WireReader::sub(*tt);
      while (!t.at_end()) {
        auto th = t.read_tag();
        if (!th) return th.error();
        if (th->field_number == 1 && th->wire_type == WireType::Varint) {
          auto et = t.read_varint();
          if (!et) return et.error();
          info.dtype = map_dtype(static_cast<int64_t>(*et));
        } else if (th->field_number == 2 && th->wire_type == WireType::LenDelim) {
          auto sh = t.read_len_delim();
          if (!sh) return sh.error();
          auto ps = parse_tensor_shape(*sh, &info.shape);
          if (!ps) return ps.error();
        } else {
          auto sk = t.skip_field(th->wire_type);
          if (!sk) return sk.error();
        }
      }
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return info;
}

// StringStringEntryProto: field 1 key, field 2 value.
Result<std::pair<std::string, std::string>> parse_string_entry(const SubRange& sr) {
  std::string key, value;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_string();
      if (!s) return s.error();
      key = std::move(*s);
    } else if (h->field_number == 2 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_string();
      if (!s) return s.error();
      value = std::move(*s);
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return std::make_pair(std::move(key), std::move(value));
}

// TensorProto -> TensorRef. NEVER reads payload bytes: raw_data (field 9) is
// captured as absolute mmap offset + length via a length-delimited sub-range
// whose bytes we do not touch; typed *_data payload fields are skipped.
Result<ir::TensorRef> parse_tensor_proto(const SubRange& sr, ir::Model& model) {
  ir::TensorRef tr;
  int64_t data_location = 0;  // 0 == DEFAULT (inline), 1 == EXTERNAL
  bool has_raw = false;
  std::string ext_location;
  bool ext_offset_set = false, ext_length_set = false;
  uint64_t ext_offset = 0, ext_length = 0;

  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 1: {  // dims (repeated int64), packed or single varint
        if (h->wire_type == WireType::LenDelim) {
          auto pk = r.read_len_delim();
          if (!pk) return pk.error();
          auto ok = read_packed_int64(*pk, &tr.shape);
          if (!ok) return ok.error();
        } else if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          tr.shape.push_back(static_cast<int64_t>(*v));
        } else {
          auto sk = r.skip_field(h->wire_type);
          if (!sk) return sk.error();
        }
        break;
      }
      case 2: {  // data_type (varint)
        if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          tr.dtype = map_dtype(static_cast<int64_t>(*v));
        } else {
          auto sk = r.skip_field(h->wire_type);
          if (!sk) return sk.error();
        }
        break;
      }
      case 8: {  // name (string)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          tr.name = model.intern(*s);
        } else {
          auto sk = r.skip_field(h->wire_type);
          if (!sk) return sk.error();
        }
        break;
      }
      case 9: {  // raw_data (bytes) — PAYLOAD: record offset+len, DO NOT read.
        if (h->wire_type == WireType::LenDelim) {
          auto sub = r.read_len_delim();  // advances without reading contents
          if (!sub) return sub.error();
          tr.file_offset = sub->offset;   // absolute mmap offset
          tr.byte_len = sub->len;
          has_raw = true;
        } else {
          auto sk = r.skip_field(h->wire_type);
          if (!sk) return sk.error();
        }
        break;
      }
      case 13: {  // external_data (repeated StringStringEntryProto)
        if (h->wire_type == WireType::LenDelim) {
          auto e = r.read_len_delim();
          if (!e) return e.error();
          auto kv = parse_string_entry(*e);
          if (!kv) return kv.error();
          const std::string& k = kv->first;
          const std::string& v = kv->second;
          if (k == "location") {
            ext_location = v;
          } else if (k == "offset") {
            ext_offset = std::strtoull(v.c_str(), nullptr, 10);
            ext_offset_set = true;
          } else if (k == "length") {
            ext_length = std::strtoull(v.c_str(), nullptr, 10);
            ext_length_set = true;
          }
        } else {
          auto sk = r.skip_field(h->wire_type);
          if (!sk) return sk.error();
        }
        break;
      }
      case 14: {  // data_location (varint); 1 == EXTERNAL
        if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          data_location = static_cast<int64_t>(*v);
        } else {
          auto sk = r.skip_field(h->wire_type);
          if (!sk) return sk.error();
        }
        break;
      }
      // PAYLOAD fields we skip WITHOUT reading contents. Packed repeated
      // scalars are length-delimited, so read_len_delim (via skip_field) only
      // advances the cursor; it never touches the weight bytes.
      case 4:   // float_data
      case 5:   // int32_data
      case 6:   // string_data
      case 7:   // int64_data
      case 10:  // double_data
      case 11:  // uint64_data
      default: {
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
        break;
      }
    }
  }

  // Resolve external data location if this tensor lives outside the file.
  if (data_location == 1 || !ext_location.empty()) {
    tr.external_path = model.intern(ext_location);
    tr.file_offset = ext_offset_set ? ext_offset : 0;
    tr.byte_len = ext_length_set ? ext_length : 0;
  } else if (!has_raw) {
    // Inline typed payload (float_data etc.) or empty: no mmap byte range.
    tr.file_offset = UINT64_MAX;
    tr.byte_len = 0;
  }
  return tr;
}

// Forward decl (attributes may embed subgraphs).
Result<int32_t> parse_graph(const SubRange& sr, ir::Model& model, int depth);

// AttributeProto -> ir::Attribute. Sets *out_subgraph to a Model::graphs index
// when the attribute carries a graph body (field 9 g).
Result<ir::Attribute> parse_attribute(const SubRange& sr, ir::Model& model,
                                      int32_t* out_subgraph, int depth) {
  ir::Attribute attr;
  int64_t attr_type = AT_UNDEFINED;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 1: {  // name (string)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          attr.name = model.intern(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 20: {  // type (varint)
        if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          attr_type = static_cast<int64_t>(*v);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 2: {  // f (float, fixed32)
        if (h->wire_type == WireType::Fixed32) {
          auto v = r.read_float();
          if (!v) return v.error();
          attr.value.f = static_cast<double>(*v);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 3: {  // i (varint)
        if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          attr.value.i = static_cast<int64_t>(*v);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 4: {  // s (bytes)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          attr.value.s = model.intern(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 7: {  // floats (repeated float): packed (LenDelim) or single Fixed32
        if (h->wire_type == WireType::LenDelim) {
          auto pk = r.read_len_delim();
          if (!pk) return pk.error();
          WireReader sub = WireReader::sub(*pk);
          while (!sub.at_end()) {
            auto v = sub.read_float();
            if (!v) return v.error();
            attr.value.floats.push_back(static_cast<double>(*v));
          }
        } else if (h->wire_type == WireType::Fixed32) {
          auto v = r.read_float();
          if (!v) return v.error();
          attr.value.floats.push_back(static_cast<double>(*v));
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 8: {  // ints (repeated int64): packed (LenDelim) or single Varint
        if (h->wire_type == WireType::LenDelim) {
          auto pk = r.read_len_delim();
          if (!pk) return pk.error();
          WireReader sub = WireReader::sub(*pk);
          while (!sub.at_end()) {
            auto v = sub.read_varint();
            if (!v) return v.error();
            attr.value.ints.push_back(static_cast<int64_t>(*v));
          }
        } else if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          attr.value.ints.push_back(static_cast<int64_t>(*v));
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 9: {  // strings (repeated bytes)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          attr.value.strings.push_back(model.intern(*s));
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 5: {  // t (TensorProto)
        if (h->wire_type == WireType::LenDelim) {
          auto tsub = r.read_len_delim();
          if (!tsub) return tsub.error();
          auto tr = parse_tensor_proto(*tsub, model);
          if (!tr) return tr.error();
          attr.value.tensor = tr.take();
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 6: {  // g (GraphProto) — primary subgraph body
        if (h->wire_type == WireType::LenDelim) {
          auto gsub = r.read_len_delim();
          if (!gsub) return gsub.error();
          auto gi = parse_graph(*gsub, model, depth + 1);
          if (!gi) return gi.error();
          attr.value.graph = *gi;
          if (out_subgraph) *out_subgraph = *gi;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 11: {  // graphs (repeated GraphProto)
        if (h->wire_type == WireType::LenDelim) {
          auto gsub = r.read_len_delim();
          if (!gsub) return gsub.error();
          auto gi = parse_graph(*gsub, model, depth + 1);
          if (!gi) return gi.error();
          if (attr.value.graph < 0) attr.value.graph = *gi;
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

  // Select the active AttrValue::Kind from the declared type (falls back to
  // presence when the type field is absent in older files).
  using Kind = ir::AttrValue::Kind;
  switch (attr_type) {
    case AT_FLOAT:   attr.value.kind = Kind::Float; break;
    case AT_INT:     attr.value.kind = Kind::Int; break;
    case AT_STRING:  attr.value.kind = Kind::String; break;
    case AT_TENSOR:  attr.value.kind = Kind::Tensor; break;
    case AT_GRAPH:   attr.value.kind = Kind::Graph; break;
    case AT_FLOATS:  attr.value.kind = Kind::Floats; break;
    case AT_INTS:    attr.value.kind = Kind::Ints; break;
    case AT_STRINGS: attr.value.kind = Kind::Strings; break;
    case AT_GRAPHS:  attr.value.kind = Kind::Graph; break;
    default:
      if (attr.value.graph >= 0) attr.value.kind = Kind::Graph;
      else if (!attr.value.strings.empty()) attr.value.kind = Kind::Strings;
      else if (!attr.value.ints.empty()) attr.value.kind = Kind::Ints;
      else if (!attr.value.floats.empty()) attr.value.kind = Kind::Floats;
      else if (attr.value.s.valid()) attr.value.kind = Kind::String;
      else if (attr.value.tensor.name.valid() ||
               attr.value.tensor.byte_len > 0) attr.value.kind = Kind::Tensor;
      else attr.value.kind = Kind::None;
      break;
  }
  return attr;
}

// Per-graph value map: distinct edge name -> ValueInfo index.
using ValueMap = std::unordered_map<uint32_t, uint32_t>;

uint32_t get_or_create_value(ir::Graph& g, ValueMap& map, StringId name) {
  auto it = map.find(name.id);
  if (it != map.end()) return it->second;
  uint32_t idx = static_cast<uint32_t>(g.values.size());
  ir::ValueInfo v;
  v.name = name;
  g.values.push_back(std::move(v));
  map.emplace(name.id, idx);
  return idx;
}

// NodeProto: 1 input(rep str), 2 output(rep str), 3 name, 4 op_type,
// 5 attribute(rep AttributeProto), 7 domain.
Result<bool> parse_node(const SubRange& sr, ir::Model& model, ir::Graph& g,
                        ValueMap& map, int depth) {
  std::vector<StringId> inputs, outputs;
  std::vector<SubRange> attr_subs;
  StringId op_type, name;

  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 1: {  // input (repeated string)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          inputs.push_back(model.intern(*s));
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 2: {  // output (repeated string)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          outputs.push_back(model.intern(*s));
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 3: {  // name
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          name = model.intern(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 4: {  // op_type
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          op_type = model.intern(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 5: {  // attribute (repeated AttributeProto) — defer parse
        if (h->wire_type == WireType::LenDelim) {
          auto a = r.read_len_delim();
          if (!a) return a.error();
          attr_subs.push_back(*a);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 7:    // domain (string) — recorded implicitly; not needed in IR
      default: {
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
        break;
      }
    }
  }

  int32_t node_idx = static_cast<int32_t>(g.nodes.size());

  ir::Node node;
  node.op_type = op_type;
  node.name = name;

  // inputs -> edge_refs
  node.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  for (StringId in : inputs) {
    g.edge_refs.push_back(get_or_create_value(g, map, in));
  }
  node.inputs.count = static_cast<uint32_t>(inputs.size());

  // outputs -> edge_refs; producer of each output is this node
  node.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  for (StringId out : outputs) {
    uint32_t vi = get_or_create_value(g, map, out);
    g.values[vi].producer = node_idx;
    g.edge_refs.push_back(vi);
  }
  node.outputs.count = static_cast<uint32_t>(outputs.size());

  // attributes (may create subgraphs, appended to model.graphs)
  node.attributes.begin = static_cast<uint32_t>(g.attributes.size());
  int32_t primary_subgraph = -1;
  for (const SubRange& asr : attr_subs) {
    int32_t sub_idx = -1;
    auto a = parse_attribute(asr, model, &sub_idx, depth);
    if (!a) return a.error();
    if (sub_idx >= 0 && primary_subgraph < 0) primary_subgraph = sub_idx;
    g.attributes.push_back(a.take());
  }
  node.attributes.count =
      static_cast<uint32_t>(g.attributes.size()) - node.attributes.begin;
  node.subgraph = primary_subgraph;

  g.nodes.push_back(std::move(node));
  return true;
}

// GraphProto: 1 node(rep), 2 name, 5 initializer(rep TensorProto),
// 11 input(rep ValueInfoProto), 12 output(rep), 13 value_info(rep).
// Reserves its own Model::graphs slot BEFORE recursing into subgraphs so that
// graph indices are stable and graphs[0] stays the main graph.
Result<int32_t> parse_graph(const SubRange& sr, ir::Model& model, int depth) {
  if (depth > kMaxGraphDepth)
    return err("subgraph nesting too deep", sr.offset);

  int32_t my_idx = static_cast<int32_t>(model.graphs.size());
  model.graphs.emplace_back();  // reserve slot; do not hold a reference

  ir::Graph g;
  ValueMap map;

  // Collect sub-ranges first so we can process in a deterministic order
  // (inputs -> initializers -> nodes -> value_info -> outputs) regardless of
  // the on-wire field ordering.
  std::vector<SubRange> node_subs, init_subs, input_subs, output_subs, vinfo_subs;
  StringId graph_name;

  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 1:
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim(); if (!s) return s.error();
          node_subs.push_back(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      case 2:
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string(); if (!s) return s.error();
          graph_name = model.intern(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      case 5:
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim(); if (!s) return s.error();
          init_subs.push_back(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      case 11:
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim(); if (!s) return s.error();
          input_subs.push_back(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      case 12:
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim(); if (!s) return s.error();
          output_subs.push_back(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      case 13:
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim(); if (!s) return s.error();
          vinfo_subs.push_back(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      default: {
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
        break;
      }
    }
  }

  g.name = graph_name;

  // ValueInfoProto: 1 name(str), 2 type(TypeProto). Returns (name, TypeInfo).
  auto parse_value_info =
      [&](const SubRange& vsr, StringId* out_name, TypeInfo* out_type) -> Result<bool> {
    WireReader vr = WireReader::sub(vsr);
    while (!vr.at_end()) {
      auto h = vr.read_tag();
      if (!h) return h.error();
      if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
        auto s = vr.read_string();
        if (!s) return s.error();
        *out_name = model.intern(*s);
      } else if (h->field_number == 2 && h->wire_type == WireType::LenDelim) {
        auto t = vr.read_len_delim();
        if (!t) return t.error();
        auto ti = parse_type_proto(*t);
        if (!ti) return ti.error();
        *out_type = ti.take();
      } else {
        auto sk = vr.skip_field(h->wire_type);
        if (!sk) return sk.error();
      }
    }
    return true;
  };

  // 1) graph inputs -> ValueInfo (producer -1), graph_inputs list.
  for (const SubRange& v : input_subs) {
    StringId nm; TypeInfo ti;
    auto ok = parse_value_info(v, &nm, &ti);
    if (!ok) return ok.error();
    uint32_t vi = get_or_create_value(g, map, nm);
    if (ti.present) { g.values[vi].dtype = ti.dtype; g.values[vi].shape = ti.shape; }
    g.graph_inputs.push_back(vi);
  }

  // 2) initializers -> TensorRef; also give them a ValueInfo (producer -1).
  for (const SubRange& t : init_subs) {
    auto tr = parse_tensor_proto(t, model);
    if (!tr) return tr.error();
    ir::TensorRef ref = tr.take();
    uint32_t vi = get_or_create_value(g, map, ref.name);
    g.values[vi].dtype = ref.dtype;
    g.values[vi].shape = ref.shape;
    g.initializers.push_back(std::move(ref));
  }

  // 3) nodes in order — sets producers and edge_refs.
  for (const SubRange& n : node_subs) {
    auto ok = parse_node(n, model, g, map, depth);
    if (!ok) return ok.error();
  }

  // 4) value_info -> fill dtype/shape of intermediate edges.
  for (const SubRange& v : vinfo_subs) {
    StringId nm; TypeInfo ti;
    auto ok = parse_value_info(v, &nm, &ti);
    if (!ok) return ok.error();
    uint32_t vi = get_or_create_value(g, map, nm);
    if (ti.present) { g.values[vi].dtype = ti.dtype; g.values[vi].shape = ti.shape; }
  }

  // 5) graph outputs.
  for (const SubRange& v : output_subs) {
    StringId nm; TypeInfo ti;
    auto ok = parse_value_info(v, &nm, &ti);
    if (!ok) return ok.error();
    uint32_t vi = get_or_create_value(g, map, nm);
    if (ti.present) { g.values[vi].dtype = ti.dtype; g.values[vi].shape = ti.shape; }
    g.graph_outputs.push_back(vi);
  }

  model.graphs[my_idx] = std::move(g);
  return my_idx;
}

}  // namespace

// ONNX entry point (declared in parsers/Parser.h). Reads structure only.
Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "Parsing ONNX");

  ir::Model model;
  model.format_name = model.intern("ONNX");
  model.has_graph = true;

  if (!file.valid() || file.data() == nullptr)
    return err("empty or unmapped file", 0);

  // ModelProto (onnx.proto3): 1 ir_version, 2 producer_name, 3 producer_version,
  // 4 domain, 5 model_version, 6 doc_string, 7 graph, 8 opset_import(rep),
  // 14 metadata_props(rep).
  int64_t ir_version = -1;
  std::string producer_name, producer_version;
  std::string opset_summary;
  bool have_graph = false;

  WireReader r(file.data(), file.size(), 0);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case 1: {  // ir_version (varint)
        if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          ir_version = static_cast<int64_t>(*v);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 2: {  // producer_name (string)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string(); if (!s) return s.error();
          producer_name = std::move(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 3: {  // producer_version (string)
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string(); if (!s) return s.error();
          producer_version = std::move(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 8: {  // opset_import (repeated OperatorSetIdProto)
        if (h->wire_type == WireType::LenDelim) {
          auto o = r.read_len_delim();
          if (!o) return o.error();
          // OperatorSetIdProto: 1 domain(str), 2 version(varint).
          WireReader os = WireReader::sub(*o);
          std::string domain;
          int64_t version = 0;
          while (!os.at_end()) {
            auto oh = os.read_tag();
            if (!oh) return oh.error();
            if (oh->field_number == 1 && oh->wire_type == WireType::LenDelim) {
              auto s = os.read_string(); if (!s) return s.error();
              domain = std::move(*s);
            } else if (oh->field_number == 2 && oh->wire_type == WireType::Varint) {
              auto v = os.read_varint(); if (!v) return v.error();
              version = static_cast<int64_t>(*v);
            } else {
              auto sk = os.skip_field(oh->wire_type); if (!sk) return sk.error();
            }
          }
          if (!opset_summary.empty()) opset_summary += ", ";
          opset_summary += (domain.empty() ? std::string("ai.onnx") : domain);
          opset_summary += ":" + std::to_string(version);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 7: {  // graph (GraphProto) -> main graph (index 0)
        if (h->wire_type == WireType::LenDelim) {
          auto gsub = r.read_len_delim();
          if (!gsub) return gsub.error();
          progress.set(0.3f, "Building graph");
          auto gi = parse_graph(*gsub, model, 0);
          if (!gi) return gi.error();
          have_graph = true;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case 14: {  // metadata_props (repeated StringStringEntryProto)
        if (h->wire_type == WireType::LenDelim) {
          auto e = r.read_len_delim();
          if (!e) return e.error();
          auto kv = parse_string_entry(*e);
          if (!kv) return kv.error();
          model.metadata.emplace_back(model.intern(kv->first),
                                      model.intern(kv->second));
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

  // If the file had no graph field, present an empty main graph so downstream
  // code always has graphs[0].
  if (!have_graph) model.graphs.emplace_back();

  // Producer + version summary strings.
  std::string producer = producer_name;
  if (!producer_version.empty()) {
    producer += producer.empty() ? producer_version : (" " + producer_version);
  }
  model.producer = model.intern(producer);

  std::string version_info;
  if (ir_version >= 0) version_info = "IR v" + std::to_string(ir_version);
  if (!opset_summary.empty()) {
    if (!version_info.empty()) version_info += "; ";
    version_info += "opset " + opset_summary;
  }
  model.version_info = model.intern(version_info);

  progress.set(1.0f, "Done");
  return model;
}

}  // namespace netvis::onnx
