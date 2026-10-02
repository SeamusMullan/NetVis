// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeNet.cpp — helpers shared by the text and binary readers:
// declared-shape sanitising (§5.4) and the native attribute builder (§3.9).
#include "parsers/caffe/CaffeNet.h"

#include <utility>

#include "core/SafeMath.h"

namespace netvis::caffe {

bool sanitize_shape(SmallVec<int64_t, 6>& dims) {
  // Negative (and below -1) -> -1, the honest unknown. Zero dims are kept: Caffe
  // allows them and they are what the file says. The product of the POSITIVE
  // dims must stay within Caffe's Blob::Reshape limit (INT32_MAX elements); that
  // bound, with no single dim above it either, is what keeps every value that
  // later reaches infer_shapes_ext far from int64 overflow.
  int64_t prod = 1;
  for (int64_t& d : dims) {
    if (d < 0) {
      d = -1;
      continue;
    }
    if (d == 0) continue;
    if (!checked_mul_i64(prod, d, &prod) || prod > INT32_MAX) return false;
  }
  return true;
}

ir::AttrValue* AttrSink::slot(const std::string& name, ir::AttrValue::Kind kind,
                              bool* fresh) {
  auto it = index_.find(name);
  if (it != index_.end()) {
    ir::AttrValue& v = out_[it->second].value;
    *fresh = false;
    if (v.kind != kind) {   // a schema/heuristic clash: the later value wins
      v = ir::AttrValue{};
      v.kind = kind;
      *fresh = true;
    }
    return &v;
  }
  ir::Attribute a;
  a.name = model_.intern(name);
  a.value.kind = kind;
  index_.emplace(name, out_.size());
  out_.push_back(std::move(a));
  *fresh = true;
  return &out_.back().value;
}

bool AttrSink::add_int(const std::string& name, int64_t v, bool repeated) {
  bool fresh = false;
  if (repeated) {
    ir::AttrValue* a = slot(name, ir::AttrValue::Kind::Ints, &fresh);
    if (a->ints.size() >= kMaxRepeatedScalars) return false;
    a->ints.push_back(v);
  } else {
    slot(name, ir::AttrValue::Kind::Int, &fresh)->i = v;
  }
  return true;
}

bool AttrSink::add_float(const std::string& name, double v, bool repeated) {
  bool fresh = false;
  if (repeated) {
    ir::AttrValue* a = slot(name, ir::AttrValue::Kind::Floats, &fresh);
    if (a->floats.size() >= kMaxRepeatedScalars) return false;
    a->floats.push_back(v);
  } else {
    slot(name, ir::AttrValue::Kind::Float, &fresh)->f = v;
  }
  return true;
}

bool AttrSink::add_string(const std::string& name, std::string_view v, bool repeated) {
  bool fresh = false;
  if (repeated) {
    ir::AttrValue* a = slot(name, ir::AttrValue::Kind::Strings, &fresh);
    if (a->strings.size() >= kMaxRepeatedScalars) return false;
    a->strings.push_back(model_.intern(v));
  } else {
    slot(name, ir::AttrValue::Kind::String, &fresh)->s = model_.intern(v);
  }
  return true;
}

}  // namespace netvis::caffe
