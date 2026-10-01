// SPDX-License-Identifier: Apache-2.0
// parsers/pytorch/TorchScriptIR.cpp — the shared TorchScript -> ir:: layer.
//
// See TorchScriptIR.h. Pure structure: no payload reads, no unordered-map
// iteration decides any emitted order, every traversal is depth-capped and
// visited-gated.
#include "parsers/pytorch/TorchScriptIR.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <utility>

namespace netvis::pytorch::ts {

std::string arg_attr_name(uint32_t position) {
  return "arg" + std::to_string(position);
}

bool is_torch_object(const Value& v) {
  return v.kind == Value::Kind::Opaque && v.module.rfind("__torch__", 0) == 0 &&
         v.inner && v.inner->kind == Value::Kind::Dict;
}

// ---- deterministic tensor traversal -----------------------------------------
namespace {

std::string dict_key(const ValuePtr& k) {
  if (k && k->kind == Value::Kind::Str) return k->s;
  if (k && k->kind == Value::Kind::Int) return std::to_string(k->i);
  return "";
}

void walk_tensors(
    const ValuePtr& v, const std::string& prefix,
    std::unordered_set<const Value*>& visited, int depth,
    const std::function<void(const std::string&, const Value&)>& fn) {
  if (!v || depth > kMaxTraversalDepth) return;
  // Containers and objects can recurse / be shared through the memo; gating
  // them on `visited` breaks cycles and dedups shared subtrees without
  // suppressing repeated leaf tensors.
  if (v->kind == Value::Kind::Dict || v->kind == Value::Kind::List ||
      v->kind == Value::Kind::Tuple || v->kind == Value::Kind::Opaque) {
    if (!visited.insert(v.get()).second) return;
  }
  auto pairs = [&](const std::vector<std::pair<ValuePtr, ValuePtr>>& ps) {
    for (const auto& kv : ps) {
      std::string key = dict_key(kv.first);
      std::string child = prefix.empty() ? key : prefix + "." + key;
      walk_tensors(kv.second, child, visited, depth + 1, fn);
    }
  };
  switch (v->kind) {
    case Value::Kind::Tensor:
      fn(prefix, *v);
      break;
    case Value::Kind::Dict:
      pairs(v->pairs);
      break;
    case Value::Kind::List:
    case Value::Kind::Tuple:
      for (size_t k = 0; k < v->items.size(); ++k) {
        std::string child =
            prefix.empty() ? std::to_string(k) : prefix + "." + std::to_string(k);
        walk_tensors(v->items[k], child, visited, depth + 1, fn);
      }
      break;
    case Value::Kind::Opaque:
      // A TorchScript object: its recorded state dict lists the attributes in
      // slot order; descend exactly like a Dict (#137). Any other Opaque is an
      // inert placeholder and is not descended.
      if (is_torch_object(*v)) pairs(v->inner->pairs);
      break;
    default:
      break;
  }
}

}  // namespace

void for_each_tensor(
    const ValuePtr& root,
    const std::function<void(const std::string& path, const Value& tensor)>& fn) {
  std::unordered_set<const Value*> visited;
  walk_tensors(root, "", visited, 0, fn);
}

// ---- constant repr -----------------------------------------------------------
namespace {

// Python repr() of a finite double: the shortest round-tripping digits,
// positional for exponents in [-4, 16), scientific otherwise.
std::string py_float(double d) {
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d < 0 ? "-inf" : "inf";
  char buf[64];
  int prec = 1;
  for (; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof(buf), "%.*e", prec - 1, d);
    if (std::strtod(buf, nullptr) == d) break;
  }
  if (prec > 17) std::snprintf(buf, sizeof(buf), "%.16e", d);
  std::string s(buf);
  bool neg = !s.empty() && s[0] == '-';
  if (neg) s.erase(0, 1);
  size_t epos = s.find('e');
  if (epos == std::string::npos) return std::string(neg ? "-" : "") + s;
  std::string mant = s.substr(0, epos);
  int exp = std::atoi(s.c_str() + epos + 1);
  std::string digits;
  for (char c : mant)
    if (c != '.') digits.push_back(c);
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  std::string out;
  if (exp >= -4 && exp < 16) {
    if (exp >= 0) {
      std::string ip, fp;
      for (size_t i = 0; i < digits.size(); ++i)
        (static_cast<int>(i) <= exp ? ip : fp).push_back(digits[i]);
      while (static_cast<int>(ip.size()) < exp + 1) ip.push_back('0');
      out = ip + "." + (fp.empty() ? "0" : fp);
    } else {
      out = "0." + std::string(static_cast<size_t>(-exp - 1), '0') + digits;
    }
  } else {
    out = digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    char eb[16];
    std::snprintf(eb, sizeof(eb), "e%c%02d", exp < 0 ? '-' : '+',
                  exp < 0 ? -exp : exp);
    out += eb;
  }
  return std::string(neg ? "-" : "") + out;
}

void quote_into(std::string& out, const std::string& s) {
  static const char* kHex = "0123456789abcdef";
  out.push_back('\'');
  for (unsigned char c : s) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '\'') {
      out += "\\'";
    } else if (c < 0x20 || c > 0x7e) {
      out += "\\x";
      out.push_back(kHex[c >> 4]);
      out.push_back(kHex[c & 0xF]);
    } else {
      out.push_back(static_cast<char>(c));
    }
  }
  out.push_back('\'');
}

std::string qualified(const Value& v) {
  if (v.module.empty()) return v.name;
  if (v.name.empty()) return v.module;
  return v.module + "." + v.name;
}

std::string tensor_label(const Value& v) {
  std::string out = "<tensor ";
  out += v.dtype_label.empty() ? ir::dtype_name(v.tensor.dtype) : v.dtype_label;
  out += "[";
  for (size_t i = 0; i < v.tensor.shape.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(v.tensor.shape[i]);
  }
  out += "]>";
  return out;
}

struct ReprWriter {
  const ConstLimits& lim;
  std::string out;
  std::vector<const Value*> on_path;  // containers being rendered (cycles)

  bool full() const { return out.size() > lim.max_repr; }
  bool cyclic(const Value* v) const {
    for (const Value* p : on_path)
      if (p == v) return true;
    return false;
  }

  void seq(const std::vector<ValuePtr>& items, const char* open,
           const char* close, bool tuple, uint32_t depth) {
    if (items.size() > lim.max_list) {
      out += open;
      out += "\xE2\x80\xA6 " + std::to_string(items.size()) + " items";
      out += close;
      return;
    }
    out += open;
    for (size_t i = 0; i < items.size() && !full(); ++i) {
      if (i) out += ", ";
      write(items[i].get(), depth + 1);
    }
    if (tuple && items.size() == 1) out += ",";
    out += close;
  }

  void write(const Value* v, uint32_t depth) {
    if (full()) return;
    if (!v) {
      out += "None";
      return;
    }
    if (depth > lim.max_depth) {
      out += "...";
      return;
    }
    switch (v->kind) {
      case Value::Kind::None: out += "None"; return;
      case Value::Kind::Bool: out += v->b ? "True" : "False"; return;
      case Value::Kind::Int: out += std::to_string(v->i); return;
      case Value::Kind::Double: out += py_float(v->d); return;
      case Value::Kind::Str: quote_into(out, v->s); return;
      case Value::Kind::Bytes:
        out += "b";
        quote_into(out, v->s);
        return;
      case Value::Kind::Tensor: out += tensor_label(*v); return;
      case Value::Kind::Persistent: out += "<persistent>"; return;
      case Value::Kind::Global: out += "<" + qualified(*v) + ">"; return;
      default: break;
    }
    // Containers and Opaque-with-args recurse: cycle-check first.
    if (cyclic(v)) {
      out += "...";
      return;
    }
    on_path.push_back(v);
    switch (v->kind) {
      case Value::Kind::List: seq(v->items, "[", "]", false, depth); break;
      case Value::Kind::Tuple: seq(v->items, "(", ")", true, depth); break;
      case Value::Kind::Dict:
        if (v->pairs.size() > lim.max_list) {
          out += "{\xE2\x80\xA6 " + std::to_string(v->pairs.size()) + " items}";
          break;
        }
        out += "{";
        for (size_t i = 0; i < v->pairs.size() && !full(); ++i) {
          if (i) out += ", ";
          write(v->pairs[i].first.get(), depth + 1);
          out += ": ";
          write(v->pairs[i].second.get(), depth + 1);
        }
        out += "}";
        break;
      case Value::Kind::Opaque:
        if (v->items.empty()) {
          out += "<" + qualified(*v) + ">";
        } else {
          out += qualified(*v);
          seq(v->items, "(", ")", false, depth);
        }
        break;
      default:
        break;
    }
    on_path.pop_back();
  }
};

}  // namespace

std::string constant_repr(const Value& v, const ConstLimits& lim) {
  ReprWriter w{lim, {}, {}};
  w.write(&v, 0);
  if (w.out.size() > lim.max_repr) {
    size_t cut = lim.max_repr;
    // Never split a UTF-8 sequence (our own "…" markers are the only non-ASCII).
    while (cut > 0 && (static_cast<unsigned char>(w.out[cut]) & 0xC0) == 0x80)
      --cut;
    w.out.resize(cut);
    w.out += "\xE2\x80\xA6";
  }
  return w.out;
}

bool constant_to_attr(const Value& v, ir::Model& m, ir::AttrValue& out,
                      const ConstLimits& lim) {
  using K = ir::AttrValue::Kind;
  ir::AttrValue a;
  auto as_string = [&](std::string_view s) {
    a.kind = K::String;
    a.s = m.intern(s);
  };
  switch (v.kind) {
    case Value::Kind::Tensor:
      return false;
    case Value::Kind::Int:
      a.kind = K::Int;
      a.i = v.i;
      break;
    case Value::Kind::Double:
      a.kind = K::Float;
      a.f = v.d;
      break;
    case Value::Kind::List:
    case Value::Kind::Tuple: {
      bool all_int = !v.items.empty(), all_dbl = !v.items.empty();
      for (const auto& it : v.items) {
        all_int = all_int && it && it->kind == Value::Kind::Int;
        all_dbl = all_dbl && it && it->kind == Value::Kind::Double;
      }
      if (v.items.size() <= lim.max_list && all_int) {
        a.kind = K::Ints;
        for (const auto& it : v.items) a.ints.push_back(it->i);
      } else if (v.items.size() <= lim.max_list && all_dbl) {
        a.kind = K::Floats;
        for (const auto& it : v.items) a.floats.push_back(it->d);
      } else {
        // Empty (element type unknowable), mixed, nested or over-long lists.
        as_string(constant_repr(v, lim));
      }
      break;
    }
    default:
      // Bool -> True/False, None, 'str', b'bytes', dicts, Opaque/Global reprs.
      as_string(constant_repr(v, lim));
      break;
  }
  out = std::move(a);
  return true;
}

// ---- ModuleEnv ---------------------------------------------------------------
namespace {

ModuleEnv::Attr classify(const std::string& name, const Value* value) {
  using K = ModuleEnv::Attr::Kind;
  ModuleEnv::Attr a;
  a.value = value;
  a.name = name;
  if (!value) {
    a.kind = K::Unknown;
    a.name.clear();
  } else if (value->kind == Value::Kind::Tensor) {
    a.kind = K::Tensor;
  } else if (is_torch_object(*value)) {
    a.kind = K::Object;
  } else if (value->kind == Value::Kind::Opaque &&
             value->module.rfind("__torch__", 0) == 0) {
    // A TorchScript object WITHOUT dict state (custom __getstate__, e.g.
    // quantized packed params): its contents are not attributes we can bind.
    a.kind = K::Opaque;
  } else {
    // None/Bool/Int/Double/Str/Bytes, containers of those, and non-object
    // Opaque values with recorded args (torch.device('cpu')): fold as data.
    a.kind = K::Constant;
  }
  return a;
}

}  // namespace

ModuleEnv::ModuleEnv(const ValuePtr& data_pkl_root)
    : keep_(data_pkl_root), loaded_(data_pkl_root != nullptr) {
  if (!data_pkl_root || !is_torch_object(*data_pkl_root)) return;
  root_ = data_pkl_root.get();
  root_class_ = qualified(*data_pkl_root);
  for_each_tensor(data_pkl_root, [this](const std::string& path, const Value& t) {
    paths_.emplace(&t, path);  // first path wins
  });
}

ModuleEnv::Attr ModuleEnv::by_slot(const Value* obj, int64_t slot) const {
  if (!obj || !is_torch_object(*obj)) return {};
  const auto& pairs = obj->inner->pairs;
  if (slot < 0 || static_cast<uint64_t>(slot) >= pairs.size()) return {};
  const auto& kv = pairs[static_cast<size_t>(slot)];
  if (!kv.first || kv.first->kind != Value::Kind::Str) return {};
  return classify(kv.first->s, kv.second.get());
}

ModuleEnv::Attr ModuleEnv::by_name(const Value* obj, std::string_view name) const {
  if (!obj || !is_torch_object(*obj)) return {};
  for (const auto& kv : obj->inner->pairs) {
    if (kv.first && kv.first->kind == Value::Kind::Str && kv.first->s == name)
      return classify(kv.first->s, kv.second.get());
  }
  return {};
}

std::string ModuleEnv::tensor_path(const Value* tensor,
                                   std::string_view fallback) const {
  auto it = paths_.find(tensor);
  if (it == paths_.end()) return std::string(fallback);
  return it->second;
}

// ---- TensorConstNames --------------------------------------------------------
void TensorConstNames::add_from_constants_pkl(const ValuePtr& tuple) {
  if (!tuple || tuple->kind != Value::Kind::Tuple) return;
  for (size_t i = 0; i < tuple->items.size(); ++i) {
    const ValuePtr& v = tuple->items[i];
    if (!v || v->kind != Value::Kind::Tensor) continue;
    if (v->tensor.file_offset == UINT64_MAX) continue;  // no record to share
    by_offset_[v->tensor.file_offset].push_back(Ent{v->tensor, i});
  }
}

std::string TensorConstNames::name_for(const ir::TensorRef& t,
                                       std::string_view fallback) const {
  if (t.file_offset != UINT64_MAX) {
    auto it = by_offset_.find(t.file_offset);
    if (it != by_offset_.end()) {
      for (const Ent& e : it->second) {  // insertion (= index) order
        if (e.ref.byte_len == t.byte_len && e.ref.dtype == t.dtype &&
            e.ref.shape == t.shape)
          return "CONSTANTS.c" + std::to_string(e.index);
      }
    }
  }
  return std::string(fallback);
}

// ---- NameCounter -------------------------------------------------------------
std::string NameCounter::next() { return "%" + std::to_string(++n_); }

// ---- GraphBuilder ------------------------------------------------------------
GraphBuilder::GraphBuilder(ir::Model& m, NameCounter& names,
                           std::string_view graph_name, uint32_t id,
                           GraphBuilder* parent)
    : m_(m), names_(names), name_(graph_name), id_(id), parent_(parent) {}

std::string GraphBuilder::unique_name(std::string_view base) {
  std::string b(base);
  if (by_name_.find(b) == by_name_.end()) return b;
  uint32_t& k = next_suffix_[b];
  if (k < 2) k = 2;
  for (;; ++k) {
    std::string c = b + "#" + std::to_string(k);
    if (by_name_.find(c) == by_name_.end()) {
      ++k;
      return c;
    }
  }
}

uint32_t GraphBuilder::new_value(std::string_view name) {
  std::string n = name.empty() ? unique_name(names_.next()) : unique_name(name);
  ir::ValueInfo vi;
  vi.name = m_.intern(n);
  auto idx = static_cast<uint32_t>(values_.size());
  values_.push_back(std::move(vi));
  by_name_.emplace(std::move(n), idx);
  return idx;
}

uint32_t GraphBuilder::add_input(std::string_view name) {
  uint32_t v = new_value(name);  // empty -> NameCounter
  inputs_.push_back(v);
  return v;
}

uint32_t GraphBuilder::initializer(std::string_view name, const ir::TensorRef& t) {
  std::string base = name.empty() ? names_.next() : std::string(name);
  for (uint32_t k = 1;; ++k) {
    std::string cand = k == 1 ? base : base + "#" + std::to_string(k);
    auto it = by_name_.find(cand);
    if (it == by_name_.end()) {
      ir::ValueInfo vi;
      vi.name = m_.intern(cand);
      vi.dtype = t.dtype;
      vi.shape = t.shape;
      auto idx = static_cast<uint32_t>(values_.size());
      values_.push_back(std::move(vi));
      by_name_.emplace(cand, idx);
      ir::TensorRef r = t;
      r.name = values_[idx].name;
      init_of_value_.emplace(idx, static_cast<uint32_t>(inits_.size()));
      inits_.push_back(std::move(r));
      return idx;
    }
    auto ii = init_of_value_.find(it->second);
    if (ii != init_of_value_.end() &&
        inits_[ii->second].file_offset == t.file_offset &&
        inits_[ii->second].byte_len == t.byte_len)
      return it->second;
  }
}

uint32_t GraphBuilder::capture(uint32_t parent_value_idx, std::string_view name) {
  auto it = capture_of_.find(parent_value_idx);
  if (it != capture_of_.end()) return it->second;
  uint32_t v = new_value(name);
  inputs_.push_back(v);
  captured_.push_back(parent_value_idx);
  capture_of_.emplace(parent_value_idx, v);
  return v;
}

uint32_t GraphBuilder::node(std::string_view op_type) {
  Staged s;
  s.op_type = m_.intern(op_type);
  nodes_.push_back(std::move(s));
  return static_cast<uint32_t>(nodes_.size() - 1);
}

void GraphBuilder::add_edge(uint32_t node, uint32_t value) {
  if (node < nodes_.size() && value < values_.size())
    nodes_[node].in.push_back(value);
}

void GraphBuilder::add_attr(uint32_t node, ir::Attribute a) {
  if (node < nodes_.size()) nodes_[node].attrs.push_back(std::move(a));
}

uint32_t GraphBuilder::add_output(uint32_t node, std::string_view name) {
  uint32_t v = new_value(name);  // empty -> NameCounter
  if (node < nodes_.size()) {
    values_[v].producer = static_cast<int32_t>(node);
    nodes_[node].out.push_back(v);
  }
  return v;
}

void GraphBuilder::set_subgraph(uint32_t node, int32_t graph_index) {
  if (node < nodes_.size()) nodes_[node].subgraph = graph_index;
}

void GraphBuilder::add_output_value(uint32_t value) {
  if (value < values_.size()) outputs_.push_back(value);
}

void GraphBuilder::prepend_input(uint32_t value) {
  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i] == value) {
      inputs_.erase(inputs_.begin() + static_cast<std::ptrdiff_t>(i));
      break;
    }
  }
  inputs_.insert(inputs_.begin(), value);
}

std::string_view GraphBuilder::value_name(uint32_t value) const {
  if (value >= values_.size()) return {};
  return m_.str(values_[value].name);
}

ir::Graph GraphBuilder::finish() {
  ir::Graph g;
  g.name = m_.intern(name_);
  g.nodes.reserve(nodes_.size());
  for (Staged& s : nodes_) {
    ir::Node n;
    n.op_type = s.op_type;
    n.inputs = {static_cast<uint32_t>(g.edge_refs.size()),
                static_cast<uint32_t>(s.in.size())};
    g.edge_refs.insert(g.edge_refs.end(), s.in.begin(), s.in.end());
    n.outputs = {static_cast<uint32_t>(g.edge_refs.size()),
                 static_cast<uint32_t>(s.out.size())};
    g.edge_refs.insert(g.edge_refs.end(), s.out.begin(), s.out.end());
    n.attributes = {static_cast<uint32_t>(g.attributes.size()),
                    static_cast<uint32_t>(s.attrs.size())};
    for (ir::Attribute& a : s.attrs) g.attributes.push_back(std::move(a));
    n.subgraph = s.subgraph;
    g.nodes.push_back(n);
  }
  g.values = std::move(values_);
  g.initializers = std::move(inits_);
  g.graph_inputs = inputs_;
  g.graph_outputs = outputs_;
  nodes_.clear();
  values_.clear();
  inits_.clear();
  return g;
}

}  // namespace netvis::pytorch::ts
