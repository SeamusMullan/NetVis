// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeOps.cpp — the Caffe -> IR op table (§3.8). See CaffeOps.h
// for the vetting rule. Every branch below either proves the ONNX op's shape
// semantics are identical under the derived attributes, or falls back to the
// layer's own (collision-safe) name.
#include "parsers/caffe/CaffeOps.h"

#include <iterator>
#include <optional>
#include <utility>

#include "core/SafeMath.h"
#include "engine/OpCategory.h"

namespace netvis::caffe {

const BvlcLayerType kBvlcLayerTypes[] = {
    {"Input", nullptr, 0, 1},
    {"Convolution", "Convolution", 1, 2},
    {"Deconvolution", "Deconvolution", 1, 2},
    {"InnerProduct", "InnerProduct", 1, 2},
    {"Pooling", "Pooling", 2, 1},
    {"ReLU", "CaffeReLU", 1, 2},
    {"PReLU", "CaffePReLU", 1, 2},
    {"ELU", "CaffeELU", 1, 2},
    {"Sigmoid", "CaffeSigmoid", 1, 2},
    {"TanH", "CaffeTanH", 1, 2},
    {"AbsVal", "AbsVal", 1, 2},
    {"BNLL", "BNLL", 1, 2},
    {"Exp", "CaffeExp", 1, 2},
    {"Log", "CaffeLog", 1, 2},
    {"Clip", "CaffeClip", 1, 2},
    {"Swish", "CaffeSwish", 1, 2},
    {"Softmax", "CaffeSoftmax", 1, 2},
    {"Concat", "CaffeConcat", 1, 2},
    {"Eltwise", "Eltwise", 1, 1},
    {"Flatten", "CaffeFlatten", 1, 2},
    {"Slice", "CaffeSlice", 2, 2},
    {"BatchNorm", "CaffeBatchNorm", 1, 2},
    {"LRN", "CaffeLRN", 1, 2},
    {"MVN", "MVN", 1, 2},
    {"Dropout", "CaffeDropout", 1, 2},
    // Never vetted: the ONNX namesakes have different inputs or semantics (a
    // Caffe Split COPIES its input; an ONNX Split partitions it).
    {"Split", "CaffeSplit", 1, 2},
    {"Reshape", "CaffeReshape", 1, 1},
    {"Tile", "CaffeTile", 1, 1},
    {"ArgMax", "CaffeArgMax", 1, 1},
    {"LSTM", "CaffeLSTM", 2, 1},
    {"RNN", "CaffeRNN", 2, 1},
    {"Parameter", "CaffeParameter", 0, 1},
    // Never vetted, no ONNX namesake to collide with: verbatim.
    {"Scale", "Scale", 1, 1},
    {"Bias", "Bias", 1, 1},
    {"Power", "Power", 1, 1},
    {"Threshold", "Threshold", 1, 1},
    {"Crop", "Crop", 2, 1},
    {"Embed", "Embed", 1, 1},
    {"Reduction", "Reduction", 1, 1},
    {"SPP", "SPP", 1, 1},
    {"Im2col", "Im2col", 1, 1},
    {"BatchReindex", "BatchReindex", 2, 1},
    {"Filter", "Filter", 2, 1},
    {"Silence", "Silence", 1, 0},
    {"Recurrent", "Recurrent", 2, 1},
    {"Python", "Python", 1, 1},
    {"Accuracy", "Accuracy", 2, 1},
    {"EuclideanLoss", "EuclideanLoss", 2, 1},
    {"InfogainLoss", "InfogainLoss", 2, 1},
    {"MultinomialLogisticLoss", "MultinomialLogisticLoss", 2, 1},
    {"SoftmaxWithLoss", "SoftmaxWithLoss", 2, 1},
    {"SigmoidCrossEntropyLoss", "SigmoidCrossEntropyLoss", 2, 1},
    {"HingeLoss", "HingeLoss", 2, 1},
    {"ContrastiveLoss", "ContrastiveLoss", 3, 1},
    {"Data", "Data", 0, 2},
    {"ImageData", "ImageData", 0, 2},
    {"MemoryData", "MemoryData", 0, 2},
    {"HDF5Data", "HDF5Data", 0, 2},
    {"HDF5Output", "HDF5Output", 2, 0},
    {"WindowData", "WindowData", 0, 2},
    {"DummyData", "DummyData", 0, 1},
};
const size_t kBvlcLayerTypeCount = std::size(kBvlcLayerTypes);

namespace {

const BvlcLayerType* bvlc_row(std::string_view type) {
  for (const BvlcLayerType& r : kBvlcLayerTypes)
    if (type == r.type) return &r;
  return nullptr;
}

// ---- native attribute accessors --------------------------------------------------

std::optional<int64_t> get_int(const LayerRaw& L, const ir::Model& m, std::string_view n) {
  const ir::AttrValue* a = layer_attr(L, m, n);
  if (a && a->kind == ir::AttrValue::Kind::Int) return a->i;
  return std::nullopt;
}

std::optional<double> get_float(const LayerRaw& L, const ir::Model& m, std::string_view n) {
  const ir::AttrValue* a = layer_attr(L, m, n);
  if (a && a->kind == ir::AttrValue::Kind::Float) return a->f;
  return std::nullopt;
}

std::vector<int64_t> get_ints(const LayerRaw& L, const ir::Model& m, std::string_view n) {
  const ir::AttrValue* a = layer_attr(L, m, n);
  if (a && a->kind == ir::AttrValue::Kind::Ints) return a->ints;
  return {};
}

std::optional<std::string_view> get_str(const LayerRaw& L, const ir::Model& m,
                                        std::string_view n) {
  const ir::AttrValue* a = layer_attr(L, m, n);
  if (a && a->kind == ir::AttrValue::Kind::String) return m.str(a->s);
  return std::nullopt;
}

bool in_bounds(int64_t v, int64_t lo) { return v >= lo && v <= kMaxGeometry; }

DerivedAttr d_int(const char* name, int64_t v) {
  DerivedAttr d;
  d.name = name;
  d.kind = ir::AttrValue::Kind::Int;
  d.i = v;
  return d;
}
DerivedAttr d_ints(const char* name, std::vector<int64_t> v) {
  DerivedAttr d;
  d.name = name;
  d.kind = ir::AttrValue::Kind::Ints;
  d.ints = std::move(v);
  return d;
}
DerivedAttr d_float(const char* name, double v) {
  DerivedAttr d;
  d.name = name;
  d.kind = ir::AttrValue::Kind::Float;
  d.f = v;
  return d;
}

MappedOp vetted(const char* op) {
  MappedOp m;
  m.op_type = op;
  m.vetted = true;
  return m;
}

MappedOp fallback(std::string_view type) {
  MappedOp m;
  const BvlcLayerType* r = bvlc_row(type);
  m.op_type = (r && r->fallback) ? std::string(r->fallback) : safe_display_name(type);
  m.vetted = false;
  return m;
}

// Resolve a Caffe "list or _h/_w pair" geometry field to R values. `def` fills
// an empty list. Returns false when unresolvable.
bool resolve_list(const std::vector<int64_t>& list, std::optional<int64_t> h,
                  std::optional<int64_t> w, int64_t R, int64_t def,
                  std::vector<int64_t>& out) {
  out.clear();
  if (h || w) {
    if (!(h && w) || R != 2 || !list.empty()) return false;
    out = {*h, *w};
    return true;
  }
  if (list.empty()) {
    out.assign(static_cast<size_t>(R), def);
    return true;
  }
  if (list.size() == 1) {
    out.assign(static_cast<size_t>(R), list[0]);
    return true;
  }
  if (static_cast<int64_t>(list.size()) == R) {
    out = list;
    return true;
  }
  return false;
}

// A pooling geometry field: exactly one of `v` or the (h, w) pair.
bool pool_pair(std::optional<int64_t> v, std::optional<int64_t> h, std::optional<int64_t> w,
               std::optional<int64_t> def, int64_t& oh, int64_t& ow) {
  if (h || w) {
    if (!(h && w) || v) return false;
    oh = *h;
    ow = *w;
    return true;
  }
  if (v) {
    oh = ow = *v;
    return true;
  }
  if (!def) return false;
  oh = ow = *def;
  return true;
}

MappedOp map_conv(const LayerRaw& L, const ir::Model& m, int weight_rank, bool transpose) {
  const char* op = transpose ? "ConvTranspose" : "Conv";
  if (L.bottom.size() != 1 || L.top.size() != 1) return fallback(L.type);
  if (auto ax = get_int(L, m, "convolution_param.axis"); ax && *ax != 1) return fallback(L.type);
  ConvGeometry g = resolve_conv_geometry(L, m, weight_rank);
  if (!g.ok) return fallback(L.type);
  MappedOp out = vetted(op);
  if (!g.determined) return out;   // no weight, no rank: harmless (nothing can fire)
  if (!g.kernel.empty()) out.derived.push_back(d_ints("kernel_shape", g.kernel));
  out.derived.push_back(d_ints("strides", g.strides));
  out.derived.push_back(d_ints("pads", g.pads));
  out.derived.push_back(d_ints("dilations", g.dilations));
  out.derived.push_back(d_int("group", g.group));
  return out;
}

MappedOp map_pool(const LayerRaw& L, const ir::Model& m) {
  const std::string_view pool = get_str(L, m, "pooling_param.pool").value_or("MAX");
  const bool is_max = pool == "MAX";
  if (!is_max && pool != "AVE") return fallback(L.type);
  if (L.bottom.size() != 1) return fallback(L.type);
  if (!(L.top.size() == 1 || (is_max && L.top.size() == 2))) return fallback(L.type);

  const auto ks = get_int(L, m, "pooling_param.kernel_size");
  const auto kh = get_int(L, m, "pooling_param.kernel_h");
  const auto kw = get_int(L, m, "pooling_param.kernel_w");
  const auto st = get_int(L, m, "pooling_param.stride");
  const auto sh = get_int(L, m, "pooling_param.stride_h");
  const auto sw = get_int(L, m, "pooling_param.stride_w");
  const auto pd = get_int(L, m, "pooling_param.pad");
  const auto ph = get_int(L, m, "pooling_param.pad_h");
  const auto pw = get_int(L, m, "pooling_param.pad_w");
  int64_t s_h = 1, s_w = 1, p_h = 0, p_w = 0;
  if (!pool_pair(st, sh, sw, int64_t{1}, s_h, s_w)) return fallback(L.type);
  if (!pool_pair(pd, ph, pw, int64_t{0}, p_h, p_w)) return fallback(L.type);

  if (get_int(L, m, "pooling_param.global_pooling").value_or(0) != 0) {
    // Caffe refuses a global pool with a kernel, padding or a stride != 1.
    if (ks || kh || kw || p_h != 0 || p_w != 0 || s_h != 1 || s_w != 1)
      return fallback(L.type);
    return vetted(is_max ? "GlobalMaxPool" : "GlobalAveragePool");
  }

  int64_t k_h = 0, k_w = 0;
  if (!pool_pair(ks, kh, kw, std::nullopt, k_h, k_w)) return fallback(L.type);
  if (!in_bounds(k_h, 1) || !in_bounds(k_w, 1) || !in_bounds(s_h, 1) || !in_bounds(s_w, 1) ||
      !in_bounds(p_h, 0) || !in_bounds(p_w, 0) || p_h >= k_h || p_w >= k_w)
    return fallback(L.type);

  int64_t ceil_mode = 1;   // Caffe's default RoundMode is CEIL
  if (auto rm = get_str(L, m, "pooling_param.round_mode")) {
    if (*rm == "FLOOR") ceil_mode = 0;
    else if (*rm != "CEIL") return fallback(L.type);
  }
  MappedOp out = vetted(is_max ? "MaxPool" : "AveragePool");
  out.derived.push_back(d_ints("kernel_shape", {k_h, k_w}));
  out.derived.push_back(d_ints("strides", {s_h, s_w}));
  out.derived.push_back(d_ints("pads", {p_h, p_w, p_h, p_w}));
  out.derived.push_back(d_int("ceil_mode", ceil_mode));
  return out;
}

MappedOp map_slice(const LayerRaw& L, const ir::Model& m) {
  if (L.bottom.size() != 1 || L.top.empty()) return fallback(L.type);
  const std::vector<int64_t> sp = get_ints(L, m, "slice_param.slice_point");
  int64_t axis = 1;
  if (auto a = get_int(L, m, "slice_param.axis")) axis = *a;
  else if (auto sd = get_int(L, m, "slice_param.slice_dim")) axis = *sd;
  std::vector<int64_t> split;
  if (!sp.empty()) {
    if (sp.size() != L.top.size() - 1) return fallback(L.type);
    int64_t prev = 0;
    for (int64_t p : sp) {
      int64_t part = 0;
      if (p <= prev || !checked_sub_i64(p, prev, &part)) return fallback(L.type);
      split.push_back(part);
      prev = p;
    }
  }
  MappedOp out = vetted("Split");
  out.derived.push_back(d_int("axis", axis));
  // tops-1 entries: the last output's size along the axis is honestly unknown
  // until the input's is (the engine writes -1 for it).
  if (!split.empty()) out.derived.push_back(d_ints("split", std::move(split)));
  return out;
}

}  // namespace

const ir::AttrValue* layer_attr(const LayerRaw& L, const ir::Model& m, std::string_view n) {
  for (const ir::Attribute& a : L.attrs)
    if (m.str(a.name) == n) return &a.value;
  return nullptr;
}

std::string safe_display_name(std::string_view type) {
  // OpCategory and CostModel normalise to the LAST dot segment, so a dotted fork
  // type could still reach an engine handler through its suffix: neutralise it.
  if (type.find('.') != std::string_view::npos) {
    std::string s = "Caffe";
    for (char c : type) s.push_back(c == '.' ? '_' : c);
    return s;
  }
  // Every CostModel explicit handler key is an OpCategory key, and
  // ShapeInferenceExt matches exact ONNX names: these four it handles without an
  // OpCategory entry, and its Reduce branch matches any "Reduce*" prefix.
  if (categorize_op(type) != OpCategory::Other || type == "Identity" || type == "Dropout" ||
      type == "LpPool" || type == "Resize" || type.substr(0, 6) == "Reduce")
    return "Caffe" + std::string(type);
  return std::string(type);
}

ConvGeometry resolve_conv_geometry(const LayerRaw& L, const ir::Model& m, int weight_rank) {
  ConvGeometry g;
  auto P = [](const char* f) { return std::string("convolution_param.") + f; };
  if (auto no = get_int(L, m, P("num_output"))) {
    g.has_num_output = true;
    g.num_output = *no;
    if (!in_bounds(*no, 0)) g.ok = false;
  }
  g.bias_term = get_int(L, m, P("bias_term")).value_or(1) != 0;
  const auto kh = get_int(L, m, P("kernel_h"));
  const auto kw = get_int(L, m, P("kernel_w"));
  const std::vector<int64_t> ks = get_ints(L, m, P("kernel_size"));
  g.group = get_int(L, m, P("group")).value_or(1);
  if (!in_bounds(g.group, 1)) g.ok = false;

  int64_t R = -1;
  if (weight_rank >= 0) {
    R = static_cast<int64_t>(weight_rank) - 2;
    if (R < 1) {
      g.determined = true;
      g.ok = false;
      return g;
    }
  } else if (kh || kw) {
    R = 2;
  } else if (ks.size() >= 2) {
    R = static_cast<int64_t>(ks.size());
  } else {
    return g;   // undetermined: no geometry attributes at all
  }
  g.determined = true;
  g.rank = R;
  if (R > static_cast<int64_t>(kMaxBlobAxes)) {
    g.ok = false;
    return g;
  }

  // kernel: kernel_h + kernel_w (R == 2), else kernel_size of length 1 (repeated),
  // R (as is) or 0 (omitted: the engine reads the weight's spatial dims).
  if (kh || kw) {
    if (!(kh && kw) || R != 2 || !ks.empty()) g.ok = false;
    else g.kernel = {*kh, *kw};
  } else if (ks.size() == 1) {
    g.kernel.assign(static_cast<size_t>(R), ks[0]);
  } else if (static_cast<int64_t>(ks.size()) == R) {
    g.kernel = ks;
  } else if (!ks.empty()) {
    g.ok = false;
  }
  for (int64_t k : g.kernel)
    if (!in_bounds(k, 1)) g.ok = false;

  if (!resolve_list(get_ints(L, m, P("stride")), get_int(L, m, P("stride_h")),
                    get_int(L, m, P("stride_w")), R, 1, g.strides))
    g.ok = false;
  std::vector<int64_t> pad;
  if (!resolve_list(get_ints(L, m, P("pad")), get_int(L, m, P("pad_h")),
                    get_int(L, m, P("pad_w")), R, 0, pad))
    g.ok = false;
  if (!resolve_list(get_ints(L, m, P("dilation")), std::nullopt, std::nullopt, R, 1,
                    g.dilations))
    g.ok = false;
  for (int64_t s : g.strides)
    if (!in_bounds(s, 1)) g.ok = false;
  for (int64_t p : pad)
    if (!in_bounds(p, 0)) g.ok = false;
  for (int64_t d : g.dilations)
    if (!in_bounds(d, 1)) g.ok = false;
  g.pads = pad;
  g.pads.insert(g.pads.end(), pad.begin(), pad.end());
  return g;
}

std::string blob_role(const LayerRaw& L, size_t index) {
  static const char* const kWB[] = {"weight", "bias"};
  static const char* const kBN[] = {"mean", "variance", "scale_factor"};
  static const char* const kSc1[] = {"scale", "bias"};
  static const char* const kBias[] = {"bias"};
  static const char* const kSlope[] = {"slope"};
  const char* const* roles = nullptr;
  size_t n = 0;
  const std::string& t = L.type;
  if (t == "Convolution" || t == "Deconvolution" || t == "InnerProduct" || t == "Embed") {
    roles = kWB;
    n = 2;
  } else if (t == "BatchNorm") {
    roles = kBN;
    n = 3;
  } else if (t == "Scale") {
    if (L.bottom.size() >= 2) {
      roles = kBias;
      n = 1;
    } else {
      roles = kSc1;
      n = 2;
    }
  } else if (t == "Bias" && L.bottom.size() == 1) {
    roles = kBias;
    n = 1;
  } else if (t == "PReLU") {
    roles = kSlope;
    n = 1;
  }
  if (index < n) return roles[index];
  return "blob" + std::to_string(index);
}

int expected_blob_count(const LayerRaw& L, const ir::Model& m) {
  const std::string& t = L.type;
  if (t == "Convolution" || t == "Deconvolution")
    return 1 + (get_int(L, m, "convolution_param.bias_term").value_or(1) != 0 ? 1 : 0);
  if (t == "InnerProduct")
    return 1 + (get_int(L, m, "inner_product_param.bias_term").value_or(1) != 0 ? 1 : 0);
  if (t == "Embed")
    return 1 + (get_int(L, m, "embed_param.bias_term").value_or(1) != 0 ? 1 : 0);
  if (t == "BatchNorm") return 3;
  if (t == "Scale") {
    const bool bias = get_int(L, m, "scale_param.bias_term").value_or(0) != 0;
    if (L.bottom.size() == 1) return 1 + (bias ? 1 : 0);
    if (L.bottom.size() == 2) return bias ? 1 : 0;
    return -1;
  }
  if (t == "Bias" && L.bottom.size() == 1) return 1;
  if (t == "PReLU") return 1;
  return -1;
}

MappedOp map_layer(const LayerRaw& L, const ir::Model& m, int weight_rank) {
  // Unresolved types: never guessed (TFLite's BUILTIN_<n> precedent).
  if (L.v1 && L.type.empty() && !L.v1_type_ident.empty()) {
    MappedOp out;
    out.op_type = "V1TYPE_" + L.v1_type_ident;
    return out;
  }
  if (!L.has_type || L.type.empty()) {
    MappedOp out;
    out.op_type = "UNTYPED";
    return out;
  }
  if (L.v0) {   // V0 types are lowercase legacy strings; shown, never mapped
    MappedOp out;
    out.op_type = safe_display_name(L.type);
    return out;
  }

  const std::string& t = L.type;
  const size_t nb = L.bottom.size(), nt = L.top.size();
  const bool one_one = nb == 1 && nt == 1;

  if (t == "Input") return vetted("Input");
  if (t == "Convolution") return map_conv(L, m, weight_rank, false);
  if (t == "Deconvolution") return map_conv(L, m, weight_rank, true);
  if (t == "InnerProduct") {
    if (!one_one) return fallback(t);
    if (auto ax = get_int(L, m, "inner_product_param.axis"); ax && *ax != 1) return fallback(t);
    MappedOp out = vetted("Gemm");
    const bool transpose = get_int(L, m, "inner_product_param.transpose").value_or(0) != 0;
    out.derived.push_back(d_int("transB", transpose ? 0 : 1));
    return out;
  }
  if (t == "Pooling") return map_pool(L, m);
  if (t == "ReLU") {
    if (!one_one) return fallback(t);
    const double slope = get_float(L, m, "relu_param.negative_slope").value_or(0.0);
    if (slope != 0.0) {
      MappedOp out = vetted("LeakyRelu");
      out.derived.push_back(d_float("alpha", slope));
      return out;
    }
    return vetted("Relu");
  }
  struct Simple {
    const char* caffe;
    const char* onnx;
  };
  static const Simple kSimple[] = {
      {"PReLU", "PRelu"},   {"ELU", "Elu"},     {"Sigmoid", "Sigmoid"},
      {"TanH", "Tanh"},     {"AbsVal", "Abs"},  {"BNLL", "Softplus"},   // BNLL == log(1+e^x)
      {"Exp", "Exp"},       {"Log", "Log"},     {"Clip", "Clip"},
      {"Swish", "Swish"},   {"BatchNorm", "BatchNormalization"},
      {"LRN", "LRN"},       {"MVN", "MeanVarianceNormalization"},
      {"Dropout", "Dropout"},
  };
  for (const Simple& s : kSimple) {
    if (t == s.caffe) return one_one ? vetted(s.onnx) : fallback(t);
  }
  if (t == "Softmax") {
    if (!one_one) return fallback(t);
    MappedOp out = vetted("Softmax");
    out.derived.push_back(d_int("axis", get_int(L, m, "softmax_param.axis").value_or(1)));
    return out;
  }
  if (t == "Concat") {
    if (nb < 1 || nt != 1) return fallback(t);
    int64_t axis = 1;
    if (auto a = get_int(L, m, "concat_param.axis")) axis = *a;
    else if (auto cd = get_int(L, m, "concat_param.concat_dim")) axis = *cd;
    MappedOp out = vetted("Concat");
    out.derived.push_back(d_int("axis", axis));
    return out;
  }
  if (t == "Eltwise") {
    if (nt != 1) return fallback(t);
    const std::string_view op = get_str(L, m, "eltwise_param.operation").value_or("SUM");
    if (op == "SUM" && nb == 2) return vetted("Add");
    if (op == "SUM" && nb > 2) return vetted("Sum");
    if (op == "PROD" && nb == 2) return vetted("Mul");
    if (op == "MAX" && nb >= 2) return vetted("Max");
    return fallback(t);
  }
  if (t == "Flatten") {
    if (!one_one) return fallback(t);
    if (auto e = get_int(L, m, "flatten_param.end_axis"); e && *e != -1) return fallback(t);
    // Caffe keeps the axes before `axis`; ONNX Flatten folds them into one. The
    // two agree only for axis == 1 (a negative axis depends on the input rank).
    if (auto a = get_int(L, m, "flatten_param.axis"); a && *a != 1) return fallback(t);
    MappedOp out = vetted("Flatten");
    out.derived.push_back(d_int("axis", 1));
    return out;
  }
  if (t == "Slice") return map_slice(L, m);
  if (bvlc_row(t)) return fallback(t);   // the never-vetted rows
  MappedOp out;
  out.op_type = safe_display_name(t);
  return out;
}

}  // namespace netvis::caffe
