// engine/ShapeInferenceExt.cpp — mmap-base-aware ONNX shape/dtype propagation.
//
// This is the real implementation of shape inference. The frozen 3-arg
// infer_shapes (ShapeInference.cpp) delegates here with base=nullptr, giving the
// pure structural subset. When `mmap_base != nullptr`, constant-driven ops
// (Reshape/Slice/Gather/Expand/Tile/Resize/Split/Squeeze/Unsqueeze/Pad/Constant
// with a shape/indices/axes/pads/repeats tensor) resolve by reading a raw_data
// initializer's bytes through a bounds-checked ByteReader over [base, base+size).
//
// THREADING: mutates a snapshot the main thread is NOT reading; the enriched
// model is republished on job completion (generation-checked). Best-effort:
// unknown ops / unknown inputs leave outputs Unknown and never fail or crash.
//
// LIMITATION: only initializers stored as raw_data (a contiguous mmap range) are
// readable. Shape tensors packed into ONNX int64_data/int32_data protobuf fields
// have no recorded offset (parser sets file_offset=UINT64_MAX) and stay Unknown.
#include "engine/ShapeInferenceExt.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

#include "core/ByteReader.h"

namespace netvis {

namespace {

using ir::DType;
using ir::Graph;
using ir::Model;
using ir::Node;
using Shape = SmallVec<int64_t, 6>;

bool shape_known(const ir::ValueInfo& v) { return !v.shape.empty(); }

// Resolve node input slot -> value index, or -1 if slot is absent/optional.
int32_t input_value(const Graph& g, const Node& n, uint32_t slot) {
  if (slot >= n.inputs.count) return -1;
  uint32_t ei = n.inputs.begin + slot;
  if (ei >= g.edge_refs.size()) return -1;
  uint32_t vi = g.edge_refs[ei];
  if (vi >= g.values.size()) return -1;
  return static_cast<int32_t>(vi);
}

int32_t output_value(const Graph& g, const Node& n, uint32_t slot) {
  if (slot >= n.outputs.count) return -1;
  uint32_t ei = n.outputs.begin + slot;
  if (ei >= g.edge_refs.size()) return -1;
  uint32_t vi = g.edge_refs[ei];
  if (vi >= g.values.size()) return -1;
  return static_cast<int32_t>(vi);
}

// Find an attribute on a node by name. Returns nullptr if absent.
const ir::AttrValue* attr(const Model& m, const Graph& g, const Node& n,
                          std::string_view name) {
  for (uint32_t i = 0; i < n.attributes.count; ++i) {
    uint32_t ai = n.attributes.begin + i;
    if (ai >= g.attributes.size()) break;
    if (m.str(g.attributes[ai].name) == name) return &g.attributes[ai].value;
  }
  return nullptr;
}

int64_t attr_int(const Model& m, const Graph& g, const Node& n,
                 std::string_view name, int64_t def) {
  const ir::AttrValue* a = attr(m, g, n, name);
  return (a && a->kind == ir::AttrValue::Kind::Int) ? a->i : def;
}

std::vector<int64_t> attr_ints(const Model& m, const Graph& g, const Node& n,
                               std::string_view name) {
  const ir::AttrValue* a = attr(m, g, n, name);
  if (a && a->kind == ir::AttrValue::Kind::Ints) return a->ints;
  return {};
}

// String attribute (e.g. RNN `direction`, recurrent `layout` when string-typed).
// Returns an empty view if absent or not a String attr.
std::string_view attr_string(const Model& m, const Graph& g, const Node& n,
                             std::string_view name) {
  const ir::AttrValue* a = attr(m, g, n, name);
  if (a && a->kind == ir::AttrValue::Kind::String) return m.str(a->s);
  return {};
}

// numpy broadcasting of two shapes (right-aligned). Returns nullopt if
// incompatible. -1 (dynamic) is treated as compatible-with-anything.
std::optional<Shape> broadcast(const Shape& a, const Shape& b) {
  size_t na = a.size(), nb = b.size();
  size_t nr = std::max(na, nb);
  Shape out;
  out.resize(nr);
  for (size_t i = 0; i < nr; ++i) {
    int64_t da = (i < nr - na) ? 1 : a[i - (nr - na)];
    int64_t db = (i < nr - nb) ? 1 : b[i - (nr - nb)];
    int64_t r;
    if (da < 0 || db < 0) {
      r = -1;
    } else if (da == 1) {
      r = db;
    } else if (db == 1) {
      r = da;
    } else if (da == db) {
      r = da;
    } else {
      return std::nullopt;  // incompatible static dims
    }
    out[i] = r;
  }
  return out;
}

// Set a value's shape/dtype if not already known; returns true if it became
// newly resolved (was unknown-shape before, now has a shape).
bool set_shape(ir::ValueInfo& v, const Shape& s, DType dt) {
  bool newly = false;
  if (!shape_known(v) && !s.empty()) {
    v.shape = s;
    newly = true;
  }
  if (v.dtype == DType::Unknown && dt != DType::Unknown) v.dtype = dt;
  return newly;
}

// Carry a dtype onto an output value without touching its shape.
void carry_dtype(ir::ValueInfo& v, DType dt) {
  if (v.dtype == DType::Unknown && dt != DType::Unknown) v.dtype = dt;
}

DType value_dtype(const Graph& g, int32_t vi) {
  if (vi < 0) return DType::Unknown;
  return g.values[static_cast<size_t>(vi)].dtype;
}

// Read a small 1-D int shape/index/axes initializer via the mmap. Only raw_data
// initializers are readable (offset recorded); int64_data/int32_data-packed
// tensors carry file_offset==UINT64_MAX and yield nullopt (documented limit).
// Capped at a tiny element count so we never touch real weights.
std::optional<std::vector<int64_t>> read_shape_initializer(
    const Model& m, const Graph& g, const ir::TensorRef& t, const uint8_t* base,
    uint64_t base_size) {
  if (t.external_path.valid()) return std::nullopt;  // don't fetch external
  if (t.file_offset == UINT64_MAX) return std::nullopt;
  if (base == nullptr) return std::nullopt;
  int64_t n = t.elem_count();
  if (n <= 0 || n > 64) return std::nullopt;  // must be tiny + fully static
  (void)m;
  (void)g;

  ByteReader r(base, base_size);
  if (!r.seek(t.file_offset)) return std::nullopt;
  std::vector<int64_t> dims;
  dims.reserve(static_cast<size_t>(n));
  if (t.dtype == DType::I64) {
    for (int64_t i = 0; i < n; ++i) {
      auto v = r.i64le();
      if (!v) return std::nullopt;
      dims.push_back(*v);
    }
  } else if (t.dtype == DType::I32) {
    for (int64_t i = 0; i < n; ++i) {
      auto v = r.i32le();
      if (!v) return std::nullopt;
      dims.push_back(*v);
    }
  } else {
    return std::nullopt;
  }
  return dims;
}

// Resolve an int vector from a node input slot: find the initializer matching
// the slot's value name and read it through the mmap. Returns nullopt when the
// slot is absent, has no matching initializer, or the payload isn't readable.
std::optional<std::vector<int64_t>> read_int_input(const Model& m, const Graph& g,
                                                   const Node& n, uint32_t slot,
                                                   const uint8_t* base,
                                                   uint64_t base_size) {
  int32_t vi = input_value(g, n, slot);
  if (vi < 0) return std::nullopt;
  std::string_view want = m.str(g.values[static_cast<size_t>(vi)].name);
  for (const ir::TensorRef& init : g.initializers) {
    if (init.name.valid() && m.str(init.name) == want) {
      return read_shape_initializer(m, g, init, base, base_size);
    }
  }
  return std::nullopt;
}

int64_t clamp64(int64_t v, int64_t lo, int64_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Convolution output shape (NCHW-style), behaviour-identical to the original
// Conv/ConvTranspose branch. `x`=input feature map, `w`=weight tensor, kernel/
// stride/pad/dilation/group read from node attrs. `transpose` selects the
// ConvTranspose weight-layout + output formula. Returns nullopt when unresolved
// (rank too small, unknown kernel, non-positive stride, or a partial result) —
// callers then fall back to dtype-only.
std::optional<Shape> conv_output_shape(const Model& m, const Graph& g,
                                       const Node& n, const Shape& x,
                                       const Shape& w, bool transpose) {
  if (x.size() < 3 || w.size() < 2) return std::nullopt;
  size_t spatial = x.size() - 2;
  auto kernel = attr_ints(m, g, n, "kernel_shape");
  auto strides = attr_ints(m, g, n, "strides");
  auto pads = attr_ints(m, g, n, "pads");
  auto dil = attr_ints(m, g, n, "dilations");
  auto out_pad = attr_ints(m, g, n, "output_padding");
  int64_t group = attr_int(m, g, n, "group", 1);

  Shape out;
  out.push_back(x[0]);  // N
  int64_t cout = transpose ? w[1] * group : w[0];
  out.push_back(cout);

  bool ok = true;
  for (size_t d = 0; d < spatial; ++d) {
    int64_t in_dim = x[2 + d];
    if (in_dim < 0) { out.push_back(-1); continue; }
    int64_t k = kernel.size() > d ? kernel[d]
               : (w.size() > 2 + d ? w[2 + d] : -1);
    if (k < 0) { ok = false; break; }
    int64_t st = strides.size() > d ? strides[d] : 1;
    if (st <= 0) { ok = false; break; }
    int64_t di = dil.size() > d ? dil[d] : 1;
    int64_t p_begin = pads.size() > d ? pads[d] : 0;
    int64_t p_end = pads.size() > spatial + d ? pads[spatial + d] : 0;
    int64_t eff_k = di * (k - 1) + 1;
    int64_t o;
    if (!transpose) {
      o = (in_dim + p_begin + p_end - eff_k) / st + 1;
    } else {
      int64_t op_pad = out_pad.size() > d ? out_pad[d] : 0;
      o = st * (in_dim - 1) + op_pad + eff_k - p_begin - p_end;
    }
    out.push_back(o);
  }
  if (ok && out.size() == x.size()) return out;
  return std::nullopt;
}

// MatMul output shape (batched, last two dims are the matrix dims), behaviour-
// identical to the original MatMul branch. Returns nullopt when either operand
// has no shape.
std::optional<Shape> matmul_output_shape(const Shape& s0, const Shape& s1) {
  if (s0.size() < 1 || s1.size() < 1) return std::nullopt;
  Shape a = s0, b = s1;
  bool a1 = a.size() == 1, b1 = b.size() == 1;
  if (a1) { Shape t; t.push_back(1); t.push_back(a[0]); a = t; }
  if (b1) { Shape t; t.push_back(b[0]); t.push_back(1); b = t; }
  if (a.size() >= 2 && b.size() >= 2) {
    int64_t M = a[a.size() - 2];
    int64_t N = b[b.size() - 1];
    Shape ab, bb;
    for (size_t i = 0; i + 2 < a.size(); ++i) ab.push_back(a[i]);
    for (size_t i = 0; i + 2 < b.size(); ++i) bb.push_back(b[i]);
    Shape batch;
    auto bc = broadcast(ab, bb);
    if (bc) batch = *bc;
    Shape s = batch;
    if (!a1) s.push_back(M);
    if (!b1) s.push_back(N);
    return s;
  }
  return std::nullopt;
}

// Compute the topological order of nodes (producer before consumer). Falls back
// to natural order for cycles / missing producers.
std::vector<uint32_t> topo_order(const Graph& g) {
  size_t nn = g.nodes.size();
  std::vector<uint32_t> order;
  order.reserve(nn);
  std::vector<uint8_t> state(nn, 0);  // 0=unvisited,1=on-stack,2=done

  std::vector<std::pair<uint32_t, uint32_t>> stack;  // (node, next-input-slot)
  for (uint32_t start = 0; start < nn; ++start) {
    if (state[start] != 0) continue;
    stack.push_back({start, 0});
    while (!stack.empty()) {
      auto& [ni, slot] = stack.back();
      if (state[ni] == 2) { stack.pop_back(); continue; }
      state[ni] = 1;
      const Node& n = g.nodes[ni];
      bool descended = false;
      while (slot < n.inputs.count) {
        int32_t vi = input_value(g, n, slot);
        ++slot;
        if (vi < 0) continue;
        int32_t prod = g.values[static_cast<size_t>(vi)].producer;
        if (prod < 0 || static_cast<size_t>(prod) >= nn) continue;
        if (state[prod] == 0) {
          stack.push_back({static_cast<uint32_t>(prod), 0});
          descended = true;
          break;
        }
      }
      if (descended) continue;
      state[ni] = 2;
      order.push_back(ni);
      stack.pop_back();
    }
  }
  return order;
}

}  // namespace

uint32_t infer_shapes_ext(Model& model, uint32_t graph_index,
                          const uint8_t* mmap_base, size_t mmap_size,
                          ProgressSink* progress) {
  if (graph_index >= model.graphs.size()) return 0;
  Graph& g = model.graphs[graph_index];

  const uint8_t* base = mmap_base;
  const uint64_t base_size = static_cast<uint64_t>(mmap_size);

  uint32_t resolved = 0;

  // Seed value dtypes/shapes from initializers by name (initializers carry both
  // shape and dtype even though their payload is unread).
  for (const ir::TensorRef& init : g.initializers) {
    for (auto& v : g.values) {
      if (v.name == init.name && init.name.valid()) {
        Shape s;
        for (int64_t d : init.shape) s.push_back(d);
        if (set_shape(v, s, init.dtype)) ++resolved;
      }
    }
  }

  std::vector<uint32_t> order = topo_order(g);
  const size_t total = order.empty() ? 1 : order.size();

  for (size_t idx = 0; idx < order.size(); ++idx) {
    if (progress && (idx % 4096 == 0)) {
      progress->set(static_cast<float>(idx) / static_cast<float>(total),
                    "shape inference");
    }
    uint32_t ni = order[idx];
    const Node& n = g.nodes[ni];
    std::string_view op = model.str(n.op_type);

    int32_t in0 = input_value(g, n, 0);
    int32_t in1 = input_value(g, n, 1);
    int32_t out0 = output_value(g, n, 0);
    if (out0 < 0) continue;
    ir::ValueInfo& outv = g.values[static_cast<size_t>(out0)];

    const Shape emptyShape{};
    const Shape& s0 =
        in0 >= 0 ? g.values[static_cast<size_t>(in0)].shape : emptyShape;
    const Shape& s1 =
        in1 >= 0 ? g.values[static_cast<size_t>(in1)].shape : emptyShape;
    DType dt0 = value_dtype(g, in0);

    // --- Elementwise / broadcasting -----------------------------------------
    if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" ||
        op == "Pow" || op == "Max" || op == "Min" || op == "Where" ||
        op == "Equal" || op == "And" || op == "Or" || op == "Xor" ||
        op == "Greater" || op == "Less" || op == "Mod" ||
        op == "GreaterOrEqual" || op == "LessOrEqual" ||
        op == "BitwiseAnd" || op == "BitwiseOr" || op == "BitwiseXor" ||
        op == "BitShift") {
      const Shape* a = &s0;
      const Shape* b = &s1;
      DType odt = dt0;
      if (op == "Where") {
        int32_t inx = input_value(g, n, 1);
        int32_t iny = input_value(g, n, 2);
        a = inx >= 0 ? &g.values[static_cast<size_t>(inx)].shape : &emptyShape;
        b = iny >= 0 ? &g.values[static_cast<size_t>(iny)].shape : &emptyShape;
        odt = value_dtype(g, inx);
      }
      if (op == "Equal" || op == "Greater" || op == "Less" || op == "And" ||
          op == "Or" || op == "Xor" || op == "GreaterOrEqual" ||
          op == "LessOrEqual")
        odt = DType::Bool;
      // BitwiseAnd/Or/Xor and BitShift preserve the integer input dtype (dt0);
      // GreaterOrEqual/LessOrEqual are comparisons -> Bool (handled above).
      if (!a->empty() && !b->empty()) {
        auto bc = broadcast(*a, *b);
        if (bc) { if (set_shape(outv, *bc, odt)) ++resolved; }
      } else if (!a->empty()) {
        if (set_shape(outv, *a, odt)) ++resolved;
      } else if (!b->empty()) {
        if (set_shape(outv, *b, odt)) ++resolved;
      } else {
        carry_dtype(outv, odt);
      }
      continue;
    }

    // --- Unary shape-preserving ---------------------------------------------
    if (op == "Relu" || op == "Sigmoid" || op == "Tanh" || op == "Softmax" ||
        op == "LogSoftmax" || op == "Gelu" || op == "Erf" || op == "Exp" ||
        op == "Log" || op == "Sqrt" || op == "Abs" || op == "Neg" ||
        op == "Reciprocal" || op == "Clip" || op == "Elu" || op == "Selu" ||
        op == "LeakyRelu" || op == "PRelu" || op == "HardSigmoid" ||
        op == "HardSwish" || op == "Softplus" || op == "Softsign" ||
        op == "Identity" || op == "Dropout" || op == "LayerNormalization" ||
        op == "BatchNormalization" || op == "InstanceNormalization" ||
        op == "Sin" || op == "Cos" || op == "Floor" || op == "Ceil" ||
        op == "Round" || op == "Sign" ||
        op == "CumSum" || op == "Sinh" || op == "Cosh" || op == "Tan" ||
        op == "Asin" || op == "Acos" || op == "Atan" || op == "Asinh" ||
        op == "Acosh" || op == "Atanh" || op == "Mish" ||
        op == "ThresholdedRelu" || op == "Celu" || op == "Shrink" ||
        op == "QuickGelu" || op == "BitwiseNot") {
      // BitwiseNot preserves the integer input dtype, like the transcendentals.
      if (!s0.empty()) { if (set_shape(outv, s0, dt0)) ++resolved; }
      else carry_dtype(outv, dt0);
      continue;
    }

    // --- Not / IsNaN / IsInf: shape-preserving, dtype -> Bool ---------------
    if (op == "Not" || op == "IsNaN" || op == "IsInf") {
      if (!s0.empty()) { if (set_shape(outv, s0, DType::Bool)) ++resolved; }
      else carry_dtype(outv, DType::Bool);
      continue;
    }

    // --- Cast: shape preserved, dtype changes via `to` ----------------------
    if (op == "Cast") {
      DType nd = dt0;
      int64_t to = attr_int(model, g, n, "to", -1);
      switch (to) {
        case 1: nd = DType::F32; break;
        case 2: nd = DType::U8; break;
        case 3: nd = DType::I8; break;
        case 4: nd = DType::U16; break;
        case 5: nd = DType::I16; break;
        case 6: nd = DType::I32; break;
        case 7: nd = DType::I64; break;
        case 9: nd = DType::Bool; break;
        case 10: nd = DType::F16; break;
        case 11: nd = DType::F64; break;
        case 12: nd = DType::U32; break;
        case 13: nd = DType::U64; break;
        case 16: nd = DType::BF16; break;
        default: break;
      }
      if (!s0.empty()) { if (set_shape(outv, s0, nd)) ++resolved; }
      else carry_dtype(outv, nd);
      continue;
    }

    // --- Gemm: (M,K)x(K,N) with transA/transB -------------------------------
    if (op == "Gemm") {
      if (s0.size() == 2 && s1.size() == 2) {
        bool ta = attr_int(model, g, n, "transA", 0) != 0;
        bool tb = attr_int(model, g, n, "transB", 0) != 0;
        int64_t M = ta ? s0[1] : s0[0];
        int64_t N = tb ? s1[0] : s1[1];
        Shape s; s.push_back(M); s.push_back(N);
        if (set_shape(outv, s, dt0)) ++resolved;
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- MatMul: batched, last two dims are matrix dims ---------------------
    if (op == "MatMul") {
      auto s = matmul_output_shape(s0, s1);
      if (s) { if (set_shape(outv, *s, dt0)) ++resolved; }
      else carry_dtype(outv, dt0);
      continue;
    }

    // --- Conv / ConvTranspose (NCHW) ----------------------------------------
    if (op == "Conv" || op == "ConvTranspose") {
      auto out = conv_output_shape(model, g, n, s0, s1, op == "ConvTranspose");
      if (out) { if (set_shape(outv, *out, dt0)) ++resolved; }
      else carry_dtype(outv, dt0);
      continue;
    }

    // --- Pooling (spatial, like Conv without channel change) ----------------
    if (op == "MaxPool" || op == "AveragePool" || op == "LpPool") {
      if (s0.size() >= 3) {
        size_t spatial = s0.size() - 2;
        auto kernel = attr_ints(model, g, n, "kernel_shape");
        auto strides = attr_ints(model, g, n, "strides");
        auto pads = attr_ints(model, g, n, "pads");
        auto dil = attr_ints(model, g, n, "dilations");
        Shape out; out.push_back(s0[0]); out.push_back(s0[1]);
        bool ok = true;
        for (size_t d = 0; d < spatial; ++d) {
          int64_t in_dim = s0[2 + d];
          if (in_dim < 0) { out.push_back(-1); continue; }
          int64_t k = kernel.size() > d ? kernel[d] : -1;
          if (k < 0) { ok = false; break; }
          int64_t st = strides.size() > d ? strides[d] : 1;
          if (st <= 0) { ok = false; break; }
          int64_t di = dil.size() > d ? dil[d] : 1;
          int64_t p_begin = pads.size() > d ? pads[d] : 0;
          int64_t p_end = pads.size() > spatial + d ? pads[spatial + d] : 0;
          int64_t eff_k = di * (k - 1) + 1;
          out.push_back((in_dim + p_begin + p_end - eff_k) / st + 1);
        }
        if (ok) { if (set_shape(outv, out, dt0)) ++resolved; }
        else carry_dtype(outv, dt0);
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
      if (s0.size() >= 2) {
        Shape out; out.push_back(s0[0]); out.push_back(s0[1]);
        for (size_t d = 2; d < s0.size(); ++d) out.push_back(1);
        if (set_shape(outv, out, dt0)) ++resolved;
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Transpose: perm (default: reverse dims) ----------------------------
    if (op == "Transpose") {
      if (!s0.empty()) {
        auto perm = attr_ints(model, g, n, "perm");
        Shape out;
        bool ok = true;
        if (perm.empty()) {
          for (size_t i = s0.size(); i > 0; --i) out.push_back(s0[i - 1]);
        } else {
          for (int64_t p : perm) {
            if (p < 0 || static_cast<size_t>(p) >= s0.size()) { ok = false; break; }
            out.push_back(s0[static_cast<size_t>(p)]);
          }
        }
        if (ok) { if (set_shape(outv, out, dt0)) ++resolved; }
        else carry_dtype(outv, dt0);
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Concat: sum along axis ---------------------------------------------
    if (op == "Concat") {
      int64_t axis = attr_int(model, g, n, "axis", 0);
      Shape out;
      bool ok = false;
      int64_t rank = -1;
      for (uint32_t s = 0; s < n.inputs.count; ++s) {
        int32_t vi = input_value(g, n, s);
        if (vi < 0) { ok = false; break; }
        const Shape& si = g.values[static_cast<size_t>(vi)].shape;
        if (si.empty()) { ok = false; break; }
        if (rank < 0) { rank = static_cast<int64_t>(si.size()); out = si; ok = true; }
        else if (static_cast<int64_t>(si.size()) != rank) { ok = false; break; }
        else {
          int64_t ax = axis < 0 ? axis + rank : axis;
          if (ax < 0 || ax >= rank) { ok = false; break; }
          if (s != 0) {
            if (out[static_cast<size_t>(ax)] < 0 ||
                si[static_cast<size_t>(ax)] < 0)
              out[static_cast<size_t>(ax)] = -1;
            else
              out[static_cast<size_t>(ax)] += si[static_cast<size_t>(ax)];
          }
        }
      }
      if (ok) { if (set_shape(outv, out, dt0)) ++resolved; }
      else carry_dtype(outv, dt0);
      continue;
    }

    // --- Flatten: (d0..da-1) x (da..dn) -> (rows, cols) ---------------------
    if (op == "Flatten") {
      if (!s0.empty()) {
        int64_t axis = attr_int(model, g, n, "axis", 1);
        int64_t rank = static_cast<int64_t>(s0.size());
        if (axis < 0) axis += rank;
        int64_t rows = 1, cols = 1;
        bool dyn_r = false, dyn_c = false;
        for (int64_t i = 0; i < rank; ++i) {
          int64_t d = s0[static_cast<size_t>(i)];
          if (i < axis) { if (d < 0) dyn_r = true; else rows *= d; }
          else { if (d < 0) dyn_c = true; else cols *= d; }
        }
        Shape out;
        out.push_back(dyn_r ? -1 : rows);
        out.push_back(dyn_c ? -1 : cols);
        if (set_shape(outv, out, dt0)) ++resolved;
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Reshape: decode small constant shape initializer (raw_data only) ----
    if (op == "Reshape") {
      std::optional<std::vector<int64_t>> target =
          read_int_input(model, g, n, 1, base, base_size);
      if (target && !s0.empty()) {
        int64_t total_in = 1; bool in_dyn = false;
        for (int64_t d : s0) { if (d < 0) in_dyn = true; else total_in *= d; }
        Shape out;
        int64_t prod = 1; int neg_idx = -1; bool any_dyn = in_dyn;
        for (size_t i = 0; i < target->size(); ++i) {
          int64_t d = (*target)[i];
          if (d == 0) {
            d = (i < s0.size()) ? s0[i] : -1;
          }
          if (d == -1) { neg_idx = static_cast<int>(i); out.push_back(-1); }
          else { out.push_back(d); if (d < 0) any_dyn = true; else prod *= d; }
        }
        if (neg_idx >= 0 && !any_dyn && prod > 0) {
          out[static_cast<size_t>(neg_idx)] = total_in / prod;
        }
        if (set_shape(outv, out, dt0)) ++resolved;
      } else if (target) {
        Shape out; bool any_neg = false;
        for (int64_t d : *target) { out.push_back(d); if (d < 0) any_neg = true; }
        if (!any_neg && set_shape(outv, out, dt0)) ++resolved;
        else carry_dtype(outv, dt0);
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Expand: broadcast input against a constant shape (input 1) ---------
    if (op == "Expand") {
      auto target = read_int_input(model, g, n, 1, base, base_size);
      if (target && !s0.empty()) {
        Shape t;
        for (int64_t d : *target) t.push_back(d);
        auto bc = broadcast(s0, t);
        if (bc) { if (set_shape(outv, *bc, dt0)) ++resolved; }
        else carry_dtype(outv, dt0);
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Tile: repeat each dim by a constant `repeats` (input 1) ------------
    if (op == "Tile") {
      auto reps = read_int_input(model, g, n, 1, base, base_size);
      if (reps && !s0.empty() && reps->size() == s0.size()) {
        Shape out;
        for (size_t i = 0; i < s0.size(); ++i) {
          int64_t d = s0[i], r = (*reps)[i];
          if (d < 0 || r < 0) out.push_back(-1);
          else out.push_back(d * r);
        }
        if (set_shape(outv, out, dt0)) ++resolved;
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Gather: out = data[:axis] + indices.shape + data[axis+1:] ----------
    if (op == "Gather") {
      if (!s0.empty()) {
        int64_t rank = static_cast<int64_t>(s0.size());
        int64_t axis = attr_int(model, g, n, "axis", 0);
        if (axis < 0) axis += rank;
        if (axis >= 0 && axis < rank) {
          const Shape& idx_shape = s1;  // indices shape (empty => scalar => rank-0)
          Shape out;
          for (int64_t i = 0; i < axis; ++i) out.push_back(s0[static_cast<size_t>(i)]);
          for (int64_t d : idx_shape) out.push_back(d);
          for (int64_t i = axis + 1; i < rank; ++i)
            out.push_back(s0[static_cast<size_t>(i)]);
          if (set_shape(outv, out, dt0)) ++resolved;
        } else {
          carry_dtype(outv, dt0);
        }
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Slice: constant starts/ends/axes/steps (inputs or opset<10 attrs) --
    if (op == "Slice") {
      if (!s0.empty()) {
        auto starts = read_int_input(model, g, n, 1, base, base_size);
        auto ends = read_int_input(model, g, n, 2, base, base_size);
        std::vector<int64_t> a_starts = starts ? *starts
                                               : attr_ints(model, g, n, "starts");
        std::vector<int64_t> a_ends = ends ? *ends
                                           : attr_ints(model, g, n, "ends");
        auto axes_in = read_int_input(model, g, n, 3, base, base_size);
        std::vector<int64_t> axes = axes_in ? *axes_in
                                            : attr_ints(model, g, n, "axes");
        auto steps_in = read_int_input(model, g, n, 4, base, base_size);
        std::vector<int64_t> steps = steps_in ? *steps_in
                                              : attr_ints(model, g, n, "steps");
        int64_t rank = static_cast<int64_t>(s0.size());
        if (!a_starts.empty() && a_starts.size() == a_ends.size()) {
          Shape dims = s0;
          for (size_t i = 0; i < a_starts.size(); ++i) {
            int64_t axis = (i < axes.size()) ? axes[i] : static_cast<int64_t>(i);
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank) continue;
            int64_t d = dims[static_cast<size_t>(axis)];
            if (d < 0) continue;
            int64_t st = (i < steps.size()) ? steps[i] : 1;
            if (st == 0) continue;
            int64_t s = a_starts[i], e = a_ends[i];
            if (s < 0) s += d;
            if (e < 0) e += d;
            int64_t outd;
            if (st > 0) {
              s = clamp64(s, 0, d);
              e = clamp64(e, 0, d);
              outd = e > s ? (e - s + st - 1) / st : 0;
            } else {
              s = clamp64(s, 0, d - 1);
              e = clamp64(e, -1, d - 1);
              int64_t ns = -st;
              outd = s > e ? (s - e + ns - 1) / ns : 0;
            }
            dims[static_cast<size_t>(axis)] = outd < 0 ? 0 : outd;
          }
          if (set_shape(outv, dims, dt0)) ++resolved;
        } else {
          carry_dtype(outv, dt0);
        }
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Resize: prefer constant `sizes` (input 3) as the full output shape --
    if (op == "Resize") {
      auto sizes = read_int_input(model, g, n, 3, base, base_size);
      if (sizes && !sizes->empty()) {
        Shape out;
        for (int64_t d : *sizes) out.push_back(d);
        if (set_shape(outv, out, dt0)) ++resolved;
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Constant: shape/dtype comes from the `value`* attribute ------------
    if (op == "Constant") {
      const ir::AttrValue* v = attr(model, g, n, "value");
      if (v && v->kind == ir::AttrValue::Kind::Tensor) {
        Shape out;
        for (int64_t d : v->tensor.shape) out.push_back(d);
        if (set_shape(outv, out, v->tensor.dtype)) ++resolved;
        else carry_dtype(outv, v->tensor.dtype);
      } else if (attr(model, g, n, "value_ints")) {
        auto ints = attr_ints(model, g, n, "value_ints");
        Shape out; out.push_back(static_cast<int64_t>(ints.size()));
        if (set_shape(outv, out, DType::I64)) ++resolved;
      } else if (attr(model, g, n, "value_floats")) {
        const ir::AttrValue* vf = attr(model, g, n, "value_floats");
        Shape out; out.push_back(static_cast<int64_t>(vf->floats.size()));
        if (set_shape(outv, out, DType::F32)) ++resolved;
      } else if (attr(model, g, n, "value_int")) {
        carry_dtype(outv, DType::I64);
      } else if (attr(model, g, n, "value_float")) {
        carry_dtype(outv, DType::F32);
      }
      continue;
    }

    // --- Split: multi-output, split along axis; resolve every output slot ---
    if (op == "Split") {
      if (!s0.empty()) {
        int64_t rank = static_cast<int64_t>(s0.size());
        int64_t axis = attr_int(model, g, n, "axis", 0);
        if (axis < 0) axis += rank;
        std::vector<int64_t> split = attr_ints(model, g, n, "split");
        if (split.empty()) {
          auto sp = read_int_input(model, g, n, 1, base, base_size);
          if (sp) split = *sp;
        }
        uint32_t nout = n.outputs.count;
        if (axis >= 0 && axis < rank && nout > 0) {
          int64_t axis_dim = s0[static_cast<size_t>(axis)];
          for (uint32_t j = 0; j < nout; ++j) {
            int32_t ov = output_value(g, n, j);
            if (ov < 0) continue;
            ir::ValueInfo& ovv = g.values[static_cast<size_t>(ov)];
            Shape out = s0;
            int64_t part;
            if (j < split.size()) {
              part = split[j];
            } else if (split.empty() && axis_dim >= 0 &&
                       axis_dim % static_cast<int64_t>(nout) == 0) {
              part = axis_dim / static_cast<int64_t>(nout);
            } else {
              part = -1;
            }
            out[static_cast<size_t>(axis)] = part;
            if (set_shape(ovv, out, dt0)) ++resolved;
          }
        } else {
          carry_dtype(outv, dt0);
        }
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- TopK: multi-output; dim `axis` becomes k. slot0=input dtype, slot1=I64
    if (op == "TopK") {
      if (!s0.empty()) {
        int64_t rank = static_cast<int64_t>(s0.size());
        int64_t axis = attr_int(model, g, n, "axis", -1);
        if (axis < 0) axis += rank;
        int64_t k = attr_int(model, g, n, "k", -1);
        if (k < 0) {
          auto kv = read_int_input(model, g, n, 1, base, base_size);
          if (kv && !kv->empty()) k = (*kv)[0];
        }
        if (axis >= 0 && axis < rank) {
          Shape out = s0;
          out[static_cast<size_t>(axis)] = k;  // -1 if k unknown
          int32_t o0 = output_value(g, n, 0);
          int32_t o1 = output_value(g, n, 1);
          if (o0 >= 0 && k >= 0) {
            if (set_shape(g.values[static_cast<size_t>(o0)], out, dt0)) ++resolved;
          } else if (o0 >= 0) {
            carry_dtype(g.values[static_cast<size_t>(o0)], dt0);
          }
          if (o1 >= 0 && k >= 0) {
            if (set_shape(g.values[static_cast<size_t>(o1)], out, DType::I64))
              ++resolved;
          } else if (o1 >= 0) {
            carry_dtype(g.values[static_cast<size_t>(o1)], DType::I64);
          }
        } else {
          carry_dtype(outv, dt0);
        }
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Squeeze / Unsqueeze (axes attr OR opset-13 axes-as-input) ----------
    if (op == "Squeeze") {
      if (!s0.empty()) {
        auto axes = attr_ints(model, g, n, "axes");  // opset<13
        if (axes.empty()) {
          auto ai = read_int_input(model, g, n, 1, base, base_size);
          if (ai) axes = *ai;
        }
        Shape out;
        int64_t rank = static_cast<int64_t>(s0.size());
        if (axes.empty()) {
          for (int64_t d : s0) if (d != 1) out.push_back(d);
        } else {
          std::vector<bool> drop(s0.size(), false);
          for (int64_t a : axes) { int64_t ax = a < 0 ? a + rank : a;
            if (ax >= 0 && ax < rank) drop[static_cast<size_t>(ax)] = true; }
          for (size_t i = 0; i < s0.size(); ++i) if (!drop[i]) out.push_back(s0[i]);
        }
        if (set_shape(outv, out, dt0)) ++resolved;
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }
    if (op == "Unsqueeze") {
      if (!s0.empty()) {
        auto axes = attr_ints(model, g, n, "axes");  // opset<13
        if (axes.empty()) {
          auto ai = read_int_input(model, g, n, 1, base, base_size);
          if (ai) axes = *ai;
        }
        if (!axes.empty()) {
          int64_t new_rank = static_cast<int64_t>(s0.size()) +
                             static_cast<int64_t>(axes.size());
          std::vector<bool> ins(static_cast<size_t>(new_rank), false);
          for (int64_t a : axes) { int64_t ax = a < 0 ? a + new_rank : a;
            if (ax >= 0 && ax < new_rank) ins[static_cast<size_t>(ax)] = true; }
          Shape out; size_t si = 0;
          for (int64_t i = 0; i < new_rank; ++i) {
            if (ins[static_cast<size_t>(i)]) out.push_back(1);
            else if (si < s0.size()) out.push_back(s0[si++]);
          }
          if (set_shape(outv, out, dt0)) ++resolved;
        } else {
          carry_dtype(outv, dt0);
        }
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Reduce* (keepdims / axes) ------------------------------------------
    if (op.substr(0, 6) == "Reduce" || op == "ArgMax" || op == "ArgMin") {
      if (!s0.empty()) {
        auto axes = attr_ints(model, g, n, "axes");
        if (axes.empty() && op.substr(0, 6) == "Reduce") {
          // opset-18+ moves axes to input 1.
          auto ai = read_int_input(model, g, n, 1, base, base_size);
          if (ai) axes = *ai;
        }
        int64_t keepdims = attr_int(model, g, n, "keepdims", 1);
        int64_t rank = static_cast<int64_t>(s0.size());
        std::vector<bool> red(s0.size(), false);
        if (axes.empty() && (op == "ArgMax" || op == "ArgMin")) {
          int64_t ax = attr_int(model, g, n, "axis", 0);
          if (ax < 0) ax += rank;
          if (ax >= 0 && ax < rank) red[static_cast<size_t>(ax)] = true;
        } else if (axes.empty()) {
          for (size_t i = 0; i < red.size(); ++i) red[i] = true;  // all
        } else {
          for (int64_t a : axes) { int64_t ax = a < 0 ? a + rank : a;
            if (ax >= 0 && ax < rank) red[static_cast<size_t>(ax)] = true; }
        }
        Shape out;
        for (size_t i = 0; i < s0.size(); ++i) {
          if (red[i]) { if (keepdims) out.push_back(1); }
          else out.push_back(s0[i]);
        }
        DType odt = (op == "ArgMax" || op == "ArgMin") ? DType::I64 : dt0;
        if (set_shape(outv, out, odt)) ++resolved;
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Pad: attr pads (opset<11) OR pads-as-input (opset-11+), axes opt ----
    if (op == "Pad") {
      if (!s0.empty()) {
        std::vector<int64_t> pads = attr_ints(model, g, n, "pads");  // opset<11
        if (pads.empty()) {
          auto pin = read_int_input(model, g, n, 1, base, base_size);
          if (pin) pads = *pin;
        }
        // Optional axes input (opset-18); default all axes in order.
        std::vector<int64_t> axes;
        {
          auto ain = read_int_input(model, g, n, 3, base, base_size);
          if (ain) axes = *ain;
        }
        int64_t rank = static_cast<int64_t>(s0.size());
        Shape out = s0;
        bool ok = false;
        if (axes.empty() && pads.size() == s0.size() * 2) {
          // pads = [begin_0..begin_{r-1}, end_0..end_{r-1}].
          ok = true;
          for (size_t i = 0; i < s0.size(); ++i) {
            int64_t d = s0[i];
            if (d < 0) { out[i] = -1; continue; }
            out[i] = d + pads[i] + pads[i + s0.size()];
          }
        } else if (!axes.empty() && pads.size() == axes.size() * 2) {
          ok = true;
          size_t na = axes.size();
          for (size_t i = 0; i < na; ++i) {
            int64_t ax = axes[i] < 0 ? axes[i] + rank : axes[i];
            if (ax < 0 || ax >= rank) { ok = false; break; }
            int64_t d = out[static_cast<size_t>(ax)];
            if (d < 0) continue;
            out[static_cast<size_t>(ax)] = d + pads[i] + pads[i + na];
          }
        }
        if (ok) { if (set_shape(outv, out, dt0)) ++resolved; }
        else carry_dtype(outv, dt0);
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Shape: output is 1-D int64 of length = rank(input) -----------------
    if (op == "Shape") {
      Shape out;
      if (!s0.empty()) out.push_back(static_cast<int64_t>(s0.size()));
      if (set_shape(outv, out, DType::I64)) ++resolved;
      else carry_dtype(outv, DType::I64);
      continue;
    }

    // --- Variadic Sum / Mean: emit a shape ONLY when ALL inputs resolve -----
    if (op == "Sum" || op == "Mean") {
      bool all_ok = n.inputs.count > 0;
      Shape acc;
      for (uint32_t s = 0; s < n.inputs.count && all_ok; ++s) {
        int32_t vi = input_value(g, n, s);
        if (vi < 0) { all_ok = false; break; }
        const Shape& si = g.values[static_cast<size_t>(vi)].shape;
        if (si.empty()) { all_ok = false; break; }
        if (s == 0) { acc = si; }
        else {
          auto bc = broadcast(acc, si);
          if (!bc) { all_ok = false; break; }
          acc = *bc;
        }
      }
      if (all_ok && !acc.empty()) { if (set_shape(outv, acc, dt0)) ++resolved; }
      else carry_dtype(outv, dt0);
      continue;
    }

    // --- Attention (com.microsoft, fused) -----------------------------------
    // out0 = copy of input[0] shape+dtype (best-effort, not claimed exact).
    // present_key (slot1) / present_value (slot2) get dtype-only.
    if (op == "Attention") {
      if (!s0.empty()) { if (set_shape(outv, s0, dt0)) ++resolved; }
      else carry_dtype(outv, dt0);
      int32_t o1 = output_value(g, n, 1);
      int32_t o2 = output_value(g, n, 2);
      if (o1 >= 0) carry_dtype(g.values[static_cast<size_t>(o1)], dt0);
      if (o2 >= 0) carry_dtype(g.values[static_cast<size_t>(o2)], dt0);
      continue;
    }

    // --- MultiHeadAttention (com.microsoft, pre-projected) ------------------
    // out0 = [B, Sq, v_hidden]; B,Sq from query(in0); v_hidden from value(in2):
    //   rank3 -> value.back(); rank4 [B,N,S,H] -> value[1]*value[3];
    //   else query.back().
    if (op == "MultiHeadAttention") {
      int32_t iv = input_value(g, n, 2);
      const Shape& qs = s0;
      const Shape& vs =
          iv >= 0 ? g.values[static_cast<size_t>(iv)].shape : emptyShape;
      if (qs.size() >= 2) {
        int64_t B = qs[0];
        int64_t Sq = qs[1];
        int64_t v_hidden = -1;
        if (vs.size() == 3) v_hidden = vs[2];
        else if (vs.size() == 4) {
          int64_t a = vs[1], b = vs[3];
          v_hidden = (a < 0 || b < 0) ? -1 : a * b;
        } else if (!qs.empty()) {
          v_hidden = qs[qs.size() - 1];
        }
        Shape out; out.push_back(B); out.push_back(Sq); out.push_back(v_hidden);
        if (set_shape(outv, out, dt0)) ++resolved;
      } else {
        carry_dtype(outv, dt0);
      }
      continue;
    }

    // --- Recurrent LSTM / GRU / RNN (one parameterized branch) --------------
    // X = in0; W = in1; R = in2. gates: LSTM=4, GRU=3, RNN=1 (LSTM emits Y_c).
    // layout==0: X=[seq,batch,input]  Y=[seq,dirs,batch,hidden]
    //            Yh/Yc=[dirs,batch,hidden]
    // layout==1: X=[batch,seq,input]  Y=[batch,seq,dirs,hidden]
    //            Yh/Yc=[batch,dirs,hidden]
    if (op == "LSTM" || op == "GRU" || op == "RNN") {
      int32_t iw = input_value(g, n, 1);
      int32_t ir = input_value(g, n, 2);
      const Shape& xs = s0;
      const Shape& ws =
          iw >= 0 ? g.values[static_cast<size_t>(iw)].shape : emptyShape;
      const Shape& rs =
          ir >= 0 ? g.values[static_cast<size_t>(ir)].shape : emptyShape;
      int gates = op == "LSTM" ? 4 : (op == "GRU" ? 3 : 1);
      int64_t layout = attr_int(model, g, n, "layout", 0);

      // dirs: direction attr, else W.shape[0], else 1.
      std::string_view dir = attr_string(model, g, n, "direction");
      int64_t dirs;
      if (dir == "bidirectional") dirs = 2;
      else if (dir == "forward" || dir == "reverse") dirs = 1;
      else if (ws.size() == 3 && ws[0] > 0) dirs = ws[0];
      else dirs = 1;

      // hidden: hidden_size attr, else R.shape[2], else W.shape[1]/gates.
      int64_t hidden = attr_int(model, g, n, "hidden_size", -1);
      if (hidden <= 0 && rs.size() == 3 && rs[2] > 0) hidden = rs[2];
      if (hidden <= 0 && ws.size() == 3 && ws[1] > 0)
        hidden = ws[1] / gates;
      if (hidden <= 0) hidden = -1;

      // seq/batch from X per layout (partial -1 allowed).
      int64_t seq = -1, batch = -1;
      if (xs.size() == 3) {
        if (layout == 1) { batch = xs[0]; seq = xs[1]; }
        else { seq = xs[0]; batch = xs[1]; }
      }

      // Y (slot0), Y_h (slot1), Y_c (slot2 for LSTM).
      if (hidden > 0) {
        Shape y, yh;
        if (layout == 1) {
          y.push_back(batch); y.push_back(seq); y.push_back(dirs);
          y.push_back(hidden);
          yh.push_back(batch); yh.push_back(dirs); yh.push_back(hidden);
        } else {
          y.push_back(seq); y.push_back(dirs); y.push_back(batch);
          y.push_back(hidden);
          yh.push_back(dirs); yh.push_back(batch); yh.push_back(hidden);
        }
        if (out0 >= 0) { if (set_shape(outv, y, dt0)) ++resolved; }
        int32_t o1 = output_value(g, n, 1);
        if (o1 >= 0) {
          if (set_shape(g.values[static_cast<size_t>(o1)], yh, dt0)) ++resolved;
        }
        if (gates == 4) {
          int32_t o2 = output_value(g, n, 2);
          if (o2 >= 0) {
            if (set_shape(g.values[static_cast<size_t>(o2)], yh, dt0))
              ++resolved;
          }
        }
      } else {
        // hidden unresolvable -> dtype-only on every output slot.
        for (uint32_t j = 0; j < n.outputs.count; ++j) {
          int32_t ov = output_value(g, n, j);
          if (ov >= 0) carry_dtype(g.values[static_cast<size_t>(ov)], dt0);
        }
      }
      continue;
    }

    // --- QLinearConv: x=slot0, w=slot3; out dtype = y_zp(slot7) else x -------
    if (op == "QLinearConv") {
      int32_t ix = input_value(g, n, 0);
      int32_t iw = input_value(g, n, 3);
      const Shape& xs =
          ix >= 0 ? g.values[static_cast<size_t>(ix)].shape : emptyShape;
      const Shape& ws =
          iw >= 0 ? g.values[static_cast<size_t>(iw)].shape : emptyShape;
      DType odt = value_dtype(g, input_value(g, n, 7));
      if (odt == DType::Unknown) odt = value_dtype(g, ix);
      auto out = conv_output_shape(model, g, n, xs, ws, false);
      if (out) { if (set_shape(outv, *out, odt)) ++resolved; }
      else carry_dtype(outv, odt);
      continue;
    }

    // --- ConvInteger: x=slot0, w=slot1; out dtype I32 -----------------------
    if (op == "ConvInteger") {
      int32_t ix = input_value(g, n, 0);
      int32_t iw = input_value(g, n, 1);
      const Shape& xs =
          ix >= 0 ? g.values[static_cast<size_t>(ix)].shape : emptyShape;
      const Shape& ws =
          iw >= 0 ? g.values[static_cast<size_t>(iw)].shape : emptyShape;
      auto out = conv_output_shape(model, g, n, xs, ws, false);
      if (out) { if (set_shape(outv, *out, DType::I32)) ++resolved; }
      else carry_dtype(outv, DType::I32);
      continue;
    }

    // --- QLinearMatMul: a=slot0, b=slot3; out dtype = y_zp(slot7) else a -----
    if (op == "QLinearMatMul") {
      int32_t ia = input_value(g, n, 0);
      int32_t ib = input_value(g, n, 3);
      const Shape& as =
          ia >= 0 ? g.values[static_cast<size_t>(ia)].shape : emptyShape;
      const Shape& bs =
          ib >= 0 ? g.values[static_cast<size_t>(ib)].shape : emptyShape;
      DType odt = value_dtype(g, input_value(g, n, 7));
      if (odt == DType::Unknown) odt = value_dtype(g, ia);
      auto out = matmul_output_shape(as, bs);
      if (out) { if (set_shape(outv, *out, odt)) ++resolved; }
      else carry_dtype(outv, odt);
      continue;
    }

    // --- MatMulInteger: a=slot0, b=slot1; out dtype I32 ---------------------
    if (op == "MatMulInteger") {
      int32_t ia = input_value(g, n, 0);
      int32_t ib = input_value(g, n, 1);
      const Shape& as =
          ia >= 0 ? g.values[static_cast<size_t>(ia)].shape : emptyShape;
      const Shape& bs =
          ib >= 0 ? g.values[static_cast<size_t>(ib)].shape : emptyShape;
      auto out = matmul_output_shape(as, bs);
      if (out) { if (set_shape(outv, *out, DType::I32)) ++resolved; }
      else carry_dtype(outv, DType::I32);
      continue;
    }

    // --- QGemm (com.microsoft): a=slot0, b=slot3, transA/transB; out y_zp else a
    // Signature A(0),a_scale(1),a_zp(2),B(3),b_scale(4),b_zp(5),C(6),y_scale(7),
    // y_zp(8) — the optional C at slot 6 shifts y_zero_point to slot 8 (slot 7 is
    // y_scale, a float). Read the output dtype from slot 8, not 7.
    if (op == "QGemm") {
      int32_t ia = input_value(g, n, 0);
      int32_t ib = input_value(g, n, 3);
      const Shape& as =
          ia >= 0 ? g.values[static_cast<size_t>(ia)].shape : emptyShape;
      const Shape& bs =
          ib >= 0 ? g.values[static_cast<size_t>(ib)].shape : emptyShape;
      DType odt = value_dtype(g, input_value(g, n, 8));
      if (odt == DType::Unknown) odt = value_dtype(g, ia);
      if (as.size() == 2 && bs.size() == 2) {
        bool ta = attr_int(model, g, n, "transA", 0) != 0;
        bool tb = attr_int(model, g, n, "transB", 0) != 0;
        int64_t M = ta ? as[1] : as[0];
        int64_t N = tb ? bs[0] : bs[1];
        Shape s; s.push_back(M); s.push_back(N);
        if (set_shape(outv, s, odt)) ++resolved;
      } else {
        carry_dtype(outv, odt);
      }
      continue;
    }

    // --- QLinearAdd / QLinearMul: broadcast slot0, slot3 --------------------
    if (op == "QLinearAdd" || op == "QLinearMul") {
      int32_t ia = input_value(g, n, 0);
      int32_t ib = input_value(g, n, 3);
      const Shape& as =
          ia >= 0 ? g.values[static_cast<size_t>(ia)].shape : emptyShape;
      const Shape& bs =
          ib >= 0 ? g.values[static_cast<size_t>(ib)].shape : emptyShape;
      // out dtype = y_zp(slot7) else a's zp(slot2) else a.
      DType odt = value_dtype(g, input_value(g, n, 7));
      if (odt == DType::Unknown) odt = value_dtype(g, ia);
      if (!as.empty() && !bs.empty()) {
        auto bc = broadcast(as, bs);
        if (bc) { if (set_shape(outv, *bc, odt)) ++resolved; }
        else carry_dtype(outv, odt);
      } else if (!as.empty()) {
        if (set_shape(outv, as, odt)) ++resolved;
      } else if (!bs.empty()) {
        if (set_shape(outv, bs, odt)) ++resolved;
      } else {
        carry_dtype(outv, odt);
      }
      continue;
    }

    // --- QLinearAveragePool: pool math, quantized out dtype -----------------
    if (op == "QLinearAveragePool") {
      int32_t ix = input_value(g, n, 0);
      const Shape& xs =
          ix >= 0 ? g.values[static_cast<size_t>(ix)].shape : emptyShape;
      DType odt = value_dtype(g, input_value(g, n, 4));  // y_zp
      if (odt == DType::Unknown) odt = value_dtype(g, ix);
      if (xs.size() >= 3) {
        size_t spatial = xs.size() - 2;
        auto kernel = attr_ints(model, g, n, "kernel_shape");
        auto strides = attr_ints(model, g, n, "strides");
        auto pads = attr_ints(model, g, n, "pads");
        auto dil = attr_ints(model, g, n, "dilations");
        Shape out; out.push_back(xs[0]); out.push_back(xs[1]);
        bool ok = true;
        for (size_t d = 0; d < spatial; ++d) {
          int64_t in_dim = xs[2 + d];
          if (in_dim < 0) { out.push_back(-1); continue; }
          int64_t k = kernel.size() > d ? kernel[d] : -1;
          if (k < 0) { ok = false; break; }
          int64_t st = strides.size() > d ? strides[d] : 1;
          if (st <= 0) { ok = false; break; }
          int64_t di = dil.size() > d ? dil[d] : 1;
          int64_t p_begin = pads.size() > d ? pads[d] : 0;
          int64_t p_end = pads.size() > spatial + d ? pads[spatial + d] : 0;
          int64_t eff_k = di * (k - 1) + 1;
          out.push_back((in_dim + p_begin + p_end - eff_k) / st + 1);
        }
        if (ok) { if (set_shape(outv, out, odt)) ++resolved; }
        else carry_dtype(outv, odt);
      } else {
        carry_dtype(outv, odt);
      }
      continue;
    }

    // --- QLinearGlobalAveragePool: reduce spatial dims to 1 -----------------
    if (op == "QLinearGlobalAveragePool") {
      int32_t ix = input_value(g, n, 0);
      const Shape& xs =
          ix >= 0 ? g.values[static_cast<size_t>(ix)].shape : emptyShape;
      DType odt = value_dtype(g, input_value(g, n, 4));  // y_zp
      if (odt == DType::Unknown) odt = value_dtype(g, ix);
      if (xs.size() >= 2) {
        Shape out; out.push_back(xs[0]); out.push_back(xs[1]);
        for (size_t d = 2; d < xs.size(); ++d) out.push_back(1);
        if (set_shape(outv, out, odt)) ++resolved;
      } else {
        carry_dtype(outv, odt);
      }
      continue;
    }

    // --- DequantizeLinear: shape-preserving; dtype = scale(slot1) else F32 --
    if (op == "DequantizeLinear") {
      DType odt = value_dtype(g, input_value(g, n, 1));  // scale
      if (odt == DType::Unknown) odt = DType::F32;
      if (!s0.empty()) { if (set_shape(outv, s0, odt)) ++resolved; }
      else carry_dtype(outv, odt);
      continue;
    }

    // --- QuantizeLinear: shape-preserving; dtype = y_zp(slot2) else U8 ------
    if (op == "QuantizeLinear") {
      DType odt = value_dtype(g, input_value(g, n, 2));  // y_zp
      if (odt == DType::Unknown) odt = DType::U8;
      if (!s0.empty()) { if (set_shape(outv, s0, odt)) ++resolved; }
      else carry_dtype(outv, odt);
      continue;
    }

    // --- DynamicQuantizeLinear: out0=in shape U8; out1 scale F32 scalar;
    //     out2 zp U8 scalar (scalars dtype-only) --------------------------
    if (op == "DynamicQuantizeLinear") {
      if (!s0.empty()) { if (set_shape(outv, s0, DType::U8)) ++resolved; }
      else carry_dtype(outv, DType::U8);
      int32_t o1 = output_value(g, n, 1);
      int32_t o2 = output_value(g, n, 2);
      if (o1 >= 0) carry_dtype(g.values[static_cast<size_t>(o1)], DType::F32);
      if (o2 >= 0) carry_dtype(g.values[static_cast<size_t>(o2)], DType::U8);
      continue;
    }

    // Unknown op: never fail. Carry input dtype forward as a weak hint.
    carry_dtype(outv, dt0);
  }

  if (progress) progress->set(1.0f, "shape inference");
  return resolved;
}

}  // namespace netvis
