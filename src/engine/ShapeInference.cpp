// engine/ShapeInference.cpp — best-effort ONNX shape/dtype propagation.
//
// DECISION (spec §7.3): propagate shapes for the common ops so edge labels fill
// in. Best-effort: unknown ops / unknown inputs leave outputs Unknown and never
// fail. We walk nodes in topological order (producer-before-consumer) and mutate
// ValueInfo.shape/dtype in place.
//
// THREADING: this mutates a snapshot the main thread is NOT reading; the
// enriched model is republished on job completion (generation-checked). No
// synchronization is needed here because we hold exclusive ownership.
#include "engine/ShapeInference.h"

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

DType value_dtype(const Graph& g, int32_t vi) {
  if (vi < 0) return DType::Unknown;
  return g.values[static_cast<size_t>(vi)].dtype;
}

// Try to read a small 1-D int64 shape initializer for Reshape/Expand targets.
// DECISION (spec §7.3): we MAY read this ONE tiny initializer's bytes via a
// bounds-checked ByteReader — it is shape metadata, not tensor payload, and is
// capped at a small element count so we never touch real weights.
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

// Compute the topological order of nodes (producer before consumer). Falls back
// to natural order for cycles / missing producers.
std::vector<uint32_t> topo_order(const Graph& g) {
  size_t nn = g.nodes.size();
  std::vector<uint32_t> order;
  order.reserve(nn);
  std::vector<uint8_t> state(nn, 0);  // 0=unvisited,1=on-stack,2=done

  // Iterative DFS to avoid stack overflow on deep graphs.
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
        // state==1 (on stack) => cycle; just skip.
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

uint32_t infer_shapes(Model& model, uint32_t graph_index,
                      ProgressSink* progress) {
  if (graph_index >= model.graphs.size()) return 0;
  Graph& g = model.graphs[graph_index];

  // We have no mmap handle here; shape initializers we can decode are the tiny
  // constant ones stored inline in the graph's initializers with a valid local
  // offset. Without a base pointer we simply skip that decode path. The base
  // may be provided in a future overload; for now we only use attribute + shape
  // propagation and initializer shapes recorded in TensorRef::shape.
  const uint8_t* base = nullptr;
  uint64_t base_size = 0;

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
        op == "Equal" || op == "And" || op == "Or" || op == "Greater" ||
        op == "Less" || op == "Mod") {
      // Where has 3 inputs (cond,x,y); use x,y for shape/dtype.
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
          op == "Or")
        odt = DType::Bool;
      if (!a->empty() && !b->empty()) {
        auto bc = broadcast(*a, *b);
        if (bc) { if (set_shape(outv, *bc, odt)) ++resolved; }
      } else if (!a->empty()) {
        if (set_shape(outv, *a, odt)) ++resolved;
      } else if (!b->empty()) {
        if (set_shape(outv, *b, odt)) ++resolved;
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
        op == "Round" || op == "Sign") {
      if (!s0.empty()) { if (set_shape(outv, s0, dt0)) ++resolved; }
      else if (dt0 != DType::Unknown && outv.dtype == DType::Unknown) {
        outv.dtype = dt0;
      }
      continue;
    }

    // --- Cast: shape preserved, dtype changes via `to` ----------------------
    if (op == "Cast") {
      DType nd = dt0;
      int64_t to = attr_int(model, g, n, "to", -1);
      // ONNX TensorProto.DataType numeric codes.
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
      else if (nd != DType::Unknown && outv.dtype == DType::Unknown)
        outv.dtype = nd;
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
      }
      continue;
    }

    // --- MatMul: batched, last two dims are matrix dims ---------------------
    if (op == "MatMul") {
      if (s0.size() >= 1 && s1.size() >= 1) {
        // Handle 1-D promotions per numpy matmul rules.
        Shape a = s0, b = s1;
        bool a1 = a.size() == 1, b1 = b.size() == 1;
        if (a1) { Shape t; t.push_back(1); t.push_back(a[0]); a = t; }
        if (b1) { Shape t; t.push_back(b[0]); t.push_back(1); b = t; }
        if (a.size() >= 2 && b.size() >= 2) {
          int64_t M = a[a.size() - 2];
          int64_t N = b[b.size() - 1];
          // Broadcast the batch dims.
          Shape ab, bb;
          for (size_t i = 0; i + 2 < a.size(); ++i) ab.push_back(a[i]);
          for (size_t i = 0; i + 2 < b.size(); ++i) bb.push_back(b[i]);
          Shape batch;
          auto bc = broadcast(ab, bb);
          if (bc) batch = *bc;
          Shape s = batch;
          if (!a1) s.push_back(M);
          if (!b1) s.push_back(N);
          if (set_shape(outv, s, dt0)) ++resolved;
        }
      }
      continue;
    }

    // --- Conv / ConvTranspose (NCHW) ----------------------------------------
    if (op == "Conv" || op == "ConvTranspose") {
      // input: (N, Cin, D...), weight: Conv (Cout, Cin/g, k...),
      //        ConvTranspose (Cin, Cout/g, k...).
      if (s0.size() >= 3 && s1.size() >= 2) {
        size_t spatial = s0.size() - 2;
        auto kernel = attr_ints(model, g, n, "kernel_shape");
        auto strides = attr_ints(model, g, n, "strides");
        auto pads = attr_ints(model, g, n, "pads");
        auto dil = attr_ints(model, g, n, "dilations");
        auto out_pad = attr_ints(model, g, n, "output_padding");
        int64_t group = attr_int(model, g, n, "group", 1);

        Shape out; out.push_back(s0[0]);  // N
        int64_t cout;
        if (op == "Conv") cout = s1[0];
        else cout = s1[1] * group;  // ConvTranspose: (Cin, Cout/g, ...)
        out.push_back(cout);

        bool ok = true;
        for (size_t d = 0; d < spatial; ++d) {
          int64_t in_dim = s0[2 + d];
          if (in_dim < 0) { out.push_back(-1); continue; }
          int64_t k = kernel.size() > d ? kernel[d]
                     : (s1.size() > 2 + d ? s1[2 + d] : -1);
          if (k < 0) { ok = false; break; }
          int64_t st = strides.size() > d ? strides[d] : 1;
          // Stride comes from an attacker-controlled attribute; a zero/negative
          // stride would divide-by-zero (SIGFPE). Best-effort inference bails on
          // this dimension rather than crashing (spec §7.3: never fail hard).
          if (st <= 0) { ok = false; break; }
          int64_t di = dil.size() > d ? dil[d] : 1;
          int64_t p_begin = pads.size() > d ? pads[d] : 0;
          int64_t p_end = pads.size() > spatial + d ? pads[spatial + d] : 0;
          int64_t eff_k = di * (k - 1) + 1;
          int64_t o;
          if (op == "Conv") {
            o = (in_dim + p_begin + p_end - eff_k) / st + 1;
          } else {
            int64_t op_pad = out_pad.size() > d ? out_pad[d] : 0;
            o = st * (in_dim - 1) + op_pad + eff_k - p_begin - p_end;
          }
          out.push_back(o);
        }
        if (ok && out.size() == s0.size()) {
          if (set_shape(outv, out, dt0)) ++resolved;
        }
      }
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
          if (st <= 0) { ok = false; break; }  // avoid div-by-zero (SIGFPE)
          int64_t di = dil.size() > d ? dil[d] : 1;
          int64_t p_begin = pads.size() > d ? pads[d] : 0;
          int64_t p_end = pads.size() > spatial + d ? pads[spatial + d] : 0;
          int64_t eff_k = di * (k - 1) + 1;
          out.push_back((in_dim + p_begin + p_end - eff_k) / st + 1);
        }
        if (ok) { if (set_shape(outv, out, dt0)) ++resolved; }
      }
      continue;
    }

    if (op == "GlobalAveragePool" || op == "GlobalMaxPool") {
      if (s0.size() >= 2) {
        Shape out; out.push_back(s0[0]); out.push_back(s0[1]);
        for (size_t d = 2; d < s0.size(); ++d) out.push_back(1);
        if (set_shape(outv, out, dt0)) ++resolved;
      }
      continue;
    }

    // --- Transpose: perm (default: reverse dims) ----------------------------
    if (op == "Transpose") {
      if (!s0.empty()) {
        auto perm = attr_ints(model, g, n, "perm");
        Shape out;
        if (perm.empty()) {
          for (size_t i = s0.size(); i > 0; --i) out.push_back(s0[i - 1]);
        } else {
          bool ok = true;
          for (int64_t p : perm) {
            if (p < 0 || static_cast<size_t>(p) >= s0.size()) { ok = false; break; }
            out.push_back(s0[static_cast<size_t>(p)]);
          }
          if (!ok) { continue; }
        }
        if (set_shape(outv, out, dt0)) ++resolved;
      }
      continue;
    }

    // --- Concat: sum along axis ---------------------------------------------
    if (op == "Concat") {
      int64_t axis = attr_int(model, g, n, "axis", 0);
      // Gather all input shapes; require rank known.
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
      }
      continue;
    }

    // --- Reshape: decode small constant shape initializer -------------------
    if (op == "Reshape") {
      std::optional<std::vector<int64_t>> target;
      // Shape is the second input; try to find it as an initializer.
      if (in1 >= 0) {
        std::string_view sname = model.str(g.values[static_cast<size_t>(in1)].name);
        for (const ir::TensorRef& init : g.initializers) {
          if (model.str(init.name) == sname && init.name.valid()) {
            target = read_shape_initializer(model, g, init, base, base_size);
            break;
          }
        }
      }
      if (target && !s0.empty()) {
        // Handle 0 (copy-dim) and -1 (infer).
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
        // No input shape but explicit static target still gives us dims.
        Shape out; bool any_neg = false;
        for (int64_t d : *target) { out.push_back(d); if (d < 0) any_neg = true; }
        if (!any_neg && set_shape(outv, out, dt0)) ++resolved;
      }
      continue;
    }

    // --- Squeeze / Unsqueeze ------------------------------------------------
    if (op == "Squeeze") {
      if (!s0.empty()) {
        auto axes = attr_ints(model, g, n, "axes");  // opset<13
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
      }
      continue;
    }
    if (op == "Unsqueeze") {
      if (!s0.empty()) {
        auto axes = attr_ints(model, g, n, "axes");
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
        }
      }
      continue;
    }

    // --- Reduce* (keepdims / axes) ------------------------------------------
    if (op.substr(0, 6) == "Reduce" || op == "ArgMax" || op == "ArgMin") {
      if (!s0.empty()) {
        auto axes = attr_ints(model, g, n, "axes");
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
      }
      continue;
    }

    // --- Pad: adds pads to spatial (best-effort, needs static pads attr) -----
    if (op == "Pad") {
      if (!s0.empty()) {
        auto pads = attr_ints(model, g, n, "pads");  // opset<11
        if (!pads.empty() && pads.size() == s0.size() * 2) {
          Shape out;
          for (size_t i = 0; i < s0.size(); ++i) {
            int64_t d = s0[i];
            if (d < 0) { out.push_back(-1); continue; }
            out.push_back(d + pads[i] + pads[i + s0.size()]);
          }
          if (set_shape(outv, out, dt0)) ++resolved;
        } else {
          // dynamic pads: keep rank/dtype only if fully unknown
          if (outv.dtype == DType::Unknown) outv.dtype = dt0;
        }
      }
      continue;
    }

    // --- Softmax family already covered; keep-shape ops covered above. -------

    // --- Fallbacks that just carry dtype (shape genuinely unknown) ----------
    if (op == "Shape") {
      // Output is 1-D int64 of length = rank(input).
      Shape out;
      if (!s0.empty()) out.push_back(static_cast<int64_t>(s0.size()));
      if (set_shape(outv, out, DType::I64)) ++resolved;
      else if (outv.dtype == DType::Unknown) outv.dtype = DType::I64;
      continue;
    }

    // Gather/Slice/Expand/Resize: leave Unknown unless already set (we avoid
    // guessing without index/scale tensors). Propagate dtype only.
    if (op == "Gather" || op == "GatherElements" || op == "Slice" ||
        op == "Expand" || op == "Resize" || op == "Split" || op == "Tile") {
      if (outv.dtype == DType::Unknown && dt0 != DType::Unknown)
        outv.dtype = dt0;
      continue;
    }

    // Unknown op: never fail. Carry input dtype forward as a weak hint.
    if (outv.dtype == DType::Unknown && dt0 != DType::Unknown)
      outv.dtype = dt0;
  }

  if (progress) progress->set(1.0f, "shape inference");
  return resolved;
}

}  // namespace netvis
