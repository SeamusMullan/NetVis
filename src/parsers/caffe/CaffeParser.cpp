// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeParser.cpp — Caffe .prototxt / .caffemodel -> ir::Model
// (#138, #109).
//
// Pipeline (all on the parse worker thread; nothing new on the UI thread):
//   sniff -> front-end (CaffeText or CaffeBinary) -> validate -> phase filter
//   -> [prototxt] pair weights from a sibling .caffemodel -> SSA graph build
//   -> infer_shapes_ext (structural, mmap_base = nullptr) -> metadata.
//
// SPEED: weights are never read. A .caffemodel's blob payloads are offset +
// length into the opened file; a paired .prototxt's are offset + length into the
// sibling, resolved lazily by TensorStats exactly like OpenVINO's .bin
// (TensorRef::external_path + resolve_payload under model_dir, with its
// path_within_root confinement).
//
// HONESTY: a Caffe layer is renamed to an ONNX op only where the semantics are
// identical (CaffeOps.h), so shapes are inferred only through those; everything
// downstream of a Caffe-only layer stays unknown. In-place layers (top ==
// bottom) become an SSA chain (conv1 -> conv1#1), never a self-loop. A sibling
// file's problems are never errors for the opened file — they are a note.
#include <cstdint>
#include <exception>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "engine/ShapeInferenceExt.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/caffe/CaffeNet.h"
#include "parsers/caffe/CaffeOps.h"
#include "parsers/caffe/CaffeSniff.h"

namespace netvis::caffe {
namespace {

namespace fs = std::filesystem;

constexpr size_t kMaxDirScan = 4096;
constexpr size_t kMaxListedNames = 16;

std::string plural(size_t n, const char* one, const char* many) {
  return std::to_string(n) + " " + (n == 1 ? one : many);
}

std::string phase_label(int32_t p) {
  if (p == 0) return "TRAIN";
  if (p == 1) return "TEST";
  if (p == -2) return "UNKNOWN";
  return std::to_string(p);
}

std::string layer_label(const LayerRaw& L, size_t i) {
  return L.name.empty() ? "layer_" + std::to_string(i) : L.name;
}

// ---- validation (§3.3) ----------------------------------------------------------------

Result<bool> validate(const NetRaw& net, bool text) {
  if (net.saw_layer && net.saw_layers)
    return Error(
        "Caffe: NetParameter has both 'layer' and 'layers' fields; Caffe refuses to load "
        "this net",
        net.second_kind_offset);
  if (net.layers.empty()) {
    if (text)
      return Error(
          "Caffe prototxt: no 'layer' or 'layers' entries \xE2\x80\x94 not a network "
          "definition (is this a solver file?)",
          0);
    return Error("Caffe caffemodel: no layers", 0);
  }
  return true;
}

// ---- phase filter (§3.4, Net::FilterNet / StateMeetsRule) --------------------------------

bool meets(const NetRaw& net, const std::unordered_set<std::string>& stages, int32_t phase,
           const RuleRaw& r) {
  if (r.phase != -1 && (r.phase == -2 || r.phase != phase)) return false;
  if (r.has_min_level && net.state_level < r.min_level) return false;
  if (r.has_max_level && net.state_level > r.max_level) return false;
  for (const std::string& s : r.stage)
    if (!stages.count(s)) return false;
  for (const std::string& s : r.not_stage)
    if (stages.count(s)) return false;
  return true;
}

// Drops the layers the state excludes; returns the metadata note ("" when no
// layer carries a rule).
std::string apply_phase_filter(NetRaw& net) {
  const int32_t phase = net.has_state ? net.state_phase : 1;   // NetState default: TEST
  const std::unordered_set<std::string> stages(net.state_stages.begin(),
                                               net.state_stages.end());   // lookup only
  bool any_rules = false;
  std::vector<std::string> excluded;
  std::vector<LayerRaw> kept;
  kept.reserve(net.layers.size());
  for (size_t i = 0; i < net.layers.size(); ++i) {
    LayerRaw& L = net.layers[i];
    if (!L.include.empty() || !L.exclude.empty()) any_rules = true;
    bool included = L.include.empty();
    for (size_t j = 0; included && j < L.exclude.size(); ++j)
      if (meets(net, stages, phase, L.exclude[j])) included = false;
    for (size_t j = 0; !included && j < L.include.size(); ++j)
      if (meets(net, stages, phase, L.include[j])) included = true;
    if (included) kept.push_back(std::move(L));
    else excluded.push_back(layer_label(L, i));
  }
  net.layers = std::move(kept);
  if (!any_rules) return {};
  std::string note =
      phase_label(phase) + " (excluded " + plural(excluded.size(), "layer", "layers");
  for (size_t i = 0; i < excluded.size() && i < 8; ++i)
    note += (i == 0 ? ": " : ", ") + excluded[i];
  if (excluded.size() > 8) note += ", ...";
  note += ")";
  return note;
}

// ---- sibling pairing (§3.7) ---------------------------------------------------------------

struct PairInfo {
  std::string note;                  // metadata `weights`
  std::vector<std::string> rejected; // metadata `unpaired_layers`
  std::string sibling;               // basename of the paired file ("" if none)
  uint32_t duplicates = 0;
};

// ASCII case-insensitive ".caffemodel" over any path character type.
template <class S>
bool is_caffemodel_ext(const S& ext) {
  static const char kExt[] = ".caffemodel";
  if (ext.size() != sizeof(kExt) - 1) return false;
  for (size_t i = 0; i < ext.size(); ++i) {
    auto c = ext[i];
    if (c >= 'A' && c <= 'Z') c = static_cast<decltype(c)>(c - 'A' + 'a');
    if (c != static_cast<decltype(c)>(kExt[i])) return false;
  }
  return true;
}

int64_t elem_count(const BlobRaw& b) {
  if (!b.shape_known) return -1;
  int64_t n = 1;
  for (int64_t d : b.shape) {
    if (d < 0) return -1;
    n *= d;   // bounded by sanitize_shape
  }
  return n;
}

int64_t int_attr(const LayerRaw& L, const ir::Model& m, std::string_view n, int64_t def,
                 bool* present = nullptr) {
  const ir::AttrValue* a = layer_attr(L, m, n);
  const bool has = a && a->kind == ir::AttrValue::Kind::Int;
  if (present) *present = has;
  return has ? a->i : def;
}

// Does the sibling's blob list fit this prototxt layer's definition? A layer that
// fails is left without weights and listed, never shown with the wrong ones.
bool verify_pair(const LayerRaw& P, const std::vector<BlobRaw>& blobs, const ir::Model& m) {
  const int want = expected_blob_count(P, m);
  if (want >= 0 && blobs.size() != static_cast<size_t>(want)) return false;
  const std::string& t = P.type;
  if (t == "Convolution" || t == "Deconvolution") {
    if (blobs.empty()) return false;
    const BlobRaw& w = blobs[0];
    if (!w.shape_known || w.shape.size() < 3) return false;
    const ConvGeometry g = resolve_conv_geometry(P, m, static_cast<int>(w.shape.size()));
    if (!g.ok) return false;
    if (g.has_num_output) {
      if (t == "Convolution") {
        if (w.shape[0] != g.num_output) return false;
      } else if (w.shape[1] < 0 || w.shape[1] * g.group != g.num_output) {
        return false;   // w[1] <= INT32_MAX and group <= kMaxGeometry: no overflow
      }
    }
    for (size_t d = 0; d < g.kernel.size(); ++d)
      if (w.shape[2 + d] != g.kernel[d]) return false;
    if (blobs.size() > 1 && g.has_num_output && elem_count(blobs[1]) != g.num_output)
      return false;
    return true;
  }
  if (t == "InnerProduct") {
    if (blobs.empty()) return false;
    const BlobRaw& w = blobs[0];
    if (!w.shape_known || w.shape.size() < 2) return false;
    bool has_n = false;
    const int64_t n = int_attr(P, m, "inner_product_param.num_output", 0, &has_n);
    const bool transpose = int_attr(P, m, "inner_product_param.transpose", 0) != 0;
    const size_t r = w.shape.size();
    if (has_n) {
      // Right-aligned (Caffe's LegacyShape): legacy [1,1,N,K] and modern [N,K] pass.
      if (w.shape[transpose ? r - 1 : r - 2] != n) return false;
      if (blobs.size() > 1 && elem_count(blobs[1]) != n) return false;
    }
    return true;
  }
  return true;
}

// Rules 1-3. Returns an empty path and sets `note` when there is no single match.
fs::path find_sibling(const fs::path& model_path, std::string& note) {
  std::error_code ec;
  const fs::path dir = model_path.parent_path();
  const std::string stem = model_path.stem().string();

  // Rule 1: <stem>.caffemodel.
  fs::path c1 = dir / stem;
  c1 += ".caffemodel";
  if (fs::is_regular_file(c1, ec)) return c1;

  // Rule 2: <stem minus _deploy / -deploy / .deploy>.caffemodel (case-sensitive).
  for (const char* suf : {"_deploy", "-deploy", ".deploy"}) {
    const std::string_view sv(suf);
    if (stem.size() > sv.size() &&
        stem.compare(stem.size() - sv.size(), sv.size(), sv) == 0) {
      fs::path c2 = dir / stem.substr(0, stem.size() - sv.size());
      c2 += ".caffemodel";
      ec.clear();
      if (fs::is_regular_file(c2, ec)) return c2;
    }
  }

  // Rule 3: the only .caffemodel in the folder — from a COMPLETE scan only, so
  // the answer never depends on directory iteration order.
  size_t seen = 0, found = 0;
  fs::path only;
  bool complete = true;
  ec.clear();
  fs::directory_iterator it(dir.empty() ? fs::path(".") : dir, ec);
  const fs::directory_iterator end;
  for (; !ec && it != end; it.increment(ec)) {
    if (++seen > kMaxDirScan) {
      complete = false;
      break;
    }
    std::error_code fec;
    if (!it->is_regular_file(fec)) continue;
    if (is_caffemodel_ext(it->path().extension().native())) {
      ++found;
      only = it->path();
    }
  }
  if (ec) complete = false;
  if (complete && found == 1) return only;
  if (complete && found >= 2)
    note = "ambiguous: " + std::to_string(found) +
           " .caffemodel files beside this one, none named " + stem + ".caffemodel";
  else
    note = "none found";
  return {};
}

void pair_sibling_weights(const std::string& path, NetRaw& net, const ir::Model& model,
                          PairInfo& info) {
  // Caffe's CopyTrainedLayersFrom matches by layer name; on duplicates the first
  // occurrence wins (and the rest are counted).
  std::unordered_map<std::string, size_t> first_index;   // lookup only
  for (size_t i = 0; i < net.layers.size(); ++i)
    if (!first_index.emplace(net.layers[i].name, i).second) ++info.duplicates;

  if (path.empty()) {
    info.note = "none found";
    return;
  }
  const fs::path sib = find_sibling(fs::path(path), info.note);
  if (sib.empty()) return;
  const std::string base = sib.filename().string();

  // The mapping is local to this function; TensorStats maps the sibling again on
  // demand through external_path, exactly as OpenVINO's .bin.
  auto mf = MappedFile::open(sib.string());
  if (!mf) {
    info.note = base + " could not be read: " + mf.error().message;
    return;
  }
  if (!looks_like_caffemodel(mf->data(), mf->size())) {
    info.note = base + " is not a Caffe binary NetParameter";
    return;
  }
  auto sr = read_caffemodel(mf->data(), mf->size(), nullptr, ReadMode::BlobsOnly, nullptr);
  if (!sr) {
    info.note = base + " could not be read: " + sr.error().message + " (at byte " +
                std::to_string(sr.error().offset) + ")";
    return;
  }

  std::vector<uint8_t> attempted(net.layers.size(), 0);
  size_t paired = 0, not_here = 0;
  for (LayerRaw& S : sr->layers) {
    if (S.blobs.empty()) continue;   // blob-less layers (relu1, ...) count nowhere
    auto it = first_index.find(S.name);
    if (it == first_index.end()) {
      ++not_here;
      continue;
    }
    const size_t idx = it->second;
    if (attempted[idx]) continue;
    LayerRaw& P = net.layers[idx];
    if (!P.blobs.empty()) continue;   // inline text blobs win
    attempted[idx] = 1;
    if (!verify_pair(P, S.blobs, model)) {
      info.rejected.push_back(layer_label(P, idx));
      continue;
    }
    for (BlobRaw& b : S.blobs) {
      b.external = true;
      if (b.unpacked) ++net.unpacked_blobs;
    }
    P.blobs = std::move(S.blobs);
    ++paired;
  }
  info.sibling = base;
  info.note = "paired with " + base + ": " + plural(paired, "layer", "layers");
  if (!info.rejected.empty()) info.note += "; " + std::to_string(info.rejected.size()) + " rejected";
  if (not_here > 0)
    info.note += "; " + plural(not_here, "sibling layer", "sibling layers") + " not in this net";
}

// ---- graph build (§3.5, §3.6, §3.8, §3.10) ------------------------------------------------

struct BuildInfo {
  std::string activation_note;
  bool input_shapes_inconsistent = false;
  size_t dangling = 0;
};

bool is_data_layer(const std::string& t) {
  return t == "Input" || t == "Data" || t == "ImageData" || t == "MemoryData" ||
         t == "HDF5Data" || t == "WindowData" || t == "DummyData";
}

ir::DType activation_dtype(const NetRaw& net, std::string& note) {
  bool f32 = false, f64 = false;
  for (const LayerRaw& L : net.layers) {
    for (const BlobRaw& b : L.blobs) {
      if (!(b.addressable || b.text)) continue;
      if (b.dtype == ir::DType::F32) f32 = true;
      else if (b.dtype == ir::DType::F64) f64 = true;
    }
  }
  if (f32 && !f64) {
    note = "f32 (from weight blob storage)";
    return ir::DType::F32;
  }
  if (f64 && !f32) {
    note = "f64 (from weight blob storage)";
    return ir::DType::F64;
  }
  note = (f32 && f64) ? "unknown (mixed blob storage)" : "unknown (no weights)";
  return ir::DType::Unknown;
}

void build_graph(const NetRaw& net, ir::Model& model, const std::string& sibling,
                 BuildInfo& info) {
  ir::Graph g;
  g.name = model.intern(net.name.empty() ? std::string("net") : net.name);
  const ir::DType act = activation_dtype(net, info.activation_note);
  const StringId sibling_id = sibling.empty() ? StringId{} : model.intern(sibling);

  // Exact-string sets/maps, lookup only (never iterated for output).
  std::unordered_set<std::string> reserved, used;
  std::unordered_map<std::string, uint32_t> current;
  std::unordered_map<std::string, uint32_t> version;
  for (const std::string& s : net.inputs) reserved.insert(s);
  for (const LayerRaw& L : net.layers) {
    for (const std::string& s : L.bottom) reserved.insert(s);
    for (const std::string& s : L.top) reserved.insert(s);
  }
  // fresh(base): `base` if free, else the first free `base#2`, `base#3`, ...
  // The search resumes where the last one for the same base stopped (`used` only
  // grows, so a rejected candidate stays rejected): a file repeating one layer
  // name N times costs O(N), not O(N^2).
  std::unordered_map<std::string, uint64_t> next_suffix;
  auto fresh = [&](const std::string& base) -> std::string {
    if (!used.count(base) && !reserved.count(base)) return base;
    uint64_t& k = next_suffix[base];
    if (k < 2) k = 2;
    for (;; ++k) {
      std::string cand = base + "#" + std::to_string(k);
      if (!used.count(cand) && !reserved.count(cand)) return cand;
    }
  };
  auto add_value = [&](const std::string& name, int32_t producer) -> uint32_t {
    ir::ValueInfo v;
    v.name = model.intern(name);
    v.producer = producer;
    used.insert(name);
    g.values.push_back(std::move(v));
    return static_cast<uint32_t>(g.values.size() - 1);
  };

  std::vector<uint32_t> net_inputs, data_tops, dangling;

  // 1. Net-level inputs (graph-input values; no synthetic Input node).
  const size_t ni = net.inputs.size();
  const bool by_shape = ni > 0 && net.input_shapes.size() == ni;
  const bool by_dim = !by_shape && ni > 0 && net.input_dims.size() == 4 * ni;
  if (!by_shape && !by_dim && (!net.input_shapes.empty() || !net.input_dims.empty()))
    info.input_shapes_inconsistent = true;
  for (size_t i = 0; i < ni; ++i) {
    const std::string& name = net.inputs[i];
    if (current.count(name)) continue;   // a repeated input name is one value
    const uint32_t vi = add_value(name, -1);
    current[name] = vi;
    ir::ValueInfo& v = g.values[vi];
    v.dtype = act;
    if (by_shape && net.input_shape_ok[i]) {
      v.shape = net.input_shapes[i];
    } else if (by_dim) {
      SmallVec<int64_t, 6> dims;
      for (size_t k = 0; k < 4; ++k) dims.push_back(net.input_dims[4 * i + k]);
      if (sanitize_shape(dims)) v.shape = dims;
    }
    net_inputs.push_back(vi);
  }

  // 2. Layers, in file order (after the phase filter).
  for (size_t li = 0; li < net.layers.size(); ++li) {
    const LayerRaw& L = net.layers[li];
    const uint32_t node_idx = static_cast<uint32_t>(g.nodes.size());
    const std::string node_name = layer_label(L, li);
    ir::Node node;
    node.name = model.intern(node_name);

    // Inputs: every bottom is read BEFORE any top is written, which is what turns
    // an in-place layer into conv1 -> conv1#1 instead of a self-loop.
    std::vector<uint32_t> ins, outs;
    for (const std::string& b : L.bottom) {
      auto it = current.find(b);
      if (it != current.end()) {
        ins.push_back(it->second);
        continue;
      }
      const uint32_t vi = add_value(b, -1);   // dangling: never produced, not declared
      current[b] = vi;
      dangling.push_back(vi);
      ins.push_back(vi);
    }

    // Params: one value + one initializer per blob, after all bottoms.
    for (size_t k = 0; k < L.blobs.size(); ++k) {
      const BlobRaw& b = L.blobs[k];
      const std::string vname = fresh(node_name + "/" + blob_role(L, k));
      const uint32_t vi = add_value(vname, -1);
      ir::ValueInfo& v = g.values[vi];
      v.dtype = b.dtype;
      if (b.shape_known) v.shape = b.shape;
      ins.push_back(vi);

      ir::TensorRef t;
      t.name = v.name;
      t.dtype = b.dtype;
      if (b.shape_known) t.shape = b.shape;
      if (b.addressable) {
        t.file_offset = b.offset;
        t.byte_len = b.byte_len;
        if (b.external) t.external_path = sibling_id;
      } else {
        // LOAD-BEARING: resolve_payload reads "UINT64_MAX + external path" as a
        // payload starting at byte 0 of the external file, so a non-addressable
        // blob must carry NO external path, even when it came from the sibling.
        t.file_offset = UINT64_MAX;
        t.byte_len = 0;
      }
      g.initializers.push_back(std::move(t));
    }

    // Outputs.
    for (const std::string& t : L.top) {
      auto it = current.find(t);
      std::string vname;
      if (it != current.end()) {
        const uint32_t ver = ++version[t];
        vname = fresh(t + "#" + std::to_string(ver));
      } else {
        vname = t;
      }
      const uint32_t vi = add_value(vname, static_cast<int32_t>(node_idx));
      current[t] = vi;
      outs.push_back(vi);
    }

    // Declared shapes of Input layer tops (InputLayer::LayerSetUp): one shape
    // for every top, or one per top; anything else is unknown.
    if (L.type == "Input" && !L.v1 && !L.v0) {
      const size_t ns = L.input_param_shapes.size();
      if (ns == 1 || (ns == outs.size() && ns > 0)) {
        for (size_t k = 0; k < outs.size(); ++k) {
          const size_t si = ns == 1 ? 0 : k;
          if (L.input_param_shape_ok[si]) g.values[outs[k]].shape = L.input_param_shapes[si];
        }
      } else if (L.has_input_param || !outs.empty()) {
        info.input_shapes_inconsistent = true;
      }
    }
    if (is_data_layer(L.type) && !L.v0) {
      for (uint32_t vi : outs) {
        g.values[vi].dtype = act;
        data_tops.push_back(vi);
      }
    }

    // Op mapping + attributes: derived (vetted ops only), caffe.type,
    // caffe.v1_type (V1 only), then the native attributes in source order.
    const int weight_rank = (!L.blobs.empty() && L.blobs[0].shape_known)
                                ? static_cast<int>(L.blobs[0].shape.size())
                                : -1;
    const MappedOp mapped = map_layer(L, model, weight_rank);
    node.op_type = model.intern(mapped.op_type);
    node.attributes.begin = static_cast<uint32_t>(g.attributes.size());
    for (const DerivedAttr& d : mapped.derived) {
      ir::Attribute a;
      a.name = model.intern(d.name);
      a.value.kind = d.kind;
      a.value.i = d.i;
      a.value.f = d.f;
      a.value.ints = d.ints;
      g.attributes.push_back(std::move(a));
    }
    {
      ir::Attribute a;
      a.name = model.intern("caffe.type");
      a.value.kind = ir::AttrValue::Kind::String;
      a.value.s = model.intern(L.type);
      g.attributes.push_back(std::move(a));
    }
    if (L.v1) {
      ir::Attribute a;
      a.name = model.intern("caffe.v1_type");
      a.value.kind = ir::AttrValue::Kind::String;
      a.value.s = model.intern(L.v1_type_ident);
      g.attributes.push_back(std::move(a));
    }
    for (const ir::Attribute& a : L.attrs) g.attributes.push_back(a);
    node.attributes.count =
        static_cast<uint32_t>(g.attributes.size()) - node.attributes.begin;

    node.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    for (uint32_t vi : ins) g.edge_refs.push_back(vi);
    node.inputs.count = static_cast<uint32_t>(ins.size());
    node.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
    for (uint32_t vi : outs) g.edge_refs.push_back(vi);
    node.outputs.count = static_cast<uint32_t>(outs.size());
    g.nodes.push_back(std::move(node));
  }

  // 3. graph_inputs: net inputs, then Input/data-layer tops, then dangling values.
  std::vector<uint8_t> listed(g.values.size(), 0);
  for (const std::vector<uint32_t>* group : {&net_inputs, &data_tops, &dangling}) {
    for (uint32_t vi : *group) {
      if (listed[vi]) continue;
      listed[vi] = 1;
      g.graph_inputs.push_back(vi);
    }
  }
  info.dangling = dangling.size();

  // 4. graph_outputs: produced values no node consumes, ascending value index.
  std::vector<uint8_t> consumed(g.values.size(), 0);
  for (const ir::Node& n : g.nodes)
    for (uint32_t k = 0; k < n.inputs.count; ++k) consumed[g.edge_refs[n.inputs.begin + k]] = 1;
  for (uint32_t vi = 0; vi < g.values.size(); ++vi)
    if (g.values[vi].producer >= 0 && !consumed[vi]) g.graph_outputs.push_back(vi);

  model.graphs.clear();
  model.graphs.push_back(std::move(g));
}

// ---- metadata (§3.11) -------------------------------------------------------------------

void put(ir::Model& m, const char* key, const std::string& value) {
  if (value.empty()) return;
  m.metadata.emplace_back(m.intern(key), m.intern(value));
}

void put_count(ir::Model& m, const char* key, uint64_t n, const char* suffix) {
  if (n == 0) return;
  put(m, key, std::to_string(n) + suffix);
}

}  // namespace

// Caffe entry point (declared in parsers/Parser.h). Reads structure only.
Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "Parsing Caffe");
  ir::Model model;
  model.format_name = model.intern("Caffe");
  model.has_graph = true;
  if (!file.valid() || file.data() == nullptr) return err("empty or unmapped file", 0);
  const uint8_t* d = file.data();
  const uint64_t n = file.size();

  // The same sniffs as detection, so the two can never disagree. A file that
  // matches neither was routed here by its extension; text gets text errors
  // (e.g. the "is this a solver file?" hint).
  const bool text = looks_like_prototxt(d, n)    ? true
                    : looks_like_caffemodel(d, n) ? false
                                                  : is_text_prefix(d, n);
  progress.set(0.2f, text ? "Reading prototxt" : "Reading caffemodel");
  auto nr = text ? read_prototxt(d, n, model, &progress)
                 : read_caffemodel(d, n, &model, ReadMode::Full, &progress);
  if (!nr) return nr.error();
  NetRaw net = nr.take();

  auto ok = validate(net, text);
  if (!ok) return ok.error();
  const std::string phase_note = apply_phase_filter(net);

  PairInfo pair;
  if (text) {
    progress.set(0.5f, "Pairing weights");
    try {
      pair_sibling_weights(file.path(), net, model, pair);
    } catch (const std::exception& e) {
      // std::filesystem path conversions are the only throwers here (bad_alloc
      // aside); pairing is never fatal for the opened file.
      pair = PairInfo{};
      pair.note = std::string("sibling lookup failed: ") + e.what();
    }
  }

  progress.set(0.7f, "Building graph");
  BuildInfo info;
  build_graph(net, model, pair.sibling, info);

  progress.set(0.9f, "Shape inference");
  // mmap_base = nullptr: the constant-reading path is off, so inference cannot
  // touch a byte of the file (Caffe has no shape-constant initializers anyway).
  infer_shapes_ext(model, 0, nullptr, 0, nullptr);

  const char* schema_name = net.gen == Generation::V1 ? "V1LayerParameter" : "LayerParameter";
  model.version_info = model.intern(std::string("Caffe ") + (text ? "prototxt" : "caffemodel") +
                                    ", " + schema_name);
  put(model, "name", net.name);
  std::string schema = net.gen == Generation::V1 ? "V1LayerParameter (layers)"
                                                 : "LayerParameter (layer)";
  if (net.v0_layers > 0)
    schema += "; " + std::to_string(net.v0_layers) + " V0 layers (parameters not decoded)";
  put(model, "schema", schema);
  put(model, "source", text ? "prototxt (text)" : "caffemodel (binary)");
  if (text) {
    put(model, "weights", pair.note);
    std::string unpaired;
    for (size_t i = 0; i < pair.rejected.size() && i < kMaxListedNames; ++i)
      unpaired += (i == 0 ? "" : ", ") + pair.rejected[i];
    if (pair.rejected.size() > kMaxListedNames)
      unpaired += " (+" + std::to_string(pair.rejected.size() - kMaxListedNames) + " more)";
    put(model, "unpaired_layers", unpaired);
  } else {
    size_t blobs = 0;
    for (const LayerRaw& L : net.layers) blobs += L.blobs.size();
    put(model, "weights", "embedded (" + plural(blobs, "blob", "blobs") + ")");
  }
  put(model, "phase_filter", phase_note);
  put(model, "activation_dtype", info.activation_note);
  if (info.input_shapes_inconsistent)
    put(model, "input_shapes", "inconsistent declarations; left unknown");
  put_count(model, "dangling_inputs", info.dangling, "");
  put_count(model, "text_blobs", net.text_blobs, " (inline text payloads are not addressable)");
  put_count(model, "unpacked_blobs", net.unpacked_blobs,
            " (payload not addressable; later fields not read)");
  put_count(model, "undecoded_fields", net.undecoded_fields, " binary fields not decoded");
  put_count(model, "ignored_fields", net.ignored_top_level,
            " unknown top-level NetParameter fields");
  put_count(model, "oversized_shapes", net.oversized_shapes,
            " shapes exceed Caffe's INT_MAX element limit; left unknown");
  put_count(model, "blob_size_mismatch", net.blob_size_mismatch,
            " blobs whose shape and payload length disagree");
  put_count(model, "duplicate_layer_names", pair.duplicates, "");

  progress.set(1.0f, "Done");
  return model;
}

}  // namespace netvis::caffe
