// parsers/coreml/CoreMlParser.cpp — CoreML .mlmodel (protobuf) -> ir::Model.
//
// DECISION (v0.5.0 plan §"#38 — CoreML .mlmodel"): a .mlmodel is a bare CoreML
// `Model` protobuf. We reuse the ONNX hand-rolled wire reader verbatim
// (onnx::WireReader is bounds-checked and NOT ONNX-specific in substance) and
// walk the message structurally — recording weight payloads only as absolute
// mmap offset+length via length-delimited sub-ranges we never read. All reads go
// through the bounds-checked WireReader/ByteReader, so a malformed/truncated file
// yields a Result error carrying a byte offset instead of crashing. Unknown
// fields/types are skipped (forward-compatible), never fatal.
//
// Mapping: the top-level `oneof Type` selects the model kind. `neuralNetwork`
// (and the classifier/regressor variants, which carry `layers` at the same field
// number) become a compute graph: each NeuralNetworkLayer is one ir::Node, edges
// wired from its input/output name lists (same name->ValueInfo scheme as ONNX),
// and every WeightParams sub-message found inside the layer-kind body is recorded
// as an initializer TensorRef (offset+len only). Exotic model types (mlProgram,
// pipeline, tree/GLM/SVM, ...) fall back to has_graph=false + a metadata note —
// never an error. `.mlpackage` directory bundles are out of scope (single-file
// `.mlmodel` only, per the milestone non-goals).
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/onnx/WireReader.h"

namespace netvis::coreml {
namespace {

// Reuse the ONNX protobuf wire reader (bounds-checked, absolute-offset). It is
// not ONNX-specific in substance — only in namespace.
using onnx::SubRange;
using onnx::WireReader;
using onnx::WireType;

// --- Hardening bounds (adversarial input) -----------------------------------
constexpr uint32_t kMaxLayers = 5'000'000;      // bounded layer count
constexpr int kMaxWeightDepth = 16;             // nested layer-body recursion cap
constexpr uint32_t kMaxWeightsPerLayer = 256;   // bounded weights per layer

// --- CoreML field numbers (verified against apple/coremltools
// mlmodel/format/{Model,NeuralNetwork}.proto) ---------------------------------
//
// Model: 1 specificationVersion (varint), 2 description (ModelDescription),
// 10 isUpdatable, and a `oneof Type` at high field numbers (below).
enum ModelField : uint32_t {
  MF_SPEC_VERSION = 1,
  MF_DESCRIPTION = 2,
  // oneof Type — neuralNetwork family (all expose `layers` at field 1):
  MF_NEURAL_NETWORK = 500,
  MF_NN_CLASSIFIER = 403,
  MF_NN_REGRESSOR = 303,
};

// NeuralNetwork(/Classifier/Regressor): repeated NeuralNetworkLayer at field 1.
constexpr uint32_t kNnLayersField = 1;

// NeuralNetworkLayer: 1 name, 2 input(rep str), 3 output(rep str),
// 4 inputTensor, 5 outputTensor, 10 isUpdatable, and a `oneof layer` at fields
// >= 100 selecting the layer kind.
enum LayerField : uint32_t {
  LF_NAME = 1,
  LF_INPUT = 2,
  LF_OUTPUT = 3,
  LF_KIND_MIN = 100,  // oneof layer starts here
};

// Human-readable op name for the common `oneof layer` field numbers. Anything
// unmapped surfaces as "CoreMLLayer_<field>" so it never crashes and still
// categorizes as Other.
const char* layer_kind_name(uint32_t field) {
  switch (field) {
    case 100: return "Convolution";
    case 120: return "Pooling";
    case 130: return "Activation";
    case 140: return "InnerProduct";
    case 150: return "Embedding";
    case 160: return "BatchNorm";
    case 165: return "MeanVarianceNormalize";
    case 175: return "Softmax";
    case 180: return "LRN";
    case 190: return "Crop";
    case 200: return "Padding";
    case 210: return "Upsample";
    case 220: return "Unary";
    case 230: return "Add";
    case 231: return "Multiply";
    case 240: return "Average";
    case 245: return "Scale";
    case 250: return "Bias";
    case 260: return "Max";
    case 261: return "Min";
    case 270: return "Dot";
    case 280: return "ReduceLayer";
    case 290: return "LoadConstant";
    case 300: return "Reshape";
    case 301: return "Flatten";
    case 310: return "Permute";
    case 320: return "Concat";
    case 330: return "Split";
    case 340: return "SequenceRepeat";
    case 400: return "SimpleRecurrent";
    case 410: return "GRU";
    case 420: return "UniDirectionalLSTM";
    case 430: return "BiDirectionalLSTM";
    case 500: return "Custom";
    case 590: return "Copy";
    case 600: return "BranchLayer";
    case 615: return "LoopLayer";
    case 620: return "LoopBreakLayer";
    case 630: return "LoopContinueLayer";
    case 985: return "Transpose";
    case 1045: return "BatchedMatMul";
    default: return nullptr;
  }
}

// Human-readable name for a top-level `oneof Type` model kind (used only for a
// metadata note / version string on the fallback path).
const char* model_type_name(uint32_t field) {
  switch (field) {
    case 500: return "neuralNetwork";
    case 403: return "neuralNetworkClassifier";
    case 303: return "neuralNetworkRegressor";
    case 502: return "mlProgram";
    case 200: return "pipelineClassifier";
    case 201: return "pipelineRegressor";
    case 202: return "pipeline";
    case 300: return "glmRegressor";
    case 400: return "glmClassifier";
    case 302: return "treeEnsembleRegressor";
    case 402: return "treeEnsembleClassifier";
    case 301: return "supportVectorRegressor";
    case 401: return "supportVectorClassifier";
    default: return nullptr;
  }
}

// A discovered weight payload: absolute mmap offset+len (never read) + dtype.
struct WeightHit {
  uint64_t offset = 0;
  uint64_t len = 0;
  ir::DType dtype = ir::DType::Unknown;
};

// Priority among the payload encodings so rawValue wins over int8 over float16
// over packed floatValue when a WeightParams carries several.
int wp_priority(bool raw, bool i8, bool f16, bool f32) {
  if (raw) return 4;
  if (i8) return 3;
  if (f16) return 2;
  if (f32) return 1;
  return 0;
}

// Try to interpret `sr` as a WeightParams. Returns true and fills *hit with the
// payload sub-range (offset+len) + dtype if the message parses cleanly AND every
// field it carries belongs to the WeightParams field set with the expected wire
// type. NEVER reads payload bytes (read_len_delim only records offset+len).
//
// WeightParams: 1 floatValue (packed float | single Fixed32), 2 float16Value
// (bytes), 30 rawValue (bytes), 31 int8RawValue (bytes), 40 quantization
// (message), 50 isUpdatable (bool). The all-fields-recognized rule is a strong
// structural signature that avoids misclassifying arbitrary nested layer bodies.
bool try_weight_params(const SubRange& sr, WeightHit* hit) {
  WireReader r = WireReader::sub(sr);
  int best_prio = 0;
  WeightHit best;
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return false;  // malformed wire -> not a clean WeightParams
    switch (h->field_number) {
      case 1: {  // floatValue: packed (LenDelim) or single Fixed32
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return false;
          int p = wp_priority(false, false, false, true);
          if (p > best_prio) { best_prio = p; best = {s->offset, s->len, ir::DType::F32}; }
        } else if (h->wire_type == WireType::Fixed32) {
          auto s = r.read_fixed32();  // single float element; no contiguous range
          if (!s) return false;
        } else {
          return false;
        }
        break;
      }
      case 2: {  // float16Value (bytes)
        if (h->wire_type != WireType::LenDelim) return false;
        auto s = r.read_len_delim();
        if (!s) return false;
        int p = wp_priority(false, false, true, false);
        if (p > best_prio) { best_prio = p; best = {s->offset, s->len, ir::DType::F16}; }
        break;
      }
      case 30: {  // rawValue (bytes) — strongest signal
        if (h->wire_type != WireType::LenDelim) return false;
        auto s = r.read_len_delim();
        if (!s) return false;
        int p = wp_priority(true, false, false, false);
        if (p > best_prio) { best_prio = p; best = {s->offset, s->len, ir::DType::Unknown}; }
        break;
      }
      case 31: {  // int8RawValue (bytes)
        if (h->wire_type != WireType::LenDelim) return false;
        auto s = r.read_len_delim();
        if (!s) return false;
        int p = wp_priority(false, true, false, false);
        if (p > best_prio) { best_prio = p; best = {s->offset, s->len, ir::DType::I8}; }
        break;
      }
      case 40: {  // quantization (QuantizationParams) — structural, skip
        if (h->wire_type != WireType::LenDelim) return false;
        auto s = r.skip_field(h->wire_type);
        if (!s) return false;
        break;
      }
      case 50: {  // isUpdatable (bool)
        if (h->wire_type != WireType::Varint) return false;
        auto s = r.read_varint();
        if (!s) return false;
        break;
      }
      default:
        return false;  // any other field -> this is not a WeightParams
    }
  }
  if (best_prio == 0) return false;  // no payload field present
  *hit = best;
  return true;
}

// Recursively walk a layer-kind body, recording every WeightParams found (at any
// nesting level). A nested length-delimited field that parses as a WeightParams
// is recorded; otherwise we descend into it (bounded depth). Best-effort: a wire
// error inside an in-bounds sub-range just stops this branch, never crashes.
void collect_weights(const SubRange& sr, int depth, std::vector<WeightHit>* out) {
  if (depth > kMaxWeightDepth) return;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    if (out->size() >= kMaxWeightsPerLayer) return;
    auto h = r.read_tag();
    if (!h) return;
    if (h->wire_type == WireType::LenDelim) {
      auto s = r.read_len_delim();
      if (!s) return;
      WeightHit hit;
      if (try_weight_params(*s, &hit)) {
        out->push_back(hit);
      } else {
        collect_weights(*s, depth + 1, out);
      }
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return;
    }
  }
}

// Per-graph value map: edge name (StringId) -> ValueInfo index (same scheme as
// the ONNX parser).
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

// Parse one NeuralNetworkLayer into an ir::Node (appended to g). Weights found in
// the layer-kind body become initializers whose synthesized names are appended
// to the node's inputs, so CostModel attributes their bytes to this node exactly
// like ONNX initializers-as-inputs.
Result<bool> parse_layer(const SubRange& sr, ir::Model& model, ir::Graph& g,
                         ValueMap& map, uint32_t layer_index) {
  StringId name;
  std::vector<StringId> inputs, outputs;
  const char* kind = nullptr;
  uint32_t kind_field = 0;
  std::vector<WeightHit> weights;

  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case LF_NAME: {
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          name = model.intern(*s);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case LF_INPUT: {
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          inputs.push_back(model.intern(*s));
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case LF_OUTPUT: {
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_string();
          if (!s) return s.error();
          outputs.push_back(model.intern(*s));
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      default: {
        // The `oneof layer` lives at fields >= 100. The first such
        // length-delimited field identifies the layer kind and carries its
        // (possibly nested) WeightParams.
        if (h->field_number >= LF_KIND_MIN && h->wire_type == WireType::LenDelim) {
          auto body = r.read_len_delim();
          if (!body) return body.error();
          if (kind_field == 0) {
            kind = layer_kind_name(h->field_number);
            kind_field = h->field_number;
          }
          collect_weights(*body, 0, &weights);
        } else {
          auto sk = r.skip_field(h->wire_type);
          if (!sk) return sk.error();
        }
        break;
      }
    }
  }

  // op_type: the mapped kind name, else a stable synthetic name from the tag.
  std::string op_name;
  if (kind != nullptr) {
    op_name = kind;
  } else if (kind_field != 0) {
    op_name = "CoreMLLayer_" + std::to_string(kind_field);
  } else {
    op_name = "CoreMLLayer";
  }

  int32_t node_idx = static_cast<int32_t>(g.nodes.size());
  ir::Node node;
  node.op_type = model.intern(op_name);
  node.name = name;

  // Synthesize an initializer + input edge per discovered weight so weights wire
  // into the graph like ONNX initializers (CostModel matches by input name). The
  // synthesized value keeps producer == -1 (a constant / graph input).
  std::string base = name.valid() ? std::string(model.str(name))
                                   : ("layer" + std::to_string(layer_index));
  for (uint32_t k = 0; k < weights.size(); ++k) {
    const WeightHit& w = weights[k];
    std::string wname = base + "_weight_" + std::to_string(k);
    StringId wid = model.intern(wname);
    ir::TensorRef tr;
    tr.name = wid;
    tr.dtype = w.dtype;
    tr.file_offset = w.offset;  // absolute mmap offset; payload never read
    tr.byte_len = w.len;
    g.initializers.push_back(std::move(tr));
    inputs.push_back(wid);
  }

  // inputs -> edge_refs
  node.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  for (StringId in : inputs) {
    g.edge_refs.push_back(get_or_create_value(g, map, in));
  }
  node.inputs.count = static_cast<uint32_t>(inputs.size());

  // outputs -> edge_refs; this node is the producer of each output value.
  node.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  for (StringId out : outputs) {
    uint32_t vi = get_or_create_value(g, map, out);
    g.values[vi].producer = node_idx;
    g.edge_refs.push_back(vi);
  }
  node.outputs.count = static_cast<uint32_t>(outputs.size());

  g.nodes.push_back(std::move(node));
  return true;
}

// Parse a NeuralNetwork(/Classifier/Regressor) message: repeated layers at
// field 1. Builds model.graphs[0].
Result<bool> parse_neural_network(const SubRange& sr, ir::Model& model,
                                  ProgressSink& progress) {
  ir::Graph g;
  ValueMap map;

  // Collect layer sub-ranges first (deterministic order), then build.
  std::vector<SubRange> layer_subs;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == kNnLayersField && h->wire_type == WireType::LenDelim) {
      auto s = r.read_len_delim();
      if (!s) return s.error();
      if (layer_subs.size() < kMaxLayers) layer_subs.push_back(*s);
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }

  progress.set(0.5f, "Building graph");
  for (uint32_t i = 0; i < layer_subs.size(); ++i) {
    auto ok = parse_layer(layer_subs[i], model, g, map, i);
    if (!ok) return ok.error();
  }

  model.graphs.push_back(std::move(g));
  return true;
}

// FeatureDescription: 1 name (string). Best-effort — returns empty on any issue.
Result<StringId> parse_feature_name(const SubRange& sr, ir::Model& model) {
  StringId nm;
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_string();
      if (!s) return s.error();
      nm = model.intern(*s);
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return nm;
}

// ModelDescription: 1 input (repeated FeatureDescription), 10 output (repeated).
// Collects feature names so we can surface graph inputs/outputs after the graph
// is built (best-effort; not required for a passing parse).
Result<bool> parse_description(const SubRange& sr, ir::Model& model,
                               std::vector<StringId>* in_names,
                               std::vector<StringId>* out_names) {
  WireReader r = WireReader::sub(sr);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    if (h->field_number == 1 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_len_delim();
      if (!s) return s.error();
      auto nm = parse_feature_name(*s, model);
      if (!nm) return nm.error();
      if (nm->valid()) in_names->push_back(*nm);
    } else if (h->field_number == 10 && h->wire_type == WireType::LenDelim) {
      auto s = r.read_len_delim();
      if (!s) return s.error();
      auto nm = parse_feature_name(*s, model);
      if (!nm) return nm.error();
      if (nm->valid()) out_names->push_back(*nm);
    } else {
      auto sk = r.skip_field(h->wire_type);
      if (!sk) return sk.error();
    }
  }
  return true;
}

}  // namespace

// CoreML entry point (declared in parsers/Parser.h). Reads structure only.
Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "Parsing CoreML");

  ir::Model model;
  model.format_name = model.intern("CoreML");

  if (!file.valid() || file.data() == nullptr)
    return err("empty or unmapped file", 0);

  int64_t spec_version = -1;
  const char* fallback_type = nullptr;   // non-null => note this exotic type
  std::vector<StringId> in_names, out_names;

  // Walk the top-level Model message. Collect the description sub-range and the
  // selected Type; build the graph for a neuralNetwork family, note the rest.
  SubRange desc_sub;
  bool have_desc = false;
  SubRange nn_sub;
  bool have_nn = false;

  WireReader r(file.data(), file.size(), 0);
  while (!r.at_end()) {
    auto h = r.read_tag();
    if (!h) return h.error();
    switch (h->field_number) {
      case MF_SPEC_VERSION: {
        if (h->wire_type == WireType::Varint) {
          auto v = r.read_varint();
          if (!v) return v.error();
          spec_version = static_cast<int64_t>(*v);
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case MF_DESCRIPTION: {
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          desc_sub = *s;
          have_desc = true;
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      case MF_NEURAL_NETWORK:
      case MF_NN_CLASSIFIER:
      case MF_NN_REGRESSOR: {
        if (h->wire_type == WireType::LenDelim) {
          auto s = r.read_len_delim();
          if (!s) return s.error();
          if (!have_nn) { nn_sub = *s; have_nn = true; }
        } else { auto sk = r.skip_field(h->wire_type); if (!sk) return sk.error(); }
        break;
      }
      default: {
        // Any other top-level Type oneof (mlProgram, pipeline, tree/GLM/SVM, …)
        // -> note it for the tensor-table fallback, then skip its body.
        if (fallback_type == nullptr) {
          const char* mt = model_type_name(h->field_number);
          if (mt != nullptr) fallback_type = mt;
        }
        auto sk = r.skip_field(h->wire_type);
        if (!sk) return sk.error();
        break;
      }
    }
  }

  // Description feature names (best-effort; wire errors propagate as a clean
  // Result error, unknown fields are skipped inside).
  if (have_desc) {
    auto ok = parse_description(desc_sub, model, &in_names, &out_names);
    if (!ok) return ok.error();
  }

  // Build the graph for a neuralNetwork family model.
  if (have_nn) {
    auto ok = parse_neural_network(nn_sub, model, progress);
    if (!ok) return ok.error();
    model.has_graph = true;
    ir::Graph& g = model.graphs[0];
    // Wire graph inputs/outputs from the ModelDescription feature names when the
    // names line up with graph values.
    ValueMap map;
    map.reserve(g.values.size());
    for (uint32_t i = 0; i < g.values.size(); ++i) map.emplace(g.values[i].name.id, i);
    for (StringId n : in_names) {
      auto it = map.find(n.id);
      if (it != map.end()) g.graph_inputs.push_back(it->second);
    }
    for (StringId n : out_names) {
      auto it = map.find(n.id);
      if (it != map.end()) g.graph_outputs.push_back(it->second);
    }
  } else {
    // No recognizable compute graph: tensor-table fallback (no error, ever).
    model.has_graph = false;
    std::string note;
    if (fallback_type != nullptr) {
      note = std::string("CoreML ") + fallback_type +
             " model: graph view not supported (structural ops not surfaced)";
      if (std::string(fallback_type) == "mlProgram") {
        note += "; weights typically reside in weights/weight.bin (.mlpackage bundle)";
      }
    } else {
      note = "CoreML model: no neuralNetwork graph found";
    }
    model.metadata.emplace_back(model.intern("note"), model.intern(note));
  }

  // Producer / version summary.
  {
    std::string vinfo;
    if (spec_version >= 0) vinfo = "spec v" + std::to_string(spec_version);
    if (have_nn) {
      vinfo += vinfo.empty() ? "neuralNetwork" : "; neuralNetwork";
    } else if (fallback_type != nullptr) {
      vinfo += vinfo.empty() ? fallback_type : (std::string("; ") + fallback_type);
    }
    model.version_info = model.intern(vinfo);
  }

  progress.set(1.0f, "Done");
  return model;
}

}  // namespace netvis::coreml
