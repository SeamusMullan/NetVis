// parsers/openvino/OpenVinoParser.cpp — OpenVINO IR (.xml + sibling .bin) parser.
//
// DECISION (v0.5.0 plan §"#39 — OpenVINO IR"): the topology is XML; we walk it
// with the bounded, hostile-input-safe openvino::XmlDocument (NO XML dependency)
// and NEVER read a weight byte. A Const layer's <data offset= size=> is recorded
// as a TensorRef pointing at the sibling <stem>.bin — the SAME external-data
// plumbing ONNX uses (resolve_payload opens <model_dir>/<stem>.bin lazily). The
// structural parse leaves ByteReader::payload_read_counter() at 0.
//
// XML -> IR mapping:
//   <net name= version=>                  -> Model (format_name="OpenVINO")
//   <layers><layer id= name= type=>       -> ir::Node (type -> normalize_ov_op)
//   <data ...>                            -> ir::Attribute(s) (String-valued)
//   <input>/<output><port id= precision=> -> ValueInfo (dtype from precision)
//     <dim>N</dim>                        -> shape dim (symbolic / -1 -> -1)
//   <edges><edge from-layer= from-port= to-layer= to-port=>
//                                         -> wire edge_refs by (layer,port) value
//   Const <data offset= size= element_type= shape=>
//                                         -> initializer TensorRef into <stem>.bin
//   Parameter                             -> graph input ; Result -> graph output
//   TensorIterator/Loop <body>, If <then_body>/<else_body>
//                                         -> nested graph (depth-bounded)
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"
#include "parsers/openvino/XmlReader.h"

namespace netvis::openvino {
namespace {

// Subgraph nesting guard (matches ONNX kMaxGraphDepth = 64).
constexpr int kMaxSubgraphDepth = 64;

// --- dtype / op normalization ------------------------------------------------

// Lowercase a copy for spelling-agnostic matching (handles f32 AND FP32, etc.).
std::string lower(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return out;
}

// OpenVINO element_type (lowercase: f32/f16/bf16/i8/...) AND port precision
// (uppercase-ish: FP32/FP16/I32/...) -> ir::DType. Unknown -> DType::Unknown.
ir::DType map_ov_dtype(std::string_view raw) {
  const std::string t = lower(raw);
  if (t == "f32" || t == "fp32") return ir::DType::F32;
  if (t == "f16" || t == "fp16") return ir::DType::F16;
  if (t == "bf16") return ir::DType::BF16;
  if (t == "f64" || t == "fp64") return ir::DType::F64;
  if (t == "i8") return ir::DType::I8;
  if (t == "i16") return ir::DType::I16;
  if (t == "i32") return ir::DType::I32;
  if (t == "i64") return ir::DType::I64;
  if (t == "u8") return ir::DType::U8;
  if (t == "u16") return ir::DType::U16;
  if (t == "u32") return ir::DType::U32;
  if (t == "u64") return ir::DType::U64;
  if (t == "boolean" || t == "bool") return ir::DType::Bool;
  // i4/u4/nf4/u1/dynamic/undefined and any exotic type -> honest Unknown.
  return ir::DType::Unknown;
}

// Map common OpenVINO op names to the vocabulary OpCategory/cost recognize
// (case-insensitive). Unmapped ops pass through verbatim (categorize as Other).
std::string normalize_ov_op(std::string_view type) {
  const std::string t = lower(type);
  // Compute
  if (t == "convolution" || t == "groupconvolution" ||
      t == "binaryconvolution" || t == "deformableconvolution")
    return "Conv";
  if (t == "convolutionbackpropdata" || t == "groupconvolutionbackpropdata")
    return "ConvTranspose";
  if (t == "matmul") return "MatMul";
  if (t == "fullyconnected") return "Gemm";
  // Activations
  if (t == "relu") return "Relu";
  if (t == "prelu") return "PRelu";
  if (t == "leakyrelu") return "LeakyRelu";
  if (t == "sigmoid") return "Sigmoid";
  if (t == "tanh") return "Tanh";
  if (t == "elu") return "Elu";
  if (t == "clamp") return "Clip";
  if (t == "softmax") return "Softmax";
  if (t == "logsoftmax") return "LogSoftmax";
  if (t == "gelu") return "Gelu";
  if (t == "swish") return "Swish";
  if (t == "hswish") return "HardSwish";
  if (t == "hsigmoid") return "HardSigmoid";
  if (t == "mish") return "Mish";
  if (t == "softplus") return "Softplus";
  // Norms
  if (t == "mvn" || t == "batchnorminference") return "BatchNormalization";
  // Pooling
  if (t == "maxpool") return "MaxPool";
  if (t == "avgpool") return "AveragePool";
  // Elementwise
  if (t == "add") return "Add";
  if (t == "subtract") return "Sub";
  if (t == "multiply") return "Mul";
  if (t == "divide") return "Div";
  if (t == "power") return "Pow";
  if (t == "maximum") return "Max";
  if (t == "minimum") return "Min";
  if (t == "sqrt") return "Sqrt";
  if (t == "exp") return "Exp";
  if (t == "log") return "Log";
  if (t == "abs") return "Abs";
  if (t == "erf") return "Erf";
  // Shape
  if (t == "reshape") return "Reshape";
  if (t == "transpose") return "Transpose";
  if (t == "concat") return "Concat";
  if (t == "split" || t == "variadicsplit") return "Split";
  if (t == "squeeze") return "Squeeze";
  if (t == "unsqueeze") return "Unsqueeze";
  if (t == "gather") return "Gather";
  // Reduce
  if (t == "reducemean") return "ReduceMean";
  if (t == "reducesum") return "ReduceSum";
  if (t == "reducemax") return "ReduceMax";
  if (t == "reducemin") return "ReduceMin";
  // Constant / graph IO markers
  if (t == "const" || t == "constant") return "Constant";
  if (t == "parameter") return "Parameter";
  if (t == "result") return "Result";
  // Control flow
  if (t == "tensoriterator") return "Loop";
  if (t == "loop") return "Loop";
  if (t == "if") return "If";
  // Unmapped: keep the original spelling (categorizes as Other, never a crash).
  return std::string(type);
}

// --- small parsing helpers ---------------------------------------------------

// Parse a signed base-10 int from `s`; symbolic / non-numeric / negative -> -1
// (OpenVINO uses -1 and symbolic names for dynamic dims).
int64_t parse_dim(std::string_view s) {
  // Trim.
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  s = s.substr(b, e - b);
  if (s.empty()) return -1;
  bool neg = false;
  size_t i = 0;
  if (s[0] == '-') { neg = true; i = 1; }
  if (i >= s.size()) return -1;
  int64_t v = 0;
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return -1;  // symbolic dim -> dynamic
    v = v * 10 + (s[i] - '0');
  }
  return neg ? -1 : v;  // any negative (incl. -1) -> dynamic
}

// Split a "2,3,224,224" shape string into dims (symbolic -> -1).
void parse_shape_csv(std::string_view s, SmallVec<int64_t, 6>* out) {
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    std::string_view tok =
        s.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                        : comma - start);
    // Skip empty tokens (e.g. scalar "" or trailing comma).
    size_t b = 0, e = tok.size();
    while (b < e && std::isspace(static_cast<unsigned char>(tok[b]))) ++b;
    if (b < e) out->push_back(parse_dim(tok));
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
}

// Read a port element's <dim> children into a shape.
void parse_port_dims(const XmlDocument& doc, const XmlNode& port,
                     SmallVec<int64_t, 6>* out) {
  for (uint32_t ci : port.children) {
    const XmlNode& c = doc.node(ci);
    if (c.name == "dim") out->push_back(parse_dim(c.text));
  }
}

// --- per-graph value plumbing (mirrors OnnxParser) ---------------------------

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

// Stable per-(layer,port) SSA value name. Interned in the model arena; the
// per-graph ValueMap keys on the resulting StringId, so identical names in two
// different subgraphs never collide (each graph owns its ValueMap).
StringId value_name(ir::Model& model, int64_t layer_id, int64_t port_id) {
  return model.intern("v" + std::to_string(layer_id) + "_" +
                      std::to_string(port_id));
}

// Edge key: (to_layer, to_port) packed; used to find each input port's source.
uint64_t edge_key(int64_t layer, int64_t port) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(layer)) << 32) |
         static_cast<uint32_t>(port);
}

// Forward decl: build one graph (net root OR a subgraph body container).
Result<int32_t> build_graph(const XmlDocument& doc, const XmlNode& container,
                            ir::Model& model, StringId bin_path, int depth);

// Process a single <layer> into a Node in graph `g`. `edge_src` maps a layer's
// input port (to_layer,to_port) -> its source (from_layer,from_port).
Result<bool> parse_layer(
    const XmlDocument& doc, const XmlNode& layer, ir::Model& model, ir::Graph& g,
    ValueMap& map, StringId bin_path,
    const std::unordered_map<uint64_t, std::pair<int64_t, int64_t>>& edge_src,
    int depth) {
  auto id_opt = layer.attr_int("id");
  if (!id_opt) return err("layer missing integer id", layer.offset);
  int64_t layer_id = *id_opt;

  const std::string* type = layer.attr("type");
  const std::string* lname = layer.attr("name");
  std::string raw_type = type ? *type : std::string();

  int32_t node_idx = static_cast<int32_t>(g.nodes.size());
  ir::Node node;
  node.op_type = model.intern(normalize_ov_op(raw_type));
  if (lname) node.name = model.intern(*lname);

  // --- data attributes (strides/pads/... and Const offset/size/etc.) --------
  node.attributes.begin = static_cast<uint32_t>(g.attributes.size());
  const XmlNode* data = doc.child(layer, "data");
  if (data) {
    for (const XmlAttr& a : data->attrs) {
      ir::Attribute attr;
      attr.name = model.intern(a.name);
      attr.value.kind = ir::AttrValue::Kind::String;
      attr.value.s = model.intern(a.value);
      g.attributes.push_back(std::move(attr));
    }
  }
  node.attributes.count =
      static_cast<uint32_t>(g.attributes.size()) - node.attributes.begin;

  const std::string lt = lower(raw_type);
  const bool is_param = (lt == "parameter");
  const bool is_result = (lt == "result");
  const bool is_const = (lt == "const" || lt == "constant");

  // --- inputs: wire each <input><port id=> to its edge source value ---------
  node.inputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  uint32_t input_count = 0;
  if (const XmlNode* inputs = doc.child(layer, "input")) {
    for (uint32_t ci : inputs->children) {
      const XmlNode& port = doc.node(ci);
      if (port.name != "port") continue;
      auto pid = port.attr_int("id");
      if (!pid) continue;
      StringId vname;
      auto it = edge_src.find(edge_key(layer_id, *pid));
      if (it != edge_src.end()) {
        vname = value_name(model, it->second.first, it->second.second);
      } else {
        // No incoming edge (dangling input): keep a stable placeholder value so
        // the slot is not silently dropped.
        vname = value_name(model, layer_id, *pid);
      }
      uint32_t vi = get_or_create_value(g, map, vname);
      // Fill shape/dtype from the input port only if the producer has not yet
      // set them (producer's output port is authoritative).
      if (g.values[vi].shape.empty()) parse_port_dims(doc, port, &g.values[vi].shape);
      if (g.values[vi].dtype == ir::DType::Unknown) {
        if (const std::string* prec = port.attr("precision"))
          g.values[vi].dtype = map_ov_dtype(*prec);
      }
      g.edge_refs.push_back(vi);
      ++input_count;
      if (is_result) g.graph_outputs.push_back(vi);
    }
  }
  node.inputs.count = input_count;

  // --- outputs: each <output><port id=> defines a value produced here -------
  node.outputs.begin = static_cast<uint32_t>(g.edge_refs.size());
  uint32_t output_count = 0;
  uint32_t const_out_value = UINT32_MAX;
  if (const XmlNode* outputs = doc.child(layer, "output")) {
    for (uint32_t ci : outputs->children) {
      const XmlNode& port = doc.node(ci);
      if (port.name != "port") continue;
      auto pid = port.attr_int("id");
      if (!pid) continue;
      StringId vname = value_name(model, layer_id, *pid);
      uint32_t vi = get_or_create_value(g, map, vname);
      g.values[vi].producer = node_idx;
      // Output port is authoritative for shape/dtype.
      SmallVec<int64_t, 6> shape;
      parse_port_dims(doc, port, &shape);
      if (!shape.empty()) g.values[vi].shape = shape;
      if (const std::string* prec = port.attr("precision"))
        g.values[vi].dtype = map_ov_dtype(*prec);
      g.edge_refs.push_back(vi);
      ++output_count;
      if (const_out_value == UINT32_MAX) const_out_value = vi;
      if (is_param) g.graph_inputs.push_back(vi);
    }
  }
  node.outputs.count = output_count;

  // --- Const weight -> initializer TensorRef (offset+len only) --------------
  if (is_const && data) {
    auto off = data->attr_int("offset");
    auto size = data->attr_int("size");
    if (off && size && *off >= 0 && *size >= 0) {
      ir::TensorRef tr;
      // Name the initializer after its output value so it aligns with the edge.
      if (const_out_value != UINT32_MAX)
        tr.name = g.values[const_out_value].name;
      else
        tr.name = node.name;
      if (const std::string* et = data->attr("element_type")) {
        tr.dtype = map_ov_dtype(*et);
        // v0.6.3 (#85): when element_type maps to Unknown (i4/u4/nf4/u1/f8* — exotic
        // quant types with no ir::DType), record the raw type as a label so the
        // inspector can still display it honestly.
        if (tr.dtype == ir::DType::Unknown) {
          tr.dtype_label = model.intern(lower(*et));
        }
      }
      if (const std::string* sh = data->attr("shape"))
        parse_shape_csv(*sh, &tr.shape);
      tr.external_path = bin_path;
      tr.file_offset = static_cast<uint64_t>(*off);
      tr.byte_len = static_cast<uint64_t>(*size);
      // Backfill the output value's dtype/shape from the Const declaration.
      if (const_out_value != UINT32_MAX) {
        if (g.values[const_out_value].dtype == ir::DType::Unknown)
          g.values[const_out_value].dtype = tr.dtype;
        if (g.values[const_out_value].shape.empty())
          g.values[const_out_value].shape = tr.shape;
      }
      g.initializers.push_back(std::move(tr));
    }
  }

  // --- subgraph bodies (TensorIterator/Loop/If) -----------------------------
  int32_t primary_subgraph = -1;
  auto attach_body = [&](const char* body_tag) -> Result<bool> {
    if (const XmlNode* body = doc.child(layer, body_tag)) {
      auto gi = build_graph(doc, *body, model, bin_path, depth + 1);
      if (!gi) return gi.error();
      if (primary_subgraph < 0) primary_subgraph = *gi;
    }
    return true;
  };
  if (lt == "tensoriterator" || lt == "loop") {
    auto ok = attach_body("body");
    if (!ok) return ok.error();
  } else if (lt == "if") {
    auto ok = attach_body("then_body");
    if (!ok) return ok.error();
    ok = attach_body("else_body");
    if (!ok) return ok.error();
  }
  node.subgraph = primary_subgraph;

  g.nodes.push_back(std::move(node));
  return true;
}

// Build a graph from a container that holds <layers> and <edges> (the <net>
// root, or a subgraph <body>/<then_body>/<else_body>). Reserves its own
// Model::graphs slot BEFORE recursing so graph indices stay stable and
// graphs[0] remains the main graph — exactly like OnnxParser::parse_graph.
Result<int32_t> build_graph(const XmlDocument& doc, const XmlNode& container,
                            ir::Model& model, StringId bin_path, int depth) {
  if (depth > kMaxSubgraphDepth)
    return err("subgraph nesting too deep", container.offset);

  int32_t my_idx = static_cast<int32_t>(model.graphs.size());
  model.graphs.emplace_back();  // reserve; do not hold a reference across recursion

  ir::Graph g;
  ValueMap map;
  if (const std::string* nm = container.attr("name")) g.name = model.intern(*nm);

  // Parse <edges> first: build (to_layer,to_port) -> (from_layer,from_port).
  std::unordered_map<uint64_t, std::pair<int64_t, int64_t>> edge_src;
  if (const XmlNode* edges = doc.child(container, "edges")) {
    for (uint32_t ci : edges->children) {
      const XmlNode& e = doc.node(ci);
      if (e.name != "edge") continue;
      auto fl = e.attr_int("from-layer");
      auto fp = e.attr_int("from-port");
      auto tl = e.attr_int("to-layer");
      auto tp = e.attr_int("to-port");
      if (!fl || !fp || !tl || !tp) continue;
      edge_src[edge_key(*tl, *tp)] = {*fl, *fp};
    }
  }

  // Process <layers><layer> in document order.
  if (const XmlNode* layers = doc.child(container, "layers")) {
    for (uint32_t ci : layers->children) {
      const XmlNode& layer = doc.node(ci);
      if (layer.name != "layer") continue;
      auto ok = parse_layer(doc, layer, model, g, map, bin_path, edge_src, depth);
      if (!ok) return ok.error();
    }
  }

  model.graphs[my_idx] = std::move(g);
  return my_idx;
}

}  // namespace

// OpenVINO IR entry point (declared in parsers/Parser.h). Reads structure only;
// weight bytes stay in the sibling .bin until the inspector reads them.
Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "Parsing OpenVINO IR");

  ir::Model model;
  model.format_name = model.intern("OpenVINO");
  model.has_graph = true;

  if (!file.valid() || file.data() == nullptr)
    return err("empty or unmapped file", 0);

  auto parsed = XmlDocument::parse(file.data(), file.size(), 0);
  if (!parsed) return parsed.error();
  const XmlDocument doc = parsed.take();

  if (!doc.has_root()) return err("OpenVINO IR: no root element", 0);
  const XmlNode& net = doc.root();
  if (net.name != "net")
    return err("OpenVINO IR: root element is not <net>", net.offset);

  // Model metadata.
  if (const std::string* nm = net.attr("name"))
    model.metadata.emplace_back(model.intern("name"), model.intern(*nm));
  if (const std::string* ver = net.attr("version"))
    model.version_info = model.intern("IR v" + *ver);

  // The weight blob is the .xml stem + ".bin", a sibling resolved relative to
  // model_dir by resolve_payload (the same path ONNX external-data uses).
  std::string bin_name = "model.bin";
  if (!file.path().empty()) {
    std::filesystem::path xmlp(file.path());
    bin_name = xmlp.stem().string() + ".bin";
  }
  StringId bin_path = model.intern(bin_name);

  progress.set(0.3f, "Building graph");
  auto gi = build_graph(doc, net, model, bin_path, 0);
  if (!gi) return gi.error();

  // v0.6.3 (#85): if the model declared at least one Const initializer and the
  // sibling .bin does not exist on disk, append a metadata note. Do NOT error —
  // the topology must still parse successfully (zero-payload invariant: we never
  // read the weight blob during structural parse, so a missing blob is not a parse
  // failure, just a metadata fact the inspector should know).
  if (!model.graphs.empty() && !model.graphs[0].initializers.empty() &&
      !file.path().empty()) {
    std::filesystem::path bin_full_path =
        std::filesystem::path(file.path()).parent_path() / bin_name;
    if (!std::filesystem::exists(bin_full_path)) {
      model.metadata.emplace_back(model.intern("note"),
                                   model.intern("weights blob not found: " + bin_name));
    }
  }

  progress.set(1.0f, "Done");
  return model;
}

}  // namespace netvis::openvino
