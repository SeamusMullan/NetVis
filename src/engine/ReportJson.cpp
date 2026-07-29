// engine/ReportJson.cpp — headless JSON model report (issue #58).
//
// See ReportJson.h for the schema. Everything here is pure structure over an
// ir::Model + the existing headless analyzers (compute_cost, GraphAdjacency);
// no tensor payload is ever read, so the report runs identically on a worker,
// in ctest, and from the `--report` CLI.
#include "engine/ReportJson.h"

#include <cstdint>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "core/MappedFile.h"
#include "engine/CostModel.h"
#include "engine/GraphAdjacency.h"
#include "engine/ModelPath.h"
#include "engine/ShapeInferenceExt.h"
#include "parsers/Parser.h"

namespace netvis {

namespace {

using json = nlohmann::ordered_json;

// Lowercased extension without the leading dot ("" if none). Mirrors the
// tiebreaker ModelSession derives; used only to route detection.
std::string ext_of(const std::string& path) {
  auto dot = path.find_last_of('.');
  auto slash = path.find_last_of("/\\");
  if (dot == std::string::npos) return {};
  if (slash != std::string::npos && dot < slash) return {};  // dot in a dir name
  std::string e = path.substr(dot + 1);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}

// Count distinct producer->consumer IR edges over one graph (deduped, self-loops
// dropped) — the same edge derivation the navigation/adjacency uses.
uint64_t edge_count(const ir::Model& model, uint32_t graph_index) {
  GraphAdjacency adj;
  adj.build(model, graph_index);
  return static_cast<uint64_t>(adj.succ_values().size());
}

}  // namespace

std::string build_report_json(const ir::Model& model) {
  json j;
  j["schema"] = kReportSchema;
  j["format"] = std::string(model.str(model.format_name));
  j["producer"] = std::string(model.str(model.producer));
  j["version"] = std::string(model.str(model.version_info));
  j["has_graph"] = model.has_graph;
  j["graph_count"] = model.graphs.size();

  json graphs = json::array();
  for (uint32_t gi = 0; gi < model.graphs.size(); ++gi) {
    const ir::Graph& g = model.graphs[gi];
    json gj;
    gj["index"] = gi;
    gj["name"] = std::string(model.str(g.name));
    gj["nodes"] = g.nodes.size();
    gj["edges"] = edge_count(model, gi);
    gj["inputs"] = g.graph_inputs.size();
    gj["outputs"] = g.graph_outputs.size();
    gj["initializers"] = g.initializers.size();
    graphs.push_back(std::move(gj));
  }
  j["graphs"] = std::move(graphs);

  // Cost over the MAIN graph (index 0). compute_cost falls back to table mode
  // (from_graph=false, built from flat_tensors) for a model with no compute
  // graph or an out-of-range index, so this is safe for every format.
  const uint32_t cost_graph = 0;
  CostReport r = compute_cost(model, cost_graph);

  json cost;
  cost["graph_index"] = cost_graph;
  cost["from_graph"] = r.from_graph;
  cost["total_flops"] = r.total_flops;
  cost["total_params"] = r.total_params;
  cost["total_weight_bytes"] = r.total_weight_bytes;
  cost["peak_activation_bytes"] = r.peak_activation_bytes;
  cost["nodes_total"] = r.nodes_total;
  cost["nodes_flops_known"] = r.nodes_flops_known;
  cost["effective_bits_per_param"] = r.effective_bits_per_param();
  cost["size_vs_fp32"] = r.size_vs_fp32();

  json dtypes = json::array();
  for (const DTypeUsage& u : r.dtype_usage) {
    json du;
    du["dtype"] = ir::dtype_name(u.dtype);
    du["params"] = u.params;
    du["bytes"] = u.bytes;
    dtypes.push_back(std::move(du));
  }
  cost["dtype_usage"] = std::move(dtypes);

  json roof;
  roof["ridge_flop_per_byte"] = r.roofline.ridge_flop_per_byte;
  roof["compute_bound_flops"] = r.roofline.compute_bound_flops;
  roof["memory_bound_flops"] = r.roofline.memory_bound_flops;
  roof["compute_bound_nodes"] = r.roofline.compute_bound_nodes;
  roof["memory_bound_nodes"] = r.roofline.memory_bound_nodes;
  roof["compute_bound_fraction"] = r.roofline.compute_bound_fraction();
  cost["roofline"] = std::move(roof);

  j["cost"] = std::move(cost);

  // Compact single line (no indent, no trailing newline). The CLI adds the
  // newline; a tool piping this into `jq` gets clean input either way.
  return j.dump();
}

Result<std::string> report_file(const std::string& path) {
  // Resolve a .mlpackage bundle to its inner model file (a plain path passes
  // through unchanged), exactly as ModelSession::open_async does, so external
  // weights resolve from the right directory if a parser needs them.
  ResolvedModelPath resolved = resolve_model_path(path);

  auto mapped = MappedFile::open(resolved.map_path);
  if (!mapped) return mapped.error();
  MappedFile file = mapped.take();

  const std::string ext = ext_of(resolved.map_path);
  ProgressSink progress;
  Result<ir::Model> parsed = parse_model(file, ext, progress);
  if (!parsed) return parsed.error();
  ir::Model model = parsed.take();

  // ONNX ships incomplete value_info; run the same best-effort shape inference
  // the ShapeInferJob runs so cost (FLOPs/activation bytes) is meaningful. Other
  // formats carry resolved shapes from the parser, so this is ONNX-only (a no-op
  // elsewhere would be harmless, but matching ModelSession keeps behavior 1:1).
  if (model.has_graph && model.str(model.format_name) == "ONNX") {
    infer_shapes_ext(model, 0, file.data(), file.size(), &progress);
  }

  return build_report_json(model);
}

}  // namespace netvis
