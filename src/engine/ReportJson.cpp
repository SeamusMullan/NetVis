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

#include "engine/CostModel.h"
#include "engine/GraphAdjacency.h"
#include "engine/QueryCli.h"

namespace netvis {

namespace {

using json = nlohmann::ordered_json;

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
  // One shared headless pipeline (resolve -> mmap -> parse -> ONNX shape
  // inference) serves both this report and every `netvis query` verb, so the
  // two CLIs can never disagree about how a model is loaded.
  Result<HeadlessModel> hm = load_model_headless(path);
  if (!hm) return hm.error();
  return build_report_json(hm->model);
}

}  // namespace netvis
