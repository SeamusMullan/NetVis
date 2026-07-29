// engine/ReportJson.h — headless JSON model report (issue #58).
//
// DECISION: `netvis --report model.onnx` must analyze a model and print JSON
// stats with NO GUI. The report logic therefore lives in netvis_core (no
// view/ImGui deps) so it runs on a worker thread, in ctest, and from the CLI
// alike. It is a PURE structural summary built from the same headless pipeline
// the GUI uses (parse_model + infer_shapes_ext + compute_cost) and NEVER reads a
// tensor payload — it upholds the zero-payload-read thesis (spec §2.1), exactly
// like compute_cost.
//
// ---------------------------------------------------------------------------
//  JSON SCHEMA ("netvis.report.v1")
// ---------------------------------------------------------------------------
//  {
//    "schema": "netvis.report.v1",
//    "format": "ONNX",              // ir::Model::format_name
//    "producer": "pytorch 2.1",     // "" if the format records none
//    "version": "",                 // ir::Model::version_info ("" if none)
//    "has_graph": true,             // false => tensor-table model (GGUF/…)
//    "graph_count": 1,
//    "graphs": [                    // one entry per ir::Model::graphs
//      {
//        "index": 0,
//        "name": "main",
//        "nodes": 3,                // graphs[i].nodes.size()
//        "edges": 2,                // distinct producer->consumer IR edges
//        "inputs": 1,               // graphs[i].graph_inputs.size()
//        "outputs": 1,              // graphs[i].graph_outputs.size()
//        "initializers": 4          // graphs[i].initializers.size()
//      }
//    ],
//    "cost": {                      // compute_cost over the MAIN graph (index 0);
//      "graph_index": 0,            // table-mode models report from flat_tensors
//      "from_graph": true,          // false => built from model.flat_tensors
//      "total_flops": 28901376,
//      "total_params": 18432,
//      "total_weight_bytes": 73728,
//      "peak_activation_bytes": 1024,
//      "nodes_total": 3,
//      "nodes_flops_known": 2,      // (nodes_total - this) have unknown FLOPs
//      "effective_bits_per_param": 32.0,
//      "size_vs_fp32": 1.0,         // <1 => compressed vs storing every param fp32
//      "dtype_usage": [             // per-dtype weight usage, bytes-desc
//        { "dtype": "f32", "params": 18432, "bytes": 73728 }
//      ],
//      "roofline": {                // default-ridge snapshot (kDefaultRidgeFlopPerByte)
//        "ridge_flop_per_byte": 40.0,
//        "compute_bound_flops": 28901376,
//        "memory_bound_flops": 0,
//        "compute_bound_nodes": 1,
//        "memory_bound_nodes": 0,
//        "compute_bound_fraction": 1.0
//      }
//    }
//  }
//
// Depends only on ir/IR.h + engine leaves (CostModel/GraphAdjacency). No view
// coupling; safe to unit-test headlessly.
#pragma once

#include <string>

#include "core/Result.h"
#include "ir/IR.h"

namespace netvis {

// Schema identifier emitted as the report's "schema" field. Bump when the shape
// of the JSON changes so consumers can gate on it.
inline constexpr const char* kReportSchema = "netvis.report.v1";

// Build the JSON report for an ALREADY-PARSED model (shapes assumed already
// inferred by the caller, as report_file does for ONNX). Cost is computed over
// the main graph (index 0); a table-mode model reports table-mode cost. PURE;
// never reads a tensor payload; never throws. Output is a compact single-line
// JSON string (no trailing newline) with keys in the documented order.
std::string build_report_json(const ir::Model& model);

// Parse `path` headlessly using the SAME synchronous pipeline the GUI drives on
// its worker (resolve .mlpackage bundle -> mmap -> parse_model -> ONNX shape
// inference on the main graph), then build the report. Returns an Error Result
// on a map/parse failure (message suitable for stderr). No window, no ImGui.
Result<std::string> report_file(const std::string& path);

}  // namespace netvis
