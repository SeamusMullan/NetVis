// engine/QueryCli.cpp — headless agent-facing query CLI.
//
// See QueryCli.h for the contract and docs/agent-cli.md for the per-verb JSON.
// Everything here is a thin JSON projection over the existing headless engine
// (parse_model, infer_shapes_ext, compute_cost, GraphAdjacency, SearchIndex,
// compute_tensor_stats); no verb invents analysis of its own, so the CLI can
// never disagree with what the GUI shows.
#include "engine/QueryCli.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "engine/CostModel.h"
#include "engine/GraphAdjacency.h"
#include "engine/ModelDiff.h"
#include "engine/ModelPath.h"
#include "engine/OpCategory.h"
#include "engine/ReportJson.h"
#include "engine/SearchIndex.h"
#include "engine/ShapeInferenceExt.h"
#include "engine/TensorStats.h"
#include "parsers/Parser.h"

namespace netvis {

namespace {

using json = nlohmann::ordered_json;

// Lowercased extension without the leading dot ("" if none) — the same
// detection tiebreaker ModelSession and report_file derive.
std::string ext_of(const std::string& path) {
  auto dot = path.find_last_of('.');
  auto slash = path.find_last_of("/\\");
  if (dot == std::string::npos) return {};
  if (slash != std::string::npos && dot < slash) return {};
  std::string e = path.substr(dot + 1);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}

std::string dir_of(const std::string& path) {
  auto slash = path.find_last_of("/\\");
  return slash == std::string::npos ? std::string(".") : path.substr(0, slash);
}

std::string lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

// ---------------------------------------------------------------------------
// Option parsing. Verbs take positionals first (model, then an optional
// target), then --key value / --key=value flags. Unknown flags are an error:
// an agent typo must fail loudly, not silently drop a filter.
// ---------------------------------------------------------------------------
struct Options {
  std::vector<std::string> positional;
  std::vector<std::pair<std::string, std::string>> flags;

  const std::string* flag(std::string_view key) const {
    for (const auto& [k, v] : flags)
      if (k == key) return &v;
    return nullptr;
  }
};

Result<Options> parse_options(const std::vector<std::string>& args, size_t start,
                              const std::vector<std::string_view>& known) {
  Options o;
  for (size_t i = start; i < args.size(); ++i) {
    std::string_view a = args[i];
    if (a.size() >= 2 && a.substr(0, 2) == "--") {
      std::string key, val;
      auto eq = a.find('=');
      if (eq != std::string_view::npos) {
        key = std::string(a.substr(2, eq - 2));
        val = std::string(a.substr(eq + 1));
      } else {
        key = std::string(a.substr(2));
        if (i + 1 >= args.size())
          return err("query: flag --" + key + " expects a value");
        val = args[++i];
      }
      bool ok = false;
      for (std::string_view k : known) ok = ok || k == key;
      if (!ok) return err("query: unknown flag --" + key);
      o.flags.emplace_back(std::move(key), std::move(val));
    } else {
      o.positional.emplace_back(a);
    }
  }
  return o;
}

Result<uint64_t> parse_u64(const std::string& s, std::string_view what) {
  if (s.empty()) return err("query: " + std::string(what) + " expects a number");
  uint64_t v = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return err("query: " + std::string(what) + " expects a number, got '" + s + "'");
    v = v * 10 + static_cast<uint64_t>(c - '0');
  }
  return v;
}

// ---------------------------------------------------------------------------
// JSON projections of IR pieces. Shapes keep -1 for dynamic dims — the agent
// must see "unknown" as unknown, never a fabricated size (spec P4.2 honesty).
// ---------------------------------------------------------------------------
json shape_json(const SmallVec<int64_t, 6>& shape) {
  json a = json::array();
  for (int64_t d : shape) a.push_back(d);
  return a;
}

json value_json(const ir::Model& m, const ir::ValueInfo& v) {
  json j;
  j["name"] = std::string(m.str(v.name));
  j["dtype"] = ir::dtype_name(v.dtype);
  j["shape"] = shape_json(v.shape);
  j["producer"] = v.producer;
  return j;
}

json tensor_ref_json(const ir::Model& m, const ir::TensorRef& t) {
  json j;
  j["name"] = std::string(m.str(t.name));
  j["dtype"] = ir::dtype_name(t.dtype);
  j["shape"] = shape_json(t.shape);
  j["elements"] = t.elem_count();
  j["bytes"] = t.byte_len;
  j["external"] = std::string(m.str(t.external_path));
  return j;
}

json attr_json(const ir::Model& m, const ir::Attribute& a) {
  json j;
  j["name"] = std::string(m.str(a.name));
  using K = ir::AttrValue::Kind;
  switch (a.value.kind) {
    case K::Int:    j["kind"] = "int";    j["value"] = a.value.i; break;
    case K::Float:  j["kind"] = "float";  j["value"] = a.value.f; break;
    case K::String: j["kind"] = "string"; j["value"] = std::string(m.str(a.value.s)); break;
    case K::Ints:   j["kind"] = "ints";   j["value"] = a.value.ints; break;
    case K::Floats: j["kind"] = "floats"; j["value"] = a.value.floats; break;
    case K::Strings: {
      j["kind"] = "strings";
      json arr = json::array();
      for (StringId s : a.value.strings) arr.push_back(std::string(m.str(s)));
      j["value"] = std::move(arr);
      break;
    }
    case K::Tensor: j["kind"] = "tensor"; j["value"] = tensor_ref_json(m, a.value.tensor); break;
    case K::Graph:  j["kind"] = "graph";  j["value"] = a.value.graph; break;
    case K::None:   j["kind"] = "none";   j["value"] = nullptr; break;
  }
  return j;
}

// One row of the `nodes` listing: enough to decide where to drill next without
// a second call per node (op, category, cost), but not the full attribute dump.
json node_row_json(const ir::Model& m, const ir::Graph& g, uint32_t idx,
                   const CostReport& cost) {
  const ir::Node& n = g.nodes[idx];
  json j;
  j["index"] = idx;
  j["name"] = std::string(m.str(n.name));
  j["op"] = std::string(m.str(n.op_type));
  j["category"] = category_name(categorize_op(m.str(n.op_type)));
  j["inputs"] = n.inputs.count;
  j["outputs"] = n.outputs.count;
  if (idx < cost.per_node.size()) {
    const NodeCost& c = cost.per_node[idx];
    j["flops"] = c.flops_known ? json(c.flops) : json(nullptr);
    j["params"] = c.params;
  }
  return j;
}

// Resolve a node from "<name>" (exact match) or "#<index>". Exactness is
// deliberate: discovery belongs to the `search` verb, so a lookup here either
// hits the one intended node or fails loudly.
Result<uint32_t> resolve_node(const ir::Model& m, const ir::Graph& g,
                              const std::string& target) {
  if (!target.empty() && target[0] == '#') {
    auto idx = parse_u64(target.substr(1), "node index");
    if (!idx) return idx.error();
    if (*idx >= g.nodes.size())
      return err("query: node index " + target + " out of range (graph has " +
                 std::to_string(g.nodes.size()) + " nodes)");
    return static_cast<uint32_t>(*idx);
  }
  for (uint32_t i = 0; i < g.nodes.size(); ++i)
    if (m.str(g.nodes[i].name) == target) return i;
  return err("query: no node named '" + target + "' (use '#<index>' or the search verb)");
}

Result<uint32_t> resolve_graph(const HeadlessModel& hm, const Options& o) {
  uint32_t gi = 0;
  if (const std::string* g = o.flag("graph")) {
    auto v = parse_u64(*g, "--graph");
    if (!v) return v.error();
    gi = static_cast<uint32_t>(*v);
  }
  if (!hm.model.has_graph || gi >= hm.model.graphs.size())
    return err("query: graph " + std::to_string(gi) + " does not exist (model has " +
               std::to_string(hm.model.graphs.size()) + " graph(s))");
  return gi;
}

json envelope(std::string_view verb, const HeadlessModel& hm) {
  json j;
  j["schema"] = kQuerySchema;
  j["verb"] = std::string(verb);
  j["model"] = hm.display_path;
  j["format"] = std::string(hm.model.str(hm.model.format_name));
  return j;
}

// ---------------------------------------------------------------------------
// Verbs.
// ---------------------------------------------------------------------------
Result<std::string> verb_summary(const HeadlessModel& hm) {
  json j = envelope("summary", hm);
  // The report is itself a documented schema; embed it whole rather than
  // duplicating its fields, so the two can never drift.
  j["report"] = json::parse(build_report_json(hm.model));
  return j.dump();
}

Result<std::string> verb_io(const HeadlessModel& hm, const Options& o) {
  auto gi = resolve_graph(hm, o);
  if (!gi) return gi.error();
  const ir::Graph& g = hm.model.graphs[*gi];
  json j = envelope("io", hm);
  j["graph"] = *gi;
  json in = json::array(), out = json::array();
  for (uint32_t v : g.graph_inputs) in.push_back(value_json(hm.model, g.values[v]));
  for (uint32_t v : g.graph_outputs) out.push_back(value_json(hm.model, g.values[v]));
  j["inputs"] = std::move(in);
  j["outputs"] = std::move(out);
  return j.dump();
}

Result<std::string> verb_nodes(const HeadlessModel& hm, const Options& o) {
  auto gi = resolve_graph(hm, o);
  if (!gi) return gi.error();
  const ir::Graph& g = hm.model.graphs[*gi];

  uint64_t limit = 100, offset = 0;
  if (const std::string* s = o.flag("limit")) {
    auto v = parse_u64(*s, "--limit");
    if (!v) return v.error();
    limit = *v;
  }
  if (const std::string* s = o.flag("offset")) {
    auto v = parse_u64(*s, "--offset");
    if (!v) return v.error();
    offset = *v;
  }
  const std::string* op = o.flag("op");
  std::string contains = o.flag("contains") ? lower(*o.flag("contains")) : std::string();

  const CostReport cost = compute_cost(hm.model, *gi);

  json rows = json::array();
  uint64_t matched = 0;
  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    const ir::Node& n = g.nodes[i];
    if (op && lower(hm.model.str(n.op_type)) != lower(*op)) continue;
    if (!contains.empty() &&
        lower(hm.model.str(n.name)).find(contains) == std::string::npos)
      continue;
    if (matched >= offset && matched < offset + limit)
      rows.push_back(node_row_json(hm.model, g, i, cost));
    ++matched;
  }

  json j = envelope("nodes", hm);
  j["graph"] = *gi;
  j["total_matched"] = matched;
  j["offset"] = offset;
  j["returned"] = rows.size();
  j["nodes"] = std::move(rows);
  return j.dump();
}

Result<std::string> verb_node(const HeadlessModel& hm, const Options& o) {
  if (o.positional.size() < 2) return err("query node: expected <model> <name|#index>");
  auto gi = resolve_graph(hm, o);
  if (!gi) return gi.error();
  const ir::Graph& g = hm.model.graphs[*gi];
  auto ni = resolve_node(hm.model, g, o.positional[1]);
  if (!ni) return ni.error();
  const ir::Node& n = g.nodes[*ni];

  json j = envelope("node", hm);
  j["graph"] = *gi;
  j["index"] = *ni;
  j["name"] = std::string(hm.model.str(n.name));
  j["op"] = std::string(hm.model.str(n.op_type));
  j["category"] = category_name(categorize_op(hm.model.str(n.op_type)));
  j["subgraph"] = n.subgraph;

  json inputs = json::array(), outputs = json::array();
  for (uint32_t s = 0; s < n.inputs.count; ++s)
    inputs.push_back(value_json(hm.model, g.values[g.edge_refs[n.inputs.begin + s]]));
  for (uint32_t s = 0; s < n.outputs.count; ++s)
    outputs.push_back(value_json(hm.model, g.values[g.edge_refs[n.outputs.begin + s]]));
  j["inputs"] = std::move(inputs);
  j["outputs"] = std::move(outputs);

  json attrs = json::array();
  for (uint32_t a = 0; a < n.attributes.count; ++a)
    attrs.push_back(attr_json(hm.model, g.attributes[n.attributes.begin + a]));
  j["attributes"] = std::move(attrs);

  const CostReport cost = compute_cost(hm.model, *gi);
  if (*ni < cost.per_node.size()) {
    const NodeCost& c = cost.per_node[*ni];
    json cj;
    cj["flops"] = c.flops_known ? json(c.flops) : json(nullptr);
    cj["params"] = c.params;
    cj["weight_bytes"] = c.weight_bytes;
    cj["activation_bytes"] = c.act_bytes;
    cj["arithmetic_intensity"] = c.intensity_known() ? json(c.arithmetic_intensity()) : json(nullptr);
    j["cost"] = std::move(cj);
  }

  GraphAdjacency adj;
  adj.build(hm.model, *gi);
  auto emit_nodes = [&](const std::vector<uint32_t>& ids) {
    json arr = json::array();
    for (uint32_t id : ids) {
      json e;
      e["index"] = id;
      e["name"] = std::string(hm.model.str(g.nodes[id].name));
      e["op"] = std::string(hm.model.str(g.nodes[id].op_type));
      arr.push_back(std::move(e));
    }
    return arr;
  };
  j["predecessors"] = emit_nodes(adj.reachable_pred(*ni, 1, 64));
  j["successors"] = emit_nodes(adj.reachable_succ(*ni, 1, 64));
  return j.dump();
}

Result<std::string> verb_tensors(const HeadlessModel& hm, const Options& o) {
  uint64_t limit = 100, offset = 0;
  if (const std::string* s = o.flag("limit")) {
    auto v = parse_u64(*s, "--limit");
    if (!v) return v.error();
    limit = *v;
  }
  if (const std::string* s = o.flag("offset")) {
    auto v = parse_u64(*s, "--offset");
    if (!v) return v.error();
    offset = *v;
  }
  std::string sort = o.flag("sort") ? *o.flag("sort") : "name";
  if (sort != "name" && sort != "bytes" && sort != "params")
    return err("query tensors: --sort expects name|bytes|params");

  // Graph models list the selected graph's initializers; table-mode models
  // (GGUF/SafeTensors/...) list flat_tensors. Same split the GUI draws.
  const std::vector<ir::TensorRef>* tensors = nullptr;
  json j = envelope("tensors", hm);
  if (hm.model.has_graph) {
    auto gi = resolve_graph(hm, o);
    if (!gi) return gi.error();
    tensors = &hm.model.graphs[*gi].initializers;
    j["graph"] = *gi;
  } else {
    tensors = &hm.model.flat_tensors;
    j["graph"] = nullptr;
  }

  std::vector<uint32_t> order(tensors->size());
  for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
  const ir::Model& m = hm.model;
  if (sort == "bytes") {
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return (*tensors)[a].byte_len > (*tensors)[b].byte_len;
    });
  } else if (sort == "params") {
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return (*tensors)[a].elem_count() > (*tensors)[b].elem_count();
    });
  } else {
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return m.str((*tensors)[a].name) < m.str((*tensors)[b].name);
    });
  }

  json rows = json::array();
  for (uint64_t i = offset; i < order.size() && i < offset + limit; ++i)
    rows.push_back(tensor_ref_json(m, (*tensors)[order[i]]));
  j["total"] = tensors->size();
  j["offset"] = offset;
  j["returned"] = rows.size();
  j["tensors"] = std::move(rows);
  return j.dump();
}

Result<std::string> verb_tensor(const HeadlessModel& hm, const Options& o) {
  if (o.positional.size() < 2) return err("query tensor: expected <model> <tensor-name>");
  const std::string& name = o.positional[1];

  const ir::TensorRef* found = nullptr;
  json j = envelope("tensor", hm);
  if (hm.model.has_graph) {
    auto gi = resolve_graph(hm, o);
    if (!gi) return gi.error();
    for (const ir::TensorRef& t : hm.model.graphs[*gi].initializers)
      if (hm.model.str(t.name) == name) { found = &t; break; }
    j["graph"] = *gi;
  } else {
    for (const ir::TensorRef& t : hm.model.flat_tensors)
      if (hm.model.str(t.name) == name) { found = &t; break; }
    j["graph"] = nullptr;
  }
  if (!found) return err("query: no tensor named '" + name + "' (use the tensors verb to list)");

  j["tensor"] = tensor_ref_json(hm.model, *found);

  // The ONE payload read in the query surface, and it says so: agents (and
  // tests) can rely on every other verb never touching tensor bytes.
  Result<TensorStats> stats = compute_tensor_stats(*found, hm.file, hm.model_dir, &hm.model);
  if (!stats) return stats.error();

  json s;
  s["count"] = stats->count;
  s["min"] = stats->min;
  s["max"] = stats->max;
  s["mean"] = stats->mean;
  s["std"] = stats->std;
  s["zero_count"] = stats->zero_count;
  s["nan_inf_count"] = stats->nan_inf_count;
  s["quantized_unsupported"] = stats->quantized_unsupported;
  json hist = json::array();
  for (uint64_t b : stats->histogram) hist.push_back(b);
  s["histogram"] = std::move(hist);
  s["hist_min"] = stats->hist_min;
  s["hist_max"] = stats->hist_max;
  j["stats"] = std::move(s);
  return j.dump();
}

Result<std::string> verb_search(const HeadlessModel& hm, const Options& o) {
  if (o.positional.size() < 2) return err("query search: expected <model> <query>");
  uint64_t limit = 50;
  if (const std::string* s = o.flag("limit")) {
    auto v = parse_u64(*s, "--limit");
    if (!v) return v.error();
    limit = *v;
  }

  SearchIndex index;
  index.build(hm.model);
  // types_ready=true is safe here: shape inference already ran synchronously in
  // load_model_headless, and this process is single-threaded — the data race
  // the GUI guards against cannot occur.
  std::vector<SearchHit> hits = index.query(o.positional[1], hm.model, true, limit);

  json rows = json::array();
  for (const SearchHit& h : hits) {
    const SearchEntry& e = index.entries()[h.entry];
    json r;
    r["display"] = e.display;
    switch (e.kind) {
      case SearchKind::Node:   r["kind"] = "node"; break;
      case SearchKind::OpType: r["kind"] = "op"; break;
      case SearchKind::Value:  r["kind"] = "value"; break;
      case SearchKind::Tensor: r["kind"] = "tensor"; break;
    }
    r["graph"] = e.graph;
    r["ref"] = e.ref;
    r["score"] = h.score;
    rows.push_back(std::move(r));
  }
  json j = envelope("search", hm);
  j["query"] = o.positional[1];
  j["returned"] = rows.size();
  j["hits"] = std::move(rows);
  return j.dump();
}

Result<std::string> verb_neighbors(const HeadlessModel& hm, const Options& o) {
  if (o.positional.size() < 2) return err("query neighbors: expected <model> <name|#index>");
  auto gi = resolve_graph(hm, o);
  if (!gi) return gi.error();
  const ir::Graph& g = hm.model.graphs[*gi];
  auto ni = resolve_node(hm.model, g, o.positional[1]);
  if (!ni) return ni.error();

  uint64_t hops = 1, cap = 256;
  if (const std::string* s = o.flag("hops")) {
    auto v = parse_u64(*s, "--hops");
    if (!v) return v.error();
    hops = *v;
  }
  if (const std::string* s = o.flag("cap")) {
    auto v = parse_u64(*s, "--cap");
    if (!v) return v.error();
    cap = *v;
  }
  std::string dir = o.flag("dir") ? *o.flag("dir") : "both";
  if (dir != "in" && dir != "out" && dir != "both")
    return err("query neighbors: --dir expects in|out|both");

  GraphAdjacency adj;
  adj.build(hm.model, *gi);
  auto emit = [&](const std::vector<uint32_t>& ids) {
    json arr = json::array();
    for (uint32_t id : ids) {
      json e;
      e["index"] = id;
      e["name"] = std::string(hm.model.str(g.nodes[id].name));
      e["op"] = std::string(hm.model.str(g.nodes[id].op_type));
      arr.push_back(std::move(e));
    }
    return arr;
  };

  json j = envelope("neighbors", hm);
  j["graph"] = *gi;
  j["node"] = *ni;
  j["hops"] = hops;
  if (dir != "out")
    j["predecessors"] = emit(adj.reachable_pred(*ni, static_cast<uint32_t>(hops),
                                                static_cast<uint32_t>(cap)));
  if (dir != "in")
    j["successors"] = emit(adj.reachable_succ(*ni, static_cast<uint32_t>(hops),
                                              static_cast<uint32_t>(cap)));
  return j.dump();
}

// Per-node cost ranking — the analyzer panel's table as JSON. Totals live in
// `summary`; this verb answers "where is the cost" rather than "how much".
Result<std::string> verb_cost(const HeadlessModel& hm, const Options& o) {
  auto gi = resolve_graph(hm, o);
  if (!gi) return gi.error();
  const ir::Graph& g = hm.model.graphs[*gi];

  uint64_t limit = 25;
  if (const std::string* s = o.flag("limit")) {
    auto v = parse_u64(*s, "--limit");
    if (!v) return v.error();
    limit = *v;
  }
  std::string by = o.flag("by") ? *o.flag("by") : "flops";
  if (by != "flops" && by != "params" && by != "weight_bytes" && by != "act_bytes" &&
      by != "intensity")
    return err("query cost: --by expects flops|params|weight_bytes|act_bytes|intensity");

  const CostReport cost = compute_cost(hm.model, *gi);
  std::vector<uint32_t> order(cost.per_node.size());
  for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
  auto key = [&](uint32_t i) -> double {
    const NodeCost& c = cost.per_node[i];
    if (by == "flops") return c.flops_known ? static_cast<double>(c.flops) : -1.0;
    if (by == "params") return static_cast<double>(c.params);
    if (by == "weight_bytes") return static_cast<double>(c.weight_bytes);
    if (by == "act_bytes") return static_cast<double>(c.act_bytes);
    return c.intensity_known() ? c.arithmetic_intensity() : -1.0;
  };
  std::stable_sort(order.begin(), order.end(),
                   [&](uint32_t a, uint32_t b) { return key(a) > key(b); });

  json rows = json::array();
  for (uint64_t i = 0; i < order.size() && i < limit; ++i) {
    const uint32_t idx = order[i];
    const NodeCost& c = cost.per_node[idx];
    json r;
    r["index"] = idx;
    r["name"] = std::string(hm.model.str(g.nodes[idx].name));
    r["op"] = std::string(hm.model.str(g.nodes[idx].op_type));
    r["flops"] = c.flops_known ? json(c.flops) : json(nullptr);
    r["params"] = c.params;
    r["weight_bytes"] = c.weight_bytes;
    r["activation_bytes"] = c.act_bytes;
    r["arithmetic_intensity"] =
        c.intensity_known() ? json(c.arithmetic_intensity()) : json(nullptr);
    rows.push_back(std::move(r));
  }

  json j = envelope("cost", hm);
  j["graph"] = *gi;
  j["by"] = by;
  j["nodes_total"] = cost.per_node.size();
  j["returned"] = rows.size();
  j["nodes"] = std::move(rows);
  return j.dump();
}

// Model diff — the GUI's added/removed/changed tinting as JSON. The second
// model loads through the same headless pipeline as the first, so both sides
// are shape-inferred identically.
Result<std::string> verb_diff(const HeadlessModel& hm, const Options& o) {
  if (o.positional.size() < 2) return err("query diff: expected <model-a> <model-b>");
  auto other = load_model_headless(o.positional[1]);
  if (!other) return other.error();

  auto gi = resolve_graph(hm, o);
  if (!gi) return gi.error();
  if (!other->model.has_graph || other->model.graphs.empty())
    return err("query diff: comparison model has no compute graph");

  uint64_t limit = 100;
  if (const std::string* s = o.flag("limit")) {
    auto v = parse_u64(*s, "--limit");
    if (!v) return v.error();
    limit = *v;
  }
  std::string match = o.flag("match") ? *o.flag("match") : "name";
  if (match != "name" && match != "topology")
    return err("query diff: --match expects name|topology");

  const ModelDiffResult d =
      diff_models(hm.model, *gi, other->model, 0,
                  match == "name" ? DiffMatch::NameThenTopology : DiffMatch::TopologyOnly);
  if (!d.valid) return err("query diff: graph index out of range");

  json j = envelope("diff", hm);
  j["model_b"] = other->display_path;
  j["graph"] = *gi;
  j["match"] = match;
  j["same"] = d.same;
  j["added"] = d.added;
  j["removed"] = d.removed;
  j["changed"] = d.changed;

  // The change lists an agent acts on. Capped by --limit per list; the counts
  // above are always exact.
  const ir::Graph& ga = hm.model.graphs[*gi];
  const ir::Graph& gb = other->model.graphs[0];
  auto emit = [&](const ir::Model& m, const ir::Graph& g, const std::vector<DiffStatus>& st,
                  DiffStatus want) {
    json arr = json::array();
    for (uint32_t i = 0; i < st.size() && arr.size() < limit; ++i) {
      if (st[i] != want) continue;
      json e;
      e["index"] = i;
      e["name"] = std::string(m.str(g.nodes[i].name));
      e["op"] = std::string(m.str(g.nodes[i].op_type));
      arr.push_back(std::move(e));
    }
    return arr;
  };
  j["changed_nodes"] = emit(hm.model, ga, d.a_status, DiffStatus::Changed);
  j["removed_nodes"] = emit(hm.model, ga, d.a_status, DiffStatus::Removed);
  j["added_nodes"] = emit(other->model, gb, d.b_status, DiffStatus::Added);
  return j.dump();
}

}  // namespace

Result<HeadlessModel> load_model_headless(const std::string& path) {
  // Resolve a .mlpackage bundle to its inner model file (a plain path passes
  // through unchanged), exactly as ModelSession::open_async does, so external
  // weights resolve from the right directory if a payload is later read.
  ResolvedModelPath resolved = resolve_model_path(path);

  auto mapped = MappedFile::open(resolved.map_path);
  if (!mapped) return mapped.error();

  HeadlessModel hm;
  hm.file = mapped.take();
  hm.display_path = resolved.display_path;
  hm.model_dir = dir_of(resolved.map_path);

  const std::string ext = ext_of(resolved.map_path);
  ProgressSink progress;
  Result<ir::Model> parsed = parse_model(hm.file, ext, progress);
  if (!parsed) return parsed.error();
  hm.model = parsed.take();

  // ONNX ships incomplete value_info; run the same best-effort shape inference
  // the GUI's ShapeInferJob runs so shapes/dtypes and cost are meaningful.
  if (hm.model.has_graph && hm.model.str(hm.model.format_name) == "ONNX") {
    infer_shapes_ext(hm.model, 0, hm.file.data(), hm.file.size(), &progress);
  }
  return hm;
}

Result<std::string> run_query(const std::vector<std::string>& args) {
  if (args.empty())
    return err(
        "query: expected a verb "
        "(summary|io|nodes|node|tensors|tensor|search|neighbors|cost|diff)");
  const std::string& verb = args[0];

  // Per-verb flag allowlists: a typo'd or misplaced flag is an error.
  std::vector<std::string_view> known;
  if (verb == "summary") known = {};
  else if (verb == "io") known = {"graph"};
  else if (verb == "nodes") known = {"graph", "op", "contains", "limit", "offset"};
  else if (verb == "node") known = {"graph"};
  else if (verb == "tensors") known = {"graph", "sort", "limit", "offset"};
  else if (verb == "tensor") known = {"graph"};
  else if (verb == "search") known = {"limit"};
  else if (verb == "neighbors") known = {"graph", "dir", "hops", "cap"};
  else if (verb == "cost") known = {"graph", "by", "limit"};
  else if (verb == "diff") known = {"graph", "match", "limit"};
  else return err("query: unknown verb '" + verb + "'");

  auto opts = parse_options(args, 1, known);
  if (!opts) return opts.error();
  if (opts->positional.empty()) return err("query " + verb + ": expected a model file path");

  auto hm = load_model_headless(opts->positional[0]);
  if (!hm) return hm.error();

  if (verb == "summary") return verb_summary(*hm);
  if (verb == "io") return verb_io(*hm, *opts);
  if (verb == "nodes") return verb_nodes(*hm, *opts);
  if (verb == "node") return verb_node(*hm, *opts);
  if (verb == "tensors") return verb_tensors(*hm, *opts);
  if (verb == "tensor") return verb_tensor(*hm, *opts);
  if (verb == "search") return verb_search(*hm, *opts);
  if (verb == "cost") return verb_cost(*hm, *opts);
  if (verb == "diff") return verb_diff(*hm, *opts);
  return verb_neighbors(*hm, *opts);
}

bool wants_query(int argc, char** argv) {
  return argc > 1 && std::string_view(argv[1]) == "query";
}

int run_query_cli(int argc, char** argv) {
  std::vector<std::string> args;
  for (int i = 2; i < argc; ++i) args.emplace_back(argv[i]);
  Result<std::string> out = run_query(args);
  if (!out) {
    std::fprintf(stderr, "netvis query: %s\n", out.error().message.c_str());
    return 1;
  }
  std::printf("%s\n", out->c_str());
  return 0;
}

}  // namespace netvis
