// engine/ModelDiff.cpp — structural diff between two models (v0.2.0 model diff).
//
// MATCHING STRATEGY (see header): the two models have INDEPENDENT StringArenas,
// so a raw StringId is meaningless across models — every cross-model comparison
// goes through model.str(id) -> string_view CONTENT. We match in two passes:
//   (a) by node NAME content, but only for names that are non-empty AND unique
//       within their own graph (an ambiguous/duplicate name is not a reliable
//       key);
//   (b) fall back to structural fingerprint (node_fingerprint) aligned by
//       topological position for the still-unmatched nodes.
// A matched pair with equal fingerprint => Same; matched-by-name but differing
// fingerprint => Changed; unmatched in A => Removed; unmatched in B => Added.
//
// SAFETY: reads only op_type / name / attributes / arity / topology (producer
// indices). NEVER reads ValueInfo.shape/dtype, so it is race-free against
// background shape inference mutating shapes on either model.
#include "engine/ModelDiff.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "engine/CollapseTree.h"  // node_fingerprint

namespace netvis {

namespace {

// Deterministic topological rank per node (Kahn, ties broken by node index).
// Mirrors CollapseTree's edge derivation. Cyclic remnants get ranks after the
// acyclic prefix, in index order. rank[i] is a total order over [0, n).
std::vector<uint32_t> topo_rank(const ir::Graph& g) {
  const size_t n = g.nodes.size();
  std::vector<uint32_t> indeg(n, 0);
  std::vector<std::vector<uint32_t>> succ(n);

  auto for_each_pred = [&](uint32_t ni, auto&& fn) {
    const ir::Node& nd = g.nodes[ni];
    const uint32_t b = nd.inputs.begin;
    const uint32_t e = nd.inputs.begin + nd.inputs.count;
    for (uint32_t k = b; k < e && k < g.edge_refs.size(); ++k) {
      uint32_t vidx = g.edge_refs[k];
      if (vidx >= g.values.size()) continue;
      int32_t p = g.values[vidx].producer;
      if (p >= 0 && static_cast<size_t>(p) < n &&
          static_cast<uint32_t>(p) != ni)
        fn(static_cast<uint32_t>(p));
    }
  };
  for (uint32_t i = 0; i < n; ++i) {
    for_each_pred(i, [&](uint32_t p) {
      ++indeg[i];
      succ[p].push_back(i);
    });
  }

  std::vector<uint32_t> ready;
  for (uint32_t i = 0; i < n; ++i)
    if (indeg[i] == 0) ready.push_back(i);

  std::vector<uint32_t> rank(n, 0);
  std::vector<bool> done(n, false);
  uint32_t next_rank = 0;
  while (!ready.empty()) {
    auto it = std::min_element(ready.begin(), ready.end());
    uint32_t v = *it;
    ready.erase(it);
    if (done[v]) continue;
    done[v] = true;
    rank[v] = next_rank++;
    for (uint32_t s : succ[v])
      if (indeg[s] > 0 && --indeg[s] == 0) ready.push_back(s);
  }
  // Cycle fallback: any node not emitted gets a rank after the acyclic prefix,
  // in index order, so the rank remains a deterministic total order.
  for (uint32_t i = 0; i < n; ++i)
    if (!done[i]) rank[i] = next_rank++;
  return rank;
}

// Map each UNIQUE non-empty node-name CONTENT to its node index. Names that are
// empty or appear more than once are excluded (value UINT32_MAX marks a
// collision so a later occurrence cannot resurrect it).
std::unordered_map<std::string_view, uint32_t> unique_name_index(
    const ir::Model& model, const ir::Graph& g) {
  std::unordered_map<std::string_view, uint32_t> idx;
  for (uint32_t i = 0; i < g.nodes.size(); ++i) {
    std::string_view nm = model.str(g.nodes[i].name);
    if (nm.empty()) continue;
    auto [it, inserted] = idx.emplace(nm, i);
    if (!inserted) it->second = UINT32_MAX;  // duplicate -> ambiguous
  }
  return idx;
}

}  // namespace

ModelDiffResult diff_models(const ir::Model& model_a, uint32_t graph_a,
                            const ir::Model& model_b, uint32_t graph_b) {
  ModelDiffResult r;
  if (graph_a >= model_a.graphs.size() || graph_b >= model_b.graphs.size()) {
    r.valid = false;
    return r;
  }
  r.valid = true;

  const ir::Graph& ga = model_a.graphs[graph_a];
  const ir::Graph& gb = model_b.graphs[graph_b];
  const size_t na = ga.nodes.size();
  const size_t nb = gb.nodes.size();

  // Per-node fingerprints (structural; name-independent).
  std::vector<uint64_t> fa(na), fb(nb);
  for (uint32_t i = 0; i < na; ++i)
    fa[i] = node_fingerprint(model_a, ga, ga.nodes[i]);
  for (uint32_t i = 0; i < nb; ++i)
    fb[i] = node_fingerprint(model_b, gb, gb.nodes[i]);

  r.a_to_b.assign(na, -1);
  r.b_to_a.assign(nb, -1);
  r.a_status.assign(na, DiffStatus::Removed);  // default until matched
  r.b_status.assign(nb, DiffStatus::Added);

  // (a) Name matching over content-unique names present in both graphs.
  auto name_a = unique_name_index(model_a, ga);
  auto name_b = unique_name_index(model_b, gb);
  for (auto& [nm, ai] : name_a) {
    if (ai == UINT32_MAX) continue;
    auto it = name_b.find(nm);
    if (it == name_b.end() || it->second == UINT32_MAX) continue;
    uint32_t bi = it->second;
    r.a_to_b[ai] = static_cast<int32_t>(bi);
    r.b_to_a[bi] = static_cast<int32_t>(ai);
  }

  // (b) Fingerprint fallback for still-unmatched nodes, aligned by topological
  // position. Bucket unmatched A/B nodes by fingerprint; within a bucket, sort
  // both sides by topo rank and pair them up positionally. Equal fingerprint =>
  // the pair is structurally identical (Same).
  {
    std::vector<uint32_t> rank_a = topo_rank(ga);
    std::vector<uint32_t> rank_b = topo_rank(gb);

    std::unordered_map<uint64_t, std::vector<uint32_t>> bucket_a, bucket_b;
    for (uint32_t i = 0; i < na; ++i)
      if (r.a_to_b[i] < 0) bucket_a[fa[i]].push_back(i);
    for (uint32_t i = 0; i < nb; ++i)
      if (r.b_to_a[i] < 0) bucket_b[fb[i]].push_back(i);

    for (auto& [fp, av] : bucket_a) {
      auto it = bucket_b.find(fp);
      if (it == bucket_b.end()) continue;
      std::vector<uint32_t>& bv = it->second;
      std::sort(av.begin(), av.end(),
                [&](uint32_t x, uint32_t y) { return rank_a[x] < rank_a[y]; });
      std::sort(bv.begin(), bv.end(),
                [&](uint32_t x, uint32_t y) { return rank_b[x] < rank_b[y]; });
      size_t m = std::min(av.size(), bv.size());
      for (size_t k = 0; k < m; ++k) {
        r.a_to_b[av[k]] = static_cast<int32_t>(bv[k]);
        r.b_to_a[bv[k]] = static_cast<int32_t>(av[k]);
      }
    }
  }

  // Classify + count. Matched pairs: equal fingerprint => Same, else Changed.
  for (uint32_t i = 0; i < na; ++i) {
    int32_t bi = r.a_to_b[i];
    if (bi < 0) {
      r.a_status[i] = DiffStatus::Removed;
      ++r.removed;
    } else if (fa[i] == fb[static_cast<size_t>(bi)]) {
      r.a_status[i] = DiffStatus::Same;
      ++r.same;
    } else {
      r.a_status[i] = DiffStatus::Changed;
      ++r.changed;
    }
  }
  for (uint32_t i = 0; i < nb; ++i) {
    int32_t ai = r.b_to_a[i];
    if (ai < 0) {
      r.b_status[i] = DiffStatus::Added;
      ++r.added;
    } else if (fb[i] == fa[static_cast<size_t>(ai)]) {
      r.b_status[i] = DiffStatus::Same;
    } else {
      r.b_status[i] = DiffStatus::Changed;
    }
  }

  return r;
}

}  // namespace netvis
