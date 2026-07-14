// engine/CollapseTree.cpp — repeated-block detection + collapse state.
//
// PERF (spec §7.1): this runs BEFORE layout on purpose. Modern models are
// dominated by repeated blocks (e.g. 32 identical decoder layers). By collapsing
// each block to a single super-node here, the default layout operates on a few
// dozen display nodes instead of 100k leaves — that is what makes the first
// paint sub-second. Detection itself is at most a couple of linear-ish passes
// over the node list (see build()); it is negligible next to layout.
#include "engine/CollapseTree.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/Hash.h"

namespace netvis {

// ---------------------------------------------------------------------------
// node_fingerprint
// ---------------------------------------------------------------------------
// Structural fingerprint of a single node (spec §7.1.1). Combines op_type,
// input/output arity and every attribute NAME + typed VALUE. EXCLUDES the node
// name and any "doc_string" attribute so that two structurally identical nodes
// in different block instances (which differ only by name) fingerprint equal.
uint64_t node_fingerprint(const ir::Model& model, const ir::Graph& g,
                          const ir::Node& n) {
  Hasher h;
  h.str(model.str(n.op_type));
  h.u32(n.inputs.count);   // input arity
  h.u32(n.outputs.count);  // output arity

  const uint32_t attr_begin = n.attributes.begin;
  const uint32_t attr_end = n.attributes.begin + n.attributes.count;
  for (uint32_t ai = attr_begin; ai < attr_end && ai < g.attributes.size();
       ++ai) {
    const ir::Attribute& a = g.attributes[ai];
    std::string_view name = model.str(a.name);
    if (name == "doc_string") continue;  // excluded per spec
    h.str(name);
    h.u32(static_cast<uint32_t>(a.value.kind));
    using K = ir::AttrValue::Kind;
    switch (a.value.kind) {
      case K::Int:
        h.i64(a.value.i);
        break;
      case K::Float:
        h.bytes(&a.value.f, sizeof(a.value.f));
        break;
      case K::String:
        h.str(model.str(a.value.s));
        break;
      case K::Ints:
        for (int64_t v : a.value.ints) h.i64(v);
        break;
      case K::Floats:
        for (double v : a.value.floats) h.bytes(&v, sizeof(v));
        break;
      case K::Strings:
        for (StringId sid : a.value.strings) h.str(model.str(sid));
        break;
      case K::Tensor:
        h.u32(static_cast<uint32_t>(a.value.tensor.dtype));
        for (int64_t d : a.value.tensor.shape) h.i64(d);
        break;
      case K::Graph:
        h.u32(static_cast<uint32_t>(a.value.graph));
        break;
      case K::None:
        break;
    }
  }
  return h.value();
}

namespace {

// Find the first run of ASCII digits in `s`. Returns [start,end); start==npos
// if there is no digit run.
struct DigitRun {
  size_t start = std::string_view::npos;
  size_t end = 0;
};
DigitRun first_digit_run(std::string_view s) {
  DigitRun r;
  for (size_t i = 0; i < s.size(); ++i) {
    if (std::isdigit(static_cast<unsigned char>(s[i]))) {
      r.start = i;
      size_t j = i;
      while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
      r.end = j;
      return r;
    }
  }
  return r;
}

// Derive a friendly label from a normalized prefix key like "/model/layers.#".
// Takes the last path-ish segment and strips the trailing ".#"/placeholder.
std::string label_from_prefix(std::string_view key) {
  // Trim trailing placeholder and separators.
  std::string_view core = key;
  while (!core.empty() &&
         (core.back() == '#' || core.back() == '.' || core.back() == '/' ||
          core.back() == '_' || core.back() == '-')) {
    core.remove_suffix(1);
  }
  size_t slash = core.find_last_of("/.");
  std::string_view seg =
      (slash == std::string_view::npos) ? core : core.substr(slash + 1);
  if (seg.empty()) return std::string(key);
  return std::string(seg);
}

// Compute a deterministic topological order of `g` (Kahn, ties broken by node
// index). Falls back to node order if the graph has cycles (remaining nodes
// appended in index order).
std::vector<uint32_t> topo_order(const ir::Graph& g) {
  const size_t n = g.nodes.size();
  std::vector<uint32_t> indeg(n, 0);
  // Build predecessor edges via value producers.
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
  for (uint32_t i = 0; i < n; ++i)
    for_each_pred(i, [&](uint32_t) { ++indeg[i]; });

  // Simple queue processed in index order for determinism.
  std::vector<uint32_t> order;
  order.reserve(n);
  std::vector<uint32_t> ready;
  for (uint32_t i = 0; i < n; ++i)
    if (indeg[i] == 0) ready.push_back(i);
  // Successor adjacency for decrementing indegree.
  std::vector<std::vector<uint32_t>> succ(n);
  for (uint32_t i = 0; i < n; ++i)
    for_each_pred(i, [&](uint32_t p) { succ[p].push_back(i); });

  std::vector<bool> done(n, false);
  while (!ready.empty()) {
    // Pop smallest index for determinism.
    auto it = std::min_element(ready.begin(), ready.end());
    uint32_t v = *it;
    ready.erase(it);
    if (done[v]) continue;
    done[v] = true;
    order.push_back(v);
    for (uint32_t s : succ[v]) {
      if (indeg[s] > 0 && --indeg[s] == 0) ready.push_back(s);
    }
  }
  // Cycle fallback: append any not-yet-emitted node in index order.
  if (order.size() < n)
    for (uint32_t i = 0; i < n; ++i)
      if (!done[i]) order.push_back(i);
  return order;
}

}  // namespace

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------
void CollapseTree::build(const ir::Model& model, uint32_t graph_index) {
  groups_.clear();
  display_.clear();
  expanded_.clear();
  structure_hash_ = 0;
  model_ = &model;
  graph_index_ = graph_index;
  if (graph_index >= model.graphs.size()) return;
  const ir::Graph& g = model.graphs[graph_index];
  const size_t n = g.nodes.size();

  // Per-node structural fingerprint.
  std::vector<uint64_t> fps(n);
  for (uint32_t i = 0; i < n; ++i)
    fps[i] = node_fingerprint(model, g, g.nodes[i]);

  // structure_hash_: full topology = node fingerprints (in index order) + edge
  // connectivity (each input value's producer node). Order-sensitive FNV-1a.
  {
    Hasher h;
    h.u64(n);
    for (uint32_t i = 0; i < n; ++i) {
      h.u64(fps[i]);
      const ir::Node& nd = g.nodes[i];
      const uint32_t b = nd.inputs.begin;
      const uint32_t e = nd.inputs.begin + nd.inputs.count;
      for (uint32_t k = b; k < e && k < g.edge_refs.size(); ++k) {
        uint32_t vidx = g.edge_refs[k];
        int32_t p = (vidx < g.values.size()) ? g.values[vidx].producer : -1;
        h.i64(p);
      }
    }
    structure_hash_ = h.value();
  }

  // node_group[i] = index into groups_ or UINT32_MAX (node in at most one group).
  std::vector<uint32_t> node_group(n, UINT32_MAX);

  // (a) Name-prefix grouping. Normalize a node name by replacing the first
  // integer run with '#'; the group key is the prefix up to and including that
  // run. Nodes sharing a key with >=2 distinct integer values form a block.
  {
    std::unordered_map<std::string, uint32_t> key_to_slot;
    std::vector<std::string> keys;                 // first-appearance order
    std::vector<std::vector<uint32_t>> members;    // nodes per key
    std::vector<std::vector<int64_t>> values;      // parsed integer per member

    for (uint32_t i = 0; i < n; ++i) {
      std::string_view name = model.str(g.nodes[i].name);
      if (name.empty()) continue;
      DigitRun dr = first_digit_run(name);
      if (dr.start == std::string_view::npos) continue;
      std::string key;
      key.reserve(dr.start + 1);
      key.append(name.substr(0, dr.start));
      key.push_back('#');
      // Node names are untrusted; a digit run longer than int64 can hold would
      // overflow (signed overflow is UB). ival is only a grouping discriminant,
      // so saturate at a safe width (<=18 digits always fits in int64).
      int64_t ival = 0;
      for (size_t p = dr.start; p < dr.end && (p - dr.start) < 18; ++p)
        ival = ival * 10 + (name[p] - '0');
      auto it = key_to_slot.find(key);
      uint32_t slot;
      if (it == key_to_slot.end()) {
        slot = static_cast<uint32_t>(keys.size());
        key_to_slot.emplace(key, slot);
        keys.push_back(std::move(key));
        members.emplace_back();
        values.emplace_back();
      } else {
        slot = it->second;
      }
      members[slot].push_back(i);
      values[slot].push_back(ival);
    }

    for (size_t slot = 0; slot < keys.size(); ++slot) {
      // Distinct integer values == number of block instances.
      std::vector<int64_t> distinct = values[slot];
      std::sort(distinct.begin(), distinct.end());
      distinct.erase(std::unique(distinct.begin(), distinct.end()),
                     distinct.end());
      if (distinct.size() < 2) continue;  // need >= 2 instances

      CollapseGroup grp;
      grp.label = label_from_prefix(keys[slot]);
      grp.instances = static_cast<uint32_t>(distinct.size());
      grp.member_nodes = members[slot];  // already in node-index order
      // Representative = the smallest-index instance's nodes.
      int64_t rep_val = distinct.front();
      Hasher fh;
      for (size_t m = 0; m < members[slot].size(); ++m) {
        uint32_t ni = members[slot][m];
        if (values[slot][m] == rep_val) grp.representative_nodes.push_back(ni);
      }
      // Fingerprint of the group = combined fingerprint of representative nodes.
      for (uint32_t ni : grp.representative_nodes) fh.u64(fps[ni]);
      grp.fingerprint = fh.value();

      uint32_t gidx = static_cast<uint32_t>(groups_.size());
      for (uint32_t ni : grp.member_nodes) node_group[ni] = gidx;
      groups_.push_back(std::move(grp));
    }
  }

  // (b) Rolling-hash over topological order: repeated CONSECUTIVE fingerprint
  // runs among nodes not already grouped by (a). A run is a block of length
  // p>=2 that repeats r>=2 times back-to-back. This catches repeated structure
  // that is not reflected in node names.
  {
    std::vector<uint32_t> order = topo_order(g);
    // Sequence of ungrouped nodes (preserving topo order) + their fingerprints.
    std::vector<uint32_t> seq;
    std::vector<uint64_t> sfp;
    seq.reserve(order.size());
    sfp.reserve(order.size());
    for (uint32_t ni : order) {
      if (node_group[ni] != UINT32_MAX) continue;
      seq.push_back(ni);
      sfp.push_back(fps[ni]);
    }
    const size_t m = seq.size();
    // PERF: period search is capped so this stays well bounded (build-time
    // only). For the model sizes we target the cap is never the bottleneck.
    constexpr size_t kMaxPeriod = 512;
    size_t i = 0;
    while (i < m) {
      size_t maxp = std::min<size_t>(kMaxPeriod, (m - i) / 2);
      size_t best_p = 0, best_r = 0;
      for (size_t p = 2; p <= maxp; ++p) {
        // Count consecutive repeats of block [i, i+p).
        size_t r = 1;
        while (i + (r + 1) * p <= m) {
          bool eq = true;
          for (size_t k = 0; k < p; ++k) {
            if (sfp[i + k] != sfp[i + r * p + k]) {
              eq = false;
              break;
            }
          }
          if (!eq) break;
          ++r;
        }
        if (r >= 2 && r * p > best_r * best_p) {
          best_p = p;
          best_r = r;
        }
      }
      if (best_r >= 2) {
        CollapseGroup grp;
        grp.label = "Block";
        grp.instances = static_cast<uint32_t>(best_r);
        size_t total = best_p * best_r;
        Hasher fh;
        for (size_t k = 0; k < total; ++k)
          grp.member_nodes.push_back(seq[i + k]);
        for (size_t k = 0; k < best_p; ++k) {
          grp.representative_nodes.push_back(seq[i + k]);
          fh.u64(sfp[i + k]);
        }
        grp.fingerprint = fh.value();
        // Keep member_nodes in node-index order for stable downstream use.
        std::sort(grp.member_nodes.begin(), grp.member_nodes.end());
        std::sort(grp.representative_nodes.begin(),
                  grp.representative_nodes.end());
        uint32_t gidx = static_cast<uint32_t>(groups_.size());
        for (uint32_t ni : grp.member_nodes) node_group[ni] = gidx;
        groups_.push_back(std::move(grp));
        i += total;
      } else {
        ++i;
      }
    }
  }

  // Default: everything EXPANDED, so the initial view shows every node (parity
  // with Netron). Repeated-block groups are still detected and can be collapsed
  // on demand (double-click), but we never hide nodes by default — a prior
  // default-collapsed view made large models look like they were missing nodes.
  expanded_.assign(groups_.size(), true);
  rebuild_display(g);
}

// ---------------------------------------------------------------------------
// rebuild_display
// ---------------------------------------------------------------------------
// One DisplayNode per collapsed group + each ungrouped leaf. An expanded group
// emits its representative-instance leaves (incremental expand). Emission is on
// first encounter in node-index order -> stable and deterministic.
void CollapseTree::rebuild_display(const ir::Graph& g) {
  display_.clear();
  const size_t n = g.nodes.size();

  // node_group + membership of representative sets, rebuilt from groups_.
  std::vector<uint32_t> node_group(n, UINT32_MAX);
  for (uint32_t gi = 0; gi < groups_.size(); ++gi)
    for (uint32_t ni : groups_[gi].member_nodes)
      if (ni < n) node_group[ni] = gi;

  std::vector<bool> group_emitted(groups_.size(), false);

  for (uint32_t i = 0; i < n; ++i) {
    uint32_t gi = node_group[i];
    if (gi == UINT32_MAX) {
      DisplayNode d;
      d.is_group = false;
      d.ir_node = i;
      display_.push_back(d);
      continue;
    }
    if (group_emitted[gi]) continue;
    group_emitted[gi] = true;
    if (expanded_[gi]) {
      // Expanded: emit ALL member nodes as leaves (every instance), so nothing
      // is hidden. (Members are stored in node-index order.)
      for (uint32_t ni : groups_[gi].member_nodes) {
        DisplayNode d;
        d.is_group = false;
        d.ir_node = ni;
        display_.push_back(d);
      }
    } else {
      DisplayNode d;
      d.is_group = true;
      d.group_index = gi;
      d.expanded = false;
      display_.push_back(d);
    }
  }
}

// ---------------------------------------------------------------------------
// toggle_group
// ---------------------------------------------------------------------------
bool CollapseTree::toggle_group(uint32_t group_index) {
  if (group_index >= expanded_.size()) return false;
  expanded_[group_index] = !expanded_[group_index];
  if (model_ && graph_index_ < model_->graphs.size())
    rebuild_display(model_->graphs[graph_index_]);
  return true;
}

// ---------------------------------------------------------------------------
// collapse_hash
// ---------------------------------------------------------------------------
uint64_t CollapseTree::collapse_hash() const {
  Hasher h;
  h.u64(expanded_.size());
  for (size_t i = 0; i < expanded_.size(); ++i) {
    uint8_t bit = expanded_[i] ? 1u : 0u;
    h.bytes(&bit, 1);
  }
  return h.value();
}

}  // namespace netvis
