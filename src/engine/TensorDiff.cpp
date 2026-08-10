// engine/TensorDiff.cpp — cross-model tensor matching + weight-stat deltas
// (#34/#50). See the header for the PHASE 1 (free) / PHASE 2 (payload) split
// this file implements.
#include "engine/TensorDiff.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace netvis {

const ir::TensorRef* resolve_tensor(const ir::Model& m, TensorLocator loc) {
  if (loc.index < 0) return nullptr;
  const size_t idx = static_cast<size_t>(loc.index);
  if (loc.graph < 0) {
    // flat_tensors mode (GGUF/SafeTensors/state-dict).
    if (idx >= m.flat_tensors.size()) return nullptr;
    return &m.flat_tensors[idx];
  }
  const size_t g = static_cast<size_t>(loc.graph);
  if (g >= m.graphs.size()) return nullptr;
  const ir::Graph& graph = m.graphs[g];
  if (idx >= graph.initializers.size()) return nullptr;
  return &graph.initializers[idx];
}

std::vector<TensorLocator> enumerate_tensors(const ir::Model& m) {
  std::vector<TensorLocator> out;
  // Reserve on the common case (flat_tensors XOR graphs' initializers is
  // populated in practice, per the header note) so the usual path is a single
  // allocation rather than repeated growth.
  size_t total = m.flat_tensors.size();
  for (const ir::Graph& g : m.graphs) total += g.initializers.size();
  out.reserve(total);

  for (size_t i = 0; i < m.flat_tensors.size(); ++i)
    out.push_back(TensorLocator{-1, static_cast<int32_t>(i)});

  for (size_t g = 0; g < m.graphs.size(); ++g) {
    const ir::Graph& graph = m.graphs[g];
    for (size_t i = 0; i < graph.initializers.size(); ++i)
      out.push_back(
          TensorLocator{static_cast<int32_t>(g), static_cast<int32_t>(i)});
  }
  return out;
}

TensorLocator find_tensor_by_name(const ir::Model& m, std::string_view name) {
  // An empty name never matches — see header: otherwise every unnamed tensor
  // would collide on the arena's canonical StringId{0} == "" entry.
  if (name.empty()) return TensorLocator{};
  for (const TensorLocator& loc : enumerate_tensors(m)) {
    const ir::TensorRef* t = resolve_tensor(m, loc);
    if (t != nullptr && m.str(t->name) == name) return loc;
  }
  return TensorLocator{};
}

std::vector<TensorMatch> match_tensors(const ir::Model& a, const ir::Model& b) {
  std::vector<TensorMatch> out;

  // Lookup structure: unordered_map<string_view, TensorLocator> keyed on B's
  // tensor-name bytes. B's StringArena backs names in a std::deque<std::string>
  // (see core/StringArena.h), whose element addresses never move on further
  // interning/growth — only on destruction — so a string_view into it stays
  // valid for the lifetime of `b`, which outlives this function call. The map
  // itself is function-local and never escapes, so there is no dangling risk
  // even though it holds views rather than owned strings.
  //
  // Only the FIRST occurrence of a name is kept, matching find_tensor_by_name's
  // "first match in enumerate_tensors order" contract — a duplicate name in B
  // resolves the same way whether you probe it via find_tensor_by_name or via
  // match_tensors.
  std::unordered_map<std::string_view, TensorLocator> b_by_name;
  b_by_name.reserve(b.flat_tensors.size());
  for (const TensorLocator& loc : enumerate_tensors(b)) {
    const ir::TensorRef* t = resolve_tensor(b, loc);
    if (t == nullptr) continue;
    std::string_view nm = b.str(t->name);
    if (nm.empty()) continue;
    b_by_name.emplace(nm, loc);  // emplace: no-op if nm already present
  }

  // Single pass over A, in A's enumeration order — O(n + m) total rather than
  // the O(n*m) a naive double loop would cost on tens-of-thousands-of-tensors
  // checkpoints (spec note in the header).
  for (const TensorLocator& loc_a : enumerate_tensors(a)) {
    const ir::TensorRef* ta = resolve_tensor(a, loc_a);
    if (ta == nullptr) continue;
    std::string_view nm = a.str(ta->name);
    if (nm.empty()) continue;  // unnamed tensors are not addressable by name

    auto it = b_by_name.find(nm);
    if (it == b_by_name.end()) continue;  // unmatched -> omitted (see header)
    const ir::TensorRef* tb = resolve_tensor(b, it->second);
    // Defensive: b_by_name only ever stores locators resolve_tensor(b, ...)
    // already accepted above, so this is unreachable in practice.
    if (tb == nullptr) continue;

    TensorMatch match;
    match.a = loc_a;
    match.b = it->second;
    match.shape_equal = (ta->shape == tb->shape);
    match.dtype_equal = (ta->dtype == tb->dtype);
    match.a_elems = ta->elem_count();
    match.b_elems = tb->elem_count();
    match.a_bytes = ta->byte_len;
    match.b_bytes = tb->byte_len;
    out.push_back(match);
  }
  return out;
}

Result<TensorStatDelta> compute_tensor_stat_delta(
    const ir::TensorRef& ta, const MappedFile& fa, const std::string& dir_a,
    const ir::Model* ma, const ir::TensorRef& tb, const MappedFile& fb,
    const std::string& dir_b, const ir::Model* mb) {
  TensorStatDelta d;

  // Each side is resolved through its OWN mapping/dir/model — never crossed
  // (see header: crossing them would resolve external-data paths against the
  // wrong model). Failures are independent; only report the Result-level error
  // if NEITHER side produced anything.
  Result<TensorStats> ra = compute_tensor_stats(ta, fa, dir_a, ma);
  if (ra.ok()) {
    d.a_ok = true;
    d.a = ra.take();
  } else {
    d.a_ok = false;
    d.a_error = ra.error().message;
  }

  Result<TensorStats> rb = compute_tensor_stats(tb, fb, dir_b, mb);
  if (rb.ok()) {
    d.b_ok = true;
    d.b = rb.take();
  } else {
    d.b_ok = false;
    d.b_error = rb.error().message;
  }

  if (!d.a_ok && !d.b_ok)
    return err("tensor delta: both sides failed (A: " + d.a_error +
               ", B: " + d.b_error + ")");
  return d;
}

}  // namespace netvis
