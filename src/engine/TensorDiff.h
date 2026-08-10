// engine/TensorDiff.h — cross-model tensor matching + weight-stat deltas
// (#34 per-tensor weight-stat diff, #50 same-tensor side-by-side compare).
//
// DECISION (v0.9.1b): ModelDiff answers "which NODES changed" and deliberately
// reads no shapes and no payload. This file answers the other half — "how did the
// WEIGHTS change" — and therefore does read payload, so it is split out rather
// than bolted onto ModelDiff, and its two phases are kept strictly apart:
//
//   PHASE 1 (free): match_tensors / enumerate_tensors / find_tensor_by_name walk
//   names, shapes and dtypes only. Zero payload reads, safe to run every frame,
//   safe to run on the main thread. This is what fills the comparison table.
//
//   PHASE 2 (on demand): compute_tensor_stat_delta decodes ONE matched pair and
//   is the expensive part. It must run on a worker, exactly like the inspector's
//   TensorDecodeJob, and it is never run implicitly for a whole model — a 7B
//   checkpoint has thousands of tensors and "diff every weight" would read the
//   entire payload of BOTH models, which is precisely what the zero-payload
//   thesis exists to prevent. The view drives it one pair at a time.
//
// Matching is by name CONTENT (model.str(id)), never by StringId: the two models
// have independent StringArenas, so a raw id is meaningless across them. Same
// rule as ModelDiff.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/MappedFile.h"
#include "core/Result.h"
#include "engine/TensorStats.h"
#include "ir/IR.h"

namespace netvis {

// Where a tensor lives in a Model. Tensors sit in one of two places: the
// graph-less flat_tensors list (GGUF/SafeTensors/state-dict) or a graph's
// initializers. One locator addresses both.
struct TensorLocator {
  int32_t graph = -1;  // -1 => Model::flat_tensors; else index into Model::graphs
  int32_t index = -1;  // index into that container
  bool valid() const { return index >= 0; }
  bool operator==(const TensorLocator& o) const {
    return graph == o.graph && index == o.index;
  }
};

// Resolve a locator against a model. Returns nullptr if the locator is invalid
// or out of range (hostile/stale input must never index OOB).
const ir::TensorRef* resolve_tensor(const ir::Model& m, TensorLocator loc);

// Every tensor in the model, in a STABLE order: flat_tensors first (graph = -1,
// ascending index), then graph 0..N-1 initializers. Deterministic. Zero payload.
std::vector<TensorLocator> enumerate_tensors(const ir::Model& m);

// First tensor whose interned name equals `name` by content, in the
// enumerate_tensors order. Returns an invalid locator if there is no such
// tensor. An empty `name` never matches (unnamed tensors are not addressable
// this way — otherwise every unnamed tensor in model B would "match" one in A).
TensorLocator find_tensor_by_name(const ir::Model& m, std::string_view name);

// A tensor present in BOTH models under the same name.
struct TensorMatch {
  TensorLocator a, b;
  bool shape_equal = false;   // identical rank + dims
  bool dtype_equal = false;   // identical ir::DType
  // Element counts, for a params-level delta that needs no payload at all. 0
  // when a dim is dynamic/unset (elem_count()'s own convention).
  int64_t a_elems = 0, b_elems = 0;
  uint64_t a_bytes = 0, b_bytes = 0;
};

// Match every named tensor of `a` against `b` by name content. Deterministic,
// ordered by `a`'s enumerate_tensors order. Tensors present in only one model are
// omitted — the node-level ModelDiff already reports added/removed structure, and
// an unmatched weight has nothing to delta against. Unnamed tensors are skipped.
// Zero payload reads.
std::vector<TensorMatch> match_tensors(const ir::Model& a, const ir::Model& b);

// The decoded result for one matched pair. Both sides are attempted; either may
// fail independently (a quantized GGUF tensor yields quantized_unsupported, a
// truncated file yields an error) and the panel reports what it got.
struct TensorStatDelta {
  bool a_ok = false, b_ok = false;
  TensorStats a, b;
  std::string a_error, b_error;  // non-empty iff the matching *_ok is false

  bool both_ok() const { return a_ok && b_ok; }
  // Deltas are B - A, matching the sign convention the #31 cost delta already
  // uses in DiffPanel ("B - A"). Only meaningful when both_ok().
  double d_min() const { return b.min - a.min; }
  double d_max() const { return b.max - a.max; }
  double d_mean() const { return b.mean - a.mean; }
  double d_std() const { return b.std - a.std; }
  // Counts are unsigned; return a signed delta so a shrink reads as negative
  // rather than wrapping to ~1.8e19 (the bug class the #31 signed_row helper
  // exists to avoid).
  int64_t d_zero_count() const {
    return static_cast<int64_t>(b.zero_count) - static_cast<int64_t>(a.zero_count);
  }
  int64_t d_nan_inf_count() const {
    return static_cast<int64_t>(b.nan_inf_count) -
           static_cast<int64_t>(a.nan_inf_count);
  }
};

// Decode ONE matched pair and return both sides' stats.
//
// PAYLOAD-READING — must be called from a worker, never the main thread. Each
// side is resolved through the normal compute_tensor_stats path (so external
// data, blob_indirect and quantized-unsupported all behave identically to the
// inspector). `fa`/`dir_a`/`ma` describe model A, `fb`/`dir_b`/`mb` model B; the
// two models have independent mappings and arenas and must not be crossed.
//
// Returns an error Result only if BOTH sides fail to even be attempted; a
// single-side failure is reported in the struct so the panel can show the half it
// has. Calls mark_payload_read() once per side.
Result<TensorStatDelta> compute_tensor_stat_delta(
    const ir::TensorRef& ta, const MappedFile& fa, const std::string& dir_a,
    const ir::Model* ma, const ir::TensorRef& tb, const MappedFile& fb,
    const std::string& dir_b, const ir::Model* mb);

}  // namespace netvis
