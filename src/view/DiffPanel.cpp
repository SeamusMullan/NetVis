// view/DiffPanel.cpp — comparison / model-diff panel (v0.2.0 model diff).
//
// Pure ImGui + reading published DiffLoader results (view -> engine only; no
// parser include). The panel loads comparison models, shows the
// added/removed/changed summary + lists, and drives the diff-color overlay
// resolved by diff_tint_for_display() (consulted by GraphCanvas).
//
// v0.9.1b adds two things:
//   #36 N-WAY — DiffLoader holds up to kMaxComparisons slots, each diffed
//       against the SAME primary baseline. The panel renders one row per slot
//       (the fp32 -> int8 -> int4 quant ladder) and lets the user pick which
//       slot is ACTIVE. Exactly one slot tints the graph, so everything below
//       the ladder table (summary, cost delta, node lists, export, tensor
//       diff) reads the ACTIVE slot and only the active slot.
//   #34 per-tensor weight-stat delta — a virtualized list of tensors present
//       in BOTH primary and the active comparison, with an explicit per-row
//       Compare that decodes exactly ONE pair on a worker.
//
// CACHE KEYING (read this before adding a cache here): every cross-frame cache
// in this file is a function-local static, and with N comparison slots a key
// that omits the slot will happily serve slot A's numbers under slot B. Every
// cache below is therefore stamped with a SlotKey — see its comment for what
// each field defends against. This exact bug class (stale diff tint after a
// primary reopen; stale cost report after async shape inference) has shipped
// twice in this codebase already.
#define IMGUI_DEFINE_MATH_OPERATORS
#include "view/DiffPanel.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "imgui.h"

#include "core/JobSystem.h"
#include "engine/CostModel.h"   // #31: compute_cost for the two-model cost delta
#include "engine/DiffLoader.h"
// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h that
// App.h pulls in without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/ModelDiff.h"
#include "engine/TensorDiff.h"  // #34: match_tensors / compute_tensor_stat_delta
#include "ir/IR.h"
#include "view/App.h"
#include "view/PanelHelpers.h"

// tinyfiledialogs: declared here (same as App.cpp) so the panel can open the
// comparison-model file dialog. Definition provided by the tinyfiledialogs TU.
extern "C" {
char* tinyfd_openFileDialog(const char* aTitle, const char* aDefaultPathAndFile,
                            int aNumOfFilterPatterns,
                            const char* const* aFilterPatterns,
                            const char* aSingleFilterDescription,
                            int aAllowMultipleSelects);
char const* tinyfd_saveFileDialog(char const* aTitle,
                                  char const* aDefaultPathAndFile,
                                  int aNumOfFilterPatterns,
                                  char const* const* aFilterPatterns,
                                  char const* aSingleFilterDescription);
}

namespace netvis {

namespace {

using panel_detail::grouped_count;
using panel_detail::human_bytes;
using panel_detail::icontains;
using panel_detail::shape_string;

// Diff overlay colors (dark-first, chosen distinct from the op-category palette).
constexpr ImU32 kColAdded = IM_COL32(76, 201, 120, 255);     // green
constexpr ImU32 kColRemoved = IM_COL32(224, 92, 92, 255);    // red
constexpr ImU32 kColChanged = IM_COL32(232, 168, 56, 255);   // amber
constexpr ImU32 kColNeutral = IM_COL32(160, 160, 160, 255);  // "no change"

// A one-glyph load indicator. ImGui has no spinner widget and the arc-drawing
// one in WeightInspector.cpp is panel-local; a rotating glyph costs nothing and
// keeps a ladder row (which may show three at once) cheap.
char spinner_glyph() {
  static const char kFrames[] = "|/-\\";
  return kFrames[static_cast<int>(ImGui::GetTime() * 8.0) & 3];
}

// Last path component, for the ladder's file column. The view aliases `path`,
// which is owned by the DiffLoader slot and outlives the frame.
std::string_view basename_of(const std::string& path) {
  const size_t slash = path.find_last_of("/\\");
  return slash == std::string::npos
             ? std::string_view(path)
             : std::string_view(path).substr(slash + 1);
}

// One "label   +/-value" row. The delta is computed in the UNSIGNED domain
// because the inputs are saturating uint64 that can exceed INT64_MAX — casting
// to int64 and negating would be signed-overflow UB — so a sign flag plus a
// magnitude is carried instead. Hoisted out of the #31 cost section so #34's
// zero/NaN counts reuse it: an unsigned count rendered raw wraps to ~1.8e19 on
// any shrink, which is precisely the bug this helper exists to prevent.
void signed_row(const char* label, uint64_t a, uint64_t b, bool bytes) {
  const bool neg = b < a;
  const uint64_t mag = neg ? a - b : b - a;
  const ImU32 col = mag == 0 ? kColNeutral : (neg ? kColRemoved : kColAdded);
  // grouped_count takes int64; clamp a saturated magnitude so the cast is safe.
  const int64_t mag_i = mag > static_cast<uint64_t>(INT64_MAX)
                            ? INT64_MAX
                            : static_cast<int64_t>(mag);
  const std::string dv = (neg ? "-" : "+") +
                         (bytes ? human_bytes(mag) : grouped_count(mag_i));
  ImGui::Text("%s", label);
  ImGui::SameLine(150.0f);
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(col), "%s", dv.c_str());
}

// The same row for a genuinely real-valued statistic (#34 min/max/mean/std).
// No unsigned-wrap hazard here, but NaN is possible (a tensor whose stats
// include NaN), and it is printed as-is rather than hidden or coerced to 0 — a
// NaN weight is exactly what this table exists to surface.
void float_row(const char* label, double a, double b) {
  const double d = b - a;
  ImU32 col = kColNeutral;  // NaN falls through to neutral (both tests false)
  if (d > 0.0) col = kColAdded;
  else if (d < 0.0) col = kColRemoved;
  ImGui::Text("%s", label);
  ImGui::SameLine(150.0f);
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(col), "%+.6g", d);
  ImGui::SameLine();
  ImGui::TextDisabled("(A %.6g  B %.6g)", a, b);
}

// #36: the per-slot validity guard. Pre-#36 this predicate was computed ONCE
// inline in draw_diff_panel, which is now wrong: every slot snapshots its own
// primary session/generation/graph when it is added, so slot 0 can be valid for
// this tab while slot 1 was loaded against a different one. Same three-part test
// the canvas tint uses (see diff_tint_for_display).
bool slot_primary_matches(const DiffLoader& dl, size_t i,
                          const ModelSession& s) {
  return dl.primary_session_of(i) == &s &&
         dl.primary_generation_of(i) == s.generation() &&
         dl.primary_graph_of(i) == s.current_graph();
}

// The key EVERY cross-frame cache in this panel is stamped with. Each field
// closes a distinct hole, and dropping any one of them reintroduces a
// stale-numbers bug this codebase has already shipped:
//   a / b     — the two models the cached value actually describes.
//   a_gen     — a primary reopen. current_graph_ resets to 0 on open, so a
//               graph-index match alone would pair a fresh model with stale data.
//   a_enrich  — ONNX shape inference mutates ValueInfo IN PLACE after the model
//               is published (same model pointer, same generation), so a cost
//               report built pre-inference has all-unknown FLOPs and would
//               otherwise be served forever. Same reason ViewState carries
//               cost_key_enrich.
//   a_graph   — which primary graph this slot was diffed against.
//   slot      — separates concurrently-loaded comparisons. Without it, switching
//               the active slot serves the previous slot's numbers, because the
//               remaining fields can all be stable across that switch.
//   b_path    — closes pointer ABA. remove_comparison() frees a slot's model and
//               a later add_comparison() can land the fresh ir::Model on that
//               freed address at the same slot index, making (b, slot) compare
//               equal for two unrelated files. Two live slots can never alias
//               (both models are allocated while the other is alive), so the
//               path is only load-bearing for that free-then-reallocate case. A
//               reload of the SAME path can still alias, but then the cached
//               value describes byte-identical content and the hit is correct.
struct SlotKey {
  const ir::Model* a = nullptr;
  const ir::Model* b = nullptr;
  uint64_t a_gen = UINT64_MAX;
  uint64_t a_enrich = UINT64_MAX;
  uint32_t a_graph = UINT32_MAX;
  size_t slot = SIZE_MAX;
  std::string b_path;
  bool operator==(const SlotKey&) const = default;
};

SlotKey make_slot_key(const DiffLoader& dl, size_t i, const ModelSession& s) {
  SlotKey k;
  k.a = s.model();
  k.b = dl.model_of(i);
  k.a_gen = s.generation();
  k.a_enrich = s.enrich_generation();
  k.a_graph = dl.primary_graph_of(i);
  k.slot = i;
  k.b_path = dl.path_of(i);
  return k;
}

// #34: state for the ONE per-tensor weight delta that may be in flight. One diff
// panel, one active comparison, so one slot suffices; a second request supersedes
// the first through the monotonic token, exactly like PendingDecode. Reached
// through a function-local static rather than captured by pointer, so the worker
// completion never writes through a pointer into per-tab state that a tab close
// could have freed.
struct TensorDeltaSlot {
  bool in_flight = false;  // the NEWEST job is still running (drives the spinner)
  bool done = false;
  bool ok = false;       // the call itself succeeded; per-side ok lives in `delta`
  int32_t row = -1;      // index into the cached match list this result is for
  std::string name;      // tensor name, so the result names itself
  TensorStatDelta delta;
  std::string error;
  uint64_t token = 0;    // only the newest completion publishes
  // Jobs dispatched but not yet completed, INCLUDING superseded ones. The token
  // decides who may publish; this decides who may still be READING. They are not
  // the same question: comparing row X then row Y leaves X's worker holding both
  // mappings after Y has published, so a lock keyed on `in_flight` alone would
  // release while X is still mid-decode. Slot removal is gated on this instead.
  uint32_t outstanding = 0;
};

TensorDeltaSlot& tensor_delta_state() {
  static TensorDeltaSlot st;
  return st;
}

// #37: serialize the diff to a change report. `tsv` selects TSV vs markdown.
// Lists each changed/added/removed node (status, op, name) + the summary counts.
// Pure over the two models + the diff result; no payload reads. `a_gi` is the
// primary graph the diff was computed for; B is always comparison graph 0. With
// #36 the caller always passes the ACTIVE slot's model + diff — exporting a
// non-active slot would produce an artifact that does not match what is on
// screen, and the summary counts alone would not say which rung it came from.
std::string build_change_report(const ir::Model& a, uint32_t a_gi,
                                const ir::Model& b, const ModelDiffResult& diff,
                                bool tsv) {
  std::string out;
  const char* sep = tsv ? "\t" : " | ";
  auto row = [&](const char* status, std::string_view op, std::string_view nm) {
    if (!tsv) out += "| ";
    out += status;
    out += sep;
    out += op.empty() ? "?" : std::string(op);
    out += sep;
    out += std::string(nm);
    if (!tsv) out += " |";
    out += "\n";
  };
  if (tsv) {
    out += "status\top\tname\n";
  } else {
    out += "# NetVis model diff\n\n";
    out += "**Summary:** " + std::to_string(diff.same) + " same, " +
           std::to_string(diff.added) + " added, " +
           std::to_string(diff.removed) + " removed, " +
           std::to_string(diff.changed) + " changed\n\n";
    out += "| status | op | name |\n|---|---|---|\n";
  }
  if (a_gi < a.graphs.size()) {
    const auto& nodes = a.graphs[a_gi].nodes;
    for (uint32_t i = 0; i < diff.a_status.size() && i < nodes.size(); ++i) {
      const char* st = diff.a_status[i] == DiffStatus::Removed  ? "removed"
                       : diff.a_status[i] == DiffStatus::Changed ? "changed"
                                                                 : nullptr;
      if (st) row(st, a.str(nodes[i].op_type), a.str(nodes[i].name));
    }
  }
  if (b.graphs.size() > 0) {
    const auto& nodes = b.graphs[0].nodes;
    for (uint32_t i = 0; i < diff.b_status.size() && i < nodes.size(); ++i) {
      if (diff.b_status[i] == DiffStatus::Added)
        row("added", b.str(nodes[i].op_type), b.str(nodes[i].name));
    }
  }
  return out;
}

// #36: one row per comparison slot — the quant ladder (fp32 -> int8 -> int4 read
// as rows against the SAME baseline). Every slot is diffed against the primary,
// never against each other, so the count columns are directly comparable. The
// radio picks which slot is ACTIVE; the active slot is the only one that tints
// the graph, so there is still exactly one set of node colors on screen.
//
// `lock_slots` disables everything that would FREE a comparison (see the
// lifetime note in draw_tensor_diff): while a per-tensor delta decode is in
// flight a worker holds the slot's MappedFile, and removing it out from under
// that worker would be a use-after-free, not merely a stale result.
void draw_comparison_ladder(App& app, bool lock_slots) {
  DiffLoader& dl = app.diff_loader();
  ModelSession& s = app.session();

  const ImGuiTableFlags flags = ImGuiTableFlags_Borders |
                                ImGuiTableFlags_RowBg |
                                ImGuiTableFlags_SizingFixedFit;
  if (!ImGui::BeginTable("cmp_slots", 8, flags)) return;
  ImGui::TableSetupColumn("act");
  ImGui::TableSetupColumn("file", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("state");
  ImGui::TableSetupColumn("same");
  ImGui::TableSetupColumn("added");
  ImGui::TableSetupColumn("removed");
  ImGui::TableSetupColumn("changed");
  ImGui::TableSetupColumn("##rm");
  ImGui::TableHeadersRow();

  // Deferred: remove_comparison() COMPACTS the slot vector, so mutating it
  // mid-loop would shift the indices this loop is still walking (and the
  // PushID/selection state bound to them). Apply it after EndTable.
  size_t remove_me = SIZE_MAX;

  const size_t n = dl.comparison_count();
  for (size_t i = 0; i < n; ++i) {
    ImGui::PushID(static_cast<int>(i));
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);
    const bool is_active = dl.active_comparison() == i;
    if (ImGui::RadioButton("##active", is_active) && !is_active)
      dl.set_active_comparison(i);

    ImGui::TableSetColumnIndex(1);
    const std::string& path = dl.path_of(i);
    const std::string_view bn = basename_of(path);
    if (bn.empty()) {
      ImGui::TextDisabled("(none)");
    } else {
      ImGui::Text("%.*s", static_cast<int>(bn.size()), bn.data());
      ImGui::SetItemTooltip("%s", path.c_str());
    }

    ImGui::TableSetColumnIndex(2);
    switch (dl.state_of(i)) {
      case DiffLoadState::Empty:
        ImGui::TextDisabled("empty");
        break;
      case DiffLoadState::Loading:
        ImGui::Text("load %c", spinner_glyph());
        break;
      case DiffLoadState::Failed:
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColRemoved),
                           "failed");
        ImGui::SetItemTooltip("%s", dl.error_of(i).c_str());
        break;
      case DiffLoadState::Ready:
        // The guard is PER SLOT (#36): this slot may have been added against a
        // different tab's session, in which case its counts are still true for
        // the pair it was computed from, but nothing A-side may be indexed with
        // the currently-active model.
        if (slot_primary_matches(dl, i, s)) {
          ImGui::TextUnformatted("ready");
        } else {
          ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColChanged),
                             "other model");
          ImGui::SetItemTooltip(
              "Diffed against a different model/tab - counts are for that "
              "pair. Re-add it here to diff the current model.");
        }
        break;
    }

    // Summary counts are model-agnostic (they describe the pair the slot was
    // diffed from), so they are safe to show even when the guard above fails.
    const ModelDiffResult* d = dl.diff_of(i);
    for (int c = 3; c <= 6; ++c) {
      ImGui::TableSetColumnIndex(c);
      if (d == nullptr || !d->valid) {
        ImGui::TextDisabled("-");
        continue;
      }
      switch (c) {
        case 3: ImGui::Text("%u", d->same); break;
        case 4:
          ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColAdded), "%u",
                             d->added);
          break;
        case 5:
          ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColRemoved), "%u",
                             d->removed);
          break;
        default:
          ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColChanged), "%u",
                             d->changed);
          break;
      }
    }

    ImGui::TableSetColumnIndex(7);
    ImGui::BeginDisabled(lock_slots);
    if (ImGui::SmallButton("remove")) remove_me = i;
    ImGui::EndDisabled();

    ImGui::PopID();
  }
  ImGui::EndTable();

  if (lock_slots)
    ImGui::TextDisabled(
        "Removal is held while a weight comparison is decoding.");

  if (remove_me != SIZE_MAX) dl.remove_comparison(remove_me);
}

// #31 two-model cost diff: dFLOPs / dparams / dweight-bytes (B - A) for the
// ACTIVE slot. Model-wide deltas come from each model's whole-graph CostReport
// (the comparison is always diffed against its graph 0). Cheap headless compute,
// but not free, so it is cached across frames — see SlotKey for why the key has
// to include the slot and the enrich epoch.
void draw_cost_delta(const DiffLoader& dl, const ModelSession& s, size_t slot,
                     const ir::Model& a_model, uint32_t a_gi,
                     const ir::Model& b_model) {
  ImGui::SeparatorText("Cost delta (B - A)");

  static SlotKey ck;
  static bool ck_valid = false;
  static uint64_t a_flops = 0, b_flops = 0, a_params = 0, b_params = 0;
  static uint64_t a_wbytes = 0, b_wbytes = 0;

  const SlotKey key = make_slot_key(dl, slot, s);
  if (!ck_valid || ck != key) {
    ck = key;
    ck_valid = true;
    const CostReport ra = compute_cost(a_model, a_gi);
    const CostReport rb = compute_cost(b_model, 0);
    a_flops = ra.total_flops; b_flops = rb.total_flops;
    a_params = ra.total_params; b_params = rb.total_params;
    a_wbytes = ra.total_weight_bytes; b_wbytes = rb.total_weight_bytes;
  }

  signed_row("d FLOPs", a_flops, b_flops, false);
  signed_row("d params", a_params, b_params, false);
  signed_row("d weights", a_wbytes, b_wbytes, true);
  // Percent size change (weights) — the headline "how much smaller after quant".
  if (a_wbytes > 0) {
    const double pct =
        100.0 * (static_cast<double>(b_wbytes) - static_cast<double>(a_wbytes)) /
        static_cast<double>(a_wbytes);
    ImGui::TextDisabled("weights %+.1f%% vs primary", pct);
  }
  ImGui::TextDisabled("Static estimate - analytical FLOPs, not measured.");
}

// #34 PHASE 2 render: the ONE decoded pair. Every failure mode gets its own
// honest line — a half-decoded pair reports which half failed, and a pair whose
// stats are metadata-only reports that instead of subtracting placeholder zeros.
void draw_tensor_delta_result(const TensorDeltaSlot& st) {
  if (st.row < 0) {
    ImGui::TextDisabled(
        "Press \"compare\" on a row to decode that ONE pair (both models' "
        "bytes for that tensor only).");
    return;
  }
  ImGui::Text("%s", st.name.c_str());
  if (st.in_flight) {
    ImGui::Text("decoding both sides %c", spinner_glyph());
    return;
  }
  if (!st.done) return;
  if (!st.ok) {
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColRemoved),
                       "compare failed:");
    ImGui::TextWrapped("%s", st.error.c_str());
    return;
  }

  const TensorStatDelta& d = st.delta;
  // Each side can fail independently; report the half that failed rather than
  // silently dropping the row.
  if (!d.a_ok)
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColRemoved),
                       "primary: %s", d.a_error.c_str());
  if (!d.b_ok)
    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColRemoved),
                       "comparison: %s", d.b_error.c_str());
  if (!d.both_ok()) {
    ImGui::TextDisabled("No delta: one side could not be decoded.");
    return;
  }
  // Honesty (spec §7.5): an unsupported quantized block layout yields
  // metadata-only stats, so a "delta" against them would be a fabricated number.
  if (d.a.quantized_unsupported || d.b.quantized_unsupported) {
    const char* which =
        d.a.quantized_unsupported
            ? (d.b.quantized_unsupported ? "Both sides use" : "The primary uses")
            : "The comparison uses";
    ImGui::TextDisabled(
        "%s a quantized layout NetVis cannot decode - no value delta.", which);
    return;
  }

  float_row("d min", d.a.min, d.b.min);
  float_row("d max", d.a.max, d.b.max);
  float_row("d mean", d.a.mean, d.b.mean);
  float_row("d std", d.a.std, d.b.std);
  // Counts are uint64: rendered through signed_row so a shrink reads as a
  // negative delta instead of wrapping to ~1.8e19.
  signed_row("d zeros", d.a.zero_count, d.b.zero_count, false);
  signed_row("d NaN/Inf", d.a.nan_inf_count, d.b.nan_inf_count, false);
  signed_row("d elements", d.a.count, d.b.count, false);
}

// #34: the per-tensor weight-stat diff, in the two strictly separated phases
// engine/TensorDiff.h defines.
//
// PHASE 1 (free): match_tensors reads names, shapes and dtypes only — no payload
// — and fills the table. "Free" is per CALL, not per frame though: a 7B
// checkpoint has thousands of tensors, so the match list is cached under a
// SlotKey and rebuilt only when that key changes (or on the first frame the
// section is open).
//
// PHASE 2 (on demand): compute_tensor_stat_delta decodes ONE pair on a worker.
// There is deliberately NO "compare all" action — decoding every matched weight
// reads the entire payload of BOTH models, which is exactly what the zero-payload
// thesis exists to prevent (spec §2.1). One pair, on an explicit click.
void draw_tensor_diff(App& app, size_t slot, const ir::Model& a_model,
                      const ir::Model& b_model) {
  ViewState& vs = app.view();
  DiffLoader& dl = app.diff_loader();
  ModelSession& s = app.session();
  TensorDeltaSlot& st = tensor_delta_state();

  // CollapsingHeader owns the open/closed state; mirror it into ViewState so the
  // rebuild below can tell that the section just flipped on, and so the flag is
  // available to the rest of the view as the frozen App.h contract promises.
  const bool open = ImGui::CollapsingHeader("Tensor weight diff");
  vs.tensor_diff_open = open;
  if (!open) return;

  static SlotKey mk;
  static bool mk_valid = false;
  static std::vector<TensorMatch> matches;
  // A-side names, resolved ONCE per rebuild. These are string_views into A's
  // StringArena, whose lifetime is exactly the `a` pointer in the key — so the
  // same condition that invalidates `matches` invalidates them. Caching them
  // keeps the per-frame filter scan free of any per-entry heap allocation.
  static std::vector<std::string_view> names;
  static std::vector<int32_t> rows;  // filtered indices into `matches`
  static std::string rows_filter;
  static bool rows_valid = false;

  const SlotKey key = make_slot_key(dl, slot, s);
  if (!mk_valid || mk != key) {
    mk = key;
    mk_valid = true;
    matches = match_tensors(a_model, b_model);
    names.clear();
    names.reserve(matches.size());
    for (const TensorMatch& m : matches) {
      const ir::TensorRef* t = resolve_tensor(a_model, m.a);
      names.push_back(t != nullptr ? a_model.str(t->name) : std::string_view());
    }
    rows_valid = false;
    vs.tensor_diff_selected = -1;
    // Any in-flight delta describes the PREVIOUS pair. Bump the token so its
    // completion is dropped rather than published under this one's name.
    const uint64_t next = st.token + 1;
    const uint32_t was_outstanding = st.outstanding;
    st = TensorDeltaSlot{};
    st.token = next;
    // `outstanding` is deliberately PRESERVED across this reset: the RESULT is
    // superseded, but those workers are still reading the previous pair's
    // mappings, so the ladder must keep slot removal locked until they land.
    // Clearing it here would re-enable "remove" on a slot a live worker holds —
    // a use-after-free, not merely a stale number. `in_flight` is display-only
    // and is cleared with the rest: there is no longer a row to spin over.
    st.outstanding = was_outstanding;
  }

  // Name filter (char buffer synced to vs.tensor_diff_filter; imgui_stdlib is
  // not built), same pattern as the tensor table's filter.
  static char filter_buf[256];
  std::snprintf(filter_buf, sizeof(filter_buf), "%s",
                vs.tensor_diff_filter.c_str());
  ImGui::SetNextItemWidth(240.0f);
  if (ImGui::InputTextWithHint("##tdiff_filter", "filter tensor name...",
                               filter_buf, sizeof(filter_buf))) {
    vs.tensor_diff_filter = filter_buf;
  }
  ImGui::SameLine();
  ImGui::TextDisabled("%d matched", static_cast<int>(matches.size()));

  if (!rows_valid || rows_filter != vs.tensor_diff_filter) {
    rows_filter = vs.tensor_diff_filter;
    rows_valid = true;
    rows.clear();
    for (size_t i = 0; i < matches.size(); ++i)
      if (icontains(names[i], rows_filter))
        rows.push_back(static_cast<int32_t>(i));
  }

  // Kick the ONE-pair decode for `mi`. Payload-reading, so it runs on a worker
  // and never inline: a multi-GB weight would otherwise stall the frame.
  auto start_compare = [&](int32_t mi) {
    if (mi < 0 || static_cast<size_t>(mi) >= matches.size()) return;
    const TensorMatch& m = matches[static_cast<size_t>(mi)];

    // Supersede whatever was in flight. `in_flight`/`outstanding` are NOT
    // cleared here: an older worker may still be reading, and the early return
    // below must not drop a lock it did not take.
    const uint64_t token = ++st.token;
    st.done = false;
    st.ok = false;
    st.error.clear();
    st.delta = TensorStatDelta{};
    st.row = mi;
    st.name = std::string(names[static_cast<size_t>(mi)]);

    const ir::TensorRef* ta = resolve_tensor(a_model, m.a);
    const ir::TensorRef* tb = resolve_tensor(b_model, m.b);
    if (ta == nullptr || tb == nullptr) {
      // A locator that no longer resolves means the cached match list outlived
      // the model it describes. Say so rather than dispatching a job that would
      // read nothing (honesty rule: unknown is reported, never approximated).
      st.done = true;
      st.error = "tensor no longer resolvable in one of the models";
      return;
    }

    // LIFETIME. The two sides have DIFFERENT owners and need different rules.
    //
    // A's mapping belongs to the tab's ModelSession and may be borrowed by
    // pointer: this job runs on the tab's own JobSystem, and ~Tab joins that pool
    // before the session dies (the same contract App::inspect_tensor relies on).
    //
    // B's mapping belongs to a DiffLoader slot and must NOT be borrowed. Nothing
    // sequences the tab's pool against the loader: App::close_tab clears the
    // loader BEFORE erasing the tab, so a job still reading a borrowed
    // DiffLoader::file_of(slot) would read a freed mapping — a use-after-free the
    // panel cannot defend against, because the user never touches this panel to
    // trigger it. So the job owns its B side outright: it pins model B with a
    // shared_ptr and opens its OWN mapping of the comparison path on the worker.
    // An mmap is ~1 ms, far cheaper than making a borrowed mapping safe, and it
    // is exactly what App::inspect_tensor_comparison does for the #50 B side.
    const MappedFile* fa = &s.file();
    const ir::Model* ma = &a_model;
    const std::shared_ptr<const ir::Model> mb = dl.model_ptr_of(slot);
    const std::string da = s.model_dir();
    const std::string db = dl.model_dir_of(slot);
    const std::string bpath = dl.path_of(slot);
    const ir::TensorRef ca = *ta;
    const ir::TensorRef cb = *tb;
    JobSystem* jobs = &app.jobs();
    if (!mb) {
      st.done = true;
      st.error = "the comparison model is no longer loaded";
      return;
    }

    st.in_flight = true;
    ++st.outstanding;
    jobs->submit([jobs, token, ca, cb, fa, da, db, bpath, ma, mb]() {
      // Open B's mapping here, on the worker, so this job depends on nothing
      // owned by DiffLoader beyond the shared_ptr it already holds.
      auto mapped = MappedFile::open(bpath);
      bool ok = false;
      TensorStatDelta d{};
      std::string err;
      if (!mapped) {
        err = mapped.error().message;
      } else {
        const MappedFile fb = std::move(*mapped);
        Result<TensorStatDelta> r =
            compute_tensor_stat_delta(ca, *fa, da, ma, cb, fb, db, mb.get());
        ok = r.ok();
        if (ok) {
          d = *r;
        } else {
          err = r.error().message;
        }
      }
      jobs->post_to_main([token, ok, d, err]() {
        // Reach the state through the accessor, never a captured pointer: this
        // completion can land after the panel has moved to another slot.
        TensorDeltaSlot& cur = tensor_delta_state();
        // Released FIRST and unconditionally: this worker has stopped reading
        // whether or not it is still the newest, and the removal lock is about
        // reading, not publishing.
        if (cur.outstanding > 0) --cur.outstanding;
        if (cur.token != token) return;  // superseded — drop the result.
        cur.delta = d;
        cur.ok = ok;
        cur.error = err;
        cur.done = true;
        cur.in_flight = false;
      });
    });
  };

  // --- Phase-1 table: free columns only (name/shape/dtype/params/bytes) -------
  const ImGuiTableFlags tflags = ImGuiTableFlags_Borders |
                                 ImGuiTableFlags_RowBg |
                                 ImGuiTableFlags_ScrollY;
  if (ImGui::BeginTable("tensor_diff", 6, tflags, ImVec2(0.0f, 220.0f))) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("name", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("shape");
    ImGui::TableSetupColumn("dtype");
    ImGui::TableSetupColumn("params");
    ImGui::TableSetupColumn("bytes");
    ImGui::TableSetupColumn("##cmp");
    ImGui::TableHeadersRow();

    const ImVec4 warn = ImGui::ColorConvertU32ToFloat4(kColChanged);

    // Virtualized: a 7B checkpoint matches thousands of tensors and only the
    // visible slice may be formatted (same clipper pattern as the per-channel
    // table in WeightInspector.cpp).
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(rows.size()));
    while (clipper.Step()) {
      for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
        const int32_t mi = rows[static_cast<size_t>(r)];
        const TensorMatch& m = matches[static_cast<size_t>(mi)];
        const ir::TensorRef* ta = resolve_tensor(a_model, m.a);
        const ir::TensorRef* tb = resolve_tensor(b_model, m.b);
        const std::string_view nm = names[static_cast<size_t>(mi)];

        ImGui::PushID(mi);
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        // Stack buffer: a std::string per visible row would allocate every
        // frame the section is open, for no benefit.
        char label[256];
        std::snprintf(label, sizeof(label), "%.*s",
                      static_cast<int>(nm.size()), nm.data());
        if (ImGui::Selectable(label, vs.tensor_diff_selected == mi,
                              ImGuiSelectableFlags_SpanAllColumns))
          vs.tensor_diff_selected = mi;

        ImGui::TableSetColumnIndex(1);
        if (ta != nullptr) {
          const std::string sa = shape_string(ta->shape);
          if (m.shape_equal) {
            ImGui::TextUnformatted(sa.c_str());
          } else {
            const std::string sb =
                tb != nullptr ? shape_string(tb->shape) : std::string("?");
            ImGui::TextColored(warn, "%s -> %s", sa.c_str(), sb.c_str());
          }
        }

        ImGui::TableSetColumnIndex(2);
        if (ta != nullptr) {
          // Prefer the parser's exact label when ir::DType could not express the
          // element type (frozen 16-enumerator DType), as the tensor table does.
          auto dtype_text = [](const ir::Model& mm, const ir::TensorRef& t) {
            if (t.dtype == ir::DType::Unknown && t.dtype_label.valid())
              return std::string(mm.str(t.dtype_label));
            return std::string(ir::dtype_name(t.dtype));
          };
          const std::string dta = dtype_text(a_model, *ta);
          if (m.dtype_equal) {
            ImGui::TextUnformatted(dta.c_str());
          } else {
            const std::string dtb =
                tb != nullptr ? dtype_text(b_model, *tb) : std::string("?");
            ImGui::TextColored(warn, "%s -> %s", dta.c_str(), dtb.c_str());
          }
        }

        ImGui::TableSetColumnIndex(3);
        if (m.a_elems == m.b_elems) {
          ImGui::TextUnformatted(grouped_count(m.a_elems).c_str());
        } else {
          ImGui::TextColored(warn, "%s -> %s", grouped_count(m.a_elems).c_str(),
                             grouped_count(m.b_elems).c_str());
        }

        ImGui::TableSetColumnIndex(4);
        if (m.a_bytes == m.b_bytes) {
          ImGui::TextUnformatted(human_bytes(m.a_bytes).c_str());
        } else {
          const ImU32 col = m.b_bytes < m.a_bytes ? kColRemoved : kColAdded;
          ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(col), "%s -> %s",
                             human_bytes(m.a_bytes).c_str(),
                             human_bytes(m.b_bytes).c_str());
        }

        ImGui::TableSetColumnIndex(5);
        if (ImGui::SmallButton("compare")) {
          vs.tensor_diff_selected = mi;
          start_compare(mi);
        }

        ImGui::PopID();
      }
    }
    ImGui::EndTable();
  }

  if (matches.empty()) {
    ImGui::TextDisabled(
        "No named tensor appears in both models (structure-only diff).");
  } else if (rows.empty()) {
    ImGui::TextDisabled("(no tensor matches \"%s\")",
                        vs.tensor_diff_filter.c_str());
  }

  ImGui::Separator();
  draw_tensor_delta_result(st);
}

}  // namespace

DiffTint diff_tint_for_display(App& app, int32_t display_id) {
  DiffTint out;
  DiffLoader& dl = app.diff_loader();
  if (!dl.active()) return out;

  ModelSession& s = app.session();
  // #62: only tint if the diff was computed against the ACTIVE tab's session. With
  // per-tab sessions, generation+graph both start at 0 per tab, so without this a
  // diff loaded for tab A would paint tab B's coincidentally-matching graph.
  // #36: `active()` / `diff()` are the ACTIVE slot, so exactly one comparison
  // tints the canvas no matter how many are loaded.
  if (dl.primary_session() != &s) return out;
  // Only tint if the diff was computed against the CURRENT primary model+graph.
  // Generation guards a primary RELOAD (current_graph resets to 0 on open, so a
  // graph-index match alone would paint a fresh model with stale diff status).
  if (dl.primary_generation() != s.generation()) return out;
  if (dl.primary_graph() != s.current_graph()) return out;

  const ModelDiffResult* diff = dl.diff();
  if (diff == nullptr || !diff->valid) return out;

  const auto& disp = s.collapse().display_nodes();
  if (display_id < 0 || static_cast<size_t>(display_id) >= disp.size())
    return out;
  const DisplayNode& dn = disp[static_cast<size_t>(display_id)];

  // Resolve the A-side (primary) node index this display node maps to. For a
  // collapsed group, use its first member as the representative.
  uint32_t a_node = UINT32_MAX;
  if (dn.is_group) {
    const auto& groups = s.collapse().groups();
    if (dn.group_index < groups.size() &&
        !groups[dn.group_index].member_nodes.empty())
      a_node = groups[dn.group_index].member_nodes.front();
  } else {
    a_node = dn.ir_node;
  }
  if (a_node == UINT32_MAX || a_node >= diff->a_status.size()) return out;

  switch (diff->a_status[a_node]) {
    case DiffStatus::Removed:
      out.active = true;
      out.color = kColRemoved;
      break;
    case DiffStatus::Changed:
      out.active = true;
      out.color = kColChanged;
      break;
    case DiffStatus::Added:  // A-side nodes are never "Added" (that's B-only).
    case DiffStatus::Same:
    default:
      out.active = false;
      break;
  }
  return out;
}

void draw_diff_panel(App& app) {
  ViewState& vs = app.view();
  if (!vs.diff_panel_open) return;

  if (!ImGui::Begin("Model Diff", &vs.diff_panel_open)) {
    ImGui::End();
    return;
  }

  DiffLoader& dl = app.diff_loader();
  ModelSession& s = app.session();

  // When every comparison is gone (Clear all, or App clearing the diff because
  // the diffed tab closed) there is no slot left to protect, so leftover #34
  // bookkeeping is dropped here. Without this the lock below could outlive a tab
  // close whose JobSystem was shut down with the completion still queued, and
  // would then disable removal for the rest of the run. Safe precisely because
  // no mapping this panel could free is still reachable at this point.
  if (dl.comparison_count() == 0) {
    TensorDeltaSlot& st = tensor_delta_state();
    if (st.outstanding > 0 || st.row >= 0) {
      const uint64_t next = st.token + 1;
      st = TensorDeltaSlot{};
      st.token = next;
    }
  }

  // Read the outstanding-job count BEFORE anything can act on it: the ladder must
  // not offer a control that frees a slot a #34 worker is currently reading. The
  // count is decremented only by a worker's completion (drained before frame()),
  // so a one-frame stale non-zero merely holds the button one extra frame — the
  // unsafe direction (zero while a worker still holds the mapping) cannot occur.
  const bool decoding = tensor_delta_state().outstanding > 0;

  // #36: ADD rather than replace. Each comparison is an independent slot diffed
  // against the same primary baseline.
  const bool at_cap = dl.comparison_count() >= kMaxComparisons;
  ImGui::BeginDisabled(at_cap);
  if (ImGui::Button("Add comparison...")) {
    const char* filters[] = {"*.onnx", "*.tflite", "*.safetensors",
                             "*.gguf", "*.pt",     "*.pth",
                             "*.bin"};
    char* picked =
        tinyfd_openFileDialog("Open comparison model", "", 7, filters,
                              "Model files", 0);
    if (picked != nullptr) dl.add_comparison(s, picked);
  }
  ImGui::EndDisabled();
  if (at_cap) {
    // A disabled widget does not hover, so the explanation is plain text rather
    // than a tooltip that would never appear.
    ImGui::SameLine();
    ImGui::TextDisabled("(%d loaded - the maximum; remove one first)",
                        static_cast<int>(kMaxComparisons));
  }
  if (dl.comparison_count() > 0) {
    ImGui::SameLine();
    ImGui::BeginDisabled(decoding);
    if (ImGui::Button("Clear all")) dl.clear();
    ImGui::EndDisabled();
  }

  // #35: node-matching strategy toggle. Changing it re-diffs EVERY loaded
  // comparison in place (DiffLoader::set_match; no reload). Loader-wide by
  // design: a ladder whose rungs used different matching rules would not be
  // comparable column-to-column.
  {
    int mode = dl.match() == DiffMatch::TopologyOnly ? 1 : 0;
    ImGui::TextDisabled("Match:");
    ImGui::SameLine();
    if (ImGui::RadioButton("name+topology", &mode, 0))
      dl.set_match(DiffMatch::NameThenTopology);
    ImGui::SameLine();
    if (ImGui::RadioButton("topology only", &mode, 1))
      dl.set_match(DiffMatch::TopologyOnly);
  }

  ImGui::Separator();

  if (dl.comparison_count() == 0) {
    ImGui::TextDisabled("No comparison loaded.");
    ImGui::End();
    return;
  }

  ImGui::SeparatorText("Comparisons");
  draw_comparison_ladder(app, decoding);

  // Everything below is the ACTIVE slot: it is the one tinting the graph, so
  // pairing the on-screen colors with another slot's numbers would misread the
  // canvas. Re-read the index AFTER the ladder — a removal there compacts.
  const size_t ai = dl.active_comparison();
  ImGui::SeparatorText("Active comparison");

  switch (dl.state_of(ai)) {
    case DiffLoadState::Empty:
      ImGui::TextDisabled("This slot is empty.");
      ImGui::End();
      return;
    case DiffLoadState::Loading:
      ImGui::Text("Loading comparison %c", spinner_glyph());
      ImGui::End();
      return;
    case DiffLoadState::Failed:
      ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1.0f), "Load failed:");
      ImGui::TextWrapped("%s", dl.error_of(ai).c_str());
      ImGui::End();
      return;
    case DiffLoadState::Ready:
      break;
  }

  ImGui::Text("Comparison: %s", dl.path_of(ai).c_str());
  const ModelDiffResult* diff = dl.diff_of(ai);
  if (diff == nullptr) {
    ImGui::TextDisabled("No diff result.");
    ImGui::End();
    return;
  }

  // The diff was computed against a SPECIFIC primary session/generation/graph. The
  // summary counts are model-agnostic and always safe to show, but the A-side node
  // lists, cost delta, tensor diff and #37 export index the ACTIVE tab's model —
  // which is only the diff's A-side when the active session IS the one this SLOT
  // was pinned to (same identity + generation + graph). On a second/reloaded tab
  // those differ, so pairing the active model with this diff's per-node status
  // would render + (in #37) EXPORT a wrong artifact. #36: evaluated PER SLOT — the
  // pre-#36 single inline computation assumed one comparison and is now wrong.
  const bool primary_matches = slot_primary_matches(dl, ai, s);

  ImGui::SeparatorText("Summary");
  ImGui::Text("same:    %u", diff->same);
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColAdded), "added:   %u",
                     diff->added);
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColRemoved), "removed: %u",
                     diff->removed);
  ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(kColChanged), "changed: %u",
                     diff->changed);

  // A-side (node lists + cost delta + tensor diff + #37 export) is only valid
  // when the active session is the one this slot was pinned to. Otherwise stop
  // after the summary.
  if (!primary_matches) {
    ImGui::TextDisabled(
        "Diff was computed for a different model/tab - switch to that tab or "
        "re-add the comparison here to diff this model.");
    ImGui::End();
    return;
  }

  const ir::Model* a_model = s.model();
  const ir::Model* b_model = dl.model_of(ai);
  const uint32_t a_gi = dl.primary_graph_of(ai);

  if (a_model != nullptr && b_model != nullptr &&
      a_gi < a_model->graphs.size())
    draw_cost_delta(dl, s, ai, *a_model, a_gi, *b_model);

  // #34: per-tensor weight-stat diff for this slot's pair.
  if (a_model != nullptr && b_model != nullptr)
    draw_tensor_diff(app, ai, *a_model, *b_model);

  // Helper to render a scrollable list of A-side nodes matching a status.
  auto list_a = [&](const char* title, DiffStatus want, ImU32 col) {
    if (a_model == nullptr || a_gi >= a_model->graphs.size()) return;
    const auto& nodes = a_model->graphs[a_gi].nodes;
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    bool open = ImGui::CollapsingHeader(title);
    ImGui::PopStyleColor();
    if (!open) return;
    if (ImGui::BeginChild(title, ImVec2(0, 140), ImGuiChildFlags_Borders)) {
      for (uint32_t i = 0; i < diff->a_status.size() && i < nodes.size(); ++i) {
        if (diff->a_status[i] != want) continue;
        const ir::Node& node = nodes[i];
        std::string_view nm = a_model->str(node.name);
        std::string_view op = a_model->str(node.op_type);
        std::string label(op.empty() ? "?" : std::string(op));
        if (!nm.empty()) {
          label += "  ";
          label += std::string(nm);
        }
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Selectable(label.c_str())) {
          // Jump the canvas to this (still-present) A-side node.
          int32_t disp =
              panel_detail::display_index_for_node(s.collapse(), i);
          if (disp >= 0) vs.selected_display = disp;
        }
        ImGui::PopID();
      }
    }
    ImGui::EndChild();
  };

  // B-side list for "added" nodes (present only in comparison).
  auto list_b_added = [&]() {
    if (b_model == nullptr) return;
    // DiffLoader always diffs against comparison graph 0 (diff_models(...,*B,0)),
    // so b_status is parallel to graph 0's node list — index that, NOT a_gi.
    constexpr uint32_t b_gi = 0u;
    if (b_gi >= b_model->graphs.size()) return;
    const auto& nodes = b_model->graphs[b_gi].nodes;
    ImGui::PushStyleColor(ImGuiCol_Text, kColAdded);
    bool open = ImGui::CollapsingHeader("Added (comparison only)");
    ImGui::PopStyleColor();
    if (!open) return;
    if (ImGui::BeginChild("added_b", ImVec2(0, 140), ImGuiChildFlags_Borders)) {
      for (uint32_t i = 0; i < diff->b_status.size() && i < nodes.size(); ++i) {
        if (diff->b_status[i] != DiffStatus::Added) continue;
        const ir::Node& node = nodes[i];
        std::string_view nm = b_model->str(node.name);
        std::string_view op = b_model->str(node.op_type);
        std::string label(op.empty() ? "?" : std::string(op));
        if (!nm.empty()) {
          label += "  ";
          label += std::string(nm);
        }
        ImGui::TextUnformatted(label.c_str());
      }
    }
    ImGui::EndChild();
  };

  // #37: export the change report (markdown / TSV) for the ACTIVE slot.
  if (a_model != nullptr && b_model != nullptr) {
    ImGui::SeparatorText("Export");
    auto do_export = [&](bool tsv) {
      const char* pat_md[] = {"*.md"};
      const char* pat_tsv[] = {"*.tsv"};
      const char* out = tinyfd_saveFileDialog(
          "Export change report", tsv ? "diff.tsv" : "diff.md", 1,
          tsv ? pat_tsv : pat_md, tsv ? "TSV" : "Markdown");
      if (out == nullptr) return;
      std::string report =
          build_change_report(*a_model, a_gi, *b_model, *diff, tsv);
      std::ofstream f(out);
      if (f) {
        f << report;
        app.add_toast(std::string("Exported ") + out, false);
      } else {
        app.add_toast("Could not write report", true);
      }
    };
    if (ImGui::Button("Export .md")) do_export(false);
    ImGui::SameLine();
    if (ImGui::Button("Export .tsv")) do_export(true);
  }

  ImGui::SeparatorText("Details");
  list_a("Removed (primary only)", DiffStatus::Removed, kColRemoved);
  list_a("Changed", DiffStatus::Changed, kColChanged);
  list_b_added();

  ImGui::End();
}

}  // namespace netvis
