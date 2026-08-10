// engine/DiffLoader.cpp — async load + diff of COMPARISON models (v0.2.0; N-way
// since v0.9.1b / #36).
//
// DESIGN (simplest provably race-free — see header + task notes):
//   * A worker maps its OWN MappedFile (job-local) and runs parse_model into its
//     OWN ir::Model. It reads NOTHING from model A, so nothing on the worker can
//     race the primary session's in-flight shape inference.
//   * The parsed model B + its mapping are published to the MAIN thread via
//     shared_ptr. The completion (token-checked, newest-load-wins) then:
//       - moves the mapping into the slot and adopts model B, and
//       - runs diff_models(*primary.model(), primary_graph, *B, 0) INLINE.
//     This diff is bounded main-thread work (~2*node fingerprints + matching)
//     and runs only when the primary model/graph still matches the snapshot, so
//     model A's immutable Node/edge/producer data is read safely and its
//     ValueInfo.shape/dtype are NEVER touched (diff_models guarantees this).
//
// SLOT LIFETIME (#36) — the one genuinely hard part of going N-way. A slot can be
// destroyed at ANY main-thread moment (remove_comparison compacts the vector,
// clear/load_comparison drop every slot) while one of its loads is still running.
// Two consequences drive the whole implementation:
//
//   1. NOTHING owned by a slot may be touched off the main thread. In particular
//      the worker does NOT write the slot's ProgressSink — it parses into a
//      JOB-LOCAL sink instead, because a remove_comparison() mid-parse would
//      otherwise leave parse_model writing into freed memory. The slot's own sink
//      carries only coarse main-thread state (loading/ready/failed).
//   2. A completion may NOT identify its slot by index (removal compacts, so
//      indices shift under it) and may NOT identify it by a captured Comparison*
//      (that pointer dangles once the slot is erased — and a later
//      make_unique<Comparison> can recycle the very same address, so even
//      comparing it for identity could match the WRONG slot).
//
//   So a load is identified by its TOKEN and nothing else. Tokens come from a
//   monotonic process-wide counter (next_token below), so a token is unique for
//   the life of the process and is never re-issued once retired. The completion
//   searches comps_ for the live slot that still carries its token; finding none
//   means the load was superseded (reload), the slot was removed, or everything
//   was cleared — in every case it drops itself without dereferencing anything.
//   Erasing a slot is therefore all that "cancelling" its in-flight load takes.
//
// LIFETIME/RACE (unchanged, now per slot): the comparison model is held by
// shared_ptr and the completion carries that shared_ptr, so model B stays alive
// for the whole job even if a newer load replaces the published one. The loader
// itself outlives its jobs (App shuts diff_jobs_ down in ~App before destroying
// diff_loader_), which is what makes capturing `this` on a worker legal.
#include "engine/DiffLoader.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h references but
// does not itself include; pull it in first so ModelSession.h compiles here.
#include "engine/LayoutEngine.h"

#include "engine/ModelDiff.h"
#include "engine/ModelPath.h"
#include "engine/ModelSession.h"
#include "parsers/Parser.h"

namespace netvis {

namespace {

// Lowercased extension without the dot; "" if none. Format detection tiebreaker.
std::string ext_of(const std::string& path) {
  std::filesystem::path p(path);
  std::string e = p.extension().string();
  if (!e.empty() && e[0] == '.') e.erase(0, 1);
  for (char& c : e)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}

// Monotonic load token, unique for the whole process (see file header: a
// completion finds its slot by token, so a token must never be re-issued — not
// even after every slot of a loader has been dropped). Deliberately a plain
// counter, not an atomic: every caller is a main-thread-only public method of
// DiffLoader. Sharing one counter across several DiffLoader instances is
// harmless and in fact desirable — it can only make tokens MORE unique.
uint64_t next_token() {
  static uint64_t counter = 0;
  return ++counter;
}

// Neutral referents for the out-of-range cases of the accessors that return a
// reference. They must be function-local statics because there is no slot to
// borrow from, and `const` so a caller that (wrongly) writes through a returned
// non-const alias cannot corrupt the shared instance — every accessor here hands
// out const references, so const is the accurate type for the fallback too.
const std::string& empty_string() {
  static const std::string kEmpty;
  return kEmpty;
}

// Value-initialized (`{}`) rather than default-initialized: MappedFile's default
// constructor is defaulted, not user-provided, so `static const MappedFile x;`
// would not compile. A default-constructed mapping has valid() == false and its
// destructor is a no-op, which is exactly the "no such comparison" answer.
const MappedFile& invalid_file() {
  static const MappedFile kNone{};
  return kNone;
}

}  // namespace

DiffLoader::DiffLoader(JobSystem& jobs) : jobs_(jobs) {}

// Dropping every slot retires every live token, so a completion that has already
// been queued finds no slot to write through and becomes a no-op. (App shuts the
// diff JobSystem down before destroying the loader — see file header — so this
// only has to cover completions already sitting in the queue.)
DiffLoader::~DiffLoader() { comps_.clear(); }

void DiffLoader::start_load(Comparison& c, const ModelSession& primary,
                            const std::string& path) {
  // MAIN THREAD. Newest load into THIS slot wins: take a fresh token (which
  // retires the slot's previous one, dropping any in-flight load into it) and
  // snapshot the primary graph index we are diffing against.
  const uint64_t token = next_token();
  c.token = token;
  c.state = DiffLoadState::Loading;
  c.error.clear();
  c.path = path;
  c.diff.reset();
  // The previously adopted mapping/model (if this is a reload) are deliberately
  // NOT dropped here: they stay published until the new ones are adopted, so a
  // #34/#50 reader that is mid-flight against this slot keeps seeing mapped bytes
  // rather than an unmapped range. state != Ready and diff == nullptr are what
  // gate the UI in the meantime.
  c.primary_graph = primary.current_graph();
  c.primary_generation = UINT64_MAX;
  c.primary_session = nullptr;
  c.progress.set(0.0f, "loading");

  const ModelSession* primary_ptr = &primary;

  jobs_.submit([this, token, path, primary_ptr] {
    // WORKER. Own the mapping + parse into a job-local model. Reads nothing from
    // model A and nothing from the slot (which may be freed under us — see file
    // header), so no race with the primary session and no use-after-free.
    // Resolve a .mlpackage bundle to its inner model file (ModelPath.h); a plain
    // file passes through.
    ResolvedModelPath resolved = resolve_model_path(path);

    // Derived exactly as ModelSession::model_dir() does — from the MAPPED path,
    // not the display path, so for a bundle it points at the inner
    // Data/com.apple.CoreML/ that holds weights/weight.bin. The diff itself is
    // topology-only and needs none of this; #34/#50 read model B's payload
    // through file_of()/model_dir_of() and do (spec §2.1, external data).
    std::string model_dir =
        std::filesystem::path(resolved.map_path).parent_path().string();

    // Job-local sink: the slot's own ProgressSink must not be written off the
    // main thread because the slot can be destroyed while this parse runs.
    ProgressSink progress;

    std::string err;
    std::shared_ptr<MappedFile> mf;
    std::shared_ptr<ir::Model> model;

    auto mapped = MappedFile::open(resolved.map_path);
    if (!mapped) {
      err = mapped.error().message;
    } else {
      mf = std::make_shared<MappedFile>(mapped.take());
      Result<ir::Model> parsed = parse_model(*mf, ext_of(resolved.map_path),
                                             progress);
      if (!parsed) {
        err = parsed.error().message;
        mf.reset();  // job-local: unmaps here, on the worker
      } else {
        model = std::make_shared<ir::Model>(parsed.take());
      }
    }

    jobs_.post_to_main([this, token, err, model_dir, mf, model,
                        primary_ptr]() mutable {
      // MAIN THREAD. Find the slot that still carries this token — the ONLY safe
      // way to identify it (see file header). No match means superseded by a
      // reload, removed, or cleared; drop silently without touching anything,
      // which is also what keeps `primary_ptr` from being dereferenced after the
      // owning tab closed (App clears the loader on tab close).
      auto it = std::find_if(comps_.begin(), comps_.end(),
                             [token](const std::unique_ptr<Comparison>& s) {
                               return s->token == token;
                             });
      if (it == comps_.end()) return;
      Comparison& c = **it;

      if (!err.empty()) {
        c.state = DiffLoadState::Failed;
        c.error = err;
        c.progress.set(1.0f, "failed");
        return;
      }

      // Adopt model B + its mapping (both were job-local until now).
      c.file = std::move(*mf);
      c.model_dir = std::move(model_dir);
      c.model = model;  // shared_ptr<ir::Model> -> shared_ptr<const ir::Model>
      c.progress.set(1.0f, "ready");

      // Diff INLINE against the primary snapshot, but only if the primary model
      // and graph still match what we snapshotted at call time (guards against a
      // primary re-open or push_graph/pop_graph having moved on). diff_models
      // reads only immutable structure/topology of model A — never shapes.
      const ir::Model* pa = primary_ptr->model();
      if (pa && primary_ptr->current_graph() == c.primary_graph) {
        ModelDiffResult d = diff_models(*pa, c.primary_graph, *c.model, 0,
                                        match_);
        const bool ok = d.valid;
        c.diff = std::make_unique<ModelDiffResult>(std::move(d));
        if (ok) {
          // Pin the primary generation the diff was computed against, so the
          // view stops tinting the moment the primary model is reloaded.
          c.primary_generation = primary_ptr->generation();
          // #62: pin the session IDENTITY too — with per-tab sessions a matching
          // generation+graph no longer means "same model". The view requires this
          // equals the session it is tinting. Pointer stored for comparison ONLY.
          c.primary_session = primary_ptr;
          c.state = DiffLoadState::Ready;
        } else {
          c.state = DiffLoadState::Failed;
          c.error = "diff graph index out of range";
          c.diff.reset();
        }
      } else {
        // Primary moved on; model B is loaded but the snapshot diff is stale.
        // Leave the diff null so active_of() is false and the view does not tint.
        c.state = DiffLoadState::Ready;
      }
    });
  });
}

void DiffLoader::rediff(Comparison& c) {
  // MAIN THREAD. Re-run this slot's diff under the current match strategy
  // (bounded structural work, same as the post-load diff — reads only immutable
  // topology of both models, never shapes). No-op unless the slot is loaded and
  // the primary STILL matches the snapshot the existing diff was computed against
  // (identity + graph + generation — the same guards the view uses before
  // tinting); if it moved on we leave the existing diff alone until a reload.
  if (c.state != DiffLoadState::Ready || !c.model || c.primary_session == nullptr)
    return;
  const ModelSession* ps = c.primary_session;
  const ir::Model* pa = ps->model();
  if (pa == nullptr || ps->current_graph() != c.primary_graph ||
      ps->generation() != c.primary_generation)
    return;
  ModelDiffResult d = diff_models(*pa, c.primary_graph, *c.model, 0, match_);
  if (d.valid) c.diff = std::make_unique<ModelDiffResult>(std::move(d));
}

bool DiffLoader::add_comparison(const ModelSession& primary,
                                const std::string& path) {
  // MAIN THREAD. Bounded at kMaxComparisons: each slot costs a full mapping + a
  // parsed model, and the panel has one column per slot.
  if (comps_.size() >= kMaxComparisons) return false;
  comps_.push_back(std::make_unique<Comparison>());
  if (comps_.size() == 1) active_ = 0;  // first comparison drives the tint
  start_load(*comps_.back(), primary, path);
  return true;
}

void DiffLoader::reload_comparison(size_t i, const ModelSession& primary,
                                   const std::string& path) {
  // Bounds-checked like every other index-taking call: the panel holds a
  // selection index across frames and a removal can invalidate it.
  Comparison* c = slot(i);
  if (c == nullptr) return;
  start_load(*c, primary, path);
}

void DiffLoader::remove_comparison(size_t i) {
  if (i >= comps_.size()) return;
  // Erasing retires the slot's token, which is exactly what invalidates any
  // in-flight load into it (file header): the completion will find no match.
  comps_.erase(comps_.begin() + static_cast<std::ptrdiff_t>(i));
  // Follow the compaction so the SAME comparison stays active when a slot below
  // it is dropped (indices above `i` shift down by one), then clamp for the case
  // where the removed slot was the last one.
  if (active_ > i) --active_;
  if (active_ >= comps_.size()) active_ = comps_.empty() ? 0 : comps_.size() - 1;
}

void DiffLoader::set_active_comparison(size_t i) {
  if (i >= comps_.size()) return;  // out of range: no-op, never UB
  active_ = i;
}

DiffLoadState DiffLoader::state_of(size_t i) const {
  const Comparison* c = slot(i);
  return c ? c->state : DiffLoadState::Empty;
}

const std::string& DiffLoader::error_of(size_t i) const {
  const Comparison* c = slot(i);
  return c ? c->error : empty_string();
}

const std::string& DiffLoader::path_of(size_t i) const {
  const Comparison* c = slot(i);
  return c ? c->path : empty_string();
}

const ir::Model* DiffLoader::model_of(size_t i) const {
  const Comparison* c = slot(i);
  return c ? c->model.get() : nullptr;
}

const ModelDiffResult* DiffLoader::diff_of(size_t i) const {
  const Comparison* c = slot(i);
  return c ? c->diff.get() : nullptr;
}

uint32_t DiffLoader::primary_graph_of(size_t i) const {
  const Comparison* c = slot(i);
  return c ? c->primary_graph : 0;
}

uint64_t DiffLoader::primary_generation_of(size_t i) const {
  const Comparison* c = slot(i);
  // UINT64_MAX is the "never" sentinel: ModelSession generations start at 0 and
  // only increase, so this can never compare equal to a live generation and the
  // view's tint guard fails closed.
  return c ? c->primary_generation : UINT64_MAX;
}

const ModelSession* DiffLoader::primary_session_of(size_t i) const {
  const Comparison* c = slot(i);
  return c ? c->primary_session : nullptr;
}

bool DiffLoader::active_of(size_t i) const {
  const Comparison* c = slot(i);
  return c != nullptr && c->state == DiffLoadState::Ready && c->diff != nullptr;
}

const MappedFile& DiffLoader::file_of(size_t i) const {
  const Comparison* c = slot(i);
  return c ? c->file : invalid_file();
}

const std::string& DiffLoader::model_dir_of(size_t i) const {
  const Comparison* c = slot(i);
  return c ? c->model_dir : empty_string();
}

void DiffLoader::load_comparison(const ModelSession& primary,
                                 const std::string& path) {
  // Pre-#36 semantics: newest load wins, replacing whatever was there. clear()
  // retires every token, so any in-flight load is dropped; the add can then never
  // hit the kMaxComparisons cap, so its result carries no information.
  clear();
  add_comparison(primary, path);
}

void DiffLoader::update() {
  // MAIN THREAD: DiffLoader uses its OWN JobSystem (see header), so drain that
  // system's completion queue here — call once per frame after the primary
  // session's update(). This runs the load/diff completions posted by workers.
  jobs_.drain_completions();
}

void DiffLoader::clear() {
  // MAIN THREAD: exit diff mode. Dropping the slots retires their tokens, so any
  // in-flight completion becomes a no-op.
  comps_.clear();
  active_ = 0;
}

void DiffLoader::set_match(DiffMatch m) {
  // MAIN THREAD. The strategy is loader-wide (header: a ladder whose rungs used
  // different matching rules would be incomparable), so every loaded slot is
  // re-diffed inline. No reload: match affects only how the two topologies are
  // aligned, never what was parsed.
  if (m == match_) return;
  match_ = m;
  for (auto& c : comps_) rediff(*c);
}

}  // namespace netvis
