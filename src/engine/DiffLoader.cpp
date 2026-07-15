// engine/DiffLoader.cpp — async load + diff of a COMPARISON model (v0.2.0).
//
// DESIGN (simplest provably race-free — see header + task notes):
//   * A worker maps its OWN MappedFile (job-local) and runs parse_model into its
//     OWN ir::Model. It reads NOTHING from model A, so nothing on the worker can
//     race the primary session's in-flight shape inference.
//   * The parsed model B + its mapping are published to the MAIN thread via
//     shared_ptr. The completion (token-checked, newest-load-wins) then:
//       - moves the mapping into file_ and adopts model B, and
//       - runs diff_models(*primary.model(), primary_graph_, *B, 0) INLINE.
//     This diff is bounded main-thread work (~2*node fingerprints + matching)
//     and runs only when the primary model/graph still matches the snapshot, so
//     model A's immutable Node/edge/producer data is read safely and its
//     ValueInfo.shape/dtype are NEVER touched (diff_models guarantees this).
//   * token_ is bumped by every load_comparison()/clear() (main thread only), so
//     an in-flight worker's completion is dropped if a newer load superseded it;
//     file_/model_ are only ever mutated by the winning completion on the main
//     thread, so an in-flight worker (which owns its own MappedFile) never sees
//     them mutated underneath it.
#include "engine/DiffLoader.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h references but
// does not itself include; pull it in first so ModelSession.h compiles here.
#include "engine/LayoutEngine.h"

#include "engine/ModelDiff.h"
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

}  // namespace

DiffLoader::DiffLoader(JobSystem& jobs) : jobs_(jobs) {}

// Mirror ModelSession's documented lifetime assumption: App destroys this before
// the JobSystem. Bump the token so any completion that has already been queued
// becomes a no-op if it still runs before shutdown.
DiffLoader::~DiffLoader() { ++token_; }

void DiffLoader::load_comparison(const ModelSession& primary,
                                 const std::string& path) {
  // MAIN THREAD. Newest load wins: bump the token so any in-flight completion is
  // dropped, and snapshot the primary graph index we are diffing against.
  const uint64_t token = ++token_;
  primary_graph_ = primary.current_graph();
  state_ = DiffLoadState::Loading;
  error_.clear();
  path_ = path;
  diff_.reset();
  progress_.set(0.0f, "loading");

  const ModelSession* primary_ptr = &primary;

  jobs_.submit([this, token, path, primary_ptr] {
    // WORKER. Own the mapping + parse into a job-local model. Reads nothing from
    // model A, so no race with the primary session.
    auto mapped = MappedFile::open(path);
    if (!mapped) {
      std::string msg = mapped.error().message;
      jobs_.post_to_main([this, token, msg] {
        if (token_ != token) return;  // stale: superseded by a newer load/clear
        state_ = DiffLoadState::Failed;
        error_ = msg;
      });
      return;
    }
    auto mf = std::make_shared<MappedFile>(mapped.take());
    const std::string ext = ext_of(path);
    Result<ir::Model> parsed = parse_model(*mf, ext, progress_);
    if (!parsed) {
      std::string msg = parsed.error().message;
      jobs_.post_to_main([this, token, msg] {
        if (token_ != token) return;  // stale
        state_ = DiffLoadState::Failed;
        error_ = msg;
      });
      return;
    }
    auto model = std::make_shared<ir::Model>(parsed.take());

    jobs_.post_to_main([this, token, model, mf, primary_ptr]() mutable {
      if (token_ != token) return;  // stale: a newer load/clear superseded us
      // MAIN THREAD: adopt model B + its mapping (both were job-local until now).
      file_ = std::move(*mf);
      model_ = model;  // shared_ptr<ir::Model> -> shared_ptr<const ir::Model>

      // Diff INLINE against the primary snapshot, but only if the primary model
      // and graph still match what we snapshotted at call time (guards against a
      // primary re-open or push_graph/pop_graph having moved on). diff_models
      // reads only immutable structure/topology of model A — never shapes.
      const ir::Model* pa = primary_ptr->model();
      if (pa && primary_ptr->current_graph() == primary_graph_) {
        ModelDiffResult d = diff_models(*pa, primary_graph_, *model_, 0);
        const bool ok = d.valid;
        diff_ = std::make_unique<ModelDiffResult>(std::move(d));
        if (ok) {
          // Pin the primary generation the diff was computed against, so the
          // view stops tinting the moment the primary model is reloaded.
          primary_generation_ = primary_ptr->generation();
          state_ = DiffLoadState::Ready;
        } else {
          state_ = DiffLoadState::Failed;
          error_ = "diff graph index out of range";
          diff_.reset();
        }
      } else {
        // Primary moved on; model B is loaded but the snapshot diff is stale.
        // Leave diff_ null so active() is false and the view does not tint.
        state_ = DiffLoadState::Ready;
      }
    });
  });
}

void DiffLoader::update() {
  // MAIN THREAD: DiffLoader uses its OWN JobSystem (see header), so drain that
  // system's completion queue here — call once per frame after the primary
  // session's update(). This runs the load/diff completions posted by workers.
  jobs_.drain_completions();
}

void DiffLoader::clear() {
  // MAIN THREAD: exit diff mode. Bump the token so any in-flight completion is
  // dropped, and reset to Empty.
  ++token_;
  state_ = DiffLoadState::Empty;
  error_.clear();
  path_.clear();
  diff_.reset();
  model_.reset();
  file_ = MappedFile{};
  primary_graph_ = 0;
  primary_generation_ = UINT64_MAX;
}

}  // namespace netvis
