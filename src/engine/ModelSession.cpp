// engine/ModelSession.cpp — implementation of the model-owning session that
// drives the background job pipeline (spec §3, §4).
//
// THREADING (spec §4): every public method here runs on the MAIN thread. Long
// work is pushed onto JobSystem workers; each job owns the data it touches until
// it hands a result back via jobs_.post_to_main(), whose completion runs on the
// MAIN thread during update()/drain_completions(). Completions capture `this`
// and the generation the job was queued under and PUBLISH ONLY IF that
// generation still matches jobs_.generation() — otherwise the user has already
// opened another file and the stale result is dropped.
//
// LIFETIME ASSUMPTION (documented, per task): the JobSystem is owned by App and
// OUTLIVES this ModelSession. App destroys the session before the JobSystem, so
// a completion that fires after destruction cannot happen for THIS session —
// but even so, every completion re-checks generation before touching state, and
// bumping the generation on open/close invalidates in-flight captures.

// LayoutEngine.h defines SizeFn, which the frozen ModelSession.h references but
// does not itself include; pull it in first so the header compiles standalone.
#include "engine/LayoutEngine.h"

#include "engine/ModelSession.h"

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "engine/LayoutCache.h"
#include "engine/ShapeInference.h"
#include "engine/ShapeInferenceExt.h"

namespace netvis {

namespace {

// Lowercased file extension without the leading dot; "" if none. Used only as a
// format-detection tiebreaker (spec §5).
std::string ext_of(const std::string& path) {
  std::filesystem::path p(path);
  std::string e = p.extension().string();
  if (!e.empty() && e[0] == '.') e.erase(0, 1);
  for (char& c : e) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return e;
}

// Wall-clock milliseconds between two steady_clock samples.
double ms_since(std::chrono::steady_clock::time_point start) {
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

ModelSession::ModelSession(JobSystem& jobs) : jobs_(jobs) {}

// Destructor: bump the generation so any in-flight worker's captured generation
// no longer matches and its completion becomes a no-op. (App is responsible for
// destroying the session before the JobSystem — see file header.)
ModelSession::~ModelSession() { jobs_.bump_generation(); }

std::string ModelSession::model_dir() const {
  if (path_.empty()) return {};
  return std::filesystem::path(path_).parent_path().string();
}

void ModelSession::open_async(const std::string& path) {
  // MAIN THREAD. Invalidate any in-flight work from a previous open and capture
  // the fresh generation for every job we submit below.
  const uint64_t gen = jobs_.bump_generation();
  generation_ = gen;

  // Reset all derived state so the view shows a clean loading state immediately.
  error_.clear();
  stage_ = LoadStage::Mapping;
  model_.reset();
  layout_.reset();
  collapse_ = CollapseTree{};
  search_ = SearchIndex{};
  current_graph_ = 0;
  graph_stack_.clear();
  timings_ = StageTimings{};
  progress_.set(0.0f, "mapping");
  path_ = path;

  // mmap on the MAIN thread: the file becomes interactive the moment the map
  // succeeds (spec §4). This is cheap even for multi-GB files — no bytes are
  // paged in until touched.
  auto t_map = std::chrono::steady_clock::now();
  auto mapped = MappedFile::open(path);
  timings_.mmap_ms = ms_since(t_map);
  if (!mapped) {
    stage_ = LoadStage::Failed;
    error_ = mapped.error().message;
    return;
  }
  // Fresh mapping in a NEW shared_ptr: any worker from a previous open still
  // holds the old shared_ptr, so replacing ours never munmaps under it.
  file_ = std::make_shared<MappedFile>(mapped.take());
  std::shared_ptr<MappedFile> file = file_;  // snapshot captured by the jobs

  const std::string ext = ext_of(path);
  stage_ = LoadStage::Parsing;
  progress_.set(0.0f, "parsing");

  // ParseJob: runs on a worker. It owns the produced Model until publication.
  // The MappedFile is immutable after mapping, so read-only concurrent access
  // from the worker is safe (see MappedFile threading note). The captured
  // shared_ptr keeps the mapping alive for the job even across a reopen.
  jobs_.submit([this, gen, ext, file] {
    auto t_parse = std::chrono::steady_clock::now();
    // parse_model reads ONLY structure through ByteReader and records tensor
    // payload offset+length; it never reads weight bytes.
    Result<ir::Model> parsed = parse_model(*file, ext, progress_);
    double parse_ms = ms_since(t_parse);

    if (!parsed) {
      // Publish the failure on the MAIN thread (generation-checked).
      std::string msg = parsed.error().message;
      jobs_.post_to_main([this, gen, msg, parse_ms] {
        if (jobs_.generation() != gen) return;  // stale: newer file opened
        timings_.parse_ms = parse_ms;
        stage_ = LoadStage::Failed;
        error_ = msg;
      });
      return;
    }

    // Move the parsed model into a heap object the worker owns until the
    // completion transfers ownership into model_ on the MAIN thread. shared_ptr
    // (not unique_ptr) because JobSystem's completion queue is std::function,
    // which requires a copy-constructible target; only the main thread ever
    // dereferences it, so there is no shared-mutation race.
    auto model = std::make_shared<ir::Model>(parsed.take());
    jobs_.post_to_main([this, gen, parse_ms, model]() mutable {
      if (jobs_.generation() != gen) return;  // stale: drop
      // MAIN THREAD: publish the model, then kick the derived-state jobs.
      timings_.parse_ms = parse_ms;
      // model_ is unique_ptr; adopt the shared_ptr's object by moving it into a
      // fresh unique_ptr. The shared_ptr held the sole reference.
      model_ = std::make_unique<ir::Model>(std::move(*model));
      const bool is_onnx = (model_->str(model_->format_name) == "ONNX");

      // Build the collapse tree for the main graph (main-thread, fast).
      collapse_.build(*model_, 0);
      current_graph_ = 0;
      stage_ = LoadStage::Laying;
      progress_.set(0.0f, "layout");

      // Layout job (may hit the persistent cache).
      request_layout();

      // SearchIndex job: build over the now-immutable model_ on a worker. The
      // model_ pointer is stable and its contents are not mutated on the main
      // thread while this runs (shape inference publishes a *new* enriched
      // state via completion, and search only reads names). We publish the
      // built index by swapping it in on the main thread.
      jobs_.submit([this, gen] {
        auto t_search = std::chrono::steady_clock::now();
        auto index = std::make_shared<SearchIndex>();
        index->build(*model_);
        double search_ms = ms_since(t_search);
        jobs_.post_to_main([this, gen, search_ms, index]() mutable {
          if (jobs_.generation() != gen) return;  // stale: drop
          search_ = std::move(*index);
          timings_.search_ms = search_ms;
        });
      });

      // ShapeInferJob (ONNX only): mutate ValueInfo shapes/dtypes in place.
      // OWNERSHIP: infer_shapes writes into model_ from a worker. The main
      // thread is not reading those ValueInfo fields for enrichment yet
      // (stage is Enriching); the writes become visible to the UI when the
      // completion runs on the main thread and flips to Ready — the completion
      // acts as the happens-before publication point for those writes.
      if (is_onnx) {
        stage_ = LoadStage::Enriching;
        // Snapshot the mapping (main thread) so the worker holds it alive even
        // if the primary is reopened mid-inference — reading file_ off-thread
        // would otherwise race a munmap.
        std::shared_ptr<MappedFile> file = file_;
        jobs_.submit([this, gen, file] {
          auto t_shapes = std::chrono::steady_clock::now();
          infer_shapes_ext(*model_, 0, file->data(), file->size(), &progress_);
          double shapes_ms = ms_since(t_shapes);
          jobs_.post_to_main([this, gen, shapes_ms] {
            if (jobs_.generation() != gen) return;  // stale: drop
            timings_.shapes_ms = shapes_ms;
            // Enrichment done; if layout has already landed we are Ready.
            if (stage_ == LoadStage::Enriching && layout_) stage_ = LoadStage::Ready;
          });
        });
      }
    });
  });
}

void ModelSession::request_layout() {
  // MAIN THREAD. Capture the current generation and the immutable inputs the
  // layout worker needs. Layout is a pure function of (model, graph, collapse
  // view, sizes) so the worker only reads state that does not change under it.
  if (!model_) return;
  const uint64_t gen = generation_;
  const uint64_t structure_hash = collapse_.structure_hash();
  const uint64_t collapse_hash = collapse_.collapse_hash();

  jobs_.submit([this, gen, structure_hash, collapse_hash] {
    // First try the persistent layout cache keyed by (structure, collapse).
    // Weights never enter the key, so a re-export with identical topology hits.
    Result<LayoutResult> cached = load_cached_layout(structure_hash, collapse_hash);
    if (cached) {
      auto layout = std::make_shared<LayoutResult>(cached.take());
      layout->from_cache = true;
      jobs_.post_to_main([this, gen, layout]() mutable {
        if (jobs_.generation() != gen) return;  // stale: drop
        layout_ = std::make_unique<LayoutResult>(std::move(*layout));
        if (stage_ != LoadStage::Enriching) stage_ = LoadStage::Ready;
      });
      return;
    }

    // Cache miss: compute the layout on this worker.
    auto t_layout = std::chrono::steady_clock::now();
    LayoutResult result =
        compute_layout(*model_, current_graph_, collapse_, size_fn_, {}, &progress_);
    double layout_ms = ms_since(t_layout);

    // Best-effort persist; a write failure is non-fatal (spec §7.2.7).
    store_cached_layout(result);

    auto layout = std::make_shared<LayoutResult>(std::move(result));
    jobs_.post_to_main([this, gen, layout_ms, layout]() mutable {
      if (jobs_.generation() != gen) return;  // stale: drop
      timings_.layout_ms = layout_ms;
      layout_ = std::make_unique<LayoutResult>(std::move(*layout));
      // Ready unless we are still enriching (ONNX shapes not done yet).
      if (stage_ != LoadStage::Enriching) stage_ = LoadStage::Ready;
    });
  });
}

void ModelSession::update() {
  // MAIN THREAD: run all completions queued by workers this frame.
  jobs_.drain_completions();
}

void ModelSession::toggle_group(uint32_t group_index) {
  // MAIN THREAD. Toggle then re-layout the whole current collapse view. A full
  // re-layout is fine here: at collapsed sizes the display-node count is small,
  // so recomputing everything is sub-frame — simpler than region-incremental.
  if (collapse_.toggle_group(group_index)) {
    stage_ = (stage_ == LoadStage::Ready) ? LoadStage::Laying : stage_;
    request_layout();
  }
}

void ModelSession::push_graph(uint32_t graph_index) {
  // MAIN THREAD: dive into a subgraph, remembering where we came from.
  if (!model_) return;
  graph_stack_.push_back(current_graph_);
  current_graph_ = graph_index;
  collapse_.build(*model_, graph_index);
  stage_ = LoadStage::Laying;
  request_layout();
}

void ModelSession::pop_graph() {
  // MAIN THREAD: back up the breadcrumb stack.
  if (graph_stack_.empty() || !model_) return;
  current_graph_ = graph_stack_.back();
  graph_stack_.pop_back();
  collapse_.build(*model_, current_graph_);
  stage_ = LoadStage::Laying;
  request_layout();
}

}  // namespace netvis
