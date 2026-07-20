// engine/ModelSession.h — owns the loaded model + all derived engine state.
//
// DECISION (spec §3, §4): the single object the view talks to. It owns the mmap,
// the parsed ir::Model, the CollapseTree, the current LayoutResult, and the
// SearchIndex, and it drives the background job pipeline
//   mmap -> ParseJob -> (ShapeInferJob ∥ SearchIndexJob ∥ LayoutJob) -> done
// The main thread NEVER blocks: open_async() returns immediately; results land
// via JobSystem completion callbacks drained once per frame. A generation
// counter invalidates in-flight jobs when a new file is opened (spec §4).
//
// THREADING CONTRACT:
//  - All public methods here are called from the MAIN thread only.
//  - Background jobs operate on data they exclusively own until publication;
//    they check generation() before posting results to the main thread.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "engine/CollapseTree.h"
#include "engine/Layout.h"
#include "engine/SearchIndex.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

namespace netvis {

// Coarse pipeline stage for the status bar / progressive UI (spec §4, §8.7).
enum class LoadStage : uint8_t {
  Empty,       // nothing loaded
  Mapping,     // mmap in progress
  Parsing,     // ParseJob running
  Laying,      // LayoutJob running (interactive already: skeleton shown)
  Enriching,   // shapes / search still filling in
  Ready,       // fully loaded
  Failed,
};

// Per-stage timing in milliseconds (spec §8.7: timings are a feature).
struct StageTimings {
  double mmap_ms = 0, parse_ms = 0, layout_ms = 0, shapes_ms = 0, search_ms = 0;
};

class ModelSession {
 public:
  explicit ModelSession(JobSystem& jobs);
  ~ModelSession();

  ModelSession(const ModelSession&) = delete;
  ModelSession& operator=(const ModelSession&) = delete;

  // Begin loading `path`. Non-blocking: bumps generation, cancels in-flight
  // work, maps the file, and kicks off the parse pipeline. Returns immediately.
  void open_async(const std::string& path);

  // Drain completed jobs and apply their results. Call once per frame from the
  // main thread (delegates to JobSystem::drain_completions()).
  void update();

  // --- State the view reads (main thread only) -------------------------------
  LoadStage stage() const { return stage_; }
  const std::string& error_message() const { return error_; }
  float progress() const { return progress_.fraction(); }
  std::string progress_stage() const { return progress_.stage(); }
  const StageTimings& timings() const { return timings_; }
  const std::string& path() const { return path_; }

  // The parsed model (null until Parsing completes). Owned here.
  const ir::Model* model() const { return model_.get(); }
  bool has_graph() const { return model_ && model_->has_graph; }

  // Current graph being viewed (breadcrumb stack for subgraph dives).
  uint32_t current_graph() const { return current_graph_; }
  void push_graph(uint32_t graph_index);   // dive into a subgraph
  void pop_graph();                        // back up the breadcrumb
  const std::vector<uint32_t>& graph_stack() const { return graph_stack_; }

  CollapseTree& collapse() { return collapse_; }
  const CollapseTree& collapse() const { return collapse_; }

  // Latest published layout (null until first LayoutJob completes).
  const LayoutResult* layout() const { return layout_.get(); }

  // Toggle a collapse group and request an incremental re-layout (spec §7.2.6).
  void toggle_group(uint32_t group_index);

  const SearchIndex& search() const { return search_; }

  const MappedFile& file() const { return *file_; }
  std::string model_dir() const;  // directory of `path_`, for external_data

  // Provided by the view so layout can measure label extents (font metrics).
  // Defaults to a headless heuristic until the view installs one.
  void set_size_fn(SizeFn fn) { size_fn_ = std::move(fn); }

  uint64_t generation() const { return generation_; }

  // Bumped on the main thread each time background enrichment mutates the
  // published model in place (currently: ONNX shape inference completing). The
  // model pointer/generation do NOT change when this happens, so derived state
  // that depends on ValueInfo shapes (e.g. the cost report) must fold this epoch
  // into its rebuild key or it serves a pre-inference (all-shapes-empty) result
  // forever. Starts at 0; only ever increases within a generation.
  uint64_t enrich_generation() const { return enrich_generation_; }

 private:
  JobSystem& jobs_;

  // --- Owned model state (main thread; mutated only via completions) ---------
  // shared_ptr, not a bare MappedFile: worker jobs (parse/search/shape) read the
  // mapping off the main thread, and a reopen would otherwise munmap it under a
  // live worker (use-after-free). Each job captures a shared_ptr snapshot so the
  // mapping outlives the job; reopen swaps in a fresh mapping without unmapping
  // the one an in-flight worker still holds. Never null after the first open;
  // file() dereferences it. Default-constructed to an empty mapping.
  std::shared_ptr<MappedFile> file_ = std::make_shared<MappedFile>();
  std::unique_ptr<ir::Model> model_;
  CollapseTree collapse_;
  std::unique_ptr<LayoutResult> layout_;
  SearchIndex search_;

  std::string path_;
  std::string error_;
  LoadStage stage_ = LoadStage::Empty;
  StageTimings timings_;
  ProgressSink progress_;

  uint32_t current_graph_ = 0;
  std::vector<uint32_t> graph_stack_;

  SizeFn size_fn_;
  uint64_t generation_ = 0;
  uint64_t enrich_generation_ = 0;  // see enrich_generation()

  // Kick a layout job for the current collapse view (used after parse + expand).
  void request_layout();
};

}  // namespace netvis
