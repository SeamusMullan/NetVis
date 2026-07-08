// core/JobSystem.h — fixed thread pool + main-thread completion queue.
//
// DECISION (spec §4): the main thread NEVER blocks. Parsing, layout, search
// indexing, and tensor decode run on a small pool (hardware_concurrency-1,
// min 2). Each job, when done, pushes a completion callback onto a queue the
// main thread drains once per frame — so results land on the UI thread without
// the UI ever waiting on a lock held by a worker.
//
// A generation counter invalidates in-flight work when the user opens a new
// file: a job captures the generation it was queued under and checks it before
// publishing; stale results are dropped. This prevents a slow parse of file A
// from overwriting the view after the user has already opened file B.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace netvis {

class JobSystem {
 public:
  // Spawn (max(2, hw-1)) workers.
  JobSystem() {
    unsigned hw = std::thread::hardware_concurrency();
    unsigned n = hw > 1 ? hw - 1 : 2;
    if (n < 2) n = 2;
    workers_.reserve(n);
    for (unsigned i = 0; i < n; ++i)
      workers_.emplace_back([this] { worker_loop(); });
  }

  ~JobSystem() { shutdown(); }

  JobSystem(const JobSystem&) = delete;
  JobSystem& operator=(const JobSystem&) = delete;

  size_t worker_count() const { return workers_.size(); }

  // Queue work to run on a background thread.
  void submit(std::function<void()> job) {
    {
      std::lock_guard<std::mutex> lk(job_mu_);
      jobs_.push(std::move(job));
    }
    job_cv_.notify_one();
  }

  // Called by a worker (via a job) to hand a result back to the main thread.
  // The callback runs on the main thread during drain_completions().
  void post_to_main(std::function<void()> completion) {
    std::lock_guard<std::mutex> lk(done_mu_);
    completions_.push(std::move(completion));
  }

  // Main thread: run all pending completion callbacks. Call once per frame.
  void drain_completions() {
    std::queue<std::function<void()>> local;
    {
      std::lock_guard<std::mutex> lk(done_mu_);
      std::swap(local, completions_);
    }
    while (!local.empty()) {
      local.front()();
      local.pop();
    }
  }

  // Current generation. Bump on new-file-open; jobs compare against this.
  uint64_t generation() const { return generation_.load(std::memory_order_acquire); }
  uint64_t bump_generation() { return generation_.fetch_add(1, std::memory_order_acq_rel) + 1; }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(job_mu_);
      if (stop_) return;
      stop_ = true;
    }
    job_cv_.notify_all();
    for (auto& t : workers_)
      if (t.joinable()) t.join();
    workers_.clear();
  }

 private:
  void worker_loop() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lk(job_mu_);
        job_cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
        if (stop_ && jobs_.empty()) return;
        job = std::move(jobs_.front());
        jobs_.pop();
      }
      job();
    }
  }

  std::vector<std::thread> workers_;

  std::mutex job_mu_;
  std::condition_variable job_cv_;
  std::queue<std::function<void()>> jobs_;
  bool stop_ = false;

  std::mutex done_mu_;
  std::queue<std::function<void()>> completions_;

  std::atomic<uint64_t> generation_{0};
};

// ProgressSink: parsers report coarse progress (0..1) and a stage label. The
// implementation is thread-safe and cheap; the UI polls the latest value.
// DECISION (spec §4, §8.7): progress is a product feature (the stage-timing
// status bar), so every long job takes a sink.
class ProgressSink {
 public:
  void set(float fraction, const char* stage) {
    fraction_.store(fraction, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(mu_);
    stage_ = stage ? stage : "";
  }
  float fraction() const { return fraction_.load(std::memory_order_relaxed); }
  std::string stage() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stage_;
  }

 private:
  std::atomic<float> fraction_{0.0f};
  mutable std::mutex mu_;
  std::string stage_;
};

}  // namespace netvis
