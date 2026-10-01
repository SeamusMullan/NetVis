// SPDX-License-Identifier: Apache-2.0
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
    // Counted BEFORE the push: a worker can only decrement a job that was
    // already counted (see idle()).
    in_flight_.fetch_add(1, std::memory_order_acq_rel);
    {
      std::lock_guard<std::mutex> lk(job_mu_);
      jobs_.push(std::move(job));
    }
    job_cv_.notify_one();
  }

  // True when nothing is queued, nothing is running, and no completion is left
  // to drain (#170: the `--screenshot` capture gate waits on this instead of on
  // LoadStage labels, which have been wrong before).
  //
  // Meaningful on the main thread AFTER drain_completions()/ModelSession::update():
  // a drained completion may submit() more work, and that is counted
  // synchronously, so a loop of {update; idle()} cannot slip between a job and
  // its follow-up.
  //
  // ORDER MATTERS. A job posts its completion (under done_mu_) BEFORE it
  // returns, and the worker decrements in_flight_ only AFTER it returns. So
  // reading in_flight_ == 0 (acquire) guarantees every completion that job
  // posted is already visible in the queue, and the queue check that follows
  // cannot miss it. The reverse order has a hole: see the queue empty, then a
  // job posts and decrements, then see 0, and report idle with a completion
  // still pending.
  bool idle() const {
    if (in_flight_.load(std::memory_order_acquire) != 0) return false;
    std::lock_guard<std::mutex> lk(done_mu_);
    return completions_.empty();
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
      // After job() returns: its completion (if any) is already queued.
      in_flight_.fetch_sub(1, std::memory_order_acq_rel);
    }
  }

  std::vector<std::thread> workers_;

  std::mutex job_mu_;
  std::condition_variable job_cv_;
  std::queue<std::function<void()>> jobs_;
  bool stop_ = false;

  mutable std::mutex done_mu_;
  std::queue<std::function<void()>> completions_;

  // Jobs submitted and not yet finished (queued + running). shutdown() drains the
  // queue before the workers exit, so the count stays consistent.
  std::atomic<size_t> in_flight_{0};

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

  // Cooperative cancellation (#61): the main thread flips the flag with
  // request_cancel(); a long job polls cancelled() at its coarse checkpoints and
  // bails out. reset_cancel() is called on the main thread before a fresh job is
  // submitted so a prior cancel never poisons the next one. Deliberately NOT
  // cleared inside set() — a checkpoint must be able to observe the flag.
  void request_cancel() { cancel_.store(true, std::memory_order_relaxed); }
  bool cancelled() const { return cancel_.load(std::memory_order_relaxed); }
  void reset_cancel() { cancel_.store(false, std::memory_order_relaxed); }

 private:
  std::atomic<float> fraction_{0.0f};
  mutable std::mutex mu_;
  std::string stage_;
  std::atomic<bool> cancel_{false};
};

}  // namespace netvis
