// SPDX-License-Identifier: Apache-2.0
// tests/test_jobsystem.cpp — JobSystem::idle() (#170).
//
// idle() is what the `--screenshot` capture gate waits on: "no job queued or
// running, and no completion left to drain". The ordering argument for why it
// cannot report idle early lives in core/JobSystem.h; these tests drive the real
// pool through it, and the 1000-job case is mostly here for ThreadSanitizer.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "core/JobSystem.h"

using namespace netvis;

namespace {

using Clock = std::chrono::steady_clock;

// Drain completions until the pool reports idle, or the deadline passes. Returns
// whether it went idle. Main thread only, like the real caller.
bool drain_until_idle(JobSystem& js, std::chrono::seconds timeout = std::chrono::seconds(5)) {
  const auto deadline = Clock::now() + timeout;
  for (;;) {
    js.drain_completions();
    if (js.idle()) return true;
    if (Clock::now() > deadline) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

}  // namespace

TEST_CASE("JobSystem::idle: a fresh pool is idle") {
  JobSystem js;
  CHECK(js.idle());
}

TEST_CASE("JobSystem::idle: not idle while a job runs or its completion is undrained") {
  JobSystem js;
  int ran = 0;  // main thread only: touched from the completion
  js.submit([&js, &ran] {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    js.post_to_main([&ran] { ++ran; });
  });
  CHECK_FALSE(js.idle());  // queued or running right now
  CHECK(drain_until_idle(js));
  CHECK(ran == 1);  // exactly once
  CHECK(js.idle());
}

TEST_CASE("JobSystem::idle: a completion that submits follow-up work keeps the pool busy") {
  JobSystem js;
  bool follow_up_done = false;  // main thread only: set from the second completion
  js.submit([&js, &follow_up_done] {
    js.post_to_main([&js, &follow_up_done] {
      // Runs on the main thread inside drain_completions(): queue more work.
      js.submit([&js, &follow_up_done] {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        js.post_to_main([&follow_up_done] { follow_up_done = true; });
      });
    });
  });

  const auto deadline = Clock::now() + std::chrono::seconds(5);
  for (;;) {
    js.drain_completions();
    if (js.idle()) break;
    // The invariant under test: idle() never reads true while the follow-up's
    // own completion has not been drained yet.
    CHECK_FALSE((js.idle() && !follow_up_done));
    if (Clock::now() > deadline) FAIL("pool never went idle");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  CHECK(follow_up_done);
}

TEST_CASE("JobSystem::idle: 1000 jobs each posting a completion all land before idle") {
  JobSystem js;
  int counter = 0;  // main thread only
  for (int i = 0; i < 1000; ++i) {
    js.submit([&js, &counter] { js.post_to_main([&counter] { ++counter; }); });
  }
  CHECK(drain_until_idle(js, std::chrono::seconds(20)));
  CHECK(counter == 1000);
  CHECK(js.idle());
}
