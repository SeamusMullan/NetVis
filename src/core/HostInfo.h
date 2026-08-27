// core/HostInfo.h — what machine produced a measurement (#154).
//
// DECISION (#154): the perf gate refuses to compare two runs recorded on
// different machine classes, and it made that call from ONE number — the
// logical core count. When the gate then said "baseline was recorded on a
// 4-core machine, this run is on 2 cores", nothing in either JSON said WHICH
// machines those were, so the only way to act on the message was to guess.
// A run that carries its own hardware is diagnosable from the artifact alone,
// after the runner that produced it is long gone.
//
// This is metadata, not measurement: nothing here is ever divided into a
// timing or used to normalize one. Wall-clock numbers are compared only
// against a baseline from the same machine class (tools/bench_gate.py), and
// scaling a number by a core count would invent precision the harness cannot
// justify — every bench stage is single-threaded anyway (engine/Bench.h).
//
// HONESTY RULE, the same one core/Rss.h follows: an empty string and 0 mean
// "this host would not tell us", never a fabricated value. A reader must
// render those as unknown rather than as a measurement of zero.
#pragma once

#include <cstdint>
#include <string>

namespace netvis {

// A snapshot of the machine a process is running on. Every field is
// independently optional: probing is best-effort per platform, and one
// unavailable counter must not blank out the ones that did answer.
struct HostInfo {
  // Human-readable CPU name, e.g. "AMD Ryzen 9 5900X 12-Core Processor" or
  // "Apple M1 Pro". Empty when the platform has no such string to give.
  std::string cpu_model;

  // Logical processors visible to the scheduler (SMT threads included), and
  // physical cores backing them. Kept as two fields precisely because their
  // RATIO is the interesting part: 8-logical/8-physical and 8-logical/4-physical
  // are different machines that a single core count reports identically.
  // 0 means unknown — notably physical_cores on ARM Linux, where /proc/cpuinfo
  // carries no core-topology fields at all.
  uint32_t logical_cores = 0;
  uint32_t physical_cores = 0;

  // The architecture and OS this binary was COMPILED for, not what the kernel
  // reports at runtime. That is the useful answer for a benchmark: an x86_64
  // build running under Rosetta on an arm64 host is executing x86_64 code, and
  // its timings belong with other x86_64 timings. One of "x86_64", "arm64",
  // "x86", "arm", "unknown"; and "linux", "macos", "windows", "unknown".
  std::string arch;
  std::string os;

  // Total physical RAM in bytes, 0 if unavailable.
  //
  // CAVEAT worth knowing before reading it as a limit: on Linux this is the
  // machine's RAM, which is NOT the cgroup memory limit a container runs
  // under. A CI runner can report 32 GB here and be OOM-killed at 7. It is
  // context for "which machine was this", not a budget to size against.
  uint64_t total_ram_bytes = 0;
};

// Probe the current host. Best-effort and non-throwing: any field the platform
// declines to answer keeps its unknown default. Cheap, but not free (Linux
// reads /proc/cpuinfo, Windows opens a registry key) — call it once per
// process and reuse the result rather than per measurement.
HostInfo host_info();

}  // namespace netvis
