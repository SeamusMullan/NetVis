// core/Rss.h — peak resident-set size of this process (#101 memory ceiling).
//
// DECISION (v0.9.3): the memory pillar needs a number the harness can assert on,
// and "peak RSS since process start" is the only one that is both portable and
// meaningful for a viewer. It captures the high-water mark, so a transient spike
// during parse or layout cannot hide behind a low steady state — which is the
// failure this check exists to catch (a 5 GB model must not materialize).
//
// Per-allocation instrumentation was rejected: it would need a global allocator
// hook, which is exactly the kind of process-wide machinery the single-binary
// thesis avoids, and it would not see mmap'd pages at all — and mmap is where a
// model viewer's memory actually goes.
//
// PLATFORM NOTE: this reads the OS high-water counter, which is monotonic for
// the process lifetime. It cannot be reset. A harness that wants per-stage
// figures must therefore either run one stage per process or report deltas and
// say so — deltas of a monotonic maximum are a LOWER bound on that stage's cost,
// never an exact figure. State which you are reporting; do not imply precision
// the counter cannot give.
#pragma once

#include <cstdint>

namespace netvis {

// Peak resident-set size in bytes since process start, or 0 if the platform
// counter is unavailable. Never throws, never allocates, cheap enough to call
// between bench stages.
//
// Linux:   /proc/self/status VmHWM
// macOS:   task_info / MACH_TASK_BASIC_INFO resident_size_max
// Windows: GetProcessMemoryInfo PeakWorkingSetSize
//
// A 0 return means "unknown", NOT "no memory used" — a caller reporting this
// must render it as unknown rather than as a zero measurement (the honesty rule
// the analyzer already follows for unresolved shapes).
uint64_t peak_rss_bytes();

// Current resident-set size in bytes, or 0 if unavailable. Unlike the peak, this
// falls as memory is released, so it is the right figure for "how much is held
// right now, after the model is open".
uint64_t current_rss_bytes();

}  // namespace netvis
