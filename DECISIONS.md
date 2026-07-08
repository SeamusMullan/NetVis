# NetVis — Design Decisions

Rationale for choices where the spec was ambiguous or where a performance-relevant
tradeoff was made. Per spec §13, ambiguities resolve toward the <1s cold-open /
60fps budgets.

## Foundation

- **`Result<T>` over exceptions.** No exceptions cross module boundaries (spec §13).
  Errors carry a byte offset for file-position diagnostics (spec §6).
- **`SmallVec<T,6>` for shapes.** Shapes are almost always ≤6 dims; inline storage
  avoids a heap allocation per tensor/value across millions of objects.
- **`StringArena` + 32-bit `StringId`.** Massive string redundancy in model files
  (repeated op types, name prefixes). Interning shrinks IR to fixed-size POD and
  makes structural hashing cheap. Backed by `std::deque` (not `vector`) so stored
  string addresses — used as `string_view` map keys — stay stable across growth.
- **`ByteReader` payload counter.** Structural parsing routes reads through checked
  accessors; a tensor-payload read is deliberately marked via `mark_payload_read()`.
  Tests assert the counter is 0 after a parse (spec §2.1, §10) — proving no eager
  weight loading. Counter is `thread_local` so parallel parses don't race.
- **FNV-1a hashing.** Tiny, dependency-free, order-sensitive. Non-cryptographic;
  a cache-key collision merely triggers a recompute, which is safe.
- **`MADV_RANDOM` on the mmap.** We touch structure front-to-back but weights
  rarely and randomly; disabling readahead avoids faulting in weight pages we
  never inspect (core of the sub-second-open thesis, spec §2.1).

## Build

- **`CMAKE_POLICY_VERSION_MINIMUM=3.5`.** miniz 3.0.2 and glfw 3.4 declare a
  `cmake_minimum_required` below what CMake 4.x supports. We relax the policy floor
  for dependencies rather than fork them; NetVis's own code targets CMake ≥3.24.
- **`GLOB_RECURSE CONFIGURE_DEPENDS` per layer.** Lets modules add source files
  without editing CMakeLists; the three targets (`netvis_core`, `netvis`,
  `netvis_tests`) map exactly to spec §9.
- **Warnings-as-errors only on our code.** `netvis_warnings` INTERFACE target is
  applied to our targets, never to `third_party` (spec §2.6).

## Architecture seams (frozen contracts)

- **`ModelSession` is the only object the view talks to** (spec §3). The view never
  includes a parser. Parsers → `ir::Model`; engine → `LayoutResult` / `SearchIndex`;
  view reads POD. See `CONTRACTS.md`.
- **Layout takes a `SizeFn`.** Node box sizes come from the view's font metrics, so
  `compute_layout` is a pure headless function — runs on a worker thread and in
  tests without a GUI (spec §7.2). Determinism per spec §2.7.
- **Collapse before layout.** `CollapseTree` produces a flat "display node" list of
  leaves + `×N` super-nodes; layout/canvas operate only on display nodes. This is
  what keeps the default layout of a 100k-node model sub-second (spec §7.1).
