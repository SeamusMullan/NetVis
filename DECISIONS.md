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
- **`std::function` completions must be copy-constructible.** Job completions
  carry results back to the main thread via a `std::function` queue, which
  requires a copyable target. Move-only payloads (`unique_ptr<Model>`,
  `LayoutResult`, `SearchIndex`) are wrapped in a `shared_ptr` before capture;
  only the main thread ever dereferences it, so there is no shared-mutation race
  (see `ModelSession.cpp`).

## Parser specifics

- **ONNX `ModelProto` field numbers.** Per real `onnx.proto3`: field 1 =
  `ir_version`, **2 = `producer_name`**, **3 = `producer_version`**, 7 = `graph`,
  **8 = `opset_import`**, 14 = `metadata_props`. (An early draft of the contract
  transposed 2/3/8; corrected after the fixture round-trip test caught it.)
- **GGUF quantized dtype bucketing.** ggml has dozens of quant types; v1 buckets
  4/5-bit and sub-4-bit K/IQ quants under `DType::Q4` and 6/8-bit under `Q8`
  (labels only — no dequant, spec §12). Documented at the mapping site.
- **`.npy` shape tuple.** NumPy convention: `(2, 3)` for rank≥2, a trailing comma
  only for 1-tuples `(3,)`, and `()` for scalars — matched exactly so exported
  tensors load in NumPy.

## Default dock layout

- The view seeds a one-time `DockBuilder` layout (graph/tensor-table center,
  properties right, weight inspector bottom-right); afterwards ImGui persists
  whatever the user rearranges. Uses `imgui_internal.h` (the only place).

## Hostile-input hardening

A file viewer must survive corrupt/malicious inputs without crashing. An
adversarial review of the parsers/engine found and fixed these (all now covered
by `tests/test_hardening.cpp` and an ASan/UBSan truncation fuzz):

- **Integer-overflow bounds checks use division, not multiplication.** e.g.
  `elem_count * dtype_size` can wrap uint64 and pass a `<= payload_len` check,
  then drive an OOB streaming read. Compare via `payload_len / dtype_size`
  instead (`TensorStats.cpp`). Same pattern for SafeTensors `data_base + end`
  (bound `end` by `file_size - data_base`) and TFLite vector lengths (bound the
  element count by remaining file bytes before `reserve`).
- **Attacker-controlled divisors are checked.** ONNX `strides` of 0 would SIGFPE
  in Conv/Pool shape inference; treated as unresolvable instead (`ShapeInference.cpp`).
- **Untrusted counts are bounded by the actual file size** before any
  value-initializing `resize()` (`LayoutCache.cpp`) — a corrupt count can't force
  a multi-GB commit.
- **Recursion over untrusted graphs is cycle- and depth-guarded.** The pickle
  value graph can be cyclic or a shared DAG with exponential path count; the
  state-dict walk carries a visited-set + depth cap (`PytorchParser.cpp`).
- **Digit runs in node names saturate** at 18 digits to avoid signed-int
  overflow UB (`CollapseTree.cpp`).
- **Crossing counting is O(E log E)** (Fenwick tree), not O(E²), so a wide
  non-collapsing graph still meets the layout budget (`LayoutEngine.cpp`).

## v0.2.0 additions

- **Layout works on internal layout-node arrays, not `out.boxes`.** To place a
  shared constant next to *each* consumer, `compute_layout` now duplicates a
  multi-consumer source (in-degree 0, non-group, ≥2 consumers) into one clone per
  consumer, and inserts Sugiyama dummy nodes along edges spanning >1 layer.
  Ordering/coordinate/routing passes run on internal `npos/nsize/nlayer/nowner`
  arrays covering *real + clones + dummies*; only real+clone nodes are emitted as
  `NodeBox`es. **Invariant relaxation:** `LayoutResult.boxes` may therefore exceed
  `display_nodes().size()`, and several boxes may share a `display_id` (a clone
  carries its source's id). Every view consumer keys off `box.display_id`
  (bounds-checked `< display_nodes().size()`), never the box index — this was
  already true. Duplication is capped at 64 consumers and dummies at 128/edge
  (deterministic fallback to a shared node / straight edge) to hold the budget.
  Any position-affecting layout change **must bump `kVersion`** in `LayoutCache.cpp`
  or a stale `.nvl` masks it (v3 covers duplication + dummy routing).
- **`view/` is not a frozen contract.** `CONTRACTS.md` lists `view/` as
  *to-implement*; the `App.h` banner comment is historical. `ViewState` is
  append-only — v0.2.0 adds `hide_const_edges`, a `unique_ptr<GraphNavState> nav`,
  and `diff_panel_open`. New panels are new `draw_*` free functions in new TUs, so
  frozen panel signatures in `App.h` are never edited.
- **Graph adjacency is built synchronously on the main thread.** `GraphAdjacency`
  (CSR forward+reverse) is one O(V+E) pass — far cheaper than `compute_layout`,
  which already runs per open — so navigation rebuilds it inline on
  generation/graph change. Backgrounding it would need a shared-ownership handle
  to the `ir::Model` the view doesn't have (the worker could read a freed model on
  re-open), so we keep it on the main thread and avoid the lifetime hazard entirely.
- **Model diff uses a second `JobSystem`.** The comparison model is loaded + diffed
  through `engine/DiffLoader` (engine may include parsers; the view must not). It
  owns its *own* `JobSystem` so its generation counter can't cross-cancel the
  primary session's in-flight parse/layout/shape jobs. The diff reads only
  op_type/name/attributes/arity/topology — never `ValueInfo.shape/dtype` — so it is
  race-free against in-flight shape inference on either model. Cross-model node
  matching compares string **content** (`model.str(id)`), since the two models have
  independent `StringArena`s and a raw `StringId` is meaningless across them.
- **Shape inference gained an mmap-aware overload.** The frozen 3-arg
  `infer_shapes` delegates to `infer_shapes_ext(…, base, size, …)`; when a base is
  supplied, shape-driven ops (Reshape/Slice/Gather/…) resolve by reading a
  *raw_data*-backed shape initializer through the mapping (every read
  bounds-checked against `base+size`). Shapes packed into `int64_data`/`int32_data`
  protobuf fields have no recorded offset (`file_offset == UINT64_MAX`) and stay
  `Unknown` — best-effort, never a crash.
