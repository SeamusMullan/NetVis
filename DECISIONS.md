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

## v0.2.1 — layout drift fix (LayoutCache kVersion 4)

The v0.2.0 x-coordinate assignment made graphs march consistently *down and to
the right*, so the minimap collapsed to a diagonal line. Three compounding
rightward biases, all fixed in `LayoutEngine.cpp`:

- **Even-count median took the upper (right) neighbor.** A 2-input node aligned
  to `cs[k/2]` — the *right* of its two inputs. Now uses the true median (mean of
  the two middle centers for even counts), so a node sits between its inputs.
- **Overlap resolution only pushed right.** A single left-to-right sweep
  (`if (x < min_x) x = min_x`) is a ratchet that never moves a node left. Now
  resolves symmetrically: average a left-packed solution (biases right) with a
  right-packed one (biases left). Averaging two gap-feasible monotone solutions
  is itself gap-feasible and centered — no net directional push.
- **Dummy lanes shear the flow.** Skip/residual edges insert dummy nodes on the
  layers they cross; a real chain node aligning against a consistently-offset
  dummy accumulates a small per-layer shift, i.e. a linear *shear*. A final
  least-squares **de-shear** fits the per-layer centroid trend (`centroid ≈
  a + b·layer`) and subtracts `b·layer` from every node. This removes only the
  global drift: a straight chain has slope 0 (untouched), and all within-layer
  order + relative offsets are preserved. O(V+L), deterministic. Result: a deep
  skip-chain that spanned ~29 node-widths of horizontal drift now stays under 1.

## v0.3.0 — analyzer mode (static cost/memory analysis)

NetVis moves from "see the model" to "understand its cost." `engine/CostModel.h`
adds `compute_cost(model, graph_idx) -> CostReport`: per-node FLOPs, params,
weight bytes, activation bytes; model totals; peak-activation estimate; and a
per-dtype quant-coverage report.

- **Static, structural, zero-payload.** `compute_cost` is a pure headless function
  over the already-parsed `ir::Model` — it reads only resolved `ValueInfo`
  shapes/dtypes and `TensorRef` metadata (`elem_count`/`byte_len`/`dtype`), never a
  tensor payload. It upholds the same zero-payload-read thesis the parsers do
  (asserted in `test_cost.cpp` via `payload_read_counter()==0`), and like
  `infer_shapes`/`GraphAdjacency` it is deterministic, never throws, and runs
  headless in tests. It is an **estimate**: an unsupported op or an unresolved
  shape is reported honestly as `flops_known=false` and excluded from `total_flops`
  (never faked); the panel shows "FLOPs unknown for k/N nodes."
- **FLOPs = 2 × MACs.** One multiply-accumulate is one multiply + one add. MatMul
  = `2·|O|·K` (K = last input dim); Conv = `2·|O|·(Cin/g)·∏kernel` reading
  Cin/group + kernel from the weight-initializer shape `[Cout, Cin/g, k…]`;
  elementwise/activation/norm = `|O|`; reduce = `|input[0]|`; pool = `|O|·∏kernel`.
  Structural ops (Reshape/Concat/Gather/Cast/control-flow) carry ~0 arithmetic and
  are `flops_known=false`. The formula table is frozen **in the header** so the
  implementation and the test agree on one source of truth. A node with a corrupt
  (out-of-range) input edge_ref is treated as structurally invalid → `flops_known
  =false` rather than estimating from partial shapes (hostile-input honesty).
- **`total_params`/`total_weight_bytes` aggregate over initializers, not
  `sum(per_node)`.** A weight feeding K consumers must count once, and an
  unconsumed initializer must still count; summing the per-node attribution would
  double-count shared weights and drop orphans. `per_node.params` remains the
  per-node view (deliberately double-counts shared weights — that is the node's
  cost); the report totals are the model view, consistent with `dtype_usage`.
- **Peak activation via single-pass liveness.** ONNX node order is topological, so
  one index-order pass tracks live activation bytes: add a node's output bytes,
  update the peak, then free each value at its last consumer unless it is a graph
  output. Freed values are bucketed by their freeing node up front (`free_at[ni]`)
  so the pass is O(V+E), not the O(V²) of rescanning all values per node.
- **Quant-coverage report.** `dtype_usage` aggregates weights per `DType`, sorted by
  bytes descending. Quantized blocks (Q4/Q8) have `dtype_size==0`, so their byte
  count comes from the recorded `TensorRef::byte_len` (the truth for block formats),
  never `elem_count*dtype_size`. Derived: `effective_bits_per_param` and
  `size_vs_fp32` — the compression story at a glance.
- **View surface mirrors existing patterns; not a frozen contract.** `view/
  CostPanel.h` follows GraphNav (lazy `ensure_cost()` keyed on
  generation/graph/collapse) and DiffPanel (a `CostTint` the canvas overrides
  category color with). Heatmap ramps by `log10(flops)`; diff tint wins if both are
  active. `draw_cost_section` renders into the Properties panel. `ViewState` gains
  append-only `unique_ptr<CostReport> cost`, its rebuild keys, and `cost_heatmap`;
  `App.cpp` includes `CostModel.h` so the `unique_ptr` deleter sees the complete
  type at `~App()`.

### v0.3.0 post-release adversarial sweep (6 bugs, all fixed)

An Opus find→verify-to-refute sweep of the cost code surfaced six confirmed
defects the tests missed; each is now covered by a `test_cost.cpp` regression.

- **(blocker) Cost report served permanently stale after shape inference.** ONNX
  shape inference runs as a worker job submitted *after* the model is published; it
  mutates `ValueInfo.shape` in place and its completion did NOT bump
  `generation_`/`current_graph_`/`collapse_hash`. `ensure_cost` keyed only on those
  three, so it built the report one frame *before* shapes landed (every FLOP
  unknown, peak 0) and then short-circuited forever. Fix: `ModelSession` exposes a
  monotonic **`enrich_generation()`**, bumped on the main thread in the
  shape-inference completion; `ensure_cost`'s key includes it, so the report
  recomputes exactly once shapes resolve. (Analogue of the v0.2.0 diff-tint-stale-
  on-reopen bug: a derived view depending on state that changes without a key bump.)
- **(major) Gemm ignored `transA`.** K was taken as `in0.shape.back()`
  unconditionally, but shape inference writes only the *output* shape, not a
  transposed input-operand view — so for `transA=1` operand A is `[K, M]` and
  `back()` is M, computing `M·N·M` instead of `M·N·K`. Fix: read the `transA`
  attribute and pick `K = transA ? shape[0] : shape.back()`.
- **(major) ConvTranspose used the Conv formula.** ConvTranspose's weight is the
  *transposed* layout `[Cin, Cout/g, k…]` and its output spatial size differs from
  the input's, so basing MACs on `|O|` and reading `shape[1]` as Cin/g is wrong.
  Fix: for ConvTranspose base MACs on `|input|` (`macs = |input|·(Cout/g)·∏kernel`,
  `Cout/g = weight.shape[1]`), disambiguating on the op string (both share
  `OpCategory::Conv`).
- **(minor) Einsum got fabricated MatMul FLOPs.** Einsum shares the MatMul category
  (for coloring) but its FLOPs depend on the `equation`, not `|O|·K`. Fix: exclude
  it (`flops_known=false`) rather than guess.
- **(minor) Peak liveness double-counted a value on duplicate output slots.** The
  add-side added output bytes per output *slot*, while the free-side frees each
  value once — so a malformed graph mapping two output slots to one value index
  inflated the peak. Fix: a `produced` set adds each activation's bytes once, at
  first production, restoring add/free symmetry.
- **(minor) Test gaps.** No `transA` Gemm, no ConvTranspose, no Einsum, no
  duplicate-slot liveness, and an unknown-op check that couldn't distinguish
  "computed unknown" from "uninitialized". All added; suite is 84 cases / 590
  assertions, ASan+UBSan clean. (The view-layer cache-keying path remains untested
  — it needs a GUI/ModelSession harness the headless `netvis_core` tests don't
  link; verified manually via the xvfb smoke run instead.)

## v0.3.1 — second sweep (2 more bugs)

A follow-up scan of the cost surface found two defects the v0.3.0 sweep missed:

- **(major) Cost heatmap saturated every collapsed group to max/hot.**
  `cost_tint_for_display` takes the group's tint as the SUM of its member nodes'
  FLOPs, but normalized that sum against a min/max computed over individual
  *per-node* FLOPs. Any multi-node group sum exceeds the largest single node, so
  every group clamped to `t=1` (hot red) — and the default view is collapsed, so
  the heatmap was useless exactly where it mattered. Fix: build the normalization
  scale over the same aggregation unit as the numerator — per *display node* (each
  aggregated via `ir_nodes_for_display`+`sum_node_costs`), so groups and leaves are
  compared on one scale.
- **(minor) Peak liveness double-counted a graph input re-emitted as an output.**
  The `graph_inputs` seed loop added a value's bytes without marking it `produced`,
  so if a malformed graph also listed that value as a node output it was added
  twice, freed once. Fix: the `produced` guard now covers the seed loop too
  (hoisted above it). Both covered by new `test_cost.cpp` regressions (suite: 85
  cases / 592 assertions, ASan+UBSan clean).

## v0.3.2 — QoL: customizable heatmap gradient + prefs persistence

Quality-of-life pass on the analyzer, headlined by a configurable cost heatmap.

- **Gradient math is a pure engine module.** `engine/HeatmapGradient` (a 3-stop
  low/mid/high ramp + reverse + preset tag) is ImGui-free, so `gradient_sample`
  and the presets are unit-tested headless (`test_gradient.cpp`) exactly like
  OpCategory/CostModel. The view converts `Rgba8 -> ImU32` and owns the picker UI.
  A heatmap is a SEQUENTIAL encoding (one increasing magnitude), so the built-in
  presets — Viridis, Magma, Grayscale — are monotonic in lightness (asserted in a
  test); Cool→Hot is kept as the prior default. "Custom" keeps user-edited stops;
  editing any stop in the UI switches the preset tag to Custom.
- **One scale, shared by tint and legend.** The heatmap min/max FLOPs range is a
  single helper, `heatmap_range`, consumed by both `cost_tint_for_display` and the
  on-canvas legend, so they can never diverge (the v0.3.1 group-scale bug was
  exactly such a divergence). The range is computed over the same aggregation unit
  the tint uses — per display node. `normalize_flops` guards the log domain
  (clamp ≥1) and the degenerate min==max case for both log and linear scales.
- **Legend mirrors the minimap.** Drawn with the canvas's own `ImDrawList`, inset
  top-left (minimap owns bottom-right), only when the heatmap is on and the range
  is valid.
- **View prefs persist in `view_prefs.json`.** Written next to the layout cache /
  recent-files list (`layout_cache_dir()`), holding the gradient (preset + stops +
  reverse), scale, theme, and toggles. Load is best-effort with per-key type
  guards and a try/catch — a missing/corrupt/hostile file falls back to defaults,
  exactly like `recent.json`. Custom stops are applied AFTER `gradient_set_preset`
  in load so a persisted Custom gradient's stops win over the preset's.
- **Copy cost to clipboard.** `cost_summary_text` builds a tab-separated block
  (model totals + quant table) for `ImGui::SetClipboardText`. Reads only the
  published report — still zero payload reads.

Suite: 92 cases / 651 assertions, GUI + ASan/UBSan clean.
