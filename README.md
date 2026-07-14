# NetVis

A speed-first native desktop viewer for neural-network model files — a from-scratch
alternative to Netron, designed so that opening and navigating a multi-gigabyte
model is effectively instant.

The core idea: **weights are never eagerly loaded.** Parsers memory-map the file
and record only *offset + length* for every tensor payload; the bytes stay on disk
until you actually open a tensor in the weight inspector. Opening a 5 GB model
touches only its structure — a few megabytes — so the graph is on screen in well
under a second.

![Graph view](docs/screenshot-graph.png)

*ONNX compute graph — nodes colored by op category, model stats in the properties
panel, minimap bottom-right.*

![Tensor-table view](docs/screenshot-tensor-table.png)

*Tensor-table mode (SafeTensors / GGUF / PyTorch state-dicts) — virtualized,
sortable table with a module hierarchy tree.*

## Features

- **Formats:** ONNX (`.onnx`), TFLite (`.tflite`), SafeTensors (`.safetensors`),
  GGUF (`.gguf`), PyTorch zip & legacy pickle checkpoints (`.pt` / `.pth` / `.bin`).
- **Instant open:** memory-mapped I/O; structure parsed off the main thread; the
  window is interactive the moment the `mmap` succeeds.
- **Compute-graph canvas:** a single custom-drawn region (no per-node widgets) with
  viewport culling, level-of-detail tiers, pan/zoom, selection, and a minimap.
  Draw cost is O(visible), not O(total).
- **Collapse tree:** repeated blocks (e.g. 32 identical decoder layers) are detected
  and collapsed into `×N` super-nodes, so even 100k-node graphs lay out in
  milliseconds. Double-click to expand.
- **Deterministic layered layout:** from-scratch Sugiyama-style layout (longest-path
  layering, barycenter crossing reduction, bezier edges). Same file → same layout,
  cached to disk keyed by structure hash.
- **Weight inspector:** lazily decodes a tensor to streaming stats (min/max/mean/std,
  zero & NaN/Inf counts, 64-bucket histogram) without materializing a converted
  copy. Export to `.npy` or raw `.bin`.
- **Shape inference (ONNX):** best-effort propagation over the common ops fills in
  edge shape labels in the background.
- **Search:** fuzzy, case-insensitive substring/subsequence search over all names;
  Enter flies the camera to the hit.
- **Tensor-table mode:** graph-less formats (GGUF/SafeTensors/PyTorch) show a
  virtualized, sortable tensor table with a dotted-key module hierarchy.
- **Safety:** the PyTorch pickle reader is a *restricted VM* with an explicit
  allowlist — unknown reduce targets become inert placeholders, never executed.
- Dark/light themes, PNG export of the current view, recent files, drag-and-drop,
  CLI open, and a status bar with per-stage load timings.

## Performance

Measured on this machine (16-core, `-O2 + LTO`), synthetic 10,000-node graph:

| Stage | Time |
|---|---|
| `mmap` (any file size) | ~1 ms |
| collapse-tree build (10k nodes) | ~1.2 ms |
| default (collapsed) layout | ~0.1 ms |
| search query over names | < 5 ms |
| tensor payload reads during parse | **0** |

The zero payload reads during parse is asserted by the test suite via a counting
`ByteReader` — it is the property the whole design exists to guarantee.

## Building

Requires CMake ≥ 3.24, a C++20 compiler, and (for the GUI) OpenGL + a windowing
system. All other dependencies are fetched and pinned via CMake `FetchContent`
(GLFW, Dear ImGui docking, nlohmann/json, miniz, stb, tinyfiledialogs, doctest).

```sh
# Full app (Release, -O2 + LTO)
cmake --preset release
cmake --build --preset release
./build/release/netvis path/to/model.onnx

# Headless core + tests only (no OpenGL/GLFW needed — CI-friendly)
cmake --preset core-only
cmake --build --preset core-only
ctest --preset core-only
```

Open a model via the `File → Open` dialog, by dragging it onto the window, or by
passing it as a CLI argument.

## Architecture

Three strictly separated layers; the view never touches parsers directly.

```
View (Dear ImGui)   — window, dockspace, graph canvas, panels, dialogs
      │  talks only to ModelSession
Engine              — ModelSession, LayoutEngine, CollapseTree, SearchIndex,
      │                ShapeInference, TensorStats, LayoutCache, JobSystem
Parsers → ir::Model — onnx/ tflite/ safetensors/ gguf/ pytorch/
```

- `core/` — `MappedFile`, bounds-checked `ByteReader`, `StringArena` (interned
  strings + 32-bit handles), `JobSystem` (thread pool + main-thread completion
  queue), `SmallVec`, FNV-1a hashing, `Result<T>` error handling.
- `ir::Model` — cache-friendly struct-of-arrays: POD nodes/values with 32-bit
  indices and interned string handles.

See `CONTRACTS.md` for the frozen interface contracts and `DECISIONS.md` for the
rationale behind every performance-relevant choice.

## Testing

`tools/gen_fixtures.py` (Python 3 stdlib only) hand-encodes tiny fixture models for
each format. The doctest suite covers each parser (asserting zero payload reads),
format detection, layout determinism, the pickle VM opcode set + allowlist
rejection, NPY export round-trip, and truncation resilience.

```sh
python3 tools/gen_fixtures.py tests/fixtures
ctest --preset core-only
```

## Non-goals (v1)

No model editing, no inference/execution, no dequantization of GGUF quant blocks,
no TorchScript/FX graph reconstruction, no plugin system, no web build.
