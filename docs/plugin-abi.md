# NetVis Plugin ABI (v0.7.0)

NetVis loads plugins that extend op coverage, add file-format parsers, and compute
analysis passes — without a rebuild. Three plugin kinds, two trust tiers:

| Kind | What it does | Trust | Default |
|---|---|---|---|
| **Declarative** (`plugin.json` + expression DSL) | op category/color/FLOPs/shape | safe by construction | **enabled** |
| **WASM** (sandboxed `.wasm`) | op handler, file parser, or analysis pass | sandboxed arbitrary code | **disabled** (per-plugin opt-in) |

The single source of truth for the WASM wire ABI is the freestanding C header
[`plugins/sdk/netvis_plugin.h`](../plugins/sdk/netvis_plugin.h). `tests/test_sdk_abi.cpp`
static-asserts it against the host C++ contracts, so the shipped header can never
silently drift from the host.

## Zero-payload thesis (per facet — stated honestly)

NetVis never eagerly decodes weights; a plugin must not be able to change that.

- **Op handler / pass**: a property of the **import set** — no host import returns a
  decoded weight buffer. An op handler reads shape/dtype/attr metadata; a pass reads
  structure + `CostReport` scalars. Neither can obtain payload bytes.
- **Parser**: NOT `counter == 0` (there *is* a byte-read import). It is a
  **bounded-window + host-marked** property: `host_read_range` is confined to an
  up-front sniff window (head `NV_SNIFF_HEAD` + tail `NV_SNIFF_TAIL`), the host reads
  through a *marked* `ByteReader`, and any read overlapping a recorded tensor range
  is rejected. Names/metadata are `(offset,len)` ranges the **host** reads from that
  window (never arbitrary guest memory); at parse-end the host re-validates that no
  rendered string overlaps a recorded tensor range and rejects the whole model if
  one does. The witness is "structural reads ≤ the declared window", not zero.

## Sandbox (WASM)

- **Memory cap** applied before load (`NV_MAX_MEMORY_PAGES`); a module declaring huge
  initial pages cannot force a giant allocation.
- **Fuel/step cap** (`NV_MAX_STEPS`) via a strong `m3_Yield` override + a loop-backedge
  patch: a runaway loop or recursion is trapped, the plugin disabled, the app survives.
- **Null-guard** offset 8 — offsets `< NV_NULL_GUARD_OFF` are treated as null.
- Exports are validated to be `() -> i32` (or `() -> ()`) before the call; a
  wrong-signature or missing export is a clean load error, never UB.
- One process-wide mutex serializes all engine load/link/call (the wasm3
  `IM3Environment` is shared mutable state; fuel is thread-local).

## Versioning + trust

- Each plugin declares `api_version`; a WASM module also exports
  `netvis_<facet>_abi_version`. A mismatch → the plugin is **refused**, never
  half-registered.
- Declarative plugins auto-load. WASM plugins are **disabled by default**; enabling
  one requires a one-time confirm dialog and is then persisted per-plugin in
  `view_prefs.json` under `"plugins"` (keyed by the discovery-subdir name). The gate
  is enforced by **table membership** — a disabled plugin is structurally absent from
  the Registry, so its `can_parse` (which executes code) never runs.

## Op-handler facet (module `"netvis_op"`)

Exports (`() -> i32` status; `0` = answered, nonzero = abstain → honest-unknown):
`netvis_op_abi_version`, `netvis_op_category`, `netvis_op_flops`,
`netvis_op_infer_shape`, `netvis_op_color` (optional).

Reads the current op via the `netvis_op` imports (counts, per-slot rank/dims/dtype,
initializer elem-count/byte-len/dtype, attrs, the one guarded `op_input_const_ints`).
Pushes its verdict via out-imports (`op_set_category/flops/output_shape/color`). The
host **clamps** a returned category to a valid `OpCategory` (a hostile `9999` →
`Other`) and forces opaque color, so no invalid value reaches the view. Per §A.1 a
WASM handler runs only on the worker cost/shape pass (memoized per op-type); the
render thread reads the cached scalar — the sandbox is never entered per frame.

## Parser facet (module `"netvis"`)

Exports: `netvis_parser_abi_version`, `netvis_can_parse` (`() -> i32`, 1 = claims),
`netvis_parse` (`() -> i32`, 0 = ok). Builds the model with append-only commands
(`host_begin_graph`, `host_add_value`, `host_add_node`, `host_add_attr_*`,
`host_record_tensor`, `host_set_graph_io`, `host_set_model_info`, `host_set_metadata`,
`host_set_error`). Every guest count is capped (`NV_MAX_RANK`, `NV_MAX_NODE_IO`,
`NV_MAX_ATTR_LEN`, …) and memory-bounds-checked. A built-in format always wins; a
plugin parser is tried only for a file no built-in claimed.

## Pass facet (module `"netvis"`, shipped v0.6.0)

Export `run` (`() -> i32`). Reads `host_node_count` / `host_total_flops` /
`host_total_params`; emits scalars via `host_emit_metric`.

## Worked examples

- [`plugins/examples/toy_parser/`](../plugins/examples/toy_parser/) — a WASM parser
  for a toy `NVTOY1` format: sniffs the magic in the window, records each weight by
  `(offset,len)`, never reads a payload byte.
- [`plugins/examples/attn_pass/`](../plugins/examples/attn_pass/) — a WASM pass
  emitting `flops_per_node` from the `CostReport`.
- [`plugins/examples/`](../plugins/examples/) declarative examples (`my-ops`,
  `linear-layer`, …) — see that directory's `README.md`.

Each WASM example ships source + `build.sh` (needs `clang --target=wasm32` +
`wasm-ld`). NetVis CI has no wasm toolchain, so the test suite uses hand-encoded
fixtures (`tools/gen_fixtures.py`) rather than building the examples.

### Porting to Rust / Zig

The imports/exports are a plain C ABI; the SDK header's signatures translate directly
(offsets/lengths are `i64`, never C `long`). Hand-translated Rust/Zig decls are
**unverified** by the drift guard (which only checks the C header) — the most likely
real-world drift source; keep them in sync with `netvis_plugin.h` by hand.
