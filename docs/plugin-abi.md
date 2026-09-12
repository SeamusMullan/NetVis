# NetVis Plugin ABI (v1, frozen)

NetVis loads plugins that extend op coverage, add file-format parsers, and compute
analysis passes — without a rebuild. Three plugin kinds, two trust tiers:

| Kind | What it does | Trust | Default |
|---|---|---|---|
| **Declarative** (`plugin.json` + expression DSL) | op category/color/FLOPs/shape | safe by construction | **enabled** |
| **WASM** (sandboxed `.wasm`) | op handler, file parser, or analysis pass | sandboxed arbitrary code | **disabled** (per-plugin opt-in) |

The single source of truth for the WASM wire ABI is the freestanding C header
[`plugins/sdk/netvis_plugin.h`](../plugins/sdk/netvis_plugin.h). `tests/test_sdk_abi.cpp`
static-asserts it against the host C++ contracts, so the shipped header can never
silently drift from the host, and `tests/test_plugin_abi_freeze.cpp` holds its
*surface* to [`plugins/sdk/abi-v1-surface.txt`](../plugins/sdk/abi-v1-surface.txt) —
see **Compatibility promise** below.

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

## Compatibility promise (ABI v1)

All three plugin ABIs — op handler, parser, pass — are at **version 1** and frozen.
The promise to a plugin that is already built and shipped:

> A plugin compiled against ABI v1 keeps loading and keeps answering, unchanged, on
> every later NetVis release that still declares ABI v1. When NetVis can no longer
> honour that, it bumps the version and **refuses** the plugin — it never loads one
> it cannot honour and hopes.

### What is frozen

Everything in [`plugins/sdk/abi-v1-surface.txt`](../plugins/sdk/abi-v1-surface.txt):
every macro name and value, every enumerator and its number, every host import's
module and name, and the typedefs. `tests/test_plugin_abi_freeze.cpp` re-derives that
inventory from `netvis_plugin.h` on every test run and fails if the two disagree — so
changing the header is only possible as a deliberate, reviewed edit to the freeze
file. The same test reads the host's own `m3_LinkRawFunctionEx` tables and requires
them to name exactly the header's import set, so an import can never exist on one
side only (a guest importing a name the host does not link fails to instantiate).

### Changes that keep ABI v1

| Change | Why an existing plugin is unaffected |
|---|---|
| Adding a host import | An ABI v1 plugin does not import it. A plugin that does simply will not run on an older host — the author's choice, made at build time. |
| Adding an optional export the host probes for | Absent export → the host falls back exactly as it does today (honest-unknown). |
| Adding a defaulted virtual to a C++ handler interface | `OpHandler::color/flops/infer_shape` are already defaulted; a handler that does not implement a newly added one is not broken by it. |
| Raising a cap (`NV_MAX_*`) | A plugin built against the old header still marshals within the old, smaller bound. |
| Anything behind `#if defined(__wasm__)` that is not a name or a number — the bump allocator's body, an attribute spelling | Guest-side convenience, not wire format. Only `NV_ARENA_BYTES` is frozen. |

Each of these still edits the freeze file. That is the point: the edit is where
someone asks "does this need a bump?", and the table above is the answer.

### Changes that force a version bump

- Renaming or removing a host import, or changing its signature.
- Renumbering an enumerator. Note that `nv_dtype_t` and `nv_category_t` byte-match
  `ir::DType` and `OpCategory`, whose honest-unknown sentinels (`NV_DT_UNKNOWN` = 15,
  `NV_CAT_OTHER` = 14) are **last** — so adding a dtype or a category renumbers the
  sentinel and is a break, not an addition.
- Lowering a cap, or changing what one means.
- Changing a wire struct's layout (`nv_tensor_hdr_t` is `_Static_assert`ed at 24
  bytes in the header itself, on both the host and the wasm32 toolchain).
- Changing an entry point's status convention (`NV_STATUS_OK` / `NV_STATUS_ABSTAIN`).

NetVis supports **one ABI version at a time**. A bump means plugin authors rebuild
against the new header; there is no shim layer, because a shim that mistranslates is
worse than a refusal that is visible.

### How a mismatch is refused

A mismatch is refused *cleanly*: nothing half-registers, no partial answer reaches
the view, and the built-in result still stands.

- **Manifest** — `plugin.json`'s `api_version` is checked before anything is
  constructed (`declarative/Manifest.cpp`, `wasm/WasmOpHandler.cpp`,
  `wasm/WasmParser.cpp`). A mismatch rejects the whole file with a diagnostic.
- **Module** — a WASM module must export `netvis_<facet>_abi_version` returning the
  host's version. A wrong value *or* a missing export sets `WasmOpDiag::abi_mismatch`
  and the handler answers honest-unknown; a parser refuses to claim the file.
- **Registry** — `register_op_handler` / `register_parser` / `register_pass` each
  re-check `api_version()` and drop the plugin rather than insert it, so even a
  loader that skipped the earlier gates cannot get a wrong-ABI plugin into the table.

`tests/test_plugin_abi_freeze.cpp` covers all three, in both directions (a plugin
from the future, and one so old it declares no version at all), and asserts that a
matching plugin still answers — so the gate cannot pass by refusing everything.

## Trust

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
