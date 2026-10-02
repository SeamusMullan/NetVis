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

- **Memory cap** applied before load; a module declaring huge initial pages cannot
  force a giant allocation.
- **Fuel/step cap** via a strong `m3_Yield` override + a loop-backedge patch: a
  runaway loop or recursion is trapped, the plugin disabled, the app survives.
- **ABI probe budget**: loading a plugin runs a little guest code (see **ABI probe
  budget** below), so that load gets its own, much smaller step budget.
- **Null-guard** offset 8 — offsets `< NV_NULL_GUARD_OFF` are treated as null.
- Exports are validated to be `() -> i32` (or `() -> ()`) before the call; a
  wrong-signature or missing export is a clean load error, never UB.
- One process-wide mutex serializes all engine load/link/call (the wasm3
  `IM3Environment` is shared mutable state; fuel is thread-local).

`NV_MAX_MEMORY_PAGES` (256 pages = 16 MiB) and `NV_MAX_STEPS` (200,000,000) are the
**ceiling** a plugin may rely on. Which facet gets how much of it:

| Facet entry point | Memory | Steps |
|---|---|---|
| Parser `netvis_parse`, pass `run` | `NV_MAX_MEMORY_PAGES` | `NV_MAX_STEPS` |
| Parser `netvis_can_parse` (the sniff) | 4 pages | 500,000 |
| Op handler (every `netvis_op_*` call, memoized per op-type) | 64 pages | 2,000,000 |
| **ABI probe** (loading a plugin: start section + `netvis_<facet>_abi_version`) | the facet's own (op 64, parser 4, pass 256) | **50,000** |

A facet may be granted less than the ceiling, never more. The marshalling caps
(`NV_MAX_RANK`, `NV_MAX_NODE_IO`, `NV_MAX_ATTR_LEN`, `NV_MAX_INTERN_LEN`,
`NV_ERR_MSG_MAX`, `NV_SNIFF_HEAD/TAIL`, `NV_READ_CHUNK_CAP`) apply as named in the
facet sections below. The host does not keep its own copies of any of these: it takes
them from the header (`src/engine/plugin/wasm/SdkCaps.h`) and `static_assert`s every
per-facet budget against the ceiling, so a cap cannot be edited in one place and left
behind in the other.

### ABI probe budget

Before anything is registered, the host loads every WASM plugin once to ask which ABI
it was built for: it instantiates the module, which runs its **start section** (the
wasm start function, if it has one), and then calls its `netvis_<facet>_abi_version`
export. That load happens wherever plugins are loaded — at start-up and each time a
plugin is toggled in the Plugins panel — so it must not be able to hold the app up. It
therefore runs under its own step budget, `kAbiProbeStepBudget` =
**50,000 steps**, far below the parse/run and sniff budgets above. A step is one
function call or one loop iteration.

**The module's start section and its ABI-version export must each finish within that
budget.** A version read is a constant function and a toolchain-built start section is
a handful of calls, so neither comes close; a module whose start section loops, or that
does real work there, is refused — not registered — with this reason in the plugin
load diagnostics (the Plugins panel's error text):

> module exceeded the ABI probe budget (50000 steps): its start section and
> `netvis_op_abi_version` must each finish within it

Put real initialisation in the entry points (`netvis_op_*`, `netvis_parse`, `run`),
which run under the facet budgets above, not in the start section. (A pass that does
not export the optional `netvis_pass_abi_version` has nothing for the probe to call, so
its start section first runs in `run()`, under the pass's own budget.)

## Compatibility promise (ABI v1)

All three plugin ABIs — op handler, parser, pass — are at **version 1** and frozen.
The promise to a plugin that is already built and shipped:

> A plugin compiled against ABI v1 keeps loading and keeps answering, unchanged, on
> every later NetVis release that still declares ABI v1. When NetVis can no longer
> honour that, it bumps the version and **refuses** the plugin — it never loads one
> it cannot honour and hopes.

### What is frozen

Everything in [`plugins/sdk/abi-v1-surface.txt`](../plugins/sdk/abi-v1-surface.txt):

- every macro name and value, every enumerator and its number, every typedef (name
  *and* underlying type), the wire struct's members in order, and the guest helpers;
- every **host import**, as `(facet, module, name, wasm signature)` plus its C
  prototype name. The signature is part of the import: wasm3 refuses to bind a guest
  import whose signature differs from the host's, so a changed signature is an import
  that stops existing, not one that changes slightly;
- every **guest export** the host looks up by name (`netvis_op_*`,
  `netvis_can_parse`, `netvis_parse`, `netvis_parser_abi_version`,
  `netvis_pass_abi_version`, `run`), per facet, all `() -> i32`.

`tests/test_plugin_abi_freeze.cpp` re-derives that inventory on every test run — the
header from `netvis_plugin.h`, and the host's side from its own `m3_LinkRawFunctionEx`
tables and `call_i32` / `invoke_facet` call sites (comments stripped, so a
commented-out link is not a link) — and fails if any of it disagrees with the freeze
file or the two sides disagree with each other. Changing the header or the host's
link tables is therefore only possible as a deliberate, reviewed edit to the freeze
file. The test also feeds mutated headers and host sources to its own extractors, so
the guard is known to notice a changed signature, a renamed export, a reordered struct
member or a hex enumerator, and to ignore a reworded comment.

Imports are scoped **per facet**. The parser and the pass both live in module
`"netvis"`, but the parser's adapter links `host_file_len`…`host_record_tensor` and
the pass's links `host_node_count`, `host_total_flops`, `host_total_params` and
`host_emit_metric`; each set is only available to its own facet. A parser module that
imports a pass function is refused at load (see **Link** below), not run until it
calls it.

### Changes that keep ABI v1

| Change | Why an existing plugin is unaffected |
|---|---|
| Adding a host import | An ABI v1 plugin does not import it. A plugin that does simply will not load on an older host — it is refused with the import named (**Link**, below), the author's choice made at build time. |
| Adding an optional export the host probes for | Absent export → the host falls back exactly as it does today (honest-unknown). |
| Adding a defaulted virtual to a C++ handler interface | `OpHandler::color/flops/infer_shape` are already defaulted; a handler that does not implement a newly added one is not broken by it. |
| Raising a cap (`NV_MAX_*`) | A plugin built against the old header still marshals within the old, smaller bound. A plugin built against the *raised* header and run on an older host is not refused — it declares v1 — but the older host never truncates or guesses: the op facet records an output whose rank exceeds its cap with its dtype and no shape (the shape stays unknown), and the parser facet's calls reject the over-cap value (`host_add_value` / `host_add_node` return -1; `host_record_tensor` records the tensor with an unknown shape). |
| Anything behind `#if defined(__wasm__)` that is not a name, a signature or a number — the bump allocator's body, an attribute spelling | Guest-side convenience, not wire format. Only the helper names and `NV_ARENA_BYTES` are frozen. |

Each of these still edits the freeze file. That is the point: the edit is where
someone asks "does this need a bump?", and the table above is the answer.

### Changes that force a version bump

- Renaming or removing a host import, or changing its signature (or moving it to a
  different facet).
- Renaming or removing a guest export the host calls, or changing its signature, or
  turning an optional export into a required one.
- Renumbering an enumerator. Note that `nv_dtype_t` and `nv_category_t` byte-match
  `ir::DType` and `OpCategory`, whose honest-unknown sentinels (`NV_DT_UNKNOWN` = 15,
  `NV_CAT_OTHER` = 14) are **last** — so adding a dtype or a category renumbers the
  sentinel and is a break, not an addition.
- Lowering a cap, or changing what one means.
- Changing the layout of a struct a v1 import or export passes by pointer. v1 has
  none: `nv_tensor_hdr_t` documents the canonical order of a tensor record and is
  **not** read by the host (`host_record_tensor` takes scalars). It is frozen anyway —
  member list and order in the freeze file, size asserted in the header, every offset
  asserted in `tests/test_sdk_abi.cpp` — so that a later import that does pass it
  inherits a layout that never moved. (The header's own `static_assert` runs on any
  compiler with `static_assert`, including MSVC, which reports `__cplusplus` as
  199711L; there is no wasm32 toolchain in NetVis CI, so the wasm32 side is checked
  by plugin authors' builds.)
- Changing an entry point's status convention (`NV_STATUS_OK` / `NV_STATUS_ABSTAIN`).

NetVis supports **one ABI version at a time**. A bump means plugin authors rebuild
against the new header; there is no shim layer, because a shim that mistranslates is
worse than a refusal that is visible.

### How a mismatch is refused

A mismatch is refused *cleanly*: nothing half-registers, no partial answer reaches
the view, and the built-in result still stands. There are four gates, in the order a
plugin meets them:

1. **Manifest** — `plugin.json`'s `api_version` must be **present**, an **unsigned
   integer**, and **exactly** the host's version, compared at full 64-bit width. A
   missing key, a string, a float, a negative number, or `4294967297` (which would
   narrow to 1) rejects the whole file with a diagnostic before anything is
   constructed. One reader serves the declarative loader and both WASM loaders
   (`engine/plugin/AbiGate.h`).
2. **Module** — building a WASM adapter loads the module once and calls its
   `netvis_<facet>_abi_version` export. The adapter's `api_version()` reports **the
   module's declared version** — not the host's constant — so the Registry check below
   is a real gate for WASM plugins. The loaders (`load_wasm_op_plugin`,
   `load_wasm_parser_plugin`) read the same probe before registering anything and
   refuse with the reason. The policy per facet:
   - **Op handler, parser**: the export is required. A different value *or* a missing
     export is a refusal (`WasmOpDiag::abi_mismatch`; the parser will not claim a
     file).
   - **Pass**: the export is *optional*. The pass facet shipped in v0.6.0 before the
     export existed, and the promise above forbids refusing a v1 plugin for lacking
     something v1 never asked for, so a pass with no `netvis_pass_abi_version` is an
     ABI v1 pass. A pass that *does* export it must return the host's version; any
     other value is refused and `run()` yields no metrics — not even one the export
     itself emitted while the module was being judged. "Does not export it" is read
     off the module's export names: an export that is present but fails (it traps, it
     will not compile, it overruns the probe budget) is refused, never taken for the
     absent-export case.

   The whole probe runs under the **ABI probe budget** above; a module that overruns
   it is refused with that reason.
3. **Link** — after linking, every function import the module declares must be bound.
   wasm3 leaves an import unbound when the host has no function of that name *and*
   when the host has one but the guest declared a different signature (it reports
   "function signature mismatch" and links nothing), and a call to an unbound import
   only fails later, in whichever entry point reaches it, which reads as a plugin that
   merely abstains. The module is refused instead, with the import named. This is also
   what makes a plugin built against a *newer* v1 header (a host import added within
   v1) a visible refusal on an older v1 host.
4. **Registry** — `register_op_handler` / `register_parser` / `register_pass` each
   re-check `api_version()` and drop the plugin rather than insert it, so a loader
   that skipped the earlier gates still cannot get a wrong-ABI plugin into the table.
   Every later invocation re-checks the module gate and the link gate too.

Why this matters beyond tidiness: an op handler registered with `override: true`
replaces the built-in answer for that op. A module that can only abstain would turn,
say, every `MatMul`'s known FLOP count into *unknown* while the Plugins panel listed
the plugin as loaded. Refusal at load keeps the built-in result and says why.

`tests/test_plugin_abi_freeze.cpp` covers, each next to a matching-plugin control so
that a gate which refuses everything cannot pass:

- **Manifest**: all three loaders, against eight bad `api_version` shapes (wrong
  version, zero, missing, string, float, negative, boolean, wrapping integer).
- **Module**: op handler and parser — a future version and a missing export; pass — a
  future version, a declared v1, and a pre-negotiation module with no export (which
  must keep running). A WASM parser with a missing export is covered by its own
  fixture.
- **ABI probe budget**: for each facet, a start section that never ends, an ABI export
  that never returns, and a start section that finishes but needs far more steps than
  the budget — each refused with the budget named (and, for the op handler, shown
  through `discover_and_load_plugins`), next to a control whose short start section is
  admitted. A pass's ABI export that emits a metric before returning an unsupported
  version leaves `run()` with no metrics, and one whose export is present but broken
  is refused rather than run as v1.
- **Link**: op handler — a real import name with the wrong signature, an import the
  host has no function for, and `op_input_const_ints` declared with the header's
  signature (which the host used to reject). The parser and pass adapters run the same
  check but have no dedicated fixtures.
- **Registry**: all three kinds with stubs, and the real op/parser/pass adapters.
- **End to end**: `load_wasm_op_plugin` → `resolve_op` → `compute_cost` for an
  override of `MatMul` (built-in FLOPs preserved on refusal, taken over by the control),
  and `discover_and_load_plugins`, which is what the Plugins panel reads.

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
`Other`) and forces opaque color, so no invalid value reaches the view. A shape is
never clamped: `op_set_output_shape` with a rank above `NV_MAX_RANK` records that
output with its dtype and no shape (the shape stays unknown) instead of recording a
truncated one as known. Per §A.1 a WASM handler runs only on the worker cost/shape
pass (memoized per op-type); the render thread reads the cached scalar — the sandbox
is never entered per frame.

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
`host_total_params`; emits scalars via `host_emit_metric`. May export
`netvis_pass_abi_version` (see **How a mismatch is refused**); a pass that does not is
ABI v1, while one that does and fails it is refused.

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
real-world drift source; keep them in sync with `netvis_plugin.h` by hand. A wrong
signature in a port is now refused when an op-handler or parser plugin loads
(**Link**), naming the import, rather than showing up as a plugin that abstains.
