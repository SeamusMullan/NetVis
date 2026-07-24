# attn_pass — example WASM pass plugin

A minimal NetVis WASM **analysis pass** (the pass facet shipped in v0.6.0). Reads
model structure + the `CostReport` scalars and emits a named metric — it cannot read
a weight buffer (the host exposes no such import).

Emits `flops_per_node = total_flops / node_count` (known when both are available).

## Build

```sh
./build.sh        # needs clang with a wasm32 target + wasm-ld (LLVM 8+)
```

Produces `attn_pass.wasm`. Drop it beside a `plugin.json` with:

```json
{ "api_version": 1, "name": "attn-pass", "wasm": "attn_pass.wasm", "pass": true }
```

WASM plugins are **disabled by default** — enable in the Plugins panel. The emitted
metric appears in the analyzer's pass-metrics list.

## Notes

- The metric name lives in guest memory; the host copies it out bounds-checked, so
  no pointer to host state crosses the sandbox.
- A trap or fuel-exhaustion mid-run simply yields whatever metrics were emitted; the
  app always survives.
