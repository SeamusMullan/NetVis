# The agent query CLI

`netvis query <verb> <model> [target] [--flags]` answers one structural question
about a model as a single line of JSON on stdout, then exits. It exists so
automated tooling — coding agents, scripts, CI — can interrogate a model the way
the GUI does, with no window, no display, and no long-lived process.

## Design invariants

- **Stateless.** Opening a model is milliseconds (mmap + structural parse — the
  zero-payload thesis), so every invocation re-opens the file. There is no
  daemon and no session; each answer is reproducible from the file alone.
- **Zero payload reads**, except the one verb whose job is reading a payload
  (`tensor`). The test suite asserts this through the `ByteReader` counter.
- **One JSON object per success**, on one line, always carrying the envelope
  `{"schema": "netvis.query.v1", "verb": ..., "model": ..., "format": ...}`.
  The schema string is bumped on any payload-shape change; gate on it.
- **Errors are loud.** Any failure — unknown verb, unknown flag, missing model,
  unresolvable node or tensor — prints plain text to stderr and exits 1, and
  stdout stays silent. A consumer never parses a half-written object. Estimates
  stay honest the same way they do in the GUI: unknown FLOPs and dynamic
  dimensions are `null` / `-1`, never fabricated.
- **Shared engine.** Every verb is a JSON projection over the same headless
  calls the GUI makes (`parse_model`, shape inference, `compute_cost`,
  `GraphAdjacency`, `SearchIndex`, `compute_tensor_stats`, `diff_models`), so
  the CLI cannot disagree with what the GUI shows.

Flags are `--key value` or `--key=value`. Where a verb takes a node, `<target>`
is an exact node name or `#<index>`; discovery belongs to `search`.

## Verbs

| Verb | Question it answers |
|---|---|
| `summary <model>` | What is this model? (embeds the `netvis.report.v1` report: format, graphs, totals, quant profile, roofline) |
| `io <model> [--graph N]` | What are the declared graph inputs/outputs, with dtype and shape? |
| `nodes <model> [--op T] [--contains S] [--limit N] [--offset N] [--graph N]` | Which nodes are here? (paged; `total_matched` is always exact) |
| `node <model> <name\|#i> [--graph N]` | Everything about one node: attributes, input/output values with shapes, per-node cost, direct predecessors/successors |
| `tensors <model> [--sort name\|bytes\|params] [--limit N] [--offset N] [--graph N]` | Which weights exist, how big, what dtype? (initializers, or `flat_tensors` for GGUF/SafeTensors-style models) |
| `tensor <model> <name> [--graph N]` | What is in this weight? min/max/mean/std, zero and NaN/Inf counts, 64-bucket histogram — **the one verb that reads payload bytes** |
| `search <model> <query> [--limit N]` | Find things by fuzzy name or field query — the GUI search bar's exact syntax: `op:`, `name:`, `dtype:`, `shape:`, `params:` (with `<`/`>` and K/M/G suffixes), `*` globs |
| `neighbors <model> <name\|#i> [--dir in\|out\|both] [--hops N] [--cap N] [--graph N]` | What feeds / consumes this node, N hops out? |
| `cost <model> [--by flops\|params\|weight_bytes\|act_bytes\|intensity] [--limit N] [--graph N]` | Where is the cost? (the analyzer's per-node ranking) |
| `diff <model-a> <model-b> [--match name\|topology] [--limit N] [--graph N]` | What changed between two models? (exact added/removed/changed counts plus capped change lists) |

## Worked examples

What is this model, and is anything obviously wrong with its quantization?

```sh
netvis query summary model.onnx | jq '.report.cost | {total_flops, effective_bits_per_param, dtype_usage}'
```

Where does the compute go?

```sh
netvis query cost model.onnx --by flops --limit 10
```

Find the attention blocks, then inspect one:

```sh
netvis query search model.onnx "op:attention" --limit 5
netvis query node model.onnx "attn_0" | jq '{attributes, inputs, cost}'
```

Trace what feeds a suspicious output:

```sh
netvis query neighbors model.onnx "#412" --dir in --hops 3
```

Is this weight dead or exploded? (the one payload read)

```sh
netvis query tensor model.onnx "decoder.layers.7.mlp.w2" | jq .stats
```

Did quantization change the graph, or only the weights?

```sh
netvis query diff model_fp32.onnx model_int8.onnx --match name
```

## Headless binary

The `core-only` preset builds a standalone **`netvis_query`** binary alongside
`netvis_bench` — netvis_core only, no GLFW/OpenGL/ImGui — so the CLI runs on CI
runners, remote debug boxes and agent sandboxes with no display stack. It
accepts the verb bare (`netvis_query nodes model.onnx`) or behind the GUI
binary's `query` word, so scripts can swap one binary name for the other.

```sh
cmake --preset core-only && cmake --build --preset core-only
./build/core/netvis_query summary model.onnx
```

Tensor listings also carry `dtype_label`, the format-native type name
(`"i4"`, `"q4_k"`, ...) recorded by the parser. Sub-byte and block-quantized
types map to a generic IR dtype, so when the two differ the label is the honest
answer to "what is this tensor really".

## MCP server

`netvis_mcp` serves the same ten answers as MCP tools (`netvis_summary`,
`netvis_nodes`, `netvis_tensor`, ...) over stdio JSON-RPC, for clients that
speak the Model Context Protocol. It is a pure adapter: every tool call is
translated into the exact `netvis query` argument vector and dispatched through
the same code, so the tools and the CLI cannot drift. Point a client at the
dedicated binary, or at `netvis mcp` / `netvis_query mcp` — all three start
the identical loop.

The server's lifetime is the client session (stdio servers are spawned and
owned by their client), and within a session an LRU cache keyed by path and
revalidated by file size and mtime keeps the last few parsed models warm — a
burst of queries against one multi-gigabyte file parses it once. The cap is
small by design (structure for a very large graph can reach ~100 MB) and can
be overridden with `NETVIS_MCP_CACHE_MODELS`. Payload bytes stay on disk;
`netvis_tensor` remains the only reader.

```json
{ "mcpServers": { "netvis": { "command": "/path/to/netvis_mcp" } } }
```

Within a session, derived analyses (the cost report, the search index, the
adjacency) are memoized per cached model, so warm tool calls pay only their own
query. Numbers come from the footprint harness, not from claims — reproduce
them with:

```sh
./build/core/netvis_bench --bench-mcp     # JSON to stdout
```

Same machine class as the README's engine table, median of 5, on the synthetic
chain ladder:

| | 1k nodes | 10k nodes | 100k nodes |
|---|---|---|---|
| cold call (load + analyze) | 0.95 ms | 9.7 ms | 122 ms |
| warm `nodes` | 0.012 ms | 0.030 ms | 0.19 ms |
| warm `search` | 0.041 ms | 0.28 ms | 3.0 ms |
| warm `cost` | 0.061 ms | 0.56 ms | 6.2 ms |
| warm `neighbors` | 0.010 ms | 0.011 ms | 0.014 ms |
| `ping` / `tools/list` | 1.3 µs / 50 µs | — | — |

The footprint case drives eight distinct models through the default cap-4
cache and reports RSS at cap and after the churn — the growth stays bounded by
the cap, not by how many models the session touches.

The repo is also installable as a **Claude Code plugin**: it doubles as its
own single-plugin marketplace, so

```
/plugin marketplace add SeamusMullan/NetVis
/plugin install netvis@netvis
```

registers the server for every session — including on a machine that has
never built or installed NetVis. The plugin launches a small Node bootstrap
(`tools/netvis-mcp-bootstrap.mjs`, stdlib only) that prefers a binary you
already have: an explicit `NETVIS_MCP` path, a build inside the checkout,
then `netvis_mcp` on PATH. When none exists it downloads the standalone
binary for the current platform from the latest GitHub Release, verifies it
against the release's `SHA256SUMS`, caches it in the plugin data directory,
and launches it. The cache survives plugin updates and is removed on
uninstall; later sessions start straight from it with no network touch.
`NETVIS_MCP_VERSION=X.Y.Z` pins the downloaded version, `NETVIS_MCP_UPDATE=1`
re-downloads, and `NETVIS_MCP_REPO=owner/name` points a fork at its own
releases.

The per-OS installers still ship the binary (the Windows installer offers the
PATH entry), and a PATH install always wins over a download; from source it is
one build and one copy:

```sh
cmake --preset core-only && cmake --build --preset core-only
# Linux/macOS                          # Windows (PowerShell)
cp build/core/netvis_mcp ~/.local/bin  # copy build\core\netvis_mcp.exe into a PATH dir
```

`/mcp` shows the connection status.

When the server is not running, it costs nothing: MCP state lives only inside
the server object, the tool table is built lazily on first use, and none of
the engine paths the performance table gates were touched — the engine ladder
reproduces the README numbers unchanged on this branch.

## Relation to `--report`

`netvis --report <model>` predates `query` and is kept as-is for compatibility;
`query summary` embeds the identical report document. Both share one loading
pipeline (`load_model_headless`), so they can never disagree about how a model
is opened.
