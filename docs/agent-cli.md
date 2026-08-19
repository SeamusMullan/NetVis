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

## Relation to `--report`

`netvis --report <model>` predates `query` and is kept as-is for compatibility;
`query summary` embeds the identical report document. Both share one loading
pipeline (`load_model_headless`), so they can never disagree about how a model
is opened.
