# NetVis plugin examples

NetVis loads **declarative plugins** at startup: a directory containing a
`plugin.json` describes ops the built-ins don't know (or overrides ones they do),
giving each a **color category**, a **FLOP formula**, and an **output-shape rule** —
with no compiling, no toolchain, and no way to crash or slow the app.

> These examples target the **declarative** backend, which ships and works in
> v0.6.1. The WASM backend (for logic beyond the integer DSL) is a documented
> follow-up — see `docs/v0.6.x-followups-plan.md`.

## Installing a plugin

Copy an example directory into NetVis's plugin directory, then restart NetVis:

| OS | Plugin directory |
|---|---|
| Linux | `$XDG_CONFIG_HOME/netvis/plugins/` (else `~/.config/netvis/plugins/`) |
| macOS | `~/Library/Application Support/NetVis/plugins/` |
| Windows | `%LOCALAPPDATA%\NetVis\plugins\` |

Each plugin is its own directory with one `plugin.json`:

```
<plugin-dir>/
  my-ops/
    plugin.json
```

Open **View → Plugins** to see what loaded, each op's category, and whether it
overrides a built-in. A rejected plugin (bad JSON, unknown category, `api_version`
mismatch) is listed with its error — it never affects the rest of the app.

## The manifest

```jsonc
{
  "api_version": 1,               // must match the host (v0.6.1 = 1), else rejected
  "name": "my-ops",               // shown in the Plugins panel
  "author": "you",                // optional
  "ops": [
    {
      "name": "MyOp",             // op_type this handles (normalized: lowercased,
                                  //   domain-stripped — "com.x.MyOp" -> "myop")
      "domain": "com.example",    // optional
      "category": "MatMul",       // REQUIRED, must be a known category (list below)
      "color": "#RRGGBB",         // optional; else the category's default color
      "override": true,           // REQUIRED to shadow a BUILT-IN op (safety)
      "vars": { "K": "in0.shape[-1]" },   // optional named subexpressions
      "flops": "2 * O * K",       // optional; omitted/empty => FLOPs unknown (honest)
      "shape": { "out0": ["in0.shape[0]", "in1.shape[-1]"] }  // optional per-output dims
    }
  ]
}
```

**Categories** (drive node color + the generic-formula fallback):
`Conv`, `MatMul`, `Activation`, `Norm`, `Pool`, `Elementwise`, `Shape`, `Reduce`,
`Tensor`, `ControlFlow`, `IO`, `Attention`, `Recurrent`, `Quantize`, `Other`.

## The expression DSL

Pure **integer** arithmetic over a node's resolved shapes and attributes. It is
total and side-effect-free: a formula can only produce a number or an *honest
unknown* — never a crash, a hang, or a fabricated value.

**Terms**
- `inK.shape[i]` / `outK.shape[i]` — dim `i` of input/output slot `K`. Negative `i`
  counts from the end (`in0.shape[-1]` = last dim). An unresolved/dynamic dim ⇒ the
  whole expression is unknown.
- `inK.rank` / `outK.rank` — number of dims.
- `O` — element count of output 0 (product of its dims). Unknown if any dim is.
- `nin` / `nout` — input / output slot counts.
- `attr("name")` — an integer attribute; **unknown if absent**.
- `attr("name", default)` — integer attribute, or `default` if absent/wrong-kind.
- `sattr("name") == "literal"` (or `!=`) — a **string**-attribute test → `1`/`0`.
- `unknown` — the explicit honest-unknown literal (use as a guard result).
- a bare name — a variable defined in `vars`.

**Operators** `+ - * / %`, comparisons `== != < <= > >=`, `&& || !`,
ternary `cond ? a : b`, `min(a,b,…)`, `max(a,b,…)`, `prod(inK.shape[a:b])`
(product over a dim slice; `a:b` is a half-open range, negatives allowed).

**Honesty rules** (why you can't shoot yourself in the foot):
- integer overflow, `/0`, `%0`, or a **negative** feeding FLOPs ⇒ unknown.
- any unresolved shape/attr is contagious ⇒ unknown (short-circuited through
  `&&`/`||`/`?:`).
- unknown FLOPs are excluded from totals and shown as such — never faked.
- `MAC = 2 FLOPs`, so an N-multiply-accumulate op writes `2 * (…)`.

## The examples

| Directory | Teaches |
|---|---|
| `my-ops/` | fused-attention FLOPs + a color-only built-in override |
| `linear-layer/` | a simple `MatMul`-family op: `flops = 2·O·K`, shape passthrough |
| `custom-norm/` | a custom norm op + honest-unknown guard + string-attr branch |
| `group-conv/` | `Conv` with grouped channels using `prod()` over a kernel slice |
| `color-theme/` | pure recolor of several built-ins (no FLOP change) — a "theme" |

Each directory's `plugin.json` is heavily commented. Copy one, edit the op name +
formula, restart NetVis, and check **View → Plugins**.

## Testing your formula

`tools/validate_plugin.py <plugin.json>` (stdlib-only) checks the JSON shape,
category names, `override` requirement, and basic DSL token sanity **before** you
install it. NetVis itself validates fully at load and reports any error in the
panel.
