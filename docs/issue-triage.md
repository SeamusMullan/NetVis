# Issue triage — priority assessment

> Pass run 2026-09-10 over all 29 open issues. Priorities are applied as `P0`–`P3`
> labels on GitHub; this document is the reasoning behind them, so a priority can be
> argued with rather than just obeyed.

## The scale

| Label | Meaning | Answer to "why not later?" |
|---|---|---|
| **P0** | Broken for users on a shipped release. | It is broken *now*. |
| **P1** | On the critical path to the next tag (v0.9.5 format parity), or a commitment whose cost rises the longer it waits. | Waiting makes it more expensive, not just later. |
| **P2** | Real value, milestoned, not blocking anything. | Scheduled work. Do it in milestone order. |
| **P3** | Post-1.0, blocked on a decision, or blocked on missing information. | It cannot be started responsibly today. |

Two rules used throughout:

- **A tracking parent inherits its highest child's priority.** #108/#109/#110/#111/#112/#132 are umbrellas; the work lives in the sub-issues.
- **Priority is not schedule.** #113 is P1 in a v1.0.0 milestone because the freeze
  gets harder every month, not because it should jump the queue ahead of v0.9.5.

## P0 — one issue

| # | Title |
|---|---|
| [#149](https://github.com/SeamusMullan/NetVis/issues/149) | Installer fails on virtual drives; PATH-too-long breaks add-to-PATH |

The only open issue that fails a user on a released build, and it fails them at the
first step — a viewer nobody can install scores zero on every other axis in this
list. It is also the only user-reported bug among 29 issues that are otherwise all
maintainer-authored roadmap work, which makes it the single datapoint about what
real usage hits.

**Blocked on information.** The body is one screenshot: no Windows version, no
installer version, no drive type (subst? VHD? network share?), no failing path. The
two symptoms in the title may also be two separate bugs — a virtual-drive resolution
failure and a `MAX_PATH` limit are different fixes. Reproduce or request specifics
before starting.

## P1 — seven issues

**TorchScript as a real graph** — [#108](https://github.com/SeamusMullan/NetVis/issues/108) (parent), [#136](https://github.com/SeamusMullan/NetVis/issues/136) (stage A, desktop archive), [#137](https://github.com/SeamusMullan/NetVis/issues/137) (stage B, mobile `.ptl`)

The largest remaining parity gap weighted by user population: PyTorch is the biggest
source of models a NetVis user will drag in, and today they get an op *inventory*
(#40's text scan), not a graph. `docs/v1.0-plan.md` ranks it P5.2, second only to
TensorFlow, which has shipped.

**Sequencing note: do #137 before #136.** #137 decodes `bytecode.pkl` through the
existing `PickleVM` — a fraction of the work — and it forces the same IR-mapping
decisions #136 will need. #136 is a language front-end for a typed Python subset and
is the single largest parser in the arc. Doing the cheap one first de-risks the
expensive one and puts a real TorchScript graph on screen sooner.

**Caffe** — [#109](https://github.com/SeamusMullan/NetVis/issues/109) (parent), [#138](https://github.com/SeamusMullan/NetVis/issues/138) (text-format protobuf reader)

P1 on schedule risk rather than user demand. Caffe is a legacy format, but #138 is
the only reader in v0.9.5 with **no precedent in the tree** — text-format protobuf is
a new grammar, and the issue already names a detection collision with
`looks_like_onnx_proto()`. Unknowns land early, not late in a release.

**[#113](https://github.com/SeamusMullan/NetVis/issues/113) — plugin ABI freeze + compatibility guarantee**

Milestoned v1.0.0, prioritised above it. The ABI is already at v1 and plugins already
load against it; every month it stays unfrozen-but-shipping is another month of
third-party code that a later correction would break. The work itself is small
(document the promise, add a version-rejection test, freeze the header) and it is a
prerequisite for #142 doing its own freeze against a stable model.

**[#114](https://github.com/SeamusMullan/NetVis/issues/114) — format-support matrix + honesty audit**

The honesty rule (*unresolved dim ⇒ honest-unknown; never a fabricated shape or
FLOP*) is a stated project invariant, which makes any violation a correctness bug,
not a docs gap. The audit half is the P1: it is the only thing that would catch a
parser quietly inventing a shape. The matrix half also happens to be the artifact a
prospective user reads before choosing NetVis over Netron.

## P2 — sixteen issues

Scheduled work with no blocking relationship. Grouped by what they are:

- **Remaining parity formats** — [#110](https://github.com/SeamusMullan/NetVis/issues/110)/[#139](https://github.com/SeamusMullan/NetVis/issues/139) (Darknet),
  [#111](https://github.com/SeamusMullan/NetVis/issues/111)/[#140](https://github.com/SeamusMullan/NetVis/issues/140)/[#141](https://github.com/SeamusMullan/NetVis/issues/141) (torch.export, ExecuTorch). Same v0.9.5 milestone as
  the P1 formats, ranked below them on user population. #139 is the sleeper: a
  `.weights` file has no offsets at all, so locating a tensor means simulating the
  whole `.cfg` — shape inference as a *precondition for reading bytes*, the inverse
  of every other parser here. Size it accordingly.
  *Worth revisiting:* the plan's user-value order puts torch.export/ExecuTorch last
  (P5.5) and Caffe/Darknet ahead of it. That ordering has aged — `.pt2` and `.pte`
  are where new PyTorch models are going while Caffe is static. If v0.9.5 has to shed
  scope, drop Darknet before dropping #140/#141.
- **View-panel plugins** — [#112](https://github.com/SeamusMullan/NetVis/issues/112) (parent), [#142](https://github.com/SeamusMullan/NetVis/issues/142) (protocol freeze), [#143](https://github.com/SeamusMullan/NetVis/issues/143) (renderer +
  hardening). **#142 strictly precedes #143** — that is the whole point of splitting
  them, and building the renderer first would leak host details into a wire format
  that then has to be frozen anyway.
- **1.0 capstone** — [#115](https://github.com/SeamusMullan/NetVis/issues/115) (user guide), [#116](https://github.com/SeamusMullan/NetVis/issues/116) (diagnostics), [#117](https://github.com/SeamusMullan/NetVis/issues/117) (coverage gaps),
  [#118](https://github.com/SeamusMullan/NetVis/issues/118) (parity scorecard). #116 is the one with leverage beyond its own scope:
  "silent best-effort everywhere" is why a report like #149 arrives as a screenshot
  instead of a log. #118 is genuinely last — it verifies work that has not landed.
- **[#146](https://github.com/SeamusMullan/NetVis/issues/146)** (fuzz harness) — standing infrastructure, not a 1.0 task. Every parser
  is a bounded reader over untrusted bytes and the current suites feed them
  well-formed fixtures. The `payload_read_counter() == 0` oracle is the interesting
  property: a mutated length field that tricks a parser into reading a payload is a
  zero-payload-thesis violation no existing test would catch.
- **[#135](https://github.com/SeamusMullan/NetVis/issues/135)** (TF checkpoint weights) — depth past parity; Netron does not read the
  checkpoint either. Its section 4 (stale file-dialog filters — `.xml`/`.npz`/
  `.keras`/`.h5`/`.mlmodel` parse fine but cannot be *picked*) is a one-line-per-site
  fix hiding inside a large issue. **Split it out and take it now**; it is a
  user-visible gap with a trivial cost.
- **Graph polish** — [#147](https://github.com/SeamusMullan/NetVis/issues/147) (edge thickness by tensor volume), [#157](https://github.com/SeamusMullan/NetVis/issues/157) (edge routing). Both are
  well-specified against real call sites and both are cheap. #157's default flip is
  one line, but confirm the Manhattan router on a dense graph first — it takes a
  fixed mid-Y per edge, which on a wide fan-out stacks horizontal segments.

## P3 — five issues

**MLIR** — [#132](https://github.com/SeamusMullan/NetVis/issues/132) (parent), [#144](https://github.com/SeamusMullan/NetVis/issues/144) (read-only parse), [#145](https://github.com/SeamusMullan/NetVis/issues/145) (editing + writeback)

`docs/v1.0-plan.md` places MLIR in the experimental tail, "explicitly post-1.0". Both
sub-issues additionally carry an open decision that must be made before any code:
#144 needs core-parser-vs-WASM-ParserPlugin settled (the plugin path already exists
and costs nothing in the core binary), and #145 asks whether NetVis is a viewer or a
viewer+editor — a product question, not a parser one. #145 in particular is not an
MLIR feature: `MappedFile` is read-only by construction and the zero-payload thesis
is built on that immutability, so a mutation layer touches the foundation. Keep it
out of the 1.0 arc.

**[#153](https://github.com/SeamusMullan/NetVis/issues/153) — graph display closer to Netron: recommend closing**

Fully decomposed. Its main note (constant placement) shipped in #161
(`feat: sink whole constant cones next to their consumer`), and its two remaining
gripes were split into #157 and #158. Nothing is tracked here that is not tracked
better elsewhere. Left open at P3 rather than closed unilaterally.

**[#158](https://github.com/SeamusMullan/NetVis/issues/158) — scroll/zoom does not match Netron: blocked on information**

The issue says so itself. "Does not match" spans wheel semantics, zoom anchoring,
zoom rate, drag-to-pan and trackpad-vs-mouse — five independent behaviours that
interact, so tuning them blind trades one mismatch for another. Needs the gesture
named and today-vs-expected stated before it is workable. Zoom anchoring (toward
pointer vs viewport centre) is the most likely culprit.

## Recommended order

1. **#149** — reproduce or request specifics, then fix.
2. **#137 → #136** — cheap TorchScript graph first, language front-end second.
3. **#138/#109** — the unknown reader, early.
4. **#113, #114** — small, and both get more expensive with time.
5. Then v0.9.5's remainder in milestone order, with **#135 §4** and **#157** as
   fillers whenever a large task needs a break.

## Housekeeping found in this pass

- **#153** is finished in substance — recommend closing in favour of #157/#158.
- **#149** and **#158** are blocked on information, not on effort. Neither should be
  picked up before that information exists.
- **#135 §4** (file-dialog filters) deserves its own issue; the parent will not be
  closed for a long time and the fix is a line per site.
- The `P0`–`P3` labels were created by this pass with GitHub's default grey and no
  description. Worth colouring them (red → grey) and describing them so the scale is
  legible in the issue list.
