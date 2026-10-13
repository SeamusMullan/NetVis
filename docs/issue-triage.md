# Issue triage — priority assessment

> Pass run 2026-09-10 over all 29 open issues. Priorities are applied as `P0`–`P3`
> labels on GitHub; this document is the reasoning behind them, so a priority can be
> argued with rather than just obeyed.

## Status since the pass

The text below is the 2026-09-10 snapshot. Things that have moved since, or that
the snapshot got wrong:

- **A CoreML regression shipped in v0.9.6 and is P0 by the scale below, but has no
  issue of its own.** The TensorFlow structural sniff added by #134 runs before the
  `.mlmodel` extension guard in `src/parsers/Detect.cpp`, and a bare CoreML `Model`
  protobuf satisfies `looks_like_saved_model` (field 1 `specificationVersion`, then
  a length-delimited field 2 whose first inner tag is field 1). Ordinary `.mlmodel`
  files are therefore handed to the TensorFlow parser and fail to open with
  "SavedModel meta_graph carries no graph_def". Draft PR #166 carries the fix. It
  should land first, or the regression should be filed as its own P0 issue so the
  tracker shows it.
- **Draft fix PRs exist for three items.** #149 → #164 (add-to-PATH edit for long
  PATHs and virtual drives; a separate install failure, if there is one, still needs
  a repro). #113 → #165 (ABI freeze). #114 → #166 (format matrix + honesty audit, and
  the CoreML fix above).
- **#153 has been closed**, as recommended below.
- **Milestone names are not release tags.** The v0.9.5 and v0.9.6 tags have shipped
  (2026-08-21 and 2026-09-08). The `v0.9.5` and `v0.9.6` milestones are still open,
  and their contents will land under later tags.
- **Filed after this pass:** #169 (P1) and #170 (P2) carry labels on GitHub but are
  not assessed in this document.

## The scale

| Label | Meaning | Answer to "why not later?" |
|---|---|---|
| **P0** | Broken for users on a shipped release. | It is broken *now*. |
| **P1** | On the critical path to the v0.9.5 *milestone* (format parity), or a commitment whose cost rises the longer it waits. | Waiting makes it more expensive, not just later. |
| **P2** | Real value, usually milestoned, not blocking anything. | Scheduled work. Do it in milestone order. |
| **P3** | Post-1.0, blocked on a decision, or blocked on missing information. | It cannot be started responsibly today. |

Two rules used throughout:

- **A tracking parent inherits its highest child's priority.** #108/#109/#110/#111/#112/#132 are umbrellas; the work lives in the sub-issues.
- **Priority is not schedule.** #113 is P1 in a v1.0.0 milestone because the freeze
  gets harder every month, not because it should jump the queue ahead of the v0.9.5
  milestone.

Where a label does not fit its definition exactly:

- **#135** is milestoned v1.1.0, which is P3 territory. It stays P2 only because of
  its §4 file-dialog fix; once that is split out (see P2 and the housekeeping list),
  #135 drops to P3.
- **#147 and #157** are P2 without a milestone: unmilestoned graph polish, scheduled
  by cost rather than by release.

## P0 — one issue, plus one unfiled regression

| # | Title |
|---|---|
| [#149](https://github.com/SeamusMullan/NetVis/issues/149) | Installer fails to properly install on virtual drives & Path too long errors cause issues with install add to path |

The only open *issue* that fails a user on a released build. The CoreML detection
regression described under "Status since the pass" also meets the P0 bar, but it has
no issue of its own and is tracked only through #166.

#149 is the only open bug report against a shipped build, and it bites at the first
step: the installer. The screenshot shows only the add-to-PATH step failing ("PATH too
long installer unable to modify PATH!"), which skips the PATH edit but lets the
install finish. #153 and #158 are also usage feedback, but about how the graph looks
and feels rather than something that does not work.

**Partly blocked on information.** The body is one screenshot: no Windows version, no
installer version, no drive type (subst? VHD? network share?), no failing path. The
two symptoms in the title may also be two separate bugs — a virtual-drive resolution
failure and a `MAX_PATH` limit are different fixes. Draft #164 already diagnoses the
PATH-length half and the virtual-drive PATH skip from the screenshot and the CPack
NSIS template. What still needs a repro is any separate "files did not land"
symptom.

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
the only reader in the v0.9.5 milestone with **no precedent in the tree** — text-format protobuf is
a new grammar, and the issue already names a detection collision with
`looks_like_onnx_proto()`. Unknowns land early, not late in a milestone.

**[#113](https://github.com/SeamusMullan/NetVis/issues/113) — plugin ABI freeze + compatibility guarantee**

Milestoned v1.0.0, prioritised above it. The ABI is already at v1 and plugins already
load against it; every month it stays unfrozen-but-shipping is another month of
third-party code that a later correction would break. The work itself is small
(document the promise, add a version-rejection test, freeze the header) and it is a
prerequisite for #142 doing its own freeze against a stable model. Draft #165 is open.

**[#114](https://github.com/SeamusMullan/NetVis/issues/114) — format-support matrix + honesty audit**

The honesty rule (*unresolved dim ⇒ honest-unknown; never a fabricated shape or
FLOP*) is a stated project invariant, which makes any violation a correctness bug,
not a docs gap. The audit half is the P1: it is the only thing that would catch a
parser quietly inventing a shape. The matrix half also happens to be the artifact a
prospective user reads before choosing NetVis over Netron. Draft #166 is open; its
audit already found the CoreML regression described under "Status since the pass".

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
  are where new PyTorch models are going while Caffe is static. If the v0.9.5 milestone has to shed
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
  checkpoint either. Its section 4 is stale file-dialog filters: `.xml`/`.npz`/
  `.keras`/`.h5`/`.hdf5`/`.pkl`/`.pickle` parse fine but cannot be *picked*, and
  `.mlmodel` is in the same position once #166 lands (today it also fails to parse).
  The two sites are `App::open_file_dialog` (`src/view/App.cpp`) and the DiffPanel
  open dialog (`src/view/DiffPanel.cpp`); since #148 the patterns are a
  `std::vector` initializer list, so it is one line per site hiding inside a large
  issue. **Split it out and take it now**; it is a user-visible gap with a trivial
  cost.
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

**[#153](https://github.com/SeamusMullan/NetVis/issues/153) — graph display closer to Netron: recommended closing (since closed)**

Fully decomposed. Its main note (constant placement) shipped in #161
(`feat: sink whole constant cones next to their consumer`), and its two remaining
gripes were split into #157 and #158. Nothing is tracked here that is not tracked
better elsewhere. It was left open at P3 rather than closed unilaterally, and has
since been closed.

**[#158](https://github.com/SeamusMullan/NetVis/issues/158) — scroll/zoom does not match Netron: blocked on information**

The issue says so itself. "Does not match" spans wheel semantics, zoom anchoring,
zoom rate, drag-to-pan and trackpad-vs-mouse — five independent behaviours that
interact, so tuning them blind trades one mismatch for another. Needs the gesture
named and today-vs-expected stated before it is workable. Zoom anchoring (toward
pointer vs viewport centre) is the most likely culprit.

## Recommended order

1. **The CoreML detection fix (draft #166)** — a regression on the current release;
   land it first, or file it as a P0 issue and link the PR.
2. **#149** — review draft #164 for the PATH half, then request specifics for
   anything beyond it.
3. **#137 → #136** — cheap TorchScript graph first, language front-end second.
4. **#138/#109** — the unknown reader, early.
5. **#113, #114** — small, and both get more expensive with time. Drafts #165 and
   #166 are open for these.
6. Then the v0.9.5 milestone's remainder in milestone order, with **#135 §4** and
   **#157** as fillers whenever a large task needs a break.

## Housekeeping found in this pass

- **#153** was finished in substance and has since been closed in favour of #157/#158.
- **#158** is blocked on information, not on effort, and should not be picked up
  before that information exists. **#149** is only partly blocked: the PATH half has
  a draft fix (#164), and only any separate install failure still needs a repro.
- **#135 §4** (file-dialog filters) deserves its own issue; the parent will not be
  closed for a long time and the fix is a line per site. Once it is split out, #135
  itself drops to P3.
- **The CoreML detection regression** deserves its own P0 issue (see "Status since the
  pass"), so the tracker reflects what is broken on the current release.
- The `P0`–`P3` labels were created by this pass with GitHub's default grey and no
  description. Worth colouring them (red → grey) and describing them so the scale is
  legible in the issue list.
