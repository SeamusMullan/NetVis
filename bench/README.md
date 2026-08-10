# Benchmark harness (#97)

This directory holds the committed performance **baseline** that
`tools/bench_gate.py` checks every push/PR against, and this file, which
explains how the workflow works.

## Running the harness locally

```sh
cmake --preset release
cmake --build --preset release --target netvis
./build/release/netvis --bench > out.json          # full 1k/10k/100k ladder
./build/release/netvis --bench-quick > out.json     # skips the 100k rung
```

`--bench` is a headless mode (see `src/engine/Bench.h` / `src/main.cpp`): it
never creates a window or a GL context, times the engine's hot paths
(`parse`, `collapse`, `shape_infer`, `layout`, `cost`, `visible_scan`)
directly against a deterministic synthetic model ladder, and prints one JSON
object to stdout. Build **Release** when you intend to compare against the
baseline — a Debug number is a different measurement, not a slower one, and
`tools/bench_gate.py` refuses to compare across build types for exactly that
reason.

`--bench-repeats=N` overrides the default median-of-5; `--bench-model=<path>`
additionally times a real model file end-to-end alongside the synthetic
ladder.

## Reading the JSON

Each entry in `cases` is one rung of the ladder (`synthetic-1k`,
`synthetic-10k`, `synthetic-100k`, or a real file name passed via
`--bench-model`), carrying the structural counts (`ir_nodes`, `ir_edges`,
`display_nodes` after collapse, `layout_boxes`) and a `stages` array. Every
stage's `ms` is the **median** across repeats, not the mean — a single
scheduler hiccup on a shared runner skews a mean; the median shrugs it off.
`min_ms`/`max_ms` are the fastest/slowest repeat, useful for eyeballing how
noisy a particular stage was on that run.

**`visible_scan` is a proxy, not a frame time.** It measures the O(total
boxes/edges) scans `GraphCanvas` performs every frame — hit-testing and edge
culling — over a layout, at a given `visible_fraction`. It is the number that
must stop growing with total node count once spatial indexing (#99) lands.
It is **not** a frame time and **must never be reported as fps** — there is
no GL context, no ImGui, no actual draw call anywhere in this number.

## Re-baselining

`bench/baseline.json` is a **deliberate, reviewed artifact**, not something
regenerated automatically. Re-baseline when a real, understood change moves
the numbers (a new optimization, an intentional algorithmic tradeoff, new
hardware for the CI runner) — never just to make a red gate turn green.

```sh
./build/release/netvis --bench > out.json
python3 tools/bench_gate.py --baseline bench/baseline.json --current out.json --update
```

Put the re-baseline in **its own commit**, separate from the code change
that caused it, with a commit message stating *why* the numbers moved (e.g.
"rebaseline: layout now O(display_nodes) after #99, was O(layout_boxes)").
That makes the baseline's git history a readable log of every perf-relevant
change NetVis has shipped, instead of a number nobody can explain.

### `bench/baseline.json` does not exist yet

As of the #97 harness landing, this file is intentionally absent: the
harness that would produce real numbers didn't exist until this same
change, so there is nothing genuine to commit yet. Any baseline written
before the harness has actually been built and run would be fabricated
data. The CI perf-gate job (`.github/workflows/ci.yml`) detects the missing
file and runs in report-only mode (uploads the bench JSON as an artifact,
does not fail the build) until a maintainer generates the first real
baseline with `--update` above and commits it as its own reviewed change.

## Threshold and noise floor

`tools/bench_gate.py --threshold` defaults to **0.30** (30%). This is
deliberately generous: shared CI runners are noisy, and a tight threshold
produces flaky failures that get ignored, which is worse than no gate at
all (see `docs/v1.0-plan.md`'s gate policy). The gate also applies a
**noise floor** (`NOISE_FLOOR_MS` in the script, currently 0.05ms): a stage
that legitimately takes ~0.02ms is dominated by timer-resolution and
scheduler jitter, not by the code under test, so a percentage delta below
the floor is not treated as signal even if it nominally exceeds the
threshold. `DEFAULT_THRESHOLD` and `NOISE_FLOOR_MS` at the top of
`tools/bench_gate.py` are the source of truth if this doc and the script
ever drift; `--threshold` on the command line overrides the former for a
one-off run.
