#!/usr/bin/env python3
"""tools/bench_gate.py — CI perf-regression gate for the benchmark harness (#97).

Compares a freshly captured `netvis --bench` JSON run ("current") against a
committed baseline and decides whether the diff represents an acceptable
change. Stdlib only (house rule — see tools/gen_fixtures.py's top comment):
this script must run on a bare `ubuntu-latest` runner with nothing installed
beyond python3 itself.

    python3 tools/bench_gate.py --baseline bench/baseline.json --current out.json
    python3 tools/bench_gate.py --baseline bench/baseline.json --current out.json --threshold 0.20
    python3 tools/bench_gate.py --baseline bench/baseline.json --current out.json --update

Exit codes (deliberately distinct so CI/a human can tell "the code got
slower" apart from "the gate itself could not run"):
    0 - clean: every matched stage is within threshold (or improved), and every
        baseline stage is present in the current run.
    1 - a real regression was found: a matched stage exceeded the threshold, or
        a stage present in the baseline is MISSING from the current run.
    2 - the run could not be judged at all: missing/unreadable file, invalid
        JSON, missing required keys, a schema tag mismatch, or a build-type
        mismatch. Silence or a pass in these cases would be worse than a loud
        failure, because the whole point of the gate is that a pass MEANS
        something (see docs/v1.0-plan.md's gate policy, quoted in the DIGEST
        for #97).

INTEGRATION NOTE on the JSON shape this script expects: engine/Bench.h (the
frozen contract) documents build_bench_json() as emitting "kBenchSchema, the
host's logical core count, and the build type" alongside the BenchCase array,
but does not spell out the exact JSON key names — that lives in Bench.cpp,
authored by a different task in this same milestone. This script assumes the
same naming convention ReportJson.cpp already uses for the sibling --report
JSON (plain lower_snake_case field-per-struct-member: "schema", "cases",
each case has "label"/"stages", each stage has "name"/"ms"), plus a top-level
"build" string for the Debug/Release guard this file's docstring
requires. If Bench.cpp lands with different key names, update
REQUIRED_TOP_KEYS/REQUIRED_CASE_KEYS/REQUIRED_STAGE_KEYS and the lookups in
flat_stages() below to match — the validation in load_bench() will fail
LOUDLY (exit 2, naming the missing key) rather than silently miscomparing,
so a mismatch here cannot masquerade as a clean run.
"""

import argparse
import json
import sys
from pathlib import Path

# The decided gate policy (docs/v1.0-plan.md, #97 DIGEST): a GENEROUS
# threshold on purpose. Shared CI runners are noisy (neighboring jobs steal
# cycles, cache state differs run to run); a tight gate flags noise as often
# as it flags real regressions, and a gate that cries wolf gets ignored,
# which is strictly worse than not having one. 30% is the agreed default;
# --threshold exists for local experimentation, not to loosen CI's copy.
DEFAULT_THRESHOLD = 0.30

# WHY a noise floor, as a NAMED constant (#97): a stage that legitimately
# takes ~0.02ms is dominated by chrono/timer resolution and OS-scheduler
# jitter on a shared runner, not by the code under test — the SAME binary
# can report a 2x swing between two back-to-back runs with zero code change.
# Below this floor a percentage delta is measurement noise, not signal, so a
# stage is judged against the threshold only once EITHER reading clears the
# floor. 0.05ms sits comfortably above the sub-0.03ms noise band synthetic-1k
# stages like "cost" can legitimately land in, and comfortably below every
# rung the gate actually exists to protect (all measured in whole
# milliseconds or more).
NOISE_FLOOR_MS = 0.05

REQUIRED_TOP_KEYS = ("schema", "build", "cases")
REQUIRED_CASE_KEYS = ("label", "stages")
REQUIRED_STAGE_KEYS = ("name", "ms")

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_ERROR = 2


class GateError(Exception):
    """Raised for anything that makes a run INCOMPARABLE or malformed.

    Always maps to exit code 2 — kept a distinct exception from "a real
    regression was found" so main() can tell the two failure modes apart at
    the point it chooses sys.exit()'s argument.
    """


def load_bench(path_str, role):
    """Load and structurally validate one bench JSON file.

    `role` is "baseline" or "current", used only to make error messages say
    which file is at fault — with two file arguments on the command line, a
    bare "missing key 'schema'" is not enough to act on.
    """
    path = Path(path_str)
    if not path.exists():
        raise GateError(f"{role} bench file not found: {path}")
    try:
        raw = path.read_text()
    except OSError as e:
        raise GateError(f"{role} bench file unreadable: {path} ({e})")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GateError(f"{role} bench file is not valid JSON: {path} ({e})")
    if not isinstance(data, dict):
        raise GateError(f"{role} bench file must be a JSON object: {path}")

    for key in REQUIRED_TOP_KEYS:
        if key not in data:
            raise GateError(f"{role} bench file missing required key {key!r}: {path}")
    if not isinstance(data["cases"], list):
        raise GateError(f"{role} bench file's 'cases' must be a list: {path}")

    for case in data["cases"]:
        if not isinstance(case, dict):
            raise GateError(f"{role} bench file has a non-object case entry: {path}")
        for key in REQUIRED_CASE_KEYS:
            if key not in case:
                raise GateError(f"{role} bench file case missing {key!r}: {path}")
        if not isinstance(case["stages"], list):
            raise GateError(
                f"{role} case {case['label']!r} 'stages' must be a list: {path}"
            )
        for stage in case["stages"]:
            if not isinstance(stage, dict):
                raise GateError(
                    f"{role} case {case['label']!r} has a non-object stage entry: {path}"
                )
            for key in REQUIRED_STAGE_KEYS:
                if key not in stage:
                    raise GateError(
                        f"{role} case {case['label']!r} stage missing {key!r}: {path}"
                    )

    return data


def check_comparable(baseline, current):
    """Refuse to compare two runs that are not measuring the same thing.

    Both checks exist because a "pass" from this gate is read as "no
    regression happened" — if the two runs are secretly incompatible (a
    schema bump changed a field's meaning, or one run is Debug and the other
    Release), a numeric comparison would produce a confident-looking answer
    that means nothing. Loud refusal beats a plausible lie.
    """
    if baseline["schema"] != current["schema"]:
        raise GateError(
            "schema mismatch: baseline={!r} current={!r} -- these are DIFFERENT "
            "wire formats (a kBenchSchema bump changed a field's meaning). "
            "Comparing them would silently read the wrong fields. If this bump "
            "is intentional, accept the new numbers as the starting point with "
            "--update.".format(baseline["schema"], current["schema"])
        )
    if baseline["build"] != current["build"]:
        raise GateError(
            "build type mismatch: baseline={!r} current={!r} -- a Debug run is "
            "meaningless against a Release baseline (or vice versa). This "
            "usually means CI's CMAKE_BUILD_TYPE configuration drifted, not "
            "that the code got faster or slower.".format(
                baseline["build"], current["build"]
            )
        )


def flat_stages(data):
    """Flatten cases/stages to {(case_label, stage_name): stage_dict}.

    Matching is by (label, stage name) per the frozen-name contract in
    engine/Bench.h: stage names are frozen constants precisely so a rename
    cannot silently drop a stage out of this comparison — it shows up as
    "missing" on one side and "added" on the other instead.

    Python dicts preserve insertion order (3.7+), so the values() here are
    the file's own case/stage order — used below to keep the printed table
    in a stable, reproducible order rather than hash order.
    """
    out = {}
    for case in data["cases"]:
        for stage in case["stages"]:
            out[(case["label"], stage["name"])] = stage
    return out


def fmt_ms(value):
    return f"{value:.4f}"


def compare(baseline, current, threshold):
    """Diff two validated bench runs. Returns (rows, any_failure)."""
    base_stages = flat_stages(baseline)
    cur_stages = flat_stages(current)

    rows = []
    any_failure = False

    # Walk the BASELINE first so a stage that used to exist and vanished is
    # never silently skipped — see the missing-stage handling below.
    for key, bstage in base_stages.items():
        label, name = key
        bms = float(bstage["ms"])

        if key not in cur_stages:
            # Hard failure, not a skip (per #97's contract): a rename or an
            # accidentally-deleted stage must not quietly drop out of the gate.
            rows.append(
                {
                    "label": label,
                    "stage": name,
                    "baseline_ms": bms,
                    "current_ms": None,
                    "delta": None,
                    "verdict": "MISSING",
                    "failure": True,
                }
            )
            any_failure = True
            continue

        cms = float(cur_stages[key]["ms"])
        both_under_floor = bms < NOISE_FLOOR_MS and cms < NOISE_FLOOR_MS

        if both_under_floor:
            rows.append(
                {
                    "label": label,
                    "stage": name,
                    "baseline_ms": bms,
                    "current_ms": cms,
                    "delta": None,
                    "verdict": "ok (< noise floor)",
                    "failure": False,
                }
            )
            continue

        if bms <= 0.0:
            # Baseline rounded to zero but current cleared the noise floor:
            # a % delta is undefined (division by zero), but going from
            # "too fast to measure" to "measurable" is a real change worth a
            # human's eyes, not something to silently wave through.
            rows.append(
                {
                    "label": label,
                    "stage": name,
                    "baseline_ms": bms,
                    "current_ms": cms,
                    "delta": None,
                    "verdict": "REGRESSION (baseline ~= 0)",
                    "failure": True,
                }
            )
            any_failure = True
            continue

        delta = (cms - bms) / bms
        if delta > threshold:
            verdict, failure = "REGRESSION", True
            any_failure = True
        elif delta <= -threshold:
            # A large unexplained SPEEDUP is reported, never failed. Per the
            # #97 contract: it as often means the work stopped happening (a
            # cache hit, an early return, a stage silently skipped) as a
            # genuine win, so a human should look — a robot cannot tell the
            # difference from the number alone.
            verdict, failure = "IMPROVED (verify)", False
        else:
            verdict, failure = "ok", False

        rows.append(
            {
                "label": label,
                "stage": name,
                "baseline_ms": bms,
                "current_ms": cms,
                "delta": delta,
                "verdict": verdict,
                "failure": failure,
            }
        )

    # Anything new in `current` is fine — report it, do not fail on it.
    for key, cstage in cur_stages.items():
        if key in base_stages:
            continue
        label, name = key
        rows.append(
            {
                "label": label,
                "stage": name,
                "baseline_ms": None,
                "current_ms": float(cstage["ms"]),
                "delta": None,
                "verdict": "added",
                "failure": False,
            }
        )

    # --- #101: the memory ceiling -------------------------------------------
    #
    # Time is not the only thing that regresses. A change that trades memory for
    # speed will sail through every stage comparison above while quietly pushing
    # peak RSS up, and "opens a multi-GB model" is a memory claim as much as a
    # latency one — the zero-payload thesis is precisely a statement about not
    # holding the file in RAM. So peak RSS is gated on the same terms as a stage.
    #
    # Compared per case, because RSS is reported per case and the 100k rung is
    # the one that matters. Reported as "unknown" rather than compared when
    # either side is 0: core/Rss.h returns 0 for "the platform counter was
    # unavailable", NOT for zero bytes, and treating an unavailable counter as a
    # 100% improvement would silently disable this check on any platform where
    # it does not work.
    base_rss = {c["label"]: c.get("peak_rss_bytes", 0) for c in baseline["cases"]}
    for case in current["cases"]:
        label = case["label"]
        cur = case.get("peak_rss_bytes", 0)
        base = base_rss.get(label, 0)
        if label not in base_rss:
            continue  # a new case; its stages already reported as "added"
        if not base or not cur:
            rows.append({
                "label": label, "stage": "peak_rss", "baseline_ms": None,
                "current_ms": None, "delta": None,
                "verdict": "unknown (RSS counter unavailable)", "failure": False,
            })
            continue
        delta = (cur - base) / base
        # The same generous threshold as timing. Allocator behaviour varies
        # between runners far less than the scheduler does, but a tight bound
        # here would fail on a glibc change rather than on a NetVis change.
        failed = delta > threshold
        rows.append({
            "label": label,
            "stage": "peak_rss",
            "baseline_ms": base / (1024.0 * 1024.0),
            "current_ms": cur / (1024.0 * 1024.0),
            "delta": delta,
            "verdict": ("REGRESSION (memory)" if failed
                        else "IMPROVED (verify)" if delta < -0.10
                        else "ok"),
            "failure": failed,
        })
        any_failure = any_failure or failed

    return rows, any_failure


def render_table(rows):
    """A GitHub-flavored Markdown table — paste straight into a PR comment."""
    lines = [
        "| case | stage | baseline (ms) | current (ms) | delta | verdict |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        bms = fmt_ms(r["baseline_ms"]) if r["baseline_ms"] is not None else "-"
        cms = fmt_ms(r["current_ms"]) if r["current_ms"] is not None else "-"
        delta = f"{r['delta'] * 100:+.1f}%" if r["delta"] is not None else "-"
        lines.append(f"| {r['label']} | {r['stage']} | {bms} | {cms} | {delta} | {r['verdict']} |")
    return "\n".join(lines)


def do_update(current_path, baseline_path):
    """--update: rewrite the baseline from a validated current run.

    Only the CURRENT file is structurally validated ("refuse to update when
    the run is incomparable" — read here as "refuse to enshrine a malformed
    run as the new baseline", since there is nothing to compare a fresh
    baseline against). If a baseline already existed under a different
    schema/build_type, that is exactly what an intentional re-baseline is
    for, so it is reported as an informational note, not blocked.
    """
    current = load_bench(current_path, "current")

    old_note = ""
    old_path = Path(baseline_path)
    if old_path.exists():
        try:
            old = load_bench(baseline_path, "baseline")
            if old["schema"] != current["schema"] or old["build"] != current["build"]:
                old_note = (
                    "\nNOTE: replacing baseline schema={!r}/build={!r} with "
                    "schema={!r}/build={!r}.".format(
                        old["schema"], old["build"],
                        current["schema"], current["build"]
                    )
                )
        except GateError:
            # The existing baseline is itself unreadable/corrupt — that is
            # exactly the situation --update exists to repair, not a reason
            # to refuse the repair.
            old_note = "\nNOTE: the previous baseline file could not be read; overwriting it."

    old_path.write_text(json.dumps(current, indent=2) + "\n")
    print(f"Wrote baseline: {old_path} (schema={current['schema']!r}, build_type={current['build']!r}){old_note}")
    return EXIT_OK


def main():
    parser = argparse.ArgumentParser(
        description="Compare a netvis --bench JSON run against a committed baseline (#97)."
    )
    parser.add_argument("--baseline", required=True, help="Path to the committed baseline JSON.")
    parser.add_argument("--current", required=True, help="Path to the freshly captured bench JSON.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Fractional regression threshold (default {DEFAULT_THRESHOLD}, i.e. {DEFAULT_THRESHOLD*100:.0f}%%).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite --baseline from --current instead of comparing. Refuses a malformed current run.",
    )
    args = parser.parse_args()

    try:
        if args.update:
            return do_update(args.current, args.baseline)

        baseline = load_bench(args.baseline, "baseline")
        current = load_bench(args.current, "current")
        check_comparable(baseline, current)
    except GateError as e:
        print(f"bench_gate: ERROR: {e}", file=sys.stderr)
        return EXIT_ERROR

    rows, any_failure = compare(baseline, current, args.threshold)

    n_regressed = sum(1 for r in rows if r["failure"])
    n_missing = sum(1 for r in rows if r["verdict"] == "MISSING")
    n_added = sum(1 for r in rows if r["verdict"] == "added")
    n_improved = sum(1 for r in rows if r["verdict"] == "IMPROVED (verify)")

    print(f"bench_gate: schema={current['schema']!r} build_type={current['build']!r}")
    print(f"bench_gate: threshold={args.threshold*100:.0f}% noise_floor={NOISE_FLOOR_MS}ms "
          f"(stages where BOTH readings are under the floor are never flagged, "
          "regardless of their percentage delta)")
    print()
    print(render_table(rows))
    print()
    print(
        f"bench_gate: {len(rows)} stage(s) compared -- "
        f"{n_regressed} regression(s), {n_missing} missing, {n_added} added, "
        f"{n_improved} large improvement(s) to double-check."
    )

    if any_failure:
        print("bench_gate: FAIL — see REGRESSION/MISSING rows above.", file=sys.stderr)
        return EXIT_REGRESSION

    print("bench_gate: PASS")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
