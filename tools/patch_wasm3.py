#!/usr/bin/env python3
"""Patch a fetched wasm3 source tree so the sandbox can kill a runaway module.

wasm3 v0.5.0 calls the weak `m3_Yield()` hook on every function Call, but a
tight WASM loop with no calls (`loop; br 0; end`) never hits Call — so a fuel
check in m3_Yield alone cannot stop it. This adds `m3_Yield()` to the loop
back-edge ops (`op_ContinueLoop` / `op_ContinueLoopIf` in m3_exec.h), the exact
"escape point" the wasm3 authors flagged in a comment there. Combined with a
strong `m3_Yield` override in the host, this lets a deadline/step cap trap ANY
runaway module — loop or recursion — so the app survives a hostile plugin.

Idempotent: running twice is a no-op. Invoked from CMake at configure time
(POST wasm3 populate) so the fetched source in the build tree carries the patch
while the upstream tag stays pinned/pristine.
"""
import sys

MARKER = "/* NETVIS_WASM_FUEL */"

CONTINUE_LOOP_ORIG = """d_m3Op  (ContinueLoop)
{
    m3StackCheck();
"""
CONTINUE_LOOP_NEW = """d_m3Op  (ContinueLoop)
{
    m3StackCheck();
    { m3ret_t nv_trap = m3_Yield (); if (UNLIKELY(nv_trap)) return nv_trap; } """ + MARKER + """
"""

CONTINUE_LOOP_IF_ORIG = """d_m3Op  (ContinueLoopIf)
{
    i32 condition = (i32) _r0;
    void * loopId = immediate (void *);

    if (condition)
    {
"""
CONTINUE_LOOP_IF_NEW = """d_m3Op  (ContinueLoopIf)
{
    i32 condition = (i32) _r0;
    void * loopId = immediate (void *);

    if (condition)
    {
        { m3ret_t nv_trap = m3_Yield (); if (UNLIKELY(nv_trap)) return nv_trap; } """ + MARKER + """
"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_wasm3.py <path-to-m3_exec.h>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"patch_wasm3: cannot read {path}: {e}", file=sys.stderr)
        return 0  # non-fatal: WASM backend still builds, loop-kill just absent

    if MARKER in text:
        return 0  # already patched (idempotent)

    ok = True
    if CONTINUE_LOOP_ORIG in text:
        text = text.replace(CONTINUE_LOOP_ORIG, CONTINUE_LOOP_NEW, 1)
    else:
        ok = False
    if CONTINUE_LOOP_IF_ORIG in text:
        text = text.replace(CONTINUE_LOOP_IF_ORIG, CONTINUE_LOOP_IF_NEW, 1)
    else:
        ok = False

    if not ok:
        print("patch_wasm3: anchors not found (wasm3 version drift?) — "
              "loop-kill patch NOT applied", file=sys.stderr)
        return 0  # non-fatal, surfaced

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"patch_wasm3: applied loop-fuel patch to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
