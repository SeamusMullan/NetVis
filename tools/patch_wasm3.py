#!/usr/bin/env python3
"""Patch a fetched wasm3 source tree so the sandbox can kill a runaway module and
so wasm3's fallback m3_Yield does not collide with the host's on MSVC.

Two edits, each idempotent and non-fatal on anchor drift:

1. m3_exec.h — fuel on the loop back-edge. wasm3 v0.5.0 calls the weak `m3_Yield()`
   hook on every function Call, but a tight WASM loop with no calls
   (`loop; br 0; end`) never hits Call — so a fuel check in m3_Yield alone cannot
   stop it. This adds `m3_Yield()` to the loop back-edge ops (`op_ContinueLoop` /
   `op_ContinueLoopIf`), the exact "escape point" the wasm3 authors flagged in a
   comment there. Combined with the host's strong `m3_Yield` override, a
   deadline/step cap traps ANY runaway module — loop or recursion.

2. m3_core.c — remove wasm3's fallback `m3_Yield` definition. The NetVis host
   (WasmRuntime.cpp) always provides a STRONG `m3_Yield` (the fuel meter). On
   GCC/Clang `M3_WEAK` is `__attribute__((weak))`, so the host override silently
   wins. On MSVC `M3_WEAK` expands to nothing (see m3_config_platforms.h), so
   wasm3's fallback is a SECOND strong definition and the two collide at link
   (LNK2005 -> LNK1169, netvis.exe). Dropping wasm3's fallback leaves the host's
   as the sole definition on every toolchain; wasm3's internal callers resolve to
   it at final link (a static lib may carry the symbol unresolved).

Idempotent: running twice is a no-op. Invoked from CMake at configure time (POST
wasm3 populate) so the fetched source in the build tree carries the patch while
the upstream tag stays pinned/pristine. Accepts either the wasm3 `source/`
directory or (back-compat) the path to `m3_exec.h`.
"""
import os
import sys

MARKER = "/* NETVIS_WASM_FUEL */"

# --- Edit 1: m3_exec.h loop back-edge fuel -----------------------------------
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

# --- Edit 2: m3_core.c drop wasm3's fallback m3_Yield ------------------------
YIELD_ORIG = """M3_WEAK
M3Result m3_Yield ()
{
    return m3Err_none;
}"""
YIELD_NEW = ("""#if 0  """ + MARKER + """  /* wasm3's fallback m3_Yield removed:
   the NetVis host provides a STRONG m3_Yield (fuel meter). On MSVC M3_WEAK is
   empty, so keeping this here is a second strong def and collides at link
   (LNK2005). Dropping it leaves the host's as the sole definition everywhere. */
M3_WEAK
M3Result m3_Yield ()
{
    return m3Err_none;
}
#endif""")


def _read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        print(f"patch_wasm3: cannot read {path}: {e}", file=sys.stderr)
        return None


def patch_exec(path):
    text = _read(path)
    if text is None:
        return  # non-fatal: WASM backend still builds, loop-kill just absent
    if MARKER in text:
        return  # already patched (idempotent)
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
        print("patch_wasm3: m3_exec.h anchors not found (wasm3 version drift?) — "
              "loop-kill patch NOT applied", file=sys.stderr)
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"patch_wasm3: applied loop-fuel patch to {path}")


def patch_core(path):
    text = _read(path)
    if text is None:
        return  # non-fatal
    if MARKER in text:
        return  # already patched (idempotent)
    if YIELD_ORIG not in text:
        print("patch_wasm3: m3_core.c m3_Yield anchor not found (wasm3 version "
              "drift?) — MSVC link-collision guard NOT applied", file=sys.stderr)
        return
    text = text.replace(YIELD_ORIG, YIELD_NEW, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"patch_wasm3: neutralized wasm3 fallback m3_Yield in {path}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_wasm3.py <wasm3-source-dir | path-to-m3_exec.h>",
              file=sys.stderr)
        return 2
    arg = sys.argv[1]
    if os.path.isdir(arg):
        src_dir = arg
    else:
        src_dir = os.path.dirname(arg)   # back-compat: given m3_exec.h directly
    patch_exec(os.path.join(src_dir, "m3_exec.h"))
    patch_core(os.path.join(src_dir, "m3_core.c"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
