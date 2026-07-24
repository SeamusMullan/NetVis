#!/usr/bin/env bash
# Build the attn_pass WASM plugin. Requires clang with a wasm32 target (LLVM 8+).
# NetVis CI has no wasm toolchain, so examples are author-compiled; the test suite
# uses hand-encoded fixtures (tools/gen_fixtures.py) instead.
set -euo pipefail
cd "$(dirname "$0")"

clang \
  --target=wasm32 \
  -nostdlib \
  -O2 \
  -Wl,--no-entry \
  -Wl,--export-dynamic \
  -Wl,--allow-undefined \
  -o attn_pass.wasm \
  attn_pass.c

echo "wrote attn_pass.wasm ($(wc -c < attn_pass.wasm) bytes)"
