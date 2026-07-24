#!/usr/bin/env bash
# Build the toy_parser WASM plugin. Requires clang with a wasm32 target (LLVM 8+).
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
  -o toy_parser.wasm \
  toy_parser.c

echo "wrote toy_parser.wasm ($(wc -c < toy_parser.wasm) bytes)"
