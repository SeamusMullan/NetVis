#!/bin/sh
# Launch the netvis MCP server for the Claude Code plugin.
#
# Prefer a binary built inside this checkout (either preset's output), fall
# back to one installed on PATH. The repo ships source, not binaries, so a
# fresh plugin install needs one `cmake --preset core-only && cmake --build
# --preset core-only` before first use -- or a system-wide netvis_mcp.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for candidate in "$ROOT/build/core/netvis_mcp" "$ROOT/build/release/netvis_mcp"; do
  if [ -x "$candidate" ]; then
    exec "$candidate"
  fi
done
if command -v netvis_mcp >/dev/null 2>&1; then
  exec netvis_mcp
fi
echo "netvis_mcp not found. Build it (cmake --preset core-only && cmake --build --preset core-only) or install netvis_mcp on PATH." >&2
exit 1
