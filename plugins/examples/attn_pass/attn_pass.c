/* plugins/examples/attn_pass/attn_pass.c — a minimal NetVis WASM PASS plugin.
 *
 * The directly-runnable example against the LIVE pass facet (shipped in v0.6.0):
 * a pass sees model STRUCTURE + the CostReport scalars and emits named metrics. It
 * can never read a weight buffer — the host has no such import (the zero-payload
 * thesis is a property of the import SET here).
 *
 * This pass emits "flops_per_node" = total_flops / node_count (known=1 when both
 * are available), a trivial derived metric to demonstrate the round-trip.
 *
 * Build:  ./build.sh   (needs clang with a wasm32 target; see README.md)
 */
#include "../../sdk/netvis_plugin.h"

#include <stdint.h>

NV_EXPORT("run")
int32_t run(void) {
  nv_arena_reset();
  double flops = nv_host_total_flops();
  int64_t nodes = nv_host_node_count();

  double per_node = 0.0;
  int32_t known = 0;
  if (nodes > 0 && flops > 0.0) {
    per_node = flops / (double)nodes;
    known = 1;
  }

  /* Emit the metric. The name lives in guest memory; the host copies it out
   * (bounds-checked) — no pointer to host state ever crosses. */
  static const char name[] = "flops_per_node";
  nv_host_emit_metric((int32_t)(intptr_t)name, (int32_t)(sizeof(name) - 1),
                      per_node, known);
  return 0;
}
