/* plugins/examples/toy_parser/toy_parser.c — a minimal NetVis WASM PARSER plugin.
 *
 * Demonstrates the zero-payload thesis for the parser facet (#10, Increment B):
 * we read the file's small TOC through the bounded, host-marked sniff window
 * (host_read_range), and record each weight as an (offset, len) TensorRef via
 * host_record_tensor — we NEVER read a weight byte. The host owns the ir::Model;
 * this plugin only issues append-only commands.
 *
 * Toy format "NVTOY1":
 *   magic  "NVTOY1\0\0"                (8 bytes)
 *   u32    n_tensors                   (LE)
 *   per tensor: u32 name_off, u32 name_len, u32 dtype, u64 data_off, u64 byte_len
 *   ... names + payload follow (payload is NEVER read here).
 *
 * Build:  ./build.sh   (needs clang with a wasm32 target; see README.md)
 */
#include "../../sdk/netvis_plugin.h"

#include <stdint.h>

/* Read a little-endian u32 from a 4-byte guest buffer. */
static uint32_t rd_u32(const unsigned char* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
         ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

NV_EXPORT("netvis_parser_abi_version")
int32_t netvis_parser_abi_version(void) { return NETVIS_PARSER_ABI_VERSION; }

/* Claim the file iff the first 8 bytes are the NVTOY1 magic (read via the host,
 * inside the sniff window). */
NV_EXPORT("netvis_can_parse")
int32_t netvis_can_parse(void) {
  nv_arena_reset();
  unsigned char* buf = (unsigned char*)nv_alloc(8);
  if (!buf) return 0;
  int32_t n = nv_host_read_range(0, 8, (int32_t)(intptr_t)buf);
  if (n != 8) return 0;
  const unsigned char m[8] = {'N', 'V', 'T', 'O', 'Y', '1', 0, 0};
  for (int i = 0; i < 8; ++i) if (buf[i] != m[i]) return 0;
  return 1;
}

NV_EXPORT("netvis_parse")
int32_t netvis_parse(void) {
  nv_arena_reset();
  unsigned char* hdr = (unsigned char*)nv_alloc(12);
  if (!hdr) return 1;
  /* Read magic(8) + n_tensors(4) from the head window. */
  if (nv_host_read_range(0, 12, (int32_t)(intptr_t)hdr) != 12) return 1;
  uint32_t n = rd_u32(hdr + 8);
  if (n > 4096) n = 4096;               /* keep the demo bounded */

  nv_host_set_model_info(-1, -1, -1);   /* format label defaulted by the host */
  int32_t g = nv_host_begin_graph(-1);
  if (g < 0) return 1;

  const int32_t REC = 24;               /* bytes per TOC record */
  for (uint32_t i = 0; i < n; ++i) {
    unsigned char* rec = (unsigned char*)nv_alloc(REC);
    if (!rec) return 1;
    int64_t rec_off = 12 + (int64_t)i * REC;
    if (nv_host_read_range(rec_off, REC, (int32_t)(intptr_t)rec) != REC) break;
    uint32_t name_off = rd_u32(rec + 0);
    uint32_t name_len = rd_u32(rec + 4);
    uint32_t dtype    = rd_u32(rec + 8);
    /* data_off(u64) + byte_len(u64) at rec+12, rec+20 wrapped in a helper: */
    int64_t data_off  = (int64_t)rd_u32(rec + 12) | ((int64_t)rd_u32(rec + 16) << 32);
    int64_t byte_len  = (int64_t)rd_u32(rec + 20);   /* demo: <4GB tensors */
    /* Intern the name the HOST reads from its window (never from guest memory). */
    int32_t name_id = nv_host_intern_range((int64_t)name_off, (int32_t)name_len);
    /* Record the weight by offset+len ONLY — zero payload bytes read. */
    nv_host_record_tensor(g, name_id, data_off, byte_len, (int32_t)dtype, 0, 0);
  }
  return 0;   /* 0 = ok */
}
