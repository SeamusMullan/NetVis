/* plugins/sdk/netvis_plugin.h — NetVis plugin SDK: the FROZEN WASM wire ABI.
 *
 * Single freestanding C11 header (<stdint.h> only). ONE source of truth for the
 * op-handler and parser wire formats shared by the host (C++) and guest (.wasm).
 *
 * TWO COMPILE MODES:
 *   - PLUGIN build (__wasm__ defined): emits `extern` host-import declarations,
 *     the NV_EXPORT macro, and a tiny bump allocator for marshalling.
 *   - HOST-BRIDGE build (plain C++ from tests/test_sdk_abi.cpp): imports are
 *     #if'd out; only enums / wire structs / constants are visible, so a
 *     _Static_assert bridge can lock them against the real C++ headers.
 *
 * INVARIANTS (see docs/v0.6.x-followups-plan.md):
 *   - All offsets/lengths/i64 returns use int64_t/uint64_t, NEVER C `long`
 *     (32-bit on wasm32).  [§0.5]
 *   - Every guest-supplied count is capped here AND host-validated.  [§0.6]
 *   - No import returns a decoded weight buffer.  [inv 3]
 *   - Enums byte-match ir::DType (16, Unknown=15) and OpCategory (15, Other=14).
 */
#ifndef NETVIS_PLUGIN_H
#define NETVIS_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- ABI versions (mirror kOpHandlerAbiVersion / kParserPluginAbiVersion /
 * kPassPluginAbiVersion in the engine plugin headers — all 1). ------------- */
#define NETVIS_OP_ABI_VERSION     1u
#define NETVIS_PARSER_ABI_VERSION 1u
#define NETVIS_PASS_ABI_VERSION   1u

/* ---- Wire enums (EXACT match to the C++ enums; test_sdk_abi.cpp asserts). -- */
/* ir::DType — 16 enumerators, Unknown == 15. */
typedef enum {
  NV_DT_F32 = 0, NV_DT_F16 = 1, NV_DT_BF16 = 2, NV_DT_F64 = 3,
  NV_DT_I8 = 4,  NV_DT_I16 = 5, NV_DT_I32 = 6,  NV_DT_I64 = 7,
  NV_DT_U8 = 8,  NV_DT_U16 = 9, NV_DT_U32 = 10, NV_DT_U64 = 11,
  NV_DT_BOOL = 12, NV_DT_Q4 = 13, NV_DT_Q8 = 14, NV_DT_UNKNOWN = 15
} nv_dtype_t;

/* OpCategory — 15 enumerators, Other == 14 (LAST). */
typedef enum {
  NV_CAT_CONV = 0, NV_CAT_MATMUL = 1, NV_CAT_ACTIVATION = 2, NV_CAT_NORM = 3,
  NV_CAT_POOL = 4, NV_CAT_ELEMENTWISE = 5, NV_CAT_SHAPE = 6, NV_CAT_REDUCE = 7,
  NV_CAT_TENSOR = 8, NV_CAT_CONTROLFLOW = 9, NV_CAT_IO = 10, NV_CAT_ATTENTION = 11,
  NV_CAT_RECURRENT = 12, NV_CAT_QUANTIZE = 13, NV_CAT_OTHER = 14
} nv_category_t;

typedef uint32_t nv_strid_t;   /* interned string id (StringId::id) */

/* ---- Sandbox caps (mirror SandboxLimits + the new marshalling caps). ------ */
#define NV_MAX_MEMORY_PAGES 256u        /* 16 MiB linear-memory cap           */
#define NV_MAX_STEPS        200000000ull/* fuel: yield-check decrements       */
#define NV_NULL_GUARD_OFF   8u          /* offsets < 8 are the null guard      */
#define NV_MAX_RANK         8           /* max tensor rank marshalled          */
#define NV_MAX_NODE_IO      4096        /* max inputs/outputs per node         */
#define NV_MAX_ATTR_LEN     65536       /* max ints in an attr array           */
#define NV_MAX_INTERN_LEN   4096        /* max bytes of one interned string    */
#define NV_ERR_MSG_MAX      1024        /* max error-message bytes (1 KiB)     */
#define NV_SNIFF_HEAD       4096        /* parser sniff-window head bytes      */
#define NV_SNIFF_TAIL       4096        /* parser sniff-window tail bytes      */
#define NV_READ_CHUNK_CAP   4096        /* max bytes per host_read_range call  */

/* Entry-point status: 0 = answered/ok, nonzero = abstain -> honest-unknown. */
#define NV_STATUS_OK      0
#define NV_STATUS_ABSTAIN 1

/* =========================================================================
 * PLUGIN-BUILD SIDE: import declarations + export helpers (guest .wasm only).
 * Excluded from the host-bridge build so C++ sees only the shared types above.
 * ========================================================================= */
#if defined(__wasm__) && !defined(NETVIS_SDK_HOST_BRIDGE)

#define NV_IMPORT(module, name) \
  __attribute__((import_module(module), import_name(name)))
#define NV_EXPORT(name) __attribute__((export_name(name)))

/* ---- Op-handler facet — module "netvis_op" (Increment A §A.2). ------------
 * Read imports (host answers about the current op); result out-imports (guest
 * pushes its verdict). Sig chars audited vs m3_LinkRawFunctionEx: i=i32, I=i64,
 * f=f32, F=f64, *=i32 memory offset. */
NV_IMPORT("netvis_op","op_input_count")        int32_t nv_op_input_count(void);
NV_IMPORT("netvis_op","op_output_count")       int32_t nv_op_output_count(void);
NV_IMPORT("netvis_op","op_default_category")   int32_t nv_op_default_category(void);
NV_IMPORT("netvis_op","op_type")               int32_t nv_op_type(int32_t dst, int32_t cap);
NV_IMPORT("netvis_op","op_input_rank")         int32_t nv_op_input_rank(int32_t slot);
NV_IMPORT("netvis_op","op_output_rank")        int32_t nv_op_output_rank(int32_t slot);
NV_IMPORT("netvis_op","op_input_dims")         int32_t nv_op_input_dims(int32_t slot, int32_t dst, int32_t cap);
NV_IMPORT("netvis_op","op_output_dims")        int32_t nv_op_output_dims(int32_t slot, int32_t dst, int32_t cap);
NV_IMPORT("netvis_op","op_input_dtype")        int32_t nv_op_input_dtype(int32_t slot);
NV_IMPORT("netvis_op","op_output_dtype")       int32_t nv_op_output_dtype(int32_t slot);
NV_IMPORT("netvis_op","op_input_is_initializer") int32_t nv_op_input_is_initializer(int32_t slot);
NV_IMPORT("netvis_op","op_input_init_elem_count") int64_t nv_op_input_init_elem_count(int32_t slot);
NV_IMPORT("netvis_op","op_input_init_byte_len")   int64_t nv_op_input_init_byte_len(int32_t slot);
NV_IMPORT("netvis_op","op_input_init_dtype")      int32_t nv_op_input_init_dtype(int32_t slot);
NV_IMPORT("netvis_op","op_has_attr")           int32_t nv_op_has_attr(int32_t name_ptr, int32_t name_len);
NV_IMPORT("netvis_op","op_attr_int")           int64_t nv_op_attr_int(int32_t name_ptr, int32_t name_len, int64_t def);
NV_IMPORT("netvis_op","op_attr_float")         double  nv_op_attr_float(int32_t name_ptr, int32_t name_len, int32_t known_ptr);
NV_IMPORT("netvis_op","op_attr_string")        int32_t nv_op_attr_string(int32_t name_ptr, int32_t name_len, int32_t dst, int32_t cap);
NV_IMPORT("netvis_op","op_attr_ints")          int32_t nv_op_attr_ints(int32_t name_ptr, int32_t name_len, int32_t dst, int32_t cap);
NV_IMPORT("netvis_op","op_input_const_ints")   int32_t nv_op_input_const_ints(int32_t slot, int32_t dst, int32_t cap);
/* result out-imports */
NV_IMPORT("netvis_op","op_set_category")       void nv_op_set_category(int32_t cat);
NV_IMPORT("netvis_op","op_set_flops")          void nv_op_set_flops(int64_t value, int32_t known);
NV_IMPORT("netvis_op","op_set_output_shape")   void nv_op_set_output_shape(int32_t slot, int32_t dims_ptr, int32_t rank, int32_t dtype);
NV_IMPORT("netvis_op","op_set_color")          void nv_op_set_color(int32_t rgba);

/* ---- Parser facet — module "netvis" (Increment B §B.2). -------------------
 * Read (window-bounded, host-marked); model-mutating (append-only commands). */
NV_IMPORT("netvis","host_file_len")        int64_t nv_host_file_len(void);
NV_IMPORT("netvis","host_read_range")      int32_t nv_host_read_range(int64_t off, int32_t len, int32_t dst);
NV_IMPORT("netvis","host_intern_range")    int32_t nv_host_intern_range(int64_t off, int32_t len);
NV_IMPORT("netvis","host_set_model_info")  void nv_host_set_model_info(int32_t fmt_id, int32_t prod_id, int32_t ver_id);
NV_IMPORT("netvis","host_set_metadata")    void nv_host_set_metadata(int32_t key_id, int32_t val_id);
NV_IMPORT("netvis","host_set_error")       void nv_host_set_error(int32_t msg_ptr, int32_t len);
NV_IMPORT("netvis","host_begin_graph")     int32_t nv_host_begin_graph(int32_t name_id);
NV_IMPORT("netvis","host_add_value")       int32_t nv_host_add_value(int32_t g, int32_t name_id, int32_t dtype, int32_t dims_ptr, int32_t rank);
NV_IMPORT("netvis","host_add_node")        int32_t nv_host_add_node(int32_t g, int32_t op_id, int32_t name_id, int32_t in_ptr, int32_t nin, int32_t out_ptr, int32_t nout);
NV_IMPORT("netvis","host_set_graph_io")    void nv_host_set_graph_io(int32_t g, int32_t in_ptr, int32_t nin, int32_t out_ptr, int32_t nout);
NV_IMPORT("netvis","host_add_attr_int")    int32_t nv_host_add_attr_int(int32_t g, int32_t node, int32_t name_id, int64_t v);
NV_IMPORT("netvis","host_add_attr_float")  int32_t nv_host_add_attr_float(int32_t g, int32_t node, int32_t name_id, double v);
NV_IMPORT("netvis","host_add_attr_string") int32_t nv_host_add_attr_string(int32_t g, int32_t node, int32_t name_id, int32_t val_id);
NV_IMPORT("netvis","host_add_attr_ints")   int32_t nv_host_add_attr_ints(int32_t g, int32_t node, int32_t name_id, int32_t ptr, int32_t count);
NV_IMPORT("netvis","host_record_tensor")   int32_t nv_host_record_tensor(int32_t g, int32_t name_id, int64_t off, int64_t len, int32_t dtype, int32_t dims_ptr, int32_t rank);

/* ---- Pass facet — module "netvis" (v0.6.0, already shipped). -------------- */
NV_IMPORT("netvis","host_node_count")   int64_t nv_host_node_count(void);
NV_IMPORT("netvis","host_total_flops")  double  nv_host_total_flops(void);
NV_IMPORT("netvis","host_total_params") double  nv_host_total_params(void);
NV_IMPORT("netvis","host_emit_metric")  void    nv_host_emit_metric(int32_t name_ptr, int32_t name_len, double value, int32_t known);

/* ---- Tiny bump allocator: guests marshal small dims/name buffers without a
 * full libc. Backed by a static arena; reset() before each entry call. ------ */
#ifndef NV_ARENA_BYTES
#define NV_ARENA_BYTES 4096
#endif
static unsigned char nv__arena[NV_ARENA_BYTES];
static uint32_t nv__arena_top = 0;
static inline void  nv_arena_reset(void) { nv__arena_top = 0; }
static inline void* nv_alloc(uint32_t n) {
  n = (n + 7u) & ~7u;                       /* 8-byte align */
  if (nv__arena_top + n > (uint32_t)NV_ARENA_BYTES) return 0;
  void* p = &nv__arena[nv__arena_top];
  nv__arena_top += n;
  return p;
}

#endif /* __wasm__ && !NETVIS_SDK_HOST_BRIDGE */

/* =========================================================================
 * Shared packed wire structs + static asserts (BOTH sides compile these).
 * A packing divergence (i64/f64 alignment, tail padding) fails at compile on
 * the wasm32 plugin toolchain AND the host toolchain. [§D.1]
 * ========================================================================= */

/* A marshalled tensor-record command payload (guest -> host_record_tensor is
 * passed by scalar args, but this struct documents the canonical field order &
 * is used by the host-side decoder for the (dims_ptr,rank) block). */
typedef struct {
  int64_t off;       /* file offset of the payload / MIL blob header */
  int64_t len;       /* byte length (0 => host derives, e.g. blob-indirect) */
  int32_t dtype;     /* nv_dtype_t */
  int32_t rank;      /* [0, NV_MAX_RANK] */
} nv_tensor_hdr_t;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(nv_tensor_hdr_t) == 24, "nv_tensor_hdr_t packing drift");
_Static_assert(NV_DT_UNKNOWN == 15, "nv_dtype_t must match ir::DType (Unknown=15)");
_Static_assert(NV_CAT_OTHER == 14, "nv_category_t must match OpCategory (Other=14)");
#elif defined(__cplusplus) && __cplusplus >= 201103L
static_assert(sizeof(nv_tensor_hdr_t) == 24, "nv_tensor_hdr_t packing drift");
static_assert((int)NV_DT_UNKNOWN == 15, "nv_dtype_t must match ir::DType (Unknown=15)");
static_assert((int)NV_CAT_OTHER == 14, "nv_category_t must match OpCategory (Other=14)");
#endif

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* NETVIS_PLUGIN_H */
