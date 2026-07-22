// parsers/keras/Hdf5Reader.h — a MINIMAL, hostile-input-safe HDF5 reader.
//
// SCOPE (issue #42, spec §6): this is NOT a general HDF5 library. It implements
// only the *structure-discovery* subset needed to surface datasets as tensors:
// locate the superblock, walk the classic symbol-table group form (v1 object
// headers + v1 B-tree "TREE" + local heap "HEAP" + symbol-table node "SNOD"),
// and for each dataset read its dataspace (dims), datatype (class+size ->
// ir::DType) and data-layout messages. A CONTIGUOUS layout yields a single
// (offset,size) into the file -> recorded as a TensorRef offset+len, the bytes
// never touched. A CHUNKED layout is NOT stitched (the chunk B-tree is a payload
// index we deliberately do not walk): the dataset is recorded with
// file_offset == UINT64_MAX and marked `chunked` (honest, no fabricated offset).
//
// EXPLICITLY OUT OF SCOPE (best-effort / noted, never a crash):
//   - the newer link-message / "fractal heap" group form (libver='latest');
//   - chunked-payload stitching;
//   - data-layout message versions 1/2 (address recorded best-effort only).
//
// INVARIANTS: every read goes through a bounds-checked ByteReader; object-header
// addresses and B-tree nodes are cycle-guarded with `visited` sets; depth,
// message-count, dataset-count and name-length are all capped. Truncated/garbage
// input yields a partial/empty result plus honest notes — never a crash, hang,
// or out-of-bounds read. Dataset PAYLOAD bytes are never read, so a parse leaves
// ByteReader::payload_read_counter() at 0.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ir/IR.h"

namespace netvis::hdf5 {

// One discovered dataset. `file_offset` is ABSOLUTE into the outer memory-mapped
// file (see walk()'s base_file_offset); UINT64_MAX when the payload is not
// linearly addressable (chunked, external, or undefined-address).
struct Dataset {
  std::string name;                    // group-path-joined, '/'-separated
  ir::DType dtype = ir::DType::Unknown;
  std::vector<int64_t> shape;          // -1 for unlimited/max dims
  uint64_t file_offset = UINT64_MAX;
  uint64_t byte_len = 0;
  bool chunked = false;                // true -> payload not linearly addressable
};

// The result of a bounded structural walk. Always returned (never throws); on
// malformed input `datasets` may be empty and `notes` explains what was skipped.
struct WalkResult {
  std::vector<Dataset> datasets;
  std::vector<std::string> notes;      // honest limitations encountered
  bool superblock_found = false;
};

// Walk the HDF5 structure in the byte range [data, data+size). `base_file_offset`
// is the offset of THIS range within the outer memory-mapped file (0 for a raw
// .h5; the ZIP entry payload offset for an embedded model.weights.h5). It is
// added to every recorded dataset file_offset so refs are absolute into the
// outer mmap. Structure-only, bounded, cycle-guarded; never reads payload bytes.
WalkResult walk(const uint8_t* data, uint64_t size, uint64_t base_file_offset);

}  // namespace netvis::hdf5
