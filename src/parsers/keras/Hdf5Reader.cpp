// parsers/keras/Hdf5Reader.cpp — implementation of the minimal HDF5 reader.
//
// See Hdf5Reader.h for scope + invariants. This walks the CLASSIC group form:
//   superblock (v0/v1) -> root symbol-table entry -> root object header
//   -> Symbol Table Message (v1 B-tree "TREE" + local heap "HEAP")
//   -> B-tree leaves -> symbol-table nodes "SNOD" -> per-object headers
//   -> dataset object headers (Dataspace + Datatype + Data Layout messages).
//
// Every field read goes through a bounds-checked random-access helper (Ra); an
// out-of-range read simply fails the current step and appends an honest note,
// so a truncated/adversarial file yields a partial WalkResult, never a crash.
// Object-header / B-tree / SNOD addresses are cycle-guarded via a visited set,
// and depth / dataset / message / symbol counts are all capped. Dataset PAYLOAD
// bytes are never touched — only offsets+lengths are recorded.
#include "parsers/keras/Hdf5Reader.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

namespace netvis::hdf5 {
namespace {

constexpr uint64_t kUndef = UINT64_MAX;              // HDF5 "undefined address"
constexpr int      kMaxDepth = 64;                   // group nesting cap
constexpr uint32_t kMaxDatasets = 100000;            // total datasets cap
constexpr uint32_t kMaxMessages = 4096;              // messages per object header
constexpr uint32_t kMaxSymbols = 65535;              // symbols per SNOD
constexpr uint32_t kMaxBtreeEntries = 65535;         // entries per B-tree node
constexpr uint32_t kMaxNodesVisited = 200000;        // global node-visit budget
constexpr size_t   kMaxNameLen = 4096;               // per-name cap
constexpr size_t   kMaxPathLen = 65536;              // joined-path cap

// Bounded random-access reader over [d, d+n). All accessors return false on any
// out-of-range access instead of reading; this is the single choke point that
// makes the walk hostile-input-safe. These are STRUCTURAL reads only — they
// never call ByteReader::mark_payload_read, so the payload counter stays 0.
struct Ra {
  const uint8_t* d = nullptr;
  uint64_t n = 0;

  bool u8(uint64_t off, uint8_t& out) const {
    if (off >= n) return false;
    out = d[off];
    return true;
  }
  bool u16(uint64_t off, uint16_t& out) const {
    if (off + 2 > n) return false;
    uint16_t v; std::memcpy(&v, d + off, 2); out = v; return true;
  }
  bool u32(uint64_t off, uint32_t& out) const {
    if (off + 4 > n) return false;
    uint32_t v; std::memcpy(&v, d + off, 4); out = v; return true;
  }
  bool u64(uint64_t off, uint64_t& out) const {
    if (off + 8 > n) return false;
    uint64_t v; std::memcpy(&v, d + off, 8); out = v; return true;
  }
  // Read a `size`-byte little-endian unsigned value (HDF5 "size of offsets" /
  // "size of lengths" is 2/4/8). Undefined (all-0xFF) stays all-0xFF so callers
  // can compare against kUndef when size==8.
  bool uvar(uint64_t off, uint8_t size, uint64_t& out) const {
    if (size == 0 || size > 8 || off + size > n) return false;
    uint64_t v = 0;
    for (uint8_t i = 0; i < size; ++i) v |= static_cast<uint64_t>(d[off + i]) << (8 * i);
    // Sign-extend an all-ones (undefined) marker to the full 64-bit sentinel.
    if (size < 8) {
      uint64_t all_ones = (size == 8) ? kUndef : ((1ULL << (8 * size)) - 1);
      if (v == all_ones) v = kUndef;
    }
    out = v;
    return true;
  }
  // Null-terminated string at `off`, capped. Returns "" on overrun.
  std::string cstr(uint64_t off) const {
    std::string s;
    while (off < n && s.size() < kMaxNameLen) {
      char c = static_cast<char>(d[off++]);
      if (c == '\0') break;
      s.push_back(c);
    }
    return s;
  }
};

// Superblock coordinates we care about after locating + validating it.
struct Superblock {
  bool ok = false;
  uint8_t size_offsets = 8;
  uint8_t size_lengths = 8;
  uint64_t base_address = 0;
  uint64_t root_oh_addr = kUndef;   // root group object-header address
};

const uint8_t kSig[8] = {0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};

// Locate + parse the superblock at an aligned offset (0, 512, 1024, ...).
Superblock read_superblock(const Ra& ra, WalkResult& out) {
  Superblock sb;
  uint64_t sig_at = kUndef;
  const uint64_t offs[] = {0, 512, 1024, 2048, 4096, 8192};
  for (uint64_t o : offs) {
    if (o + 8 <= ra.n && std::memcmp(ra.d + o, kSig, 8) == 0) { sig_at = o; break; }
  }
  if (sig_at == kUndef) return sb;  // no superblock -> caller notes it
  out.superblock_found = true;

  uint8_t version = 0;
  if (!ra.u8(sig_at + 8, version)) return sb;
  if (!ra.u8(sig_at + 13, sb.size_offsets)) return sb;
  if (!ra.u8(sig_at + 14, sb.size_lengths)) return sb;
  if (sb.size_offsets == 0 || sb.size_offsets > 8 ||
      sb.size_lengths == 0 || sb.size_lengths > 8) {
    out.notes.push_back("HDF5: unsupported offset/length size");
    return sb;
  }

  // v0/v1 fixed portion: after the 8-byte "consistency flags" region the file
  // carries base/freespace/eof/driver addresses, then the root symbol-table
  // entry. v1 inserts 4 extra bytes (indexed-storage node K + reserved) before
  // those addresses; anything newer (v2/v3) uses a different layout we skip.
  uint64_t addrs_at;
  if (version == 0) {
    addrs_at = sig_at + 24;
  } else if (version == 1) {
    addrs_at = sig_at + 28;
  } else {
    out.notes.push_back("HDF5: superblock v2+/paged layout not supported (best-effort skip)");
    return sb;
  }

  const uint8_t so = sb.size_offsets;
  if (!ra.uvar(addrs_at, so, sb.base_address)) return sb;
  // addrs: base(0), free-space(1), eof(2), driver(3); then root STE.
  uint64_t root_ste = addrs_at + 4ULL * so;
  // Symbol Table Entry: link-name-offset(so), object-header-address(so), ...
  uint64_t oh_addr;
  if (!ra.uvar(root_ste + so, so, oh_addr)) return sb;
  sb.root_oh_addr = oh_addr;
  sb.ok = (oh_addr != kUndef);
  return sb;
}

// Parsed subset of an object header.
struct ObjectHeader {
  bool ok = false;
  bool is_group = false;
  uint64_t btree_addr = kUndef;     // group: v1 B-tree
  uint64_t heap_addr = kUndef;      // group: local heap
  // dataset messages:
  bool has_dataspace = false, has_datatype = false, has_layout = false;
  std::vector<int64_t> shape;
  ir::DType dtype = ir::DType::Unknown;
  bool chunked = false;
  uint64_t data_addr = kUndef;
  uint64_t data_size = 0;
};

// --- individual message parsers (operate on the message DATA sub-range) ------

// Dataspace message (v1): version, rank, flags, then rank * length dims.
void parse_dataspace(const Ra& ra, uint64_t off, uint64_t size, uint8_t so,
                     ObjectHeader& oh) {
  (void)so;
  uint8_t ver = 0, rank = 0, flags = 0;
  if (!ra.u8(off, ver) || !ra.u8(off + 1, rank) || !ra.u8(off + 2, flags)) return;
  uint64_t p;
  if (ver == 1) {
    p = off + 8;               // v1: 8-byte fixed header (incl. reserved(1)+reserved(4))
  } else if (ver == 2) {
    p = off + 4;               // v2: version, rank, flags, type
  } else {
    return;                    // unknown dataspace version
  }
  oh.has_dataspace = true;
  for (uint8_t i = 0; i < rank && i < 32; ++i) {
    uint64_t dim;
    if (p + 8 > off + size) break;
    if (!ra.u64(p, dim)) break;
    // Guard absurd dims; HDF5 unlimited dims use all-ones -> mark dynamic (-1).
    if (dim == kUndef || dim > (1ULL << 62)) oh.shape.push_back(-1);
    else oh.shape.push_back(static_cast<int64_t>(dim));
    p += 8;
  }
}

// Datatype message: class+version byte, 3-byte bitfield, 4-byte size, props.
void parse_datatype(const Ra& ra, uint64_t off, uint64_t size, ObjectHeader& oh) {
  (void)size;
  uint8_t cv = 0;
  if (!ra.u8(off, cv)) return;
  uint8_t cls = cv & 0x0F;
  uint8_t bitfield0 = 0;
  ra.u8(off + 1, bitfield0);   // low 8 bits of the 24-bit class bit field
  uint32_t dsize = 0;
  if (!ra.u32(off + 4, dsize)) return;
  oh.has_datatype = true;
  using D = ir::DType;
  if (cls == 1) {              // floating-point
    if (dsize == 4) oh.dtype = D::F32;
    else if (dsize == 2) oh.dtype = D::F16;
    else if (dsize == 8) oh.dtype = D::F64;
  } else if (cls == 0) {       // fixed-point (integer)
    bool is_signed = (bitfield0 & 0x08) != 0;   // bit 3 = signed
    if (dsize == 1) oh.dtype = is_signed ? D::I8 : D::U8;
    else if (dsize == 2) oh.dtype = is_signed ? D::I16 : D::U16;
    else if (dsize == 4) oh.dtype = is_signed ? D::I32 : D::U32;
    else if (dsize == 8) oh.dtype = is_signed ? D::I64 : D::U64;
  }
  // Other classes (string/compound/etc.) -> DType::Unknown (honest).
}

// Data Layout message (v3): version, class; contiguous -> (address,size),
// chunked -> mark chunked, compact -> inline (address = message-relative).
void parse_layout(const Ra& ra, uint64_t off, uint64_t size, uint8_t so,
                  uint8_t sl, ObjectHeader& oh) {
  (void)size;
  uint8_t ver = 0;
  if (!ra.u8(off, ver)) return;
  oh.has_layout = true;
  if (ver == 3) {
    uint8_t cls = 0;
    if (!ra.u8(off + 1, cls)) return;
    if (cls == 1) {            // contiguous
      uint64_t addr, len;
      if (ra.uvar(off + 2, so, addr) && ra.uvar(off + 2 + so, sl, len)) {
        oh.data_addr = addr;
        oh.data_size = len;
      }
    } else if (cls == 2) {     // chunked -> payload index, not stitched
      oh.chunked = true;
    } else if (cls == 0) {     // compact -> inline size(2) + data
      // Payload is inside the header; not linearly addressable as a weight blob.
      oh.chunked = false;      // leave data_addr undefined; recorded as no-offset
    }
  } else if (ver == 1 || ver == 2) {
    // Older layout messages: dimensionality(1), class(1), then addr for
    // contiguous. Best-effort: try class at off+2, address at off+8.
    uint8_t cls = 0;
    ra.u8(off + 2, cls);
    if (cls == 1) {
      uint64_t addr;
      if (ra.uvar(off + 8, so, addr)) oh.data_addr = addr;
      // size in these versions is derived from dims*elem; leave data_size 0 and
      // let the caller compute from shape*dtype if needed.
    } else if (cls == 2) {
      oh.chunked = true;
    }
  }
}

// Parse a run of header messages in [msg_start, msg_start+msg_len). Follows
// Object Header Continuation (type 0x10) blocks up to the visit budget.
struct WalkState {
  const Ra& ra;
  uint8_t so, sl;
  std::unordered_set<uint64_t>& visited;
  uint32_t& nodes;
  WalkResult& out;
};

void parse_message_block(WalkState& st, ObjectHeader& oh, uint64_t msg_start,
                         uint64_t msg_len, uint32_t& msg_budget, int cont_depth);

void handle_message(WalkState& st, ObjectHeader& oh, uint16_t type, uint64_t data,
                    uint64_t dsize, uint32_t& msg_budget, int cont_depth) {
  switch (type) {
    case 0x0001: parse_dataspace(st.ra, data, dsize, st.so, oh); break;
    case 0x0003: parse_datatype(st.ra, data, dsize, oh); break;
    case 0x0008: parse_layout(st.ra, data, dsize, st.so, st.sl, oh); break;
    case 0x0011: {  // Symbol Table Message -> this object is a group
      uint64_t bt, hp;
      if (st.ra.uvar(data, st.so, bt) && st.ra.uvar(data + st.so, st.so, hp)) {
        oh.is_group = true;
        oh.btree_addr = bt;
        oh.heap_addr = hp;
      }
      break;
    }
    case 0x0010: {  // Object Header Continuation -> (offset, length) block
      if (cont_depth >= 8) break;  // bounded continuation chain
      uint64_t coff, clen;
      if (st.ra.uvar(data, st.so, coff) && st.ra.uvar(data + st.so, st.sl, clen)) {
        parse_message_block(st, oh, coff, clen, msg_budget, cont_depth + 1);
      }
      break;
    }
    default: break;  // NIL / fill-value / attribute / etc. -> ignored
  }
}

void parse_message_block(WalkState& st, ObjectHeader& oh, uint64_t msg_start,
                         uint64_t msg_len, uint32_t& msg_budget, int cont_depth) {
  uint64_t p = msg_start;
  const uint64_t end = msg_start + msg_len;
  while (p + 8 <= end && msg_budget > 0) {
    --msg_budget;
    uint16_t type = 0, dsize = 0;
    if (!st.ra.u16(p, type) || !st.ra.u16(p + 2, dsize)) break;
    uint64_t data = p + 8;               // after 8-byte message header
    if (data + dsize > end) break;       // message data overruns the block
    handle_message(st, oh, type, data, dsize, msg_budget, cont_depth);
    p = data + dsize;                    // message data is already 8-byte padded
  }
}

// Parse a v1 object header at `addr`. Cycle-guarded via st.visited.
ObjectHeader parse_object_header(WalkState& st, uint64_t addr) {
  ObjectHeader oh;
  if (addr == kUndef || ++st.nodes > kMaxNodesVisited) return oh;
  if (!st.visited.insert(addr).second) return oh;  // already visited -> cycle

  uint8_t version = 0;
  if (!st.ra.u8(addr, version)) return oh;
  if (version != 1) {
    // v2 object headers ("OHDR" magic) are the newer form; not supported.
    st.out.notes.push_back("HDF5: v2 object header not supported (best-effort skip)");
    return oh;
  }
  uint16_t num_msgs = 0;
  uint32_t hdr_size = 0;
  if (!st.ra.u16(addr + 2, num_msgs)) return oh;
  if (!st.ra.u32(addr + 8, hdr_size)) return oh;
  if (num_msgs > kMaxMessages) num_msgs = kMaxMessages;

  uint32_t msg_budget = num_msgs;
  parse_message_block(st, oh, addr + 16, hdr_size, msg_budget, 0);
  oh.ok = true;
  return oh;
}

// Forward decl (B-tree <-> object-header recursion).
void walk_object(WalkState& st, uint64_t oh_addr, const std::string& path,
                 int depth);

// Read the local-heap data-segment base address.
bool read_heap_data_seg(const Ra& ra, uint64_t heap_addr, uint8_t so, uint8_t sl,
                        uint64_t& data_seg_out) {
  if (heap_addr == kUndef || heap_addr + 4 > ra.n) return false;
  if (std::memcmp(ra.d + heap_addr, "HEAP", 4) != 0) return false;
  // version(1)+reserved(3) = 4; data-seg size(sl); free-list off(sl); data-seg addr(so)
  uint64_t seg;
  if (!ra.uvar(heap_addr + 8 + 2ULL * sl, so, seg)) return false;
  data_seg_out = seg;
  return true;
}

// Walk one SNOD (symbol-table node): each entry names a child object header.
void walk_snod(WalkState& st, uint64_t snod_addr, uint64_t heap_data_seg,
               const std::string& path, int depth) {
  if (snod_addr == kUndef || ++st.nodes > kMaxNodesVisited) return;
  if (!st.visited.insert(snod_addr | (1ULL << 63)).second) return;  // tag-space for SNOD
  if (snod_addr + 8 > st.ra.n) return;
  if (std::memcmp(st.ra.d + snod_addr, "SNOD", 4) != 0) return;
  uint16_t nsym = 0;
  if (!st.ra.u16(snod_addr + 6, nsym)) return;
  if (nsym > kMaxSymbols) nsym = kMaxSymbols;

  const uint8_t so = st.so;
  // Symbol Table Entry size: link-name-off(so) + oh-addr(so) + cache-type(4)
  //   + reserved(4) + scratch-pad(16).
  const uint64_t ste_size = 2ULL * so + 4 + 4 + 16;
  uint64_t p = snod_addr + 8;
  for (uint16_t i = 0; i < nsym; ++i) {
    uint64_t name_off, child_oh;
    if (!st.ra.uvar(p, so, name_off) || !st.ra.uvar(p + so, so, child_oh)) break;
    std::string name;
    if (heap_data_seg != kUndef) name = st.ra.cstr(heap_data_seg + name_off);
    std::string child_path = name.empty()
        ? path
        : (path.empty() ? name : path + "/" + name);
    if (child_path.size() > kMaxPathLen) child_path.resize(kMaxPathLen);
    walk_object(st, child_oh, child_path, depth + 1);
    p += ste_size;
  }
}

// Walk a v1 B-tree of group nodes. type 0 = group; level 0 = leaves -> SNOD.
void walk_group_btree(WalkState& st, uint64_t btree_addr, uint64_t heap_data_seg,
                      const std::string& path, int depth) {
  if (btree_addr == kUndef || ++st.nodes > kMaxNodesVisited) return;
  if (!st.visited.insert(btree_addr | (1ULL << 62)).second) return;  // tag-space for TREE
  if (btree_addr + 8 > st.ra.n) return;
  if (std::memcmp(st.ra.d + btree_addr, "TREE", 4) != 0) return;
  uint8_t node_type = 0, node_level = 0;
  uint16_t entries = 0;
  if (!st.ra.u8(btree_addr + 4, node_type) || !st.ra.u8(btree_addr + 5, node_level) ||
      !st.ra.u16(btree_addr + 6, entries))
    return;
  if (node_type != 0) return;  // type 1 = chunked raw-data B-tree: not walked
  if (entries > kMaxBtreeEntries) entries = kMaxBtreeEntries;

  const uint8_t so = st.so, sl = st.sl;
  // Header: sig(4)+type(1)+level(1)+entries(2)+left-sib(so)+right-sib(so).
  // Then interleaved: key[0], child[0], key[1], child[1], ..., key[entries].
  // Group-node keys are length-sized (heap offsets); children are offset-sized.
  uint64_t p = btree_addr + 8 + 2ULL * so;
  for (uint16_t i = 0; i < entries; ++i) {
    p += sl;                    // skip key[i]
    uint64_t child;
    if (!st.ra.uvar(p, so, child)) break;
    p += so;
    if (node_level == 0) walk_snod(st, child, heap_data_seg, path, depth);
    else walk_group_btree(st, child, heap_data_seg, path, depth);  // sub-tree
  }
}

// Dispatch one object: group -> recurse into its B-tree; dataset -> record.
void walk_object(WalkState& st, uint64_t oh_addr, const std::string& path,
                 int depth) {
  if (depth > kMaxDepth) return;
  if (st.out.datasets.size() >= kMaxDatasets) return;
  ObjectHeader oh = parse_object_header(st, oh_addr);
  if (!oh.ok) return;

  if (oh.is_group) {
    uint64_t heap_data_seg = kUndef;
    read_heap_data_seg(st.ra, oh.heap_addr, st.so, st.sl, heap_data_seg);
    walk_group_btree(st, oh.btree_addr, heap_data_seg, path, depth);
    return;
  }

  // Dataset: needs a dataspace + datatype + layout to be meaningful.
  if (oh.has_dataspace || oh.has_datatype || oh.has_layout) {
    Dataset ds;
    ds.name = path;
    ds.dtype = oh.dtype;
    ds.shape = oh.shape;
    if (oh.chunked) {
      ds.chunked = true;
      ds.file_offset = kUndef;
      ds.byte_len = 0;
    } else if (oh.data_addr != kUndef) {
      ds.file_offset = oh.data_addr;   // absolute-in-range; base added by caller
      ds.byte_len = oh.data_size;
    } else {
      ds.file_offset = kUndef;         // compact/undefined -> not addressable
      ds.byte_len = 0;
    }
    st.out.datasets.push_back(std::move(ds));
  }
}

}  // namespace

WalkResult walk(const uint8_t* data, uint64_t size, uint64_t base_file_offset) {
  WalkResult out;
  if (data == nullptr || size < 16) {
    out.notes.push_back("HDF5: file too small / unmapped");
    return out;
  }
  Ra ra{data, size};
  Superblock sb = read_superblock(ra, out);
  if (!out.superblock_found) {
    out.notes.push_back("HDF5: superblock signature not found");
    return out;
  }
  if (!sb.ok) {
    out.notes.push_back("HDF5: unsupported/invalid superblock (best-effort)");
    return out;
  }

  std::unordered_set<uint64_t> visited;
  uint32_t nodes = 0;
  WalkState st{ra, sb.size_offsets, sb.size_lengths, visited, nodes, out};
  walk_object(st, sb.root_oh_addr, "", 0);

  // Rebase recorded dataset offsets to absolute positions in the OUTER mmap.
  if (base_file_offset != 0) {
    for (auto& ds : out.datasets) {
      if (ds.file_offset != kUndef) ds.file_offset += base_file_offset;
    }
  }
  return out;
}

}  // namespace netvis::hdf5
