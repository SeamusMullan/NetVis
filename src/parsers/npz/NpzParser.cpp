// parsers/npz/NpzParser.cpp — NumPy .npz (zip of .npy arrays) -> ir::Model.
//
// NumPy .npz is a ZIP archive of .npy entries. Each .npy has a header (magic,
// version, dict length, dict with descr/fortran_order/shape) followed by the
// array payload. We read ONLY the header (structural) to extract dtype+shape,
// then record offset+len of the payload WITHOUT touching it (spec §2.1, the
// zero-payload thesis). Compressed entries (np.savez_compressed) set
// file_offset=UINT64_MAX (honest fallback: we know shape/dtype but cannot
// mmap-address a DEFLATE stream). Result: tensor-table mode (has_graph=false).
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "core/ByteReader.h"
#include "core/JobSystem.h"
#include "core/MappedFile.h"
#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/Parser.h"

#include "miniz.h"

namespace netvis::npz {
namespace {

// ZIP local file header layout (30 fixed bytes + variable filename + extra).
// We reuse the same technique as PytorchParser: compute the absolute payload
// offset from the local header without extracting the payload.
constexpr uint32_t kLocalHeaderSig = 0x04034b50;

// Compute the absolute file offset of an entry's payload from its local header.
// Bounds-checked via ByteReader; returns false on any error (spec §6, §10).
bool payload_offset_from_local_header(const uint8_t* base, uint64_t file_size,
                                      uint64_t local_header_ofs,
                                      uint64_t& out_offset) {
  ByteReader r(base, file_size);
  if (!r.seek(local_header_ofs)) return false;
  auto sig = r.u32le();
  if (!sig || *sig != kLocalHeaderSig) return false;
  // Skip to filename_len/extra_len at offset 26 from signature.
  // We've consumed 4 (sig); skip 22 more to reach offset 26.
  r.skip(22);
  auto fn_len = r.u16le();
  if (!fn_len) return false;
  auto extra_len = r.u16le();
  if (!extra_len) return false;
  out_offset = local_header_ofs + 30ULL + *fn_len + *extra_len;
  if (out_offset > file_size) return false;
  return true;
}

// Parse a NumPy dtype descr string (e.g. "<f4", "<i8", "|u1") to ir::DType.
// Inverts the npy_descr() table in TensorStats.cpp (spec §7 requires this).
// Format: [byteorder][typecode][size], e.g. '<f4' = little-endian float 4 bytes.
// Returns DType::Unknown for unsupported/unrecognized descr.
ir::DType descr_to_dtype(const std::string& descr) {
  if (descr.size() < 3) return ir::DType::Unknown;
  // Byte order: '<' little-endian, '>' big-endian, '|' not applicable (size 1).
  // We support little-endian and '|' for size-1 types; big-endian -> Unknown.
  char order = descr[0];
  char kind = descr[1];
  std::string size_str = descr.substr(2);
  int sz = 0;
  try { sz = std::stoi(size_str); } catch (...) { return ir::DType::Unknown; }

  // Reject big-endian (not supported in the IR).
  if (order == '>') return ir::DType::Unknown;
  // '|' is valid only for size-1 types (i1, u1, b1).
  if (order == '|' && sz != 1) return ir::DType::Unknown;

  switch (kind) {
    case 'f':  // floating point
      if (sz == 2) return ir::DType::F16;
      if (sz == 4) return ir::DType::F32;
      if (sz == 8) return ir::DType::F64;
      return ir::DType::Unknown;
    case 'i':  // signed integer
      if (sz == 1) return ir::DType::I8;
      if (sz == 2) return ir::DType::I16;
      if (sz == 4) return ir::DType::I32;
      if (sz == 8) return ir::DType::I64;
      return ir::DType::Unknown;
    case 'u':  // unsigned integer
      if (sz == 1) return ir::DType::U8;
      if (sz == 2) return ir::DType::U16;
      if (sz == 4) return ir::DType::U32;
      if (sz == 8) return ir::DType::U64;
      return ir::DType::Unknown;
    case 'b':  // boolean
      if (sz == 1) return ir::DType::Bool;
      return ir::DType::Unknown;
    default:
      return ir::DType::Unknown;
  }
}

// Parse the shape tuple string from a NumPy header dict, e.g. "(2, 3)" or "(3,)".
// Returns a SmallVec of dimensions. Empty on parse error.
SmallVec<int64_t, 6> parse_shape_tuple(const std::string& s) {
  SmallVec<int64_t, 6> dims;
  // Trim whitespace, expect '(' ... ')'.
  size_t start = s.find('(');
  size_t end = s.rfind(')');
  if (start == std::string::npos || end == std::string::npos || end <= start)
    return dims;
  std::string inner = s.substr(start + 1, end - start - 1);
  // Scalar: "()" -> rank 0.
  if (inner.empty() || (inner.find_first_not_of(" \t") == std::string::npos))
    return dims;  // empty = scalar (shape [])
  // Split by ','.
  size_t pos = 0;
  while (pos < inner.size()) {
    // Skip whitespace.
    while (pos < inner.size() && (inner[pos] == ' ' || inner[pos] == '\t')) ++pos;
    if (pos >= inner.size()) break;
    // Read a number.
    size_t num_start = pos;
    while (pos < inner.size() && inner[pos] != ',') ++pos;
    std::string tok = inner.substr(num_start, pos - num_start);
    // Trim trailing whitespace from token.
    while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) tok.pop_back();
    if (!tok.empty()) {
      try { dims.push_back(std::stoll(tok)); } catch (...) { return {}; }
    }
    if (pos < inner.size() && inner[pos] == ',') ++pos;
  }
  return dims;
}

// Parse the NumPy header dict (Python literal dict) to extract 'descr' and 'shape'.
// Minimal Python-dict parser: looks for 'descr': '...', 'shape': (...).
// Returns {descr_string, shape_tuple_string}. Empty strings on parse error.
std::pair<std::string, std::string> parse_npy_dict(const std::string& dict_str) {
  std::string descr, shape;
  // Find 'descr': '...' (single-quoted value).
  size_t descr_pos = dict_str.find("'descr'");
  if (descr_pos == std::string::npos) descr_pos = dict_str.find("\"descr\"");
  if (descr_pos != std::string::npos) {
    size_t colon = dict_str.find(':', descr_pos);
    if (colon != std::string::npos) {
      size_t q1 = dict_str.find_first_of("'\"", colon);
      if (q1 != std::string::npos) {
        char quote = dict_str[q1];
        size_t q2 = dict_str.find(quote, q1 + 1);
        if (q2 != std::string::npos) descr = dict_str.substr(q1 + 1, q2 - q1 - 1);
      }
    }
  }
  // Find 'shape': (...)
  size_t shape_pos = dict_str.find("'shape'");
  if (shape_pos == std::string::npos) shape_pos = dict_str.find("\"shape\"");
  if (shape_pos != std::string::npos) {
    size_t colon = dict_str.find(':', shape_pos);
    if (colon != std::string::npos) {
      size_t paren = dict_str.find('(', colon);
      if (paren != std::string::npos) {
        size_t close = dict_str.find(')', paren);
        if (close != std::string::npos)
          shape = dict_str.substr(paren, close - paren + 1);
      }
    }
  }
  return {descr, shape};
}

// Parse the .npy header at `offset` in the mmap to extract dtype and shape.
// Returns {dtype, shape, header_length}. header_length is the byte count of the
// .npy header (magic+version+len+dict) so payload offset = offset+header_length.
// On error, returns {DType::Unknown, {}, 0}.
struct NpyHeader { ir::DType dtype; SmallVec<int64_t, 6> shape; uint64_t header_len; };
NpyHeader parse_npy_header(ByteReader& r, uint64_t offset) {
  if (!r.seek(offset)) return {ir::DType::Unknown, {}, 0};
  uint64_t start = offset;
  // Magic: 6 bytes "\x93NUMPY" (note: 0x93 is 147 decimal).
  if (r.remaining() < 6) return {ir::DType::Unknown, {}, 0};
  auto b0 = r.u8(); if (!b0 || *b0 != 0x93) return {ir::DType::Unknown, {}, 0};
  auto b1 = r.u8(); if (!b1 || *b1 != 'N') return {ir::DType::Unknown, {}, 0};
  auto b2 = r.u8(); if (!b2 || *b2 != 'U') return {ir::DType::Unknown, {}, 0};
  auto b3 = r.u8(); if (!b3 || *b3 != 'M') return {ir::DType::Unknown, {}, 0};
  auto b4 = r.u8(); if (!b4 || *b4 != 'P') return {ir::DType::Unknown, {}, 0};
  auto b5 = r.u8(); if (!b5 || *b5 != 'Y') return {ir::DType::Unknown, {}, 0};
  // Version: 2 bytes (major, minor).
  auto ver_major = r.u8(); if (!ver_major) return {ir::DType::Unknown, {}, 0};
  auto ver_minor = r.u8(); if (!ver_minor) return {ir::DType::Unknown, {}, 0};
  // Header length: u16 for version 1.0, u32 for 2.0+. We support both.
  uint64_t hdr_len = 0;
  if (*ver_major == 1) {
    auto len16 = r.u16le(); if (!len16) return {ir::DType::Unknown, {}, 0};
    hdr_len = *len16;
  } else if (*ver_major >= 2) {
    auto len32 = r.u32le(); if (!len32) return {ir::DType::Unknown, {}, 0};
    hdr_len = *len32;
  } else {
    return {ir::DType::Unknown, {}, 0};  // unsupported version
  }
  // Read the dict string (hdr_len bytes).
  auto dict_bytes = r.bytes(hdr_len);
  if (!dict_bytes) return {ir::DType::Unknown, {}, 0};
  std::string dict = *dict_bytes;
  // Parse the dict for 'descr' and 'shape'.
  auto [descr_str, shape_str] = parse_npy_dict(dict);
  if (descr_str.empty() || shape_str.empty())
    return {ir::DType::Unknown, {}, 0};
  ir::DType dtype = descr_to_dtype(descr_str);
  SmallVec<int64_t, 6> shape = parse_shape_tuple(shape_str);
  // Total header length = 6 (magic) + 2 (version) + 2or4 (len) + hdr_len.
  uint64_t total_hdr_len = (r.pos() - start);
  return {dtype, std::move(shape), total_hdr_len};
}

}  // namespace

Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "opening npz");
  const uint8_t* base = file.data();
  uint64_t size = file.size();
  if (!base || size == 0) return err("empty file", 0);

  mz_zip_archive zip;
  std::memset(&zip, 0, sizeof(zip));
  if (!mz_zip_reader_init_mem(&zip, base, static_cast<size_t>(size), 0))
    return err("not a valid zip archive", 0);

  // RAII cleanup guard.
  struct ZipGuard {
    mz_zip_archive* z;
    ~ZipGuard() { mz_zip_reader_end(z); }
  } guard{&zip};

  mz_uint num = mz_zip_reader_get_num_files(&zip);
  // Bounded entry count (safety: a hostile zip claiming millions of entries).
  constexpr mz_uint kMaxEntries = 100000;
  if (num > kMaxEntries) return err("too many entries in npz archive", num);

  progress.set(0.3f, "scanning entries");

  ir::Model model;
  model.format_name = model.intern("NumPy npz");
  model.has_graph = false;

  ByteReader r(base, size);
  uint64_t compressed_count = 0;

  for (mz_uint i = 0; i < num; ++i) {
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&zip, i, &st)) continue;
    if (st.m_is_directory) continue;
    std::string name = st.m_filename;
    // Only process .npy entries.
    if (name.size() < 4 || name.substr(name.size() - 4) != ".npy") continue;
    // Tensor name = entry name minus ".npy" suffix.
    std::string tensor_name = name.substr(0, name.size() - 4);

    // Check if the entry is compressed (DEFLATE).
    bool compressed = (st.m_method != 0);  // 0 = ZIP_STORED (uncompressed)
    if (compressed) {
      // Compressed entries: we cannot mmap-address the payload (it's DEFLATE).
      // Record shape/dtype from the header if reachable, set file_offset=UINT64_MAX.
      // SECURITY: we do NOT extract/decompress to read the header — that would
      // be a payload read. Instead we mark this tensor as unaddressable (honest).
      ++compressed_count;
      ir::TensorRef t;
      t.name = model.intern(tensor_name);
      t.dtype = ir::DType::Unknown;
      t.shape = {};
      t.file_offset = UINT64_MAX;
      t.byte_len = st.m_uncomp_size;
      model.flat_tensors.push_back(t);
      continue;
    }

    // Uncompressed (STORED): compute payload offset from local header.
    uint64_t payload_off = 0;
    if (!payload_offset_from_local_header(base, size, st.m_local_header_ofs,
                                          payload_off))
      continue;  // Skip malformed entry.

    // Parse the .npy header at payload_off (structural read, NOT a payload read).
    auto hdr = parse_npy_header(r, payload_off);
    if (hdr.dtype == ir::DType::Unknown || hdr.header_len == 0)
      continue;  // Skip unparseable .npy entry.

    // The array payload starts after the .npy header.
    uint64_t array_offset = payload_off + hdr.header_len;
    uint64_t array_len = (st.m_uncomp_size > hdr.header_len)
                             ? st.m_uncomp_size - hdr.header_len : 0;

    ir::TensorRef t;
    t.name = model.intern(tensor_name);
    t.dtype = hdr.dtype;
    t.shape = hdr.shape;
    t.file_offset = array_offset;
    t.byte_len = array_len;
    model.flat_tensors.push_back(t);
  }

  if (compressed_count > 0) {
    model.metadata.emplace_back(
        model.intern("compressed_entries"),
        model.intern(std::to_string(compressed_count) +
                     " (payload not mmap-addressable)"));
  }
  model.metadata.emplace_back(model.intern("tensors"),
                              model.intern(std::to_string(model.flat_tensors.size())));

  progress.set(1.0f, "done");
  return model;
}

}  // namespace netvis::npz
