// parsers/keras/KerasParser.cpp — Keras .h5 / .keras -> ir::Model (tensor table).
//
// Two on-disk shapes under one Format::Keras:
//   - raw HDF5 (.h5, legacy .keras): run the minimal Hdf5Reader over the whole
//     mmap and surface every discovered dataset as a flat tensor.
//   - Keras v3 zip (.keras): a ZIP containing config.json + metadata.json +
//     model.weights.h5. We read config.json (small, structural) for metadata and
//     locate the embedded model.weights.h5; when it is STORED (uncompressed) we
//     resolve its payload offset from the local file header and run Hdf5Reader
//     over that sub-range, rebasing dataset offsets to absolute mmap positions.
//     A DEFLATE-compressed inner .h5 is not linearly addressable -> metadata note.
//
// WEIGHTS: dataset payload bytes are NEVER read. Hdf5Reader records only
// offset+len; a parse leaves ByteReader::payload_read_counter() at 0.
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
#include "parsers/keras/Hdf5Reader.h"

#include "miniz.h"

namespace netvis::keras {
namespace {

// Absolute file offset of a STORED zip entry's uncompressed payload, from its
// local file header. Bounds-checked; returns false on any error. (Mirrors the
// helper in PytorchParser.cpp; duplicated to keep this TU self-contained.)
constexpr uint32_t kLocalHeaderSig = 0x04034b50;

bool payload_offset_from_local_header(const uint8_t* base, uint64_t file_size,
                                      uint64_t local_header_ofs,
                                      uint64_t& out_offset) {
  ByteReader r(base, file_size);
  if (!r.seek(local_header_ofs)) return false;
  auto sig = r.u32le();
  if (!sig || *sig != kLocalHeaderSig) return false;
  r.skip(22);  // to filename_len at header+26
  auto fn_len = r.u16le();
  if (!fn_len) return false;
  auto extra_len = r.u16le();
  if (!extra_len) return false;
  out_offset = local_header_ofs + 30ULL + *fn_len + *extra_len;
  if (out_offset > file_size) return false;
  return true;
}

// Turn a WalkResult into flat tensors + carry notes into metadata.
void ingest_walk(const hdf5::WalkResult& wr, ir::Model& model) {
  for (const auto& ds : wr.datasets) {
    ir::TensorRef t;
    t.name = model.intern(ds.name);
    t.dtype = ds.dtype;
    for (int64_t d : ds.shape) t.shape.push_back(d);
    t.file_offset = ds.file_offset;   // UINT64_MAX if chunked/compact/undefined
    t.byte_len = ds.byte_len;
    model.flat_tensors.push_back(std::move(t));
  }
  for (const auto& n : wr.notes) {
    model.metadata.emplace_back(model.intern("hdf5"), model.intern(n));
  }
}

Result<ir::Model> parse_zip_keras(const MappedFile& file) {
  const uint8_t* base = file.data();
  uint64_t size = file.size();

  mz_zip_archive zip;
  std::memset(&zip, 0, sizeof(zip));
  if (!mz_zip_reader_init_mem(&zip, base, static_cast<size_t>(size), 0))
    return err("not a valid .keras zip archive", 0);
  struct ZipGuard { mz_zip_archive* z; ~ZipGuard() { mz_zip_reader_end(z); } } guard{&zip};

  ir::Model model;
  model.format_name = model.intern("Keras");
  model.has_graph = false;

  mz_uint num = mz_zip_reader_get_num_files(&zip);
  mz_uint weights_index = num;               // sentinel
  bool weights_stored = false;
  uint64_t weights_uncomp = 0;
  mz_uint config_index = num;

  for (mz_uint i = 0; i < num; ++i) {
    mz_zip_archive_file_stat st;
    if (!mz_zip_reader_file_stat(&zip, i, &st)) continue;
    if (st.m_is_directory) continue;
    std::string name = st.m_filename;
    std::string bn = name;
    auto sl = name.find_last_of('/');
    if (sl != std::string::npos) bn = name.substr(sl + 1);
    if (bn == "model.weights.h5" && weights_index == num) {
      weights_index = i;
      weights_stored = (st.m_method == 0);   // 0 == STORED
      weights_uncomp = st.m_uncomp_size;
    }
    if (bn == "config.json" && config_index == num) config_index = i;
  }

  // config.json: small structural JSON — record its presence + a class-name hint
  // if cheaply available. We avoid a full JSON parse here (names come from the
  // HDF5 group paths); just note the model was a Keras v3 archive.
  if (config_index != num) {
    model.metadata.emplace_back(model.intern("keras"), model.intern("v3 archive"));
  }

  if (weights_index == num) {
    model.metadata.emplace_back(model.intern("keras"),
                                model.intern("no model.weights.h5 in archive"));
    return model;
  }

  if (!weights_stored) {
    model.metadata.emplace_back(
        model.intern("keras"),
        model.intern("model.weights.h5 is compressed; weights not addressable"));
    return model;
  }

  mz_zip_archive_file_stat wst;
  if (!mz_zip_reader_file_stat(&zip, weights_index, &wst))
    return err("failed to stat model.weights.h5", 0);
  uint64_t payload_off = 0;
  if (!payload_offset_from_local_header(base, size, wst.m_local_header_ofs, payload_off)) {
    model.metadata.emplace_back(model.intern("keras"),
                                model.intern("could not locate embedded weights payload"));
    return model;
  }
  if (payload_off + weights_uncomp > size)
    return err("embedded weights payload out of range", payload_off);

  // Walk the embedded .h5 sub-range; rebase offsets to absolute mmap positions.
  hdf5::WalkResult wr = hdf5::walk(base + payload_off, weights_uncomp, payload_off);
  ingest_walk(wr, model);
  return model;
}

}  // namespace

Result<ir::Model> parse(const MappedFile& file, ProgressSink& progress) {
  progress.set(0.0f, "Parsing Keras");
  const uint8_t* base = file.data();
  uint64_t size = file.size();
  if (!base || size == 0) return err("empty or unmapped file", 0);

  // .keras v3 archives are ZIPs (PK\x03\x04); everything else is treated as raw
  // HDF5 and walked directly.
  if (size >= 4 && std::memcmp(base, "PK\x03\x04", 4) == 0) {
    auto r = parse_zip_keras(file);
    progress.set(1.0f, "done");
    return r;
  }

  ir::Model model;
  model.format_name = model.intern("Keras");
  model.has_graph = false;
  progress.set(0.3f, "walking HDF5");
  hdf5::WalkResult wr = hdf5::walk(base, size, 0);
  if (!wr.superblock_found)
    return err("not an HDF5 / Keras file (no superblock)", 0);
  ingest_walk(wr, model);
  model.metadata.emplace_back(model.intern("tensors"),
                              model.intern(std::to_string(model.flat_tensors.size())));
  progress.set(1.0f, "done");
  return model;
}

}  // namespace netvis::keras
