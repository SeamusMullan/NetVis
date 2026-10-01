// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeNet.h — the format-neutral raw Caffe network both
// front-ends produce (#138/#109).
//
// read_prototxt (CaffeText.cpp, text NetParameter through TextProto) and
// read_caffemodel (CaffeBinary.cpp, binary NetParameter through
// onnx::WireReader) both fill a NetRaw; CaffeParser.cpp turns a NetRaw into an
// ir::Model. Keeping the two readers to one output shape is what lets the
// sibling-pairing step graft binary blobs onto text layers, and lets tests
// prove a prototxt and its caffemodel describe the same graph.
//
// ZERO PAYLOAD: a BlobRaw records the absolute offset + length of its single
// packed payload chunk; nothing here ever reads weight bytes.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/JobSystem.h"
#include "core/Result.h"
#include "core/SmallVec.h"
#include "ir/IR.h"

namespace netvis::caffe {

// ---- limits (spec §5.1) -----------------------------------------------------------
constexpr uint64_t kMaxNameBytes = 64 * 1024;       // layer / blob / type names
constexpr size_t kMaxLayers = 1000000;
constexpr size_t kMaxBlobsPerLayer = 1024;
constexpr size_t kMaxBottomsOrTops = 65536;         // each, per layer
constexpr size_t kMaxBlobAxes = 32;                 // Caffe's own kMaxBlobAxes
constexpr size_t kMaxRepeatedScalars = 4096;        // per non-blob repeated field
// Binary string fields share the text reader's decoded-string cap (TextProto.h).
constexpr uint64_t kMaxBinaryStringBytes = 1ull << 20;

enum class Generation : uint8_t { None, Layer /*NetParameter.layer=100*/, V1 /*NetParameter.layers=2*/ };
enum class ReadMode : uint8_t { Full, BlobsOnly };

struct BlobRaw {
  SmallVec<int64_t, 6> shape;          // BlobShape.dim, else legacy {num,channels,height,width}
  bool shape_known = false;
  bool legacy_dims = false;            // came from fields 1..4
  ir::DType dtype = ir::DType::Unknown;// F32 from `data`(5), F64 from `double_data`(8)
  uint64_t offset = UINT64_MAX;        // absolute offset of the ONE packed payload chunk
  uint64_t byte_len = 0;
  uint64_t value_count = 0;            // byte_len/4|8, or elided text value count
  bool addressable = false;            // exactly one packed chunk in a binary file
  bool unpacked = false;               // saw wire-type-5/1 elements -> abandoned (§3.6)
  bool text = false;                   // inline text-format blob (never addressable)
  bool external = false;               // set by pairing: lives in the sibling file
  uint64_t proto_offset = 0;           // offset of the BlobProto, for notes/errors
};

struct RuleRaw {                       // NetStateRule
  int32_t phase = -1;                  // -1 = absent
  bool has_min_level = false;
  int32_t min_level = 0;
  bool has_max_level = false;
  int32_t max_level = 0;
  std::vector<std::string> stage, not_stage;
};

struct LayerRaw {
  std::string name;
  std::string type;                    // modern type string; V1 enum translated (§3.3)
  std::string v1_type_ident;           // "CONVOLUTION" / raw number text; empty for modern
  bool v1 = false, v0 = false;
  bool has_type = false;               // a type field was present at all
  std::vector<std::string> bottom, top;
  std::vector<RuleRaw> include, exclude;
  std::vector<BlobRaw> blobs;
  std::vector<ir::Attribute> attrs;    // native, dotted names (§3.9), interned in the target model
  // input_param.shape entries, already sanitised (§5.4); has_input_param when the
  // layer carried an input_param at all.
  bool has_input_param = false;
  std::vector<SmallVec<int64_t, 6>> input_param_shapes;
  std::vector<bool> input_param_shape_ok;   // false => oversized, left unknown
  uint64_t offset = 0;                 // absolute offset of the layer's first byte/token
};

struct NetRaw {
  std::string name;
  Generation gen = Generation::None;
  bool saw_layer = false, saw_layers = false;
  uint64_t second_kind_offset = UINT64_MAX;     // first field of whichever kind came second
  std::vector<std::string> inputs;              // NetParameter.input (3)
  std::vector<SmallVec<int64_t, 6>> input_shapes;   // input_shape (8), sanitised
  std::vector<bool> input_shape_ok;             // false => oversized, left unknown
  std::vector<int64_t> input_dims;              // input_dim (4)
  bool has_state = false;
  int32_t state_phase = 1 /*TEST*/;
  int32_t state_level = 0;
  std::vector<std::string> state_stages;
  std::vector<LayerRaw> layers;
  // counters surfaced as metadata (§3.11)
  uint32_t v0_layers = 0, text_blobs = 0, unpacked_blobs = 0, undecoded_fields = 0,
           ignored_top_level = 0, oversized_shapes = 0, blob_size_mismatch = 0;
};

Result<NetRaw> read_prototxt(const uint8_t* d, uint64_t n, ir::Model& model, ProgressSink* p);
Result<NetRaw> read_caffemodel(const uint8_t* d, uint64_t n, ir::Model* model /*null ok in BlobsOnly*/,
                               ReadMode mode, ProgressSink* p);

// Record that a generation was seen at `offset` (shared by both readers so the
// "both 'layer' and 'layers'" error points at the same place).
inline void note_generation(NetRaw& net, Generation g, uint64_t offset) {
  const bool first_of_kind = g == Generation::Layer ? !net.saw_layer : !net.saw_layers;
  if (g == Generation::Layer) net.saw_layer = true;
  else net.saw_layers = true;
  if (net.gen == Generation::None) {
    net.gen = g;
  } else if (net.gen != g && first_of_kind && net.second_kind_offset == UINT64_MAX) {
    net.second_kind_offset = offset;
  }
}

// §5.4: sanitise a declared shape in place. Negative dims (and anything below
// -1) become -1; a product of the non-negative dims that overflows int64 or
// exceeds Caffe's INT32_MAX element limit makes the shape unknown (returns false).
bool sanitize_shape(SmallVec<int64_t, 6>& dims);

// ---- native attribute builder (both front-ends) -------------------------------------
// Flattens values into dotted-name attributes with one rule set: a repeated scalar
// is ONE list attribute in occurrence order (created at its first occurrence); a
// non-repeated scalar seen twice keeps its first position and takes the last value.
class AttrSink {
 public:
  AttrSink(ir::Model& model, std::vector<ir::Attribute>& out) : model_(model), out_(out) {}

  // Each returns false when a repeated list would exceed kMaxRepeatedScalars.
  bool add_int(const std::string& name, int64_t v, bool repeated);
  bool add_float(const std::string& name, double v, bool repeated);
  bool add_string(const std::string& name, std::string_view v, bool repeated);

 private:
  ir::AttrValue* slot(const std::string& name, ir::AttrValue::Kind kind, bool* fresh);

  ir::Model& model_;
  std::vector<ir::Attribute>& out_;
  std::unordered_map<std::string, size_t> index_;   // lookup only, never iterated
};

}  // namespace netvis::caffe
