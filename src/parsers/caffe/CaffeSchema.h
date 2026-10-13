// SPDX-License-Identifier: Apache-2.0
// parsers/caffe/CaffeSchema.h — the subset of BVLC caffe.proto that NetVis
// decodes, as static constexpr tables (#138/#109).
//
// DECISION: one schema drives BOTH front-ends. The text reader looks fields up
// by name, the binary reader by number, and both flatten what they find into
// the same dotted attribute names with the same kinds — so a .prototxt and the
// .caffemodel written from it carry identical native attributes. Field numbers,
// kinds and enum values are verbatim from upstream src/caffe/proto/caffe.proto
// (master). A field missing from these tables is not an error: text shows it
// through generic flattening, binary counts it as undecoded.
//
// The tables are constexpr (no dynamic initialisation, nothing for TSan) and are
// reached through accessor functions.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace netvis::caffe {

enum class FieldKind : uint8_t {
  Int32, Int64, UInt32, UInt64, Bool, Float, Double, Enum, String, Message
};

struct EnumValue {
  int32_t number;
  const char* name;
};

struct EnumSpec {
  const char* name;
  const EnumValue* values;
  size_t count;
};

struct MessageSpec;

struct FieldSpec {
  uint32_t number;
  const char* name;
  FieldKind kind;
  bool repeated;
  const EnumSpec* enum_spec;    // Enum only
  const MessageSpec* message;   // Message only; nullptr => contents not decoded
};

struct MessageSpec {
  const char* name;
  const FieldSpec* fields;
  size_t count;
};

// Lookups (linear; the largest table has ~60 entries). nullptr when absent.
const FieldSpec* find_field(const MessageSpec& m, uint32_t number);
const FieldSpec* find_field(const MessageSpec& m, std::string_view name);

// Enum identifier for `number`, or nullptr when the value is not in the enum.
const char* enum_name(const EnumSpec& e, int64_t number);
// Enum number for `name`, or nullopt when the identifier is not in the enum.
std::optional<int32_t> enum_number(const EnumSpec& e, std::string_view name);

// Message specs.
const MessageSpec& net_parameter_spec();       // only for the top-level field list
const MessageSpec& layer_parameter_spec();     // core fields + *_param slots 100..148
const MessageSpec& v1_layer_parameter_spec();  // core fields + *_param slots 9..42
const MessageSpec& blob_shape_spec();
const MessageSpec& net_state_spec();
const MessageSpec& net_state_rule_spec();

// Enums the parser reads directly.
const EnumSpec& phase_enum();          // TRAIN 0, TEST 1
const EnumSpec& v1_layer_type_enum();  // V1LayerParameter.LayerType

// V1 LayerType number -> modern `type` string, verbatim from UpgradeV1LayerType
// (src/caffe/util/upgrade_proto.cpp). "" for NONE and for unknown numbers.
const char* v1_modern_type(int64_t number);
constexpr int32_t kMaxV1LayerType = 39;   // DECONVOLUTION

}  // namespace netvis::caffe
