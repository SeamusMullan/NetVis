// parsers/onnx/WireReader.h — minimal hand-rolled protobuf wire reader.
//
// DECISION: NetVis avoids libprotobuf/protoc entirely. ONNX files are just
// protobuf wire data, and we only need to walk fields structurally (record
// tensor offsets, never decode payloads). A ~200-line reader over the
// bounds-checked ByteReader is faster to build, has zero dependency, and lets
// us guarantee no out-of-bounds read on malformed input (truncation -> Result
// error carrying the byte offset).
//
// Wire format recap (protobuf): each field is preceded by a varint "tag" whose
// low 3 bits are the wire type and the rest is the field number:
//   field_number = tag >> 3 ; wire_type = tag & 7
// Wire types: 0 varint, 1 fixed64, 2 length-delimited, 5 fixed32.
#pragma once

#include <cstdint>

#include "core/ByteReader.h"
#include "core/Result.h"

namespace netvis::onnx {

enum class WireType : uint8_t {
  Varint = 0,
  Fixed64 = 1,
  LenDelim = 2,
  StartGroup = 3,  // deprecated groups; we reject
  EndGroup = 4,
  Fixed32 = 5,
};

// A parsed field header.
struct FieldHeader {
  uint32_t field_number = 0;
  WireType wire_type = WireType::Varint;
};

// A length-delimited sub-range as an absolute [offset,len) into the source
// buffer, plus a pointer to its first byte. Used to snapshot nested messages
// and to record tensor payload locations WITHOUT reading the bytes.
struct SubRange {
  const uint8_t* ptr = nullptr;
  uint64_t offset = 0;  // absolute offset into the mmap'd file
  uint64_t len = 0;
};

// WireReader wraps a ByteReader and adds protobuf-specific decoding. It carries
// a `base_offset`: the absolute file offset of byte 0 of the underlying reader,
// so nested sub-ranges can report absolute mmap offsets for TensorRef.
class WireReader {
 public:
  WireReader(const uint8_t* data, uint64_t size, uint64_t base_offset = 0)
      : br_(data, size), base_(base_offset) {}

  bool at_end() const { return br_.at_end(); }
  uint64_t pos() const { return br_.pos(); }
  uint64_t base_offset() const { return base_; }
  // Absolute file offset of the current cursor position.
  uint64_t abs_pos() const { return base_ + br_.pos(); }

  // Read a base-128 varint (LEB128) as u64. Rejects overlong (>10 byte)
  // encodings and truncation with a byte-offset error.
  Result<uint64_t> read_varint() {
    uint64_t result = 0;
    int shift = 0;
    uint64_t start = br_.pos();
    for (int i = 0; i < 10; ++i) {
      auto b = br_.u8();
      if (!b) return b.error();
      uint8_t byte = *b;
      result |= (static_cast<uint64_t>(byte & 0x7F) << shift);
      if ((byte & 0x80) == 0) return result;
      shift += 7;
    }
    return err("varint too long / unterminated", base_ + start);
  }

  // Read a field header (tag varint -> field number + wire type).
  Result<FieldHeader> read_tag() {
    uint64_t start = br_.pos();
    auto t = read_varint();
    if (!t) return t.error();
    uint64_t tag = *t;
    WireType wt = static_cast<WireType>(tag & 0x7);
    FieldHeader h;
    h.field_number = static_cast<uint32_t>(tag >> 3);
    h.wire_type = wt;
    if (h.field_number == 0)
      return err("invalid field number 0", base_ + start);
    return h;
  }

  Result<uint32_t> read_fixed32() {
    auto r = br_.u32le();
    if (!r) return r.error();
    return *r;
  }

  Result<uint64_t> read_fixed64() {
    auto r = br_.u64le();
    if (!r) return r.error();
    return *r;
  }

  Result<float> read_float() {
    auto r = br_.f32le();
    if (!r) return r.error();
    return *r;
  }

  // Read a length-delimited chunk, returning an absolute sub-range WITHOUT
  // copying. Advances past the chunk. Used for nested messages, strings, and
  // (crucially) tensor payloads whose bytes we must NOT touch.
  Result<SubRange> read_len_delim() {
    uint64_t lenpos = br_.pos();
    auto ln = read_varint();
    if (!ln) return ln.error();
    uint64_t len = *ln;
    uint64_t data_pos = br_.pos();
    auto p = br_.raw(len);  // bounds-checked; does NOT read the bytes' contents
    if (!p) return err("length-delimited field truncated", base_ + lenpos);
    SubRange sr;
    sr.ptr = *p;
    sr.offset = base_ + data_pos;
    sr.len = len;
    return sr;
  }

  // Read a length-delimited field as a std::string (names/keys only).
  Result<std::string> read_string() {
    auto sr = read_len_delim();
    if (!sr) return sr.error();
    // Copy structural strings (never payloads); safe because the sub-range was
    // bounds-checked above.
    return std::string(reinterpret_cast<const char*>(sr->ptr),
                       static_cast<size_t>(sr->len));
  }

  // Skip a field whose wire type we don't handle for this message, so unknown
  // fields never abort a parse (forward-compat with newer ONNX).
  Result<bool> skip_field(WireType wt) {
    switch (wt) {
      case WireType::Varint: {
        auto r = read_varint();
        if (!r) return r.error();
        return true;
      }
      case WireType::Fixed64: {
        auto r = read_fixed64();
        if (!r) return r.error();
        return true;
      }
      case WireType::Fixed32: {
        auto r = read_fixed32();
        if (!r) return r.error();
        return true;
      }
      case WireType::LenDelim: {
        auto r = read_len_delim();
        if (!r) return r.error();
        return true;
      }
      default:
        return err("unsupported wire type (group?)", abs_pos());
    }
  }

  // Sub-reader over a nested sub-range, carrying the correct absolute base.
  static WireReader sub(const SubRange& sr) {
    return WireReader(sr.ptr, sr.len, sr.offset);
  }

 private:
  ByteReader br_;
  uint64_t base_;  // absolute file offset of br_ byte 0
};

}  // namespace netvis::onnx
