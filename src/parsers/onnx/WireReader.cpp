// parsers/onnx/WireReader.cpp — the WireReader is entirely header-inline (its
// methods are small and hot, so we let them inline at call sites). This TU
// exists to satisfy the module layout and to anchor a static_assert on the
// wire-type numbering the parser relies on.
#include "parsers/onnx/WireReader.h"

namespace netvis::onnx {

static_assert(static_cast<int>(WireType::Varint) == 0);
static_assert(static_cast<int>(WireType::Fixed64) == 1);
static_assert(static_cast<int>(WireType::LenDelim) == 2);
static_assert(static_cast<int>(WireType::Fixed32) == 5);

}  // namespace netvis::onnx
