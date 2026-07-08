// ir/IR.cpp — dtype metadata tables.
#include "ir/IR.h"

namespace netvis::ir {

const char* dtype_name(DType d) {
  switch (d) {
    case DType::F32: return "f32";
    case DType::F16: return "f16";
    case DType::BF16: return "bf16";
    case DType::F64: return "f64";
    case DType::I8: return "i8";
    case DType::I16: return "i16";
    case DType::I32: return "i32";
    case DType::I64: return "i64";
    case DType::U8: return "u8";
    case DType::U16: return "u16";
    case DType::U32: return "u32";
    case DType::U64: return "u64";
    case DType::Bool: return "bool";
    case DType::Q4: return "q4";
    case DType::Q8: return "q8";
    case DType::Unknown: return "?";
  }
  return "?";
}

uint32_t dtype_size(DType d) {
  switch (d) {
    case DType::F32: return 4;
    case DType::F16: return 2;
    case DType::BF16: return 2;
    case DType::F64: return 8;
    case DType::I8: return 1;
    case DType::I16: return 2;
    case DType::I32: return 4;
    case DType::I64: return 8;
    case DType::U8: return 1;
    case DType::U16: return 2;
    case DType::U32: return 4;
    case DType::U64: return 8;
    case DType::Bool: return 1;
    // Quantized blocks are not element-addressable; callers use metadata.
    case DType::Q4: return 0;
    case DType::Q8: return 0;
    case DType::Unknown: return 0;
  }
  return 0;
}

}  // namespace netvis::ir
