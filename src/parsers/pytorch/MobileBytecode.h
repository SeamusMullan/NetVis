// SPDX-License-Identifier: Apache-2.0
// parsers/pytorch/MobileBytecode.h — PyTorch Mobile lite-interpreter bytecode.
//
// A .ptl written by torch.jit._save_for_lite_interpreter is a torch zip archive
// whose bytecode.pkl holds, per method, the instruction table of the lite
// interpreter's register/stack VM plus its operator, constant and type tables
// and (v4+) the method schema. This header is the format layer (#137):
//
//   decode_*            bytecode.pkl -> typed tables, every count/length capped
//                       and every field type-checked at a fixed index (upstream
//                       parse_bytecode.cpp expect_field semantics). Unknown
//                       opcode strings are KEPT (BcOp::Unknown) so the operator
//                       inventory still works; upstream would map them to OP.
//   build_method_graph  symbolic execution of one method's instruction stream
//                       (bytecode v6+, where each operator records its input
//                       count) into ir::Graphs: operators become nodes, registers
//                       and stack slots become edges, GET_ATTR chains on `self`
//                       bind module parameters, JF/JMP diamonds become prim::If
//                       and LOOP becomes prim::Loop with drill-down subgraphs.
//
// HONESTY: nothing is guessed. An instruction whose stack effect is unknowable,
// an operator without a recorded arity, or a stream that contradicts the JIT
// emitter's invariants (spec §2.5) is a method error with the instruction's
// byte offset; the caller then shows the exact operator inventory instead.
// Activation dtypes/shapes are unknown (the format carries none).
//
// SAFETY: no payload reads, no recursion beyond max_cf_depth, every index and
// count range-checked in int64 before use, the symbolic stack capped.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/Result.h"
#include "ir/IR.h"
#include "parsers/pytorch/PickleVM.h"
#include "parsers/pytorch/TorchScriptIR.h"

namespace netvis::pytorch::mobile {

// FORALL_OPCODES order. __IS__/__ISNOT__/__NOT__ are reserved identifiers in C++,
// so the enum spells them IS/ISNOT/NOT; the string table keeps upstream spelling.
enum class BcOp : uint8_t {
  OP, OPN, LOAD, MOVE, STOREN, STORE, DROP, DROPR, LOADC, JF, JMP, LOOP, RET, WAIT,
  CALL, GUARD, TYPECHECK, FAIL_GUARD, PROFILE_OP, TAIL_CALL, INTERFACE_CALL,
  GET_ATTR, SET_ATTR, LIST_UNPACK, TUPLE_CONSTRUCT, NAMED_TUPLE_CONSTRUCT,
  LIST_CONSTRUCT, DICT_CONSTRUCT, CREATE_OBJECT, ISINSTANCE, TUPLE_SLICE,
  TUPLE_INDEX, RAISE_EXCEPTION, DICT_INDEX, UNCHECKED_CAST, IS, UN_INITIALIZED,
  ISNOT, FORMAT, DEVICE, DTYPE, DIM, NOT, TO_LIST, NUM_TO_TENSOR, IS_CUDA, FORK,
  WARN, ENTER, EXIT, AWAITABLE,
  Unknown  // NOT upstream: an unrecognised string (upstream would map it to OP)
};
BcOp opcode_from_string(std::string_view s);  // exact, case-sensitive; else Unknown
const char* opcode_name(BcOp op);             // upstream spelling; "?" for Unknown
bool opcode_valid_in_mobile(BcOp op);         // isOpSupportedInMobile() minus CALL

struct BytecodeLimits {
  uint64_t max_pickle_bytes = 64ull << 20;    // bytecode.pkl / constants.pkl uncompressed
  uint64_t max_pickle_values = 2'000'000;     // PickleLimits for those two pickles
  uint32_t max_functions = 4096;
  uint32_t max_instructions = 1'000'000;      // per function
  uint32_t max_operators = 65'536;            // per function
  uint32_t max_constants = 262'144;           // per function
  uint32_t max_types = 65'536;                // per function
  int64_t max_register_size = 1 << 20;
  uint32_t max_schema_args = 4096;            // per arguments / returns list
  uint32_t max_string = 4096;                 // any decoded name/type string, bytes
  uint32_t max_opcode_string = 32;            // longest real opcode is 21 chars
  uint32_t max_stack = 65'536;                // symbolic stack depth
  uint32_t max_cf_depth = 64;                 // nested If/Loop
  uint32_t max_subgraphs = 4096;              // per method
  uint32_t max_attr_path = 1024;              // module attribute path, bytes
};

struct BcInstruction {
  BcOp op = BcOp::Unknown;
  int32_t x = 0;                       // validated to fit int32 (upstream Instruction::X)
  int32_t n = 0;                       // validated to [0, 65535] (upstream uint16 N)
  uint32_t unknown_name = UINT32_MAX;  // index into BcFunction::unknown_opcodes
  uint64_t pos = UINT64_MAX;           // src_pos of the instruction Tuple
};
struct BcOperator {
  std::string name, overload;  // verbatim
  int64_t num_args = -1;       // valid only if has_num_args
  bool has_num_args = false;   // v6+ 3-element entry with an Int
  bool well_formed = true;     // false: entry not a >=2 Tuple or name/overload not Str
  uint64_t pos = UINT64_MAX;
};
struct BcArg {
  std::string name, type;
  ValuePtr default_value;
};
struct BcFunction {
  std::string qualified_name;                // e.g. "__torch__.Model.forward"
  std::vector<BcInstruction> instructions;
  std::vector<std::string> unknown_opcodes;  // raw (<= max_opcode_string bytes + "…")
  std::vector<BcOperator> operators;
  std::vector<ValuePtr> constants;           // any pickled value; tensors are Kind::Tensor
  std::vector<std::string> types;            // verbatim annotation strings
  std::vector<uint8_t> type_ok;              // 1:1 with types; 0 => entry was not a Str ("?")
  int64_t register_size = 0;
  bool has_schema = false;
  std::vector<BcArg> arguments, returns;
  uint64_t pos = UINT64_MAX;
};
struct BcModule {
  int64_t version = 0;
  bool version_implicit = false;      // v3 (no version int)
  std::vector<BcFunction> functions;  // file order
};

// Decode an unpickled bytecode.pkl root. Errors carry the offending Value's
// src_pos (an offset within bytecode.pkl).
Result<BcModule> decode_bytecode(const ValuePtr& root, const BytecodeLimits& lim = {});
// Runs PickleVM(data,size,resolver,PickleLimits{lim.max_pickle_values}) then
// decode_bytecode.
Result<BcModule> decode_bytecode_pickle(const uint8_t* data, uint64_t size,
                                        const StorageResolver& resolver,
                                        const BytecodeLimits& lim = {});

// Index into `names` (zip central-directory order) of the bytecode.pkl to use:
// exactly dir_prefix(data_pkl_path)+"bytecode.pkl" when data_pkl_path is
// non-empty; otherwise the basename-"bytecode.pkl" entry with the fewest '/'
// (tie: lowest index). -1 if none. Pure, so the two-archives-in-one-zip rule is
// unit-testable.
int32_t select_bytecode_entry(const std::vector<std::string>& names,
                              std::string_view data_pkl_path);
// The method to graph, in file order: the root class's forward, else any
// *.forward, else the first root-class method that is not __getstate__ /
// __setstate__; -1 when none.
int32_t select_main_function(const BcModule& m, std::string_view root_class);

struct BuildContext {
  ir::Model& model;                         // model.graphs must be empty on entry
  const ts::ModuleEnv& env;                 // empty ModuleEnv when there is no data.pkl
  const ts::TensorConstNames& const_names;  // empty when constants.pkl is absent/failed
  BytecodeLimits lim;
};
struct BuildResult {
  int32_t main_graph = -1;                         // always 0 on success
  std::vector<const Value*> bound_module_tensors;  // distinct, first-bound order
  std::string self_note;                           // non-empty iff `self` stayed unbound
};
// Builds graphs for m.functions[fn] (+ its If/Loop subgraphs) into
// ctx.model.graphs. On error, ctx.model.graphs is left EMPTY (the caller falls
// back); offsets are relative to bytecode.pkl.
Result<BuildResult> build_method_graph(const BcModule& m, size_t fn, BuildContext& ctx);
// "forward(self: __torch__.Model, x: Tensor, flag: bool) -> Tensor"; "" without
// a schema.
std::string method_signature(const BcFunction& f);

}  // namespace netvis::pytorch::mobile
