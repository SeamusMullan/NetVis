#!/usr/bin/env python3
"""NetVis test-fixture generator (spec §10).

Emits tiny, hand-encoded model files exercising every parser path, using the
Python 3 standard library ONLY (no ML frameworks, no protobuf/flatbuffers libs).
Every byte is produced by hand so the fixtures document the wire formats and stay
stable across machines.

The generator is DETERMINISTIC and IDEMPOTENT: running it twice writes identical
bytes. Output directory is argv[1] (default: <repo>/tests/fixtures).

Formats emitted:
  model.onnx        - 3-node Conv->Relu->MatMul ONNX ModelProto with one raw_data
                      initializer and one external_data initializer (>4GB offset).
  model.safetensors - JSON header + 36-byte payload, two F32 tensors.
  model.gguf        - GGUF v3, 2 tensors (F32 + Q4_0), 2 KV pairs, aligned data.
  model.pt          - PyTorch zip: hand-written protocol-2 pickle + storage blob.
  model.tflite      - Minimal-but-valid TFLite flatbuffer (ADD op, 2 tensors).
"""

import os
import struct
import sys
import zipfile


# ---------------------------------------------------------------------------
# Generic little-endian / protobuf helpers
# ---------------------------------------------------------------------------

def varint(n):
    """Encode an unsigned LEB128 varint (protobuf wire type 0)."""
    if n < 0:
        n &= (1 << 64) - 1  # two's-complement wrap for int64 fields
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def pb_tag(field, wire):
    """Protobuf key = (field_number << 3) | wire_type."""
    return varint((field << 3) | wire)


def pb_varint(field, n):
    return pb_tag(field, 0) + varint(n)


def pb_len(field, data):
    """Length-delimited field (wire type 2): tag, length, bytes."""
    return pb_tag(field, 2) + varint(len(data)) + data


def pb_string(field, s):
    return pb_len(field, s.encode("utf-8"))


# ---------------------------------------------------------------------------
# ONNX (protobuf ModelProto / GraphProto / NodeProto / TensorProto)
# ---------------------------------------------------------------------------

def onnx_attr_ints(name, ints):
    """AttributeProto with an int list: name(1) string, type(20)=INTS(7),
    ints(8) repeated varint. Field numbers per real onnx.proto3."""
    b = bytearray()
    b += pb_string(1, name)
    b += pb_varint(20, 7)          # AttributeType INTS = 7
    for v in ints:
        b += pb_varint(8, v)       # ints (repeated, unpacked)
    return bytes(b)


def onnx_node(op_type, name, inputs, outputs, attrs=()):
    """NodeProto: input(1)*, output(2)*, name(3), op_type(4), attribute(5)*."""
    b = bytearray()
    for i in inputs:
        b += pb_string(1, i)
    for o in outputs:
        b += pb_string(2, o)
    b += pb_string(3, name)
    b += pb_string(4, op_type)
    for a in attrs:
        b += pb_len(5, a)          # attribute (repeated AttributeProto)
    return bytes(b)


def onnx_sse(key, value):
    """StringStringEntryProto: key(1) string, value(2) string."""
    return pb_string(1, key) + pb_string(2, value)


def onnx_tensor_rawdata(name, dims, floats):
    """TensorProto with inline raw_data (field 9). The ONNX parser records the
    byte offset+length of raw_data into a TensorRef and never decodes it."""
    b = bytearray()
    for d in dims:
        b += pb_varint(1, d)          # dims (repeated int64)
    b += pb_varint(2, 1)              # data_type = FLOAT (1)
    b += pb_string(8, name)          # name
    raw = b"".join(struct.pack("<f", f) for f in floats)
    b += pb_len(9, raw)              # raw_data (bytes) -> payload region
    return bytes(b)


def onnx_tensor_external(name, dims, location, offset, length):
    """TensorProto using external_data (field 13) with a huge (>4GB) offset, so
    the parser must carry 64-bit offsets and set external_path/byte_len."""
    b = bytearray()
    for d in dims:
        b += pb_varint(1, d)          # dims
    b += pb_varint(2, 1)              # data_type = FLOAT (1)
    b += pb_string(8, name)          # name
    b += pb_len(13, onnx_sse("location", location))
    b += pb_len(13, onnx_sse("offset", str(offset)))   # ">2GB" as decimal string
    b += pb_len(13, onnx_sse("length", str(length)))
    b += pb_varint(14, 1)            # data_location = EXTERNAL (1)
    return bytes(b)


def build_onnx():
    # Conv carries int-list attributes (kernel_shape, strides) so tests exercise
    # AttributeProto field 8 (ints) — the field most likely to be mismapped.
    conv = onnx_node("Conv", "conv0", ["input", "W"], ["conv_out"],
                     attrs=[onnx_attr_ints("kernel_shape", [3, 3]),
                            onnx_attr_ints("strides", [1, 1])])
    relu = onnx_node("Relu", "relu0", ["conv_out"], ["relu_out"])
    matmul = onnx_node("MatMul", "matmul0", ["relu_out", "B"], ["output"])

    init_w = onnx_tensor_rawdata("W", [2, 2], [1.0, 2.0, 3.0, 4.0])
    init_b = onnx_tensor_external("B", [2, 2], "weights.bin",
                                  5000000000, 16)  # 5e9 bytes > 2GB
    # Add a small resolvable external initializer: offset 8, length 8 (2 floats).
    init_c = onnx_tensor_external("C", [2], "weights.bin", 8, 8)

    graph = bytearray()
    graph += pb_len(1, conv)          # node
    graph += pb_len(1, relu)
    graph += pb_len(1, matmul)
    graph += pb_string(2, "test_graph")  # graph name
    graph += pb_len(5, init_w)        # initializer (repeated)
    graph += pb_len(5, init_b)
    graph += pb_len(5, init_c)

    model = bytearray()
    model += pb_varint(1, 1)          # ir_version
    model += pb_string(2, "netvis-test")  # producer_name
    model += pb_len(7, bytes(graph))  # graph
    return bytes(model)


# ---------------------------------------------------------------------------
# SafeTensors (u64 LE header length + JSON header + payload)
# ---------------------------------------------------------------------------

def build_safetensors():
    # Exact header from spec §10 (compact JSON, deterministic key order).
    header = (
        '{"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},'
        '"b":{"dtype":"F32","shape":[3],"data_offsets":[24,36]},'
        '"__metadata__":{"framework":"test"}}'
    )
    hb = header.encode("utf-8")
    # 36 payload bytes: 6 F32 (w) + 3 F32 (b). Deterministic values.
    payload = b"".join(struct.pack("<f", float(i)) for i in range(9))
    assert len(payload) == 36
    return struct.pack("<Q", len(hb)) + hb + payload


# ---------------------------------------------------------------------------
# GGUF v3
# ---------------------------------------------------------------------------

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8

GGML_TYPE_F32 = 0
GGML_TYPE_Q4_0 = 2


def gguf_string(s):
    """GGUF string: u64 length + raw bytes."""
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def gguf_kv_u32(key, value):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", value)


def gguf_kv_string(key, value):
    return gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + gguf_string(value)


def gguf_tensor_info(name, dims, ggml_type, offset):
    b = bytearray()
    b += gguf_string(name)
    b += struct.pack("<I", len(dims))          # n_dimensions
    for d in dims:
        b += struct.pack("<Q", d)              # dimensions (u64 each)
    b += struct.pack("<I", ggml_type)          # ggml type
    b += struct.pack("<Q", offset)             # offset relative to data section
    return bytes(b)


def build_gguf():
    alignment = 32
    # F32 tensor: 2x3 -> 6 floats -> 24 bytes at data offset 0.
    # Q4_0 tensor: 32 elems -> one Q4_0 block -> 18 bytes, aligned to 32.
    f32_bytes = 24
    q4_off = ((f32_bytes + alignment - 1) // alignment) * alignment  # 32
    q4_bytes = 18

    header = bytearray()
    header += b"GGUF"                          # magic
    header += struct.pack("<I", 3)             # version 3
    header += struct.pack("<Q", 2)             # tensor_count
    header += struct.pack("<Q", 2)             # metadata_kv_count
    header += gguf_kv_u32("general.alignment", alignment)
    header += gguf_kv_string("general.name", "test")
    header += gguf_tensor_info("weight_f32", [2, 3], GGML_TYPE_F32, 0)
    header += gguf_tensor_info("weight_q4", [32], GGML_TYPE_Q4_0, q4_off)

    # Pad header end up to alignment; that padded position is the data section.
    pad = (-len(header)) % alignment
    header += b"\x00" * pad

    data = bytearray()
    data += b"".join(struct.pack("<f", float(i)) for i in range(6))  # 24 bytes
    data += b"\x00" * (q4_off - len(data))     # pad to q4 offset (32)
    data += b"\x00" * q4_bytes                 # 18 bytes Q4_0 block (opaque)

    return bytes(header) + bytes(data)


# ---------------------------------------------------------------------------
# PyTorch zip: hand-written protocol-2 pickle + raw storage blob
# ---------------------------------------------------------------------------

class Pickler:
    """Minimal protocol-2 pickle opcode emitter (spec §10 PyTorch path)."""

    def __init__(self):
        self.b = bytearray()
        self.memo = 0

    def _put(self):
        # BINPUT / LONG_BINPUT: memoize the top of stack (exercises the memo path).
        if self.memo < 256:
            self.b += b"q" + bytes([self.memo])
        else:
            self.b += b"r" + struct.pack("<I", self.memo)
        self.memo += 1

    def proto(self):
        self.b += b"\x80\x02"

    def stop(self):
        self.b += b"."

    def unicode(self, s):
        data = s.encode("utf-8")
        self.b += b"X" + struct.pack("<I", len(data)) + data
        self._put()

    def global_(self, module, name):
        self.b += b"c" + module.encode("utf-8") + b"\n" + name.encode("utf-8") + b"\n"
        self._put()

    def mark(self):
        self.b += b"("

    def tuple(self):
        self.b += b"t"

    def tuple2(self):
        self.b += b"\x86"

    def empty_tuple(self):
        self.b += b")"

    def reduce(self):
        self.b += b"R"
        self._put()

    def binpersid(self):
        self.b += b"Q"

    def binint1(self, n):
        self.b += b"K" + bytes([n])

    def newfalse(self):
        self.b += b"\x89"

    def setitems(self):
        self.b += b"u"


def build_pytorch_pickle():
    p = Pickler()
    p.proto()

    # od = OrderedDict()
    p.global_("collections", "OrderedDict")
    p.empty_tuple()
    p.reduce()

    p.mark()

    # ---- key "w" -> torch._utils._rebuild_tensor_v2(...) ------------------
    p.unicode("w")
    p.global_("torch._utils", "_rebuild_tensor_v2")
    p.mark()
    #   storage persistent id: ("storage", FloatStorage, "0", "cpu", 6)
    p.mark()
    p.unicode("storage")
    p.global_("torch", "FloatStorage")
    p.unicode("0")            # storage key -> archive/data/0
    p.unicode("cpu")
    p.binint1(6)              # numel
    p.tuple()
    p.binpersid()
    #   storage_offset, size (2,3), stride (3,1), requires_grad, backward_hooks
    p.binint1(0)
    p.binint1(2); p.binint1(3); p.tuple2()
    p.binint1(3); p.binint1(1); p.tuple2()
    p.newfalse()
    p.global_("collections", "OrderedDict"); p.empty_tuple(); p.reduce()
    p.tuple()                 # 6-arg tuple for _rebuild_tensor_v2
    p.reduce()

    # ---- key "junk" -> GLOBAL to a NON-allowlisted target (opaque path) ---
    p.unicode("junk")
    p.global_("numpy.core", "foo")

    p.setitems()
    p.stop()
    return bytes(p.b)


def build_pytorch(path):
    pickle_bytes = build_pytorch_pickle()
    storage_bytes = b"".join(struct.pack("<f", float(i)) for i in range(6))  # 24 bytes

    # Deterministic zip: fixed timestamps, STORED (uncompressed) so payload
    # offsets map directly into the mmap (the whole product thesis, spec §2.1).
    if os.path.exists(path):
        os.remove(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for arcname, data in (("archive/data.pkl", pickle_bytes),
                              ("archive/data/0", storage_bytes)):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, data)


def build_torchscript_constants():
    """Minimal protocol-2 pickle for constants.pkl (allowlist-safe)."""
    p = Pickler()
    p.proto()
    # Empty tuple — allowlist-safe, exercises constants.pkl path
    p.empty_tuple()
    p.stop()
    return bytes(p.b)


def build_torchscript(path):
    """TorchScript archive: data.pkl + constants.pkl + code/*.py entries."""
    pickle_bytes = build_pytorch_pickle()
    storage_bytes = b"".join(struct.pack("<f", float(i)) for i in range(6))  # 24 bytes
    constants_bytes = build_torchscript_constants()

    # Synthesized TorchScript code with method definitions and op calls
    code_py = b"""
def forward(self, x):
    y = torch.relu(x)
    z = torch.add(y, x)
    return aten::matmul(z, z)

def helper(a, b):
    return ops.custom_op(a, b)
"""

    if os.path.exists(path):
        os.remove(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        entries = [
            ("archive/data.pkl", pickle_bytes),
            ("archive/data/0", storage_bytes),
            ("archive/constants.pkl", constants_bytes),
            ("archive/code/model.py", code_py),
        ]
        for arcname, data in entries:
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, data)


# ---------------------------------------------------------------------------
# TFLite: a hand-built FlatBufferBuilder (bottom-up, vtables, alignment)
# ---------------------------------------------------------------------------

class FlatBufferBuilder:
    """Minimal FlatBuffers builder that writes back-to-front, matching the
    reference algorithm (Prep/vtable/soffset). Enough to emit a valid TFLite
    Model table with the TFL3 file_identifier at bytes 4..8 (spec §10)."""

    def __init__(self, size=4096):
        self.Bytes = bytearray(size)
        self.head = size
        self.minalign = 1
        self.current_vtable = None
        self.object_end = None

    def offset(self):
        return len(self.Bytes) - self.head

    def _pad(self, n):
        for _ in range(n):
            self.head -= 1
            self.Bytes[self.head] = 0

    def prep(self, size, additional):
        if size > self.minalign:
            self.minalign = size
        used = len(self.Bytes) - self.head + additional
        alignsize = ((~used) + 1) & (size - 1)
        assert self.head >= alignsize + size + additional, "fixed buffer too small"
        self._pad(alignsize)

    def _place(self, x, fmt):
        sz = struct.calcsize(fmt)
        self.head -= sz
        struct.pack_into(fmt, self.Bytes, self.head, x)

    def prepend_u32(self, x):
        self.prep(4, 0); self._place(x, "<I")

    def prepend_i32(self, x):
        self.prep(4, 0); self._place(x, "<i")

    def prepend_bool(self, x):
        self.prep(1, 0); self._place(1 if x else 0, "<B")

    def prepend_u8(self, x):
        self.prep(1, 0); self._place(x, "<B")

    def prepend_uoffset(self, off):
        self.prep(4, 0)
        assert off <= self.offset()
        self._place(self.offset() - off + 4, "<I")

    # -- strings / byte vectors ------------------------------------------
    def create_string(self, s):
        data = s.encode("utf-8")
        self.prep(4, len(data) + 1)
        self.head -= 1
        self.Bytes[self.head] = 0                       # null terminator
        self.head -= len(data)
        self.Bytes[self.head:self.head + len(data)] = data
        self._place(len(data), "<I")                    # length prefix
        return self.offset()

    def create_byte_vector(self, data):
        self.start_vector(1, len(data), 1)
        self.head -= len(data)
        self.Bytes[self.head:self.head + len(data)] = data
        return self.end_vector(len(data))

    # -- vectors ----------------------------------------------------------
    def start_vector(self, elem_size, num_elems, alignment):
        self.prep(4, elem_size * num_elems)
        self.prep(alignment, elem_size * num_elems)

    def end_vector(self, num_elems):
        self.head -= 4
        struct.pack_into("<I", self.Bytes, self.head, num_elems)
        return self.offset()

    def create_int_vector(self, values):
        self.start_vector(4, len(values), 4)
        for v in reversed(values):
            self.prepend_i32(v)
        return self.end_vector(len(values))

    def create_offset_vector(self, offsets):
        self.start_vector(4, len(offsets), 4)
        for o in reversed(offsets):
            self.prepend_uoffset(o)
        return self.end_vector(len(offsets))

    # -- tables -----------------------------------------------------------
    def start_object(self, numfields):
        self.current_vtable = [0] * numfields
        self.object_end = self.offset()

    def slot(self, o):
        self.current_vtable[o] = self.offset()

    def add_i32(self, o, x, default):
        if x != default:
            self.prepend_i32(x); self.slot(o)

    def add_u32(self, o, x, default):
        if x != default:
            self.prepend_u32(x); self.slot(o)

    def add_u8(self, o, x, default):
        if x != default:
            self.prepend_u8(x); self.slot(o)

    def add_bool(self, o, x, default):
        if x != default:
            self.prepend_bool(x); self.slot(o)

    def add_offset(self, o, off, default):
        if off != default:
            self.prepend_uoffset(off); self.slot(o)

    def end_object(self):
        # Placeholder soffset (patched below).
        self.prep(4, 0)
        self._place(0, "<i")
        object_offset = self.offset()

        # Write field voffsets in reverse, then the two metadata voffsets.
        i = len(self.current_vtable) - 1
        while i >= 0:
            off = 0 if self.current_vtable[i] == 0 else object_offset - self.current_vtable[i]
            self.prep(2, 0); self._place(off, "<H")
            i -= 1
        self.prep(2, 0); self._place(object_offset - self.object_end, "<H")  # object size
        self.prep(2, 0); self._place((len(self.current_vtable) + 2) * 2, "<H")  # vtable size

        # Patch the object's soffset to point back to this vtable.
        obj_pos = len(self.Bytes) - object_offset
        struct.pack_into("<i", self.Bytes, obj_pos, self.offset() - object_offset)
        return object_offset

    def finish(self, root, file_identifier):
        assert len(file_identifier) == 4
        self.prep(self.minalign, 4 + 4)
        for i in range(3, -1, -1):
            self.head -= 1
            self.Bytes[self.head] = file_identifier[i]
        self.prepend_uoffset(root)
        return bytes(self.Bytes[self.head:])


# TensorType.FLOAT32 == 0 (default -> omitted).
def build_tflite():
    b = FlatBufferBuilder()

    # --- buffers ---------------------------------------------------------
    # buffer0 is the conventional empty buffer; buffer1 holds tensor payload.
    payload = b"".join(struct.pack("<f", float(i)) for i in range(6))  # 24 bytes
    data_vec = b.create_byte_vector(payload)
    b.start_object(1)
    b.add_offset(0, data_vec, 0)     # Buffer.data
    buffer1 = b.end_object()
    b.start_object(1)
    buffer0 = b.end_object()         # empty buffer
    buffers = b.create_offset_vector([buffer0, buffer1])

    # --- tensors ---------------------------------------------------------
    name_in = b.create_string("input")
    shape_in = b.create_int_vector([2, 3])
    b.start_object(8)
    b.add_offset(0, shape_in, 0)     # shape
    # type FLOAT32 == 0 -> default, omitted (slot 1)
    b.add_u32(2, 1, 0)               # buffer index 1 (has payload)
    b.add_offset(3, name_in, 0)      # name
    tensor_in = b.end_object()

    name_out = b.create_string("output")
    shape_out = b.create_int_vector([2, 3])
    b.start_object(8)
    b.add_offset(0, shape_out, 0)
    b.add_u32(2, 0, 0)               # buffer 0 (empty) -> default, omitted
    b.add_offset(3, name_out, 0)
    tensor_out = b.end_object()

    tensors = b.create_offset_vector([tensor_in, tensor_out])

    # --- operator --------------------------------------------------------
    op_inputs = b.create_int_vector([0])
    op_outputs = b.create_int_vector([1])
    b.start_object(6)
    b.add_u32(0, 0, 0xFFFFFFFF)      # opcode_index 0 (force present)
    b.add_offset(1, op_inputs, 0)
    b.add_offset(2, op_outputs, 0)
    operator = b.end_object()
    operators = b.create_offset_vector([operator])

    # --- subgraph --------------------------------------------------------
    sg_name = b.create_string("main")
    sg_inputs = b.create_int_vector([0])
    sg_outputs = b.create_int_vector([1])
    b.start_object(5)
    b.add_offset(0, tensors, 0)
    b.add_offset(1, sg_inputs, 0)
    b.add_offset(2, sg_outputs, 0)
    b.add_offset(3, operators, 0)
    b.add_offset(4, sg_name, 0)
    subgraph = b.end_object()
    subgraphs = b.create_offset_vector([subgraph])

    # --- operator_codes (ADD: builtin_code == 0 == default, empty table) --
    b.start_object(4)
    opcode0 = b.end_object()
    operator_codes = b.create_offset_vector([opcode0])

    # --- model -----------------------------------------------------------
    description = b.create_string("netvis-test")
    b.start_object(7)
    b.add_u32(0, 3, 0)               # version 3
    b.add_offset(1, operator_codes, 0)
    b.add_offset(2, subgraphs, 0)
    b.add_offset(3, description, 0)
    b.add_offset(4, buffers, 0)
    model = b.end_object()

    return b.finish(model, b"TFL3")


# TFLite with control flow: main subgraph holds one IF operator whose
# builtin_options (union tag 92 = IfOptions in the canonical schema.fbs
# BuiltinOptions enum) references then_subgraph_index=1. Exercises the
# f_operator field-3/4 fix + subgraph linking (spec §7.x).
def build_tflite_ctrlflow():
    b = FlatBufferBuilder(size=8192)

    # --- buffers: single empty buffer -----------------------------------
    b.start_object(1)
    buffer0 = b.end_object()
    buffers = b.create_offset_vector([buffer0])

    # ================= then-branch subgraph (index 1) ===================
    # One tensor, no operators; just needs to be a valid subgraph.
    t1_name = b.create_string("then_out")
    t1_shape = b.create_int_vector([1])
    b.start_object(8)
    b.add_offset(0, t1_shape, 0)
    b.add_offset(3, t1_name, 0)
    then_tensor = b.end_object()
    then_tensors = b.create_offset_vector([then_tensor])
    then_name = b.create_string("then_branch")
    b.start_object(5)
    b.add_offset(0, then_tensors, 0)
    b.add_offset(4, then_name, 0)
    then_subgraph = b.end_object()

    # ================= main subgraph (index 0) ==========================
    mt_name = b.create_string("cond")
    mt_shape = b.create_int_vector([1])
    b.start_object(8)
    b.add_offset(0, mt_shape, 0)
    b.add_offset(3, mt_name, 0)
    main_tensor = b.end_object()
    main_tensors = b.create_offset_vector([main_tensor])

    # IfOptions table: then_subgraph_index=1 (field 0), else=0 (field 1 default).
    b.start_object(2)
    b.add_i32(0, 1, 0x7FFFFFFF)   # then_subgraph_index = 1 (force present)
    if_options = b.end_object()

    op_inputs = b.create_int_vector([0])
    op_outputs = b.create_int_vector([0])
    b.start_object(6)
    b.add_u32(0, 0, 0xFFFFFFFF)   # opcode_index 0 (force present)
    b.add_offset(1, op_inputs, 0)
    b.add_offset(2, op_outputs, 0)
    b.add_u8(3, 92, 0)            # builtin_options_type = IfOptions (schema ord 92)
    b.add_offset(4, if_options, 0)
    operator = b.end_object()
    operators = b.create_offset_vector([operator])

    main_name = b.create_string("main")
    main_inputs = b.create_int_vector([0])
    main_outputs = b.create_int_vector([0])
    b.start_object(5)
    b.add_offset(0, main_tensors, 0)
    b.add_offset(1, main_inputs, 0)
    b.add_offset(2, main_outputs, 0)
    b.add_offset(3, operators, 0)
    b.add_offset(4, main_name, 0)
    main_subgraph = b.end_object()

    subgraphs = b.create_offset_vector([main_subgraph, then_subgraph])

    # --- operator_codes: IF builtin_code == 118 -------------------------
    b.start_object(4)
    b.add_i32(3, 118, 0)          # builtin_code = IF
    opcode0 = b.end_object()
    operator_codes = b.create_offset_vector([opcode0])

    description = b.create_string("netvis-ctrlflow")
    b.start_object(7)
    b.add_u32(0, 3, 0)
    b.add_offset(1, operator_codes, 0)
    b.add_offset(2, subgraphs, 0)
    b.add_offset(3, description, 0)
    b.add_offset(4, buffers, 0)
    model = b.end_object()

    return b.finish(model, b"TFL3")


# ---------------------------------------------------------------------------
# OpenVINO IR (.xml topology + sibling .bin weight blob)  — issue #39
# ---------------------------------------------------------------------------

def build_openvino_xml():
    """A tiny OpenVINO IR v11 net: Parameter -> Convolution(Const weight) ->
    ReLU -> Result. The Const's <data offset="0" size="16" ...> points into the
    16-byte model.bin sibling, exercising the external-data resolution path. Hand
    written as text so the fixture documents the XML subset the reader accepts."""
    return (
        '<?xml version="1.0"?>\n'
        '<net name="tiny_ov" version="11">\n'
        '  <layers>\n'
        '    <layer id="0" name="in" type="Parameter">\n'
        '      <data shape="1,1,2,2" element_type="f32"/>\n'
        '      <output>\n'
        '        <port id="0" precision="FP32">\n'
        '          <dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim>\n'
        '        </port>\n'
        '      </output>\n'
        '    </layer>\n'
        '    <layer id="1" name="weights" type="Const">\n'
        '      <data offset="0" size="16" element_type="f32" shape="2,2"/>\n'
        '      <output>\n'
        '        <port id="1" precision="FP32">\n'
        '          <dim>2</dim><dim>2</dim>\n'
        '        </port>\n'
        '      </output>\n'
        '    </layer>\n'
        '    <layer id="2" name="conv" type="Convolution">\n'
        '      <data strides="1,1" pads_begin="0,0" pads_end="0,0" dilations="1,1"/>\n'
        '      <input>\n'
        '        <port id="0" precision="FP32">\n'
        '          <dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim>\n'
        '        </port>\n'
        '        <port id="1" precision="FP32">\n'
        '          <dim>2</dim><dim>2</dim>\n'
        '        </port>\n'
        '      </input>\n'
        '      <output>\n'
        '        <port id="2" precision="FP32">\n'
        '          <dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim>\n'
        '        </port>\n'
        '      </output>\n'
        '    </layer>\n'
        '    <layer id="3" name="relu" type="ReLU">\n'
        '      <input>\n'
        '        <port id="0" precision="FP32">\n'
        '          <dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim>\n'
        '        </port>\n'
        '      </input>\n'
        '      <output>\n'
        '        <port id="1" precision="FP32">\n'
        '          <dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim>\n'
        '        </port>\n'
        '      </output>\n'
        '    </layer>\n'
        '    <layer id="4" name="out" type="Result">\n'
        '      <input>\n'
        '        <port id="0" precision="FP32">\n'
        '          <dim>1</dim><dim>1</dim><dim>2</dim><dim>2</dim>\n'
        '        </port>\n'
        '      </input>\n'
        '    </layer>\n'
        '  </layers>\n'
        '  <edges>\n'
        '    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>\n'
        '    <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>\n'
        '    <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>\n'
        '    <edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>\n'
        '  </edges>\n'
        '  <rt_info/>\n'
        '</net>\n'
    ).encode("utf-8")


def build_openvino_bin():
    """The sibling weight blob: 16 bytes = four F32 (a 2x2 Const)."""
    return b"".join(struct.pack("<f", f) for f in (1.0, 2.0, 3.0, 4.0))


def build_openvino_quant_xml():
    """OpenVINO IR v11 with a Const whose element_type is a sub-byte quant type
    (nf4) that has no ir::DType. The parser must keep dtype=Unknown but record the
    raw element_type as TensorRef::dtype_label. There is intentionally NO sibling
    .bin for this fixture (issue #85 "weights blob not found" path): the topology
    must still parse and a metadata note must be emitted. (#85)"""
    return (
        '<?xml version="1.0"?>\n'
        '<net name="quant_ov" version="11">\n'
        '  <layers>\n'
        '    <layer id="0" name="w" type="Const">\n'
        '      <data offset="0" size="8" element_type="nf4" shape="4,4"/>\n'
        '      <output>\n'
        '        <port id="0" precision="NF4">\n'
        '          <dim>4</dim><dim>4</dim>\n'
        '        </port>\n'
        '      </output>\n'
        '    </layer>\n'
        '    <layer id="1" name="out" type="Result">\n'
        '      <input>\n'
        '        <port id="0" precision="NF4">\n'
        '          <dim>4</dim><dim>4</dim>\n'
        '        </port>\n'
        '      </input>\n'
        '    </layer>\n'
        '  </layers>\n'
        '  <edges>\n'
        '    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>\n'
        '  </edges>\n'
        '</net>\n'
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# NumPy .npz (zip of .npy arrays)
# ---------------------------------------------------------------------------

def build_npz_npy(shape, dtype_descr, values):
    """Build a single .npy array: magic, version, header dict, payload.
    shape: tuple of dims, e.g. (2, 3).
    dtype_descr: NumPy descr string, e.g. '<f4'.
    values: flat list of numeric values (will be packed per descr).
    """
    # Shape tuple string: "(2, 3)" for rank>=2, "(3,)" for rank 1, "()" for scalar.
    if len(shape) == 0:
        shape_str = "()"
    elif len(shape) == 1:
        shape_str = f"({shape[0]},)"
    else:
        shape_str = "(" + ", ".join(str(d) for d in shape) + ")"

    # Header dict (Python literal).
    dict_str = f"{{'descr': '{dtype_descr}', 'fortran_order': False, 'shape': {shape_str}, }}"

    # Version 1.0: magic(6) + version(2) + u16 header_len + dict + padding to 64-byte boundary.
    magic = b"\x93NUMPY"
    version = b"\x01\x00"
    preamble_size = 6 + 2 + 2  # magic + version + u16
    dict_bytes = dict_str.encode("utf-8")
    # Total before padding: preamble + dict + '\n'.
    unpadded = preamble_size + len(dict_bytes) + 1
    # Pad to 64-byte boundary.
    total = ((unpadded + 63) // 64) * 64
    pad = total - unpadded
    dict_bytes += b" " * pad + b"\n"
    header_len = len(dict_bytes)

    # Payload: pack values according to dtype_descr.
    payload = bytearray()
    if dtype_descr == "<f4":
        for v in values:
            payload += struct.pack("<f", float(v))
    elif dtype_descr == "<i4":
        for v in values:
            payload += struct.pack("<i", int(v))
    else:
        # Extend as needed; for the fixture we only need f4.
        raise ValueError(f"unsupported dtype_descr: {dtype_descr}")

    out = bytearray()
    out += magic
    out += version
    out += struct.pack("<H", header_len)
    out += dict_bytes
    out += payload
    return bytes(out)


def build_npz(path):
    """Build a .npz fixture: a ZIP (STORED) of two hand-written .npy arrays."""
    # Two arrays: w (2,3) f32, b (3,) f32.
    w_npy = build_npz_npy((2, 3), "<f4", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b_npy = build_npz_npy((3,), "<f4", [0.1, 0.2, 0.3])

    # Deterministic zip: STORED, fixed timestamps (spec §10, same as build_pytorch).
    if os.path.exists(path):
        os.remove(path)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for arcname, data in (("w.npy", w_npy), ("b.npy", b_npy)):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, data)


# ---------------------------------------------------------------------------
# CoreML .mlmodel (protobuf Model / NeuralNetwork / NeuralNetworkLayer)
# ---------------------------------------------------------------------------
# Reuses the generic pb_* helpers above. Field numbers verified against Apple
# coremltools mlmodel/format/{Model,NeuralNetwork}.proto:
#   Model:              1 specificationVersion, 2 description, 500 neuralNetwork
#   NeuralNetwork:      1 layers (repeated NeuralNetworkLayer)
#   NeuralNetworkLayer: 1 name, 2 input(rep), 3 output(rep), 140 innerProduct
#   InnerProductLayer:  ... 5 weights (WeightParams)
#   WeightParams:       30 rawValue (bytes)  <- payload; parser records offset+len
#   ModelDescription:   1 input, 10 output (repeated FeatureDescription)
#   FeatureDescription: 1 name


def coreml_weight_params_raw(raw):
    """WeightParams with rawValue(30) bytes — the payload the parser records as a
    TensorRef offset+len and never decodes."""
    return pb_len(30, raw)


def coreml_inner_product(weight_raw):
    """InnerProductLayer with a weights(5) WeightParams sub-message."""
    return pb_len(5, coreml_weight_params_raw(weight_raw))


def coreml_layer(name, inputs, outputs, kind_field, kind_body):
    """NeuralNetworkLayer: name(1), input(2)*, output(3)*, oneof layer(kind_field)."""
    b = bytearray()
    b += pb_string(1, name)
    for i in inputs:
        b += pb_string(2, i)
    for o in outputs:
        b += pb_string(3, o)
    b += pb_len(kind_field, kind_body)   # oneof layer (e.g. 140 = innerProduct)
    return bytes(b)


def coreml_feature(name):
    """FeatureDescription: name(1)."""
    return pb_string(1, name)


def build_coreml_mlmodel():
    # One innerProduct layer: input "data" -> output "fc_out", with a rawValue
    # weight of four F32 (16 bytes). The parser records the rawValue sub-range as
    # a TensorRef offset+len (never read) and wires data->fc_out edges.
    weight_raw = b"".join(struct.pack("<f", f) for f in (1.0, 2.0, 3.0, 4.0))
    ip = coreml_inner_product(weight_raw)
    layer = coreml_layer("fc1", ["data"], ["fc_out"], 140, ip)

    neural_network = pb_len(1, layer)     # NeuralNetwork.layers (repeated)

    description = bytearray()
    description += pb_len(1, coreml_feature("data"))     # input FeatureDescription
    description += pb_len(10, coreml_feature("fc_out"))  # output FeatureDescription

    model = bytearray()
    model += pb_varint(1, 4)              # specificationVersion = 4
    model += pb_len(2, bytes(description))  # description (ModelDescription)
    model += pb_len(500, neural_network)  # neuralNetwork (oneof Type)
    return bytes(model)


# ---------------------------------------------------------------------------
# CoreML .mlpackage (mlProgram / MIL) — issue #85
# ---------------------------------------------------------------------------
# A modern Apple export: a DIRECTORY bundle, not a single file.
#
#   model.mlpackage/
#     Manifest.json
#     Data/com.apple.CoreML/model.mlmodel     (Model proto, mlProgram=field 502)
#     Data/com.apple.CoreML/weights/weight.bin (MIL blob v2 storage)
#
# Field numbers verified against apple/coremltools mlmodel/format/{Model,MIL}.proto:
#   Model:            502 mlProgram (MILSpec.Program)
#   Program:          1 version(int64), 2 functions(map<string,Function>)
#   Function:         2 opset(string), 3 block_specializations(map<string,Block>)
#   Block:            1 inputs(NamedValueType*), 2 outputs(string*), 3 operations*
#   Operation:        1 type(string), 2 inputs(map<string,Argument>), 3 outputs(NVT*)
#   NamedValueType:   1 name(string), 2 type(ValueType)
#   Argument:         1 arguments(Binding*);  Binding: 1 name(string) | 2 value(Value)
#   Value:            2 type(ValueType), 3 immediateValue, 5 blobFileValue
#   BlobFileValue:    1 fileName(string), 2 offset(uint64)
#   ValueType:        1 tensorType(TensorType)
#   TensorType:       1 dataType(enum), 2 rank(int64), 3 dimensions(Dimension*)
#   Dimension:        1 constant(ConstantDimension{1 size uint64})
#   DataType enum:    FLOAT32=11, FLOAT16=10, INT8=21  (distinct from blob enum!)
#
# weight.bin MIL blob v2 (MILBlob/Blob/StorageFormat.hpp):
#   storage_header 64B: count u32@0, version u32@4(=2), 7x reserved u64
#   blob_metadata  64B: sentinel u32@0 =0xDEADBEEF, mil_dtype u32@4,
#                       sizeInBytes u64@8, data_offset u64@16, padding_bits u64@24,
#                       4x reserved u64.  Everything 64-byte aligned.
#   BlobFileValue.offset -> the blob_metadata header (NOT the raw data).


def pb_map_entry(field, key, value_bytes):
    """A proto3 map<string,MSG> field on the wire: one length-delimited entry per
    key, entry = { 1: key(string), 2: value(message) }."""
    entry = pb_string(1, key) + pb_len(2, value_bytes)
    return pb_len(field, entry)


def mil_tensor_type(data_type, dims):
    """ValueType{ tensorType(1) = TensorType{ dataType(1), rank(2), dimensions(3) } }."""
    tt = bytearray()
    tt += pb_varint(1, data_type)          # dataType enum
    tt += pb_varint(2, len(dims))          # rank
    for d in dims:
        # Dimension{ constant(1) = ConstantDimension{ size(1) uint64 } }
        const_dim = pb_varint(1, d)
        tt += pb_len(3, pb_len(1, const_dim))
    return pb_len(1, bytes(tt))            # ValueType.tensorType = field 1


def mil_named_value_type(name, data_type, dims):
    """NamedValueType{ name(1), type(2) = ValueType }."""
    b = pb_string(1, name) + pb_len(2, mil_tensor_type(data_type, dims))
    return bytes(b)


def mil_arg_name(value_name):
    """Argument{ arguments(1) = [ Binding{ name(1) = value_name } ] } — an op input
    that references another value by its SSA name."""
    binding = pb_string(1, value_name)     # Binding.name (oneof binding, field 1)
    return pb_len(1, binding)              # Argument.arguments (repeated Binding)


def mil_blob_value(data_type, dims, file_name, offset):
    """Value{ type(2)=ValueType, blobFileValue(5)=BlobFileValue{fileName(1),offset(2)} }
    — a weight stored in weight.bin at `offset` (points at the blob_metadata hdr)."""
    bfv = pb_string(1, file_name) + pb_varint(2, offset)
    b = pb_len(2, mil_tensor_type(data_type, dims)) + pb_len(5, bytes(bfv))
    return bytes(b)


def build_weight_bin():
    """MIL blob v2 weight.bin with one Float32 blob of four F32 (16 bytes).
    Returns (bytes, metadata_offset) — the offset the BlobFileValue must carry."""
    ALIGN = 64
    SENTINEL = 0xDEADBEEF
    BLOB_FLOAT32 = 2                        # BlobDataType.Float32 (blob enum!)
    raw = b"".join(struct.pack("<f", f) for f in (1.0, 2.0, 3.0, 4.0))

    # storage_header @0 (64B): count=1, version=2, reserved.
    hdr = bytearray()
    hdr += struct.pack("<I", 1)            # count
    hdr += struct.pack("<I", 2)            # version = 2
    hdr += b"\x00" * (64 - 8)              # 7x reserved u64
    assert len(hdr) == 64

    meta_off = ALIGN                       # first metadata at 64
    data_off = meta_off + ALIGN            # its data at 128 (64-aligned)

    meta = bytearray()
    meta += struct.pack("<I", SENTINEL)    # sentinel @0
    meta += struct.pack("<I", BLOB_FLOAT32)  # mil_dtype @4
    meta += struct.pack("<Q", len(raw))    # sizeInBytes @8
    meta += struct.pack("<Q", data_off)    # data_offset @16
    meta += struct.pack("<Q", 0)           # padding_bits @24
    meta += b"\x00" * (64 - 32)            # 4x reserved u64
    assert len(meta) == 64

    blob = bytearray()
    blob += hdr
    blob += meta
    blob += raw
    return bytes(blob), meta_off


def build_coreml_mlprogram(weight_offset):
    """Model proto carrying an mlProgram (field 502). One function "main" with one
    block whose single op `linear` takes input "x" + a blobFileValue weight and
    produces "out". weight_offset points at the blob_metadata header in weight.bin.
    """
    # Operation: type="linear", inputs{ x: name-binding "x", weight: blob },
    #            outputs=[ NamedValueType("out", FLOAT32 [2,2]) ]
    FLOAT32 = 11                           # MIL proto DataType.FLOAT32
    op = bytearray()
    op += pb_string(1, "linear")           # Operation.type
    op += pb_map_entry(2, "x", bytes(mil_arg_name("x")))       # inputs["x"]
    op += pb_map_entry(2, "weight",
                       bytes(mil_blob_value(FLOAT32, [2, 2],
                                            "weights/weight.bin", weight_offset)))
    op += pb_len(3, mil_named_value_type("out", FLOAT32, [2, 2]))  # outputs

    block = bytearray()
    block += pb_len(1, mil_named_value_type("x", FLOAT32, [2, 2]))  # Block.inputs
    block += pb_string(2, "out")           # Block.outputs (repeated string)
    block += pb_len(3, bytes(op))          # Block.operations

    func = bytearray()
    func += pb_string(2, "CoreML6")        # Function.opset
    func += pb_map_entry(3, "CoreML6", bytes(block))  # block_specializations

    program = bytearray()
    program += pb_varint(1, 1)             # Program.version
    program += pb_map_entry(2, "main", bytes(func))   # functions["main"]

    model = bytearray()
    model += pb_varint(1, 7)               # specificationVersion = 7 (iOS16)
    model += pb_len(502, bytes(program))   # mlProgram (oneof Type)
    return bytes(model)


def build_mlpackage(out_dir):
    """Write a model.mlpackage DIRECTORY bundle under out_dir."""
    import json
    pkg = os.path.join(out_dir, "model.mlpackage")
    inner_dir = os.path.join(pkg, "Data", "com.apple.CoreML")
    weights_dir = os.path.join(inner_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    weight_bin, meta_off = build_weight_bin()
    with open(os.path.join(weights_dir, "weight.bin"), "wb") as f:
        f.write(weight_bin)
    with open(os.path.join(inner_dir, "model.mlmodel"), "wb") as f:
        f.write(build_coreml_mlprogram(meta_off))

    # Manifest.json: rootModelIdentifier -> itemInfoEntries[uuid].path (=<author>/<name>).
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            "11111111-1111-1111-1111-111111111111": {
                "path": "com.apple.CoreML/model.mlmodel",
                "name": "model.mlmodel",
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
            }
        },
        "rootModelIdentifier": "11111111-1111-1111-1111-111111111111",
    }
    with open(os.path.join(pkg, "Manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return pkg


# ---------------------------------------------------------------------------
# Keras / HDF5 (classic symbol-table form) — hand-encoded, matches the minimal
# reader in src/parsers/keras/Hdf5Reader.cpp exactly.
#
# Layout (all 8-byte offsets/lengths, superblock v0, v1 object headers):
#   superblock -> root-group symbol-table entry -> root-group object header
#   (one Symbol Table Message giving B-tree + local-heap addresses) ->
#   v1 B-tree "TREE" (1 leaf entry) -> "SNOD" (1 symbol "kernel") ->
#   dataset object header (Dataspace [2,2] + Datatype F32 + contiguous Layout)
#   -> 16 bytes of raw F32 data. The reader records only (offset,len).
# ---------------------------------------------------------------------------

def _h5_object_header(messages):
    """v1 object header: 16-byte prefix (version, msg count, refcount, size,
    pad) + concatenated messages. Each message's data is already 8-padded."""
    body = b"".join(messages)
    hdr = bytearray()
    hdr += bytes([1, 0])                       # version 1, reserved
    hdr += struct.pack("<H", len(messages))    # number of header messages
    hdr += struct.pack("<I", 1)                # object reference count
    hdr += struct.pack("<I", len(body))        # header size (all message bytes)
    hdr += bytes(4)                            # pad prefix to 16 bytes
    return bytes(hdr) + body


def _h5_message(type_id, data):
    """v1 header message: type(2), size(2), flags(1), reserved(3), then data.
    Data is padded to a multiple of 8 as the format requires."""
    pad = (-len(data)) % 8
    data = data + bytes(pad)
    return struct.pack("<H", type_id) + struct.pack("<H", len(data)) + bytes([0]) + bytes(3) + data


def build_keras_h5():
    U = 0xFFFFFFFFFFFFFFFF  # HDF5 "undefined address"

    # --- fixed-size blocks; compute absolute addresses up front --------------
    SB_LEN = 96
    ROOT_OH_LEN = 16 + 24          # prefix + one 24-byte Symbol Table Message
    HEAP_HDR_LEN = 32
    HEAPDATA_LEN = 16
    BTREE_LEN = 48
    SNOD_LEN = 48
    DSET_OH_LEN = 16 + 96          # prefix + 3 * (8 hdr + 24 data)
    DATA_LEN = 16

    a_root_oh = SB_LEN
    a_heap = a_root_oh + ROOT_OH_LEN
    a_heapdata = a_heap + HEAP_HDR_LEN
    a_btree = a_heapdata + HEAPDATA_LEN
    a_snod = a_btree + BTREE_LEN
    a_dset_oh = a_snod + SNOD_LEN
    a_data = a_dset_oh + DSET_OH_LEN
    eof = a_data + DATA_LEN

    # --- superblock v0 (96 bytes) -------------------------------------------
    sb = bytearray()
    sb += b"\x89HDF\r\n\x1a\n"                  # 0-7  signature
    sb += bytes([0, 0, 0, 0, 0])               # 8-12 versions + reserved
    sb += bytes([8])                           # 13   size of offsets
    sb += bytes([8])                           # 14   size of lengths
    sb += bytes([0])                           # 15   reserved
    sb += struct.pack("<H", 4)                 # 16-17 group leaf node K
    sb += struct.pack("<H", 16)                # 18-19 group internal node K
    sb += struct.pack("<I", 0)                 # 20-23 consistency flags
    sb += struct.pack("<Q", 0)                 # 24-31 base address
    sb += struct.pack("<Q", U)                 # 32-39 free-space address
    sb += struct.pack("<Q", eof)               # 40-47 end-of-file address
    sb += struct.pack("<Q", U)                 # 48-55 driver info address
    # root-group symbol-table entry (40 bytes) at offset 56
    sb += struct.pack("<Q", 0)                 # 56-63 link name offset
    sb += struct.pack("<Q", a_root_oh)         # 64-71 object header address
    sb += struct.pack("<I", 1)                 # 72-75 cache type 1
    sb += struct.pack("<I", 0)                 # 76-79 reserved
    sb += struct.pack("<Q", a_btree)           # 80-87 scratch: B-tree addr
    sb += struct.pack("<Q", a_heap)            # 88-95 scratch: heap addr
    assert len(sb) == SB_LEN

    # --- root group object header (Symbol Table Message: btree + heap) -------
    stm = struct.pack("<Q", a_btree) + struct.pack("<Q", a_heap)   # 16 bytes
    root_oh = _h5_object_header([_h5_message(0x0011, stm)])
    assert len(root_oh) == ROOT_OH_LEN, (len(root_oh), ROOT_OH_LEN)

    # --- local heap: header + data segment ("kernel\0" at offset 8) ----------
    heap = bytearray()
    heap += b"HEAP" + bytes([0]) + bytes(3)    # signature, version, reserved
    heap += struct.pack("<Q", HEAPDATA_LEN)    # data segment size
    heap += struct.pack("<Q", 0)               # free-list head offset
    heap += struct.pack("<Q", a_heapdata)      # data segment address
    assert len(heap) == HEAP_HDR_LEN
    heapdata = bytearray(HEAPDATA_LEN)
    name = b"kernel\x00"
    heapdata[8:8 + len(name)] = name           # name offset 8

    # --- v1 B-tree (group nodes, 1 leaf entry -> SNOD) -----------------------
    bt = bytearray()
    bt += b"TREE" + bytes([0]) + bytes([0]) + struct.pack("<H", 1)  # sig,type0,level0,entries1
    bt += struct.pack("<Q", U) + struct.pack("<Q", U)               # left/right sibling
    bt += struct.pack("<Q", 0)                 # key0 (heap offset)
    bt += struct.pack("<Q", a_snod)            # child0 -> SNOD
    bt += struct.pack("<Q", 0)                 # key1 (heap offset)
    assert len(bt) == BTREE_LEN

    # --- SNOD: one symbol "kernel" -> dataset object header ------------------
    snod = bytearray()
    snod += b"SNOD" + bytes([1]) + bytes([0]) + struct.pack("<H", 1)
    snod += struct.pack("<Q", 8)               # name offset in heap -> "kernel"
    snod += struct.pack("<Q", a_dset_oh)       # object header address
    snod += struct.pack("<I", 0)               # cache type 0
    snod += struct.pack("<I", 0)               # reserved
    snod += bytes(16)                          # scratch pad
    assert len(snod) == SNOD_LEN

    # --- dataset object header: Dataspace + Datatype + Layout ---------------
    # Dataspace v1: version, rank, flags, reserved(1), reserved(4), dims[2,2].
    dspace = bytes([1, 2, 0, 0]) + bytes(4) + struct.pack("<Q", 2) + struct.pack("<Q", 2)
    # Datatype v1: class=1 (float), size=4, IEEE-F32LE properties (ignored by reader).
    dtype = bytes([0x11]) + bytes([0x20, 0x3f, 0x00]) + struct.pack("<I", 4)
    dtype += struct.pack("<H", 0) + struct.pack("<H", 32) + bytes([23, 8, 0, 23]) + struct.pack("<I", 127)
    # Data Layout v3: contiguous -> (address, size).
    layout = bytes([3, 1]) + struct.pack("<Q", a_data) + struct.pack("<Q", DATA_LEN)
    dset_oh = _h5_object_header([
        _h5_message(0x0001, dspace),
        _h5_message(0x0003, dtype),
        _h5_message(0x0008, layout),
    ])
    assert len(dset_oh) == DSET_OH_LEN, (len(dset_oh), DSET_OH_LEN)

    data = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)

    blob = bytes(sb) + root_oh + bytes(heap) + bytes(heapdata) + bytes(bt) + \
        bytes(snod) + dset_oh + data
    assert len(blob) == eof, (len(blob), eof)
    return blob


def build_keras_v3(out_dir):
    """A Keras-v3 archive: config.json + metadata.json + model.weights.h5, STORED
    so the embedded HDF5 payload offset is resolvable from the local file header."""
    path = os.path.join(out_dir, "model.keras")
    if os.path.exists(path):
        os.remove(path)
    config = b'{"module":"keras","class_name":"Sequential","config":{"name":"seq"}}'
    metadata = b'{"keras_version":"3.0.0","date_saved":"2026-07-22"}'
    weights = build_keras_h5()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for arcname, blob in (("config.json", config),
                              ("metadata.json", metadata),
                              ("model.weights.h5", weights)):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            zf.writestr(info, blob)


# ---------------------------------------------------------------------------
# WebAssembly plugin fixtures (v0.6.0 #10). Hand-encoded, stdlib-only — no wasm
# toolchain needed (same discipline as the hand-built TFLite/HDF5 fixtures).
# ---------------------------------------------------------------------------
def _uleb(n):
    """Unsigned LEB128."""
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _wasm_section(sid, body):
    return bytes([sid]) + _uleb(len(body)) + body


def _wasm_module(type_sec, func_sec, export_sec, code_sec):
    """Assemble a minimal module with type/function/export/code sections."""
    out = bytearray(b"\x00asm\x01\x00\x00\x00")  # magic + version 1
    out += _wasm_section(1, type_sec)    # Type
    out += _wasm_section(3, func_sec)    # Function
    out += _wasm_section(7, export_sec)  # Export
    out += _wasm_section(10, code_sec)   # Code
    return bytes(out)


def build_wasm_const():
    """A tiny valid module: export "run" : () -> i32 that returns 42.
    Proves the sandbox runs a well-behaved module to completion."""
    # Type section: one type () -> i32.  0x60 form, 0 params, 1 result i32(0x7F).
    types = _uleb(1) + b"\x60" + _uleb(0) + _uleb(1) + b"\x7f"
    funcs = _uleb(1) + _uleb(0)                       # one func, type index 0
    name = b"run"
    exports = _uleb(1) + _uleb(len(name)) + name + b"\x00" + _uleb(0)  # export func 0 as "run"
    # Code: locals=0; body: i32.const 42 ; end
    body = _uleb(0) + b"\x41" + _uleb(42) + b"\x0b"
    code = _uleb(1) + _uleb(len(body)) + body
    return _wasm_module(types, funcs, exports, code)


def build_wasm_pass():
    """A PASS plugin: imports netvis.host_total_flops : ()->f64 and
    netvis.host_emit_metric : (i32 ptr, i32 len, f64 value, i32 known)->void.
    On run() it emits a metric "double_flops" = 2 * host_total_flops(), known=1.
    Proves the capability host API + marshalling round-trip (and that NO weight
    buffer ever crosses — the host has no such import)."""
    # --- Type section: 3 types ---
    #  t0: () -> f64            (host_total_flops)
    #  t1: (i32,i32,f64,i32)->()(host_emit_metric)
    #  t2: () -> ()             (run)
    t0 = b"\x60" + _uleb(0) + _uleb(1) + b"\x7c"                       # ()->f64
    t1 = b"\x60" + _uleb(4) + b"\x7f\x7f\x7c\x7f" + _uleb(0)           # (i32,i32,f64,i32)->()
    t2 = b"\x60" + _uleb(0) + _uleb(0)                                 # ()->()
    types = _uleb(3) + t0 + t1 + t2

    def name(s):
        b = s.encode()
        return _uleb(len(b)) + b

    # --- Import section: two funcs from module "netvis" ---
    imp = _uleb(2)
    imp += name("netvis") + name("host_total_flops") + b"\x00" + _uleb(0)  # func type 0
    imp += name("netvis") + name("host_emit_metric") + b"\x00" + _uleb(1)  # func type 1
    # --- Function section: one local func "run" of type 2 ---
    funcs = _uleb(1) + _uleb(2)
    # --- Memory section: one memory, min 1 page (for the metric name bytes) ---
    mem = _uleb(1) + b"\x00" + _uleb(1)   # limits: flag 0 (min only), min=1
    # --- Export section: "run" (func index 2 = after 2 imported funcs) + memory ---
    exp = _uleb(2)
    exp += name("run") + b"\x00" + _uleb(2)
    exp += name("memory") + b"\x02" + _uleb(0)
    # --- Data section: metric name "double_flops" at memory offset 8 ---
    # NOT offset 0: wasm3's m3ApiIsNullPtr treats a pointer == memory base (offset
    # 0) as null and traps host_read/CheckMem. Real toolchains reserve low memory
    # for the same reason; we mirror that so a bounds-checked host read succeeds.
    NAME_OFF = 8
    mname = b"double_flops"
    data = _uleb(1) + _uleb(0) + b"\x41" + _uleb(NAME_OFF) + b"\x0b" + _uleb(len(mname)) + mname
    # --- Code for run(): value = 2.0 * host_total_flops(); emit(0, len, value, 1) ---
    #   call 0 (host_total_flops) -> f64 ; f64.const 2 ; f64.mul  => value on stack
    #   then push args for emit: ptr=0, len, <value>, known=1 ; call 1
    # WASM args are pushed in order, so: i32.const 0; i32.const len; <value>; i32.const 1; call 1
    import struct as _struct
    f64_2 = b"\x44" + _struct.pack("<d", 2.0)     # f64.const 2.0
    body = _uleb(0)                                # 0 locals
    body += b"\x41" + _uleb(NAME_OFF)              # i32.const 8 (name ptr, non-null)
    body += b"\x41" + _uleb(len(mname))            # i32.const len
    body += b"\x10" + _uleb(0)                     # call 0 (host_total_flops) -> f64
    body += f64_2 + b"\xa2"                        # f64.const 2 ; f64.mul
    body += b"\x41\x01"                            # i32.const 1 (known)
    body += b"\x10" + _uleb(1)                     # call 1 (host_emit_metric)
    body += b"\x0b"                                # end
    code = _uleb(1) + _uleb(len(body)) + body

    # Sections MUST appear in canonical id order: type(1) import(2) func(3)
    # memory(5) export(7) code(10) data(11). (code before data.)
    out = bytearray(b"\x00asm\x01\x00\x00\x00")
    out += _wasm_section(1, types)
    out += _wasm_section(2, imp)
    out += _wasm_section(3, funcs)
    out += _wasm_section(5, mem)
    out += _wasm_section(7, exp)
    out += _wasm_section(10, code)
    out += _wasm_section(11, data)
    return bytes(out)


def build_wasm_badsig():
    """A module exporting run:(i32)->() — WRONG signature. call_i32 must REJECT it
    cleanly (not va_arg past the empty arg list → UB). Regression for the review's
    export-signature finding."""
    # Type () with ONE i32 param, no result: 0x60, 1 param i32, 0 results.
    types = _uleb(1) + b"\x60" + _uleb(1) + b"\x7f" + _uleb(0)
    funcs = _uleb(1) + _uleb(0)
    name = b"run"
    exports = _uleb(1) + _uleb(len(name)) + name + b"\x00" + _uleb(0)
    body = _uleb(0) + b"\x0b"   # empty body, just end
    code = _uleb(1) + _uleb(len(body)) + body
    return _wasm_module(types, funcs, exports, code)


def build_wasm_bigmem():
    """A module declaring 30000 initial memory pages (~1.9 GiB) + run:()->i32.
    The sandbox's 256-page cap must bound the allocation at LOAD time (regression
    for the review's memory-cap-set-too-late finding)."""
    types = _uleb(1) + b"\x60" + _uleb(0) + _uleb(1) + b"\x7f"
    funcs = _uleb(1) + _uleb(0)
    mem = _uleb(1) + b"\x00" + _uleb(30000)   # limits flag 0 (min only), min=30000 pages
    name = b"run"
    exports = _uleb(1) + _uleb(len(name)) + name + b"\x00" + _uleb(0)
    body = _uleb(0) + b"\x41\x00\x0b"
    code = _uleb(1) + _uleb(len(body)) + body
    # order: type(1) func(3) memory(5) export(7) code(10)
    out = bytearray(b"\x00asm\x01\x00\x00\x00")
    out += _wasm_section(1, types)
    out += _wasm_section(3, funcs)
    out += _wasm_section(5, mem)
    out += _wasm_section(7, exports)
    out += _wasm_section(10, code)
    return bytes(out)


def build_wasm_loop():
    """A HOSTILE module: export "run" : () -> i32 with an infinite loop
    (loop; br 0; end). Must be KILLED by the sandbox fuel cap, not hang."""
    types = _uleb(1) + b"\x60" + _uleb(0) + _uleb(1) + b"\x7f"
    funcs = _uleb(1) + _uleb(0)
    name = b"run"
    exports = _uleb(1) + _uleb(len(name)) + name + b"\x00" + _uleb(0)
    # body: loop (void 0x40) ; br 0 ; end ; i32.const 0 ; end
    #   0x03 loop, 0x40 void blocktype, 0x0c br, 0x00 depth, 0x0b end(loop),
    #   0x41 0x00 i32.const 0, 0x0b end(func)
    body = _uleb(0) + b"\x03\x40" + b"\x0c" + _uleb(0) + b"\x0b" + b"\x41\x00" + b"\x0b"
    code = _uleb(1) + _uleb(len(body)) + body
    return _wasm_module(types, funcs, exports, code)


def build_wasm_start_loop():
    """A HOSTILE module whose START section runs an infinite loop, plus an exported
    run():()->i32. wasm3 runs the start fn lazily on first FindFunction, BEFORE the
    call — so the sandbox must meter that path too (regression for the adversarial
    review's start-section fuel-bypass finding). f0 = run (returns 0),
    f1 = start (loops). NB: the start index must be NON-ZERO — wasm3's lazy-start
    check is `if (module->startFunction)` (truthiness), so index 0 is treated as
    'no start'; we put the looping start fn at index 1."""
    # Types: t0 ()->i32 [run], t1 ()->() [start]
    t0 = b"\x60" + _uleb(0) + _uleb(1) + b"\x7f"
    t1 = b"\x60" + _uleb(0) + _uleb(0)
    types = _uleb(2) + t0 + t1
    funcs = _uleb(2) + _uleb(0) + _uleb(1)       # f0:t0 (run), f1:t1 (start)
    # Export "run" = func index 0.
    rn = b"run"
    exports = _uleb(1) + _uleb(len(rn)) + rn + b"\x00" + _uleb(0)
    # Start section (id 8): start func index 1 (non-zero so wasm3 actually runs it).
    start = _uleb(1)
    # Code: f0 = i32.const 0;end   f1 = loop;br 0;end;end
    b0 = _uleb(0) + b"\x41\x00\x0b"
    b1 = _uleb(0) + b"\x03\x40" + b"\x0c" + _uleb(0) + b"\x0b" + b"\x0b"
    code = _uleb(2) + _uleb(len(b0)) + b0 + _uleb(len(b1)) + b1
    # Section order: type(1) func(3) export(7) start(8) code(10)
    out = bytearray(b"\x00asm\x01\x00\x00\x00")
    out += _wasm_section(1, types)
    out += _wasm_section(3, funcs)
    out += _wasm_section(7, exports)
    out += _wasm_section(8, start)
    out += _wasm_section(10, code)
    return bytes(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tests", "fixtures")
    os.makedirs(out_dir, exist_ok=True)

    def write(name, data):
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data)

    write("model.onnx", build_onnx())
    # Write sibling weights.bin for ONNX external-data resolution test.
    # Layout: 8 bytes padding + 8 bytes resolvable data (2 F32: 5.0, 6.0).
    weights_bin = b"\x00" * 8 + struct.pack("<f", 5.0) + struct.pack("<f", 6.0)
    write("weights.bin", weights_bin)
    write("model.safetensors", build_safetensors())
    write("model.gguf", build_gguf())
    build_pytorch(os.path.join(out_dir, "model.pt"))
    build_npz(os.path.join(out_dir, "model.npz"))
    build_torchscript(os.path.join(out_dir, "model_ts.pt"))
    tfl = build_tflite()
    # Self-check: the file_identifier must land at bytes 4..8 and the root
    # uoffset must point inside the buffer (spec §10 priorities).
    assert tfl[4:8] == b"TFL3", "TFLite identifier misplaced"
    root = struct.unpack_from("<I", tfl, 0)[0]
    assert 0 < root < len(tfl), "TFLite root offset out of range"
    write("model.tflite", tfl)

    tfl_cf = build_tflite_ctrlflow()
    assert tfl_cf[4:8] == b"TFL3", "TFLite ctrlflow identifier misplaced"
    root_cf = struct.unpack_from("<I", tfl_cf, 0)[0]
    assert 0 < root_cf < len(tfl_cf), "TFLite ctrlflow root offset out of range"
    write("model_ctrlflow.tflite", tfl_cf)

    # OpenVINO IR: .xml topology + sibling .bin weight blob (issue #39). The
    # .bin MUST be a real sibling so external-data resolution is exercised.
    write("model.xml", build_openvino_xml())
    write("model.bin", build_openvino_bin())
    # OpenVINO quant IR (#85): nf4 Const, NO sibling .bin -> dtype_label + note.
    write("model_quant.xml", build_openvino_quant_xml())
    write("model.mlmodel", build_coreml_mlmodel())
    # CoreML .mlpackage (mlProgram/MIL) DIRECTORY bundle (#85).
    build_mlpackage(out_dir)
    h5 = build_keras_h5()
    assert h5[:8] == b"\x89HDF\r\n\x1a\n", "HDF5 signature misplaced"
    write("model.h5", h5)
    build_keras_v3(out_dir)

    # WASM plugin fixtures (#10): a well-behaved module + a hostile infinite loop.
    wc = build_wasm_const()
    assert wc[:4] == b"\x00asm", "wasm magic misplaced"
    write("plugin_const.wasm", wc)
    write("plugin_loop.wasm", build_wasm_loop())
    write("plugin_start_loop.wasm", build_wasm_start_loop())
    write("plugin_badsig.wasm", build_wasm_badsig())
    write("plugin_bigmem.wasm", build_wasm_bigmem())
    write("plugin_pass.wasm", build_wasm_pass())

    print("wrote fixtures to", out_dir)
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            print("  {:20s} {:6d} bytes".format(name, os.path.getsize(p)))


if __name__ == "__main__":
    main()
