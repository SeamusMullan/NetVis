# Format support matrix

What NetVis actually does with each file format it claims to open, and where the
gaps are. The table below is checkable: it is the input to
`tests/test_format_matrix.cpp`, which opens every fixture named here exactly the
way the app does and asserts the row it was promised.

**What is checked, and what is not.** Only the table between the `BEGIN MATRIX` /
`END MATRIX` markers is machine-checked. The prose under *Known gaps* is
hand-written; a gap there is pinned by a test only where a table row shows it
(called out per format). Treat the prose as a reviewed claim, not a verified one.

**Scope.** This is the support matrix and the list of known gaps in what each
*parser* records. It is not a complete audit of every honest-unknown path: the
cost and shape-inference code in `src/engine/` has not been walked for places that
fall back to `0`, a default dtype or a guessed size. That audit is still open under
#114.

That the table is checked is not ceremony. A support table maintained by hand goes
quietly wrong, and this one did: #107 added a TensorFlow structural sniff ahead of
the `.mlmodel` extension guard in `Detect.cpp`, so every ordinary CoreML file was
routed to the TensorFlow parser and failed to open with *"SavedModel meta_graph
carries no graph_def"* — while this document still said CoreML worked. The table is
now re-derived on every test run.

## How to read the columns

| Column | Meaning |
|---|---|
| **Format** | The label the parser puts on `ir::Model::format_name` — what the title bar and the report show. |
| **Fixture** | The shipped file the row is measured from, opened through `resolve_model_path` → `MappedFile` → `parse_model`, exactly as the app opens it. |
| **Graph** | `yes` if the parse yields a compute graph (`has_graph`). `no` means **tensor-table mode**: the file carries weights but no topology, so NetVis shows the tensor table and the module tree, not a node graph. That is a property of the format, not a shortfall. |
| **Tensors** | Whether any tensors were recorded at all (graph initializers plus flat tensors). |
| **Shapes** | How many recorded tensors came back with a non-empty shape: `all`, `some`, or `none`. The IR has no separate "shape known" flag, so a genuine rank-0 scalar reads as unshaped; the column errs toward claiming less. |
| **Weight offsets** | How many recorded tensors are *addressable* — a byte range that lies inside the mapped file, or an external-data path the weight inspector resolves on demand. Anything short of `all` means some tensor's bytes cannot be located, and the inspector says so rather than showing zeros. |

**Every column describes the tensors a parser recorded, not the tensors the file
holds.** A format that records nothing for some of its weights still reads `all`
for the ones it did. Two such gaps exist today and are named under their formats
below: a TensorFlow SavedModel's `variables/` checkpoint, and an ONNX
`sparse_initializer`.

Two rules apply to every row:

- **Zero payload.** A structural open records `offset + len` and does not decode
  weights. The test asserts `ByteReader::payload_read_counter()` is 0 after every
  parse. That counter is only bumped by the payload decoders (the weight inspector
  and the WASM host's file reader), not by plain `ByteReader` reads, and it is
  thread-local; so what is verified is that an open never routes through the
  decoders. It does not prove that no parser ever touches a weight page — that
  holds because parsers record offsets by construction.
- **Honest unknown shapes.** An unresolved dimension is `-1`, and nothing below
  `-1` is ever written. A `0` is not a stand-in for unknown: it would claim an empty
  tensor and silently zero every FLOP and byte count downstream. A dimension a file
  states as `0` stays `0`. The test asserts no recorded dimension anywhere in the
  table is below `-1`, and pins the ONNX case directly (a symbolic or absent
  dimension becomes `-1`, a stated `0` stays `0`).

<!-- BEGIN MATRIX (checked by tests/test_format_matrix.cpp — regenerate with NETVIS_EMIT_FORMAT_MATRIX=1) -->
| Format | Fixture | Graph | Tensors | Shapes | Weight offsets |
|---|---|---|---|---|---|
| ONNX | `tests/fixtures/model.onnx` | yes | yes | all | all |
| ONNX | `tests/fixtures/model_typed_data.onnx` | yes | yes | all | some |
| TFLite | `tests/fixtures/model.tflite` | yes | yes | all | all |
| TFLite | `tests/fixtures/model_ctrlflow.tflite` | yes | no | none | none |
| SafeTensors | `tests/fixtures/model.safetensors` | no | yes | all | all |
| GGUF | `tests/fixtures/model.gguf` | no | yes | all | all |
| GGUF | `tests/fixtures/model_quant.gguf` | no | yes | all | all |
| PyTorch | `tests/fixtures/model.pt` | no | yes | all | all |
| PyTorch | `tests/fixtures/model_ts.pt` | no | yes | all | all |
| OpenVINO | `tests/fixtures/model.xml` | yes | yes | all | all |
| OpenVINO | `tests/fixtures/model_quant.xml` | yes | yes | all | all |
| NumPy npz | `tests/fixtures/model.npz` | no | yes | all | all |
| NumPy npz | `tests/fixtures/model_compressed.npz` | no | yes | none | none |
| Keras | `tests/fixtures/model.h5` | no | yes | all | all |
| Keras | `tests/fixtures/model.keras` | no | yes | all | all |
| CoreML | `tests/fixtures/model.mlmodel` | yes | yes | none | all |
| CoreML | `tests/fixtures/model.mlpackage/Data/com.apple.CoreML/model.mlmodel` | yes | yes | all | all |
| CoreML | `tests/fixtures/model.mlpackage` | yes | yes | all | all |
| TensorFlow | `tests/fixtures/model_frozen.pb` | yes | yes | all | all |
| TensorFlow | `tests/fixtures/saved_model` | yes | yes | all | all |
| Caffe | `tests/fixtures/model_caffe.prototxt` | yes | yes | all | all |
| Caffe | `tests/fixtures/model_caffe.caffemodel` | yes | yes | all | all |
| Caffe | `tests/fixtures/model_caffe_alone.prototxt` | yes | no | none | none |
| Caffe | `tests/fixtures/model_caffe_v1_deploy.prototxt` | yes | yes | all | all |
| Caffe | `tests/fixtures/model_caffe_v1.caffemodel` | yes | yes | all | all |
<!-- END MATRIX -->

## Known gaps, by format

Each of these is a place where NetVis knows less than the format can express. They
are listed because the honest-unknown rule makes them *quiet*: a missing shape
renders as `—`, not as a wrong number, so nothing in the UI draws attention to
them. This is the list.

**ONNX** — external data (`weights.bin` beside the model) is recorded as an
external path and resolved by the weight inspector on demand, which is why the
matrix reports those tensors as addressable without any payload read at open. Three
things the parser does not record, none of which the UI flags:

- *Typed-field initializers.* A `TensorProto` whose payload sits in `float_data`,
  `int32_data`, `int64_data`, `double_data` or `uint64_data` (exporters write small
  shape constants this way) is listed with its dims and dtype but has no mmap byte
  range: `file_offset = UINT64_MAX`, `byte_len = 0`. `model_typed_data.onnx` shows
  it as `Weight offsets = some`.
- *`sparse_initializer`* (`GraphProto` field 15) is skipped, so those weights never
  appear in the tensor table at all.
- *Model-local `functions`* (`ModelProto` field 25) are skipped, so a call to a
  local function is an opaque op with no drill-down, unlike a TensorFlow library
  function (below).

FP4 / FP8 element types (`FLOAT4E2M1`, `FLOAT8E8M0`, `FLOAT8E4M3FN`, …) are carried
as the exact name in `dtype_label` with `DType::Unknown`: the tensor lists under
its real type name, but the weight inspector shows no histogram or statistics for it
(`stats_unavailable_reason`), because there is no decoder.

**TFLite** — subgraphs and control flow are parsed, including branch bodies. A
subgraph with no operators is still a real subgraph and is kept, which is why
`model_ctrlflow.tflite` reports no tensors: the fixture's branches are empty.

**SafeTensors** — the format carries no topology, so tensor-table mode is the
complete answer, not a shortfall. A sharded checkpoint's `.index.json` is not
followed: each shard opens as its own file. `F4`, `F8_E8M0`, `F8_E4M3`, `F8_E5M2`
and the other element types `ir::DType` cannot express are carried as the exact
dtype string in `dtype_label` with `DType::Unknown`: shape and offset are known, the
name reads correctly, but the inspector shows no histogram for them.

**GGUF** — tensor-table mode, as above. Quantized block types have no `ir::DType`
that describes them exactly, so the element type is carried as a label
(`dtype_label`) alongside a best-fit `DType`, and `dtype_size` returns 0 for them
rather than a plausible-looking guess. Anything downstream that needs an element
size therefore reports unknown instead of a wrong byte count. `MXFP4` and `NVFP4`
(ggml types 39 / 40) follow the same rule: the coarse `Q4` bucket plus the exact
type id, decodable only in the inspector's one-block preview.

**PyTorch** — `.pt`/`.pth`/`.bin` open as a tensor table of the state dict, via
the restricted non-executing `PickleVM`. `float4_e2m1fn_x2` and `float8_*` tensors
are carried as a label with `DType::Unknown`, as above (no histogram). **A
TorchScript archive does not yield a compute graph.** `scan_torchscript_code()` is a
bounded text scan over `code/*.py` that harvests an op *inventory* into
`torchscript.ops` / `torchscript.methods` metadata; `has_graph` stays `false`. That
is why `model_ts.pt` sits in the table with `Graph = no`. Issues #108 / #136 / #137
cover the real parse.

**OpenVINO** — full graph from the `.xml`, weights recorded into the sibling
`.bin` through the same external-data plumbing ONNX uses. Sub-byte element types
(`i4`, `u4`, `nf4`, `u1`, `f8*`) are carried as labels for the same reason GGUF's
quantized types are.

**NumPy npz** — tensor-table mode. An entry written by `np.savez_compressed` is a
DEFLATE stream, which has no linear mmap address, and the `.npy` header that holds
its dtype and shape is inside that same stream. NetVis does not inflate it
(that would be a payload read), so the entry is listed by name only: unknown dtype,
no shape, `file_offset = UINT64_MAX`. `savez_compressed` compresses every entry, so
such an archive reads `Shapes = none` and `Weight offsets = none`, as
`model_compressed.npz` shows.

**Keras** — tensor-table mode only. In a `.keras` v3 archive, `config.json` is
detected and noted in metadata (`keras = v3 archive`) but **not parsed**: no
topology and no class name are extracted, and a legacy `.h5`'s `model_config` is not
read either. An archive whose inner `model.weights.h5` is DEFLATE-compressed rather
than stored is not linearly addressable; the parse leaves a metadata note instead of
inventing offsets.

**CoreML** — a `neuralNetwork` (and the classifier/regressor variants) becomes a
real graph. Its initializers carry dtype, offset and length but **no shape**:
CoreML's `WeightParams` genuinely does not encode one — the dimensions live in the
layer's typed parameters — so the shape is left empty rather than reconstructed.
That is the `Shapes = none` row for `model.mlmodel`. An `mlProgram` (MIL) also
becomes a real graph, whether it arrives as an `.mlpackage` bundle (resolved through
its `Manifest.json` to the inner spec) or as a bare `.mlmodel`, and MIL carries
shapes, so those rows read `Shapes = all`. Model types with no graph here
(pipelines, tree/GLM/SVM) fall back to `has_graph = false` plus a metadata note
rather than an error, and so does an `mlProgram` whose MIL breaks a structural
limit (for example nesting past the recursion cap, which
`tests/test_mlpackage.cpp` pins with a hostile-input fixture that is deliberately
not a row here).

CoreML is recognised by content, not just by suffix: its `Model` carries the model
type at a field number of 200 or more, which no TensorFlow or ONNX protobuf uses.
The `.mlmodel` extension only decides a spec that carries no model type at all.

**TensorFlow** — frozen GraphDef and SavedModel both yield graphs, and
`GraphDef.library.function[]` bodies become their own graphs so a TF2 SavedModel's
`PartitionedCall` stubs can be drilled into. **A SavedModel's checkpoint weights
are not decoded.** They live under `variables/` in a compressed sstable; NetVis
records `variables: checkpoint present (payloads not decoded)` in metadata rather
than inventing offsets for tensors it cannot locate. Note what that does to the
table: the `saved_model` row reads `all` / `all` for the one constant it recorded,
while none of the trained weights are recorded at all. Issue #135 covers decoding
them, along with `.pbtxt` and multi-`meta_graph` files.

**Caffe** — `.prototxt` and `.caffemodel` both yield graphs, in the current `layer`
schema and the legacy V1 `layers` schema (`schema` in the metadata says which). A
`.prototxt` takes its weights from a `.caffemodel` beside it — same name, same name
without `_deploy`, or the only one in the folder — matched by layer name. A layer whose
weights do not fit its definition is left unpaired and listed in `unpaired_layers`. The
`.caffemodel` itself records weights as file offsets. Caffe declares only input shapes, so
every other shape is inferred, and **only through layers with an exact ONNX equivalent**:
`Scale`, `Bias`, `Power`, `Threshold`, `Split`, `Crop`, `Reshape`, `Tile`, `Embed`,
`LSTM`/`RNN`, `ArgMax`, `Reduction`, `SPP` and fork layers leave their outputs unknown, and so
does everything after them. A ResNet-style `BatchNorm → Scale` net therefore has no shapes past
its first `Scale`. `InnerProduct` on a rank > 2 input stays unknown too, as does any
V1 `InnerProduct`, whose weights keep their legacy 4-D `[1,1,N,K]` dims. A `.prototxt`
opened without weights has no shapes past its first convolution or inner product,
because the weight blob is what fixes the output channels. Activation element types are
known only when weights are present: Caffe stores float weights as `data` and double
weights as `double_data`, and activations share that type. Train/test files are shown as
the TEST phase (or the net's own `state`), and the metadata lists what was excluded. Not
decoded, and counted in the metadata: V0 (pre-2014) layer parameters, weights written as text
inside a `.prototxt` (shaped but not addressable), unpacked or non-BVLC blob storage, and
fork parameter messages in a `.caffemodel` (a `.prototxt` shows every field).

## Formats NetVis does not open

Named here so the absence is a documented answer rather than a silent one. Each
has an open issue:

| Format | Issue |
|---|---|
| Darknet (`.cfg` + `.weights`) | #110, #139 |
| torch.export (`.pt2`) / ExecuTorch (`.pte`) | #111, #140, #141 |
| TorchScript compute graph (desktop archive and mobile `.ptl`) | #108, #136, #137 |
| MLIR | #132, #144 |

A file none of the built-in formats claims is offered to registered parser
plugins, and only then reported as an unrecognized format — see
[`docs/plugin-abi.md`](plugin-abi.md).

## Keeping this document true

`tests/test_format_matrix.cpp` reads the table above and checks every row. To
regenerate it after an intentional change:

```sh
NETVIS_EMIT_FORMAT_MATRIX=1 ./build/core/netvis_tests \
  -tc="format matrix: every documented row*"
```

That prints the rows the current build produces, in this document's format. Paste
them back between the `BEGIN MATRIX` / `END MATRIX` markers, and update the prose
above to match — a row that changed is a support claim that changed. The prose is
not checked by the test, so keeping it true is on whoever changes the row.
