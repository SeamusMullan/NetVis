# Format support matrix

What NetVis actually does with each file format it claims to open, and where the
gaps are. Written to be checkable: the table below is the input to
`tests/test_format_matrix.cpp`, which opens every fixture named here exactly the
way the app does and asserts the row it was promised.

That is not ceremony. A support table maintained by hand goes quietly wrong, and
this one did: #107 added a TensorFlow structural sniff ahead of the `.mlmodel`
extension guard in `Detect.cpp`, so every ordinary CoreML file was routed to the
TensorFlow parser and failed to open with *"SavedModel meta_graph carries no
graph_def"* — while this document still said CoreML worked. The table is now
re-derived on every test run.

## How to read the columns

| Column | Meaning |
|---|---|
| **Format** | The label the parser puts on `ir::Model::format_name` — what the title bar and the report show. |
| **Fixture** | The shipped file the row is measured from, opened through `resolve_model_path` → `MappedFile` → `parse_model`, exactly as the app opens it. |
| **Graph** | `yes` if the parse yields a compute graph (`has_graph`). `no` means **tensor-table mode**: the file carries weights but no topology, so NetVis shows the tensor table and the module tree, not a node graph. That is a property of the format, not a shortfall. |
| **Tensors** | Whether any tensors were recorded at all (graph initializers plus flat tensors). |
| **Shapes** | How many recorded tensors came back with a shape: `all`, `some`, or `none`. |
| **Weight offsets** | How many recorded tensors are *addressable* — a real mmap offset, or an external-data path the weight inspector resolves on demand. Anything short of `all` means some tensor's bytes cannot be located, and the inspector says so rather than showing zeros. |

Two rules hold for **every** row and are asserted for every row rather than
written in a column:

- **Zero payload.** A structural open records `offset + len` and never reads a
  weight byte: `ByteReader::payload_read_counter()` is 0 after the parse.
- **Honest unknown shapes.** An unresolved dimension is `-1`. Never `0`, which
  would claim an empty tensor and silently zero every FLOP and byte count
  downstream — a fabricated answer wearing a real one's clothes.

<!-- BEGIN MATRIX (checked by tests/test_format_matrix.cpp — regenerate with NETVIS_EMIT_FORMAT_MATRIX=1) -->
| Format | Fixture | Graph | Tensors | Shapes | Weight offsets |
|---|---|---|---|---|---|
| ONNX | `tests/fixtures/model.onnx` | yes | yes | all | all |
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
| Keras | `tests/fixtures/model.h5` | no | yes | all | all |
| Keras | `tests/fixtures/model.keras` | no | yes | all | all |
| CoreML | `tests/fixtures/model.mlmodel` | yes | yes | none | all |
| CoreML | `tests/fixtures/model_mlprogram_deep.mlmodel` | no | no | none | none |
| CoreML | `tests/fixtures/model.mlpackage` | yes | yes | all | all |
| TensorFlow | `tests/fixtures/model_frozen.pb` | yes | yes | all | all |
| TensorFlow | `tests/fixtures/saved_model` | yes | yes | all | all |
<!-- END MATRIX -->

## Known gaps, by format

Each of these is a place where NetVis knows less than the format can express. They
are listed because the honest-unknown rule makes them *quiet*: a missing shape
renders as `—`, not as a wrong number, so nothing in the UI draws attention to
them. This is the list.

**ONNX** — no known structural gap. External data (`weights.bin` beside the model)
is recorded as an external path and resolved by the weight inspector on demand,
which is why the matrix reports those tensors as addressable without any payload
read at open.

**TFLite** — subgraphs and control flow are parsed, including branch bodies. A
subgraph with no operators is still a real subgraph and is kept, which is why
`model_ctrlflow.tflite` reports no tensors: the fixture's branches are empty.

**SafeTensors** — the format carries no topology, so tensor-table mode is the
complete answer, not a shortfall. A sharded checkpoint's `.index.json` is not
followed: each shard opens as its own file.

**GGUF** — tensor-table mode, as above. Quantized block types have no `ir::DType`
that describes them exactly, so the element type is carried as a label
(`dtype_label`) alongside a best-fit `DType`, and `dtype_size` returns 0 for them
rather than a plausible-looking guess. Anything downstream that needs an element
size therefore reports unknown instead of a wrong byte count.

**PyTorch** — `.pt`/`.pth`/`.bin` open as a tensor table of the state dict, via
the restricted non-executing `PickleVM`. **A TorchScript archive does not yield a
compute graph.** `scan_torchscript_code()` is a bounded text scan over `code/*.py`
that harvests an op *inventory* into `torchscript.ops` / `torchscript.methods`
metadata; `has_graph` stays `false`. That is why `model_ts.pt` sits in the table
with `Graph = no`. Issues #108 / #136 / #137 cover the real parse.

**OpenVINO** — full graph from the `.xml`, weights recorded into the sibling
`.bin` through the same external-data plumbing ONNX uses. Sub-byte element types
(`i4`, `u4`, `nf4`, `u1`, `f8*`) are carried as labels for the same reason GGUF's
quantized types are.

**NumPy npz** — tensor-table mode. An entry written by `np.savez_compressed` is a
DEFLATE stream, which has no linear mmap address: NetVis records its shape and
dtype from the header and sets `file_offset = UINT64_MAX`. The tensor is listed
and described but its bytes are not addressable, and the matrix would report that
row's weight offsets as `some`. Decompressing to find out would break the
zero-payload thesis, so it does not.

**Keras** — tensor-table mode only: the `config.json` topology is read for
metadata but is **not** turned into a graph. A `.keras` v3 archive whose inner
`model.weights.h5` is DEFLATE-compressed rather than stored is not linearly
addressable either; the parse leaves a metadata note instead of inventing offsets.

**CoreML** — a `neuralNetwork` (and the classifier/regressor variants) becomes a
real graph. Its initializers carry dtype, offset and length but **no shape**:
CoreML's `WeightParams` genuinely does not encode one — the dimensions live in the
layer's typed parameters — so the shape is left empty rather than reconstructed.
That is the `Shapes = none` row for `model.mlmodel`. Exotic model types
(`mlProgram` in a bare `.mlmodel`, pipelines, tree/GLM/SVM) fall back to
`has_graph = false` plus a metadata note rather than an error. A `.mlpackage`
bundle resolves through its `Manifest.json` to the inner model and does produce
shapes, since MIL carries them.

**TensorFlow** — frozen GraphDef and SavedModel both yield graphs, and
`GraphDef.library.function[]` bodies become their own graphs so a TF2 SavedModel's
`PartitionedCall` stubs can be drilled into. **A SavedModel's checkpoint weights
are not decoded.** They live under `variables/` in a compressed sstable; NetVis
records `variables: checkpoint present (payloads not decoded)` in metadata rather
than inventing offsets for tensors it cannot locate. Issue #135 covers decoding
them, along with `.pbtxt` and multi-`meta_graph` files.

## Formats NetVis does not open

Named here so the absence is a documented answer rather than a silent one. Each
has an open issue:

| Format | Issue |
|---|---|
| Caffe (`.prototxt` + `.caffemodel`) | #109, #138 |
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
NETVIS_EMIT_FORMAT_MATRIX=1 ./build/core-only/netvis_tests \
  -tc="format matrix: every documented row*"
```

That prints the rows the current build produces, in this document's format. Paste
them back between the `BEGIN MATRIX` / `END MATRIX` markers, and update the prose
above to match — a row that changed is a support claim that changed.
