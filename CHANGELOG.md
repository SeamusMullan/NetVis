# Changelog

This file starts at the relicensing; earlier history is in the git tags and GitHub releases.

## Unreleased

- **Changed default: the scroll wheel now pans the graph instead of zooming (#158).** The canvas now follows Netron: scrolling, or a two-finger swipe on a trackpad, pans up/down and left/right, and Shift+scroll pans sideways with a mouse. Ctrl+scroll zooms at the pointer (Cmd+scroll or Control+scroll on macOS). On macOS the app also reads trackpad pinch (zoom) and precise scroll deltas (1:1 finger tracking) through a small native bridge; that bridge has not yet been tried on real trackpad hardware, so treat both as expected rather than confirmed. Dragging with the left mouse button anywhere on the canvas pans (middle-drag and Space+drag still work). A node is now selected when you release a click, so starting a drag on a node no longer selects it. New keyboard zoom: Shift+Up / Shift+Down zoom in and out about the centre of the view, Shift+Backspace returns to 100% (Ctrl+= / Ctrl+- / Ctrl+0 also work, Cmd on macOS), and the arrow keys pan. **To get the old scroll-to-zoom back, choose View > Scroll wheel > Zoom** (also in Preferences and the command palette). The choice is remembered. Anyone who has used NetVis before (a saved preferences file, recent files, a saved session or a cached layout) sees a one-time notice on the first launch after upgrading; a fresh install does not. That first launch writes only `wheel_mode` to `view_prefs.json` (merged into an existing file, never replacing it), so no other setting is pinned to today's default. Two differences from Netron are deliberate: Shift+scroll pans sideways rather than zooming, and Cmd+scroll zooms on macOS. See `docs/canvas-controls.md`.
- The shortcut reference no longer claims Ctrl means the Control key on macOS: Dear ImGui maps Ctrl shortcuts to Cmd there.
- The Open dialog and the diff panel's "Add comparison..." picker now list every extension the format detector recognises: `.xml` (OpenVINO), `.npz`, `.keras`, `.h5` / `.hdf5`, `.mlmodel` and `.mlpackage`, `.pkl` / `.pickle`, on top of the previous `.onnx`, `.tflite`, `.safetensors`, `.gguf`, `.pt` / `.pth` / `.bin` and `.pb`. Those formats always parsed; they just could not be picked from the GUI. The first filter is now "All supported models", followed by "All files" on Linux (zenity-family choosers and kdialog) and Windows; the macOS chooser takes a list of types only, so it shows the model types without an "All files" entry. A file with another or no extension (which content detection would still accept) or a WASM-plugin format is therefore opened with "All files", or by drag-and-drop or the command line on macOS. Both dialogs build their filter from the detector's own extension table (plus `.mlpackage`), and a test fails if the detector, including its zip and `.mlmodel` tiebreaks, learns an extension that table lacks. On Linux the Open filter now also matches upper-case extensions such as `Model.ONNX`; Save dialogs keep their plain, single filter, so KDE's save dialog still appends the extension and asks before overwriting. A `.mlpackage` directory can be picked only where the system chooser treats packages as files (macOS); elsewhere drag it onto the window or pass it on the command line. A bundle picked on macOS comes back with a trailing slash; it is now normalised, so its tab and the diff panel show its name and it does not appear twice in Recent. Part of #135.
- Relicensed from PolyForm Noncommercial 1.0.0 to Apache-2.0. All releases up to and including v0.9.5 remain under PolyForm-NC; this tag onward is Apache-2.0.
- **Plugin ABI frozen at v1** (#113). `plugins/sdk/abi-v1-surface.txt` inventories
  everything a plugin binds to: every macro, enumerator, typedef and wire-struct
  member the SDK header exposes, every host import *with its wasm signature*, and
  every guest export the host calls by name. `tests/test_plugin_abi_freeze.cpp`
  re-derives it from the header and from the host's own link tables and call sites on
  every run, requires each import's facet, name and signature to agree between the
  two, and checks its own extractors against mutated sources. `docs/plugin-abi.md`
  gains a compatibility promise: what v1 guarantees, which surface changes keep it,
  and which force a bump.
- **WASM plugins that the host cannot honour are now refused at load, with the reason.**
  Previously a wrong-ABI op plugin could be registered, and with `override: true`
  turn a built-in answer (say every `MatMul`'s FLOPs) into "unknown" while the
  Plugins panel showed it as loaded. Now: the manifest `api_version` must be present,
  an unsigned integer and exactly the host's version (a missing, string, float or
  `4294967297` value used to slip through; the declarative loader had the same
  narrowing bug); a module must declare the host's ABI through its
  `netvis_<facet>_abi_version` export (a pass module without the export is still ABI
  v1, because that facet shipped before the export existed); and every import a
  module declares must bind, with the unbound one named. WASM adapters now report the
  version their module declared, so the Registry's own version check applies to them.
  That load-time probe runs guest code (the module's start section, then its
  ABI-version export) wherever plugins are loaded, so it has its own 50,000-step
  budget, far below the run budgets, shared by the two (the budget is not refilled
  between them): a module whose start section and ABI export do not finish within it
  together is refused, with the budget named in the Plugins panel, instead of freezing
  start-up for as long as the module's run budget lasts. The budget counts steps, not
  wall-clock time: a module that spends it on bulk-memory instructions can still stall
  the probe for seconds (see `docs/plugin-abi.md`; a fix is tracked in #182). A pass whose ABI
  export is present but broken is refused rather than taken for a pre-negotiation v1
  pass, and a refused pass reports no metrics.
  Negotiation is tested for the manifest (all three loaders), the module (op, parser
  and pass), the link step (op handlers), and the Registry (all three kinds), each next
  to a matching-plugin control, plus end to end through `load_wasm_op_plugin` ->
  `compute_cost`.
- Fixed `op_input_const_ints` for WASM op plugins: the host linked it with two
  parameters while the SDK header declares three, so wasm3 rejected every SDK-built
  plugin's import and the call never worked. Fixed `op_set_output_shape` clamping a
  rank above `NV_MAX_RANK` to its first 8 dims and recording that as a known shape; it
  now records the output with its dtype and no shape, so the shape stays unknown. The
  WASM adapters take their marshalling and sandbox caps from the SDK header instead of
  keeping their own copies.
- Native Wayland support on Linux. The GLFW Wayland backend is now built whenever its dev packages are present (`NETVIS_GLFW_WAYLAND=AUTO`, the new default) and is always built for release packages; the X11 backend stays in the same binary as a fallback. `NETVIS_PLATFORM=x11|wayland` forces a backend at runtime. A `netvis.desktop` entry is installed so Wayland compositors show the app icon, and GLFW errors are now reported on stderr instead of the app exiting silently.
- MXFP4 and NVFP4 support. GGUF `MXFP4` (ggml type 39) and `NVFP4` (type 40) tensors are recognised, labelled by name, and decodable in the weight inspector's one-block preview (the preview now shows up to 64 values, the NVFP4 block size). SafeTensors `F4`/`F8_E8M0`/`F8_E4M3`/…, ONNX `FLOAT4E2M1`/`FLOAT8E8M0`/`FLOAT8E4M3FN`/…, and PyTorch `float4_e2m1fn_x2`/`float8_*` tensors (saved via `_rebuild_tensor_v3`, which torch uses for these dtypes) now show their exact element type instead of `?`.
- PyTorch Mobile `.ptl` (lite-interpreter) archives open as real compute graphs (#137). `bytecode.pkl` is decoded with the existing restricted pickle VM (instructions, operator table, constants, types, register count and per-method schemas) and `forward` is rebuilt by symbolically executing the interpreter's register/stack bytecode: operators become nodes with their exact `aten::`/`prim::` names (overloads as an attribute, constant arguments folded in by position), registers become edges, module parameters reached through `GET_ATTR` become initializers named by module path (`fc.weight`), tensor constants become initializers named `CONSTANTS.c<i>`, and `if`/loops become `prim::If`/`prim::Loop` nodes whose branches and bodies you can dive into. Bytecode v6–v9 is graphed. v3–v5 files (which do not record operator arity), files with an unknown instruction, and files whose layout contradicts the interpreter's emitter show an exact operator inventory and the reason instead of a guessed graph. Weight bytes are never read; activation shapes are reported as unknown because the format carries none. `.ptl` is accepted by the file dialogs and the extension tiebreak, and a flatbuffer-format mobile model (`PTMF`) is reported as unsupported instead of unrecognised.
- TorchScript desktop archives (`torch.jit.save`), and `.ptl` files shown as an inventory, now list their module parameters in the tensor table; graphed `.ptl` files show them as graph initializers. Previously none were found, because the pickled module object's state was discarded; the restricted pickle VM now records object state as inert data (it still executes nothing). Weights kept in the state of packed objects (for example the prepacked conv/linear contexts that `optimize_for_mobile` creates) are located too. Packed values NetVis cannot decode (for example quantized tensors) are counted in the `torchscript.packed_unlocated` metadata, and the parameter totals in the properties and cost panels (and `unlocated_weights` in the JSON report) then say they are a lower bound.
- `--screenshot` mode (#170): `netvis --screenshot out.png [--size WxH] [--view file.netvis-view] [--canvas-only] [--fit] [--theme dark|light] [--timeout s] [--no-layout-cache] model` opens the model through the normal loading pipeline, waits until parsing, layout and shape inference have all finished, renders into a hidden window's offscreen framebuffer at exactly the requested size (independent of HiDPI scale), writes a PNG and exits. Exit code 0 on success; 2 bad arguments, 3 model failed to load, 4 bad view file, 5 timeout, 6 OpenGL/window failure, 7 PNG not written, each with the reason on stderr. Captures ignore saved preferences, recent files, `imgui.ini` and plugins so the same command gives the same picture, and read no weight payloads (only the small shape constants ONNX shape inference reads, as in the GUI). `--canvas-only` drops menus, panels and the status bar. The output name must end in `.png`; the PNG is written through a temporary file created exclusively next to it (an unpredictable name, never opened through a symbolic link), the target is re-checked just before it is replaced, and an existing file is replaced only if it is a PNG. Full-window captures show the model's file name rather than its absolute path in the status bar and leave out the model-tab row (which ImGui draws into an implicit "Debug" window); `--help` prints the usage.
- View-state files (`.netvis-view`, File → Save/Load View State) now also record and restore edge routing, the collapse state, the cost heatmap with its metric and scale, the critical-path overlay, the navigation mode and pins, and the search/attribute/table filters, and accept a model-independent `"collapse": "all"` or `"none"`. Files saved by earlier versions load as before. A view saved for a different model still applies only its model-independent settings; the same-model check now treats `./model.onnx` and `/abs/path/model.onnx` as the same file, and a relative `"model"` is also resolved against the view file's own directory (Save View State writes the model relative to the file when it sits inside the file's directory, so a view file can live next to its model). Load View State now reports what it ignored ("View loaded (2 settings ignored)" plus the reasons) and an apply that was refused is an error rather than "View loaded"; a restored attribute filter is kept for the restored selection instead of being cleared on the next frame; restored filters are cut to the size of their boxes; a save too large to load back is refused with a message (the readable indented form falls back to a compact one first), and anything a save had to leave out is named. Model-specific parts of a load wait for any in-flight layout or shape inference to finish before applying.
- Fixed: an ONNX model whose shape inference finished before its first layout stayed in the "Enriching" state forever (a spinning status bar, and search's type filters never enabled).
- A finished layout of a graph with no nodes now says "This graph has no nodes" instead of "Laying out graph...".
- The JSON nesting pre-scan (`core/JsonNesting.h`, used by plugin manifests and view files) can skip `//` and `/* */` comments for callers that parse with comments ignored, and plugin manifests now do: a quote inside a comment could previously hide a very deeply nested document from the depth check.
- **Fixed: the Windows installer could not modify PATH on a machine with a long PATH** (#149). CPack's stock NSIS
  template reads PATH into a 1024-character NSIS string, so on any machine with a
  longer PATH it aborted with "Warning! PATH too long installer unable to modify
  PATH!". The template also skips the edit silently when it cannot see the install
  directory (a suspected cause of the "virtual drive" report, not a confirmed one).
  The edit now goes through `packaging/windows/netvis-path.ps1`, which has no length
  limit, preserves a `REG_EXPAND_SZ` PATH's unexpanded `%VARS%`, and touches only
  NetVis's own entry (an uninstall with nothing to remove writes nothing). The
  "Do not add / all users / current user" choice is now honoured even when "Do not
  create shortcuts" is ticked. A failed edit is non-fatal: the install completes,
  the template's own PATH edit gets a turn, and the installer names the folder to
  add by hand. Not fixed: a drive-letter install that fails outright on a virtual
  or mapped drive; that needs its own repro (#149 stays open).
- **Fixed: CoreML `.mlmodel` files failed to open** (#114). v0.9.5 added a TensorFlow
  structural sniff ahead of the `.mlmodel` extension guard in format detection. A
  CoreML `Model` protobuf satisfies that sniff exactly — field 1 varint, field 2
  length-delimited, whose own first field is length-delimited — so ordinary
  `.mlmodel` files were handed to the TensorFlow parser and died with "SavedModel
  meta_graph carries no graph_def". The extension guard now runs first, and CoreML is
  also recognised from content (its model type sits at a field number of 200 or
  more, which no TensorFlow or ONNX protobuf uses), so a CoreML file renamed without
  the `.mlmodel` suffix no longer falls through to TensorFlow or ONNX.
- Format detection: length checks in the protobuf sniffs are now overflow-safe. A
  hostile length varint near 2^64 could wrap the bounds check and be accepted.
- Added [`docs/format-support.md`](docs/format-support.md): what every supported
  format yields (graph, shapes, addressable weights) and the known gaps per format.
  `tests/test_format_matrix.cpp` re-derives the table from the shipped fixtures on
  every test run, so a stale row cannot go unnoticed. Only the table is checked, not
  the prose around it, and the cost and shape-inference code has not yet been audited
  for fabricated values (still open under #114).
- CI builds the GUI app and runs `ctest` on Linux (both the `ubuntu-22.04` release image and `ubuntu-latest`), macOS and Windows for every pull request and push to master (#169), so a platform break shows up on the PR instead of when a release tag is built. The test suite was made portable to MSVC with test-only changes; no library or app behaviour changes.
- Added a `.gitattributes` rule (`CHANGELOG.md merge=union`, #169) so the append-only changelog merges without conflicts when two branches each add entries; check the merged text afterwards, since a union merge keeps both sides' lines as they are.
- **Caffe support** (#138, #109). NetVis opens `.prototxt` (text) and `.caffemodel` (binary) Caffe networks, in both the current `layer` schema and the legacy V1 `layers` schema, and says which one it saw in the model metadata. A `.prototxt` picks up weights from a `.caffemodel` beside it: the one with the same name, the same name without a `_deploy` suffix, or the only `.caffemodel` in the folder. Layers are matched by name, and a layer whose weights do not fit its definition is left without weights and listed, rather than shown with the wrong ones. A `.caffemodel` also opens on its own, from the layers stored inside it, and a training snapshot is shown in the phase its layers record (TRAIN) rather than re-filtered as TEST. Weights are located in the file and read only when you inspect one. In-place layers such as `ReLU` with `top == bottom` become a proper chain (`conv1` → `conv1#1`) instead of a loop. Layer types map onto the existing op categories, so colouring and the cost analyzer work. Shapes are inferred only where a layer has an exact equivalent, so Caffe-only layers such as `Scale` and `Split` leave their outputs unknown rather than guessed. Caffe files are recognised by content: a `.prototxt` could otherwise be mistaken for ONNX, and so could a `.caffemodel` that declares `input_shape`; the binary check also refuses TensorFlow GraphDefs that start with their function library. Both extensions are listed in the Open and "Add comparison..." dialogs.
- **Fixed: some pooling and convolution output shapes were wrong.** Shape inference ignored `ceil_mode` on `MaxPool`/`AveragePool` and always rounded down, so an ONNX pool with `ceil_mode=1` over an input the window does not divide evenly got a dimension one short. `ceil_mode` is now honoured. In the one case where runtimes disagree (the last window would start in the right padding) the dimension is reported as unknown. A pooling or convolution window larger than its padded input now gives an unknown shape instead of a 0 or negative dimension. `LRN` and `MeanVarianceNormalization` now carry shapes through, since their output has the same shape as their input.
