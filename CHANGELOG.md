# Changelog

This file starts at the relicensing; earlier history is in the git tags and GitHub releases.

## Unreleased

- The Open dialog and the diff panel's "Add comparison..." picker now list every extension the format detector recognises: `.xml` (OpenVINO), `.npz`, `.keras`, `.h5` / `.hdf5`, `.mlmodel` and `.mlpackage`, `.pkl` / `.pickle`, on top of the previous `.onnx`, `.tflite`, `.safetensors`, `.gguf`, `.pt` / `.pth` / `.bin` and `.pb`. Those formats always parsed; they just could not be picked from the GUI. The first filter is now "All supported models", followed by "All files" on Linux (zenity-family choosers and kdialog) and Windows; the macOS chooser takes a list of types only, so it shows the model types without an "All files" entry. A file with another or no extension (which content detection would still accept) or a WASM-plugin format is therefore opened with "All files", or by drag-and-drop or the command line on macOS. Both dialogs build their filter from the detector's own extension table (plus `.mlpackage`), and a test fails if the detector, including its zip and `.mlmodel` tiebreaks, learns an extension that table lacks. On Linux the filter now also matches upper-case extensions such as `Model.ONNX`. A `.mlpackage` directory can be picked only where the system chooser treats packages as files (macOS); elsewhere drag it onto the window or pass it on the command line. A bundle picked on macOS comes back with a trailing slash; it is now normalised, so its tab and the diff panel show its name and it does not appear twice in Recent. Part of #135.
- Relicensed from PolyForm Noncommercial 1.0.0 to Apache-2.0. All releases up to and including v0.9.5 remain under PolyForm-NC; this tag onward is Apache-2.0.
- Native Wayland support on Linux. The GLFW Wayland backend is now built whenever its dev packages are present (`NETVIS_GLFW_WAYLAND=AUTO`, the new default) and is always built for release packages; the X11 backend stays in the same binary as a fallback. `NETVIS_PLATFORM=x11|wayland` forces a backend at runtime. A `netvis.desktop` entry is installed so Wayland compositors show the app icon, and GLFW errors are now reported on stderr instead of the app exiting silently.
- MXFP4 and NVFP4 support. GGUF `MXFP4` (ggml type 39) and `NVFP4` (type 40) tensors are recognised, labelled by name, and decodable in the weight inspector's one-block preview (the preview now shows up to 64 values, the NVFP4 block size). SafeTensors `F4`/`F8_E8M0`/`F8_E4M3`/…, ONNX `FLOAT4E2M1`/`FLOAT8E8M0`/`FLOAT8E4M3FN`/…, and PyTorch `float4_e2m1fn_x2`/`float8_*` tensors (saved via `_rebuild_tensor_v3`, which torch uses for these dtypes) now show their exact element type instead of `?`.
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
