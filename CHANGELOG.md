# Changelog

This file starts at the relicensing; earlier history is in the git tags and GitHub releases.

## Unreleased

- Relicensed from PolyForm Noncommercial 1.0.0 to Apache-2.0. All releases up to and including v0.9.5 remain under PolyForm-NC; this tag onward is Apache-2.0.
- Native Wayland support on Linux. The GLFW Wayland backend is now built whenever its dev packages are present (`NETVIS_GLFW_WAYLAND=AUTO`, the new default) and is always built for release packages; the X11 backend stays in the same binary as a fallback. `NETVIS_PLATFORM=x11|wayland` forces a backend at runtime. A `netvis.desktop` entry is installed so Wayland compositors show the app icon, and GLFW errors are now reported on stderr instead of the app exiting silently.
- MXFP4 and NVFP4 support. GGUF `MXFP4` (ggml type 39) and `NVFP4` (type 40) tensors are recognised, labelled by name, and decodable in the weight inspector's one-block preview (the preview now shows up to 64 values, the NVFP4 block size). SafeTensors `F4`/`F8_E8M0`/`F8_E4M3`/…, ONNX `FLOAT4E2M1`/`FLOAT8E8M0`/`FLOAT8E4M3FN`/…, and PyTorch `float4_e2m1fn_x2`/`float8_*` tensors (saved via `_rebuild_tensor_v3`, which torch uses for these dtypes) now show their exact element type instead of `?`.
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
