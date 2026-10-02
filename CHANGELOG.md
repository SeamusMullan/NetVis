# Changelog

This file starts at the relicensing; earlier history is in the git tags and GitHub releases.

## Unreleased

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
  budget, far below the run budgets: a module whose start section or ABI export does
  not finish within it is refused, with the budget named in the Plugins panel, instead
  of freezing start-up for as long as the module's run budget lasts. A pass whose ABI
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
