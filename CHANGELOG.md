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
  Negotiation is tested for the manifest (all three loaders), the module (op, parser
  and pass), the link step (op handlers), and the Registry (all three kinds), each next
  to a matching-plugin control, plus end to end through `load_wasm_op_plugin` ->
  `compute_cost`.
- Fixed `op_input_const_ints` for WASM op plugins: the host linked it with two
  parameters while the SDK header declares three, so wasm3 rejected every SDK-built
  plugin's import and the call never worked. Fixed `op_set_output_shape` clamping a
  rank above `NV_MAX_RANK` to its first 8 dims and recording that as a known shape; it
  now drops the output, so the shape stays unknown. The WASM adapters take their
  marshalling and sandbox caps from the SDK header instead of keeping their own
  copies.
- Native Wayland support on Linux. The GLFW Wayland backend is now built whenever its dev packages are present (`NETVIS_GLFW_WAYLAND=AUTO`, the new default) and is always built for release packages; the X11 backend stays in the same binary as a fallback. `NETVIS_PLATFORM=x11|wayland` forces a backend at runtime. A `netvis.desktop` entry is installed so Wayland compositors show the app icon, and GLFW errors are now reported on stderr instead of the app exiting silently.
- MXFP4 and NVFP4 support. GGUF `MXFP4` (ggml type 39) and `NVFP4` (type 40) tensors are recognised, labelled by name, and decodable in the weight inspector's one-block preview (the preview now shows up to 64 values, the NVFP4 block size). SafeTensors `F4`/`F8_E8M0`/`F8_E4M3`/…, ONNX `FLOAT4E2M1`/`FLOAT8E8M0`/`FLOAT8E4M3FN`/…, and PyTorch `float4_e2m1fn_x2`/`float8_*` tensors (saved via `_rebuild_tensor_v3`, which torch uses for these dtypes) now show their exact element type instead of `?`.
