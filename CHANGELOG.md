# Changelog

This file starts at the relicensing; earlier history is in the git tags and GitHub releases.

## Unreleased

- **Changed default: the scroll wheel now pans the graph instead of zooming (#158).** The canvas now follows Netron: scrolling, or a two-finger swipe on a trackpad, pans up/down and left/right, and Shift+scroll pans sideways with a mouse. Ctrl+scroll zooms at the pointer (Cmd+scroll or Control+scroll on macOS), and so does pinching on a macOS trackpad. On macOS, trackpad scrolling now tracks your fingers 1:1. Dragging with the left mouse button anywhere on the canvas pans (middle-drag and Space+drag still work). A node is now selected when you release a click, so starting a drag on a node no longer selects it. New keyboard zoom: Shift+Up / Shift+Down zoom in and out about the centre of the view, Shift+Backspace returns to 100% (Ctrl+= / Ctrl+- / Ctrl+0 also work, Cmd on macOS), and the arrow keys pan. **To get the old scroll-to-zoom back, choose View > Scroll wheel > Zoom** (also in Preferences and the command palette). The choice is remembered, and existing users see a one-time notice after upgrading. Two differences from Netron are deliberate: Shift+scroll pans sideways rather than zooming, and Cmd+scroll zooms on macOS. See `docs/canvas-controls.md`.
- The shortcut reference no longer claims Ctrl means the Control key on macOS: Dear ImGui maps Ctrl shortcuts to Cmd there.
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
