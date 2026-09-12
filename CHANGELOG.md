# Changelog

This file starts at the relicensing; earlier history is in the git tags and GitHub releases.

## Unreleased

- Relicensed from PolyForm Noncommercial 1.0.0 to Apache-2.0. All releases up to and including v0.9.5 remain under PolyForm-NC; this tag onward is Apache-2.0.
- **Fixed: the Windows installer could not modify PATH** (#149). CPack's stock NSIS
  template reads PATH into a 1024-character NSIS string, so on any machine with a
  longer PATH it aborted with "Warning! PATH too long installer unable to modify
  PATH!"; it also gated the edit on `IfFileExists "<dir>\*.*"` and a
  `GetFullPathName /SHORT` dedupe, both unreliable on virtual drives and on volumes
  with 8.3 names disabled, where it skipped the edit silently. The edit now goes
  through `packaging/windows/netvis-path.ps1`, which has neither limit and preserves
  a `REG_EXPAND_SZ` PATH's unexpanded `%VARS%`. A failed edit is non-fatal: the
  install completes and names the folder to add by hand.
