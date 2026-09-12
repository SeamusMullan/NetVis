# Changelog

This file starts at the relicensing; earlier history is in the git tags and GitHub releases.

## Unreleased

- Relicensed from PolyForm Noncommercial 1.0.0 to Apache-2.0. All releases up to and including v0.9.5 remain under PolyForm-NC; this tag onward is Apache-2.0.
- **Plugin ABI frozen at v1** (#113). `plugins/sdk/abi-v1-surface.txt` inventories
  every macro, enumerator, host import and typedef the SDK header exposes;
  `tests/test_plugin_abi_freeze.cpp` re-derives it from the header on every run, and
  additionally requires the host's own link tables to name exactly that import set.
  `docs/plugin-abi.md` gains a compatibility promise: what v1 guarantees, which
  surface changes keep it, and which force a bump. Version negotiation is now tested
  for all three plugin kinds in both directions - a plugin from the future, and one
  that declares no version at all.
