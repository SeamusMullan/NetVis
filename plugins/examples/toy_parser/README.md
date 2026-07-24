# toy_parser — example WASM parser plugin

A minimal NetVis WASM **parser** for a toy `NVTOY1` format. Demonstrates the
parser-facet zero-payload thesis: the plugin sniffs the magic + walks a small TOC
through the bounded, host-marked sniff window and records each weight as an
`(offset, len)` `TensorRef` — it never reads a weight byte.

## Build

```sh
./build.sh        # needs clang with a wasm32 target + wasm-ld (LLVM 8+)
```

Produces `toy_parser.wasm`. Drop it beside a `plugin.json` under the NetVis plugin
directory (shown in the Plugins panel) with:

```json
{ "api_version": 1, "name": "toy-parser", "parser_wasm": "toy_parser.wasm" }
```

WASM plugins are **disabled by default** — enable this one in the Plugins panel
(a one-time confirm dialog), and NetVis will offer it for any file no built-in
format claims.

## The format (`NVTOY1`)

```
magic  "NVTOY1\0\0"                       (8 bytes)
u32    n_tensors                          (LE)
per tensor: u32 name_off, u32 name_len, u32 dtype,
            u64 data_off, u64 byte_len    (24-byte records)
... names + payload region (payload NEVER read by the parser).
```

## Notes

- `netvis_can_parse` returns 1 only when the first 8 bytes match the magic.
- `host_record_tensor` copies zero bytes — it records the range; NetVis's weight
  inspector reads the payload lazily later (the one sanctioned read).
- NetVis CI builds no `.wasm`; its tests use a hand-encoded equivalent
  (`plugin_toyparser.wasm` from `tools/gen_fixtures.py`).
