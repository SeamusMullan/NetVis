#!/usr/bin/env python3
"""Validate a NetVis declarative plugin manifest before installing it.

Stdlib-only. Mirrors the host loader's up-front checks (JSON shape, api_version,
known category, override-required-to-shadow-builtin, basic DSL token sanity) so an
author catches mistakes without launching NetVis. The HOST is still the final
authority — it validates fully at load and reports errors in the Plugins panel.

Usage: tools/validate_plugin.py <path/to/plugin.json> [more.json ...]
Exit 0 if all files pass, 1 otherwise.
"""
import json
import re
import sys

API_VERSION = 1

CATEGORIES = {
    "Conv", "MatMul", "Activation", "Norm", "Pool", "Elementwise", "Shape",
    "Reduce", "Tensor", "ControlFlow", "IO", "Attention", "Recurrent",
    "Quantize", "Other",
}

# Op names the built-ins already know (categorize_op != Other). A plugin op whose
# NORMALIZED name is one of these shadows a built-in and needs "override": true.
# This is a representative subset — the host has the authoritative table; we flag
# the common ones so authors aren't surprised.
BUILTIN_OPS = {
    "conv", "convtranspose", "qlinearconv", "convinteger",
    "matmul", "gemm", "einsum", "linear", "qlinearmatmul", "matmulinteger", "qgemm",
    "relu", "gelu", "sigmoid", "tanh", "softmax", "elu", "leakyrelu", "prelu",
    "batchnormalization", "layernormalization", "groupnormalization", "rmsnorm",
    "maxpool", "averagepool", "globalaveragepool",
    "add", "sub", "mul", "div", "pow", "sqrt", "exp", "log",
    "reshape", "transpose", "concat", "slice", "gather", "squeeze", "unsqueeze",
    "reducesum", "reducemean", "argmax", "argmin",
    "attention", "multiheadattention", "lstm", "gru", "rnn",
    "quantizelinear", "dequantizelinear", "dynamicquantizelinear",
}

# Token allowlist for a crude DSL sanity pass (identifiers only; operators/brackets
# are stripped first). Anything else is a likely typo.
DSL_KEYWORDS = {"O", "nin", "nout", "unknown", "attr", "sattr", "min", "max", "prod",
                "shape", "rank"}
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
OPERAND_RE = re.compile(r"^(in|out)\d+$")


def norm(op: str) -> str:
    tail = op.rsplit(".", 1)[-1]
    return tail.lower()


def check_expr(expr: str, var_names, where, errs):
    # Strip string literals (attr/sattr names + comparison literals) before token scan.
    stripped = re.sub(r'"[^"]*"', " ", expr)
    for m in IDENT_RE.finditer(stripped):
        tok = m.group(0)
        if tok in DSL_KEYWORDS or tok in var_names or OPERAND_RE.match(tok):
            continue
        errs.append(f"{where}: unknown identifier '{tok}' "
                    f"(not a var/operand/keyword)")


def validate(path: str) -> bool:
    errs = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"FAIL {path}: {e}")
        return False

    # nlohmann parses JSONC (comments); Python json does not. Strip // and /* */ so
    # the validator accepts the same files NetVis does.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(^|[^:])//[^\n]*", lambda m: m.group(1), text)
    try:
        j = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"FAIL {path}: JSON parse error: {e}")
        return False

    if not isinstance(j, dict):
        errs.append("root is not an object")
    if j.get("api_version") != API_VERSION:
        errs.append(f"api_version must be {API_VERSION} (got {j.get('api_version')!r})")
    ops = j.get("ops")
    if not isinstance(ops, list):
        errs.append("'ops' must be an array")
        ops = []

    for i, op in enumerate(ops):
        w = f"ops[{i}]"
        if not isinstance(op, dict):
            errs.append(f"{w}: not an object")
            continue
        name = op.get("name")
        if not isinstance(name, str) or not name:
            errs.append(f"{w}: missing string 'name'")
            name = ""
        cat = op.get("category")
        if cat not in CATEGORIES:
            errs.append(f"{w} '{name}': category {cat!r} is not a known category")
        color = op.get("color")
        if color is not None and not re.match(r"^#[0-9A-Fa-f]{6}$", str(color)):
            errs.append(f"{w} '{name}': color {color!r} must be '#RRGGBB'")
        if name and norm(name) in BUILTIN_OPS and not op.get("override"):
            errs.append(f"{w} '{name}': shadows a built-in op — set \"override\": true")
        var_names = set()
        vars_ = op.get("vars", {})
        if vars_ and not isinstance(vars_, dict):
            errs.append(f"{w}: 'vars' must be an object")
            vars_ = {}
        var_names = set(vars_.keys())
        for vn, ve in vars_.items():
            if not isinstance(ve, str):
                errs.append(f"{w} var '{vn}': expression must be a string")
            else:
                check_expr(ve, var_names, f"{w} var '{vn}'", errs)
        flops = op.get("flops")
        if flops is not None:
            if not isinstance(flops, str):
                errs.append(f"{w}: 'flops' must be a string")
            elif flops:
                check_expr(flops, var_names, f"{w} flops", errs)
        shape = op.get("shape", {})
        if shape and not isinstance(shape, dict):
            errs.append(f"{w}: 'shape' must be an object")
            shape = {}
        for k, dims in shape.items():
            if not re.match(r"^out\d+$", k):
                errs.append(f"{w} shape key '{k}': must be 'outN'")
            if not isinstance(dims, list):
                errs.append(f"{w} shape '{k}': must be an array of dim expressions")
                continue
            for d in dims:
                if not isinstance(d, str):
                    errs.append(f"{w} shape '{k}': dim must be a string expression")
                else:
                    check_expr(d, var_names, f"{w} shape '{k}' dim", errs)

    if errs:
        print(f"FAIL {path}")
        for e in errs:
            print(f"  - {e}")
        return False
    print(f"OK   {path}  ({len(ops)} op(s))")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: validate_plugin.py <plugin.json> [...]", file=sys.stderr)
        return 2
    ok = True
    for p in sys.argv[1:]:
        ok = validate(p) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
