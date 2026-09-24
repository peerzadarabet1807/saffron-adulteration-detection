"""Rewrite broadcast-shaped biases of fused ops in a TF.js graph model to rank 1.

Grappler folds ResNet50V2's Conv -> BatchNorm -> ReLU blocks into `_FusedConv2D(BiasAdd, Relu)` with a
bias constant of shape [1, 1, 1, C]. TF.js's WebGL/CPU kernels broadcast it, but the WASM kernel only accepts
rank-1 biases ("FusedConv2D only supports rank-1 bias"). [1, 1, 1, C] and [C] have the same memory layout, so
only the declared shape changes; the weight bytes are untouched. The parity test verifies the result.

Usage: python scripts/fix_tfjs_bias.py <tfjs_model_dir>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

FUSED_OPS = {"_FusedConv2D", "_FusedDepthwiseConv2dNative", "FusedDepthwiseConv2dNative", "_FusedMatMul"}


def main(model_dir: str) -> int:
    path = Path(model_dir) / "model.json"
    model = json.loads(path.read_text())
    nodes = model["modelTopology"]["node"]
    by_name = {n["name"]: n for n in nodes}
    consumers = defaultdict(set)
    for n in nodes:
        for inp in n.get("input", []):
            consumers[inp.split(":")[0].lstrip("^")].add(n["op"])

    weights = {w["name"]: w for g in model["weightsManifest"] for w in g["weights"]}
    fixed = 0
    for n in nodes:
        if n["op"] not in FUSED_OPS:
            continue
        for inp in n["input"][2:]:
            name = inp.split(":")[0]
            w = weights.get(name)
            if w is None or len(w["shape"]) <= 1 or any(d != 1 for d in w["shape"][:-1]):
                continue
            if not consumers[name] <= FUSED_OPS:
                continue  # also used elsewhere, keep the broadcast shape
            w["shape"] = [w["shape"][-1]]
            dims = by_name[name].get("attr", {}).get("value", {}).get("tensor", {}).get("tensorShape", {})
            if "dim" in dims:
                dims["dim"] = [dims["dim"][-1]]
            fixed += 1

    path.write_text(json.dumps(model, separators=(",", ":")))
    print(f"{path}: reshaped {fixed} fused-op biases to rank 1")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
