"""Python reference embeddings for the input check's TF.js parity test (web/tests/parity.gate.node.test.ts).

Fixtures (all committed, none of the owner's raw photos): the 10 shipped samples, the EXIF-rotated fixture and
the 7 original OOD photos. Two steps, each in the venv that has its framework:

1. ``seen`` (TensorFlow venv, scripts/requirements.txt): the training preprocessing to 300 x 300, then the area
   resize to the gate's 224 x 224 (scripts/gate_extract.py). Writes .work/gate/fixtures.npz and .json.
2. ``embed`` (PyTorch venv, scripts/requirements-gate.txt): embed those images with the original model
   (Hugging Face weights) and apply the rule in models/input_check.json. Writes
   web/tests/fixtures/input_check/reference.json with the SHA-256 of each 224 x 224 image, so the Node test can
   check that the TypeScript preprocessing produced exactly the same pixels.

Usage:
    python scripts/make_gate_references.py seen
    python scripts/make_gate_references.py embed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work" / "gate"
OUT = ROOT / "web" / "tests" / "fixtures" / "input_check"
CONFIG = ROOT / "models" / "input_check.json"


def fixtures() -> list[dict]:
    manifest = json.loads((ROOT / "web" / "public" / "samples" / "manifest.json").read_text(encoding="utf-8"))
    rows = [{"id": s["file"], "file": f"web/public/samples/{s['file']}", "kind": "saffron"} for s in manifest["samples"]]
    rows.append({"id": "exif-rotated.jpg", "file": "web/tests/fixtures/exif-rotated.jpg", "kind": "saffron"})
    credits = json.loads((ROOT / "docs" / "ood" / "credits.json").read_text(encoding="utf-8"))["photos"]
    rows += [{"id": "ood_" + p["file"], "file": f"docs/ood/{p['file']}", "kind": "ood"} for p in credits if p.get("original")]
    return rows


def seen_step() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from gate_extract import gate_input
    from saffron_common import load_image_rgb, to_model_input

    rows, images = [], []
    for r in fixtures():
        rgb, _, err = load_image_rgb(ROOT / r["file"])
        if err:
            raise ValueError(f"{r['file']}: {err}")
        g = gate_input(to_model_input(rgb))
        images.append(g)
        rows.append({**r, "gateSha256": hashlib.sha256(np.ascontiguousarray(g).tobytes()).hexdigest()})
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(WORK / "fixtures.npz", gate=np.stack(images))
    (WORK / "fixtures.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{len(rows)} fixtures -> {WORK / 'fixtures.npz'}")
    return 0


def embed_step() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_input_check import encoder

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = json.loads((WORK / "fixtures.json").read_text(encoding="utf-8"))
    emb = encoder()(np.load(WORK / "fixtures.npz")["gate"]).astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    protos = np.asarray(config["prototypes"]["vectors"], dtype=np.float64)
    scores = (emb @ protos.T).max(axis=1)
    t = config["threshold"]
    images = []
    for r, e, s in zip(rows, emb, scores):
        images.append({**r, "score": float(s), "passed": bool(s >= t), "margin": float(s - t),
                       "embedding": np.round(e, 7).tolist()})
        print(f"{r['id']:<44} {s:.6f} {'pass' if s >= t else 'reject'}")
    doc = {
        "description": "PyTorch (Hugging Face) reference embeddings of the input check's image encoder on the 224 x "
        "224 image (training preprocessing to 300 x 300, then area resize). scripts/make_gate_references.py",
        "model": config["model"]["name"],
        "revision": config["model"]["revision"],
        "version": config["version"],
        "threshold": t,
        "images": images,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reference.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"min |score - threshold| = {min(abs(i['margin']) for i in images):.6f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["seen", "embed"])
    args = ap.parse_args()
    return seen_step() if args.step == "seen" else embed_step()


if __name__ == "__main__":
    sys.exit(main())
