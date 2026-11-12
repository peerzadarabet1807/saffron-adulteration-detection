"""Images and classifier outputs for the input check study.  [needs dataset for train / val / test]

For every set, the image the input check sees and the classifier's probabilities:

* the classifier's input, exactly as in training: PIL decode -> EXIF transpose -> RGB ->
  cv2.resize(..., (300, 300), INTER_AREA) (saffron_common.load_image_rgb + to_model_input);
* the input check's input: that 300 x 300 image area-resized again to 224 x 224 (cv2 INTER_AREA), the size of
  the TinyCLIP image encoder. The web app does the same with web/src/lib/resize.ts.

Sets: train / val / test from build/test_split.json (scripts/reproduce_test_split.py), the 10 shipped samples
(web/public/samples) and the OOD photos (docs/ood/credits.json). Writes .work/gate/<set>.npz (gate, probs) and
.work/gate/<set>.json (id, label). .work/ is gitignored: the owner's raw photos never leave this machine.

Usage: python scripts/gate_extract.py [--set test --set ood ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from saffron_common import BEST_MODEL, BUILD_DIR, DATA_DIR, ROOT, load_image_rgb, to_model_input

GATE_SIZE = 224
OUT = ROOT / ".work" / "gate"
SETS = ("train", "val", "test", "samples", "ood")


def gate_input(seen: np.ndarray) -> np.ndarray:
    return cv2.resize(seen, (GATE_SIZE, GATE_SIZE), interpolation=cv2.INTER_AREA)


def items(set_name: str) -> list[dict]:
    if set_name in ("train", "val", "test"):
        split = json.loads((BUILD_DIR / "test_split.json").read_text())
        return [{"id": r["path"], "path": DATA_DIR / r["path"], "label": r["class"]} for r in split[set_name]]
    if set_name == "samples":
        manifest = json.loads((ROOT / "web" / "public" / "samples" / "manifest.json").read_text(encoding="utf-8"))
        return [
            {"id": s["file"], "path": ROOT / "web" / "public" / "samples" / s["file"], "label": s["trueClass"]}
            for s in manifest["samples"]
        ]
    credits = json.loads((ROOT / "docs" / "ood" / "credits.json").read_text(encoding="utf-8"))
    photos = credits if isinstance(credits, list) else credits["photos"]
    return [{"id": p["file"], "path": ROOT / "docs" / "ood" / p["file"], "label": p.get("category", "")}
            for p in photos]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", choices=SETS, help="default: all sets")
    args = ap.parse_args()

    import keras

    model = keras.models.load_model(BEST_MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    for set_name in args.set or SETS:
        rows = items(set_name)
        seen = []
        for r in rows:
            rgb, _, err = load_image_rgb(Path(r["path"]))
            if err:
                raise ValueError(f"{r['id']}: {err}")
            seen.append(to_model_input(rgb))
        seen = np.stack(seen)
        probs = model.predict(seen.astype("float32"), batch_size=16, verbose=0)
        gate = np.stack([gate_input(s) for s in seen])
        np.savez(OUT / f"{set_name}.npz", gate=gate, probs=probs.astype(np.float32))
        meta = [{"id": r["id"], "label": r["label"]} for r in rows]
        (OUT / f"{set_name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"{set_name:<8} {len(rows)} images", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
