"""Reproduce the notebook's exact train / validation / test split from the raw dataset.  [needs dataset]

Mirrors notebook section 2: classes = sorted sub-folders, files = sorted rglob with the same extension and
hidden-file filters, MD5 de-duplication in iteration order, the same validity checks, then two stratified
`train_test_split` calls with random_state=42.

Writes build/test_split.json and asserts the published counts (349 / 75 / 75, 15 per class in test).

Usage: python scripts/reproduce_test_split.py [--data data]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from saffron_common import (BUILD_DIR, DATA_DIR, ROOT, SEED, TEST_FRACTION, VAL_FRACTION, class_files,
                            load_image_rgb, md5)

EXPECTED = {"train": 349, "val": 75, "test": 75}


def scan(data_dir: Path):
    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    ordered = [(label, cls, f) for label, cls in enumerate(classes) for f in class_files(data_dir, cls)]

    # Hashing and decoding run in parallel; the de-duplication below replays the notebook's sequential logic.
    with ThreadPoolExecutor() as pool:
        digests = list(pool.map(lambda t: md5(t[2]), ordered))
        checks = list(pool.map(lambda t: load_image_rgb(t[2])[2], ordered))

    records, rejected, seen = [], [], {}
    for (label, cls, f), digest, err in zip(ordered, digests, checks):
        if digest in seen:
            rejected.append({"file": f"{cls}/{f.name}", "reason": f"duplicate of {seen[digest]}"})
            continue
        if err:
            rejected.append({"file": f"{cls}/{f.name}", "reason": err})
            continue
        seen[digest] = f.name
        records.append({"path": f.relative_to(data_dir).as_posix(), "class": cls, "label": label})
    return classes, records, rejected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA_DIR))
    args = ap.parse_args()
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"Dataset not found at {data_dir}; this step needs the raw images.", file=sys.stderr)
        return 1

    classes, records, rejected = scan(data_dir)
    print(f"{len(records)} valid images, {len(rejected)} rejected: {rejected}")

    published = list(csv.DictReader(open(ROOT / "results" / "rejected_images.csv", encoding="utf-8")))
    published = [{"file": "/".join(Path(r["file"]).parts[-2:]), "reason": r["reason"]} for r in published]
    assert rejected == published, f"rejected images differ from results/rejected_images.csv: {published}"

    idx_all = np.arange(len(records))
    labels = np.array([r["label"] for r in records])
    idx_trainval, idx_test = train_test_split(idx_all, test_size=TEST_FRACTION, stratify=labels,
                                              random_state=SEED)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=VAL_FRACTION / (1 - TEST_FRACTION),
                                          stratify=labels[idx_trainval], random_state=SEED)

    split = {"train": idx_train, "val": idx_val, "test": idx_test}
    counts = {s: {c: int((labels[i] == k).sum()) for k, c in enumerate(classes)} for s, i in split.items()}
    for s, i in split.items():
        print(f"{s:5s} {len(i):3d}  {counts[s]}")
        assert len(i) == EXPECTED[s], f"{s} has {len(i)} images, expected {EXPECTED[s]}"
    assert all(v == 15 for v in counts["test"].values()), "test split is not 15 per class"

    published_counts = json.loads((ROOT / "results" / "metrics.json").read_text())["data"]["split_counts"]
    assert counts == published_counts, "split counts differ from results/metrics.json"

    out = {
        "classes": classes,
        "seed": SEED,
        "counts": counts,
        "rejected": rejected,
        # order matters: it is the order of X_test / y_test in the notebook
        **{s: [{"index": int(j), **records[j]} for j in i] for s, i in split.items()},
    }
    BUILD_DIR.mkdir(exist_ok=True)
    (BUILD_DIR / "test_split.json").write_text(json.dumps(out, indent=1))
    print(f"\nSplit reproduced; counts match results/metrics.json. Wrote {BUILD_DIR / 'test_split.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
