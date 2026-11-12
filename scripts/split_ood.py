"""Split the out-of-distribution photos once into ood-tune (about 40%) and ood-test (about 60%).

Stratified by category with a fixed seed. The original 7 photos (measured before the input check existed)
all go to ood-test, so their numbers stay comparable; they count towards each category's test share.
Run once, before any threshold is tuned; the committed docs/ood/split.json is the split from then on.

Usage:
    python scripts/split_ood.py            # refuses to overwrite an existing split
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OOD = ROOT / "docs" / "ood"
SEED = 20261112
TUNE_SHARE = 0.4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = ap.parse_args()
    out = OOD / "split.json"
    if out.exists() and not args.force:
        print(f"{out} exists; the split is fixed. Pass --force only if the photo set itself changed.")
        return 1

    photos = json.loads((OOD / "credits.json").read_text(encoding="utf-8"))["photos"]
    rng = random.Random(SEED)
    tune, test, per_category = [], [], {}
    for cat in sorted({p["category"] for p in photos}):
        members = sorted(p["file"] for p in photos if p["category"] == cat)
        originals = [f for f in members if "/" not in f]
        rest = [f for f in members if "/" in f]
        rng.shuffle(rest)
        n_tune = min(len(rest), round(TUNE_SHARE * len(members)))
        tune += rest[:n_tune]
        test += rest[n_tune:] + originals
        per_category[cat] = {"tune": n_tune, "test": len(members) - n_tune}
    doc = {
        "description": "Fixed split of docs/ood/ into ood-tune and ood-test, written once by scripts/split_ood.py.",
        "seed": SEED,
        "tune_share": TUNE_SHARE,
        "per_category": per_category,
        "tune": sorted(tune),
        "test": sorted(test),
    }
    out.write_text(json.dumps(doc, indent=2) + "\n", "utf-8", newline="\n")
    print(f"tune {len(tune)}, test {len(test)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
