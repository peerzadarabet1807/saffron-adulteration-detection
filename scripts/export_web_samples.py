"""Export small held-out test images for the web app's "Try a sample" gallery.  [needs dataset]

Picks 2 images per class from the reproduced *test* split (never seen in training, validation or model
selection), using a fixed rule instead of hand-picking:
  typical  the image whose true-class probability is the class median
  hardest  the image with the lowest true-class probability (for C4 this is a misclassification)

Images are EXIF-transposed and downscaled to 600 px on the long side (JPEG, Display-P3 ICC profile kept,
like a phone photo) under neutral, shuffled names so thumbnails do not give the label away.

Also writes the e2e fixtures in web/tests/fixtures/: an EXIF-rotated JPEG, a HEIC file and a non-image.

Needs build/reference/test/reference.json (run make_reference_predictions.py first).
Usage: python scripts/export_web_samples.py
"""
from __future__ import annotations

import io
import json
import random
import sys

import numpy as np
from PIL import Image, ImageOps

from saffron_common import CLASS_NAMES, DATA_DIR, REFERENCE_DIR, SEED, WEB_DIR

LONG_SIDE = 600
THUMB_SIDE = 200
MAX_BYTES = 150_000
SAMPLES_DIR = WEB_DIR / "public" / "samples"
FIXTURES = WEB_DIR / "tests" / "fixtures"

DESCRIPTIONS = {
    "C1": "Pure saffron stigma, dried with a dryer",
    "C2": "Pure saffron stigma, dried by pressing",
    "C3": "Dried saffron stem (counterfeit)",
    "C4": "Saffron stem coloured with food dye (counterfeit)",
    "C5": "Synthetic / artificial saffron (counterfeit)",
}


def load_upright(path):
    with Image.open(path) as im:
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im).convert("RGB")
    return im, icc


def save_jpeg(im: Image.Image, path, icc, exif=None) -> int:
    for q in (88, 84, 80, 76, 72):
        buf = io.BytesIO()
        kw = {"quality": q, "optimize": True, "progressive": True}
        if icc:
            kw["icc_profile"] = icc
        if exif is not None:
            kw["exif"] = exif
        im.save(buf, "JPEG", **kw)
        if buf.tell() <= MAX_BYTES:
            path.write_bytes(buf.getvalue())
            return buf.tell()
    raise RuntimeError(f"could not fit {path.name} under {MAX_BYTES} bytes")


def downscale(im: Image.Image, side: int) -> Image.Image:
    im = im.copy()
    im.thumbnail((side, side), Image.LANCZOS, reducing_gap=3.0)
    return im


def main() -> int:
    ref_file = REFERENCE_DIR / "test" / "reference.json"
    if not (DATA_DIR.exists() and ref_file.exists()):
        print("Needs data/ and build/reference/test/reference.json (run the previous scripts).", file=sys.stderr)
        return 1
    refs = json.loads(ref_file.read_text())

    picks = []
    for c, cls in enumerate(CLASS_NAMES):
        rows = sorted((r for r in refs if r["trueClass"] == cls), key=lambda r: (r["probabilities"][c], r["id"]))
        hardest, typical = rows[0], rows[len(rows) // 2]
        picks.append((typical, "typical", f"A typical {cls} test image (median {cls} probability)."))
        picks.append((hardest, "hardest", f"The hardest {cls} test image (lowest {cls} probability in the test set)."))

    random.Random(SEED).shuffle(picks)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLES_DIR / "thumbs").mkdir(exist_ok=True)
    for old in list(SAMPLES_DIR.glob("*.jpg")) + list((SAMPLES_DIR / "thumbs").glob("*")):
        old.unlink()

    manifest = []
    for n, (ref, kind, note) in enumerate(picks, start=1):
        name = f"sample-{n:02d}.jpg"
        im, icc = load_upright(DATA_DIR / ref["id"])
        size = save_jpeg(downscale(im, LONG_SIDE), SAMPLES_DIR / name, icc)
        downscale(im, THUMB_SIDE).save(SAMPLES_DIR / "thumbs" / name.replace(".jpg", ".webp"), "WEBP", quality=80)
        manifest.append({
            "file": name,
            "thumb": f"thumbs/{name.replace('.jpg', '.webp')}",
            "trueClass": ref["trueClass"],
            "kind": kind,
            "description": f"{DESCRIPTIONS[ref['trueClass']]}. {note}",
            "source": ref["id"],
        })
        print(f"{name}  {ref['trueClass']} {kind:7s} {size / 1000:5.1f} KB  <- {ref['id']} "
              f"(full-res prediction {ref['predicted']}, {max(ref['probabilities']):.1%})")

    (SAMPLES_DIR / "manifest.json").write_text(json.dumps({
        "note": "Held-out test images (never used for training, validation or model selection), "
                "downscaled to 600 px. Chosen per class by rule: the median and the lowest true-class probability.",
        "samples": manifest,
    }, indent=1) + "\n")

    # ---- e2e fixtures ----
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fixtures = []
    # 1) EXIF orientation 6: pixels stored rotated, the tag says "rotate 90 deg clockwise to display".
    src = manifest[0]
    upright, icc = load_upright(SAMPLES_DIR / src["file"])
    exif = Image.Exif()
    exif[0x0112] = 6
    save_jpeg(upright.transpose(Image.Transpose.ROTATE_90), FIXTURES / "exif-rotated.jpg", icc, exif.tobytes())
    fixtures.append({"file": "exif-rotated.jpg", "trueClass": src["trueClass"], "sameAs": src["file"],
                     "note": "Same image as the sample, stored rotated with EXIF Orientation=6."})
    # 2) HEIC (iPhone default format)
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        src = manifest[1]
        upright, icc = load_upright(SAMPLES_DIR / src["file"])
        upright.save(FIXTURES / "sample.heic", "HEIF", quality=90)
        fixtures.append({"file": "sample.heic", "trueClass": src["trueClass"], "sameAs": src["file"],
                         "pilReadable": False, "note": "The sample re-encoded as HEIC."})
    except ImportError:
        print("pillow-heif not installed: skipping the HEIC fixture")
    # 3) not an image
    (FIXTURES / "not-an-image.txt").write_text("This is a text file, not a photo of saffron.\n")

    (FIXTURES / "fixtures.json").write_text(json.dumps(fixtures, indent=1) + "\n")
    total = sum(p.stat().st_size for p in SAMPLES_DIR.rglob("*") if p.is_file())
    print(f"\nWrote {len(manifest)} samples ({total / 1000:.0f} KB incl. thumbnails) and {len(fixtures)} fixtures.")
    assert all((SAMPLES_DIR / m["file"]).stat().st_size <= MAX_BYTES for m in manifest)
    assert all(max(Image.open(SAMPLES_DIR / m["file"]).size) <= LONG_SIDE for m in manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
