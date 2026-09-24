"""Downscale figures from results/ for the web app's About section (WebP, a few hundred KB).

Usage: python scripts/export_web_figures.py
"""
from __future__ import annotations

from PIL import Image

from saffron_common import ROOT, WEB_DIR

FIGURES = {
    # source (relative to results/)          output name          width
    "gradcam/gradcam_overview.png": ("gradcam_overview.webp", 1100),
    "dataset_samples.png": ("dataset_samples.webp", 1100),
}


def main() -> None:
    out_dir = WEB_DIR / "public" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for src, (name, width) in FIGURES.items():
        with Image.open(ROOT / "results" / src) as im:
            im = im.convert("RGB")
            h = round(im.height * width / im.width)
            im.resize((width, h), Image.LANCZOS, reducing_gap=3.0).save(out_dir / name, "WEBP", quality=82, method=6)
        print(f"{src} -> web/public/figures/{name} ({(out_dir / name).stat().st_size / 1000:.0f} KB, {width}x{h})")


if __name__ == "__main__":
    main()
