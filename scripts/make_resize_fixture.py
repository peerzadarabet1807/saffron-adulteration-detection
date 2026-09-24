"""Generate web/tests/fixtures/inter_area.json: cv2.INTER_AREA outputs the JS port must reproduce.

Inputs are synthetic and defined by a formula that the unit test recomputes, so only the (small) outputs are
stored. Covers non-integer factors (the real 3024 -> 300 case is ~10.08x), integer 2x / 3x (OpenCV's fast
path) and non-square inputs.

Usage: python scripts/make_resize_fixture.py
"""
from __future__ import annotations

import base64
import json

import cv2
import numpy as np

from saffron_common import WEB_DIR

CASES = [((97, 61), (13, 9)), ((64, 48), (32, 24)), ((90, 60), (30, 20)), ((403, 302), (30, 30)),
         ((1511, 1511), (150, 150)), ((3024, 3024), (30, 30)), ((4032, 3024), (40, 30))]


def synthetic(w: int, h: int) -> np.ndarray:
    """Must match `synthetic()` in web/tests/unit/resize.test.ts."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    c = np.arange(3, dtype=np.int64)
    v = (x[..., None] * 7 + y[..., None] * 13 + c * 29 + ((x * y) % 17)[..., None] * 5) % 256
    return v.astype(np.uint8)


def main() -> None:
    out = {"generator": "scripts/make_resize_fixture.py", "opencv": cv2.__version__, "cases": []}
    for (sw, sh), (dw, dh) in CASES:
        dst = cv2.resize(synthetic(sw, sh), (dw, dh), interpolation=cv2.INTER_AREA)
        out["cases"].append({"src": [sw, sh], "dst": [dw, dh],
                             "rgb": base64.b64encode(dst.tobytes()).decode()})
    path = WEB_DIR / "tests" / "fixtures" / "inter_area.json"
    path.write_text(json.dumps(out) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1000:.0f} KB, {len(CASES)} cases)")


if __name__ == "__main__":
    main()
