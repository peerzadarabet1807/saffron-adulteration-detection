"""Shared constants and preprocessing, copied verbatim from the training notebook.

Every script that feeds images to the model goes through `load_image_rgb` + `to_model_input`, so the
arrays match what the notebook's evaluation cache (`img_eval`) contained.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "saffron_adulteration_artifacts"
BEST_MODEL = ARTIFACTS_DIR / "models" / "best_model.keras"
BUILD_DIR = ROOT / "build"
WEB_SAVED_MODEL = BUILD_DIR / "web_saved_model"
REFERENCE_DIR = BUILD_DIR / "reference"
WEB_DIR = ROOT / "web"

CLASS_NAMES: list[str] = json.loads((ROOT / "models" / "class_names.json").read_text())
NUM_CLASSES = len(CLASS_NAMES)

# ---- identical to the notebook configuration cell ----
SEED = 42
IMG_SIZE = 300
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MIN_SIDE_PX = 32


def load_image_rgb(path: Path):
    """Return (rgb_uint8_array, original_mode, error_message). Same checks as the notebook."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            mode = im.mode
            im = ImageOps.exif_transpose(im)
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception as e:  # noqa: BLE001
        return None, None, f"unreadable ({type(e).__name__})"
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None, mode, f"bad shape {arr.shape}"
    if min(arr.shape[:2]) < MIN_SIDE_PX:
        return None, mode, f"too small {arr.shape[:2]}"
    if arr.std() < 1.0:
        return None, mode, "blank image"
    return arr, mode, None


def to_model_input(rgb: np.ndarray) -> np.ndarray:
    """Full-frame squash to IMG_SIZE x IMG_SIZE with area interpolation (no crop, no letterbox)."""
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def class_files(data_dir: Path, cls: str) -> list[Path]:
    """Files of one class in the notebook's order.

    The notebook ran on Linux, where `sorted()` on paths compares path components case-sensitively.
    Sorting by the relative POSIX parts reproduces that order on any OS.
    """
    root = data_dir / cls
    files = [p for p in root.rglob("*")
             if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VALID_EXTS]
    return sorted(files, key=lambda p: p.relative_to(root).parts)


def relu_minmax(cam: np.ndarray) -> np.ndarray:
    """ReLU followed by min-max normalisation to [0, 1] (all-zero maps stay zero)."""
    cam = np.maximum(cam.astype(np.float64), 0.0)
    lo, hi = cam.min(), cam.max()
    return ((cam - lo) / (hi - lo)).astype(np.float32) if hi > lo else np.zeros_like(cam, dtype=np.float32)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])
