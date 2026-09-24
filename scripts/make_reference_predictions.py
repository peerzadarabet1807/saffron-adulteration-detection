"""Reference inputs and outputs for the Keras <-> TF.js parity test.

For every image it saves the exact 300x300 uint8 model input (.npy, preprocessed as in training: PIL decode,
EXIF transpose, RGB, cv2.INTER_AREA full-frame squash), the Keras probabilities from best_model.keras and the
ReLU + min-max normalised CAM of the predicted class from the web SavedModel.

Image sets:
  test     the reproduced held-out test split (needs data/ and build/test_split.json)   [needs dataset]
           -> also re-checks accuracy 0.9733, the confusion matrix and the two C4 -> C5 errors
  samples  the downscaled sample images shipped with the web app (web/public/samples/manifest.json)
           and the e2e fixtures in web/tests/fixtures/ -> web/tests/fixtures/sample-references.json

Outputs: build/reference/<set>/NNN.npy and build/reference/<set>/reference.json

Usage: python scripts/make_reference_predictions.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from saffron_common import (BEST_MODEL, BUILD_DIR, CLASS_NAMES, DATA_DIR, REFERENCE_DIR, ROOT, WEB_DIR,
                            WEB_SAVED_MODEL, load_image_rgb, relu_minmax, to_model_input)

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

FIXTURES = WEB_DIR / "tests" / "fixtures"


def run_set(name: str, items: list[dict], keras_model, web_fn) -> list[dict]:
    """items: [{"file": Path, "id": str, "trueClass": str | None}] -> reference rows (also written to disk)."""
    out_dir = REFERENCE_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.npy"):
        old.unlink()

    x = []
    for it in items:
        arr, _, err = load_image_rgb(it["file"])
        assert err is None, f"{it['file']}: {err}"
        x.append(to_model_input(arr))
    x = np.stack(x)

    probs = keras_model.predict(x.astype("float32"), batch_size=16, verbose=0)
    cams = np.concatenate([web_fn(image=tf.constant(x[i:i + 16].astype("float32")))["cam"].numpy()
                           for i in range(0, len(x), 16)])

    rows = []
    for i, it in enumerate(items):
        npy = f"{i:03d}.npy"
        np.save(out_dir / npy, x[i])
        pred = int(probs[i].argmax())
        rows.append({
            "id": it["id"],
            "npy": npy,
            "file": it["file"].relative_to(ROOT).as_posix() if it["file"].is_relative_to(ROOT) else str(it["file"]),
            "trueClass": it.get("trueClass"),
            "predicted": CLASS_NAMES[pred],
            "probabilities": [round(float(p), 7) for p in probs[i]],
            "cam": [round(float(v), 6) for v in relu_minmax(cams[i, :, :, pred]).ravel()],
        })
    (out_dir / "reference.json").write_text(json.dumps(rows, indent=1))
    print(f"[{name}] wrote {len(rows)} references to {out_dir}")
    return rows


def check_test_set(rows: list[dict], y_true: np.ndarray) -> None:
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())["test"]
    y_pred = np.array([CLASS_NAMES.index(r["predicted"]) for r in rows])
    proba = np.array([r["probabilities"] for r in rows])
    acc = float((y_pred == y_true).mean())
    cm = np.zeros((len(CLASS_NAMES),) * 2, dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    errors = [(CLASS_NAMES[t], CLASS_NAMES[p], float(proba[i, p]))
              for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    print(f"[test] reproduced accuracy {acc:.4f} ({int((y_pred == y_true).sum())}/{len(y_true)}); "
          f"published {metrics['accuracy']:.4f}")
    print(f"[test] errors (true, predicted, confidence): {errors}")
    print(f"[test] confusion matrix:\n{cm}")
    assert abs(acc - metrics["accuracy"]) < 1e-9, "accuracy differs from results/metrics.json"
    assert cm.tolist() == metrics["confusion_matrix"], "confusion matrix differs from results/metrics.json"

    # The notebook picked Grad-CAM examples deterministically; replaying its selection lets us compare the
    # per-image confidences with results/gradcam/gradcam_index.csv.
    conf = proba.max(1)
    chosen = []
    for c in range(len(CLASS_NAMES)):
        idx_c = np.where(y_true == c)[0]
        correct_c = idx_c[y_pred[idx_c] == c]
        if len(correct_c):
            chosen.append(int(correct_c[np.argmax(conf[correct_c])]))
        lowest = int(idx_c[np.argmin(proba[idx_c, c])])
        if lowest not in chosen:
            chosen.append(lowest)
    for i in np.where(y_pred != y_true)[0]:
        if int(i) not in chosen:
            chosen.append(int(i))
    published = list(csv.DictReader(open(ROOT / "results" / "gradcam" / "gradcam_index.csv", encoding="utf-8")))
    diffs = []
    for k, i in enumerate(chosen[:len(published)]):
        pub = published[k]
        assert (pub["true"], pub["predicted"]) == (CLASS_NAMES[y_true[i]], CLASS_NAMES[y_pred[i]]), (k, pub)
        diffs.append(abs(float(pub["confidence"]) - float(conf[i])))
    print(f"[test] Grad-CAM index: {len(diffs)} images match on (true, predicted); "
          f"max |confidence diff| vs Colab = {max(diffs):.4f}")


def main() -> int:
    if not WEB_SAVED_MODEL.exists():
        print("Run scripts/build_web_model.py first.", file=sys.stderr)
        return 1
    keras_model = keras.models.load_model(BEST_MODEL)
    web_fn = tf.saved_model.load(str(WEB_SAVED_MODEL)).signatures["serving_default"]

    split_file = BUILD_DIR / "test_split.json"
    if DATA_DIR.exists() and split_file.exists():
        split = json.loads(split_file.read_text())
        items = [{"file": DATA_DIR / r["path"], "id": r["path"], "trueClass": r["class"]} for r in split["test"]]
        rows = run_set("test", items, keras_model, web_fn)
        check_test_set(rows, np.array([CLASS_NAMES.index(r["class"]) for r in split["test"]]))
    else:
        print("[test] skipped: needs data/ and build/test_split.json (run reproduce_test_split.py)")

    manifest = WEB_DIR / "public" / "samples" / "manifest.json"
    if manifest.exists():
        samples = json.loads(manifest.read_text())["samples"]
        items = [{"file": manifest.parent / s["file"], "id": s["file"], "trueClass": s["trueClass"]}
                 for s in samples]
        fixture_manifest = FIXTURES / "fixtures.json"
        if fixture_manifest.exists():
            for fx in json.loads(fixture_manifest.read_text()):
                if fx.get("pilReadable", True):
                    items.append({"file": FIXTURES / fx["file"], "id": f"fixtures/{fx['file']}",
                                  "trueClass": fx.get("trueClass")})
        rows = run_set("samples", items, keras_model, web_fn)
        FIXTURES.mkdir(parents=True, exist_ok=True)
        slim = [{k: r[k] for k in ("id", "trueClass", "predicted", "probabilities")} for r in rows]
        (FIXTURES / "sample-references.json").write_text(json.dumps(slim, indent=1) + "\n")
        print(f"[samples] wrote {FIXTURES / 'sample-references.json'}")
    else:
        print("[samples] skipped: web/public/samples/manifest.json not found (run export_web_samples.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
