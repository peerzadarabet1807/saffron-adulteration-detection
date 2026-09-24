"""Build the browser model: one graph that returns class probabilities *and* class activation maps.

TF.js graph models cannot compute gradients, so Grad-CAM is folded into the forward pass. The head of the
trained model is GlobalAveragePooling -> Dense ("logits"), so for class c and channel k the Grad-CAM weight is

    alpha_k^c = d y^c / d A^k_ij  averaged over (i, j)  =  w_k^c / (H * W)

and Grad-CAM = ReLU(sum_k alpha_k^c A^k) = ReLU(sum_k w_k^c A^k) / (H * W), i.e. classic CAM up to a constant
that disappears after normalisation. `sum_k w_k^c A^k` is a 1x1 convolution of the feature map with the
logits kernel, which is what the extra "cam" output computes (a 1x1 Conv2D is the same op as a Dense layer
applied at every spatial position, and converts to a plain Conv2D in TF.js).

Outputs of the exported signature (all float32):
    probabilities  [N, 5]           softmax, identical to best_model.keras
    cam            [N, 10, 10, 5]   un-normalised class maps (apply ReLU + min-max on the client)

Usage:
    python scripts/build_web_model.py [--model PATH] [--out build/web_saved_model] [--per-class 6]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from saffron_common import (BEST_MODEL, CLASS_NAMES, DATA_DIR, NUM_CLASSES, WEB_SAVED_MODEL, class_files,
                            load_image_rgb, pearson, relu_minmax, to_model_input)

import keras  # noqa: E402  (after saffron_common sets TF_CPP_MIN_LOG_LEVEL)
import tensorflow as tf  # noqa: E402

PROB_TOL = 1e-5
CAM_MIN_CORR = 0.99


def build_web_model(src: keras.Model) -> keras.Model:
    """Reuse the trained layer objects; only the 1x1 "cam" convolution is new (weights copied)."""
    preprocess = src.get_layer("preprocess")
    backbone = src.get_layer("resnet50v2")
    gap = src.get_layer("gap")
    logits = src.get_layer("logits")

    image = keras.Input((300, 300, 3), dtype="float32", name="image")
    fmap = backbone(preprocess(image), training=False)
    probabilities = keras.layers.Activation("softmax", name="probabilities")(logits(gap(fmap)))

    cam_layer = keras.layers.Conv2D(NUM_CLASSES, 1, use_bias=False, name="cam")
    cam = cam_layer(fmap)
    kernel = logits.get_weights()[0]                       # (2048, NUM_CLASSES)
    cam_layer.set_weights([kernel.reshape(1, 1, *kernel.shape)])

    return keras.Model(image, {"probabilities": probabilities, "cam": cam}, name="saffron_web")


def gradient_grad_cam(src: keras.Model, x: np.ndarray, class_index: int) -> np.ndarray:
    """Gradient-based Grad-CAM at feature-map resolution, as in the notebook (d logit / d feature map)."""
    pre, backbone = src.get_layer("preprocess"), src.get_layer("resnet50v2")
    gap, drop, logits = src.get_layer("gap"), src.get_layer("dropout"), src.get_layer("logits")
    t = tf.convert_to_tensor(x[None].astype("float32"))
    with tf.GradientTape() as tape:
        fmap = backbone(pre(t), training=False)
        tape.watch(fmap)
        score = logits(drop(gap(fmap), training=False))[:, class_index]
    grads = tape.gradient(score, fmap)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    return tf.reduce_sum(fmap[0] * weights, axis=-1).numpy()


def verification_images(per_class: int) -> np.ndarray:
    if DATA_DIR.exists():
        imgs = []
        for cls in CLASS_NAMES:
            for f in class_files(DATA_DIR, cls)[:per_class]:
                arr, _, err = load_image_rgb(f)
                if err is None:
                    imgs.append(to_model_input(arr))
        print(f"Verifying on {len(imgs)} dataset images ({per_class} per class).")
        return np.stack(imgs)
    print("WARNING: data/ not found; verifying on synthetic images only.")
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(8, 300, 300, 3), dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(BEST_MODEL))
    ap.add_argument("--out", default=str(WEB_SAVED_MODEL))
    ap.add_argument("--per-class", type=int, default=6, help="dataset images per class used for checks")
    args = ap.parse_args()

    src = keras.models.load_model(args.model)
    web = build_web_model(src)
    web.summary(line_length=110)

    x = verification_images(args.per_class).astype("float32")
    p_ref = src.predict(x, batch_size=8, verbose=0)
    out = web.predict(x, batch_size=8, verbose=0)
    p_web, cam_web = out["probabilities"], out["cam"]

    max_diff = float(np.abs(p_ref - p_web).max())
    print(f"\n[check] probabilities vs best_model: max |diff| = {max_diff:.2e} (tolerance {PROB_TOL:.0e})")
    assert max_diff < PROB_TOL, "web model probabilities differ from best_model"

    corrs = []
    for i in range(len(x)):
        c = int(p_web[i].argmax())
        corrs.append(pearson(relu_minmax(cam_web[i, :, :, c]), relu_minmax(gradient_grad_cam(src, x[i], c))))
    corrs = np.array(corrs)
    print(f"[check] CAM vs gradient Grad-CAM (ReLU + min-max, 10x10): Pearson min {corrs.min():.6f}, "
          f"mean {corrs.mean():.6f} (threshold {CAM_MIN_CORR})")
    assert corrs.min() > CAM_MIN_CORR, "CAM output does not match gradient Grad-CAM"

    t0 = time.time()
    web.export(args.out, verbose=False)
    print(f"\nExported SavedModel -> {args.out} ({time.time() - t0:.1f}s)")

    loaded = tf.saved_model.load(args.out)
    sig = loaded.signatures["serving_default"]
    print("serving_default inputs :", {k: (v.shape, v.dtype.name) for k, v in sig.structured_input_signature[1].items()})
    print("serving_default outputs:", {k: (v.shape, v.dtype.name) for k, v in sig.structured_outputs.items()})
    served = sig(image=tf.constant(x[:4]))
    served_diff = float(np.abs(served["probabilities"].numpy() - p_ref[:4]).max())
    cam_diff = float(np.abs(served["cam"].numpy() - cam_web[:4]).max())
    print(f"[check] SavedModel vs Keras: max |prob diff| {served_diff:.2e}, max |cam diff| {cam_diff:.2e}")
    assert served_diff < PROB_TOL
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
