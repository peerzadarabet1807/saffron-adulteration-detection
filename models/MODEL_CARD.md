# Model card: saffron adulteration classifier (ResNet50V2)

| | |
|---|---|
| **Version** | 1.0.0 (September 2026) |
| **Author** | Peerzada Mohammad Rabet (B.Tech, Agricultural and Food Engineering, IIT Kharagpur) |
| **Task** | 5-class image classification of a single saffron thread |
| **Architecture** | ImageNet ResNet50V2 → global average pooling → dropout 0.2 → dense (5) → softmax |
| **Weights** | [`best_model.keras`](README.md) (95 MB, float32); browser: TF.js graph model, float16 weights (47.2 MB) |
| **License** | MIT (code and weights). The training photos are not redistributed. |
| **Status** | Research prototype. **Not** a certified food-safety or laboratory test. |

## Classes

| Id | Group | Description |
|---|---|---|
| C1 | Pure | Saffron stigma dried with a dryer |
| C2 | Pure | Saffron stigma dried by pressing |
| C3 | Counterfeit | Dried saffron stem |
| C4 | Counterfeit | Saffron stem coloured with food dye |
| C5 | Counterfeit | Synthetic / artificial saffron |

## Intended use

- Demonstrating and teaching image-based food-fraud screening, transfer learning and Grad-CAM.
- A first, informal look at a *single saffron thread photographed like the training data*: one thread, centred,
  on plain light paper, in even daylight, with a phone camera.

## Out-of-scope use

- Any certification, legal, regulatory or commercial decision, or accusing a seller or producer.
- Mixed samples, powdered saffron, jars, packaging, several threads, other backgrounds, lighting or cameras.
- Anything that is not saffron: the model has no "not saffron" class and **always** outputs one of the five
  classes (see [non-saffron test](../docs/ood/README.md)).
- Adulterants not in the five classes (e.g. safflower, corn silk, coloured paper strips).

## Training data

- 500 phone photos (median 3024 × 3024 px) taken by the author of saffron collected in Pampore, Kashmir;
  100 per class. One exact duplicate removed (MD5) → 499 images. All Display-P3 JPEGs.
- Stratified split with seed 42: **349 train / 75 validation / 75 test** (15 per class in validation and test).
- Preprocessing: EXIF transpose, RGB, full-frame squash to 300 × 300 with `cv2.INTER_AREA`; the model
  normalises internally.
- Augmentation (train only): random resized crop, flips, 90° rotations, mild affine, brightness/contrast,
  deliberately mild hue (±4) and saturation (±12) shifts, CLAHE, Gaussian blur.

## Training procedure

Three ImageNet backbones (EfficientNetB3, ResNet50V2, DenseNet121) were each tuned (4 of 8 grid configurations,
8 frozen-backbone epochs) and trained in two phases: frozen backbone, then the top 30% unfrozen at a 10× lower
learning rate with BatchNorm kept frozen. AdamW (weight decay 1e-4), cosine schedule with warm-up plus plateau
reduction, label smoothing 0.1, mixed precision on a Colab T4. Selected on validation macro-F1: ResNet50V2
(lr 1e-3, batch 16, dropout 0.2), rebuilt in float32 (100% agreement with the mixed-precision model).

## Evaluation (held-out test set, 75 images, evaluated once)

All numbers are from [`results/metrics.json`](../results/metrics.json).

| Metric | Value |
|---|---|
| Accuracy | 0.9733 (73 / 75) |
| Macro-F1 | 0.9732 |
| Macro ROC-AUC (one-vs-rest) | 0.9993 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| C1 | 1.000 | 1.000 | 1.000 |
| C2 | 1.000 | 1.000 | 1.000 |
| C3 | 1.000 | 1.000 | 1.000 |
| C4 | 1.000 | 0.867 | 0.929 |
| C5 | 0.882 | 1.000 | 0.938 |

Both errors are dyed stems (C4) predicted as artificial saffron (C5); one of them with about 79% confidence
([`results/gradcam/gradcam_index.csv`](../results/gradcam/gradcam_index.csv)).

**Browser model parity** ([`tfjs_parity.json`](tfjs_parity.json)): on the 75 test images, the TF.js model
(float16 weights, WASM backend) agrees with Keras on every top class; the largest probability difference is
0.0021 and the Grad-CAM maps correlate at r ≥ 0.9999. The uint8-quantised variant (23.7 MB) was rejected
(largest probability difference 0.122).

## Limitations

1. **Background confound.** Grad-CAM shows the model often attends to the background paper for C1 and C2. C1
   was photographed on a beige/grey paper and C2 mostly on bluish or vignetted white paper, while C3–C5 share a
   similar white vignetted background. The perfect C1/C2 scores are therefore likely helped by background cues.
2. **Small test set:** 15 images per class; one error changes a class's recall by about 6.7 points.
3. **Optimistic validation scores:** the validation set drove early stopping, the hyperparameter search and
   model selection. Quote the test score.
4. **Narrow acquisition:** one phone, one region, single threads on light paper.
5. **Closed set and no calibration:** always outputs one of five classes; probabilities are not calibrated.
   Of seven red thread/yarn/fabric photos, five triggered the app's low-confidence warning and two were
   classified confidently ([details](../docs/ood/README.md)).

Recommended next steps: re-photograph every class on one standard background with fixed lighting;
background-randomisation augmentation or segment-and-crop of the thread; an external test set collected under
different conditions; a "not saffron" class or OOD detector; temperature scaling; k-fold cross-validation.

## Ethical considerations

- **False "pure":** a buyer could trust adulterated saffron. **False "counterfeit":** an honest seller or
  farmer could be wrongly suspected; Kashmiri saffron growers depend on the crop's reputation. Treat every
  output as a hint to seek a laboratory test (e.g. ISO 3632), never as proof.
- **Privacy:** the web app runs entirely in the browser; photos are not uploaded anywhere.
- **Provenance:** all training photos were taken by the author; no third-party or personal data.

## Citation

See [`CITATION.cff`](../CITATION.cff).
