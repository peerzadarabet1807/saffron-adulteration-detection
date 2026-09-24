# Model files

The trained weights are too large for git. They are attached to the
[**v1.0.0 GitHub Release**](https://github.com/peerzadarabet1807/saffron-adulteration-detection/releases/tag/v1.0.0).

| File | Size | What it is | SHA-256 |
|---|---|---|---|
| `best_model.keras` | 95.0 MB | Selected ResNet50V2 model, float32, preprocessing built in (Keras 3.13 / TF 2.20) | `3f5cec9dd588af2a892dc27a2d200552e3d6ee32e7cc4ebc8008631811f1045c` |
| `saved_model.zip` | 175.6 MB | The same model exported as a TensorFlow SavedModel (`serving_default` signature) | `4eeb5d0c80d891ad0edc47aee8be47803c39a88ed82fec16918cfd2de359a523` |

In this folder:

| File | What it is |
|---|---|
| [`class_names.json`](class_names.json) | Output index → class id (`["C1","C2","C3","C4","C5"]`) |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Intended use, data, metrics, limitations, ethical considerations |
| [`tfjs_parity.json`](tfjs_parity.json) | Keras vs TF.js agreement for the float16 (shipped) and uint8 (rejected) browser models |

The browser model (probabilities + class activation maps, float16 weights, 47.2 MB in 4 MiB shards) is
committed in [`web/public/model/`](../web/public/model). It is rebuilt from `best_model.keras` with
[`scripts/build_web_model.py`](../scripts/build_web_model.py) and
[`scripts/convert_to_tfjs.sh`](../scripts/convert_to_tfjs.sh).

## Download and use

```bash
gh release download v1.0.0 -R peerzadarabet1807/saffron-adulteration-detection -D saffron_adulteration_artifacts/models
sha256sum saffron_adulteration_artifacts/models/best_model.keras
```

```python
import json, cv2, numpy as np, keras
from PIL import Image, ImageOps

model = keras.models.load_model("saffron_adulteration_artifacts/models/best_model.keras")
classes = json.load(open("models/class_names.json"))

# exactly the training preprocessing: EXIF transpose, RGB, full-frame squash to 300 x 300 with INTER_AREA
img = np.asarray(ImageOps.exif_transpose(Image.open("thread.jpg")).convert("RGB"))
x = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)[None].astype("float32")  # raw 0-255, no scaling

probs = model.predict(x, verbose=0)[0]
print(dict(zip(classes, probs.round(3))))
```

**Input contract:** tensor `image`, shape `[N, 300, 300, 3]`, `float32` RGB in **[0, 255]**. The model
normalises internally (ResNet50V2 `[-1, 1]` rescaling), so do not normalise the input yourself.
**Output:** softmax probabilities in the order of `class_names.json`.

The scripts in [`scripts/`](../scripts) expect the weights at `saffron_adulteration_artifacts/models/`
(the folder layout of the training notebook's Google Drive export).
