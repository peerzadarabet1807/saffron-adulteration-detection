"""Fix the input check and evaluate it once on the test sets.  [needs dataset; PyTorch venv]

The gate family chosen in the poultry repository's candidate study is reused here, with saffron's own
prototypes and threshold: the TinyCLIP ViT-8M/16 image encoder (Microsoft, MIT) embeds the 224 x 224 image
(scripts/gate_extract.py), and a photo passes when its cosine similarity with the nearest of five class
prototypes is at least the threshold.

* prototypes: the normalised mean embedding of every training photo of each class (349 photos);
* threshold: the highest value that keeps false rejection on the validation split (75 photos) at or below 1%.
  With 75 photos that means rejecting none of them, so the threshold is the lowest validation score;
* everything else is reported on the test split (75 photos) and on ood-test only, with 95% Wilson intervals.
  "grey" OOD categories (safflower, corn silk, loose saffron) are reported on their own.

Inputs: .work/gate/<set>.npz and .json (gate_extract.py), docs/ood/split.json. Embeddings are cached in
.work/gate/emb/. Writes:
    models/input_check.json                    the rule: model, prototypes, threshold (read by the web app)
    results/input_check/evaluation.json, .md   rates with intervals; summary.json for the app
    results/input_check/figures/input_check_roc.png
    docs/ood/input_check.json                  score and decision for every OOD photo

Usage (venv with scripts/requirements-gate.txt):
    python scripts/build_input_check.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".work" / "gate"
OUT = ROOT / "results" / "input_check"
OOD = ROOT / "docs" / "ood"
CONFIG = ROOT / "models" / "input_check.json"
CLASSES: list[str] = json.loads((ROOT / "models" / "class_names.json").read_text())
NAME = "tinyclip-vit-8m-16-yfcc15m"
REPO = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
REVISION = "a2a8c6eaa2549ad66eb7c31b85022bf58273a26c"
VERSION = "saffron-tinyclip-8m16-proto-1"
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
MAX_FRR = 0.01
DECIMALS = 8
MIN_TOP, MIN_MARGIN = 0.60, 0.20  # web/src/lib/prediction.ts: the app's low-confidence rule
SETS = ("train", "val", "test", "samples", "ood")


# ----------------------------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else float("nan"), "ci95": [lo, hi]}


def threshold_for(scores: np.ndarray, max_frr: float = MAX_FRR) -> float:
    """Highest t such that the share of tuning scores below t is at most max_frr (pass: score >= t)."""
    s = np.sort(scores)
    return float(s[int(math.floor(max_frr * len(s)))])


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(in-domain score > OOD score), ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def pct(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']}; {100 * lo:.1f} to {100 * hi:.1f})"


# ---------------------------------------------------------------------------------------------- embeddings


def encoder():
    import torch
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(REPO, revision=REVISION).eval()
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1) * 255.0
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1) * 255.0

    @torch.no_grad()
    def embed(images: np.ndarray, batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i : i + batch].astype(np.float32)).permute(0, 3, 1, 2)
            f = model.get_image_features((x - mean) / std)
            out.append((f / f.norm(dim=-1, keepdim=True)).numpy())
        return np.concatenate(out).astype(np.float32)

    return embed


def load() -> tuple[dict, dict, dict]:
    emb_dir = WORK / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    embed = None
    emb, meta, probs = {}, {}, {}
    for s in SETS:
        data = np.load(WORK / f"{s}.npz")
        meta[s] = json.loads((WORK / f"{s}.json").read_text(encoding="utf-8"))
        probs[s] = data["probs"]
        cache = emb_dir / f"{s}.npy"
        if not cache.exists() or len(np.load(cache)) != len(meta[s]):
            embed = embed or encoder()
            np.save(cache, embed(data["gate"]))
        emb[s] = np.load(cache).astype(np.float64)
        assert len(emb[s]) == len(meta[s]), s
    return emb, meta, probs


# -------------------------------------------------------------------------------------------------- rule


def prototypes(emb: np.ndarray, labels: list[str]) -> np.ndarray:
    protos = np.stack([emb[[lab == c for lab in labels]].mean(axis=0) for c in CLASSES])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    return np.round(protos, DECIMALS)  # stored rounded; every score below uses the stored values


def score(emb: np.ndarray, protos: np.ndarray) -> np.ndarray:
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return (unit @ protos.T).max(axis=1)


def low_confidence(p: np.ndarray) -> bool:
    top2 = np.sort(p)[-2:]
    return bool(top2[1] < MIN_TOP or top2[1] - top2[0] < MIN_MARGIN)


def frr_block(s: np.ndarray, labels: list[str], t: float) -> dict:
    return {
        "overall": rate(int((s < t).sum()), len(s)),
        "by_class": {
            c: rate(int(sum(x < t for x, lab in zip(s, labels) if lab == c)), labels.count(c)) for c in CLASSES
        },
    }


def far_block(s: np.ndarray, rows: list[dict], t: float) -> dict:
    def acc(sel):
        idx = [i for i, r in enumerate(rows) if sel(r)]
        return rate(int(sum(s[i] >= t for i in idx)), len(idx))

    prox = {r["category"]: r["proximity"] for r in rows}
    order = {"near": 0, "grey": 1, "far": 2}
    cats = sorted(prox, key=lambda c: (order[prox[c]], c))
    return {
        "near": acc(lambda r: r["proximity"] == "near"),
        "far": acc(lambda r: r["proximity"] == "far"),
        "near_and_far": acc(lambda r: r["proximity"] in ("near", "far")),
        "grey": acc(lambda r: r["proximity"] == "grey"),
        "by_category": {c: {"proximity": prox[c], **acc(lambda r, c=c: r["category"] == c)} for c in cats},
        "accepted": sorted(r["file"] for r, x in zip(rows, s) if x >= t and r["proximity"] != "grey"),
        "grey_accepted": sorted(r["file"] for r, x in zip(rows, s) if x >= t and r["proximity"] == "grey"),
    }


def figure(roc: dict, at_t: dict, t: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.plot(roc["far"], roc["frr"], label="near and far ood-test photos", color="#1f6f8b")
    ax.plot(roc["far_near"], roc["frr"], label="near ood-test photos", color="#b5651d", linestyle="--")
    ax.scatter([at_t["far_near"]], [at_t["frr"]], color="#b5651d", zorder=3)
    ax.scatter([at_t["far"]], [at_t["frr"]], color="#1f6f8b", zorder=3, label=f"shipped threshold {t:.4f}")
    ax.set_xlabel("false acceptance (OOD photos that pass)")
    ax.set_ylabel("false rejection (test saffron-thread photos that fail)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Input check on the test sets (TinyCLIP 8M/16, prototype score)", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    emb, meta, probs = load()
    labels = {s: [m["label"] for m in meta[s]] for s in SETS}
    protos = prototypes(emb["train"], labels["train"])
    scores = {s: score(emb[s], protos) for s in SETS if s != "train"}
    t = threshold_for(scores["val"])
    tuning = rate(int((scores["val"] < t).sum()), len(scores["val"]))

    config = {
        "description": (
            "Input check: is this a single saffron thread on paper, like the training photos? Pass when the cosine "
            "similarity between the unit image embedding and the nearest class prototype is at least the threshold. "
            "It says nothing about whether the thread is pure or counterfeit. A safeguard, not a guarantee; measured "
            "rates are in results/input_check/evaluation.json."
        ),
        "version": VERSION,
        "model": {
            "name": NAME,
            "repo": REPO,
            "revision": REVISION,
            "license": "MIT, Copyright (c) Microsoft Corporation (model card; microsoft/Cream TinyCLIP/LICENSE)",
            "input": "the 300 x 300 image the classifier sees, area-resized to 224 x 224, RGB float32 0..255, NHWC",
            "output": "unit-length image embedding",
            "embedding_dim": int(protos.shape[1]),
            "browser_weights": "web/public/gate/ (TF.js graph model, float16 weights)",
        },
        "score": "max over classes of cosine(image embedding, class prototype)",
        "threshold": t,
        "rule": f"pass if score >= threshold; threshold = highest value with false rejection <= {MAX_FRR:.0%} on the "
        "validation split (75 photos, so: the lowest validation score)",
        "tuning": {"set": "validation split", "false_rejection": tuning},
        "prototypes": {
            "source": f"normalised mean embedding of all training photos per class, rounded to {DECIMALS} decimals",
            "classes": CLASSES,
            "photos_per_class": {c: labels["train"].count(c) for c in CLASSES},
            "vectors": protos.tolist(),
        },
        "source": "scripts/build_input_check.py",
    }
    CONFIG.write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8", newline="\n")

    credits = json.loads((OOD / "credits.json").read_text(encoding="utf-8"))["photos"]
    by_file = {p["file"]: p for p in credits}
    split = json.loads((OOD / "split.json").read_text(encoding="utf-8"))
    part = {f: "tune" for f in split["tune"]} | {f: "test" for f in split["test"]}
    ood_rows = []
    for m, s, p in zip(meta["ood"], scores["ood"], probs["ood"]):
        c = by_file[m["id"]]
        order = np.argsort(p)[::-1]
        ood_rows.append(
            {
                "file": m["id"],
                "category": c["category"],
                "proximity": c["proximity"],
                "split": part[m["id"]],
                "original": bool(c.get("original")),
                "score": round(float(s), 6),
                "passed": bool(s >= t),
                "classifier_top_class": CLASSES[int(order[0])],
                "classifier_top_probability": round(float(p[order[0]]), 6),
                "classifier_confident": not low_confidence(p),
            }
        )
    test_idx = [i for i, r in enumerate(ood_rows) if r["split"] == "test"]
    tune_idx = [i for i, r in enumerate(ood_rows) if r["split"] == "tune"]
    ood_test = [ood_rows[i] for i in test_idx]
    s_test = scores["ood"][test_idx]
    in_test = scores["test"]
    nf = np.array([r["proximity"] in ("near", "far") for r in ood_test])
    near = np.array([r["proximity"] == "near" for r in ood_test])

    grid = np.unique(np.quantile(np.concatenate([in_test, s_test]), np.linspace(0, 1, 401)))
    roc = {
        "threshold": grid.tolist(),
        "frr": [float((in_test < g).mean()) for g in grid],
        "far": [float((s_test[nf] >= g).mean()) for g in grid],
        "far_near": [float((s_test[near] >= g).mean()) for g in grid],
    }
    at_t = {
        "frr": float((in_test < t).mean()),
        "far": float((s_test[nf] >= t).mean()),
        "far_near": float((s_test[near] >= t).mean()),
    }

    test_labels = labels["test"]
    wrong = np.array([CLASSES[int(np.argmax(q))] != lab for q, lab in zip(probs["test"], test_labels)])
    rej = in_test < t
    confident = [r["classifier_confident"] for r in ood_test if r["proximity"] in ("near", "far")]
    passed_nf = [r["passed"] for r in ood_test if r["proximity"] in ("near", "far")]
    evaluation = {
        "description": (
            "Evaluation of the shipped input check. The threshold comes from the validation split only; every rate "
            "below except 'tuning' is on the test split (75 photos, 15 per class) and on ood-test. 95% Wilson "
            "intervals. 'grey' categories are look-alikes outside the five classes, reported on their own."
        ),
        "version": VERSION,
        "model": NAME,
        "threshold": t,
        "tuning": {
            "false_rejection_val": frr_block(scores["val"], labels["val"], t),
            "false_acceptance_ood_tune": far_block(scores["ood"][tune_idx], [ood_rows[i] for i in tune_idx], t),
        },
        "false_rejection": {"test": frr_block(in_test, test_labels, t)},
        "false_acceptance": far_block(s_test, ood_test, t),
        "auroc_test_vs_ood_test": auroc(in_test, s_test[nf]),
        "auroc_test_vs_near_ood_test": auroc(in_test, s_test[near]),
        "score_range": {
            s: {"min": float(scores[s].min()), "median": float(np.median(scores[s])), "max": float(scores[s].max())}
            for s in ("val", "test")
        },
        "samples": [
            {"id": m["id"], "label": m["label"], "score": round(float(s), 6), "passed": bool(s >= t)}
            for m, s in zip(meta["samples"], scores["samples"])
        ],
        "posthoc": {
            "note": "Post-hoc observations, not used for any choice.",
            "classifier_errors_test": {
                "classifier_wrong": int(wrong.sum()),
                "rejected_when_wrong": rate(int((rej & wrong).sum()), int(wrong.sum())),
                "rejected_when_right": rate(int((rej & ~wrong).sum()), int((~wrong).sum())),
            },
            "ood_test_confident_answer": {
                "rule": f"the app's rule: top >= {MIN_TOP} and lead >= {MIN_MARGIN} (otherwise 'low confidence')",
                "classifier_alone": rate(sum(confident), len(confident)),
                "after_input_check": rate(sum(c and p for c, p in zip(confident, passed_nf)), len(confident)),
            },
        },
        "roc_test": roc,
        "source": "scripts/build_input_check.py",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(evaluation, indent=1) + "\n", encoding="utf-8", newline="\n")
    (OOD / "input_check.json").write_text(
        json.dumps(
            {
                "description": "Input check score and decision for every OOD photo, with the classifier's answer for "
                "comparison (scripts/build_input_check.py).",
                "version": VERSION,
                "threshold": t,
                "photos": ood_rows,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    figure(roc, at_t, t, OUT / "figures" / "input_check_roc.png")
    fr, fa = evaluation["false_rejection"]["test"], evaluation["false_acceptance"]
    summary = {
        "description": "The numbers the web app shows about the input check (from evaluation.json).",
        "version": VERSION,
        "model": {"name": NAME, "repo": REPO, "license": "MIT"},
        "threshold": t,
        "prototype_photos_per_class": config["prototypes"]["photos_per_class"],
        "false_rejection": {"val": tuning, "test": fr["overall"], "test_by_class": fr["by_class"]},
        "false_acceptance": {k: fa[k] for k in ("near", "far", "near_and_far", "grey")},
        "samples": {"passed": sum(s["passed"] for s in evaluation["samples"]), "n": len(evaluation["samples"])},
        "ood_test_confident_answer": evaluation["posthoc"]["ood_test_confident_answer"],
        "source": "scripts/build_input_check.py",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    write_markdown(evaluation)

    print(f"threshold {t:.6f}  tuning FRR {pct(tuning)}")
    print(f"FRR test {pct(fr['overall'])}  " + ", ".join(f"{c} {pct(r)}" for c, r in fr["by_class"].items()))
    print(f"FAR near {pct(fa['near'])}  far {pct(fa['far'])}  grey {pct(fa['grey'])}")
    print(f"samples failing: {[s['id'] for s in evaluation['samples'] if not s['passed']]}")
    return 0


def write_markdown(ev: dict) -> None:
    fr, fa, ph = ev["false_rejection"]["test"], ev["false_acceptance"], ev["posthoc"]
    val = ev["tuning"]["false_rejection_val"]
    tune = ev["tuning"]["false_acceptance_ood_tune"]
    lines = [
        "# Input check: evaluation of the shipped gate",
        "",
        f"Written by [`scripts/build_input_check.py`](../../scripts/build_input_check.py). Gate `{ev['version']}`: "
        "TinyCLIP ViT-8M/16 image embedding of the 224 x 224 image, cosine to the nearest of five class prototypes "
        f"(mean embeddings of the training photos), threshold {ev['threshold']:.6f}: the highest value with at "
        "most 1% false rejection on the 75 validation photos, which here means the lowest validation score. The "
        "test split and ood-test played no part in any choice. Rates: percent (count; 95% Wilson interval). With "
        "15 test photos per class the intervals are wide.",
        "",
        "## False rejection (saffron-thread photos the check turns away)",
        "",
        "| Class | Validation (tuning) | Test |",
        "|---|---|---|",
    ]
    for c in CLASSES:
        lines.append(f"| {c} | {pct(val['by_class'][c])} | {pct(fr['by_class'][c])} |")
    lines += [
        f"| **All** | {pct(val['overall'])} | {pct(fr['overall'])} |",
        "",
        "## False acceptance (ood-test photos that pass)",
        "",
        "| Category | Near, grey or far | Passed |",
        "|---|---|---|",
    ]
    for c, r in fa["by_category"].items():
        lines.append(f"| {c.replace('_', ' ')} | {r['proximity']} | {pct(r)} |")
    lines += [
        f"| **All near** | near | {pct(fa['near'])} |",
        f"| **All far** | far | {pct(fa['far'])} |",
        f"| **All grey zone** | grey | {pct(fa['grey'])} |",
        "",
        f"On ood-tune (seen while setting up the categories): near {pct(tune['near'])}, far {pct(tune['far'])}, "
        f"grey {pct(tune['grey'])}.",
        "",
        f"Passing near or far ood-test photos: {', '.join(f'`{f}`' for f in fa['accepted']) or 'none'}.",
        f"Passing grey-zone ood-test photos: {', '.join(f'`{f}`' for f in fa['grey_accepted']) or 'none'}.",
        "",
        f"AUROC, test photos against near and far ood-test: {ev['auroc_test_vs_ood_test']:.4f} (near only: "
        f"{ev['auroc_test_vs_near_ood_test']:.4f}).",
        "",
        "![False rejection against false acceptance on the test sets](figures/input_check_roc.png)",
        "",
        "## Shipped sample photos",
        "",
        f"{sum(s['passed'] for s in ev['samples'])} of {len(ev['samples'])} pass.",
        "",
        "## Post-hoc observations",
        "",
        "Not used for any choice.",
        "",
    ]
    ce = ph["classifier_errors_test"]
    lines.append(
        f"The classifier gets {ce['classifier_wrong']} of 75 test photos wrong; the check turns away "
        f"{pct(ce['rejected_when_wrong'])} of those and {pct(ce['rejected_when_right'])} of the photos it gets right."
    )
    v = ph["ood_test_confident_answer"]
    lines += [
        "",
        f"On near and far ood-test photos the classifier alone gives a confident answer (not 'low confidence') for "
        f"{pct(v['classifier_alone'])}; with the input check in front, for {pct(v['after_input_check'])}.",
        "",
    ]
    (OUT / "evaluation.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
