"""Combined Phase B L5 confusion-matrix grid (PRD V2 §4.5 fig 22, FX-06).

FX-06 fix: per-panel colormap scaling makes cross-model comparison
misleading when the 6 panels are embedded independently. This script
re-renders all 6 (model x dataset) confusion matrices in one matplotlib
figure with a single shared colorbar (`vmin=0, vmax=1` row-normalized
recall). Cached val logits at `runs/final/<tag>/logits/val.npz` are the
data source — no model re-load required.

Output:
  Final_Report/figures/confusion/B_L5_grid.png
  Final_Report/figures/confusion/_grid_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "final"
OUT_DIR = ROOT / "Final_Report" / "figures" / "confusion"
OUT_PNG = OUT_DIR / "B_L5_grid.png"
META_JSON = OUT_DIR / "_grid_meta.json"


MODELS = ("resnet50", "densenet121", "transnext_tiny")
MODEL_LABEL = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "transnext_tiny": "TransNeXt-tiny",
}
DATASETS = ("cifar10", "mnist")
DATASET_LABEL = {"cifar10": "CIFAR-10", "mnist": "MNIST"}

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
MNIST_CLASSES = [str(i) for i in range(10)]


def _load_confusion(tag: str) -> tuple[np.ndarray, float]:
    npz = RUN_ROOT / tag / "logits" / "val.npz"
    if not npz.exists():
        raise SystemExit(f"[combine_confusion_grid] missing logits cache: {npz}")
    d = np.load(npz)
    logits, labels = d["logits"], d["labels"]
    preds = logits.argmax(axis=1)
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    diag_acc = float(np.trace(cm) / cm.sum())
    return cm, diag_acc


def _most_confused(cm: np.ndarray, classes: list[str]) -> dict:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    ridx, cidx = np.unravel_index(np.argmax(off), off.shape)
    return {
        "true": classes[int(ridx)],
        "predicted": classes[int(cidx)],
        "count": int(off[ridx, cidx]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(MODELS),
        ncols=len(DATASETS),
        figsize=(11.5, 14.0),
        dpi=150,
        gridspec_kw={"hspace": 0.32, "wspace": 0.22},
    )

    panels: list[dict] = []
    im_handle = None
    for r, model in enumerate(MODELS):
        for c, dataset in enumerate(DATASETS):
            tag = f"final_B_L5_{model}_{dataset}"
            cm, diag_acc = _load_confusion(tag)
            classes = MNIST_CLASSES if dataset == "mnist" else CIFAR_CLASSES
            row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
            cm_norm = cm / row_sums
            ax = axes[r][c]
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
            im_handle = im
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            if r == len(MODELS) - 1:
                ax.set_xlabel("Predicted", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{MODEL_LABEL[model]}\nTrue", fontsize=9)
            if r == 0:
                ax.set_title(DATASET_LABEL[dataset], fontsize=11, pad=8)
            mc = _most_confused(cm, classes)
            sub = f"acc = {diag_acc * 100:.2f}%   most-confused: {mc['true']} -> {mc['predicted']} (n={mc['count']})"
            ax.text(
                0.5, -0.22, sub,
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
            )
            # Annotate counts; flip text color over dark cells.
            for i in range(10):
                for j in range(10):
                    v = int(cm[i, j])
                    if v == 0:
                        continue
                    color = "white" if cm_norm[i, j] > 0.5 else "black"
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=5.5, color=color)
            panels.append({
                "tag": tag,
                "model": model,
                "dataset": dataset,
                "level": 5,
                "diag_acc": diag_acc,
                "most_confused": mc,
                "row": r,
                "col": c,
            })

    fig.suptitle(
        "Phase B confusion matrices at L5 (row-normalized recall; shared 0 to 1 scale)",
        fontsize=12, y=0.995,
    )
    # Single shared colorbar on the right.
    cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
    cb = fig.colorbar(im_handle, cax=cbar_ax)
    cb.set_label("Row-normalized recall", fontsize=9)

    fig.subplots_adjust(left=0.07, right=0.92, top=0.96, bottom=0.05)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(
            {"png": str(OUT_PNG.relative_to(ROOT)).replace("\\", "/"),
             "panels": panels},
            f, indent=2, ensure_ascii=False,
        )
    print(f"[combine_confusion_grid] wrote {OUT_PNG}")
    print(f"[combine_confusion_grid] wrote {META_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
