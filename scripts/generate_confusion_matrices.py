"""US-045 (b) — Confusion matrices for 18 L5 headline cells at seed=42.

PRD spec: 24 cells across B1 + D-T3 + B2 + B2-nr. B2-nr is L3-only per
US-041, so actual on-disk count is 18 cells (B1 + D-T3 + B2 each x 6
(model, dataset) pairs). Logits are sourced from runs/final/<dir>/
logits/val.npz (or re-inferred via build_val_loader).

Outputs:
  - artifacts/figures/confusion/<tag>_L5.png (18 files)
  - artifacts/figures/confusion/_index.json (run summary)

Idempotency: --skip-existing checks both PNG presence and the cached
logits; otherwise re-generates the PNG.

Usage:
  python scripts/generate_confusion_matrices.py [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.diagnostics_common import (
    get_or_run_val_logits,
    headline_l5_bases,
    load_cell_spec,
)

OUT_DIR = ROOT / "artifacts" / "figures" / "confusion"
INDEX = OUT_DIR / "_index.json"

CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MNIST_CLASSES = [str(i) for i in range(10)]


def _confusion(logits: np.ndarray, labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    pred = logits.argmax(axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels, pred):
        cm[int(t), int(p)] += 1
    return cm


def _render(cm: np.ndarray, classes: list[str], title: str, out_png: Path) -> None:
    # Normalize by row (true class) for color, keep counts as annotation
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums
    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=130)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    # Annotate counts; flip text color over dark cells
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            if v == 0:
                continue
            txt_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=6.5, color=txt_color)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalized recall", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[confusion] device={dev}")
    bases = headline_l5_bases()
    print(f"[confusion] {len(bases)} L5 cells to process")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    t0 = time.time()
    n_skipped = 0
    for base in bases:
        run_dir = ROOT / "runs" / "final" / base
        if not (run_dir / "metrics.json").exists():
            print(f"[skip] {base} — no metrics.json")
            continue
        spec = load_cell_spec(run_dir)
        out_png = OUT_DIR / f"{base}_L5.png"
        if args.skip_existing and out_png.exists():
            print(f"[cached-png] {base}")
            n_skipped += 1
            continue
        logits, labels = get_or_run_val_logits(spec, dev, force=False)
        cm = _confusion(logits, labels, n_classes=10)
        classes = MNIST_CLASSES if spec.dataset == "mnist" else CIFAR_CLASSES
        diag_acc = float(np.trace(cm) / cm.sum())
        title = f"{base} — {spec.model_name} / {spec.dataset} L5  (n={int(cm.sum())}, acc={diag_acc*100:.2f}%)"
        _render(cm, classes, title, out_png)
        # Per-class diagonal recall + most-confused-pair
        recalls = (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).tolist()
        np.fill_diagonal(cm, 0)
        # Most-confused: max off-diagonal
        ridx, cidx = np.unravel_index(np.argmax(cm), cm.shape)
        most_confused = {"true": classes[int(ridx)], "predicted": classes[int(cidx)], "count": int(cm[ridx, cidx])}
        entries.append({
            "tag": base,
            "phase": spec.phase,
            "treatment": spec.treatment,
            "model": spec.model_name,
            "dataset": spec.dataset,
            "level": spec.level,
            "val_acc_diag": diag_acc,
            "per_class_recall": recalls,
            "most_confused": most_confused,
            "png": str(out_png.relative_to(ROOT)).replace("\\", "/"),
        })
        print(f"[done] {base}: acc={diag_acc*100:.2f}% top-confused={most_confused}")

    summary = {
        "n_cells": len(entries) + n_skipped,
        "n_rendered": len(entries),
        "n_cached_skip": n_skipped,
        "elapsed_s": round(time.time() - t0, 1),
    }
    INDEX.write_text(json.dumps({"summary": summary, "entries": entries}, indent=2, sort_keys=True))
    print(f"\n[confusion] wrote {INDEX}; {summary['n_rendered']} new PNGs, {summary['n_cached_skip']} cached; elapsed={summary['elapsed_s']}s")


if __name__ == "__main__":
    main()
