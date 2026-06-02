"""Combine Phase B reliability-diagram PNGs into 3x2 grid composites (FX-05).

Per PRD V2 §4.5 figures 19 / 20 and FX-05: keep all 12 Phase B reliability
panels in the body (Q7 = A) but compose them into two grid PNGs so they
cost two figure embeds in the page-count budget instead of six per level.

Layout: rows = models (ResNet50, DenseNet121, TransNeXt-tiny);
        cols = datasets (CIFAR-10, MNIST).

Outputs:
  Final_Report/figures/calibration/B_L3_grid.png
  Final_Report/figures/calibration/B_L5_grid.png
  Final_Report/figures/calibration/_grid_meta.json

FX-05's "sharex / sharey / single colorbar" language is from confusion-grid
territory; reliability diagrams are bar charts with no colormap, so the
shared-axis goal is met visually (uniform white-space borders + row / col
titles) rather than via matplotlib's `sharex='all'` machinery.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "artifacts" / "figures" / "calibration"
OUT_DIR = ROOT / "Final_Report" / "figures" / "calibration"
META_JSON = OUT_DIR / "_grid_meta.json"


MODELS = ("resnet50", "densenet121", "transnext_tiny")
MODEL_LABEL = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "transnext_tiny": "TransNeXt-tiny",
}
DATASETS = ("cifar10", "mnist")
DATASET_LABEL = {"cifar10": "CIFAR-10", "mnist": "MNIST"}
LEVELS = (3, 5)


def _panel_path(model: str, dataset: str, level: int) -> Path:
    # Phase B reliability diagrams under artifacts/figures/calibration/.
    return SRC_DIR / f"final_B_L{level}_{model}_{dataset}_L{level}.png"


def _build_grid(level: int, out_png: Path) -> dict:
    fig, axes = plt.subplots(
        nrows=len(MODELS),
        ncols=len(DATASETS),
        figsize=(9.0, 12.0),
        dpi=160,
    )
    panels = []
    for r, model in enumerate(MODELS):
        for c, dataset in enumerate(DATASETS):
            src = _panel_path(model, dataset, level)
            if not src.exists():
                raise SystemExit(f"[combine_calibration_grids] missing panel: {src}")
            ax = axes[r][c]
            ax.imshow(mpimg.imread(src))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(DATASET_LABEL[dataset], fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(MODEL_LABEL[model], fontsize=11, rotation=90, labelpad=6)
            panels.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "level": level,
                    "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                    "row": r,
                    "col": c,
                }
            )
    fig.suptitle(
        f"Phase B reliability diagrams at L{level} "
        f"(test split; perfect-calibration dashed; ECE annotated per panel)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return {
        "level": level,
        "panels": panels,
        "png": str(out_png.relative_to(ROOT)).replace("\\", "/"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}
    for level in LEVELS:
        out_png = OUT_DIR / f"B_L{level}_grid.png"
        meta[f"B_L{level}"] = _build_grid(level, out_png)
        print(f"[combine_calibration_grids] wrote {out_png}")

    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[combine_calibration_grids] wrote {META_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
