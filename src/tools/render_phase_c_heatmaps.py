"""Render Phase C 5x5 (axis x level) heatmaps per (model, dataset).

Reads artifacts/Final_Exp.json and writes 6 PNGs to docs/_autogen/figs/:

    docs/_autogen/figs/phase_c_heatmap_{model}_{dataset}.png

Each PNG is a 5x5 grid colored by val_acc with the numeric value annotated in
every cell. Axes are labeled (rows: AXES, columns: L1..L5). Used by
docs/Final_Report.tex via \\includegraphics (see US-024).

Idempotent: re-running with the same Final_Exp.json produces byte-identical
PNGs (matplotlib seeded via rcParams; no random elements).

Usage:
    python -m src.tools.render_phase_c_heatmaps
    python -m src.tools.render_phase_c_heatmaps --src artifacts/Final_Exp.json --out-dir docs/_autogen/figs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SRC = _REPO_ROOT / "artifacts" / "Final_Exp.json"
_DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "_autogen" / "figs"

AXES = ["resolution", "noise", "blur", "saturation", "salt_pepper"]
MODELS = ["resnet50", "densenet121", "transnext_tiny"]
DATASETS = ["cifar10", "mnist"]
LEVELS = [1, 2, 3, 4, 5]

MODEL_PRETTY = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "transnext_tiny": "TransNeXt-tiny",
}
DATASET_PRETTY = {"cifar10": "CIFAR-10", "mnist": "MNIST"}
AXIS_PRETTY = {
    "resolution": "resolution",
    "noise": "noise (v2)",
    "blur": "blur",
    "saturation": "saturation",
    "salt_pepper": "salt-and-pepper (v2)",
}


def _build_matrix(rows: list[dict], model: str, dataset: str) -> np.ndarray:
    """Build a 5x5 matrix of val_acc indexed by (axis, level)."""
    mat = np.full((len(AXES), len(LEVELS)), np.nan, dtype=np.float64)
    for i, axis in enumerate(AXES):
        for j, level in enumerate(LEVELS):
            tag = f"final_C_L{level}_{axis}_{model}_{dataset}"
            for r in rows:
                if r.get("tag") == tag:
                    v = r.get("val_acc")
                    if isinstance(v, (int, float)) and v >= 0:
                        mat[i, j] = float(v)
                    break
    return mat


def render_one(rows: list[dict], model: str, dataset: str, out_path: Path) -> None:
    """Render a single 5x5 heatmap PNG for (model, dataset)."""
    mat = _build_matrix(rows, model, dataset)
    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=180)
    im = ax.imshow(
        mat,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(LEVELS)), [f"L{l}" for l in LEVELS])
    ax.set_yticks(range(len(AXES)), [AXIS_PRETTY[a] for a in AXES])
    ax.set_xlabel("Severity level")
    ax.set_ylabel("Phase C axis")
    ax.set_title(f"{MODEL_PRETTY[model]} · {DATASET_PRETTY[dataset]} · val\\_acc")
    for i in range(len(AXES)):
        for j in range(len(LEVELS)):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.55 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("val\\_acc")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render Phase C 5x5 heatmaps for the IEEEtran report.")
    p.add_argument("--src", default=str(_DEFAULT_SRC))
    p.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR))
    args = p.parse_args(argv)

    src = Path(args.src)
    out_dir = Path(args.out_dir)

    if not src.exists():
        print(f"ERROR: source JSON not found: {src}", file=sys.stderr)
        return 1

    with src.open(encoding="utf-8") as f:
        rows = json.load(f)["rows"]

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for model in MODELS:
        for dataset in DATASETS:
            out_path = out_dir / f"phase_c_heatmap_{model}_{dataset}.png"
            render_one(rows, model, dataset, out_path)
            print(f"wrote {out_path.relative_to(_REPO_ROOT)}")
            written += 1

    print(f"render_phase_c_heatmaps: {written} PNGs written under {out_dir.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
