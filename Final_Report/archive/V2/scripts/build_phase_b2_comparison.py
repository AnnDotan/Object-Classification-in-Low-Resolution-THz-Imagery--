"""Phase B2 comparison summary at publication size (PRD V2 §8 FX-03).

V1 sourced the composite from `artifacts/figures/phase_b2_comparison_summary.png`
which renders at thumbnail size with overflowing per-panel subtitles. FX-03
re-renders at figsize=(10, 6), dpi=200, 2x3 grid with sharey='row',
constrained_layout=True, and a single shared legend at the top.

Data sources (read-only):
  runs/final/<tag>/metrics.json    -> best_val_acc per cell

Output:
  Final_Report/figures/phase_b2/comparison_summary.{png,pdf}

Reads four series per (model, dataset) panel: B1 / D-T3 / B2 / B2-nr (L3 only).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "final"
OUT_DIR = ROOT / "Final_Report" / "figures" / "phase_b2"


MODELS = ("resnet50", "densenet121", "transnext_tiny")
DATASETS = ("cifar10", "mnist")
LEVELS = (1, 2, 3, 4, 5)
MODEL_LABEL = {"resnet50": "ResNet50",
               "densenet121": "DenseNet121",
               "transnext_tiny": "TransNeXt-tiny"}
DATASET_LABEL = {"cifar10": "CIFAR-10", "mnist": "MNIST"}

SERIES = (
    ("A (clean)", "final_clean_{m}_{d}",      "*", "#000000", "single"),
    ("B1",        "final_B_L{l}_{m}_{d}",     "o", "#4c72b0", "line"),
    ("D-T3",      "final_D_T3_L{l}_{m}_{d}",  "s", "#55a868", "line"),
    ("B2",        "final_B2_L{l}_{m}_{d}",    "^", "#c44e52", "line"),
    ("B2-nr",     "final_B2nr_L3_{m}_{d}",    "D", "#8172b2", "single-L3"),
)


def _val_acc(tag: str) -> Optional[float]:
    p = RUNS / tag / "metrics.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text()).get("best_val_acc", float("nan")))


def _draw_panel(ax, m: str, d: str) -> None:
    for label, tag_fmt, marker, color, kind in SERIES:
        if kind == "single":
            v = _val_acc(tag_fmt.format(m=m, d=d))
            if v is not None:
                ax.scatter([0], [v * 100], color=color, marker=marker,
                           s=80, zorder=5)
        elif kind == "single-L3":
            v = _val_acc(tag_fmt.format(m=m, d=d))
            if v is not None:
                ax.scatter([3], [v * 100], color=color, marker=marker,
                           s=60, zorder=4)
        else:
            ys = [_val_acc(tag_fmt.format(l=l, m=m, d=d)) for l in LEVELS]
            valid = [(x, y * 100) for x, y in zip(LEVELS, ys) if y is not None]
            if valid:
                vx, vy = zip(*valid)
                ax.plot(vx, vy, color=color, marker=marker,
                        linewidth=1.5, markersize=6, zorder=3)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(["A", "L1", "L2", "L3", "L4", "L5"], fontsize=8)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"{MODEL_LABEL[m]} × {DATASET_LABEL[d]}", fontsize=9, pad=4)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(DATASETS),
        ncols=len(MODELS),
        figsize=(10.0, 6.0),
        dpi=200,
        sharey="row",
        constrained_layout=True,
    )
    for di, d in enumerate(DATASETS):
        for mi, m in enumerate(MODELS):
            ax = axes[di][mi]
            _draw_panel(ax, m, d)
            if mi == 0:
                ax.set_ylabel(f"{DATASET_LABEL[d]}\nval acc (%)", fontsize=9)
            if di == len(DATASETS) - 1:
                ax.set_xlabel("Degradation level", fontsize=9)

    # Single shared legend at top (FX-03 spec).
    legend_handles = []
    for label, _, marker, color, kind in SERIES:
        if kind == "line":
            legend_handles.append(plt.Line2D(
                [0], [0], color=color, marker=marker, linestyle="-",
                linewidth=1.5, markersize=6, label=label,
            ))
        else:
            legend_handles.append(plt.Line2D(
                [0], [0], color=color, marker=marker, linestyle="",
                markersize=8, label=label,
            ))
    fig.legend(handles=legend_handles, loc="upper center",
               ncols=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.04))

    out_png = OUT_DIR / "comparison_summary.png"
    out_pdf = OUT_DIR / "comparison_summary.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[build_phase_b2_comparison] wrote {out_png}")
    print(f"[build_phase_b2_comparison] wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
