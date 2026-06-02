"""Build per-axis accuracy vs PSNR / SSIM curves (PRD V2 §4.2, FX-15).

Reads `Final_Report/data/tables/per_cell_with_iq.csv` (built by
`build_per_cell_with_iq.py`) and emits 8 body figures + a caption-metadata
JSON:

    figures/curves/acc_vs_psnr_phase{C,C2}_<dataset>.{png,pdf}
    figures/curves/acc_vs_ssim_phase{C,C2}_<dataset>.{png,pdf}
    figures/curves/_axis_curves_meta.json

Plot spec (PRD V2 §4.2 + FX-15):
  - figsize=(8, 5), dpi=160
  - x: psnr_mean | ssim_mean   y: val_acc (%)
  - Phase C: 5 axes; Phase C2: 3 axes
  - Three model traces overlaid:
      ResNet50 = blue circle ; DenseNet121 = green square ;
      TransNeXt-tiny = red triangle
  - Per-axis line-styles within a model (solid / dashed / dotted / dash-dot
    / long-dash)
  - Shared y-range across CIFAR-10 vs MNIST within (phase, metric)
  - Caption-metadata JSON carries per-axis Spearman ρ + a one-sentence
    ranking note for LaTeX caption injection.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Final_Report" / "data" / "tables" / "per_cell_with_iq.csv"
OUT_DIR = ROOT / "Final_Report" / "figures" / "curves"
META_JSON = OUT_DIR / "_axis_curves_meta.json"


MODEL_STYLE = {
    "resnet50":       {"color": "#1f77b4", "marker": "o", "label": "ResNet50"},
    "densenet121":    {"color": "#2ca02c", "marker": "s", "label": "DenseNet121"},
    "transnext_tiny": {"color": "#d62728", "marker": "^", "label": "TransNeXt-tiny"},
}
MODELS_ORDER = ("resnet50", "densenet121", "transnext_tiny")

AXIS_STYLE = {
    "resolution":  {"linestyle": "-",                 "label": "Resolution"},
    "blur":        {"linestyle": "--",                "label": "Blur"},
    "noise":       {"linestyle": ":",                 "label": "Noise"},
    "saturation":  {"linestyle": "-.",                "label": "Saturation"},
    "salt_pepper": {"linestyle": (0, (5, 1)),         "label": "S&P"},
}
PHASE_C_AXES = ("resolution", "blur", "noise", "saturation", "salt_pepper")
PHASE_C2_AXES = ("resolution", "blur", "salt_pepper")
DATASETS = ("cifar10", "mnist")
METRICS = ("psnr", "ssim")
DATASET_LABEL = {"cifar10": "CIFAR-10", "mnist": "MNIST"}


def _load_rows() -> list[dict]:
    out = []
    with SRC.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] not in ("C", "C2"):
                continue
            if row["psnr_mean"] == "" or row["ssim_mean"] == "":
                continue
            out.append(
                {
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "phase": row["phase"],
                    "level": int(row["level"]),
                    "axis": row["axis"],
                    "val_acc": float(row["val_acc"]) * 100.0,
                    "psnr": float(row["psnr_mean"]),
                    "ssim": float(row["ssim_mean"]),
                }
            )
    return out


def _y_range(rows: list[dict], phase: str) -> tuple[float, float]:
    ys = [r["val_acc"] for r in rows if r["phase"] == phase]
    lo, hi = min(ys), max(ys)
    pad = max(2.0, (hi - lo) * 0.05)
    return (max(0.0, lo - pad), min(100.0, hi + pad))


def _make_plot(
    rows: list[dict],
    phase: str,
    dataset: str,
    metric: str,
    axes_list: tuple[str, ...],
    y_range: tuple[float, float],
    out_png: Path,
    out_pdf: Path,
) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    subset = [r for r in rows if r["phase"] == phase and r["dataset"] == dataset]
    per_axis_rho: dict[str, float] = {}
    for axis in axes_list:
        ax_pts = [r for r in subset if r["axis"] == axis]
        if not ax_pts:
            continue
        # Spearman over all 3 models * 5 levels combined for this axis.
        xs_all = [r[metric] for r in ax_pts]
        ys_all = [r["val_acc"] for r in ax_pts]
        rho_val: float
        if len(xs_all) >= 3 and len(set(xs_all)) > 1:
            rho, _p = spearmanr(xs_all, ys_all)
            rho_val = float(rho)
        else:
            rho_val = float("nan")
        per_axis_rho[axis] = rho_val

        for model in MODELS_ORDER:
            pts = sorted(
                (r for r in ax_pts if r["model"] == model),
                key=lambda r: r["level"],
            )
            if not pts:
                continue
            xs = [r[metric] for r in pts]
            ys = [r["val_acc"] for r in pts]
            ms = MODEL_STYLE[model]
            ax.plot(
                xs,
                ys,
                color=ms["color"],
                marker=ms["marker"],
                linestyle=AXIS_STYLE[axis]["linestyle"],
                linewidth=1.4,
                markersize=6,
                alpha=0.85,
            )
    metric_label = "PSNR (dB)" if metric == "psnr" else "SSIM"
    ax.set_xlabel(metric_label)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(*y_range)
    ax.grid(True, alpha=0.25)

    # Two-part legend: models (color + marker) and axes (linestyle).
    model_handles = [
        plt.Line2D(
            [0], [0],
            color=MODEL_STYLE[m]["color"],
            marker=MODEL_STYLE[m]["marker"],
            linestyle="",
            markersize=7,
            label=MODEL_STYLE[m]["label"],
        )
        for m in MODELS_ORDER
    ]
    axis_handles = [
        plt.Line2D(
            [0], [0],
            color="black",
            linestyle=AXIS_STYLE[a]["linestyle"],
            linewidth=1.4,
            label=AXIS_STYLE[a]["label"],
        )
        for a in axes_list
    ]
    leg1 = ax.legend(handles=model_handles, loc="upper left", fontsize=8, framealpha=0.85, title="Model")
    ax.add_artist(leg1)
    ax.legend(handles=axis_handles, loc="lower right", fontsize=8, framealpha=0.85, title="Axis")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)
    return per_axis_rho


def _ranking_sentence(rho_by_axis: dict[str, float], metric: str) -> str:
    valid = {a: r for a, r in rho_by_axis.items() if r == r}  # filter NaN
    if not valid:
        return ""
    ranked = sorted(valid.items(), key=lambda kv: abs(kv[1]), reverse=True)
    dominant_axis, dominant_rho = ranked[0]
    label = AXIS_STYLE[dominant_axis]["label"]
    metric_word = "PSNR" if metric == "psnr" else "SSIM"
    return (
        f"{label} carries the strongest accuracy-{metric_word} association "
        f"(Spearman rho = {dominant_rho:+.2f})."
    )


def main() -> int:
    rows = _load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta: dict[str, dict] = {}
    for phase, axes_list in (("C", PHASE_C_AXES), ("C2", PHASE_C2_AXES)):
        y_range = _y_range(rows, phase)
        for metric in METRICS:
            for dataset in DATASETS:
                tag = f"phase{phase}_{dataset}_{metric}"
                out_png = OUT_DIR / f"acc_vs_{metric}_phase{phase}_{dataset}.png"
                out_pdf = OUT_DIR / f"acc_vs_{metric}_phase{phase}_{dataset}.pdf"
                rho_by_axis = _make_plot(
                    rows, phase, dataset, metric, axes_list, y_range, out_png, out_pdf,
                )
                meta[tag] = {
                    "phase": phase,
                    "dataset": dataset,
                    "metric": metric,
                    "y_range_pct": list(y_range),
                    "spearman_rho_by_axis": rho_by_axis,
                    "ranking_sentence": _ranking_sentence(rho_by_axis, metric),
                    "png": str(out_png.relative_to(ROOT)).replace("\\", "/"),
                    "pdf": str(out_pdf.relative_to(ROOT)).replace("\\", "/"),
                }

    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[build_psnr_axis_curves] wrote {len(meta)} figure pairs under {OUT_DIR}")
    print(f"[build_psnr_axis_curves] wrote caption metadata -> {META_JSON}")
    if len(meta) != 8:
        raise SystemExit(
            f"[build_psnr_axis_curves] expected 8 figure pairs, got {len(meta)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
