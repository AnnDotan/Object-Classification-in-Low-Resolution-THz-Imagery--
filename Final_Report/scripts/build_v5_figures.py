"""Build the clean, focused V5 report figures (B1 + C2 only).

The V5 report gives full graphical treatment to exactly two phases:

  * Phase B1 (combined degradation) -- accuracy vs combined level, 3 model
    traces per dataset, with the Phase A clean baseline as a dotted line.
    (The L1-L5 axis is legitimate here: B1 sweeps a single combined recipe.)

  * Phase C2 (single-axis under the THz protocol) -- two focused families:
      family 1 (degradation-TYPE comparison): accuracy vs PSNR and vs SSIM,
        one line per axis = mean over the 3 models.  NEVER plotted against the
        L1-L5 index (the user constraint: level is a parameter, not a quality
        measure; PSNR / SSIM are the quality measures).
      family 2 (MODEL comparison): one subplot per axis, 3 model traces, vs
        PSNR and vs SSIM.

Outputs (all under Final_Report/figures/curves/, both PNG + PDF):
  b1_acc_vs_level_{cifar10,mnist}
  c2_type_{psnr,ssim}_{cifar10,mnist}
  c2_model_{psnr,ssim}_{cifar10,mnist}
  _v5_meta.json   (per-axis Spearman rho + ranking sentence for captions)

This script is self-contained and leaves the V4 figure scripts
(build_figures.py, build_psnr_axis_curves.py) untouched: it writes new
filenames, so the V4 report still builds.

Run from the repository root:
  python Final_Report/scripts/build_v5_figures.py
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

# Deterministic, embeddable fonts (TrueType-42), serif to match the report.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["svg.hashsalt"] = "v5_figures"

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "Final_Report" / "data" / "tables"
B1_CSV = TABLES / "tab_V_phase_b.csv"
A_CSV = TABLES / "tab_IV_phase_a.csv"
C2_CSV = TABLES / "per_cell_with_iq.csv"
OUT_DIR = ROOT / "Final_Report" / "figures" / "curves"
META_JSON = OUT_DIR / "_v5_meta.json"

MODELS_ORDER = ("resnet50", "densenet121", "transnext_tiny")
MODEL_STYLE = {
    "resnet50":       {"color": "#1f77b4", "marker": "o", "label": "ResNet50"},
    "densenet121":    {"color": "#2ca02c", "marker": "s", "label": "DenseNet121"},
    "transnext_tiny": {"color": "#d62728", "marker": "^", "label": "TransNeXt-tiny"},
}

C2_AXES = ("resolution", "blur", "salt_pepper")
AXIS_STYLE = {
    "resolution":  {"color": "#1f77b4", "marker": "o", "label": "Resolution"},
    "blur":        {"color": "#ff7f0e", "marker": "s", "label": "Blur"},
    "salt_pepper": {"color": "#2ca02c", "marker": "^", "label": "Salt & pepper"},
}

DATASETS = ("cifar10", "mnist")
DATASET_LABEL = {"cifar10": "CIFAR-10", "mnist": "MNIST"}
METRICS = ("psnr", "ssim")
METRIC_LABEL = {"psnr": "PSNR (dB)", "ssim": "SSIM"}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load_b1() -> dict[tuple[str, str], list[tuple[int, float]]]:
    """(model, dataset) -> sorted [(level, acc%)]."""
    out: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with B1_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["dataset"])
            out[key].append((int(row["level"]), float(row["val_acc"]) * 100.0))
    for key in out:
        out[key].sort(key=lambda t: t[0])
    return out


def _load_clean() -> dict[tuple[str, str], float]:
    """(model, dataset) -> clean baseline acc%."""
    out: dict[tuple[str, str], float] = {}
    with A_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["model"], row["dataset"])] = float(row["val_acc"]) * 100.0
    return out


def _load_c2() -> list[dict]:
    out: list[dict] = []
    with C2_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] != "C2":
                continue
            if row["psnr_mean"] == "" or row["ssim_mean"] == "":
                continue
            out.append(
                {
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "level": int(row["level"]),
                    "axis": row["axis"],
                    "val_acc": float(row["val_acc"]) * 100.0,
                    "psnr": float(row["psnr_mean"]),
                    "ssim": float(row["ssim_mean"]),
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Phase B1 -- accuracy vs combined level
# --------------------------------------------------------------------------- #
def _plot_b1(b1, clean, dataset: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    for model in MODELS_ORDER:
        pts = b1[(model, dataset)]
        xs = [lv for lv, _ in pts]
        ys = [acc for _, acc in pts]
        ms = MODEL_STYLE[model]
        ax.plot(xs, ys, color=ms["color"], marker=ms["marker"], linestyle="-",
                linewidth=1.8, markersize=6, label=ms["label"])
        base = clean.get((model, dataset))
        if base is not None:
            ax.axhline(base, color=ms["color"], linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Combined degradation level")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["L1", "L2", "L3", "L4", "L5"])
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Phase B1 combined degradation -- {DATASET_LABEL[dataset]}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
              title="Model (dotted = clean baseline)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"b1_acc_vs_level_{dataset}.png", dpi=150)
    fig.savefig(OUT_DIR / f"b1_acc_vs_level_{dataset}.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Phase C2 family 1 -- type comparison (mean over models) vs PSNR / SSIM
# --------------------------------------------------------------------------- #
def _c2_type_curve(rows, dataset, axis, metric):
    """Return sorted (x, mean_acc) over levels, plus the raw (x, acc) cloud."""
    by_level_acc: dict[int, list[float]] = defaultdict(list)
    by_level_x: dict[int, list[float]] = defaultdict(list)
    cloud_x: list[float] = []
    cloud_y: list[float] = []
    for r in rows:
        if r["dataset"] != dataset or r["axis"] != axis:
            continue
        by_level_acc[r["level"]].append(r["val_acc"])
        by_level_x[r["level"]].append(r[metric])
        cloud_x.append(r[metric])
        cloud_y.append(r["val_acc"])
    pts = []
    for lv in sorted(by_level_acc):
        x = sum(by_level_x[lv]) / len(by_level_x[lv])
        y = sum(by_level_acc[lv]) / len(by_level_acc[lv])
        pts.append((x, y))
    pts.sort(key=lambda t: t[0])
    return pts, cloud_x, cloud_y


def _spearman(xs, ys) -> float:
    if len(xs) >= 3 and len(set(xs)) > 1:
        rho, _p = spearmanr(xs, ys)
        return float(rho)
    return float("nan")


def _plot_c2_type(rows, dataset, metric, y_range) -> dict[str, float]:
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    rho_by_axis: dict[str, float] = {}
    for axis in C2_AXES:
        pts, cx, cy = _c2_type_curve(rows, dataset, axis, metric)
        if not pts:
            continue
        rho_by_axis[axis] = _spearman(cx, cy)
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        st = AXIS_STYLE[axis]
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], linestyle="-",
                linewidth=1.8, markersize=6, label=st["label"])
    ax.set_xlabel(METRIC_LABEL[metric])
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_ylim(*y_range)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"Phase C2 axis comparison -- {DATASET_LABEL[dataset]}")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, title="Degradation axis")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"c2_type_{metric}_{dataset}.png", dpi=150)
    fig.savefig(OUT_DIR / f"c2_type_{metric}_{dataset}.pdf")
    plt.close(fig)
    return rho_by_axis


# --------------------------------------------------------------------------- #
# Phase C2 family 2 -- model comparison (per axis) vs PSNR / SSIM
# --------------------------------------------------------------------------- #
def _plot_c2_model(rows, dataset, metric, y_range) -> None:
    fig, axarr = plt.subplots(1, len(C2_AXES), figsize=(11.0, 3.8), dpi=150, sharey=True)
    for ax, axis in zip(axarr, C2_AXES):
        for model in MODELS_ORDER:
            pts = sorted(
                ((r[metric], r["val_acc"]) for r in rows
                 if r["dataset"] == dataset and r["axis"] == axis and r["model"] == model),
                key=lambda t: t[0],
            )
            if not pts:
                continue
            xs = [x for x, _ in pts]
            ys = [y for _, y in pts]
            ms = MODEL_STYLE[model]
            ax.plot(xs, ys, color=ms["color"], marker=ms["marker"], linestyle="-",
                    linewidth=1.6, markersize=5, label=ms["label"])
        ax.set_title(AXIS_STYLE[axis]["label"])
        ax.set_xlabel(METRIC_LABEL[metric])
        ax.grid(True, alpha=0.25)
    axarr[0].set_ylabel("Validation accuracy (%)")
    axarr[0].set_ylim(*y_range)
    axarr[-1].legend(loc="lower right", fontsize=8, framealpha=0.9, title="Model")
    fig.suptitle(f"Phase C2 model comparison -- {DATASET_LABEL[dataset]}", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"c2_model_{metric}_{dataset}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"c2_model_{metric}_{dataset}.pdf", bbox_inches="tight")
    plt.close(fig)


def _ranking_sentence(rho_by_axis: dict[str, float], metric: str) -> str:
    valid = {a: r for a, r in rho_by_axis.items() if r == r}  # drop NaN
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    b1, clean = _load_b1(), _load_clean()
    c2 = _load_c2()

    # Phase B1
    for dataset in DATASETS:
        _plot_b1(b1, clean, dataset)

    # Shared y-range across datasets within (phase C2, metric) for comparability.
    all_acc = [r["val_acc"] for r in c2]
    lo, hi = min(all_acc), max(all_acc)
    pad = max(2.0, (hi - lo) * 0.05)
    y_range = (max(0.0, lo - pad), min(100.0, hi + pad))

    meta: dict[str, dict] = {}
    for metric in METRICS:
        for dataset in DATASETS:
            rho = _plot_c2_type(c2, dataset, metric, y_range)
            _plot_c2_model(c2, dataset, metric, y_range)
            meta[f"c2_{dataset}_{metric}"] = {
                "dataset": dataset,
                "metric": metric,
                "y_range_pct": list(y_range),
                "spearman_rho_by_axis": rho,
                "ranking_sentence": _ranking_sentence(rho, metric),
            }

    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    n_pairs = 2 + len(METRICS) * len(DATASETS) * 2  # B1(2) + C2 type(4) + C2 model(4)
    print(f"[build_v5_figures] wrote {n_pairs} figure pairs under {OUT_DIR}")
    print(f"[build_v5_figures] wrote caption metadata -> {META_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
