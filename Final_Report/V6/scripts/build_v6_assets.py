"""V6 report asset builder: readable figures + deepened-analysis tables.

Fixes the V5 figure legibility problems and adds the V6 full-depth analysis
artifacts (Phase B1 delta-vs-clean, per-level model gaps, full Phase C2
Spearman table).

Figures (PNG + PDF) -> Final_Report/figures/v6/
  block_diagrams/system_overview      (rebuilt: no overlapping labels)
  block_diagrams/pipeline_overview    (rebuilt: no literal LaTeX escapes)
  curves/b1_acc_vs_level              (1x2 panel: CIFAR-10 | MNIST)
  curves/c2_type_psnr / c2_type_ssim  (1x2 panel, one curve per axis)
  curves/c2_model_psnr / c2_model_ssim(2x3 grid, one curve per model)

Tables (LaTeX) -> Final_Report/V6/source/tables/
  tab_b1_delta.tex     Phase B1 accuracy drop vs the Phase A clean baseline
  tab_b1_gap.tex       Per-level TransNeXt margin over the CNNs
  tab_c2_spearman.tex  Spearman rho accuracy-vs-PSNR/SSIM per axis

Meta -> Final_Report/figures/v6/_v6_meta.json (every number quoted in prose)

Run from the repository root:
  python Final_Report/V6/scripts/build_v6_assets.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import spearmanr

# Readable-at-print-size typography: these figures render at 8-16 cm width in
# the report, so the base font is deliberately large relative to the canvas.
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11.5,
        "legend.title_fontsize": 12,
        "svg.hashsalt": "v6_figures",
    }
)

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "Final_Report" / "data" / "tables"
B1_CSV = TABLES / "tab_V_phase_b.csv"
A_CSV = TABLES / "tab_IV_phase_a.csv"
IQ_CSV = TABLES / "per_cell_with_iq.csv"
FIG_DIR = ROOT / "Final_Report" / "figures" / "v6"
TEX_DIR = ROOT / "Final_Report" / "V6" / "source" / "tables"
META_JSON = FIG_DIR / "_v6_meta.json"

MODELS_ORDER = ("resnet50", "densenet121", "transnext_tiny")
MODEL_STYLE = {
    "resnet50": {"color": "#1f77b4", "marker": "o", "label": "ResNet50"},
    "densenet121": {"color": "#2ca02c", "marker": "s", "label": "DenseNet121"},
    "transnext_tiny": {"color": "#d62728", "marker": "^", "label": "TransNeXt-tiny"},
}
MODEL_TEX = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "transnext_tiny": "TransNeXt-tiny",
}

C2_AXES = ("resolution", "blur", "salt_pepper")
AXIS_STYLE = {
    "resolution": {"color": "#1f77b4", "marker": "o", "label": "Resolution"},
    "blur": {"color": "#ff7f0e", "marker": "s", "label": "Blur"},
    "salt_pepper": {"color": "#2ca02c", "marker": "^", "label": "Salt & pepper"},
}
AXIS_TEX = {"resolution": "Resolution", "blur": "Blur", "salt_pepper": "Salt \\& pepper"}

DATASETS = ("cifar10", "mnist")
DATASET_LABEL = {"cifar10": "CIFAR-10", "mnist": "MNIST"}
METRICS = ("psnr", "ssim")
METRIC_LABEL = {"psnr": "PSNR (dB)", "ssim": "SSIM"}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _load_b1() -> dict[tuple[str, str], dict[int, float]]:
    """Prefer the multi-seed mean where audited (L3) so every derived table
    quotes the same value as the headline accuracy table."""
    out: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    with B1_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            acc = row["val_acc_mean"] or row["val_acc"]
            out[(row["model"], row["dataset"])][int(row["level"])] = (
                float(acc) * 100.0
            )
    return out


def _load_clean() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    with A_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[(row["model"], row["dataset"])] = float(row["val_acc"]) * 100.0
    return out


def _load_c2() -> list[dict]:
    out: list[dict] = []
    with IQ_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] != "C2" or not row["psnr_mean"] or not row["ssim_mean"]:
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
# Block diagrams (rebuilt so every label fits inside its box)
# --------------------------------------------------------------------------- #
def _draw_chain(ax, boxes: list[tuple[str, str]], box_w: float, box_h: float,
                gap: float, fontsize: float, sub: dict[int, str] | None = None) -> None:
    """boxes: list of (label, facecolor). Sub: optional annotation below box i."""
    x = 0.0
    centers = []
    for label, color in boxes:
        ax.add_patch(
            Rectangle((x, 0.0), box_w, box_h, facecolor=color,
                      edgecolor="black", linewidth=1.1)
        )
        ax.text(x + box_w / 2.0, box_h / 2.0, label, ha="center", va="center",
                fontsize=fontsize, linespacing=1.35)
        centers.append(x + box_w / 2.0)
        x += box_w + gap
    for i in range(len(boxes) - 1):
        x0 = centers[i] + box_w / 2.0
        x1 = centers[i + 1] - box_w / 2.0
        ax.add_patch(
            FancyArrowPatch((x0 + 0.02, box_h / 2.0), (x1 - 0.02, box_h / 2.0),
                            arrowstyle="-|>", mutation_scale=16,
                            linewidth=1.4, color="black")
        )
    if sub:
        for i, note in sub.items():
            ax.text(centers[i], -0.16, note, ha="center", va="top",
                    fontsize=fontsize - 1.5, style="italic", color="#444444")
    ax.set_xlim(-0.15, x - gap + 0.15)
    ax.set_ylim(-0.55 if sub else -0.15, box_h + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_block_diagrams() -> None:
    out = FIG_DIR / "block_diagrams"
    out.mkdir(parents=True, exist_ok=True)

    # System overview: 6 stages.
    fig, ax = plt.subplots(figsize=(13.2, 2.6), dpi=200)
    boxes = [
        ("Input image\nCIFAR-10 / MNIST\n(32 / 28 px)", "#fce5cd"),
        ("THz-like\ndegradation\npipeline (v2)", "#cfe2f3"),
        ("ImageNet\nnormalization\n(mean / std)", "#d9ead3"),
        ("Backbone\nResNet50 /\nDenseNet121 /\nTransNeXt-tiny", "#f4cccc"),
        ("Classification\nhead\n(GAP + linear)", "#d9d2e9"),
        ("Predicted\nclass\n(10-way)", "#fff2cc"),
    ]
    _draw_chain(ax, boxes, box_w=1.95, box_h=1.45, gap=0.55, fontsize=11.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "system_overview.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "system_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    # Degradation pipeline v2: 8 stages, faithful to src/data/degrade.py order
    # (noise and salt-and-pepper act at the low resolution, blur after upsample).
    fig, ax = plt.subplots(figsize=(13.2, 2.9), dpi=200)
    steps = [
        ("Clean\nimage", "#ffffff"),
        ("Saturation\nlerp (s)", "#cfe2f3"),
        ("Downsample\nto r × r\n(bilinear)", "#cfe2f3"),
        ("Gaussian\nnoise (σ)", "#a4c2f4"),
        ("Salt &\npepper (p)", "#a4c2f4"),
        ("Upsample to\n224 × 224\n(bicubic)", "#cfe2f3"),
        ("Gaussian\nblur (K, σ)", "#cfe2f3"),
        ("Clamp to\n[0, 1]", "#ffffff"),
    ]
    sub = {3: "applied at the low\nresolution r × r", 4: "applied at the low\nresolution r × r"}
    _draw_chain(ax, steps, box_w=1.62, box_h=1.45, gap=0.42, fontsize=11.0, sub=sub)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "pipeline_overview.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "pipeline_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[v6-assets] block diagrams rebuilt")


# --------------------------------------------------------------------------- #
# Phase B1 -- accuracy vs combined level (1x2 panel)
# --------------------------------------------------------------------------- #
def fig_b1(b1, clean) -> None:
    out = FIG_DIR / "curves"
    out.mkdir(parents=True, exist_ok=True)
    fig, axarr = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=200, sharey=True)
    for ax, dataset in zip(axarr, DATASETS):
        for model in MODELS_ORDER:
            levels = sorted(b1[(model, dataset)])
            ys = [b1[(model, dataset)][lv] for lv in levels]
            ms = MODEL_STYLE[model]
            ax.plot(levels, ys, color=ms["color"], marker=ms["marker"],
                    linewidth=2.2, markersize=8, label=ms["label"])
            ax.axhline(clean[(model, dataset)], color=ms["color"],
                       linestyle=":", linewidth=1.4, alpha=0.75)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["L1", "L2", "L3", "L4", "L5"])
        ax.set_xlabel("Combined degradation level")
        ax.set_ylim(0, 104)
        ax.grid(True, alpha=0.3)
        ax.set_title(DATASET_LABEL[dataset])
    axarr[0].set_ylabel("Validation accuracy (%)")
    handles = [Line2D([], [], **{k: v for k, v in MODEL_STYLE[m].items()
                                 if k in ("color", "marker", "label")},
                      linewidth=2.2, markersize=8) for m in MODELS_ORDER]
    handles.append(Line2D([], [], color="#555555", linestyle=":", linewidth=1.6,
                          label="Clean baseline (Phase A)"))
    axarr[0].legend(handles=handles, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out / "b1_acc_vs_level.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "b1_acc_vs_level.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[v6-assets] b1_acc_vs_level rebuilt")


# --------------------------------------------------------------------------- #
# Phase C2 -- axis (type) comparison vs PSNR / SSIM (1x2 panel per metric)
# --------------------------------------------------------------------------- #
def _c2_axis_mean_curve(rows, dataset, axis, metric):
    by_level: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r["dataset"] == dataset and r["axis"] == axis:
            by_level[r["level"]].append((r[metric], r["val_acc"]))
    pts = []
    for lv in sorted(by_level):
        xs = [x for x, _ in by_level[lv]]
        ys = [y for _, y in by_level[lv]]
        pts.append((lv, sum(xs) / len(xs), sum(ys) / len(ys)))
    return pts  # [(level, mean_metric, mean_acc)]


def fig_c2_type(rows, y_range) -> None:
    out = FIG_DIR / "curves"
    for metric in METRICS:
        fig, axarr = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=200, sharey=True)
        for ax, dataset in zip(axarr, DATASETS):
            for axis in C2_AXES:
                pts = _c2_axis_mean_curve(rows, dataset, axis, metric)
                if not pts:
                    continue
                pts_sorted = sorted(pts, key=lambda t: t[1])
                xs = [x for _, x, _ in pts_sorted]
                ys = [y for _, _, y in pts_sorted]
                st = AXIS_STYLE[axis]
                ax.plot(xs, ys, color=st["color"], marker=st["marker"],
                        linewidth=2.2, markersize=8, label=st["label"])
                # Severity direction annotations: L5 at the degraded end,
                # L1 at the high-quality end (resolution axis only -- the
                # curve the reading hinges on; avoids label clutter).
                if axis == "resolution":
                    lv_by_x = {x: lv for lv, x, _ in pts}
                    x_lo, x_hi = min(xs), max(xs)
                    y_lo = ys[xs.index(x_lo)]
                    y_hi = ys[xs.index(x_hi)]
                    ax.annotate(f"L{lv_by_x[x_lo]}", (x_lo, y_lo),
                                textcoords="offset points", xytext=(-2, -16),
                                fontsize=10.5, color=st["color"])
                    ax.annotate(f"L{lv_by_x[x_hi]}", (x_hi, y_hi),
                                textcoords="offset points", xytext=(4, -14),
                                fontsize=10.5, color=st["color"])
            ax.set_xlabel(METRIC_LABEL[metric])
            ax.set_ylim(*y_range)
            ax.grid(True, alpha=0.3)
            ax.set_title(DATASET_LABEL[dataset])
        axarr[0].set_ylabel("Validation accuracy (%)")
        axarr[0].legend(loc="lower right", framealpha=0.95, title="Degradation axis")
        fig.tight_layout()
        fig.savefig(out / f"c2_type_{metric}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / f"c2_type_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)
    print("[v6-assets] c2_type_{psnr,ssim} rebuilt")


# --------------------------------------------------------------------------- #
# Phase C2 -- model comparison (2x3 grid per metric)
# --------------------------------------------------------------------------- #
def fig_c2_model(rows, y_range) -> None:
    out = FIG_DIR / "curves"
    for metric in METRICS:
        fig, axarr = plt.subplots(2, 3, figsize=(12.6, 7.4), dpi=200,
                                  sharey=True)
        for r_i, dataset in enumerate(DATASETS):
            for c_i, axis in enumerate(C2_AXES):
                ax = axarr[r_i][c_i]
                for model in MODELS_ORDER:
                    pts = sorted(
                        (r[metric], r["val_acc"])
                        for r in rows
                        if r["dataset"] == dataset and r["axis"] == axis
                        and r["model"] == model
                    )
                    xs = [x for x, _ in pts]
                    ys = [y for _, y in pts]
                    ms = MODEL_STYLE[model]
                    ax.plot(xs, ys, color=ms["color"], marker=ms["marker"],
                            linewidth=2.0, markersize=6.5, label=ms["label"])
                if r_i == 0:
                    ax.set_title(AXIS_STYLE[axis]["label"])
                ax.set_xlabel(METRIC_LABEL[metric])
                ax.grid(True, alpha=0.3)
            axarr[r_i][0].set_ylabel(
                f"{DATASET_LABEL[dataset]}\nValidation accuracy (%)"
            )
        axarr[0][0].set_ylim(*y_range)
        handles, labels = axarr[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   frameon=True, framealpha=0.95, bbox_to_anchor=(0.5, -0.005))
        fig.tight_layout(rect=(0.0, 0.055, 1.0, 1.0))
        fig.savefig(out / f"c2_model_{metric}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out / f"c2_model_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)
    print("[v6-assets] c2_model_{psnr,ssim} rebuilt")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def tab_b1_delta(b1, clean) -> dict:
    """Phase B1 drop vs the Phase A clean baseline, in percentage points."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Phase B1 accuracy drop versus the Phase A clean baseline"
        " (percentage points); the Clean column is the Phase A accuracy"
        " (\\%). L3 uses the multi-seed mean.}",
        "\\label{tab:b1_delta}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Model & Dataset & Clean & $\\Delta$L1 & $\\Delta$L2 & $\\Delta$L3"
        " & $\\Delta$L4 & $\\Delta$L5 \\\\",
        "\\midrule",
    ]
    deltas: dict[str, dict[str, list[float]]] = {}
    for model in MODELS_ORDER:
        for dataset in DATASETS:
            base = clean[(model, dataset)]
            ds = [b1[(model, dataset)][lv] - base for lv in range(1, 6)]
            deltas.setdefault(dataset, {})[model] = ds
            cells = " & ".join(f"{d:+.2f}" for d in ds)
            lines.append(
                f"{MODEL_TEX[model]} & {DATASET_LABEL[dataset]} & "
                f"{base:.2f} & {cells} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX_DIR / "tab_b1_delta.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return deltas


def tab_b1_gap(b1) -> dict:
    """TransNeXt-tiny margin over the CNNs per level (pp)."""
    gaps: dict[str, dict[str, list[float]]] = {}
    for dataset in DATASETS:
        tnx = [b1[("transnext_tiny", dataset)][lv] for lv in range(1, 6)]
        cnn_mean = [
            (b1[("resnet50", dataset)][lv] + b1[("densenet121", dataset)][lv]) / 2.0
            for lv in range(1, 6)
        ]
        cnn_best = [
            max(b1[("resnet50", dataset)][lv], b1[("densenet121", dataset)][lv])
            for lv in range(1, 6)
        ]
        gaps[dataset] = {
            "vs_cnn_mean": [t - c for t, c in zip(tnx, cnn_mean)],
            "vs_cnn_best": [t - c for t, c in zip(tnx, cnn_best)],
        }
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{TransNeXt-tiny accuracy margin over the two CNNs on the"
        " Phase B1 curve (percentage points); L3 uses the multi-seed mean"
        " (seed noise $\\sigma < 1.2$).}",
        "\\label{tab:b1_gap}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset & Margin & L1 & L2 & L3 & L4 & L5 \\\\",
        "\\midrule",
    ]
    for dataset in DATASETS:
        for key, label in (("vs_cnn_mean", "vs CNN mean"),
                           ("vs_cnn_best", "vs best CNN")):
            cells = " & ".join(f"{g:+.2f}" for g in gaps[dataset][key])
            lines.append(f"{DATASET_LABEL[dataset]} & {label} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX_DIR / "tab_b1_gap.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return gaps


def tab_c2_spearman(rows) -> dict:
    """Spearman rho between accuracy and measured image quality per axis.

    Pooled over the three models (15 points per cell: 3 models x 5 levels);
    the per-model range documents how stable the association is.
    """
    stats: dict[str, dict[str, dict[str, dict]]] = {}
    for dataset in DATASETS:
        stats[dataset] = {}
        for axis in C2_AXES:
            stats[dataset][axis] = {}
            for metric in METRICS:
                pooled = [
                    (r[metric], r["val_acc"])
                    for r in rows
                    if r["dataset"] == dataset and r["axis"] == axis
                ]
                rho_pooled = float(spearmanr([p[0] for p in pooled],
                                             [p[1] for p in pooled]).statistic)
                per_model = []
                for model in MODELS_ORDER:
                    pm = [
                        (r[metric], r["val_acc"])
                        for r in rows
                        if r["dataset"] == dataset and r["axis"] == axis
                        and r["model"] == model
                    ]
                    per_model.append(float(spearmanr(
                        [p[0] for p in pm], [p[1] for p in pm]).statistic))
                stats[dataset][axis][metric] = {
                    "pooled": rho_pooled,
                    "per_model_min": min(per_model),
                    "per_model_max": max(per_model),
                }
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Phase C2: Spearman rank correlation $\\rho$ between"
        " accuracy and measured image quality, pooled over the three"
        " backbones ($n = 15$ per entry).}",
        "\\label{tab:c2_spearman}",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{CIFAR-10} & \\multicolumn{2}{c}{MNIST} \\\\",
        "Axis & $\\rho$ vs PSNR & $\\rho$ vs SSIM & $\\rho$ vs PSNR"
        " & $\\rho$ vs SSIM \\\\",
        "\\midrule",
    ]
    for axis in C2_AXES:
        cells = []
        for dataset in DATASETS:
            for metric in METRICS:
                cells.append(f"{stats[dataset][axis][metric]['pooled']:+.2f}")
        lines.append(f"{AXIS_TEX[axis]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TEX_DIR / "tab_c2_spearman.tex").write_text("\n".join(lines) + "\n",
                                                 encoding="utf-8")
    return stats


def c2_axis_attribution(rows, clean) -> dict:
    """Mean Delta_A->C2 per axis across the 6 (model, dataset) pairs.

    Cross-checked against PRD v5 section 4.6 (resolution -27.45 / blur -6.58 /
    salt_pepper -5.13 mean L1-L5).
    """
    per_axis: dict[str, dict[str, float]] = {}
    for axis in C2_AXES:
        per_level = []
        for lv in range(1, 6):
            ds = [
                r["val_acc"] - clean[(r["model"], r["dataset"])]
                for r in rows
                if r["axis"] == axis and r["level"] == lv
            ]
            per_level.append(sum(ds) / len(ds))
        per_axis[axis] = {
            "per_level": per_level,
            "mean_l1_l5": sum(per_level) / len(per_level),
        }
    return per_axis


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    b1, clean, c2 = _load_b1(), _load_clean(), _load_c2()

    fig_block_diagrams()
    fig_b1(b1, clean)

    all_acc = [r["val_acc"] for r in c2]
    lo, hi = min(all_acc), max(all_acc)
    pad = max(2.0, (hi - lo) * 0.05)
    y_range = (max(0.0, lo - pad), min(102.0, hi + pad))
    fig_c2_type(c2, y_range)
    fig_c2_model(c2, y_range)

    deltas = tab_b1_delta(b1, clean)
    gaps = tab_b1_gap(b1)
    spear = tab_c2_spearman(c2)
    attrib = c2_axis_attribution(c2, clean)

    # Cross-check the PRD v5 section 4.6 attribution numbers.
    prd = {"resolution": -27.45, "blur": -6.58, "salt_pepper": -5.13}
    for axis, want in prd.items():
        got = attrib[axis]["mean_l1_l5"]
        assert abs(got - want) < 0.05, f"attribution drift on {axis}: {got:.2f} vs {want}"

    meta = {
        "b1_delta_vs_clean_pp": deltas,
        "b1_transnext_gap_pp": gaps,
        "c2_spearman": spear,
        "c2_axis_attribution_pp": attrib,
    }
    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[v6-assets] tables -> {TEX_DIR}")
    print(f"[v6-assets] meta   -> {META_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
