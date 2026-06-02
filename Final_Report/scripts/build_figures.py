"""Generate the §VII figures (curves, scatters, attribution, calibration).

All plots are produced from Final_Exp.json + image_quality/_index.csv. No data
is fabricated. Outputs land under Final_Report/figures/ in PNG and PDF.

Figures:
  curves/accuracy_vs_level_<dataset>.{png,pdf}
  curves/drop_vs_clean_<dataset>.{png,pdf}
  curves/accuracy_vs_psnr.{png,pdf}
  curves/accuracy_vs_ssim.{png,pdf}
  attribution/axis_attribution_L3_<dataset>.{png,pdf}
  multiseed/L3_variance_<dataset>.{png,pdf}
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "Final_Exp.json"
PIP = ROOT / "Final_Report" / "data" / "image_quality" / "_index.csv"
FIG = ROOT / "Final_Report" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.unicode_minus": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODELS = ("resnet50", "densenet121", "transnext_tiny")
MODEL_LABELS = {"resnet50": "ResNet50", "densenet121": "DenseNet121", "transnext_tiny": "TransNeXt-tiny"}
MODEL_COLORS = {"resnet50": "#1f77b4", "densenet121": "#2ca02c", "transnext_tiny": "#d62728"}
MODEL_MARKERS = {"resnet50": "o", "densenet121": "s", "transnext_tiny": "^"}
DATASETS = ("cifar10", "mnist")
DATASET_LABELS = {"cifar10": "CIFAR-10", "mnist": "MNIST"}
AXES = ("resolution", "blur", "noise", "salt_pepper", "saturation")
AXIS_LABELS = {"resolution": "Resolution", "blur": "Blur", "noise": "Noise",
               "salt_pepper": "Salt & Pepper", "saturation": "Saturation"}
PHASE_COLORS = {
    "A": "#000000",
    "B": "#1f77b4",
    "C": "#ff7f0e",
    "D": "#9467bd",
    "B2": "#2ca02c",
    "B2nr": "#7f7f7f",
    "C2": "#17becf",
}


def _save(fig, name: str) -> None:
    out_png = FIG / f"{name}.png"
    out_pdf = FIG / f"{name}.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  wrote {out_png.relative_to(ROOT)}")


def _load_rows() -> list[dict]:
    with SRC.open("r", encoding="utf-8") as f:
        return json.load(f)["rows"]


def _val_acc(r: dict) -> float | None:
    return r.get("val_acc_mean") or r.get("val_acc")


def _multiseed_label(r: dict) -> str:
    """Human-readable cell label for the FX-04 multi-seed boxplot.

    Examples: "B / DenseNet121 / no reg", "C / ResNet50 / resolution",
              "D / TransNeXt-tiny / T3", "B2 / ResNet50 / T3".
    """
    phase = r["phase"]
    model = MODEL_LABELS.get(r["model"], r["model"])
    axis = r.get("axis")
    treatment = r.get("treatment")
    if axis:
        third = AXIS_LABELS.get(axis, axis).lower()
    elif treatment:
        third = treatment
    elif phase in ("B", "B2nr"):
        third = "no reg"
    else:
        third = "-"
    return f"{phase} / {model} / {third}"


# ---------------------------------------------------------------------------
# Fig. 6: Accuracy vs degradation level, one panel per dataset, line per model x phase.
# ---------------------------------------------------------------------------

def fig_accuracy_vs_level(rows: list[dict]) -> None:
    for d in DATASETS:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        # Phase A: horizontal reference per model.
        for m in MODELS:
            clean = [r for r in rows if r["phase"] == "A" and r["model"] == m and r["dataset"] == d]
            if clean:
                clean_acc = _val_acc(clean[0]) * 100
                ax.axhline(clean_acc, color=MODEL_COLORS[m], linestyle=":", alpha=0.5)

        # Phase B curve per model.
        for m in MODELS:
            xs, ys = [], []
            for L in (1, 2, 3, 4, 5):
                cells = [r for r in rows if r["phase"] == "B" and r["model"] == m
                         and r["dataset"] == d and r["level"] == L]
                if cells:
                    xs.append(L)
                    ys.append(_val_acc(cells[0]) * 100)
            ax.plot(xs, ys, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                    label=f"{MODEL_LABELS[m]} (Phase B)", linewidth=1.6)

        # Phase D-T3 curve per model (regularization recovery).
        for m in MODELS:
            xs, ys = [], []
            for L in (1, 2, 3, 4, 5):
                cells = [r for r in rows if r["phase"] == "D" and r["model"] == m
                         and r["dataset"] == d and r["level"] == L and r.get("treatment") == "T3"]
                if cells:
                    xs.append(L)
                    ys.append(_val_acc(cells[0]) * 100)
            ax.plot(xs, ys, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                    linestyle="--", linewidth=1.0, alpha=0.7,
                    label=f"{MODEL_LABELS[m]} (Phase D-T3)")

        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["L1\nMild", "L2\nLight", "L3\nModerate", "L4\nSevere", "L5\nExtreme"])
        ax.set_ylabel("Validation accuracy (\\%)")
        ax.set_xlabel("Degradation level")
        ax.set_ylim(0, 100)
        ax.set_title(f"{DATASET_LABELS[d]} - accuracy vs combined degradation level")
        ax.legend(fontsize=7, loc="lower left", ncol=2, frameon=False)
        _save(fig, f"curves/accuracy_vs_level_{d}")


# ---------------------------------------------------------------------------
# Fig. 7 / 8: Accuracy vs PSNR / SSIM scatter, colored by phase.
# ---------------------------------------------------------------------------

def fig_accuracy_vs_quality(rows: list[dict]) -> None:
    # Build dataset of (psnr, ssim, acc, model, phase, dataset).
    points = []
    for r in rows:
        if r.get("psnr_mean") is None:
            continue
        acc = _val_acc(r)
        if acc is None:
            continue
        points.append({
            "psnr": r["psnr_mean"],
            "ssim": r["ssim_mean"],
            "acc": acc * 100,
            "model": r["model"],
            "phase": r["phase"],
            "dataset": r["dataset"],
        })

    for metric, axlabel in (("psnr", "Mean PSNR (dB)"), ("ssim", "Mean SSIM")):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
        for ax, d in zip(axes, DATASETS):
            pts = [p for p in points if p["dataset"] == d]
            for m in MODELS:
                pp = [p for p in pts if p["model"] == m]
                xs = [p[metric] for p in pp]
                ys = [p["acc"] for p in pp]
                ax.scatter(xs, ys, s=18, alpha=0.55, color=MODEL_COLORS[m],
                           marker=MODEL_MARKERS[m], label=MODEL_LABELS[m])
            # FX-01: Spearman rho moved to bottom-right (was top-left overlapping
            # MNIST cluster); legend moved to upper-left (was lower-right
            # overlapping CIFAR-10 TransNeXt cluster).
            from scipy.stats import spearmanr  # type: ignore
            if len(pts) > 4:
                rho, _ = spearmanr([p[metric] for p in pts], [p["acc"] for p in pts])
                ax.text(0.97, 0.03, f"Spearman $\\rho = {rho:+.3f}$",
                        transform=ax.transAxes, fontsize=8,
                        horizontalalignment="right", verticalalignment="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))
            ax.set_xlabel(axlabel)
            ax.set_title(DATASET_LABELS[d])
            ax.grid(True, linestyle=":", alpha=0.4)
        axes[0].set_ylabel("Validation accuracy (\\%)")
        axes[0].legend(fontsize=7, loc="upper left",
                       bbox_to_anchor=(0.02, 0.98), framealpha=0.85)
        fig.subplots_adjust(wspace=0.18)
        _save(fig, f"curves/accuracy_vs_{metric}")


# ---------------------------------------------------------------------------
# Per-axis attribution bar chart at L3, per dataset.
# ---------------------------------------------------------------------------

def fig_axis_attribution(rows: list[dict]) -> None:
    for d in DATASETS:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        n_axes = len(AXES)
        x = np.arange(n_axes)
        width = 0.27
        for i, m in enumerate(MODELS):
            clean = [r for r in rows if r["phase"] == "A" and r["model"] == m and r["dataset"] == d]
            if not clean:
                continue
            clean_acc = _val_acc(clean[0]) * 100
            deltas = []
            for ax_name in AXES:
                cells = [r for r in rows if r["phase"] == "C" and r["model"] == m
                         and r["dataset"] == d and r["level"] == 3 and r.get("axis") == ax_name]
                if cells:
                    deltas.append(_val_acc(cells[0]) * 100 - clean_acc)
                else:
                    deltas.append(0.0)
            ax.bar(x + (i - 1) * width, deltas, width, color=MODEL_COLORS[m], label=MODEL_LABELS[m])

        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([AXIS_LABELS[a] for a in AXES], rotation=0)
        ax.set_ylabel("Accuracy delta vs clean (pp)")
        ax.set_title(f"{DATASET_LABELS[d]} - per-axis attribution at L3")
        # FX-02: ylim caps the resolution bar above the bottom edge; legend
        # moved outside the data box (was lower-left covering the Resolution
        # group). axes.unicode_minus is already off via rcParams.
        ax.set_ylim(bottom=-32, top=2)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  framealpha=0.85)
        ax.grid(True, linestyle=":", alpha=0.4, axis="y")
        fig.tight_layout()
        _save(fig, f"attribution/axis_attribution_L3_{d}")


# ---------------------------------------------------------------------------
# Multi-seed variance: per-cell mean and std at L3.
# ---------------------------------------------------------------------------

def fig_multiseed_variance(rows: list[dict]) -> None:
    for d in DATASETS:
        cells = [r for r in rows
                 if r["dataset"] == d
                 and r.get("val_acc_mean") is not None
                 and r.get("val_acc_std") is not None
                 and len(r.get("seeds_observed", []) or []) >= 2]
        if not cells:
            continue
        # Plot mean accuracy with std error bars, one bar per cell, color by phase.
        cells.sort(key=lambda r: (r["phase"], r["model"], r.get("axis") or "", r.get("treatment") or ""))
        n = len(cells)
        # FX-04: fixed figsize=(11, 5.5), 45-degree x-labels, legend outside,
        # human-readable cell labels (was `B/res/-` truncations).
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(n)
        means = [r["val_acc_mean"] * 100 for r in cells]
        stds = [r["val_acc_std"] * 100 for r in cells]
        colors = [PHASE_COLORS.get(r["phase"], "#7f7f7f") for r in cells]
        ax.bar(x, means, yerr=stds, color=colors, capsize=2, edgecolor="black", linewidth=0.3)
        labels = [_multiseed_label(r) for r in cells]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Validation accuracy (\\%)")
        ax.set_title(f"{DATASET_LABELS[d]} - multi-seed variance at L3 (mean $\\pm$ $\\sigma$, n=3)")
        ax.grid(True, linestyle=":", alpha=0.4, axis="y")
        # Phase legend outside the data box.
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=PHASE_COLORS[p], label=p)
                   for p in ("B", "C", "D", "B2", "B2nr", "C2") if p in {r["phase"] for r in cells}]
        ax.legend(handles=handles, fontsize=8, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), framealpha=0.85)
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        _save(fig, f"multiseed/L3_variance_{d}")


# ---------------------------------------------------------------------------
# ECE vs level: read each cell's `runs/final/<tag>/metrics.json` for ECE.
# ---------------------------------------------------------------------------

def _load_ece(tag: str) -> float | None:
    p = ROOT / "runs" / "final" / tag / "metrics.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            j = json.load(f)
        return j.get("ece") or j.get("val_ece")
    except (json.JSONDecodeError, OSError):
        return None


def fig_ece_vs_level(rows: list[dict]) -> None:
    for d in DATASETS:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        any_data = False
        for m in MODELS:
            xs, ys = [], []
            for L in (1, 2, 3, 4, 5):
                cells = [r for r in rows if r["phase"] == "B" and r["model"] == m
                         and r["dataset"] == d and r["level"] == L]
                if not cells:
                    continue
                ece = _load_ece(cells[0]["tag"])
                if ece is None:
                    continue
                xs.append(L)
                ys.append(ece * 100 if ece < 1.5 else ece)
            if xs:
                any_data = True
                ax.plot(xs, ys, marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                        label=MODEL_LABELS[m])
        if not any_data:
            plt.close(fig)
            continue
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["L1", "L2", "L3", "L4", "L5"])
        ax.set_xlabel("Degradation level")
        ax.set_ylabel("Expected Calibration Error (\\%)")
        ax.set_title(f"{DATASET_LABELS[d]} - ECE vs degradation level (Phase B)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left", frameon=False)
        _save(fig, f"calibration/ece_vs_level_{d}")


# ---------------------------------------------------------------------------
# Gantt: planned vs actual.
# ---------------------------------------------------------------------------

def fig_gantt() -> None:
    planned = [
        ("Data Simulation & Setup", "2026-01-01", "2026-02-28"),
        ("Model Implementation", "2026-02-01", "2026-04-30"),
        ("Progress Submission", "2026-03-15", "2026-03-15"),
        ("Advanced Research (TransNeXt)", "2026-04-01", "2026-05-31"),
        ("Poster & Abstracts", "2026-05-31", "2026-05-31"),
        ("Final Presentation", "2026-06-21", "2026-06-21"),
        ("Results Analysis & Final Report", "2026-06-01", "2026-07-26"),
        ("Final Submission", "2026-07-26", "2026-07-26"),
    ]
    actual = [
        ("Data Simulation & Setup", "2026-01-15", "2026-02-28"),
        ("Model Implementation", "2026-02-15", "2026-04-30"),
        ("Progress Submission", "2026-03-15", "2026-03-15"),
        ("Advanced Research (TransNeXt)", "2026-04-15", "2026-05-31"),
        ("Phase D regularization sweep", "2026-05-20", "2026-05-26"),
        ("Phase B2 / B2nr / C2", "2026-05-27", "2026-05-31"),
        ("Multi-seed audit (US-042)", "2026-05-31", "2026-06-01"),
        ("v4 closure (US-045..US-053)", "2026-06-01", "2026-06-01"),
        ("Results Analysis & Final Report", "2026-06-01", "2026-07-26"),
        ("Final Submission", "2026-07-26", "2026-07-26"),
    ]
    from datetime import date
    def _to_day(s: str) -> int:
        y, m, d = map(int, s.split("-"))
        return (date(y, m, d) - date(2026, 1, 1)).days

    fig, (axp, axa) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax, data, title in ((axp, planned, "Planned schedule (PDR)"),
                            (axa, actual, "Actual schedule (v4 closure)")):
        for i, (name, start, end) in enumerate(reversed(data)):
            s = _to_day(start); e = _to_day(end)
            ax.barh(i, max(e - s, 1), left=s, height=0.6,
                    color="#1f77b4" if title.startswith("Planned") else "#2ca02c",
                    edgecolor="black", linewidth=0.4)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([n for n, _, _ in reversed(data)], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    # Month ticks
    month_starts = [_to_day(f"2026-{mm:02d}-01") for mm in range(1, 8)]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
    for ax in (axp, axa):
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels)
        ax.set_xlim(-5, _to_day("2026-07-28"))
    axa.set_xlabel("2026 calendar month")
    _save(fig, "gantt/gantt_planned_vs_actual")


# ---------------------------------------------------------------------------
# Block diagrams (text-only schematic via matplotlib).
# ---------------------------------------------------------------------------

def fig_block_diagrams() -> None:
    # System block diagram (Fig. 1)
    fig, ax = plt.subplots(figsize=(8, 2.2))
    boxes = [
        ("Input Image\n(CIFAR-10 / MNIST)", 0.5, "#fce5cd"),
        ("Degradation Pipeline\n(5-level THz-like)", 2.0, "#cfe2f3"),
        ("ImageNet Normalize\n(mean / std)", 3.5, "#d9ead3"),
        ("Backbone\n(ResNet50 / DenseNet121 /\nTransNeXt-tiny)", 5.0, "#f4cccc"),
        ("Classification Head\n(GAP + Linear)", 6.7, "#d9d2e9"),
        ("Predicted Class", 8.2, "#fff2cc"),
    ]
    for label, x, color in boxes:
        ax.add_patch(plt.Rectangle((x - 0.55, 0.3), 1.1, 0.9,
                                   facecolor=color, edgecolor="black", linewidth=0.8))
        ax.text(x, 0.75, label, ha="center", va="center", fontsize=7.5)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + 0.55
        x1 = boxes[i + 1][1] - 0.55
        ax.annotate("", xy=(x1, 0.75), xytext=(x0, 0.75),
                    arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlim(-0.5, 9)
    ax.set_ylim(0, 1.5)
    ax.axis("off")
    ax.set_title("System block diagram", fontsize=9)
    _save(fig, "block_diagrams/system_overview")

    # Pipeline block diagram (Fig. 2)
    fig, ax = plt.subplots(figsize=(8, 2.2))
    steps = [
        ("Clean\nimage", 0.5, "#ffffff"),
        ("Saturation\nlerp (s)", 1.7, "#cfe2f3"),
        ("Downsample\n(low\\_res)", 2.9, "#cfe2f3"),
        ("Additive\nnoise ($\\sigma$)", 4.1, "#cfe2f3"),
        ("Salt \\&\nPepper (p)", 5.3, "#cfe2f3"),
        ("Bicubic\nupsample 224", 6.5, "#cfe2f3"),
        ("Blur\n(K, $\\sigma$)", 7.7, "#cfe2f3"),
        ("Clamp\n$[0, 1]$", 8.9, "#ffffff"),
    ]
    for label, x, color in steps:
        ax.add_patch(plt.Rectangle((x - 0.5, 0.3), 1.0, 0.9,
                                   facecolor=color, edgecolor="black", linewidth=0.8))
        ax.text(x, 0.75, label, ha="center", va="center", fontsize=7)
    for i in range(len(steps) - 1):
        x0 = steps[i][1] + 0.5
        x1 = steps[i + 1][1] - 0.5
        ax.annotate("", xy=(x1, 0.75), xytext=(x0, 0.75),
                    arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlim(-0.3, 9.7)
    ax.set_ylim(0, 1.5)
    ax.axis("off")
    ax.set_title("THz-like degradation pipeline (pipeline v2)", fontsize=9)
    _save(fig, "block_diagrams/pipeline_overview")


def main() -> None:
    rows = _load_rows()
    print("[build_figures] accuracy vs level")
    fig_accuracy_vs_level(rows)
    print("[build_figures] accuracy vs quality (PSNR / SSIM)")
    fig_accuracy_vs_quality(rows)
    print("[build_figures] per-axis attribution")
    fig_axis_attribution(rows)
    print("[build_figures] multi-seed variance")
    fig_multiseed_variance(rows)
    print("[build_figures] ECE vs level")
    fig_ece_vs_level(rows)
    print("[build_figures] Gantt")
    fig_gantt()
    print("[build_figures] block diagrams")
    fig_block_diagrams()
    print("[build_figures] DONE")


if __name__ == "__main__":
    main()
