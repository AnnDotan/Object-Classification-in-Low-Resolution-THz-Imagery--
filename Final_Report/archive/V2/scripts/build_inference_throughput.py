"""Inference-throughput bar chart at publication size (PRD V2 §8 FX-07).

V1 PNG had the top bar riding the upper axis with its value label clipped.
FX-07: `ax.set_ylim(0, max_value * 1.1)`; annotate each bar value above
the bar (mean +/- std ms).

Source CSV:
  Final_Report/data/tables/inference_throughput.csv (copied at v4 close
  from artifacts/figures/inference_throughput.csv).

Output:
  Final_Report/figures/inference_throughput.{png,pdf}
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Final_Report" / "data" / "tables" / "inference_throughput.csv"
OUT_PNG = ROOT / "Final_Report" / "figures" / "inference_throughput.png"
OUT_PDF = ROOT / "Final_Report" / "figures" / "inference_throughput.pdf"


MODEL_LABEL = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "transnext_tiny": "TransNeXt-tiny",
}
MODEL_COLOR = {
    "resnet50": "#1f77b4",
    "densenet121": "#2ca02c",
    "transnext_tiny": "#d62728",
}
ORDER = ("resnet50", "densenet121", "transnext_tiny")


def main() -> int:
    rows: dict[str, dict] = {}
    with SRC.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r

    labels = [MODEL_LABEL[m] for m in ORDER]
    means = [float(rows[m]["mean_ms"]) for m in ORDER]
    stds = [float(rows[m]["std_ms"]) for m in ORDER]
    thrs = [float(rows[m]["throughput_images_per_s"]) for m in ORDER]
    colors = [MODEL_COLOR[m] for m in ORDER]

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=200)
    x = range(len(ORDER))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                  edgecolor="black", linewidth=0.5, alpha=0.9)

    # FX-07: ylim caps the tallest bar with 10% headroom; annotate value above.
    max_top = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(0, max_top * 1.1)

    for xi, mean, std, thr in zip(x, means, stds, thrs):
        ax.text(
            xi, mean + std + max_top * 0.02,
            f"{mean:.2f} $\\pm$ {std:.2f} ms\n({thr:.0f} img/s)",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Per-image latency (ms)", fontsize=10)
    ax.set_title(
        "Inference throughput on RTX 5070 (bf16-mixed, batch=1, "
        "100 warmup / 1000 measurement)", fontsize=10,
    )
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"[build_inference_throughput] wrote {OUT_PNG}")
    print(f"[build_inference_throughput] wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
