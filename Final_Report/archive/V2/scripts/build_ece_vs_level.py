"""Build ECE vs level line chart (PRD V2 §4.3 / §4.5 figure 21 / §6.2 step 1).

Reads `artifacts/figures/calibration/_index.json` (42 entries: 7 phase / level
groups x 6 model / dataset pairs) and emits one chart per dataset showing how
expected calibration error rises from L3 -> L5 across the three phases that
have entries at both levels (B, D-T3, B2-T3). Phase B2-nr is L3-only and is
reported as a footnote in the metadata JSON.

Outputs:
  figures/curves/ece_vs_level_cifar10.{png,pdf}
  figures/curves/ece_vs_level_mnist.{png,pdf}
  figures/curves/_ece_vs_level_meta.json

Plot spec aligned with FX-15 styling: figsize=(8, 5), dpi=160; three model
markers (circle / square / triangle); three phase line-styles. Y-axis in
percentage points (matches PRD §4.8 / Finding #8 framing).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "figures" / "calibration" / "_index.json"
OUT_DIR = ROOT / "Final_Report" / "figures" / "curves"
META_JSON = OUT_DIR / "_ece_vs_level_meta.json"


MODEL_STYLE = {
    "resnet50":       {"color": "#1f77b4", "marker": "o", "label": "ResNet50"},
    "densenet121":    {"color": "#2ca02c", "marker": "s", "label": "DenseNet121"},
    "transnext_tiny": {"color": "#d62728", "marker": "^", "label": "TransNeXt-tiny"},
}
MODELS_ORDER = ("resnet50", "densenet121", "transnext_tiny")
DATASETS = ("cifar10", "mnist")
DATASET_LABEL = {"cifar10": "CIFAR-10", "mnist": "MNIST"}

# Phases plotted as lines (both L3 and L5 available).
LINE_PHASES = (
    {"key": ("B",   None),  "label": "B1 (no reg)",       "linestyle": "-"},
    {"key": ("D",   "T3"),  "label": "D-T3 (full reg)",   "linestyle": "--"},
    {"key": ("B2",  "T3"),  "label": "B2-T3 (THz prot.)", "linestyle": ":"},
)


def _load_ece() -> dict[tuple[str, str | None, str, str, int], float]:
    """Map (phase, treatment, model, dataset, level) -> ECE in pp."""
    with SRC.open("r", encoding="utf-8") as f:
        idx = json.load(f)
    out: dict = {}
    for e in idx["entries"]:
        key = (e["phase"], e.get("treatment"), e["model"], e["dataset"], int(e["level"]))
        out[key] = float(e["ece_pp"])
    return out


def _shared_y_range(ece: dict, datasets: tuple[str, ...]) -> tuple[float, float]:
    vals = []
    for (phase, treatment, model, dataset, level), v in ece.items():
        if dataset in datasets and any(
            (phase, treatment) == lp["key"] for lp in LINE_PHASES
        ):
            vals.append(v)
    lo = min(0.0, min(vals) - 1.0)
    hi = max(vals) + 2.0
    return (lo, hi)


def _make_plot(
    ece: dict,
    dataset: str,
    y_range: tuple[float, float],
    out_png: Path,
    out_pdf: Path,
) -> dict:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    series_data: dict[str, dict] = {}
    for model in MODELS_ORDER:
        ms = MODEL_STYLE[model]
        for lp in LINE_PHASES:
            phase, treatment = lp["key"]
            xs = [3, 5]
            ys = [ece.get((phase, treatment, model, dataset, lv)) for lv in xs]
            if None in ys:
                continue
            ax.plot(
                xs,
                ys,
                color=ms["color"],
                marker=ms["marker"],
                linestyle=lp["linestyle"],
                linewidth=1.4,
                markersize=7,
                alpha=0.85,
            )
            series_data[f"{model}/{lp['label']}"] = {
                "L3_pp": ys[0],
                "L5_pp": ys[1],
                "delta_pp": ys[1] - ys[0],
            }

    ax.set_xticks([3, 5])
    ax.set_xticklabels(["L3", "L5"])
    ax.set_xlim(2.7, 5.3)
    ax.set_ylim(*y_range)
    ax.set_xlabel("Degradation level")
    ax.set_ylabel("Expected calibration error (pp)")
    ax.grid(True, alpha=0.25)

    model_handles = [
        plt.Line2D([0], [0],
                   color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                   linestyle="", markersize=7, label=MODEL_STYLE[m]["label"])
        for m in MODELS_ORDER
    ]
    phase_handles = [
        plt.Line2D([0], [0],
                   color="black", linestyle=lp["linestyle"], linewidth=1.4, label=lp["label"])
        for lp in LINE_PHASES
    ]
    leg1 = ax.legend(handles=model_handles, loc="upper left",
                     fontsize=8, framealpha=0.85, title="Model")
    ax.add_artist(leg1)
    ax.legend(handles=phase_handles, loc="lower right",
              fontsize=8, framealpha=0.85, title="Phase")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)

    deltas = [s["delta_pp"] for s in series_data.values()]
    mean_l3 = sum(s["L3_pp"] for s in series_data.values()) / len(series_data)
    mean_l5 = sum(s["L5_pp"] for s in series_data.values()) / len(series_data)
    return {
        "dataset": dataset,
        "y_range_pp": list(y_range),
        "series": series_data,
        "mean_L3_pp": mean_l3,
        "mean_L5_pp": mean_l5,
        "mean_delta_L3_to_L5_pp": (sum(deltas) / len(deltas)) if deltas else None,
        "png": str(out_png.relative_to(ROOT)).replace("\\", "/"),
        "pdf": str(out_pdf.relative_to(ROOT)).replace("\\", "/"),
    }


def _b2nr_footnote(ece: dict) -> dict:
    """Phase B2-nr is L3-only. Report the per-(model, dataset) values for the
    caption footnote so they aren't lost from the figure surface."""
    out = {}
    for (phase, treatment, model, dataset, level), v in ece.items():
        if phase == "B2nr" and level == 3:
            out[f"{model}/{dataset}"] = v
    return out


def main() -> int:
    ece = _load_ece()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_range = _shared_y_range(ece, DATASETS)

    meta: dict = {"b2nr_L3_only_pp": _b2nr_footnote(ece)}
    for dataset in DATASETS:
        out_png = OUT_DIR / f"ece_vs_level_{dataset}.png"
        out_pdf = OUT_DIR / f"ece_vs_level_{dataset}.pdf"
        meta[dataset] = _make_plot(ece, dataset, y_range, out_png, out_pdf)

    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[build_ece_vs_level] wrote {OUT_DIR / 'ece_vs_level_cifar10.png'}")
    print(f"[build_ece_vs_level] wrote {OUT_DIR / 'ece_vs_level_mnist.png'}")
    print(f"[build_ece_vs_level] mean L3 pp -> CIFAR-10: {meta['cifar10']['mean_L3_pp']:.2f}, "
          f"MNIST: {meta['mnist']['mean_L3_pp']:.2f}")
    print(f"[build_ece_vs_level] mean L5 pp -> CIFAR-10: {meta['cifar10']['mean_L5_pp']:.2f}, "
          f"MNIST: {meta['mnist']['mean_L5_pp']:.2f}")
    print(f"[build_ece_vs_level] wrote caption metadata -> {META_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
