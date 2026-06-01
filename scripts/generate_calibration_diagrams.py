"""US-045 (c) — Calibration / ECE reliability diagrams for L3 + L5 cells.

Cells (seed = 42 only):
  - 24 L3 headline cells (B1, D-T3, B2, B2-nr × 6 (model, dataset) pairs)
  - 18 L5 cells (B1, D-T3, B2 × 6 pairs; B2-nr is L3-only)
  Total: 42 PNGs + 1 LaTeX table.

ECE computed per torchmetrics.CalibrationError(n_bins=15, task=multiclass)
on softmax probabilities. L3 cells use test-split logits (cached by
run_test_set_inference.py); L5 cells use val-set logits.

Outputs:
  - artifacts/figures/calibration/<tag>_L{3,5}.png (42 files)
  - docs/_autogen/calibration_table.tex
  - artifacts/figures/calibration/_index.json
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
import torchmetrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.diagnostics_common import (
    get_or_run_test_logits,
    get_or_run_val_logits,
    headline_l3_bases,
    headline_l5_bases,
    load_cell_spec,
)

OUT_DIR = ROOT / "artifacts" / "figures" / "calibration"
TEX_OUT = ROOT / "docs" / "_autogen" / "calibration_table.tex"
INDEX = OUT_DIR / "_index.json"
TEST_SPLIT = ROOT / "artifacts" / "validation" / "test_split_indices.json"
N_BINS = 15


def _ece_and_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """Numpy ECE + per-bin (conf, acc, count) for reliability diagram.

    Returns ece (float), bin_centers (n_bins,), bin_conf (n_bins,),
    bin_acc (n_bins,), bin_count (n_bins,). Empty bins return (0, 0, 0).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    total = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_b = int(mask.sum())
        bin_count[i] = n_b
        if n_b == 0:
            continue
        bin_conf[i] = float(confidences[mask].mean())
        bin_acc[i] = float(correct[mask].mean())
        ece += (n_b / total) * abs(bin_conf[i] - bin_acc[i])
    return float(ece), bin_centers, bin_conf, bin_acc, bin_count


def _render(title: str, ece: float, bin_centers, bin_acc, bin_conf, bin_count, out_png: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.6, 6.2), dpi=130,
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    width = (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 0.05
    nonzero = bin_count > 0
    ax1.bar(bin_centers[nonzero], bin_acc[nonzero], width=width * 0.95,
            color="#4c72b0", edgecolor="black", linewidth=0.4, label="empirical accuracy")
    # Gap = acc - conf, drawn as a hatched gap to bin_conf
    for c, a, cf, n in zip(bin_centers, bin_acc, bin_conf, bin_count):
        if n == 0:
            continue
        ax1.plot([c - width * 0.475, c + width * 0.475], [cf, cf], color="red", linewidth=1.3)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title(f"{title}\nECE = {ece*100:.2f}%", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # Confidence histogram
    ax2.bar(bin_centers[nonzero], bin_count[nonzero], width=width * 0.95,
            color="#c44e52", edgecolor="black", linewidth=0.4)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("# samples")
    ax2.set_yscale("symlog", linthresh=10)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _process(base: str, level: int, run_dir: Path, dev: torch.device,
             test_idxs: list[int] | None, skip_existing: bool) -> dict | None:
    if not (run_dir / "metrics.json").exists():
        print(f"[skip] {run_dir.name} — no metrics.json")
        return None
    spec = load_cell_spec(run_dir)
    out_png = OUT_DIR / f"{base}_L{level}.png"
    if skip_existing and out_png.exists():
        print(f"[cached-png] {run_dir.name}_L{level}")
        # Still need ECE for table — reload logits cache
    # Pick logits source
    if level == 3 and test_idxs is not None:
        logits, labels = get_or_run_test_logits(spec, test_idxs, dev, force=False)
        split = "test"
    else:
        logits, labels = get_or_run_val_logits(spec, dev, force=False)
        split = "val"
    # Numerical stability via float64 softmax
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    ece, bcent, bconf, bacc, bcnt = _ece_and_bins(probs, labels, n_bins=N_BINS)
    # Cross-check with torchmetrics for the table value
    tm = torchmetrics.CalibrationError(task="multiclass", num_classes=10, n_bins=N_BINS)
    tm.update(torch.from_numpy(probs.astype(np.float32)), torch.from_numpy(labels.astype(np.int64)))
    ece_tm = float(tm.compute())
    title = f"{base} — {spec.model_name} / {spec.dataset} L{level}  ({split} split, n={int(len(labels))})"
    if not (skip_existing and out_png.exists()):
        _render(title, ece, bcent, bacc, bconf, bcnt, out_png)
    print(f"[done] {run_dir.name}_L{level}: ECE_numpy={ece*100:.2f}% ECE_tm={ece_tm*100:.2f}% split={split}")
    return {
        "tag": base,
        "phase": spec.phase,
        "treatment": spec.treatment,
        "model": spec.model_name,
        "dataset": spec.dataset,
        "level": level,
        "split": split,
        "ece_pp": ece * 100.0,
        "ece_tm_pp": ece_tm * 100.0,
        "png": str(out_png.relative_to(ROOT)).replace("\\", "/"),
    }


def _emit_latex(entries: list[dict]) -> str:
    """Group by level then by (phase, model, dataset). Compact table."""
    lines = []
    lines.append(r"% Auto-generated by scripts/generate_calibration_diagrams.py - do not edit by hand")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\caption{Expected Calibration Error (ECE, \%), $n_{\text{bins}}=15$. L3 cells use the held-out test split; L5 cells use the validation set.}")
    lines.append(r"\label{tab:calibration}")
    lines.append(r"\begin{tabular}{lll rr}")
    lines.append(r"\toprule")
    lines.append(r"Phase & Model & Dataset & L3 ECE (\%) & L5 ECE (\%) \\")
    lines.append(r"\midrule")
    # Group
    keyed: dict[tuple, dict[int, float]] = {}
    for e in entries:
        k = (e["phase"], e["treatment"] or "--", e["model"], e["dataset"])
        keyed.setdefault(k, {})[int(e["level"])] = float(e["ece_pp"])
    # Sort by phase order
    phase_order = {"B": 0, "D": 1, "B2": 2, "B2nr": 3}
    for k in sorted(keyed, key=lambda kk: (phase_order.get(kk[0], 99), kk[2], kk[3])):
        phase_disp = {"B": "B1", "D": "D-T3", "B2": "B2", "B2nr": "B2-nr"}.get(k[0], k[0])
        model_disp = k[2].replace("_", r"\_")
        dataset_disp = k[3]
        l3 = keyed[k].get(3, None)
        l5 = keyed[k].get(5, None)
        l3_s = f"{l3:.2f}" if l3 is not None else "--"
        l5_s = f"{l5:.2f}" if l5 is not None else "--"
        lines.append(f"{phase_disp} & {model_disp} & {dataset_disp} & {l3_s} & {l5_s} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    test_split = json.loads(TEST_SPLIT.read_text())

    entries = []
    t0 = time.time()
    # L3 cells
    for base in headline_l3_bases():
        run_dir = ROOT / "runs" / "final" / base
        spec_dataset = base.rsplit("_", 1)[-1]
        idxs = test_split["datasets"][spec_dataset]["indices"]
        e = _process(base, 3, run_dir, dev, idxs, args.skip_existing)
        if e:
            entries.append(e)
    # L5 cells
    for base in headline_l5_bases():
        run_dir = ROOT / "runs" / "final" / base
        e = _process(base, 5, run_dir, dev, None, args.skip_existing)
        if e:
            entries.append(e)

    tex = _emit_latex(entries)
    TEX_OUT.write_text(tex)

    summary = {
        "n_cells": len(entries),
        "n_bins": N_BINS,
        "tex_table": str(TEX_OUT.relative_to(ROOT)).replace("\\", "/"),
        "mean_ece_l3_pp": float(np.mean([e["ece_pp"] for e in entries if e["level"] == 3])) if any(e["level"] == 3 for e in entries) else None,
        "mean_ece_l5_pp": float(np.mean([e["ece_pp"] for e in entries if e["level"] == 5])) if any(e["level"] == 5 for e in entries) else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    INDEX.write_text(json.dumps({"summary": summary, "entries": entries}, indent=2, sort_keys=True))
    print(f"\n[calibration] wrote {len(entries)} entries; mean ECE L3={summary['mean_ece_l3_pp']:.2f}pp L5={summary['mean_ece_l5_pp']:.2f}pp; elapsed={summary['elapsed_s']}s")
    print(f"[calibration] LaTeX table -> {TEX_OUT}")


if __name__ == "__main__":
    main()
