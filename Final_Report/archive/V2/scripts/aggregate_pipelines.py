"""Aggregate per-pipeline PSNR / SSIM and per-cell accuracy from Final_Exp.json.

The Final_Exp.json file (schema_version=3) already carries per-cell PSNR/SSIM
values computed during dashboard build. Same-pipeline cells share identical
values by construction (PSNR/SSIM depend only on the degradation operator and
the dataset, not on the model), so we collapse along the model/treatment/seed
axes and emit one row per unique pipeline (dataset, phase, level, axis).

Outputs (all under Final_Report/data/):
  image_quality/_index.csv  - long-form per-pipeline (PSNR, SSIM)
  tables/per_cell.csv       - flattened per-cell accuracy joined with PSNR/SSIM
  tables/pipelines.json     - the same long-form per-pipeline data as JSON
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "Final_Exp.json"
OUT_DIR = ROOT / "Final_Report" / "data"
IQ_DIR = OUT_DIR / "image_quality"
TBL_DIR = OUT_DIR / "tables"
IQ_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    with SRC.open("r", encoding="utf-8") as f:
        j = json.load(f)
    rows = j["rows"]

    # ------------------------------------------------------------------
    # Per-pipeline aggregation. A pipeline is (dataset, phase, level, axis).
    # Phase D shares pipelines with Phase B1: the regularization treatment
    # changes training, not the degradation operator. We document that by
    # mapping Phase D -> Phase B for the pipeline key when the level
    # parameters match (they do, by construction).
    # ------------------------------------------------------------------
    pipelines = defaultdict(list)
    for r in rows:
        psnr = r.get("psnr_mean")
        ssim = r.get("ssim_mean")
        if psnr is None or ssim is None:
            continue
        phase = r["phase"]
        # Collapse D into B for image-quality purposes, but keep B2 and B2nr
        # separate because they pin saturation=0 and noise=0.
        pipeline_phase = "B" if phase == "D" else phase
        key = (
            r["dataset"],
            pipeline_phase,
            r["level"],
            r["axis"],
        )
        pipelines[key].append(
            {
                "psnr_mean": r["psnr_mean"],
                "psnr_std": r["psnr_std"],
                "ssim_mean": r["ssim_mean"],
                "ssim_std": r["ssim_std"],
                "tag": r["tag"],
            }
        )

    pipeline_rows = []
    for (dataset, phase, level, axis), members in sorted(pipelines.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2] or 0, kv[0][3] or "")):
        # Deterministic representative (first by tag).
        rep = sorted(members, key=lambda m: m["tag"])[0]
        # Within a pipeline the PSNR/SSIM should be identical (or differ only
        # in floating-point noise). Verify and report max deviation.
        psnrs = [m["psnr_mean"] for m in members]
        ssims = [m["ssim_mean"] for m in members]
        max_psnr_dev = max(psnrs) - min(psnrs)
        max_ssim_dev = max(ssims) - min(ssims)
        pipeline_rows.append(
            {
                "dataset": dataset,
                "phase": phase,
                "level": level,
                "axis": axis or "",
                "psnr_mean": rep["psnr_mean"],
                "psnr_std": rep["psnr_std"],
                "ssim_mean": rep["ssim_mean"],
                "ssim_std": rep["ssim_std"],
                "n_cells": len(members),
                "max_psnr_deviation": max_psnr_dev,
                "max_ssim_deviation": max_ssim_dev,
            }
        )

    index_csv = IQ_DIR / "_index.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pipeline_rows[0].keys()))
        w.writeheader()
        w.writerows(pipeline_rows)

    with (TBL_DIR / "pipelines.json").open("w", encoding="utf-8") as f:
        json.dump(pipeline_rows, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Per-cell flat table joined with PSNR/SSIM (used by the §VI tables).
    # ------------------------------------------------------------------
    per_cell_csv = TBL_DIR / "per_cell.csv"
    fields = [
        "tag",
        "phase",
        "model",
        "dataset",
        "level",
        "axis",
        "treatment",
        "seed",
        "val_acc",
        "val_acc_mean",
        "val_acc_std",
        "n_seeds",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "epochs_run",
        "runtime_s",
        "status",
    ]
    with per_cell_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "tag": r["tag"],
                    "phase": r["phase"],
                    "model": r["model"],
                    "dataset": r["dataset"],
                    "level": r.get("level"),
                    "axis": r.get("axis") or "",
                    "treatment": r.get("treatment") or "",
                    "seed": r.get("seed"),
                    "val_acc": r.get("val_acc"),
                    "val_acc_mean": r.get("val_acc_mean"),
                    "val_acc_std": r.get("val_acc_std"),
                    "n_seeds": len(r.get("seeds_observed", []) or []),
                    "psnr_mean": r.get("psnr_mean"),
                    "psnr_std": r.get("psnr_std"),
                    "ssim_mean": r.get("ssim_mean"),
                    "ssim_std": r.get("ssim_std"),
                    "epochs_run": r.get("epochs_run"),
                    "runtime_s": r.get("runtime_s"),
                    "status": r.get("status"),
                }
            )

    n_pipelines = len(pipeline_rows)
    max_dev_p = max(p["max_psnr_deviation"] for p in pipeline_rows)
    max_dev_s = max(p["max_ssim_deviation"] for p in pipeline_rows)
    print(f"[aggregate_pipelines] {len(rows)} cells -> {n_pipelines} unique pipelines")
    print(f"[aggregate_pipelines] max PSNR deviation across same-pipeline cells: {max_dev_p:.6f} dB")
    print(f"[aggregate_pipelines] max SSIM deviation across same-pipeline cells: {max_dev_s:.6f}")
    print(f"[aggregate_pipelines] wrote {index_csv}")
    print(f"[aggregate_pipelines] wrote {per_cell_csv}")


if __name__ == "__main__":
    main()
