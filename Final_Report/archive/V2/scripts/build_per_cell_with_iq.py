"""Build per_cell_with_iq.csv — PRD V2 §4.1 contract.

Joins the 402 canonical accuracy rows in artifacts/Final_Exp.json with the
per-pipeline PSNR / SSIM aggregation in Final_Report/data/image_quality/_index.csv.

Join keys (PRD V2 §4.1):
  - accuracy: tag (one row per canonical cell, seed=42)
  - image quality: (dataset, phase, level, axis)
  - Phase D rows reuse the Phase B1 image-quality pipeline (pixels identical;
    regularization is training-side). Footnote in the §4.1 spec documents the
    re-use; this script implements it by mapping D -> B on the IQ join key.

Output columns (PRD V2 §4.1 contract + multi-seed bonus columns):
  tag, model, dataset, phase, level, axis, regularization, seed,
  val_acc, psnr_mean, psnr_std, ssim_mean, ssim_std,
  val_acc_mean, val_acc_std, n_seeds

Phase A rows carry null PSNR / SSIM (identity pipeline; IQ undefined / perfect
by construction). All non-A rows are asserted to have populated IQ.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_JSON = ROOT / "artifacts" / "Final_Exp.json"
SRC_IQ = ROOT / "Final_Report" / "data" / "image_quality" / "_index.csv"
OUT_CSV = ROOT / "Final_Report" / "data" / "tables" / "per_cell_with_iq.csv"


FIELDS = [
    "tag",
    "model",
    "dataset",
    "phase",
    "level",
    "axis",
    "regularization",
    "seed",
    "val_acc",
    "psnr_mean",
    "psnr_std",
    "ssim_mean",
    "ssim_std",
    "val_acc_mean",
    "val_acc_std",
    "n_seeds",
]


def _load_iq_index(path: Path) -> dict[tuple[str, str, int | None, str], dict[str, float]]:
    iq: dict[tuple[str, str, int | None, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            level_raw = row["level"]
            level = int(level_raw) if level_raw not in (None, "") else None
            key = (row["dataset"], row["phase"], level, row["axis"] or "")
            iq[key] = {
                "psnr_mean": float(row["psnr_mean"]),
                "psnr_std": float(row["psnr_std"]),
                "ssim_mean": float(row["ssim_mean"]),
                "ssim_std": float(row["ssim_std"]),
            }
    return iq


def main() -> int:
    with SRC_JSON.open("r", encoding="utf-8") as f:
        j = json.load(f)
    rows = j["rows"]
    iq = _load_iq_index(SRC_IQ)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    missing_iq: list[str] = []
    out_rows: list[dict] = []
    for r in rows:
        phase = r["phase"]
        # Phase D shares the Phase B1 degradation pipeline. IQ key collapses D -> B.
        iq_phase = "B" if phase == "D" else phase
        level = r.get("level")
        axis = r.get("axis") or ""

        iq_row = iq.get((r["dataset"], iq_phase, level, axis))
        if iq_row is None and phase != "A":
            missing_iq.append(r["tag"])

        seeds = r.get("seeds_observed") or []
        out_rows.append(
            {
                "tag": r["tag"],
                "model": r["model"],
                "dataset": r["dataset"],
                "phase": phase,
                "level": level if level is not None else "",
                "axis": axis,
                "regularization": r.get("treatment") or "",
                "seed": r.get("seed"),
                "val_acc": r.get("val_acc"),
                "psnr_mean": iq_row["psnr_mean"] if iq_row else "",
                "psnr_std": iq_row["psnr_std"] if iq_row else "",
                "ssim_mean": iq_row["ssim_mean"] if iq_row else "",
                "ssim_std": iq_row["ssim_std"] if iq_row else "",
                "val_acc_mean": r.get("val_acc_mean"),
                "val_acc_std": r.get("val_acc_std"),
                "n_seeds": len(seeds),
            }
        )

    if missing_iq:
        raise SystemExit(
            f"[build_per_cell_with_iq] {len(missing_iq)} non-A cells missing IQ join: "
            f"{missing_iq[:5]}{'...' if len(missing_iq) > 5 else ''}"
        )

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    n_total = len(out_rows)
    n_with_iq = sum(1 for r in out_rows if r["psnr_mean"] != "")
    n_phase_a = sum(1 for r in out_rows if r["phase"] == "A")
    print(f"[build_per_cell_with_iq] wrote {OUT_CSV} ({n_total} rows)")
    print(f"[build_per_cell_with_iq] phase=A (null IQ by construction): {n_phase_a}")
    print(f"[build_per_cell_with_iq] non-A rows with IQ: {n_with_iq}")

    expected = 402
    if n_total != expected:
        raise SystemExit(
            f"[build_per_cell_with_iq] row count {n_total} != expected canonical {expected}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
