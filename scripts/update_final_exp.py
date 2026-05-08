"""Regenerate `Final_Exp.md` from the on-disk state of `runs/final/`.

The 186-cell tracker is generated, never hand-edited. This script scans
each `runs/final/<tag>/metrics.json` (and `image_quality.json` for
PSNR/SSIM, US-002) and rewrites `Final_Exp.md` in canonical order
defined by `src.experiments.matrix.build_final_matrix()`.

Idempotency: identical disk state -> byte-identical output. CI/hooks can
use `--check` to assert no drift.

CLI:
    python scripts/update_final_exp.py
    python scripts/update_final_exp.py --runs-root runs/final --out Final_Exp.md
    python scripts/update_final_exp.py --check        # exit 1 on drift
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import Optional

# Make the repo root importable when invoked as `python scripts/update_final_exp.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.cells import (  # noqa: E402
    EXPECTED_COUNTS,
    EXPECTED_TOTAL,
    all_cells,
)
from src.experiments.run_status import (  # noqa: E402
    QUARANTINE_REASON,
    detect_status,
    is_quarantined,
    read_image_quality,
    read_metrics,
)

LEVEL_NAMES: dict[int, str] = {
    1: "L1 Mild",
    2: "L2 Light",
    3: "L3 Moderate",
    4: "L4 Severe",
    5: "L5 Extreme",
}

EM_DASH = "—"


def _fmt_acc(metrics: Optional[dict]) -> str:
    if metrics is None:
        return EM_DASH
    for key in ("best_val_acc", "final_val_acc", "last_val_acc"):
        v = metrics.get(key)
        if isinstance(v, (int, float)) and v >= 0.0:
            return f"{v:.4f}"
    return EM_DASH


def _fmt_psnr(quality: Optional[dict]) -> str:
    if quality is None:
        return EM_DASH
    v = quality.get("psnr_mean")
    if not isinstance(v, (int, float)):
        return EM_DASH
    std = quality.get("psnr_std")
    if isinstance(std, (int, float)):
        return f"{v:.2f}±{std:.2f}"
    return f"{v:.2f}"


def _fmt_ssim(quality: Optional[dict]) -> str:
    if quality is None:
        return EM_DASH
    v = quality.get("ssim_mean")
    if not isinstance(v, (int, float)):
        return EM_DASH
    std = quality.get("ssim_std")
    if isinstance(std, (int, float)):
        return f"{v:.4f}±{std:.4f}"
    return f"{v:.4f}"


def _fmt_started(run_dir: Path, metrics: Optional[dict]) -> str:
    if metrics and isinstance(metrics.get("started_at"), str):
        return metrics["started_at"][:19].replace("T", " ")
    log = run_dir / "log.txt"
    if log.exists():
        ts = _dt.datetime.fromtimestamp(log.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M")
    if run_dir.exists():
        ts = _dt.datetime.fromtimestamp(run_dir.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M")
    return EM_DASH


def _fmt_duration(run_dir: Path, metrics: Optional[dict]) -> str:
    if metrics and isinstance(metrics.get("duration_sec"), (int, float)):
        secs = float(metrics["duration_sec"])
    else:
        log = run_dir / "log.txt"
        m_path = run_dir / "metrics.json"
        if not (log.exists() and m_path.exists()):
            return EM_DASH
        secs = m_path.stat().st_mtime - log.stat().st_mtime
        if secs <= 0:
            return EM_DASH
    if secs < 90:
        return f"{secs:.0f}s"
    mins = secs / 60.0
    if mins < 90:
        return f"{mins:.0f}m"
    return f"{mins / 60.0:.1f}h"


def _row(idx: int, columns: list[str]) -> str:
    return "| " + " | ".join([str(idx), *columns]) + " |"


def _empty_counts() -> dict:
    return {"Pending": 0, "Running": 0, "Complete": 0, "Failed": 0, "Deferred": 0}


def _resolve_status(spec, runs_root: Path, metrics: Optional[dict]) -> str:
    """Detect status, then apply US-014 quarantine override for TransNeXt."""
    if is_quarantined(spec.model):
        return "Deferred"
    return detect_status(spec.tag, runs_root, metrics)


def _fmt_status_md(status: str) -> str:
    """Render status for the Markdown table; Deferred rows annotate the reason."""
    if status == "Deferred":
        return f"Deferred — {QUARANTINE_REASON}"
    return status


def _phase_a_table(cells: list, runs_root: Path, start_idx: int) -> tuple[str, dict]:
    rows = [
        "| # | Model | Dataset | Tag | Status | Best Val Acc | PSNR | SSIM | Started | Duration |",
        "|---|-------|---------|-----|--------|--------------|------|------|---------|----------|",
    ]
    counts = _empty_counts()
    for i, spec in enumerate(cells, start=start_idx):
        m = read_metrics(spec.tag, runs_root)
        q = read_image_quality(spec.tag, runs_root)
        status = _resolve_status(spec, runs_root, m)
        counts[status] += 1
        rd = runs_root / spec.tag
        rows.append(_row(i, [
            spec.model, spec.dataset, f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(m), _fmt_psnr(q), _fmt_ssim(q),
            _fmt_started(rd, m), _fmt_duration(rd, m),
        ]))
    return "\n".join(rows), counts


def _phase_b_table(cells: list, runs_root: Path, start_idx: int) -> tuple[str, dict]:
    rows = [
        "| # | Model | Dataset | Level | Tag | Status | Best Val Acc | PSNR | SSIM | Started | Duration |",
        "|---|-------|---------|-------|-----|--------|--------------|------|------|---------|----------|",
    ]
    counts = _empty_counts()
    for i, spec in enumerate(cells, start=start_idx):
        m = read_metrics(spec.tag, runs_root)
        q = read_image_quality(spec.tag, runs_root)
        status = _resolve_status(spec, runs_root, m)
        counts[status] += 1
        rd = runs_root / spec.tag
        rows.append(_row(i, [
            spec.model, spec.dataset, LEVEL_NAMES[spec.level],
            f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(m), _fmt_psnr(q), _fmt_ssim(q),
            _fmt_started(rd, m), _fmt_duration(rd, m),
        ]))
    return "\n".join(rows), counts


def _phase_c_table(cells: list, runs_root: Path, start_idx: int) -> tuple[str, dict]:
    rows = [
        "| # | Model | Dataset | Level | Axis | Tag | Status | Best Val Acc | PSNR | SSIM | Started | Duration |",
        "|---|-------|---------|-------|------|-----|--------|--------------|------|------|---------|----------|",
    ]
    counts = _empty_counts()
    for i, spec in enumerate(cells, start=start_idx):
        m = read_metrics(spec.tag, runs_root)
        q = read_image_quality(spec.tag, runs_root)
        status = _resolve_status(spec, runs_root, m)
        counts[status] += 1
        rd = runs_root / spec.tag
        rows.append(_row(i, [
            spec.model, spec.dataset, LEVEL_NAMES[spec.level], spec.axis,
            f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(m), _fmt_psnr(q), _fmt_ssim(q),
            _fmt_started(rd, m), _fmt_duration(rd, m),
        ]))
    return "\n".join(rows), counts


def render(runs_root: Path, today: Optional[_dt.date] = None) -> str:
    """Return the full Markdown content of `Final_Exp.md`."""
    by_phase: dict[str, list] = {"A": [], "B": [], "C": []}
    for c in all_cells():
        by_phase[c.phase].append(c)
    assert sum(len(v) for v in by_phase.values()) == EXPECTED_TOTAL

    a_table, a_counts = _phase_a_table(by_phase["A"], runs_root, 1)
    b_start = 1 + EXPECTED_COUNTS["A"]
    b_table, b_counts = _phase_b_table(by_phase["B"], runs_root, b_start)
    c_start = b_start + EXPECTED_COUNTS["B"]
    c_table, c_counts = _phase_c_table(by_phase["C"], runs_root, c_start)

    total_complete = a_counts["Complete"] + b_counts["Complete"] + c_counts["Complete"]
    total_deferred = a_counts["Deferred"] + b_counts["Deferred"] + c_counts["Deferred"]
    today = today or _dt.date.today()

    return f"""# Final Experiment Matrix — 186 Runs

Master tracker for the 186-cell final research campaign.
Source of truth: filesystem scan of `runs/final/`.
Regenerated by `scripts/update_final_exp.py` after every run; do not hand-edit row data.

## Status Summary
- Phase A (Clean): {a_counts['Complete']}/{EXPECTED_COUNTS['A']}
- Phase B (Combined): {b_counts['Complete']}/{EXPECTED_COUNTS['B']}
- Phase C (Isolation): {c_counts['Complete']}/{EXPECTED_COUNTS['C']}
- Deferred (TransNeXt — {QUARANTINE_REASON}): {total_deferred}/{EXPECTED_TOTAL}
- **Total: {total_complete}/{EXPECTED_TOTAL}**
- Last updated: {today.isoformat()}

> TransNeXt rows are quarantined per `docs/prds/PHASE_B_VISUAL_CORE.md` US-014 and excluded from execution. The deferral reason "{QUARANTINE_REASON}" reflects two prerequisites: (1) a Blackwell-tier GPU online (RTX 5070 12 GB, migration packaged 2026-05-08) AND (2) the TransNeXt wrapper refactored to consume the THz pipeline at native (non-224×224) resolution. They remain in the matrix so the 186 denominator is preserved.

Models: `resnet50`, `densenet121`, `transnext_base` (Final-campaign default; full FT per US-006).
Datasets: `cifar10`, `mnist`.
Levels: L1 Mild, L2 Light, L3 Moderate, L4 Severe, L5 Extreme.
Axes (Phase C): `resolution`, `noise`, `blur`, `saturation`, `salt_pepper`.

### `metrics.json` Schema (Final-campaign rows)

Every row's `runs/final/<tag>/metrics.json` carries the following keys (US-008 + US-012):

| Key | Type | Source |
|---|---|---|
| `best_val_acc` / `best_epoch` | float / int | `LegacyJSONMetricsCallback` |
| `last_val_acc` / `last_train_acc` / `last_val_loss` / `last_train_loss` | float | `LegacyJSONMetricsCallback` |
| `epochs_run` | int | `LegacyJSONMetricsCallback` |
| `wandb_run_id` / `wandb_entity` / `wandb_project` | string \\| null | `LegacyJSONMetricsCallback` (US-012) — `null` when offline |
| `hparams` | object | `run_systematic._merge_metadata_into_metrics_json` (US-008) |
| `hparams_source` | object — `study_name`, `best_value`, `n_trials_completed`, `priors_file_hash` | US-008 |
| `cell_tag` / `phase` / `level` / `axis` / `model` / `dataset` | strings / int / null | US-008 |

PSNR/SSIM live in a sibling `image_quality.json` written by
`src/tools/measure_image_quality.py` (US-002) when invoked per-cell.

## Phase A — Clean Baselines ({EXPECTED_COUNTS['A']})

{a_table}

## Phase B — Combined Degradation ({EXPECTED_COUNTS['B']})

{b_table}

## Phase C — Single-Axis Isolation ({EXPECTED_COUNTS['C']})

Phase C pins every axis at L1 (mild) values and sweeps only the named axis through L1→L5.
At L1, every isolation cell collapses to the Phase B L1 row for the same (model, dataset);
`run_all_phases.py --skip-existing` deduplicates these at runtime.

{c_table}

## How this file is updated

- Every Lightning run writes a `metrics.json` and `metrics.csv` under `runs/final/<tag>/`.
- `scripts/update_final_exp.py` (invoked by `DashboardRefreshCallback` on `on_train_end`) scans that tree and rewrites this file in place.
- Manual refresh: `python scripts/update_final_exp.py` (idempotent).
- Status values: `Pending` / `Running` / `Complete` / `Failed`.
"""


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Regenerate Final_Exp.md from runs/final/.")
    p.add_argument("--runs-root", default="runs/final",
                   help="Root directory holding per-cell metrics.json files.")
    p.add_argument("--out", default="Final_Exp.md",
                   help="Path to the tracker Markdown file.")
    p.add_argument("--check", action="store_true",
                   help="Exit non-zero if regenerated content differs from --out.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    runs_root = Path(args.runs_root)
    out_path = Path(args.out)

    new_content = render(runs_root)

    if args.check:
        if not out_path.exists():
            print(f"[update_final_exp] {out_path} does not exist", file=sys.stderr)
            return 1
        cur = out_path.read_text(encoding="utf-8")
        if cur != new_content:
            print(f"[update_final_exp] DRIFT: {out_path} is out of date", file=sys.stderr)
            return 1
        print(f"[update_final_exp] OK: {out_path} matches disk state")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(new_content, encoding="utf-8")
    print(f"[update_final_exp] wrote {out_path} ({EXPECTED_TOTAL} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
