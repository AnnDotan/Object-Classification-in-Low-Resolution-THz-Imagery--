"""Regenerate `Final_Exp.md` from the on-disk state of `runs/final/`.

The 186/276-cell tracker is generated, never hand-edited. This script scans
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
    EXPECTED_COUNTS_WITH_ALL,
    EXPECTED_COUNTS_WITH_D,
    EXPECTED_TOTAL,
    EXPECTED_TOTAL_WITH_ALL,
    EXPECTED_TOTAL_WITH_D,
    all_cells,
    phase_b2_present_on_disk,
    phase_b2nr_present_on_disk,
    phase_c2_present_on_disk,
    phase_d_present_on_disk,
)
from src.experiments.run_status import (  # noqa: E402
    QUARANTINE_REASON,
    demote_v2_pending,
    detect_status,
    is_quarantined,
    is_v2_affected,
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
    """Detect status, then apply US-014 quarantine override for TransNeXt
    AND the US-019B v2-pending demoter for the 90 v2-affected cells whose
    on-disk metrics.json predates PIPELINE_VERSION=2."""
    if is_quarantined(spec.model):
        return "Deferred"
    status = detect_status(spec.tag, runs_root, metrics)
    return demote_v2_pending(status, metrics, spec.phase, spec.axis)


def _fmt_status_md(status: str) -> str:
    """Render status for the Markdown table; Deferred rows annotate the reason."""
    if status == "Deferred":
        return f"Deferred — {QUARANTINE_REASON}"
    return status


def _display_payload(spec, status: str, m: Optional[dict], q: Optional[dict]):
    """Hide v1-vintage metrics + image_quality when the row was demoted to
    Pending by `demote_v2_pending`. Keeps Best Val Acc / PSNR / SSIM /
    Started / Duration columns as `EM_DASH` for the 90 v2-affected cells
    awaiting US-020/021/022 re-run."""
    if status == "Pending" and is_v2_affected(spec.phase, spec.axis):
        return None, None
    return m, q


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
        dm, dq = _display_payload(spec, status, m, q)
        rows.append(_row(i, [
            spec.model, spec.dataset, f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(dm), _fmt_psnr(dq), _fmt_ssim(dq),
            _fmt_started(rd, dm), _fmt_duration(rd, dm),
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
        dm, dq = _display_payload(spec, status, m, q)
        rows.append(_row(i, [
            spec.model, spec.dataset, LEVEL_NAMES[spec.level],
            f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(dm), _fmt_psnr(dq), _fmt_ssim(dq),
            _fmt_started(rd, dm), _fmt_duration(rd, dm),
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
        dm, dq = _display_payload(spec, status, m, q)
        rows.append(_row(i, [
            spec.model, spec.dataset, LEVEL_NAMES[spec.level], spec.axis,
            f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(dm), _fmt_psnr(dq), _fmt_ssim(dq),
            _fmt_started(rd, dm), _fmt_duration(rd, dm),
        ]))
    return "\n".join(rows), counts


def _phase_d_table(cells: list, runs_root: Path, start_idx: int) -> tuple[str, dict]:
    """Render the Phase D — Regularization Sweep table (US-026).

    Columns add a `Treatment` after Level so the T1/T2/T3 dimension is
    visible alongside the standard Phase B metrics. Same status/PSNR/SSIM
    formatting as Phase B.
    """
    rows = [
        "| # | Model | Dataset | Level | Treatment | Tag | Status | Best Val Acc | PSNR | SSIM | Started | Duration |",
        "|---|-------|---------|-------|-----------|-----|--------|--------------|------|------|---------|----------|",
    ]
    counts = _empty_counts()
    for i, spec in enumerate(cells, start=start_idx):
        m = read_metrics(spec.tag, runs_root)
        q = read_image_quality(spec.tag, runs_root)
        status = _resolve_status(spec, runs_root, m)
        counts[status] += 1
        rd = runs_root / spec.tag
        dm, dq = _display_payload(spec, status, m, q)
        rows.append(_row(i, [
            spec.model, spec.dataset, LEVEL_NAMES[spec.level],
            spec.treatment or EM_DASH,
            f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(dm), _fmt_psnr(dq), _fmt_ssim(dq),
            _fmt_started(rd, dm), _fmt_duration(rd, dm),
        ]))
    return "\n".join(rows), counts


def _phase_b2_table(cells: list, runs_root: Path, start_idx: int) -> tuple[str, dict]:
    """Render Phase B2 — THz-protocol simplification table (US-038)."""
    rows = [
        "| # | Model | Dataset | Level | Treatment | Tag | Status | Best Val Acc | PSNR | SSIM | Started | Duration |",
        "|---|-------|---------|-------|-----------|-----|--------|--------------|------|------|---------|----------|",
    ]
    counts = _empty_counts()
    for i, spec in enumerate(cells, start=start_idx):
        m = read_metrics(spec.tag, runs_root)
        q = read_image_quality(spec.tag, runs_root)
        status = _resolve_status(spec, runs_root, m)
        counts[status] += 1
        rd = runs_root / spec.tag
        dm, dq = _display_payload(spec, status, m, q)
        rows.append(_row(i, [
            spec.model, spec.dataset, LEVEL_NAMES[spec.level],
            spec.treatment or EM_DASH,
            f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(dm), _fmt_psnr(dq), _fmt_ssim(dq),
            _fmt_started(rd, dm), _fmt_duration(rd, dm),
        ]))
    return "\n".join(rows), counts


def _phase_b2nr_table(cells: list, runs_root: Path, start_idx: int) -> tuple[str, dict]:
    """Render Phase B2-nr — no-regularization L3 arm table (US-038)."""
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
        dm, dq = _display_payload(spec, status, m, q)
        rows.append(_row(i, [
            spec.model, spec.dataset, LEVEL_NAMES[spec.level],
            f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(dm), _fmt_psnr(dq), _fmt_ssim(dq),
            _fmt_started(rd, dm), _fmt_duration(rd, dm),
        ]))
    return "\n".join(rows), counts


def _phase_c2_table(cells: list, runs_root: Path, start_idx: int) -> tuple[str, dict]:
    """Render Phase C2 — THz-protocol single-axis isolation table (US-038)."""
    rows = [
        "| # | Model | Dataset | Level | Axis | Treatment | Tag | Status | Best Val Acc | PSNR | SSIM | Started | Duration |",
        "|---|-------|---------|-------|------|-----------|-----|--------|--------------|------|------|---------|----------|",
    ]
    counts = _empty_counts()
    for i, spec in enumerate(cells, start=start_idx):
        m = read_metrics(spec.tag, runs_root)
        q = read_image_quality(spec.tag, runs_root)
        status = _resolve_status(spec, runs_root, m)
        counts[status] += 1
        rd = runs_root / spec.tag
        dm, dq = _display_payload(spec, status, m, q)
        rows.append(_row(i, [
            spec.model, spec.dataset, LEVEL_NAMES[spec.level],
            spec.axis or EM_DASH,
            spec.treatment or EM_DASH,
            f"`{spec.tag}`", _fmt_status_md(status),
            _fmt_acc(dm), _fmt_psnr(dq), _fmt_ssim(dq),
            _fmt_started(rd, dm), _fmt_duration(rd, dm),
        ]))
    return "\n".join(rows), counts


# Backward-compat alias (US-029.5): the canonical helper now lives in
# `src.experiments.cells.phase_d_present_on_disk`. This private wrapper is
# preserved for any internal call sites that already reference the original
# leading-underscore name; new code should import directly from `cells`.
def _phase_d_present_on_disk(runs_root: Path) -> bool:
    return phase_d_present_on_disk(runs_root)


def render(runs_root: Path, today: Optional[_dt.date] = None) -> str:
    """Return the full Markdown content of `Final_Exp.md`.

    Phase D rendering (US-026): when at least one `runs/final/final_D_*`
    directory exists, the 90-cell Phase D table is appended after Phase C
    and totals reflect 276 cells. Empty disk state preserves the historical
    186-cell view byte-identical.

    Phase B2 / B2nr / C2 rendering (US-038): each is gated by its
    `phase_*_present_on_disk` helper so the legacy 186- and 276-cell views
    stay byte-identical until the corresponding phase launches.
    """
    include_phase_d   = _phase_d_present_on_disk(runs_root)
    include_phase_b2  = phase_b2_present_on_disk(runs_root)
    include_phase_b2nr = phase_b2nr_present_on_disk(runs_root)
    include_phase_c2  = phase_c2_present_on_disk(runs_root)

    expected_counts = dict(EXPECTED_COUNTS)
    if include_phase_b2:
        expected_counts["B2"] = EXPECTED_COUNTS_WITH_ALL["B2"]
    if include_phase_b2nr:
        expected_counts["B2nr"] = EXPECTED_COUNTS_WITH_ALL["B2nr"]
    if include_phase_c2:
        expected_counts["C2"] = EXPECTED_COUNTS_WITH_ALL["C2"]
    if include_phase_d:
        expected_counts["D"] = EXPECTED_COUNTS_WITH_ALL["D"]
    expected_total = sum(expected_counts.values())

    by_phase: dict[str, list] = {
        "A": [], "B": [], "B2": [], "B2nr": [], "C": [], "C2": [], "D": [],
    }
    for c in all_cells(
        include_phase_d=include_phase_d,
        include_phase_b2=include_phase_b2,
        include_phase_b2nr=include_phase_b2nr,
        include_phase_c2=include_phase_c2,
    ):
        by_phase[c.phase].append(c)
    assert sum(len(v) for v in by_phase.values()) == expected_total

    a_table, a_counts = _phase_a_table(by_phase["A"], runs_root, 1)
    b_start = 1 + expected_counts["A"]
    b_table, b_counts = _phase_b_table(by_phase["B"], runs_root, b_start)

    next_start = b_start + expected_counts["B"]

    phase_b2_section = ""
    phase_b2_summary_line = ""
    if include_phase_b2:
        b2_table, b2_counts = _phase_b2_table(by_phase["B2"], runs_root, next_start)
        next_start += expected_counts["B2"]
        phase_b2_summary_line = (
            f"\n- Phase B2 (THz protocol + T3): {b2_counts['Complete']}/{expected_counts['B2']}"
        )
        phase_b2_section = f"""
## Phase B2 — THz Protocol + T3 ({expected_counts['B2']})

US-038 (2026-05-26): duplicate of Phase B with `saturation = 0.0` +
`gaussian_noise_std = 0.0` DegradeConfig overrides AND T3 regularization
layered on top. Rationale: [`docs/phase_b2.md`](../docs/phase_b2.md).

{b2_table}
"""
    else:
        b2_counts = _empty_counts()

    phase_b2nr_section = ""
    phase_b2nr_summary_line = ""
    if include_phase_b2nr:
        b2nr_table, b2nr_counts = _phase_b2nr_table(by_phase["B2nr"], runs_root, next_start)
        next_start += expected_counts["B2nr"]
        phase_b2nr_summary_line = (
            f"\n- Phase B2-nr (B2 protocol, no T3): "
            f"{b2nr_counts['Complete']}/{expected_counts['B2nr']}"
        )
        phase_b2nr_section = f"""
## Phase B2-no-regularization — L3-only ({expected_counts['B2nr']})

US-038 (2026-05-26): same DegradeConfig as Phase B2 L3 but WITHOUT T3
deltas. Provides the pure-protocol Δ_B1→B2nr at L3.

{b2nr_table}
"""
    else:
        b2nr_counts = _empty_counts()

    c_table, c_counts = _phase_c_table(by_phase["C"], runs_root, next_start)
    next_start += expected_counts["C"]

    phase_c2_section = ""
    phase_c2_summary_line = ""
    if include_phase_c2:
        c2_table, c2_counts = _phase_c2_table(by_phase["C2"], runs_root, next_start)
        next_start += expected_counts["C2"]
        phase_c2_summary_line = (
            f"\n- Phase C2 (THz axis attribution): "
            f"{c2_counts['Complete']}/{expected_counts['C2']}"
        )
        phase_c2_section = f"""
## Phase C2 — THz Single-Axis Attribution ({expected_counts['C2']})

US-038 (2026-05-26): legacy Phase C single-axis isolation under the B2
protocol (sat = 0 + noise = 0) + T3 regularization; axes restricted to
{{resolution, blur, salt_pepper}}. Rationale: [`docs/phase_c2.md`](../docs/phase_c2.md).

{c2_table}
"""
    else:
        c2_counts = _empty_counts()

    phase_d_section = ""
    phase_d_summary_line = ""
    if include_phase_d:
        d_table, d_counts = _phase_d_table(by_phase["D"], runs_root, next_start)
        next_start += expected_counts["D"]
        phase_d_summary_line = (
            f"\n- Phase D (Regularization): {d_counts['Complete']}/{expected_counts['D']}"
        )
        phase_d_section = f"""
## Phase D — Regularization Sweep ({expected_counts['D']})

US-026 (2026-05-23): duplicate of Phase B with regularization treatments layered
on top of the L3-Optuna baselines. Treatments:
- **T1** — architectural dropout (`dropout=0.2` for CNNs, `drop_path_rate=0.2` for TransNeXt)
- **T2** — label-mixing (`mixup_alpha=0.2`, cutmix off)
- **T3** — combo (T1 ∪ T2 ∪ `cutmix_alpha=1.0`)

Phase D directories use `final_D_*` tags so they cannot collide with the 30
existing Phase B runs. Each Phase D row may be compared against its
`final_B_L<l>_<m>_<d>` sibling for the regularization delta at that
(model, dataset, level).

{d_table}
"""
    else:
        d_counts = _empty_counts()

    total_complete = (
        a_counts["Complete"] + b_counts["Complete"] + c_counts["Complete"]
        + b2_counts["Complete"] + b2nr_counts["Complete"]
        + c2_counts["Complete"] + d_counts["Complete"]
    )
    total_deferred = (
        a_counts["Deferred"] + b_counts["Deferred"] + c_counts["Deferred"]
        + b2_counts["Deferred"] + b2nr_counts["Deferred"]
        + c2_counts["Deferred"] + d_counts["Deferred"]
    )

    today = today or _dt.date.today()
    header_title = f"# Final Experiment Matrix — {expected_total} Runs"
    return f"""{header_title}

Master tracker for the {expected_total}-cell final research campaign (186 v2 + 90 Phase D when present).
Source of truth: filesystem scan of `runs/final/`.
Regenerated by `scripts/update_final_exp.py` after every run; do not hand-edit row data.

## Status Summary
- Phase A (Clean): {a_counts['Complete']}/{expected_counts['A']}
- Phase B (Combined): {b_counts['Complete']}/{expected_counts['B']}{phase_b2_summary_line}{phase_b2nr_summary_line}
- Phase C (Isolation): {c_counts['Complete']}/{expected_counts['C']}{phase_c2_summary_line}{phase_d_summary_line}
- **Total: {total_complete}/{expected_total}**
- Last updated: {today.isoformat()}

> TransNeXt-tiny replaced TransNeXt-small across the matrix on 2026-05-14 (US-016): ~28M params capacity-match to ResNet50 (~25M) and frees ~310 GPU-h vs the small variant across the 62 TransNeXt rows. The earlier base→small swap (US-004, same day) is preserved in the history. All 62 TransNeXt rows train at 224×224 with full FT + differential LR (head 5e-4 / backbone 5e-5).

Models: `resnet50`, `densenet121`, `transnext_tiny` (Final-campaign default; full FT per US-006; swapped from transnext_small by US-016 on 2026-05-14, which itself swapped from transnext_base by US-004 on the same day).
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

## Phase A — Clean Baselines ({expected_counts['A']})

{a_table}

## Phase B — Combined Degradation ({expected_counts['B']})

{b_table}
{phase_b2_section}{phase_b2nr_section}
## Phase C — Single-Axis Isolation ({expected_counts['C']})

Phase C pins every axis at L1 (mild) values and sweeps only the named axis through L1→L5.
At L1, every isolation cell collapses to the Phase B L1 row for the same (model, dataset);
`run_all_phases.py --skip-existing` deduplicates these at runtime.

{c_table}
{phase_c2_section}{phase_d_section}
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
    # US-038: total reflects whichever opt-in phases are present on disk.
    n_cells = EXPECTED_TOTAL
    if phase_b2_present_on_disk(runs_root):
        n_cells += EXPECTED_COUNTS_WITH_ALL["B2"]
    if phase_b2nr_present_on_disk(runs_root):
        n_cells += EXPECTED_COUNTS_WITH_ALL["B2nr"]
    if phase_c2_present_on_disk(runs_root):
        n_cells += EXPECTED_COUNTS_WITH_ALL["C2"]
    if _phase_d_present_on_disk(runs_root):
        n_cells += EXPECTED_COUNTS_WITH_ALL["D"]
    print(f"[update_final_exp] wrote {out_path} ({n_cells} cells; "
          f"canonical = {EXPECTED_TOTAL_WITH_ALL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
