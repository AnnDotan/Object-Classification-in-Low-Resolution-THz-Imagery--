"""Shared run-status detection for the 186-cell campaign.

Single source of truth for `Pending | Running | Complete | Failed` so the
HTML dashboard (`src/tools/build_final_dashboard.py`) and the Markdown
tracker updater (`scripts/update_final_exp.py`) cannot drift.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.data.degradation_levels import PIPELINE_VERSION

RUNS_ROOT_DEFAULT = Path("runs/final")

# Keys in metrics.json that, when present and non-negative, mean the run
# reached a published validation accuracy and is considered Complete.
_COMPLETE_KEYS: tuple[str, ...] = ("final_val_acc", "best_val_acc", "last_val_acc")

# Substrings in log.txt (lower-cased) that mark a run as Failed when no
# completed metrics are present.
_FAILURE_MARKERS: tuple[str, ...] = ("traceback", "[error]")

# Quarantine policy (US-014, docs/prds/PHASE_B_VISUAL_CORE.md).
# TransNeXt is deferred until a Blackwell-tier GPU is online AND the model
# wrapper is refactored to consume the THz pipeline at native (non-224×224)
# resolution. Rows render as a separate "Deferred" status so the 186-cell
# denominator stays intact while excluding TransNeXt from any execution-driving
# iteration. Label updated 2026-05-08 for the RTX 5070 migration package.
QUARANTINE_REASON = "Awaiting Native-Resolution Refactor"

# Filename of the SIGINT sentinel dropped by run_all_phases when a cell is
# interrupted mid-train (US-016 + US-019). Presence -> the cell is Failed
# regardless of any incomplete metrics.json the trainer wrote before exiting.
INTERRUPTED_SENTINEL = "INTERRUPTED"

# Filename of the §6.4 second-pass failure sentinel written by the ralph
# driver (scripts/run_ralph_loop.py:run_remediate) when the retry-after-
# NEEDS_FULL_FT itself fails the pathology guard. Body is `second_failure:
# <verdict>` (e.g. `second_failure:overfitting`). Presence -> the cell is
# Failed and quarantined; the original metrics.json is preserved for audit
# but the row should NOT count as Complete in the Final_Exp counts.
# (Iteration 12, 2026-05-15 — wired into detect_status precedence + read by
# build_final_exp_json for quarantine fields.)
QUARANTINED_AFTER_RETRY_SENTINEL = "QUARANTINED_AFTER_RETRY"


def read_quarantine_sentinel(
    tag: str, runs_root: Optional[Path] = None
) -> Optional[str]:
    """Return the body of the QUARANTINED_AFTER_RETRY sentinel (e.g.
    `second_failure:overfitting`) for `tag`, or None if absent.

    Used by the Final_Exp aggregator to flag rows that the §6.4 retry path
    permanently quarantined. The sentinel body documents the verdict so the
    dashboard can render `Failed · second_failure:overfitting` directly.
    """
    if runs_root is None:
        runs_root = RUNS_ROOT_DEFAULT
    p = runs_root / tag / QUARANTINED_AFTER_RETRY_SENTINEL
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").strip() or "second_failure"
    except OSError:
        return None


def is_quarantined(model: str) -> bool:
    """Return True if `model` is currently deferred from execution.

    V3 quarantine lift (ratified 2026-05-12): the predicate is now a permanent
    no-op. Pre-V3 it returned True for `transnext_*` so the runner skipped
    those cells while the project waited for Blackwell hardware. The hardware
    arrived (RTX 5070, sm_120) and TransNeXt cells are now part of the active
    campaign, so this returns False unconditionally. Kept as a function (vs
    deleted) so the call sites in `tune_all.py`, `scripts/update_final_exp.py`,
    and `src/tools/build_final_exp_json.py` continue to work without edits —
    they just stop quarantining anything.
    """
    return False


def read_metrics(tag: str, runs_root: Path = RUNS_ROOT_DEFAULT) -> Optional[dict]:
    """Return the parsed `metrics.json` for a cell or None if absent/unreadable."""
    path = runs_root / tag / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_image_quality(tag: str, runs_root: Path = RUNS_ROOT_DEFAULT) -> Optional[dict]:
    """Return parsed `image_quality.json` (PSNR/SSIM, US-002) or None."""
    path = runs_root / tag / "image_quality.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_v2_affected(phase: str, axis: Optional[str]) -> bool:
    """True iff a cell's v1 metrics were invalidated by the
    `PIPELINE_VERSION=2` noise/S&P pre-upsample move (US-017). Covers the
    90 v2-affected cells:
      - Phase B (all 30; noise_std>0 and salt_pepper>0 at every level)
      - Phase C `noise` axis (30)
      - Phase C `salt_pepper` axis (30)
    The remaining 96 cells (Phase A clean + Phase C resolution/blur/
    saturation) are unaffected because US-017 moved steps that are
    no-ops at their identity values.
    """
    if phase == "B":
        return True
    if phase == "C" and axis in ("noise", "salt_pepper"):
        return True
    return False


def demote_v2_pending(
    status: str,
    metrics: Optional[dict],
    phase: str,
    axis: Optional[str],
) -> str:
    """Dashboard-only status override: demote `Complete` to `Pending` when
    the cell is in the v2-affected set AND its `metrics.json` predates
    `PIPELINE_VERSION=2`. The on-disk v1 metrics are NOT deleted (that's
    US-020/021/022 pre-flight); this only changes what the operator sees
    in the dashboards until the v2 re-run lands.
    """
    if status != "Complete":
        return status
    if not is_v2_affected(phase, axis):
        return status
    pv = (metrics or {}).get("pipeline_version")
    if isinstance(pv, (int, float)) and int(pv) >= PIPELINE_VERSION:
        return status
    return "Pending"


def detect_status(
    tag: str,
    runs_root: Path = RUNS_ROOT_DEFAULT,
    metrics: Optional[dict] = None,
) -> str:
    """Return one of `Pending | Running | Complete | Failed`.

    Precedence:
      1. INTERRUPTED sentinel (US-016) -> Failed regardless of any partial
         metrics.json the trainer flushed before the SIGINT.
      1.5 QUARANTINED_AFTER_RETRY sentinel (US-005 §6.4, wired 2026-05-15) ->
         Failed regardless of any metrics.json the retry produced. The
         retry's best_val_acc is preserved for audit but a second-pass
         failure means the cell is permanently quarantined.
      2. Published val_acc in metrics.json -> Complete even if log.txt also
         contains a traceback (training succeeded; something downstream
         printed an error).
      3. Tracebacked log without metrics.json -> Failed.
      4. Run dir present, none of the above -> Running.
      5. No run dir -> Pending.
    """
    run_dir = runs_root / tag
    if (run_dir / INTERRUPTED_SENTINEL).exists():
        return "Failed"
    if (run_dir / QUARANTINED_AFTER_RETRY_SENTINEL).exists():
        return "Failed"

    if metrics is None:
        metrics = read_metrics(tag, runs_root)

    if metrics is not None:
        for key in _COMPLETE_KEYS:
            v = metrics.get(key)
            if isinstance(v, (int, float)) and v >= 0.0:
                return "Complete"

    if run_dir.exists():
        log = run_dir / "log.txt"
        if log.exists():
            try:
                tail = log.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                tail = ""
            if any(marker in tail for marker in _FAILURE_MARKERS):
                return "Failed"
        return "Running"
    return "Pending"
