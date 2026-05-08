"""Shared run-status detection for the 186-cell campaign.

Single source of truth for `Pending | Running | Complete | Failed` so the
HTML dashboard (`src/tools/build_final_dashboard.py`) and the Markdown
tracker updater (`scripts/update_final_exp.py`) cannot drift.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

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


def is_quarantined(model: str) -> bool:
    """Return True if `model` is currently deferred from execution."""
    return "transnext" in (model or "").lower()


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


def detect_status(
    tag: str,
    runs_root: Path = RUNS_ROOT_DEFAULT,
    metrics: Optional[dict] = None,
) -> str:
    """Return one of `Pending | Running | Complete | Failed`.

    Precedence:
      1. INTERRUPTED sentinel (US-016) -> Failed regardless of any partial
         metrics.json the trainer flushed before the SIGINT.
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
