"""TypedDict schema for `artifacts/Final_Exp.json` (FINAL_EXP Dashboard, US-001).

Stable contract between `build_final_exp_json.py` (writer) and
`build_final_dashboard.py` / the browser (reader). Bumping
`SCHEMA_VERSION` is a breaking change for all consumers.

Field shapes match what is already on disk:
- `level` is `int | None` (1..5; None for Phase A clean), not a stringified
  ``"L1"``/``"clean"`` enum — the renderer formats for display.
- `axis` is `str | None` (one of `AXIS_KEYS`; None for Phase A and Phase B).
- `status` matches `src.experiments.run_status.detect_status` exactly:
  ``"Pending" | "Running" | "Complete" | "Failed"`` (note: ``"Complete"``,
  not ``"Done"``).

Privacy: this schema deliberately excludes checkpoint paths, weight URIs,
and host paths. Only metric scalars and config keys are carried.
"""
from __future__ import annotations

from typing import Literal, TypedDict

SCHEMA_VERSION: Literal[1] = 1

Phase = Literal["A", "B", "C"]
Status = Literal["Pending", "Running", "Complete", "Failed", "Deferred"]

PHASES: tuple[Phase, ...] = ("A", "B", "C")
# `"Deferred"` (US-014) is for cells whose model is quarantined pending
# hardware (e.g. TransNeXt). Deferred cells stay in the matrix so the 186
# denominator is preserved, but they are excluded from execution-driving
# iterators in `run_all_phases.py` and rendered with a distinct badge.
STATUSES: tuple[Status, ...] = (
    "Pending", "Running", "Complete", "Failed", "Deferred",
)


class ParamsDict(TypedDict):
    low_res: int
    blur_kernel: int
    blur_sigma: float
    noise_std: float
    salt_pepper: float
    saturation: float


class FinalExpRow(TypedDict):
    tag: str
    phase: Phase
    model: str
    dataset: str
    level: int | None
    axis: str | None
    params: ParamsDict
    status: Status
    val_acc: float | None
    val_loss: float | None
    epochs_run: int | None
    runtime_s: float | None
    started_at: str | None
    finished_at: str | None
    # US-014 quarantine: True when the cell is excluded from execution
    # (e.g. TransNeXt rows pending hardware). `quarantine_reason` is a
    # short human-readable string the dashboard renders in the badge.
    quarantined: bool
    quarantine_reason: str | None
    # US-017 Visual Core: relative path (from artifacts/Final_Exp.html) to
    # the side-by-side Original|Degraded preview PNG produced by
    # `src/tools/render_cell_thumbs.py`. None when the PNG is missing
    # (e.g. before the renderer has run, or in a torch-free environment).
    visual_core: str | None
    # PSNR/SSIM of the degraded sample vs the clean original, sourced from
    # `runs/final/<tag>/image_quality.json` (US-002). The aggregator copies
    # these onto the row so the dashboard can render them directly under
    # the Visual Core thumbnail. Null for Phase A clean baselines (where
    # the comparison is trivially identity) and for any cell whose
    # measurement step has not run yet.
    psnr_mean: float | None
    psnr_std: float | None
    ssim_mean: float | None
    ssim_std: float | None
    # US-018: True iff `runs/final/<tag>/history.json` exists. The dashboard
    # uses this to decide whether to render a 📈 indicator on the row and
    # whether clicking should attempt the lazy fetch. Aggregator never
    # opens the file; presence is checked via Path.exists().
    has_history: bool


class CountsDict(TypedDict):
    total: int
    pending: int
    running: int
    complete: int
    failed: int
    deferred: int


class FinalExpDoc(TypedDict):
    schema_version: Literal[1]
    generated_at: str
    rows: list[FinalExpRow]
    counts: CountsDict


__all__ = [
    "SCHEMA_VERSION",
    "PHASES",
    "STATUSES",
    "Phase",
    "Status",
    "ParamsDict",
    "FinalExpRow",
    "CountsDict",
    "FinalExpDoc",
]
