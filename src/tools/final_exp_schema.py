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
Status = Literal["Pending", "Running", "Complete", "Failed"]

PHASES: tuple[Phase, ...] = ("A", "B", "C")
STATUSES: tuple[Status, ...] = ("Pending", "Running", "Complete", "Failed")


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


class CountsDict(TypedDict):
    total: int
    pending: int
    running: int
    complete: int
    failed: int


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
