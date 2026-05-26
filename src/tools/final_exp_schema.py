"""TypedDict schema for `artifacts/Final_Exp.json` (FINAL_EXP Dashboard, US-001).

Stable contract between `build_final_exp_json.py` (writer) and
`build_final_dashboard.py` / the browser (reader). Bumping
`SCHEMA_VERSION` is a breaking change for all consumers.

Field shapes match what is already on disk:
- `level` is `int | None` (1..5; None for Phase A clean), not a stringified
  ``"L1"``/``"clean"`` enum — the renderer formats for display.
- `axis` is `str | None` (one of `AXIS_KEYS`; None for Phase A and Phase B).
- `treatment` is `str | None` (one of ``"T1"``/``"T2"``/``"T3"``;
  None for Phase A/B/B2nr/C; set on Phase B2/C2/D regularization-sweep rows).
- `status` matches `src.experiments.run_status.detect_status` exactly:
  ``"Pending" | "Running" | "Complete" | "Failed"`` (note: ``"Complete"``,
  not ``"Done"``).
- `seed` is the canonical run's `pl.seed_everything` value (US-038, v4).
  Always 42 on canonical cells; multi-seed audit replicates (seed != 42)
  are aggregated into the canonical row's `val_acc_mean` / `val_acc_std`
  rather than emitted as separate rows.

Privacy: this schema deliberately excludes checkpoint paths, weight URIs,
and host paths. Only metric scalars and config keys are carried.

Schema history
--------------
- v1 (US-001): initial contract.
- v2 (US-029.5, 2026-05-23): add `treatment: str | None` for Phase D
  regularization-sweep rows. Phase A/B/C carry `treatment=None` so the
  field is uniformly present on every row.
- v3 (US-046, 2026-05-26): extend `Phase` Literal with B2 / B2nr / C2.
  Add `seed: int` (canonical = 42), `val_acc_mean` and `val_acc_std`
  (populated from multi-seed audit replicates when present; None
  otherwise), and `seeds_observed: list[int]` (sorted seeds with a
  Complete run on disk for this base tag, always includes 42).
"""
from __future__ import annotations

from typing import Literal, TypedDict

SCHEMA_VERSION: Literal[3] = 3

Phase = Literal["A", "B", "B2", "B2nr", "C", "C2", "D"]
Status = Literal["Pending", "Running", "Complete", "Failed", "Deferred"]

PHASES: tuple[Phase, ...] = ("A", "B", "B2", "B2nr", "C", "C2", "D")
# `"Deferred"` (US-014) is for cells whose model is quarantined pending
# hardware (e.g. TransNeXt). Deferred cells stay in the matrix so the 402
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
    # US-029.5 (schema v2): Phase D regularization treatment.
    # US-046 (schema v3): also set on Phase B2 ("T3") and Phase C2 ("T3");
    # remains None on Phase A/B/B2nr/C rows so the field is uniformly present.
    treatment: str | None
    # US-046 (schema v3): canonical seed. Always 42 for the row's
    # canonical run; multi-seed audit replicates collapse into the same
    # canonical row via val_acc_mean / val_acc_std / seeds_observed.
    seed: int
    params: ParamsDict
    status: Status
    val_acc: float | None
    # US-046 (schema v3): aggregate across {seed=42} ∪ {seeds_observed}
    # when more than one seed has a Complete run on disk for this base tag.
    # None when only the canonical (seed=42) run exists.
    val_acc_mean: float | None
    val_acc_std: float | None
    seeds_observed: list[int]
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
    # Inline learning-curve series, embedded directly into Final_Exp.json so
    # the drawer can render under `file://` (where fetch() of cross-origin
    # local files is blocked by every modern browser). Each entry mirrors
    # the HistoryJSONCallback schema:
    #     {"epoch": int, "train_loss": float|None, "val_loss": float|None,
    #      "train_acc": float|None, "val_acc": float|None}
    # Null when the cell has no history.json. The dashboard JS prefers
    # `row.history` if present and falls back to fetch() only when it is
    # null (HTTP-served live updates).
    history: list[dict] | None


class CountsDict(TypedDict):
    total: int
    pending: int
    running: int
    complete: int
    failed: int
    deferred: int


class FinalExpDoc(TypedDict):
    schema_version: Literal[3]
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
