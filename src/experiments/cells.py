"""Torch-free enumeration of the 186-cell campaign.

`matrix.py` builds the full `CellSpec` list (with a `DegradeConfig` per cell,
which requires torch). This module defines just the tags + metadata, so
analysis tools (Markdown tracker updater, dashboards, CI lint) can iterate
the matrix without importing torch.

If you change MODELS / DATASETS / LEVELS or the tag scheme here, also
update `matrix.py` — both must agree on the 186 tags.

Phase D — Regularization Sweep (US-026, 2026-05-23)
---------------------------------------------------
Phase D is a *duplicate* of Phase B with regularization treatments added on
top of the L3-Optuna-tuned baselines. It is opt-in via `include_phase_d=True`;
the default 186-cell matrix is unchanged.

  3 treatments × 5 levels × 3 models × 2 datasets = 90 cells

  T1 = architectural dropout    (CNN: dropout=0.2 / TransNeXt: drop_path_rate=0.2)
  T2 = label-mixing             (mixup_alpha=0.2, cutmix off)
  T3 = combo (kitchen-sink)     (T1 ∪ T2 ∪ cutmix_alpha=1.0)

Tag scheme: `final_D_{T}_L{l}_{m}_{d}` — cannot collide with `final_B_*`.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

from src.data.degradation_levels import AXES


MODELS: tuple[str, ...] = ("resnet50", "densenet121", "transnext_tiny")
DATASETS: tuple[str, ...] = ("cifar10", "mnist")
LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)

# Phase D — Regularization Sweep treatments (US-026).
PHASE_D_TREATMENTS: tuple[str, ...] = ("T1", "T2", "T3")

EXPECTED_COUNTS: dict[str, int] = {"A": 6, "B": 30, "C": 150}
EXPECTED_TOTAL: int = 186

# When Phase D is included, totals grow by 3*5*3*2 = 90 cells.
EXPECTED_COUNTS_WITH_D: dict[str, int] = {**EXPECTED_COUNTS, "D": 90}
EXPECTED_TOTAL_WITH_D: int = EXPECTED_TOTAL + EXPECTED_COUNTS_WITH_D["D"]


class CellMeta(NamedTuple):
    tag: str
    phase: str                       # "A" | "B" | "C" | "D"
    model: str
    dataset: str
    level: Optional[int]             # None for Phase A; 1..5 otherwise
    axis: Optional[str]              # Phase C only (one of AXES)
    treatment: Optional[str] = None  # Phase D only (one of PHASE_D_TREATMENTS)


def iter_cells(include_phase_d: bool = False) -> Iterator[CellMeta]:
    """Yield cells in canonical order (Phase A, B, C, optionally D).

    The default 186-cell ordering is unchanged from the original campaign;
    Phase D rows append after Phase C when `include_phase_d=True`.
    """
    for model, dataset in product(MODELS, DATASETS):
        yield CellMeta(
            tag=f"final_clean_{model}_{dataset}",
            phase="A", model=model, dataset=dataset, level=None, axis=None,
        )
    for level, model, dataset in product(LEVELS, MODELS, DATASETS):
        yield CellMeta(
            tag=f"final_B_L{level}_{model}_{dataset}",
            phase="B", model=model, dataset=dataset, level=level, axis=None,
        )
    for level, axis, model, dataset in product(LEVELS, AXES, MODELS, DATASETS):
        yield CellMeta(
            tag=f"final_C_L{level}_{axis}_{model}_{dataset}",
            phase="C", model=model, dataset=dataset, level=level, axis=axis,
        )
    if include_phase_d:
        for treatment, level, model, dataset in product(
            PHASE_D_TREATMENTS, LEVELS, MODELS, DATASETS
        ):
            yield CellMeta(
                tag=f"final_D_{treatment}_L{level}_{model}_{dataset}",
                phase="D", model=model, dataset=dataset, level=level, axis=None,
                treatment=treatment,
            )


def all_cells(include_phase_d: bool = False) -> list[CellMeta]:
    cells = list(iter_cells(include_phase_d=include_phase_d))
    expected = EXPECTED_TOTAL_WITH_D if include_phase_d else EXPECTED_TOTAL
    assert len(cells) == expected, f"expected {expected} cells, got {len(cells)}"
    return cells


# Default `runs/final` root used by `phase_d_present_on_disk` when the caller
# omits `runs_root`. Kept as a module-level constant (not a function default
# arg of `Path("runs/final")`) so tests can monkeypatch if ever needed.
_DEFAULT_RUNS_ROOT: Path = Path("runs/final")


def phase_d_present_on_disk(runs_root: Path = _DEFAULT_RUNS_ROOT) -> bool:
    """True iff at least one `final_D_*` run directory exists under
    `runs_root`. Used to gate Phase D inclusion in the dashboard JSON
    aggregator and the `Final_Exp.md` renderer — the default 186-cell
    view is preserved byte-identical until Phase D launches.

    Moved from `scripts/update_final_exp.py` to this torch-free module
    (US-029.5) so `src/tools/build_final_exp_json.py` can import it
    without the forbidden `scripts/` reverse-import.
    """
    if not runs_root.exists():
        return False
    for child in runs_root.iterdir():
        if child.is_dir() and child.name.startswith("final_D_"):
            return True
    return False
