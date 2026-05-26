"""Torch-free enumeration of the 186-cell campaign.

`matrix.py` builds the full `CellSpec` list (with a `DegradeConfig` per cell,
which requires torch). This module defines just the tags + metadata, so
analysis tools (Markdown tracker updater, dashboards, CI lint) can iterate
the matrix without importing torch.

If you change MODELS / DATASETS / LEVELS or the tag scheme here, also
update `matrix.py` — both must agree on the cell counts.

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

Phase B2 / B2nr / C2 — THz-protocol simplification (US-038, 2026-05-26)
----------------------------------------------------------------------
Phase B2 is a *duplicate* of Phase D T3 with the DegradeConfig overridden
to zero `saturation` and `gaussian_noise_std` (THz protocol). Opt-in via
`include_phase_b2=True`. Tag: `final_B2_L{l}_{m}_{d}`. 30 cells.

Phase B2nr (no-regularization) is a 6-cell L3-only arm: same DegradeConfig
as Phase B2 L3 but WITHOUT the T3 deltas. Tag: `final_B2nr_L3_{m}_{d}`.
Provides the pure-protocol Δ_B1→B2nr decomposition.

Phase C2 is *legacy Phase C single-axis isolation under the B2 protocol* +
T3 regularization. The named axis ∈ {"resolution", "blur", "salt_pepper"}
is at level L; every other non-{saturation, noise_std} axis is at identity.
Tag: `final_C2_L{l}_{axis}_{m}_{d}`. 90 cells.

Canonical total when all four opt-in flags are True:
  6 + 30 + 30 + 6 + 150 + 90 + 90 = 402 cells.
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

# Phase B2 / C2 — THz-protocol simplification (US-038).
# Both phases reuse Phase D's T3 regularization byte-for-byte; the dispatcher
# routes through `_phase_d_treatment_deltas("T3", model)` so the values live
# in exactly one place (run_systematic.py:_phase_d_treatment_deltas).
PHASE_B2_TREATMENT: str = "T3"
PHASE_C2_TREATMENT: str = "T3"
PHASE_C2_AXES: tuple[str, ...] = ("resolution", "blur", "salt_pepper")

EXPECTED_COUNTS: dict[str, int] = {"A": 6, "B": 30, "C": 150}
EXPECTED_TOTAL: int = 186

# When Phase D is included, totals grow by 3*5*3*2 = 90 cells.
EXPECTED_COUNTS_WITH_D: dict[str, int] = {**EXPECTED_COUNTS, "D": 90}
EXPECTED_TOTAL_WITH_D: int = EXPECTED_TOTAL + EXPECTED_COUNTS_WITH_D["D"]

# Phase B2 / B2nr / C2 cell counts (US-038).
EXPECTED_COUNTS_WITH_B2: dict[str, int] = {**EXPECTED_COUNTS, "B2": 30}
EXPECTED_COUNTS_WITH_B2NR: dict[str, int] = {**EXPECTED_COUNTS, "B2nr": 6}
EXPECTED_COUNTS_WITH_C2: dict[str, int] = {**EXPECTED_COUNTS, "C2": 90}
EXPECTED_COUNTS_WITH_ALL: dict[str, int] = {
    "A": 6, "B": 30, "B2": 30, "B2nr": 6, "C": 150, "C2": 90, "D": 90,
}
EXPECTED_TOTAL_WITH_ALL: int = sum(EXPECTED_COUNTS_WITH_ALL.values())  # = 402


class CellMeta(NamedTuple):
    tag: str
    phase: str                       # "A" | "B" | "B2" | "B2nr" | "C" | "C2" | "D"
    model: str
    dataset: str
    level: Optional[int]             # None for Phase A; 1..5 otherwise
    axis: Optional[str]              # Phase C / C2 only
    treatment: Optional[str] = None  # Phase D / B2 / C2 only


def iter_cells(
    include_phase_d: bool = False,
    include_phase_b2: bool = False,
    include_phase_b2nr: bool = False,
    include_phase_c2: bool = False,
) -> Iterator[CellMeta]:
    """Yield cells in canonical order: A, B, B2, B2nr, C, C2, D.

    The default 186-cell ordering (A → B → C) is unchanged from the original
    campaign; opt-in phases append after their natural neighbours per the
    canonical order documented above.
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
    if include_phase_b2:
        for level, model, dataset in product(LEVELS, MODELS, DATASETS):
            yield CellMeta(
                tag=f"final_B2_L{level}_{model}_{dataset}",
                phase="B2", model=model, dataset=dataset, level=level, axis=None,
                treatment=PHASE_B2_TREATMENT,
            )
    if include_phase_b2nr:
        for model, dataset in product(MODELS, DATASETS):
            yield CellMeta(
                tag=f"final_B2nr_L3_{model}_{dataset}",
                phase="B2nr", model=model, dataset=dataset, level=3, axis=None,
                treatment=None,
            )
    for level, axis, model, dataset in product(LEVELS, AXES, MODELS, DATASETS):
        yield CellMeta(
            tag=f"final_C_L{level}_{axis}_{model}_{dataset}",
            phase="C", model=model, dataset=dataset, level=level, axis=axis,
        )
    if include_phase_c2:
        for level, axis, model, dataset in product(
            LEVELS, PHASE_C2_AXES, MODELS, DATASETS
        ):
            yield CellMeta(
                tag=f"final_C2_L{level}_{axis}_{model}_{dataset}",
                phase="C2", model=model, dataset=dataset, level=level, axis=axis,
                treatment=PHASE_C2_TREATMENT,
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


def _expected_total(
    include_phase_d: bool,
    include_phase_b2: bool,
    include_phase_b2nr: bool,
    include_phase_c2: bool,
) -> int:
    n = EXPECTED_TOTAL
    if include_phase_b2:
        n += EXPECTED_COUNTS_WITH_ALL["B2"]
    if include_phase_b2nr:
        n += EXPECTED_COUNTS_WITH_ALL["B2nr"]
    if include_phase_c2:
        n += EXPECTED_COUNTS_WITH_ALL["C2"]
    if include_phase_d:
        n += EXPECTED_COUNTS_WITH_ALL["D"]
    return n


def all_cells(
    include_phase_d: bool = False,
    include_phase_b2: bool = False,
    include_phase_b2nr: bool = False,
    include_phase_c2: bool = False,
) -> list[CellMeta]:
    cells = list(iter_cells(
        include_phase_d=include_phase_d,
        include_phase_b2=include_phase_b2,
        include_phase_b2nr=include_phase_b2nr,
        include_phase_c2=include_phase_c2,
    ))
    expected = _expected_total(
        include_phase_d, include_phase_b2, include_phase_b2nr, include_phase_c2
    )
    assert len(cells) == expected, f"expected {expected} cells, got {len(cells)}"
    return cells


# Default `runs/final` root used by `phase_*_present_on_disk` when the caller
# omits `runs_root`. Kept as a module-level constant (not a function default
# arg of `Path("runs/final")`) so tests can monkeypatch if ever needed.
_DEFAULT_RUNS_ROOT: Path = Path("runs/final")


def _phase_present(runs_root: Path, prefix: str) -> bool:
    if not runs_root.exists():
        return False
    for child in runs_root.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            return True
    return False


def phase_d_present_on_disk(runs_root: Path = _DEFAULT_RUNS_ROOT) -> bool:
    """True iff at least one `final_D_*` run directory exists under
    `runs_root`. Used to gate Phase D inclusion in the dashboard JSON
    aggregator and the `Final_Exp.md` renderer — the default 186-cell
    view is preserved byte-identical until Phase D launches.

    Moved from `scripts/update_final_exp.py` to this torch-free module
    (US-029.5) so `src/tools/build_final_exp_json.py` can import it
    without the forbidden `scripts/` reverse-import.
    """
    return _phase_present(runs_root, "final_D_")


def phase_b2_present_on_disk(runs_root: Path = _DEFAULT_RUNS_ROOT) -> bool:
    """True iff at least one `final_B2_L*` run directory exists under
    `runs_root` (US-038). Excludes `final_B2nr_*` — those have their own
    presence helper.
    """
    if not runs_root.exists():
        return False
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        # `final_B2_L*` distinguishes from `final_B2nr_*` (no L-prefix).
        if child.name.startswith("final_B2_L"):
            return True
    return False


def phase_b2nr_present_on_disk(runs_root: Path = _DEFAULT_RUNS_ROOT) -> bool:
    """True iff at least one `final_B2nr_*` run directory exists (US-038)."""
    return _phase_present(runs_root, "final_B2nr_")


def phase_c2_present_on_disk(runs_root: Path = _DEFAULT_RUNS_ROOT) -> bool:
    """True iff at least one `final_C2_*` run directory exists (US-038)."""
    return _phase_present(runs_root, "final_C2_")
