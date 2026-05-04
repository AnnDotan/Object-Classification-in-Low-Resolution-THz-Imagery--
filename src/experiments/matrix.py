"""186-cell campaign matrix — single source of truth for the Final Phase.

Phase A (clean baselines) ............... 3 models x 2 datasets x 1 = 6
Phase B (combined degradation) .......... 3 x 2 x 5 levels         = 30
Phase C (single-axis isolation) ......... 3 x 2 x 5 levels x 5 axes = 150
                                                            total   = 186

Per the PRD (choice 2B), L1 cells in Phase C are NOT deduplicated against
the Phase B L1 row even though their DegradeConfig is identical — they are
trained again as a redundancy / sanity check, and the test below asserts
that the configs do match (same pixels under deterministic seeding).

Tag scheme (CLAUDE.md):
    final_clean_{model}_{dataset}                   (Phase A)
    final_B_L{level}_{model}_{dataset}              (Phase B)
    final_C_L{level}_{axis}_{model}_{dataset}       (Phase C)
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

from src.data.degrade import DegradeConfig, degrade_config_for
from src.data.degradation_levels import AXES


MODELS: tuple[str, ...] = ("resnet50", "densenet121", "transnext_base")
DATASETS: tuple[str, ...] = ("cifar10", "mnist")
LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)

EXPECTED_COUNTS: dict[str, int] = {"A": 6, "B": 30, "C": 150}
EXPECTED_TOTAL: int = 186


@dataclass(frozen=True)
class CellSpec:
    """One cell of the 186-cell campaign.

    `tag` is the on-disk identifier (under runs/final/<tag>/) and the
    primary key for matrix lookups. `degrade_config` is the precomputed
    DegradeConfig that the THzDataModule will consume — note that
    DegradeConfig is a mutable @dataclass so the spec is frozen but its
    nested config is technically mutable; downstream code must not mutate it.
    """
    tag: str
    phase: str            # "A" | "B" | "C"
    model: str            # one of MODELS
    dataset: str          # one of DATASETS
    level: Optional[int]  # None for Phase A; 1..5 otherwise
    axis: Optional[str]   # None for Phase A and B; one of AXES for Phase C
    degrade_config: DegradeConfig


def build_final_matrix(out_size: int = 224) -> list[CellSpec]:
    """Return the 186 CellSpecs that drive the Final Research Phase.

    Iteration order: Phase A first, then Phase B (level outer, model/dataset
    inner), then Phase C (level outer, axis next, model/dataset innermost).
    Downstream tools (run_all_phases.py, dashboard, dedupe filters) rely
    on this stable ordering.
    """
    cells: list[CellSpec] = []

    # Phase A: clean baselines (no degradation, just upsample to out_size).
    for model, dataset in product(MODELS, DATASETS):
        cells.append(CellSpec(
            tag=f"final_clean_{model}_{dataset}",
            phase="A",
            model=model,
            dataset=dataset,
            level=None,
            axis=None,
            degrade_config=degrade_config_for(None, axis=None, out_size=out_size),
        ))

    # Phase B: combined degradation across all axes at level L.
    for level, model, dataset in product(LEVELS, MODELS, DATASETS):
        cells.append(CellSpec(
            tag=f"final_B_L{level}_{model}_{dataset}",
            phase="B",
            model=model,
            dataset=dataset,
            level=level,
            axis=None,
            degrade_config=degrade_config_for(level, axis=None, out_size=out_size),
        ))

    # Phase C: single-axis isolation — one axis at level L, others pinned to L1.
    for level, axis, model, dataset in product(LEVELS, AXES, MODELS, DATASETS):
        cells.append(CellSpec(
            tag=f"final_C_L{level}_{axis}_{model}_{dataset}",
            phase="C",
            model=model,
            dataset=dataset,
            level=level,
            axis=axis,
            degrade_config=degrade_config_for(level, axis=axis, out_size=out_size),
        ))

    return cells


def cells_by_tag(matrix: Optional[list[CellSpec]] = None) -> dict[str, CellSpec]:
    """Convenience index: tag -> CellSpec. Lazily builds the matrix if not given."""
    if matrix is None:
        matrix = build_final_matrix()
    return {c.tag: c for c in matrix}
