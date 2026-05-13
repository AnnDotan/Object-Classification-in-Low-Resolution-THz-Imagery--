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

Resolution policy: every model — CNNs and TransNeXt alike — trains at 224x224.
CIFAR-10 (32) and MNIST (28) inputs are upsampled to 224 by the data pipeline
before reaching the model, matching the ImageNet-pretrained regime end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.data.degrade import DegradeConfig, degrade_config_for
from src.experiments.cells import (
    DATASETS,
    EXPECTED_COUNTS,
    EXPECTED_TOTAL,
    LEVELS,
    MODELS,
    iter_cells,
)


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
    out_size: int = 224
    img_size: int = 224
    patch_size: int = 4
    pretrain_size: Optional[int] = None
    compile_mode: str = "none"
    precision: str = "bf16-mixed"


def build_final_matrix(out_size: int = 224) -> list[CellSpec]:
    """Return the 186 CellSpecs that drive the Final Research Phase.

    Tag enumeration is delegated to `src.experiments.cells.iter_cells()`
    (torch-free) so analysis tools can list cells without importing torch.
    Iteration order — Phase A first, then Phase B (level outer, model/dataset
    inner), then Phase C (level outer, axis next, model/dataset innermost) —
    is the contract downstream tools (run_all_phases.py, dashboard, dedupe
    filters) rely on.
    """
    specs: list[CellSpec] = []
    for meta in iter_cells():
        deg = degrade_config_for(meta.level, axis=meta.axis, out_size=out_size)
        pretrain_size = 224 if meta.model.startswith("transnext_") else None
        specs.append(
            CellSpec(
                tag=meta.tag,
                phase=meta.phase,
                model=meta.model,
                dataset=meta.dataset,
                level=meta.level,
                axis=meta.axis,
                degrade_config=deg,
                out_size=out_size,
                img_size=out_size,
                patch_size=4,
                pretrain_size=pretrain_size,
                compile_mode="none",
                precision="bf16-mixed",
            )
        )
    return specs


def cells_by_tag(matrix: Optional[list[CellSpec]] = None) -> dict[str, CellSpec]:
    """Convenience index: tag -> CellSpec. Lazily builds the matrix if not given."""
    if matrix is None:
        matrix = build_final_matrix()
    return {c.tag: c for c in matrix}
