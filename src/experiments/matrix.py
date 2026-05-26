"""186-cell campaign matrix — single source of truth for the Final Phase.

Phase A (clean baselines) ............... 3 models x 2 datasets x 1 = 6
Phase B (combined degradation) .......... 3 x 2 x 5 levels         = 30
Phase C (single-axis isolation) ......... 3 x 2 x 5 levels x 5 axes = 150
                                                            total   = 186

Phase D (regularization sweep, US-026, opt-in via include_phase_d=True):
  3 treatments x 5 levels x 3 models x 2 datasets = 90 — duplicate of
  Phase B with regularization deltas layered on top of L3-Optuna baselines.

Phase B2 (US-038, opt-in via include_phase_b2=True):
  5 levels x 3 models x 2 datasets = 30 — duplicate of Phase B with the
  DegradeConfig overridden to sat = 0 + noise = 0 (THz protocol), and
  T3 regularization layered on top.

Phase B2nr (US-038, opt-in via include_phase_b2nr=True):
  3 models x 2 datasets = 6 — L3 only, same DegradeConfig override as
  Phase B2 L3 but WITHOUT T3 deltas. Pure-protocol Δ_B1→B2nr measurement.

Phase C2 (US-038, opt-in via include_phase_c2=True):
  5 levels x 3 axes x 3 models x 2 datasets = 90 — legacy Phase C
  single-axis isolation under the THz protocol; axes restricted to
  {"resolution", "blur", "salt_pepper"} (noise + saturation are
  protocol-invariant under C2 so they would be no-ops).

Canonical total when all four opt-in flags are True:
  6 + 30 + 30 + 6 + 150 + 90 + 90 = 402 cells.

Per the PRD (choice 2B), L1 cells in Phase C are NOT deduplicated against
the Phase B L1 row even though their DegradeConfig is identical — they are
trained again as a redundancy / sanity check, and the test below asserts
that the configs do match (same pixels under deterministic seeding).

Tag scheme (CLAUDE.md):
    final_clean_{model}_{dataset}                    (Phase A)
    final_B_L{level}_{model}_{dataset}               (Phase B)
    final_B2_L{level}_{model}_{dataset}              (Phase B2, US-038)
    final_B2nr_L3_{model}_{dataset}                  (Phase B2nr, US-038)
    final_C_L{level}_{axis}_{model}_{dataset}        (Phase C)
    final_C2_L{level}_{axis}_{model}_{dataset}       (Phase C2, US-038)
    final_D_{treatment}_L{level}_{model}_{dataset}   (Phase D, US-026)

Resolution policy: every model — CNNs and TransNeXt alike — trains at 224x224.
CIFAR-10 (32) and MNIST (28) inputs are upsampled to 224 by the data pipeline
before reaching the model, matching the ImageNet-pretrained regime end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.data.degrade import (
    DegradeConfig,
    degrade_config_for,
    degrade_config_for_b2,
    degrade_config_for_c2,
)
from src.experiments.cells import (
    DATASETS,
    EXPECTED_COUNTS,
    EXPECTED_COUNTS_WITH_ALL,
    EXPECTED_COUNTS_WITH_D,
    EXPECTED_TOTAL,
    EXPECTED_TOTAL_WITH_ALL,
    EXPECTED_TOTAL_WITH_D,
    LEVELS,
    MODELS,
    PHASE_B2_TREATMENT,
    PHASE_C2_AXES,
    PHASE_C2_TREATMENT,
    PHASE_D_TREATMENTS,
    iter_cells,
)


@dataclass(frozen=True)
class CellSpec:
    """One cell of the campaign matrix.

    `tag` is the on-disk identifier (under runs/final/<tag>/) and the
    primary key for matrix lookups. `degrade_config` is the precomputed
    DegradeConfig that the THzDataModule will consume — note that
    DegradeConfig is a mutable @dataclass so the spec is frozen but its
    nested config is technically mutable; downstream code must not mutate it.

    `treatment` is None for Phase A/B/B2nr/C and one of the treatment strings
    for the regularized phases:
      - Phase D: one of `PHASE_D_TREATMENTS` (T1/T2/T3)
      - Phase B2 / Phase C2: always "T3" (PHASE_B2_TREATMENT / PHASE_C2_TREATMENT)
    Treatment-specific hparam deltas are applied in `run_systematic` (see
    `_apply_phase_d_treatment` / `_apply_b2_or_c2_treatment`), not here —
    matrix.py only carries the marker so the dispatcher knows which
    regularization recipe to load.
    """
    tag: str
    phase: str                       # "A" | "B" | "B2" | "B2nr" | "C" | "C2" | "D"
    model: str                       # one of MODELS
    dataset: str                     # one of DATASETS
    level: Optional[int]             # None for Phase A; 1..5 otherwise
    axis: Optional[str]              # Phase C / C2 only
    degrade_config: DegradeConfig
    treatment: Optional[str] = None  # See class docstring
    out_size: int = 224
    img_size: int = 224
    patch_size: int = 4
    pretrain_size: Optional[int] = None
    compile_mode: str = "none"
    precision: str = "bf16-mixed"


def _degrade_config_for_meta(meta, out_size: int) -> DegradeConfig:
    """Route a CellMeta to its phase-specific DegradeConfig builder."""
    if meta.phase in ("B2", "B2nr"):
        assert meta.level is not None
        return degrade_config_for_b2(meta.level, out_size=out_size)
    if meta.phase == "C2":
        assert meta.level is not None and meta.axis is not None
        return degrade_config_for_c2(meta.level, meta.axis, out_size=out_size)
    return degrade_config_for(meta.level, axis=meta.axis, out_size=out_size)


def build_final_matrix(
    out_size: int = 224,
    include_phase_d: bool = False,
    include_phase_b2: bool = False,
    include_phase_b2nr: bool = False,
    include_phase_c2: bool = False,
) -> list[CellSpec]:
    """Return the campaign CellSpecs — 186 by default; up to 402 with all
    opt-in flags set.

    Tag enumeration is delegated to `src.experiments.cells.iter_cells()`
    (torch-free) so analysis tools can list cells without importing torch.
    Iteration order — Phase A → B → B2 → B2nr → C → C2 → D — is the
    contract downstream tools (run_all_phases.py, dashboard, dedupe filters)
    rely on.

    Phase D cells share Phase B's DegradeConfig (all 5 axes at level L); the
    treatment dimension is encoded only in `tag` and `treatment`, not in the
    pixel pipeline. Phase B2 / B2nr / C2 cells override the DegradeConfig
    (sat = 0 + noise = 0) per PRD §4.1 / §4.3.
    """
    specs: list[CellSpec] = []
    for meta in iter_cells(
        include_phase_d=include_phase_d,
        include_phase_b2=include_phase_b2,
        include_phase_b2nr=include_phase_b2nr,
        include_phase_c2=include_phase_c2,
    ):
        deg = _degrade_config_for_meta(meta, out_size=out_size)
        pretrain_size = 224 if meta.model.startswith("transnext_") else None
        specs.append(
            CellSpec(
                tag=meta.tag,
                phase=meta.phase,
                model=meta.model,
                dataset=meta.dataset,
                level=meta.level,
                axis=meta.axis,
                treatment=meta.treatment,
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
    """Convenience index: tag -> CellSpec. Lazily builds the full 402-cell
    matrix (all opt-in flags True) if not given, so Phase B2 / B2nr / C2 / D
    tags also resolve."""
    if matrix is None:
        matrix = build_final_matrix(
            include_phase_d=True,
            include_phase_b2=True,
            include_phase_b2nr=True,
            include_phase_c2=True,
        )
    return {c.tag: c for c in matrix}
