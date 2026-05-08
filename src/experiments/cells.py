"""Torch-free enumeration of the 186-cell campaign.

`matrix.py` builds the full `CellSpec` list (with a `DegradeConfig` per cell,
which requires torch). This module defines just the tags + metadata, so
analysis tools (Markdown tracker updater, dashboards, CI lint) can iterate
the matrix without importing torch.

If you change MODELS / DATASETS / LEVELS or the tag scheme here, also
update `matrix.py` — both must agree on the 186 tags.
"""
from __future__ import annotations

from itertools import product
from typing import Iterator, NamedTuple, Optional

from src.data.degradation_levels import AXES


MODELS: tuple[str, ...] = ("resnet50", "densenet121", "transnext_base")
DATASETS: tuple[str, ...] = ("cifar10", "mnist")
LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)

EXPECTED_COUNTS: dict[str, int] = {"A": 6, "B": 30, "C": 150}
EXPECTED_TOTAL: int = 186


class CellMeta(NamedTuple):
    tag: str
    phase: str            # "A" | "B" | "C"
    model: str
    dataset: str
    level: Optional[int]  # None for Phase A; 1..5 otherwise
    axis: Optional[str]   # None for Phase A and B; one of AXES for Phase C


def iter_cells() -> Iterator[CellMeta]:
    """Yield the 186 cells in canonical order (Phase A, then B, then C)."""
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


def all_cells() -> list[CellMeta]:
    cells = list(iter_cells())
    assert len(cells) == EXPECTED_TOTAL, f"expected {EXPECTED_TOTAL} cells, got {len(cells)}"
    return cells
