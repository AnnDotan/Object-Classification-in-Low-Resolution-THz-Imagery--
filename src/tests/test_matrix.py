"""186-cell matrix invariants (US-007 contract).

Asserts:
  - exactly 186 cells, with per-phase counts A=6, B=30, C=150
  - every tag is unique (the tag is the primary key for the run dir)
  - each Phase C L1 cell's degrade_config equals the Phase B L1 config
    for the same (model, dataset) — an L1 isolation collapses by design
  - the cells_by_tag index round-trips

Run: ``python -m src.tests.test_matrix``
"""
from __future__ import annotations

import sys

from src.experiments.matrix import (
    EXPECTED_COUNTS,
    EXPECTED_TOTAL,
    build_final_matrix,
    cells_by_tag,
)


def _check_counts() -> None:
    cells = build_final_matrix()
    assert len(cells) == EXPECTED_TOTAL, f"expected 186 cells, got {len(cells)}"
    by_phase: dict[str, int] = {}
    for c in cells:
        by_phase[c.phase] = by_phase.get(c.phase, 0) + 1
    for phase, expected in EXPECTED_COUNTS.items():
        actual = by_phase.get(phase, 0)
        assert actual == expected, (
            f"phase {phase}: expected {expected} cells, got {actual}"
        )
    print(f"OK [counts] — total=186 (A=6, B=30, C=150) confirmed.")


def _check_tag_uniqueness() -> None:
    cells = build_final_matrix()
    tags = [c.tag for c in cells]
    assert len(set(tags)) == len(tags), (
        f"duplicate tags found: {len(tags)} cells, {len(set(tags))} unique"
    )
    assert len(set(tags)) == EXPECTED_TOTAL
    print(f"OK [unique] — all 186 tags unique.")


def _check_phase_c_l1_collapses_to_phase_b_l1() -> None:
    """At L1, every Phase C single-axis isolation must equal Phase B L1
    for the same (model, dataset) — by construction in degrade_config_for."""
    matrix = build_final_matrix()
    by_tag = cells_by_tag(matrix)

    phase_b_l1_configs = {
        (c.model, c.dataset): c.degrade_config
        for c in matrix
        if c.phase == "B" and c.level == 1
    }
    assert len(phase_b_l1_configs) == 6, (
        f"expected 6 Phase B L1 cells, got {len(phase_b_l1_configs)}"
    )

    phase_c_l1 = [c for c in matrix if c.phase == "C" and c.level == 1]
    assert len(phase_c_l1) == 30, (
        f"expected 30 Phase C L1 cells (5 axes x 3 models x 2 datasets), got {len(phase_c_l1)}"
    )

    mismatches: list[str] = []
    for c in phase_c_l1:
        ref = phase_b_l1_configs[(c.model, c.dataset)]
        if c.degrade_config != ref:
            mismatches.append(c.tag)
    assert not mismatches, (
        f"Phase C L1 configs differ from Phase B L1 for: {mismatches[:3]}"
    )
    print(f"OK [collapse] — all 30 Phase C L1 configs match Phase B L1.")


def _check_tag_format_examples() -> None:
    """Smoke-check the canonical tag patterns CLAUDE.md documents."""
    by_tag = cells_by_tag()
    for tag in (
        "final_clean_resnet50_cifar10",
        "final_clean_transnext_base_mnist",
        "final_B_L3_densenet121_cifar10",
        "final_B_L5_transnext_base_mnist",
        "final_C_L4_noise_resnet50_cifar10",
        "final_C_L1_saturation_transnext_base_mnist",
    ):
        assert tag in by_tag, f"missing canonical tag: {tag}"
    print(f"OK [format] — canonical tag examples all resolve.")


def main() -> int:
    _check_counts()
    _check_tag_uniqueness()
    _check_phase_c_l1_collapses_to_phase_b_l1()
    _check_tag_format_examples()
    return 0


if __name__ == "__main__":
    sys.exit(main())
