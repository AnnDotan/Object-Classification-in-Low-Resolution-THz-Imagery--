"""186-cell matrix invariants (US-007 contract, US-003 refresh 2026-05-14).

Asserts:
  - exactly 186 cells, with per-phase counts A=6, B=30, C=150
  - every tag is unique (the tag is the primary key for the run dir)
  - every Phase C cell has exactly one non-identity axis (US-003 invariant)
  - the non-identity axis matches the cell's `axis` field at the cell's level
  - the cells_by_tag index round-trips

Pre-US-003 the test asserted "Phase C L1 collapses to Phase B L1" — that
invariant is gone: Phase C L1 now has one axis at L1 mild and four at
identity, while Phase B L1 has all five at L1 mild.

Run: ``python -m src.tests.test_matrix``
"""
from __future__ import annotations

import sys

from src.data.degradation_levels import AXIS_KEYS, IDENTITY_VALUES, level_params
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


def _check_phase_c_isolates_single_axis() -> None:
    """US-003 invariant: every Phase C cell has exactly one non-identity axis,
    and that axis matches the cell's `axis` field at the cell's level.

    Replaces the pre-US-003 "Phase C L1 collapses to Phase B L1" invariant
    (which no longer holds — Phase C L1 has 4 identity axes, Phase B L1
    has 5 L1-mild axes).
    """
    matrix = build_final_matrix()
    phase_c = [c for c in matrix if c.phase == "C"]
    assert len(phase_c) == 150, (
        f"expected 150 Phase C cells, got {len(phase_c)}"
    )

    mismatches: list[str] = []
    wrong_axis_count: list[tuple[str, int]] = []

    for c in phase_c:
        assert c.axis is not None, f"Phase C cell missing axis: {c.tag}"
        assert c.level is not None, f"Phase C cell missing level: {c.tag}"
        params = level_params(c.level, axis=c.axis)
        active_axes = [
            ax
            for ax, keys in AXIS_KEYS.items()
            if any(params[k] != IDENTITY_VALUES[k] for k in keys)
        ]
        # At L1 the active axis's L1-mild value MAY equal the identity value
        # for axes where DEGRADATION_LEVELS[1] == IDENTITY_VALUES (none do
        # today — all 5 axes have non-identity L1 values), but we keep the
        # generic check to survive future curve tweaks.
        if len(active_axes) != 1:
            wrong_axis_count.append((c.tag, len(active_axes)))
            continue
        if active_axes[0] != c.axis:
            mismatches.append(f"{c.tag}: active={active_axes[0]} declared={c.axis}")

    assert not wrong_axis_count, (
        f"Phase C cells with != 1 active axis: {wrong_axis_count[:3]} "
        f"(total {len(wrong_axis_count)})"
    )
    assert not mismatches, (
        f"Phase C active axis != declared axis: {mismatches[:3]}"
    )
    print(f"OK [isolation] — all 150 Phase C cells isolate exactly one axis.")


def _check_tag_format_examples() -> None:
    """Smoke-check the canonical tag patterns CLAUDE.md documents."""
    by_tag = cells_by_tag()
    for tag in (
        "final_clean_resnet50_cifar10",
        "final_clean_transnext_tiny_mnist",
        "final_B_L3_densenet121_cifar10",
        "final_B_L5_transnext_tiny_mnist",
        "final_C_L4_noise_resnet50_cifar10",
        "final_C_L1_saturation_transnext_tiny_mnist",
    ):
        assert tag in by_tag, f"missing canonical tag: {tag}"
    print(f"OK [format] — canonical tag examples all resolve.")


def main() -> int:
    _check_counts()
    _check_tag_uniqueness()
    _check_phase_c_isolates_single_axis()
    _check_tag_format_examples()
    return 0


if __name__ == "__main__":
    sys.exit(main())
