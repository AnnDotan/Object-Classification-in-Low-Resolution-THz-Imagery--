"""186-cell matrix invariants (US-007 contract, US-003 refresh 2026-05-14,
US-038 extension 2026-05-26 for Phase B2 / B2nr / C2 / multi-seed).

Asserts:
  - exactly 186 cells by default; per-phase counts A=6, B=30, C=150
  - 276 cells when include_phase_d=True (Phase D adds 90)
  - 402 cells when all four opt-in flags are True (B2 + B2nr + C2 + D)
  - every tag is unique across all phases
  - every Phase C cell has exactly one non-identity axis (US-003 invariant)
  - the non-identity axis matches the cell's `axis` field at the cell's level
  - the cells_by_tag index round-trips
  - Phase B2 cells carry saturation = 0 and gaussian_noise_std = 0 in their
    DegradeConfig (PRD §4.1 byte-for-byte)
  - Phase C2 cells carry the same sat/noise zeros AND isolate exactly one
    of {resolution, blur, salt_pepper}; treatment is "T3" (PRD §4.3)
  - Phase B2nr cells carry the B2 DegradeConfig but treatment is None
  - The 7 tag prefixes are mutually exclusive (final_clean_ / final_B_L /
    final_B2_L / final_B2nr_ / final_C_L / final_C2_ / final_D_) — the legacy
    Phase B check must use startswith("final_B_L"), not startswith("final_B").

Pre-US-003 the test asserted "Phase C L1 collapses to Phase B L1" — that
invariant is gone: Phase C L1 now has one axis at L1 mild and four at
identity, while Phase B L1 has all five at L1 mild.

Run: ``python -m src.tests.test_matrix``
"""
from __future__ import annotations

import sys

from src.data.degradation_levels import AXIS_KEYS, IDENTITY_VALUES, level_params
from src.experiments.cells import (
    PHASE_B2_TREATMENT,
    PHASE_C2_AXES,
    PHASE_C2_TREATMENT,
)
from src.experiments.matrix import (
    EXPECTED_COUNTS,
    EXPECTED_COUNTS_WITH_ALL,
    EXPECTED_TOTAL,
    EXPECTED_TOTAL_WITH_ALL,
    EXPECTED_TOTAL_WITH_D,
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
        # US-038 — new phase tag examples
        "final_B2_L3_resnet50_cifar10",
        "final_B2_L5_transnext_tiny_mnist",
        "final_B2nr_L3_densenet121_cifar10",
        "final_C2_L3_resolution_resnet50_cifar10",
        "final_C2_L5_salt_pepper_transnext_tiny_mnist",
        "final_D_T3_L3_densenet121_mnist",
    ):
        assert tag in by_tag, f"missing canonical tag: {tag}"
    print(f"OK [format] — canonical tag examples all resolve (12 examples).")


# -----------------------------------------------------------------------
# US-038 — Phase B2 / B2nr / C2 / matrix-count assertions
# -----------------------------------------------------------------------


def _check_b2_count_30() -> None:
    cells = build_final_matrix(include_phase_b2=True)
    b2 = [c for c in cells if c.phase == "B2"]
    assert len(b2) == 30, f"expected 30 Phase B2 cells, got {len(b2)}"
    # 5 levels x 3 models x 2 datasets = 30; every level represented 6 times.
    by_level: dict[int, int] = {}
    for c in b2:
        assert c.level is not None
        by_level[c.level] = by_level.get(c.level, 0) + 1
    for L in (1, 2, 3, 4, 5):
        assert by_level.get(L, 0) == 6, f"Phase B2 level {L}: expected 6 cells, got {by_level.get(L, 0)}"
    print("OK [b2-count] — Phase B2 enumerates 30 cells (5 levels × 6 (m, d) pairs).")


def _check_b2nr_count_6() -> None:
    cells = build_final_matrix(include_phase_b2nr=True)
    b2nr = [c for c in cells if c.phase == "B2nr"]
    assert len(b2nr) == 6, f"expected 6 Phase B2nr cells, got {len(b2nr)}"
    # All B2nr cells are L3 only.
    levels = {c.level for c in b2nr}
    assert levels == {3}, f"Phase B2nr should be L3-only; got levels={levels}"
    print("OK [b2nr-count] — Phase B2nr enumerates 6 cells (L3 × 6 (m, d) pairs).")


def _check_c2_count_90() -> None:
    cells = build_final_matrix(include_phase_c2=True)
    c2 = [c for c in cells if c.phase == "C2"]
    assert len(c2) == 90, f"expected 90 Phase C2 cells, got {len(c2)}"
    # 5 levels x 3 axes x 3 models x 2 datasets = 90.
    axes = {c.axis for c in c2}
    assert axes == set(PHASE_C2_AXES), (
        f"Phase C2 axes must be {set(PHASE_C2_AXES)}; got {axes}"
    )
    print("OK [c2-count] — Phase C2 enumerates 90 cells (5 levels × 3 axes × 6 (m, d) pairs).")


def _check_all_phases_total_402() -> None:
    cells = build_final_matrix(
        include_phase_d=True,
        include_phase_b2=True,
        include_phase_b2nr=True,
        include_phase_c2=True,
    )
    assert len(cells) == EXPECTED_TOTAL_WITH_ALL, (
        f"expected {EXPECTED_TOTAL_WITH_ALL} cells with all opt-ins; got {len(cells)}"
    )
    assert EXPECTED_TOTAL_WITH_ALL == 402, (
        f"canonical total must be 402; got {EXPECTED_TOTAL_WITH_ALL}"
    )
    by_phase: dict[str, int] = {}
    for c in cells:
        by_phase[c.phase] = by_phase.get(c.phase, 0) + 1
    for phase, expected in EXPECTED_COUNTS_WITH_ALL.items():
        actual = by_phase.get(phase, 0)
        assert actual == expected, (
            f"phase {phase}: expected {expected} cells, got {actual}"
        )
    print("OK [all-phases-total] — 402 cells across 7 phases (A=6, B=30, B2=30, B2nr=6, C=150, C2=90, D=90).")


def _check_b2_tags_distinct_from_b1() -> None:
    """Phase B2 tags use the `final_B2_L` prefix; Phase B uses `final_B_L`.
    The shared `final_B` substring would collide under a naive startswith
    check — the legacy Phase B check MUST use startswith("final_B_L")."""
    cells = build_final_matrix(include_phase_b2=True, include_phase_b2nr=True)
    b1_tags = {c.tag for c in cells if c.phase == "B"}
    b2_tags = {c.tag for c in cells if c.phase == "B2"}
    b2nr_tags = {c.tag for c in cells if c.phase == "B2nr"}
    assert b1_tags.isdisjoint(b2_tags), "Phase B1 / B2 tag collision"
    assert b1_tags.isdisjoint(b2nr_tags), "Phase B1 / B2nr tag collision"
    assert b2_tags.isdisjoint(b2nr_tags), "Phase B2 / B2nr tag collision"
    # The naive startswith("final_B") would match all three; the canonical
    # check uses startswith("final_B_L") for legacy Phase B.
    for tag in b1_tags:
        assert tag.startswith("final_B_L"), f"Phase B1 tag malformed: {tag}"
    for tag in b2_tags:
        assert tag.startswith("final_B2_L"), f"Phase B2 tag malformed: {tag}"
    for tag in b2nr_tags:
        assert tag.startswith("final_B2nr_"), f"Phase B2nr tag malformed: {tag}"
    print("OK [b2-tags-distinct] — Phase B1 / B2 / B2nr tag sets are pairwise disjoint.")


def _check_c2_tags_distinct_from_c() -> None:
    cells = build_final_matrix(include_phase_c2=True)
    c1_tags = {c.tag for c in cells if c.phase == "C"}
    c2_tags = {c.tag for c in cells if c.phase == "C2"}
    assert c1_tags.isdisjoint(c2_tags), "Phase C / C2 tag collision"
    for tag in c1_tags:
        assert tag.startswith("final_C_L"), f"Phase C tag malformed: {tag}"
    for tag in c2_tags:
        assert tag.startswith("final_C2_L"), f"Phase C2 tag malformed: {tag}"
    print("OK [c2-tags-distinct] — Phase C / C2 tag sets are disjoint.")


def _check_seven_prefix_mutual_exclusion() -> None:
    """The seven canonical phase prefixes must be mutually exclusive under
    `startswith` — i.e. no tag matches more than one prefix. The legacy
    Phase B check uses the L-prefix to disambiguate from final_B2_*.

    This is the load-bearing collision-safety assertion that PRD §7 + US-038
    require run_all_phases.py to honor at dispatch time.
    """
    prefixes = (
        "final_clean_",
        "final_B_L",     # legacy Phase B — L-prefix required
        "final_B2_L",    # Phase B2
        "final_B2nr_",   # Phase B2nr
        "final_C_L",     # legacy Phase C — L-prefix required (distinguishes from C2)
        "final_C2_L",    # Phase C2
        "final_D_",      # Phase D
    )
    cells = build_final_matrix(
        include_phase_d=True,
        include_phase_b2=True,
        include_phase_b2nr=True,
        include_phase_c2=True,
    )
    for c in cells:
        matched = [p for p in prefixes if c.tag.startswith(p)]
        assert len(matched) == 1, (
            f"tag {c.tag} matches {len(matched)} prefixes: {matched}"
        )
    print("OK [7-prefix-mutex] — all 402 tags match exactly one canonical phase prefix.")


def _check_b2_degrade_config_overrides() -> None:
    """PRD §4.1: Phase B2 DegradeConfig sets `saturation = 0.0` and
    `gaussian_noise_std = 0.0` byte-for-byte. All other axes flow through
    DEGRADATION_LEVELS[level] unchanged."""
    cells = build_final_matrix(include_phase_b2=True, include_phase_b2nr=True)
    b2_like = [c for c in cells if c.phase in ("B2", "B2nr")]
    assert len(b2_like) == 30 + 6
    for c in b2_like:
        cfg = c.degrade_config
        assert cfg.saturation == 0.0, (
            f"{c.tag}: saturation must be 0.0; got {cfg.saturation}"
        )
        assert cfg.gaussian_noise_std == 0.0, (
            f"{c.tag}: gaussian_noise_std must be 0.0; got {cfg.gaussian_noise_std}"
        )
        # Other axes must match the level's table values.
        params = level_params(c.level)  # type: ignore[arg-type]
        assert cfg.low_res == int(params["low_res"]), (
            f"{c.tag}: low_res={cfg.low_res} != table {params['low_res']}"
        )
        assert cfg.blur_kernel == int(params["blur_kernel"]), (
            f"{c.tag}: blur_kernel={cfg.blur_kernel} != table {params['blur_kernel']}"
        )
        assert cfg.salt_pepper_amount == float(params["salt_pepper"]), (
            f"{c.tag}: salt_pepper_amount={cfg.salt_pepper_amount} "
            f"!= table {params['salt_pepper']}"
        )
    print("OK [b2-degrade-overrides] — Phase B2 / B2nr DegradeConfig stamps sat=0 + noise=0; other axes pass through.")


def _check_c2_degrade_config_overrides_and_axis_isolation() -> None:
    """PRD §4.3: Phase C2 DegradeConfig sets `saturation = 0.0` and
    `gaussian_noise_std = 0.0` AND only the named axis is non-identity."""
    cells = build_final_matrix(include_phase_c2=True)
    c2 = [c for c in cells if c.phase == "C2"]
    for c in c2:
        cfg = c.degrade_config
        assert cfg.saturation == 0.0, f"{c.tag}: saturation must be 0.0"
        assert cfg.gaussian_noise_std == 0.0, f"{c.tag}: gaussian_noise_std must be 0.0"
        # Only the declared axis (resolution / blur / salt_pepper) differs from identity.
        active_axes = []
        if cfg.low_res != IDENTITY_VALUES["low_res"]:
            active_axes.append("resolution")
        if cfg.blur_kernel != IDENTITY_VALUES["blur_kernel"]:
            active_axes.append("blur")
        if cfg.salt_pepper_amount != IDENTITY_VALUES["salt_pepper"]:
            active_axes.append("salt_pepper")
        assert active_axes == [c.axis], (
            f"{c.tag}: active C2 axes {active_axes} != declared {[c.axis]}"
        )
    print("OK [c2-degrade-overrides] — Phase C2 stamps sat=0 + noise=0 + isolates exactly the declared axis.")


def _check_treatment_routing() -> None:
    """B2 carries treatment='T3'; C2 carries treatment='T3'; B2nr carries
    treatment=None. PRD §4.7 treatment-field summary."""
    cells = build_final_matrix(
        include_phase_b2=True,
        include_phase_b2nr=True,
        include_phase_c2=True,
        include_phase_d=True,
    )
    for c in cells:
        if c.phase == "B2":
            assert c.treatment == PHASE_B2_TREATMENT == "T3", (
                f"{c.tag}: Phase B2 treatment must be T3; got {c.treatment}"
            )
        elif c.phase == "C2":
            assert c.treatment == PHASE_C2_TREATMENT == "T3", (
                f"{c.tag}: Phase C2 treatment must be T3; got {c.treatment}"
            )
        elif c.phase == "B2nr":
            assert c.treatment is None, (
                f"{c.tag}: Phase B2nr treatment must be None; got {c.treatment!r}"
            )
        elif c.phase in ("A", "B", "C"):
            assert c.treatment is None, (
                f"{c.tag}: Phase {c.phase} treatment must be None; got {c.treatment!r}"
            )
        elif c.phase == "D":
            assert c.treatment in ("T1", "T2", "T3"), (
                f"{c.tag}: Phase D treatment must be T1/T2/T3; got {c.treatment!r}"
            )
    print("OK [treatment-routing] — B2='T3', C2='T3', B2nr=None, A/B/C=None, D in {T1,T2,T3}.")


def _check_resolution_blur_sp_match_level_for_b2_and_c2() -> None:
    """For each Phase B2 / C2 cell, the active spatial-domain axis value at
    level L must match DEGRADATION_LEVELS[L] byte-for-byte."""
    cells = build_final_matrix(include_phase_b2=True, include_phase_c2=True)
    b2 = [c for c in cells if c.phase == "B2"]
    c2 = [c for c in cells if c.phase == "C2"]

    for c in b2:
        params = level_params(c.level)  # type: ignore[arg-type]
        assert c.degrade_config.low_res == int(params["low_res"])
        assert c.degrade_config.blur_kernel == int(params["blur_kernel"])
        assert c.degrade_config.salt_pepper_amount == float(params["salt_pepper"])

    for c in c2:
        cfg = c.degrade_config
        if c.axis == "resolution":
            params = level_params(c.level)  # type: ignore[arg-type]
            assert cfg.low_res == int(params["low_res"]), (
                f"{c.tag}: low_res mismatch ({cfg.low_res} vs {params['low_res']})"
            )
        elif c.axis == "blur":
            params = level_params(c.level)  # type: ignore[arg-type]
            assert cfg.blur_kernel == int(params["blur_kernel"]), (
                f"{c.tag}: blur_kernel mismatch"
            )
            assert cfg.blur_sigma == float(params["blur_sigma"]), (
                f"{c.tag}: blur_sigma mismatch"
            )
        elif c.axis == "salt_pepper":
            params = level_params(c.level)  # type: ignore[arg-type]
            assert cfg.salt_pepper_amount == float(params["salt_pepper"]), (
                f"{c.tag}: salt_pepper mismatch"
            )
    print("OK [b2-c2-active-axis-matches-level] — every active spatial-domain axis equals DEGRADATION_LEVELS[L].")


def _check_phase_d_total_276() -> None:
    """Regression — Phase D alone still produces 276 cells (no spillover from
    the new opt-in flags)."""
    cells = build_final_matrix(include_phase_d=True)
    assert len(cells) == EXPECTED_TOTAL_WITH_D == 276, (
        f"Phase D regression: expected 276, got {len(cells)}"
    )
    print("OK [phase-d-regression] — Phase D alone still produces 276 cells.")


def main() -> int:
    _check_counts()
    _check_tag_uniqueness()
    _check_phase_c_isolates_single_axis()
    _check_tag_format_examples()
    # US-038 extensions
    _check_b2_count_30()
    _check_b2nr_count_6()
    _check_c2_count_90()
    _check_all_phases_total_402()
    _check_b2_tags_distinct_from_b1()
    _check_c2_tags_distinct_from_c()
    _check_seven_prefix_mutual_exclusion()
    _check_b2_degrade_config_overrides()
    _check_c2_degrade_config_overrides_and_axis_isolation()
    _check_treatment_routing()
    _check_resolution_blur_sp_match_level_for_b2_and_c2()
    _check_phase_d_total_276()
    return 0


if __name__ == "__main__":
    sys.exit(main())
