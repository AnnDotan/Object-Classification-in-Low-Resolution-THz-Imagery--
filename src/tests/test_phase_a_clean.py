"""Phase A clean-baseline invariants for the degradation pipeline.

US-001 contract: a Phase A "clean" cell must produce pixels that are a
pure bilinear resize of the source image, with no degradation applied.

This is already implemented in degrade_image via the ``degradation_type='none'``
early-return path (see src/data/degrade.py:84-93). build_final_matrix() emits
Phase A CellSpecs through degrade_config_for(level=None), which sets that
field. This test pins the contract so future refactors cannot silently
re-enable degradation on Phase A cells.

Three invariants:
  (a) Phase A clean output equals a pure bilinear resize + clamp(0,1) — MSE=0.
  (b) Different RNG seeds yield identical Phase A clean output (the 'none'
      path consumes no randomness).
  (c) build_final_matrix() returns CellSpecs whose degrade_config has
      degradation_type='none' for every Phase A cell.

Run: ``python -m src.tests.test_phase_a_clean``
"""
from __future__ import annotations

import sys

import torch

from src.data.degrade import degrade_config_for, degrade_image
from src.experiments.matrix import build_final_matrix


def _rand_rgb(shape: tuple[int, int, int], seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(*shape, generator=g)


def _check_phase_a_clean_is_pure_resize() -> None:
    """Phase A clean output == bilinear resize ל-out_size + clamp(0,1)."""
    img = _rand_rgb((3, 32, 32), seed=11)  # CIFAR-10-shaped input
    cfg = degrade_config_for(level=None, axis=None, out_size=224)

    assert cfg.degradation_type == "none", (
        f"degrade_config_for(level=None) must yield type='none', got {cfg.degradation_type!r}"
    )

    out = degrade_image(img, cfg, seed=42)
    assert out.shape == (3, 224, 224), f"unexpected shape {out.shape}"

    # Reference: pure bilinear resize ל-224 + clamp (mirrors the early-return body).
    ref = (
        torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .clamp(0, 1)
    )
    mse = torch.mean((out - ref) ** 2).item()
    assert mse == 0.0, f"Phase A clean is not pure resize: MSE={mse}"

    print("OK [phase-a-pure-resize] — clean output == bilinear resize + clamp (MSE=0).")


def _check_phase_a_clean_seed_invariant() -> None:
    """The 'none' path must not consume the RNG — seeds must not change pixels."""
    img = _rand_rgb((3, 28, 28), seed=12)  # MNIST-shaped (post 3-channel repeat)
    cfg = degrade_config_for(level=None, axis=None, out_size=224)

    out_a = degrade_image(img, cfg, seed=42)
    out_b = degrade_image(img, cfg, seed=999_999)
    mse = torch.mean((out_a - out_b) ** 2).item()
    assert mse == 0.0, (
        f"Phase A clean is RNG-dependent (seed leaked into output): MSE={mse}"
    )

    print("OK [phase-a-seed-invariant] — different seeds yield identical clean output.")


def _check_phase_a_no_op_when_already_at_out_size() -> None:
    """If input is already at out_size, the 'none' path must return clamp-only."""
    img = _rand_rgb((3, 224, 224), seed=13).clamp(0, 1)
    cfg = degrade_config_for(level=None, axis=None, out_size=224)

    out = degrade_image(img, cfg, seed=42)
    mse = torch.mean((out - img) ** 2).item()
    assert mse == 0.0, (
        f"Phase A clean altered already-at-out_size pixels: MSE={mse}"
    )

    print("OK [phase-a-noop-at-out_size] — input already at out_size passes through.")


def _check_matrix_phase_a_cells_are_all_clean() -> None:
    """Every Phase A CellSpec must carry degradation_type='none'."""
    matrix = build_final_matrix(out_size=224)
    phase_a = [c for c in matrix if c.phase == "A"]
    assert len(phase_a) == 6, f"expected 6 Phase A cells, got {len(phase_a)}"

    for c in phase_a:
        deg = c.degrade_config
        assert deg.degradation_type == "none", (
            f"{c.tag}: degradation_type={deg.degradation_type!r} (expected 'none')"
        )
        assert deg.out_size == 224, f"{c.tag}: out_size={deg.out_size} (expected 224)"
        assert deg.gaussian_noise_std == 0.0, f"{c.tag}: noise_std non-zero on Phase A"
        assert deg.salt_pepper_amount == 0.0, f"{c.tag}: S&P non-zero on Phase A"
        assert deg.saturation == 1.0, f"{c.tag}: saturation != 1.0 on Phase A"

    print("OK [matrix-phase-a-clean] — all 6 Phase A CellSpecs carry type='none'.")


def main() -> int:
    _check_phase_a_clean_is_pure_resize()
    _check_phase_a_clean_seed_invariant()
    _check_phase_a_no_op_when_already_at_out_size()
    _check_matrix_phase_a_cells_are_all_clean()
    return 0


if __name__ == "__main__":
    sys.exit(main())
