"""Saturation-axis invariants for the degradation pipeline.

Two contracts required by the 186-cell campaign:
  (a) saturation=0.0 collapses RGB to a 3-channel grayscale where all
      three channels are equal per pixel (deterministic lerp endpoint).
  (b) saturation=1.0 is a no-op vs. a pipeline that never received the
      saturation field — byte-identical pixels (regression guard against
      accidentally activating the lerp at s=1.0).

Run: ``python -m src.tests.test_saturation``
"""
from __future__ import annotations

import sys

import torch

from src.data.degrade import DegradeConfig, degrade_image


def _rand_rgb(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, 32, 32, generator=g)


def _check_grayscale_endpoint() -> None:
    """saturation=0.0 -> all 3 channels equal per pixel (no noise/S&P/blur)."""
    img = _rand_rgb(seed=1)
    cfg = DegradeConfig(
        low_res=32,
        out_size=32,
        blur_kernel=0,
        blur_sigma=0.0,
        gaussian_noise_std=0.0,
        salt_pepper_amount=0.0,
        saturation=0.0,
        degradation_type='saturation',
    )
    out = degrade_image(img, cfg, seed=42)
    assert out.shape == (3, 32, 32), f"unexpected shape {out.shape}"

    rg_diff = (out[0] - out[1]).abs().max().item()
    rb_diff = (out[0] - out[2]).abs().max().item()
    assert rg_diff == 0.0, f"saturation=0.0 R/G channels differ: max abs diff={rg_diff}"
    assert rb_diff == 0.0, f"saturation=0.0 R/B channels differ: max abs diff={rb_diff}"

    print("OK [saturation=0.0] — 3 channels equal per pixel (full grayscale).")


def _check_unit_saturation_is_no_op() -> None:
    """saturation=1.0 must produce the same pixels as the legacy pipeline.

    We compare two configs that differ only in saturation (1.0 vs default).
    Both should yield byte-identical output because (1-1)·gray + 1·img == img.
    """
    img = _rand_rgb(seed=2)

    cfg_default = DegradeConfig(
        low_res=16,
        out_size=32,
        blur_kernel=5,
        blur_sigma=1.0,
        gaussian_noise_std=0.05,
        salt_pepper_amount=0.03,
        degradation_type='all',
    )
    cfg_explicit = DegradeConfig(
        low_res=16,
        out_size=32,
        blur_kernel=5,
        blur_sigma=1.0,
        gaussian_noise_std=0.05,
        salt_pepper_amount=0.03,
        saturation=1.0,
        degradation_type='all',
    )

    out_default = degrade_image(img, cfg_default, seed=123)
    out_explicit = degrade_image(img, cfg_explicit, seed=123)
    mse = torch.mean((out_default - out_explicit) ** 2).item()
    assert mse == 0.0, (
        f"saturation=1.0 is not a no-op vs default: MSE={mse} "
        "(the lerp activated at s=1.0 — should be byte-identical)"
    )

    print("OK [saturation=1.0] — byte-identical to baseline pipeline (MSE=0).")


def main() -> int:
    _check_grayscale_endpoint()
    _check_unit_saturation_is_no_op()
    return 0


if __name__ == "__main__":
    sys.exit(main())
