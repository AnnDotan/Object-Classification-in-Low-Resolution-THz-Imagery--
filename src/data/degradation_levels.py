# src/data/degradation_levels.py
"""Single source of truth for the 5-level degradation curve.

L1 = Mild (anchor, close to clean) ... L5 = Extreme (breaking point).
Imported by both run_systematic.py / run_all_phases.py and the dashboards
so there is exactly one definition of "what level X means" across the
186-cell final research matrix.
"""
from __future__ import annotations

from typing import Optional


# 5-level degradation table.
#
# saturation:  1.0 = full color, 0.0 = full grayscale (deterministic lerp).
#              Replaces the legacy stochastic `p_grayscale` field.
# noise_std:   additive Gaussian noise std on [0,1] image.
# salt_pepper: fraction of pixels flipped to {0, 1}.
# low_res:     downsample target before bilinear upsample to model input size.
# blur_kernel / blur_sigma:
#   The Gaussian blur runs AFTER the upsample to out_size=224, so kernel
#   and sigma are pixel-domain values at 224x224. The 2026-05-12 curve
#   used kernels designed for native-size (32x32) imagery (K=3..13,
#   sigma=0.8..2.3); at 224x224 those kernels cover ~1-6% of the image
#   width and produce essentially invisible blur (operator observation
#   on Phase C blur thumbs, 2026-05-14). The values below are rescaled
#   to produce perceptually meaningful blur on 224x224:
#     - kernel = 2 * ceil(2.5 * sigma) + 1 (captures 2.5-sigma footprint,
#       snapped to odd)
#     - sigma curve roughly doubles per level (visible -> heavy -> smeared)
# Severity bumped 2026-05-12 for the RTX 5070 RALPH Loop campaign:
# the prior table compressed accuracy across L1..L3 (delta < 4pp on
# DenseNet x CIFAR-10), so the curve below pushes each axis one notch
# harder while keeping L1 close-to-clean and L5 a true breaking point.
# Blur axis re-scaled 2026-05-14 (operator-approved) for 224x224 visibility.
DEGRADATION_LEVELS: dict[int, dict[str, float | int]] = {
    1: {"low_res": 18, "blur_kernel": 13, "blur_sigma":  2.50, "noise_std": 0.04, "salt_pepper": 0.03, "saturation": 0.95},
    2: {"low_res": 12, "blur_kernel": 25, "blur_sigma":  5.00, "noise_std": 0.08, "salt_pepper": 0.06, "saturation": 0.65},
    3: {"low_res":  8, "blur_kernel": 41, "blur_sigma":  8.00, "noise_std": 0.12, "salt_pepper": 0.10, "saturation": 0.40},
    4: {"low_res":  6, "blur_kernel": 61, "blur_sigma": 12.00, "noise_std": 0.16, "salt_pepper": 0.14, "saturation": 0.15},
    5: {"low_res":  3, "blur_kernel": 91, "blur_sigma": 18.00, "noise_std": 0.22, "salt_pepper": 0.18, "saturation": 0.00},
}

LEVEL_NAMES: dict[int, str] = {
    1: "Mild",
    2: "Light",
    3: "Moderate",
    4: "Severe",
    5: "Extreme",
}

# axis name -> the keys it controls in DEGRADATION_LEVELS.
# Phase C single-axis isolation (US-003, 2026-05-14): named axis at level L,
# every other axis at IDENTITY (no degradation). The pre-US-003 semantics
# ("every other axis pinned to L1 mild") confounded the per-axis signal —
# a "blur at L5" cell still carried L1 noise+S&P+resolution loss, so the
# measured drop attributed to blur was contaminated by the L1 floor.
AXIS_KEYS: dict[str, tuple[str, ...]] = {
    "resolution":  ("low_res",),
    "blur":        ("blur_kernel", "blur_sigma"),
    "noise":       ("noise_std",),
    "salt_pepper": ("salt_pepper",),
    "saturation":  ("saturation",),
}

# Canonical iteration order for Phase C (matches run_all_phases.py product()).
AXES: tuple[str, ...] = ("resolution", "noise", "blur", "saturation", "salt_pepper")


# Identity (no-degradation) values for inactive axes in Phase C isolation.
# Each value is the pass-through for its axis:
#   low_res=224     -> downsample stage short-circuits (degrade.py) to a single
#                      bicubic upsample from input to out_size, matching the
#                      Phase A clean baseline upsampling.
#   blur_kernel=1   -> _gaussian_blur_torch returns img unchanged (kernel<=1 guard).
#   blur_sigma=0.0  -> consistent with kernel=1 short-circuit.
#   noise_std=0.0   -> additive-noise step skipped (>0 guard in degrade.py).
#   salt_pepper=0.0 -> S&P step skipped (>0 guard in degrade.py).
#   saturation=1.0  -> lerp short-circuit (s<1.0 guard in degrade.py).
IDENTITY_VALUES: dict[str, float | int] = {
    "low_res":     224,
    "blur_kernel": 1,
    "blur_sigma":  0.0,
    "noise_std":   0.0,
    "salt_pepper": 0.0,
    "saturation":  1.0,
}


def level_params(level: int, axis: Optional[str] = None) -> dict[str, float | int]:
    """Return the parameter dict for a given level.

    - level=L, axis=None     -> Phase B combined: all axes at level L.
    - level=L, axis="blur"   -> Phase C isolation: blur at L, every other
                                axis at IDENTITY (no degradation) per
                                IDENTITY_VALUES. Pre-US-003 this returned
                                inactive axes at L1 mild — the change rotates
                                `degradation_levels_hash` so cross-batch
                                comparisons across the US-003 boundary are
                                explicitly forbidden (see docs/phase_c.md).
    """
    if level not in DEGRADATION_LEVELS:
        raise ValueError(f"Unknown level {level}; expected 1..5")
    if axis is None:
        return dict(DEGRADATION_LEVELS[level])
    if axis not in AXIS_KEYS:
        raise ValueError(f"Unknown axis {axis!r}; expected one of {list(AXIS_KEYS)}")
    base = dict(IDENTITY_VALUES)
    for k in AXIS_KEYS[axis]:
        base[k] = DEGRADATION_LEVELS[level][k]
    return base
