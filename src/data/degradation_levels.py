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
# Severity bumped 2026-05-12 for the RTX 5070 RALPH Loop campaign:
# the prior table compressed accuracy across L1..L3 (delta < 4pp on
# DenseNet × CIFAR-10), so the curve below pushes each axis one notch
# harder while keeping L1 close-to-clean and L5 a true breaking point.
DEGRADATION_LEVELS: dict[int, dict[str, float | int]] = {
    1: {"low_res": 18, "blur_kernel": 3,  "blur_sigma": 0.80, "noise_std": 0.04, "salt_pepper": 0.03, "saturation": 0.95},
    2: {"low_res": 12, "blur_kernel": 5,  "blur_sigma": 1.15, "noise_std": 0.08, "salt_pepper": 0.06, "saturation": 0.65},
    3: {"low_res":  8, "blur_kernel": 7,  "blur_sigma": 1.50, "noise_std": 0.12, "salt_pepper": 0.10, "saturation": 0.40},
    4: {"low_res":  6, "blur_kernel": 9,  "blur_sigma": 1.85, "noise_std": 0.16, "salt_pepper": 0.14, "saturation": 0.15},
    5: {"low_res":  3, "blur_kernel": 13, "blur_sigma": 2.30, "noise_std": 0.22, "salt_pepper": 0.18, "saturation": 0.00},
}

LEVEL_NAMES: dict[int, str] = {
    1: "Mild",
    2: "Light",
    3: "Moderate",
    4: "Severe",
    5: "Extreme",
}

# axis name -> the keys it controls in DEGRADATION_LEVELS.
# Phase C single-axis isolation: named axis at level L, every other axis
# pinned to its L1 (mild) value.
AXIS_KEYS: dict[str, tuple[str, ...]] = {
    "resolution":  ("low_res",),
    "blur":        ("blur_kernel", "blur_sigma"),
    "noise":       ("noise_std",),
    "salt_pepper": ("salt_pepper",),
    "saturation":  ("saturation",),
}

# Canonical iteration order for Phase C (matches run_all_phases.py product()).
AXES: tuple[str, ...] = ("resolution", "noise", "blur", "saturation", "salt_pepper")


def level_params(level: int, axis: Optional[str] = None) -> dict[str, float | int]:
    """Return the parameter dict for a given level.

    - level=L, axis=None     -> Phase B combined: all axes at level L.
    - level=L, axis="blur"   -> Phase C isolation: blur at L, others pinned to L1.
    """
    if level not in DEGRADATION_LEVELS:
        raise ValueError(f"Unknown level {level}; expected 1..5")
    if axis is None:
        return dict(DEGRADATION_LEVELS[level])
    if axis not in AXIS_KEYS:
        raise ValueError(f"Unknown axis {axis!r}; expected one of {list(AXIS_KEYS)}")
    base = dict(DEGRADATION_LEVELS[1])
    for k in AXIS_KEYS[axis]:
        base[k] = DEGRADATION_LEVELS[level][k]
    return base
