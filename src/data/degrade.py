# src/data/degrade.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


SEED_OFFSET_TRAIN = 0
SEED_OFFSET_VAL = 10_000_000


@dataclass
class DegradeConfig:
    # downsample to low_res (e.g. 20/14/10/7/4) then upsample back to out_size.
    low_res: int = 16
    out_size: int = 224  # set 32 if you decide to stay CIFAR-size

    # Deprecated: stochastic grayscale flip. Replaced by the deterministic
    # `saturation` lerp below. Field is kept so older configs/JSONs still load,
    # but the new pipeline ignores it.
    p_grayscale: float = 0.3

    gaussian_noise_std: float = 0.08
    blur_kernel: int = 5  # odd number, e.g. 3/5/7/9/11
    blur_sigma: float = 1.0

    # Salt-and-pepper noise parameters
    salt_pepper_amount: float = 0.05  # fraction of pixels affected

    # Saturation axis: 1.0 = full color, 0.0 = full grayscale (deterministic lerp).
    saturation: float = 1.0

    # Degradation type isolation:
    #   'all'         -> apply every axis (saturation, low_res, blur, noise, salt_pepper)
    #   'none'        -> Phase A clean baseline: no degradation, only upsample to out_size
    #   'downsampling','blur','noise','salt_pepper','saturation' -> apply only that axis
    degradation_type: str = 'all'


def _gaussian_blur_torch(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """img: [C,H,W] float in [0,1]"""
    if kernel_size <= 1:
        return img

    k = kernel_size
    x = torch.arange(k, device=img.device, dtype=img.dtype) - (k - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * (sigma ** 2)))
    g = g / g.sum()

    c = img.shape[0]
    g_h = g.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    g_w = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)

    img_b = img.unsqueeze(0)
    pad = k // 2
    img_b = torch.nn.functional.pad(img_b, (0, 0, pad, pad), mode="reflect")
    img_b = torch.nn.functional.conv2d(img_b, g_h, groups=c)
    img_b = torch.nn.functional.pad(img_b, (pad, pad, 0, 0), mode="reflect")
    img_b = torch.nn.functional.conv2d(img_b, g_w, groups=c)
    return img_b.squeeze(0)


def degrade_image(img: torch.Tensor, cfg: DegradeConfig, seed: Optional[int] = None) -> torch.Tensor:
    """
    img: [C,H,W] float tensor in [0,1]
    returns: [C,out_size,out_size] float in [0,1]

    Pipeline v2 (US-017, 2026-05-20): noise + salt-and-pepper now operate at the
    pre-upsample resolution. Each stochastic pixel-level sample is then spread
    across the bicubic kernel's footprint by the upsample, producing the
    coarse-grain noise structure a real low-resolution sensor would record.
    Pipeline v1 applied noise/S&P at 224x224 (post-upsample), which made each
    sample land on a single 224-grid pixel and severely understated the
    perceived noise relative to the configured noise_std.

    Step order (saturation BEFORE noise/S&P so noise color stays correct):
      0. degradation_type=='none' -> early-return (clean baseline, only upsample).
      1. saturation lerp (replaces stochastic p_grayscale)
      2. downsample to low_res (bilinear)        <- pre-upsample begins
      3. additive gaussian noise (at low_res)    <- v2 move
      4. salt-and-pepper (at low_res)            <- v2 move
      5. upsample to out_size (bicubic)          <- pre-upsample ends
      6. gaussian blur (at out_size; kernels were calibrated for 224)
      7. final clamp to [0,1]

    Determinism: the local torch.Generator (lines below) is seeded per-sample
    by THzLikeCIFAR10/MNIST.__getitem__ (seed = idx + SEED_OFFSET_*). Moving
    the noise/S&P consumers earlier in the function changes how many random
    samples each step draws (low_res^2 instead of out_size^2) but the per-sample
    seed determinism still holds: two reads of the same val index yield
    byte-identical pixels under v2. The MSE=0 gate in test_degradation_determinism
    remains valid; only the cross-version comparison breaks (intentionally —
    `degradation_levels_hash` rotates so v1 vs v2 metrics.json are distinguishable).
    """
    # Local RNG only — never mutate global state from inside __getitem__,
    # or we'd clobber the DataLoader shuffler, model init, dropout, etc.
    torch_rng = torch.Generator(device=img.device)
    if seed is not None:
        torch_rng.manual_seed(int(seed))

    # 0) Clean baseline early-return (Phase A): no degradation, only upsample.
    # US-043 (operator mandate 2026-05-12): bicubic interpolation on the
    # 32->224 upsample. Reason: bilinear introduces aliasing artifacts that
    # handicap the ImageNet-pretrained ResNet50 / DenseNet121 receptive-field
    # hierarchy. Bicubic preserves high-frequency content for the CNN baselines.
    # Bicubic is deterministic in PyTorch >= 1.10 so the determinism gate still
    # passes (MSE=0). Phase A is unaffected by the v2 noise/S&P move.
    if cfg.degradation_type == 'none':
        if img.shape[-1] != cfg.out_size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(cfg.out_size, cfg.out_size),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)
        return img.clamp(0, 1)

    # 1) Saturation lerp (deterministic; replaces stochastic p_grayscale).
    # Applied first, at the native input resolution, so the grayscale luminance
    # is computed from un-degraded RGB values.
    if cfg.degradation_type in ('all', 'saturation'):
        s = float(cfg.saturation)
        if img.shape[0] == 3 and s < 1.0:
            gray = (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).clamp(0, 1)
            gray3 = torch.stack([gray, gray, gray], dim=0)
            img = (1.0 - s) * gray3 + s * img

    # 2) downsample to low_res (bilinear). Pre-US-017 the upsample-to-out_size
    # was fused into the same block; v2 splits them so noise/S&P can run between
    # the two interpolations at the lower resolution.
    # US-003 (2026-05-14): when low_res == out_size the resolution axis is at
    # identity. The bilinear downsample is skipped; only the bicubic upsample
    # at step 5 (which may be a no-op or a 32->224 enlargement, depending on
    # input shape) is left to run.
    if cfg.degradation_type in ('all', 'downsampling'):
        if cfg.low_res != cfg.out_size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(cfg.low_res, cfg.low_res),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

    # 3) additive gaussian noise (at pre-upsample resolution; v2 move).
    # Drawing noise at low_res instead of out_size means far fewer random
    # samples (e.g. 3*3 = 9 instead of 3*224*224 = 150K at L5 resolution),
    # each then bicubic-upsampled into a kernel-shaped patch. The perceived
    # noise std at 224 is lower than `gaussian_noise_std`, but the noise
    # *structure* is realistic (sensor-grain rather than fine-grain).
    if cfg.degradation_type in ('all', 'noise'):
        if cfg.gaussian_noise_std and cfg.gaussian_noise_std > 0:
            noise = torch.randn(
                img.shape, generator=torch_rng, device=img.device, dtype=img.dtype
            ) * cfg.gaussian_noise_std
            img = (img + noise).clamp(0, 1)

    # 4) salt-and-pepper noise (at pre-upsample resolution; v2 move).
    # A "salt" pixel at low_res becomes a small bright blob in the upsampled
    # image rather than a single isolated bright pixel — much closer to what
    # a real defective-pixel detector would record at THz frequencies.
    if cfg.degradation_type in ('all', 'salt_pepper'):
        if cfg.salt_pepper_amount and cfg.salt_pepper_amount > 0:
            mask = torch.rand(
                img[0:1].shape, generator=torch_rng, device=img.device, dtype=img.dtype
            )
            salt = mask < (cfg.salt_pepper_amount / 2.0)
            pepper = mask > (1.0 - cfg.salt_pepper_amount / 2.0)
            img = img.clone()
            img[:, salt.squeeze(0)] = 1.0
            img[:, pepper.squeeze(0)] = 0.0

    # 5) upsample to out_size (bicubic). Always runs when the input is not
    # already at out_size — covers the Phase B/C path (after the optional
    # bilinear downsample) and the resolution-identity Phase C path (where
    # we still need to bring the native 32x32 / 28x28 input up to 224x224
    # before the blur step).
    if img.shape[-1] != cfg.out_size or img.shape[-2] != cfg.out_size:
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(cfg.out_size, cfg.out_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)

    # 6) blur (at out_size; kernel/sigma table was calibrated for 224x224 on
    # 2026-05-14, so it stays post-upsample).
    if cfg.degradation_type in ('all', 'blur'):
        if cfg.blur_kernel and cfg.blur_kernel > 1:
            img = _gaussian_blur_torch(img, kernel_size=cfg.blur_kernel, sigma=cfg.blur_sigma)

    # 7) bicubic upsample can slightly overshoot [0,1]; clamp once at the end.
    return img.clamp(0, 1)


def degrade_config_for(
    level: Optional[int],
    axis: Optional[str] = None,
    out_size: int = 224,
) -> DegradeConfig:
    """Build a DegradeConfig from the canonical 5-level table.

    - level=None              -> Phase A clean baseline (no degradation, upsample only).
    - level=L, axis=None      -> Phase B combined: every axis at level L.
    - level=L, axis="<name>"  -> Phase C isolation: named axis at L, others pinned to L1.
    """
    from .degradation_levels import level_params

    if level is None:
        return DegradeConfig(
            low_res=out_size,
            out_size=out_size,
            blur_kernel=0,
            blur_sigma=0.0,
            gaussian_noise_std=0.0,
            salt_pepper_amount=0.0,
            saturation=1.0,
            p_grayscale=0.0,
            degradation_type='none',
        )

    p = level_params(level, axis=axis)
    return DegradeConfig(
        low_res=int(p['low_res']),
        out_size=out_size,
        blur_kernel=int(p['blur_kernel']),
        blur_sigma=float(p['blur_sigma']),
        gaussian_noise_std=float(p['noise_std']),
        salt_pepper_amount=float(p['salt_pepper']),
        saturation=float(p['saturation']),
        p_grayscale=0.0,  # deprecated; saturation lerp replaces stochastic grayscale
        degradation_type='all',
    )


def degrade_config_for_b2(
    level: int,
    out_size: int = 224,
) -> DegradeConfig:
    """Build a Phase B2 DegradeConfig (THz-protocol simplification).

    US-038 (v4, 2026-05-26). Identical to `degrade_config_for(level)` except:
      - `saturation = 0.0`         (full grayscale via deterministic lerp)
      - `gaussian_noise_std = 0.0` (additive-Gaussian noise step skipped by
                                    the `> 0` guard in `degrade_image`)
    All other axes (low_res, blur_kernel, blur_sigma, salt_pepper_amount)
    flow through `DEGRADATION_LEVELS[level]` unchanged.

    Used by Phase B2 (with T3 regularization layered on top) AND Phase
    B2-no-regularization (B2nr — bare Optuna L3 winners). Source of truth
    locked in PRD §4.1.
    """
    if level not in (1, 2, 3, 4, 5):
        raise ValueError(f"Phase B2 level must be 1..5; got {level!r}")
    cfg = degrade_config_for(level, axis=None, out_size=out_size)
    cfg.saturation = 0.0
    cfg.gaussian_noise_std = 0.0
    return cfg


_PHASE_C2_AXES: tuple[str, ...] = ("resolution", "blur", "salt_pepper")


def degrade_config_for_c2(
    level: int,
    axis: str,
    out_size: int = 224,
) -> DegradeConfig:
    """Build a Phase C2 DegradeConfig (THz-protocol single-axis isolation).

    US-038 (v4, 2026-05-26). Phase C2 = legacy Phase C single-axis isolation
    (US-003 identity semantics) under the THz protocol — the named `axis` is
    at level L, every other non-{saturation, noise_std} axis is at its
    `IDENTITY_VALUES`, and `saturation = 0.0` + `gaussian_noise_std = 0.0`
    are forced regardless of level.

    Valid axes: {"resolution", "blur", "salt_pepper"} — `noise` and
    `saturation` are protocol-invariant (always zero) and would produce a
    trivial no-op cell under the C2 override, so they are rejected here.

    Source of truth locked in PRD §4.3.
    """
    if level not in (1, 2, 3, 4, 5):
        raise ValueError(f"Phase C2 level must be 1..5; got {level!r}")
    if axis not in _PHASE_C2_AXES:
        raise ValueError(
            f"Phase C2 axis must be one of {_PHASE_C2_AXES}; got {axis!r}"
        )
    cfg = degrade_config_for(level, axis=axis, out_size=out_size)
    cfg.saturation = 0.0
    cfg.gaussian_noise_std = 0.0
    return cfg
