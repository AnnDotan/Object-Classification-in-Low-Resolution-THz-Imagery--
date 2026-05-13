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

    Step order (saturation BEFORE noise/S&P so noise color stays correct):
      0. degradation_type=='none' -> early-return (clean baseline, only upsample).
      1. saturation lerp (replaces stochastic p_grayscale)
      2. downsample -> upsample
      3. gaussian blur
      4. additive gaussian noise
      5. salt-and-pepper
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
    # Cost: rotates `metrics.json.degradation_levels_hash`; prior Phase A
    # baselines (frozen pre-US-043) are non-comparable to runs after this
    # commit. Re-baseline is part of the US-043 scope. Bicubic is deterministic
    # in PyTorch >= 1.10 so the determinism gate still passes (MSE=0).
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
    if cfg.degradation_type in ('all', 'saturation'):
        s = float(cfg.saturation)
        if img.shape[0] == 3 and s < 1.0:
            gray = (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).clamp(0, 1)
            gray3 = torch.stack([gray, gray, gray], dim=0)
            img = (1.0 - s) * gray3 + s * img

    # 2) downsample to low_res then upsample to out_size.
    # US-043: downsample stays bilinear (anti-aliased pooling is sensible for
    # resolution loss simulation); upsample switches to bicubic to match the
    # clean-baseline upsample mode. Both stages remain deterministic.
    # US-003 (2026-05-14): when low_res == out_size the resolution axis is at
    # identity (Phase C inactive-axis semantic). Skip the bilinear downsample
    # and do a single bicubic upsample so the pixels match the Phase A clean
    # baseline upsampling exactly. Without this short-circuit, the cascade
    # bilinear(32->224) -> bicubic(224->224) would diverge from a single
    # bicubic(32->224), polluting the per-axis isolation signal.
    if cfg.degradation_type in ('all', 'downsampling'):
        img = img.unsqueeze(0)
        if cfg.low_res != cfg.out_size:
            img = torch.nn.functional.interpolate(img, size=(cfg.low_res, cfg.low_res), mode="bilinear", align_corners=False)
        img = torch.nn.functional.interpolate(img, size=(cfg.out_size, cfg.out_size), mode="bicubic", align_corners=False)
        img = img.squeeze(0)

    # 3) blur
    if cfg.degradation_type in ('all', 'blur'):
        if cfg.blur_kernel and cfg.blur_kernel > 1:
            img = _gaussian_blur_torch(img, kernel_size=cfg.blur_kernel, sigma=cfg.blur_sigma)

    # 4) additive gaussian noise
    if cfg.degradation_type in ('all', 'noise'):
        if cfg.gaussian_noise_std and cfg.gaussian_noise_std > 0:
            noise = torch.randn(
                img.shape, generator=torch_rng, device=img.device, dtype=img.dtype
            ) * cfg.gaussian_noise_std
            img = (img + noise).clamp(0, 1)

    # 5) salt-and-pepper noise
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

    return img


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
