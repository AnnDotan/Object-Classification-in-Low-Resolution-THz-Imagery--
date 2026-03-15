# src/data/degrade.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class DegradeConfig:
    # downsample to low_res (e.g., 16 or 8) then upsample back to out_size
    low_res: int = 16
    out_size: int = 224  # set 32 if you decide to stay CIFAR-size
    p_grayscale: float = 0.3
    gaussian_noise_std: float = 0.08  # adjust later
    blur_kernel: int = 5  # odd number, e.g. 3/5/7
    blur_sigma: float = 1.0


def _gaussian_blur_torch(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    img: [C,H,W] float in [0,1]
    """
    if kernel_size <= 1:
        return img

    # create 1D gaussian
    k = kernel_size
    x = torch.arange(k, device=img.device, dtype=img.dtype) - (k - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2 * (sigma ** 2)))
    g = g / g.sum()

    # separable conv: first H then W
    # make depthwise conv weights
    c = img.shape[0]
    g_h = g.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    g_w = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)

    img_b = img.unsqueeze(0)  # [1,C,H,W]
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
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 1) optional grayscale
    if img.shape[0] == 3 and random.random() < cfg.p_grayscale:
        gray = (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).clamp(0, 1)
        img = torch.stack([gray, gray, gray], dim=0)

    # 2) downsample to low_res then upsample to out_size
    img = img.unsqueeze(0)  # [1,C,H,W]
    img = torch.nn.functional.interpolate(img, size=(cfg.low_res, cfg.low_res), mode="bilinear", align_corners=False)
    img = torch.nn.functional.interpolate(img, size=(cfg.out_size, cfg.out_size), mode="bilinear", align_corners=False)
    img = img.squeeze(0)

    # 3) blur
    if cfg.blur_kernel and cfg.blur_kernel > 1:
        img = _gaussian_blur_torch(img, kernel_size=cfg.blur_kernel, sigma=cfg.blur_sigma)

    # 4) additive gaussian noise
    if cfg.gaussian_noise_std and cfg.gaussian_noise_std > 0:
        noise = torch.randn_like(img) * cfg.gaussian_noise_std
        img = (img + noise).clamp(0, 1)

    return img