"""Verify Final_Exp.json PSNR/SSIM by recomputing them on a 1000-image probe.

Probes a single (dataset, phase, level, axis) pipeline and compares the
recomputed mean PSNR/SSIM to the value already stored in Final_Exp.json.
The PRD §4.1 methodology (1000 images, seed 2026) is implemented here as
the authoritative reference; cross-check that the existing trusted values
agree within a small tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.degrade import (  # noqa: E402
    DegradeConfig,
    SEED_OFFSET_VAL,
    degrade_config_for,
    degrade_config_for_b2,
    degrade_config_for_c2,
    degrade_image,
)

DATA_ROOT = ROOT / "data"


def _build_cfg(phase: str, level: int | None, axis: str | None) -> DegradeConfig:
    if phase == "A":
        return degrade_config_for(level=None)
    if phase == "B":
        return degrade_config_for(level=level)
    if phase == "C":
        return degrade_config_for(level=level, axis=axis)
    if phase == "B2":
        return degrade_config_for_b2(level=level)
    if phase == "B2nr":
        return degrade_config_for_b2(level=level)
    if phase == "C2":
        return degrade_config_for_c2(level=level, axis=axis)
    raise ValueError(f"unknown phase {phase!r}")


def _load_val_dataset(dataset: str):
    to_tensor = transforms.ToTensor()
    if dataset == "cifar10":
        ds = datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True, transform=to_tensor)
        n_channels_in = 3
    elif dataset == "mnist":
        ds = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=to_tensor)
        n_channels_in = 1
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return ds, n_channels_in


def _clean_reference(x: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    """The bicubic-upsampled clean image at out_size (Phase A early-return)."""
    if x.shape[-1] != out_size:
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(out_size, out_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
    return x.clamp(0, 1)


def probe(dataset: str, phase: str, level: int | None, axis: str | None, n: int = 1000, seed: int = 2026) -> dict:
    ds, n_channels_in = _load_val_dataset(dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), n, replace=False)

    cfg = _build_cfg(phase, level, axis)
    psnrs = np.empty(n, dtype=np.float64)
    ssims = np.empty(n, dtype=np.float64)
    for i, j in enumerate(idx):
        x, _ = ds[int(j)]  # [C,H,W] in [0,1]
        if n_channels_in == 1:
            x = x.repeat(3, 1, 1)
        ref = _clean_reference(x, out_size=cfg.out_size).numpy().transpose(1, 2, 0)
        deg = degrade_image(x, cfg, seed=int(j) + SEED_OFFSET_VAL).numpy().transpose(1, 2, 0)
        psnrs[i] = peak_signal_noise_ratio(ref, deg, data_range=1.0)
        ssims[i] = structural_similarity(ref, deg, channel_axis=-1, data_range=1.0)
    return {
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "n": n,
        "seed": seed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cifar10", "mnist"])
    p.add_argument("--phase", required=True, choices=["A", "B", "C", "D", "B2", "B2nr", "C2"])
    p.add_argument("--level", type=int, default=None)
    p.add_argument("--axis", type=str, default=None)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    res = probe(args.dataset, args.phase, args.level, args.axis, n=args.n, seed=args.seed)

    # Look up the stored value
    with (ROOT / "artifacts" / "Final_Exp.json").open("r", encoding="utf-8") as f:
        store = json.load(f)
    pipeline_phase = "B" if args.phase == "D" else args.phase
    candidates = [
        r for r in store["rows"]
        if r["dataset"] == args.dataset
        and r["phase"] == pipeline_phase
        and r.get("level") == args.level
        and (r.get("axis") or None) == args.axis
        and r.get("psnr_mean") is not None
    ]
    stored = candidates[0] if candidates else None

    print(json.dumps({
        "probe_result": res,
        "stored": None if stored is None else {
            "psnr_mean": stored["psnr_mean"],
            "psnr_std": stored["psnr_std"],
            "ssim_mean": stored["ssim_mean"],
            "ssim_std": stored["ssim_std"],
            "example_tag": stored["tag"],
        },
        "delta": None if stored is None else {
            "psnr": res["psnr_mean"] - stored["psnr_mean"],
            "ssim": res["ssim_mean"] - stored["ssim_mean"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
