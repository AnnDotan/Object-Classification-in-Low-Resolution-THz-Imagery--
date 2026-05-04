"""Offline PSNR/SSIM measurement over a fixed 256-sample subset.

Decoupled from training (per PRD US-002, choice 3D): one CLI invocation per
cell, dumps `image_quality.json` next to (or under) `--out`. Same code path
as training (degrade_config_for + degrade_image), so values are identical
to the pixels the model actually sees.

CLI:
    python -m src.tools.measure_image_quality \\
        --cell-tag final_B_L3_resnet50_cifar10 \\
        --dataset cifar10 \\
        --n-samples 256 \\
        --out runs/final/final_B_L3_resnet50_cifar10/image_quality.json

Tag scheme (per CLAUDE.md):
    final_clean_{model}_{dataset}
    final_B_L{level}_{model}_{dataset}
    final_C_L{level}_{axis}_{model}_{dataset}

The model token is parsed but ignored for measurement (degradation is
model-independent). Including it in the tag keeps the file naming
1-to-1 with the campaign matrix.

`--levels-spec` is an optional override that bypasses the tag and lets
you specify the cell raw, e.g. `--levels-spec L3` (Phase B L3) or
`--levels-spec L3:noise` (Phase C L3 noise-only). Useful for ad-hoc runs
that don't need a tag.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from src.data.degrade import DegradeConfig, degrade_config_for, degrade_image
from src.data.degradation_levels import AXES


# Sample-index seed offset used for the 256-sample fixed subset.
# Distinct from SEED_OFFSET_TRAIN (=0) and SEED_OFFSET_VAL (=1e7) so the
# image-quality subset never collides with the training/validation streams.
QUALITY_SUBSET_SEED = 20_000_000


_TAG_CLEAN = re.compile(r"^final_clean_(?P<model>[^_]+)_(?P<dataset>[^_]+)$")
_TAG_B = re.compile(r"^final_B_L(?P<level>\d)_(?P<model>[^_]+)_(?P<dataset>[^_]+)$")
_TAG_C = re.compile(
    r"^final_C_L(?P<level>\d)_(?P<axis>[a-z_]+)_(?P<model>[^_]+)_(?P<dataset>[^_]+)$"
)


def parse_cell_tag(tag: str) -> dict:
    """Decode a campaign tag into {phase, level, axis, model, dataset}.

    Order matters: Phase C tag also matches Phase B's prefix structure, so
    we try the most specific (C) pattern first.
    """
    m = _TAG_C.match(tag)
    if m:
        axis = m.group("axis")
        if axis not in AXES:
            raise ValueError(
                f"unknown axis {axis!r} in tag {tag!r}; expected one of {AXES}"
            )
        return {
            "phase": "C",
            "level": int(m.group("level")),
            "axis": axis,
            "model": m.group("model"),
            "dataset": m.group("dataset"),
        }
    m = _TAG_B.match(tag)
    if m:
        return {
            "phase": "B",
            "level": int(m.group("level")),
            "axis": None,
            "model": m.group("model"),
            "dataset": m.group("dataset"),
        }
    m = _TAG_CLEAN.match(tag)
    if m:
        return {
            "phase": "A",
            "level": None,
            "axis": None,
            "model": m.group("model"),
            "dataset": m.group("dataset"),
        }
    raise ValueError(f"unrecognized cell tag: {tag!r}")


def parse_levels_spec(spec: str) -> tuple[Optional[int], Optional[str]]:
    """Decode `--levels-spec` short form.

    Forms:
        "clean"          -> (None, None)        Phase A
        "L<n>"           -> (n, None)           Phase B level n
        "L<n>:<axis>"    -> (n, axis)           Phase C level n, single axis
    """
    s = spec.strip().lower()
    if s in ("clean", "none", "a"):
        return None, None
    m = re.match(r"^l(\d)(?::([a-z_]+))?$", s)
    if not m:
        raise ValueError(
            f"invalid --levels-spec {spec!r}; expected 'clean', 'L3', or 'L3:noise'"
        )
    level = int(m.group(1))
    axis = m.group(2)
    if axis is not None and axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
    return level, axis


def _load_val_dataset(dataset: str, root: str = "./data"):
    """Return the raw torchvision val dataset returning [C,H,W] tensors in [0,1].

    We don't go through THzLikeCIFAR10/MNIST because we want the *clean* image
    to compare against — we apply degradation manually below, mirroring the
    training pipeline exactly.
    """
    from torchvision import datasets, transforms

    tf = transforms.ToTensor()
    if dataset == "cifar10":
        return datasets.CIFAR10(root=root, train=False, download=True, transform=tf)
    if dataset == "mnist":
        return datasets.MNIST(root=root, train=False, download=True, transform=tf)
    raise ValueError(f"unknown dataset {dataset!r}; expected cifar10 or mnist")


def _to_3channel_224(img: torch.Tensor, out_size: int) -> torch.Tensor:
    """Upsample to (3, out_size, out_size). Used to build the clean reference.

    MNIST returns [1, 28, 28] -> repeat to 3 channels first. CIFAR-10 is
    already [3, 32, 32]. Both then bilinear-upsample to (3, out_size, out_size).
    """
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[-1] != out_size:
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(out_size, out_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return img.clamp(0, 1)


def measure(
    dataset: str,
    deg_cfg: DegradeConfig,
    n_samples: int,
    out_size: int = 224,
) -> dict:
    """Compute PSNR + SSIM mean/std over a deterministic n-sample subset.

    Sample indices are `range(n_samples)` (the first n images of the val split),
    seeded into the existing val-stream SEED_OFFSET so the same image index
    always yields the same degraded pixels — matching what the model sees.
    """
    ds = _load_val_dataset(dataset)
    n = min(n_samples, len(ds))
    indices = list(range(n))

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    # SSIM kernel default 11x11 is too large for tiny input edge cases — at
    # 224×224 that's fine. data_range=1.0 because both clean and degraded
    # are clipped to [0,1].
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    psnr_vals: list[float] = []
    ssim_vals: list[float] = []

    for idx in indices:
        img, _ = ds[idx]
        clean = _to_3channel_224(img, out_size=out_size)

        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        # Match training: seed = idx + SEED_OFFSET_VAL is what the val loader
        # uses. The image-quality subset uses its own offset to stay disjoint
        # from train/val streams (so noise patterns don't accidentally line up).
        seed = idx + QUALITY_SUBSET_SEED
        degraded = degrade_image(img, deg_cfg, seed=seed).clamp(0, 1)

        # torchmetrics expects [B, C, H, W]
        c_b = clean.unsqueeze(0)
        d_b = degraded.unsqueeze(0)
        psnr_vals.append(float(psnr_metric(d_b, c_b).item()))
        ssim_vals.append(float(ssim_metric(d_b, c_b).item()))

    psnr_t = torch.tensor(psnr_vals)
    ssim_t = torch.tensor(ssim_vals)

    return {
        "psnr_mean": float(psnr_t.mean().item()),
        "psnr_std": float(psnr_t.std(unbiased=False).item()),
        "ssim_mean": float(ssim_t.mean().item()),
        "ssim_std": float(ssim_t.std(unbiased=False).item()),
        "n_samples": n,
        "sample_indices": indices,
        "degrade_config": asdict(deg_cfg),
        "subset_seed_offset": QUALITY_SUBSET_SEED,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure PSNR/SSIM for a single campaign cell."
    )
    p.add_argument(
        "--cell-tag",
        type=str,
        default=None,
        help="Campaign cell tag (final_clean_*, final_B_L*, final_C_L*_*).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "mnist"],
        default=None,
        help="Override dataset (defaults to dataset parsed from --cell-tag).",
    )
    p.add_argument(
        "--levels-spec",
        type=str,
        default=None,
        help="Alternative to --cell-tag: 'clean', 'L3', or 'L3:noise'.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=256,
        help="Number of val samples to measure (default 256).",
    )
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help="Destination path for image_quality.json.",
    )
    p.add_argument(
        "--out-size",
        type=int,
        default=224,
        help="Model input size (default 224).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    if not args.cell_tag and not args.levels_spec:
        print("error: must supply --cell-tag or --levels-spec", file=sys.stderr)
        return 2
    if args.levels_spec and args.cell_tag:
        print(
            "error: --cell-tag and --levels-spec are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    if args.cell_tag:
        info = parse_cell_tag(args.cell_tag)
        level = info["level"]
        axis = info["axis"]
        dataset = args.dataset or info["dataset"]
        cell_tag = args.cell_tag
    else:
        level, axis = parse_levels_spec(args.levels_spec)
        if not args.dataset:
            print(
                "error: --dataset is required when using --levels-spec",
                file=sys.stderr,
            )
            return 2
        dataset = args.dataset
        cell_tag = None

    deg_cfg = degrade_config_for(level, axis=axis, out_size=args.out_size)
    result = measure(
        dataset=dataset,
        deg_cfg=deg_cfg,
        n_samples=args.n_samples,
        out_size=args.out_size,
    )
    result["cell_tag"] = cell_tag
    result["dataset"] = dataset
    result["level"] = level
    result["axis"] = axis

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"wrote {out_path} "
        f"(psnr={result['psnr_mean']:.2f}±{result['psnr_std']:.2f}, "
        f"ssim={result['ssim_mean']:.4f}±{result['ssim_std']:.4f}, "
        f"n={result['n_samples']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
