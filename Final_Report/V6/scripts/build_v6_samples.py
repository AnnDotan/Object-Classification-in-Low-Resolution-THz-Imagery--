"""V6 sample-image grids with print-readable row labels.

Rebuilds the two grids that enter the V6 report body, at a larger label
typography than the V5 originals (whose row labels were unreadable at print
size):

  figures/v6/samples/grid_THz_L3_{cifar10,mnist}.png
      rows: Clean / B1 L3 / B2 L3 / C2 resolution / C2 blur / C2 salt&pepper
  figures/v6/samples/grid_B1_levels_{cifar10,mnist}.png
      rows: Clean / L1 / L2 / L3 / L4 / L5 (combined Phase B1 recipe)

Same fixed validation indices as the V5 grids (default_rng(2026), first 6)
and the same per-sample degradation seeds (idx + SEED_OFFSET_VAL), so the
images are pixel-comparable with previous revisions.

Run from the repository root:
  python Final_Report/V6/scripts/build_v6_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.degrade import (  # noqa: E402
    SEED_OFFSET_VAL,
    degrade_config_for,
    degrade_config_for_b2,
    degrade_config_for_c2,
    degrade_image,
)

DATA_ROOT = ROOT / "data"
OUT = ROOT / "Final_Report" / "figures" / "v6" / "samples"
N_FIXED = 6
PROBE_SEED = 2026

CIFAR10_CLASSES = ("airplane", "auto", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")
MNIST_CLASSES = tuple(str(i) for i in range(10))


def _load_val(dataset: str):
    tf = transforms.ToTensor()
    if dataset == "cifar10":
        ds = datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True, transform=tf)
        return ds, CIFAR10_CLASSES
    ds = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=tf)
    return ds, MNIST_CLASSES


def _fixed_indices(n_val: int) -> np.ndarray:
    rng = np.random.default_rng(PROBE_SEED)
    return rng.choice(n_val, N_FIXED, replace=False)


def _to_disp(x: torch.Tensor) -> np.ndarray:
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x.clamp(0, 1).numpy().transpose(1, 2, 0)


def _clean_224(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if x.shape[-1] != 224:
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(224, 224), mode="bicubic", align_corners=False
        ).squeeze(0)
    return x.clamp(0, 1)


def _apply(cfg):
    def f(x, idx):
        x3 = x.repeat(3, 1, 1) if x.shape[0] == 1 else x
        return degrade_image(x3, cfg, seed=idx + SEED_OFFSET_VAL)
    return f


def _make_grid(dataset: str, row_specs, out_name: str) -> None:
    ds, classes = _load_val(dataset)
    indices = _fixed_indices(len(ds))

    n_rows, n_cols = len(row_specs), N_FIXED
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.75, n_rows * 1.85), dpi=200)
    for r, (label, fn) in enumerate(row_specs):
        for c, idx in enumerate(indices):
            x, y = ds[int(idx)]
            axes[r, c].imshow(_to_disp(fn(x, int(idx))))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(classes[int(y)], fontsize=13)
        axes[r, 0].set_ylabel(label, fontsize=13, rotation=0,
                              labelpad=8, ha="right", va="center")
    plt.subplots_adjust(left=0.16, right=0.995, top=0.94, bottom=0.01,
                        hspace=0.06, wspace=0.05)
    out_path = OUT / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for d in ("cifar10", "mnist"):
        # Combined Phase B1 severity strip: clean + all five levels.
        rows = [("Clean", lambda x, idx: _clean_224(x))]
        rows += [(f"B1 L{lv}", _apply(degrade_config_for(lv))) for lv in range(1, 6)]
        _make_grid(d, rows, f"grid_B1_levels_{d}.png")

        # THz-protocol comparison at L3.
        rows = [
            ("Clean", lambda x, idx: _clean_224(x)),
            ("B1 L3", _apply(degrade_config_for(3))),
            ("B2 L3", _apply(degrade_config_for_b2(3))),
            ("C2 res. L3", _apply(degrade_config_for_c2(3, "resolution"))),
            ("C2 blur L3", _apply(degrade_config_for_c2(3, "blur"))),
            ("C2 S&P L3", _apply(degrade_config_for_c2(3, "salt_pepper"))),
        ]
        _make_grid(d, rows, f"grid_THz_L3_{d}.png")
    print("[v6-samples] DONE")


if __name__ == "__main__":
    main()
