"""Sample-image gallery generator.

Produces 4-panel strips (clean / degraded / |difference| / class label) for
each pipeline that appears in the experiments. Strips use the same fixed
6 indices per dataset to keep the visual comparison paired.

Indices are derived deterministically: numpy default_rng(2026).choice over
the validation split, take the first 6. Saved under
Final_Report/figures/samples/<dataset>/<pipeline_key>/sample_<k>.png.

We also produce two curated grids that go INTO the report body:
  figures/samples/grid_phase_B_levels_<dataset>.png  - rows L1, L3, L5; cols 6 fixed examples
  figures/samples/grid_phase_C_axes_<dataset>.png    - rows 5 axes at L3; cols 6 fixed examples
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.degrade import (  # noqa: E402
    SEED_OFFSET_VAL,
    degrade_config_for,
    degrade_config_for_b2,
    degrade_config_for_c2,
    degrade_image,
)

DATA_ROOT = ROOT / "data"
OUT = ROOT / "Final_Report" / "figures" / "samples"
N_FIXED = 6
PROBE_SEED = 2026

DATASETS = ("cifar10", "mnist")
LEVELS = (1, 3, 5)
AXES = ("resolution", "blur", "noise", "salt_pepper", "saturation")

CIFAR10_CLASSES = ("airplane", "auto", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")
MNIST_CLASSES = tuple(str(i) for i in range(10))


def _load_val(dataset: str):
    tf = transforms.ToTensor()
    if dataset == "cifar10":
        ds = datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True, transform=tf)
        return ds, CIFAR10_CLASSES, 3
    ds = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True, transform=tf)
    return ds, MNIST_CLASSES, 1


def _fixed_indices(n_val: int) -> np.ndarray:
    rng = np.random.default_rng(PROBE_SEED)
    return rng.choice(n_val, N_FIXED, replace=False)


def _to_disp(x: torch.Tensor) -> np.ndarray:
    """[C,H,W] in [0,1] -> HxWxC numpy in [0,1]."""
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


def _make_grid(dataset: str, row_specs: list[tuple[str, callable]], out_name: str) -> None:
    """row_specs: list of (label, degrade_fn). degrade_fn(x_3hw_01, idx_int) -> degraded x."""
    ds, classes, _ = _load_val(dataset)
    indices = _fixed_indices(len(ds))

    n_rows = len(row_specs)
    n_cols = N_FIXED
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.4, n_rows * 1.55))
    if n_rows == 1:
        axes = np.array([axes])
    for r, (label, fn) in enumerate(row_specs):
        for c, idx in enumerate(indices):
            x, y = ds[int(idx)]
            x_out = fn(x, int(idx))
            axes[r, c].imshow(_to_disp(x_out))
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(classes[int(y)], fontsize=7)
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0,
                              labelpad=42, ha="right", va="center")
    plt.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.02,
                        hspace=0.05, wspace=0.05)
    out_path = OUT / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(ROOT)}")


def grids() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for d in DATASETS:
        # Phase B levels grid: clean, L1, L3, L5
        cfg_clean = degrade_config_for(None)
        cfg_l1 = degrade_config_for(1)
        cfg_l3 = degrade_config_for(3)
        cfg_l5 = degrade_config_for(5)

        def _apply(cfg):
            def f(x, idx):
                x3 = x.repeat(3, 1, 1) if x.shape[0] == 1 else x
                return degrade_image(x3, cfg, seed=idx + SEED_OFFSET_VAL)
            return f

        rows = [
            ("Clean", lambda x, idx: _clean_224(x)),
            ("L1 Mild", _apply(cfg_l1)),
            ("L3 Moderate", _apply(cfg_l3)),
            ("L5 Extreme", _apply(cfg_l5)),
        ]
        _make_grid(d, rows, f"grid_phase_B_levels_{d}.png")

        # Phase C single-axis at L3
        rows = [("Clean", lambda x, idx: _clean_224(x))]
        for ax_name in AXES:
            cfg = degrade_config_for(level=3, axis=ax_name)
            rows.append((f"{ax_name} L3", _apply(cfg)))
        _make_grid(d, rows, f"grid_phase_C_axes_L3_{d}.png")

        # Phase B2 vs B comparison at L3 (saturation=0, noise=0)
        rows = [
            ("Clean", lambda x, idx: _clean_224(x)),
            ("B L3", _apply(degrade_config_for(3))),
            ("B2 L3", _apply(degrade_config_for_b2(3))),
            ("C2 res L3", _apply(degrade_config_for_c2(3, "resolution"))),
            ("C2 blur L3", _apply(degrade_config_for_c2(3, "blur"))),
            ("C2 S\\&P L3", _apply(degrade_config_for_c2(3, "salt_pepper"))),
        ]
        _make_grid(d, rows, f"grid_phase_THz_L3_{d}.png")


def main() -> None:
    print("[build_samples] building grids")
    grids()
    print("[build_samples] DONE")


if __name__ == "__main__":
    main()
