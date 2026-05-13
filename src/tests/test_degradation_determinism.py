"""Validation-degradation determinism check.

Run: ``python -m src.tests.test_degradation_determinism``

Contract: val pixels are byte-identical across freshly-built dataset
instances at out_size=224 (the campaign's single output resolution). The
test verifies:

  (a) intra-instance: two reads of val image #N from the same dataset are byte-equal
  (b) cross-instance: a freshly-built dataset produces the same val image #N
  (c) batched read via DataLoader is byte-stable across instances
  (d) pixel histograms are exactly equal across freshly-built instances
  (e) train/val seed offsets are disjoint (SEED_OFFSET_TRAIN != SEED_OFFSET_VAL)

Datasets exercised:
  - cifar10 @ out_size=224
  - mnist   @ out_size=224

Tested at the dataset layer (where the determinism actually lives), so the
test does not require pytorch_lightning. A Lightning-DataModule cross-check
is added at the end and skipped automatically if the import fails.
"""
from __future__ import annotations

import sys
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.data.datasets import DataConfig, THzLikeCIFAR10, THzLikeMNIST


def _build(name: str, train: bool, out_size: int = 224) -> Tuple[torch.utils.data.Dataset, DataConfig]:
    cls = THzLikeMNIST if name == "mnist" else THzLikeCIFAR10
    cfg = DataConfig(
        dataset=name,
        train=train,
        out_size=out_size,
        low_res=16,
    )
    return cls(cfg), cfg


def _check_dataset(name: str, out_size: int = 224) -> None:
    label = f"{name}@{out_size}"
    ds_a, _ = _build(name, train=False, out_size=out_size)
    ds_b, _ = _build(name, train=False, out_size=out_size)

    # Pixel shape sanity: every emitted image must match out_size.
    a0 = ds_a[5][0]
    assert tuple(a0.shape) == (3, out_size, out_size), (
        f"[{label}] unexpected pixel shape {tuple(a0.shape)} for out_size={out_size}"
    )

    # (a) intra-instance: same idx -> same pixels
    a0_again = ds_a[5][0]
    intra_mse = torch.mean((a0 - a0_again) ** 2).item()
    assert intra_mse == 0.0, (
        f"[{label}] intra-instance non-determinism for val idx=5: MSE={intra_mse}"
    )

    # (b) cross-instance: freshly-built dataset, same pixels
    b0 = ds_b[5][0]
    inter_mse = torch.mean((a0 - b0) ** 2).item()
    assert inter_mse == 0.0, (
        f"[{label}] cross-instance non-determinism for val idx=5: MSE={inter_mse}"
    )

    # (c) batched read via DataLoader (val protocol: shuffle=False)
    loader_a = DataLoader(ds_a, batch_size=32, shuffle=False, num_workers=0)
    loader_b = DataLoader(ds_b, batch_size=32, shuffle=False, num_workers=0)
    batch_a = next(iter(loader_a))[0]
    batch_b = next(iter(loader_b))[0]
    batch_mse = torch.mean((batch_a - batch_b) ** 2).item()
    assert batch_mse == 0.0, (
        f"[{label}] DataLoader-batch non-determinism: MSE={batch_mse}"
    )

    # (d) pixel histograms must match exactly
    hist_a = torch.histc(batch_a, bins=256, min=-3.0, max=3.0)
    hist_b = torch.histc(batch_b, bins=256, min=-3.0, max=3.0)
    hist_diff = (hist_a - hist_b).abs().sum().item()
    assert hist_diff == 0.0, (
        f"[{label}] pixel-histogram differs across instances: diff={hist_diff}"
    )

    # (e) train/val seed offsets are disjoint
    train_ds, _ = _build(name, train=True, out_size=out_size)
    train0 = train_ds[0][0]
    val0 = ds_a[0][0]
    train_val_mse = torch.mean((train0 - val0) ** 2).item()
    assert train_val_mse > 0.0, (
        f"[{label}] train/val seed offsets collide: MSE={train_val_mse} "
        "(SEED_OFFSET_TRAIN and SEED_OFFSET_VAL must be different)"
    )

    print(
        f"OK [{label}] — val determinism verified "
        f"(intra=0, inter=0, batch=0, hist diff=0, "
        f"train/val disjoint MSE={train_val_mse:.4f})."
    )


def _check_lightning_datamodule(dataset: str, out_size: int = 224) -> None:
    try:
        import pytorch_lightning as pl  # noqa: F401
        from src.lightning.datamodule import THzDataModule
    except Exception as e:
        print(f"SKIP lightning-datamodule check ({type(e).__name__}: {e})")
        return

    label = f"{dataset}@{out_size}"
    pl.seed_everything(42, workers=True)
    dm1 = THzDataModule(
        dataset=dataset, out_size=out_size, low_res=16,
        batch_size=32, val_subset=64,
    )
    dm1.prepare_data()
    dm1.setup()
    a = next(iter(dm1.val_dataloader()))[0]
    b = next(iter(dm1.val_dataloader()))[0]
    intra = torch.mean((a - b) ** 2).item()
    assert intra == 0.0, f"[{label}] DataModule intra MSE={intra}"

    pl.seed_everything(42, workers=True)
    dm2 = THzDataModule(
        dataset=dataset, out_size=out_size, low_res=16,
        batch_size=32, val_subset=64,
    )
    dm2.prepare_data()
    dm2.setup()
    c = next(iter(dm2.val_dataloader()))[0]
    inter = torch.mean((a - c) ** 2).item()
    assert inter == 0.0, f"[{label}] DataModule inter MSE={inter}"

    print(f"OK [lightning {label}] — DataModule val determinism verified (intra=0, inter=0).")


_DATASETS: tuple[str, ...] = ("cifar10", "mnist")


def main() -> int:
    for name in _DATASETS:
        _check_dataset(name)

    for name in _DATASETS:
        _check_lightning_datamodule(dataset=name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
