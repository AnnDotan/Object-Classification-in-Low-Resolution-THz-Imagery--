"""LightningDataModule wrapping the existing degradation pipeline.

Preserves verbatim:
- DegradeConfig parameter passing from src/data/datasets.py
- ImageNet normalization order (after degradation)
- MNIST grayscale -> 3-channel repeat before degradation
- num_workers=0 default (matches legacy runner)
- val_loader batch_size = max(batch_size, 64)
"""
from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from src.data.datasets import DataConfig, THzLikeCIFAR10, THzLikeMNIST


def _dataset_class(name: str):
    return THzLikeMNIST if name == "mnist" else THzLikeCIFAR10


class THzDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str = "cifar10",
        out_size: int = 224,
        low_res: int = 16,
        batch_size: int = 32,
        train_subset: int = 0,
        val_subset: int = 0,
        degradation_type: str = "all",
        blur_kernel: Optional[int] = None,
        blur_sigma: Optional[float] = None,
        gaussian_noise_std: Optional[float] = None,
        salt_pepper_amount: Optional[float] = None,
        p_grayscale: Optional[float] = None,
        saturation: Optional[float] = None,
        root: str = "./data",
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _build_cfg(self, train: bool) -> DataConfig:
        kw = dict(
            dataset=self.hparams.dataset,
            train=train,
            out_size=self.hparams.out_size,
            low_res=self.hparams.low_res,
            root=self.hparams.root,
            degradation_type=self.hparams.degradation_type,
        )
        for k in ("blur_kernel", "blur_sigma", "gaussian_noise_std",
                  "salt_pepper_amount", "p_grayscale", "saturation"):
            v = getattr(self.hparams, k)
            if v is not None:
                kw[k] = v
        return DataConfig(**kw)

    def prepare_data(self) -> None:
        Cls = _dataset_class(self.hparams.dataset)
        # one-shot download
        Cls(self._build_cfg(train=True))
        Cls(self._build_cfg(train=False))

    def setup(self, stage: Optional[str] = None) -> None:
        Cls = _dataset_class(self.hparams.dataset)
        self.train_ds = Cls(self._build_cfg(train=True))
        self.val_ds = Cls(self._build_cfg(train=False))
        if self.hparams.train_subset > 0:
            self.train_ds = Subset(
                self.train_ds, range(min(self.hparams.train_subset, len(self.train_ds)))
            )
        if self.hparams.val_subset > 0:
            self.val_ds = Subset(
                self.val_ds, range(min(self.hparams.val_subset, len(self.val_ds)))
            )

    def _loader_extras(self) -> dict:
        """US-043 DataLoader throughput knobs (operator mandate 2026-05-12).

        pin_memory and persistent_workers are only meaningful with
        num_workers > 0. The CPU-only test path keeps num_workers=0, so we
        guard both flags on that condition to avoid PyTorch's "persistent
        workers requires num_workers>0" hard error.
        """
        if self.hparams.num_workers > 0:
            return dict(pin_memory=True, persistent_workers=True, prefetch_factor=2)
        return dict(pin_memory=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            **self._loader_extras(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=max(self.hparams.batch_size, 64),
            shuffle=False,
            num_workers=self.hparams.num_workers,
            **self._loader_extras(),
        )
