# src/data/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .degrade import DegradeConfig, SEED_OFFSET_TRAIN, SEED_OFFSET_VAL, degrade_image


@dataclass
class DataConfig:
    dataset: str = "cifar10"  # "cifar10" or "mnist"
    root: str = "./data"
##################################################################################Change to 224 after training
    #out_size: int = 224
    out_size: int = 32
    low_res: int = 16
    train: bool = True
    download: bool = True
    degradation_type: str = "all"  # "all", "downsampling", "blur", "noise"
    # Optional custom degradation overrides (None = use DegradeConfig defaults)
    blur_kernel: int | None = None
    blur_sigma: float | None = None
    gaussian_noise_std: float | None = None
    salt_pepper_amount: float | None = None
    p_grayscale: float | None = None


class THzLikeCIFAR10(Dataset):
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

        # CIFAR-10 returns PIL images -> we convert to tensor in [0,1]
        self.base_tf = transforms.ToTensor()

        self.ds = datasets.CIFAR10(
            root=cfg.root,
            train=cfg.train,
            download=cfg.download,
            transform=self.base_tf,
        )

        deg_kwargs = dict(
            low_res=cfg.low_res,
            out_size=cfg.out_size,
            degradation_type=cfg.degradation_type,
        )
        # Apply custom degradation overrides if provided
        if cfg.blur_kernel is not None:
            deg_kwargs["blur_kernel"] = cfg.blur_kernel
        if cfg.blur_sigma is not None:
            deg_kwargs["blur_sigma"] = cfg.blur_sigma
        if cfg.gaussian_noise_std is not None:
            deg_kwargs["gaussian_noise_std"] = cfg.gaussian_noise_std
        if cfg.salt_pepper_amount is not None:
            deg_kwargs["salt_pepper_amount"] = cfg.salt_pepper_amount
        if cfg.p_grayscale is not None:
            deg_kwargs["p_grayscale"] = cfg.p_grayscale
        self.deg_cfg = DegradeConfig(**deg_kwargs)

        self.norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        )
    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.ds[idx]  # x: [3,32,32] float in [0,1]
        seed = idx + (SEED_OFFSET_TRAIN if self.cfg.train else SEED_OFFSET_VAL)
        x_deg = degrade_image(x, self.deg_cfg, seed=seed)
        x_deg = self.norm(x_deg)
        return x_deg, y


class THzLikeMNIST(Dataset):
    """MNIST with THz-like degradation pipeline.

    Grayscale images are repeated to 3 channels for pretrained backbone
    compatibility, then degraded identically to CIFAR-10.
    """

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

        self.base_tf = transforms.ToTensor()  # [1, 28, 28] float [0,1]

        self.ds = datasets.MNIST(
            root=cfg.root,
            train=cfg.train,
            download=cfg.download,
            transform=self.base_tf,
        )

        deg_kwargs = dict(
            low_res=cfg.low_res,
            out_size=cfg.out_size,
            degradation_type=cfg.degradation_type,
        )
        if cfg.blur_kernel is not None:
            deg_kwargs["blur_kernel"] = cfg.blur_kernel
        if cfg.blur_sigma is not None:
            deg_kwargs["blur_sigma"] = cfg.blur_sigma
        if cfg.gaussian_noise_std is not None:
            deg_kwargs["gaussian_noise_std"] = cfg.gaussian_noise_std
        if cfg.salt_pepper_amount is not None:
            deg_kwargs["salt_pepper_amount"] = cfg.salt_pepper_amount
        if cfg.p_grayscale is not None:
            deg_kwargs["p_grayscale"] = cfg.p_grayscale
        self.deg_cfg = DegradeConfig(**deg_kwargs)

        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.ds[idx]  # x: [1, 28, 28] float in [0,1]
        x = x.repeat(3, 1, 1)  # [3, 28, 28] — grayscale to 3-channel
        seed = idx + (SEED_OFFSET_TRAIN if self.cfg.train else SEED_OFFSET_VAL)
        x_deg = degrade_image(x, self.deg_cfg, seed=seed)
        x_deg = self.norm(x_deg)
        return x_deg, y


