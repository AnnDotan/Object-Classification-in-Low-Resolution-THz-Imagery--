# src/data/datasets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .degrade import DegradeConfig, degrade_image


@dataclass
class DataConfig:
    dataset: str = "cifar10"  # "cifar10" only for now
    root: str = "./data"
##################################################################################Change to 224 after training
    #out_size: int = 224
    out_size: int = 32
    low_res: int = 16
    train: bool = True
    download: bool = True


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

        self.deg_cfg = DegradeConfig(
            low_res=cfg.low_res,
            out_size=cfg.out_size,
        )

        self.norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        )
    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        x, y = self.ds[idx]  # x: [3,32,32] float in [0,1]
        x_deg = degrade_image(x, self.deg_cfg)
        x_deg = self.norm(x_deg)
        return x_deg, y


