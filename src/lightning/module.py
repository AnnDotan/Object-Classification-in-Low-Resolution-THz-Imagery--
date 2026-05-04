"""LightningModule wrapping ResNet50 / DenseNet121 / TransNeXt Micro.

Mirrors the semantics of the hand-rolled training loop in src/runner.py:
- Same model factories (timm + create_transnext_model)
- Same differential-LR param groups when backbone_lr is set
- Same Mixup / CutMix wiring (timm.data.Mixup)
- Same label smoothing, weight decay, gradient clipping (clipping is on Trainer)
- Cosine LR schedule with optional warmup (LambdaLR replicates timm's warmup_prefix curve)
"""
from __future__ import annotations

import math
from typing import Optional

import pytorch_lightning as pl
import timm
import torch
import torchmetrics
from timm.data import Mixup
from torch import nn

from src.models.transnext_wrapper import create_transnext_model


def _warmup_cosine_lambda(epochs: int, warmup_epochs: int, lr_min_ratio: float = 1e-2):
    """Return a function(epoch) -> multiplier that mirrors timm's CosineLRScheduler
    with warmup_prefix=True and lr_min=1e-5 for a base LR of 1e-3 (ratio 1e-2)."""
    warmup = max(0, int(warmup_epochs))
    total = max(1, int(epochs))

    def _lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine

    return _lr_lambda


class THzClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 10,
        pretrained: bool = True,
        lr: float = 1e-3,
        backbone_lr: Optional[float] = None,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
        max_grad_norm: float = 0.0,
        scheduler_type: str = "none",
        epochs: int = 30,
        warmup_epochs: int = 0,
        freeze_backbone: bool = False,
        drop_path_rate: float = 0.0,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        pos_bias_interp: str = "bilinear",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model()

        if mixup_alpha > 0 or cutmix_alpha > 0:
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=label_smoothing,
                num_classes=num_classes,
            )
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.mixup_fn = None
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc5 = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, top_k=min(5, num_classes)
        )

    def _build_model(self) -> nn.Module:
        TRANSNEXT_NAMES = {"transnext_micro", "transnext_tiny", "transnext_small", "transnext_base"}
        if self.hparams.model_name in TRANSNEXT_NAMES:
            transnext_ckpt = None
            if self.hparams.pretrained:
                from pathlib import Path
                from src.models.transnext_weights import find_or_download_weights
                project_root = Path(__file__).resolve().parents[2]
                weights_dir = project_root / "artifacts" / "weights"
                # find_or_download_weights raises FileNotFoundError with a clear
                # remediation message if neither local file nor download work.
                transnext_ckpt = str(
                    find_or_download_weights(
                        self.hparams.model_name, weights_dir=weights_dir
                    )
                )
            model = create_transnext_model(
                model_name=self.hparams.model_name,
                num_classes=self.hparams.num_classes,
                pretrained=self.hparams.pretrained,
                checkpoint_path=transnext_ckpt,
                drop_path_rate=self.hparams.drop_path_rate,
            )
            if self.hparams.pos_bias_interp != "bilinear":
                self._retarget_pos_bias_interp(model, mode=self.hparams.pos_bias_interp)
        else:
            model = timm.create_model(
                self.hparams.model_name,
                pretrained=self.hparams.pretrained,
                num_classes=self.hparams.num_classes,
            )

        if self.hparams.freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
            if hasattr(model, "head"):
                for p in model.head.parameters():
                    p.requires_grad = True
            else:
                raise RuntimeError("freeze_backbone=True but model has no attribute 'head'")
        return model

    @staticmethod
    def _retarget_pos_bias_interp(model: nn.Module, mode: str) -> None:
        """Hook for OPTIMIZER agent's positional-bias interpolation work.

        TransNeXt's relative positional bias is interpolated from the pretrained
        14x14 token grid when the runtime grid differs. The default in the upstream
        repo is bilinear; this hook lets Optuna sweep {bilinear, bicubic, nearest}.
        """
        for m in model.modules():
            if hasattr(m, "pos_bias_interp_mode"):
                m.pos_bias_interp_mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        if self.mixup_fn is None:
            self.train_acc.update(logits, y)
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.val_acc.update(logits, y)
        self.val_acc5.update(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_acc5", self.val_acc5, on_epoch=True)

    def configure_optimizers(self):
        if self.hparams.backbone_lr is not None:
            backbone_params = [
                p for n, p in self.model.named_parameters()
                if "head" not in n and p.requires_grad
            ]
            head_params = [
                p for n, p in self.model.named_parameters()
                if "head" in n and p.requires_grad
            ]
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": self.hparams.backbone_lr,
                 "weight_decay": self.hparams.weight_decay},
                {"params": head_params, "lr": self.hparams.lr,
                 "weight_decay": self.hparams.weight_decay},
            ])
        else:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )

        if self.hparams.scheduler_type == "cosine":
            lr_lambda = _warmup_cosine_lambda(
                epochs=self.hparams.epochs,
                warmup_epochs=self.hparams.warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        return optimizer
