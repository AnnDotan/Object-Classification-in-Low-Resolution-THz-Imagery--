"""Optuna study for TransNeXt Micro on CIFAR-10 (Level 2 Moderate degradation).

Targets the largest measured CNN-vs-ViT gap:
    DenseNet121: 80.4%   |   TransNeXt Micro: 63.6%   (CIFAR-10 L2)

Search space includes the OPTIMIZER agent's two declared research targets:
    pos_bias_interp  -- {bilinear, bicubic, nearest}  (positional-bias interpolation)
    backbone_lr      -- frozen vs partial unfreeze    (resolution-discrepancy mitigation)

Storage: SQLite at artifacts/optuna_thz.db so the study survives interruptions.
Default budget: 50 trials x 10 epochs each (per the approved plan).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Disable HF symlink warning on Windows (must be set early)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import optuna
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from src.lightning.datamodule import THzDataModule
from src.lightning.module import THzClassifier


# Level 2 (Moderate) degradation per CLAUDE.md
L2_DEG = dict(
    out_size=224,
    low_res=16,
    blur_kernel=5,
    blur_sigma=1.0,
    gaussian_noise_std=0.08,
    salt_pepper_amount=0.05,
    p_grayscale=0.3,
)


def build_objective(max_epochs: int, train_subset: int, val_subset: int):
    def objective(trial: optuna.Trial) -> float:
        freeze_backbone = trial.suggest_categorical("freeze_backbone", [True, False])
        if freeze_backbone:
            backbone_lr = None
        else:
            backbone_lr = trial.suggest_float("backbone_lr", 1e-7, 1e-3, log=True)

        hp = dict(
            lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            backbone_lr=backbone_lr,
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            drop_path_rate=trial.suggest_float("drop_path_rate", 0.0, 0.3),
            label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.2),
            warmup_epochs=trial.suggest_categorical("warmup_epochs", [0, 1, 2, 3]),
            pos_bias_interp=trial.suggest_categorical(
                "pos_bias_interp", ["bilinear", "bicubic", "nearest"]
            ),
        )
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = THzClassifier(
            model_name="transnext_micro",
            num_classes=10,
            pretrained=True,
            epochs=max_epochs,
            scheduler_type="cosine",
            max_grad_norm=1.0,
            freeze_backbone=freeze_backbone,
            **hp,
        )
        dm = THzDataModule(
            dataset="cifar10",
            degradation_type="all",
            batch_size=batch_size,
            train_subset=train_subset,
            val_subset=val_subset,
            **L2_DEG,
        )

        wandb_logger = None
        try:
            from pytorch_lightning.loggers import WandbLogger
            wandb_logger = WandbLogger(
                project="thz-tune",
                name=f"trial_{trial.number}",
                save_dir="wandb/",
                offline=True,
                log_model=False,
            )
        except Exception:
            wandb_logger = None

        precision = "16-mixed" if pl.pytorch.accelerators.cuda.CUDAAccelerator.is_available() else "32-true"
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            precision=precision,
            logger=wandb_logger if wandb_logger is not None else False,
            enable_progress_bar=False,
            enable_model_summary=False,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            callbacks=[
                EarlyStopping(monitor="val_acc", mode="max", patience=3),
                PyTorchLightningPruningCallback(trial, monitor="val_acc"),
            ],
        )
        trainer.fit(model, datamodule=dm)
        val_acc = trainer.callback_metrics.get("val_acc", None)
        return float(val_acc) if val_acc is not None else 0.0

    return objective


def main() -> None:
    p = argparse.ArgumentParser(description="Optuna study: TransNeXt Micro on CIFAR-10 L2")
    p.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials (plan default: 50).")
    p.add_argument("--max-epochs", type=int, default=10, help="Max epochs per trial.")
    p.add_argument("--train-subset", type=int, default=10000)
    p.add_argument("--val-subset", type=int, default=5000)
    p.add_argument("--study-name", type=str,
                   default="transnext_cifar10_l2_2026-04-11")
    p.add_argument("--storage", type=str,
                   default="sqlite:///artifacts/optuna_thz.db")
    args = p.parse_args()

    Path("artifacts").mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    objective = build_objective(
        max_epochs=args.max_epochs,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
    )
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print("\n=== OPTUNA BEST ===")
    print(f"Best val_acc: {study.best_value:.4f}")
    print(f"Best params:  {json.dumps(study.best_params, indent=2)}")
    out_path = Path("artifacts") / "optuna_best_params.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "study_name": args.study_name,
        }, f, indent=2)
    print(f"Saved best params -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
