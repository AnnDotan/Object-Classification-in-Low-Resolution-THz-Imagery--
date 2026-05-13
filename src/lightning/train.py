"""Lightning entry point. Provides a run_experiment(...) shim with the same
keyword surface as src.runner.run_experiment so existing callers (run_systematic.py,
run_all_phases.py) can dispatch via --engine lightning.

Reuses the legacy artifact schema so dashboards keep working without modification.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# Disable HF symlink warning on Windows (must be set early)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", message=r"dtype\(\): align")

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from .module import THzClassifier
from .datamodule import THzDataModule
from .callbacks import (
    DashboardRefreshCallback,
    HistoryJSONCallback,
    LegacyCheckpointCallback,
    LegacyJSONMetricsCallback,
    LegacyMetricsCSVCallback,
    LogSampleImagesCallback,
)


def _format_lr_for_name(lr: float) -> str:
    if lr < 1e-2:
        return f"{lr:.0e}"
    return str(lr).replace(".", "p")


def _make_unique_run_dir(base_dir: Path, base_name: str) -> Path:
    candidate = base_dir / base_name
    if not candidate.exists():
        return candidate
    version = 2
    while True:
        candidate = base_dir / f"{base_name}__v{version}"
        if not candidate.exists():
            return candidate
        version += 1


def _write_run_config(run_dir: Path, config: dict) -> None:
    with open(run_dir / "run_config.txt", "w", encoding="utf-8") as f:
        for k, v in config.items():
            f.write(f"{k}={v}\n")


def _build_wandb_logger(run_name: str, run_dir: Path):
    """Construct WandbLogger in offline mode by default. Returns None on import error."""
    try:
        from pytorch_lightning.loggers import WandbLogger
    except Exception:
        return None
    os.environ.setdefault("WANDB_MODE", "offline")
    wandb_dir = Path("wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)
    try:
        return WandbLogger(
            project="thz-classification",
            name=run_name,
            save_dir=str(wandb_dir),
            offline=os.environ.get("WANDB_MODE", "offline") == "offline",
            log_model=False,
        )
    except Exception as e:
        print(f"[WARN] WandbLogger init failed ({e}); falling back to CSV-only logging.")
        return None


def run_experiment(
    model_name: str,
    pretrained: bool,
    out_size: int,
    low_res: int,
    epochs: int,
    batch_size: int,
    train_subset: int,
    val_subset: int,
    lr: float = 1e-3,
    tag: str = "",
    group: str = "pilot",
    freeze_backbone: bool = False,
    backbone_lr: Optional[float] = None,
    degradation_type: str = "all",
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    scheduler_type: str = "none",
    warmup_epochs: int = 0,
    max_grad_norm: float = 0.0,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    drop_path_rate: float = 0.0,
    blur_kernel: Optional[int] = None,
    blur_sigma: Optional[float] = None,
    gaussian_noise_std: Optional[float] = None,
    salt_pepper_amount: Optional[float] = None,
    p_grayscale: Optional[float] = None,
    saturation: Optional[float] = None,
    early_stopping_patience: int = 0,
    dataset: str = "cifar10",
    pos_bias_interp: str = "bilinear",
    img_size: int = 224,
    patch_size: int = 4,
    pretrain_size: Optional[int] = None,
    compile_mode: str = "none",
    precision: Optional[str] = None,
    run_name_override: Optional[str] = None,
    num_workers: Optional[int] = None,  # US-043: None -> auto-pick on CUDA, 0 on CPU
):
    # All models — CNNs and TransNeXt alike — train at 224x224. The data
    # pipeline upsamples CIFAR (32) / MNIST (28) to out_size before reaching
    # the model, and TransNeXt's pretrained checkpoints assume img_size=224.
    if model_name.startswith("transnext_") and out_size != img_size:
        raise ValueError(
            f"TransNeXt requires out_size == img_size; got out_size={out_size} "
            f"img_size={img_size}. Set --img_size to match --out_size."
        )

    pl.seed_everything(42, workers=True)

    if run_name_override:
        run_name = run_name_override
    else:
        tag_prefix = f"{tag}_" if tag else ""
        weights_tag = "pt" if pretrained else "scratch"
        lr_str = _format_lr_for_name(lr)
        run_name = (
            f"{tag_prefix}{model_name}_{weights_tag}_out{out_size}_"
            f"lowres{low_res}_lr{lr_str}"
        )

    base_group_dir = Path("runs") / group
    base_group_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _make_unique_run_dir(base_group_dir, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    run_name = run_dir.name

    config_snapshot = {
        "engine": "lightning",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_name": model_name,
        "pretrained": pretrained,
        "out_size": out_size,
        "low_res": low_res,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_subset": train_subset,
        "val_subset": val_subset,
        "lr": lr,
        "backbone_lr": backbone_lr,
        "degradation_type": degradation_type,
        "tag": tag,
        "group": group,
        "run_name": run_name,
        "freeze_backbone": freeze_backbone,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "scheduler_type": scheduler_type,
        "warmup_epochs": warmup_epochs,
        "max_grad_norm": max_grad_norm,
        "mixup_alpha": mixup_alpha,
        "cutmix_alpha": cutmix_alpha,
        "drop_path_rate": drop_path_rate,
        "blur_kernel": blur_kernel,
        "blur_sigma": blur_sigma,
        "gaussian_noise_std": gaussian_noise_std,
        "salt_pepper_amount": salt_pepper_amount,
        "p_grayscale": p_grayscale,
        "saturation": saturation,
        "early_stopping_patience": early_stopping_patience,
        "dataset": dataset,
        "pos_bias_interp": pos_bias_interp,
        "img_size": img_size,
        "patch_size": patch_size,
        "pretrain_size": pretrain_size,
        "compile_mode": compile_mode,
        "precision": precision,
        "seed": 42,
    }
    _write_run_config(run_dir, config_snapshot)

    log_path = run_dir / "log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[lightning] run_name={run_name}\n")
        f.write(f"[lightning] dataset={dataset} group={group}\n")
        f.write(f"[lightning] model={model_name} pretrained={pretrained}\n")

    # US-043: auto-pick num_workers on CUDA (operator mandate: high throughput,
    # >=90% GPU utilization on the 32->224 bicubic upsample). On CPU, keep 0
    # for test determinism + Windows fork safety.
    if num_workers is None:
        if torch.cuda.is_available():
            import os
            cpu_count = os.cpu_count() or 4
            num_workers = min(max(cpu_count // 2, 2), 8)
        else:
            num_workers = 0

    dm = THzDataModule(
        dataset=dataset,
        out_size=out_size,
        low_res=low_res,
        batch_size=batch_size,
        train_subset=train_subset,
        val_subset=val_subset,
        degradation_type=degradation_type,
        blur_kernel=blur_kernel,
        blur_sigma=blur_sigma,
        gaussian_noise_std=gaussian_noise_std,
        salt_pepper_amount=salt_pepper_amount,
        p_grayscale=p_grayscale,
        saturation=saturation,
        num_workers=num_workers,
    )

    model = THzClassifier(
        model_name=model_name,
        num_classes=10,
        pretrained=pretrained,
        lr=lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        max_grad_norm=max_grad_norm,
        scheduler_type=scheduler_type,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        freeze_backbone=freeze_backbone,
        drop_path_rate=drop_path_rate,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        pos_bias_interp=pos_bias_interp,
        img_size=img_size,
        patch_size=patch_size,
        pretrain_size=pretrain_size,
        compile_mode=compile_mode,
    )

    run_meta = {
        "model_name": model_name,
        "pretrained": pretrained,
        "out_size": out_size,
        "low_res": low_res,
        "lr": lr,
        "group": group,
        "run_name": run_name,
    }

    callbacks = [
        LegacyMetricsCSVCallback(run_dir),
        LegacyJSONMetricsCallback(run_dir),
        # US-018: per-epoch learning curves for the dashboard's lazy Plotly drawer.
        HistoryJSONCallback(run_dir, tag=run_name),
        LegacyCheckpointCallback(run_dir, run_meta),
        LogSampleImagesCallback(run_dir),
        DashboardRefreshCallback(),
    ]
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(monitor="val_acc", mode="max", patience=early_stopping_patience)
        )

    csv_logger = CSVLogger(save_dir=str(run_dir), name="lightning_csv")
    loggers = [csv_logger]
    wandb_logger = _build_wandb_logger(run_name=run_name, run_dir=run_dir)
    if wandb_logger is not None:
        loggers.append(wandb_logger)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # Precision: explicit `precision` arg wins. Default per US-043 operator
    # mandate: bf16-mixed on CUDA (Blackwell sm_120 native; no loss scaler;
    # fp32 dynamic range avoids fp16 underflow). 32-true on CPU.
    resolved_precision = precision or ("bf16-mixed" if accelerator == "gpu" else "32-true")

    # US-043 operator mandate: enable TF32 on the FP32 matmul path. Free 1.3x
    # throughput on Blackwell with no accuracy regression validated by the
    # determinism gate. Idempotent — safe to call before every Trainer.
    if accelerator == "gpu":
        torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=1,
        precision=resolved_precision,
        logger=loggers,
        callbacks=callbacks,
        gradient_clip_val=(max_grad_norm if max_grad_norm and max_grad_norm > 0 else None),
        gradient_clip_algorithm="norm",
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=10,
        deterministic="warn",
    )

    trainer.fit(model, datamodule=dm)

    try:
        from src.tools.visualize_run import RunVisualizer
        RunVisualizer(run_dir).generate_all()
    except Exception as e:
        print(f"[WARN] RunVisualizer failed: {e}")

    return run_dir


def main() -> None:
    """CLI entry mirroring the legacy src.runner argparse surface."""
    import argparse

    p = argparse.ArgumentParser(description="THz classification — Lightning engine")
    p.add_argument("--model", required=True)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--out_size", type=int, default=32)
    p.add_argument("--low_res", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--train_subset", type=int, default=2000)
    p.add_argument("--val_subset", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone_lr", type=float, default=None)
    p.add_argument("--degradation_type", type=str, default="all")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--group", type=str, default="pilot", choices=["pilot", "official", "systematic"])
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--scheduler_type", type=str, default="none", choices=["none", "cosine"])
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--cutmix_alpha", type=float, default=0.0)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--blur_kernel", type=int, default=None)
    p.add_argument("--blur_sigma", type=float, default=None)
    p.add_argument("--gaussian_noise_std", type=float, default=None)
    p.add_argument("--salt_pepper_amount", type=float, default=None)
    p.add_argument("--p_grayscale", type=float, default=None)
    p.add_argument("--early_stopping_patience", type=int, default=0)
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist"])
    p.add_argument("--pos_bias_interp", type=str, default="bilinear",
                   choices=["bilinear", "bicubic", "nearest"])
    # V3 native-res / Blackwell knobs. Defaults preserve legacy behaviour;
    # All models train at 224x224.
    p.add_argument("--img_size", type=int, default=224,
                   help="Spatial dim the model expects (always 224).")
    p.add_argument("--patch_size", type=int, default=4,
                   help="TransNeXt stage-1 stride (always 4 at 224).")
    p.add_argument("--pretrain_size", type=int, default=None,
                   help="CPB coord scale; defaults to 224 if pretrained else img_size.")
    p.add_argument("--compile_mode", type=str, default="none",
                   choices=["none", "default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode. 'none' disables compile.")
    p.add_argument("--precision", type=str, default=None,
                   choices=["16-mixed", "bf16-mixed", "32-true", "16-true", "bf16-true"],
                   help="Trainer precision. Default: bf16-mixed on CUDA, 32-true on CPU.")
    args = p.parse_args()

    run_experiment(
        model_name=args.model,
        pretrained=args.pretrained,
        out_size=args.out_size,
        low_res=args.low_res,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        degradation_type=args.degradation_type,
        tag=args.tag,
        group=args.group,
        freeze_backbone=args.freeze_backbone,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        drop_path_rate=args.drop_path_rate,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        gaussian_noise_std=args.gaussian_noise_std,
        salt_pepper_amount=args.salt_pepper_amount,
        p_grayscale=args.p_grayscale,
        early_stopping_patience=args.early_stopping_patience,
        dataset=args.dataset,
        pos_bias_interp=args.pos_bias_interp,
        img_size=args.img_size,
        patch_size=args.patch_size,
        pretrain_size=args.pretrain_size,
        compile_mode=args.compile_mode,
        precision=args.precision,
    )


if __name__ == "__main__":
    sys.exit(main())
