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
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

# Disable HF symlink warning on Windows (must be set early)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("WANDB_MODE", "offline")

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from src.data.degradation_levels import level_params
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

        # US-043: bf16-mixed default on CUDA (Blackwell sm_120 native); 32-true on CPU.
        cuda_avail = pl.pytorch.accelerators.cuda.CUDAAccelerator.is_available()
        precision = "bf16-mixed" if cuda_avail else "32-true"
        if cuda_avail:
            torch.set_float32_matmul_precision("high")  # TF32 on Blackwell
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


# ---------------------------------------------------------------------------
# Final-campaign runner (US-005): paper-anchored Optuna sweep at Phase B L3
# Moderate, persisted to artifacts/optuna_thz.db, one study per (model,dataset).
# ---------------------------------------------------------------------------

# Optuna trial param names. Stored in study DB; do not rename without resetting.
_PARAM_NAMES = (
    "head_lr",
    "backbone_lr",
    "weight_decay",
    "label_smoothing",
    "warmup_epochs",
)


def _suggest_from_priors(trial: optuna.Trial, priors: dict) -> dict:
    """Map a priors dict to Optuna trial.suggest_* calls.

    `warmup_epochs` is suggested as a float and rounded to int — Optuna's
    integer suggesters don't support priors-driven log distributions cleanly,
    and the search range is small (0..10) so float-then-round is fine.
    """
    hp = priors["hparams"]
    out: dict = {}
    for name in _PARAM_NAMES:
        spec = hp[name]
        dist = spec["distribution"]
        if dist in ("uniform", "loguniform"):
            val = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]),
                log=(dist == "loguniform"),
            )
        elif dist == "categorical":
            val = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"unsupported distribution {dist!r} for {name}")
        out[name] = val
    out["warmup_epochs"] = int(round(out["warmup_epochs"]))
    return out


def _l3_train_one_trial(
    *, model: str, dataset: str, params: dict,
    max_epochs: int, train_subset: int, val_subset: int,
) -> float:
    """Train one Optuna trial on Phase B L3 Moderate. Returns val_acc.

    Phase B L3: every degradation axis at level 3 — low_res=10, blur 7/1.30,
    noise 0.09, S&P 0.08, saturation 0.50 (per src/data/degradation_levels.py).
    """
    p = level_params(3, axis=None)
    pl.seed_everything(42, workers=True)

    classifier_kwargs = dict(
        model_name=model,
        num_classes=10,
        pretrained=True,
        lr=params["head_lr"],
        backbone_lr=params["backbone_lr"],
        weight_decay=params["weight_decay"],
        label_smoothing=params["label_smoothing"],
        warmup_epochs=params["warmup_epochs"],
        scheduler_type="cosine",
        max_grad_norm=1.0,
        epochs=max_epochs,
        freeze_backbone=False,  # full FT for all 3 models in the final campaign
    )
    classifier = THzClassifier(**classifier_kwargs)

    # US-043 (post-iter4b fix): force num_workers=0 in the Optuna trial loop.
    # Reason: a multi-trial in-process loop on Windows trips a Lightning
    # `combined_loader` + persistent_workers interaction at trial ~11
    # (RuntimeError: Please call iter(combined_loader) first.). The 2k-train
    # subset is small enough that single-process loading is not the
    # bottleneck — pre-crash throughput was ~24 it/s, compute-bound on bf16.
    # Production paths (run_experiment via run_cell / run_phase_a) keep the
    # auto-pick num_workers > 0 because they're single Trainer.fit calls.
    _nw = 0

    dm = THzDataModule(
        dataset=dataset,
        out_size=224,
        low_res=int(p["low_res"]),
        batch_size=32,
        train_subset=train_subset,
        val_subset=val_subset,
        degradation_type="all",
        blur_kernel=int(p["blur_kernel"]),
        blur_sigma=float(p["blur_sigma"]),
        gaussian_noise_std=float(p["noise_std"]),
        salt_pepper_amount=float(p["salt_pepper"]),
        saturation=float(p["saturation"]),
        num_workers=_nw,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    # US-043: bf16-mixed default on CUDA (Blackwell sm_120 native).
    precision = "bf16-mixed" if torch.cuda.is_available() else "32-true"
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")  # TF32 throughput on Blackwell
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[
            EarlyStopping(monitor="val_acc", mode="max", patience=3),
        ],
    )
    trainer.fit(classifier, datamodule=dm)
    val_acc = trainer.callback_metrics.get("val_acc", None)
    return float(val_acc) if val_acc is not None else 0.0


def _priors_file_hash(model: str, priors_dir: Optional[Path] = None) -> str:
    """SHA-256 of the priors file used for this study, recorded in winner JSON.

    Lets the campaign verify that hparams were actually selected from the
    priors checked into git at the time of the study.
    """
    priors_dir = priors_dir or Path("artifacts/priors")
    return hashlib.sha256(
        (priors_dir / f"{model}.json").read_bytes()
    ).hexdigest()


def run_studies(
    *,
    models: Iterable[str],
    datasets: Iterable[str],
    n_trials: int,
    storage: str,
    out_dir: Path,
    load_priors: Callable[[str], dict],
    train_fn: Optional[Callable[..., float]] = None,
    max_epochs: int = 5,
    train_subset: int = 2000,
    val_subset: int = 1000,
    priors_dir: Optional[Path] = None,
) -> int:
    """Run one Optuna study per (model, dataset) pair. Returns 0 on success.

    Resumable: each study's trials live in `storage` (SQLite) keyed by study_name,
    so re-running with the same model/dataset/storage continues from where it
    crashed. `train_fn` is injectable so tests can replace heavy training with
    a stub.
    """
    train_fn = train_fn or _l3_train_one_trial
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    for model in models:
        priors = load_priors(model)
        priors_hash = _priors_file_hash(model, priors_dir=priors_dir)

        for dataset in datasets:
            study_name = f"{model}_{dataset}_L3"
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42),
            )

            def objective(
                trial: optuna.Trial,
                _model=model, _dataset=dataset, _priors=priors,
            ) -> float:
                params = _suggest_from_priors(trial, _priors)
                return float(train_fn(
                    model=_model, dataset=_dataset, params=params,
                    max_epochs=max_epochs,
                    train_subset=train_subset, val_subset=val_subset,
                ))

            try:
                study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
            except Exception as e:
                failures.append(f"{study_name}: {type(e).__name__}: {e}")
                print(f"FAIL [{study_name}] {e}", file=sys.stderr)
                continue

            n_done = sum(1 for t in study.trials if t.state.is_finished())
            try:
                best_value = float(study.best_value)
                best_params = dict(study.best_params)
            except ValueError:
                best_value = float("nan")
                best_params = {}

            winner = {
                "model": model,
                "dataset": dataset,
                "study_name": study_name,
                "best_value": best_value,
                "best_params": best_params,
                "n_trials_completed": n_done,
                "priors_file_hash": priors_hash,
                "phase": "B",
                "level": 3,
            }
            target = out_dir / f"{model}_{dataset}.json"
            target.write_text(json.dumps(winner, indent=2), encoding="utf-8")
            print(
                f"OK [{study_name}] best_val_acc={best_value:.4f} "
                f"({n_done} trials) -> {target}"
            )

    if failures:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Legacy single-pair main() preserved below for backwards compat.
# ---------------------------------------------------------------------------


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
