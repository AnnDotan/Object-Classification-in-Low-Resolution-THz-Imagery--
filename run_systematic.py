#!/usr/bin/env python3
"""
Systematic Degradation Experiments
====================================

For each degradation level combination, run all 3 models with
optimized hyperparameters and overfitting prevention.

Degradation Levels (combined pipeline):
  Level 1 (Mild):     low_res=16, blur_kernel=3, blur_sigma=0.5, noise_std=0.04, salt_pepper=0.02
  Level 2 (Moderate):  low_res=16, blur_kernel=5, blur_sigma=1.0, noise_std=0.08, salt_pepper=0.05
  Level 3 (Severe):    low_res=8,  blur_kernel=7, blur_sigma=1.5, noise_std=0.12, salt_pepper=0.08

Overfitting Prevention:
  - Early stopping (patience=5 epochs)
  - Cosine LR scheduler
  - Weight decay 1e-4
  - Gradient clipping (max_norm=1.0)
  - Label smoothing 0.1 (ResNet50/DenseNet121)
  - TransNeXt: frozen backbone (linear probe) to prevent overfitting

Models:
  1. ResNet50       - differential LR fine-tuning
  2. DenseNet121    - differential LR fine-tuning
  3. TransNeXt Micro - linear probe (frozen backbone)

Usage:
  python run_systematic.py --level 2 --mode pilot     # quick test, level 2
  python run_systematic.py --level all --mode full     # all levels, full training
  python run_systematic.py --level 1 --mode full       # level 1 only, full training
"""

import os
import sys
import csv
import time
import json
import argparse
from pathlib import Path

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def _resolve_run_experiment(engine: str):
    """Pick the legacy or Lightning training entry. Same call signature."""
    if engine == "lightning":
        from src.lightning.train import run_experiment
    else:
        from src.runner import run_experiment
    return run_experiment

# ============================================================
# DEGRADATION LEVELS
# ============================================================

DEGRADATION_LEVELS = {
    1: {
        "name": "mild",
        "low_res": 16,
        "blur_kernel": 3,
        "blur_sigma": 0.5,
        "gaussian_noise_std": 0.04,
        "salt_pepper_amount": 0.02,
        "p_grayscale": 0.3,
    },
    2: {
        "name": "moderate",
        "low_res": 16,
        "blur_kernel": 5,
        "blur_sigma": 1.0,
        "gaussian_noise_std": 0.08,
        "salt_pepper_amount": 0.05,
        "p_grayscale": 0.3,
    },
    3: {
        "name": "severe",
        "low_res": 8,
        "blur_kernel": 7,
        "blur_sigma": 1.5,
        "gaussian_noise_std": 0.12,
        "salt_pepper_amount": 0.08,
        "p_grayscale": 0.3,
    },
}

# ============================================================
# MODEL CONFIGURATIONS (with overfitting prevention)
# ============================================================

MODEL_CONFIGS = [
    # ResNet50 — TResNet/EfficientNetV2 paper recommendations
    {
        "model_name": "resnet50",
        "lr": 1e-3,
        "backbone_lr": 5e-5,          # was 1e-4: slower adaptation, less forgetting
        "freeze_backbone": False,
        "weight_decay": 5e-4,          # was 1e-4: stronger L2 regularization
        "label_smoothing": 0.15,       # was 0.1: moderate increase
        "warmup_epochs": 3,            # EfficientNetV2 warmup recommendation
    },
    # DenseNet121 — DenseNet paper recommendations
    {
        "model_name": "densenet121",
        "lr": 1e-3,
        "backbone_lr": 5e-5,          # was 1e-4: more conservative
        "freeze_backbone": False,
        "weight_decay": 5e-4,          # was 1e-4: stronger regularization
        "label_smoothing": 0.2,        # was 0.1: strong smoothing for 99%+ train acc
        "warmup_epochs": 2,            # shorter warmup (DenseNet converges fast)
    },
    # TransNeXt Micro — TransNeXt paper recommendations
    {
        "model_name": "transnext_micro",
        "lr": 1e-3,
        "backbone_lr": 1e-5,           # was None/frozen: very conservative unfreeze
        "freeze_backbone": False,      # was True: allow backbone adaptation
        "weight_decay": 0.05,          # TransNeXt paper fine-tuning value
        "label_smoothing": 0.1,        # was 0.0: light smoothing for fine-tuning
        "warmup_epochs": 5,            # TransNeXt paper training protocol
        "drop_path_rate": 0.1,         # TransNeXt paper stochastic depth
    },
]

# ============================================================
# TRAINING MODES
# ============================================================

PILOT_SETTINGS = {
    "epochs": 5,
    "train_subset": 2000,
    "val_subset": 1000,
}

FULL_SETTINGS = {
    "epochs": 30,
    "train_subset": 10000,
    "val_subset": 5000,
}

COMMON = {
    "pretrained": True,
    "out_size": 224,
    "batch_size": 32,
    "degradation_type": "all",
    "scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "group": "systematic",
}


# ============================================================
# 186-CELL CAMPAIGN: single-cell dispatch (US-008)
# ============================================================

# Mode -> training settings for the final campaign. `pilot` is for ad-hoc
# smoke tests only; run_all_phases.py rejects it for --plan final.
FINAL_PILOT = {
    "epochs": 5,
    "train_subset": 2000,
    "val_subset": 1000,
    "early_stopping_patience": 2,
}
FINAL_FULL = {
    "epochs": 60,                    # CLAUDE.md: max_epochs 60
    "train_subset": 10000,
    "val_subset": 5000,
    "early_stopping_patience": 10,   # CLAUDE.md: patience 10, min_delta 1e-4
}


def _load_best_hparams(model: str, dataset: str) -> dict:
    """Load Optuna winner JSON for a (model, dataset) pair."""
    path = Path("artifacts/best_hparams") / f"{model}_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing best_hparams: {path}\n"
            f"  Run `python tune_all.py --n-trials 20 --model {model} --dataset {dataset}` "
            f"first, or `python tune_all.py --n-trials 20` for all 6 pairs."
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    if "best_params" not in blob:
        raise ValueError(
            f"{path}: missing 'best_params' key — file may be from an interrupted study"
        )
    return blob


def _cell_config(spec, hparams: dict, mode: str) -> dict:
    """Build run_experiment kwargs from a CellSpec + best_hparams.

    The DegradeConfig fields override the legacy CLI flag names because
    that's what THzDataModule consumes. `saturation` is included now that
    train.py forwards it to THzDataModule (US-008 prerequisite).
    """
    settings = FINAL_PILOT if mode == "pilot" else FINAL_FULL
    deg = spec.degrade_config
    bp = hparams["best_params"]

    return {
        # Model + training
        "model_name": spec.model,
        "pretrained": True,
        "out_size": 224,
        "batch_size": 32,
        "freeze_backbone": False,        # all 3 models full FT in final campaign
        "scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "lr": float(bp["head_lr"]),
        "backbone_lr": float(bp["backbone_lr"]),
        "weight_decay": float(bp["weight_decay"]),
        "label_smoothing": float(bp["label_smoothing"]),
        "warmup_epochs": int(round(float(bp["warmup_epochs"]))),

        # Degradation (from CellSpec.degrade_config)
        "low_res": deg.low_res,
        "blur_kernel": deg.blur_kernel,
        "blur_sigma": deg.blur_sigma,
        "gaussian_noise_std": deg.gaussian_noise_std,
        "salt_pepper_amount": deg.salt_pepper_amount,
        "saturation": deg.saturation,
        "degradation_type": deg.degradation_type,

        # Bookkeeping
        "tag": spec.tag,
        "dataset": spec.dataset,
        "group": "final",
        "run_name_override": spec.tag,    # forces runs/final/<tag>/

        # Phase-aware training schedule
        "epochs": settings["epochs"],
        "train_subset": settings["train_subset"],
        "val_subset": settings["val_subset"],
        "early_stopping_patience": settings["early_stopping_patience"],
    }


def _merge_metadata_into_metrics_json(
    run_dir: Path, spec, hparams: dict,
) -> None:
    """Extend metrics.json with hparams + cell metadata for round-trip reproducibility.

    Acceptance: "All hparam values logged to metrics.json under a `hparams` key."
    """
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        return  # training crashed before LegacyJSONMetricsCallback fired
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["hparams"] = hparams.get("best_params", {})
    metrics["hparams_source"] = {
        "study_name": hparams.get("study_name"),
        "best_value": hparams.get("best_value"),
        "n_trials_completed": hparams.get("n_trials_completed"),
        "priors_file_hash": hparams.get("priors_file_hash"),
    }
    metrics["cell_tag"] = spec.tag
    metrics["phase"] = spec.phase
    metrics["level"] = spec.level
    metrics["axis"] = spec.axis
    metrics["model"] = spec.model
    metrics["dataset"] = spec.dataset
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_cell(
    cell_tag: str,
    mode: str = "full",
    engine: str = "lightning",
    *,
    run_experiment_fn=None,         # injectable for testing
):
    """Dispatch a single cell from the 186-cell matrix.

    Resolves the tag against build_final_matrix(), loads the (model, dataset)
    Optuna winner, calls run_experiment with run_name_override=tag so output
    lands at runs/final/<tag>/, then merges hparams + cell metadata into
    metrics.json for downstream dashboards.
    """
    from src.experiments.matrix import cells_by_tag

    by_tag = cells_by_tag()
    if cell_tag not in by_tag:
        raise ValueError(
            f"unknown cell tag: {cell_tag!r}. "
            f"Expected one of the 186 tags from build_final_matrix()."
        )
    spec = by_tag[cell_tag]
    hparams = _load_best_hparams(spec.model, spec.dataset)
    config = _cell_config(spec, hparams, mode)

    run_experiment = run_experiment_fn or _resolve_run_experiment(engine)
    run_dir = run_experiment(**config)
    _merge_metadata_into_metrics_json(Path(run_dir), spec, hparams)
    return Path(run_dir)


def run_systematic(levels: list[int], mode: str = "pilot", dataset: str = "cifar10",
                   engine: str = "lightning"):
    """Run all 3 models for each specified degradation level."""
    settings = PILOT_SETTINGS if mode == "pilot" else FULL_SETTINGS
    ds_label = "CIFAR-10" if dataset == "cifar10" else "MNIST"
    run_experiment = _resolve_run_experiment(engine)

    total_experiments = len(levels) * len(MODEL_CONFIGS)

    print("\n" + "=" * 70)
    print(f"  SYSTEMATIC DEGRADATION EXPERIMENTS ({mode.upper()} MODE)")
    print("=" * 70)
    print(f"  Engine: {engine}")
    print(f"  Dataset: {ds_label}")
    print(f"  Degradation levels: {levels}")
    print(f"  Models: ResNet50, DenseNet121, TransNeXt Micro")
    print(f"  Epochs: {settings['epochs']}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Overfitting prevention: early stopping, cosine LR, weight decay,")
    print(f"                          gradient clipping, label smoothing")
    print("=" * 70)

    results = []
    exp_num = 0

    for level_id in levels:
        level = DEGRADATION_LEVELS[level_id]
        level_name = level["name"]

        print(f"\n{'━' * 70}")
        print(f"  DEGRADATION LEVEL {level_id} ({level_name.upper()})")
        print(f"  low_res={level['low_res']}, blur_kernel={level['blur_kernel']}, "
              f"blur_sigma={level['blur_sigma']}")
        print(f"  noise_std={level['gaussian_noise_std']}, "
              f"salt_pepper={level['salt_pepper_amount']}, "
              f"grayscale_p={level['p_grayscale']}")
        print(f"{'━' * 70}")

        for model_cfg in MODEL_CONFIGS:
            exp_num += 1
            model = model_cfg["model_name"]
            ds_suffix = f"_{dataset}" if dataset != "cifar10" else ""
            tag = f"sys_L{level_id}_{level_name}_{model}{ds_suffix}"

            print(f"\n{'─' * 70}")
            print(f"  [{exp_num}/{total_experiments}] {model} @ Level {level_id} ({level_name})")
            print(f"  LR={model_cfg['lr']}, backbone_lr={model_cfg.get('backbone_lr')}")
            print(f"  freeze={model_cfg['freeze_backbone']}, wd={model_cfg['weight_decay']}")
            print(f"{'─' * 70}\n")

            config = {
                **COMMON,
                **settings,
                **model_cfg,
                "low_res": level["low_res"],
                "blur_kernel": level["blur_kernel"],
                "blur_sigma": level["blur_sigma"],
                "gaussian_noise_std": level["gaussian_noise_std"],
                "salt_pepper_amount": level["salt_pepper_amount"],
                "p_grayscale": level["p_grayscale"],
                "early_stopping_patience": 5,
                "tag": tag,
                "dataset": dataset,
            }

            t0 = time.time()
            try:
                run_experiment(**config)
                dt = time.time() - t0
                results.append({
                    "level": level_id,
                    "level_name": level_name,
                    "model": model,
                    "tag": tag,
                    "status": "OK",
                    "time": f"{dt:.1f}s",
                })
            except Exception as e:
                dt = time.time() - t0
                print(f"[ERROR] {model} @ Level {level_id}: {e}")
                results.append({
                    "level": level_id,
                    "level_name": level_name,
                    "model": model,
                    "tag": tag,
                    "status": f"FAILED: {e}",
                    "time": f"{dt:.1f}s",
                })

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SYSTEMATIC EXPERIMENTS - RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Level':<10} {'Model':<20} {'Status':<10} {'Time':<10}")
    print(f"  {'─'*10} {'─'*20} {'─'*10} {'─'*10}")
    for r in results:
        icon = "✓" if r["status"] == "OK" else "✗"
        print(f"  {icon} L{r['level']} ({r['level_name']:<8}) {r['model']:<20} {r['status']:<10} {r['time']}")
    print("=" * 70)

    # Save summary JSON for dashboard
    summary_path = Path("artifacts") / "systematic_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Summary saved to {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Systematic degradation experiments")
    p.add_argument("--level", default="all",
                   help="Degradation level: 1, 2, 3, or 'all'")
    p.add_argument("--mode", default="pilot", choices=["pilot", "full"],
                   help="pilot (5 epochs, small data) or full (30 epochs)")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"],
                   help="Dataset to use: cifar10 or mnist")
    p.add_argument("--engine", default="lightning", choices=["lightning", "legacy"],
                   help="Training engine: lightning (default) or legacy hand-rolled trainer")
    p.add_argument(
        "--transnext_size",
        default="small",
        choices=["micro", "tiny", "small", "base"],
        help="TransNeXt size variant. Default 'small' (CLAUDE.md baseline); "
             "use 'base' for the final 186-cell campaign per US-006.",
    )
    p.add_argument(
        "--transnext_mode",
        default="ft",
        choices=["lp", "ft"],
        help="TransNeXt training mode: 'lp' (linear probe, frozen backbone) "
             "or 'ft' (full fine-tuning). Default 'ft' for the final campaign.",
    )
    p.add_argument(
        "--cell-tag",
        default=None,
        help="Run a single cell from the 186-cell matrix (US-008). Tag form: "
             "final_clean_{m}_{d} | final_B_L{l}_{m}_{d} | final_C_L{l}_{ax}_{m}_{d}. "
             "Loads hparams from artifacts/best_hparams/{m}_{d}.json. When set, "
             "the legacy --level loop is bypassed.",
    )
    args = p.parse_args()

    # Re-target the TransNeXt entry of MODEL_CONFIGS based on CLI flags.
    for _cfg in MODEL_CONFIGS:
        if _cfg["model_name"].startswith("transnext_"):
            _cfg["model_name"] = f"transnext_{args.transnext_size}"
            _cfg["freeze_backbone"] = (args.transnext_mode == "lp")
            break

    # 186-cell single-cell dispatch (US-008): --cell-tag bypasses the legacy loop.
    if args.cell_tag:
        run_cell(args.cell_tag, mode=args.mode, engine=args.engine)
        sys.exit(0)

    if args.level == "all":
        levels = [1, 2, 3]
    else:
        levels = [int(x) for x in args.level.split(",")]
        for lv in levels:
            if lv not in DEGRADATION_LEVELS:
                print(f"[ERROR] Invalid level {lv}. Choose from 1, 2, 3")
                sys.exit(1)

    run_systematic(levels, args.mode, args.dataset, args.engine)
