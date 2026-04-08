#!/usr/bin/env python3
"""
Combined Degradation Experiments - Optimized Hyperparameters
=============================================================

3 Models on combined (all) degradation with paper-informed HP tuning.

HP Rationale (from papers):
---------------------------
ResNet50:
  - TResNet paper: AdamW + cosine schedule yields 79.0% on ImageNet
  - Differential LR: backbone=1e-4, head=1e-3 (standard transfer learning)
  - Weight decay 1e-4 (DenseNet paper standard)
  - Label smoothing 0.1 (TResNet, EfficientNetV2)
  - 30 epochs with cosine annealing

DenseNet121:
  - DenseNet paper: feature reuse + parameter efficiency = good for degraded data
  - Dense connections preserve low-level features across layers
  - Same differential LR strategy as ResNet50
  - Weight decay 1e-4, label smoothing 0.1
  - 30 epochs with cosine annealing

TransNeXt Micro:
  - CLAUDE.md: full fine-tuning fails (~10%) due to overfitting
  - Linear probe (freeze backbone) proved effective: 68.75% on low_res=16
  - Head-only training: lr=1e-3, weight_decay=1e-4
  - 30 epochs with cosine annealing
  - No label smoothing (simpler head training avoids over-regularization)

All use:
  - degradation_type='all' (combined: downsampling + blur + noise + salt_pepper + grayscale)
  - low_res=16, out_size=224
  - Cosine LR scheduler
  - Gradient clipping (max_norm=1.0)
"""

import os
import sys
from pathlib import Path

# Fix CUDA paging
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.runner import run_experiment


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

COMMON = {
    "pretrained": True,
    "low_res": 16,
    "out_size": 224,
    "batch_size": 32,
    "degradation_type": "all",
    "scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "group": "official",
}

# Pilot mode: quick 5-epoch runs to validate HP before full training
PILOT_OVERRIDES = {
    "epochs": 5,
    "train_subset": 2000,
    "val_subset": 1000,
    "group": "pilot",
}

# Full mode: serious training
FULL_OVERRIDES = {
    "epochs": 30,
    "train_subset": 10000,
    "val_subset": 5000,
}

EXPERIMENTS = [
    # ---- Experiment 1: ResNet50 (differential LR fine-tuning) ----
    {
        "model_name": "resnet50",
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "freeze_backbone": False,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "tag": "combined_resnet50_tuned",
    },
    # ---- Experiment 2: DenseNet121 (differential LR fine-tuning) ----
    {
        "model_name": "densenet121",
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "freeze_backbone": False,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "tag": "combined_densenet121_tuned",
    },
    # ---- Experiment 3: TransNeXt Micro (linear probe) ----
    {
        "model_name": "transnext_micro",
        "lr": 1e-3,
        "backbone_lr": None,
        "freeze_backbone": True,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "tag": "combined_transnext_lp_tuned",
    },
]


def run_all(mode: str = "pilot"):
    """Run all 3 experiments. mode='pilot' for quick validation, 'full' for real."""
    overrides = PILOT_OVERRIDES if mode == "pilot" else FULL_OVERRIDES

    print("\n" + "=" * 70)
    print(f"  COMBINED DEGRADATION EXPERIMENTS ({mode.upper()} MODE)")
    print("=" * 70)
    print(f"  Models: ResNet50, DenseNet121, TransNeXt Micro")
    print(f"  Degradation: ALL (combined)")
    print(f"  Epochs: {overrides.get('epochs', COMMON.get('epochs'))}")
    print(f"  LR Scheduler: Cosine Annealing")
    print("=" * 70)

    results = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        config = {**COMMON, **overrides, **exp}
        model = config["model_name"]
        print(f"\n{'─' * 70}")
        print(f"  [{i}/3] {model}")
        print(f"  LR={config['lr']}, backbone_lr={config.get('backbone_lr')}")
        print(f"  freeze={config['freeze_backbone']}, wd={config['weight_decay']}")
        print(f"  label_smoothing={config['label_smoothing']}")
        print(f"{'─' * 70}\n")

        try:
            run_experiment(**config)
            results.append({"model": model, "status": "OK"})
        except Exception as e:
            print(f"[ERROR] {model}: {e}")
            results.append({"model": model, "status": f"FAILED: {e}"})

    # Summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    for r in results:
        status = "✓" if r["status"] == "OK" else "✗"
        print(f"  {status} {r['model']:20s} {r['status']}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    args = p.parse_args()
    run_all(args.mode)
