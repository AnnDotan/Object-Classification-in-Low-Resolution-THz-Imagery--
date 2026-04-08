#!/usr/bin/env python3
"""
TransNeXt Improved Fine-Tuning Experiment
==========================================

Based on analysis of the TransNeXt paper (CVPR 2024) official training recipe.

Previous results:
  - Linear probe (frozen backbone): 64.2%  — too limited
  - Full fine-tuning (no regularization): ~10%  — severe overfitting

Key missing ingredients identified from paper:
  1. Mixup (α=0.8) + CutMix (α=1.0) — essential for ViT fine-tuning
  2. Weight decay = 0.05 (paper default, vs our 1e-4)
  3. Drop path rate = 0.15 (micro config)
  4. LR warmup (5 epochs from 1e-6)
  5. Label smoothing = 0.1

Strategy: Full fine-tuning with paper's heavy regularization.
  - Backbone LR = 5e-5 (careful adaptation)
  - Head LR = 5e-4 (fresh head needs faster learning)
  - Cosine LR with warmup
  - All regularization active
"""

import os
import sys
from pathlib import Path

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.runner import run_experiment


# ============================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================

COMMON = {
    "model_name": "transnext_micro",
    "pretrained": True,
    "low_res": 16,
    "out_size": 224,
    "batch_size": 32,
    "degradation_type": "all",
    "group": "official",
}

PILOT_OVERRIDES = {
    "epochs": 10,
    "train_subset": 2000,
    "val_subset": 1000,
    "group": "pilot",
}

FULL_OVERRIDES = {
    "epochs": 40,
    "train_subset": 10000,
    "val_subset": 5000,
}

EXPERIMENTS = [
    # ---- V1: Full fine-tuning with paper recipe ----
    {
        "tag": "transnext_v1_paper_recipe",
        "freeze_backbone": False,
        "lr": 5e-4,
        "backbone_lr": 5e-5,
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
        "scheduler_type": "cosine",
        "warmup_epochs": 5,
        "max_grad_norm": 1.0,
        "mixup_alpha": 0.8,
        "cutmix_alpha": 1.0,
        "drop_path_rate": 0.15,
    },
    # ---- V2: Moderate regularization (lower mixup, lower wd) ----
    {
        "tag": "transnext_v2_moderate",
        "freeze_backbone": False,
        "lr": 3e-4,
        "backbone_lr": 3e-5,
        "weight_decay": 0.02,
        "label_smoothing": 0.1,
        "scheduler_type": "cosine",
        "warmup_epochs": 5,
        "max_grad_norm": 1.0,
        "mixup_alpha": 0.4,
        "cutmix_alpha": 0.5,
        "drop_path_rate": 0.15,
    },
    # ---- V3: Partial freeze (stages 1-2 frozen, 3-4 + head trainable) ----
    {
        "tag": "transnext_v3_partial_freeze",
        "freeze_backbone": False,   # we handle partial freeze specially below
        "lr": 5e-4,
        "backbone_lr": 1e-5,        # very low for unfrozen early stages
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
        "scheduler_type": "cosine",
        "warmup_epochs": 5,
        "max_grad_norm": 1.0,
        "mixup_alpha": 0.8,
        "cutmix_alpha": 1.0,
        "drop_path_rate": 0.15,
    },
]


def run_all(mode: str = "pilot"):
    overrides = PILOT_OVERRIDES if mode == "pilot" else FULL_OVERRIDES

    print("\n" + "=" * 70)
    print(f"  TRANSNEXT IMPROVED EXPERIMENTS ({mode.upper()} MODE)")
    print("=" * 70)
    print(f"  Variants: V1 (paper recipe), V2 (moderate), V3 (partial freeze)")
    print(f"  Degradation: ALL (combined)")
    print(f"  Epochs: {overrides.get('epochs')}")
    print(f"  Key additions: Mixup + CutMix + DropPath + Warmup + wd=0.05")
    print("=" * 70)

    results = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        config = {**COMMON, **overrides, **exp}
        tag = config["tag"]
        print(f"\n{'─' * 70}")
        print(f"  [{i}/{len(EXPERIMENTS)}] {tag}")
        print(f"  LR={config['lr']}, backbone_lr={config.get('backbone_lr')}")
        print(f"  wd={config['weight_decay']}, mixup={config['mixup_alpha']}, "
              f"cutmix={config['cutmix_alpha']}")
        print(f"  drop_path={config['drop_path_rate']}, warmup={config['warmup_epochs']}")
        print(f"{'─' * 70}\n")

        try:
            run_experiment(**config)
            results.append({"tag": tag, "status": "OK"})
        except Exception as e:
            print(f"[ERROR] {tag}: {e}")
            import traceback; traceback.print_exc()
            results.append({"tag": tag, "status": f"FAILED: {e}"})

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['tag']}: {r['status']}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    args = p.parse_args()
    run_all(args.mode)
