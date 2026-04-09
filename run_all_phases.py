#!/usr/bin/env python3
"""
Run ALL 36 experiments from the experiment plan sequentially.

Phase A: CIFAR-10 systematic (9 experiments) — 3 levels × 3 models
Phase B: MNIST systematic (9 experiments) — 3 levels × 3 models
Phase C: Single-degradation isolation (12 experiments) — 4 types × 3 models
Phase D: Clean baselines (6 experiments) — 2 datasets × 3 models

Usage:
    python run_all_phases.py                    # run everything
    python run_all_phases.py --phase A          # Phase A only
    python run_all_phases.py --phase A,B        # Phase A + B
    python run_all_phases.py --phase C,D        # Phase C + D
    python run_all_phases.py --skip-existing    # skip experiments whose run dir exists
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.runner import run_experiment

# ── Shared config ──
COMMON = {
    "pretrained": True,
    "out_size": 224,
    "batch_size": 32,
    "scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "group": "systematic",
    "epochs": 30,
    "train_subset": 10000,
    "val_subset": 5000,
    "early_stopping_patience": 5,
}

MODEL_CONFIGS = {
    "resnet50": {
        "model_name": "resnet50",
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "freeze_backbone": False,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
    },
    "densenet121": {
        "model_name": "densenet121",
        "lr": 1e-3,
        "backbone_lr": 1e-4,
        "freeze_backbone": False,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
    },
    "transnext_micro": {
        "model_name": "transnext_micro",
        "lr": 1e-3,
        "backbone_lr": None,
        "freeze_backbone": True,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
    },
}

DEGRADATION_LEVELS = {
    1: {"name": "mild",     "low_res": 16, "blur_kernel": 3, "blur_sigma": 0.5,
        "gaussian_noise_std": 0.04, "salt_pepper_amount": 0.02, "p_grayscale": 0.3},
    2: {"name": "moderate", "low_res": 16, "blur_kernel": 5, "blur_sigma": 1.0,
        "gaussian_noise_std": 0.08, "salt_pepper_amount": 0.05, "p_grayscale": 0.3},
    3: {"name": "severe",   "low_res": 8,  "blur_kernel": 7, "blur_sigma": 1.5,
        "gaussian_noise_std": 0.12, "salt_pepper_amount": 0.08, "p_grayscale": 0.3},
}

MODELS = ["resnet50", "densenet121", "transnext_micro"]


def build_all_experiments():
    """Build list of all 36 experiments."""
    experiments = []

    # ── Phase A: CIFAR-10 systematic (9) ──
    for level_id, level in DEGRADATION_LEVELS.items():
        for model in MODELS:
            tag = f"sys_L{level_id}_{level['name']}_{model}"
            experiments.append({
                "phase": "A",
                "tag": tag,
                "dataset": "cifar10",
                "degradation_type": "all",
                **COMMON,
                **MODEL_CONFIGS[model],
                "low_res": level["low_res"],
                "blur_kernel": level["blur_kernel"],
                "blur_sigma": level["blur_sigma"],
                "gaussian_noise_std": level["gaussian_noise_std"],
                "salt_pepper_amount": level["salt_pepper_amount"],
                "p_grayscale": level["p_grayscale"],
            })

    # ── Phase B: MNIST systematic (9) ──
    for level_id, level in DEGRADATION_LEVELS.items():
        for model in MODELS:
            tag = f"sys_L{level_id}_{level['name']}_{model}_mnist"
            experiments.append({
                "phase": "B",
                "tag": tag,
                "dataset": "mnist",
                "degradation_type": "all",
                **COMMON,
                **MODEL_CONFIGS[model],
                "low_res": level["low_res"],
                "blur_kernel": level["blur_kernel"],
                "blur_sigma": level["blur_sigma"],
                "gaussian_noise_std": level["gaussian_noise_std"],
                "salt_pepper_amount": level["salt_pepper_amount"],
                "p_grayscale": level["p_grayscale"],
            })

    # ── Phase C: Single-degradation isolation (12) — CIFAR-10, Level 2 params ──
    isolation_configs = {
        "downsampling": {
            "low_res": 16, "blur_kernel": None, "blur_sigma": None,
            "gaussian_noise_std": None, "salt_pepper_amount": None, "p_grayscale": None,
        },
        "blur": {
            "low_res": 16, "blur_kernel": 5, "blur_sigma": 1.0,
            "gaussian_noise_std": None, "salt_pepper_amount": None, "p_grayscale": None,
        },
        "noise": {
            "low_res": 16, "blur_kernel": None, "blur_sigma": None,
            "gaussian_noise_std": 0.08, "salt_pepper_amount": None, "p_grayscale": None,
        },
        "salt_pepper": {
            "low_res": 16, "blur_kernel": None, "blur_sigma": None,
            "gaussian_noise_std": None, "salt_pepper_amount": 0.05, "p_grayscale": None,
        },
    }
    for iso_type, iso_params in isolation_configs.items():
        for model in MODELS:
            tag = f"iso_{iso_type}_{model}"
            exp = {
                "phase": "C",
                "tag": tag,
                "dataset": "cifar10",
                "degradation_type": iso_type,
                **COMMON,
                **MODEL_CONFIGS[model],
                "low_res": iso_params["low_res"],
            }
            # Only set non-None params
            for k in ["blur_kernel", "blur_sigma", "gaussian_noise_std",
                       "salt_pepper_amount", "p_grayscale"]:
                if iso_params[k] is not None:
                    exp[k] = iso_params[k]
            experiments.append(exp)

    # ── Phase D: Clean baselines (6) — no degradation ──
    for ds in ["cifar10", "mnist"]:
        for model in MODELS:
            ds_suffix = f"_{ds}" if ds == "mnist" else ""
            tag = f"clean_{model}{ds_suffix}"
            experiments.append({
                "phase": "D",
                "tag": tag,
                "dataset": ds,
                "degradation_type": "all",
                **COMMON,
                **MODEL_CONFIGS[model],
                "low_res": 224,   # no downsampling
                "blur_kernel": 1,
                "blur_sigma": 0.0,
                "gaussian_noise_std": 0.0,
                "salt_pepper_amount": 0.0,
                "p_grayscale": 0.0,
            })

    return experiments


def find_existing_run(tag: str) -> bool:
    """Check if a run with this tag already exists in runs/systematic/."""
    sys_dir = Path("runs") / "systematic"
    if not sys_dir.exists():
        return False
    for d in sys_dir.iterdir():
        if d.is_dir() and d.name.startswith(tag):
            # Check if it has metrics.csv with >0 rows
            metrics = d / "metrics.csv"
            if metrics.exists() and metrics.stat().st_size > 50:
                return True
    return False


def run_single(exp: dict, exp_num: int, total: int):
    """Run a single experiment, stripping out internal keys."""
    tag = exp["tag"]
    phase = exp["phase"]

    # Remove 'phase' key before passing to run_experiment
    config = {k: v for k, v in exp.items() if k != "phase"}

    print(f"\n{'━' * 70}")
    print(f"  [{exp_num}/{total}] Phase {phase} | {config['model_name']} | {tag}")
    print(f"  dataset={config['dataset']}, degradation={config['degradation_type']}, "
          f"low_res={config['low_res']}")
    print(f"{'━' * 70}\n")

    t0 = time.time()
    try:
        run_experiment(**config)
        dt = time.time() - t0
        return {"tag": tag, "phase": phase, "status": "OK", "time": f"{dt:.1f}s"}
    except Exception as e:
        dt = time.time() - t0
        print(f"\n[ERROR] {tag}: {e}")
        return {"tag": tag, "phase": phase, "status": f"FAILED: {e}", "time": f"{dt:.1f}s"}


def main():
    parser = argparse.ArgumentParser(description="Run all experiment plan phases")
    parser.add_argument("--phase", default="A,B,C,D",
                        help="Phases to run (comma-separated): A,B,C,D")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments that already have results")
    args = parser.parse_args()

    phases = [p.strip().upper() for p in args.phase.split(",")]

    all_experiments = build_all_experiments()
    experiments = [e for e in all_experiments if e["phase"] in phases]

    print("\n" + "=" * 70)
    print("  FULL EXPERIMENT PLAN RUNNER")
    print("=" * 70)
    print(f"  Phases: {', '.join(phases)}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Epochs per experiment: {COMMON['epochs']}")
    print(f"  Early stopping patience: {COMMON['early_stopping_patience']}")
    print("=" * 70)

    results = []
    skipped = 0

    for i, exp in enumerate(experiments, 1):
        if args.skip_existing and find_existing_run(exp["tag"]):
            print(f"  [SKIP] {exp['tag']} — already exists")
            skipped += 1
            continue

        result = run_single(exp, i - skipped, len(experiments) - skipped)
        results.append(result)

        # Print running tally
        ok = sum(1 for r in results if r["status"] == "OK")
        failed = sum(1 for r in results if r["status"] != "OK")
        remaining = len(experiments) - skipped - len(results)
        print(f"\n  Progress: {ok} OK, {failed} failed, {remaining} remaining, {skipped} skipped")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("  EXPERIMENT PLAN — FINAL SUMMARY")
    print("=" * 70)
    for phase in phases:
        phase_results = [r for r in results if r["phase"] == phase]
        if phase_results:
            ok = sum(1 for r in phase_results if r["status"] == "OK")
            print(f"\n  Phase {phase}: {ok}/{len(phase_results)} completed")
            for r in phase_results:
                icon = "✓" if r["status"] == "OK" else "✗"
                print(f"    {icon} {r['tag']:<45} {r['status']:<10} {r['time']}")
    if skipped:
        print(f"\n  Skipped (already existed): {skipped}")
    print("=" * 70)

    # Save results
    summary_path = Path("artifacts") / "all_phases_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Full results saved to {summary_path}")


if __name__ == "__main__":
    main()
