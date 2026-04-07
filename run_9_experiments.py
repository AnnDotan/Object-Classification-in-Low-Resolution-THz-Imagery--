#!/usr/bin/env python3
"""
9 Systematic Degradation Experiments
=====================================

3 Models x 3 Degradation Types = 9 Experiments

Models:
  - ResNet50      (CNN baseline)
  - DenseNet121   (feature-reuse CNN)
  - TransNeXt     (aggregated attention, linear probe)

Degradation Types:
  - downsampling  (resolution loss: 32 -> 16 -> upscale to 224)
  - blur          (Gaussian blur, kernel=5, sigma=1.0)
  - salt_pepper   (salt & pepper noise, 5% of pixels)

All experiments use IDENTICAL base settings for fair comparison.

Usage:
    python run_9_experiments.py           # run all 9
    python run_9_experiments.py --dry-run # show commands only
"""

import os
import sys
import subprocess
from pathlib import Path

# Fix Windows CUDA paging file issue - must be set BEFORE torch import
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# ============================================================
# EXPERIMENTAL SETUP - identical across all 9 runs
# ============================================================
COMMON_PARAMS = {
    "pretrained": True,
    "low_res": 16,
    "out_size": 224,       # all models use 224 for fair comparison
    "epochs": 20,
    "batch_size": 32,
    "train_subset": 5000,
    "val_subset": 2000,
    "lr": 1e-3,
    "group": "official",
}

MODELS = ["resnet50", "densenet121", "transnext_micro"]
DEGRADATION_TYPES = ["downsampling", "blur", "salt_pepper"]


def build_command(model: str, degradation_type: str) -> list[str]:
    """Build the CLI command for a single experiment."""
    cmd = [sys.executable, "main.py"]

    cmd.extend(["--model", model])
    cmd.extend(["--degradation_type", degradation_type])

    if COMMON_PARAMS["pretrained"]:
        cmd.append("--pretrained")

    cmd.extend(["--low_res", str(COMMON_PARAMS["low_res"])])
    cmd.extend(["--out_size", str(COMMON_PARAMS["out_size"])])
    cmd.extend(["--epochs", str(COMMON_PARAMS["epochs"])])
    cmd.extend(["--batch_size", str(COMMON_PARAMS["batch_size"])])
    cmd.extend(["--train_subset", str(COMMON_PARAMS["train_subset"])])
    cmd.extend(["--val_subset", str(COMMON_PARAMS["val_subset"])])
    cmd.extend(["--lr", str(COMMON_PARAMS["lr"])])
    cmd.extend(["--group", COMMON_PARAMS["group"]])

    tag = f"{degradation_type}_{model}"
    cmd.extend(["--tag", tag])

    return cmd


def run_all(dry_run: bool = False):
    experiments = [(m, d) for m in MODELS for d in DEGRADATION_TYPES]

    print("=" * 70)
    print("  9 SYSTEMATIC DEGRADATION EXPERIMENTS")
    print("=" * 70)
    print()
    print("  Models:       ", ", ".join(MODELS))
    print("  Degradations: ", ", ".join(DEGRADATION_TYPES))
    print(f"  Total:         {len(experiments)} experiments")
    print()
    print("  Common Settings:")
    for k, v in COMMON_PARAMS.items():
        print(f"    {k}: {v}")
    print()
    print("=" * 70)

    completed = 0
    failed = 0
    results = []

    for idx, (model, deg) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] {model} + {deg}")
        print("-" * 70)

        cmd = build_command(model, deg)
        print(f"  CMD: {' '.join(cmd)}")

        if dry_run:
            print("  [DRY RUN] skipped")
            completed += 1
            results.append((model, deg, "DRY_RUN"))
            continue

        try:
            env = os.environ.copy()
            env["CUDA_MODULE_LOADING"] = "LAZY"
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                env=env,
                timeout=3600,  # 1 hour max per experiment
            )
            if result.returncode == 0:
                completed += 1
                results.append((model, deg, "OK"))
                print(f"  [OK] {model} + {deg}")
            else:
                failed += 1
                results.append((model, deg, f"FAIL(rc={result.returncode})"))
                print(f"  [FAIL] exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            failed += 1
            results.append((model, deg, "TIMEOUT"))
            print(f"  [TIMEOUT] exceeded 1 hour")
        except Exception as e:
            failed += 1
            results.append((model, deg, f"ERROR({e})"))
            print(f"  [ERROR] {e}")

    # Summary table
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Model':<18} {'Degradation':<16} {'Status'}")
    print(f"  {'-'*18} {'-'*16} {'-'*10}")
    for model, deg, status in results:
        marker = "[OK]" if status in ("OK", "DRY_RUN") else "[!!]"
        print(f"  {model:<18} {deg:<16} {marker} {status}")
    print(f"\n  Completed: {completed}/{len(experiments)}")
    print(f"  Failed:    {failed}/{len(experiments)}")
    print("=" * 70)

    if not dry_run and completed == len(experiments):
        print("\n  All 9 experiments completed!")
        print("\n  Next steps:")
        print("    1. python src/tools/refresh_dashboards.py")
        print("    2. Open artifacts/dashboard_advanced.html")
    elif failed > 0:
        print(f"\n  {failed} experiments failed")
        return 1

    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run 9 degradation experiments")
    parser.add_argument("--dry-run", action="store_true", help="Show commands only")
    args = parser.parse_args()
    sys.exit(run_all(dry_run=args.dry_run))
