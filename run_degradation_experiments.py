#!/usr/bin/env python3
"""
Automated Degradation Type Experiments

Runs 9 experiments: 3 models × 3 degradation types
All experiments use consistent parameters for fair comparison.

Models:
  - resnet50
  - densenet121
  - transnext_micro

Degradation Types:
  - downsampling (low resolution)
  - blur (gaussian blur)
  - noise (gaussian noise)

Total: 9 experiments = 3 models × 3 degradation types
"""

import subprocess
import sys
from pathlib import Path


# Consistent experimental setup
COMMON_PARAMS = {
    'pretrained': True,
    'low_res': 16,
    'out_size': 224,  # TransNeXt needs 224
    'epochs': 20,
    'batch_size': 32,
    'train_subset': 5000,
    'val_subset': 2000,
    'lr': 1e-3,
    'group': 'official',
}

MODELS = ['resnet50', 'densenet121', 'transnext_micro']
DEGRADATION_TYPES = ['downsampling', 'blur', 'noise']


def build_command(model: str, degradation_type: str) -> list:
    """Build command line for an experiment."""
    cmd = ['python', 'main.py']

    cmd.extend(['--model', model])
    cmd.extend(['--degradation_type', degradation_type])

    if COMMON_PARAMS['pretrained']:
        cmd.append('--pretrained')

    cmd.extend(['--low_res', str(COMMON_PARAMS['low_res'])])
    cmd.extend(['--out_size', str(COMMON_PARAMS['out_size'])])
    cmd.extend(['--epochs', str(COMMON_PARAMS['epochs'])])
    cmd.extend(['--batch_size', str(COMMON_PARAMS['batch_size'])])
    cmd.extend(['--train_subset', str(COMMON_PARAMS['train_subset'])])
    cmd.extend(['--val_subset', str(COMMON_PARAMS['val_subset'])])
    cmd.extend(['--lr', str(COMMON_PARAMS['lr'])])
    cmd.extend(['--group', COMMON_PARAMS['group']])

    # Tag for identification
    tag = f"{degradation_type}_robustness"
    cmd.extend(['--tag', tag])

    return cmd


def print_header():
    """Print experiment overview."""
    print("\n" + "=" * 80)
    print("DEGRADATION TYPE ROBUSTNESS EXPERIMENTS")
    print("=" * 80)
    print("\nExperimental Setup:")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Degradation Types: {', '.join(DEGRADATION_TYPES)}")
    print(f"  Total Experiments: {len(MODELS) * len(DEGRADATION_TYPES)}")
    print(f"\nCommon Parameters:")
    for key, value in COMMON_PARAMS.items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 80 + "\n")


def run_experiments(dry_run: bool = False):
    """Run all experiments."""
    experiments = []

    for model in MODELS:
        for degradation_type in DEGRADATION_TYPES:
            experiments.append((model, degradation_type))

    print_header()

    completed = 0
    failed = 0

    for idx, (model, degradation_type) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] Running: {model} + {degradation_type}")
        print("-" * 80)

        cmd = build_command(model, degradation_type)
        cmd_str = " ".join(cmd)

        print(f"Command: {cmd_str}\n")

        if dry_run:
            print("[DRY RUN] Would execute above command")
            completed += 1
        else:
            try:
                result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
                if result.returncode == 0:
                    completed += 1
                    print(f"\n[OK] Completed: {model} + {degradation_type}")
                else:
                    failed += 1
                    print(f"\n[ERROR] Failed: {model} + {degradation_type}")
            except subprocess.CalledProcessError as e:
                failed += 1
                print(f"\n[ERROR] Exception: {model} + {degradation_type}")
                print(f"Return code: {e.returncode}")

        print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"Total Experiments: {len(experiments)}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print("=" * 80 + "\n")

    if not dry_run and completed == len(experiments):
        print("[OK] All experiments completed successfully!")
        print("\nNext steps:")
        print("  1. Run: python -m src.tools.degradation_robustness")
        print("  2. Check: artifacts/figures/degradation_robustness_*.png")
        print("  3. Analyze results and make conclusions")
    elif failed > 0:
        print(f"[WARN] {failed} experiments failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run degradation type robustness experiments"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )

    args = parser.parse_args()

    run_experiments(dry_run=args.dry_run)
