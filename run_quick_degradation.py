#!/usr/bin/env python3
"""
Quick Degradation Type Experiments (Short Duration)

9 experiments with shorter runtime for quick results:
- 5 epochs (instead of 20)
- 2000 train samples (instead of 5000)
- 1000 val samples (instead of 2000)

Total estimated time: ~40-45 minutes for all 9
"""

import subprocess
from pathlib import Path


MODELS = ['resnet50', 'densenet121', 'transnext_micro']
DEGRADATION_TYPES = ['downsampling', 'blur', 'noise']


def build_command(model: str, degradation_type: str) -> list:
    """Build quick experiment command."""
    return [
        'python', 'main.py',
        '--model', model,
        '--pretrained',
        '--degradation_type', degradation_type,
        '--out_size', '224',
        '--low_res', '16',
        '--epochs', '5',  # SHORT
        '--batch_size', '32',
        '--train_subset', '2000',  # SHORT
        '--val_subset', '1000',  # SHORT
        '--lr', '1e-3',
        '--group', 'quick_runs',
        '--tag', f'{degradation_type}_quick'
    ]


def run_quick_experiments():
    """Run all 9 quick experiments."""
    experiments = []
    for model in MODELS:
        for deg_type in DEGRADATION_TYPES:
            experiments.append((model, deg_type))

    print("\n" + "=" * 80)
    print("QUICK DEGRADATION EXPERIMENTS (5 EPOCHS)")
    print("=" * 80)
    print(f"Total: {len(experiments)} experiments")
    print(f"Estimated time: 40-45 minutes")
    print("=" * 80 + "\n")

    completed = 0

    for idx, (model, deg_type) in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] {model} + {deg_type}")
        print("-" * 80)

        cmd = build_command(model, deg_type)

        try:
            result = subprocess.run(cmd, cwd=Path(__file__).parent)
            if result.returncode == 0:
                completed += 1
                print(f"[OK] Completed")
            else:
                print(f"[ERROR] Failed with code {result.returncode}")
        except Exception as e:
            print(f"[ERROR] {e}")

        print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print(f"COMPLETED: {completed}/{len(experiments)} experiments")
    print("=" * 80)

    if completed == len(experiments):
        print("\nNext: python -m src.tools.degradation_robustness")


if __name__ == "__main__":
    run_quick_experiments()
