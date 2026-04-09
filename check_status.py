#!/usr/bin/env python3
"""Quick status check for running experiments.

Usage:
    python check_status.py
"""
import os
import csv
from pathlib import Path

EXPECTED_TAGS = {
    "A": [
        "sys_L1_mild_resnet50", "sys_L1_mild_densenet121", "sys_L1_mild_transnext_micro",
        "sys_L2_moderate_resnet50", "sys_L2_moderate_densenet121", "sys_L2_moderate_transnext_micro",
        "sys_L3_severe_resnet50", "sys_L3_severe_densenet121", "sys_L3_severe_transnext_micro",
    ],
    "B": [
        "sys_L1_mild_resnet50_mnist", "sys_L1_mild_densenet121_mnist", "sys_L1_mild_transnext_micro_mnist",
        "sys_L2_moderate_resnet50_mnist", "sys_L2_moderate_densenet121_mnist", "sys_L2_moderate_transnext_micro_mnist",
        "sys_L3_severe_resnet50_mnist", "sys_L3_severe_densenet121_mnist", "sys_L3_severe_transnext_micro_mnist",
    ],
    "C": [
        "iso_downsampling_resnet50", "iso_downsampling_densenet121", "iso_downsampling_transnext_micro",
        "iso_blur_resnet50", "iso_blur_densenet121", "iso_blur_transnext_micro",
        "iso_noise_resnet50", "iso_noise_densenet121", "iso_noise_transnext_micro",
        "iso_salt_pepper_resnet50", "iso_salt_pepper_densenet121", "iso_salt_pepper_transnext_micro",
    ],
    "D": [
        "clean_resnet50", "clean_densenet121", "clean_transnext_micro",
        "clean_resnet50_mnist", "clean_densenet121_mnist", "clean_transnext_micro_mnist",
    ],
}


def find_run(tag):
    sys_dir = Path("runs/systematic")
    if not sys_dir.exists():
        return None
    for d in sorted(sys_dir.iterdir()):
        if d.is_dir() and d.name.startswith(tag):
            return d
    return None


def get_run_info(run_dir):
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return 0, 0.0
    rows = []
    try:
        with open(metrics_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("epoch"):
                    rows.append(row)
    except Exception:
        return 0, 0.0
    if not rows:
        return 0, 0.0
    best_acc = max(float(r["val_acc"]) for r in rows)
    return len(rows), best_acc


def main():
    print("=" * 75)
    print("  EXPERIMENT STATUS CHECK")
    print("=" * 75)

    total = 0
    done = 0
    running = 0

    for phase, tags in EXPECTED_TAGS.items():
        print(f"\n  Phase {phase} ({len(tags)} experiments)")
        print(f"  {'Tag':<48} {'Epochs':>6} {'Best Acc':>9}  Status")
        print(f"  {'─'*48} {'─'*6} {'─'*9}  {'─'*10}")

        for tag in tags:
            total += 1
            run_dir = find_run(tag)
            if run_dir is None:
                print(f"  {tag:<48} {'—':>6} {'—':>9}  ⏳ Pending")
                continue

            epochs, best_acc = get_run_info(run_dir)
            if epochs == 0:
                print(f"  {tag:<48} {'0':>6} {'—':>9}  🔄 Starting")
                running += 1
            elif epochs < 10:
                print(f"  {tag:<48} {epochs:>6} {best_acc*100:>8.1f}%  🔄 Running")
                running += 1
            else:
                print(f"  {tag:<48} {epochs:>6} {best_acc*100:>8.1f}%  ✅ Done")
                done += 1

    pending = total - done - running
    print(f"\n{'=' * 75}")
    print(f"  TOTAL: {done} done, {running} running, {pending} pending (out of {total})")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
