#!/usr/bin/env python3
"""
Direct experiment runner - execute 9 experiments with consistent settings
This script can be run directly with: python run_experiments_batch.py
"""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up environment
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from src.runner import run_experiment

# Consistent background settings for all experiments
EPOCHS = 20
BATCH_SIZE = 32
LOW_RES = 16
PRETRAINED = True
GROUP = "official"
FREEZE_BACKBONE = False  # Use linear probe approach like TransNeXt

# Experiment matrix: (model, degradation_type, tag)
EXPERIMENTS = [
    ("resnet50", "downsampling", "exp1_resnet50_downsampling"),
    ("resnet50", "blur", "exp2_resnet50_blur"),
    ("resnet50", "noise", "exp3_resnet50_noise"),
    ("densenet121", "downsampling", "exp4_densenet_downsampling"),
    ("densenet121", "blur", "exp5_densenet_blur"),
    ("densenet121", "noise", "exp6_densenet_noise"),
    ("transnext_micro", "downsampling", "exp7_transnext_downsampling"),
    ("transnext_micro", "blur", "exp8_transnext_blur"),
    ("transnext_micro", "noise", "exp9_transnext_noise"),
]

def main():
    print("\n" + "="*80)
    print("STARTING 9 SYSTEMATIC EXPERIMENTS - DEGRADATION TYPE ISOLATION")
    print("="*80)
    print(f"\nConsistent Background Settings (all experiments):")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Low Resolution: {LOW_RES}")
    print(f"  - Pretrained: {PRETRAINED}")
    print(f"  - Group: {GROUP}")
    print(f"  - Dataset: CIFAR-10")
    print(f"  - Freeze Backbone: {FREEZE_BACKBONE}")
    
    results = []
    
    for i, (model, degradation, tag) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'-'*80}")
        print(f"EXPERIMENT {i}/9: {model:20s} + {degradation:15s}")
        print(f"{'-'*80}\n")
        
        try:
            run_experiment(
                model_name=model,
                pretrained=PRETRAINED,
                out_size=224,
                low_res=LOW_RES,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                train_subset=5000,
                val_subset=2000,
                lr=1e-3,
                tag=tag,
                group=GROUP,
                freeze_backbone=FREEZE_BACKBONE,
                degradation_type=degradation,
            )
            results.append({
                "num": i,
                "model": model,
                "degradation": degradation,
                "tag": tag,
                "success": True,
            })
            print(f"\n[OK] Experiment {i} COMPLETED")
        except Exception as e:
            results.append({
                "num": i,
                "model": model,
                "degradation": degradation,
                "tag": tag,
                "success": False,
                "error": str(e),
            })
            print(f"\n[FAIL] Experiment {i} FAILED: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r["success"])
    print(f"\nCompleted: {successful}/9")
    
    for result in results:
        status = "[OK]" if result["success"] else "[FAIL]"
        print(f"{status} [{result['num']}] {result['model']:20s} + {result['degradation']:15s}")
        if not result["success"]:
            print(f"     Error: {result.get('error', 'Unknown error')}")
    
    if successful == 9:
        print("\n[SUCCESS] ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("  1. Generate dashboards: python src/tools/refresh_dashboards.py")
        print("  2. Open: artifacts/dashboard_advanced.html")
        print("  3. Analyze results and compare models")
        return 0
    else:
        print(f"\n[WARNING] {9 - successful} experiment(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
