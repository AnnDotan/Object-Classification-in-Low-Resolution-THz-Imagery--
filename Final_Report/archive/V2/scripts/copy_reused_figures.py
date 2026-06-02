"""Copy curated subsets of existing figures into Final_Report/figures/.

The campaign produced ~80 figures under artifacts/figures/. The report uses a
curated subset. This script copies only those, into clean per-topic folders,
so the LaTeX source references stable paths inside Final_Report/.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "figures"
DST = ROOT / "Final_Report" / "figures"

COPIES = [
    # Phase B2 / C2 / D summary plots (already publishable quality).
    ("phase_b2_comparison_summary.png", "phase_b2/comparison_summary.png"),
    ("phase_b2_comparison_summary.pdf", "phase_b2/comparison_summary.pdf"),
    ("phase_c2_attribution_summary.png", "phase_c2/attribution_summary.png"),
    ("phase_c2_attribution_summary.pdf", "phase_c2/attribution_summary.pdf"),
    ("phase_d_recovery_summary.png", "phase_d/recovery_summary.png"),
    ("phase_d_recovery_summary.pdf", "phase_d/recovery_summary.pdf"),
    ("phase_d_comparison.png", "phase_d/comparison.png"),
    ("phase_d_comparison.pdf", "phase_d/comparison.pdf"),
    # Per-(m,d) Phase B2 panels (6).
    ("phase_b2_comparison_resnet50_cifar10.png", "phase_b2/comparison_resnet50_cifar10.png"),
    ("phase_b2_comparison_resnet50_mnist.png", "phase_b2/comparison_resnet50_mnist.png"),
    ("phase_b2_comparison_densenet121_cifar10.png", "phase_b2/comparison_densenet121_cifar10.png"),
    ("phase_b2_comparison_densenet121_mnist.png", "phase_b2/comparison_densenet121_mnist.png"),
    ("phase_b2_comparison_transnext_tiny_cifar10.png", "phase_b2/comparison_transnext_tiny_cifar10.png"),
    ("phase_b2_comparison_transnext_tiny_mnist.png", "phase_b2/comparison_transnext_tiny_mnist.png"),
    # Inference throughput.
    ("inference_throughput.png", "../figures/inference_throughput.png"),
    ("inference_throughput.csv", "../data/tables/inference_throughput.csv"),
    # Confusion matrices: Phase B L5 (6 panels).
    ("confusion/final_B_L5_resnet50_cifar10_L5.png", "confusion/B_L5_resnet50_cifar10.png"),
    ("confusion/final_B_L5_resnet50_mnist_L5.png", "confusion/B_L5_resnet50_mnist.png"),
    ("confusion/final_B_L5_densenet121_cifar10_L5.png", "confusion/B_L5_densenet121_cifar10.png"),
    ("confusion/final_B_L5_densenet121_mnist_L5.png", "confusion/B_L5_densenet121_mnist.png"),
    ("confusion/final_B_L5_transnext_tiny_cifar10_L5.png", "confusion/B_L5_transnext_tiny_cifar10.png"),
    ("confusion/final_B_L5_transnext_tiny_mnist_L5.png", "confusion/B_L5_transnext_tiny_mnist.png"),
    # Calibration: Phase B L3 and L5 (12 panels).
    ("calibration/final_B_L3_resnet50_cifar10_L3.png", "calibration/B_L3_resnet50_cifar10.png"),
    ("calibration/final_B_L3_resnet50_mnist_L3.png", "calibration/B_L3_resnet50_mnist.png"),
    ("calibration/final_B_L3_densenet121_cifar10_L3.png", "calibration/B_L3_densenet121_cifar10.png"),
    ("calibration/final_B_L3_densenet121_mnist_L3.png", "calibration/B_L3_densenet121_mnist.png"),
    ("calibration/final_B_L3_transnext_tiny_cifar10_L3.png", "calibration/B_L3_transnext_tiny_cifar10.png"),
    ("calibration/final_B_L3_transnext_tiny_mnist_L3.png", "calibration/B_L3_transnext_tiny_mnist.png"),
    ("calibration/final_B_L5_resnet50_cifar10_L5.png", "calibration/B_L5_resnet50_cifar10.png"),
    ("calibration/final_B_L5_resnet50_mnist_L5.png", "calibration/B_L5_resnet50_mnist.png"),
    ("calibration/final_B_L5_densenet121_cifar10_L5.png", "calibration/B_L5_densenet121_cifar10.png"),
    ("calibration/final_B_L5_densenet121_mnist_L5.png", "calibration/B_L5_densenet121_mnist.png"),
    ("calibration/final_B_L5_transnext_tiny_cifar10_L5.png", "calibration/B_L5_transnext_tiny_cifar10.png"),
    ("calibration/final_B_L5_transnext_tiny_mnist_L5.png", "calibration/B_L5_transnext_tiny_mnist.png"),
]


def main() -> None:
    copied = 0
    missing = []
    for src_rel, dst_rel in COPIES:
        src = SRC / src_rel
        dst = DST / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(src_rel)
    print(f"[copy_reused_figures] copied {copied} files")
    if missing:
        print(f"[copy_reused_figures] WARNING: missing {len(missing)} sources:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
