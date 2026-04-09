# Results Summary — Object Classification in Low-Resolution THz Imagery

**Date**: April 2025
**Status**: Phase A complete, Phase B in progress (6/9), Phases C/D pending

---

## Executive Summary

We evaluate three architectures under systematic degradation simulating THz imaging conditions on CIFAR-10 and MNIST. DenseNet121 is the most robust model on CIFAR-10, while ResNet50 leads on MNIST.

---

## Phase A: CIFAR-10 Systematic — COMPLETE (9/9)

| Level | ResNet50 | DenseNet121 | TransNeXt Micro |
|-------|----------|-------------|-----------------|
| **L1 Mild** (16px, blur k=3) | 81.2% (26 ep) | **81.9%** (16 ep) | 68.2% (19 ep) |
| **L2 Moderate** (16px, blur k=5) | 78.7% (26 ep) | **80.4%** (26 ep) | 63.6% (15 ep) |
| **L3 Severe** (8px, blur k=7) | 61.5% (27 ep) | **63.4%** (15 ep) | 44.9% (18 ep) |

**Key Findings (CIFAR-10)**:
- DenseNet121 leads at every degradation level — dense connections preserve low-level features
- Mild → Severe accuracy drop: DenseNet ~18.5%, ResNet ~19.7%, TransNeXt ~23.3%
- TransNeXt frozen backbone trails by ~15-18% vs fine-tuned CNNs
- All models converge in 15-27 epochs with early stopping (patience=5)

---

## Phase B: MNIST Systematic — IN PROGRESS (6/9)

| Level | ResNet50 | DenseNet121 | TransNeXt Micro |
|-------|----------|-------------|-----------------|
| **L1 Mild** | **99.1%** (14 ep) | 98.7% (7 ep) | 92.5% (30 ep) |
| **L2 Moderate** | **99.1%** (22 ep) | 99.0% (15 ep) | 92.8% (26 ep) |
| **L3 Severe** | Pending | Pending | Pending |

**Key Findings (MNIST)**:
- MNIST is significantly easier — even mild degradation yields 92%+ for all models
- ResNet50 leads slightly over DenseNet121 (99.1% vs 98.7-99.0%)
- TransNeXt achieves 92%+ even though it trails on CIFAR-10
- Model ranking differs from CIFAR-10 (ResNet > DenseNet on MNIST)

---

## Cross-Dataset Comparison

| Model | CIFAR-10 L2 | MNIST L2 | Gap |
|-------|-------------|----------|-----|
| ResNet50 | 78.7% | 99.1% | +20.4% |
| DenseNet121 | 80.4% | 99.0% | +18.6% |
| TransNeXt Micro | 63.6% | 92.8% | +29.2% |

MNIST is ~20-30% easier than CIFAR-10 under the same degradation, with TransNeXt benefiting most from the simpler visual structure.

---

## Previous Best Results (Pre-Systematic, CIFAR-10)

These reference runs used the same combined degradation at Level 2 equivalent parameters:

| Model | Val Acc | Run |
|-------|---------|-----|
| DenseNet121 | 80.7% | combined_densenet121_tuned |
| ResNet50 | 78.8% | combined_resnet50_tuned |
| TransNeXt Micro | 64.2% | combined_transnext_lp_tuned |

Systematic results at L2 closely match: DenseNet 80.4%, ResNet 78.7%, TransNeXt 63.6%.

---

## Remaining Experiments

| Phase | Count | Description |
|-------|-------|-------------|
| B (L3 Severe) | 3 | MNIST: ResNet50, DenseNet121, TransNeXt at 8px |
| C (Isolation) | 12 | Single-degradation: downsampling/blur/noise/s&p x 3 models |
| D (Baselines) | 6 | Clean (no degradation): 3 models x 2 datasets |
