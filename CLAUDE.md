# Project Context – Low-Resolution Image Classification (THz-like)

## Overview
This project investigates the robustness of deep learning models for image classification under severe visual degradation, simulating Terahertz (THz) imaging conditions.

The core idea is that standard CNNs rely on high-frequency details that are lost under low resolution, blur, and noise. The project evaluates whether advanced architectures can preserve classification performance under these constraints.

---

## Research Question
How robust can image classification remain when visual information is severely degraded (low resolution, blur, noise, grayscale)?

---

## Objectives and Success Criteria

1. Accuracy on degraded data  
   Target: Top-1 accuracy ≥ 80%

2. Robustness  
   Accuracy drop ≤ 15% compared to original images

3. Architecture comparison  
   Identify a model that outperforms ResNet50 baseline (>5%)

4. Efficiency  
   Inference time < 50ms per image (GPU)

---

## Dataset and Degradation Pipeline

Datasets:
- CIFAR-10 (primary benchmark)
- MNIST (optional extension)

Degradation:
- Downsampling (32 → 16 → 8)
- Gaussian blur
- Noise (Gaussian / Speckle)
- Grayscale

Pipeline:
Original → Degradation → Upsample → Normalize → Model

---

## Models

Baseline:
- ResNet50

Intermediate:
- DenseNet121 (feature reuse)

Advanced:
- TransNeXt (aggregated attention)

---

## Current Project Status

DONE:
- Full repo structure (src, runs, artifacts)
- Degradation pipeline
- Training pipeline (PyTorch)
- Logging, checkpoints, metrics
- Summary scripts
- Baseline runs (ResNet50, DenseNet121)
- TransNeXt integration (wrapper + pretrained + GPU)

RESULTS:
- ResNet50 ≈ 54.5% (low_res=16)
- DenseNet slightly lower
- Accuracy decreases with degradation → benchmark valid

INSIGHT:
The system is stable and reproducible.
Project is in evaluation phase.

---

## Main Problem

TransNeXt underperforms:
- Accuracy ~9–15%
- Not a bug → likely transfer-learning mismatch

---

## Current Focus (CRITICAL)

Before full fine-tuning:

1. Head-only training  
   Freeze backbone, train classifier

2. Linear probe  
   Extract features, train linear layer

Goal:
Verify whether pretrained features are usable.

---

## Next Steps

- Validate transfer strategy
- Compare ResNet / DenseNet / TransNeXt
- Generate plots (accuracy, robustness)
- Build final conclusions

---

## Codebase Structure

src/ – main code  
src/models/ – models  
src/data/ – datasets + degradation  
src/tools/ – utilities  
runs/ – experiments  
artifacts/ – outputs  

Each run must include:
- metrics.csv
- log.txt
- run_config.txt
- checkpoints

---

## Experiment Rules

- Same protocol across models
- Same dataset split
- Same degradation
- Same preprocessing

Goal: fair comparison only

---

## Key Concepts

- Information preservation > depth
- Low resolution → rely on low-level features
- Transfer learning may fail under domain shift
- Robustness measured across degradation severity

---

## Coding Guidelines

- Minimal changes (no rewrites)
- Keep pipeline intact
- Log everything
- Ensure reproducibility

---

## Experiment Strategy

- Start small (pilot runs)
- Scale gradually
- Tag runs clearly
- Always compare to baseline

---

## Analysis Guidelines

- Focus on trends, not single runs
- Compare across models and degradations
- Highlight failures clearly
- Avoid overclaiming

---

## Final Goal

Even if TransNeXt fails:

The project is successful if it clearly demonstrates:
- which architecture is most robust
- under what conditions
- supported by strong experiments