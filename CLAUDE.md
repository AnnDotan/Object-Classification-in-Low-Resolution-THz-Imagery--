# Project Context – Low-Resolution Image Classification (THz-like)

## Overview
This project investigates the robustness of deep learning models for image classification under severe visual degradation, simulating Terahertz (THz) imaging conditions.

The core idea is that standard CNNs rely on high-frequency details that are lost under low resolution, blur, and noise. The project evaluates whether advanced architectures can preserve classification performance under these constraints.

---

## Research Question
How robust can image classification remain when visual information is severely degraded (low resolution, blur, noise, grayscale)?

---

## Objectives and Success Criteria

1. **Accuracy on degraded data** ✅ PROGRESSING
   - Target: Top-1 accuracy ≥ 80%
   - Current: 68.75% on low_res=16 (significant improvement from 54.5% baseline)
   - Status: Approaching target with further fine-tuning potential

2. **Robustness** ✅ IN PROGRESS
   - Target: Accuracy drop ≤ 15% compared to original images
   - Status: Measuring across low_res=8 and low_res=16

3. **Architecture comparison** ✅ ACHIEVED
   - Target: Identify model that outperforms ResNet50 (>5%)
   - Result: **TransNeXt achieves 14.25% improvement** ✓
   - Baselines: ResNet50 54.5%, DenseNet121 51.15%

4. **Efficiency** ⏳ PENDING
   - Target: Inference time < 50ms per image (GPU)
   - Note: GPU infrastructure ready, inference benchmarking next

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

### ✅ STAGE 1 COMPLETE

DONE:
- Full repo structure (src, runs, artifacts)
- Degradation pipeline (downsampling, blur, noise, grayscale)
- Training pipeline (PyTorch with GPU support)
- Logging, checkpoints, metrics
- Summary scripts (summarize_runs.py, plot_experiments.py)
- Baseline runs (ResNet50, DenseNet121) ✓
- TransNeXt integration (wrapper + pretrained + GPU) ✓
- **TransNeXt Linear Probe validation ✓**

RESULTS (STAGE 1):
- TransNeXt Linear Probe: **68.75%** (low_res=16, epoch 11) ✓
- ResNet50 baseline: 54.50% (low_res=16)
- DenseNet121: 51.15% (low_res=16)
- Improvement: +14.25% vs ResNet50 ✓
- Accuracy decreases with degradation (as expected)
- Benchmark is valid and stable

INSIGHT:
- TransNeXt pretrained features ARE highly useful for degraded images
- Frozen backbone + head training is effective
- Full fine-tuning fails (~10%) due to overfitting
- GPU acceleration (RTX 4050) reduced runtime from hours to ~82 minutes
- System is stable, reproducible, and ready for STAGE 2

---

## Key Achievements

1. **Python 3.14 Compatibility Fix**
   - Replaced `pkg_resources` with importlib-based check in transnext.py
   - Enables use of latest Python versions

2. **GPU Acceleration**
   - Installed PyTorch 2.11 with CUDA 13.0
   - RTX 4050 support confirmed and working
   - 10x faster training on GPU

3. **Code Enhancement**
   - Added `--backbone_lr` parameter for selective fine-tuning
   - Supports different learning rates for backbone vs head
   - Ready for STAGE 2 fine-tuning experiments

4. **Analysis Complete**
   - 3 publication-ready plots generated
   - Summary table with 38 runs aggregated
   - Results documentation (RESULTS_SUMMARY.md)

---

## Current Focus (STAGE 2)

Tasks IN PROGRESS:

1. Complete TransNeXt low_res=8 experiment
   - Currently initialized but no epochs recorded
   - Investigate and restart if needed

2. Build complete robustness curves
   - Compare across low_res=16 and low_res=8
   - Measure accuracy drop vs degradation

3. Final analysis and conclusions
   - Which architecture is most robust?
   - Under what conditions does each fail?
   - Why does TransNeXt succeed while fine-tuning fails?

---

## Next Steps

[ ] Complete TransNeXt low_res=8 experiment (or investigate failure)
[ ] Generate final robustness comparison plots
[ ] Write analysis explaining WHY results occur
[ ] Prepare figures for poster (deadline: 31/05/2026)
[ ] Final presentation prep (deadline: 21/06/2026)

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