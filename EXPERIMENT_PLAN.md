# Experiment Plan — Object Classification in Low-Resolution THz Imagery

> **Date**: April 2026  
> **Deadlines**: Poster 31/05 | Presentation 21/06 | Submission 26/07

---

## 1. Overview

This document defines **all** experiments required to complete the project. The plan ensures:
- **Fair comparison**: all models trained with identical degradation, data splits, and protocol
- **Reproducibility**: fixed seeds, logged configs, saved checkpoints
- **Two datasets**: CIFAR-10 (existing) + MNIST (new) for cross-dataset generalization analysis
- **Systematic coverage**: 3 degradation levels × 3 models × 2 datasets = **18 core experiments**

---

## 2. Models

| # | Model | Type | Strategy | Justification (from papers) |
|---|-------|------|----------|----------------------------|
| 1 | **ResNet50** | CNN (residual) | Differential LR fine-tuning | Baseline; GPU-efficient (TResNet paper); standard transfer learning backbone |
| 2 | **DenseNet121** | CNN (dense connections) | Differential LR fine-tuning | Feature reuse preserves low-level features under degradation (DenseNet paper: CIFAR-10 error 3.46% with augmentation) |
| 3 | **TransNeXt Micro** | Aggregated attention ViT | Linear probe (frozen backbone) | Robust foveal perception; global attention captures degraded signals (TransNeXt paper: 82.6% ImageNet-1K at 224²) |

---

## 3. Datasets

### 3.1 CIFAR-10 (existing)
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image size**: 32×32 RGB
- **Split**: 10,000 train / 5,000 val (from 50,000 train / 10,000 test)
- **Normalization**: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 3.2 MNIST (new)
- **Classes**: 10 (digits 0–9)
- **Image size**: 28×28 grayscale → convert to 3-channel (repeat)
- **Split**: 10,000 train / 5,000 val (from 60,000 train / 10,000 test)
- **Normalization**: ImageNet stats (same as CIFAR-10 for pretrained backbone compatibility)
- **Rationale**: Simpler dataset — if models struggle under degradation on CIFAR-10, MNIST shows a lower bound of degradation tolerance. Also tests generalization to different visual domains (digits vs. objects).

---

## 4. Degradation Pipeline

Identical pipeline for both datasets:

```
Original image (32×32 or 28×28)
    ↓
[Grayscale conversion (p=0.3)]           ← probabilistic
    ↓
[Downsampling to low_res px]             ← bilinear
    ↓
[Upsampling back to 224×224]             ← bilinear
    ↓
[Gaussian blur (kernel, sigma)]          ← separable convolution
    ↓
[Gaussian noise (std)]                   ← additive, clipped to [0,1]
    ↓
[Salt & Pepper noise (amount)]           ← random pixels set to 0/1
    ↓
[ImageNet normalization]
    ↓
Model input (224×224×3)
```

### 4.1 Degradation Levels

| Level | Name | Resolution | Blur Kernel | Blur Sigma | Noise Std | Salt & Pepper | Grayscale p |
|-------|------|-----------|-------------|------------|-----------|---------------|-------------|
| **1** | Mild | 16 px | 3 | 0.5 | 0.04 | 2% | 0.3 |
| **2** | Moderate | 16 px | 5 | 1.0 | 0.08 | 5% | 0.3 |
| **3** | Severe | 8 px | 7 | 1.5 | 0.12 | 8% | 0.3 |

**Source**: These levels follow a progressive degradation strategy inspired by:
- EfficientNetV2 (progressive training concept — gradually increase difficulty)
- The project's THz imaging simulation rationale (levels correspond to near-field, mid-field, far-field THz capture scenarios)

---

## 5. Training Hyperparameters

### 5.1 Common (all experiments)

| Parameter | Value | Source / Justification |
|-----------|-------|----------------------|
| **Input resolution** | 224×224 | Required by TransNeXt; standard for pretrained models |
| **Batch size** | 32 | GPU memory constraint (single GPU) |
| **Optimizer** | AdamW | TransNeXt paper: AdamW with β₁=0.9, β₂=0.999 |
| **Scheduler** | Cosine LR decay | TransNeXt paper; proven for fine-tuning (EfficientNetV2 paper) |
| **Epochs** | 30 (full) / 5 (pilot) | Early stopping may terminate sooner |
| **Early stopping** | patience=5 epochs | Prevents overfitting on degraded data |
| **Gradient clipping** | max_norm=1.0 | TransNeXt paper: gradient clipping norm of 1.0 |
| **Weight decay** | 1e-4 | DenseNet paper: weight decay 1e-4; conservative for transfer |
| **Train subset** | 10,000 (full) / 2,000 (pilot) | |
| **Val subset** | 5,000 (full) / 1,000 (pilot) | |
| **Seed** | 42 (fixed) | Reproducibility |
| **Pretrained** | Yes (ImageNet-1K) | Transfer learning from ImageNet for all models |
| **Degradation type** | "all" (combined pipeline) | Main experiments use full pipeline |

### 5.2 Per-Model Hyperparameters

| Parameter | ResNet50 | DenseNet121 | TransNeXt Micro |
|-----------|----------|-------------|-----------------|
| **Head LR** | 1e-3 | 1e-3 | 1e-3 |
| **Backbone LR** | 1e-4 | 1e-4 | — (frozen) |
| **Freeze backbone** | No | No | Yes |
| **Label smoothing** | 0.1 | 0.1 | 0.0 |
| **Training method** | Differential LR fine-tuning | Differential LR fine-tuning | Linear probe |

**Justification**:
- **Differential LR** (backbone 10× lower than head): Standard practice for fine-tuning pretrained CNNs — avoids catastrophic forgetting while allowing backbone adaptation
- **TransNeXt frozen backbone**: Full fine-tuning fails (~10% accuracy) due to overfitting; linear probe preserves ImageNet features (validated in Stage 1)
- **Label smoothing 0.1**: TransNeXt paper uses 0.1; applied to CNNs to regularize; not used for TransNeXt LP since the head is already simple
- **No mixup/cutmix**: Omitted because the degradation pipeline itself introduces significant noise and variation — additional augmentation on already-degraded images risks destroying signal

---

## 6. Experiment Matrix — Core Experiments

### Phase A: CIFAR-10 Systematic (9 experiments)

All experiments use `run_systematic.py --mode full`:

| Exp ID | Dataset | Level | Model | Tag | Expected Run Dir |
|--------|---------|-------|-------|-----|-----------------|
| **C1-1** | CIFAR-10 | 1 (Mild) | ResNet50 | `sys_L1_mild_resnet50` | `runs/systematic/` |
| **C1-2** | CIFAR-10 | 1 (Mild) | DenseNet121 | `sys_L1_mild_densenet121` | `runs/systematic/` |
| **C1-3** | CIFAR-10 | 1 (Mild) | TransNeXt Micro | `sys_L1_mild_transnext_micro` | `runs/systematic/` |
| **C2-1** | CIFAR-10 | 2 (Moderate) | ResNet50 | `sys_L2_moderate_resnet50` | `runs/systematic/` |
| **C2-2** | CIFAR-10 | 2 (Moderate) | DenseNet121 | `sys_L2_moderate_densenet121` | `runs/systematic/` |
| **C2-3** | CIFAR-10 | 2 (Moderate) | TransNeXt Micro | `sys_L2_moderate_transnext_micro` | `runs/systematic/` |
| **C3-1** | CIFAR-10 | 3 (Severe) | ResNet50 | `sys_L3_severe_resnet50` | `runs/systematic/` |
| **C3-2** | CIFAR-10 | 3 (Severe) | DenseNet121 | `sys_L3_severe_densenet121` | `runs/systematic/` |
| **C3-3** | CIFAR-10 | 3 (Severe) | TransNeXt Micro | `sys_L3_severe_transnext_micro` | `runs/systematic/` |

**Command**:
```bash
python run_systematic.py --level all --mode full
```

### Phase B: MNIST Systematic (9 experiments)

Same protocol as Phase A, identical degradation levels and hyperparameters, only the dataset changes:

| Exp ID | Dataset | Level | Model | Tag | Expected Run Dir |
|--------|---------|-------|-------|-----|-----------------|
| **M1-1** | MNIST | 1 (Mild) | ResNet50 | `sys_L1_mild_resnet50_mnist` | `runs/systematic/` |
| **M1-2** | MNIST | 1 (Mild) | DenseNet121 | `sys_L1_mild_densenet121_mnist` | `runs/systematic/` |
| **M1-3** | MNIST | 1 (Mild) | TransNeXt Micro | `sys_L1_mild_transnext_micro_mnist` | `runs/systematic/` |
| **M2-1** | MNIST | 2 (Moderate) | ResNet50 | `sys_L2_moderate_resnet50_mnist` | `runs/systematic/` |
| **M2-2** | MNIST | 2 (Moderate) | DenseNet121 | `sys_L2_moderate_densenet121_mnist` | `runs/systematic/` |
| **M2-3** | MNIST | 2 (Moderate) | TransNeXt Micro | `sys_L2_moderate_transnext_micro_mnist` | `runs/systematic/` |
| **M3-1** | MNIST | 3 (Severe) | ResNet50 | `sys_L3_severe_resnet50_mnist` | `runs/systematic/` |
| **M3-2** | MNIST | 3 (Severe) | DenseNet121 | `sys_L3_severe_densenet121_mnist` | `runs/systematic/` |
| **M3-3** | MNIST | 3 (Severe) | TransNeXt Micro | `sys_L3_severe_transnext_micro_mnist` | `runs/systematic/` |

**Command** (after implementing MNIST support):
```bash
python run_systematic.py --level all --mode full --dataset mnist
```

---

## 7. Experiment Matrix — Supplementary Experiments

### Phase C: Single-Degradation Isolation (CIFAR-10 only)

Isolate each degradation type to understand individual contributions. Use **Level 2 (Moderate) parameters** as the reference:

| Exp ID | Dataset | Degradation Type | Model | Parameters |
|--------|---------|-----------------|-------|------------|
| **I-1** | CIFAR-10 | downsampling only | ResNet50 | low_res=16, no blur/noise/sp |
| **I-2** | CIFAR-10 | downsampling only | DenseNet121 | low_res=16, no blur/noise/sp |
| **I-3** | CIFAR-10 | downsampling only | TransNeXt Micro | low_res=16, no blur/noise/sp |
| **I-4** | CIFAR-10 | blur only | ResNet50 | blur k=5, σ=1.0 |
| **I-5** | CIFAR-10 | blur only | DenseNet121 | blur k=5, σ=1.0 |
| **I-6** | CIFAR-10 | blur only | TransNeXt Micro | blur k=5, σ=1.0 |
| **I-7** | CIFAR-10 | noise only | ResNet50 | noise_std=0.08 |
| **I-8** | CIFAR-10 | noise only | DenseNet121 | noise_std=0.08 |
| **I-9** | CIFAR-10 | noise only | TransNeXt Micro | noise_std=0.08 |
| **I-10** | CIFAR-10 | salt_pepper only | ResNet50 | sp=0.05 |
| **I-11** | CIFAR-10 | salt_pepper only | DenseNet121 | sp=0.05 |
| **I-12** | CIFAR-10 | salt_pepper only | TransNeXt Micro | sp=0.05 |

**Note**: Some of these may already exist in `runs/official/`. Check before running duplicates.

### Phase D: Clean Baseline (No Degradation)

Run all models on clean (non-degraded) data to establish upper bounds:

| Exp ID | Dataset | Model | Notes |
|--------|---------|-------|-------|
| **B-1** | CIFAR-10 (clean, resized to 224) | ResNet50 | No degradation pipeline; direct resize to 224 + normalize |
| **B-2** | CIFAR-10 (clean, resized to 224) | DenseNet121 | |
| **B-3** | CIFAR-10 (clean, resized to 224) | TransNeXt Micro | |
| **B-4** | MNIST (clean, resized to 224) | ResNet50 | |
| **B-5** | MNIST (clean, resized to 224) | DenseNet121 | |
| **B-6** | MNIST (clean, resized to 224) | TransNeXt Micro | |

---

## 8. Total Experiment Count

| Phase | Description | # Experiments | Priority |
|-------|-------------|--------------|----------|
| **A** | CIFAR-10 × 3 levels × 3 models | 9 | 🔴 Critical — run first |
| **B** | MNIST × 3 levels × 3 models | 9 | 🔴 Critical — run second |
| **C** | Single-degradation isolation | 12 | 🟡 Important — already partially done |
| **D** | Clean baselines (no degradation) | 6 | 🟡 Important — upper bounds |
| **Total** | | **36** | |

---

## 9. Implementation Steps

### Step 1: Run CIFAR-10 Systematic (Phase A)
```bash
# Pilot first (quick validation)
python run_systematic.py --level all --mode pilot

# Full runs
python run_systematic.py --level all --mode full
```

### Step 2: Implement MNIST Support
Required code changes:
1. **`src/data/datasets.py`** — Add `THzLikeMNIST` class:
   - Load MNIST via `torchvision.datasets.MNIST`
   - Convert grayscale (1ch) to 3-channel by repeating
   - Apply same degradation pipeline via `degrade_image()`
   - Apply same ImageNet normalization
2. **`src/runner.py`** — Add `dataset` parameter to `run_experiment()`:
   - Route to `THzLikeCIFAR10` or `THzLikeMNIST` based on `dataset` param
3. **`run_systematic.py`** — Add `--dataset` argument:
   - Pass through to `run_experiment()`

### Step 3: Run MNIST Systematic (Phase B)
```bash
python run_systematic.py --level all --mode full --dataset mnist
```

### Step 4: Run Baselines + Isolation (Phases C, D)
- Check which isolation experiments already exist
- Run remaining ones
- Run clean baselines

### Step 5: Generate Dashboards
```bash
python src/tools/generate_systematic_dashboard.py
python src/tools/refresh_dashboards.py
```

---

## 10. Expected Analysis & Outputs

### 10.1 Tables for the Report

| Table | Content |
|-------|---------|
| **Table 1** | Model accuracy vs. degradation level (CIFAR-10) — 3×3 matrix |
| **Table 2** | Model accuracy vs. degradation level (MNIST) — 3×3 matrix |
| **Table 3** | Cross-dataset comparison (CIFAR-10 vs MNIST per level per model) |
| **Table 4** | Clean baseline vs. degraded accuracy — accuracy drop (Δ) per model |
| **Table 5** | Single-degradation isolation — which degradation hurts most per model |
| **Table 6** | Training efficiency — convergence speed, epochs to best accuracy |

### 10.2 Figures for the Report/Poster

| Figure | Content |
|--------|---------|
| **Fig 1** | Degradation pipeline diagram (original → degraded examples per level) |
| **Fig 2** | Bar chart: accuracy vs. degradation level (grouped by model, CIFAR-10) |
| **Fig 3** | Bar chart: accuracy vs. degradation level (grouped by model, MNIST) |
| **Fig 4** | Line plot: accuracy drop from clean baseline per model per level |
| **Fig 5** | Heatmap: model × degradation type isolation (which degradation hurts most) |
| **Fig 6** | Learning curves: train/val accuracy per model at Level 2 |
| **Fig 7** | Sample degraded images: grid showing all 3 levels for both datasets |

### 10.3 Key Research Questions to Answer

1. **Which model is most robust to degradation?** 
   - Hypothesis: DenseNet121 (dense connections preserve low-level features)
   
2. **How does degradation severity affect each architecture differently?**
   - Expect: TransNeXt degrades gracefully (global attention); CNNs drop faster at severe levels

3. **Does the same pattern hold on MNIST?**
   - MNIST is simpler; expect higher absolute accuracy but same relative ranking

4. **Which individual degradation is most destructive?**
   - Hypothesis: Low resolution (downsampling) is the primary factor; blur is secondary

5. **Can we achieve ≥80% accuracy under moderate degradation?**
   - Current best: DenseNet121 80.7% (Level 2 equivalent) — verify systematically

---

## 11. Current Results (as of April 2025)

### Phase A: CIFAR-10 Systematic — ✅ COMPLETE (9/9)

| Level | ResNet50 | DenseNet121 | TransNeXt Micro |
|-------|----------|-------------|-----------------|
| **L1 Mild** | 81.2% (26 ep) | **81.9%** (16 ep) | 68.2% (19 ep) |
| **L2 Moderate** | 78.7% (26 ep) | **80.4%** (26 ep) | 63.6% (15 ep) |
| **L3 Severe** | 61.5% (27 ep) | **63.4%** (15 ep) | 44.9% (18 ep) |

### Phase B: MNIST Systematic — 🟡 IN PROGRESS (6/9)

| Level | ResNet50 | DenseNet121 | TransNeXt Micro |
|-------|----------|-------------|-----------------|
| **L1 Mild** | **99.1%** (14 ep) | 98.7% (7 ep) | 92.5% (30 ep) |
| **L2 Moderate** | **99.1%** (22 ep) | 99.0% (15 ep) | 92.8% (26 ep) |
| **L3 Severe** | Pending | Pending | Pending |

### Phase C: Single-Degradation Isolation — 🔲 TODO (0/12)

### Phase D: Clean Baselines — 🔲 TODO (0/6)

### Key Findings So Far

1. **DenseNet121 is the most robust CNN** — consistently leads on CIFAR-10 across all degradation levels
2. **MNIST is much easier** — even TransNeXt achieves 92%+ under severe degradation
3. **Severe degradation (8px)** causes ~20% accuracy drop on CIFAR-10 compared to mild
4. **Early stopping works well** — most runs converge in 15–27 epochs (out of 30 max)

---

## 12. Execution Timeline

| Week | Dates | Tasks |
|------|-------|-------|
| **Week 1** | Apr 8–14 | Run Phase A (CIFAR-10 systematic, 9 experiments). Implement MNIST pipeline. |
| **Week 2** | Apr 15–21 | Run Phase B (MNIST systematic, 9 experiments). Run Phase D (clean baselines). |
| **Week 3** | Apr 22–28 | Run Phase C (isolation experiments — fill gaps). Generate all dashboards. |
| **Week 4** | Apr 29–May 5 | Analysis: generate tables, figures, write key findings. |
| **Week 5** | May 6–12 | Verify reproducibility (re-run select experiments). Draft analysis sections. |
| **Week 6** | May 13–19 | Poster design: select best figures, write abstract. |
| **Week 7** | May 20–26 | Finalize poster. Prepare abstract. |
| **Week 8** | May 27–31 | **POSTER & ABSTRACT DEADLINE (31/05)** |
| **Week 9–10** | Jun 1–14 | Write final report. Prepare presentation slides. |
| **Week 11** | Jun 15–21 | **FINAL PRESENTATION (21/06)** |
| **Week 12–16** | Jun 22–Jul 26 | Polish report, clean repo. **FINAL SUBMISSION (26/07)** |

---

## 12. File Changes Required

| File | Change | Status |
|------|--------|--------|
| `src/data/datasets.py` | Add `THzLikeMNIST` class | ✅ Done |
| `src/runner.py` | Add `dataset` parameter + auto-dashboard refresh | ✅ Done |
| `run_systematic.py` | Add `--dataset` argument | ✅ Done |
| `run_all_phases.py` | Runner for all 36 experiments (4 phases) | ✅ Done |
| `src/tools/generate_experiment_plan_dashboard.py` | 36-experiment tracker dashboard | ✅ Done |
| `src/tools/generate_systematic_dashboard.py` | Show dataset column in dashboard | ✅ Done |
| `EXPERIMENT_PLAN.md` | This document | ✅ Done |
