# Object Classification in Low-Resolution THz Imagery

## About

This project investigates how robust deep learning models can remain when visual information is severely degraded, simulating Terahertz (THz) imaging conditions. Standard CNNs rely on high-frequency details that are lost under low resolution, blur, and noise. We evaluate whether advanced architectures can preserve classification performance under these constraints.

**Research Question**: How robust can image classification remain when visual information is severely degraded (low resolution, blur, noise, grayscale)?

## Objectives

1. **Accuracy on degraded data** — Target: Top-1 accuracy >= 80% on combined degradation
2. **Robustness** — Accuracy drop <= 15% compared to clean images
3. **Architecture comparison** — Identify which architecture is most robust and why
4. **Efficiency** — Inference time < 50ms per image (GPU)

## Experiment Plan

> Full details: [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)

**36 total experiments** organized in 4 phases across 2 datasets:

| Phase | Description | # Experiments | Priority |
|-------|-------------|--------------|----------|
| **A** | CIFAR-10 × 3 degradation levels × 3 models | 9 | 🔴 Critical |
| **B** | MNIST × 3 degradation levels × 3 models | 9 | 🔴 Critical |
| **C** | Single-degradation isolation (CIFAR-10) | 12 | 🟡 Important |
| **D** | Clean baselines (no degradation, both datasets) | 6 | 🟡 Important |

## Models

| Model | Type | Strategy | Paper Reference |
|-------|------|----------|----------------|
| **ResNet50** | CNN (baseline) | Differential LR fine-tuning (backbone=1e-4, head=1e-3) | TResNet (Ridnik et al., 2020) |
| **DenseNet121** | CNN (feature reuse) | Differential LR fine-tuning (backbone=1e-4, head=1e-3) | DenseNet (Huang et al., 2017) |
| **TransNeXt Micro** | Vision Transformer (aggregated attention) | Linear probe — frozen ImageNet backbone, head-only training | TransNeXt (Shi, 2024) |

## Datasets

| Dataset | Image Size | Classes | Train / Val | Notes |
|---------|-----------|---------|-------------|-------|
| **CIFAR-10** | 32×32 RGB | 10 (objects) | 10,000 / 5,000 | Primary dataset |
| **MNIST** | 28×28 grayscale | 10 (digits) | 10,000 / 5,000 | Cross-domain validation (grayscale → 3ch repeat) |

Both datasets use the same degradation pipeline, normalization (ImageNet stats), and training protocol for fair comparison.

## Best Results So Far — Combined Degradation (low_res=16, CIFAR-10)

All results below are on the **same degradation pipeline** (downsampling + blur + noise + salt & pepper + grayscale) with the same data split (10,000 train / 5,000 val), making them directly comparable:

```
 DenseNet121 ██████████████████████████████████████████  80.7%
  ResNet50   ████████████████████████████████████████    78.8%
TransNeXt LP ████████████████████████████████            64.2%  (frozen backbone)
```

| Model | Best Val Acc | Epochs | Method | Overfitting Prevention |
|-------|-------------|--------|--------|----------------------|
| **DenseNet121** | **80.7%** | 30 | Differential LR + cosine schedule | Weight decay, label smoothing 0.1, grad clipping |
| ResNet50 | 78.8% | 30 | Differential LR + cosine schedule | Weight decay, label smoothing 0.1, grad clipping |
| TransNeXt Micro | 64.2% | 30 | Linear probe (frozen backbone) | Weight decay, grad clipping |

**Key findings**:
- DenseNet121's dense connections preserve low-level features better under degradation
- ResNet50 benefits greatly from differential LR and cosine scheduling (+24% vs baseline)
- TransNeXt frozen backbone retains useful features but trails behind fine-tuned CNNs on combined degradation
- Full fine-tuning of TransNeXt fails (~10%) due to overfitting — linear probe is essential

## Degradation Pipeline

Images pass through a multi-stage degradation pipeline simulating THz-like conditions:

```
Original (32×32 CIFAR-10 / 28×28 MNIST)
    |
    v
[Grayscale conversion (p=0.3)]
    |
    v
[Downsampling to low_res (16px or 8px)]
    |
    v
[Upsampling back to 224x224]
    |
    v
[Gaussian blur (kernel, sigma)]
    |
    v
[Gaussian noise (std)]
    |
    v
[Salt & Pepper noise]
    |
    v
[ImageNet normalization]
    |
    v
Model input (224×224×3)
```

### Systematic Degradation Levels

| Level | Name | Resolution | Blur | Noise | Salt & Pepper | Grayscale p |
|-------|------|-----------|------|-------|---------------|-------------|
| 1 | Mild | 16px | k=3, σ=0.5 | std=0.04 | 2% | 0.3 |
| 2 | Moderate | 16px | k=5, σ=1.0 | std=0.08 | 5% | 0.3 |
| 3 | Severe | 8px | k=7, σ=1.5 | std=0.12 | 8% | 0.3 |

### Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Input resolution | 224×224 | TransNeXt requirement |
| Batch size | 32 | GPU memory constraint |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) | TransNeXt paper |
| Scheduler | Cosine LR decay | TransNeXt / EfficientNetV2 papers |
| Epochs | 30 (full) / 5 (pilot) | Early stopping patience=5 |
| Gradient clipping | max_norm=1.0 | TransNeXt paper |
| Weight decay | 1e-4 | DenseNet paper |
| Label smoothing | 0.1 (CNNs) / 0.0 (TransNeXt LP) | TransNeXt paper |

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt

# Run systematic experiments (3 degradation levels x 3 models)
python run_systematic.py --level all --mode pilot   # quick validation (5 epochs)
python run_systematic.py --level 2 --mode full      # full training (30 epochs)
python run_systematic.py --level 1,3 --mode full    # specific levels

# Run with MNIST dataset
python run_systematic.py --level all --mode full --dataset mnist

# Run a single experiment
python main.py --model resnet50 --pretrained --epochs 20 --degradation_type all

# Generate dashboards
python src/tools/generate_systematic_dashboard.py
python src/tools/generate_experiment_plan_dashboard.py
python src/tools/generate_dashboard.py
python src/tools/generate_advanced_dashboard.py
```

## Dashboards

Three interactive HTML dashboards in `artifacts/`:

| Dashboard | File | Purpose |
|-----------|------|---------|
| **Experiment Plan** | `dashboard_experiment_plan.html` | Tracks all 36 planned experiments (4 phases). Scans only `runs/systematic/` for plan runs. Dark theme, white text. Comparative bar charts per degradation level, cross-level robustness chart, clickable learning curves. Filters by phase/model/dataset/status. |
| **Systematic** | `dashboard_systematic.html` | Groups experiments by degradation config. Shows original vs degraded sample images. Filters out incomplete and low-accuracy (<30%) runs. Click any row for learning curves. |
| **Basic** | `dashboard.html` | Overview with model comparison charts, degradation comparison, accuracy distribution histogram, and filterable results table with color-coded rows. |
| **Advanced** | `dashboard_advanced.html` | Full-pipeline vs single-degradation separation. Top runs with interactive learning curves. 3x3 grid for single-degradation type isolation analysis. |

Regenerate all after new experiments:
```bash
python src/tools/generate_experiment_plan_dashboard.py
python src/tools/generate_systematic_dashboard.py
python src/tools/refresh_dashboards.py
```

## Repository Structure

```
EXPERIMENT_PLAN.md      — full experiment plan (36 experiments, 4 phases)
run_systematic.py       — systematic experiment launcher (3 levels x 3 models x 2 datasets)
main.py                 — single experiment entry point
requirements.txt        — Python dependencies
src/
  runner.py             — training loop (early stopping, custom degradation, cosine LR)
  data/
    degrade.py          — degradation pipeline (DegradeConfig)
    datasets.py         — CIFAR-10 + MNIST dataset wrappers with degradation
  models/
    transnext_wrapper.py — TransNeXt model wrapper
  tools/
    generate_experiment_plan_dashboard.py  — experiment plan dashboard (36 experiments tracker)
    generate_systematic_dashboard.py  — systematic dashboard generator
    generate_dashboard.py             — basic dashboard generator
    generate_advanced_dashboard.py    — advanced dashboard generator
    refresh_dashboards.py             — refresh all dashboards
    summarize_runs.py                 — aggregate run results to CSV
    visualize_run.py                  — per-run learning curve plots
runs/                   — experiment outputs (systematic/, official/, pilot/)
artifacts/              — generated dashboards, figures, tables
papers/                 — reference papers (TransNeXt, DenseNet, TResNet, EfficientNetV2, NASNet)
scripts/archive/        — old experiment scripts (not active)
docs/                   — detailed documentation
```

## Papers & References

| Paper | Key Contribution to This Project |
|-------|----------------------------------|
| **TransNeXt** (Shi, 2024) | Model architecture; training hyperparameters (AdamW, cosine LR, grad clip 1.0, label smoothing 0.1) |
| **DenseNet** (Huang et al., 2017) | Model architecture; weight decay 1e-4; feature reuse under degradation |
| **TResNet** (Ridnik et al., 2020) | GPU-efficient architectures; CIFAR-10 transfer learning reference |
| **EfficientNetV2** (Tan & Le, 2021) | Progressive training concept; transfer learning protocol |
| **NASNet** (Zoph et al., 2018) | Architecture search on CIFAR-10; transferability to larger datasets |
| **Deep Learning Models for Image Classification** (Sharma & Guleria) | Survey of model comparison methodology |

## Tech Stack

- Python 3.12+
- PyTorch 2.x (CUDA supported), torchvision, timm, torchmetrics
- **PyTorch Lightning** — training engine (LightningModule + LightningDataModule, `pl.seed_everything(42, workers=True)` lock)
- **Weights & Biases** — experiment tracking (offline mode by default; CSVLogger as fallback)
- **Optuna** (+ `optuna-integration[pytorch-lightning]`) — hyper-parameter search
- Chart.js / Plotly.js — interactive dashboards

## Reproducibility

The pipeline is **fully deterministic**. Each validation image is degraded with a per-sample local RNG keyed on its dataset index (`seed = idx + SEED_OFFSET_VAL`), so image *N* receives byte-identical noise across every epoch, every model, and every process re-launch. Two reads of the same val batch satisfy MSE = 0; verified by [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py).

## Contributors

- Itamar Bahat

## License

Academic and research use.
