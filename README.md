# Object Classification in Low-Resolution THz Imagery

## About

This project investigates how robust deep learning models can remain when visual information is severely degraded, simulating Terahertz (THz) imaging conditions. Standard CNNs rely on high-frequency details that are lost under low resolution, blur, and noise. We evaluate whether advanced architectures can preserve classification performance under these constraints.

**Research Question**: How robust can image classification remain when visual information is severely degraded (low resolution, blur, noise, grayscale)?

## Objectives

1. **Accuracy on degraded data** — Target: Top-1 accuracy >= 80% on combined degradation
2. **Robustness** — Accuracy drop <= 15% compared to clean images
3. **Architecture comparison** — Identify which architecture is most robust and why
4. **Efficiency** — Inference time < 50ms per image (GPU)

## Models

| Model | Type | Strategy |
|-------|------|----------|
| **ResNet50** | CNN (baseline) | Differential LR fine-tuning (backbone=1e-4, head=1e-3) |
| **DenseNet121** | CNN (feature reuse) | Differential LR fine-tuning (backbone=1e-4, head=1e-3) |
| **TransNeXt Micro** | Vision Transformer (aggregated attention) | Linear probe — frozen ImageNet backbone, head-only training |

## Best Results So Far — Combined Degradation (low_res=16)

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
Original CIFAR-10 (32x32)
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
[Gaussian blur (kernel=5, sigma=1.0)]
    |
    v
[Gaussian noise (std=0.08)]
    |
    v
[Salt & Pepper noise (5%)]
    |
    v
[ImageNet normalization]
    |
    v
Model input
```

### Systematic Degradation Levels

| Level | Resolution | Blur | Noise | Salt & Pepper |
|-------|-----------|------|-------|---------------|
| 1 (Mild) | 16px | k=3, sigma=0.5 | std=0.04 | 2% |
| 2 (Moderate) | 16px | k=5, sigma=1.0 | std=0.08 | 5% |
| 3 (Severe) | 8px | k=7, sigma=1.5 | std=0.12 | 8% |

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

# Run a single experiment
python main.py --model resnet50 --pretrained --epochs 20 --degradation_type all

# Generate dashboards
python src/tools/generate_systematic_dashboard.py
python src/tools/generate_dashboard.py
python src/tools/generate_advanced_dashboard.py
```

## Dashboards

Three interactive HTML dashboards in `artifacts/`:

| Dashboard | File | Purpose |
|-----------|------|---------|
| **Systematic** | `dashboard_systematic.html` | Main dashboard. Groups experiments by degradation config. Shows original vs degraded sample images. Filters out incomplete and low-accuracy (<30%) runs. Click any row for learning curves. |
| **Basic** | `dashboard.html` | Overview with model comparison charts, degradation comparison, accuracy distribution histogram, and filterable results table with color-coded rows. |
| **Advanced** | `dashboard_advanced.html` | Full-pipeline vs single-degradation separation. Top runs with interactive learning curves. 3x3 grid for single-degradation type isolation analysis. |

Regenerate all after new experiments:
```bash
python src/tools/generate_systematic_dashboard.py
python src/tools/refresh_dashboards.py
```

## Repository Structure

```
run_systematic.py       — systematic experiment launcher (3 levels x 3 models)
main.py                 — single experiment entry point
requirements.txt        — Python dependencies
src/
  runner.py             — training loop (early stopping, custom degradation, cosine LR)
  data/
    degrade.py          — degradation pipeline (DegradeConfig)
    datasets.py         — CIFAR-10 dataset with degradation
  models/
    transnext_wrapper.py — TransNeXt model wrapper
  tools/
    generate_systematic_dashboard.py  — systematic dashboard generator
    generate_dashboard.py             — basic dashboard generator
    generate_advanced_dashboard.py    — advanced dashboard generator
    refresh_dashboards.py             — refresh all dashboards
    summarize_runs.py                 — aggregate run results to CSV
    visualize_run.py                  — per-run learning curve plots
runs/                   — experiment outputs (systematic/, official/, pilot/)
artifacts/              — generated dashboards, figures, tables
scripts/archive/        — old experiment scripts (not active)
docs/                   — detailed documentation
```

## Tech Stack

- Python 3.12+
- PyTorch 2.x (CUDA supported)
- torchvision, timm
- Chart.js / Plotly.js (dashboards)

## Contributors

- Itamar Bahat

## License

Academic and research use.
