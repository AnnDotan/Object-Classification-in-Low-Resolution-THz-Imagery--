# Project: Low-Resolution Image Classification (THz-like)

## Goal
Evaluate deep learning robustness under severe visual degradation (low resolution, blur, noise, grayscale) simulating THz imaging.

## Models
- **ResNet50** — baseline CNN (differential LR fine-tuning)
- **DenseNet121** — feature reuse CNN (differential LR fine-tuning)
- **TransNeXt Micro** — aggregated attention ViT (linear probe, frozen backbone)

## Datasets
- **CIFAR-10** — 10 object classes, 32×32 RGB, 10K train / 5K val
- **MNIST** — 10 digit classes, 28×28 grayscale→3ch, 10K train / 5K val

Both use identical degradation pipeline: Original → Degrade → Upsample 224×224 → Normalize → Model

## Current Best Results (combined degradation, CIFAR-10)
| Model | Val Acc | Method |
|-------|---------|--------|
| DenseNet121 | 80.7% | differential LR fine-tuning |
| ResNet50 | 78.8% | differential LR fine-tuning |
| TransNeXt | 64.2% | linear probe (frozen backbone) |

## Experiment Plan (36 experiments)

> Full details: `EXPERIMENT_PLAN.md`

| Phase | Description | # Experiments | Status |
|-------|-------------|--------------|--------|
| **A** | CIFAR-10 × 3 levels × 3 models | 9 | 🔲 TODO |
| **B** | MNIST × 3 levels × 3 models | 9 | 🔲 TODO (needs MNIST pipeline) |
| **C** | Single-degradation isolation | 12 | 🟡 7/12 done |
| **D** | Clean baselines (no degradation) | 6 | 🔲 TODO |

## Degradation Levels
| Level | Resolution | Blur | Noise | Salt & Pepper |
|-------|-----------|------|-------|---------------|
| 1 Mild | 16px | k=3, σ=0.5 | std=0.04 | 2% |
| 2 Moderate | 16px | k=5, σ=1.0 | std=0.08 | 5% |
| 3 Severe | 8px | k=7, σ=1.5 | std=0.12 | 8% |

## Training Hyperparameters (from papers)
- **Optimizer**: AdamW (β₁=0.9, β₂=0.999) — TransNeXt paper
- **Scheduler**: Cosine LR decay — TransNeXt / EfficientNetV2 papers
- **Head LR**: 1e-3, **Backbone LR**: 1e-4 (CNNs) / frozen (TransNeXt)
- **Weight decay**: 1e-4 — DenseNet paper
- **Label smoothing**: 0.1 (CNNs) / 0.0 (TransNeXt LP) — TransNeXt paper
- **Gradient clipping**: max_norm=1.0 — TransNeXt paper
- **Early stopping**: patience=5 epochs
- **Batch size**: 32, **Epochs**: 30 (full) / 5 (pilot)

## Experiment System

**Systematic runner**: `run_systematic.py` — 3 degradation levels × 3 models with early stopping
```bash
python run_systematic.py --level all --mode pilot              # quick test
python run_systematic.py --level 1,2,3 --mode full             # full CIFAR-10
python run_systematic.py --level all --mode full --dataset mnist # full MNIST
```

**Dashboard (systematic)**: `python src/tools/generate_systematic_dashboard.py` → `artifacts/dashboard_systematic.html`
- Filters out incomplete runs and <30% accuracy
- Shows original vs degraded images per config
- Interactive learning curves

**Dashboard (experiment plan)**: `python src/tools/generate_experiment_plan_dashboard.py` → `artifacts/dashboard_experiment_plan.html`
- Dark theme with white text, tracks all 36 planned experiments
- Scans only `runs/systematic/` — matches runs by tag (e.g. `sys_L1_mild_resnet50`)
- Shows experiment status (completed/pending), date, duration, degradation sample per config
- Comparative bar charts per degradation level (all 3 models side-by-side)
- Cross-level robustness line chart (accuracy vs. severity)
- Clickable rows → learning curve modal (accuracy + loss)
- Filters: phase, model, dataset, status

## Codebase Structure
```
EXPERIMENT_PLAN.md — full experiment plan (36 experiments, 4 phases, timeline)
src/               — main code (models, data, tools)
src/data/          — datasets + degradation pipeline (degrade.py, datasets.py)
src/models/        — model wrappers (transnext_wrapper.py)
src/tools/         — dashboard generators (incl. experiment plan dashboard), summarizers, visualizers
runs/              — experiment outputs (systematic/, official/, pilot/)
artifacts/         — dashboards, figures, tables
papers/            — reference papers (TransNeXt, DenseNet, TResNet, EfficientNetV2, NASNet)
scripts/archive/   — old experiment scripts (not active)
docs/              — detailed documentation
```

## Key Files
- `src/runner.py` — training loop with early stopping, custom degradation params
- `src/data/degrade.py` — DegradeConfig: low_res, blur, noise, salt_pepper, grayscale
- `src/data/datasets.py` — THzLikeCIFAR10 + THzLikeMNIST dataset wrappers
- `run_systematic.py` — systematic experiment launcher (3 levels × 3 models × 2 datasets)
- `src/tools/generate_experiment_plan_dashboard.py` — 36-experiment plan dashboard (scans `runs/systematic/` only)

## Coding Rules
- Minimal changes, no rewrites
- Keep pipeline intact, log everything
- Same protocol/dataset/degradation across models (fair comparison)
- Reproducibility first

## Deadlines
- Poster: 31/05/2026
- Presentation: 21/06/2026
- Submission: 26/07/2026
