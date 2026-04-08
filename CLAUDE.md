# Project: Low-Resolution Image Classification (THz-like)

## Goal
Evaluate deep learning robustness under severe visual degradation (low resolution, blur, noise, grayscale) simulating THz imaging.

## Models
- **ResNet50** — baseline CNN
- **DenseNet121** — feature reuse CNN
- **TransNeXt Micro** — aggregated attention (primary model, linear probe)

## Dataset
CIFAR-10 with degradation pipeline: Original -> Degrade -> Upsample -> Normalize -> Model

## Current Best Results (combined degradation)
| Model | Val Acc | Method |
|-------|---------|--------|
| DenseNet121 | 80.7% | differential LR fine-tuning |
| ResNet50 | 78.8% | differential LR fine-tuning |
| TransNeXt | 68.8% | linear probe (frozen backbone) |

## Experiment System

**Systematic runner**: `run_systematic.py` — 3 degradation levels x 3 models with early stopping
```bash
python run_systematic.py --level all --mode pilot   # quick test
python run_systematic.py --level 1,2,3 --mode full  # full training
```

**Dashboard**: `python src/tools/generate_systematic_dashboard.py` -> `artifacts/dashboard_systematic.html`
- Filters out incomplete runs and <30% accuracy
- Shows original vs degraded images per config
- Interactive learning curves

## Codebase Structure
```
src/              — main code (models, data, tools)
src/data/         — datasets + degradation pipeline (degrade.py, datasets.py)
src/models/       — model wrappers (transnext_wrapper.py)
src/tools/        — dashboard generators, summarizers, visualizers
runs/             — experiment outputs (systematic/, official/, pilot/)
artifacts/        — dashboards, figures, tables
scripts/archive/  — old experiment scripts (not active)
docs/             — detailed documentation
```

## Key Files
- `src/runner.py` — training loop with early stopping, custom degradation params
- `src/data/degrade.py` — DegradeConfig: low_res, blur, noise, salt_pepper, grayscale
- `run_systematic.py` — systematic experiment launcher (3 levels x 3 models)

## Coding Rules
- Minimal changes, no rewrites
- Keep pipeline intact, log everything
- Same protocol/dataset/degradation across models (fair comparison)
- Reproducibility first

## Deadlines
- Poster: 31/05/2026
- Presentation: 21/06/2026
- Submission: 26/07/2026
