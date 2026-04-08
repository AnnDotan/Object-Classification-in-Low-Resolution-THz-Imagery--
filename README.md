# Object Classification in Low-Resolution THz Imagery

Deep learning robustness evaluation under simulated THz-like imaging: low resolution, blur, noise, and grayscale degradation.

## Key Results

| Model | Combined Degradation | Method |
|-------|---------------------|--------|
| **DenseNet121** | **80.7%** | differential LR fine-tuning |
| ResNet50 | 78.8% | differential LR fine-tuning |
| TransNeXt Micro | 68.8% | linear probe (frozen backbone) |

## Quick Start

```bash
# Setup
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Run systematic experiments (3 degradation levels x 3 models)
python run_systematic.py --level all --mode pilot   # quick test (5 epochs)
python run_systematic.py --level 2 --mode full      # full training (30 epochs)

# Generate dashboard
python src/tools/generate_systematic_dashboard.py
# Open artifacts/dashboard_systematic.html in browser

# Single experiment
python main.py --model resnet50 --pretrained --epochs 20 --degradation_type all
```

## Degradation Levels (Systematic)

| Level | Resolution | Blur | Noise | Salt & Pepper |
|-------|-----------|------|-------|---------------|
| 1 (Mild) | 16px | k=3, σ=0.5 | std=0.04 | 2% |
| 2 (Moderate) | 16px | k=5, σ=1.0 | std=0.08 | 5% |
| 3 (Severe) | 8px | k=7, σ=1.5 | std=0.12 | 8% |

## Repository Structure

```
src/                — source code (models, data pipeline, tools)
run_systematic.py   — systematic experiment launcher
main.py             — single experiment entry point
runs/               — experiment outputs and checkpoints
artifacts/          — dashboards, figures, tables
scripts/archive/    — old experiment scripts
docs/               — detailed documentation
```

## Dashboard

Interactive HTML dashboard at `artifacts/dashboard_systematic.html`:
- Original vs degraded sample images per configuration
- Filterable results tables (by group, model)
- Learning curves on click
- Auto-filters incomplete runs and <30% accuracy

Regenerate after new experiments: `python src/tools/generate_systematic_dashboard.py`

## Models

- **ResNet50** — CNN baseline
- **DenseNet121** — dense connections for feature reuse
- **TransNeXt Micro** — aggregated attention (ImageNet pretrained, linear probe)

## Contributors

- Itamar Bahat

## License

Academic and research use.
