# Visualization Guide

## Per-Experiment Visualizations

After each training run, 3 visualizations are automatically generated in the run directory:

### 1. learning_curves.png
2x2 grid showing:
- Training Loss over epochs
- Validation Loss over epochs
- Training Accuracy (%) over epochs
- Validation Accuracy with best epoch highlighted

### 2. overfitting_analysis.png
Side-by-side train vs validation comparison:
- Left: Train vs Val Loss (gap = overfitting)
- Right: Train vs Val Accuracy (gap = overfitting)

### 3. metrics_summary.png
Text summary card with:
- Model name, device, pretrained status
- Resolution, batch size
- Best validation accuracy & epoch
- Convergence status

## Dashboards

### Experiment Plan Dashboard
`python src/tools/generate_experiment_plan_dashboard.py` → `artifacts/dashboard_experiment_plan.html`

Features:
- Tracks all 36 planned experiments
- Original vs degraded sample images
- Comparative bar charts per degradation level
- Cross-level robustness chart
- Learning curve modal (click any row)
- Auto-generated interim conclusions

### Systematic Dashboard
`python src/tools/generate_systematic_dashboard.py` → `artifacts/dashboard_systematic.html`

Features:
- Groups by degradation config
- Sample image comparison
- Filters out incomplete/low-accuracy runs
- Learning curve viewer

## Regenerating Visualizations

```bash
# Regenerate dashboards
python src/tools/generate_experiment_plan_dashboard.py
python src/tools/generate_systematic_dashboard.py
```
