# Visualization Pack - Learning Curve Guide

## Overview

The Visualization Pack automatically generates learning curves and metrics summaries after every experiment. This helps you quickly understand model behavior and make research decisions without manual plotting.

---

## What Gets Generated

After each training run, **3 visualizations are automatically created** in the run directory:

### 1. **learning_curves.png** 
📊 Comprehensive 2×2 grid showing:
- Training Loss over epochs
- Validation Loss over epochs
- Training Accuracy (%) over epochs
- **Validation Accuracy** with best epoch highlighted in gold

**Why**: Quickly spot when the model converges, whether overfitting occurs, and what your best validation accuracy is.

### 2. **overfitting_analysis.png**
🔍 Side-by-side train vs validation comparison:
- Left: Train vs Val Loss (gap indicates overfitting)
- Right: Train vs Val Accuracy (gap indicates overfitting)

**Why**: Instantly detect overfitting. Large gaps = your model memorized training data instead of learning general patterns.

### 3. **metrics_summary.png**
📋 Text summary showing:
- Model configuration (name, device, pretrained)
- Data configuration (resolution, batch size)
- Best validation accuracy & epoch
- Loss reduction statistics
- Convergence status

**Why**: One-page reference for all key metrics without opening CSV files.

---

## Auto-Integration

Visualizations are **automatically generated** at the end of each training run. No manual steps needed!

```
Training completes...
  ✅ Saved checkpoint
  ✅ Saved metrics
  🎨 Generating learning curve visualizations...
  ✅ learning_curves.png created
  ✅ overfitting_analysis.png created
  ✅ metrics_summary.png created
```

---

## Manual Visualization

If you want to regenerate visualizations for an existing run:

```bash
# Method 1: From Python
python -m src.tools.visualize_run runs/official/your_experiment

# Method 2: From Python script
from pathlib import Path
from src.tools.visualize_run import visualize_run

visualize_run(Path("runs/official/your_experiment"))
```

---

## Research Decision Making

### How to Use These Plots

**1. Best Validation Accuracy**
- Look at validation accuracy curve
- Gold star shows the best epoch
- Compare best_val_acc across different models
- Use this for final comparison tables

**2. Convergence Speed**
- Look at how quickly loss decreases in first epochs
- Faster convergence = efficient learning
- Plateau = model has learned what it can

**3. Overfitting Detection**
- Large gap between train and val loss = overfitting
- Similar curves = good generalization
- Decision: increase regularization or data if overfitting detected

**4. Noise in Validation**
- Smooth curve = stable learning
- Jagged curve = high noise (small batch, noisy data)
- Helps decide: increase batch size? More epochs?

---

## File Locations

All visualizations save in the run directory:

```
runs/official/transnext_linear_probe_gpu_transnext_micro_pt_out224_lowres16_lr1e-03/
├── metrics.csv                    (raw data)
├── log.txt                        (training log)
├── best.pt                        (best model)
├── learning_curves.png            ✨ (auto-generated)
├── overfitting_analysis.png       ✨ (auto-generated)
└── metrics_summary.png            ✨ (auto-generated)
```

---

## Customization

To change DPI (resolution) or colors, edit `src/tools/visualize_run.py`:

```python
# Change DPI for publication quality
visualizer = RunVisualizer(run_dir, dpi=150)  # Default is 100

# Or change colors in the plot functions
color_train = '#2E86AB'  # Current: blue
color_val = '#A23B72'    # Current: purple
```

---

## Future Enhancements

Potential additions (not yet implemented):
- [ ] Cross-run comparison plots (overlay multiple runs)
- [ ] Degradation robustness curves
- [ ] Inference time benchmarking
- [ ] Interactive HTML dashboard
- [ ] PDF report generation

---

## Troubleshooting

**Problem**: "metrics.csv not found"
- Solution: Ensure training completed successfully

**Problem**: "Visualization failed"
- Solution: Check that metrics.csv has data in it (not just headers)

**Problem**: Plots look blurry
- Solution: Use higher DPI: `visualizer = RunVisualizer(run_dir, dpi=150)`

---

## Example Workflow

```bash
# 1. Run an experiment
python main.py --model resnet50 --pretrained \
  --low_res 16 --epochs 20 \
  --group official --tag test

# 2. Visualizations auto-generate
# 3. Check the run directory for PNG files
# 4. Open learning_curves.png to see results
# 5. Make decisions based on plots:
#    - "Overfitting after epoch 15? Stop early next time"
#    - "Model converged after epoch 8? Could use fewer epochs"
#    - "Good accuracy? Compare with other models"
```

---

**Created**: April 2026  
**Status**: Ready for production use
