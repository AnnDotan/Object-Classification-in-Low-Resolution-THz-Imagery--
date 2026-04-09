# Implementation Notes

## Architecture

### Data Flow
```
run_all_phases.py / run_systematic.py
    |
    v
src/runner.py  (run_experiment)
    |
    ├── src/data/datasets.py  (THzLikeCIFAR10 / THzLikeMNIST)
    │       |
    │       └── src/data/degrade.py  (DegradeConfig → degrade_image)
    |
    ├── torchvision / timm  (model loading)
    │       |
    │       └── src/models/transnext_wrapper.py  (TransNeXt integration)
    |
    ├── Training loop (AdamW, cosine LR, early stopping)
    |
    └── Post-training:
        ├── Save checkpoint (best.pt, model_last.pt)
        ├── Save metrics (metrics.csv, run_config.txt, log.txt)
        ├── Generate learning curves (learning_curves.png, overfitting_analysis.png)
        └── Auto-refresh dashboard (generate_experiment_plan_dashboard.py)
```

### Run Directory Structure
Each experiment creates a directory under `runs/systematic/`:
```
runs/systematic/<tag>_<model>_<config>/
    ├── run_config.txt      — experiment parameters
    ├── log.txt             — training log
    ├── metrics.csv         — per-epoch metrics (epoch, train_loss, val_loss, train_acc, val_acc)
    ├── best.pt             — best checkpoint (by val accuracy)
    ├── model_last.pt       — final epoch checkpoint
    ├── learning_curves.png — 2x2 grid (loss, accuracy)
    ├── overfitting_analysis.png — train vs val comparison
    └── metrics_summary.png — text summary card
```

### Experiment Tagging
Tags follow the pattern: `sys_L{level}_{severity}_{model}[_mnist]`

Examples:
- `sys_L1_mild_resnet50` — CIFAR-10, Level 1, ResNet50
- `sys_L2_moderate_densenet121_mnist` — MNIST, Level 2, DenseNet121
- `iso_downsampling_resnet50` — Phase C isolation, downsampling only
- `clean_cifar10_resnet50` — Phase D clean baseline

### Dashboard Generation
The experiment plan dashboard (`generate_experiment_plan_dashboard.py`) uses exact tag matching with boundary checks to avoid cross-dataset collisions. For each planned experiment, it scans `runs/systematic/` for a matching directory.

## Key Design Decisions

1. **Differential Learning Rate** (CNNs): Backbone at 1e-4, head at 1e-3 — prevents catastrophic forgetting while allowing adaptation to degraded inputs.

2. **Frozen Backbone for TransNeXt**: Full fine-tuning fails (~10% accuracy) due to overfitting on small dataset. Linear probe preserves ImageNet features.

3. **Identical Pipeline for Both Datasets**: MNIST (28x28 grayscale) is converted to 3-channel and processed through the same degradation pipeline as CIFAR-10 (32x32 RGB) for fair comparison.

4. **Early Stopping (patience=5)**: Most models converge in 15-27 epochs. Prevents wasted compute and overfitting.

5. **Label Smoothing 0.1 (CNNs only)**: Regularizes CNN training. Not used for TransNeXt LP because the head is already simple (single linear layer).
