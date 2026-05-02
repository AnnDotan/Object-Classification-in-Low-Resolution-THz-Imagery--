# Object Classification in Low-Resolution THz Imagery

> **Final Research Campaign — 186-cell experiment matrix.**
> 5-level degradation curve (L1 Mild → L5 Extreme), deterministic saturation axis, FP16 mixed precision, paper-anchored Optuna pre-tuning.
> Master tracker: [`Final_Exp.md`](Final_Exp.md) — interactive dashboard: `artifacts/Final_Exp.html`.

## Research Question

How robust does image classification remain when visual information is severely degraded — low resolution, blur, noise, salt-and-pepper, desaturation — simulating Terahertz (THz) imaging conditions? We evaluate three architectures (one residual CNN, one densely-connected CNN, one aggregated-attention ViT) across a 5-level degradation curve on two datasets (CIFAR-10, MNIST), with single-axis isolation runs to attribute the contribution of each degradation type.

## Implementation Status

This README describes the **target runbook**. The repo is mid-pivot from a legacy 36-experiment plan to the 186-cell final plan. Live vs. pending:

| Component | Status |
|---|---|
| 5-level degradation table — [`src/data/degradation_levels.py`](src/data/degradation_levels.py) | ✅ live |
| Saturation axis (deterministic lerp) — [`src/data/degrade.py`](src/data/degrade.py) | ✅ live |
| `degrade_config_for(level, axis)` helper | ✅ live |
| `Final_Exp.md` master tracker (186 cells, all Pending) | ✅ live |
| PyTorch Lightning training loop — [`src/lightning/`](src/lightning/) | ✅ live |
| Optuna pre-tuning — [`src/tune_hyperparams.py`](src/tune_hyperparams.py) | ✅ live (broad search; needs paper-anchored narrowing) |
| `saturation` plumbed through `DataConfig` → `DegradeConfig` | 🟡 pending |
| PSNR/SSIM logging in `setup()` (`metrics.json`) | 🟡 pending |
| FP16 mixed precision in main Trainer | 🟡 pending (live in tune_hyperparams.py only) |
| TransNeXt size selector (`--transnext_size {micro,small,base}`) + auto-download | 🟡 pending |
| `tune_all.py`, `setup.sh`, `scripts/verify_env.py`, `scripts/update_final_exp.py` | 🟡 pending |
| `src/tools/generate_final_dashboard.py` (`Final_Exp.html`) | 🟡 pending |
| `runs/final/` tree + `--plan final` switch in `run_all_phases.py` | 🟡 pending |

Until the pending pieces land, the working entry-point remains the legacy [`src/lightning/train.py`](src/lightning/train.py) / [`run_all_phases.py`](run_all_phases.py) flow against `runs/systematic/`. The legacy 33/36 results live in `runs/systematic/` and `runs/official/` and are frozen.

## Hardware & OS Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| GPU | 8 GB VRAM, CUDA 11.8+ | 24 GB (RTX 3090 / A100) |
| CPU | 8 cores | 16+ cores |
| RAM | 16 GB | 32 GB |
| Disk | 100 GB free | 250 GB SSD |
| OS | Windows 10/11, Ubuntu 20.04+, macOS 13+ | — |
| Python | 3.10+ | 3.12 |
| CUDA toolkit | 11.8 or 12.x | matching `torch` build |

## Step 0 — One-shot Environment Setup

```bash
git clone <repo-url>
cd Object-Classification-in-Low-Resolution-THz-Imagery--

# Bootstraps venv, pins torch+CUDA, installs requirements,
# downloads TransNeXt pretrained weights, runs FP16 smoke test.
bash setup.sh        # planned — see "Implementation Status"

# Manual fallback (works today):
python -m venv .venv
.venv\Scripts\activate            # Windows PowerShell
source .venv/bin/activate         # Linux / macOS
pip install -r requirements.txt
python scripts/verify_env.py      # planned — exits non-zero if CUDA missing
```

`scripts/verify_env.py` prints `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()`, device name, free VRAM, cuDNN version, and runs a tiny FP16 matmul. Exit non-zero ⇒ stop.

## Step 1 — Optuna Pre-Tuning (~20 GPU-hours)

Tune hyperparameters on a representative mid-level (L3 Moderate) for each `(model, dataset)` pair, then freeze them for the entire 186-cell sweep. Search space is **anchored on paper-derived priors** (TransNeXt, DenseNet, TResNet papers in `papers/`) — Optuna refines around them, doesn't blindly explore.

```bash
python tune_all.py --n-trials 20         # 6 pairs × 20 trials
```

Output: `artifacts/best_hparams/{model}_{dataset}.json` per pair. Resume on crash is automatic via the `artifacts/optuna_thz.db` SQLite store.

For a single-pair smoke test:

```bash
python src/tune_hyperparams.py --model resnet50 --dataset cifar10 --n-trials 5 --level 3
```

## Step 2 — Run the 186-cell Matrix

```bash
# Phase A — clean baselines (6 runs, ~1.5 h on RTX 3090)
python run_all_phases.py --plan final --phase A

# Phase B — combined degradation (30 runs, ~10 h)
python run_all_phases.py --plan final --phase B

# Phase C — single-axis isolation (150 runs, ~50 h)
python run_all_phases.py --plan final --phase C

# Full pipeline (idempotent — safe to Ctrl-C and resume)
python run_all_phases.py --plan final --phase all --skip-existing --tune-first
```

Outputs land in `runs/final/<tag>/`. Each run writes:
- `metrics.json` — `psnr_mean`, `psnr_std`, `ssim_mean`, `ssim_std`, `best_val_acc`, `best_epoch`, dataset config
- `metrics.csv` — per-epoch loss/accuracy
- `best.ckpt`, `last.ckpt` — Lightning checkpoints (gitignored, claudeignored, **never leave the local machine**)
- `log.txt` — training log

## Step 3 — Monitor

| Surface | What it shows |
|---|---|
| `artifacts/Final_Exp.html` | 186-cell visual grid: side-by-side original/degraded thumbnails, status badges, PSNR/SSIM, click → learning-curve modal. Auto-refreshes every 30 s. |
| [`Final_Exp.md`](Final_Exp.md) | Master tracker; Markdown table for every cell. Regenerated by `scripts/update_final_exp.py` on every `on_train_end`. |
| `runs/final/<tag>/log.txt` | Live training output — `tail -f` it during a run. |

## Degradation Pipeline & Levels

Identical pipeline for both datasets (saturation applied **before** noise/S&P so additive noise stays color-correct):

```
Original (32×32 CIFAR-10 / 28×28 MNIST → 3-channel)
   ↓ Saturation lerp:  (1−s)·gray + s·img            ← deterministic
   ↓ Downsample → bilinear upsample to 224×224
   ↓ Gaussian blur (separable conv)
   ↓ Additive Gaussian noise
   ↓ Salt-and-pepper noise
   ↓ ImageNet normalization
Model input (224×224×3)
```

5-level degradation table (single source of truth: [`src/data/degradation_levels.py`](src/data/degradation_levels.py)):

| Level | Name | low_res | blur kernel | blur σ | noise std | S&P | saturation |
|---|---|---|---|---|---|---|---|
| L1 | Mild | 20 | 3 | 0.70 | 0.03 | 0.02 | 1.00 |
| L2 | Light | 14 | 5 | 1.00 | 0.06 | 0.05 | 0.75 |
| L3 | Moderate | 10 | 7 | 1.30 | 0.09 | 0.08 | 0.50 |
| L4 | Severe | 7 | 9 | 1.65 | 0.13 | 0.11 | 0.25 |
| L5 | Extreme | 4 | 11 | 2.00 | 0.18 | 0.15 | 0.00 |

Phase C (single-axis isolation) sweeps one axis through L1→L5 with every other axis pinned at L1 mild values.

## Models

| Model | Type | Strategy | Paper |
|---|---|---|---|
| **ResNet50** | residual CNN | differential LR fine-tuning (head 1e-3, backbone 1e-4) | TResNet (Ridnik et al., 2020) |
| **DenseNet121** | densely-connected CNN | differential LR fine-tuning | DenseNet (Huang et al., 2017) |
| **TransNeXt** (default size: `small`) | aggregated-attention ViT | linear probe — frozen ImageNet backbone, head-only | TransNeXt (Shi, 2024) |

## The 186-cell Matrix

| Phase | Description | Count |
|---|---|---|
| **A** Clean baselines | 3 models × 2 datasets × 1 (no degradation) | **6** |
| **B** Combined degradation | 3 models × 2 datasets × 5 levels (all axes at L) | **30** |
| **C** Single-axis isolation | 3 models × 2 datasets × 5 levels × 5 axes | **150** |
| | | **186** |

Tag scheme: `final_clean_{model}_{dataset}` / `final_B_L{level}_{model}_{dataset}` / `final_C_L{level}_{axis}_{model}_{dataset}`. At L1 every isolation cell collapses to the Phase B L1 row for the same `(model, dataset)`; `--skip-existing` deduplicates these at runtime.

## Training Defaults — Convergence-First

| Parameter | Value | Source |
|---|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) | TransNeXt paper |
| Scheduler | Cosine LR decay | TransNeXt / EfficientNetV2 |
| Head LR / Backbone LR (CNNs) | 1e-3 / 1e-4 | TResNet, DenseNet |
| TransNeXt LP | head 1e-3, backbone frozen | TransNeXt paper §A.3 |
| Weight decay | 1e-4 (CNNs) / 5e-2 (TransNeXt) | DenseNet / TransNeXt |
| Label smoothing | 0.1 (CNNs) / 0.0 (TransNeXt LP) | TransNeXt paper |
| Gradient clipping | max_norm = 1.0 | TransNeXt paper |
| Batch size | 32 | GPU memory |
| **Max epochs** | **60** | quality-over-speed |
| **Early stopping** | **patience=10, min_delta=1e-4, monitor=val_acc, mode=max** | quality-over-speed |
| Precision | `16-mixed` (auto fallback to `32-true` if no CUDA) | FP16 mixed precision |
| Seed | `pl.seed_everything(42, workers=True)` | reproducibility lock |

Pilot mode (`--mode pilot`) keeps shorter numbers (5 epochs / patience 2) for smoke tests **only**. `run_all_phases.py --plan final --mode pilot` is rejected with an assertion to prevent contamination of final results.

## Reproducibility

Per-sample local RNG keyed on dataset index (`seed = idx + SEED_OFFSET_VAL`) gives byte-identical validation pixels across every model and run. Two reads of the same val batch satisfy MSE = 0; gated by [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py). PSNR/SSIM are recomputed deterministically on a fixed 256-sample subset and asserted byte-identical across re-runs.

```bash
pytest src/tests/test_degradation_determinism.py -v
```

## Weight Privacy

Trained checkpoints (`*.ckpt`, `*.pt`, `*.pth`) and the Optuna SQLite store are **local-only**. They are excluded from Git via `.gitignore` and from AI assistant context via `.claudeignore`. Future Claude / Codex sessions analyze runs by reading `metrics.json` / `metrics.csv` / the dashboard HTML — never the binary artifacts. TransNeXt pretrained weights under `artifacts/weights/` are auto-downloaded on demand and also gitignored.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `verify_env.py` exits non-zero | CUDA / `torch` build mismatch | re-install matching `torch` per [pytorch.org/get-started](https://pytorch.org/get-started/locally/) |
| OOM on TransNeXt-S | batch 32 too large for 8 GB VRAM | `--batch-size 16` or `--transnext_size micro` |
| TransNeXt weight download blocked | corporate proxy / firewall | manual fallback URL printed in log; place file in `artifacts/weights/` |
| Optuna trial pruned | MedianPruner — expected | no action — pruned trials still record |
| `--plan final --mode pilot` rejected | guardrail prevents short-training contamination | drop `--mode pilot` or run a real pilot via `--plan legacy` |

## Codebase Structure

```
Final_Exp.md                       — master tracker, 186 rows
src/data/degradation_levels.py     — single-source 5-level table
src/data/degrade.py                — DegradeConfig + degrade_config_for
src/data/datasets.py               — THzLikeCIFAR10 / THzLikeMNIST
src/lightning/                     — Lightning training engine
src/models/                        — model wrappers (TransNeXt + size selector)
src/tools/                         — dashboard generators
src/tune_hyperparams.py            — Optuna single-pair sweep
src/tests/                         — determinism + saturation invariants
tune_all.py                        — Optuna driver across all 6 pairs (planned)
run_all_phases.py                  — final & legacy phase runner
runs/final/                        — 186-cell campaign outputs (gitignored)
runs/systematic/                   — frozen legacy 33-run results (kept for reference)
artifacts/best_hparams/            — Optuna outputs per (model, dataset)
artifacts/Final_Exp.html           — interactive 186-cell dashboard (planned)
artifacts/weights/                 — auto-downloaded pretrained weights (gitignored)
papers/                            — TransNeXt, DenseNet, TResNet, EfficientNetV2, NASNet
agents/                            — sub-agent specs (MASTER + 9 specialists)
```

## Deadlines

| Date | Deliverable |
|---|---|
| 2026-05-31 | Poster & abstract |
| 2026-06-21 | Final presentation |
| 2026-07-26 | Final submission |

## Contributors

Itamar Bahat

## License

Academic and research use.
