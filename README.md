# Object Classification in Low-Resolution THz Imagery

> **Final Research Campaign — 186-cell experiment matrix.**
> 5-level degradation curve (L1 Mild → L5 Extreme), deterministic saturation axis, FP16 mixed precision, paper-anchored Optuna pre-tuning.
> Master tracker: [`Final_Exp.md`](Final_Exp.md) — interactive dashboard: `artifacts/Final_Exp.html`.

## Research Question

How robust does image classification remain when visual information is severely degraded — low resolution, blur, noise, salt-and-pepper, desaturation — simulating Terahertz (THz) imaging conditions? We evaluate three architectures (one residual CNN, one densely-connected CNN, one aggregated-attention ViT) across a 5-level degradation curve on two datasets (CIFAR-10, MNIST), with single-axis isolation runs to attribute the contribution of each degradation type.

## Implementation Status

| Component | Status |
|---|---|
| 5-level degradation table — [`src/data/degradation_levels.py`](src/data/degradation_levels.py) | ✅ live |
| Saturation axis plumbed through `DataConfig` → `DegradeConfig` | ✅ live (US-001) |
| `degrade_config_for(level, axis)` helper | ✅ live |
| 186-cell matrix generator — [`src/experiments/matrix.py`](src/experiments/matrix.py) | ✅ live (US-007) |
| PyTorch Lightning training loop — [`src/lightning/`](src/lightning/) | ✅ live |
| Offline PSNR/SSIM script — [`src/tools/measure_image_quality.py`](src/tools/measure_image_quality.py) | ✅ live (US-002) |
| Per-model paper-anchored priors — [`artifacts/priors/*.json`](artifacts/priors/) | ✅ live (US-003) |
| `tune_all.py` priors loader + Optuna study runner | ✅ live (US-004 + US-005) |
| TransNeXt `base` + full-FT mode + auto-download | ✅ live (US-006) |
| `--cell-tag` matrix consumer in [`run_systematic.py`](run_systematic.py) | ✅ live (US-008) |
| `--plan final` orchestration in [`run_all_phases.py`](run_all_phases.py) | ✅ live (US-009) |
| Pre-rendered Original-vs-Degraded thumbs — [`src/tools/render_cell_thumbs.py`](src/tools/render_cell_thumbs.py) | ✅ live (US-010) |
| Final_Exp.html dashboard — [`src/tools/build_final_dashboard.py`](src/tools/build_final_dashboard.py) | ✅ live (US-011, rewritten as the FINAL_EXP Dashboard — pilot-styled tabs / chip filters / 30 s polling — see [`docs/prds/FINAL_EXP_DASHBOARD.md`](docs/prds/FINAL_EXP_DASHBOARD.md)) |
| `artifacts/Final_Exp.json` aggregate + `src/tools/build_final_exp_json.py` | ✅ live (FINAL_EXP US-001..US-003) |
| W&B `run_id` / `entity` / `project` in `metrics.json` | ✅ live (US-012) |
| Weight-privacy hardening (`.gitignore`, `.claudeignore`, `scripts/check_ignores.sh`) | ✅ live (US-013) |
| `scripts/update_final_exp.py` (regenerates `Final_Exp.md` row data after each run) | ✅ live |
| `scripts/fetch_transnext_weights.py` (pre-fetch checkpoint into `artifacts/weights/`) | ✅ live |
| Click-to-toggle learning-curve panels in `Final_Exp.html` (`render_curve_thumbs.py`) | 🗑 deprecated — superseded by the chip-filtered FINAL_EXP Dashboard; thumbs still rendered for ad-hoc inspection. |

Legacy 33/36 results in `runs/systematic/` are frozen and kept for reference.

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

## Step 0 — Environment Setup

**Bash (Linux / macOS / git-bash):**
```bash
git clone <repo-url>
cd Object-Classification-in-Low-Resolution-THz-Imagery--
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.tests.test_degradation_determinism   # MSE=0 sanity check
```

**PowerShell (Windows):**
```powershell
git clone <repo-url>
Set-Location Object-Classification-in-Low-Resolution-THz-Imagery--
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\venv\Scripts\python.exe -m src.tests.test_degradation_determinism
```

If the determinism test prints `OK [cifar10] ...`, `OK [mnist] ...`, `OK [lightning] ...` you are ready. The first run will download CIFAR-10 / MNIST into `./data/` (≈ 200 MB combined).

## Step 1 — Dataset Setup

CIFAR-10 and MNIST are downloaded automatically into `./data/` on first use by torchvision. To pre-warm the cache without starting training:

**Bash:**
```bash
python -c "from torchvision import datasets; datasets.CIFAR10('./data', train=True, download=True); datasets.MNIST('./data', train=True, download=True)"
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe -c "from torchvision import datasets; datasets.CIFAR10('./data', train=True, download=True); datasets.MNIST('./data', train=True, download=True)"
```

TransNeXt-Base ImageNet-1K weights auto-download to `artifacts/weights/transnext_base_224_1k.pth` on first training run. To pre-fetch them upfront (recommended — surfaces network problems before Phase A starts and avoids latency on the first epoch):

**Bash:**
```bash
python scripts/fetch_transnext_weights.py                  # base only (campaign default)
python scripts/fetch_transnext_weights.py --all            # every size
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe scripts\fetch_transnext_weights.py
.\venv\Scripts\python.exe scripts\fetch_transnext_weights.py --all
```

Files smaller than 1 MiB (Git-LFS pointers, partial downloads, accidental placeholders) are detected and re-downloaded automatically. Override the source URL via the `THZ_TRANSNEXT_BASE_URL` env var (or `THZ_TRANSNEXT_<SIZE>_URL` for any size) if the default GitHub release URL is unreachable; if both fail the training script raises `FileNotFoundError` with a manual-download instruction.

## Step 2 — Optuna Pre-Tuning (~20 GPU-hours)

Tune hyperparameters on Phase B L3 Moderate for each `(model, dataset)` pair, then freeze them for the 186-cell sweep. Search space is **anchored on paper-derived priors** (`artifacts/priors/{resnet50,densenet121,transnext_base}.json`) — every range is held to ≤ 1 decade around the paper anchor.

**Bash:**
```bash
python tune_all.py --validate-only          # dry-run: validates priors files
python tune_all.py --n-trials 20            # 6 pairs × 20 trials
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe tune_all.py --validate-only
.\venv\Scripts\python.exe tune_all.py --n-trials 20
```

Output: `artifacts/best_hparams/{model}_{dataset}.json` per pair. Each winner JSON carries `priors_file_hash` (SHA-256 of the priors used) so you can audit which search bounds produced the result. Resume on crash is automatic via the `artifacts/optuna_thz.db` SQLite store.

To tune just one pair:

**Bash:**
```bash
python tune_all.py --n-trials 20 --model resnet50 --dataset cifar10
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe tune_all.py --n-trials 20 --model resnet50 --dataset cifar10
```

## Step 3 — Run the 186-cell Matrix

**Bash:**
```bash
# Phase A — clean baselines (6 runs)
python run_all_phases.py --plan final --phase A

# Phase B — combined degradation (30 runs)
python run_all_phases.py --plan final --phase B

# Phase C — single-axis isolation (150 runs)
python run_all_phases.py --plan final --phase C

# Full pipeline (idempotent — safe to Ctrl-C and resume; auto-tunes if any best_hparams missing)
python run_all_phases.py --plan final --phase all --skip-existing --tune-first
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe run_all_phases.py --plan final --phase A
.\venv\Scripts\python.exe run_all_phases.py --plan final --phase B
.\venv\Scripts\python.exe run_all_phases.py --plan final --phase C
.\venv\Scripts\python.exe run_all_phases.py --plan final --phase all --skip-existing --tune-first
```

To run a single cell ad-hoc:

**Bash:**
```bash
python run_systematic.py --cell-tag final_B_L3_resnet50_cifar10
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe run_systematic.py --cell-tag final_B_L3_resnet50_cifar10
```

Outputs land in `runs/final/<tag>/`. Each run writes:
- `metrics.json` — `best_val_acc`, `best_epoch`, `last_*_acc/loss`, `epochs_run`, plus `wandb_run_id` / `wandb_entity` / `wandb_project` (US-012, may be `null` offline), `hparams` + `hparams_source` (US-008), and `cell_tag` / `phase` / `level` / `axis` / `model` / `dataset`.
- `metrics.csv` — per-epoch loss/accuracy
- `best.pt`, `model_last.pt` — Lightning checkpoints (gitignored, claudeignored, **never leave the local machine**)
- `log.txt` — training log

## Step 4 — Inspect Results

| Surface | What it shows |
|---|---|
| `artifacts/Final_Exp.html` | FINAL_EXP Dashboard — pilot-styled, tabbed by Phase A/B/C, with multi-select chip filters (model / dataset / status / axis), graded `L1`–`L5` level badges with parameter tooltips, status pills, `val_acc` / epochs / runtime per row, and 30 s JSON polling for live updates. Initial data is embedded inline so the dashboard works directly via `file://`; polling activates when served over HTTP. |
| `artifacts/Final_Exp.json` | Aggregate of all 186 cells in a stable schema (`src/tools/final_exp_schema.py`) — the data contract the dashboard polls and the source of truth for downstream tools. |
| [`Final_Exp.md`](Final_Exp.md) | Master tracker — Markdown table per cell + `metrics.json` schema reference. |
| `runs/final/<tag>/log.txt` | Live training output. `Get-Content -Wait` (PowerShell) or `tail -f` (Bash). |
| `artifacts/priors/*.json` | Paper-anchored Optuna search bounds (tracked in git). |
| `artifacts/best_hparams/*.json` | Optuna winner per `(model, dataset)` — locally ignored (machine-specific). |

Refresh / regenerate after a run:

**Bash:**
```bash
python scripts/update_final_exp.py                    # regenerate Final_Exp.md row data
python -m src.tools.render_cell_thumbs                # 186 Original|Degraded PNG pairs (idempotent)
python -m src.tools.render_curve_thumbs               # per-cell val_acc / val_loss curves (skips cells with no metrics.csv)
python -m src.tools.build_final_dashboard             # rebuild Final_Exp.html
python -m src.tools.measure_image_quality \
    --cell-tag final_B_L3_resnet50_cifar10 \
    --out runs/final/final_B_L3_resnet50_cifar10/image_quality.json
```

**PowerShell:**
```powershell
.\venv\Scripts\python.exe scripts\update_final_exp.py
.\venv\Scripts\python.exe -m src.tools.render_cell_thumbs
.\venv\Scripts\python.exe -m src.tools.render_curve_thumbs
.\venv\Scripts\python.exe -m src.tools.build_final_dashboard
.\venv\Scripts\python.exe -m src.tools.measure_image_quality `
    --cell-tag final_B_L3_resnet50_cifar10 `
    --out runs/final/final_B_L3_resnet50_cifar10/image_quality.json
```

`scripts/update_final_exp.py --check` exits non-zero if `Final_Exp.md` is out of date — useful for CI / pre-commit hooks.

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
| **TransNeXt-Base** (default size for the 186-cell campaign per US-006) | aggregated-attention ViT | full fine-tuning (no frozen backbone) — `--transnext_size base --transnext_mode ft` | TransNeXt (Shi, 2024) §A.3 |

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

## Privacy Notes

Binary weight artifacts never leave the local machine: trained checkpoints (`*.ckpt`, `*.pt`, `*.pth`), the Optuna SQLite store, dashboard thumbs, and pretrained TransNeXt weights under `artifacts/weights/` are excluded from Git via [`.gitignore`](.gitignore) and from AI assistant context via [`.claudeignore`](.claudeignore). The contract is gated by [`scripts/check_ignores.sh`](scripts/check_ignores.sh) and its Python sibling [`src/tests/test_ignores.py`](src/tests/test_ignores.py) — both confirm 13 excluded categories ignore correctly, the 4 paper-anchored priors files stay tracked, and zero binary weight files are in the git index. Future AI sessions analyze runs by reading `metrics.json` / `metrics.csv` / the dashboard HTML — never the binary artifacts.

```bash
bash scripts/check_ignores.sh                          # smoke test the contract
python -m src.tests.test_ignores                       # cross-platform variant
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `import torch` fails | venv not activated | activate `venv/` (note: NOT `.venv/`); see Step 0 |
| `FileNotFoundError: missing best_hparams: artifacts/best_hparams/<m>_<d>.json` | tuning never ran for this `(model, dataset)` | `python tune_all.py --n-trials 20 --model M --dataset D` (or `--tune-first` on `run_all_phases.py`) |
| `FileNotFoundError: Pretrained TransNeXt weights missing: artifacts/weights/transnext_base_224_1k.pth` | auto-download URL unreachable | set `THZ_TRANSNEXT_BASE_URL` to a working mirror, or manually drop the file at the printed path |
| OOM on TransNeXt-Base | batch 32 too large for 8 GB VRAM | `--batch-size 16` or fall back to `--transnext_size small` |
| `--plan final --mode pilot` rejected with `AssertionError` | guardrail prevents short-training contamination | drop `--mode pilot`, or run pilot via `--plan legacy` |
| Optuna trial pruned | normal pruner behavior | no action — trial state still recorded in `artifacts/optuna_thz.db` |
| Determinism test fails (MSE > 0) | data pipeline mutation broke seed-per-index contract | revert recent changes to `src/data/datasets.py` or `src/data/degrade.py` |
| `git check-ignore` returns wrong category | `.gitignore` regression | run `python -m src.tests.test_ignores` to see the failing rule |

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
