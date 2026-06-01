# Project: Low-Resolution Image Classification (THz-like)

## Goal
Evaluate deep-learning robustness under severe visual degradation — low resolution, blur, noise, salt-and-pepper, desaturation — simulating Terahertz imaging. Quality-over-speed: train every cell to convergence, no shortcuts.

## Project Status (2026-06-01)

- **Uniform 224×224 protocol (ratified 2026-05-13):** every model — ResNet50, DenseNet121, **TransNeXt** — trains at **224×224**. The V3 native-resolution refactor (32×32 for TransNeXt, 28→32 pad for MNIST) was rolled back; the fair-comparison invariant is back to *same tensor shape* (224×224 after upsample). TransNeXt loads its 224-pretrained checkpoint and uses full-FT with differential LR (head 5e-4 / backbone 5e-5).
- **v4 Campaign closed (2026-06-01):** the v3 276-cell baseline (186 v2 + 90 Phase D) was extended to **402 canonical cells** plus 48 multi-seed audit replicates (`runs/final/`). The new scope landed under PRD v4 on the `PHASE_B2` branch: 30 Phase B2 + 6 Phase B2-nr (L3) + 90 Phase C2 + 48 multi-seed runs across the 24 L3 headline cells at seeds {43, 44}. **TransNeXt US-014 quarantine lifted** — RTX 5070 / sm_120 Blackwell hardware is online; quarantine predicate is a permanent no-op.
- **Engine:** PyTorch Lightning in [`src/lightning/`](src/lightning/) (`THzClassifier`, `THzDataModule`); `pl.seed_everything(42, workers=True)` for canonical cells; multi-seed audit uses `--seed 43` / `--seed 44`.
- **Deterministic validation:** per-sample seeded degradation (`seed = idx + SEED_OFFSET_VAL`) gives byte-identical val pixels at out_size=224 across freshly-built datasets. Multi-seed audits verify byte-identity across {42, 43, 44}. Gated by [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py) — extended for the v4 Phase B2 / B2-nr / C2 DegradeConfig overrides.
- **Master tracker:** [`Final_Exp.md`](Final_Exp.md) — **402 canonical rows** (186 v2 + 90 Phase D + 30 Phase B2 + 6 Phase B2-nr + 90 Phase C2). Multi-seed `[mean ± std]` aggregation lives in `artifacts/Final_Exp.json` (`schema_version=3`).
- **Single source of truth for levels:** [`src/data/degradation_levels.py`](src/data/degradation_levels.py). Phase B2 + B2-nr + C2 apply DegradeConfig overrides on top (sat=0, noise_std=0).
- **2026-06-01 — v4 full scientific-depth scope closed (US-036..US-053 in flight; US-036..US-049 closed end-to-end).** 5 new operator-locked findings: (5) Phase B2 protocol simplification recovers ~3.24 pp pure-protocol Δ_D→B2 and +4.01 pp combined Δ_B1→B2, ~80% of which is pure protocol rather than regularization; (6) Resolution dominates the THz axis attribution by ~4× (−27.45 pp mean L1–L5 vs blur −6.58 / S&P −5.13); (7) Multi-seed audit confirms headline accuracies stable to σ < 1.5 pp on all 24 L3 base tags; (8) Calibration ECE rises from < 5 pp at L3 to > 15 pp at L5 on CIFAR-10 — the model retains confidence as the spatial information collapses. See [docs/phase_b2.md](docs/phase_b2.md), [docs/phase_c2.md](docs/phase_c2.md), and `Final_Report.pdf` rev3 §VIII–§XIII.
- **2026-05-26 — Phase D regularization sweep closed (90 / 90 cells healthy, 56.5 GPU-h, zero pathology-guard sentinels).** Operator-locked answer to the §1 research question *"Does regularization recover the Phase B v2 −12.49 pp mean collapse vs v1?"*: **Regularization recovers a small fraction of the v2 collapse (mean Δ = +0.77 pp under T3 vs the −12.49 pp v2 drop). The collapse is largely a fundamental information bottleneck, not an overfit signal.** Best segment: T3 on TransNeXt × CIFAR-10 at **+2.20 pp mean** (5 levels). Superseded by the v4 Finding #5 above — the v3 +0.77 pp T3 recovery on the full pipeline is consistent with the v4 Phase B2-nr → B2 +1.39 pp T3 recovery on the simpler protocol (T3 is ~80% more effective when protocol is simpler).

## Models

| Model | Input | Training | Source |
|---|---|---|---|
| **ResNet50** | 224×224 | Differential-LR full FT (ImageNet pretrain) | TResNet / EfficientNetV2 |
| **DenseNet121** | 224×224 | Differential-LR full FT (ImageNet pretrain) | DenseNet paper |
| **TransNeXt (tiny)** | **224×224** | **Full FT** with differential LR (head 5e-4 / backbone 5e-5); `pretrain_size=224` matches the upstream checkpoint. Swapped from `transnext_small` by US-016 on 2026-05-14 (~28M vs ~50M params; tighter capacity-match to ResNet50 ~25M and ~310 GPU-h saved across the 62 TransNeXt rows vs small). Predecessor swap from `transnext_base` (US-004, same day) preserved in history. | TransNeXt §A.3 |

All three models share the same input resolution. CIFAR-10 (32) and MNIST (28) inputs are upsampled to 224 by the data pipeline before reaching any model. This preserves the ImageNet-pretrained receptive-field hierarchy for the CNNs and matches the TransNeXt paper's 224 training distribution end-to-end.

## Datasets

- **CIFAR-10** — 10 object classes, 32×32 RGB, 10K train / 5K val
- **MNIST** — 10 digit classes, 28×28 grayscale → 3-channel, 10K train / 5K val. Upsampled 28→224 by the degradation pipeline (no separate pad-to-32 path).

Pipeline (uniform across models):
- `Original → saturation lerp → downsample(low_res) → upsample 224×224 (bicubic) → blur → noise → S&P → ImageNet normalize → Model`

## 186/276/402-cell Experiment Plan

| Phase | Description | Count |
|---|---|---|
| **A** | Clean baselines (3 models × 2 datasets × 1 no-degradation) | **6** |
| **B** (B1) | Combined degradation (× 5 levels, all axes at L) | **30** |
| **C** | Single-axis isolation (× 5 levels × 5 axes) | **150** |
| **D** | Regularization sweep (× 3 treatments × 5 levels × all axes at L) | **90** |
| **B2** | THz-protocol simplification — combined sweep with sat=0, noise=0 (× 5 levels × all axes-minus-2 at L, T3 regularization) | **30** |
| **B2-nr** | Phase B2 no-regularization arm at L3 only (pure-protocol decomposition) | **6** |
| **C2** | Single-axis isolation under the THz protocol (× 3 axes × 5 levels, T3) | **90** |
| | | **402** |
| **Multi-seed audit (US-042, 2026-06-01)** | 24 L3 base tags × 2 extra seeds {43, 44} | **48 audit replicates** |

Tag scheme: `final_clean_{m}_{d}` / `final_B_L{l}_{m}_{d}` (Phase B1, code `phase=="B"`) / `final_C_L{l}_{ax}_{m}_{d}` / `final_D_{T}_L{l}_{m}_{d}` / `final_B2_L{l}_{m}_{d}` / `final_B2nr_L3_{m}_{d}` / `final_C2_L{l}_{ax}_{m}_{d}`. Multi-seed audit replicates append `_seed{N}` to the base tag. Phase C axes: `resolution`, `noise`, `blur`, `saturation`, `salt_pepper`. Phase C2 axes: `resolution`, `blur`, `salt_pepper` (the THz-relevant spatial-domain subset). Phase D treatments: `T1`, `T2`, `T3` (see [docs/phase_d.md](docs/phase_d.md)). Phase B2 + B2-nr + C2 rationale: [docs/phase_b2.md](docs/phase_b2.md), [docs/phase_c2.md](docs/phase_c2.md).

## Degradation Levels (5-level curve)

| Level | Name | low_res | blur kernel | blur σ | noise std | S&P | saturation |
|---|---|---|---|---|---|---|---|
| L1 | Mild | 18 | 13 | 2.50 | 0.04 | 0.03 | 0.95 |
| L2 | Light | 12 | 25 | 5.00 | 0.08 | 0.06 | 0.65 |
| L3 | Moderate | 8 | 41 | 8.00 | 0.12 | 0.10 | 0.40 |
| L4 | Severe | 6 | 61 | 12.00 | 0.16 | 0.14 | 0.15 |
| L5 | Extreme | 3 | 91 | 18.00 | 0.22 | 0.18 | 0.00 |

Blur kernel/σ rescaled 2026-05-14: the prior values (K=3..13, σ=0.80..2.30) were designed for native-size (32×32) imagery; on the 224×224 upsampled tensor they covered ~1–6% of image width and produced essentially invisible blur. The values above are pixel-domain at 224×224 (kernel = 2·⌈2.5σ⌉+1). Other axes unchanged from the 2026-05-12 severity bump.

Saturation is a deterministic lerp: `(1−s)·gray + s·img`, applied **before** noise/S&P so noise color stays correct. Replaces the legacy stochastic `p_grayscale` (kept as a no-op field for backwards-compat).

Phase C single-axis isolation (US-003, 2026-05-14): named axis at level L, every other axis at **identity** (no degradation). At L1, the active axis is at L1 mild while the four inactive axes are at identity — so a Phase C L1 cell is NOT equivalent to Phase B L1 (which has all five axes at L1 mild). See `docs/phase_c.md` for the scientific rationale and the `degradation_levels_hash` rotation note.

## Training Hyperparameters (paper-anchored, convergence-first)

| Parameter | Value | Source |
|---|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) | TransNeXt paper |
| Scheduler | Cosine LR decay | TransNeXt / EfficientNetV2 |
| Head LR / Backbone LR (CNNs) | 1e-3 / 1e-4 | TResNet, DenseNet |
| **TransNeXt FT** | **head 5e-4, backbone 5e-5** (10× differential) | TransNeXt §A.3 + CNN differential-LR convention |
| Weight decay | 1e-4 (CNNs) / 5e-2 (TransNeXt) | DenseNet / TransNeXt |
| Label smoothing | 0.1 (CNNs) / **0.1 (TransNeXt FT)** | TransNeXt paper |
| Drop-path rate | 0.0 (CNNs) / **0.1 (TransNeXt FT)** | TransNeXt paper stochastic depth |
| Gradient clipping | max_norm = 1.0 | TransNeXt paper |
| Batch size | 32 | GPU memory |
| **Max epochs** | **60** | quality over speed |
| **Early stopping** | **patience=10, min_delta=1e-4, monitor=val_acc, mode=max** | quality over speed |
| **Precision** | **`bf16-mixed`** (Blackwell) / `32-true` (CPU) | RTX 5070 sm_120; no loss-scaler reproducibility win over fp16 |
| **`torch.compile`** | `none` (all models) | TransNeXt's `attention_native` path is not Inductor-graph-capturable and the CNN compile path is unvalidated |
| Seed | 42 (workers=True) | reproducibility lock |

TransNeXt full-FT priors (used when `artifacts/best_hparams/transnext_tiny_{dataset}.json` is absent) live in [`run_systematic.py`](run_systematic.py) `V3_TRANSNEXT_FT_PRIORS`. CNN priors for Phase A continue to live in `PHASE_A_FROZEN_HPARAMS`; Phase B/C cells without an Optuna JSON are a hard error.

`--mode pilot` (5 epochs / patience 2) is for smoke tests **only**. `run_all_phases.py --plan final --mode pilot` is rejected with an assertion.

## Optuna Pre-Tuning (Sprint 4, planned)

Tune each `(model, dataset)` pair on L3 Moderate, freeze the winning hyperparameters for the entire 186-cell sweep. Search space narrowed to **paper-derived priors ± 1 decade max** (no blind exploration). Budget: 6 pairs × 20 trials ≈ 20 GPU-hours.

```bash
python tune_all.py --n-trials 20
```

Outputs: `artifacts/best_hparams/{model}_{dataset}.json`. Resume on crash via `artifacts/optuna_thz.db`.

## Experiment System

```bash
# Optuna pre-tune (planned)
python tune_all.py --n-trials 20

# Run the 186 (planned --plan final flag; falls back to legacy paths today)
python run_all_phases.py --plan final --phase A                            # 6 runs ~1.5h
python run_all_phases.py --plan final --phase B                            # 30 runs ~10h
python run_all_phases.py --plan final --phase C                            # 150 runs ~50h
python run_all_phases.py --plan final --phase all --skip-existing --tune-first

# Phase D regularization sweep (90 cells, ~52.5 GPU-h)
python run_all_phases.py --plan final --phase D

# v4 phases (PRD v4)
python scripts/run_ralph_loop.py --phase B2          # 30 cells THz-protocol simplification ~15 GPU-h
python scripts/run_ralph_loop.py --phase B2nr        # 6 L3 cells no-regularization arm ~3.5 GPU-h
python scripts/run_ralph_loop.py --phase C2          # 90 cells single-axis THz attribution ~48 GPU-h
python scripts/run_ralph_loop.py --phase multiseed --seeds 43,44 --cells <24 tag list>   # 48 audit replicates ~24 GPU-h

# US-045 post-training diagnostics (no re-train; ~3.5 GPU-h)
python scripts/build_test_split_indices.py            # seed=99 stratified test split
python scripts/run_test_set_inference.py [--skip-existing]
python scripts/generate_confusion_matrices.py [--skip-existing]
python scripts/generate_calibration_diagrams.py [--skip-existing]
python scripts/measure_inference_throughput.py
python scripts/build_diagnostics_latex.py             # emit §XI / §XII tables

# US-048 plots (no GPU)
python scripts/plot_phase_b2_comparison.py
python scripts/plot_phase_c2_attribution.py
python scripts/plot_multi_seed_variance.py

# Live monitor: artifacts/Final_Exp.html (7-tab dashboard) + Final_Exp.md (402 rows)
```

### Phase D treatments (regularization recovery layer)

| Treatment | CNN delta | TransNeXt delta | Purpose |
|---|---|---|---|
| **T1** | `dropout=0.2` | `drop_path_rate=0.2` | Architectural regularization only |
| **T2** | `mixup_alpha=0.2`, `cutmix_alpha=0.0` | same | Label-mixing only |
| **T3** | `dropout=0.2` + `mixup_alpha=0.2` + `cutmix_alpha=1.0` | `drop_path_rate=0.2` + `mixup_alpha=0.2` + `cutmix_alpha=1.0` | Combo / kitchen-sink |

Treatments layer on top of the frozen v2 L3-Optuna winners — no re-tune. CNN routing uses timm's `drop_rate`; TransNeXt routing uses `drop_path_rate` (stochastic depth) because timm CNNs do not honor `drop_path_rate`. Rationale + per-(model, dataset) recovery table: [docs/phase_d.md](docs/phase_d.md).

## Codebase Structure

```
Final_Exp.md                       — 402-cell master tracker
src/data/degradation_levels.py     — 5-level table + level_params + AXIS_KEYS + IDENTITY_VALUES
src/data/degrade.py                — DegradeConfig + degrade_config_for + degrade_image
src/data/datasets.py               — THzLikeCIFAR10 / THzLikeMNIST
src/lightning/                     — Lightning training engine (THzClassifier + THzDataModule)
src/models/                        — model wrappers (TransNeXt-tiny + weights loader)
src/tools/                         — dashboard generators, schemas, diagnostics_common helpers
src/tune_hyperparams.py            — Optuna single-pair sweep
src/tests/                         — determinism + saturation invariants + multi-seed byte-id
runs/final/                        — 402 canonical cells + 48 multi-seed audit replicates (gitignored)
runs/systematic/                   — frozen 33-run legacy results
artifacts/best_hparams/            — Optuna outputs per (model, dataset)
artifacts/Final_Exp.html           — interactive 7-tab dashboard (US-047)
artifacts/Final_Exp.json           — schema_version=3 aggregator with multi-seed [mean ± std]
artifacts/validation/              — manifest snapshots + test_split_indices.json + diagnostics outputs
artifacts/figures/                 — comparison plots + per-(m,d) panels + confusion + calibration + throughput
artifacts/weights/                 — auto-downloaded pretrained weights (gitignored)
docs/Final_Report.tex              — IEEEtran source (8 sections + §VIII–§XIII v4 appendices)
docs/_autogen/                     — auto-generated LaTeX tables (B2/C2/multi-seed/calibration/throughput)
docs/phase_b2.md / phase_c2.md     — Phase B2 / C2 rationale (US-037)
papers/                            — TransNeXt, DenseNet, TResNet, EfficientNetV2, NASNet
agents/                            — MASTER + 11 specialist sub-agent specs
```

## Weight Privacy (mandatory for AI sessions)

Trained checkpoints (`*.ckpt`, `*.pt`, `*.pth`), the Optuna SQLite store, and dashboard thumbnail caches are **local-only**. They are excluded from Git via `.gitignore` and from AI assistant context via `.claudeignore`. Pretrained TransNeXt weights under `artifacts/weights/` are auto-downloaded on demand and also gitignored.

**Future Claude / Codex sessions must:**
- Never `Read`, `Bash cat`, or otherwise inspect `*.ckpt`, `*.pt`, `*.pth`, or files under `artifacts/weights/` or `runs/**/*.ckpt`.
- Analyze runs only via `metrics.json`, `metrics.csv`, `Final_Exp.md`, or `artifacts/Final_Exp.html`.
- If a binary weight artifact appears in context, treat it as out-of-scope and decline to introspect it.

## Coding Rules

- Minimal changes, no rewrites of working code.
- Keep the pipeline intact; log everything.
- Same protocol / dataset / degradation across all models (fair-comparison invariant — DATA_ARCHITECT enforces).
- Reproducibility first — every config knob must round-trip through `metrics.json`.

## Multi-Agent Protocol

The repo runs a hierarchical multi-agent structure (see [`AGENTS.md`](AGENTS.md) and [`agents/`](agents/)). **Plan-Mode is mandatory** before any code change — submit a plan to MASTER for approval first. Each sub-agent has a scoped file-system; respect those scopes.

## Deadline

- Final submission: 26/07/2026
