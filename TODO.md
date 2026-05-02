# TODO — Final Research Campaign

> **Pivot — 2026-05-02:** moved from legacy 36-experiment plan (33/36 frozen in `runs/systematic/`) to the **186-cell Final Research Phase**.
> Master tracker: [`Final_Exp.md`](Final_Exp.md). Plan body: see CLAUDE.md and the original prompt thread.

## Deadlines

- Poster & abstract: **31/05/2026**
- Final presentation: **21/06/2026**
- Final submission: **26/07/2026**

---

## Sprint 1 — Foundations (in progress, started 2026-05-02)

### ✅ Done

- [x] `src/data/degradation_levels.py` — 5-level table (`DEGRADATION_LEVELS`, `LEVEL_NAMES`, `AXIS_KEYS`, `level_params`)
- [x] `src/data/degrade.py` — saturation axis (deterministic lerp), `'none'` clean-baseline early-return, `'saturation'` isolation mode, `degrade_config_for(level, axis, out_size)` helper
- [x] `Final_Exp.md` — 186-cell master tracker, all rows seeded `Pending`
- [x] Smoke tests: Phase A pass-through, saturation=0 produces exact grayscale (max channel diff = 0.000000)

### 🟡 Up next (Sprint 1 remaining)

- [ ] Plumb `saturation` through `DataConfig` → `DegradeConfig` in [`src/data/datasets.py`](src/data/datasets.py)
- [ ] Add `torchmetrics` to `requirements.txt` (PSNR/SSIM)
- [ ] PSNR/SSIM in [`src/lightning/datamodule.py`](src/lightning/datamodule.py) `setup()` — 256 fixed val indices, write `psnr_mean/std`, `ssim_mean/std` to `metrics.json`
- [ ] Extend [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py) — assert PSNR/SSIM byte-identical across two `setup()` calls
- [ ] Add `'saturation'` to the test's degradation-type matrix

### Sprint 2 — Training Engine

- [ ] FP16 mixed precision in [`src/lightning/train.py`](src/lightning/train.py) Trainer (`precision="16-mixed"`, `accelerator="auto"`)
- [ ] CLI flag `--precision` (debug override)
- [ ] Raise defaults: `max_epochs=60`, `EarlyStopping(patience=10, min_delta=1e-4, monitor="val_acc", mode="max")`
- [ ] `ModelCheckpoint(dirpath=runs/final/<tag>/, save_top_k=1, save_last=True)` — confirm no checkpoint escapes `runs/final/`
- [ ] Assertion that refuses `--plan final --mode pilot`

### Sprint 3 — TransNeXt Size Selector

- [ ] [`src/models/transnext_wrapper.py`](src/models/transnext_wrapper.py) — add `size: Literal["micro","small","base"]` arg
- [ ] Auto-download from official TransNeXt GitHub release with SHA256 verification, cache in `artifacts/weights/`
- [ ] Route through wrapper from [`src/lightning/module.py`](src/lightning/module.py) when `model_name.startswith("transnext")`
- [ ] CLI flag `--transnext_size {micro,small,base}` (default `small`) in `train.py`, `run_systematic.py`, `run_all_phases.py`

### Sprint 4 — Optuna Pre-Tuning

- [ ] Narrow [`src/tune_hyperparams.py`](src/tune_hyperparams.py) search to paper-anchored ranges (lr, weight_decay, label_smoothing, drop_path_rate, batch_size, warmup_epochs, pos_bias_interp)
- [ ] CLI surface: `--model {resnet50,densenet121,transnext}`, `--dataset {cifar10,mnist}`, `--transnext_size`, `--n-trials`, `--level`
- [ ] Write `artifacts/best_hparams/{model}_{dataset}.json` on completion
- [ ] New `tune_all.py` — loops over 6 `(model, dataset)` pairs, fails loudly on any incomplete sweep
- [ ] Wire `run_all_phases.py --tune-first` to load best_hparams or fall back to paper centers

### Sprint 5 — Phase Runner

- [ ] `run_all_phases.py --plan final` — replace 36-cell logic with the three product loops (Phase A 6, Phase B 30, Phase C 150)
- [ ] `--skip-existing` checks `runs/final/<tag>/metrics.json`
- [ ] Tag scheme: `final_clean_{m}_{d}` / `final_B_L{l}_{m}_{d}` / `final_C_L{l}_{ax}_{m}_{d}`
- [ ] Replace inline degradation dicts with `from src.data.degradation_levels import DEGRADATION_LEVELS`

### Sprint 6 — Dashboard & Tracker Updates

- [ ] `src/tools/generate_final_dashboard.py` → `artifacts/Final_Exp.html`
  - [ ] Pre-render all 186 placeholder cells with side-by-side original + degraded thumbnails (cached in `artifacts/.cell_samples/`)
  - [ ] Phase A 3×2 grid, Phase B 5×3 with dataset toggle, Phase C 5-axis × 5-level sub-grid per (model, dataset)
  - [ ] Phase end analytics: bars for A, scatter `Val Acc vs SSIM/PSNR` for B, robustness curves per axis for C
  - [ ] Click cell → modal with learning curves
  - [ ] `<meta http-equiv="refresh" content="30">` for 30-s polling
- [ ] `scripts/update_final_exp.py` — scan `runs/final/`, rewrite `Final_Exp.md` in place
- [ ] `src/lightning/callbacks.py` — `DashboardRefreshCallback` invokes both generators on `on_train_end`

### Sprint 7 — Privacy & Autonomy Hygiene

- [ ] `.gitignore` — add `runs/final/`, `artifacts/weights/*.pth`, `artifacts/optuna_thz.db`, `artifacts/best_hparams/*.json`, `artifacts/.cell_samples/`
- [ ] `.claudeignore` — add `runs/**/*.pt|.ckpt|.pth`, `artifacts/weights/`, `artifacts/optuna_thz.db`, `artifacts/.cell_samples/`
- [ ] CLAUDE.md — Weight Privacy addendum (no `Read`/`Bash cat` on weight binaries)
- [ ] `setup.sh` — env bootstrap (venv, torch+CUDA, deps, TransNeXt weights, verify_env)
- [ ] `scripts/verify_env.py` — torch/CUDA/cuDNN print + FP16 smoke test
- [ ] Pre-commit guard — fail if any `.pt`/`.ckpt`/`.pth` is staged

---

## 186-cell Campaign Status

> Updated by `scripts/update_final_exp.py` on every `on_train_end`. Live counts: see [`Final_Exp.md`](Final_Exp.md).

| Phase | Description | Count | Status |
|---|---|---|---|
| **A** | Clean baselines (3 models × 2 datasets) | 6 | 🔲 0/6 |
| **B** | Combined degradation (× 5 levels) | 30 | 🔲 0/30 |
| **C** | Single-axis isolation (× 5 axes × 5 levels) | 150 | 🔲 0/150 |
| **Total** | | **186** | 🔲 0/186 |

### Optuna Pre-tuning (`(model, dataset)` pairs)

| # | Model | Dataset | Status |
|---|---|---|---|
| 1 | resnet50 | cifar10 | 🔲 |
| 2 | resnet50 | mnist | 🔲 |
| 3 | densenet121 | cifar10 | 🔲 |
| 4 | densenet121 | mnist | 🔲 |
| 5 | transnext (small) | cifar10 | 🔲 |
| 6 | transnext (small) | mnist | 🔲 |

---

## Stage 5 — Poster & Abstract (31/05) 🔲

- [ ] Select best figures (robustness curves, isolation heatmap, sample-grid)
- [ ] Write abstract
- [ ] Design poster layout

## Stage 6 — Final Report & Presentation (21/06) 🔲

- [ ] Write full report
- [ ] Prepare presentation slides
- [ ] **FINAL PRESENTATION (21/06)**

## Stage 7 — Final Submission (26/07) 🔲

- [ ] Clean repository (verify weight privacy via `.gitignore` / `.claudeignore`)
- [ ] Re-run determinism + autonomy tests from a fresh shell
- [ ] Submit

---

## Frozen Legacy Reference (33/36)

The previous 36-experiment plan (3 levels × 3 models × 2 datasets + isolation + clean) completed 33/36 runs. Results frozen in `runs/systematic/` and `runs/official/`. Final 3 (Phase B L3 MNIST × 3 models) were superseded by the pivot to L1–L5 and never run. **Do not retroactively run them** — they would not be comparable to the new 5-level grid.

| Phase | Status | Notes |
|---|---|---|
| Legacy Phase A (CIFAR-10, 3 levels × 3 models) | 9/9 ✅ | DenseNet 81.9% / 80.4% / 63.4% (L1/L2/L3) |
| Legacy Phase B (MNIST, 3 levels × 3 models) | 9/9 ✅ | ResNet 99.1%/99.1%/92.5%, DenseNet 98.7%/99.0%/93.0%, TransNeXt 92.5%/92.8%/72.7% |
| Legacy Phase C (CIFAR-10 isolation, 4 axes × 3 models) | 12/12 ✅ | TransNeXt collapses on blur/noise/S&P (0%) |
| Legacy Phase D (clean baselines) | 6/6 ✅ | CIFAR-10: ResNet 94.2%, DenseNet 93.2%, TransNeXt 91.7% |
