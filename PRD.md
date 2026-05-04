# PRD: THz Final Experimental Phase — 186-Run Campaign Infrastructure

## 1. Introduction

This PRD covers the infrastructure work required to execute the Final Experimental Phase of the Low-Resolution Image Classification project (THz-like degradation). The 186-cell matrix (Phase A clean baselines + Phase B combined degradation + Phase C single-axis isolation) must run autonomously across three architectures (ResNet50, DenseNet121, TransNeXt-Base) and two datasets (CIFAR-10, MNIST), with paper-anchored hyperparameter tuning, deterministic per-sample degradation, image-quality measurement (PSNR/SSIM), and a live W&B-backed dashboard.

The training engine, deterministic seeding, and 5-level degradation table already exist (see [`src/lightning/`](src/lightning/), [`src/data/degradation_levels.py`](src/data/degradation_levels.py), [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py)). What remains is wiring saturation through every layer, building the orchestration + tuning + dashboard layers, and locking weight privacy.

## 2. Goals

- Plumb the saturation axis through `datasets.py` and the degradation pipeline so all 186 cells produce byte-identical val pixels for any given (cell, image-index).
- Produce a standalone offline PSNR/SSIM measurement script that emits one `image_quality.json` per cell over a fixed 256-sample subset.
- Carry the matrix size at exactly **186 runs** (Phase A=6, Phase B=30, Phase C=150). No L1 deduplication.
- Tune hyperparameters once per (model, dataset) at L3 Moderate using Optuna anchored to per-model priors at `artifacts/priors/{model}.json`, then freeze winners for the campaign.
- Promote TransNeXt to **size `base` with full fine-tuning** (no frozen backbone) for the 186-cell campaign.
- Generate `Final_Exp.html` — an interactive dashboard with one tile per cell, embedding W&B iframe panels for live learning curves and pre-rendered Original-vs-Degraded image pairs.
- Make `README.md` a standalone human runbook (clone → run → inspect results) and harden weight privacy in `.gitignore` and `.claudeignore`.

## 3. User Stories

### US-001: Plumb saturation axis through `datasets.py`
**Description:** As a researcher, I want `THzLikeCIFAR10` and `THzLikeMNIST` to consume a saturation parameter from `DegradeConfig` so every cell can drive saturation independently of other axes.

**Acceptance Criteria:**
- [ ] `THzLikeCIFAR10.__getitem__` and `THzLikeMNIST.__getitem__` apply saturation lerp via `degrade_image(...)` before noise/S&P
- [ ] `DegradeConfig.saturation` defaults to `1.0` (no-op) when not specified
- [ ] [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py) still passes (MSE = 0 across runs)
- [ ] New test asserts `saturation=0.0` produces a 3-channel grayscale image (all 3 channels equal per pixel)
- [ ] New test asserts `saturation=1.0` matches the legacy non-saturation pipeline byte-for-byte
- [ ] Typecheck passes

### US-002: Offline PSNR/SSIM measurement script
**Description:** As a researcher, I want a standalone script that computes PSNR and SSIM between original and degraded images over a fixed 256-sample subset for any single cell, so I can characterize image quality independently of training.

**Acceptance Criteria:**
- [ ] New file `src/tools/measure_image_quality.py` with CLI flags `--cell-tag`, `--dataset`, `--levels-spec`, `--n-samples 256`, `--out`
- [ ] Uses the same `degrade_config_for(...)` path as training so values match training pixels exactly
- [ ] Writes `image_quality.json` containing `{psnr_mean, psnr_std, ssim_mean, ssim_std, n_samples, sample_indices, degrade_config}`
- [ ] Sample indices are deterministic (seeded `0..255`) and identical across models for the same cell
- [ ] Unit test in `src/tests/test_image_quality.py` runs the script on 8 samples for one Phase B L3 cell and asserts non-negative PSNR / SSIM in `[0,1]`
- [ ] Typecheck passes

### US-003: Per-model hparam priors JSON files
**Description:** As a researcher, I want hand-curated `artifacts/priors/{model}.json` files that specify Optuna search ranges anchored to paper citations, so tuning never explores blindly.

**Acceptance Criteria:**
- [ ] Three files created: `artifacts/priors/resnet50.json`, `artifacts/priors/densenet121.json`, `artifacts/priors/transnext_base.json`
- [ ] Each file declares: `head_lr`, `backbone_lr`, `weight_decay`, `label_smoothing`, `warmup_epochs`, each as `{distribution, low, high, citation}` where `distribution ∈ {loguniform, uniform, categorical}`
- [ ] Ranges are bounded to paper-derived priors ± 1 decade max (per CLAUDE.md hparam table)
- [ ] Each entry includes a `citation` field naming the source paper (e.g., `"papers/transnext.pdf §A.3"`)
- [ ] JSON schema validation file at `artifacts/priors/_schema.json` (JSON Schema Draft-07)
- [ ] Typecheck passes

### US-004: Priors loader + validator in `tune_all.py`
**Description:** As a developer, I want `tune_all.py` to load and validate priors JSON against the schema at startup, so misconfigured priors fail fast.

**Acceptance Criteria:**
- [ ] New file `tune_all.py` at repo root with function `load_priors(model_name) -> dict`
- [ ] Loader validates against `artifacts/priors/_schema.json`, raises `ValueError` with the offending field on failure
- [ ] CLI flag `--validate-only` exits 0 if all 3 priors files validate, exits 1 otherwise
- [ ] Unit test asserts a malformed priors file (negative `low`, or `low > high`) is rejected
- [ ] Typecheck passes

### US-005: Optuna study runner in `tune_all.py`
**Description:** As a researcher, I want `tune_all.py` to run an Optuna study per (model, dataset) pair on L3 Moderate using the priors, persisting trials in `artifacts/optuna_thz.db` and writing winners to `artifacts/best_hparams/{model}_{dataset}.json`.

**Acceptance Criteria:**
- [ ] CLI: `python tune_all.py --n-trials 20 [--model M] [--dataset D]`; default tunes all 6 pairs
- [ ] Uses `optuna.create_study(direction="maximize", storage="sqlite:///artifacts/optuna_thz.db", study_name=f"{model}_{dataset}", load_if_exists=True)`
- [ ] Each trial trains on Phase B L3 config (all axes at L3 Moderate) and reports `val_acc`
- [ ] Winner JSON includes `{best_value, best_params, n_trials_completed, study_name, priors_file_hash}`
- [ ] On crash, rerunning resumes from the SQLite store
- [ ] Unit test runs `--n-trials 1` against a stub dataset and verifies the winner JSON is produced
- [ ] Typecheck passes

### US-006: TransNeXt size=`base` + full fine-tuning mode
**Description:** As a researcher, I want the TransNeXt wrapper to support `size=base` and a full fine-tuning mode (no frozen backbone) so the 186-cell campaign can use the heavy variant.

**Acceptance Criteria:**
- [ ] `--transnext_size {micro,small,base}` flag wired through `run_systematic.py` and the model factory
- [ ] `--transnext_mode {lp,ft}` flag selects linear-probe vs full fine-tuning; default `ft` for the final campaign
- [ ] In `ft` mode, all backbone parameters have `requires_grad=True` and use the backbone LR from `best_hparams`
- [ ] In `lp` mode, behavior matches existing frozen-backbone implementation (regression test)
- [ ] Pretrained weights for `transnext_base` auto-download to `artifacts/weights/` (gitignored)
- [ ] Smoke test: 1-epoch run on CIFAR-10 with `--transnext_size base --transnext_mode ft` completes without OOM at batch 32 in `16-mixed`
- [ ] Typecheck passes

### US-007: 186-cell matrix generator
**Description:** As a developer, I want a single function that emits the full 186-cell matrix as a list of cell specs (tag, model, dataset, phase, level, axis, degrade_config), so all downstream tools share one source of truth.

**Acceptance Criteria:**
- [ ] New module `src/experiments/matrix.py` with `build_final_matrix() -> list[CellSpec]`
- [ ] Returns exactly **186** entries (no L1 deduplication; Phase C L1 cells run as redundancy)
- [ ] Counts per phase: A=6, B=30, C=150
- [ ] Tag scheme matches CLAUDE.md: `final_clean_{m}_{d}` / `final_B_L{l}_{m}_{d}` / `final_C_L{l}_{ax}_{m}_{d}`
- [ ] Unit test asserts the exact set of 186 tags is unique and matches `len(set(tags)) == 186`
- [ ] Unit test asserts each Phase C L1 cell's `degrade_config` equals the Phase B L1 config for the same (model, dataset)
- [ ] Typecheck passes

### US-008: `run_systematic.py` consumes matrix + best_hparams
**Description:** As a developer, I want `run_systematic.py` to accept a single CellSpec and load the corresponding `best_hparams/{model}_{dataset}.json` automatically, so each cell trains with the tuned hyperparameters.

**Acceptance Criteria:**
- [ ] New flag `--cell-tag <TAG>` resolves to a `CellSpec` from the matrix
- [ ] Hparams loaded from `artifacts/best_hparams/{model}_{dataset}.json`; missing file raises a clear error with remediation hint (`run tune_all.py first`)
- [ ] All hparam values logged to `metrics.json` under a `hparams` key for round-trip reproducibility
- [ ] Existing CLI flags (legacy direct degradation flags) remain functional for ad-hoc runs
- [ ] Smoke test: `--cell-tag final_clean_resnet50_cifar10 --mode pilot` completes and writes `runs/final/<tag>/metrics.json`
- [ ] Typecheck passes

### US-009: `run_all_phases.py --plan final` orchestration
**Description:** As a researcher, I want `run_all_phases.py --plan final --phase {A,B,C,all}` to iterate the matrix and dispatch each cell to `run_systematic.py`, with `--skip-existing` and `--tune-first` options, so the 186 runs execute autonomously.

**Acceptance Criteria:**
- [ ] `--plan final` accepted; `--plan final --mode pilot` rejected with assertion (per CLAUDE.md)
- [ ] `--phase A|B|C|all` filters the matrix accordingly (6, 30, 150, or 186 cells)
- [ ] `--skip-existing` skips cells where `runs/final/<tag>/metrics.json` exists and contains `final_val_acc`
- [ ] `--tune-first` runs `tune_all.py` if any required `best_hparams/{model}_{dataset}.json` is missing
- [ ] After every cell, calls `scripts/update_final_exp.py` to refresh `Final_Exp.md`
- [ ] On cell failure, logs the error and continues to the next cell (no crash-stop)
- [ ] Unit test asserts `--phase A` produces 6 dispatch calls when no skips apply (mocked)
- [ ] Typecheck passes

### US-010: Pre-rendered Original-vs-Degraded image pairs
**Description:** As a researcher, I want a script that pre-renders one Original-vs-Degraded PNG pair per cell (using a fixed sample index) into `artifacts/dashboard_thumbs/<tag>.png`, so the dashboard loads instantly without on-the-fly rendering.

**Acceptance Criteria:**
- [ ] New script `src/tools/render_cell_thumbs.py` iterates the 186 cells from `build_final_matrix()`
- [ ] Each PNG is a side-by-side composite (Original | Degraded) at 224×448 px
- [ ] Sample index is fixed per dataset (e.g., index 0 of the val split) so the same image is shown across models
- [ ] Output directory `artifacts/dashboard_thumbs/` is gitignored and claudeignored
- [ ] Idempotent: rerunning skips existing thumbs unless `--force` is passed
- [ ] Typecheck passes

### US-011: `Final_Exp.html` dashboard with W&B iframe embeds
**Description:** As a researcher, I want `artifacts/Final_Exp.html` to display all 186 cells as a grid of tiles, each showing the pre-rendered image pair plus a W&B iframe panel for that run's live learning curves.

**Acceptance Criteria:**
- [ ] New script `src/tools/build_final_dashboard.py` writes `artifacts/Final_Exp.html`
- [ ] HTML grid contains exactly 186 tiles, organized into 3 collapsible sections (Phase A, B, C)
- [ ] Each tile renders: cell tag, model, dataset, level, axis (if Phase C), thumbnail PNG, status badge (Pending/Running/Complete/Failed), final `val_acc` if complete, and an iframe `<iframe src="https://wandb.ai/{entity}/{project}/runs/{run_id}?panelId=...">` for the run's training curves
- [ ] W&B `entity`, `project`, and per-cell `run_id` are read from `runs/final/<tag>/metrics.json` (`wandb_run_id` field) — fallback to placeholder iframe if missing
- [ ] Tiles without W&B run_id show a "Pending" placeholder instead of the iframe
- [ ] Page loads without console errors when opened directly via `file://`
- [ ] Typecheck passes
- [ ] Verify changes work in browser

### US-012: Capture W&B run IDs in `metrics.json`
**Description:** As a developer, I want every Lightning run to persist its W&B `run_id`, `entity`, and `project` into `metrics.json`, so the dashboard can construct iframe URLs deterministically.

**Acceptance Criteria:**
- [ ] [`src/lightning/`](src/lightning/) writes `wandb_run_id`, `wandb_entity`, `wandb_project` to `metrics.json` at run end
- [ ] When W&B is offline / disabled, fields are written as `null` rather than omitted
- [ ] Existing `metrics.json` schema additions are documented in [`Final_Exp.md`](Final_Exp.md) header
- [ ] Unit test runs a 1-epoch CPU job in offline mode and asserts the three fields exist (may be `null`)
- [ ] Typecheck passes

### US-013: Weight privacy hardening (`.gitignore` + `.claudeignore`)
**Description:** As the project owner, I want `.gitignore` and `.claudeignore` to strictly exclude all model checkpoints, pretrained weights, the Optuna SQLite store, and dashboard thumbnail caches, so binary weight artifacts never leave the local machine.

**Acceptance Criteria:**
- [ ] `.gitignore` excludes: `*.ckpt`, `*.pt`, `*.pth`, `artifacts/weights/`, `artifacts/optuna_thz.db`, `artifacts/dashboard_thumbs/`, `runs/final/**/*.ckpt`
- [ ] `.claudeignore` mirrors the same exclusions plus `runs/systematic/**/*.ckpt`
- [ ] `git check-ignore` returns true for at least one sample path in each excluded category (smoke tested in CI script `scripts/check_ignores.sh`)
- [ ] No tracked checkpoint files appear in `git ls-files | grep -E '\.(ckpt|pt|pth)$'`
- [ ] Typecheck passes

### US-014: `README.md` standalone runbook
**Description:** As a new contributor, I want `README.md` to be a complete top-to-bottom runbook so I can clone the repo, set up the environment, run the 186-cell campaign, and read results without consulting any other document.

**Acceptance Criteria:**
- [ ] Sections: Overview, Prerequisites, Environment Setup, Dataset Setup, Run Tuning, Run Campaign, Inspect Results, Privacy Notes, Troubleshooting
- [ ] Every command in the runbook is copy-pasteable PowerShell + Bash
- [ ] Documents the exact CLI invocations: `python tune_all.py --n-trials 20`, `python run_all_phases.py --plan final --phase all --skip-existing --tune-first`
- [ ] Links to [`Final_Exp.md`](Final_Exp.md), [`CLAUDE.md`](CLAUDE.md), and [`AGENTS.md`](AGENTS.md) but does not require the reader to open them to complete a run
- [ ] Includes a one-line policy statement on weight privacy referencing `.claudeignore`
- [ ] Typecheck passes
- [ ] Verify changes render correctly on GitHub (preview)

## 4. Non-Goals

- **No new degradation axes** beyond the existing 5 (resolution, blur, noise, salt_pepper, saturation). Color hue, JPEG, motion blur, etc. are out of scope.
- **No automatic PDF parsing** of `/papers/` for hparam priors — priors are hand-curated (per 5D).
- **No L1 deduplication** in Phase C — all 186 cells run, even when redundant (per 2B).
- **No live training-time PSNR/SSIM** — image quality is computed offline only (per 3D).
- **No local web server** for the dashboard — static HTML + W&B iframes only (per 4C).
- **No new model architectures** beyond ResNet50, DenseNet121, TransNeXt-Base.
- **No retraining of the legacy 33-run `runs/systematic/` results** — they are frozen.
- **No editing of pretrained weight files** or `metrics.json` files outside of the run that produced them.
- **No `--mode pilot` runs against `--plan final`** — rejected with assertion (per CLAUDE.md).

## 5. Technical Notes

### Reuse
- Degradation pipeline already lives in [`src/data/degrade.py`](src/data/degrade.py) and [`src/data/degradation_levels.py`](src/data/degradation_levels.py) — extend, don't replace.
- Lightning engine in [`src/lightning/`](src/lightning/) (`THzClassifier`, `THzDataModule`) is the canonical training entry — every cell goes through it.
- Determinism gate in [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py) is the contract — every change must keep MSE = 0.
- `pl.seed_everything(42, workers=True)` and `seed = idx + SEED_OFFSET_VAL` per-sample are non-negotiable.

### Constraints
- Runtime budget: tuning ≈ 20 GPU-hours, campaign ≈ 60 GPU-hours. TransNeXt-Base + full FT will be the dominant cost.
- Hparam priors must stay within ±1 decade of paper-derived values (DATA_ARCHITECT enforces).
- All binary weight artifacts (`*.ckpt`, `*.pt`, `*.pth`, `artifacts/weights/**`) are local-only — never commit, never inspect via `Read`/`Bash cat` in AI sessions (per CLAUDE.md "Weight Privacy" section).
- Same protocol / dataset / degradation across all models (fair-comparison invariant).

### File-System Touchpoints
| Component | Files |
|---|---|
| Saturation plumbing | [`src/data/datasets.py`](src/data/datasets.py), [`src/data/degrade.py`](src/data/degrade.py) |
| Image quality | `src/tools/measure_image_quality.py` (new), `src/tests/test_image_quality.py` (new) |
| Priors | `artifacts/priors/{resnet50,densenet121,transnext_base}.json` (new), `artifacts/priors/_schema.json` (new) |
| Tuning | `tune_all.py` (new), `src/tune_hyperparams.py` (existing, refactor) |
| TransNeXt | [`src/models/`](src/models/) — wrapper additions for `base` size + `ft` mode |
| Matrix | `src/experiments/matrix.py` (new) |
| Orchestration | `run_systematic.py`, `run_all_phases.py` (existing, extend) |
| Dashboard | `src/tools/render_cell_thumbs.py` (new), `src/tools/build_final_dashboard.py` (new), `artifacts/Final_Exp.html` (output) |
| Privacy | `.gitignore`, `.claudeignore`, `scripts/check_ignores.sh` (new) |
| Docs | `README.md` |
