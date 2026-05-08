# PRD: Phase B Execution — CNN Sweep (20 cells)

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Phase Mapping:** Phase B (Combined Degradation, 30 cells) — CNN scope only: 4 (model, dataset) pairs × 5 levels = **20 executable cells**. The 10 TransNeXt rows remain quarantined per US-014.
**Continues:** [PHASE_B_VISUAL_CORE.md](PHASE_B_VISUAL_CORE.md) US-014 → US-019 (infrastructure complete). User-story numbering resumes at **US-020**.
**Supersedes:** the US-020 GPU-bootstrap section of [US_020_GPU_BOOTSTRAP.md](US_020_GPU_BOOTSTRAP.md). The Phase C / results-synthesis / figures sections of that draft remain valid as a future-PRD roadmap and are out of scope here.
**Operator runbook:** [docs/runbooks/PHASE_B_EXECUTION.md](../runbooks/PHASE_B_EXECUTION.md) — this PRD turns the runbook stages into atomic, verifiable user stories.
**Date:** 2026-05-07

---

## Handoff Status — 2026-05-07 (paused for stronger GPU)

The 20-cell Phase B campaign is partially executed on the local RTX 4050 Laptop (6 GiB VRAM); remaining work is queued for a stronger GPU box. **All Stage 1 fast-tune results, the SQLite study DB, and infrastructure code are durable on disk and resumable.**

| Story | Status | Notes |
|---|---|---|
| US-020 | ✅ DONE | `.venv-gpu` (Py 3.12.10) bootstrapped on local box. `requirements.lock.txt` written. Smoke test PASS at `--min-vram-gib 5.9` (4 MiB tolerance). On the new box, re-run `python scripts/setup_gpu_env.py` from a fresh checkout — `requirements.lock.txt` lets you `pip install -r` for an exact replay. |
| US-021 | ✅ DONE (with documented exception) | All 5 substantive checks pass. The `test_quarantine_transnext` 7/9 PASS with 2 false positives from VSCode Pylance scanning project `*transnext*.py` files — runs/training are unaffected. See "Known issues" below. |
| **Methodology lock — Option C** | ✅ LOCKED 2026-05-07 | Hybrid: fast Optuna ranking (5ep / 2k train / 1k val per `run_studies` defaults) + top-3 re-validation at production protocol (60ep / 10k / 5k / patience=10). Replaces the originally-implied "convergence-quality Optuna" path because actual per-trial wall-clock would have been ~100 GPU-h vs. ~18 GPU-h for the hybrid. See Section 8. |
| US-022 (resnet50_cifar10 fast tune) | ✅ DONE | 26 trials in SQLite. Best fast val_acc=0.444 (trial #1). |
| US-023 (densenet121_cifar10 fast tune) | ✅ DONE | 26 trials in SQLite. Best fast val_acc=0.535 (trial #15). |
| US-024 (resnet50_mnist fast tune) | ✅ DONE | 20 trials in SQLite. Best fast val_acc=0.949 (trial #0). |
| US-025 (densenet121_mnist fast tune) | ✅ DONE | 20 trials in SQLite. Best fast val_acc=0.956 (trial #0). |
| **Stage 1.5 — top-3 validation** (60ep / 10k / 5k / patience=10) | ⏳ PENDING (stronger GPU) | First attempt failed with `ModuleNotFoundError: No module named 'src'`. Root cause + fix: [scripts/validate_top3.py](../../scripts/validate_top3.py) needed `sys.path.insert(0, REPO_ROOT)` because `python scripts/X.py` only puts `scripts/` on sys.path. Fix is committed; re-launch picks up cleanly. |
| US-026 (resnet50_cifar10 sweep × L1..L5) | ⏳ PENDING | Gated on Stage 1.5 winner JSON. |
| US-027 (densenet121_cifar10 sweep × L1..L5) | ⏳ PENDING | Same. |
| US-028 (resnet50_mnist sweep × L1..L5) | ⏳ PENDING | Same. |
| US-029 (densenet121_mnist sweep × L1..L5) | ⏳ PENDING | Same. |
| US-030 (verify + SYNCHRONIZER push) | ⏳ PENDING | Single per-phase commit at end. |

**Resume on the new box:**

```powershell
# 1. Bootstrap (10-20 min, ~3-5 GB)
python scripts/setup_gpu_env.py
# 2. Pre-flight (30 sec)
.venv-gpu\Scripts\python.exe -m src.tests.test_priors
.venv-gpu\Scripts\python.exe -m src.tests.test_build_final_exp_json
.venv-gpu\Scripts\python.exe tune_all.py --validate-only
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
# 3. Resume the chain (Stage 1 will skip-existing on all 4; Stage 2 will run validation)
$proc = Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-WindowStyle","Hidden","-File","scripts\run_tune_chain.ps1" -PassThru
"PID=$($proc.Id)"  # save for monitoring
# 4. After Stage 2 winners land in artifacts/best_hparams/{m}_{d}.json (with `validated_at_full_convergence: true`):
.venv-gpu\Scripts\python.exe run_all_phases.py --plan final --phase B --skip-existing
# 5. End-of-phase
.venv-gpu\Scripts\python.exe scripts\refresh_trackers.py
.venv-gpu\Scripts\python.exe scripts\sync_trackers_git.py
```

**Durable state on disk (all gitignored — copy to new box manually if needed):**
- `artifacts/optuna_thz.db` — Optuna SQLite, all 92 fast-tune trials across 4 studies. Resumable.
- `artifacts/best_hparams/*.json` — current 4 fast-tune winners. Will be overwritten by Stage 1.5 validation.
- `artifacts/validation/` — empty (Stage 1.5 never produced cache files; first attempt failed on `src` import before any training).
- `runs/final/final_clean_*` — 4 Phase A CNN clean baselines (untouched).

**Known issues to be aware of on the new box:**
- `setup_gpu_env.py:make_venv` only iterates `RECOMMENDED_PY_MINORS = (11, 10)` — won't pick 3.12 even though `MAX_SUPPORTED_PY_MINOR = 12`. If the new box has 3.12 (no 3.11/3.10), bootstrap manually: `py -3.12 -m venv .venv-gpu`, then `.venv-gpu\Scripts\python.exe scripts\setup_gpu_env.py --skip-install` (audit + smoke test only).
- Default `--min-vram-gib 6.0` strict-fails on a 6 GiB card (actual `5.997 GiB < 6.0`). On any GPU strictly < 6 GiB, append `--min-vram-gib 5.9` and accept the OOM-recovery path documented in the runbook.
- `scripts/quarantine_transnext.py:find_running_transnext_processes` does a substring match on `"transnext"` against any python process's full cmdline. VSCode Pylance and similar LSP daemons scan project files including `*transnext*.py`, producing false positives in `test_quarantine_transnext`. Substantive check — `nvidia-smi --query-compute-apps` showing zero `python.exe` — is what matters.
- PowerShell 5.1 `*>>` redirect uses UTF-16 LE BOM by default and wraps native-exe stderr in ErrorRecord (`python.exe :` prefix). [scripts/run_tune_chain.ps1](../../scripts/run_tune_chain.ps1) uses `cmd /c "... >> log 2>&1"` + `PYTHONIOENCODING=utf-8` to bypass — log is pure UTF-8, grep-clean.
- `scripts/run_tune_chain.ps1` skips a Stage 1 pair if its SQLite study has ≥ 20 completed trials, so re-running the chain on the new box won't double the trial count.

---

## 1. Context & Phase Mapping

Phase A (CNN) is complete: 4/4 cells in `runs/final/final_clean_{resnet50,densenet121}_{cifar10,mnist}`. The 186-cell tracker reads `Phase A: 4/6, Phase B: 0/30, Phase C: 0/150, Deferred: 62/186, Total: 4/186` ([Final_Exp.md](../../Final_Exp.md):8-12).

The Phase B infrastructure (US-014 → US-019) is wired and validated by the torch-free pytest suite:

- **Quarantine** — TransNeXt rows reset to `Pending` and skipped by both `tune_all.py` and `run_all_phases.py` (US-014).
- **Priors** — paper-anchored, decade-bounded priors live in [`artifacts/priors/`](../../artifacts/priors/) for `resnet50`, `densenet121`, `transnext_base`; SHA-256 round-trips into `metrics.json.hparams_source.priors_file_hash` (US-015).
- **Tuning + execution wiring** — [`tune_all.py`](../../tune_all.py) runs per-pair Optuna studies on L3; [`run_systematic.run_cell`](../../run_systematic.py) auto-writes `image_quality.json` (256-sample PSNR/SSIM); the SIGINT handler in [`run_all_phases.py`](../../run_all_phases.py) drops an `INTERRUPTED` sentinel so a Ctrl-C'd cell renders as `Failed` (US-016).
- **Visual Core + lazy curves** — every cell has a deterministic Original|Degraded thumbnail and an on-click Plotly drawer (US-017, US-018).
- **Incremental refresh + 10-min polling** — `refresh_trackers.py --cell <tag>` patches a single row in `Final_Exp.json`; the dashboard auto-refreshes every 600 s with a manual "🔄 Refresh now" pill (US-019).

What's missing is the actual GPU run. The current dev shell is Python 3.14.2 (no PyTorch wheel), so Stage -1 (GPU env bootstrap) is the gating prerequisite. After that, Stage 1 (Optuna) and Stage 2 (sweep) execute via the runbook commands.

**Cells in scope (20):** rows tagged `final_B_L{1..5}_{resnet50|densenet121}_{cifar10|mnist}` in [`artifacts/Final_Exp.json`](../../artifacts/Final_Exp.json), all currently `status: "Pending"`.

**Cells explicitly out of scope (10):** `final_B_L{1..5}_transnext_base_{cifar10|mnist}` — quarantined; remain `Pending` for the 186 denominator.

---

## 2. Core Goals & Non-Goals

### Goals
- **G1.** Bring a fresh GPU box from `git clone` to `torch.cuda.is_available() == True` with one command, and freeze the resolved dependency set as `requirements.lock.txt`.
- **G2.** Execute Optuna pre-tune at L3 Moderate for the 4 CNN pairs and freeze the winners to `artifacts/best_hparams/{model}_{dataset}.json` with `priors_file_hash` round-tripped.
- **G3.** Run all 20 Phase B CNN cells in 4 sub-batches of 5 (one per pair, L1 → L5), each with `metrics.json` (convergence) + `image_quality.json` (PSNR/SSIM) + per-cell tracker refresh + visual-core thumb.
- **G4.** End-of-phase verification: `Final_Exp.md` shows `Phase B: 20/30 (TransNeXt deferred)`, dashboard renders all 20 cells with curves drawer + thumbs, and tracker artifacts are committed and pushed by [`scripts/sync_trackers_git.py`](../../scripts/sync_trackers_git.py).

### Non-Goals
- **NG1.** No TransNeXt training, tuning, or any modification to the quarantine guard.
- **NG2.** No Phase C execution (single-axis isolation, 100 CNN cells) — separate PRD.
- **NG3.** No new training-engine refactors; Lightning callbacks, datamodule, and `run_cell` dispatcher stay as US-014..US-019 left them.
- **NG4.** No new Optuna search space — the priors at [`artifacts/priors/`](../../artifacts/priors/) are frozen by US-015. Decade-bound enforcement (`MAX_LOGUNIFORM_RATIO = 100`) holds.
- **NG5.** No live results synthesis (master CSV, paper figures) — those are US-022/US-023 of the [US_020_GPU_BOOTSTRAP.md](US_020_GPU_BOOTSTRAP.md) roadmap and run after Phase B + C.
- **NG6.** No reading or copying of `*.ckpt`/`*.pt`/`*.pth` files. Phase B run dirs are local-only (`runs/final/**` is gitignored + claudeignored).

---

## 3. Technical Constraints & Guardrails (THz Protocol)

- **Deterministic Integrity.** Per-sample seeded degradation (`seed = idx + SEED_OFFSET_VAL`) gives byte-identical val pixels across all models and runs (MSE = 0). The pre-flight gate (US-021) re-runs [`src/tests/test_degradation_determinism.py`](../../src/tests/test_degradation_determinism.py) before any GPU work begins. Visual-core PNGs already on disk (US-017) must remain byte-stable across the run.
- **Weight Isolation (Privacy First).** No story in this PRD opens a weight binary. The runbook's Stage 3 sync explicitly excludes `runs/final/**`, `artifacts/optuna_thz.db`, and `artifacts/weights/`. Verified by [`src/tests/test_sync_trackers_git.py`](../../src/tests/test_sync_trackers_git.py) and [`src/tests/test_ignores.py`](../../src/tests/test_ignores.py).
- **The 5x5 Matrix.** Phase B uses all 5 axes simultaneously at one of 5 levels. Levels resolved through [`src/data/degradation_levels.py`](../../src/data/degradation_levels.py) `level_params(level)` — single source of truth.
- **Metric Synthesis.** Each cell produces `(best_val_acc, PSNR_mean ± std, SSIM_mean ± std)` so downstream Phase C analysis can correlate accuracy ↔ visual quality. PSNR/SSIM is auto-written by `_measure_image_quality_for_cell` (US-016) on every Phase B cell completion.
- **Convergence-First.** `--mode pilot` is rejected for `--plan final` (assertion in [`run_all_phases.py`](../../run_all_phases.py)). Max epochs 60, early-stop patience 10, monitor `val_acc`. Frozen per CLAUDE.md; not re-tunable inside this PRD.
- **Git/Sync Hygiene.** Tracker writes flow through [`scripts/refresh_trackers.py`](../../scripts/refresh_trackers.py) only. Git commits/pushes flow through [`scripts/sync_trackers_git.py`](../../scripts/sync_trackers_git.py) only — `FORBIDDEN_FLAGS = {--force, --no-verify, -f, -i}` already enforced. No story below may bypass either script.
- **SYNCHRONIZER boundary.** Per US-016 (PHASE_B_VISUAL_CORE), per-phase boundary push only — never per-cell. The Stage 3 verification story (US-027) is the single push for Phase B.

---

## 4. Implementation Plan (User Stories)

Stories are dependency-ordered: **bootstrap → pre-flight → tune → execute (4 sub-batches) → verify+sync**. Each story is one Ralph iteration (~10 min of focused operator work, plus the wall-clock GPU time noted in the description).

### US-020: GPU Environment Bootstrap — Lock & Smoke Test

**Status (2026-05-07):** ✅ DONE on local RTX 4050 Laptop (Py 3.12.10, torch 2.11.0+cu128, smoke test PASS at `--min-vram-gib 5.9`). Re-execute on the new GPU box from a clean checkout.

**Description:** Bring the GPU box from a fresh clone to a verified `torch.cuda.is_available() == True` with the project deps frozen. The script [`scripts/setup_gpu_env.py`](../../scripts/setup_gpu_env.py) already exists; this story is to execute it on the target hardware, freeze `requirements.lock.txt`, and confirm the audit + smoke-test bands. Refuses Python ≥ 3.13 (no PyTorch wheels yet) and prints the exact `py -3.11 -m venv` command to bootstrap a 3.11 venv.

**Technical Implementation:**
- Operator commands (from [PHASE_B_EXECUTION.md](../runbooks/PHASE_B_EXECUTION.md) Stage -1):
  - `python scripts/setup_gpu_env.py --audit-only` (zero-side-effect hardware report).
  - `python scripts/setup_gpu_env.py --make-venv .venv-gpu` if Python ≥ 3.13 detected.
  - `python scripts/setup_gpu_env.py` (full bootstrap: install + smoke-test).
- Output verification:
  - `requirements.lock.txt` written in UTF-8 (so future drift checks compare cleanly with the legacy UTF-16 `requirements.txt`).
  - Stdout includes `Stage -1 complete — environment ready for Stage 1 (Optuna pre-tune)`.

**Acceptance Criteria:**
- [ ] Logic: `python scripts/setup_gpu_env.py --audit-only` exits 0 on the GPU box and reports driver, CUDA, and per-device VRAM.
- [ ] Logic: post-install, `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` prints `True <gpu>`.
- [ ] Logic: `requirements.lock.txt` is committed and contains `torch`, `torchvision`, `pytorch-lightning`, `optuna`, `optuna-integration`, `plotly`, `pandas`, `opencv-python`, `matplotlib`, `timm`, `scikit-learn`, `torchmetrics`, `rich`, `pyyaml`, `tqdm`, `psutil` — at the resolved version pins.
- [ ] Privacy: bootstrap never reads `*.ckpt`/`*.pt`/`*.pth`; `.venv-gpu/` is gitignored (verified by `git status` after the run).
- [ ] Quality: `mypy scripts/setup_gpu_env.py` passes; `pytest src/tests` 84-test baseline still green (no torch-required tests run on dev box, but torch-free suite must remain unaffected).
- [ ] Verification: stdout ends with `Stage -1 complete — environment ready for Stage 1 (Optuna pre-tune)`. Operator pastes the audit summary into [`progress.txt`](../../progress.txt) under a new `## US-020` heading.

---

### US-021: Pre-Flight Gate — Test Suite + Quarantine + GPU Sanity

**Status (2026-05-07):** ✅ DONE — substantively PASS. `test_priors` 8/8, `test_build_final_exp_json` 14/14, `tune_all.py --validate-only` clean, `nvidia-smi` shows zero `python.exe` on GPU. `test_quarantine_transnext` 7/9 with 2 false positives from VSCode Pylance LSP scanning project files; non-blocking. Re-run on the new box; expect identical pattern unless the LSP daemon isn't open.

**Description:** A single command sequence that fails fast if any of the Phase B preconditions are off — torch-free test suite green, priors valid, TransNeXt quarantine intact, no zombie GPU processes holding VRAM. Burns ~30 seconds; saves multi-GPU-hour mistakes.

**Technical Implementation:**
- Operator commands (from [PHASE_B_EXECUTION.md](../runbooks/PHASE_B_EXECUTION.md) Pre-flight):
  - `python -m src.tests.test_priors` → expect 8 OK lines.
  - `python -m src.tests.test_quarantine_transnext` → expect 9 PASS lines.
  - `python -m src.tests.test_build_final_exp_json` → expect 13 OK lines (US-019 added 3).
  - `python tune_all.py --validate-only` → expect `All 3 priors files valid.`
  - `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` → expect zero `python.exe` rows.
- Optional regression smoke (recommended on a fresh clone):
  - `pytest src/tests -x --ignore=src/tests/test_degradation_determinism.py` — full torch-free suite green.
  - `python -m src.tests.test_degradation_determinism` — torch-required determinism gate (MSE = 0 across seeds).

**Acceptance Criteria:**
- [ ] Logic: all five pre-flight commands return exit code 0 and the expected output strings.
- [ ] Logic: `python -c "import json; d=json.load(open('artifacts/Final_Exp.json')); print(sum(1 for r in d['rows'] if 'transnext' in r['model'] and r.get('status','Pending') != 'Pending'))"` prints `0`.
- [ ] Logic: `nvidia-smi --query-compute-apps=...` returns no `python.exe` processes — if any are present, halt and run `python scripts/quarantine_transnext.py --yes` before proceeding.
- [ ] Privacy: pre-flight reads only test fixtures, priors JSON, and `Final_Exp.json`. No `*.ckpt`/`*.pt`/`*.pth` access.
- [ ] Quality: torch-free pytest suite green (≥ 84 tests); `mypy src scripts` green.
- [ ] Verification: a one-line summary appended to [`progress.txt`](../../progress.txt) under `## US-021` listing each command + its observed exit code.

---

### US-022: Optuna Pre-Tune — `resnet50` × `cifar10` at L3 Moderate (≈ 3.5 GPU-h)

**Status (2026-05-07):** Stage 1 fast tune ✅ DONE (26 trials, best fast val_acc=0.444 @ trial #1). Stage 1.5 validation ⏳ PENDING on stronger GPU per O4.

**Description:** First of four per-pair Optuna studies (decision O1 = split). Runs 20 trials seeded from the US-015 `resnet50` priors, all at L3 Moderate. Winner freezes to `artifacts/best_hparams/resnet50_cifar10.json` with `priors_file_hash` round-tripped. Splitting into per-pair calls gives finer-grained checkpointing — a stall on one pair never blocks progress on the others.

**Technical Implementation:**
- One command:
  - `python tune_all.py --n-trials 20 --model resnet50 --dataset cifar10`
- Resumable: the Optuna SQLite DB at [`artifacts/optuna_thz.db`](../../artifacts/optuna_thz.db) persists trial state per (model, dataset) study name. Re-run the same command after a crash and trials resume from where they died.
- Fail-soft: `tune_all.py` already wraps `Trainer.fit` failures into pruned Optuna trials, not hard crashes; OOM trials are pruned and the study continues.

**Acceptance Criteria:**
- [ ] Logic: `artifacts/best_hparams/resnet50_cifar10.json` exists.
- [ ] Logic: the JSON's `priors_file_hash` equals `python -c "import tune_all; print(tune_all.priors_file_hash('resnet50'))"`.
- [ ] Logic: the JSON contains `lr_head`, `lr_backbone`, `weight_decay`, `label_smoothing`, `batch_size`, plus the `study_name`, `best_value`, `n_trials_completed` ≥ 18 (allowing for ≤ 2 pruned trials).
- [ ] Privacy: no `*.ckpt`/`*.pt`/`*.pth` files staged. `artifacts/optuna_thz.db` is gitignored — verify with `git check-ignore artifacts/optuna_thz.db`.
- [ ] Quality: existing torch-free pytest suite still green; `priors_file_hash` round-trip asserted by `test_priors` and `test_quarantine_transnext`.
- [ ] Verification: append a one-line entry to [`progress.txt`](../../progress.txt): `US-022: resnet50_cifar10 winner frozen (best_value=<x>, trials=<n>)`.

---

### US-023: Optuna Pre-Tune — `densenet121` × `cifar10` at L3 Moderate (≈ 3.5 GPU-h)

**Status (2026-05-07):** Stage 1 fast tune ✅ DONE (26 trials, best fast val_acc=0.535 @ trial #15). Stage 1.5 validation ⏳ PENDING on stronger GPU per O4.

**Description:** Second per-pair Optuna study. Runs 20 trials seeded from the US-015 `densenet121` priors at L3 Moderate. Winner freezes to `artifacts/best_hparams/densenet121_cifar10.json`. Independent of US-022; can run on a separate GPU concurrently.

**Technical Implementation:**
- One command:
  - `python tune_all.py --n-trials 20 --model densenet121 --dataset cifar10`
- Resumable as US-022.

**Acceptance Criteria:**
- [ ] Logic: `artifacts/best_hparams/densenet121_cifar10.json` exists.
- [ ] Logic: the JSON's `priors_file_hash` equals `python -c "import tune_all; print(tune_all.priors_file_hash('densenet121'))"`.
- [ ] Logic: the JSON contains the same shape as US-022 (5 hparams + 4 metadata fields), `n_trials_completed` ≥ 18.
- [ ] Privacy: as US-022 — no weight binaries staged.
- [ ] Quality: torch-free pytest suite still green.
- [ ] Verification: append `US-023: densenet121_cifar10 winner frozen (best_value=<x>, trials=<n>)` to [`progress.txt`](../../progress.txt).

---

### US-024: Optuna Pre-Tune — `resnet50` × `mnist` at L3 Moderate (≈ 3 GPU-h)

**Status (2026-05-07):** Stage 1 fast tune ✅ DONE (20 trials, best fast val_acc=0.949 @ trial #0). Stage 1.5 validation ⏳ PENDING on stronger GPU per O4.

**Description:** Third per-pair Optuna study. MNIST is faster than CIFAR per trial (smaller effective dataset, easier convergence), so this study typically finishes in ~3 GPU-h.

**Technical Implementation:**
- One command:
  - `python tune_all.py --n-trials 20 --model resnet50 --dataset mnist`
- Resumable as US-022.

**Acceptance Criteria:**
- [ ] Logic: `artifacts/best_hparams/resnet50_mnist.json` exists with the standard shape.
- [ ] Logic: `priors_file_hash` matches `tune_all.priors_file_hash('resnet50')` (same priors file as US-022; the hash is per-model, not per-pair).
- [ ] Logic: `n_trials_completed` ≥ 18.
- [ ] Privacy: as US-022.
- [ ] Quality: torch-free pytest suite still green.
- [ ] Verification: append `US-024: resnet50_mnist winner frozen (best_value=<x>, trials=<n>)` to [`progress.txt`](../../progress.txt).

---

### US-025: Optuna Pre-Tune — `densenet121` × `mnist` at L3 Moderate (≈ 3 GPU-h)

**Status (2026-05-07):** Stage 1 fast tune ✅ DONE (20 trials, best fast val_acc=0.956 @ trial #0). Stage 1.5 validation ⏳ PENDING on stronger GPU per O4.

**Description:** Fourth and final per-pair Optuna study. Closes the pre-tune phase; once US-022..US-025 land, all 4 winner JSONs are frozen and the sweep stories (US-026..US-029) become unblocked.

**Technical Implementation:**
- One command:
  - `python tune_all.py --n-trials 20 --model densenet121 --dataset mnist`
- Resumable as US-022.

**Acceptance Criteria:**
- [ ] Logic: `artifacts/best_hparams/densenet121_mnist.json` exists with the standard shape.
- [ ] Logic: `priors_file_hash` matches `tune_all.priors_file_hash('densenet121')` (same priors file as US-023).
- [ ] Logic: `n_trials_completed` ≥ 18.
- [ ] Logic: closing check — `python -c "import json; from pathlib import Path; print(all(Path(f'artifacts/best_hparams/{m}_{d}.json').exists() for m in ('resnet50','densenet121') for d in ('cifar10','mnist')))"` prints `True`.
- [ ] Logic: `artifacts/best_hparams/transnext_base_*.json` does NOT exist (US-014 quarantine respected end-to-end).
- [ ] Privacy: as US-022.
- [ ] Quality: torch-free pytest suite still green.
- [ ] Verification: append `US-025: densenet121_mnist winner frozen — all 4 winners present` to [`progress.txt`](../../progress.txt).

---

### US-026: Phase B Execution — Sub-Batch 1 (`resnet50` × `cifar10` × L1..L5) (≈ 2.5 GPU-h)

**Status (2026-05-07):** ⏳ PENDING — gated on Stage 1.5 validation (US-022 winner JSON must have `validated_at_full_convergence: true`).

**Description:** Run the 5 cells `final_B_L{1..5}_resnet50_cifar10` using the frozen winner from US-022. Per-cell side effects (already wired by US-016 → US-019): metrics + PSNR/SSIM + visual-core thumb + per-cell incremental tracker refresh. Sub-batch boundary chosen so a crash loses ≤ 5 cells of work.

**Technical Implementation:**
- One command:
  - `python run_all_phases.py --plan final --phase B --model resnet50 --dataset cifar10 --skip-existing`
- Per-cell behavior in [`run_all_phases.py`](../../run_all_phases.py):
  1. Lightning trains to convergence (60 epochs / patience 10 / monitor val_acc).
  2. `_merge_metadata_into_metrics_json` (US-008) round-trips `hparams_source.priors_file_hash`.
  3. `_measure_image_quality_for_cell` (US-016) writes `runs/final/<tag>/image_quality.json`.
  4. `refresh_fn(cell=spec.tag)` (US-019) patches the single row in [`artifacts/Final_Exp.json`](../../artifacts/Final_Exp.json) and rebuilds the dashboard HTML.
- Ctrl-C semantics: SIGINT writes `runs/final/<tag>/INTERRUPTED`; the dashboard renders that cell as `Failed`. Re-run the same command to resume — `--skip-existing` won't re-skip a Failed cell (no completed `metrics.json`).

**Acceptance Criteria:**
- [ ] Logic: 5 directories exist under `runs/final/final_B_L{1..5}_resnet50_cifar10/`, each containing `metrics.json` with non-null `best_val_acc`, `epochs_run`, `hparams_source.priors_file_hash`.
- [ ] Logic: each cell has a sibling `image_quality.json` with `psnr_mean`, `psnr_std`, `ssim_mean`, `ssim_std`.
- [ ] Logic: each cell has `runs/final/<tag>/history.json` (US-018 callback) with `len(history) ≥ 1`.
- [ ] Logic: the dashboard at [`artifacts/Final_Exp.html`](../../artifacts/Final_Exp.html) shows all 5 rows with status `Complete`, the visual-core thumbnail rendered, and clicking the row opens the curves drawer.
- [ ] Privacy: `git status runs/final/` shows the 5 new dirs untracked (gitignored). No `*.ckpt`/`*.pt`/`*.pth` ever staged.
- [ ] Quality: torch-free pytest suite still green after the run (no test fixtures should depend on absent run dirs).
- [ ] Verification: `python -c "import json; d=json.load(open('artifacts/Final_Exp.json')); print(sum(1 for r in d['rows'] if r['model']=='resnet50' and r['dataset']=='cifar10' and r['phase']=='B' and r['status']=='Complete'))"` prints `5`. One-line entry in [`progress.txt`](../../progress.txt).

---

### US-027: Phase B Execution — Sub-Batch 2 (`densenet121` × `cifar10` × L1..L5) (≈ 2.5 GPU-h)

**Status (2026-05-07):** ⏳ PENDING — gated on Stage 1.5 validation of US-023.

**Description:** Same shape as US-026 for the `densenet121` × `cifar10` pair. Independent of US-026 once US-023 has frozen `densenet121_cifar10.json`.

**Technical Implementation:**
- `python run_all_phases.py --plan final --phase B --model densenet121 --dataset cifar10 --skip-existing`

**Acceptance Criteria:**
- [ ] Logic: 5 directories under `runs/final/final_B_L{1..5}_densenet121_cifar10/` each with `metrics.json` + `image_quality.json` + `history.json`.
- [ ] Logic: dashboard renders all 5 rows `Complete` with thumbs and curves.
- [ ] Logic: `metrics.json.hparams_source.priors_file_hash` equals `tune_all.priors_file_hash('densenet121')` for every cell.
- [ ] Privacy: as US-026 — no weight binaries staged.
- [ ] Quality: torch-free pytest suite still green.
- [ ] Verification: `Final_Exp.json` count for `(densenet121, cifar10, B, Complete)` is `5`. Entry in [`progress.txt`](../../progress.txt).

---

### US-028: Phase B Execution — Sub-Batch 3 (`resnet50` × `mnist` × L1..L5) (≈ 2 GPU-h)

**Status (2026-05-07):** ⏳ PENDING — gated on Stage 1.5 validation of US-024.

**Description:** Same shape for the `resnet50` × `mnist` pair.

**Technical Implementation:**
- `python run_all_phases.py --plan final --phase B --model resnet50 --dataset mnist --skip-existing`

**Acceptance Criteria:**
- [ ] Logic: 5 directories under `runs/final/final_B_L{1..5}_resnet50_mnist/` each with `metrics.json` + `image_quality.json` + `history.json`.
- [ ] Logic: dashboard renders all 5 rows `Complete` with thumbs and curves.
- [ ] Logic: `priors_file_hash` matches `tune_all.priors_file_hash('resnet50')`.
- [ ] Privacy: as US-026.
- [ ] Quality: torch-free pytest suite still green.
- [ ] Verification: `Final_Exp.json` count for `(resnet50, mnist, B, Complete)` is `5`. Entry in [`progress.txt`](../../progress.txt).

---

### US-029: Phase B Execution — Sub-Batch 4 (`densenet121` × `mnist` × L1..L5) (≈ 2 GPU-h)

**Status (2026-05-07):** ⏳ PENDING — gated on Stage 1.5 validation of US-025.

**Description:** Final sub-batch; closes the 20-cell Phase B CNN sweep.

**Technical Implementation:**
- `python run_all_phases.py --plan final --phase B --model densenet121 --dataset mnist --skip-existing`

**Acceptance Criteria:**
- [ ] Logic: 5 directories under `runs/final/final_B_L{1..5}_densenet121_mnist/` each with `metrics.json` + `image_quality.json` + `history.json`.
- [ ] Logic: dashboard renders all 5 rows `Complete` with thumbs and curves.
- [ ] Logic: `priors_file_hash` matches `tune_all.priors_file_hash('densenet121')`.
- [ ] Privacy: as US-026.
- [ ] Quality: torch-free pytest suite still green.
- [ ] Verification: `Final_Exp.json` count for `(densenet121, mnist, B, Complete)` is `5`. Entry in [`progress.txt`](../../progress.txt).

---

### US-030: End-of-Phase Verification + SYNCHRONIZER Push

**Status (2026-05-07):** ⏳ PENDING — gated on US-026..US-029 sweep completion.

**Description:** Final tracker refresh, dashboard inspection, and the single per-phase boundary push for Phase B (per US-016 SYNCHRONIZER decision Q3=C). Closes Phase B.

**Technical Implementation:**
- Operator commands (from [PHASE_B_EXECUTION.md](../runbooks/PHASE_B_EXECUTION.md) Stage 3):
  - `python scripts/refresh_trackers.py` (full refresh — all 186 rows; not the `--cell` incremental form).
  - `python -c "import json; d=json.load(open('artifacts/Final_Exp.json')); print(d['counts'])"` — expect `complete >= 24` (4 Phase A + 20 Phase B).
  - Open [`artifacts/Final_Exp.html`](../../artifacts/Final_Exp.html) in a browser; confirm all 4 visual checks below.
  - `git status` — review staged paths.
  - `python scripts/sync_trackers_git.py` — commit + push tracker artifacts (pathspec: `Final_Exp.md`, `artifacts/Final_Exp.html`, `artifacts/Final_Exp.json`, `artifacts/best_hparams/*.json`, `progress.txt`).
- Visual checks on [`artifacts/Final_Exp.html`](../../artifacts/Final_Exp.html):
  1. Visual column shows Original|Degraded thumbs for every Phase A + Phase B row.
  2. Status pills: Phase A green, Phase B mostly green, TransNeXt rows dashed-purple `Deferred · Awaiting Native-Resolution Refactor`.
  3. Best `val_acc` populates the per-phase stat card.
  4. Clicking any Phase B row opens the curves drawer with both Loss and Accuracy panels populated.

**Acceptance Criteria:**
- [ ] Logic: [`Final_Exp.md`](../../Final_Exp.md) status block reads `Phase A (Clean): 4/6`, `Phase B (Combined): 20/30`, `Deferred (TransNeXt — Awaiting Native-Resolution Refactor): 62/186`, `Total: 24/186`.
- [ ] Logic: `python scripts/sync_trackers_git.py` succeeds with `commit_ok=True, push_ok=True, errors=[]`. The commit message matches `chore(trackers): refresh after Phase B (20 cells)` (from US-016 contract).
- [ ] Logic: `git log --name-only -1` shows only the tracker pathspec — no `runs/final/**`, no `artifacts/optuna_thz.db`, no `artifacts/weights/**`, no `*.ckpt`/`*.pt`/`*.pth`.
- [ ] Privacy: dashboard HTML embeds no host paths or weight URIs (manual grep on the rendered file: `grep -E '\\.(ckpt|pt|pth)|C:\\\\|/home/' artifacts/Final_Exp.html` returns zero matches).
- [ ] Quality: full torch-free pytest suite green; `mypy src scripts` green; [`src/tests/test_sync_trackers_git.py`](../../src/tests/test_sync_trackers_git.py) (9/9) and [`src/tests/test_ignores.py`](../../src/tests/test_ignores.py) green to confirm no privacy regression.
- [ ] Verification: a final entry in [`progress.txt`](../../progress.txt): `Phase B (CNN) closed — 20/20 cells Complete, 4 best_hparams winners frozen, sync_trackers_git push OK at <commit_sha>`.

---

## 5. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Stage -1 fails on Py 3.13+ / unmatched CUDA wheel** | High | High (blocks everything) | `setup_gpu_env.py` exits 1 with the exact `py -3.11 -m venv` command; `--cuda-override 12.1` lets the operator force a specific wheel index. US-020 acceptance includes the exit-band verification. |
| **Optuna trial OOM on CIFAR-10 + ResNet50** | Med | High | Priors `batch_size` choice list capped at 64; `tune_all.py` already wraps `Trainer.fit` failures into pruned trials, not hard crashes. Re-run the same command — the SQLite study resumes. |
| **Cell crash mid-sub-batch (CUDA, disk full, NaN)** | Med | Med | `--skip-existing` + `INTERRUPTED` sentinel = safe re-run idempotency. The atomic `metrics.json` write (US-016) means partial cells leave no stale stats. |
| **`history.json` corrupted by partial write on Ctrl-C** | Low | Low | `HistoryJSONCallback` uses `tmp + os.replace` atomic write on every `on_validation_epoch_end` (US-018); SIGINT'd cells still produce a usable curve. |
| **TransNeXt zombie process still holds GPU** | Med | High | US-021 pre-flight asserts zero `python.exe` PIDs in `nvidia-smi --query-compute-apps`. If present, halt and run `python scripts/quarantine_transnext.py --yes` before any tuning or execution. |
| **Visual Core PNG drift due to seed offset change** | Low | High | Existing `test_render_thumbs.py` (torch-gated) covers byte-stable composite shape. US-021 pre-flight surfaces the drift before GPU hours are spent. |
| **SYNCHRONIZER push leaks weight binaries** | Low | Critical | Pathspec is explicit; `FORBIDDEN_FLAGS` blocks `--force`/`--no-verify`; `test_sync_trackers_git` (9/9) covers all leak paths. US-027 verification includes a manual `git log --name-only -1` review. |
| **Dashboard 10-min polling masks fast-failing cells** | Low | Low | Per-cell `refresh_fn(cell=spec.tag)` (US-019) writes JSON immediately; the 600 s interval is only the dashboard's auto-pull. The "🔄 Refresh now" button forces an instant fetch. |
| **Operator runs `--mode pilot` by mistake** | Low | High | `run_all_phases.py` rejects the combination `--plan final --mode pilot` with an assertion (CLAUDE.md "quality over speed"). Documented in US-022..US-026 commands; no story uses `--mode pilot`. |
| **Sub-batch finishes but tracker is stale** | Low | Low | Each `Trainer.fit` returns through `refresh_fn(cell=spec.tag)` before the next cell starts (US-019). End-of-batch sanity is the per-pair `Final_Exp.json` count check in US-023..US-026 acceptance. |

### Fail-Soft Contract

1. **Skip-existing.** `run_all_phases.py --skip-existing` no-ops any cell whose `runs/final/<tag>/metrics.json` shows convergence — re-running a sub-batch after partial crash is always safe.
2. **INTERRUPTED sentinel.** SIGINT on any cell drops `runs/final/<tag>/INTERRUPTED`; the dashboard renders it `Failed`; the next `--skip-existing` run will re-attempt that cell (no completed metrics to skip).
3. **Optuna resumability.** [`artifacts/optuna_thz.db`](../../artifacts/optuna_thz.db) persists every trial; the same `python tune_all.py --n-trials 20` after a crash continues until 20/20 completed.
4. **Aggregator preserves last-known-good.** `update_cell` falls back to a full rebuild on schema mismatch / corrupt JSON (US-019); a single bad row never wipes the dashboard.
5. **Tracker writes are atomic.** Both `Final_Exp.json` and `history.json` use `tmp + os.replace`; a Ctrl-C mid-write leaves the previous good file intact.
6. **Sync is fail-soft.** `sync_trackers_git.py` logs `[sync][WARN]` and populates `errors` on any subprocess failure but never raises — a transient network hiccup at US-027 doesn't poison the run.

### Batching Strategy

- **20 CNN cells in 4 sub-batches of 5** (one per `(model, dataset)` pair × all 5 levels). A crashed sub-batch loses at most 5 cells of work.
- **Optuna pre-tune is 4 studies × 20 trials ≈ 12–14 GPU-hours.** Run sequentially in one command; resume on crash via the SQLite store.
- **Total wall-clock budget:** ≈ 22–24 GPU-hours (12–14 tune + ~10 sweep). With the poster deadline at 2026-05-31 (T-24d), Stage 1 should kick off no later than 2026-05-12 to leave headroom for Phase C.

### State Persistence

- `artifacts/Final_Exp.json` updates per-cell via `refresh_trackers.py --cell <tag>` (US-019 incremental mode) — progress is never lost.
- `Final_Exp.md` regenerates on the same boundary; the `## Status Summary` block always reflects the on-disk truth.
- `artifacts/best_hparams/{model}_{dataset}.json` is the durable winner record per pair; once written, US-023..US-026 are independent of the Optuna DB.
- `progress.txt` is the operator's append-only log; US-020..US-027 each write one entry.

---

## 6. Dependency-Ordered Story Map

```
US-020 (GPU bootstrap) ──► US-021 (pre-flight)
                                    │
                                    ├──► US-022 (tune resnet50 × cifar10) ──► US-026 (sweep resnet50 × cifar10) ──┐
                                    │                                                                              │
                                    ├──► US-023 (tune densenet121 × cifar10) ─► US-027 (sweep densenet121 × cifar10)─┤
                                    │                                                                              ├──► US-030 (verify + sync)
                                    ├──► US-024 (tune resnet50 × mnist) ──────► US-028 (sweep resnet50 × mnist) ───┤
                                    │                                                                              │
                                    └──► US-025 (tune densenet121 × mnist) ───► US-029 (sweep densenet121 × mnist)─┘
```

US-020 → US-021 are strictly sequential. The 4 tuning stories (US-022..US-025) are independent once pre-flight passes — operator can parallelize across GPUs or serialize on a single GPU; CIFAR pairs come first per the locked dataset ordering (decision O2 = CIFAR-first). Each sweep story (US-026..US-029) is gated on its matching tune story's winner JSON; sweeps are also pairwise-independent. US-030 is the single per-phase boundary push.

---

## 7. Definition-of-Done (PRD-level)

- [ ] All 11 story acceptance-criteria sets above check green.
- [ ] [`Final_Exp.md`](../../Final_Exp.md) status block: `Phase A: 4/6`, `Phase B: 20/30`, `Phase C: 0/150`, `Deferred: 62/186`, `Total: 24/186`.
- [ ] [`artifacts/Final_Exp.html`](../../artifacts/Final_Exp.html) renders all 4 features end-to-end (visual-core column, lazy curves drawer, 10-min auto-poll, manual refresh) on the 20 new Phase B rows.
- [ ] `pytest src/tests` green (84-test torch-free baseline + any torch-gated tests on the GPU box).
- [ ] `mypy src scripts` green.
- [ ] No `*.ckpt`/`*.pt`/`*.pth` paths leaked into `Final_Exp.json` / `Final_Exp.md` / `Final_Exp.html` / `priors.json` / `best_hparams/*.json` / `history.json` / commit pathspec.
- [ ] Determinism gate ([`src/tests/test_degradation_determinism.py`](../../src/tests/test_degradation_determinism.py)) green (MSE = 0).
- [ ] SYNCHRONIZER push at US-030: single commit `chore(trackers): refresh after Phase B (20 cells)` with the explicit pathspec only.

---

## 8. Decisions Locked (operator 2026-05-07)

- **O1 — Optuna scheduling:** **split** into 4 per-pair invocations (US-022..US-025). Finer-grained checkpointing per pair; a stall on one pair never blocks the others. Single-invocation form is no longer in scope.
- **O2 — Sub-batch order:** **CIFAR-first**. Tune order: US-022 (resnet50 × cifar10) → US-023 (densenet121 × cifar10) → US-024 (resnet50 × mnist) → US-025 (densenet121 × mnist). Sweep order mirrors: US-026 → US-027 → US-028 → US-029.
- **O3 — Commit cadence:** single per-phase boundary push at US-030, per the US-016 SYNCHRONIZER contract (Q3=C). No per-sub-batch pushes.
- **O4 — Tuning protocol (Option C hybrid):** the Optuna pre-tune at [`src/tune_hyperparams.py:run_studies`](../../src/tune_hyperparams.py) ships with `max_epochs=5, train_subset=2000, val_subset=1000` — a fast proxy ranking, NOT convergence-quality. Production sweep (Phase B / Phase C) uses `max_epochs=60, patience=10, full data` per CLAUDE.md. To bridge the gap, US-022..US-025 each have **two stages**:
  - **Stage 1 — fast tune** (existing `tune_all.py --n-trials 20 --model M --dataset D`): 20 Optuna trials at the proxy budget; ranks hparams cheaply.
  - **Stage 1.5 — top-3 validation** ([`scripts/validate_top3.py`](../../scripts/validate_top3.py), new): pulls top-3 trials by fast val_acc from the SQLite study, re-trains each at `max_epochs=60 / train_subset=10000 / val_subset=5000 / patience=10` (production protocol), and writes the best-by-full-val-acc to `artifacts/best_hparams/{model}_{dataset}.json` with `validated_at_full_convergence: true`. Per-trial cache at `artifacts/validation/{m}_{d}_rank{N}.json`.
  - Phase B sweep (US-026..US-029) reads only validated winner JSONs.
  - **Why C, not B-full**: the proxy-based ranking + 60-epoch top-3 validation is ~18 GPU-h on a workstation GPU vs. ~100 GPU-h for full-Optuna-at-60-epochs, while still producing winners that reflect the 60-epoch / 10k-train production protocol the paper claims. Defensible methodology: "Optuna ranked candidates with a 5-epoch / 2k-subset proxy; the top-3 per pair were validated at full convergence."

---

**End of PRD. Operator confirmed 2026-05-07. Stage 1 fast tune executed on local RTX 4050 Laptop. Stage 1.5 validation paused 2026-05-07 — to be resumed on a stronger GPU box per the Handoff Status table at the top of this PRD.**
