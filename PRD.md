# PRD: RTX 5070 RALPH Loop — Autonomous 186-Cell Execution Phase

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Phase Mapping:** Full 186-cell Final Research Phase (`runs/final/`) — Phase A (6) + Phase B (30) + Phase C (150), three models × two datasets across the 5-level degradation curve.
**Continues:** [PHASE_B_RUN.md](PHASE_B_RUN.md) US-020..US-030 — the CNN-only Phase B campaign now extends to TransNeXt (un-quarantined per architecture adaptation below) and to Phase C single-axis isolation. User-story numbering resumes at **US-040**.
**Operator runbook:** [docs/runbooks/PHASE_B_EXECUTION.md](../runbooks/PHASE_B_EXECUTION.md) (Stage -1 / Stage 1 stages still apply); a new Stage 4 — RALPH self-correction loop — is documented here.
**Date:** 2026-05-12
**Target hardware:** NVIDIA RTX 5070 (12 GB GDDR7 VRAM, sm_120 / CUDA 12.8+).

---

## 1. Introduction

The repo has shipped the **infrastructure** for the 186-cell final research matrix (visual core, lazy curve drawer, incremental tracker, priors validation, INTERRUPTED sentinel) and the **CNN-only Phase B** fast-tune stage on an RTX 4050 Laptop (6 GB VRAM). The remaining work is the actual *campaign execution* — every cell trained to convergence with paper-anchored hparams — on a stronger GPU.

This PRD covers the **RTX 5070 RALPH Loop**: an autonomous, self-correcting, sequential execution of all 186 cells on a single RTX 5070 box. "RALPH" (Run-And-Loop-with-Patch-and-Heal) is the convention adopted from the `.claude/skills/prd/SKILL.md` planning skill — each cell is one ~10-minute RALPH iteration (or, where convergence demands, one multi-hour iteration with a single in-loop decision point at the end). The loop:

1. **Resets** prior partial state (Final_Exp, Optuna DB, artifacts) to a known clean baseline.
2. **Optimizes** hyperparameters per `(model, dataset)` pair on L3 Moderate using paper-derived priors only.
3. **Executes** each cell to convergence (60 epochs / patience 10) under the locked protocol.
4. **Self-corrects** on detected pathology (failed convergence, divergence, severe overfit) by switching from linear probe / differential LR to **Full Fine-Tuning** with adjusted hparams, then re-running that single cell exactly once.
5. **Refreshes** the dashboard incrementally and pushes tracker artifacts only at phase boundaries.

The loop is single-GPU sequential by design — the 12 GB VRAM ceiling forecloses naive parallelism, and per-cell determinism (seed=42, workers=True) is non-negotiable for the fair-comparison invariant.

---

## 2. Goals

- **G1.** Wipe four stateful artifacts (`Final_Exp.md`, `Final_Exp.json`, `optuna_thz.db`, runtime caches under `artifacts/`) to a verified clean baseline without losing the priors JSON, the lock file, or the frozen Phase A clean-baseline run dirs.
- **G2.** Bring up the RTX 5070 with `torch.cuda.is_available() == True` on the **cu128** wheel index, freeze `requirements.lock.txt`, and pass the determinism gate ([test_degradation_determinism.py](../../src/tests/test_degradation_determinism.py)) with MSE = 0.
- **G3.** Land a **scaled-down TransNeXt variant** (native-resolution 32×32 / 28×28 stem, see §5) that fits in 12 GB at batch 32 and lifts the US-014 quarantine.
- **G4.** Execute Optuna pre-tune (Stage 1 fast rank, Stage 1.5 top-3 validate at full convergence) for **all 6** `(model, dataset)` pairs at L3 Moderate, freezing winner JSONs to `artifacts/best_hparams/`.
- **G5.** Run all 186 cells to convergence under the RALPH self-correction loop, producing `metrics.json` + `image_quality.json` + `history.json` + visual-core thumb per cell.
- **G6.** End-state: `Final_Exp.md` reads `Phase A: 6/6, Phase B: 30/30, Phase C: 150/150, Total: 186/186`; dashboard renders all 186 rows; single per-phase boundary push at A-close, B-close, C-close.

---

## 3. Non-Goals

- **NG1.** No multi-GPU or DDP. Single RTX 5070, sequential cells.
- **NG2.** No new degradation axes, no new dataset, no new model family. The 186-cell matrix shape is locked by [`src/experiments/cells.py`](../../src/experiments/cells.py).
- **NG3.** No live ensembling, no test-time augmentation, no model-soup post-processing — the campaign measures *training-time* robustness only.
- **NG4.** No paper figures, no LaTeX, no poster export in this PRD — synthesis is a downstream PRD (post-Phase-C).
- **NG5.** No reading/copying of `*.ckpt`/`*.pt`/`*.pth` (CLAUDE.md "Weight Privacy"). The loop produces them locally and never stages them.
- **NG6.** No `--mode pilot` against `--plan final`. The existing assertion in [`run_all_phases.py`](../../run_all_phases.py) blocks the combination.

---

## 4. The Reset Protocol

Before kicking off the loop, four state surfaces must be wiped to a clean baseline. The protocol is **idempotent** and **opt-in destructive** — every step prints what it will remove and requires `--yes` to actually delete.

### 4.1 Files in scope of the reset

| Target | Reset action | Why |
|---|---|---|
| `Final_Exp.md` | Regenerated from `artifacts/Final_Exp.json` by `scripts/refresh_trackers.py` after wipe. | Stale completion counts will mis-render the dashboard banner. |
| `artifacts/Final_Exp.json` | Rebuilt from `runs/final/<tag>/metrics.json` discovery (re-running `src/tools/build_final_exp_json.py`). | The browser fetches this JSON; stale rows leak old `val_acc` into stat cards. |
| `artifacts/optuna_thz.db` | **Backed up** to `artifacts/optuna_thz.db.pre-5070.bak`, then deleted. | Fast-tune trials from the RTX 4050 box used different VRAM/batch defaults — re-tuning on the 5070 must start fresh, but the prior store is kept as a comparison witness. |
| `artifacts/best_hparams/*.json` | Moved to `artifacts/best_hparams/_archive_pre_5070/`. | Pre-5070 winners may have been validated at proxy budget only; re-derive on the new GPU. |
| `artifacts/validation/` | Wiped (rank{1..3}.json caches). | Stale top-3 cache shortcuts will leak old hparams into the new sweep. |
| `artifacts/Final_Exp.html` | Rebuilt by `src/tools/build_final_dashboard.py` after Step 3. | The page caches `safe_json` inline; without rebuild, the polled `Final_Exp.json` won't bind. |
| `runs/final/final_clean_*` | **Preserved**. | Phase A clean baselines are model-clean (no degradation); they don't need a re-run. |
| `runs/final/final_B_*` / `final_C_*` | **Preserved** (gitignored). | A user may still want to introspect prior partial cells. The loop's `--skip-existing` ignores them unless an empty/INTERRUPTED dir is found. |
| `artifacts/priors/*.json` | **Preserved**. | Paper-anchored priors are frozen by US-015; deleting them would re-introduce blind search. |
| `requirements.lock.txt` | **Preserved**. | Cross-box pin; the 5070 may diverge to a new lock written by the next bootstrap. |

### 4.2 Reset commands (RALPH iteration US-040)

```powershell
# DRY RUN — prints every file that would be touched, exits 0.
.venv-gpu\Scripts\python.exe scripts\reset_state.py --dry-run

# REAL run — moves Optuna DB / best_hparams to dated backups, wipes
# Final_Exp.json + Final_Exp.html + validation cache. Requires --yes.
.venv-gpu\Scripts\python.exe scripts\reset_state.py --yes
.venv-gpu\Scripts\python.exe scripts\refresh_trackers.py    # rebuild empty trackers
.venv-gpu\Scripts\python.exe -m src.tools.build_final_dashboard
```

### 4.3 Post-reset verification

- `Final_Exp.md` status block reads `Phase A: <existing>/6, Phase B: 0/30, Phase C: 0/150, Total: <existing>/186`.
- `artifacts/Final_Exp.json` `rows` array has 186 entries, all `status: "Pending"` except the preserved `final_clean_*` Phase A cells.
- `artifacts/optuna_thz.db.pre-5070.bak` exists; the live `artifacts/optuna_thz.db` does NOT.
- `artifacts/best_hparams/` is empty (or contains only `_archive_pre_5070/`).
- `python tune_all.py --validate-only` prints `All 3 priors files valid.` (priors preserved).

---

## 5. Environment & Architecture Specs

### 5.1 RTX 5070 (12 GB GDDR7) — CUDA Workspace

| Setting | Value | Reason |
|---|---|---|
| CUDA toolkit | 12.8+ | sm_120 (Blackwell) is exposed only on cu128 wheels. |
| PyTorch wheel | `torch==2.11.* +cu128` | First wheel index with RTX 5070 support. Lock via `pip install --index-url https://download.pytorch.org/whl/cu128`. |
| TF32 (matmul / cudnn) | `True` / `True` | Free 1.3× throughput on FP32 paths; trains within the convergence-quality envelope per the determinism gate (MSE = 0 across runs, validated post-toggle). |
| `cudnn.benchmark` | `True` (training) | All cells have fixed input shape (224×224 after upsample); autotuner picks once and caches. |
| `cudnn.deterministic` | `False` (training), `True` (val) | Determinism gate runs against val pixels — already byte-identical via per-sample seed offset. Training-time non-determinism is documented as a known design choice. |
| `precision` | `bf16-mixed` (Blackwell native) | RTX 5070 supports bf16 at sm_120 with no accumulator drift; the prior `16-mixed` (fp16) is retained as fallback. |
| `torch.compile` | `mode="reduce-overhead"` on ResNet50 / DenseNet121; `disable` on TransNeXt | TransNeXt's `swattention` path is uncompilable; the wrapper already sets `TRANSNEXT_FORCE_NATIVE=1` (see [`src/models/transnext_wrapper.py`](../../src/models/transnext_wrapper.py)) but Inductor still chokes on the native attention. |
| Pinned memory + persistent workers | `True` | Removes the 5-15% per-epoch dataloader stall measured on the 4050. |
| `CUDA_LAUNCH_BLOCKING` | unset | Set to `1` only when debugging OOM. |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512,expandable_segments:True` | Reduces fragmentation under TransNeXt's bursty allocation. |
| VRAM headroom budget | 1.5 GB reserve | Leaves room for the `cudnn.benchmark` autotuner + a single fallback batch retry on OOM. |

### 5.2 Dynamic Batch Sizing

| Model | Default batch | OOM-fallback batch | Effective batch (accumulation) |
|---|---|---|---|
| ResNet50 | 64 | 32 | 64 (×2 grad accum if 32 wins) |
| DenseNet121 | 64 | 32 | 64 |
| TransNeXt-micro (native 32×32) | 96 | 48 | 96 |
| TransNeXt-small (native 32×32) | 48 | 24 | 96 (×2) |

The loop computes the *effective* batch (after grad-accum) at boot via [`scripts/setup_gpu_env.py`](../../scripts/setup_gpu_env.py) probing a single forward+backward on each model — the first run that fits wins and is frozen into `artifacts/best_hparams/{m}_{d}.json` under a new `effective_batch_size` field. Subsequent cells of the same `(model, dataset)` pair don't re-probe.

### 5.3 TransNeXt Native-Resolution Refactor (lifts the US-014 quarantine)

The vendored upstream TransNeXt assumes a 224×224 input with `patch_size=4`. The project's upsample-to-224 pipeline gives it 224×224 to chew on, but for **native-resolution** training (the route this PRD chooses to bring TransNeXt within 12 GB at full convergence) the wrapper now supports:

| Variant | `img_size` | `patch_size` | `embed_dims` | `depths` | Approx params | Approx peak VRAM @ batch 48 |
|---|---|---|---|---|---|---|
| `transnext_micro_native` | 32 (CIFAR) / 28 (MNIST) | 2 | [48, 96, 192, 384] | [2, 2, 6, 2] | ~6 M | ~3.5 GB |
| `transnext_small_native` | 32 / 28 | 2 | [72, 144, 288, 576] | [3, 3, 12, 3] | ~12 M | ~6 GB |
| `transnext_base_native` | 32 / 28 | 4 | [96, 192, 384, 768] | [3, 3, 12, 3] | ~24 M | ~9 GB |

Implementation lives in [`src/models/transnext_wrapper.py`](../../src/models/transnext_wrapper.py) — extended via a `--transnext_size {micro,small,base}` CLI flag and a `--transnext_native_res` boolean. The wrapper's `_TRANSNEXT_SPECS` dict (already keyed by variant) is augmented with `_native` siblings that override `img_size`, `patch_size`, and pretrain_size (the latter cleared because no public TransNeXt checkpoint exists at sub-32 patch). The native-resolution path is **train-from-scratch**, head-and-backbone tunable, matching the differential-LR protocol for the CNNs rather than the linear-probe-only protocol the upstream paper used.

Acceptance for the un-quarantine step: [`src/tests/test_quarantine_transnext.py`](../../src/tests/test_quarantine_transnext.py) is **inverted** — it now asserts that `--transnext_native_res` paths through `tune_all.py` and `run_all_phases.py` are reachable, and that the legacy 224-upsample TransNeXt path is **only** entered under an explicit `--transnext_legacy_upsample` opt-in flag.

---

## 6. Autonomous Logic — The RALPH Loop

### 6.1 Sequential Execution Order

The 186 cells are executed in a fixed, dependency-ordered sweep. The order is encoded by [`src/experiments/cells.py:iter_cells()`](../../src/experiments/cells.py) (already torch-free; no edit needed). The RALPH loop driver — `scripts/run_ralph_loop.py` (new, see §8 user stories) — iterates this generator and dispatches each cell through `run_systematic.run_cell()`.

```
Phase A  (6 cells)   ─► clean baselines        ─► no tuning required
Phase B  (30 cells)  ─► combined degradation   ─► tuned at L3 (Stage 1 + 1.5)
Phase C  (150 cells) ─► single-axis isolation  ─► reuses Phase B winners
```

Per-cell timeline:

1. **Probe.** Re-read `artifacts/best_hparams/{m}_{d}.json`. Crash if missing for Phase B/C cells (Phase A reads paper-defaults).
2. **Plan.** Compute `DegradeConfig` via [`degradation_levels.py:level_params`](../../src/data/degradation_levels.py); confirm SHA against `metrics.json.degradation_levels_hash` of any sibling cell (drift detector).
3. **Train.** `Trainer.fit` with the locked protocol — 60 epochs, patience 10, monitor val_acc, gradient clip 1.0, AdamW + cosine.
4. **Score.** `_measure_image_quality_for_cell` writes `image_quality.json`; `HistoryJSONCallback` flushes `history.json`.
5. **Diagnose.** The RALPH guard reads `metrics.json` and the last 10 entries of `history.json`. If the pathology matrix (see §6.3) triggers, mark the cell **needs-Full-FT** and write `runs/final/<tag>/NEEDS_FULL_FT` sentinel — do **not** re-run inline; queue it for the post-phase remediation pass.
6. **Refresh.** `scripts/refresh_trackers.py --cell <tag>` patches the single row in `Final_Exp.json`; dashboard auto-polls.
7. **Advance.** Next cell.

A SIGINT mid-cell still drops `runs/final/<tag>/INTERRUPTED` (US-016 contract preserved). On loop resume, `run_ralph_loop.py --skip-existing` re-attempts INTERRUPTED cells and skips Complete ones.

### 6.2 Mandatory Optuna HPO Stage (per `(model, dataset)` pair)

Source-of-truth for the search space: the **`papers/`** directory. The priors loader at [`tune_all.py:load_priors`](../../tune_all.py) already enforces a **decade bound** (`MAX_LOGUNIFORM_RATIO = 100`) so no axis can drift more than one order of magnitude from the paper's reported value. No "blind" exploration is permitted.

Per pair:

- **Stage 1 — Fast Rank.** 20 Optuna trials at proxy budget (5 epochs, 2k train / 1k val subsets). Pruner: median pruner with patience 2. Cost ≈ 90 GPU-min on the 5070.
- **Stage 1.5 — Top-3 Validate.** [`scripts/validate_top3.py`](../../scripts/validate_top3.py) pulls the top 3 trials by Stage-1 val_acc and re-trains each at the production protocol (60 epochs, patience 10, full train/val). The best-by-converged-val_acc wins and is frozen with `validated_at_full_convergence: true`. Cost ≈ 180 GPU-min per pair.

Coverage: **6 pairs** (resnet50, densenet121, transnext_small_native) × (cifar10, mnist) = 6 winner JSONs at `artifacts/best_hparams/{m}_{d}.json`.

Acceptance: every JSON includes `lr_head`, `lr_backbone`, `weight_decay`, `label_smoothing`, `batch_size`, `effective_batch_size`, `study_name`, `best_value`, `n_trials_completed` ≥ 18, `priors_file_hash`, `validated_at_full_convergence: true`.

### 6.3 Scientific Self-Correction Protocol

The RALPH guard inspects each completed cell and decides between three outcomes:

| Verdict | Triggers (any one is sufficient) | Remediation |
|---|---|---|
| **Healthy** | • `best_val_acc` within 5pp of the L3 sweep's median for the pair, **and**<br>• `epochs_run` ≤ 55 (i.e. converged before max-epochs cap), **and**<br>• `val_acc[-1] - val_acc[-5]` ≥ -0.5pp (no late-stage divergence), **and**<br>• `train_acc - val_acc` < 12pp at best-epoch (generalization gap) | None. Mark **Complete**. |
| **Failed Convergence** | • `best_val_acc` < 1.3× `random_baseline` (i.e. < 13% on 10-class), **or**<br>• `epochs_run` == `max_epochs` AND val_acc still rising at +0.3pp/epoch (under-trained: hit the cap), **or**<br>• loss is NaN/Inf in `history.json` after epoch 2 | Re-run **once** under **Full FT** (see §6.4) with `lr_head` ÷ 3, `weight_decay` × 1.5, `label_smoothing` += 0.05 (capped at 0.15). |
| **Overfitting** | • `train_acc - val_acc` ≥ 18pp at the best-val epoch, **or**<br>• val_acc peaks before epoch 15 AND then drops > 4pp from peak over the next 10 epochs, **or**<br>• `best_val_acc - val_acc[-1]` > 6pp (i.e. early-stop fired but the drift is large enough to suggest spurious peak) | Re-run **once** under **Full FT** with `lr_backbone` ÷ 2, `weight_decay` × 2, `dropout` += 0.1 (cap 0.3), `mixup_alpha` = 0.2 if not already set. |

**Each cell is re-runnable at most once.** A second failure leaves the cell with a `runs/final/<tag>/QUARANTINED_AFTER_RETRY` sentinel and a `metrics.json.remediation_attempted: "full_ft", final_status: "failed_after_retry"` field — the dashboard renders it as Failed with a quarantine tooltip; the row is excluded from headline accuracy curves but kept in the 186 denominator (mirroring the US-014 deferral convention).

### 6.4 "Full FT" Switch

For models trained under linear-probe (TransNeXt native variants in pretrained mode) or differential-LR (CNNs), the Full FT switch:

1. Sets `freeze_backbone = False`, `lr_backbone = lr_head / 5` (CNNs) or `lr_backbone = lr_head / 10` (TransNeXt).
2. Disables label smoothing for the retry (`label_smoothing = 0.0`) on the **Failed Convergence** branch only; preserves on **Overfitting**.
3. Re-uses the Optuna-winner `batch_size` and `effective_batch_size` — does not re-probe VRAM.
4. Writes the modified config snapshot to `runs/final/<tag>/retry_config.json` so the retry's `metrics.json` is round-trip-comparable to the original.
5. Emits a one-line entry to `progress.txt`: `RALPH retry <tag> reason=<failed_convergence|overfitting> full_ft=true`.

### 6.5 Loop Termination

The loop terminates cleanly when:

- All 186 rows in `Final_Exp.json` show `status ∈ {Complete, Failed}`, AND
- The end-of-phase verify story (US-046) confirms `complete + failed == 186`, AND
- The single per-phase boundary push has succeeded (`sync_trackers_git push_ok=True`).

A SIGINT at any point leaves the on-disk state safely resumable per the fail-soft contract inherited from [PHASE_B_RUN.md §5](PHASE_B_RUN.md).

---

## 7. Technical Constraints & Guardrails (THz Protocol)

- **Deterministic Integrity.** Per-sample seeded degradation (`seed = idx + SEED_OFFSET_VAL`) gives byte-identical val pixels across runs (MSE = 0). The reset of [`src/data/degradation_levels.py`](../../src/data/degradation_levels.py) on 2026-05-12 (severity bump) is itself a determinism event — the gate must re-pass with the *new* table before any 5070 cell is launched. The hash recorded in every `metrics.json.degradation_levels_hash` will rotate; cross-batch comparisons across the table change are explicitly disallowed.
- **Weight Isolation.** No story below opens a weight binary. `runs/final/**`, `artifacts/optuna_thz.db`, `artifacts/weights/` remain gitignored + claudeignored. Verified by [`src/tests/test_sync_trackers_git.py`](../../src/tests/test_sync_trackers_git.py) and [`src/tests/test_ignores.py`](../../src/tests/test_ignores.py).
- **The 5x5 Matrix.** Phase B uses all 5 axes at one level; Phase C pins 4 axes to L1 and varies one. Single source of truth: [`degradation_levels.py:level_params`](../../src/data/degradation_levels.py) (severity-bumped 2026-05-12).
- **Metric Synthesis.** Every cell produces `(best_val_acc, PSNR_mean ± std, SSIM_mean ± std)`; PSNR/SSIM is auto-written by `_measure_image_quality_for_cell` (US-016).
- **Convergence-First.** 60 epochs / patience 10 / monitor val_acc. Frozen in CLAUDE.md. `run_all_phases.py` rejects `--mode pilot` against `--plan final`.
- **Git/Sync Hygiene.** Tracker writes through [`scripts/refresh_trackers.py`](../../scripts/refresh_trackers.py); commits through [`scripts/sync_trackers_git.py`](../../scripts/sync_trackers_git.py) (FORBIDDEN_FLAGS = `{--force, --no-verify, -f, -i}` enforced). One per-phase boundary push.
- **Plan-Mode Authority.** Per AGENTS.md, MASTER approval is required before any code change. This PRD itself is the approval submission.

---

## 8. Implementation Plan (User Stories)

Stories are dependency-ordered: **reset → bootstrap → architecture refactor → tune (6 pairs) → execute (186 cells) → verify+sync**. Each story is one RALPH iteration — sized to a single AI context window, with explicit acceptance criteria and a "Typecheck passes" final criterion. Numbering resumes at US-040 (the prior PRD closed at US-030).

### US-040: State Reset — Wipe Final_Exp, Optuna DB, validation cache

**Description:** Move the prior Optuna DB and best_hparams to a dated backup, wipe `Final_Exp.json`/`.html`/`validation/`, and verify the priors + lock file survive. Idempotent dry-run by default; destructive run requires `--yes`.

**Acceptance Criteria:**
- [x] `scripts/reset_state.py --dry-run` lists every file it would touch and exits 0.
- [x] `scripts/reset_state.py --yes` moves `artifacts/optuna_thz.db` → `artifacts/optuna_thz.db.pre-5070.bak` and `artifacts/best_hparams/*.json` → `artifacts/best_hparams/_archive_pre_5070/`.
- [x] After the run, `artifacts/Final_Exp.json` and `artifacts/Final_Exp.html` are regenerated empty-but-valid (186 rows, all status `Pending` except preserved Phase A clean baselines).
- [x] `artifacts/priors/*.json` and `requirements.lock.txt` are untouched (verified by SHA-256 round-trip).
- [x] `python tune_all.py --validate-only` prints `All 3 priors files valid.` (priors preserved).
- [x] Typecheck passes (`mypy scripts/reset_state.py`).

---

### US-041: RTX 5070 GPU Bootstrap — cu128 wheel + bf16 smoke test

**Description:** Bring the 5070 box from a fresh checkout to `torch.cuda.is_available() == True` on the cu128 wheel index, with bf16 forward/backward verified and `requirements.lock.txt` re-frozen.

**Acceptance Criteria:**
- [x] `python scripts/setup_gpu_env.py --audit-only` reports driver ≥ R555 and CUDA ≥ 12.8.
- [x] `python scripts/setup_gpu_env.py --index-url https://download.pytorch.org/whl/cu128` resolves `torch==2.11.*+cu128`.
- [x] Smoke test: `python -c "import torch; x=torch.randn(2,3,32,32,device='cuda',dtype=torch.bfloat16); print(x.float().sum().item())"` exits 0.
- [x] `requirements.lock.txt` is re-written in UTF-8 with the new cu128 pins; the prior lock survives as `requirements.lock.txt.pre-5070.bak`.
- [x] [`src/tests/test_degradation_determinism.py`](../../src/tests/test_degradation_determinism.py) passes (MSE = 0) against the **new** 2026-05-12 severity-bumped levels.
- [x] Typecheck passes (`mypy scripts/setup_gpu_env.py`).

---

### US-042: TransNeXt Native-Resolution Variants — Un-quarantine

**Description:** Implement `transnext_{micro,small,base}_native` per §5.3. Lift the US-014 quarantine for the native-res path while keeping the legacy 224-upsample path quarantined behind `--transnext_legacy_upsample`.

**Acceptance Criteria:**
- [ ] `src/models/transnext_wrapper.py:_TRANSNEXT_SPECS` includes the 3 `_native` variants with `img_size`, `patch_size`, `pretrain_size=None`.
- [ ] `python run_all_phases.py --plan final --phase A --model transnext_small_native --dataset cifar10 --dry-run` exits 0 and prints the resolved DegradeConfig (no actual training).
- [ ] Inverted quarantine test: [`src/tests/test_quarantine_transnext.py`](../../src/tests/test_quarantine_transnext.py) updated so `transnext_*_native` is **reachable** but legacy `transnext_base` (224-upsample) is still blocked unless `--transnext_legacy_upsample` is set.
- [ ] `artifacts/priors/transnext_small_native.json` exists (paper-derived priors, decade-bounded), and `python tune_all.py --validate-only` includes it in the validated set.
- [ ] Typecheck passes (`mypy src/models src/experiments`).

---

### US-043 .. US-045: Optuna Pre-Tune (Stage 1 + Stage 1.5) for the 6 pairs

**Description:** Three RALPH iterations, each tuning two pairs in series (CIFAR pair → MNIST pair) per model. Each story drives Stage 1 (fast) + Stage 1.5 (validate) and freezes 2 winner JSONs.

- **US-043:** `resnet50` × {cifar10, mnist} — Stage 1 fast + Stage 1.5 validate.
- **US-044:** `densenet121` × {cifar10, mnist} — same shape.
- **US-045:** `transnext_small_native` × {cifar10, mnist} — same shape; uses the new priors from US-042.

**Acceptance Criteria (per story):**
- [ ] Stage 1 trial count ≥ 18 per study in `artifacts/optuna_thz.db`.
- [ ] Stage 1.5 produces `artifacts/best_hparams/{m}_{d}.json` with `validated_at_full_convergence: true`.
- [ ] `priors_file_hash` round-trips: matches `tune_all.priors_file_hash(model)`.
- [ ] `effective_batch_size` field populated by the bootstrap probe.
- [ ] One-line entry in `progress.txt`: `US-04N: <m>_<d> winner frozen (best_value=<x>, trials=<n>)`.
- [ ] Typecheck passes; torch-free pytest suite still green.

---

### US-046: RALPH Loop Driver — Sequential 186-cell sweep with self-correction

**Description:** `scripts/run_ralph_loop.py` (new) wraps `run_all_phases.py` with the §6.1 sequential dispatch and the §6.3 pathology matrix. Reads the 6 frozen winners; for each cell, after `Trainer.fit` returns, evaluates the matrix and either marks Complete or writes `NEEDS_FULL_FT`. A second pass over `NEEDS_FULL_FT` cells re-runs each once under §6.4 Full FT.

**Acceptance Criteria:**
- [ ] First pass: `python scripts/run_ralph_loop.py --plan final --phase all --skip-existing` walks all 186 cells, writing `metrics.json` + `image_quality.json` + `history.json` + visual-core thumb per cell.
- [ ] Pathology guard: on a synthesized cell with NaN loss at epoch 2 (test fixture), the guard emits `NEEDS_FULL_FT`, not silent Complete.
- [ ] Retry pass: `python scripts/run_ralph_loop.py --remediate-only` consumes `NEEDS_FULL_FT` sentinels, runs each cell once under Full FT, writes `retry_config.json`, and clears the sentinel on success or replaces it with `QUARANTINED_AFTER_RETRY` on second failure.
- [ ] No retried cell runs more than once.
- [ ] `Final_Exp.json` `counts.complete + counts.failed == 186` after both passes.
- [ ] Typecheck passes; new tests in `src/tests/test_ralph_loop.py` cover the pathology matrix + retry idempotency.

---

### US-047: End-of-Campaign Verification + Three Phase-Boundary Pushes

**Description:** After each phase closes (A → B → C), refresh trackers and push the tracker pathspec via `sync_trackers_git.py`. Three commits total: `chore(trackers): refresh after Phase A (6 cells)`, `...Phase B (30 cells)`, `...Phase C (150 cells)`. No per-cell or per-sub-batch pushes.

**Acceptance Criteria:**
- [ ] `Final_Exp.md` reads `Phase A: 6/6, Phase B: 30/30, Phase C: 150/150, Total: 186/186` (with up to N retries-after-retry counted as `Failed`, never `Pending`).
- [ ] Three commits land on origin/NEW_GPU_1; each commit's `git log --name-only -1` shows only the tracker pathspec.
- [ ] No `*.ckpt`/`*.pt`/`*.pth` paths leak into any tracker artifact (manual grep gate from PHASE_B_RUN US-030 reused).
- [ ] [`src/tests/test_sync_trackers_git.py`](../../src/tests/test_sync_trackers_git.py) and [`src/tests/test_ignores.py`](../../src/tests/test_ignores.py) green after all three pushes.
- [ ] Final entry in `progress.txt`: `RTX 5070 RALPH Loop closed — 186/186 cells, 6 winners frozen, 3 phase-boundary pushes OK at <shaA> <shaB> <shaC>`.
- [ ] Typecheck passes.

---

## 9. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **cu128 wheel not yet on PyPI** | Med | High | `setup_gpu_env.py --index-url https://download.pytorch.org/whl/cu128` points at the official cu128 wheelhouse; falls back to the nightly index with operator confirmation. |
| **bf16 destabilizes TransNeXt attention** | Med | Med | Pathology matrix catches NaN loss at epoch 2; Full FT retry under fp32 attention (`autocast_dtype=torch.float32` override on `attention_native` path) is the documented escape hatch. |
| **TransNeXt native variants OOM despite §5.2 sizing** | Med | High | Dynamic batch-size probe at boot retries at /2 once; on second OOM the cell records `QUARANTINED_AFTER_RETRY` and the row stays in the 186 denominator. No silent dropout. |
| **Optuna proxy ranks the wrong winner** | Low | Med | Stage 1.5 top-3 validation at full convergence corrects this — the winner JSON's `validated_at_full_convergence: true` is a gating contract for the sweep. |
| **RALPH retry loop infinite-loops on a degenerate cell** | Low | High | Retry budget is hard-capped at 1 per cell; second failure → `QUARANTINED_AFTER_RETRY` sentinel and the loop advances. Asserted in `src/tests/test_ralph_loop.py`. |
| **Severity bump invalidates legacy comparisons** | Cert | Low | `metrics.json.degradation_levels_hash` rotates; tracker rendering surfaces the new hash; no cross-batch comparison is run against the pre-2026-05-12 table. |
| **SYNCHRONIZER push leaks weight binaries** | Low | Critical | Three explicit pathspec pushes (US-047), `FORBIDDEN_FLAGS` enforced, `test_sync_trackers_git` 9/9 covers leak paths. |
| **Dashboard first-row clipped under sticky header** | Cert (pre-fix) | Low | Fixed 2026-05-12 in this session: `scroll-padding-top: 120px` on `html`, thead `top: 76px`, `scroll-margin-top` + first-row padding. Dashboard rebuild verifies. |

---

## 10. Dependency-Ordered Story Map

```
US-040 (reset) ──► US-041 (5070 bootstrap) ──► US-042 (TransNeXt native variants)
                                                        │
                                                        ├──► US-043 (resnet50 × 2 tune) ──┐
                                                        ├──► US-044 (densenet121 × 2 tune)─┤
                                                        └──► US-045 (transnext × 2 tune) ─┤
                                                                                          ▼
                                                                                  US-046 (RALPH loop)
                                                                                          │
                                                                                          ▼
                                                                                  US-047 (verify + 3 pushes)
```

US-040 → US-041 → US-042 strictly sequential. US-043/044/045 can interleave on a single GPU (priors are independent per model); CIFAR pair within each story comes first. US-046 is gated on all 6 winner JSONs present and `validated_at_full_convergence: true`. US-047 is the single end-of-campaign closer.

---

## 11. Definition-of-Done (PRD-level)

- [ ] All 8 story acceptance-criteria sets above check green.
- [ ] `Final_Exp.md` status block: `Phase A: 6/6, Phase B: 30/30, Phase C: 150/150, Total: 186/186`.
- [ ] `artifacts/Final_Exp.html` renders all 186 rows; first row visible under the sticky header (regression covered by manual visual check + the CSS fix landed 2026-05-12).
- [ ] `pytest src/tests` green (torch-free baseline + new `test_ralph_loop.py` + inverted `test_quarantine_transnext.py`).
- [ ] `mypy src scripts` green.
- [ ] No `*.ckpt`/`*.pt`/`*.pth` paths leaked into `Final_Exp.json` / `.md` / `.html` / `priors.json` / `best_hparams/*.json` / `history.json` / `progress.txt` / any commit pathspec.
- [ ] Determinism gate green (MSE = 0) against the 2026-05-12 severity-bumped table.
- [ ] Three SYNCHRONIZER pushes at US-047, each with the explicit tracker pathspec only.

---

## 12. Decisions Locked (operator 2026-05-12)

- **D1 — Reset scope:** preserve Phase A clean baselines, priors, and lock file; back up Optuna DB and best_hparams under dated suffixes (not deletion) so the pre-5070 study is auditable.
- **D2 — Architecture:** un-quarantine TransNeXt via native-resolution variants (32×32 / 28×28, `patch_size=2`); the legacy 224-upsample path stays quarantined behind an explicit opt-in.
- **D3 — Tuning protocol:** continue Option C hybrid (Stage 1 fast rank + Stage 1.5 top-3 validate) for all 6 pairs. Budget ~27 GPU-hours total tuning (90+180 GPU-min × 6).
- **D4 — Self-correction:** single retry per cell; second failure records `QUARANTINED_AFTER_RETRY` and the row counts toward `Failed` in the 186 denominator.
- **D5 — Commit cadence:** three per-phase boundary pushes (A-close, B-close, C-close), never per-cell. Matches the US-016 SYNCHRONIZER contract (Q3=C, extended).
- **D6 — Severity bump:** the 2026-05-12 [`degradation_levels.py`](../../src/data/degradation_levels.py) bump replaces the 2026-04 table; no cross-batch comparison crosses that hash boundary.

---

**End of PRD. Awaiting MASTER approval. On approval: execute US-040 (reset) first; do not begin tuning until US-041 + US-042 are green.**
