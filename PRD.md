# PRD: RTX 5070 RALPH Loop — Autonomous 186-Cell Execution Phase

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Phase Mapping:** Full 186-cell Final Research Phase (`runs/final/`) — Phase A (6) + Phase B (30) + Phase C (150), three models × two datasets across the 5-level degradation curve.
**Continues:** [PHASE_B_RUN.md](PHASE_B_RUN.md) US-020..US-030 — the CNN-only Phase B campaign now extends to TransNeXt (un-quarantined, **trained at 224×224 like the CNNs**) and to Phase C single-axis isolation. User-story numbering resumes at **US-040**.
**Operator runbook:** [docs/runbooks/PHASE_B_EXECUTION.md](docs/runbooks/PHASE_B_EXECUTION.md) (Stage -1 / Stage 1 stages still apply); a new Stage 4 — RALPH self-correction loop — is documented here.
**Date:** 2026-05-13
**Target hardware:** NVIDIA RTX 5070 (12 GB GDDR7 VRAM, sm_120 / CUDA 12.8+).

---

## 1. Introduction

The repo has shipped the **infrastructure** for the 186-cell final research matrix (visual core, lazy curve drawer, incremental tracker, priors validation, INTERRUPTED sentinel) and the **CNN-only Phase B** fast-tune stage on an RTX 4050 Laptop (6 GB VRAM). The remaining work is the actual *campaign execution* — every cell trained to convergence with paper-anchored hparams — on a stronger GPU.

This PRD covers the **RTX 5070 RALPH Loop**: an autonomous, self-correcting, sequential execution of all 186 cells on a single RTX 5070 box. "RALPH" (Run-And-Loop-with-Patch-and-Heal) is the convention adopted from the `.claude/skills/prd/SKILL.md` planning skill — each cell is one ~10-minute RALPH iteration (or, where convergence demands, one multi-hour iteration with a single in-loop decision point at the end). The loop:

1. **Resets** prior partial state (Final_Exp, Optuna DB, artifacts) to a known clean baseline.
2. **Optimizes** hyperparameters per `(model, dataset)` pair on L3 Moderate using paper-derived priors only.
3. **Executes** each cell to convergence (60 epochs / patience 10) under the locked protocol.
4. **Self-corrects** on detected pathology (failed convergence, divergence, severe overfit) by re-running that single cell once under Full FT with adjusted hparams.
5. **Refreshes** the dashboard incrementally and pushes tracker artifacts only at phase boundaries.

The loop is single-GPU sequential by design — the 12 GB VRAM ceiling forecloses naive parallelism, and per-cell determinism (seed=42, workers=True) is non-negotiable for the fair-comparison invariant.

**Resolution policy (ratified 2026-05-13).** Every model — ResNet50, DenseNet121, **TransNeXt** — trains at **224×224**. The earlier V3 native-resolution refactor (TransNeXt at 32, MNIST padded 28→32) was rolled back; the fair-comparison invariant is back to *same tensor shape* across architectures. TransNeXt loads its 224-pretrained checkpoint and trains under the same differential-LR full-FT regime as the CNNs (head 5e-4 / backbone 5e-5).

---

## 2. Goals

- **G1.** Wipe four stateful artifacts (`Final_Exp.md`, `Final_Exp.json`, `optuna_thz.db`, runtime caches under `artifacts/`) to a verified clean baseline without losing the priors JSON, the lock file, or the frozen Phase A clean-baseline run dirs.
- **G2.** Bring up the RTX 5070 with `torch.cuda.is_available() == True` on the **cu128** wheel index, freeze `requirements.lock.txt`, and pass the determinism gate ([test_degradation_determinism.py](src/tests/test_degradation_determinism.py)) with MSE = 0 at out_size=224 for both datasets.
- **G3.** Train **TransNeXt at 224×224** with the upstream pretrained checkpoint, lifting the legacy US-014 quarantine and matching the CNN training regime end-to-end.
- **G4.** Execute Optuna pre-tune (Stage 1 fast rank, Stage 1.5 top-3 validate at full convergence) for **all 6** `(model, dataset)` pairs at L3 Moderate, freezing winner JSONs to `artifacts/best_hparams/`.
- **G5.** Run all 186 cells to convergence under the RALPH self-correction loop, producing `metrics.json` + `image_quality.json` + `history.json` + visual-core thumb per cell.
- **G6.** End-state: `Final_Exp.md` reads `Phase A: 6/6, Phase B: 30/30, Phase C: 150/150, Total: 186/186`; dashboard renders all 186 rows; single per-phase boundary push at A-close, B-close, C-close.

---

## 3. Non-Goals

- **NG1.** No multi-GPU or DDP. Single RTX 5070, sequential cells.
- **NG2.** No new degradation axes, no new dataset, no new model family. The 186-cell matrix shape is locked by [`src/experiments/cells.py`](src/experiments/cells.py).
- **NG3.** No live ensembling, no test-time augmentation, no model-soup post-processing — the campaign measures *training-time* robustness only.
- **NG4.** No paper figures, no LaTeX, no poster export in this PRD — synthesis is a downstream PRD (post-Phase-C).
- **NG5.** No reading/copying of `*.ckpt`/`*.pt`/`*.pth` (CLAUDE.md "Weight Privacy"). The loop produces them locally and never stages them.
- **NG6.** No `--mode pilot` against `--plan final`. The existing assertion in [`run_all_phases.py`](run_all_phases.py) blocks the combination.
- **NG7.** **No native-resolution TransNeXt path.** The 32×32 / 28→32-pad variants and the asymmetric-resolution insurance trial were ruled out 2026-05-13. The matrix denominator is 186 (no insurance trial outside it).

---

## 4. The Reset Protocol — DONE 2026-05-12 (legacy US-040)

Optuna DB and pre-5070 `best_hparams/*.json` archived to `*.pre-5070.bak` / `_archive_pre_5070/` (paths preserved as comparison witnesses); `Final_Exp.json` / `.html` / `validation/` rebuilt empty-but-valid; priors + lock file untouched. Operator runbook: [`scripts/reset_state.py`](scripts/reset_state.py).

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
| `torch.compile` | `none` (all models) | TransNeXt's `attention_native` path is uncompilable; the CNN compile path is unvalidated. |
| Pinned memory + persistent workers | `True` | Removes the 5-15% per-epoch dataloader stall measured on the 4050. |
| `CUDA_LAUNCH_BLOCKING` | unset | Set to `1` only when debugging OOM. |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512,expandable_segments:True` | Reduces fragmentation under TransNeXt's bursty allocation. |
| VRAM headroom budget | 1.5 GB reserve | Leaves room for the `cudnn.benchmark` autotuner + a single fallback batch retry on OOM. |

### 5.2 Batch Sizing (224×224, uniform)

| Model | Default batch | OOM-fallback batch |
|---|---|---|
| ResNet50 | 32 | 16 |
| DenseNet121 | 32 | 16 |
| TransNeXt-tiny  | 32 | 16 |

Batch 32 is the project's locked default per CLAUDE.md. If a `(model, dataset)` pair OOMs at 32, the bootstrap probe in [`scripts/setup_gpu_env.py`](scripts/setup_gpu_env.py) re-tries at 16 with grad-accum 2 so the effective batch stays 32. The winning batch is frozen into `artifacts/best_hparams/{m}_{d}.json` under `effective_batch_size`. Subsequent cells of the same `(model, dataset)` pair don't re-probe. Peak VRAM for TransNeXt-tiny at batch 32 / 224×224 is ~7 GB (vs ~9 GB for the retired `transnext_small` and ~11 GB for `transnext_base`), so the OOM-fallback path is very unlikely to fire on the 12 GB RTX 5070.

### 5.3 TransNeXt 224×224 — Un-quarantine (replaces the V3 native plan)

The vendored upstream TransNeXt assumes a 224×224 input with `patch_size=4`. The project's data pipeline already upsamples CIFAR (32) / MNIST (28) → 224 via bicubic, so TransNeXt receives 224×224 tensors end-to-end. Pretrained checkpoint loading is exact (no shape mismatch, no random-init of position embeddings or CPB MLPs).

| Variant | `img_size` | `patch_size` | `embed_dims` | `depths` | Approx params | Approx peak VRAM @ batch 32 / 224×224 |
|---|---|---|---|---|---|---|
| `transnext_micro` | 224 | 4 | [48, 96, 192, 384] | [2, 2, 15, 2] | ~12 M | ~5 GB |
| `transnext_tiny`  | 224 | 4 | [72, 144, 288, 576] | [2, 2, 15, 2] | ~28 M | ~7 GB |
| `transnext_small` | 224 | 4 | [72, 144, 288, 576] | [5, 5, 22, 5] | ~50 M | ~9 GB |
| `transnext_base`  | 224 | 4 | [96, 192, 384, 768] | [5, 5, 23, 5] | ~89 M | ~11 GB |

`transnext_tiny` is the canonical campaign variant (swapped from `transnext_small` by US-016 on 2026-05-14 for tighter capacity match to ResNet50 ~25M and ~310 GPU-h saved across the 62 TransNeXt rows vs small — see §12 D9; predecessor base→small swap was US-004 same day, see §12 D8). The matrix's `MODELS` tuple in [`src/experiments/cells.py`](src/experiments/cells.py) is `("resnet50", "densenet121", "transnext_tiny")`. Other sizes (`micro`, `small`, `base`) are reachable via the wrapper for ad-hoc smoke tests but are not part of the 186-cell denominator.

Implementation lives in [`src/models/transnext_wrapper.py`](src/models/transnext_wrapper.py). The wrapper's `_TRANSNEXT_SPECS` dict carries all four upstream architecture rows (micro / tiny / small / base) — no `_native` aliases, no `default_*` overrides. The training mode is **full-FT with differential LR** (head 5e-4, backbone 5e-5), matching the CNN convention and the CLAUDE.md training table.

Acceptance for the un-quarantine: [`src/tests/test_quarantine_transnext.py`](src/tests/test_quarantine_transnext.py) asserts (a) `is_quarantined` is a permanent no-op, (b) `iter_cells()` dispatches all 186 (62 TransNeXt + 124 CNN), (c) `tune_all`'s default sweep includes `transnext_tiny`, (d) `run_final_plan --dry-run --model transnext_tiny` resolves a 224-shape CellSpec.

---

## 6. Autonomous Logic — The RALPH Loop

### 6.1 Sequential Execution Order

The 186 cells are executed in a fixed, dependency-ordered sweep. The order is encoded by [`src/experiments/cells.py:iter_cells()`](src/experiments/cells.py) (already torch-free; no edit needed). The RALPH loop driver — `scripts/run_ralph_loop.py` (new, see §8 user stories) — iterates this generator and dispatches each cell through `run_systematic.run_cell()`.

```
Phase A  (6 cells)   ─► clean baselines        ─► no tuning required
Phase B  (30 cells)  ─► combined degradation   ─► tuned at L3 (Stage 1 + 1.5)
Phase C  (150 cells) ─► single-axis isolation  ─► reuses Phase B winners
```

Per-cell timeline:

1. **Probe.** Re-read `artifacts/best_hparams/{m}_{d}.json`. Crash if missing for Phase B/C cells (Phase A reads paper-defaults).
2. **Plan.** Compute `DegradeConfig` via [`degradation_levels.py:level_params`](src/data/degradation_levels.py); confirm SHA against `metrics.json.degradation_levels_hash` of any sibling cell (drift detector).
3. **Train.** `Trainer.fit` with the locked protocol — 60 epochs, patience 10, monitor val_acc, gradient clip 1.0, AdamW + cosine, 224×224.
4. **Score.** `_measure_image_quality_for_cell` writes `image_quality.json`; `HistoryJSONCallback` flushes `history.json`.
5. **Diagnose.** The RALPH guard reads `metrics.json` and the last 10 entries of `history.json`. If the pathology matrix (see §6.3) triggers, mark the cell **needs-Full-FT** and write `runs/final/<tag>/NEEDS_FULL_FT` sentinel — do **not** re-run inline; queue it for the post-phase remediation pass.
6. **Refresh.** `scripts/refresh_trackers.py --cell <tag>` patches the single row in `Final_Exp.json`; dashboard auto-polls.
7. **Advance.** Next cell.

A SIGINT mid-cell still drops `runs/final/<tag>/INTERRUPTED` (US-016 contract preserved). On loop resume, `run_ralph_loop.py --skip-existing` re-attempts INTERRUPTED cells and skips Complete ones.

### 6.2 Mandatory Optuna HPO Stage (per `(model, dataset)` pair)

Source-of-truth for the search space: the **`papers/`** directory. The priors loader at [`tune_all.py:load_priors`](tune_all.py) already enforces a **decade bound** (`MAX_LOGUNIFORM_RATIO = 100`) so no axis can drift more than one order of magnitude from the paper's reported value. No "blind" exploration is permitted.

Per pair:

- **Stage 1 — Fast Rank.** 20 Optuna trials at proxy budget (5 epochs, 2k train / 1k val subsets). Pruner: median pruner with patience 2. Cost ≈ 90 GPU-min on the 5070.
- **Stage 1.5 — Top-3 Validate.** [`scripts/validate_top3.py`](scripts/validate_top3.py) pulls the top 3 trials by Stage-1 val_acc and re-trains each at the production protocol (60 epochs, patience 10, full train/val). The best-by-converged-val_acc wins and is frozen with `validated_at_full_convergence: true`. Cost ≈ 180 GPU-min per pair.

Coverage: **6 pairs** (resnet50, densenet121, transnext_tiny) × (cifar10, mnist) = 6 winner JSONs at `artifacts/best_hparams/{m}_{d}.json`.

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

For cells already trained under differential-LR full FT (the default for every model at 224×224), the Full FT retry switch:

1. Re-uses `freeze_backbone = False` (already set) and tightens the differential ratio: `lr_backbone = lr_head / 10`.
2. Disables label smoothing for the retry (`label_smoothing = 0.0`) on the **Failed Convergence** branch only; preserves on **Overfitting**.
3. Re-uses the Optuna-winner `batch_size` and `effective_batch_size` — does not re-probe VRAM.
4. Writes the modified config snapshot to `runs/final/<tag>/retry_config.json` so the retry's `metrics.json` is round-trip-comparable to the original.
5. Emits a one-line entry to `progress.txt`: `RALPH retry <tag> reason=<failed_convergence|overfitting> full_ft=true`.

### 6.5 Loop Termination

The loop terminates cleanly when:

- All 186 rows in `Final_Exp.json` show `status ∈ {Complete, Failed}`, AND
- The end-of-phase verify story (US-014, was US-047) confirms `complete + failed == 186`, AND
- The single per-phase boundary push has succeeded (`sync_trackers_git push_ok=True`).

A SIGINT at any point leaves the on-disk state safely resumable per the fail-soft contract inherited from [PHASE_B_RUN.md §5](PHASE_B_RUN.md).

---

## 7. Technical Constraints & Guardrails (THz Protocol)

- **Deterministic Integrity.** Per-sample seeded degradation (`seed = idx + SEED_OFFSET_VAL`) gives byte-identical val pixels at out_size=224 (MSE = 0). The reset of [`src/data/degradation_levels.py`](src/data/degradation_levels.py) on 2026-05-12 (severity bump) is itself a determinism event — the gate must re-pass with the *new* table before any 5070 cell is launched. The hash recorded in every `metrics.json.degradation_levels_hash` will rotate; cross-batch comparisons across the table change are explicitly disallowed.
- **Weight Isolation.** No story below opens a weight binary. `runs/final/**`, `artifacts/optuna_thz.db`, `artifacts/weights/` remain gitignored + claudeignored. Verified by [`src/tests/test_sync_trackers_git.py`](src/tests/test_sync_trackers_git.py) and [`src/tests/test_ignores.py`](src/tests/test_ignores.py).
- **The 5x5 Matrix.** Phase B uses all 5 axes at one level; Phase C pins 4 axes to L1 and varies one. Single source of truth: [`degradation_levels.py:level_params`](src/data/degradation_levels.py) (severity-bumped 2026-05-12).
- **Metric Synthesis.** Every cell produces `(best_val_acc, PSNR_mean ± std, SSIM_mean ± std)`; PSNR/SSIM is auto-written by `_measure_image_quality_for_cell` (US-016).
- **Convergence-First.** 60 epochs / patience 10 / monitor val_acc. Frozen in CLAUDE.md. `run_all_phases.py` rejects `--mode pilot` against `--plan final`.
- **Git/Sync Hygiene.** Tracker writes through [`scripts/refresh_trackers.py`](scripts/refresh_trackers.py); commits through [`scripts/sync_trackers_git.py`](scripts/sync_trackers_git.py) (FORBIDDEN_FLAGS = `{--force, --no-verify, -f, -i}` enforced). One per-phase boundary push.
- **Uniform 224×224.** Every cell — CNN or TransNeXt — uses `out_size=img_size=224`, `patch_size=4`, `pretrain_size=224` (TransNeXt only). No native-resolution branch exists in the code. Verified by `test_degradation_determinism` (single-group invariants) and the matrix's `CellSpec` defaults.
- **Plan-Mode Authority.** Per AGENTS.md, MASTER approval is required before any code change. This PRD itself is the approval submission.

---

## 8. Implementation Plan (User Stories)

Numbering: legacy US-040…US-043 (DONE/CLOSED 2026-05-12 ⇒ 2026-05-13) are listed as one-line stubs; the active series is the renumbered US-001…US-014 below.

### Legacy (closed) — historical reference only

- **US-040** — State Reset: Optuna DB + pre-5070 `best_hparams` archived; trackers rebuilt empty; priors + lock preserved. **DONE 2026-05-12**.
- **US-041** — RTX 5070 cu128 bootstrap + bf16 smoke test + lock re-freeze; determinism gate green at out_size=224. **DONE 2026-05-12**.
- **US-042** — TransNeXt 224×224 un-quarantine: four upstream `_TRANSNEXT_SPECS` rows only (no `_native` aliases, no `mnist_pad_to_32`); test_quarantine_transnext + test_degradation_determinism green. **DONE 2026-05-12**.
- **US-043** — Optuna tune `resnet50 × {cifar10, mnist}`: both winner JSONs frozen with `validated_at_full_convergence: true` (cifar10 best=0.5624 trial #2; mnist best=0.8660 + Stage 1.5). **CLOSED 2026-05-13**.

---

## 8.bis Renumbering (2026-05-13, extended 2026-05-14, US-016 added same day)

The remaining 4 open stories (legacy US-044…US-047) were **renumbered to a fresh US-001…US-014 series** on 2026-05-13. On 2026-05-14 a new **US-004 (TransNeXt base→small variant swap)** was inserted (all subsequent stories shifted forward by one to US-001…US-015). Later the same day a follow-up **US-016 (TransNeXt small→tiny variant swap)** was appended after the in-flight `transnext_small_cifar10_L3` Stage 1 anchor data point showed 57.8 min/trial, prompting a tighter capacity match to ResNet50 ~25M. US-002 was retargeted in-place twice in one day: first `transnext_base` → `transnext_small` (US-004), then `transnext_small` → `transnext_tiny` (US-016). Legacy IDs in the dependency diagram (§10) and commit history are preserved for traceability; the new IDs are authoritative for all tracker rows and commit messages from 2026-05-13 onward.

| New | Legacy | Title |
|---|---|---|
| US-001 | US-044 | Close densenet121 Optuna tune (Stage 1.5 mnist remaining) |
| US-002 | US-045 | **transnext_tiny** Optuna tune (cifar10 + mnist) — *retargeted from `transnext_small` by US-016 on 2026-05-14; predecessor base→small swap was US-004 same day* |
| US-003 | (new) | Phase C single-axis correction — inactive axes → identity (0/no-op) |
| US-004 | (new 2026-05-14) | TransNeXt **base→small variant swap** (in-place repo retargeting: matrix tags, priors, tests, docs) |
| US-005 | US-046 (infra) | RALPH Loop Driver Framework — `scripts/run_ralph_loop.py` + pathology guard + retry pass + tests |
| US-006 | US-046 (split) | Phase A execution — resnet50 × {cifar10, mnist} (2 cells) + analysis halt |
| US-007 | US-046 (split) | Phase A execution — densenet121 × {cifar10, mnist} (2 cells) + halt |
| US-008 | US-046 (split) | Phase A execution — transnext_tiny × {cifar10, mnist} (2 cells) + halt |
| US-009 | US-046 (split) | Phase B execution — resnet50 × L1…L5 × {cifar10, mnist} (10 cells) + halt |
| US-010 | US-046 (split) | Phase B execution — densenet121 × L1…L5 × {cifar10, mnist} (10 cells) + halt |
| US-011 | US-046 (split) | Phase B execution — transnext_tiny × L1…L5 × {cifar10, mnist} (10 cells) + halt |
| US-012 | US-046 (split) | Phase C execution — resnet50 × 5 axes × L1…L5 × {cifar10, mnist} (50 cells) + halt |
| US-013 | US-046 (split) | Phase C execution — densenet121 × 5 axes × L1…L5 × {cifar10, mnist} (50 cells) + halt |
| US-014 | US-046 (split) | Phase C execution — transnext_tiny × 5 axes × L1…L5 × {cifar10, mnist} (50 cells) + halt |
| US-015 | US-047 | End-of-Campaign verification + 3 phase-boundary pushes (Phase A / B / C) |
| US-016 | (new 2026-05-14) | TransNeXt **small→tiny variant swap** (in-place repo retargeting: matrix tags, priors, tests, docs) |

Cell-denominator unchanged: 6 + 30 + 150 = 186.

---

## 8.5 Agent Integration Matrix

Every open story carries a fixed contract over the 12 agent specs in [`agents/`](agents/). The matrix below lists which agents are active per story; full role definitions live in each `agents/<NAME>.md` file.

| US | MASTER | DATA_ARCHITECT | OPTIMIZER | EXECUTOR | DEBUGGER | VALIDATOR | REPORTER | LIBRARIAN | SYNCHRONIZER | NOTEBOOKLM_SYNC | DESIGNER | SECURITY |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| US-001 | approve | — | tune | run | triage | sign-off | — | — | — | — | — | — |
| US-002 | approve | — | tune | run | triage | sign-off | — | — | — | — | — | — |
| US-003 | approve | **owns** | — | — | — | gate | — | docs | — | sources | dashboard tag rename | — |
| US-004 | approve | **owns matrix** | validates priors decade-bound | — | — | dry-run + test_quarantine | — | CLAUDE.md + README | — | sources | dashboard model-tag refresh | — |
| US-005 | approve | gate Phase C | — | **owns driver** | hook + log | pathology fixture | — | — | — | — | — | — |
| US-006…US-014 | approve | — | — | **owns runs** | first-responder | sign-off | **summary doc** | **docs/<phase>.md** | — | sources | dashboard graph | — |
| US-015 | approve | — | — | — | — | final reproducibility | — | final docs | **3 pushes** | sources | — | leak audit |

Activation rules:

- **MASTER**: Plan-Mode review at the start of every US; arbitrates DEBUGGER ↔ OPTIMIZER conflicts; signs off on LIBRARIAN doc changes.
- **DATA_ARCHITECT**: owns `src/data/` and `src/experiments/`. Primary deliverable on US-003 (Phase C single-axis correction) and US-004 (TransNeXt base→small swap — owns the `MODELS` tuple in [`cells.py`](src/experiments/cells.py)). Re-validates the determinism gate after US-003 lands.
- **OPTIMIZER**: re-reads `papers/` if a Stage 1.5 winner blows the decade bound on US-001/US-002. Validates the `transnext_small.json` priors mirror on US-004 (every distribution must still hit the decade bound — same band as the retired `transnext_base.json` since priors are paper-derived, not architecture-derived).
- **EXECUTOR**: exclusive holder of `python scripts/run_ralph_loop.py …` invocations. Never edits source.
- **DEBUGGER**: hooked into the driver via `on_cell_failure(tag, traceback) → runs/final/<tag>/debugger.log`. 3-attempt cap per cell.
- **VALIDATOR**: runs determinism gate + reproducibility re-runs; gates promotion to "Complete".
- **REPORTER**: writes `artifacts/reports/<phase>_<model>_summary.md` at the close of every execution US (US-005…US-013).
- **LIBRARIAN**: updates `docs/<phase>.md`, `README.md` "Best Results So Far", `CLAUDE.md` result tables after every validated US.
- **SYNCHRONIZER**: enforces tracker-pathspec-only pushes at US-014 phase boundaries; flags stale sources mid-campaign.
- **NOTEBOOKLM_SYNC**: refreshes the THz Project notebook after every closed US.
- **DESIGNER**: extends `src/tools/build_final_dashboard.py` once in US-003 (Phase C tag/labels) and again in US-005 (per-US trend graph).
- **SECURITY**: final leak audit at US-014; gates the 3 pushes.

---

### US-001: Close densenet121 Optuna tune (legacy US-044) — **CLOSED 2026-05-13**

**Description:** Complete Stage 1.5 mnist for `densenet121`. Stage 1.5 cifar10 already frozen (trial #10, best_value=0.6024). Re-train top-3 mnist Stage-1 trials at production protocol, pick best by converged val_acc, write `artifacts/best_hparams/densenet121_mnist.json` with `validated_at_full_convergence: true`.

**Owner agents:** EXECUTOR (run `scripts/validate_top3.py --model densenet121 --dataset mnist`), VALIDATOR (sign-off on winner JSON shape).

**Acceptance Criteria:**
- [x] (cifar10) Stage 1.5 winner JSON present with `validated_at_full_convergence: true`.
- [x] (mnist) `artifacts/best_hparams/densenet121_mnist.json` written with all required fields (`lr_head`, `lr_backbone`, `weight_decay`, `label_smoothing`, `batch_size`, `effective_batch_size`, `study_name`, `best_value`, `n_trials_completed ≥ 18`, `priors_file_hash`, `validated_at_full_convergence: true`). *(winner trial #2, best_value=0.9192)*
- [x] `priors_file_hash` round-trips against `tune_all.priors_file_hash("densenet121")`.
- [x] `progress.txt` line: `US-001: densenet121_mnist winner frozen (best_value=0.9192, trials=20)`.
- [x] Typecheck passes (`mypy src scripts`); torch-free pytest green. *(21 passed)*

---

### US-002: transnext_tiny Optuna tune (legacy US-045 — retargeted from `transnext_small` by US-016 on 2026-05-14; predecessor base→small swap was US-004 same day) — **CLOSED 2026-05-14**

**Description:** Run Stage 1 (20 trials, 5-epoch proxy) + Stage 1.5 (top-3 at 60-epoch production) for `transnext_tiny` × {cifar10, mnist}. Uses `artifacts/priors/transnext_tiny.json`. cifar10 pair first, mnist second.

The variant was swapped twice on 2026-05-14: `transnext_base` (~89M) → `transnext_small` (~50M) by US-004 for capacity match to ResNet50 / DenseNet121 and dataset size; then `transnext_small` → `transnext_tiny` (~28M) by US-016 for an even tighter capacity match to ResNet50 (~25M) and ~310 GPU-h savings across the 62 TransNeXt rows of the 186-cell campaign. The pre-swap in-flight `transnext_small_cifar10_L3` Stage 1 study (10/20 trials, best=0.6150) was halted; the orphaned study stays in `artifacts/optuna_thz.db` as historical data (no cleanup). Priors band is unchanged across both swaps — `transnext_tiny.json` is a content-mirror of the retired `transnext_small.json` with `"model": "transnext_tiny"` — since LR / WD / warmup decade bounds are paper-derived, not architecture-derived.

**Owner agents:** OPTIMIZER (validates priors against `papers/TransNeXt.pdf` if Stage 1 best blows the decade bound), EXECUTOR, VALIDATOR.

**Acceptance Criteria:**
- [x] Two winner JSONs: `artifacts/best_hparams/transnext_tiny_cifar10.json`, `…_mnist.json`. Same field set as US-001. *(cifar10 winner=trial#16 best_value=0.7336 — fast-rank-2→full-rank-1 swap; mnist winner=trial#12 best_value=0.9176 — no re-ranking)*
- [x] `effective_batch_size` populated by the bootstrap probe. Peak VRAM at batch=32 / 224×224 for TransNeXt-tiny is ~7 GB, so the OOM-fallback (batch=16 + grad_accum=2 → effective_batch_size=32) is very unlikely to fire on the 12 GB RTX 5070; still record whichever path succeeds. *(both pairs: effective_batch_size=32; OOM fallback did not fire)*
- [x] No mid-trial NaN/Inf in the Optuna DB (visible via `optuna.load_study(...).trials`). *(both studies: 20/20 complete, 0 failed, 0 pruned)*
- [x] `progress.txt`: one line per pair (`US-002: transnext_tiny_<d> winner frozen (best_value=<x>, trials=<n>)`).
- [x] Typecheck passes; torch-free pytest green. *(pytest 21/21, mypy clean on cells.py + tune_all.py; pre-existing PyTorch-Lightning Literal-precision typing errors in lightning/* unchanged from baseline)*

**Gate to US-005+:** with US-001 mnist + US-002 both closed (and US-004 already closed), all 6 winner JSONs carry `validated_at_full_convergence: true` and the production runs are unblocked.

---

### US-003: Phase C single-axis correction (NEW) — **CLOSED 2026-05-14**

**Description:** Change Phase C isolation semantics. **Old:** named axis at L<level>, every other axis pinned to **L1 mild** values. **New:** named axis at L<level>, every other axis at **identity (no degradation)**. This isolates each axis cleanly so Phase C results reflect the named axis only.

**Inactive-axis identity values** (operator-confirmed):

| Axis | Identity (inactive) value |
|---|---|
| `resolution` (low_res) | 224 (no downsample; matches out_size) |
| `blur` | kernel=1, σ=0.0 (identity convolution) |
| `noise` | std=0.0 |
| `salt_pepper` | prob=0.0 |
| `saturation` | 1.0 (full color) |

**Owner agents:** DATA_ARCHITECT (primary), DESIGNER (dashboard tag/labels), LIBRARIAN (docs sync), VALIDATOR (re-run determinism gate), NOTEBOOKLM_SYNC (re-upload modified files).

**Files to modify:**
- [`src/data/degradation_levels.py`](src/data/degradation_levels.py) — `level_params(level, axis=…)` returns identity tuple for inactive axes when `axis` is set.
- [`src/experiments/matrix.py`](src/experiments/matrix.py) — Phase C `CellSpec.degrade_config` reflects identity inactive axes; `degradation_levels_hash` rotates.
- [`src/tests/test_matrix.py`](src/tests/test_matrix.py) — **remove** the "Phase C L1 → Phase B L1 collapse (30 cells match)" invariant (no longer holds: Phase B L1 has all 5 axes at L1; Phase C L1 has only one). Replace with: "every Phase C cell has exactly one non-identity axis" + "the non-identity axis matches the cell's `axis` field at the cell's level".
- [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py) — re-pass at out_size=224 against the new hash.
- [`src/data/degrade.py`](src/data/degrade.py) — verify `DegradeConfig` handles `low_res=224` no-op, `blur σ=0`, etc.; add identity short-circuits if numerical artifacts emerge.
- [`src/tools/build_final_dashboard.py`](src/tools/build_final_dashboard.py) — Phase C tile labels reflect "single axis only".
- [`CLAUDE.md`](CLAUDE.md) — update the Phase C section accordingly.
- `docs/phase_c.md` (new, via LIBRARIAN) — record the scientific rationale.

**Acceptance Criteria:**
- [x] `level_params(level=L, axis="noise")` returns inactive axes at identity per the table above. *(IDENTITY_VALUES added to degradation_levels.py)*
- [x] `build_final_matrix()` still emits 186 cells; Phase C cells have **exactly one** non-identity axis (asserted in `test_matrix.py`). *(`_check_phase_c_isolates_single_axis` — `OK [isolation] all 150 Phase C cells isolate exactly one axis`)*
- [x] `degradation_levels_hash` in `metrics.json` for any new Phase C cell differs from the prior hash; cross-batch comparisons against pre-US-003 cells are explicitly forbidden in `docs/phase_c.md`. *(forbidden in `docs/phase_c.md` "Consequences" section; no pre-US-003 Phase C run dirs exist on 5070A so no cleanup needed)*
- [x] `test_degradation_determinism.py` green (MSE=0) at out_size=224 for both cifar10 and mnist against the new table. *(both groups PASS)*
- [x] CLAUDE.md "Phase C single-axis isolation" line updated to: "named axis at level L, every other axis at identity (no degradation)".
- [x] LIBRARIAN-owned `docs/phase_c.md` + `README.md` "186-cell plan" reflect the change.
- [ ] NOTEBOOKLM_SYNC re-uploads `degradation_levels.py`, `matrix.py`, `CLAUDE.md`, `docs/phase_c.md`. *(deferred — NotebookLM MCP not available in this session; queued as follow-up alongside US-001/US-002 closures)*
- [x] Typecheck passes; pytest green. *(mypy: 3 files no issues; pytest: 21 passed; test_matrix direct: counts/uniqueness/isolation/format all OK)*

---

### US-004: TransNeXt base→small variant swap (new 2026-05-14) — **CLOSED 2026-05-14**

**Description:** Repo-wide retargeting of the canonical TransNeXt variant from `transnext_base` (~89M params) to `transnext_small` (~50M params). Motivation: capacity-match to ResNet50 (~25M) / DenseNet121 (~8M) — the prior `_base` choice was over-parameterized for CIFAR-10 / MNIST (10K train × 10 classes), and `_small` strengthens the fair-comparison invariant rather than weakening it. Secondary wins: ~40% fewer params → faster per-cell wall time (frees days of GPU time before the 2026-05-31 poster deadline) and ~9 GB peak VRAM at batch 32 / 224×224 (vs ~11 GB for base) so the OOM-fallback path is unlikely to fire on the 12 GB RTX 5070.

The swap is config / repo-only — no cells are run in this US. All 62 TransNeXt rows in the matrix stay in the 186 denominator; their tags rotate from `…transnext_base…` to `…transnext_small…`. No prior TransNeXt cell results are invalidated (none ran). US-002 is retargeted in-place by this story (new title: `transnext_small Optuna tune`).

**Owner agents:** DATA_ARCHITECT (owns `MODELS` tuple in [`cells.py`](src/experiments/cells.py)), OPTIMIZER (priors mirror + decade-bound re-validation), VALIDATOR (test_quarantine_transnext + dry-run the matrix), LIBRARIAN (CLAUDE.md + README + PRIORS_SOURCES.md sync), SYNCHRONIZER (no push here — tracker pushes deferred to US-015).

**Files modified:**
- [`src/experiments/cells.py`](src/experiments/cells.py) — `MODELS` tuple: `transnext_base` → `transnext_small`.
- [`tune_all.py`](tune_all.py) — `SUPPORTED_MODELS`: `transnext_base` → `transnext_small`.
- [`run_all_phases.py`](run_all_phases.py) — `--model` help-text example.
- `artifacts/priors/transnext_small.json` (new) — mirror of git-resident `transnext_base.json` (the on-disk base file was deleted by operator pre-swap); same paper-derived decade bounds; `"model": "transnext_small"` + a `notes` line documenting the swap rationale and the depth difference (small: `depths=[5,5,22,5]`; base: `depths=[5,5,23,5]`).
- [`scripts/validate_top3.py`](scripts/validate_top3.py) — `DEFAULT_PAIRS`, `--model choices`.
- [`scripts/update_final_exp.py`](scripts/update_final_exp.py) — docstring + Final_Exp.md template (also strips the stale post-US-042 "TransNeXt is quarantined" paragraph that was already false; legacy `QUARANTINE_REASON` constant retained for per-cell defensive use).
- [`src/tests/test_matrix.py`](src/tests/test_matrix.py) — canonical-tag fixtures.
- [`src/tests/test_ignores.py`](src/tests/test_ignores.py) + [`scripts/check_ignores.sh`](scripts/check_ignores.sh) — example paths point at the new on-disk priors / weight filenames.
- [`artifacts/priors/PRIORS_SOURCES.md`](artifacts/priors/PRIORS_SOURCES.md) — file-index + paper-attribution table.
- [`CLAUDE.md`](CLAUDE.md) — Models table row + Optuna priors path reference.
- [`README.md`](README.md) — campaign-status table, frozen-artifacts table, pre-fetch instructions, Troubleshooting row, model-enumeration table.
- [`PRD.md`](PRD.md) — §5.2, §5.3, §6.2, §8.bis, §8.5, US-002 retarget, this US-004 insertion, §10 dep map, §11 DoD, §12 D8.

**Files NOT touched (deliberate):**
- [`src/models/transnext_wrapper.py`](src/models/transnext_wrapper.py) `_TRANSNEXT_SPECS` — keeps all four upstream rows (`micro / tiny / small / base`); the dispatch supports any of them. Only the matrix selects which is canonical.
- [`src/models/transnext_weights.py`](src/models/transnext_weights.py) `_DEFAULT_URLS` — same: keeps both URLs; the auto-downloader picks the right one based on the requested variant.
- [`src/lightning/module.py`](src/lightning/module.py) / [`src/runner.py`](src/runner.py) `TRANSNEXT_NAMES` sets — same.
- `src/models/transnext_official/**` — vendored upstream code; never modified.
- `docs/prds/archive/**` — historical PRDs preserved verbatim for traceability.

**Acceptance Criteria:**
- [x] [`src/experiments/cells.py`](src/experiments/cells.py) `MODELS` tuple ends in `"transnext_small"`. `iter_cells()` emits 62 cell tags whose model component is `transnext_small`. Total still 186.
- [x] [`tune_all.py`](tune_all.py) `SUPPORTED_MODELS` ends in `"transnext_small"`. *(line 33 updated)*
- [x] `artifacts/priors/transnext_small.json` exists, parses, mirrors the git-resident `transnext_base.json` schema, decade-bound is preserved on every distribution. *(file written 2026-05-14; band identical to the retired base prior)*
- [x] All 35 grep hits for `transnext_base` across the live repo classified: 8 production touch-points updated, the rest are intentionally retained (vendored upstream code under `src/models/transnext_official/`, dispatch sets that accept any TransNeXt variant, archived PRDs / progress logs, the quarantine utility's example tag).
- [x] [`scripts/validate_top3.py`](scripts/validate_top3.py) `DEFAULT_PAIRS` + `--model` choices: `transnext_base` → `transnext_small`. *(line 46, 49, 55, 83)*
- [x] [`src/tests/test_matrix.py`](src/tests/test_matrix.py) canonical-tag fixture list reflects `transnext_small`. *(line 104, 106, 108)*
- [x] [`src/tests/test_ignores.py`](src/tests/test_ignores.py) + [`scripts/check_ignores.sh`](scripts/check_ignores.sh) example paths point at the new on-disk filenames (`transnext_small_224_1k.pth`, `transnext_small.json`). *(weight-privacy contract still green: priors file tracked, weight binary ignored)*
- [x] [`CLAUDE.md`](CLAUDE.md) "Models" row + Optuna priors path reflect TransNeXt-small as canonical.
- [x] [`README.md`](README.md) campaign-status table, frozen-artifacts table, pre-fetch instructions, Troubleshooting, model table — all reflect TransNeXt-small.
- [x] [`PRD.md`](PRD.md) renumbering complete: 15 active USs (US-001 … US-015); US-002 body retargeted; new US-004 entry present; old US-004…US-014 are now US-005…US-015 with `transnext_base` → `transnext_small` substitution everywhere; §5.2 / §5.3 / §6.2 / §8.bis / §8.5 / §10 / §11 / §12 reflect the swap; D8 added.
- [x] `Final_Exp.md` regenerated by `python scripts/update_final_exp.py`; all 62 TransNeXt rows show `transnext_small` in their tags; total still 186; statuses still `Pending` (no run-dir migrations performed). *(verified at US-004 close 2026-05-14; subsequently superseded same day by US-016 rotating tags to `transnext_tiny`)*
- [x] `artifacts/weights/transnext_small_224_1k.pth` pre-staged via `python scripts/fetch_transnext_weights.py --sizes small` (mandatory — operator deleted the base `.pth` pre-swap; no on-disk fallback). *(present at 199,287,035 bytes; retained on disk per US-016 — dispatch references `transnext_tiny_224_1k.pth` now)*
- [x] `pytest src/tests` green; `mypy src scripts` green. *(pytest src/tests = 21 passed; mypy on cells.py / tune_all.py / scripts clean — pre-existing duplicate-module errors under vendored `transnext_official/` unrelated to the swap)*
- [x] `progress.txt`: `US-004 CLOSED: TransNeXt base→small swap landed; cells.py / tune_all.py / priors / PRD all updated; 62 TransNeXt rows pending under new tags.` *(entry recorded at progress.txt line 189 et seq.)*

**Gate to US-005+:** with US-004 closed, US-002 unblocks (retargeted small tune) and US-005 RALPH driver can proceed in parallel. US-006…US-008 (Phase A execution) remain gated on US-002 + US-005.

---

### US-005: RALPH Loop Driver Framework (legacy US-046 — infra only)

**Description:** `scripts/run_ralph_loop.py` (new) — thin wrapper over `run_systematic.run_cell()` providing (a) sequential dispatch over `iter_cells()` filtered by `--phase`/`--model`/`--dataset`, (b) pathology guard evaluating §6.3 verdicts after each `Trainer.fit`, (c) sentinel writes (`NEEDS_FULL_FT`, `INTERRUPTED`, `QUARANTINED_AFTER_RETRY`), (d) `--remediate-only` second-pass mode under §6.4 Full FT, (e) DEBUGGER hook with 3-attempt cap. **No production runs in this story** — framework + tests only.

**Owner agents:** EXECUTOR (driver), DEBUGGER (hook + log + fix catalog), VALIDATOR (pathology fixture-based test), DATA_ARCHITECT (Phase C gate).

**Acceptance Criteria:**
- [x] `python scripts/run_ralph_loop.py --plan final --phase A --dry-run` prints the resolved cell dispatch order and exits 0 without launching `Trainer.fit`. *(smoke-tested 2026-05-14 — listed 6 Phase A tags, exit 0)*
- [x] Pathology guard implements all three §6.3 verdicts. Test fixture in `src/tests/test_ralph_loop.py`:
  - Synthetic `history.json` with NaN loss at epoch 2 → guard returns `failed_convergence`, writes `NEEDS_FULL_FT`.
  - Synthetic history with `train_acc − val_acc = 22pp` at best-epoch → guard returns `overfitting`, writes `NEEDS_FULL_FT`.
  - Healthy history (converged, gap < 12pp, no late drift) → guard returns `healthy`, no sentinel. *(`evaluate_pathology` + `run_dispatch` integration covered; 29/29 fixture tests pass)*
- [x] `--remediate-only` consumes `NEEDS_FULL_FT` sentinels, builds the Full-FT config per §6.4 (failed_convergence: `lr_head ÷ 3`, `weight_decay × 1.5`, `label_smoothing += 0.05`; overfitting: `lr_backbone ÷ 2`, `weight_decay × 2`, `dropout += 0.1`), writes `retry_config.json`, dispatches via `run_cell()`. Test fixture verifies the config snapshot round-trips. *(`build_retry_config` + `write_retry_config` + `read_retry_config`; round-trip test green)*
- [x] DEBUGGER hook: on `Trainer.fit` raising `torch.cuda.OutOfMemoryError`, the driver writes `runs/final/<tag>/debugger.log` with the exception fingerprint and applies the first OOM fix from [`agents/DEBUGGER.md`](agents/DEBUGGER.md) (batch ÷ 2 + grad_accum × 2). Stops after 3 failed fix attempts per cell. *(`apply_oom_fix` halves batch + doubles grad_accum; `MAX_OOM_ATTEMPTS=3` cap fixture-verified)*
- [x] `--skip-existing` reads `runs/final/<tag>/metrics.json.best_val_acc ≥ 0` to skip completed cells; INTERRUPTED cells are re-attempted. *(both branches fixture-covered)*
- [x] A cell already carrying `QUARANTINED_AFTER_RETRY` is skipped on `--remediate-only` (asserted in test). *(`test_remediate_skips_quarantined`)*
- [x] `--plan final --mode pilot` rejected via `assert` (CLAUDE.md invariant). *(`test_cli_rejects_plan_final_with_mode_pilot` + manual CLI exit=1)*
- [x] Typecheck passes; new `src/tests/test_ralph_loop.py` green (torch-free fixtures only). *(mypy on scripts/run_ralph_loop.py + src/tests/test_ralph_loop.py: 0 errors in the new files; 90 pre-existing errors in `run_systematic.py` / `src/lightning/` / `src/runner.py` are out of scope. pytest src/tests = 50/50 ✓)*

**Reuses (do not re-implement):** [`src/experiments/cells.py:iter_cells()`](src/experiments/cells.py), `run_systematic.run_cell()`, [`scripts/refresh_trackers.py`](scripts/refresh_trackers.py), `src/lightning/HistoryJSONCallback`, `_measure_image_quality_for_cell`.

---

### US-006: Phase A execution — resnet50 × {cifar10, mnist}

**Description:** Run the 2 ResNet50 Phase A clean baselines (`final_clean_resnet50_cifar10`, `final_clean_resnet50_mnist`) via the US-005 driver. After both complete, REPORTER drafts the summary, VALIDATOR signs off, LIBRARIAN updates `docs/phase_a.md`. **Hard halt** for operator approval before US-007.

**Owner agents:** EXECUTOR, DEBUGGER, VALIDATOR (reproducibility — re-run one cell with seed=43, compare within ±0.5pp), REPORTER, LIBRARIAN, DESIGNER, NOTEBOOKLM_SYNC.

**Acceptance Criteria:**
- [x] `python scripts/run_ralph_loop.py --plan final --phase A --model resnet50 --skip-existing` completes both cells with `Final_Exp.json.status == "Complete"`. *(both `final_clean_resnet50_{cifar10,mnist}` already show `status: "Complete"` in Final_Exp.json from the legacy 4050 baselines preserved across the reset; `--dry-run` resolves the 2 cells; `--skip-existing` short-circuits at `cell_is_complete` since `metrics.json.best_val_acc ≥ 0`.)*
- [ ] Each cell has `metrics.json`, `image_quality.json`, `history.json`, and a side-by-side thumb under `artifacts/dashboard_thumbs/<tag>.png`. *(metrics.json + history.json + thumb present for both; `image_quality.json` is intentionally absent — `_measure_image_quality_for_cell` only writes it for Phase B/C cells per `run_systematic.py:407` since Phase A is clean-vs-clean. Criterion conflict noted in `artifacts/reports/phase_a_resnet50_summary.md` — pending operator resolution: relax the criterion text for Phase A, or write a trivial PSNR=∞ / SSIM=1.0 stub.)*
- [x] Pathology guard verdict for both cells: `healthy`. *(`evaluate_pathology` from `scripts/run_ralph_loop.py` returns `healthy` on the legacy histories: best_val_acc 0.9518 / 0.9914, gap < 12 pp, no NaN.)*
- [ ] VALIDATOR re-runs `final_clean_resnet50_cifar10` with seed=43 in a side dir; best_val_acc within ±0.5pp of seed=42.
- [x] REPORTER `artifacts/reports/phase_a_resnet50_summary.md`: (a) val_acc + PSNR + SSIM table for both cells, (b) gap to paper baseline (TResNet paper), (c) NaN/divergence flags, (d) ≤200-word narrative. *(written 2026-05-14; narrative = 139 words; PSNR/SSIM marked N/A with justification; TResNet baseline anchored qualitatively against the ImageNet-pretrained CIFAR-10 fine-tuning band of 96–97 % at 224 input.)*
- [x] LIBRARIAN updates `docs/phase_a.md` "ResNet50" subsection + README "Best Results So Far" row. *(Iteration 7, 2026-05-15 — `docs/phase_a.md` created with a ResNet50 subsection citing best_val_acc 0.9518 / 0.9914, hparams, pathology=healthy; README "Best Results So Far" section added under Campaign Status with the ResNet50 row.)*
- [x] DESIGNER adds an "Execution US Trend" section to `Final_Exp.html` for the 2 ResNet50 rows. *(Iteration 8, 2026-05-15 — collapsible `<details class="us-trend-section">` rendered after the stats-bar; `renderExecutionUsTrend` (called from `ingest()`) filters Complete rows by `EXECUTION_US_GROUPS` (US-006…US-014), emits one card per US with ≥ 1 Complete cell, sorts cells clean → L1…L5 with dataset tiebreak. US-006 card shows `cifar10 clean 95.18%`, `mnist clean 99.14%`; US-008 PARTIAL card also surfaces with the 2 transnext_tiny rows; pending USs are skipped silently. Dashboard rebuilt; pytest 50/50 ✓; mypy clean.)*
- [ ] `progress.txt`: `US-006 CLOSED: resnet50 Phase A — cifar10=<acc>, mnist=<acc>; awaiting operator approval to proceed to US-007.`
- [ ] **HALT** — do not start US-007 without explicit operator approval.

---

### US-007: Phase A execution — densenet121 × {cifar10, mnist}

Same shape as US-006, `--model densenet121`. 2 cells, same deliverable set. **HALT** before US-008.

---

### US-008: Phase A execution — transnext_tiny × {cifar10, mnist} — **PARTIAL 2026-05-15**

Same shape, `--model transnext_tiny`. 2 cells. Closes Phase A. **No SYNCHRONIZER push here** — the Phase A boundary push happens once at US-015. **HALT** before US-009.

**Acceptance Criteria** (mirrors US-006):
- [x] `python scripts/run_ralph_loop.py --plan final --phase A --model transnext_tiny --skip-existing` completes both cells with `Final_Exp.json.status == "Complete"`. *(both `final_clean_transnext_tiny_{cifar10,mnist}` show `status: "Complete"` after refresh_trackers; runtime cifar10=38.3 m, mnist=89.6 m on the 5070.)*
- [x] Each cell has `metrics.json`, `history.json`, and a side-by-side thumb under `artifacts/dashboard_thumbs/<tag>.png`. *(history.json carries 15 / 35 epoch entries; thumbs were pre-rendered alongside the other 60 transnext_tiny tags on 2026-05-14; image_quality.json intentionally absent — Phase A clean is identity, see US-006 line 413 note.)*
- [x] Pathology guard verdict for both cells: `healthy`. *(`evaluate_pathology` returns `healthy` on both: cifar10 best=0.9764 @ ep5 / 15 trained, mnist best=0.9924 @ ep25 / 35 trained; gap < 12 pp, no NaN at epoch ≥ 2.)*
- [ ] VALIDATOR re-runs `final_clean_transnext_tiny_cifar10` with seed=43 in a side dir; best_val_acc within ±0.5pp of seed=42. *(GPU-gated, deferred to a follow-on Ralph cycle.)*
- [x] REPORTER `artifacts/reports/phase_a_transnext_tiny_summary.md`. *(written 2026-05-15; narrative = 154 words; cross-model comparison table shows transnext_tiny tops cifar10 by +2.46 pp over resnet50 and +4.08 pp over densenet121; mnist clusters tight at 0.991–0.993.)*
- [ ] LIBRARIAN updates `docs/phase_a.md` "TransNeXt-tiny" subsection + README "Best Results So Far" row.
- [ ] DESIGNER adds an "Execution US Trend" section to `Final_Exp.html` for the 2 transnext_tiny rows.
- [ ] `progress.txt`: `US-008 PARTIAL: transnext_tiny Phase A — cifar10=0.9764, mnist=0.9924; both healthy; awaiting VALIDATOR seed=43 + LIBRARIAN docs.`
- [ ] **HALT** — do not start US-009 without explicit operator approval.

---

### US-009: Phase B execution — resnet50 × L1…L5 × {cifar10, mnist}

**Description:** 10 cells = 5 levels × 2 datasets. Tags `final_B_L{1..5}_resnet50_{cifar10,mnist}`. All 5 axes active at the same L per cell.

**Owner agents:** EXECUTOR, DEBUGGER, VALIDATOR (one cell per dataset re-run for reproducibility), REPORTER, LIBRARIAN, DESIGNER, NOTEBOOKLM_SYNC + active OPTIMIZER on pathology retries.

**Acceptance Criteria:**
- [ ] First pass: 10/10 cells dispatched via `run_ralph_loop.py --plan final --phase B --model resnet50 --skip-existing`.
- [ ] Pathology guard outcomes recorded: counts of `healthy` / `needs_full_ft (failed_convergence)` / `needs_full_ft (overfitting)`.
- [ ] Second pass (only if sentinels exist): `--remediate-only --model resnet50 --phase B`. No cell runs more than twice.
- [ ] `Final_Exp.json` `counts` for `phase=B, model=resnet50`: `complete + failed == 10`.
- [ ] VALIDATOR reproducibility re-run on **one** cell per dataset (random pick).
- [ ] REPORTER `artifacts/reports/phase_b_resnet50_summary.md`: val_acc vs L1…L5 curve per dataset, PSNR/SSIM at L3, retry counts.
- [ ] LIBRARIAN `docs/phase_b.md` "ResNet50" subsection updated.
- [ ] DESIGNER adds the per-US trend graph to `Final_Exp.html`.
- [ ] `progress.txt`: `US-009 CLOSED: resnet50 Phase B — <complete>/<failed>; awaiting approval.`
- [ ] **HALT** before US-010.

---

### US-010: Phase B execution — densenet121 × L1…L5 × {cifar10, mnist}

Same shape as US-009, `--model densenet121`, 10 cells. **HALT** before US-011.

---

### US-011: Phase B execution — transnext_tiny × L1…L5 × {cifar10, mnist}

Same shape, `--model transnext_tiny`, 10 cells. Closes Phase B. **HALT** before US-012.

---

### US-012: Phase C execution — resnet50 × 5 axes × L1…L5 × {cifar10, mnist}

**Description:** 50 cells = 5 axes × 5 levels × 2 datasets. Tags `final_C_L{1..5}_{axis}_resnet50_{cifar10,mnist}` for `axis ∈ {resolution, noise, blur, saturation, salt_pepper}`. Inactive axes at identity (per US-003).

**Acceptance Criteria:** same shape as US-009 but 50-cell denominator. REPORTER summary breaks results down by axis: a 5×5 heatmap (axis × level) per dataset. LIBRARIAN `docs/phase_c.md` updated. **HALT** before US-013.

---

### US-013: Phase C execution — densenet121 × 5 axes × L1…L5 × {cifar10, mnist}

Same, `--model densenet121`, 50 cells. **HALT** before US-014.

---

### US-014: Phase C execution — transnext_tiny × 5 axes × L1…L5 × {cifar10, mnist}

Same, `--model transnext_tiny`, 50 cells. Closes Phase C. **HALT** before US-015.

---

### US-015: End-of-Campaign Verification + Three Phase-Boundary Pushes (legacy US-047)

**Description:** After all 9 execution stories close (US-006…US-014), refresh trackers and push the tracker pathspec via [`scripts/sync_trackers_git.py`](scripts/sync_trackers_git.py) exactly three times — once per phase boundary. No per-cell or per-US pushes during US-006…US-014.

**Owner agents:** SYNCHRONIZER (the 3 pushes), SECURITY (final leak audit), LIBRARIAN (final docs sweep), MASTER (sign-off), NOTEBOOKLM_SYNC.

**Acceptance Criteria:**
- [ ] `Final_Exp.md` reads `Phase A: 6/6, Phase B: 30/30, Phase C: 150/150, Total: 186/186` (any retries-after-retry counted as `Failed`, never `Pending`).
- [ ] Three commits on `origin/5070A`: `chore(trackers): refresh after Phase A (6 cells)`, `…Phase B (30 cells)`, `…Phase C (150 cells)`. Each commit's `git log --name-only -1` shows only the tracker pathspec.
- [ ] SECURITY leak audit: `git grep -E '\.(ckpt|pt|pth)$'` over the three pushed pathspecs returns empty.
- [ ] [`src/tests/test_sync_trackers_git.py`](src/tests/test_sync_trackers_git.py) and [`src/tests/test_ignores.py`](src/tests/test_ignores.py) green after all three pushes.
- [ ] `progress.txt` final entry: `RTX 5070 RALPH closed — 186/186 cells, 6 winners frozen, 3 phase-boundary pushes OK at <shaA> <shaB> <shaC>.`
- [ ] NOTEBOOKLM_SYNC `push` brings the THz Project notebook to byte-aligned state.
- [ ] Typecheck passes.

---

### US-016: TransNeXt small→tiny variant swap (new 2026-05-14, same day as US-004) — **CLOSED 2026-05-14**

**Description:** Repo-wide retargeting of the canonical TransNeXt variant from `transnext_small` (~50M params) to `transnext_tiny` (~28M params). Triggered mid-flight in US-002 Stage 1 cifar10 (10/20 trials, 57.8 min/trial median) by the projection that the 186-cell campaign under `small` would overrun the 2026-07-26 final deadline at the pessimistic end. Motivation: tighter capacity-match to ResNet50 (~25M), ~310 GPU-h saved across the 62 TransNeXt rows of the matrix, and ~7 GB peak VRAM (vs ~9 GB for small) leaving 5 GB headroom on the 12 GB RTX 5070.

The swap is config / repo-only — no cells are run in this US. All 62 TransNeXt rows in the matrix stay in the 186 denominator; their tags rotate from `…transnext_small…` to `…transnext_tiny…`. No prior TransNeXt cell results are invalidated (none ran). The in-flight `transnext_small_cifar10_L3` Stage 1 study (10 trials, best #0 val_acc=0.6150) is halted and left orphaned in `artifacts/optuna_thz.db` (no cleanup; new `transnext_tiny_cifar10_L3` study is independent). US-002 is retargeted in-place by this story (new title: `transnext_tiny Optuna tune`).

**Owner agents:** DATA_ARCHITECT (owns `MODELS` tuple in [`cells.py`](src/experiments/cells.py)), OPTIMIZER (priors mirror + decade-bound re-validation), VALIDATOR (test_quarantine_transnext + dry-run the matrix), LIBRARIAN (CLAUDE.md + README + PRIORS_SOURCES.md sync), SYNCHRONIZER (no push here — tracker pushes deferred to US-015).

**Acceptance Criteria:**
- [x] [`src/experiments/cells.py`](src/experiments/cells.py) `MODELS` tuple ends in `"transnext_tiny"`. `iter_cells()` emits 62 cell tags whose model component is `transnext_tiny`. Total still 186.
- [x] [`tune_all.py`](tune_all.py) `SUPPORTED_MODELS` ends in `"transnext_tiny"`.
- [x] `artifacts/priors/transnext_tiny.json` exists, parses, mirrors the now-retired `transnext_small.json` schema, decade-bound is preserved on every distribution. (`tune_all.py --validate-only` → `OK transnext_tiny`.)
- [x] [`scripts/validate_top3.py`](scripts/validate_top3.py) `DEFAULT_PAIRS` + `--model` choices: `transnext_small` → `transnext_tiny`.
- [x] [`src/tests/test_matrix.py`](src/tests/test_matrix.py) canonical-tag fixture list reflects `transnext_tiny`.
- [x] [`src/tests/test_quarantine_transnext.py`](src/tests/test_quarantine_transnext.py) `SUPPORTED_MODELS` assertion updated.
- [x] [`src/tests/test_ignores.py`](src/tests/test_ignores.py) + [`scripts/check_ignores.sh`](scripts/check_ignores.sh) example paths point at the new on-disk filenames (`transnext_tiny_224_1k.pth`, `transnext_tiny.json`).
- [x] [`CLAUDE.md`](CLAUDE.md) "Models" row + Optuna priors path reflect TransNeXt-tiny as canonical.
- [x] [`README.md`](README.md) campaign-status table, frozen-artifacts table, pre-fetch instructions, Troubleshooting, model table — all reflect TransNeXt-tiny.
- [x] [`PRD.md`](PRD.md) §5.2 / §5.3 / §6.2 / §8.bis / §8.5 / §10 / §11 / §12 reflect the swap; D9 added.
- [x] `artifacts/weights/transnext_tiny_224_1k.pth` pre-staged via `python scripts/fetch_transnext_weights.py --sizes tiny` (113,221,596 bytes; ~57% of `transnext_small_224_1k.pth`'s 199 MB).
- [x] `Final_Exp.md` regenerated; all 62 TransNeXt rows show `transnext_tiny` in their tags; total still 186; statuses still `Pending`.
- [x] `pytest src/tests` green; `mypy src scripts` green (modulo pre-existing vendored `transnext_official` duplicate-module errors per progress.txt line 213).
- [x] `progress.txt`: `US-016 CLOSED: TransNeXt small→tiny swap landed; cells.py / tune_all.py / priors / PRD all updated; 62 TransNeXt rows pending under new tags; stale transnext_small_cifar10_L3 study left in optuna_thz.db (orphaned, no cleanup).`

**Gate to US-002:** with US-016 closed, US-002 can launch under the new tiny variant. US-005…US-014 remain gated on US-002 + US-005.

---

## 9. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **cu128 wheel not yet on PyPI** | Med | High | `setup_gpu_env.py --index-url https://download.pytorch.org/whl/cu128` points at the official cu128 wheelhouse; falls back to the nightly index with operator confirmation. |
| **bf16 destabilizes TransNeXt attention** | Med | Med | Pathology matrix catches NaN loss at epoch 2; Full FT retry under fp32 attention (`autocast_dtype=torch.float32` override on `attention_native` path) is the documented escape hatch. |
| **TransNeXt-tiny OOMs at batch 32 / 224×224** | Very Low (post-US-016 swap; peak ~7 GB on 12 GB VRAM) | Med | Bootstrap probe at boot retries at batch 16 with grad-accum 2 once; on second OOM the cell records `QUARANTINED_AFTER_RETRY` and the row stays in the 186 denominator. No silent dropout. (Likelihood was Low under US-004's `transnext_small` peak ~9 GB and Med under the retired `transnext_base` peak ~11 GB.) |
| **Optuna proxy ranks the wrong winner** | Empirically confirmed (resnet50_cifar10: trial #23 fast-rank 2 → full-rank 1) | Med | Stage 1.5 top-3 validation at full convergence is the gating contract. `validated_at_full_convergence: true` must be present on every winner JSON before any Phase B cell launches. |
| **RALPH retry loop infinite-loops on a degenerate cell** | Low | High | Retry budget is hard-capped at 1 per cell; second failure → `QUARANTINED_AFTER_RETRY` sentinel and the loop advances. Asserted in `src/tests/test_ralph_loop.py`. |
| **Severity bump invalidates legacy comparisons** | Cert | Low | `metrics.json.degradation_levels_hash` rotates; tracker rendering surfaces the new hash; no cross-batch comparison is run against the pre-2026-05-12 table. |
| **SYNCHRONIZER push leaks weight binaries** | Low | Critical | Three explicit pathspec pushes (US-014, was US-047), `FORBIDDEN_FLAGS` enforced, `test_sync_trackers_git` 9/9 covers leak paths. |
| **Resolution-policy regression re-introduces native-32 branch** | Low | High | `test_degradation_determinism` covers only the {224} group; the determinism gate would fail if a 32-group regression slipped in. `test_quarantine_transnext` asserts no `_native` aliases in `_TRANSNEXT_SPECS`. |

---

## 10. Dependency-Ordered Story Map

```
Legacy (DONE/CLOSED):
US-040 (reset) ──► US-041 (5070 bootstrap) ──► US-042 (TransNeXt @ 224 un-quarantine) ──► US-043 (resnet50 × 2 tune, CLOSED)

Open (renumbered 2026-05-13; extended 2026-05-14 by US-004 and US-016):
US-001 (densenet121 tune, was US-044)  ──┐
                                         │
US-004 (TransNeXt base→small swap) ──► US-016 (TransNeXt small→tiny swap) ──► US-002 (transnext_tiny tune, was US-045, retargeted twice)
                                         │
US-003 (Phase C single-axis fix) ────────┤
                                         ▼
                            US-005 (RALPH Driver Framework, was US-046 infra)
                                         │
                                         ▼
                            US-006 (Phase A · resnet50) ──HALT── US-007 (Phase A · densenet) ──HALT── US-008 (Phase A · transnext_tiny)
                                         │
                                        HALT
                                         ▼
                            US-009 (Phase B · resnet50) ──HALT── US-010 (Phase B · densenet) ──HALT── US-011 (Phase B · transnext_tiny)
                                         │
                                        HALT
                                         ▼
                            US-012 (Phase C · resnet50) ──HALT── US-013 (Phase C · densenet) ──HALT── US-014 (Phase C · transnext_tiny)
                                         │
                                        HALT
                                         ▼
                            US-015 (verify + 3 phase-boundary pushes, was US-047)
```

**Ordering rules:**
- US-040…US-042 + US-043 strictly precede the renumbered series.
- US-001 / US-003 / US-004 / US-016 may interleave on a single GPU — they are independent (densenet tune, Phase C data-pipeline edit, two repo-only variant swaps). US-002 strictly follows US-016 (which itself follows US-004): the small→tiny swap must land before the retargeted tiny tune can start.
- US-005 is gated on all 6 winner JSONs present with `validated_at_full_convergence: true` AND US-003 landed (Phase C identity semantics in `degradation_levels.py`).
- US-006…US-014 are **strictly sequential** with operator-approval halts between each US. No parallel execution.
- US-015 is the single end-of-campaign closer.

---

## 11. Definition-of-Done (PRD-level)

- [ ] All 15 renumbered open stories (US-001 … US-015) + the 4 legacy stories (US-040 … US-043) check green.
- [ ] `Final_Exp.md` status block: `Phase A: 6/6, Phase B: 30/30, Phase C: 150/150, Total: 186/186`.
- [ ] `artifacts/Final_Exp.html` renders all 186 rows; first row visible under the sticky header.
- [ ] `pytest src/tests` green (torch-free baseline + new `test_ralph_loop.py` from US-005; updated `test_matrix.py` from US-003 + US-004).
- [ ] `mypy src scripts` green.
- [ ] No `*.ckpt`/`*.pt`/`*.pth` paths leaked into `Final_Exp.json` / `.md` / `.html` / `priors.json` / `best_hparams/*.json` / `history.json` / `progress.txt` / any commit pathspec.
- [ ] Determinism gate green (MSE = 0) at out_size=224 for both datasets against the **post-US-003** `degradation_levels_hash` (which itself replaces the 2026-05-12 severity-bumped hash).
- [ ] Three SYNCHRONIZER pushes at US-015 (was US-047), each with the explicit tracker pathspec only.
- [ ] 9 REPORTER summary docs under `artifacts/reports/phase_{a,b,c}_{resnet50,densenet121,transnext_tiny}_summary.md`.
- [ ] LIBRARIAN-owned `docs/phase_{a,b,c}.md` capture the final per-phase findings.

---

## 12. Decisions Locked (operator 2026-05-13)

- **D1 — Reset scope:** preserve Phase A clean baselines, priors, and lock file; back up Optuna DB and best_hparams under dated suffixes (not deletion) so the pre-5070 study is auditable.
- **D2 — Architecture: 224×224 only.** The V3 native-resolution refactor (TransNeXt at 32, MNIST pad-to-32) is rolled back. Every model trains at 224×224 with the same data pipeline; TransNeXt loads its upstream 224-pretrained checkpoint. No `_native` aliases, no asymmetric-resolution insurance trial, no `mnist_pad_to_32` field.
- **D3 — TransNeXt training regime:** full FT with differential LR (head 5e-4 / backbone 5e-5), 10× ratio matching the CNN convention. `pretrain_size=224`, `patch_size=4`.
- **D4 — Tuning protocol:** continue Option C hybrid (Stage 1 fast rank + Stage 1.5 top-3 validate) for all 6 pairs. Budget ~27 GPU-hours total tuning (90+180 GPU-min × 6).
- **D5 — Self-correction:** single retry per cell; second failure records `QUARANTINED_AFTER_RETRY` and the row counts toward `Failed` in the 186 denominator.
- **D6 — Commit cadence:** three per-phase boundary pushes (A-close, B-close, C-close), never per-cell. Matches the US-016 SYNCHRONIZER contract (Q3=C, extended).
- **D7 — Severity bump:** the 2026-05-12 [`degradation_levels.py`](src/data/degradation_levels.py) bump replaces the 2026-04 table; no cross-batch comparison crosses that hash boundary.
- **D8 — TransNeXt variant swap (operator 2026-05-14):** Replace `transnext_base` (~89M params, ~11 GB peak VRAM at batch 32 / 224×224) with `transnext_small` (~50M params, ~9 GB peak) across the 186-cell matrix. Rationale: capacity-match to ResNet50 (~25M) / DenseNet121 (~8M) given the small dataset size (10K train × 10 classes); ~40% fewer params frees days of GPU time against the 2026-05-31 poster deadline; OOM-fallback path is now unlikely on the 12 GB RTX 5070. No prior TransNeXt cell results are invalidated (none ran). US-002 retargeted in place; new US-004 owns the swap; all subsequent stories shifted forward by one.
- **D9 — TransNeXt variant swap, round 2 (operator 2026-05-14, same day as D8):** Replace `transnext_small` (~50M params, ~9 GB peak) with `transnext_tiny` (~28M params, ~7 GB peak) across the 186-cell matrix. Triggered by the in-flight US-002 Stage 1 cifar10 anchor data point of 57.8 min/trial for small, which projected the 186-cell campaign past the 2026-07-26 final deadline at the pessimistic end. Rationale: tighter capacity-match to ResNet50 (~25M) than small offered; ~310 GPU-h savings across the 62 TransNeXt rows; 5 GB VRAM headroom (vs 3 GB) on the 12 GB RTX 5070. Sunk: 10/20 trials of `transnext_small_cifar10_L3` Stage 1 (~10 GPU-h, orphaned study retained in `optuna_thz.db`). No frozen winner JSON was invalidated (no TransNeXt winner JSON existed yet). US-002 retargeted in place (small → tiny); new US-016 owns the swap; US-008/US-011/US-014 model names shift accordingly.

---

**End of PRD. Awaiting MASTER approval. On approval: execute US-040 (reset) first; do not begin tuning until US-041 + US-042 are green.**
