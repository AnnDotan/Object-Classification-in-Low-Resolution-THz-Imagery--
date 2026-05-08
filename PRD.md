# PRD: Final Research Campaign — Phase A (CNN Clean Baselines)

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Phase:** A (Clean Baselines, no degradation)
**Scope (this PRD):** ResNet50 and DenseNet121 only — TransNeXt deferred to a later GPU upgrade
**Active cells:** 4 of 6 Phase A rows in [Final_Exp.md](Final_Exp.md)
**Date:** 2026-05-05

> **Note:** This document supersedes the prior infrastructure PRD (US-001..US-014, delivered in commit `3dcf969` and logged in [progress.txt](progress.txt)). Story IDs reset to US-001 for the Phase A execution scope. Infrastructure stories from the prior PRD are preserved in git history.

---

## 1. Introduction

Phase A establishes clean-baseline accuracy for two CNN backbones (ResNet50, DenseNet121) on two datasets (MNIST, CIFAR-10) with no visual degradation applied. These four numbers anchor the entire 186-cell campaign — every Phase B/C degradation result will be reported as a delta against its Phase A clean reference.

The PyTorch Lightning engine, W&B logging, deterministic seeds, 186-cell matrix tracker, dashboard generator, and orchestrator (`run_all_phases.py --plan final`) already exist (per the 14-story infrastructure delivery on 2026-05-03). This PRD scopes the residual work needed to execute Phase A end-to-end, autonomously, on a local CUDA GPU, with paper-aligned validation gates and a Phase-A-scoped HTML dashboard view.

TransNeXt rows in `Final_Exp.md` remain `Pending` until a stronger GPU is available — they are not in scope here.

## 2. Goals

- Run 4 sequential clean-baseline experiments on local CUDA: MNIST (ResNet50, DenseNet121) → CIFAR-10 (ResNet50, DenseNet121).
- Bypass `degrade_image` entirely for Phase A — resize + ImageNet normalize only.
- Apply a **two-tier alignment gate** vs. paper baselines:
  - **Green** (within −3pp of paper number) → auto-continue.
  - **Yellow** (−3pp to −7pp) → log warning, auto-continue.
  - **Red** (worse than −7pp) → halt and prompt user.
- Apply an **autonomous continuation rule** independent of the gate:
  - No CUDA / NaN / OOM / runtime errors.
  - `(train_acc − val_acc) < 15pp` at the best epoch.
  - `val_loss` non-increasing for at least 3 of the last 5 epochs before early stopping fires.
- After every run (success or halt): regenerate `Final_Exp.md` and `artifacts/Final_Exp.html` (lightweight, static, Phase A only).
- Produce `agents/DEBUGGER.md` sub-agent that handles OOM / CUDA / code-level bugs before escalating (companion deliverable, tracked outside this PRD).

## 3. User Stories

Stories are dependency-ordered. Each fits a single Ralph iteration (~10 min of focused work).

### US-001: Pin the Phase A clean data path with a contract test ✅ DONE
**Description:** As a researcher, I want a regression test that pins the Phase A clean pipeline (resize + clamp only, no degradation) so future refactors cannot silently re-enable degradation on Phase A cells.

**Discovery (2026-05-05):** The bypass already exists end-to-end. [src/data/degrade.py:84-93](src/data/degrade.py#L84-L93) early-returns for `degradation_type='none'`, and [src/experiments/matrix.py:71](src/experiments/matrix.py#L71) wires Phase A CellSpecs through `degrade_config_for(level=None)` which sets that field. Per CLAUDE.md ("minimal changes, no rewrites of working code"), this story was narrowed from "add a DataConfig flag" to "test the existing path."

**Acceptance Criteria:**
- [x] [src/tests/test_phase_a_clean.py](src/tests/test_phase_a_clean.py) asserts: clean output == pure bilinear resize + clamp (MSE=0).
- [x] Asserts: the 'none' path is RNG-independent (different seeds → identical pixels).
- [x] Asserts: input already at `out_size` passes through unchanged.
- [x] Asserts: all 6 Phase A CellSpecs from `build_final_matrix()` carry `degradation_type='none'`, `noise_std=0`, `salt_pepper=0`, `saturation=1.0`.
- [x] Existing [src/tests/test_degradation_determinism.py](src/tests/test_degradation_determinism.py) still passes (intra=0, inter=0, train/val disjoint).
- [x] Typecheck passes.

### US-001b: `run_cell` Phase A hparams fallback ✅ DONE
**Description:** As the operator, I want `run_cell` to fall back to CLAUDE.md frozen hyperparameters for Phase A cells when `artifacts/best_hparams/{model}_{dataset}.json` is absent, so Phase A can run without Optuna pre-tuning while Phase B/C still mandate it.

**Acceptance Criteria:**
- [x] `PHASE_A_FROZEN_HPARAMS` dict in [run_systematic.py](run_systematic.py) carries CLAUDE.md values for `resnet50`, `densenet121`, `transnext_{micro,small,base}`.
- [x] `_load_hparams_for_cell(spec)` resolves: real JSON > Phase A frozen fallback > FileNotFoundError (Phase B/C).
- [x] `run_cell()` uses `_load_hparams_for_cell` instead of `_load_best_hparams` directly.
- [x] [src/tests/test_phase_a_hparams_fallback.py](src/tests/test_phase_a_hparams_fallback.py): Phase A falls back, Phase B/C still raise, real JSON takes precedence over fallback.
- [x] Existing [src/tests/test_run_cell.py](src/tests/test_run_cell.py) regression checks: 6/6 still PASS.
- [x] Typecheck passes.

### US-002: Phase A alignment-gate thresholds module ✅ DONE
**Description:** As an autonomous runner, I want a single module that returns the green/yellow/red verdict for a `(model, dataset, val_acc)` triple, so gate logic is centralized and testable.

**Acceptance Criteria:**
- [x] New file [src/experiments/phase_a_gate.py](src/experiments/phase_a_gate.py) exporting `PAPER_BASELINES: dict[tuple[str, str], float]` and `evaluate_gate(model, dataset, val_acc) -> Literal["green", "yellow", "red"]`.
- [x] Baselines encoded: `(resnet50, cifar10) = 0.93`, `(densenet121, cifar10) = 0.95`, `(resnet50, mnist) = 0.995`, `(densenet121, mnist) = 0.995`.
- [x] Bands: green if `val_acc >= paper − 0.03`, yellow if `paper − 0.07 <= val_acc < paper − 0.03`, red otherwise (closed boundaries on the high side).
- [x] [src/tests/test_phase_a_gate.py](src/tests/test_phase_a_gate.py) covers all three bands per `(model, dataset)`, plus boundary cases, above-paper, unknown pairs (KeyError), and gate_explanation shape.
- [x] Bonus: `gate_explanation()` helper for human-readable rationale in `gate_verdict.json`.
- [x] Typecheck passes.

### US-003: Auto-continue evaluator (overfit gap + val_loss trend + error state) ✅ DONE
**Description:** As an autonomous runner, I want a function that decides `continue | halt` from a completed run's `metrics.csv`, so progression rules are declarative and inspectable.

**Acceptance Criteria:**
- [x] `evaluate_continuation(metrics_csv_path, error_log_path) -> dict` added to [src/experiments/phase_a_gate.py](src/experiments/phase_a_gate.py).
- [x] Returns `{decision, reasons, gap, best_epoch, epochs_run, val_loss_trend_ok, trend_non_increasing, trend_window_size, had_errors, error_marker}`.
- [x] Halts on: error markers in log (Traceback / [ERROR] / CUDA error / out of memory), `gap >= 0.15` at best-val-acc epoch, fewer than 3 of last 5 val_loss transitions non-increasing.
- [x] Edge cases: fewer than 5 epochs → halt with reason `too_few_epochs:N<5`; missing metrics.csv → halt; missing error_log_path is NOT treated as an error.
- [x] [src/tests/test_phase_a_continuation.py](src/tests/test_phase_a_continuation.py) covers happy path, all 4 halt reasons individually, and a stacked-reasons case (3+ halts at once). 9/9 PASS.
- [x] Stdlib csv only — no pandas dependency added.
- [x] Typecheck passes.

### US-004: Phase A sequential runner ✅ DONE
**Description:** As the operator, I want one entry-point that executes the 4 Phase A cells in order (MNIST→CIFAR-10, ResNet50 then DenseNet121 within each), invoking the gate + continuation evaluator and halting on red/halt verdicts.

**Acceptance Criteria:**
- [x] New [scripts/run_phase_a.py](scripts/run_phase_a.py) orchestrating the 4 cells via `run_systematic.run_cell(...)` (canonical dispatcher, not bypassed).
- [x] Run order: `resnet50_mnist -> densenet121_mnist -> resnet50_cifar10 -> densenet121_cifar10`.
- [x] Frozen hyperparameters routed through US-001b's `_load_hparams_for_cell` Phase A fallback (CLAUDE.md values).
- [x] `clean` data mode comes from US-001 (Phase A CellSpec carries `degradation_type='none'`).
- [x] Run tag = `final_clean_{model}_{dataset}` (verified via tags in `PHASE_A_CELL_ORDER`).
- [x] `runs/final/<tag>/gate_verdict.json` written after each cell — even on training crash (so DEBUGGER agent has a fingerprint).
- [x] Decision precedence: run_cell exception > continuation halt > red band > continue. Halt exits non-zero; remaining cells skipped.
- [x] Green/yellow + continue → auto-proceed.
- [x] `--dry-run` prints planned order, no training.
- [x] `--only <model>_<dataset>` runs a single cell; unknown target raises SystemExit with valid choices.
- [x] `post_cell_hook` reserved for US-005 (tracker refresh) / US-006 (HTML rebuild). Hook failures are logged but never abort the runner.
- [x] [src/tests/test_run_phase_a.py](src/tests/test_run_phase_a.py): 9/9 PASS — orchestration verified end-to-end with mocked `run_cell_fn`.
- [x] Typecheck passes.

### US-005: Per-run hook to refresh `Final_Exp.md` ✅ DONE
**Description:** As a tracker user, I want `Final_Exp.md` updated immediately after every Phase A cell so the master tracker is always in sync.

**Acceptance Criteria:**
- [x] [scripts/run_phase_a.py](scripts/run_phase_a.py) defines `refresh_final_exp_hook(verdict)` and wires it as the default `post_cell_hook` for CLI users (in-process import via `importlib.util` since `scripts/` is not a package).
- [x] After each cell (success or halt), `update_final_exp.main([])` regenerates `Final_Exp.md` from disk state — picks up the new `metrics.json`, status flips to `Complete`/`Failed` automatically.
- [x] TransNeXt rows remain `Pending` (verified end-to-end: only the staged Phase A cell flips status; other 185 stay Pending).
- [x] Hook raising does NOT halt the runner — re-asserted via [src/tests/test_run_phase_a_hook.py](src/tests/test_run_phase_a_hook.py)'s `hook-failure-tolerated` case with the production hook + forced failure.
- [x] [src/tests/test_run_phase_a_hook.py](src/tests/test_run_phase_a_hook.py): 4/4 PASS — hook-calls-once, hook-reraises-on-nonzero, end-to-end Final_Exp.md write, hook-failure-tolerated.
- [x] US-004 runner tests still 9/9 PASS — runner internals unchanged.
- [x] Typecheck passes.

### US-006: Lightweight `Final_Exp.html` Phase A view ✅ DONE
**Description:** As a researcher, I want a static HTML page showing the 4 Phase A cells in a table with collapsible per-run details (loss/acc curves, hyperparams, gate verdict reasoning), regenerated after each run.

**Acceptance Criteria:**
- [x] Extended [src/tools/build_final_dashboard.py](src/tools/build_final_dashboard.py) with a `--phase a` flag (per CLAUDE.md "minimal changes" — sibling output `artifacts/Final_Exp_PhaseA.html` so the existing 186-tile dashboard remains untouched).
- [x] Table columns: tag, model, dataset, status, `best_val_acc`, `train_acc_at_best`, `gap`, `epochs_run`, gate band (color-coded), continuation decision.
- [x] `<details>` per row → 3-column grid: learning curves PNG, JSON `<pre>` of `hparams` + `hparams_source`, gate-verdict explanation + reasons list. No JS framework, no CDN, no live polling.
- [x] Idempotent — missing cells render with placeholder messages ("no learning curves yet", "no metrics.json yet", "cell not run yet").
- [x] [scripts/run_phase_a.py](scripts/run_phase_a.py) now wires `refresh_all_trackers_hook` as the default `post_cell_hook` — refreshes BOTH `Final_Exp.md` and `Final_Exp_PhaseA.html` per cell, with isolated failure handling.
- [x] Browser-renderable static file verified end-to-end via [src/tests/test_phase_a_dashboard.py](src/tests/test_phase_a_dashboard.py) (6/6 PASS).
- [x] No regressions: `test_run_phase_a` (9/9), `test_run_phase_a_hook` (4/4), `test_dashboard` (3/3) all still PASS.
- [x] Typecheck passes.

### US-007: Execute cell `final_clean_resnet50_mnist`
**Description:** As the operator, I want to run the first Phase A cell and verify the runner halts or continues correctly based on its result.

**Acceptance Criteria:**
- [ ] `python scripts/run_phase_a.py --only resnet50_mnist` completes without CUDA / NaN / OOM errors.
- [ ] `runs/final/final_clean_resnet50_mnist/metrics.json` exists with `final_val_acc` (or `best_val_acc`), `train_acc`, `epochs_run`.
- [ ] `gate_verdict.json` written with band + continuation decision.
- [ ] `Final_Exp.md` and `artifacts/Final_Exp.html` reflect the result.
- [ ] If band is red → runner halted and operator reviewed before proceeding.
- [ ] Typecheck passes.

### US-008: Execute cell `final_clean_densenet121_mnist`
**Description:** As the operator, I want the second Phase A cell to run automatically after US-007 if its verdict was green/yellow.

**Acceptance Criteria:**
- [ ] Cell completes; metrics + verdict files written.
- [ ] `Final_Exp.md` and `Final_Exp.html` updated.
- [ ] Runner logs the auto-continue or halt decision before next cell.
- [ ] Typecheck passes.

### US-009: Execute cell `final_clean_resnet50_cifar10`
**Description:** As the operator, I want the third Phase A cell to run automatically (or be the next manually-resumed cell after a halt).

**Acceptance Criteria:**
- [ ] Cell completes; metrics + verdict files written.
- [ ] `Final_Exp.md` and `Final_Exp.html` updated.
- [ ] Typecheck passes.

### US-010: Execute cell `final_clean_densenet121_cifar10`
**Description:** As the operator, I want the final Phase A cell to run and close out the phase.

**Acceptance Criteria:**
- [ ] Cell completes; metrics + verdict files written.
- [ ] `Final_Exp.md` shows all 4 active Phase A rows resolved (Done or Halted) and TransNeXt rows still `Pending`.
- [ ] `Final_Exp.html` shows the 4 expandable rows with curves and verdicts.
- [ ] A short summary line appended to `progress.txt` confirms Phase A end-of-phase status.
- [ ] Typecheck passes.

### US-015: Gold Standard dashboard (`Final_Exp.html` unified table view) ✅ DONE
**Description:** As a researcher entering Phase B, I want a single canonical `artifacts/Final_Exp.html` that replicates the legacy 36-cell grid structure, divided into three phase sections, with parameter-rich rows and the existing thumbnails / collapsible curves / gate-verdict panels folded in.

**Decisions locked (operator 2026-05-05):**
- Q1=A: replace the 186-tile view; `Final_Exp.html` becomes the new layout, tile-grid renderer retired.
- Q2=B: fold the Phase A operator dashboard ([artifacts/Final_Exp_PhaseA.html](artifacts/Final_Exp_PhaseA.html), US-006) into `Final_Exp.html`; gate-verdict info lands in the same per-row `<details>` panel.

**Acceptance Criteria:**
- [x] [src/tools/build_final_dashboard.py](src/tools/build_final_dashboard.py) `build_dashboard()` (default mode) writes `artifacts/Final_Exp.html` in the new layout. The legacy tile-grid `_section_html` / `_tile_html` paths and `--phase a` flag are deleted (no orphaned code).
- [x] HTML structure: three `<div class="phase-section" id="phase-a|b|c">` sections, each containing a `<table>` with one `<tr>` per cell.
- [x] Table columns (in this exact order):
  | ID | Model | Level | Parameters | PSNR/SSIM | Accuracy | Visuals |
  - **ID**: 1..186, the canonical position from `build_final_matrix()`.
  - **Model**: `resnet50` / `densenet121` / `transnext_*` (no `_size` decoration).
  - **Level**: `clean` (Phase A) | `L{n}` (Phase B) | `L{n} / {axis}` (Phase C).
  - **Parameters**: human-readable degrade summary built from the cell's `DegradeConfig` (e.g., `clean (no degradation)`, `low_res=10, blur 7×1.30, noise=0.09, S&P=0.08, sat=0.50`, or per-axis form for Phase C).
  - **PSNR/SSIM**: `mean±std` for each metric from `image_quality.json`; em-dash when absent.
  - **Accuracy**: `best_val_acc` formatted to 4 decimals; Phase A rows additionally show the green/yellow/red band as a colored badge.
  - **Visuals**: inline Original|Degraded thumbnail (224×448 PNG from `artifacts/dashboard_thumbs/<tag>.png`) + a `<details>` panel with learning curves PNG, hyperparameters dump, and (Phase A only) gate-verdict explanation + reasons + manual_override note if present.
- [x] No external CDN, no JS framework — pure static HTML + inline CSS, opens via `file://`.
- [x] Phase counts in section headers (e.g., "Phase A — Clean Baselines (6 cells)" + "X complete · Y running · Z pending · W failed").
- [x] [src/tests/test_dashboard.py](src/tests/test_dashboard.py): 6/6 PASS — section/row counts, parameter strings, status transition, failed badge, Phase A gate verdict + manual_override visible, Phase B placeholder for verdict.
- [x] [src/tests/test_phase_a_dashboard.py](src/tests/test_phase_a_dashboard.py): 2/2 PASS — combined hook writes only `Final_Exp.html` (no `Final_Exp_PhaseA.html` sibling); failure isolation preserved.
- [x] [scripts/run_phase_a.py](scripts/run_phase_a.py) `refresh_phase_a_dashboard_hook` calls `build_final_dashboard.main([])` (no `--phase a`).
- [x] 59/59 PASS across all touched test suites — no regressions.
- [x] Typecheck passes.

### US-016: Autonomous dashboard trigger in `run_all_phases.py` + SYNCHRONIZER phase-boundary push ✅ DONE
**Description:** As the operator running the 186-cell campaign, I want `run_all_phases.py` to refresh both trackers (`Final_Exp.md` + `Final_Exp.html`) on every Stop event (cell complete, training failure, manual SIGINT), and SYNCHRONIZER to commit + push tracker files to the upstream remote at each phase boundary so the remote is always within one phase of local state.

**Decisions locked (operator 2026-05-05):**
- Q3=C: per-phase boundary push (after Phase A, B, C complete), not per-cell. Per-cell tracker refresh stays purely local.

**Acceptance Criteria:**

*Autonomous trigger:*
- [x] [run_all_phases.py](run_all_phases.py) `_refresh_trackers_default` calls the shared helper after every cell.
- [x] [scripts/refresh_trackers.py](scripts/refresh_trackers.py) — new shared module with `refresh_all()`. Failures isolated; both trackers always run.
- [x] SIGINT handler in `run_all_phases.py`: flushes a tracker refresh before re-raising `KeyboardInterrupt`.
- [x] Failed cells still trigger refresh + count toward phase boundary (verified by 6/6 of `test_run_all_phases_dashboard`).
- [x] Live smoke test (`--plan final --phase A --skip-existing` with mocked SYNCHRONIZER): all 4 complete cells SKIP, 2 TransNeXt cells attempted, HTML refreshed after each, sync fired exactly once for Phase A with `('A', 6)`.

*SYNCHRONIZER push at phase boundaries:*
- [x] New [scripts/sync_trackers_git.py](scripts/sync_trackers_git.py) — `commit_and_push_phase_boundary(phase, n_cells)` returns the result dict.
- [x] Pathspec explicit: `Final_Exp.md`, `artifacts/Final_Exp.html`, `progress.txt`. Tested to never include weight files, `runs/final/**`, `artifacts/optuna_thz.db`.
- [x] No-op path: `git diff --cached --quiet` rc=0 → skip commit + push.
- [x] Commit message: `chore(trackers): refresh after Phase {A|B|C} ({n} cells)`.
- [x] `git push` with no override (uses upstream); `FORBIDDEN_FLAGS` constant + assertion blocks `--force`, `--no-verify`, `-f`, `-i`.
- [x] Fail-soft: every subprocess failure logs `[sync][WARN]` + populates `errors`; never raises. Defense-in-depth: even if `phase_boundary_fn` raises, the runner catches and continues.
- [x] [agents/SYNCHRONIZER.md](agents/SYNCHRONIZER.md) — new "Phase Boundary Sync" section with pathspec, forbidden flags, commit format, fail-soft contract, test coverage.

*Tests:*
- [x] [src/tests/test_refresh_trackers.py](src/tests/test_refresh_trackers.py): 4/4 PASS.
- [x] [src/tests/test_sync_trackers_git.py](src/tests/test_sync_trackers_git.py): 9/9 PASS — happy path, no forbidden flags, explicit pathspec safety, no-op, push/commit/add fail-soft, unknown phase, message format.
- [x] [src/tests/test_run_all_phases_dashboard.py](src/tests/test_run_all_phases_dashboard.py): 6/6 PASS — single boundary per phase, never per-cell; skipped cells count; failures don't skip; sync raising doesn't halt.
- [x] [src/tests/test_final_plan.py](src/tests/test_final_plan.py): 6/6 PASS — no regressions in the legacy `run_all_phases.py` API.
- [x] Total: 84/84 PASS across 14 test suites.
- [x] Typecheck passes.

## 4. Non-Goals

- **TransNeXt runs of any kind.** Their 2 Phase A rows remain `Pending`; do not download weights, do not warm the cache, do not edit TransNeXt code paths.
- **Phase B and Phase C cells.** The 30 + 150 degradation cells are out of scope here.
- **Optuna pre-tuning.** Hyperparameters are frozen per CLAUDE.md; do not run `tune_all.py` for Phase A.
- **Multi-GPU, distributed, or cloud execution.** Local single-CUDA only.
- **Live-polling dashboard.** No JS framework, no auto-refresh; static HTML only.
- **W&B project restructure.** Use the existing W&B setup unchanged.
- **`agents/DEBUGGER.md` content.** Drafted as a separate companion deliverable after this PRD; not a user story here.
- **Reading model checkpoints** (`*.ckpt`, `*.pt`, `*.pth`). Privacy rule from CLAUDE.md applies — analyze runs only via `metrics.json` / `metrics.csv` / tracker files.
- **Refactoring delivered infrastructure.** US-001..US-014 from the prior PRD ship as-is; this PRD only adds the Phase A execution layer on top.

## 5. Technical Notes

**Reuse, don't rebuild:**
- Lightning training entry-point and `THzClassifier` / `THzDataModule` already exist in [src/lightning/](src/lightning/).
- [run_systematic.py](run_systematic.py) `run_cell(tag, mode, engine, ...)` is the canonical per-cell dispatcher (delivered in prior PRD US-008).
- [scripts/update_final_exp.py](scripts/update_final_exp.py) is the canonical tracker regenerator.
- [src/tools/render_curve_thumbs.py](src/tools/render_curve_thumbs.py) is reused for dashboard curves.
- [src/tools/build_final_dashboard.py](src/tools/build_final_dashboard.py) already builds the 186-tile dashboard; extend with a Phase-A view rather than fork.
- Hyperparameters live in CLAUDE.md as the single source of truth — read, do not duplicate.

**CUDA optimization checklist (verify in US-004 wiring, do not refactor existing code):**
- `precision="16-mixed"` already set when CUDA is available — confirm in `run_cell` path.
- DataLoader: `pin_memory=True`, `num_workers ≥ 4`, `persistent_workers=True`, `prefetch_factor ≥ 2` — confirm and adjust only if measurably slow.
- `torch.set_float32_matmul_precision("high")` for Tensor Cores — set once at runner startup.

**Paper-baseline numbers used in `phase_a_gate.py`:**
- CIFAR-10: ResNet50 ≈ 0.93, DenseNet121 ≈ 0.95 (full-data paper numbers — gate band acknowledges 10K-subset shortfall via the −3pp / −7pp tolerance).
- MNIST: 0.995 floor for both (standard saturated benchmark).

**Gate logic precedence (US-004 runner):**
1. Hard errors (CUDA / OOM / NaN / Inf) → always halt regardless of accuracy.
2. Continuation rule (gap, val_loss trend, epoch count) from US-003 → halt overrides green band.
3. Alignment gate (red) from US-002 → halt.
4. Otherwise → continue (green or yellow).

**Out-of-scope companion deliverable:** [agents/DEBUGGER.md](agents/DEBUGGER.md) will be authored after PRD acceptance. Its scope: triage and fix OOM (lower batch / accumulate grad), CUDA driver / mixed-precision issues, and code-level bugs surfaced by the runner — escalating to the user only when fixes exceed its scoped file system or risk altering frozen hyperparameters.
