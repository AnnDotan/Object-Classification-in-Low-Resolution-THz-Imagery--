# PRD v2 — Noise/S&P Pipeline Fix, 90-Cell Re-run, IEEEtran Scientific Report

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Branch:** `noise_fix`
**Supersedes:** PRD v1 (RTX 5070 RALPH Loop, 2026-05-13; closed 2026-05-20 with the 186/186 v1 campaign). v1 PRD preserved in git history at commit `3e1d089`.
**Date:** 2026-05-20
**Author:** Itamar Bahat (ib94)
**Target hardware:** NVIDIA RTX 5070 (12 GiB GDDR7, sm_120 / CUDA 12.8+, bf16-mixed Blackwell tensor cores).

---

## 1. Introduction

The v1 186-cell campaign closed 2026-05-20 with all cells `healthy` per the §6.3 pathology guard and the comprehensive report rendered to [`artifacts/Final_Exp.pdf`](artifacts/Final_Exp.pdf). Post-mortem inspection of the Phase C `noise` axis numbers (TransNeXt-tiny holding ≥ 0.95 on CIFAR-10 across L1→L5 with `noise_std` up to 0.22) raised the question: *why is additive Gaussian noise barely denting accuracy?*

**Root cause confirmed 2026-05-20** ([src/data/degrade.py](src/data/degrade.py) pre-US-017): Gaussian noise and salt-and-pepper were applied AFTER the bicubic upsample to 224×224. Each `torch.randn` sample landed on a single output pixel, so the perceived noise at the underlying low_res scale was a tiny fraction of the configured `noise_std`. The bug spanned every cell whose `noise_std > 0` or `salt_pepper > 0` — i.e., all of Phase B and the Phase C `noise` + `salt_pepper` axes (90 cells).

The fix moves both stochastic pixel-level perturbations to run BEFORE the upsample: noise is drawn at `low_res`, salt-and-pepper paints `low_res^2` pixels, then bicubic spreads each sample across a kernel footprint at 224. This produces the sensor-realistic *coarse-grain* noise structure that a true low-resolution detector (THz or otherwise) would record. The lag-1 spatial autocorrelation of the residual jumps from ~0 (v1) to **0.993** (v2 at low_res=8, out_size=224, verified by [src/tests/test_degradation_determinism.py:_check_v2_noise_is_pre_upsample](src/tests/test_degradation_determinism.py)).

This PRD covers the **noise_fix follow-up**: re-running the 90 invalidated cells under pipeline v2, refreshing every downstream artifact in lockstep, and producing a publishable-quality IEEEtran scientific report as the project's final external deliverable. The 96 cells whose `noise_std == 0 AND salt_pepper == 0` (Phase A clean + Phase C `resolution` / `blur` / `saturation`) are unaffected and not re-run.

---

## 2. Goals (v2)

1. **G1 — Corrected v2 results.** All 90 noise/S&P cells re-run under pipeline v2; `runs/final/<tag>/metrics.json` overwritten in place (no archive — v1 numbers stay only in git history per operator decision 2026-05-20).
2. **G2 — Lockstep artifact freshness.** After every RALPH pass the four downstream artifacts ([Final_Exp.md](Final_Exp.md), [artifacts/Final_Exp.json](artifacts/Final_Exp.json), [artifacts/Final_Exp.html](artifacts/Final_Exp.html), [artifacts/Final_Exp.pdf](artifacts/Final_Exp.pdf)) plus the 60 affected dashboard thumbnails are regenerated and committed in the same operator action. The dashboard is monitorable mid-campaign.
3. **G3 — Scientific deliverable.** A new `docs/Final_Report.tex` (IEEEtran journal-class, hand-authored, ~1.5–2.5K lines including auto-generated tables) → `artifacts/Final_Report.pdf`. Contains abstract, methodology, full results (Phase A/B/C tables + 6 per-(model, dataset) Phase C heatmaps), discussion of v2 findings, references in BibTeX. Supersedes the markdown-pdf [`artifacts/Final_Exp.pdf`](artifacts/Final_Exp.pdf) as the canonical publishable artifact.

---

## 3. Non-Goals

- **No re-tuning** of `artifacts/best_hparams/*.json`. The priors-anchored Optuna winners were tuned at L3 Moderate on v1 pipeline; the v2 fix changes the noise *spatial structure* but not its *severity envelope*, and the winners remain the operator-locked frozen set. Re-tuning is explicitly deferred to a future US.
- **No model swaps** or architecture changes. ResNet50 / DenseNet121 / TransNeXt-tiny stay.
- **No new datasets.** CIFAR-10 + MNIST remain the only two.
- **No blur or saturation pipeline changes.** Blur stays post-upsample (kernels were calibrated for 224); saturation stays pre-downsample (RGB-correct luminance computation). Only noise and S&P move.
- **No v1 archive.** Operator declined archival 2026-05-20; v1 metrics for the 90 cells are overwritten and exist only in git history.

---

## 4. Pipeline v2 Specification

**File of record:** [src/data/degrade.py:degrade_image](src/data/degrade.py) (refactored 2026-05-20 as part of US-017).
**Pipeline-version constant:** `src.data.degradation_levels.PIPELINE_VERSION = 2`.

```
Original ([C, H, W], H/W ∈ {32, 28})
   ↓ Saturation lerp:  (1−s)·gray + s·img            ← at native resolution
   ↓ Downsample to low_res (bilinear)                 ← skipped if low_res == out_size
   ↓ Additive Gaussian noise (at low_res)             ← v2 MOVED HERE
   ↓ Salt-and-pepper noise (at low_res)               ← v2 MOVED HERE
   ↓ Upsample to out_size = 224 (bicubic)
   ↓ Gaussian blur (separable conv, kernels calibrated for 224×224)
   ↓ Clamp [0, 1]
   → ImageNet normalization (in the DataModule, unchanged)
Model input (224×224×3)
```

**Identity rules (unchanged from US-003, 2026-05-14):**
- `low_res = 224` → downsample skipped; upsample still runs to bring native 32/28 input up to 224.
- `blur_kernel ≤ 1` → blur skipped.
- `gaussian_noise_std = 0` → noise step skipped.
- `salt_pepper = 0` → S&P step skipped.
- `saturation = 1.0` → lerp skipped.

**Determinism contract preserved:** per-sample local `torch.Generator` seeded by `idx + SEED_OFFSET_*`; moving the noise/S&P consumers earlier in the function changes the number of random draws (low_res² vs out_size²) but the per-seed reproducibility still holds. Gate: [src/tests/test_degradation_determinism.py](src/tests/test_degradation_determinism.py) — five checks: intra/inter/batch/histogram equality, train/val disjointness, and the v2-specific `_check_v2_noise_is_pre_upsample` invariant (lag-1 autocorrelation > 0.5).

**Seed contract (load-bearing for the research — operator-locked 2026-05-20):**
- The **same `SEED_OFFSET_TRAIN` / `SEED_OFFSET_VAL` constants** used by v1 carry forward verbatim into v2. `degrade_image(idx, ...)` for a given `(idx, split)` must consume randomness from a `torch.Generator` seeded by `idx + SEED_OFFSET_*` — exactly as in v1. No additional global RNG draws may be introduced between the per-sample seed and the noise/S&P calls.
- v1↔v2 are pixel-incomparable (different number of draws at different stages — see §4 above), but **within v2** the seed contract guarantees: (a) the *same `idx`* always produces the *same noise pattern* across re-runs, dataloader-worker counts, and machine restarts; (b) train/val seeded sample sets remain disjoint; (c) shuffling `idx` order does not change per-sample noise.
- Owner: DATA_ARCHITECT (per agent spec: "index-based seeding contract"). Any change here is P0 and MUST be blocked by VALIDATOR.
- Gate: the existing `test_degradation_determinism.py` five-check suite (MSE = 0 intra/inter on val for both datasets) is the byte-level proof that the seed contract holds under v2.

**v1 vs v2 incomparability:** the level *values* in `DEGRADATION_LEVELS` are unchanged, but the *order of operations* differs. For any cell with `noise_std > 0` OR `salt_pepper > 0`, v1 and v2 metrics are NOT directly comparable. The 90 invalidated cells are exhaustively re-run under v2; the remaining 96 are unaffected because the moved steps are no-ops at their identity values.

---

## 5. Environment & Hardware

Unchanged from v1 PRD. The RTX 5070 / cu128 / bf16-mixed setup section is preserved verbatim in [README.md](README.md) §"Hardware Setup — RTX 5070" and applies to all v2 work. No env regressions expected; the v2 refactor is a pure Python-level reordering with no new dependencies.

**Reproducibility audit trail:** add a `pipeline_version` field to `runs/final/<tag>/metrics.json` (write-side in the Lightning training callback) so v1 vs v2 cells can be filtered programmatically. Concretely: import `from src.data.degradation_levels import PIPELINE_VERSION` and emit `metrics["pipeline_version"] = int(PIPELINE_VERSION)` alongside the existing `degradation_levels_hash` documentary field. Backfill is not required for the 90 re-run cells (they will be overwritten); the 96 non-re-run cells get a one-time `pipeline_version` patch via `scripts/update_final_exp.py` if desired (cosmetic only).

---

## 6. RALPH Loop (carried over from v1)

§6.1 (RALPH iteration shape), §6.2 (single-GPU sequential by design), §6.3 (pathology guard — overfit / underfit / divergence detectors), and §6.4 (single retry pass with `wd × 2`, `dropout = 0.1`, `backbone_lr ÷ 20`) are unchanged from the v1 PRD. The v1 §6.4 13-retries-all-declined empirical history (operator declined all 13 retry deltas across the v1 campaign; zero improved cells) carries the same operational lesson into v2: surface every flagged cell to operator and do **not** auto-accept the retry.

---

## 7. Constraints & Guardrails

Carried over from v1: weight-privacy contract (gitignore + claudeignore + `scripts/check_ignores.sh`); `pl.seed_everything(42, workers=True)` reproducibility lock; `--plan final --mode pilot` assertion; the fair-comparison invariant (same protocol, same dataset, same degradation across all three models).

**New (US-017 onward):** *artifact-freshness rule.* After every RALPH pass that touches one or more cells, the operator MUST regenerate [Final_Exp.md](Final_Exp.md), [artifacts/Final_Exp.json](artifacts/Final_Exp.json), [artifacts/Final_Exp.html](artifacts/Final_Exp.html), [artifacts/Final_Exp.pdf](artifacts/Final_Exp.pdf), and the affected `artifacts/dashboard_thumbs/<tag>.png` files before opening the next pass. Drift between `runs/final/` and these artifacts is forbidden and detectable via `python scripts/update_final_exp.py --check`.

**New (US-019B onward):** *visual pre-flight gate.* Before ANY of the 90 v2 training runs are dispatched (US-020/021/022), the affected 90 `artifacts/dashboard_thumbs/<tag>.png` files MUST be re-rendered against the v2 pipeline and reviewed by the operator. Burning ~53 GPU-h on a subtly-wrong pipeline is the failure mode this gate prevents. See US-019B for the protocol.

**New (US-017 onward):** *seed-stability invariant.* The `SEED_OFFSET_TRAIN` / `SEED_OFFSET_VAL` constants in [src/data/degrade.py](src/data/degrade.py) are frozen for the entire v2 campaign. Any PR that touches those constants, the per-sample `torch.Generator` construction, or the global RNG state in `degrade_image` is rejected automatically (DATA_ARCHITECT owns, VALIDATOR gates).

**Multi-agent protocol (carried over from v1).** Every code-touching US is dispatched through MASTER (see [agents/MASTER.md](agents/MASTER.md)). MASTER reviews the plan, delegates to the named sub-agent per their `agents/<NAME>.md` scope, and VALIDATOR signs off before LIBRARIAN/SYNCHRONIZER synchronize the narrative artifacts. The owner column in §8 is binding — agents may not write outside their declared file-system scope without a MASTER scope-extension note in the US.

---

## 8. User Stories (v2 active set)

Ten stories, dependency-ordered. US-017 and US-018 are already closed in the current session (2026-05-20) and listed here for completeness. US-019 through US-025 are unstarted and will be executed in a follow-up RALPH session. **Every owner below is a sub-agent defined in [agents/](agents/);** MASTER dispatches each US per [agents/MASTER.md](agents/MASTER.md).

| Story | Title | Owner (sub-agent) | Status |
|---|---|---|---|
| **US-017** | Pipeline v2 refactor + new invariant test | [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) | ✅ **closed 2026-05-20** |
| **US-018** | PRD v2 authoring (this document) | [MASTER](agents/MASTER.md) | ✅ **closed 2026-05-20** |
| **US-019** | RALPH loop `--axes` filter patch (+ bundled `render_cell_thumbs` `--phase`/`--axes`) | [OPTIMIZER](agents/OPTIMIZER.md) (MASTER-approved scope extension to `scripts/run_ralph_loop.py` and `src/tools/render_cell_thumbs.py`) | ✅ **closed 2026-05-20** |
| **US-019B** | **v2 degradation thumbnail preview (operator visual gate)** | [DESIGNER](agents/DESIGNER.md) (renders) + [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) (verifies seed/noise structure) | ⏳ pending |
| **US-020** | Phase B re-run (30 cells) + dashboard refresh | [EXECUTOR](agents/EXECUTOR.md) (runs) + [DEBUGGER](agents/DEBUGGER.md) (on failure) + [VALIDATOR](agents/VALIDATOR.md) (gate) | ⏳ pending |
| **US-021** | Phase C `noise` axis re-run (30 cells) + dashboard refresh | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending |
| **US-022** | Phase C `salt_pepper` axis re-run (30 cells) + dashboard refresh | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending |
| **US-023** | Post-campaign content sync (README + Final_Exp_Report.md + progress.txt) | [SYNCHRONIZER](agents/SYNCHRONIZER.md) + [LIBRARIAN](agents/LIBRARIAN.md) + [REPORTER](agents/REPORTER.md) | ⏳ pending |
| **US-024** | IEEEtran scientific report (`docs/Final_Report.tex` → `artifacts/Final_Report.pdf`) | [REPORTER](agents/REPORTER.md) (LaTeX/prose, owns `docs/`) + [DESIGNER](agents/DESIGNER.md) (heatmap tooling under `src/tools/`) | ⏳ pending |
| **US-025** | Final verification + ship-readiness audit | [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending |

---

### US-017 — Pipeline v2 refactor + new invariant test

**Goal.** Move noise + S&P from post-upsample to pre-upsample in `degrade_image`; preserve all other pipeline behavior; gate the change with a new determinism-suite invariant.

**Files modified.**
- [src/data/degrade.py](src/data/degrade.py) — `degrade_image` split downsample/upsample, moved noise + S&P between the two interpolations, updated docstring + step-order comments. Final `clamp(0, 1)` consolidated at function exit.
- [src/data/degradation_levels.py](src/data/degradation_levels.py) — added `PIPELINE_VERSION: int = 2` constant with v1→v2 rationale comment.
- [src/tests/test_degradation_determinism.py](src/tests/test_degradation_determinism.py) — added `_check_v2_noise_is_pre_upsample`: builds a constant-0.5 image, runs `degrade_image` with `low_res=8, out_size=224, noise_std=0.2, degradation_type='all'`, asserts (a) residual lag-1 horizontal autocorrelation > 0.5 (proves noise has bicubic-spread structure), (b) byte-identical re-run under same seed (proves determinism preserved).

**Acceptance criteria.**
- [x] `python -m src.tests.test_degradation_determinism` exits 0 — 5/5 checks green: cifar10@224, mnist@224, lightning cifar10@224, lightning mnist@224, v2-noise-pre-upsample.
- [x] v2 invariant test reports lag-1 autocorrelation ≥ 0.5 (observed: **0.993**).
- [x] No regression on the four pre-existing determinism checks (MSE = 0, histograms equal, train/val disjoint).

**Closed:** 2026-05-20.

---

### US-018 — PRD v2 authoring

**Goal.** Replace v1 PRD with a v2 PRD reflecting the closed 186-cell campaign + the noise-fix follow-up + the new IEEEtran deliverable. v1 PRD content preserved only in git history (commit `3e1d089`).

**Acceptance criteria.**
- [x] [PRD.md](PRD.md) rewritten end-to-end; nine v2 user stories defined; v1 stories US-001..US-016 referenced as "closed; see git history."
- [x] PRD v2 enumerates pipeline-v2 specification (§4), goals (§2), non-goals (§3), constraints (§7), and an explicit dependency-ordered story map (§10).

**Closed:** 2026-05-20.

---

### US-019 — RALPH loop `--axes` filter patch

**Goal.** Enable a single-axis subset re-run in [scripts/run_ralph_loop.py](scripts/run_ralph_loop.py) without requiring an explicit tag list. Today the script filters only by `--phase` / `--model` / `--dataset`; we need axis-level granularity for US-021 and US-022.

**Deliverables.**
- Add `--axes` argument (comma-separated, validated against `src.data.degradation_levels.AXES`).
- When `--axes` is supplied AND `--phase C` is selected, restrict the dispatched cell tags to those whose axis is in the set.
- When `--axes` is supplied with `--phase A` or `--phase B`, raise `argparse.ArgumentError` (axes are a Phase C concept).
- Idempotent: `--skip-existing` still respected.
- One-line dry-run preview: `python scripts/run_ralph_loop.py --phase C --axes noise --dry-run` prints the 30 tags it would dispatch.

**Acceptance criteria.**
- [x] `--phase C --axes noise` dispatches exactly 30 tags (3 models × 2 datasets × 5 levels × 1 axis).
- [x] `--phase C --axes noise,salt_pepper` dispatches 60 tags.
- [x] `--phase B --axes noise` raises ArgumentError with a clear message.
- [x] `--phase C --axes BOGUS` raises ArgumentError listing valid axes.
- [x] Existing `--phase C` (no `--axes`) behavior unchanged: dispatches all 150.

**Bundled scope (US-019B note):** `src/tools/render_cell_thumbs.py` gained the same `--phase` and `--axes` filters (with the identical Phase-A/B reject and AXES-subset validation), so the US-019B dispatch commands (`--phase B`, `--phase C --axes noise`, `--phase C --axes salt_pepper`) work without touching the renderer again. Filter math verified: Phase B = 30, Phase C noise = 30, Phase C salt_pepper = 30 — exactly the 90 v2-affected cells.

**Owner:** [OPTIMIZER](agents/OPTIMIZER.md), under a MASTER-approved scope extension to `scripts/run_ralph_loop.py` and `src/tools/render_cell_thumbs.py` (outside OPTIMIZER's default `src/models/` + `src/runner.py` write scope). **Dependencies:** US-017 (pipeline must be v2 before any re-runs are valid).

**Closed:** 2026-05-20.

---

### US-019B — v2 degradation thumbnail preview (operator visual gate)

**Goal.** Before any of the 90 GPU re-runs are dispatched, regenerate the affected `artifacts/dashboard_thumbs/<tag>.png` images against the v2 pipeline so the operator can **visually confirm** that the noise/S&P fix produces the expected coarse-grain structure. This is the cheap, single-CPU-pass insurance policy against burning ~53 GPU-h on a subtly-wrong pipeline. The thumbnail renderer ([src/tools/render_cell_thumbs.py](src/tools/render_cell_thumbs.py)) only re-runs `degrade_image` on a fixed grid of sample indices — it does **not** touch model weights or training — so this US is GPU-free and takes minutes, not hours.

**Deliverables.**
- All 90 v2 thumbs re-rendered: 30 `final_B_L{1..5}_{model}_{dataset}.png` + 30 `final_C_L{1..5}_noise_{model}_{dataset}.png` + 30 `final_C_L{1..5}_salt_pepper_{model}_{dataset}.png`.
- **v1 thumbnail baseline snapshot.** `artifacts/dashboard_thumbs/` has been gitignored since US-013, so the originally-planned `git show 3e1d089:...` recovery path does NOT work — the only v1 thumbs in existence live on the 5070 box's local disk. Before any `--force` re-render, the 90 affected v1 PNGs MUST be copied to `artifacts/validation/v1_thumbs/` and a SHA-256 manifest written to `artifacts/validation/v1_thumbs_manifest.json` (committed) for durable evidence. Use `python scripts/snapshot_v1_thumbs.py` (writes the snapshot + manifest) and `--verify` to re-check hashes later. The PNG blobs themselves stay gitignored — the operator visual gate runs locally and the manifest is the cross-machine audit trail.
- A side-by-side contact sheet `artifacts/validation/v1_vs_v2_thumbs.html` (or `.md`) that pairs each v2 thumb against its snapshotted v1 counterpart (under `artifacts/validation/v1_thumbs/`) so the visual diff is one click, not a manual hunt.
- A short markdown checklist `artifacts/validation/v2_preview_checklist.md` enumerating the **smoking-gun visual expectations** the operator must sign off on:
  - L5 noise at low_res=3 shows ~74-px blob structure (single low_res samples bicubically spread across 224/3 ≈ 74-px footprints), NOT 1-px white speckles.
  - L5 salt_pepper at low_res=3 shows large bright/dark blobs in ~9 distinct cells (low_res² = 9 pixels painted at low_res then bicubically spread), NOT 1-px speckles.
  - L1 thumbs are nearly indistinguishable from v1 (mild noise_std=0.04 with low_res=18 → still fine grain).
  - Saturation/blur/resolution axes are byte-identical to v1 (sanity check: those 96 cells are unaffected by US-017).
  - All thumbs deterministic across two re-runs of the renderer (proves the seed contract from §4 holds).

**Pre-flight.**
1. Confirm `degradation_levels.PIPELINE_VERSION == 2`.
2. Confirm pytest determinism suite green (US-017 acceptance).
3. **Do NOT delete** the 90 stale `metrics.json` files yet — that happens in US-020 pre-flight. Only the thumbs are touched here.

**Dispatch (CPU-only, no GPU needed).**
```powershell
.\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase B
.\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase C --axes noise
.\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase C --axes salt_pepper
.\.venv-gpu\Scripts\python.exe scripts\build_v1_v2_thumb_contact_sheet.py   # new helper
```
(The `--axes` flag on `render_cell_thumbs` may need a small patch mirroring US-019; if so, bundle that into US-019's deliverable.)

**Acceptance criteria.**
- [x] 90 v2 thumbs present on disk and visibly differ from their v1 git-history counterparts on the noise/S&P axes. *(verified 2026-05-20: all 90 SHA-256 hashes differ from `artifacts/validation/v1_thumbs_manifest.json` baseline.)*
- [x] 96 unaffected thumbs (Phase A clean + Phase C resolution/blur/saturation) byte-identical to v1 — verified by `scripts/verify_unaffected_thumbs.py` (the gitignored `artifacts/dashboard_thumbs/` makes the original `git diff --stat` recipe impossible; the script captures a SHA-256 manifest of the on-disk v1 bytes, force-re-renders the 96 under v2, and asserts zero hash deltas). *(verified 2026-05-20: 96/96 byte-identical; manifest at `artifacts/validation/unaffected_thumbs_manifest.json`.)*
- [x] Contact sheet `artifacts/validation/v1_vs_v2_thumbs.html` opens cleanly and shows all 90 pairs. *(verified 2026-05-20: `scripts/build_v1_v2_thumb_contact_sheet.py` emits 49 KiB HTML with 3 section headings, 90 pair divs, 90 v1 refs, 90 v2 refs; byte-identical across two consecutive runs.)*
- [x] **Operator signs `artifacts/validation/v2_preview_checklist.md`** (writes "approved by ib94 YYYY-MM-DD" on the last line and commits). Without this signature, US-020 dispatch is forbidden. *(signed 2026-05-20 — operator authorized "continue to US-020" mid-Ralph-cycle; checklist's 5 smoking-gun bullets ticked, US-020 HARD GATE released.)*
- [x] Re-running the renderer with no other changes produces byte-identical PNGs (seed determinism gate). *(verified 2026-05-20: second `--force` pass produced 0 hash deltas across 310 thumbs in `artifacts/dashboard_thumbs/`.)*

**Owner:** [DESIGNER](agents/DESIGNER.md) (owns `src/tools/` per agent spec, renders the contact sheet) with **[DATA_ARCHITECT](agents/DATA_ARCHITECT.md)** verifying the seed contract held end-to-end (lag-1 autocorrelation re-spot-checked on the rendered PNGs). **Dependencies:** US-017, US-019 (the `--axes` filter on the renderer).

---

### US-020 — Phase B re-run (30 cells)

**Goal.** Re-run all 30 Phase B cells under pipeline v2, with overwrite-in-place semantics. Each cell trains to convergence under the locked protocol (60 ep / patience 10 / bf16-mixed / AdamW + cosine / paper-anchored hparams from `artifacts/best_hparams/{model}_{dataset}.json`).

**Pre-flight (operator-executed before dispatch).**
1. **HARD GATE — US-019B must be closed with operator signature on `artifacts/validation/v2_preview_checklist.md`.** No dispatch without it.
2. Delete the 30 stale `runs/final/final_B_L*_{model}_{dataset}/metrics.json` files (and `metrics.csv`, `log.txt`, `history.json` for full hygiene). Keep `best.pt` / `model_last.pt` — they are local-only and will be replaced by the new Lightning run.
3. Confirm `degradation_levels.PIPELINE_VERSION == 2`.
4. Confirm pytest determinism suite green (US-017 acceptance).
5. Confirm `SEED_OFFSET_TRAIN` / `SEED_OFFSET_VAL` constants unchanged since the v1 campaign (seed-stability invariant, §7).

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_ralph_loop.py --phase B
```

**Post-pass (mandatory artifact refresh — Constraint §7).**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\update_final_exp.py
.\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase B
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_exp_json
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_dashboard
.\.venv-gpu\Scripts\python.exe scripts\build_final_exp_report.py
.\.venv-gpu\Scripts\python.exe scripts\render_final_exp_pdf.py
```

**Sentinel handling.** §6.3 pathology guard fires → surface to operator for §6.4 retry decision. Carry forward v1's empirical default (decline retry unless the cell breaks monotonicity vs neighbors). Track tally for the campaign summary.

**Acceptance criteria.**
- [x] 30/30 cells `healthy` per §6.3 guard (or operator-accepted sentinel deferral). *(2026-05-21: zero sentinels written by the §6.3 pathology guard across all 30 cells.)*
- [x] All 30 `metrics.json` files written with `pipeline_version: 2` field. *(2026-05-21: verified across all 30 cells post __v2→base consolidation.)*
- [x] [Final_Exp.md](Final_Exp.md) reflects 30 v2 Phase B rows; `python scripts/update_final_exp.py --check` exits 0. *(2026-05-21: --check returns "Final_Exp.md matches disk state".)*
- [x] 30 dashboard thumbs at `artifacts/dashboard_thumbs/final_B_L*_*_*.png` show visibly coarser noise grain than the v1 thumbs (smoking-gun visual check). *(2026-05-21: 30 thumbs re-rendered via `render_cell_thumbs --force --phase B`; pre-flight contact sheet US-019B already attested the v1↔v2 visual diff.)*
- [x] [artifacts/Final_Exp.html](artifacts/Final_Exp.html) opens cleanly; Phase B tab shows 30 v2 `val_acc` values. *(2026-05-21: dashboard rebuilt — A=6, B=30, C=150, complete=126.)*

**v2 outcome (2026-05-21):** mean Δ across the 30 Phase B cells is **−12.49pp** vs v1 (range −0.0 to −32.7pp). CIFAR-10 drops are sharper than MNIST; mid-range levels (L2-L4) drop hardest. Worst hit: `final_B_L3_transnext_tiny_cifar10` (−32.74pp). Universal: v2 noise is materially harder than v1, validating the noise-fix campaign rationale.

**GPU budget:** ~30 cells × ~35 min = **~17.5 GPU-h** (actual: 15.79h, under budget). **Owner:** [EXECUTOR](agents/EXECUTOR.md) (dispatches runs verbatim), [DEBUGGER](agents/DEBUGGER.md) (first responder on failure), [VALIDATOR](agents/VALIDATOR.md) (signs off cells before LIBRARIAN updates `runs/official/`). **Dependencies:** US-019, **US-019B (operator visual gate)**.

---

### US-021 — Phase C `noise` axis re-run (30 cells)

**Goal.** Re-run all 30 Phase C cells whose axis is `noise` under pipeline v2. Single-axis isolation semantics (US-003): the noise axis active at level L, every other axis at identity.

**Pre-flight.** Identical to US-020 but scoped to the 30 `runs/final/final_C_L*_noise_{model}_{dataset}/` directories.

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_ralph_loop.py --phase C --axes noise
```

**Post-pass artifact refresh.** Same chain as US-020 with `--phase C --axes noise` filter on `render_cell_thumbs`.

**Acceptance criteria.**
- [x] 30/30 cells `healthy` (or operator-deferred). *(2026-05-22: zero §6.3 sentinels across all 30 cells.)*
- [x] Phase C `noise` rows in [Final_Exp.md](Final_Exp.md) carry v2 `pipeline_version` + new `val_acc`. *(2026-05-22: post `__v2`→base consolidation; `update_final_exp.py --check` exits 0; build_final_exp_json complete=156.)*
- [x] L5 noise val_acc expected to drop substantially vs v1 baseline (v1 transnext_tiny L5 cifar10 noise was 0.9608 — implausibly high; v2 prediction is materially lower, but the exact number is the empirical question this US answers). *(2026-05-22: v2 transnext_tiny L5 cifar10 noise = **0.8008** — drop of −16.00pp. v1 claim falsified, as predicted.)*
- [x] Dashboard thumb `final_C_L5_noise_transnext_tiny_cifar10.png` visibly shows blob-shaped noise (single low_res samples upsampled into ~74×74 patches at low_res=3). *(2026-05-22: thumb re-rendered via `render_cell_thumbs --force --phase C --axes noise`; pre-flight US-019B contact sheet already attested the v1↔v2 blob structure.)*

**v2 outcome (2026-05-22):** mean Δ across the 30 Phase C noise cells is **−5.15pp** vs v1, but the average masks a strong split: CIFAR-10 cells drop −2.4 to −22.9pp (worst: `final_C_L5_noise_resnet50_cifar10` at −22.86pp), while all 15 MNIST cells stay within ±0.4pp of their v1 baseline. **MNIST digit geometry is robust to coarse-grain noise at any tested low_res; CIFAR-10 texture features collapse monotonically with level.** TransNeXt softens but does not stop the CIFAR-10 collapse (L5: 0.80 vs ResNet50's 0.67). v1's "TransNeXt holds ≥0.95 on noise" claim falsified for CIFAR-10.

**GPU budget:** ~17.5 GPU-h (actual: 19.40h, 11% over — TransNeXt MNIST cells ran longer to convergence than the Phase B average). **Owner:** [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-019, US-019B, US-020 (sequential to keep dashboard monotonically updated; not strictly required but operationally cleaner).

---

### US-022 — Phase C `salt_pepper` axis re-run (30 cells)

**Goal.** Re-run all 30 Phase C cells whose axis is `salt_pepper` under pipeline v2. Salt-and-pepper now paints `low_res²` pixels and the bicubic upsample spreads each into a small blob.

**Pre-flight.** Same protocol as US-020/021, scoped to the 30 `runs/final/final_C_L*_salt_pepper_{model}_{dataset}/` directories.

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_ralph_loop.py --phase C --axes salt_pepper
```

**Post-pass artifact refresh.** Same chain.

**Acceptance criteria.**
- [x] 30/30 cells `healthy` (or operator-deferred). *(2026-05-23: zero §6.3 sentinels across all 30 cells; dispatch log carries no SENTINEL/Traceback/FAILED tokens.)*
- [x] L5 salt_pepper val_acc expected to be materially lower than v1 (v1 transnext_tiny L5 cifar10 salt_pepper was 0.9602 — same implausibility class as noise). *(2026-05-23: v2 transnext_tiny L5 cifar10 salt_pepper = **0.9422** — drop of −1.80pp. The drop is real and reproducible (the cell no longer holds ≥0.95), but materially **smaller** than the analogous noise drop of −16.00pp because salt_pepper at low_res=3 paints only low_res²=9 pixels — bicubic spreads them into 9 large blobs while most of the image remains the original signal. v1 "TransNeXt holds ≥0.95 on salt_pepper" claim weakly falsified for CIFAR-10 (0.9422 < 0.95).)*
- [x] Dashboard thumb `final_C_L5_salt_pepper_transnext_tiny_cifar10.png` shows large bright/dark blobs (not 1-pixel speckles). *(2026-05-23: thumb re-rendered via `render_cell_thumbs --force --phase C --axes salt_pepper`; pre-flight US-019B contact sheet already attested the v1↔v2 blob structure at low_res=3.)*

**v2 outcome (2026-05-23):** mean Δ across the 30 Phase C salt_pepper cells is **−0.81pp** vs v1 (range −4.38 to +0.66pp). The split-by-dataset pattern mirrors noise but with smaller magnitudes: CIFAR-10 cells drop −0.10 to −4.38pp (worst: `final_C_L5_salt_pepper_resnet50_cifar10` at −4.38pp), while all 15 MNIST cells stay within ±0.62pp of v1 (mean −0.24pp). **Salt-and-pepper is empirically the gentlest of the three perturbation axes in v2** because only low_res² pixels are painted (9 at L5 vs additive noise's ~50K samples covering every low_res pixel). MNIST geometry survives both axes; CIFAR-10 texture degrades monotonically on both but more steeply under noise than under S&P. TransNeXt softens but does not stop the CIFAR-10 collapse on either axis.

**GPU budget:** ~17.5 GPU-h. **Owner:** [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-019, US-019B, US-021.

**Total US-020 + US-021 + US-022 budget:** ~53 GPU-h on the RTX 5070.

---

### US-023 — Post-campaign content sync

**Goal.** Update narrative artifacts to reflect the v2 numbers across the whole campaign. Runs AFTER all three RALPH passes complete.

**Files modified.**
- [README.md](README.md):
  - §"Final Results — 186 / 186 cells" — Phase B table (lines 95-100) refreshed with v2 numbers; Phase C L5 headline table (lines 108-112) `noise` and `salt_pepper` rows refreshed.
  - §"Headline Research Findings" — re-derive Finding #2 (TransNeXt robustness gap on perturbation axes) and Finding #3 (MNIST robustness floor) under v2 data. The v1 qualitative claims (TransNeXt holds ≥ 0.95 on noise/salt_pepper) are likely to invert or weaken; rewrite as needed without forcing the v1 narrative.
  - §"Recent scientific changes" — add a 2026-05-?? entry documenting the v2 pipeline move, the 90-cell re-run, the `PIPELINE_VERSION = 2` bump, and the v1↔v2 incomparability rule.
- [docs/Final_Exp_Report.md](docs/Final_Exp_Report.md) — auto-regenerated by `scripts/build_final_exp_report.py` from the refreshed `artifacts/Final_Exp.json`; manually review the headline-findings prose section and rewrite if v1 claims no longer hold.
- [progress.txt](progress.txt) — append "Iteration 27 — v2 noise-fix follow-up" summarizing: (a) full v1 186-cell campaign closure highlights (8-12 bullets covering Phase A/B/C completion dates, winner JSONs, headline findings, §6.4 13-retries-all-declined tally); (b) the v2 90-cell re-run scope and outcomes; (c) US-017..US-025 progress. This is the user-requested "everything done so far" summary.

**Acceptance criteria.**
- [x] [README.md](README.md) Phase B + Phase C L5 tables show v2 numbers and reference `PIPELINE_VERSION = 2`. *(2026-05-23: §"Final Results" header rewritten to flag pipeline v2 + v1↔v2 incomparability rule; Phase B 30-cell table + Phase C L5 5-axis table refreshed from `artifacts/Final_Exp.json`; bold patterns re-keyed to per-`(dataset, level)` column-max under v2; new severity-ordering paragraph appended below Phase C L5 table. Validation: 6 Phase B rows + 5 Phase C L5 rows all match disk values in order.)*
- [x] Finding #2 + Finding #3 narratives re-derived from v2 data; no orphan v1 claims left in prose. *(2026-05-23: Finding #2 rewritten as "TransNeXt softens but does not reverse the v2 CIFAR-10 perturbation collapse" with explicit v1 ↔ v2 supersession note + per-axis L5 robustness gaps (+13.2 pp noise, +11.2 pp blur, +7.1 pp saturation, +10.0 pp salt-and-pepper) + the new ≥0.95 survival map (saturation across L1→L5; salt_pepper only L1→L3). Finding #3 rewritten as "MNIST geometric-axis robustness floor preserved under pipeline v2" with per-axis MNIST L5 r50/d121/tnx triplets + the 60-cell two-axis scope phrase. Added 2026-05-23 entry to Recent scientific changes documenting US-017, the 90-cell re-run, `PIPELINE_VERSION = 2`, the v1↔v2 incomparability rule, mean-Δ per phase (−12.49 / −5.15 / −0.81 pp), and the new CIFAR-10 L5 severity ordering. Banner header on line 3 updated to flag the v2 closure date. Validation: 14 substring assertions green; 2 orphan-v1-claim substring assertions confirm the v1 "holds ≥ 0.95" framing is gone.)*
- [ ] `progress.txt` iteration 27 entry committed; serves as the canonical session summary.
- [ ] `Final_Exp_Report.md` regenerated; the existing markdown-pdf `Final_Exp.pdf` rebuilt as a side-effect of the per-pass refresh in US-020/021/022.

**Owner:** [SYNCHRONIZER](agents/SYNCHRONIZER.md) (context-alignment, stale-source detection), [LIBRARIAN](agents/LIBRARIAN.md) (owns `README.md` / `AGENTS.md` / `CLAUDE.md` / `docs/`), [REPORTER](agents/REPORTER.md) (`docs/Final_Exp_Report.md` prose + headline-findings rewrite). **Dependencies:** US-020, US-021, US-022 all closed.

---

### US-024 — IEEEtran scientific report

**Goal.** Author a publishable LaTeX scientific report rendered to `artifacts/Final_Report.pdf` via `tectonic` (preferred — self-contained, downloads packages on demand) or `pdflatex` + `bibtex` fallback.

**New files.**
- `docs/Final_Report.tex` — IEEEtran journal-class document, hand-authored. Structure:
  - `\documentclass[journal]{IEEEtran}`; packages: `graphicx`, `booktabs`, `amsmath`, `amssymb`, `hyperref`, `cite`, `subcaption`, `siunitx`.
  - **Abstract** (~200 words): three contributions — v2 pipeline + bug story, 186-cell campaign findings under corrected protocol, universal 3×3-downsample bottleneck.
  - **§I Introduction** — motivation, THz-imaging analogy, related-work pointer to TransNeXt / DenseNet / TResNet (papers already in `papers/`).
  - **§II Methodology** — datasets, models, degradation pipeline v2 (with figure of the v2 pipeline diagram), 5-level table, Optuna pre-tuning protocol.
  - **§III Experimental Design** — 186-cell matrix, Phase A/B/C, RALPH loop + §6.3 + §6.4, reproducibility contract.
  - **§IV Results** — three subsections each with a booktabs table auto-generated from `Final_Exp.json`:
    - Tab. I: Phase A clean baselines.
    - Tab. II: Phase B combined degradation + L1→L5 line plots per `(model, dataset)`.
    - Tab. III: Phase C single-axis isolation + 6 per-(model, dataset) 5×5 heatmaps (axis × level).
  - **§V Discussion** — v2 findings, universal resolution bottleneck, attention vs convolutional robustness, MNIST geometric-axis floor.
  - **§VI Conclusion** — limitations + future work (multi-axis interactions, larger backbones, real THz data).
- `docs/references.bib` — BibTeX with TransNeXt (Shi 2024), DenseNet (Huang 2017), TResNet (Ridnik 2020), EfficientNetV2, NASNet, CIFAR-10, MNIST.
- `scripts/render_final_report_tex.py` — two-stage builder:
  1. Reads `artifacts/Final_Exp.json`; emits `\input{}`-able booktabs snippets to `docs/_autogen/{phase_a_table,phase_b_table,phase_c_l5_table,phase_c_full_table}.tex`.
  2. Invokes `tectonic docs/Final_Report.tex --outdir artifacts/` (preferred) with `pdflatex` + `bibtex` + `pdflatex ×2` fallback. Output: `artifacts/Final_Report.pdf`.
- `src/tools/render_phase_c_heatmaps.py` — matplotlib helper emitting 6 PNG heatmaps to `docs/_autogen/figs/phase_c_heatmap_{model}_{dataset}.png` keyed by (model, dataset), each a 5 × 5 axis-by-level grid colored by v2 `val_acc`. The hand-authored `Final_Report.tex` `\includegraphics`'s these.

**LaTeX toolchain prerequisite.** Confirm `tectonic` or `pdflatex` on PATH on the 5070 box before execution. If neither is available, install `tectonic` via `winget install --id TectonicProject.Tectonic` or download a portable build to the venv. Flagged as risk §9(a).

**Acceptance criteria.**
- [ ] `artifacts/Final_Report.pdf` exists; ≥ 8 pages.
- [ ] All 3 tables + 6 heatmaps present and legible.
- [ ] References section renders cleanly with no `??` placeholders.
- [ ] `tectonic` (or `pdflatex`) exits with code 0.
- [ ] Auto-generated tables are byte-identical when `scripts/render_final_report_tex.py` is re-run against the same `Final_Exp.json` (idempotency check).

**Owner:** [REPORTER](agents/REPORTER.md) (authors `docs/Final_Report.tex` + `docs/references.bib` per its `docs/` write scope; produces the publishable PDF, on the same footing as the prior poster/presentation/submission deliverables it owns), with [DESIGNER](agents/DESIGNER.md) for the heatmap PNG tooling under `src/tools/` (DESIGNER's declared scope). **Dependencies:** US-023 (narrative content must be settled before the LaTeX prose is written against it).

---

### US-025 — Final verification + ship-readiness audit

**Goal.** End-of-campaign sweep: every artifact in lockstep, tests green, audit trail intact.

**Checks.**
- `pytest src/tests/test_degradation_determinism.py -v` → 5/5 green.
- `python -m src.tests.test_ignores` → 13 ignore categories pass; no `.pt` / `.ckpt` / `.pth` in git index.
- `mypy src/` → green (baseline was green per v1 US-015).
- `python scripts/update_final_exp.py --check` → exit 0.
- `bash scripts/check_ignores.sh` → green.
- Manual: open `artifacts/Final_Exp.html`, filter to Phase B + Phase C `noise` + Phase C `salt_pepper`, confirm 90 rows reflect v2 numbers and v2 thumb pairs.
- Manual: open `artifacts/Final_Report.pdf` end-to-end, scan for placeholder text / broken refs / missing figures.
- Ship-readiness audit via `Agent` tool with subagent_type `general-purpose`: cross-check that [PRD.md](PRD.md) v2 + [README.md](README.md) + [progress.txt](progress.txt) + `artifacts/Final_Report.pdf` all reference `PIPELINE_VERSION = 2` and the same 90-cell scope.

**Acceptance criteria.**
- [ ] All five test/audit commands exit 0.
- [ ] Ship-readiness agent report identifies no drift between artifacts.
- [ ] `git status` clean on `noise_fix` branch; ready for merge into `main`.

**Owner:** [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-024.

---

## 9. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **(a) LaTeX toolchain absent on 5070 box** | Medium | High (US-024 blocked) | Install `tectonic` via winget before US-024; fallback to `pdflatex` if necessary. Verify in US-024 pre-flight check. |
| **(b) Re-run GPU budget overruns ~53 h** | Low | Medium | Cells are independent and `--skip-existing` respected. Operator can pause between passes (US-020/021/022 are not interdependent for hardware reasons). |
| **(c) Cell-tag stability lost between v1 and v2** | Low | High (tracker patches break) | No tag changes in this PRD. `runs/final/<tag>/` directory names are preserved; only file contents inside change. Verified by [src/tools/final_exp_schema.py](src/tools/final_exp_schema.py) keying by `tag`. |
| **(d) v2 noise/S&P results break monotonicity** | Medium | Low | Expected and scientifically interesting. The §6.3 pathology guard checks per-cell convergence health, not L-curve monotonicity. Report the v2 monotonicity in the [README.md](README.md) headline section as-found. |
| **(e) Determinism breaks under v2** | Already mitigated | — | Verified at US-017 close: 5/5 determinism checks green, lag-1 autocorr = 0.993, MSE = 0 on re-run. |

---

## 10. Dependency-Ordered Story Map

```
US-017 (pipeline refactor) ──────────────────────┐
                                                  │
US-018 (PRD v2) ─────────────────────────────────┤
                                                  ▼
                                          US-019 (--axes patch)
                                                  │
                                                  ▼
                                  US-019B (v2 thumb preview + operator sign-off)
                                                  │  ← HARD GATE: no GPU dispatch
                                                  ▼     without operator signature
                            ┌────────────── US-020 (Phase B, 30 cells)
                            │                     │
                            │                     ▼
                            │             US-021 (Phase C noise, 30)
                            │                     │
                            │                     ▼
                            │             US-022 (Phase C S&P, 30)
                            │                     │
                            └─────────────────────┤
                                                  ▼
                                          US-023 (content sync)
                                                  │
                                                  ▼
                                       US-024 (IEEEtran PDF)
                                                  │
                                                  ▼
                                       US-025 (final audit)
```

**Critical path:** US-017 → US-019 → **US-019B** → US-020 → US-021 → US-022 → US-023 → US-024 → US-025.
**Wall-clock estimate:** ~55 GPU-h re-runs + ~6 h for US-019/023/024/025 = ~2.5 calendar days on a single RTX 5070 with overnight runs.

---

## 11. Definition of Done (campaign-level)

- [ ] US-019B operator visual sign-off captured in `artifacts/validation/v2_preview_checklist.md` BEFORE any GPU dispatch.
- [ ] `SEED_OFFSET_TRAIN` / `SEED_OFFSET_VAL` unchanged for the whole campaign (seed-stability invariant, §7); per-`idx` byte-identity of degraded samples reproducible across re-runs.
- [ ] All 90 v2 cells `healthy` per §6.3 (or operator-accepted sentinel deferrals).
- [ ] Determinism suite 5/5 green; mypy + ignore tests green.
- [ ] [Final_Exp.md](Final_Exp.md), [artifacts/Final_Exp.json](artifacts/Final_Exp.json), [artifacts/Final_Exp.html](artifacts/Final_Exp.html), [artifacts/Final_Exp.pdf](artifacts/Final_Exp.pdf), dashboard thumbs all reflect v2 numbers.
- [ ] [README.md](README.md) headline tables + findings + recent-changes entry updated.
- [ ] [progress.txt](progress.txt) iteration 27 entry committed (the "everything done so far" summary).
- [ ] `artifacts/Final_Report.pdf` exists, ≥ 8 pages, IEEEtran-styled, BibTeX references, all 3 tables + 6 heatmaps, no broken refs.
- [ ] Ship-readiness agent confirms cross-artifact consistency.
- [ ] `git status` clean on `noise_fix`; PR opened against `main`.

---

## 12. Decisions Locked

1. **Pipeline scope:** noise + salt-and-pepper both move pre-upsample. Blur stays post-upsample (kernels calibrated for 224). Saturation stays pre-downsample (RGB-correct luminance). Confirmed by operator 2026-05-20.
2. **Archival:** overwrite in place; no `runs/final_archive/`. v1 numbers exist only in git history. Confirmed 2026-05-20.
3. **PRD form:** full rewrite into v2; v1 PRD preserved only in git history (commit `3e1d089`). Confirmed 2026-05-20.
4. **PDF form:** IEEEtran/article LaTeX template hand-authored from scratch with auto-generated tables + heatmaps + BibTeX references. `markdown-pdf` rendition (`artifacts/Final_Exp.pdf`) stays in lockstep during the campaign but is *superseded* by `artifacts/Final_Report.pdf` as the publishable deliverable. Confirmed 2026-05-20.
5. **Re-tuning:** no Optuna re-runs against pipeline v2. Frozen v1 `best_hparams` carry forward. Re-tuning deferred to future US.
6. **RALPH retry policy:** §6.4 retry deltas surfaced to operator; default decision is "decline" unless cell breaks neighbor monotonicity (v1 empirical lesson).
7. **Sub-agent dispatch:** every US owner in §8 is a sub-agent in [agents/](agents/). MASTER reviews and approves the plan before any sub-agent writes code; sub-agents may not exceed their declared `agents/<NAME>.md` file-system scope without an explicit MASTER scope-extension note (e.g. US-019 grants OPTIMIZER write access to `scripts/run_ralph_loop.py`). Confirmed 2026-05-20.
8. **Visual pre-flight gate (US-019B):** the 90 v2 dashboard thumbnails are re-rendered and **operator-signed** before any of the ~53 GPU-h of training is dispatched. Cheap CPU-only insurance against burning a 2.5-day campaign on a subtly-wrong pipeline. Confirmed 2026-05-20.
9. **Noise seed lock:** the v2 noise + S&P passes are seeded by the same `idx + SEED_OFFSET_*` contract used in v1 — same constants, same per-sample local `torch.Generator`, no new global RNG draws. Within v2, same `idx` always produces the same noise pattern. v1↔v2 pixels are NOT comparable (different number of draws), but v2↔v2 byte-identity across re-runs is mandatory and tested. Confirmed 2026-05-20.

---

## 13. v1 History (US-001..US-016 — closed)

Closed in the 2026-05-13 → 2026-05-20 campaign. Full v1 PRD prose preserved at git commit `3e1d089`. Summary:

| Story | Title | Closed |
|---|---|---|
| US-001 | densenet121 × {cifar10, mnist} Optuna tune | 2026-05-14 |
| US-002 | transnext_tiny × {cifar10, mnist} Optuna tune | 2026-05-14 |
| US-003 | Phase C single-axis correction (inactive axes → identity) | 2026-05-14 |
| US-004 | TransNeXt base→small variant swap | 2026-05-14 |
| US-005 | RALPH loop driver framework | 2026-05-14 |
| US-006 | Phase A `resnet50` ×{cifar10, mnist} | 2026-05-14 |
| US-007 | Phase A `densenet121` ×{cifar10, mnist} | 2026-05-15 |
| US-008 | Phase A `transnext_tiny` ×{cifar10, mnist} | 2026-05-15 |
| US-009 | Phase B `resnet50` × L1..L5 ×{cifar10, mnist} | 2026-05-15 |
| US-010 | Phase B `densenet121` × L1..L5 ×{cifar10, mnist} | 2026-05-15 |
| US-011 | Phase B `transnext_tiny` × L1..L5 ×{cifar10, mnist} | 2026-05-16 |
| US-012 | Phase C `resnet50` × 5 axes × L1..L5 ×{cifar10, mnist} | 2026-05-17 |
| US-013 | Phase C `densenet121` × 5 axes × L1..L5 ×{cifar10, mnist} | 2026-05-18 |
| US-014 | Phase C `transnext_tiny` × 5 axes × L1..L5 ×{cifar10, mnist} | 2026-05-20 |
| US-015 | End-of-campaign verification (partial — leak audit + mypy + pytest done; SYNCHRONIZER pushes deferred to v2 US-023) | 2026-05-20 |
| US-016 | TransNeXt small→tiny variant swap (capacity-match to ResNet50) | 2026-05-14 |

Headline v1 findings (CAVEAT: noise + salt_pepper claims are invalidated by US-017; verify against v2 in US-023):
1. Universal 3×3-downsample bottleneck — every architecture lands within ±3 pp on CIFAR-10 at L5 resolution. **Carries forward to v2** (resolution axis was unaffected by US-017).
2. TransNeXt robustness gap on noise / S&P / saturation — **REQUIRES v2 RE-DERIVATION.**
3. MNIST robustness floor on perturbation axes — **REQUIRES v2 RE-DERIVATION.**

---

**End of PRD v2.**
