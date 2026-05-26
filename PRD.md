# PRD v3 — Phase D Regularization Sweep + Documentation / Dashboard / NotebookLM Sync

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Branch:** `PHASE_D` (current working branch; merge target `main`)
**Supersedes:** PRD v2 (Noise/S&P Pipeline Fix + 90-Cell Re-run + IEEEtran Scientific Report, 2026-05-20; closed 2026-05-23 with US-017..US-025 all green). v2 PRD preserved in git history at commit `ced98ec`.
**Date:** 2026-05-23
**Author:** Itamar Bahat (ib94)
**Target hardware:** NVIDIA RTX 5070 (12 GiB GDDR7, sm_120 / CUDA 12.8+, bf16-mixed Blackwell tensor cores).

---

## 1. Introduction

The v2 campaign closed 2026-05-23 with US-017..US-025 all green: the noise / salt-and-pepper pipeline was corrected (`PIPELINE_VERSION = 2`), 90 cells re-ran under the corrected pipeline, and a publishable IEEEtran scientific report shipped to [`artifacts/Final_Report.pdf`](artifacts/Final_Report.pdf) (8 pages, 13 BibTeX entries, 3 tables + 6 heatmaps).

The v2 numbers revealed a **−12.49 pp mean val_acc collapse** on Phase B vs v1 (range −0.0 to −32.74 pp), with the worst hit being `final_B_L3_transnext_tiny_cifar10` at −32.74 pp. The Optuna winners stored in `artifacts/best_hparams/*.json` were tuned at L3 Moderate against the v1 pipeline; they had no exposure to v2's coarse-grain noise structure (lag-1 spatial autocorrelation 0.993, verified by [src/tests/test_degradation_determinism.py:_check_v2_noise_is_pre_upsample](src/tests/test_degradation_determinism.py)).

This raises the **operator-locked headline research question**:

> *Does targeted regularization recover the Phase B v2 −12.49 pp mean collapse vs v1?*

PRD v3 introduces **Phase D — Regularization Sweep**: 90 new cells (3 treatments × 5 levels × 3 models × 2 datasets) that layer regularization deltas on top of the frozen v2 Phase B baselines. The pipeline is unchanged (`PIPELINE_VERSION` stays at 2), the Optuna winners are unchanged, and the 5-level degradation table is unchanged. The only axis of variation is the regularization treatment.

Phase D is **purely additive**: the default 186-cell matrix is preserved byte-identical when `include_phase_d=False`. The 90 Phase D cells become reachable when `include_phase_d=True` is passed to `iter_cells()`, `build_final_matrix()`, or `run_all_phases.py --phase D`. Total campaign size with Phase D included is 276 cells; the v1/v2 186-cell view continues to render byte-identical when no `final_D_*` directories exist on disk.

The treatments — already wired in [src/experiments/cells.py:37-87](src/experiments/cells.py) and [run_systematic.py:243-282](run_systematic.py):

| Treatment | CNN delta | TransNeXt delta | Rationale |
|---|---|---|---|
| **T1 — Architectural dropout** | `dropout=0.2` (timm `drop_rate`) | `drop_path_rate=0.2` (stochastic depth) | Isolates architectural regularization. CNN dropout flows through the classifier head; TransNeXt routes to stochastic depth because timm CNNs don't honor `drop_path_rate`. |
| **T2 — Label-mixing** | `mixup_alpha=0.2, cutmix_alpha=0.0` | identical | Isolates label-mixing regularization. Model-agnostic; `timm.data.Mixup` wraps targets. |
| **T3 — Combo (kitchen-sink)** | T1 ∪ T2 ∪ `cutmix_alpha=1.0` | T1 ∪ T2 ∪ `cutmix_alpha=1.0` | Most aggressive combined regularization. If recovery is achievable, T3 should show the strongest signal. |

Source of truth: [run_systematic.py:_phase_d_treatment_deltas](run_systematic.py).

---

## 2. Goals (v3)

1. **G1 — Recovery quantification.** All 90 Phase D cells produce v3 metrics that enable per-(model, dataset, level, treatment) Δ-vs-Phase-B-v2 measurement. Outputs: `artifacts/figures/phase_d_recovery_{model}_{dataset}.png` × 6 + a composite summary + an L3-anchored recovery table.

2. **G2 — Lockstep artifact freshness.** After every Phase D RALPH pass the dashboard, `Final_Exp.json` / `.html` / `.md`, the 90 new dashboard thumbs, and the recovery PNGs regenerate in the same operator pass — extending v2 §7's artifact-freshness rule to Phase D.

3. **G3 — Final_Report.pdf rev2 with §VII Phase D appendix.** A new §VII appendix added to [docs/Final_Report.tex](docs/Final_Report.tex) carrying the 6 per-(model, dataset) recovery plots, an L3-anchored Δ table, and a 1-paragraph headline answer to the §1 research question. Target ≥ 10 pages total, pdflatex/bibtex exit 0, zero `??` placeholders.

4. **G4 — Two-pass NotebookLM alignment.** Pass 1 uploads PRD v3 NOW (immediately after this PRD lands) so subsequent planning context is grounded in the v3 work plan. Pass 2 (end-of-campaign, post-US-033) re-sources the full curated manifest after `README.md`, `CLAUDE.md`, `Final_Exp_Report.md`, `Final_Report.pdf`, `docs/phase_d.md`, and `progress.txt` close.

5. **G5 — Test coverage for Phase D.** Add pytest assertions for `_phase_d_treatment_deltas` arithmetic (per-model routing of T1 to dropout vs drop_path_rate), matrix tag uniqueness (no `final_B_*` vs `final_D_*` collision), schema flow (treatment field round-trips through `Final_Exp.json`), and dashboard rendering (Phase D tab + chip group visibility logic).

---

## 3. Non-Goals (v3)

- **No Phase D re-tune.** Optuna L3 winners in `artifacts/best_hparams/*.json` are reused verbatim as the baseline. Treatments are pure deltas. Re-tuning under regularization is explicitly deferred to a hypothetical v4 future-work bullet.
- **No pipeline changes.** `PIPELINE_VERSION = 2` stays. `degradation_levels_hash` does NOT rotate. The Phase D cells share Phase B's exact `DegradeConfig` byte-for-byte.
- **No model swaps.** ResNet50 / DenseNet121 / TransNeXt-tiny stay.
- **No new datasets.** CIFAR-10 + MNIST remain the only two.
- **No re-run of Phase A/B/C v2.** The 186 prior cells stay as-is. Phase D is purely additive.
- **No archival of v2 PRD into the tree.** v2 prose lives only in git history at commit `ced98ec`.
- **No new degradation axes.** The 5-axis × 5-level matrix is unchanged.

---

## 4. Phase D Specification (carries v2 pipeline forward + treatment deltas)

**Pipeline-version constant unchanged:** `PIPELINE_VERSION = 2`. v3 introduces **NO** step-order changes to the data pipeline. The v2 §4 pipeline diagram carries forward verbatim.

**Treatment delta contract.** From [run_systematic.py:243-262](run_systematic.py):

```python
def _phase_d_treatment_deltas(treatment: str, model_name: str) -> dict[str, float]:
    is_transnext = model_name.startswith("transnext_")
    if treatment == "T1":
        return {"drop_path_rate": 0.2} if is_transnext else {"dropout": 0.2}
    if treatment == "T2":
        return {"mixup_alpha": 0.2, "cutmix_alpha": 0.0}
    if treatment == "T3":
        arch = {"drop_path_rate": 0.2} if is_transnext else {"dropout": 0.2}
        return {**arch, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}
```

These deltas merge into `best_params` via `_apply_phase_d_treatment(spec, hparams)` ([run_systematic.py:265-282](run_systematic.py)). Each Phase D `metrics.json` carries:

- `hparams.best_params` (with treatment deltas merged in)
- `phase_d_treatment` (one of `T1`, `T2`, `T3`)
- `phase_d_deltas` (the exact delta dict that was applied)
- `pipeline_version: 2`

Per-cell delta attribution is post-hoc programmatic.

**Tag scheme.** `final_D_{T}_L{l}_{m}_{d}` (e.g., `final_D_T3_L3_transnext_tiny_cifar10`). The leading `final_D_` segment is the dashboard's Phase D filter key. Cannot collide with `final_B_L*_*_*` (defense-in-depth assertion at [run_all_phases.py:385-390](run_all_phases.py)).

**Baseline identity.** Each Phase D `final_D_T{x}_L{l}_{m}_{d}` cell compares directly against `final_B_L{l}_{m}_{d}` (same (model, dataset, level), no other axes activated). Phase B v2 numbers are the reference; v1 numbers are out of scope for v3 analysis.

**Determinism contract preserved.** Carried from v2 §4: `SEED_OFFSET_TRAIN` / `SEED_OFFSET_VAL` constants frozen; per-sample `torch.Generator` seeded by `idx + SEED_OFFSET_*`. Phase D adds **NO** additional global RNG draws beyond mixup/cutmix label-sampling, which gets its own seeded generator inside `THzClassifier`. The byte-identical pixel guarantee from v2's `test_degradation_determinism.py` holds for Phase D cells by construction (same DegradeConfig as Phase B siblings).

**Lightning loop wiring (verified 2026-05-23).** All four treatment hyperparameters are first-class kwargs through the full Lightning call stack:
- [src/lightning/train.py:102-105, 174-177, 253-256, 355-357, 403-405](src/lightning/train.py) — `mixup_alpha`, `cutmix_alpha`, `drop_path_rate`, `dropout` flow through.
- [src/lightning/module.py:57-83](src/lightning/module.py) — `THzClassifier.__init__` instantiates `timm.data.Mixup` when `mixup_alpha > 0 or cutmix_alpha > 0` with `prob=1.0, switch_prob=0.5, mode="batch"`.
- [src/lightning/module.py:113](src/lightning/module.py) — `drop_path_rate` flows into `create_transnext_model(...)`.
- [src/lightning/module.py:131](src/lightning/module.py) — `dropout` flows into `timm.create_model(..., drop_rate=...)` for CNN backbones.
- [src/lightning/module.py:177-185](src/lightning/module.py) — `training_step` calls `self.mixup_fn(x, y)` before forward when active.

**No code change needed for US-028 pilot to train correctly.** The Lightning wiring is end-to-end complete; this is the load-bearing finding from the 2026-05-23 sub-agent research.

---

## 5. Environment & Hardware

Unchanged from PRD v2 §5. The RTX 5070 / cu128 / bf16-mixed setup section in [README.md](README.md) §"Hardware Setup — RTX 5070" applies to all v3 work. No new dependencies; mixup/cutmix via `timm.data.Mixup` is already in the existing requirements lock.

**Reproducibility audit trail (extended for Phase D).** Each Phase D `metrics.json` must round-trip `pipeline_version=2` AND a non-empty `phase_d_deltas` dict AND a non-null `phase_d_treatment` field. The dashboard's Phase D row-count assertion (`EXPECTED_COUNTS_WITH_D["D"] == 90`) is the campaign-level integrity check.

---

## 6. RALPH Loop (carried over from v1/v2)

§6.1 (RALPH iteration shape), §6.2 (single-GPU sequential), §6.3 (pathology guard — overfit / underfit / divergence detectors), and §6.4 (single retry pass with `wd × 2`, `dropout = 0.1`, `backbone_lr ÷ 20`) are unchanged from PRD v1/v2.

**New §6.5 — Phase D dispatch.** Phase D is dispatched via:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_ralph_loop.py --phase D
```

The `--phase D` flag activates `include_phase_d=True` in `cells.iter_cells()` and `matrix.build_final_matrix()`. Optional `--treatments T1,T2,T3` filter (mirroring v2's `--axes` pattern from US-019) — OPTIMIZER scope-extension under MASTER approval — restricts dispatch to a treatment subset. Bundle the same `--treatments` filter into [src/tools/render_cell_thumbs.py](src/tools/render_cell_thumbs.py) for consistency.

**§6.4 retry policy under Phase D.** Carried forward verbatim: §6.3 pathology guard flags → surface to operator → operator decides retry. v1's empirical lesson (13/13 retries declined across the v1 campaign, zero improved cells) carries the same operational default into v3. Track tally for the campaign summary.

---

## 7. Constraints & Guardrails

All v2 guardrails carried forward:

- **Weight-privacy contract** (`.gitignore` + `.claudeignore` + `scripts/check_ignores.sh`).
- **`pl.seed_everything(42, workers=True)`** reproducibility lock.
- **`--plan final --mode pilot` is rejected** with an assertion (pilot mode is for smoke tests only; the 6-cell US-028 pilot uses `--mode pilot` on individual cell tags, NOT the full plan).
- **Fair-comparison invariant**: same protocol / dataset / degradation across all three models within each (model, dataset, level, treatment) tuple.
- **Artifact-freshness rule** (v2 §7): after every RALPH pass the operator MUST regenerate `Final_Exp.md` / `.json` / `.html` / `.pdf` + affected dashboard thumbs.
- **Seed-stability invariant** (v2 §7): `SEED_OFFSET_TRAIN` / `SEED_OFFSET_VAL` frozen for the entire v3 campaign.

**New (v3 only):**

- **Baseline-comparability invariant.** A Phase D cell must reference its Phase B v2 baseline by tag explicitly. The recovery plot script ([scripts/plot_phase_d_comparison.py](scripts/plot_phase_d_comparison.py)) rejects any Phase D row whose Phase B partner is missing or whose `pipeline_version != 2`.
- **Phase B v2 baseline manifest lock.** At US-028 pre-flight, capture SHA-256 of every `final_B_L*_*/metrics.json` into `artifacts/validation/phase_b_v2_baseline_manifest.json` and commit. Re-verify hashes at US-031 plot rendering. Prevents silent baseline drift during the ~2.5-day campaign.
- **6-cell pilot HARD GATE.** Like v2's US-019B visual gate, no 84-cell sweep dispatches (US-029) until US-028 (6-cell T3_L3 × all (model, dataset) pairs) lands healthy AND operator signs `artifacts/validation/phase_d_pilot_checklist.md`. Burning ~52 GPU-h on subtly-wrong delta arithmetic is the failure mode this gate prevents.
- **NotebookLM 2-pass discipline.** Pass 1 (PRD v3 upload) executes immediately AFTER this PRD lands. Pass 2 (full re-source) executes AFTER US-033 closes. No mid-campaign incremental refresh.

**Multi-agent protocol carried over from v1/v2.** Every code-touching US is dispatched through MASTER ([agents/MASTER.md](agents/MASTER.md)). MASTER reviews the plan, delegates to the named sub-agent per its `agents/<NAME>.md` scope, and VALIDATOR signs off before LIBRARIAN / SYNCHRONIZER synchronize the narrative artifacts. The owner column in §8 is binding — agents may not write outside their declared file-system scope without an explicit MASTER scope-extension note.

---

## 8. User Stories (v3 active set)

Eleven stories, dependency-ordered. US-026 (this PRD) is in-progress this session; US-027 through US-035 are unstarted and execute in follow-up RALPH sessions. Every owner below is a sub-agent defined in [agents/](agents/).

| Story | Title | Owner (sub-agent) | Status |
|---|---|---|---|
| **US-026** | PRD v3 authoring + progress.txt rewrite (this document) | [MASTER](agents/MASTER.md) | ⏳ in-progress this session |
| **US-027** | `docs/phase_d.md` rationale + treatment table | [REPORTER](agents/REPORTER.md) + [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) (review) | ⏳ pending |
| **US-028** | 6-cell pilot smoke test (T3_L3 × all 6 (model, dataset) pairs) + baseline-manifest snapshot | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending |
| **US-029** | Phase D full sweep (remaining 84 cells) + post-pass artifact refresh | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending |
| **US-029.5** | `Final_Exp.json` schema + aggregator wiring for Phase D | [DESIGNER](agents/DESIGNER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending |
| **US-030** | Dashboard Phase D tab + Treatment chip group + Phase B-vs-Phase D `<details>` strip | [DESIGNER](agents/DESIGNER.md) | ⏳ pending |
| **US-031** | `scripts/plot_phase_d_comparison.py` per-panel outputs + LaTeX recovery table | [DESIGNER](agents/DESIGNER.md) (plot) + [REPORTER](agents/REPORTER.md) (LaTeX review) | ⏳ pending |
| **US-032** | Post-campaign content sync (README + CLAUDE.md + Final_Exp_Report.md + progress.txt) | [SYNCHRONIZER](agents/SYNCHRONIZER.md) + [LIBRARIAN](agents/LIBRARIAN.md) + [REPORTER](agents/REPORTER.md) | ⏳ pending |
| **US-033** | `Final_Report.pdf` rev2 — new §VII Phase D appendix | [REPORTER](agents/REPORTER.md) + [DESIGNER](agents/DESIGNER.md) (heatmap helper) | ⏳ pending |
| **US-034** | NotebookLM 2-pass sync (pass 1 immediate / pass 2 post-US-033) | [NOTEBOOKLM_SYNC](agents/NOTEBOOKLM_SYNC.md) | ⏳ pending |
| **US-035** | Final verification + ship-readiness audit | [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending |

---

### US-026 — PRD v3 authoring + progress.txt rewrite

**Goal.** Replace PRD v2 with a v3 PRD reflecting the closed v2 campaign + the Phase D regularization-sweep follow-up + the documentation / dashboard / NotebookLM sync. v2 PRD content preserved only in git history (commit `ced98ec`). progress.txt preamble retired (the ≤200-line rule is formally rescinded; the file becomes an append-only journal).

**Acceptance criteria.**
- [ ] [PRD.md](PRD.md) rewritten end-to-end; eleven v3 user stories defined; v2 stories US-017..US-025 referenced as "closed; see git history."
- [ ] PRD v3 enumerates Phase D specification (§4), goals (§2), non-goals (§3), constraints (§7), and an explicit dependency-ordered story map (§10).
- [ ] [progress.txt](progress.txt) preamble (lines 1-3) updated: date stamp refreshed to 2026-05-23, ≤200-line rule formally retired with new "append-only journal" policy.
- [ ] [progress.txt](progress.txt) Iteration 47 entry appended summarizing the PRD v3 authoring session.

**Owner:** [MASTER](agents/MASTER.md).

---

### US-027 — `docs/phase_d.md` rationale + treatment table

**Goal.** Create `docs/phase_d.md` mirroring [docs/phase_c.md](docs/phase_c.md)'s 56-line structure. Document the headline research question, the three treatment deltas, the tag scheme + collision-safety note, the `degradation_levels_hash` invariance note (Phase D does NOT touch the degradation table — same v2 pipeline, same hash, byte-identical per-sample noise patterns between Phase B and Phase D siblings), and the per-cell delta attribution invariant.

**Deliverables.**
- New file [docs/phase_d.md](docs/phase_d.md), ~60-100 lines.
- One sentence in [README.md](README.md) §"Implementation Status" pointing to it (LIBRARIAN scope; defer full README sync to US-032).

**Acceptance criteria.**
- [ ] Sections present: title, "What changed" table, rationale paragraph, treatment table (T1/T2/T3 × CNN/TransNeXt), tag scheme block, hash-invariance note, delta-attribution note.
- [ ] Treatment table values match [run_systematic.py:_phase_d_treatment_deltas](run_systematic.py) byte-for-byte (CI substring assertion: `dropout=0.2`, `drop_path_rate=0.2`, `mixup_alpha=0.2`, `cutmix_alpha=1.0`).
- [ ] Explicit `PIPELINE_VERSION = 2` carry-forward note present.
- [ ] Explicit baseline-comparison rule documented (Phase D Tx_L{l} compares against `final_B_L{l}_{m}_{d}`).
- [ ] Every claim cited to a file path; no invented code paths or values.

**Owner:** [REPORTER](agents/REPORTER.md) (authors). [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) reviews the treatment table for parity with the actual code. **Dependencies:** US-026.

---

### US-028 — 6-cell pilot smoke test + Phase B v2 baseline manifest snapshot

**Goal.** Validate the Phase D pipeline end-to-end on 6 cells (T3_L3 × all 6 (model, dataset) pairs) BEFORE dispatching the full 84-cell sweep. T3 exercises all four delta keys (dropout / drop_path_rate + mixup_alpha + cutmix_alpha) per-model-routing; L3 is the Optuna-tuned operating point (strongest collapse signal); the 6-cell breadth validates per-model delta routing across CNN dropout vs TransNeXt drop_path_rate.

Concurrently, capture the SHA-256 manifest of every Phase B v2 `final_B_L*_*/metrics.json` as the immutable baseline reference for US-031's recovery plot.

**Pre-flight (operator-executed before dispatch).**
1. Confirm `degradation_levels.PIPELINE_VERSION == 2`.
2. Confirm pytest determinism suite green.
3. Capture baseline: `python scripts/snapshot_phase_b_v2_baseline.py` writes `artifacts/validation/phase_b_v2_baseline_manifest.json` with `{tag: sha256(metrics.json)}` for all 30 Phase B v2 cells. Commit.
4. Confirm `runs/final/final_D_T3_L3_*/` are empty (no leftover pilot dirs).

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_D_T3_L3_resnet50_cifar10 --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_D_T3_L3_resnet50_mnist --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_D_T3_L3_densenet121_cifar10 --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_D_T3_L3_densenet121_mnist --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_D_T3_L3_transnext_tiny_cifar10 --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_D_T3_L3_transnext_tiny_mnist --mode pilot
```

(Pilot mode is the v2 smoke-test path: 5 epochs / patience 2. Convergence is NOT the gate; the gate is that `metrics.json` is well-formed and carries `phase_d_treatment=T3` + `phase_d_deltas`.)

**Operator visual gate.** After the 6 pilot cells complete, operator inspects `runs/final/final_D_T3_L3_*/metrics.json` and signs `artifacts/validation/phase_d_pilot_checklist.md` confirming:
- All 6 `metrics.json` carry `pipeline_version=2`, `phase_d_treatment=T3`, and non-empty `phase_d_deltas`.
- CNN dropout vs TransNeXt drop_path_rate routing visible in the `phase_d_deltas` field (CNN cells have `dropout`, TransNeXt cells have `drop_path_rate`).
- Mixup labels show up in the training log (training accuracy decoupled from labels for the first epoch — signature of active Mixup).
- No CUDA / bf16 crashes in the dispatch log; no SENTINEL tokens.

**Pilot cleanup.** After operator sign-off, delete the 6 pilot `runs/final/final_D_T3_L3_*/` directories so the full sweep (US-029) re-trains those cells at full convergence (60 epochs / patience 10).

**Acceptance criteria.**
- [ ] `phase_b_v2_baseline_manifest.json` committed with 30/30 SHA-256 entries.
- [ ] 6/6 `final_D_T3_L3_*/metrics.json` written with `phase_d_treatment=T3` and the expected delta keys (CNN: `dropout` + `mixup_alpha` + `cutmix_alpha`; TransNeXt: `drop_path_rate` + `mixup_alpha` + `cutmix_alpha`).
- [ ] Zero §6.3 sentinels across the 6 pilot cells; no Traceback / FAILED tokens in dispatch logs.
- [ ] Operator signs `artifacts/validation/phase_d_pilot_checklist.md` (last line: `approved by ib94 YYYY-MM-DD`). HARD GATE for US-029.
- [ ] Pilot dirs cleaned up post-sign-off (`runs/final/final_D_T3_L3_*/` empty before US-029 dispatch).
- [ ] GPU-h tracked (~3.5h expected at pilot speed).

**GPU budget.** ~6 × 35 min = ~3.5 GPU-h (pilot mode at 5 epochs is much faster than full FT; actual likely ~3 GPU-h).

**Owner:** [EXECUTOR](agents/EXECUTOR.md) (dispatches), [DEBUGGER](agents/DEBUGGER.md) (failure triage), [VALIDATOR](agents/VALIDATOR.md) (signs off the pilot checklist). **Dependencies:** US-026, US-027.

---

### US-029 — Phase D full sweep (remaining 84 cells)

**Goal.** Re-run the 84 non-pilot Phase D cells at full convergence (60 epochs / patience 10 / bf16-mixed / AdamW + cosine / paper-anchored hparams from `artifacts/best_hparams/{model}_{dataset}.json` PLUS the treatment delta).

**Pre-flight.**
1. **HARD GATE — US-028 must be closed with operator signature on `artifacts/validation/phase_d_pilot_checklist.md`.** No dispatch without it.
2. Confirm pilot dirs cleaned (`runs/final/final_D_T3_L3_*/` empty).
3. Confirm `PIPELINE_VERSION == 2`.
4. Confirm `SEED_OFFSET_TRAIN` / `SEED_OFFSET_VAL` constants unchanged since v2 campaign.

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_ralph_loop.py --phase D
```

The 6 T3_L3 cells re-train under full FT (operator chose this over `--skip-existing` to ensure all 90 cells share the 60-epoch convergence regime).

**Post-pass (mandatory artifact refresh — Constraint §7).**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\update_final_exp.py
.\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase D
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_exp_json
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_dashboard
.\.venv-gpu\Scripts\python.exe scripts\plot_phase_d_comparison.py
```

(The `--phase D` filter on `render_cell_thumbs` requires the bundled scope-extension; fall back to `--force` un-filtered if not yet landed. Thumbs for Phase D cells are visually identical to their Phase B siblings — same DegradeConfig — so the render is a thin tagging pass.)

**Sentinel handling.** §6.3 pathology guard fires → surface to operator for §6.4 retry decision. Carry forward v1/v2's empirical default (decline retry unless cell breaks monotonicity vs neighbors).

**Acceptance criteria.**
- [ ] 90/90 cells `healthy` per §6.3 guard (or operator-accepted sentinel deferrals). Track retry-acceptance tally.
- [ ] All 90 `metrics.json` files written with `pipeline_version: 2`, non-null `phase_d_treatment`, and non-empty `phase_d_deltas`.
- [ ] [Final_Exp.md](Final_Exp.md) reflects 90 Phase D rows; `python scripts/update_final_exp.py --check` exits 0; total cell count = 276 (186 + 90).
- [ ] 90 dashboard thumbs at `artifacts/dashboard_thumbs/final_D_*.png` rendered (thin pass; visually identical to Phase B siblings by construction).
- [ ] [artifacts/Final_Exp.html](artifacts/Final_Exp.html) opens cleanly; Phase D tab shows 90 v3 `val_acc` values (requires US-029.5 + US-030 landed).

**v3 outcome (to be filled in at close).** Mean Δ across the 90 Phase D cells vs their Phase B v2 baseline siblings, broken out by treatment and dataset. Headline answer to the §1 research question.

**GPU budget.** ~90 cells × ~35 min = **~52.5 GPU-h** (full FT, comparable to v2 Phase B's 17.5h × 3). Actual variance ±15% expected. **Owner:** [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-028. **Recommended (not blocking):** US-029.5 landed before final post-pass so dashboard refreshes pick up Phase D rows.

---

### US-029.5 — `Final_Exp.json` schema + aggregator wiring for Phase D

**Goal.** Wire Phase D into the `Final_Exp.json` aggregator pipeline so the dashboard, plot script, and external consumers see Phase D rows once they exist on disk. **Currently a HARD blocker** — [src/tools/build_final_exp_json.py:243](src/tools/build_final_exp_json.py) calls `iter_cells()` *without* `include_phase_d=True` and [src/tools/final_exp_schema.py](src/tools/final_exp_schema.py) `FinalExpRow` does not carry `treatment`. The dashboard cannot render Phase D until this US lands.

**Deliverables.**
- Patch [src/tools/build_final_exp_json.py:243](src/tools/build_final_exp_json.py): `iter_cells()` → `iter_cells(include_phase_d=_phase_d_present_on_disk(runs_root))`. Move `_phase_d_present_on_disk` from `scripts/update_final_exp.py` to `src/experiments/cells.py` as a public helper to avoid a `scripts/` reverse-import.
- Patch [src/tools/build_final_exp_json.py:245](src/tools/build_final_exp_json.py): hardcoded `EXPECTED_TOTAL == 186` assertion → `EXPECTED_TOTAL_WITH_D if include_phase_d else EXPECTED_TOTAL`.
- Extend `_row_for(meta)` at [src/tools/build_final_exp_json.py:210-238](src/tools/build_final_exp_json.py) to populate `"treatment": meta.treatment`.
- Patch `update_cell(tag, ...)` at [src/tools/build_final_exp_json.py:294](src/tools/build_final_exp_json.py) to apply the same `include_phase_d` gating.
- Extend [src/tools/final_exp_schema.py:FinalExpRow](src/tools/final_exp_schema.py) TypedDict with `treatment: Optional[str]`; bump `SCHEMA_VERSION` by 1.
- New tests in [src/tests/test_build_final_exp_json.py](src/tests/test_build_final_exp_json.py):
  - `test_phase_d_absent_returns_186_rows` — clean tree → 186 rows, no Phase D.
  - `test_phase_d_present_returns_276_rows` — with `runs/final/final_D_*` dirs → 276 rows, 90 with `treatment in (T1, T2, T3)`.
  - `test_schema_treatment_field_present` — every row dict has a `treatment` key (None for A/B/C, str for D).
  - `test_schema_version_bumped` — `SCHEMA_VERSION` strictly greater than the v2-era constant.

**Acceptance criteria.**
- [ ] [src/tools/build_final_exp_json.py](src/tools/build_final_exp_json.py) emits a 276-row JSON when Phase D dirs exist; 186-row JSON byte-identical to today's when no `final_D_*` dirs exist (regression-guarded).
- [ ] `treatment` field present on every row (None for A/B/C, "T1"/"T2"/"T3" for D).
- [ ] `SCHEMA_VERSION` bumped; cached dashboard v1 invalidates on next page load.
- [ ] 4 new pytest assertions green.
- [ ] `mypy src/tools/` reports no new type errors.

**Owner:** [DESIGNER](agents/DESIGNER.md) (owns `src/tools/`). [VALIDATOR](agents/VALIDATOR.md) signs off the schema bump + regression assertion. **Dependencies:** US-026 (PRD lock).

---

### US-030 — Dashboard Phase D tab + Treatment chip group + recovery `<details>` strip

**Goal.** Extend [src/tools/build_final_dashboard.py](src/tools/build_final_dashboard.py) so the static HTML dashboard renders a 4th tab "Phase D" with a multi-select Treatment chip group (T1 / T2 / T3 — union semantics), shows the 90 Phase D rows with a new Treatment column, and embeds a per-(model, dataset) Phase B-vs-T1/T2/T3 recovery `<details>` strip below the table (mirroring the existing US Trend `<details>` pattern).

**Precise line edits (from 2026-05-23 sub-agent dashboard spec).**
- [src/tools/build_final_dashboard.py:40](src/tools/build_final_dashboard.py) — import `EXPECTED_COUNTS_WITH_D` alongside `EXPECTED_COUNTS`; import `PHASE_D_TREATMENTS`.
- [src/tools/build_final_dashboard.py:645](src/tools/build_final_dashboard.py) — `PHASES = ['A','B','C']` → `['A','B','C','D']`.
- [src/tools/build_final_dashboard.py:655-657](src/tools/build_final_dashboard.py) — extend `countsByPhase` and `state.filters` literals with `D: {}`.
- [src/tools/build_final_dashboard.py:840-863](src/tools/build_final_dashboard.py) — add `tr.dataset.treatment = row.treatment || ''` in row rendering; emit Treatment column cell after Level (empty `—` for A/B/C rows).
- [src/tools/build_final_dashboard.py:870](src/tools/build_final_dashboard.py) — `recomputeCountsByPhase` initializer adds `D: {}`.
- [src/tools/build_final_dashboard.py:907-924](src/tools/build_final_dashboard.py) — `rowMatchesFilters` add treatment-on-Phase-D guard symmetric to the axis-on-Phase-C guard.
- [src/tools/build_final_dashboard.py:949-951](src/tools/build_final_dashboard.py) — `applyActivePhase` add `treatmentGroup.style.display = (phase === 'D') ? '' : 'none'`.
- [src/tools/build_final_dashboard.py:1505-1511](src/tools/build_final_dashboard.py) — new `chip_treatment = _chip_html("treatment", "TREATMENT", PHASE_D_TREATMENTS)` (multi-select by default).
- [src/tools/build_final_dashboard.py:1565-1569](src/tools/build_final_dashboard.py) — 4th `<button data-phase="D">Phase D <span>90</span></button>` tab.
- [src/tools/build_final_dashboard.py:1587-1600](src/tools/build_final_dashboard.py) — insert `<th>Treatment</th>` after Level `<th>`.
- New `renderPhaseDRecoveryStrip(rows)` JS function alongside `renderExecutionUsTrend` ([line 1021](src/tools/build_final_dashboard.py)); wire from `ingest()` at [line 1166](src/tools/build_final_dashboard.py).
- New `<details class="phase-d-recovery-section">` mount HTML after [line 1563](src/tools/build_final_dashboard.py); empty-state fallback "Awaiting Phase D runs." matching the Execution US Trend pattern.

Tests in [src/tests/test_dashboard.py](src/tests/test_dashboard.py):
- `test_phase_d_tab_renders_with_count_90`.
- `test_treatment_chip_group_hidden_on_phases_abc`.
- `test_phase_d_row_has_treatment_dataset_attribute`.
- `test_multi_select_treatment_filter_persists_to_localstorage` (or HTML snapshot test if Playwright is infeasible).
- `test_186_cell_dashboard_byte_identical_when_phase_d_absent` (regression).

**Acceptance criteria.**
- [ ] Dashboard HTML contains 4th `data-phase="D"` tab; count reads "90" when `EXPECTED_COUNTS_WITH_D['D']==90`.
- [ ] Treatment chip group hidden on A/B/C; visible only on D.
- [ ] Phase D rows render with Treatment column populated (T1/T2/T3); A/B/C rows show `—` and remain byte-identical otherwise.
- [ ] Multi-select on Treatment chips works (T1∪T2 visible together); state round-trips through `localStorage.final_exp.filters.D`.
- [ ] Per-(model, dataset) "Phase B vs T1/T2/T3" recovery strip mounts under existing `<details class="us-trend-section">` pattern; draws only when ≥1 Phase D cell is Complete.
- [ ] 10-minute polling cycle auto-picks-up Phase D rows once `Final_Exp.json` carries them (POLL_INTERVAL_MS = 600000).
- [ ] 186-cell view byte-identical when no `final_D_*` directories exist (regression).

**Owner:** [DESIGNER](agents/DESIGNER.md). **Dependencies:** US-029.5 (load-bearing — without `treatment` in the JSON, the chip filter has nothing to filter on).

---

### US-031 — Phase D recovery PNGs + LaTeX recovery table

**Goal.** Extend [scripts/plot_phase_d_comparison.py](scripts/plot_phase_d_comparison.py) to emit per-(model, dataset) PNGs at `artifacts/figures/phase_d_recovery_{model}_{dataset}.png` (6 figs) PLUS the existing composite at `artifacts/figures/phase_d_recovery_summary.png`. Also emit a LaTeX recovery table at `docs/_autogen/phase_d_recovery_table.tex` (6 rows × 4 cols: (model_dataset), Δ_T1_at_L3, Δ_T2_at_L3, Δ_T3_at_L3) for `\input{}` into the Final_Report rev2 §VII appendix.

**Deliverables.**
- Extend `render()` to loop over (model, dataset) and `fig.savefig(per_panel_path)` per cell at `figsize=(5, 3.5), dpi=150`.
- Rename existing composite output to `phase_d_recovery_summary.png`.
- New `_emit_recovery_table(out_path)` helper invoked from `main()`.
- Pin determinism: `matplotlib.rcParams["pdf.fonttype"] = 42`, `rcParams["svg.hashsalt"] = "phase_d_recovery"`, `metadata={"CreationDate": None}` passed to `fig.savefig`.
- Baseline re-verification: read `artifacts/validation/phase_b_v2_baseline_manifest.json`, re-hash every `final_B_L*_*/metrics.json`, assert equality. Script exits 1 with a clear error on any drift.
- New test [src/tests/test_phase_d_plot.py:test_byte_identical_across_runs](src/tests/test_phase_d_plot.py) — two consecutive script invocations produce identical SHA-256 PNGs.

**Acceptance criteria.**
- [ ] 6 per-panel PNGs at `artifacts/figures/phase_d_recovery_{m}_{d}.png` with `dpi=150`, `figsize=(5,3.5)`, baseline `#777777` dashed + T1 `#1f77b4` + T2 `#2ca02c` + T3 `#d62728` solid.
- [ ] Composite at `artifacts/figures/phase_d_recovery_summary.png` (3×2 small-multiple, renamed from current output).
- [ ] LaTeX recovery table at `docs/_autogen/phase_d_recovery_table.tex` with 6 rows and L3-anchored Δ columns.
- [ ] Baseline-manifest re-verification passes (SHA-256 of every `final_B_L*_*/metrics.json` matches `phase_b_v2_baseline_manifest.json`).
- [ ] Two consecutive runs produce byte-identical PNGs (idempotency assertion).

**Owner:** [DESIGNER](agents/DESIGNER.md) (plot helper, `src/tools/` + `scripts/` scope). [REPORTER](agents/REPORTER.md) reviews the LaTeX table format. **Dependencies:** US-029, US-028 (manifest snapshot must exist).

---

### US-032 — Post-campaign content sync

**Goal.** Update narrative artifacts to reflect Phase D recovery numbers across the whole campaign. Runs AFTER US-029, US-030, US-031 complete. Mirrors v2 US-023's pattern.

**Files modified.**
- [README.md](README.md):
  - §"Final Results" — append a "Phase D — Regularization Recovery" subsection with the headline per-treatment per-(model, dataset) recovery table.
  - §"Headline Research Findings" — append Finding #4 documenting whether regularization recovers the v2 collapse (yes/partially/no per treatment, with magnitudes).
  - §"Recent scientific changes" — add a 2026-05-?? entry documenting Phase D scope, results, and the (model, dataset, treatment) × (level) recovery surface.
- [CLAUDE.md](CLAUDE.md):
  - §"Project Status" — append Phase D opens / closes entries.
  - §"186-cell Experiment Plan" — rename to "186/276-cell Experiment Plan"; add Phase D row to the table.
  - §"Experiment System" — add the `--phase D` dispatch command.
- [docs/Final_Exp_Report.md](docs/Final_Exp_Report.md) — auto-regenerated by `scripts/build_final_exp_report.py`; manually review and rewrite the conclusions section if Phase D inverts any v2 findings.
- [progress.txt](progress.txt) — append "Iteration N — v3 Phase D close" with the campaign-level summary.

**Acceptance criteria.**
- [ ] [README.md](README.md) carries a Phase D Recovery subsection + Finding #4 + 2026-05-?? entry in Recent scientific changes.
- [ ] [CLAUDE.md](CLAUDE.md) "186-cell" → "186/276-cell" rename done; Phase D row in the experiment plan table.
- [ ] Finding #4 narrative derived from v3 data; explicit per-treatment recovery magnitudes (e.g., "T3 recovers +X pp on (resnet50, cifar10, L3); T2 alone recovers +Y pp; T1 alone recovers +Z pp").
- [ ] `progress.txt` iteration entry committed; canonical session summary.
- [ ] `Final_Exp_Report.md` regenerated; the `Final_Exp.pdf` (markdown-pdf rendition) rebuilt as a side-effect.

**Owner:** [SYNCHRONIZER](agents/SYNCHRONIZER.md) (context alignment, stale-source detection), [LIBRARIAN](agents/LIBRARIAN.md) (owns `README.md` / `AGENTS.md` / `CLAUDE.md` / `docs/`), [REPORTER](agents/REPORTER.md) (`docs/Final_Exp_Report.md` prose). **Dependencies:** US-029, US-030, US-031.

---

### US-033 — `Final_Report.pdf` rev2 — new §VII Phase D appendix

**Goal.** Add a new §VII appendix to [docs/Final_Report.tex](docs/Final_Report.tex) carrying the 6 recovery PNGs (from US-031), the L3-anchored Δ recovery table (`\input{_autogen/phase_d_recovery_table}`), and a 1-paragraph headline answer to the §1 research question ("Does regularization recover the v2 collapse?"). Rebuild `artifacts/Final_Report.pdf` rev2 via the existing `scripts/render_final_report_tex.py` pipeline.

**New §VII structure.**
- §VII Phase D — Regularization Recovery
  - §VII.A Design: 3 treatments × 5 levels × 3 models × 2 datasets = 90 cells. Treatment definitions (cite the T1/T2/T3 table from §II.C).
  - §VII.B Results: `\includegraphics` × 6 recovery panels arranged as 3×2 subfigures. Table I (auto-imported from `_autogen/phase_d_recovery_table.tex`).
  - §VII.C Discussion: 1-paragraph answer to the headline question, per-treatment per-dataset interpretation, attribution to architectural (T1) vs label-mixing (T2) vs combo (T3) effects.

**Acceptance criteria.**
- [ ] `artifacts/Final_Report.pdf` rev2 exists; ≥ 10 pages (was 8 in v2).
- [ ] §VII Phase D appendix present in TOC; section heading and 3 subsections rendered.
- [ ] All 6 recovery PNGs `\includegraphics`'d successfully; no `??` placeholders.
- [ ] Recovery table renders with 6 rows; column headers visible (model_dataset, Δ_T1, Δ_T2, Δ_T3 at L3).
- [ ] `pdflatex` + `bibtex` + `pdflatex ×2` exit 0; no missing-reference warnings.
- [ ] Headline answer paragraph quotes specific pp values from the recovery table.

**Owner:** [REPORTER](agents/REPORTER.md) (LaTeX/prose). [DESIGNER](agents/DESIGNER.md) reviews the heatmap PNG renders for legibility. **Dependencies:** US-031, US-032.

---

### US-034 — NotebookLM 2-pass sync

**Goal.** Two-pass synchronization of the THz Project notebook (`notebook_id: e244f8b6-30ab-4966-a74c-426ff88be39a`):

- **Pass 1 (immediate, after US-026 lands)** — upload PRD v3 only so subsequent planning context is grounded in the v3 work plan.
- **Pass 2 (post-US-033)** — full re-source: refresh all 8 manifest-tracked files (`README.md`, `PRD.md`, `CLAUDE.md`, `docs/Final_Exp_Report.md`, `docs/phase_a.md`, `docs/phase_c.md`, new `docs/phase_d.md`, `progress.txt`) + add `artifacts/Final_Report.pdf` rev2.

**Pre-flight.** `python -m notebooklm status` to verify auth before any upload. Storage state path drift between `.notebooklm/storage_state.json` (documented in [agents/NOTEBOOKLM_SYNC.md](agents/NOTEBOOKLM_SYNC.md):36) and `.notebooklm/profiles/default/storage_state.json` (operator-stated) flagged as risk §9(g); reconcile before pass 1.

**Dispatch.**
```powershell
# Pass 1 (immediate, after this PRD lands)
python scripts\notebooklm_sync.py refresh --apply --only PRD.md

# Pass 2 (post-US-033)
python scripts\notebooklm_sync.py full --apply
```

**Acceptance criteria.**
- [ ] Pass 1: `scripts/notebooklm_sync.log` carries a `[refresh]` entry with `PRD.md` and timestamp matching this session's commit; commit prefixed `[notebooklm-sync]`.
- [ ] Pass 2: `scripts/notebooklm_sync.log` carries a `[full]` entry with `push`, `prune`, `refresh` counts; commit prefixed `[notebooklm-sync]`.
- [ ] Both passes' source lists match `scripts/notebooklm_manifest.yml` include globs.
- [ ] `python -m notebooklm status` confirms the THz Project notebook source list matches the manifest after Pass 2.
- [ ] No `*.ckpt` / `*.pt` / `*.pth` files uploaded (deny-glob enforcement verified).

**Owner:** [NOTEBOOKLM_SYNC](agents/NOTEBOOKLM_SYNC.md). **Dependencies:** US-026 (pass 1), US-033 (pass 2).

---

### US-035 — Final verification + ship-readiness audit

**Goal.** End-of-Phase-D sweep: every artifact in lockstep, tests green, audit trail intact. Mirrors v2 US-025 pattern.

**Checks.**
- `pytest src/tests/test_degradation_determinism.py -v` → 5/5 green (carried from v2).
- `pytest src/tests/test_matrix.py -v` → green INCLUDING new Phase D assertions (count_276, tags_distinct, shares_phase_b_degrade_config).
- `pytest src/tests/test_build_final_exp_json.py -v` → green INCLUDING new Phase D schema assertions.
- `pytest src/tests/test_update_final_exp.py -v` → green INCLUDING new Phase D rendering assertion.
- `pytest src/tests/test_dashboard.py -v` → green INCLUDING new Phase D UI assertions.
- `pytest src/tests/test_phase_d_plot.py -v` → green (idempotency).
- `python -m src.tests.test_ignores` → green; no weight blobs in git index.
- `mypy src/` on the v3-touched files → no new errors (baseline carry-forward from v2; the 159 pre-existing errors are not Phase D's responsibility).
- `python scripts/update_final_exp.py --check` → exit 0; total cell count = 276.
- `bash scripts/check_ignores.sh` → green.
- Manual: open `artifacts/Final_Exp.html`, filter to Phase D + each treatment chip, confirm 90 rows render with treatment column populated.
- Manual: open `artifacts/Final_Report.pdf` rev2 end-to-end, scan for placeholder text / broken refs / missing figures; §VII present in TOC.
- Ship-readiness audit via Agent tool (subagent_type `general-purpose`): cross-check that [PRD.md](PRD.md) v3 + [README.md](README.md) + [CLAUDE.md](CLAUDE.md) + [progress.txt](progress.txt) + `artifacts/Final_Report.pdf` rev2 all reference the same 90-cell Phase D scope, the same treatment deltas, and the same headline answer to §1.

**Acceptance criteria.**
- [ ] All eight pytest commands exit 0.
- [ ] Ship-readiness agent report identifies no drift between artifacts (in particular: the §1 headline answer must appear verbatim in README Finding #4, CLAUDE.md status block, Final_Report §VII.C, and progress.txt iteration close).
- [ ] `git status` clean; PR draft URL captured for review.

**Owner:** [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-033, US-034.

---

## 9. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **(a) Treatment delta arithmetic mistake** (T3 fails to layer correctly when Optuna JSON already carries `dropout` / `drop_path_rate`) | Low | High (US-029 burns ~52 GPU-h on wrong deltas) | VALIDATOR unit test on `_apply_phase_d_treatment` covering all 3 treatments × 3 model families pre-dispatch. US-028 6-cell pilot catches integration bugs at ~3.5 GPU-h cost. |
| **(b) Mixup/cutmix collator interaction with bf16-mixed** | Medium | High | Pilot (US-028) catches it on 6 cells before the 84-cell sweep. Fallback: `bf16-mixed` → `32-true` for affected cells if a stability issue surfaces (operator decision). |
| **(c) GPU budget overrun beyond 60h** | Low | Medium | Cells are independent and `--skip-existing` respected. Operator can stage T1 → T2 → T3 in separate dispatches. |
| **(d) Phase D L1 trivially equals Phase B L1 under T2-mixup-only on MNIST** | High | Low | Expected and informative; report as-found. MNIST geometric-axis robustness floor (v2 finding #3) should carry forward; T1/T2/T3 expected to be near-no-op on MNIST. |
| **(e) Phase B v2 baseline metrics drift** (someone overwrites `final_B_*` mid-Phase-D campaign) | Low | High | `phase_b_v2_baseline_manifest.json` captured at US-028 pre-flight; US-031 plot script re-verifies SHA-256 before rendering. |
| **(f) Aggregator schema bump breaks v1/v2 cached dashboards** | Low | Low | Frontend gracefully reloads on `SCHEMA_VERSION` mismatch. One-time hard refresh on operator side. |
| **(g) NotebookLM auth state drift** (`.notebooklm/storage_state.json` vs `.notebooklm/profiles/default/storage_state.json` path mismatch) | Medium | Low | Pre-flight `python -m notebooklm status` to verify auth before pass 1; operator re-runs `notebooklm login` if status fails. |

---

## 10. Dependency-Ordered Story Map

```
US-026 (PRD v3 + progress.txt) ───────────────────────────────────┐
                                                                    │
                                                                    ▼
                                            US-027 (docs/phase_d.md)
                                                                    │
                                            ┌───────────────────────┤
                                            ▼                       ▼
                              US-034 pass 1 (NotebookLM PRD)   US-028 (6-cell pilot T3_L3)
                                                                    │  ← HARD GATE: pilot checklist
                                                                    ▼      signed by operator
                                                  US-029 (84-cell sweep)
                                                                    │
                                                                    ▼
                                                  US-029.5 (JSON schema + aggregator)
                                                                    │
                                            ┌───────────────────────┤
                                            ▼                       ▼
                                  US-030 (dashboard)          US-031 (recovery PNGs)
                                            │                       │
                                            └───────────┬───────────┘
                                                        ▼
                                            US-032 (README/CLAUDE/Final_Exp_Report/progress sync)
                                                        │
                                                        ▼
                                            US-033 (Final_Report.pdf rev2 + §VII)
                                                        │
                                                        ▼
                                            US-034 pass 2 (NotebookLM full re-source)
                                                        │
                                                        ▼
                                            US-035 (ship-readiness audit)
```

**Critical path:** US-026 → US-027 → US-028 → US-029 → US-029.5 → (US-030, US-031) → US-032 → US-033 → US-034 pass 2 → US-035.

**Parallelizable forks:**
- US-034 pass 1 (NotebookLM PRD upload) runs in parallel with US-028 pilot — pass 1 only needs the PRD on disk.
- US-030 (dashboard) and US-031 (plot) can run in parallel after US-029.5 lands.

**Wall-clock estimate:** ~3.5 GPU-h pilot + ~52.5 GPU-h sweep + ~8h non-GPU US closures = **~3 calendar days on a single RTX 5070** with overnight runs.

---

## 11. Definition of Done (campaign-level)

- [ ] PRD v3 lands on disk (US-026); v2 referenced via git commit `ced98ec`.
- [ ] [progress.txt](progress.txt) preamble retired the ≤200-line rule; Iteration 47+ entries appended per US closure.
- [ ] [docs/phase_d.md](docs/phase_d.md) published; ~60-100 lines mirroring `docs/phase_c.md` structure.
- [ ] US-028 6-cell pilot signed off in `artifacts/validation/phase_d_pilot_checklist.md` BEFORE US-029 dispatch.
- [ ] `artifacts/validation/phase_b_v2_baseline_manifest.json` captured at US-028 pre-flight and re-verified at US-031.
- [ ] 90/90 Phase D cells `healthy` per §6.3 (or operator-accepted sentinel deferrals); all carry `pipeline_version=2`, non-null `phase_d_treatment`, non-empty `phase_d_deltas`.
- [ ] [src/tools/build_final_exp_json.py](src/tools/build_final_exp_json.py) + [src/tools/final_exp_schema.py](src/tools/final_exp_schema.py) wired for Phase D; `SCHEMA_VERSION` bumped; 4 new pytest assertions green.
- [ ] Dashboard renders 4th Phase D tab + multi-select Treatment chip group + recovery `<details>` strip; 90 Phase D rows visible with treatment column; 186-cell view byte-identical when Phase D absent.
- [ ] 6 per-(model, dataset) recovery PNGs at `artifacts/figures/phase_d_recovery_{m}_{d}.png` + composite summary + LaTeX table at `docs/_autogen/phase_d_recovery_table.tex`; idempotent (byte-identical across re-runs).
- [ ] [artifacts/Final_Report.pdf](artifacts/Final_Report.pdf) rev2 ≥ 10 pages; §VII Phase D appendix present; pdflatex/bibtex exit 0; no `??` placeholders.
- [ ] [README.md](README.md) carries Phase D Recovery subsection + Finding #4; [CLAUDE.md](CLAUDE.md) renamed 186 → 186/276 cells; both reference v3 deltas.
- [ ] NotebookLM passes 1 + 2 logged in `scripts/notebooklm_sync.log`; commit prefix `[notebooklm-sync]`.
- [ ] `pytest`, `test_ignores`, `mypy` (on v3-touched files) green; `git status` clean on `PHASE_D` branch.
- [ ] Ship-readiness audit (US-035) returns zero cross-artifact drift.
- [ ] Headline §1 research question answered verbatim in README Finding #4, CLAUDE.md status block, Final_Report §VII.C, and progress.txt iteration close.

---

## 12. Decisions Locked

1. **Regularization deltas frozen.** T1 = `dropout=0.2` (CNN) / `drop_path_rate=0.2` (TransNeXt); T2 = `mixup_alpha=0.2, cutmix_alpha=0.0`; T3 = T1 ∪ T2 ∪ `cutmix_alpha=1.0`. Source of truth: [run_systematic.py:_phase_d_treatment_deltas](run_systematic.py). No further variation in v3. Operator-confirmed 2026-05-23.
2. **Baseline = Phase B v2.** Every Phase D recovery measurement compares against the matching `final_B_L{l}_{m}_{d}` Phase B v2 cell. SHA-256 of all 30 Phase B v2 `metrics.json` files captured at US-028 pre-flight into `phase_b_v2_baseline_manifest.json` and re-verified at US-031 plot rendering. Operator-confirmed 2026-05-23.
3. **No Phase D re-tune.** Optuna L3 winners stay frozen. Re-tuning under regularization deferred to a hypothetical v4 future-work bullet. Operator-confirmed 2026-05-23.
4. **6-cell pilot scope = T3_L3 × all 6 (model, dataset) pairs.** Pilot is the strongest treatment at the strongest collapse level — fastest signal-to-failure ratio with the broadest per-model delta routing coverage. Operator must sign `phase_d_pilot_checklist.md` BEFORE the 84-cell sweep dispatches. Operator-confirmed 2026-05-23.
5. **2-pass NotebookLM sync.** Pass 1 (PRD v3 only, immediate after US-026 lands) and pass 2 (`full --apply` post-US-033) are both mandatory. No mid-campaign incremental refresh. Operator-confirmed 2026-05-23.
6. **Headline research question is operator-locked.** *"Does regularization recover the Phase B v2 −12.49 pp mean collapse vs v1?"* — must be answered explicitly (per-treatment, with magnitude) in README Finding #4, CLAUDE.md status block, Final_Report §VII.C, and the US-029 outcome paragraph. Operator-confirmed 2026-05-23.
7. **Treatment chip group is multi-select.** T1/T2/T3 chips behave like Model/Dataset/Status chips on the dashboard — union semantics, persisted to `localStorage.final_exp.filters.D`. Operator-confirmed 2026-05-23.
8. **Final_Report rev2 adds §VII Phase D appendix** (does NOT extend §V Discussion inline). Keeps v2 body intact; new content scannable. Operator-confirmed 2026-05-23.
9. **`progress.txt` ≤200-line rule formally retired.** Preamble updated to document the new "append-only journal" policy. File is 4774+ lines and growing; the rule has been broken since iter 17. Operator-confirmed 2026-05-23.
10. **JSON aggregator gap split into US-029.5.** `Final_Exp.json` schema bump + Phase D aggregator wiring are a standalone US between sweep (US-029) and dashboard (US-030). Tests for the aggregator become first-class. Operator-confirmed 2026-05-23.
11. **Sub-agent dispatch carried forward from v1/v2.** Every US owner is a sub-agent defined in [agents/](agents/). MASTER reviews and approves the plan before any sub-agent writes code; sub-agents may not exceed their declared `agents/<NAME>.md` file-system scope without an explicit MASTER scope-extension note. Operator-confirmed 2026-05-23.

---

## 13. v2 History (US-017..US-025 — closed)

Closed in the 2026-05-20 → 2026-05-23 campaign. Full v2 PRD prose preserved at git commit `ced98ec`. Summary:

| Story | Title | Closed |
|---|---|---|
| US-017 | Pipeline v2 refactor + new invariant test (noise + S&P pre-upsample) | 2026-05-20 |
| US-018 | PRD v2 authoring | 2026-05-20 |
| US-019 | RALPH loop `--axes` filter patch (+ bundled `render_cell_thumbs` `--phase`/`--axes`) | 2026-05-20 |
| US-019B | v2 degradation thumbnail preview (operator visual gate) | 2026-05-20 |
| US-020 | Phase B re-run (30 cells) + dashboard refresh | 2026-05-21 |
| US-021 | Phase C `noise` axis re-run (30 cells) + dashboard refresh | 2026-05-22 |
| US-022 | Phase C `salt_pepper` axis re-run (30 cells) + dashboard refresh | 2026-05-23 |
| US-023 | Post-campaign content sync (README + Final_Exp_Report.md + progress.txt) | 2026-05-23 |
| US-024 | IEEEtran scientific report (`docs/Final_Report.tex` → `artifacts/Final_Report.pdf`, 8 pages) | 2026-05-23 |
| US-025 | Final verification + ship-readiness audit | 2026-05-23 |

Headline v2 findings carried forward to v3 baseline:
1. **Phase B v2 mean Δ = −12.49 pp vs v1** across 30 cells (range −0.0 to −32.74 pp). CIFAR-10 mid-range levels (L2-L4) drop hardest; MNIST geometric-axis floor preserved. **This is the recovery target Phase D measures against.**
2. **Phase C noise mean Δ = −5.15 pp vs v1; salt_pepper mean Δ = −0.81 pp vs v1.** Severity scales with pixel-count arithmetic (noise samples every `low_res²` pixel; salt_pepper paints only `low_res² × (salt_pepper × 2)` pixels per image).
3. **CIFAR-10 L5 severity ordering under v2:** `resolution > noise > blur > salt_pepper > saturation`. Noise leapfrogs blur — masked in v1 because post-upsample noise was averaged out by the bicubic kernel.
4. **Universal 3×3-downsample bottleneck** carries forward unchanged (resolution axis unaffected by US-017).

---
