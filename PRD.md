# PRD v4 — Phase B2 + Phase C2 + Full Scientific-Depth Closure

**Project:** p-2026-061 — Object Classification in Low-Resolution THz Imagery
**Branch:** `PHASE_B2` (current working branch; merge target `main`)
**Supersedes:** PRD v3 (Phase D Regularization Sweep + Documentation / Dashboard / NotebookLM Sync, 2026-05-23; closed 2026-05-26 with US-026..US-035 all green). v3 PRD prose preserved in git history at the v3-close commit `898d007`. PRD v4 first-draft (single-phase B2 scope, 12 stories, ~17.5 GPU-h) preserved in this branch's pre-amendment commit; superseded by this revision after the 2026-05-26 operator decision to maximize scientific depth.
**Date:** 2026-05-26
**Author:** Itamar Bahat (ib94)
**Target hardware:** NVIDIA RTX 5070 (12 GiB GDDR7, sm_120 / CUDA 12.8+, bf16-mixed Blackwell tensor cores).

---

## 1. Introduction

The v3 campaign closed 2026-05-26 with US-026..US-035 all green: 90 Phase D regularization cells executed under `PIPELINE_VERSION = 2`, the dashboard grew a 4th Phase D tab + Treatment chip group, `artifacts/Final_Report.pdf` rev2 shipped at ≥ 10 pages with a new §VII Phase D appendix, and the operator-locked headline question — *"Does targeted regularization recover the Phase B v2 −12.49 pp mean collapse vs v1?"* — was answered: **regularization recovers only a small fraction (mean Δ = +0.77 pp under T3 vs the −12.49 pp v2 drop). The collapse is largely a fundamental information bottleneck, not an overfit signal.**

That finding closes the regularization line of attack but opens two siblings. **First**, the Phase B v2 degradation pipeline applies five axes simultaneously (resolution, blur, noise, salt-and-pepper, saturation) and produces a single combined accuracy curve. THz imaging in the real world is grayscale by construction (no chrominance) and has low or zero additive-Gaussian noise after standard preprocessing — the Phase B sweep mixes THz-relevant axes (resolution, blur, salt-and-pepper) with THz-irrelevant axes (color, additive noise). **Second**, every headline number in the v2 + v3 campaign is a single-seed point estimate evaluated on `val_acc`; the IEEE report (`Final_Report.pdf` rev2) cannot defend its claims with variance bounds or a held-out test confirmation.

PRD v4 closes the project with maximum scientific depth. The locked scope:

- **Phase B2 — No-Color / No-Noise THz Protocol** (30 cells, T3 regularization). Strips saturation and additive noise; keeps resolution + blur + salt-and-pepper. The operator-locked Phase B2 question: *"Under T3 regularization, how does the model behave when color and additive noise are absent from the input distribution?"*
- **Phase B2-no-regularization L3 arm** (6 cells). Same DegradeConfig override as Phase B2 but WITHOUT T3 deltas. Provides the pure-protocol Δ vs Phase B1 (no-reg ↔ no-reg) at L3, factoring T3 out of the Phase B2 attribution.
- **Phase C2 — Single-Axis Isolation Under the THz Protocol** (90 cells, T3 regularization). Three axes (resolution / blur / salt-and-pepper) × 5 levels × 3 models × 2 datasets. Attributes Phase B2's degradation to the specific axes that matter under the THz protocol.
- **Multi-seed L3 expansion** (48 training runs, no new tags). Re-trains the 24 L3 headline cells (Phase B1 + Phase D T3 + Phase B2 + Phase B2-nr, all at L3 × 6 (model, dataset) combos) with seeds 43 and 44, giving [mean ± std] variance bands on every headline pp number in the report.
- **Post-training diagnostics**: held-out **test-set evaluation** for the 24 L3 headline cells across all 3 seeds (72 inferences), **confusion matrices** at L5 (24 cells), **calibration / ECE** at L3 + L5 (48 cells), **inference-throughput report** (3 models × per-image timing).
- **`Final_Report.pdf` rev3** — six new appendices (§VIII Phase B2, §IX Phase C2, §X Multi-Seed Variance, §XI Test-Set Confirmation, §XII Diagnostics, §XIII Final Conclusions). Target ≥ 16 pages total. Submission-ready.

**Combined new scope:** 30 + 6 + 90 = **126 new canonical cells** + 48 multi-seed audit runs = **174 new training runs**, ~**100 GPU-h** (~5–6 calendar days on a single RTX 5070 with overnight runs). Total campaign size at close: **402 canonical cells** (186 v2/v3 baselines + 90 Phase D + 30 Phase B2 + 6 Phase B2-nr + 90 Phase C2) + multi-seed audit replicates.

**Naming convention.** In the scientific narrative the original 30-cell Phase B is referred to as **Phase B1**. The underlying on-disk identifier remains `final_B_L{level}_{model}_{dataset}` and the `phase` enum value remains `"B"` — no physical rename of `runs/final/final_B_*` directories. The Phase B1 label is a documentation alias; code paths continue to read `phase == "B"`. New phases introduce new `phase` enum values: `"B2"`, `"B2nr"`, `"C2"`.

---

## 2. Goals (v4 — full scientific-depth scope)

1. **G1 — Phase B2 quantification (Δ_B1→B2 and Δ_D→B2).** All 30 Phase B2 cells produce v4 metrics enabling per-(model, dataset, level) Δ measurements vs both the Phase B1 sibling (combined-effect read) and the Phase D T3 sibling (pure protocol-simplification read at fixed T3 regularization). Outputs: `artifacts/figures/phase_b2_comparison_{model}_{dataset}.png` × 6 + composite + LaTeX table.

2. **G2 — Pure-protocol Δ via Phase B2-no-regularization L3 arm (Δ_B1→B2nr).** Six L3 cells with the B2 DegradeConfig override BUT no T3 deltas, providing a clean (no-reg ↔ no-reg) Δ vs Phase B1 L3. Factors T3 out of the Phase B2 attribution.

3. **G3 — Phase C2 axis attribution (Δ_C2_axis_l vs Phase A).** All 90 Phase C2 cells produce per-axis-severity recovery surfaces under the THz protocol (sat = 0, noise = 0, one of {resolution, blur, salt_pepper} at level L). Establishes which spatial-domain axis dominates Phase B2's drop.

4. **G4 — Multi-seed variance bars on every headline number.** Re-train the 24 L3 headline cells (B1 + D-T3 + B2 + B2nr) with seeds 43 and 44. Every pp number quoted in `Final_Report.pdf` rev3's discussion sections carries an explicit [mean ± std] across 3 seeds.

5. **G5 — Held-out test-set confirmation.** Inference-only pass on a stratified held-out test split for the 24 L3 headline cells × 3 seeds = 72 evaluations. The val_acc → test_acc gap is reported per cell; any gap > 2 pp is flagged in §XI of the report.

6. **G6 — Qualitative diagnostics — confusion matrices at L5 + calibration / ECE at L3 + L5.** 24 confusion-matrix heatmaps (one per L5 cell from B1, D-T3, B2, B2nr) and 48 reliability diagrams + ECE values (one per L3 and L5 cell). Establishes WHICH classes degrade fastest and whether the model is overconfident under degradation.

7. **G7 — Inference-throughput report.** Per-image latency on the RTX 5070 for all 3 model families at 224×224, bf16-mixed precision. Single-paragraph deployment-readiness card in `Final_Report.pdf` §XII.

8. **G8 — Lockstep artifact freshness.** After every RALPH pass the dashboard, `Final_Exp.json` / `.html` / `.md`, the new dashboard thumbs, and the comparison PNGs regenerate in the same operator pass — extending v3 §7's artifact-freshness rule to all new phases.

9. **G9 — `Final_Report.pdf` rev3 with six new appendices PLUS Final Conclusions chapter.** §VIII Phase B2 + §IX Phase C2 + §X Multi-Seed Variance + §XI Test-Set Confirmation + §XII Diagnostics (confusion + calibration + throughput) + §XIII Final Conclusions. Target ≥ 16 pages total. pdflatex/bibtex exit 0, zero `??` placeholders.

10. **G10 — Two-pass NotebookLM alignment.** Pass 1 uploads PRD v4 immediately. Pass 2 (end-of-campaign, post-US-051) re-sources the full curated manifest including new `docs/phase_b2.md`, `docs/phase_c2.md`, and `Final_Report.pdf` rev3.

11. **G11 — Test coverage for the full new scope.** New pytest assertions covering: matrix counts (276 / 282 / 372 / 402 across flag combinations), tag uniqueness across all 7 prefixes, B2 + C2 DegradeConfig overrides, per-sample determinism under each override, treatment-field round-trip for B2 / B2nr / C2, multi-seed tag-suffix routing, dashboard rendering for all new tabs.

12. **G12 — Submission readiness.** By the time US-053 closes, the project state is: 402 canonical cells healthy + 48 multi-seed audit runs healthy, all artifacts in lockstep, IEEE report carrying a closing chapter with variance bars and held-out confirmation, NotebookLM aligned, and the operator has a single "ready to defend" checkpoint to draw the 26/07/2026 submission from. The deadline gives ~2 calendar months of buffer after US-053 closes — comfortable margin for any post-submission revisions.

---

## 3. Non-Goals (v4)

- **No Optuna re-tune for Phase B2 / B2-nr / C2 or multi-seed.** All new cells reuse the frozen Phase B Optuna L3 winners from `artifacts/best_hparams/*.json`. T3 deltas (when applicable) layer on top identically to Phase D. Re-tuning under the new distributions is explicitly deferred to a hypothetical v5 post-submission scope.
- **No physical rename of `runs/final/final_B_*` directories.** Phase B1 = `phase == "B"` in code; Phase B1 is a documentation alias only. The Phase D recovery artifacts (`phase_b_v2_baseline_manifest.json`, `phase_d_recovery_*.png`, `Final_Report.pdf` rev2 §VII) continue to reference `final_B_*` directly with no edits required.
- **No pipeline-version change.** `PIPELINE_VERSION = 2` stays. `degradation_levels_hash` does NOT rotate. All new cells share v3's exact per-sample deterministic seeding (`seed = idx + SEED_OFFSET_VAL`); only per-cell DegradeConfig overrides and (for multi-seed) `pl.seed_everything(seed, workers=True)` differ from the v3 contract.
- **No model swaps.** ResNet50 / DenseNet121 / TransNeXt-tiny stay.
- **No new datasets.** CIFAR-10 + MNIST remain the only two.
- **No new degradation axes.** The 5-axis × 5-level matrix is unchanged; B2 / C2 zero existing axes rather than introducing new ones.
- **No re-run of Phase A / B1 / C / D at the canonical seed = 42.** The 276 prior cells stay as-is at seed 42. **Exception:** Phase B1 L3 (6 cells) and Phase D T3 L3 (6 cells) get **additional** runs at seeds 43 and 44 in US-042 — these are additive multi-seed audits, not re-runs of the canonical cell.
- **No multi-treatment Phase B2 / Phase C2 sweep.** B2 and C2 are locked to T3 only (operator decision — §12 lock 12 + 17). The Δ_D→B2 and Δ_A→C2 comparisons at T3 give the cleanest single-treatment reads; T1-only / T2-only sweeps under the new protocols are deferred.
- **No archival of v3 PRD or v4 first-draft into the tree.** Earlier prose lives only in git history at the commits referenced in the header.
- **No deviation from seed = 42 for the canonical reported number.** When a cell exists at both seed = 42 and additional seeds, the operator's headline pp number is the seed = 42 value with [mean ± std across {42, 43, 44}] reported as the variance band. This keeps every published pp number traceable to a single deterministic run.

---

## 4. Phase B2 / B2-nr / C2 / Multi-Seed Specification

**Pipeline-version constant unchanged:** `PIPELINE_VERSION = 2`. v4 introduces **NO** step-order changes to the data pipeline. The v2 §4 pipeline diagram and v3 §4 carry forward verbatim. The only per-phase behavior change is a DegradeConfig override (Phase B2 / B2-nr / C2) and an optional `--seed` CLI knob (multi-seed audits).

### 4.1 Phase B2 DegradeConfig override

New helper in [`src/data/degrade.py`](src/data/degrade.py): `degrade_config_for_b2(level: int, out_size: int = 224) -> DegradeConfig`. Returns a `DegradeConfig` populated from `DEGRADATION_LEVELS[level]` with two explicit overrides:

- `saturation = 0.0` (full grayscale via deterministic lerp)
- `noise_std = 0.0` (additive-Gaussian noise step skipped by the `> 0` guard in `degrade.py`)

All other axes flow through unchanged. Byte-deterministic by construction.

### 4.2 Phase B2-no-regularization arm

Same DegradeConfig as Phase B2 (`degrade_config_for_b2(level=3, ...)`) but the T3 deltas are **NOT** applied. Cells route through `run_systematic.py` with `phase == "B2nr"` recognized as a no-treatment carrier: `phase_b2nr_treatment = None`, `phase_b2nr_deltas = {}` (empty dict written explicitly so the schema field stays non-null). Only 6 cells total — L3 × 3 models × 2 datasets.

### 4.3 Phase C2 DegradeConfig override

New helper: `degrade_config_for_c2(level: int, axis: str, out_size: int = 224) -> DegradeConfig`. Returns a `DegradeConfig` with:

- The named `axis` ∈ {`"resolution"`, `"blur"`, `"salt_pepper"`} at its `DEGRADATION_LEVELS[level]` value
- All other non-{saturation, noise_std} axes at `IDENTITY_VALUES`
- `saturation = 0.0` and `noise_std = 0.0` (THz protocol invariant — same as B2)

Phase C2 is **C-isolation under the B2 protocol**. The legacy Phase C (v1/v2) sets all inactive axes to identity AND keeps `saturation` and `noise_std` free at their identity values (sat = 1.0, noise = 0.0 already at identity in v3's `IDENTITY_VALUES`). Phase C2 overrides saturation to 0.0 explicitly; noise_std stays at 0.0 (identical to the identity value, but stamped explicitly into the override for symmetry). The three active axes are `resolution` / `blur` / `salt_pepper` — the THz-relevant spatial-domain axes that Phase B2 isolated.

### 4.4 T3 reuse contract (B2 + C2)

Phase B2 AND Phase C2 inherit T3 regularization (CNN: `dropout=0.2` + `mixup_alpha=0.2` + `cutmix_alpha=1.0`; TransNeXt: `drop_path_rate=0.2` + `mixup_alpha=0.2` + `cutmix_alpha=1.0`). Identical to Phase D T3 at the hparam level. The dispatcher merges T3 deltas into the frozen Phase B Optuna L3 winners via the existing `_phase_d_treatment_deltas("T3", spec.model)` route, with a new helper `_apply_b2_or_c2_treatment(spec, hparams)` stamping `phase_b2_treatment` / `phase_c2_treatment = "T3"` and `phase_b2_deltas` / `phase_c2_deltas = ...` onto the hparams blob. Phase B2-nr skips this routing — its hparams are the bare Optuna winners.

### 4.5 Multi-seed audit

Operator-locked scope: re-train the **24 L3 headline cells** (Phase B1 L3 × 6 + Phase D T3 L3 × 6 + Phase B2 L3 × 6 + Phase B2-nr L3 × 6) at seeds 43 and 44, in addition to the canonical seed = 42 run. **48 additional training runs** total. Tag scheme: base tag with `_seed{N}` suffix when seed != 42 (e.g., `final_B2_L3_resnet50_cifar10_seed43`).

`run_systematic.py` accepts a new `--seed N` CLI arg; when omitted, defaults to 42 (carries the v3 contract). `pl.seed_everything(N, workers=True)` honors the override. Tag construction inside the dispatcher appends `_seed{N}` when N != 42; the tag-suffix logic is the only multi-seed code path beyond the CLI plumbing.

Multi-seed `metrics.json` carries `seed: int` as a first-class field. The aggregator (US-046) groups by base tag and reports `val_acc_mean` and `val_acc_std` across seeds in `Final_Exp.json`.

### 4.6 Tag schemes (full set after v4)

| Phase | Tag pattern | Count | DegradeConfig source |
|---|---|---|---|
| A | `final_clean_{m}_{d}` | 6 | clean |
| B (B1) | `final_B_L{l}_{m}_{d}` | 30 | `degrade_config_for(l)` (all 5 axes at L) |
| B2 | `final_B2_L{l}_{m}_{d}` | 30 | `degrade_config_for_b2(l)` (sat = noise = 0) |
| B2-nr | `final_B2nr_L3_{m}_{d}` | 6 | `degrade_config_for_b2(3)` (same as B2 L3) |
| C | `final_C_L{l}_{axis}_{m}_{d}` | 150 | `degrade_config_for(l, axis=axis)` (legacy) |
| C2 | `final_C2_L{l}_{axis}_{m}_{d}` | 90 | `degrade_config_for_c2(l, axis)` (sat = noise = 0; axis ∈ {resolution, blur, salt_pepper}) |
| D | `final_D_T{x}_L{l}_{m}_{d}` | 90 | `degrade_config_for(l)` (all 5 axes at L; T1/T2/T3 deltas) |
| **Multi-seed audits** | `<base_tag>_seed{N}` for N ∈ {43, 44} on the 24 L3 headline cells | 48 | inherits base-tag DegradeConfig |

**Canonical campaign total: 6 + 30 + 30 + 6 + 150 + 90 + 90 = 402 cells** (counting each unique base tag exactly once). Multi-seed audits add 48 training runs but no new cells.

### 4.7 Treatment field summary

| Phase | `treatment` field | Notes |
|---|---|---|
| A | `None` | clean baselines, no regularization |
| B (B1) | `None` | no regularization (Optuna L3 winners only) |
| B2 | `"T3"` | T3 deltas layered on Optuna winners |
| B2-nr | `None` | no regularization (clean read of B2 protocol) |
| C | `None` | legacy single-axis isolation, no regularization |
| C2 | `"T3"` | T3 deltas layered on Optuna winners |
| D | `"T1"` / `"T2"` / `"T3"` | per cell, set at dispatch time |

### 4.8 Baseline manifest lock (extended)

At US-039 pre-flight (Phase B2 pilot), capture SHA-256 of every reference `metrics.json` into `artifacts/validation/phase_b2_baseline_manifest.json` (or extend the v3 `phase_b_v2_baseline_manifest.json` in place). Coverage:

- 30 Phase B1 cells (already in v3's manifest)
- 30 Phase D T3 cells (newly captured at US-039)
- 6 Phase A clean cells (newly captured at US-039 — needed as the Phase C2 baseline for Δ_A→C2 readings)

**66 total entries.** Re-verify hashes at US-048 plot rendering.

### 4.9 Determinism contract preserved + extended

Carried from v3 §4: `SEED_OFFSET_TRAIN` / `SEED_OFFSET_VAL` constants frozen; per-sample `torch.Generator` seeded by `idx + SEED_OFFSET_*`. Phase B2 / B2-nr / C2 add **NO** additional global RNG draws beyond mixup/cutmix label-sampling (already wired in v3). Multi-seed audits change `pl.seed_everything(N, workers=True)` but do NOT touch the per-sample degradation seed offsets — the same val pixels are byte-identical across seeds 42 / 43 / 44, only the training-time RNG (weight init, mixup sampling, augmentation order) differs.

US-038 adds three new groups to [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py):
- `b2_cifar10_l3` and `b2_mnist_l3` — byte-identical val pixels at out_size = 224 under the B2 override; assert `saturation == 0.0` and `noise_std == 0.0` in every cell's DegradeConfig.
- `c2_cifar10_l3_resolution` and `c2_mnist_l3_resolution` — same assertions plus axis-isolation check (only `low_res` differs from `IDENTITY_VALUES`).
- `multi_seed_val_byte_identical_42_43_44` — verifies val pixels are byte-identical across `pl.seed_everything(N)` for N ∈ {42, 43, 44} (the per-sample seed offsets are independent of the global seed).

---

## 5. Environment & Hardware

Unchanged from PRD v2 §5 / v3 §5. RTX 5070 / cu128 / bf16-mixed. No new dependencies; mixup/cutmix via `timm.data.Mixup` already in the requirements lock. Post-training diagnostics (US-045) use `torchmetrics.CalibrationError` (ECE) — already pulled in by Lightning.

**Reproducibility audit trail (extended).** Each new `metrics.json` must round-trip:
- `pipeline_version == 2`
- `phase` ∈ {`"B2"`, `"B2nr"`, `"C2"`, …} per cell type
- For B2 / C2: `treatment == "T3"`, non-empty `phase_b2_deltas` / `phase_c2_deltas`
- For multi-seed runs: `seed` ∈ {42, 43, 44}; tag carries `_seed{N}` suffix when N != 42
- `degrade_config` carrying the expected per-phase overrides

The campaign-level integrity check at US-053 confirms total reads 402 canonical cells + 48 multi-seed audit runs.

---

## 6. RALPH Loop (carried over from v1/v2/v3)

§6.1 through §6.6 unchanged from PRD v1/v2/v3.

**New §6.7 — Phase B2 dispatch.** `scripts\run_ralph_loop.py --phase B2` → `include_phase_b2=True`.

**New §6.8 — Phase B2-nr dispatch.** `scripts\run_ralph_loop.py --phase B2nr` → `include_phase_b2nr=True`. Filters to L3 only (the only level B2-nr covers).

**New §6.9 — Phase C2 dispatch.** `scripts\run_ralph_loop.py --phase C2` → `include_phase_c2=True`. Optional `--axes resolution,blur,salt_pepper` for staged dispatch (mirrors v2/v3 `--axes`).

**New §6.10 — Multi-seed dispatch.** `scripts\run_ralph_loop.py --phase multiseed --seeds 43,44 --cells <tag_list>` re-dispatches the named cells at seeds 43 and 44. Each (tag, seed) pair produces a `<tag>_seed{N}` directory under `runs/final/`.

**§6.4 retry policy carries forward verbatim.** Track per-phase retry-acceptance tally; report in campaign summary.

---

## 7. Constraints & Guardrails

All v2 + v3 guardrails carried forward:

- **Weight-privacy contract** (`.gitignore` + `.claudeignore` + `scripts/check_ignores.sh`).
- **`pl.seed_everything(42, workers=True)` default lock.** Multi-seed audits override with `--seed 43` / `--seed 44`; canonical reported pp numbers are still the seed = 42 cells (see §3 non-goal).
- **`--plan final --mode pilot` rejected** at the full-plan dispatch level.
- **Fair-comparison invariant** carries forward per-phase.
- **Artifact-freshness rule** carries forward.
- **Seed-stability invariant** for the per-sample degradation seed offsets carries forward.
- **Baseline-comparability invariant** carries forward; extended to cover the new Phase A baseline for Δ_A→C2.
- **NotebookLM 2-pass discipline.** Pass 1 (PRD v4 upload) immediate. Pass 2 (full re-source) post-US-051.

**New (v4 only):**

- **Multi-phase baseline manifest lock** (§4.8): 66 reference `metrics.json` SHA-256s captured at US-039.
- **Three HARD GATE pilot checklists.** US-039 (Phase B2 6-cell pilot) gates US-040. US-043 (Phase C2 6-cell pilot) gates US-044. No 24-cell or 84-cell sweep dispatches without operator signature.
- **Tag-prefix collision-safety assertions for all seven prefixes.** `run_all_phases.py` rejects any non-matching tag in a `--phase X` matrix. The legacy Phase B check must use `startswith("final_B_L")` (the L-prefix distinguishes it from `final_B2_*` and `final_B2nr_*`).
- **Docs-only Phase B → Phase B1 rename discipline** (carries from v4 first-draft). SYNCHRONIZER cross-checks in US-049.
- **Multi-seed canonicalization rule.** Seed = 42 is the canonical run; seeds 43 / 44 are audit replicates. Every published pp number in the report is the seed = 42 value (canonical) with [mean ± std across {42, 43, 44}] as the variance band. Mean-of-3-seeds is NOT used as the headline number — keeps every published value traceable to a deterministic run.
- **Test-set inference uses a stratified held-out split.** US-045 defines the split deterministically (operator chooses 1000 samples per class via a fixed seed = 99 split on the dataset's official test partition). The split index file `artifacts/validation/test_split_indices.json` is committed at US-045 pre-flight and frozen for the audit.
- **Confusion-matrix + calibration consume saved logits where available; re-inference where not.** US-038 extends `THzClassifier.predict_step` to log per-batch logits to disk under `runs/final/<tag>/logits/` IF the cell is a multi-seed audit target or a diagnostics target. Older v2/v3 cells without saved logits get a re-inference pass in US-045 (~0.5 GPU-h total for the 24 L5 cells).
- **Final-conclusions chapter is non-optional and is a US-053 gate.**

**Multi-agent protocol** carried forward from v1/v2/v3.

---

## 8. User Stories (v4 active set — full scientific-depth scope)

Eighteen stories, dependency-ordered. US-036 (this PRD) closed 2026-05-26 (Iteration 49); US-037 through US-053 are unstarted and execute in follow-up RALPH sessions. Every owner below is a sub-agent defined in [agents/](agents/).

| Story | Title | Owner | Status | GPU-h |
|---|---|---|---|---|
| **US-036** | PRD v4 authoring + progress.txt v4-kickoff entry (this document) | [MASTER](agents/MASTER.md) | ✅ CLOSED 2026-05-26 | 0 |
| **US-037** | `docs/phase_b2.md` + `docs/phase_c2.md` rationale documents | [REPORTER](agents/REPORTER.md) + [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) (review) | ✅ CLOSED 2026-05-26 | 0 |
| **US-038** | Code wiring: cells.py + matrix.py + degrade.py + run_systematic.py + run_all_phases.py + multi-seed CLI + logit logging + determinism test extensions | [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) + [DESIGNER](agents/DESIGNER.md) + [VALIDATOR](agents/VALIDATOR.md) | ✅ CLOSED 2026-05-26 (code+tests; mypy + pilot deferred) | 0 |
| **US-039** | Phase B2 6-cell pilot (B2_L3 × all 6 (m, d) pairs) + extended baseline manifest snapshot (66 refs) | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending | ~2.5 |
| **US-040** | Phase B2 full sweep (remaining 24 cells) + post-pass artifact refresh | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending | ~15 |
| **US-041** | Phase B2-no-regularization L3 arm (6 cells, no T3 deltas) | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending | ~3.5 |
| **US-042** | Multi-seed L3 expansion (48 runs across B1 + D-T3 + B2 + B2-nr at seeds 43, 44) | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending | ~24 |
| **US-043** | Phase C2 6-cell pilot (C2_L3_resolution × all 6 (m, d) pairs) | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending | ~3.5 |
| **US-044** | Phase C2 full sweep (remaining 84 cells across all 3 axes) | [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending | ~48 |
| **US-045** | Post-training diagnostics: held-out test-set inference (72 evals) + confusion matrices (24 L5 cells) + calibration / ECE (48 cells at L3 + L5) + inference throughput (3 models) | [DESIGNER](agents/DESIGNER.md) + [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending | ~3.5 |
| **US-046** | `Final_Exp.json` schema + aggregator wiring for B2 + B2-nr + C2 + multi-seed (group-by base tag, emit val_acc_mean / val_acc_std) | [DESIGNER](agents/DESIGNER.md) + [VALIDATOR](agents/VALIDATOR.md) | ✅ CLOSED 2026-05-26 (code+tests; mypy deferred) | 0 |
| **US-047** | Dashboard updates: 7 phase tabs (A / B (B1) / B2 / B2-nr / C / C2 / D) + multi-seed variance indicator + diagnostic galleries (confusion + calibration) + throughput card | [DESIGNER](agents/DESIGNER.md) | ⏳ pending | 0 |
| **US-048** | Comparison plots + LaTeX tables (Δ_B1→B2, Δ_D→B2, Δ_B1→B2nr, Phase C2 axis attribution, multi-seed [mean ± std] bars, confusion matrices, reliability diagrams, throughput chart) | [DESIGNER](agents/DESIGNER.md) (plots) + [REPORTER](agents/REPORTER.md) (LaTeX review) | ⏳ pending | 0 |
| **US-049** | Post-campaign content sync (README + CLAUDE.md + Final_Exp_Report.md + progress.txt; B → B1 rename discipline; new Findings #5–#8) | [SYNCHRONIZER](agents/SYNCHRONIZER.md) + [LIBRARIAN](agents/LIBRARIAN.md) + [REPORTER](agents/REPORTER.md) | ⏳ pending | 0 |
| **US-050** | `Final_Report.pdf` rev3 — appendices §VIII Phase B2 + §IX Phase C2 + §X Multi-Seed Variance + §XI Test-Set Confirmation + §XII Diagnostics (confusion + calibration + throughput) | [REPORTER](agents/REPORTER.md) + [DESIGNER](agents/DESIGNER.md) (heatmap helper) | ⏳ pending | 0 |
| **US-051** | `Final_Report.pdf` rev3 — §XIII Final Conclusions chapter (v2 + v3 + v4 closing narrative) | [REPORTER](agents/REPORTER.md) | ⏳ pending | 0 |
| **US-052** | NotebookLM 2-pass sync (pass 1 immediate / pass 2 post-US-051) | [NOTEBOOKLM_SYNC](agents/NOTEBOOKLM_SYNC.md) | ⏳ pending | 0 |
| **US-053** | Final verification + ship-readiness audit + submission close | [VALIDATOR](agents/VALIDATOR.md) | ⏳ pending | 0 |
| **TOTAL** | | | | **~100 GPU-h** |

---

### US-036 — PRD v4 authoring + progress.txt v4-kickoff entry

**Goal.** Replace PRD v3 with a v4 PRD reflecting the closed v3 campaign + the maximal-depth Phase B2 / B2-nr / C2 / multi-seed / diagnostics follow-up + project-closure narrative. v3 PRD content preserved only in git history. v4 first-draft (single-phase B2 scope, ~17.5 GPU-h) also preserved in git history at the pre-amendment commit.

**Acceptance criteria.**
- [x] [PRD.md](PRD.md) rewritten end-to-end; eighteen v4 user stories defined; v3 stories US-026..US-035 referenced as "closed; see git history" in §13.
- [x] PRD v4 enumerates the full new spec (§4 — B2 + B2-nr + C2 + multi-seed + diagnostics), goals (§2), non-goals (§3), constraints (§7), dependency-ordered story map (§10), and §12 decisions locked.
- [x] [progress.txt](progress.txt) Iteration N entry appended summarizing the PRD v4 authoring session and the locked operator decisions (§12 entries 12–18).

**Owner:** [MASTER](agents/MASTER.md).

---

### US-037 — `docs/phase_b2.md` + `docs/phase_c2.md` rationale documents

**Goal.** Two new docs mirroring `docs/phase_d.md`'s structure. Document the headline research questions, the DegradeConfig overrides, the T3 reuse contracts, the tag schemes + collision-safety notes, the hash-invariance notes, and the baseline-comparison rules.

**Deliverables.**
- New file [`docs/phase_b2.md`](docs/phase_b2.md), ~80–120 lines. Covers Phase B2 (T3) AND Phase B2-nr in adjacent subsections (the latter is a 6-cell L3-only arm; documenting both in one file keeps the THz-protocol scope co-located).
- New file [`docs/phase_c2.md`](docs/phase_c2.md), ~80–120 lines. Covers Phase C2 axis attribution; references `docs/phase_c.md` for the legacy Phase C single-axis isolation rationale.
- One sentence in [README.md](README.md) §"Implementation Status" pointing to each (LIBRARIAN scope; defer full README sync to US-049).

**Acceptance criteria.**
- [x] Both new files present with the section structure listed in §4 of this PRD.
- [x] DegradeConfig override values match [`src/data/degrade.py`](src/data/degrade.py) byte-for-byte (CI substring assertions).
- [x] T3 delta values match [`run_systematic.py:_phase_d_treatment_deltas`](run_systematic.py) byte-for-byte.
- [x] Explicit `PIPELINE_VERSION = 2` carry-forward note present in each.
- [x] Every claim cited to a file path; no invented code paths or values.

**Owner:** [REPORTER](agents/REPORTER.md) (authors). [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) reviews. **Dependencies:** US-036.

---

### US-038 — Code wiring (B2 + B2-nr + C2 + multi-seed + logit logging + tests)

**Goal.** Wire all four new phases AND the multi-seed CLI + logit-logging hook end-to-end through the cell enumerator, matrix builder, degrade module, dispatcher, and test suite. Single coordinated landing; may be staged as multiple commits within one PR for review.

**Deliverables.**

**(a) Degrade helpers.** [`src/data/degrade.py`](src/data/degrade.py):
- `degrade_config_for_b2(level: int, out_size: int = 224) -> DegradeConfig` — sat = 0, noise_std = 0 override.
- `degrade_config_for_c2(level: int, axis: str, out_size: int = 224) -> DegradeConfig` — sat = 0, noise_std = 0 + one of {resolution, blur, salt_pepper} at level L, others at identity.

**(b) Cell enumeration.** [`src/experiments/cells.py`](src/experiments/cells.py):
- New constants `PHASE_B2_TREATMENT = "T3"`, `PHASE_C2_TREATMENT = "T3"`, `PHASE_C2_AXES = ("resolution", "blur", "salt_pepper")`.
- `EXPECTED_COUNTS_WITH_B2 / _C2 / _ALL` dicts; `EXPECTED_TOTAL_WITH_ALL = 402` (186 + 90 + 30 + 6 + 90).
- `iter_cells(include_phase_d=False, include_phase_b2=False, include_phase_b2nr=False, include_phase_c2=False)` — emits the appropriate tags in canonical order (A → B → B2 → B2-nr → C → C2 → D).
- New `phase_b2_present_on_disk`, `phase_b2nr_present_on_disk`, `phase_c2_present_on_disk` helpers mirroring `phase_d_present_on_disk`.

**(c) Matrix builder.** [`src/experiments/matrix.py`](src/experiments/matrix.py):
- `build_final_matrix(out_size=224, include_phase_d=False, include_phase_b2=False, include_phase_b2nr=False, include_phase_c2=False)` — routes B2 / B2-nr cells through `degrade_config_for_b2(...)` and C2 cells through `degrade_config_for_c2(...)`.
- Module docstring updated with the new phase count math.

**(d) Dispatcher.** [`run_systematic.py`](run_systematic.py):
- New `--seed N` CLI arg (default 42). When N != 42, tag construction appends `_seed{N}`; `pl.seed_everything(N, workers=True)` is honored.
- Treatment routing recognizes `phase ∈ {"B2", "C2"}` as T3 carriers (delegates to existing `_phase_d_treatment_deltas("T3", spec.model)`); `phase == "B2nr"` skips treatment routing entirely; new helper `_apply_b2_or_c2_treatment(spec, hparams)` stamps `phase_b2_*` / `phase_c2_*` fields.
- `seed` written as a first-class field in `metrics.json`.

**(e) Top-level driver.** [`run_all_phases.py`](run_all_phases.py):
- `FINAL_PHASES = ("A", "B", "B2", "B2nr", "C", "C2", "D")` (canonical order).
- `--phase {B2, B2nr, C2}` flags route the corresponding `include_phase_*` kwargs.
- Tag-prefix assertions extended for all seven prefixes (use `startswith("final_B_L")` for the legacy Phase B check).

**(f) RALPH loop.** [`scripts/run_ralph_loop.py`](scripts/run_ralph_loop.py):
- Accept `--phase {B2, B2nr, C2, multiseed}`.
- For `--phase multiseed`: require `--seeds 43,44` and `--cells <tag_list>`; iterate (cell, seed) products.
- For `--phase C2`: optional `--axes resolution,blur,salt_pepper` filter.

**(g) Tracker.** [`scripts/update_final_exp.py`](scripts/update_final_exp.py):
- New `_phase_b2_table`, `_phase_b2nr_table`, `_phase_c2_table` renderers gated on the corresponding `*_present_on_disk` helpers.
- `--check` arithmetic: total = 402 canonical cells when all flags present. Multi-seed audit runs counted separately under a new "Multi-Seed Audits" section.

**(h) Thumbnail renderer.** [`src/tools/render_cell_thumbs.py`](src/tools/render_cell_thumbs.py):
- Accept `--phase {B2, B2nr, C2}`; thumb pipeline reads the corresponding DegradeConfig and renders thumbs at `artifacts/dashboard_thumbs/final_{B2,B2nr,C2}_*.png`.

**(i) Logit logging.** [`src/lightning/module.py`](src/lightning/module.py):
- Extend `THzClassifier.predict_step` (or add `THzClassifier.on_predict_batch_end` hook) to dump per-batch logits + labels + sample indices to `runs/final/<tag>/logits/test_batch_{idx:04d}.pt` when `self.hparams.get("log_logits", False) == True`.
- New `--log-logits` CLI flag on `run_systematic.py` defaulting to False; auto-enabled for cells flagged as multi-seed or diagnostics targets.

**(j) Determinism tests.** [`src/tests/test_degradation_determinism.py`](src/tests/test_degradation_determinism.py):
- New groups: `b2_cifar10_l3`, `b2_mnist_l3`, `c2_cifar10_l3_resolution`, `c2_mnist_l3_resolution`, `multi_seed_val_byte_identical_42_43_44`. Acceptance: same intra/inter/batch/histogram equalities as existing groups, plus the per-phase override assertions.

**(k) Matrix tests.** [`src/tests/test_matrix.py`](src/tests/test_matrix.py):
- `test_b2_count_30`, `test_b2nr_count_6`, `test_c2_count_90`, `test_all_phases_total_402`.
- `test_b2_tags_distinct_from_b1` / `test_c2_tags_distinct_from_c`.
- `test_b2_degrade_config_overrides` / `test_c2_degrade_config_overrides_and_axis_isolation`.
- `test_b2_treatment_is_t3` / `test_c2_treatment_is_t3` / `test_b2nr_treatment_is_none`.
- `test_resolution_blur_sp_match_level` for both B2 and C2.
- `test_multi_seed_tag_suffix` — assert `_seed43` / `_seed44` suffixes are appended correctly.

**Acceptance criteria.**
- [x] All new pytest assertions green (≥ 15 new tests).
- [x] `python scripts/update_final_exp.py --check` exits 0 on a clean tree; reports total 402 canonical when all phases present.
- [ ] `mypy src/` reports no new type errors. *(deferred — not run; operator audit on next pass)*
- [x] `python -c "from src.experiments.matrix import build_final_matrix; print(len(build_final_matrix(include_phase_d=True, include_phase_b2=True, include_phase_b2nr=True, include_phase_c2=True)))"` prints `402`.
- [ ] `run_systematic.py --cell-tag final_B2_L3_resnet50_cifar10 --seed 43 --mode pilot` writes to `runs/final/final_B2_L3_resnet50_cifar10_seed43/`. *(deferred — operator-side GPU pilot dispatch)*

**Owner:** [DATA_ARCHITECT](agents/DATA_ARCHITECT.md) (cells / matrix / degrade), [DESIGNER](agents/DESIGNER.md) (`src/tools/` + `scripts/` + logit hook), [VALIDATOR](agents/VALIDATOR.md) (test assertions). **Dependencies:** US-036, US-037.

---

### US-039 — Phase B2 6-cell pilot + extended baseline manifest snapshot

**Goal.** Validate the Phase B2 pipeline end-to-end on 6 cells (B2_L3 × all 6 (m, d) pairs) BEFORE dispatching the full 24-cell remainder. Concurrently, capture SHA-256 of every reference `metrics.json` needed for downstream comparisons: 30 Phase B1 + 30 Phase D T3 + 6 Phase A clean = **66 entries** into `artifacts/validation/phase_b2_baseline_manifest.json` (or extension of v3 manifest).

**Pre-flight.**
1. `PIPELINE_VERSION == 2`.
2. Determinism + matrix + ignores suites green.
3. Capture extended manifest (66 references).
4. `runs/final/final_B2_L3_*/` empty.

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_B2_L3_resnet50_cifar10 --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_B2_L3_resnet50_mnist --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_B2_L3_densenet121_cifar10 --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_B2_L3_densenet121_mnist --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_B2_L3_transnext_tiny_cifar10 --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_B2_L3_transnext_tiny_mnist --mode pilot
```

**Operator visual gate.** Sign `artifacts/validation/phase_b2_pilot_checklist.md` confirming all 6 `metrics.json` carry the expected fields + the rendered thumb is visibly grayscale with no additive-noise grain. HARD GATE for US-040.

**Pilot cleanup.** Delete `runs/final/final_B2_L3_*/` after sign-off.

**Acceptance criteria.**
- [ ] Extended baseline manifest committed with 66/66 SHA-256 entries.
- [ ] 6/6 pilot `metrics.json` written with expected fields.
- [ ] Zero §6.3 sentinels.
- [ ] Operator signs `phase_b2_pilot_checklist.md`. **HARD GATE.**
- [ ] Pilot dirs cleaned.

**GPU budget.** ~2.5 GPU-h.

**Owner:** [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-036, US-037, US-038.

---

### US-040 — Phase B2 full sweep (remaining 24 cells)

**Goal.** Re-run the 24 non-pilot Phase B2 cells at full convergence (60 epochs / patience 10 / bf16-mixed / Optuna L3 winners + T3 deltas + B2 DegradeConfig override). The 6 pilot L3 cells also re-train under full FT (matches v3 US-029 pattern).

**Pre-flight.**
1. **HARD GATE — US-039 must be closed with operator signature.**
2. Pilot dirs cleaned.
3. `PIPELINE_VERSION == 2`; `SEED_OFFSET_*` unchanged.

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_ralph_loop.py --phase B2
```

**Post-pass.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\update_final_exp.py
.\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase B2
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_exp_json
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_dashboard
```

**Acceptance criteria.**
- [ ] 30/30 cells `healthy` per §6.3 (or operator-accepted deferrals).
- [ ] All 30 `metrics.json` carry `pipeline_version: 2`, `phase: "B2"`, `treatment: "T3"`, B2 DegradeConfig override.
- [ ] [Final_Exp.md](Final_Exp.md) reflects 30 Phase B2 rows; total cell count = 306 (without C2/B2nr yet).
- [ ] 30 dashboard thumbs rendered.

**v4 outcome (filled at close).** Mean Δ_B1→B2 and Δ_D→B2 across the 30 Phase B2 cells, broken out by dataset and level. Headline answer to the §1 Phase B2 question.

**GPU budget.** ~15 GPU-h.

**Owner:** [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-039.

---

### US-041 — Phase B2-no-regularization L3 arm (6 cells, no T3 deltas)

**Goal.** Train 6 L3 cells with the B2 DegradeConfig override but WITHOUT T3 regularization deltas. Provides the pure (no-reg ↔ no-reg) Δ_B1→B2nr at L3, factoring T3 out of the Phase B2 attribution.

**Pre-flight.**
1. US-040 closed (Phase B2 sweep landed — confirms B2 pipeline correctness).
2. `runs/final/final_B2nr_L3_*/` empty.

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_ralph_loop.py --phase B2nr
```

(Equivalent to dispatching the 6 cells individually with `--cell-tag final_B2nr_L3_{m}_{d}` at full FT.)

**Post-pass.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\update_final_exp.py
.\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase B2nr
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_exp_json
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_dashboard
```

**Acceptance criteria.**
- [ ] 6/6 cells `healthy`.
- [ ] All 6 `metrics.json` carry `pipeline_version: 2`, `phase: "B2nr"`, `treatment: null` (or absent), `phase_b2nr_deltas: {}`, B2 DegradeConfig override.
- [ ] [Final_Exp.md](Final_Exp.md) reflects 6 Phase B2-nr rows.
- [ ] 6 dashboard thumbs rendered (visually identical to Phase B2 L3 thumbs — same DegradeConfig).

**v4 outcome (filled at close).** Per-(model, dataset) Δ_B1→B2nr at L3. With Δ_B1→B2 (US-040) and Δ_D→B2 (also US-040), reader can decompose Phase B2's drop into protocol-simplification effect (Δ_B1→B2nr) vs T3-regularization effect (Δ_B2nr→B2 ≈ Δ_D→B2 at L3).

**GPU budget.** ~3.5 GPU-h.

**Owner:** [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-040.

---

### US-042 — Multi-seed L3 expansion (48 runs across B1 + D-T3 + B2 + B2-nr)

**Goal.** Re-train the 24 L3 headline cells at seeds 43 and 44. Provides [mean ± std] variance bands on every L3 headline pp number cited in `Final_Report.pdf` rev3.

**Cells in scope (24 base tags × 2 extra seeds = 48 runs):**
- `final_B_L3_{m}_{d}` × 6 (Phase B1 at L3)
- `final_D_T3_L3_{m}_{d}` × 6 (Phase D T3 at L3)
- `final_B2_L3_{m}_{d}` × 6 (Phase B2 at L3)
- `final_B2nr_L3_{m}_{d}` × 6 (Phase B2-nr at L3)

**Pre-flight.**
1. US-040 and US-041 closed (Phase B2 + B2-nr seed = 42 baselines exist).
2. `runs/final/*_seed43/` and `runs/final/*_seed44/` empty.
3. Multi-seed determinism test green (`multi_seed_val_byte_identical_42_43_44`).

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_ralph_loop.py --phase multiseed --seeds 43,44 --cells final_B_L3_resnet50_cifar10,final_B_L3_resnet50_mnist,final_B_L3_densenet121_cifar10,final_B_L3_densenet121_mnist,final_B_L3_transnext_tiny_cifar10,final_B_L3_transnext_tiny_mnist,final_D_T3_L3_resnet50_cifar10,final_D_T3_L3_resnet50_mnist,final_D_T3_L3_densenet121_cifar10,final_D_T3_L3_densenet121_mnist,final_D_T3_L3_transnext_tiny_cifar10,final_D_T3_L3_transnext_tiny_mnist,final_B2_L3_resnet50_cifar10,final_B2_L3_resnet50_mnist,final_B2_L3_densenet121_cifar10,final_B2_L3_densenet121_mnist,final_B2_L3_transnext_tiny_cifar10,final_B2_L3_transnext_tiny_mnist,final_B2nr_L3_resnet50_cifar10,final_B2nr_L3_resnet50_mnist,final_B2nr_L3_densenet121_cifar10,final_B2nr_L3_densenet121_mnist,final_B2nr_L3_transnext_tiny_cifar10,final_B2nr_L3_transnext_tiny_mnist
```

(Operator may stage the dispatch by phase to keep individual sessions short. The 48 runs are independent.)

**Acceptance criteria.**
- [ ] 48/48 runs `healthy`.
- [ ] Each `<base_tag>_seed{N}/metrics.json` carries `seed: N`, `pipeline_version: 2`, and the base-tag-inherited fields.
- [ ] `Final_Exp.json` aggregator (US-046) groups by base tag and emits `val_acc_mean` / `val_acc_std` across the 3 seeds {42, 43, 44} per cell.
- [ ] Variance check: for each of the 24 base tags, `val_acc_std` < 1.5 pp (operator threshold for "stable cell"). Cells exceeding this threshold flagged for §X discussion.

**v4 outcome (filled at close).** [mean ± std] across 3 seeds for all 24 L3 headline cells, used in `Final_Report.pdf` rev3 §X Multi-Seed Variance.

**GPU budget.** ~24 GPU-h (48 runs × ~30 min).

**Owner:** [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-040, US-041.

---

### US-043 — Phase C2 6-cell pilot (C2_L3_resolution × all 6 (m, d) pairs)

**Goal.** Validate the Phase C2 pipeline on 6 cells: L3 × `resolution` axis × all 6 (m, d) pairs. The resolution axis exercises the DegradeConfig override most aggressively (sat = 0 + noise = 0 + low_res = 8 at L3 + all other axes at identity) and validates per-axis routing.

**Pre-flight.**
1. US-038 closed (Phase C2 wiring landed).
2. Determinism tests for C2 group green.
3. `runs/final/final_C2_L3_resolution_*/` empty.

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_C2_L3_resolution_resnet50_cifar10 --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_C2_L3_resolution_resnet50_mnist --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_C2_L3_resolution_densenet121_cifar10 --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_C2_L3_resolution_densenet121_mnist --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_C2_L3_resolution_transnext_tiny_cifar10 --mode pilot
.\.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag final_C2_L3_resolution_transnext_tiny_mnist --mode pilot
```

**Operator visual gate.** Sign `artifacts/validation/phase_c2_pilot_checklist.md` confirming the rendered thumb shows ONLY the resolution-axis effect (no blur, no salt-pepper, grayscale, no additive noise). HARD GATE for US-044.

**Acceptance criteria.**
- [ ] 6/6 pilot `metrics.json` written with `phase: "C2"`, `treatment: "T3"`, `axis: "resolution"`, C2 DegradeConfig override.
- [ ] Operator signs `phase_c2_pilot_checklist.md`. **HARD GATE.**
- [ ] Pilot dirs cleaned.

**GPU budget.** ~3.5 GPU-h.

**Owner:** [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-038. (Phase C2 pilot can run in parallel with the Phase B2 sweep.)

---

### US-044 — Phase C2 full sweep (remaining 84 cells across 3 axes)

**Goal.** Run the 84 non-pilot Phase C2 cells at full convergence: 3 axes (resolution / blur / salt_pepper) × 5 levels × 3 models × 2 datasets = 90 total, minus 6 pilot = 84 remaining. Pilot cells also re-train at full FT.

**Pre-flight.**
1. **HARD GATE — US-043 must be closed.**
2. Pilot dirs cleaned.
3. `PIPELINE_VERSION == 2`.

**Dispatch.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_ralph_loop.py --phase C2
```

Optional staging: `--axes resolution`, `--axes blur`, `--axes salt_pepper` for three separate ~17 GPU-h sessions.

**Post-pass.**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\update_final_exp.py
.\.venv-gpu\Scripts\python.exe -m src.tools.render_cell_thumbs --force --phase C2
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_exp_json
.\.venv-gpu\Scripts\python.exe -m src.tools.build_final_dashboard
```

**Acceptance criteria.**
- [ ] 90/90 cells `healthy`.
- [ ] All 90 `metrics.json` carry `pipeline_version: 2`, `phase: "C2"`, `axis ∈ {resolution, blur, salt_pepper}`, `treatment: "T3"`, C2 DegradeConfig override with the named axis at level L and others at identity.
- [ ] [Final_Exp.md](Final_Exp.md) reflects 90 Phase C2 rows; total cell count = 402 (with all phases).
- [ ] 90 dashboard thumbs rendered.

**v4 outcome (filled at close).** Per-axis recovery curve under the THz protocol. Attributes Phase B2's drop to which axis (resolution / blur / salt_pepper) carries the most weight under T3 regularization.

**GPU budget.** ~48 GPU-h (90 cells × ~32 min; slightly faster than Phase C since 2 axes are no-ops).

**Owner:** [EXECUTOR](agents/EXECUTOR.md) + [DEBUGGER](agents/DEBUGGER.md) + [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-043. (Can run in parallel with US-040, US-041, US-042 if operator has sufficient overnight slots.)

---

### US-045 — Post-training diagnostics (test-set + confusion + calibration + throughput)

**Goal.** Four post-hoc analysis passes producing the qualitative figures + variance numbers for `Final_Report.pdf` rev3 §XI + §XII.

**(a) Held-out test-set inference (72 evaluations).** Inference-only pass on a stratified held-out test split for all 24 L3 headline cells × 3 seeds = 72 evaluations. The split is defined by `artifacts/validation/test_split_indices.json` (committed at this US's pre-flight, frozen by a `seed=99` deterministic stratified-sample on each dataset's official test partition: 1000 per class).

**(b) Confusion matrices (24 L5 cells).** For each L5 cell of B1, D-T3, B2, B2-nr at seed = 42, generate a 10×10 confusion matrix as PNG. Source data: saved logits (when available) or a re-inference pass on the val set.

**(c) Calibration / ECE (48 cells at L3 + L5).** For each cell from (b) plus the 24 L3 headline cells (all at seed = 42), compute ECE via `torchmetrics.CalibrationError(n_bins=15)` and generate a reliability diagram. 24 L5 + 24 L3 = 48 figures.

**(d) Inference throughput (3 models).** Per-image latency measurement: warmup = 100 batches, measurement = 1000 batches at batch size 1 on the RTX 5070 at bf16-mixed precision. Report mean ± std (ms / image) for each of the 3 model families at 224×224 input. Single CSV at `artifacts/figures/inference_throughput.csv` + a horizontal-bar PNG.

**Pre-flight.**
1. US-040, US-041, US-042, US-044 closed (all required checkpoints exist).
2. `test_split_indices.json` generated and committed.
3. For cells without saved logits: confirm checkpoints are loadable (no weight-format drift since training).

**Dispatch (recommended sequence).**
```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_test_set_inference.py --cells <72 tag@seed list> --split-indices artifacts/validation/test_split_indices.json
.\.venv-gpu\Scripts\python.exe scripts\generate_confusion_matrices.py --cells <24 L5 tag list>
.\.venv-gpu\Scripts\python.exe scripts\generate_calibration_diagrams.py --cells <48 L3+L5 tag list>
.\.venv-gpu\Scripts\python.exe scripts\measure_inference_throughput.py --models resnet50,densenet121,transnext_tiny
```

**Acceptance criteria.**
- [ ] 72 test-set evaluations recorded in `artifacts/validation/test_set_results.json`; per cell records `val_acc`, `test_acc`, and `test_acc - val_acc` gap. Gaps > 2 pp flagged for §XI discussion.
- [ ] 24 confusion-matrix PNGs at `artifacts/figures/confusion/{tag}_L5.png` rendered with consistent colormap (`Blues`, no annotation overlap).
- [ ] 48 reliability diagrams at `artifacts/figures/calibration/{tag}_{L3,L5}.png` with ECE value in the title; aggregated ECE table at `docs/_autogen/calibration_table.tex`.
- [ ] Throughput CSV + PNG at `artifacts/figures/inference_throughput.{csv,png}`; report mean ± std per model.
- [ ] All four new scripts pass an idempotency assertion (re-running produces byte-identical PNGs).

**v4 outcome (filled at close).** Test-set confirmation table (§XI), L5 confusion-matrix gallery (§XII.A), L3 + L5 calibration table (§XII.B), throughput card (§XII.C).

**GPU budget.** ~3.5 GPU-h (mostly inference; throughput measurement is < 30 min).

**Owner:** [DESIGNER](agents/DESIGNER.md) (script implementation) + [VALIDATOR](agents/VALIDATOR.md) (idempotency assertions). **Dependencies:** US-040, US-041, US-042, US-044.

---

### US-046 — `Final_Exp.json` schema + aggregator wiring (B2 + B2-nr + C2 + multi-seed)

**Goal.** Wire all four new phases AND the multi-seed grouping into the `Final_Exp.json` aggregator. Mirrors v3 US-029.5 with extended scope.

**Deliverables.**
- Patch [`src/tools/build_final_exp_json.py`](src/tools/build_final_exp_json.py):
  - `iter_cells(...)` call site honors all four `*_present_on_disk` helpers.
  - `EXPECTED_TOTAL` assertion picks the right constant based on which phases are present.
  - `_row_for(meta)` populates `treatment` (`"T3"` for B2/C2, `None` for B2nr, existing for D).
  - New aggregator pass: group rows by base tag (strip `_seed{N}` suffix); for each base tag emit `val_acc_mean`, `val_acc_std`, `seeds_observed: [42, 43, 44]` (or subset).
- Extend [`src/tools/final_exp_schema.py:FinalExpRow`](src/tools/final_exp_schema.py) with `phase: Literal["A", "B", "B2", "B2nr", "C", "C2", "D"]`, `seed: int`, `val_acc_mean: Optional[float]`, `val_acc_std: Optional[float]`. Bump `SCHEMA_VERSION` by 1.
- New tests in [`src/tests/test_build_final_exp_json.py`](src/tests/test_build_final_exp_json.py):
  - Row count assertions across all flag combinations (276 / 282 / 306 / 312 / 336 / 372 / 402).
  - `test_phase_b2_row_has_treatment_t3` / `test_phase_b2nr_row_has_treatment_none` / `test_phase_c2_row_has_treatment_t3`.
  - `test_multi_seed_grouping` — 3 seed runs per base tag → 1 aggregated row with `val_acc_mean` / `val_acc_std` populated.
  - `test_schema_version_bumped`.

**Acceptance criteria.**
- [x] Aggregator emits the right row count for every flag combination.
- [x] `phase`, `treatment`, `seed`, `val_acc_mean`, `val_acc_std` fields correctly populated per row.
- [x] All new pytest assertions green.
- [ ] `mypy src/tools/` reports no new type errors. *(deferred — operator audit pass)*
- [x] 276-cell v3 view byte-identical when no new phases on disk (regression).

**Owner:** [DESIGNER](agents/DESIGNER.md). [VALIDATOR](agents/VALIDATOR.md) signs off. **Dependencies:** US-040, US-041, US-042, US-044, US-045.

---

### US-047 — Dashboard updates (7 phase tabs + multi-seed + diagnostic galleries + throughput card)

**Goal.** Extend [`src/tools/build_final_dashboard.py`](src/tools/build_final_dashboard.py) to render the full v4 surface area.

**Deliverables.**
- **7 phase tabs.** Tab order: A | B (B1) | B2 | B2-nr | C | C2 | D. Phase B tab display text relabel "Phase B" → "Phase B (B1)" (data-phase attribute unchanged).
- **Treatment chip group** visible on B2, C2, D tabs; hidden on A, B (B1), B2-nr, C.
- **Multi-seed variance indicator.** L3 cells with multi-seed audits show `val_acc` as `mean ± std` in the table; an info icon on hover lists the individual seed values.
- **Confusion-matrix gallery** under a new `<details>` section on the L5 cells; lazy-load 24 PNGs.
- **Calibration-diagram gallery** under a new `<details>` section on the L3 + L5 cells; lazy-load 48 PNGs + the ECE table.
- **Inference-throughput card** in the dashboard header (one row per model, ms / image ± std).
- **Comparison strips** per-(model, dataset):
  - Phase D recovery (existing — v3 US-030).
  - Phase B2 comparison: B1 vs D-T3 vs B2 vs B2-nr (4 series).
  - Phase C2 axis attribution: B1 baseline vs C2 resolution / blur / salt_pepper at each L (4 series).

Tests in [`src/tests/test_dashboard.py`](src/tests/test_dashboard.py):
- Tab-count + tab-label assertions.
- Treatment chip visibility per tab.
- Multi-seed indicator presence on L3 cells with audits.
- Gallery section presence on L5 cells with confusion matrices.
- Throughput card presence in header.
- Regression: 276-cell v3 dashboard byte-identical when no new phases on disk.

**Acceptance criteria.**
- [ ] All seven `data-phase` tabs render with correct counts.
- [ ] Phase B tab display reads "Phase B (B1)".
- [ ] Treatment chip group visibility matches per-tab spec.
- [ ] Multi-seed indicator + diagnostic galleries + throughput card render.
- [ ] All comparison strips draw when ≥ 1 cell is Complete.
- [ ] 276-cell v3 view byte-identical when no new phases on disk.

**Owner:** [DESIGNER](agents/DESIGNER.md). **Dependencies:** US-046.

---

### US-048 — Comparison plots + LaTeX tables (full v4 surface)

**Goal.** Create / extend the plot scripts that emit every figure + LaTeX table cited in `Final_Report.pdf` rev3.

**Deliverables.**
- **`scripts/plot_phase_b2_comparison.py`** — per-(model, dataset) 4-series panels (B1 / D-T3 / B2 / B2-nr) at L1..L5 + composite + LaTeX table at `docs/_autogen/phase_b2_comparison_table.tex`.
- **`scripts/plot_phase_c2_attribution.py`** — per-(model, dataset) panels showing C2 resolution / blur / salt_pepper recovery vs Phase A clean baseline at L1..L5 + composite + LaTeX table at `docs/_autogen/phase_c2_attribution_table.tex`.
- **`scripts/plot_multi_seed_variance.py`** — per-(model, dataset, phase) bar chart of L3 [mean ± std] across 3 seeds for B1 / D-T3 / B2 / B2-nr; 4 panels per (m, d) = 24 panels + composite + LaTeX table at `docs/_autogen/multi_seed_variance_table.tex`.
- Confusion-matrix + calibration figures already generated in US-045 — this US adds a `docs/_autogen/confusion_summary_table.tex` and `docs/_autogen/calibration_table.tex` for LaTeX `\input{}`.
- Throughput figure generated in US-045 — this US adds `docs/_autogen/throughput_table.tex`.
- Pin determinism on all new scripts: same matplotlib settings as v3's `plot_phase_d_comparison.py`.
- Baseline re-verification: read `phase_b2_baseline_manifest.json`, re-hash all 66 references, assert equality.
- Idempotency tests for each new plot script.

**Acceptance criteria.**
- [ ] All plot scripts emit the specified PNGs + LaTeX tables.
- [ ] Baseline-manifest re-verification passes for all 66 reference entries.
- [ ] Two consecutive runs of each script produce byte-identical PNGs.
- [ ] No `??` placeholders in any LaTeX table.

**Owner:** [DESIGNER](agents/DESIGNER.md) (plots). [REPORTER](agents/REPORTER.md) reviews LaTeX. **Dependencies:** US-040, US-041, US-042, US-044, US-045, US-046.

---

### US-049 — Post-campaign content sync (B → B1 rename + full v4 narrative)

**Goal.** Update narrative artifacts to reflect Phase B2 + B2-nr + C2 + multi-seed + diagnostics findings AND introduce the Phase B1 documentation alias consistently. Mirrors v3 US-032's pattern with expanded scope.

**Files modified.**
- [README.md](README.md):
  - §"Final Results" — append four subsections: "Phase B2 — No-Color / No-Noise Protocol", "Phase B2-nr — Pure-Protocol Read at L3", "Phase C2 — Axis Attribution Under THz Protocol", "Multi-Seed Variance + Test-Set Confirmation + Diagnostics".
  - §"Headline Research Findings" — append Findings #5 (Phase B2 protocol-simplification effect), #6 (Phase C2 dominant axis), #7 (multi-seed variance + test-set gap), #8 (calibration + throughput).
  - §"Recent scientific changes" — add a 2026-05-?? entry covering the full v4 scope.
  - Throughout: replace "Phase B" with "Phase B1" where the context distinguishes from B2; preserve `final_B_*` / `phase == "B"` code references.
- [CLAUDE.md](CLAUDE.md):
  - §"Project Status" — append all new phase opens / closes entries.
  - §"186/276-cell Experiment Plan" → "186/276/402-cell Experiment Plan"; add rows for Phase B2, B2-nr, C2.
  - §"Experiment System" — add `--phase B2`, `--phase B2nr`, `--phase C2`, `--phase multiseed` dispatch commands.
  - Same B → B1 alias discipline as README.
- [docs/Final_Exp_Report.md](docs/Final_Exp_Report.md) — auto-regenerated; manually review conclusions section.
- [progress.txt](progress.txt) — append "Iteration N — v4 full scientific-depth close" with the campaign-level summary.

**Rename-discipline cross-check.** SYNCHRONIZER grep pass over touched files confirms `final_B_*` tag strings / `phase == "B"` / `"--phase B"` (code constructs) UNCHANGED; narrative "Phase B" → "Phase B1" where context demands.

**Acceptance criteria.**
- [ ] All listed files updated with the v4 scope.
- [ ] Findings #5–#8 quote specific pp values from US-040, US-041, US-042, US-044, US-045 outputs.
- [ ] CLAUDE.md "186/276" → "186/276/402" rename done.
- [ ] Rename-discipline cross-check passes (no accidental tag corruption).

**Owner:** [SYNCHRONIZER](agents/SYNCHRONIZER.md) + [LIBRARIAN](agents/LIBRARIAN.md) + [REPORTER](agents/REPORTER.md). **Dependencies:** US-040, US-041, US-042, US-044, US-045, US-047, US-048.

---

### US-050 — `Final_Report.pdf` rev3 — appendices §VIII–§XII

**Goal.** Add five new appendices to [docs/Final_Report.tex](docs/Final_Report.tex):

- **§VIII Phase B2 — No-Color / No-Noise Protocol.** Design (cite §II degradation table + §VII.A T3 reuse). Results (6 comparison panels + LaTeX table). Discussion (Δ_B1→B2, Δ_D→B2, Δ_B1→B2nr decomposition; 1-paragraph answer to the §1 Phase B2 question).
- **§IX Phase C2 — Single-Axis Isolation Under the THz Protocol.** Design. Results (6 axis-attribution panels + LaTeX table). Discussion (per-(model, dataset) dominant axis under T3; cross-reference to legacy Phase C for the no-reg comparison).
- **§X Multi-Seed Variance.** Design (24 L3 headline cells × 3 seeds). Results (variance-bar bar chart + LaTeX table). Discussion (which cells are stable, which show high seed-dependence; implications for the v2 collapse magnitude).
- **§XI Test-Set Confirmation.** Design (stratified held-out split). Results (72 test_acc values + val→test gap per cell). Discussion (any val_acc cell that doesn't transfer to test; flagged cells discussed individually).
- **§XII Diagnostics.** Three subsections:
  - §XII.A Confusion matrices at L5 (24 heatmaps as small-multiples; observations on per-class robustness).
  - §XII.B Calibration / ECE at L3 + L5 (48 reliability diagrams + ECE table; observations on overconfidence under degradation).
  - §XII.C Inference throughput (per-model latency card; deployment-readiness one-paragraph).

**Acceptance criteria.**
- [ ] All five appendices present in TOC; section headings + subsections rendered.
- [ ] All figures `\includegraphics`'d successfully; no `??` placeholders.
- [ ] All LaTeX tables (`phase_b2_comparison_table.tex`, `phase_c2_attribution_table.tex`, `multi_seed_variance_table.tex`, `calibration_table.tex`, `throughput_table.tex`) render correctly.
- [ ] `pdflatex` + `bibtex` + `pdflatex ×2` exit 0; no missing-reference warnings.
- [ ] Each discussion subsection quotes specific pp values from the corresponding US outputs.

**Owner:** [REPORTER](agents/REPORTER.md) (LaTeX/prose). [DESIGNER](agents/DESIGNER.md) reviews figure renders. **Dependencies:** US-045, US-048, US-049.

---

### US-051 — `Final_Report.pdf` rev3 — §XIII Final Conclusions chapter

**Goal.** Closing chapter consolidating v2 + v3 + v4 findings into a single project-closure narrative. This is the chapter the operator's supervisor reads first.

**§XIII structure (recommended; REPORTER may restructure with MASTER approval).**
- §XIII.A Recap per phase:
  - v2: Pipeline correction → −12.49 pp mean Phase B collapse.
  - v3 Phase D: T3 regularization recovers only +0.77 pp of the collapse — collapse is largely an information bottleneck.
  - v4 Phase B2: At T3, stripping color + noise from the input moves accuracy by Δ_D→B2 (filled at close).
  - v4 Phase B2-nr: Pure protocol-simplification Δ at L3 (factors T3 out of the B2 read).
  - v4 Phase C2: Dominant axis under the THz protocol (filled at close).
  - v4 Multi-seed: Variance bands on every headline number — confirms / qualifies the v2/v3 single-seed claims.
  - v4 Test-set: val_acc → test_acc gap (filled at close).
- §XIII.B Cross-cut: per-axis severity ordering at L5 (carries v2 finding #3 forward; B2 + C2 sharpen the spatial-vs-color contribution split).
- §XIII.C Cross-cut: per-architecture pattern (ResNet50 vs DenseNet121 vs TransNeXt-tiny robustness ranking under each phase).
- §XIII.D Limitations and threats to validity:
  - Synthetic degradation curve, not measured THz imagery (carries forward).
  - 5-level discretization, not continuous severity.
  - Phase B2 / C2 use a single regularization recipe (T3); no T1-only / T2-only sweep under the THz protocol.
  - Phase A baseline is also single-seed (multi-seed expansion deferred to v5 post-submission).
- §XIII.E Practical recommendations for deploying THz-classification models (regularization tactics, preprocessing choices, architecture selection guidance) grounded in the data.
- §XIII.F Future work — bullets for any post-submission scope (e.g., true THz-data validation, continuous severity sweep, multi-treatment B2 / C2).

**Acceptance criteria.**
- [ ] `artifacts/Final_Report.pdf` rev3 exists; ≥ 16 pages (was ≥ 10 in rev2, +6 for v4 appendices).
- [ ] §XIII Final Conclusions present in TOC; six subsections rendered.
- [ ] Per-phase headline answers quoted verbatim in §XIII.A with consistent pp magnitudes across the corresponding appendix.
- [ ] No `??` placeholders; `pdflatex` + `bibtex` + `pdflatex ×2` exit 0.
- [ ] §XIII.D and §XIII.F reference the deferred v5 scope items explicitly.

**Owner:** [REPORTER](agents/REPORTER.md). **Dependencies:** US-050.

---

### US-052 — NotebookLM 2-pass sync

**Goal.** Two-pass synchronization of the THz Project notebook (`notebook_id: e244f8b6-30ab-4966-a74c-426ff88be39a`):

- **Pass 1 (immediate, after US-036 lands)** — upload PRD v4 only.
- **Pass 2 (post-US-051)** — full re-source: refresh all manifest-tracked files (`README.md`, `PRD.md`, `CLAUDE.md`, `docs/Final_Exp_Report.md`, `docs/phase_a.md`, `docs/phase_c.md`, `docs/phase_d.md`, new `docs/phase_b2.md`, new `docs/phase_c2.md`, `progress.txt`) + add `artifacts/Final_Report.pdf` rev3.

**Pre-flight.** `python -m notebooklm status` to verify auth before any upload.

**Dispatch.**
```powershell
# Pass 1 (immediate, after this PRD lands)
python scripts\notebooklm_sync.py refresh --apply --only PRD.md

# Pass 2 (post-US-051)
python scripts\notebooklm_sync.py full --apply
```

**Acceptance criteria.**
- [ ] Pass 1: `scripts/notebooklm_sync.log` carries a `[refresh]` entry with `PRD.md`.
- [ ] Pass 2: `[full]` entry with `push`, `prune`, `refresh` counts.
- [ ] Manifest updated to include `docs/phase_b2.md` and `docs/phase_c2.md`.
- [ ] `python -m notebooklm status` confirms source list matches manifest.
- [ ] No `*.ckpt` / `*.pt` / `*.pth` uploaded.

**Owner:** [NOTEBOOKLM_SYNC](agents/NOTEBOOKLM_SYNC.md). **Dependencies:** US-036 (pass 1), US-051 (pass 2).

---

### US-053 — Final verification + ship-readiness audit + submission close

**Goal.** End-of-campaign closure. Every artifact in lockstep, all tests green, audit trail intact, submission-ready.

**Checks.**
- `pytest src/tests/test_degradation_determinism.py -v` → green INCLUDING new B2, C2, multi-seed groups.
- `pytest src/tests/test_matrix.py -v` → green INCLUDING all new B2/B2nr/C2/multi-seed assertions.
- `pytest src/tests/test_build_final_exp_json.py -v` → green INCLUDING new schema assertions.
- `pytest src/tests/test_update_final_exp.py -v` → green.
- `pytest src/tests/test_dashboard.py -v` → green INCLUDING new UI assertions.
- `pytest src/tests/test_phase_b2_plot.py -v` → green (idempotency).
- `pytest src/tests/test_phase_c2_plot.py -v` → green (idempotency).
- `pytest src/tests/test_multi_seed_plot.py -v` → green (idempotency).
- `pytest src/tests/test_phase_d_plot.py -v` → still green (regression).
- `python -m src.tests.test_ignores` → green.
- `mypy src/` on v4-touched files → no new errors.
- `python scripts/update_final_exp.py --check` → exit 0; total cell count = 402 canonical + 48 multi-seed audit runs.
- `bash scripts/check_ignores.sh` → green.
- Manual: open `artifacts/Final_Exp.html`, filter to each new phase, confirm all rows render correctly; Phase B tab reads "Phase B (B1)"; multi-seed variance bars visible on L3 cells; diagnostic galleries lazy-load.
- Manual: open `artifacts/Final_Report.pdf` rev3 end-to-end; scan for placeholders / broken refs / missing figures; §VIII–§XIII all present in TOC.
- Ship-readiness audit via Agent (subagent_type `general-purpose`): cross-check PRD + README + CLAUDE.md + progress.txt + Final_Report.pdf rev3 reference the same scope, the same numbers, the same headline answers to each §1 / §IX question.
- Submission-readiness audit: confirm `Final_Report.pdf` rev3 ≥ 16 pages, §XIII present, BibTeX up-to-date, no compile warnings.

**Acceptance criteria.**
- [ ] All ten pytest commands exit 0.
- [ ] Ship-readiness audit returns zero cross-artifact drift.
- [ ] Submission-readiness audit passes.
- [ ] `git status` clean; PR draft URL captured.
- [ ] Operator sign-off in `progress.txt` as campaign close marker.

**Owner:** [VALIDATOR](agents/VALIDATOR.md). **Dependencies:** US-051, US-052.

---

## 9. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **(a) DegradeConfig override applied at wrong layer (B2 or C2)** | Low | High | US-038 unit tests + US-039 / US-043 6-cell pilots + operator visual gates (sat=0, noise=0 visible on thumb). |
| **(b) Tag-prefix collisions across 7 prefixes** | Medium | High | US-038 switches legacy Phase B check to `startswith("final_B_L")`; defense-in-depth assertions in `run_all_phases.py` for every `--phase X` dispatch. |
| **(c) T3 delta carrier slot ambiguity** (B2 + C2 both stamp `treatment="T3"` — downstream may confuse them) | Low | Medium | US-038 + US-046: `phase` field disambiguates; `treatment` is a regularization recipe identifier, `phase` is the cell-design identifier. Schema tests enforce both. |
| **(d) Multi-seed determinism drift** (val pixels NOT byte-identical across seeds 42/43/44) | Low | High | US-038 adds `multi_seed_val_byte_identical_42_43_44` determinism test; pre-flight guard in US-042. |
| **(e) Logit logging not enabled for cells we need diagnostics on** | Medium | Medium | US-038 auto-enables `--log-logits` for multi-seed + diagnostic targets; US-045 falls back to re-inference if logits absent. Cost: ~0.5 GPU-h re-inference for v3 cells (D T3 L5). |
| **(f) Test-set split contamination** (the "held-out" split overlaps training data) | Low | High | `test_split_indices.json` derived from the OFFICIAL test partition of each dataset (CIFAR-10 / MNIST); pre-flight assertion confirms indices are from the test split, not the train split. |
| **(g) Calibration bins empty for some cells** (low prediction diversity at extreme degradation) | Medium | Low | `torchmetrics.CalibrationError(n_bins=15)` is robust to empty bins; reliability diagrams visually show empty bins as gaps. Acceptable. |
| **(h) Baseline drift during the ~5-day campaign** | Low | High | `phase_b2_baseline_manifest.json` (66 refs) captured at US-039; re-verified at US-048 plot rendering. |
| **(i) Aggregator regression on 276-cell v3 view** (US-046 wiring breaks v3 dashboard) | Low | Medium | US-046 includes byte-identical-when-no-new-phases-on-disk regression assertion. |
| **(j) Project-closure §XIII scope creep** (US-051 expands into a multi-week write-up) | Medium | High | US-051 specifies six subsections; over-scope content deflected to §XIII.F future-work bullets. MASTER reviews before commit. |
| **(k) Attribution conflation re-emerges** (readers misread Δ_B1→B2 as pure protocol) | Medium | Low | Δ_B1→B2nr (US-041) + Δ_D→B2 (US-040) provide the decomposed reads; §VIII.C must frame them explicitly. |
| **(l) GPU budget overrun beyond 110 GPU-h** | Low | Low | Cells are independent and `--skip-existing` respected. Operator can stage phases (B2 → B2-nr → C2 → multi-seed → diagnostics) in separate sessions. Worst case: defer multi-seed audits for B2-nr (12 runs → ~6 GPU-h saved); preserves the main claims. |
| **(m) Single-seed Phase A baseline limits Phase C2 attribution** | Medium | Low | Listed in §XIII.D as a known limitation. v5 future-work bullet (§XIII.F) covers multi-seed Phase A if a post-submission scope opens. |

---

## 10. Dependency-Ordered Story Map

```
US-036 (PRD v4 + progress.txt) ────────────────────────────────────┐
                                                                    │
                                                                    ▼
                                            US-037 (docs/phase_b2.md + docs/phase_c2.md)
                                                                    │
                                                                    ▼
                                            US-038 (full code wiring + logit hook + tests)
                                                                    │
                                            ┌───────────────────────┼───────────────────────┐
                                            ▼                       ▼                       ▼
                              US-052 pass 1 (NotebookLM)   US-039 (B2 pilot)        US-043 (C2 pilot)
                                                                    │  HARD GATE             │  HARD GATE
                                                                    ▼                       ▼
                                                          US-040 (B2 sweep)         US-044 (C2 sweep)
                                                                    │                       │
                                                                    ▼                       │
                                                          US-041 (B2-nr L3 arm)             │
                                                                    │                       │
                                                                    ▼                       │
                                                          US-042 (multi-seed L3)            │
                                                                    │                       │
                                                                    └───────────┬───────────┘
                                                                                ▼
                                                                    US-045 (diagnostics: test-set + confusion + calibration + throughput)
                                                                                │
                                                                                ▼
                                                                    US-046 (Final_Exp.json schema + aggregator)
                                                                                │
                                                                    ┌───────────┼───────────┐
                                                                    ▼                       ▼
                                                          US-047 (dashboard)         US-048 (plots + LaTeX)
                                                                    │                       │
                                                                    └───────────┬───────────┘
                                                                                ▼
                                                                    US-049 (README / CLAUDE.md / Final_Exp_Report.md / progress.txt sync)
                                                                                │
                                                                                ▼
                                                                    US-050 (Final_Report rev3 §VIII–§XII)
                                                                                │
                                                                                ▼
                                                                    US-051 (Final_Report rev3 §XIII Final Conclusions)
                                                                                │
                                                                                ▼
                                                                    US-052 pass 2 (NotebookLM full)
                                                                                │
                                                                                ▼
                                                                    US-053 (verification + submission close)
```

**Critical path:** US-036 → US-037 → US-038 → (US-039 → US-040 → US-041 → US-042 in series; US-043 → US-044 in parallel) → US-045 → US-046 → (US-047, US-048 in parallel) → US-049 → US-050 → US-051 → US-052 pass 2 → US-053.

**Parallelizable forks:**
- US-052 pass 1 (NotebookLM PRD upload) runs in parallel with US-038 / US-039.
- US-043 (C2 pilot) and US-040 (B2 sweep) can dispatch in parallel after US-038 closes.
- US-044 (C2 sweep) can run in parallel with US-041 + US-042 if operator has overnight slots.
- US-047 (dashboard) and US-048 (plots) parallelize after US-046.

**Wall-clock estimate.** Total ~100 GPU-h on a single RTX 5070. With 12–14 hour overnight runs and parallel dispatch of B2 + C2 trains, the campaign closes in **~5–6 calendar days of GPU time** + ~4 calendar days of non-GPU US closures (US-046 through US-053). Net: **~10 calendar days from US-036 close to US-053 close**. Submission deadline 26/07/2026 — ample buffer.

---

## 11. Definition of Done (campaign-level)

- [ ] PRD v4 lands on disk (US-036); v3 + v4 first-draft referenced via git commits in the header.
- [ ] [progress.txt](progress.txt) Iteration N entries appended per US closure.
- [ ] [docs/phase_b2.md](docs/phase_b2.md) and [docs/phase_c2.md](docs/phase_c2.md) published.
- [ ] US-039 + US-043 pilot checklists signed off BEFORE the corresponding sweep dispatches.
- [ ] Extended baseline manifest (66 references) captured at US-039 and re-verified at US-048.
- [ ] 30/30 Phase B2 cells healthy with B2 DegradeConfig override + T3.
- [ ] 6/6 Phase B2-nr cells healthy with B2 DegradeConfig override + no T3.
- [ ] 90/90 Phase C2 cells healthy with C2 DegradeConfig override + T3.
- [ ] 48/48 multi-seed audit runs healthy across B1, D-T3, B2, B2-nr at L3.
- [ ] 72 test-set evaluations, 24 confusion matrices, 48 calibration diagrams, throughput report all generated.
- [ ] Aggregator wired for all new phases + multi-seed grouping; `SCHEMA_VERSION` bumped; pytest assertions green.
- [ ] Dashboard renders 7 phase tabs, multi-seed variance indicator, diagnostic galleries, throughput card; 276-cell v3 view byte-identical when no new phases on disk.
- [ ] All comparison PNGs + LaTeX tables emitted idempotently.
- [ ] [README.md](README.md) carries Phase B2 / B2-nr / C2 subsections + Findings #5–#8; [CLAUDE.md](CLAUDE.md) renamed "186/276" → "186/276/402" cells; both reference v4 numbers with B → B1 alias applied consistently.
- [ ] NotebookLM passes 1 + 2 logged.
- [ ] [artifacts/Final_Report.pdf](artifacts/Final_Report.pdf) rev3 ≥ 16 pages; §VIII–§XIII present; pdflatex/bibtex exit 0; zero `??` placeholders.
- [ ] All pytest commands exit 0; `mypy` on v4-touched files green; `git status` clean.
- [ ] Ship-readiness audit returns zero cross-artifact drift.
- [ ] Submission-readiness audit passes; operator sign-off recorded as campaign close.

---

## 12. Decisions Locked

Carries v3 decisions 1–11 forward (re-stated in summary). New v4-specific decisions begin at item 12.

1–11. **Carry-forward from v3.** Regularization deltas frozen; baselines locked; no re-tune; 6-cell pilot pattern; 2-pass NotebookLM; operator-locked research questions; multi-select treatment chip; new-appendix report pattern; progress.txt is append-only; aggregator gap handled in dedicated US; sub-agent dispatch.

**New for v4 (operator-confirmed 2026-05-26):**

12. **Phase B2 = sat=0 + noise=0 + T3 regularization, frozen Optuna L3 winners.** 30 cells. Source of truth: `degrade_config_for_b2(level)` + `_phase_d_treatment_deltas("T3", model)`.
13. **Doc-only Phase B → Phase B1 rename.** `runs/final/final_B_*` dirs UNCHANGED; `phase == "B"` enum UNCHANGED; `final_B_*` tag strings UNCHANGED. SYNCHRONIZER enforces in US-049.
14. **PRD v4 is the closing scientific PRD.** Any v5 reserved for hypothetical post-submission scope.
15. **§XIII Final Conclusions is non-optional and is a US-053 gate.**
16. **Phase B2 attribution decomposition.** Δ_D→B2 is the pure-protocol metric at fixed T3 (from B2 vs D-T3 cells). Δ_B1→B2nr is the pure-protocol metric at no regularization (from B2-nr vs B1 cells at L3). Δ_B1→B2 is the combined-effect read.
17. **Phase C2 = sat=0 + noise=0 + T3 regularization, single-axis isolation across {resolution, blur, salt_pepper}.** 90 cells. Matches Phase B2's protocol; provides per-axis attribution of B2's drop. Source of truth: `degrade_config_for_c2(level, axis)`.
18. **Multi-seed scope = 24 L3 headline cells × seeds {43, 44}.** Canonical reported pp number is the seed = 42 value; variance band is [mean ± std] across {42, 43, 44}. Mean-of-3 is NOT the headline. Multi-seed for Phase A / Phase C / Phase C2 / non-L3 levels deferred to v5.
19. **Held-out test-set split is deterministic and frozen.** `test_split_indices.json` derived from the OFFICIAL CIFAR-10 / MNIST test partitions at seed = 99, stratified 1000 per class. Committed at US-045 pre-flight, never modified.
20. **Test-set inference covers 72 evaluations** (24 L3 base tags × 3 seeds). Confusion matrices cover 24 L5 cells at seed = 42 only. Calibration / ECE covers 48 cells (24 L3 + 24 L5) at seed = 42 only.
21. **Inference-throughput report uses bf16-mixed precision at batch size 1**, 100 warmup batches + 1000 measurement batches per model. Reported per-image mean ± std in ms.
22. **`Final_Report.pdf` rev3 target = ≥ 16 pages** (rev2 was ≥ 10; +6 for new appendices). §VIII through §XIII split as defined in US-050 + US-051.
23. **No deferrals to v5 within the v4 active set.** All §14 items from the v4 first-draft PRD are folded into US-040 / US-041 / US-042 / US-044 / US-045 as first-class user stories. Future-work bullets in §XIII.F reference v5 only for items genuinely outside this campaign's scope (true THz-data validation, continuous severity sweep, multi-treatment B2 / C2).

---

## 13. v2 + v3 History (US-017..US-035 — closed)

**v2 — closed 2026-05-23.** Full v2 PRD prose preserved at git commit `ced98ec`.

| Story | Title | Closed |
|---|---|---|
| US-017 | Pipeline v2 refactor + new invariant test | 2026-05-20 |
| US-018 | PRD v2 authoring | 2026-05-20 |
| US-019 | RALPH loop `--axes` filter patch | 2026-05-20 |
| US-019B | v2 degradation thumbnail preview | 2026-05-20 |
| US-020 | Phase B re-run (30 cells) | 2026-05-21 |
| US-021 | Phase C `noise` axis re-run (30 cells) | 2026-05-22 |
| US-022 | Phase C `salt_pepper` axis re-run (30 cells) | 2026-05-23 |
| US-023 | Post-campaign content sync | 2026-05-23 |
| US-024 | IEEEtran scientific report (8 pages) | 2026-05-23 |
| US-025 | Final verification + ship-readiness audit | 2026-05-23 |

**v3 — closed 2026-05-26.** Full v3 PRD prose preserved at git commit `898d007`.

| Story | Title | Closed |
|---|---|---|
| US-026 | PRD v3 authoring | 2026-05-23 |
| US-027 | `docs/phase_d.md` rationale | 2026-05-23 |
| US-028 | 6-cell pilot + Phase B v2 baseline manifest | 2026-05-24 |
| US-029 | Phase D full sweep (84 cells) | 2026-05-25 |
| US-029.5 | `Final_Exp.json` schema + aggregator | 2026-05-25 |
| US-030 | Dashboard Phase D tab + Treatment chip | 2026-05-26 |
| US-031 | Phase D recovery PNGs + LaTeX table | 2026-05-26 |
| US-032 | Post-campaign content sync | 2026-05-26 |
| US-033 | `Final_Report.pdf` rev2 §VII | 2026-05-26 |
| US-034 | NotebookLM 2-pass sync | 2026-05-26 |
| US-035 | Final verification | 2026-05-26 |

**Headline findings carried forward to v4 baselines:**
1. Phase B v2 mean Δ = **−12.49 pp vs v1** (the collapse).
2. Phase D T3 recovery = **+0.77 pp mean** (small fraction of the collapse).
3. T1 / T2 negligible recovery; T3 is the load-bearing recipe.
4. MNIST geometric-axis floor preserved across v2 / v3.
5. CIFAR-10 protocol sensitivity dominates the recovery surface.

---

## 14. Open Scientific Questions — All Folded Into Active Set

**Per operator decision 2026-05-26, all seven §14 items from the v4 first-draft PRD are now first-class user stories in the active set.** Tracking the original mapping for audit purposes:

| First-draft §14 item | Now lives at | Status |
|---|---|---|
| §14.1 — Pure-protocol Δ via 6-cell B2-no-reg L3 arm | US-041 | active |
| §14.2 — Multi-seed at L3 for variance bars | US-042 | active (24 L3 cells × 2 extra seeds = 48 runs) |
| §14.3 — Test-set evaluation on headline L3 cells | US-045(a) | active (72 evaluations) |
| §14.4 — Phase C2 single-axis isolation under B2 protocol | US-043 (pilot) + US-044 (sweep) | active (90 cells) |
| §14.5 — Confusion-matrix dump at L5 | US-045(b) | active (24 heatmaps) |
| §14.6 — Calibration / ECE at L3 and L5 | US-045(c) | active (48 reliability diagrams) |
| §14.7 — Inference throughput report | US-045(d) | active (3 models) |

**No items deferred to v5 from the first-draft §14 list.** Future-work bullets in §XIII.F of `Final_Report.pdf` rev3 reference only genuinely out-of-scope ideas surfaced during v4 execution (e.g., true THz-data validation, multi-treatment B2 / C2, continuous severity sweep).

---

## End of PRD v4

This PRD ships the project into its closing scientific phase with maximum depth. After US-053 signs off, the operator has a single ≥ 16-page IEEE-formatted submission-ready document carrying [mean ± std] variance bands, held-out test confirmation, per-class confusion analysis, calibration audit, and deployment-readiness throughput — anchored in 402 canonical cells + 48 multi-seed audit runs. Submission deadline 26/07/2026.
