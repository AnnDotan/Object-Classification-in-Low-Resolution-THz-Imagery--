# Phase C2 — THz-protocol single-axis attribution

**Status:** US-036 closed 2026-05-26, US-037 closed 2026-05-26 — rationale documented here; code wiring lands at US-038 ([PRD.md §8 US-038](../PRD.md)). Phase C2 dispatch is scheduled at US-043 (pilot) + US-044 (full sweep).

---

## What changed

| | Pre-Phase C2 (Phase C v2 single-axis baseline) | Post-Phase C2 (v4) |
|---|---|---|
| Hparams | L3-Optuna winners (no regularization deltas) | L3-Optuna winners **+ T3 regularization delta** layered identically to Phase D / Phase B2 |
| DegradeConfig | Named axis at level L; every other axis at `IDENTITY_VALUES` per US-003 | Named axis at level L; **`saturation = 0.0` + `gaussian_noise_std = 0.0` overrides** (THz-protocol invariant); every other non-{saturation, noise_std} axis at `IDENTITY_VALUES` |
| Axes in scope | All five: resolution, blur, noise, salt_pepper, saturation (150 cells) | **Three: resolution, blur, salt_pepper** (90 cells) — saturation and noise are now part of the protocol invariant, not free axes |
| Cells | 276 (Phases A + B + C + D) | **366** when `include_phase_c2=True` — 90 new Phase C2 cells (5 levels × 3 axes × 3 models × 2 datasets). **402** when all four `include_phase_*` flags are set |

## Why

*Under the THz protocol (sat = 0 + noise = 0 + T3), which spatial-domain axis (resolution / blur / salt_pepper) dominates Phase B2's drop?*

This is one of the **operator-locked headline research questions** from [PRD.md §1](../PRD.md). Phase B2 collapses five axes into a single sat-and-noise-zeroed combined cell — interpretation is "the THz-protocol drop is X pp at level L." Phase C2 is the per-axis-attribution counterpart: it isolates each remaining spatial-domain axis under the same THz protocol so the reader can read which axis contributes most to Phase B2's drop. Legacy Phase C (US-003) was the v1/v2/v3 single-axis isolation under the full sat/noise-enabled protocol; Phase C2 is **C-isolation under the B2 protocol** — same isolation semantics, but with sat and noise zeroed out so the per-axis read is THz-faithful.

The axis-restriction to `{resolution, blur, salt_pepper}` is deliberate: saturation and noise are *part of the B2 protocol invariant* (always zero), so isolating them under the B2 protocol would produce a trivial no-op cell (no-degradation baseline = Phase A). Of the remaining three axes, `resolution` is the dominant information-bottleneck (3×3 downsample at L5; v1 finding #1 + v2 finding #1), `blur` is the smoothing nuisance (pixel-domain at 224; rescaled 2026-05-14), and `salt_pepper` is the defective-pixel nuisance (sensor-grain at low_res; v2 pipeline). Phase C2 measures how each contributes independently to the B2 drop.

## DegradeConfig override

| Axis | Legacy Phase C value (at axis = active, level L) | Phase C2 value (axis ∈ {resolution, blur, salt_pepper}, level L) |
|---|---|---|
| `low_res` | active axis: `DEGRADATION_LEVELS[L]["low_res"]`; otherwise `IDENTITY_VALUES["low_res"]` = 224 | active axis: `DEGRADATION_LEVELS[L]["low_res"]`; otherwise 224 (identical to legacy C for this axis pair) |
| `blur_kernel` | active axis: `DEGRADATION_LEVELS[L]["blur_kernel"]`; otherwise 1 | active axis: `DEGRADATION_LEVELS[L]["blur_kernel"]`; otherwise 1 |
| `blur_sigma` | active axis: `DEGRADATION_LEVELS[L]["blur_sigma"]`; otherwise 0.0 | active axis: `DEGRADATION_LEVELS[L]["blur_sigma"]`; otherwise 0.0 |
| `gaussian_noise_std` | active axis: `DEGRADATION_LEVELS[L]["noise_std"]`; otherwise 0.0 | **0.0 (always — protocol invariant)** |
| `salt_pepper_amount` | active axis: `DEGRADATION_LEVELS[L]["salt_pepper"]`; otherwise 0.0 | active axis: `DEGRADATION_LEVELS[L]["salt_pepper"]`; otherwise 0.0 |
| `saturation` | active axis: `DEGRADATION_LEVELS[L]["saturation"]`; otherwise 1.0 (identity) | **0.0 (always — protocol invariant)** |

Source of truth = `degrade_config_for_c2(level, axis, out_size=224)` landing at US-038 in [`src/data/degrade.py`](../src/data/degrade.py); semantics frozen in [PRD.md §4.3](../PRD.md). The helper composes `degrade_config_for(level, axis=axis)` (the legacy Phase C path, which already pins inactive axes to `IDENTITY_VALUES` per [US-003 / `docs/phase_c.md`](phase_c.md)) and then forcibly overrides `saturation = 0.0` and `gaussian_noise_std = 0.0` — making the two protocol-invariant axes constant across every Phase C2 cell. Field-name mapping (`noise_std` table key → `gaussian_noise_std` DegradeConfig field, `salt_pepper` table key → `salt_pepper_amount` DegradeConfig field) is the same byte-for-byte mapping `degrade_config_for` already performs.

## T3 reuse contract

Phase C2 inherits T3 regularization byte-for-byte from Phase D + Phase B2. The dispatcher routes through the existing [`_phase_d_treatment_deltas("T3", spec.model)`](../run_systematic.py#L243-L262) helper:

- **CNN** (`resnet50`, `densenet121`): `{"dropout": 0.2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}`
- **TransNeXt** (`transnext_tiny`): `{"drop_path_rate": 0.2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}`

US-038's `_apply_b2_or_c2_treatment(spec, hparams)` recognizes `spec.phase == "C2"` as a T3 carrier (alongside `"B2"`) and stamps `phase_c2_treatment = "T3"` + `phase_c2_deltas = {...}` + `phase_c2_axis = "<resolution|blur|salt_pepper>"` onto the hparams blob. No re-tune, no per-axis hparam deviation; the L3-Optuna winners frozen in `artifacts/best_hparams/*.json` carry through unchanged.

## Tag scheme

`final_C2_L{l}_{axis}_{m}_{d}` (e.g. `final_C2_L3_resolution_resnet50_cifar10`). The `_C2_` infix + explicit axis token guarantee Phase C2 directories **cannot collide** with legacy Phase C's `final_C_L{l}_{axis}_{m}_{d}` scheme (no overlap because the prefix differs at character 6: `C_` vs `C2`). PRD §7 + US-038 mandate seven-prefix collision-safety assertions in [`run_all_phases.py`](../run_all_phases.py); the legacy Phase C check must remain `startswith("final_C_L")` (the L-prefix distinguishes it from `final_C2_*`). Active-axis token validation: only `{resolution, blur, salt_pepper}` accepted at dispatch — `noise` and `saturation` are rejected because they are protocol-invariant (always zero under the B2 protocol).

## Relationship to legacy Phase C

Phase C2 is the v4-era spiritual successor to legacy Phase C; the two are **NOT** comparable cell-by-cell because:

- Legacy Phase C has free `saturation` and `noise` axes (at identity or at the active-axis level); Phase C2 zeroes both unconditionally.
- Legacy Phase C has no T3 regularization (bare Optuna L3 winners); Phase C2 layers T3 on top.
- Legacy Phase C covers all 5 axes (150 cells); Phase C2 covers only the 3 THz-relevant spatial-domain axes (90 cells).

The Phase C2 reader should compare against the Phase A clean baseline (`final_clean_{m}_{d}`) for the absolute drop attributable to the named axis under the THz protocol, NOT against legacy Phase C. The Δ_A→C2 read is what feeds the v4 axis-attribution surface; see [PRD.md §G3](../PRD.md). For legacy Phase C rationale + the US-003 identity-rewrite, see [`docs/phase_c.md`](phase_c.md).

## degradation_levels_hash invariance

Phase C2 does **NOT** modify [`src/data/degradation_levels.py`](../src/data/degradation_levels.py). `DEGRADATION_LEVELS`, `IDENTITY_VALUES`, and `AXIS_KEYS` carry forward verbatim from v3 (US-003 identity-rewrite + 2026-05-14 blur-rescale stay in place). The C2 DegradeConfig override is constructed at *DegradeConfig instantiation time* (in `degrade_config_for_c2`), not by editing the 5-level table or the identity dict — so `degradation_levels_hash` does **not** rotate. The per-sample deterministic seeding contract (`seed = idx + SEED_OFFSET_VAL`) holds across Phase C, Phase C2, Phase B1, Phase B2, and Phase D under any given (val-idx, level) pair. **`PIPELINE_VERSION = 2`** carries forward to every Phase C2 `metrics.json` — the v3 contract is preserved exactly.

## Code paths

(All references below are post-US-038 — the v4 code-wiring story. US-037 documents the design; US-038 lands the code.)

- [`src/data/degrade.py`](../src/data/degrade.py) — `degrade_config_for_c2(level, axis, out_size=224) -> DegradeConfig` (US-038). Composes `degrade_config_for(level, axis=axis)` and then overrides `saturation = 0.0` + `gaussian_noise_std = 0.0`.
- [`src/experiments/cells.py`](../src/experiments/cells.py) — `PHASE_C2_TREATMENT = "T3"`; `PHASE_C2_AXES = ("resolution", "blur", "salt_pepper")`; `iter_cells(include_phase_c2=True)` emits Phase C2 tags in canonical (level, axis, model, dataset) order. `EXPECTED_TOTAL_WITH_ALL = 402`.
- [`src/experiments/matrix.py`](../src/experiments/matrix.py) — `build_final_matrix(include_phase_c2=True)` routes Phase C2 cells through `degrade_config_for_c2(level, axis)`; treatment marker propagates into `CellSpec.treatment`.
- [`run_systematic.py`](../run_systematic.py) — `_apply_b2_or_c2_treatment(spec, hparams)` recognizes `phase == "C2"` as a T3 carrier; merges T3 deltas into best_params; stamps `phase_c2_treatment`, `phase_c2_deltas`, `phase_c2_axis` onto the hparams blob.
- [`run_all_phases.py`](../run_all_phases.py) — `FINAL_PHASES = ("A", "B", "B2", "B2nr", "C", "C2", "D")`; `--phase C2` routes `include_phase_c2=True`; optional `--axes resolution,blur,salt_pepper` filter for staged dispatch (mirrors v2/v3 Phase C `--axes` flag).
- [`scripts/run_ralph_loop.py`](../scripts/run_ralph_loop.py) — accepts `--phase C2` and `--axes` filter; iterates the (level, axis) products.
- [`scripts/update_final_exp.py`](../scripts/update_final_exp.py) — `_phase_c2_table` renderer gated on `phase_c2_present_on_disk`. `--check` arithmetic reports 402 canonical when all phases present.

## Migration note

Phase C2 has no v1/v2/v3 counterpart — it is a v4-era construct executed for the first time on the v2 pipeline under the THz-protocol DegradeConfig override. Every `metrics.json` under `runs/final/final_C2_*` carries:
- `phase: "C2"`
- `treatment: "T3"`
- `phase_c2_treatment: "T3"`
- `phase_c2_deltas` = the dict literal returned by `_phase_d_treatment_deltas("T3", model_name)` (same byte-for-byte as Phase B2 / Phase D T3 deltas)
- `phase_c2_axis` ∈ `{"resolution", "blur", "salt_pepper"}`
- `pipeline_version: 2`
- `degrade_config` with `saturation = 0.0` and `gaussian_noise_std = 0.0` explicit (both verifiable via determinism-test substring assertions)

These fields give per-axis post-hoc attribution: any Phase C2 cell's delta vs its Phase A clean sibling can be traced to the exact (active axis, level, treatment) triple that produced it without re-reading dispatch logs.

## Baseline comparison rule

Every Phase C2 cell `final_C2_L{l}_{axis}_{m}_{d}` is compared against its **Phase A clean sibling** `final_clean_{m}_{d}`. The Δ_A→C2 read is the absolute drop attributable to the named axis at level L under the THz protocol (sat = 0 + noise = 0 + T3). This is the primary axis-attribution metric.

Secondary comparisons (for context, not for headline numbers):
- Phase B2 sibling `final_B2_L{l}_{m}_{d}` at the same level — gives the *combined-axes vs single-axis* delta, indicating how much of the Phase B2 drop is reproduced by isolating a single axis. Sum of per-axis Δ_A→C2 should approximately bound (not equal) Δ_A→B2.
- Legacy Phase C sibling `final_C_L{l}_{axis}_{m}_{d}` — **not** directly comparable (different protocol; see "Relationship to legacy Phase C" above). Useful only as a sanity check that the active-axis severity reaches each model the same way under both protocols.

US-039 captures the 6 Phase A clean `metrics.json` SHA-256 hashes as part of the extended 66-entry baseline manifest at `artifacts/validation/phase_b2_baseline_manifest.json`; the manifest is re-verified at US-048 plot rendering so the comparison set is frozen for the duration of the Phase C2 campaign.
