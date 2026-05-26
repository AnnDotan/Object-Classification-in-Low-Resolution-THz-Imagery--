# Phase B2 (+ Phase B2-no-regularization) — THz-protocol simplification

**Status:** US-036 closed 2026-05-26, US-037 closed 2026-05-26 — rationale documented here; code wiring lands at US-038 ([PRD.md §8 US-038](../PRD.md)). Phase B2 dispatch is scheduled at US-039 (pilot) + US-040 (full sweep); Phase B2-no-regularization at US-041.

---

## What changed

| | Pre-Phase B2 (Phase D T3 baseline @ v3 close) | Post-Phase B2 (v4) |
|---|---|---|
| Hparams | L3-Optuna winners + T3 regularization delta (full color + additive Gaussian noise distribution) | L3-Optuna winners + T3 delta **applied under a THz-protocol DegradeConfig override** (sat = 0 + noise_std = 0) |
| Regularization scope | T3 (`dropout=0.2` + `mixup_alpha=0.2` + `cutmix_alpha=1.0` for CNN; `drop_path_rate=0.2` + same mixup/cutmix for TransNeXt) | **identical T3 deltas reused byte-for-byte** — Phase B2 is a DegradeConfig change, not an hparam change |
| Cells | 276 (Phases A + B + C + D) | **306** when `include_phase_b2=True` — 30 new Phase B2 cells (5 levels × 3 models × 2 datasets) appended. **312** when both `include_phase_b2=True` and `include_phase_b2nr=True` (6 additional L3 cells for the no-regularization arm) |

## Why

*Does removing saturation and additive Gaussian noise (THz-protocol simplification) close the v2 −12.49 pp collapse?*

This is one of the **operator-locked headline research questions** from [PRD.md §1](../PRD.md). The v2 Phase B pipeline applies five orthogonal degradation axes simultaneously: resolution + blur + additive Gaussian noise + salt-and-pepper + saturation lerp. True low-resolution THz imagery has a different noise structure — sensor read-noise dominates at low frequencies and is closer to monochrome/quasi-grayscale than to RGB. Phase B2 strips the two axes that are *least* THz-faithful (color saturation lerp + additive Gaussian noise) and re-trains under that simplified protocol. The Δ_B1→B2 read decomposes into two components: a *protocol-simplification* effect (the pure DegradeConfig change) and a *regularization-helps-with-easier-distribution* effect (the T3 deltas that were already in place at Phase D). US-041's Phase B2-no-regularization arm separates these — see "Phase B2-no-regularization (B2nr)" below.

## DegradeConfig override

| Axis | Phase B1 value at level L | Phase B2 value at level L |
|---|---|---|
| `low_res` | `DEGRADATION_LEVELS[L]["low_res"]` (18/12/8/6/3 for L1..L5) | **same** — flows through unchanged |
| `blur_kernel` | `DEGRADATION_LEVELS[L]["blur_kernel"]` (13/25/41/61/91) | **same** — flows through unchanged |
| `blur_sigma` | `DEGRADATION_LEVELS[L]["blur_sigma"]` (2.50/5.00/8.00/12.00/18.00) | **same** — flows through unchanged |
| `gaussian_noise_std` | `DEGRADATION_LEVELS[L]["noise_std"]` (0.04/0.08/0.12/0.16/0.22) | **0.0 (override)** — additive-noise step skipped by `> 0` guard at [`src/data/degrade.py:153`](../src/data/degrade.py#L153) |
| `salt_pepper_amount` | `DEGRADATION_LEVELS[L]["salt_pepper"]` (0.03/0.06/0.10/0.14/0.18) | **same** — flows through unchanged |
| `saturation` | `DEGRADATION_LEVELS[L]["saturation"]` (0.95/0.65/0.40/0.15/0.00) | **0.0 (override)** — full-grayscale via deterministic lerp at [`src/data/degrade.py:120-128`](../src/data/degrade.py#L120-L128) |

Source of truth = `degrade_config_for_b2(level, out_size=224)` landing at US-038 in [`src/data/degrade.py`](../src/data/degrade.py); semantics frozen in [PRD.md §4.1](../PRD.md). DegradeConfig field names: `gaussian_noise_std`, `salt_pepper_amount`, `saturation` (the `level_params` table uses the shorter keys `noise_std` / `salt_pepper`; the helper maps table-key → DegradeConfig-field byte-for-byte the same way [`degrade_config_for`](../src/data/degrade.py#L197-L234) already does).

## T3 reuse contract

Phase B2 inherits T3 regularization byte-for-byte from Phase D. The dispatcher routes through the existing [`_phase_d_treatment_deltas("T3", spec.model)`](../run_systematic.py#L243-L262) helper:

- **CNN** (`resnet50`, `densenet121`): `{"dropout": 0.2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}`
- **TransNeXt** (`transnext_tiny`): `{"drop_path_rate": 0.2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}`

US-038 adds a new helper `_apply_b2_or_c2_treatment(spec, hparams)` that recognizes `spec.phase ∈ {"B2", "C2"}`, calls the same `_phase_d_treatment_deltas("T3", spec.model)` route, and stamps `phase_b2_treatment = "T3"` + `phase_b2_deltas = {...}` (or the C2 equivalent) onto the hparams blob. No re-tune, no per-cell deviation; the L3-Optuna winners frozen in `artifacts/best_hparams/*.json` carry through unchanged.

## Phase B2-no-regularization (B2nr)

Six L3 cells (L3 × 3 models × 2 datasets) re-train under the **B2 DegradeConfig** (`degrade_config_for_b2(level=3)` — sat = 0 + noise = 0) **without** the T3 deltas. The bare Optuna L3 winners (no `dropout` override, no `mixup_alpha`, no `cutmix_alpha`) are used directly. `metrics.json` carries `phase: "B2nr"`, `treatment: null` (or absent), and `phase_b2nr_deltas: {}` (explicit empty dict so the schema field stays non-null).

The B2nr arm gives the pure-protocol Δ_B1→B2nr at L3 — factoring T3 out of the Phase B2 attribution. Combined with Δ_D→B2 (the pure protocol-simplification at fixed T3 regularization) and Δ_B1→B2 (combined-effect read), the report reader can decompose Phase B2's drop into:

- **Protocol-simplification effect:** Δ_B1→B2nr (L3 only)
- **T3-regularization-helps-with-easier-distribution effect:** Δ_B2nr→B2 ≈ Δ_D→B2 at L3

See [PRD.md §12 decision #16](../PRD.md) for the operator-locked attribution decomposition.

## Tag scheme

`final_B2_L{l}_{m}_{d}` (e.g. `final_B2_L3_resnet50_cifar10`) for Phase B2; `final_B2nr_L3_{m}_{d}` (e.g. `final_B2nr_L3_transnext_tiny_mnist`) for Phase B2-nr. The `_B2_` / `_B2nr_` infixes guarantee Phase B2 directories **cannot collide** with the legacy Phase B's `final_B_L{l}_{m}_{d}` scheme — but a naive `startswith("final_B")` would match all three. PRD §7 + US-038 mandate the legacy Phase B check use `startswith("final_B_L")` (the L-prefix distinguishes Phase B from Phase B2 / B2-nr). [`run_all_phases.py`](../run_all_phases.py) enforces this with seven-prefix collision-safety assertions; multi-seed audit tags (`<base>_seed{N}`) are matched against the base prefix.

## degradation_levels_hash invariance

Phase B2 does **NOT** modify [`src/data/degradation_levels.py`](../src/data/degradation_levels.py). `DEGRADATION_LEVELS` and `IDENTITY_VALUES` carry forward verbatim from v3. The B2 DegradeConfig override is constructed at *DegradeConfig instantiation time* (in `degrade_config_for_b2`), not by editing the 5-level table — so `degradation_levels_hash` does **not** rotate. The per-sample deterministic seeding contract (`seed = idx + SEED_OFFSET_VAL`) holds; Phase B1, Phase D, and Phase B2 share the same per-sample seeds at any given (val-idx, level) pair. **`PIPELINE_VERSION = 2`** carries forward to every Phase B2 / B2-nr `metrics.json` — the v3 contract is preserved exactly.

## Code paths

(All references below are post-US-038 — the v4 code-wiring story. US-037 documents the design; US-038 lands the code.)

- [`src/data/degrade.py`](../src/data/degrade.py) — `degrade_config_for_b2(level, out_size=224) -> DegradeConfig` (US-038). Builds from `DEGRADATION_LEVELS[level]` with `saturation = 0.0` and `gaussian_noise_std = 0.0` explicit overrides; all other axes flow through unchanged.
- [`src/experiments/cells.py`](../src/experiments/cells.py) — `PHASE_B2_TREATMENT = "T3"`; `iter_cells(include_phase_b2=True, include_phase_b2nr=True)` emits Phase B2 / B2-nr tags in canonical order (A → B → B2 → B2-nr → C → C2 → D). `EXPECTED_TOTAL_WITH_ALL = 402`.
- [`src/experiments/matrix.py`](../src/experiments/matrix.py) — `build_final_matrix(include_phase_b2=True, include_phase_b2nr=True)` routes Phase B2 / B2-nr cells through `degrade_config_for_b2(level)`; treatment marker propagates into `CellSpec.treatment`.
- [`run_systematic.py`](../run_systematic.py) — `_apply_b2_or_c2_treatment(spec, hparams)` recognizes `phase ∈ {"B2"}` as a T3 carrier; `phase == "B2nr"` skips treatment routing entirely (`treatment = None`, `phase_b2nr_deltas = {}`).
- [`run_all_phases.py`](../run_all_phases.py) — `FINAL_PHASES = ("A", "B", "B2", "B2nr", "C", "C2", "D")`; `--phase {B2, B2nr}` flags route the corresponding `include_phase_*` kwargs; seven-prefix collision assertions enforce `startswith("final_B_L")` for legacy Phase B.
- [`scripts/update_final_exp.py`](../scripts/update_final_exp.py) — `_phase_b2_table` / `_phase_b2nr_table` renderers gated on `phase_b2_present_on_disk` / `phase_b2nr_present_on_disk`. `--check` arithmetic reports 402 canonical when all phases present.

## Migration note

Phase B2 has no v1/v2/v3 counterpart — it is a v4-era construct executed for the first time on the v2 pipeline under the simplified DegradeConfig. Every `metrics.json` under `runs/final/final_B2_*` carries:
- `phase_b2_treatment` = `"T3"`
- `phase_b2_deltas` = the dict literal returned by `_phase_d_treatment_deltas("T3", model_name)` (same byte-for-byte as Phase D T3 deltas)
- `phase: "B2"`
- `treatment: "T3"`
- `pipeline_version: 2`

Every `metrics.json` under `runs/final/final_B2nr_*` carries:
- `phase: "B2nr"`
- `treatment: null` (or absent)
- `phase_b2nr_deltas: {}` (explicit empty dict for schema stability)
- `pipeline_version: 2`

These fields give post-hoc attribution: any Phase B2 cell's delta vs its Phase B1 sibling, its Phase D T3 sibling, or its Phase B2-nr L3 sibling can be traced to the exact DegradeConfig override + treatment recipe that produced it without re-reading dispatch logs.

## Baseline comparison rules

Every Phase B2 cell `final_B2_L{l}_{m}_{d}` is compared against three reference cells:

1. **Phase B1 v2 sibling** `final_B_L{l}_{m}_{d}` — gives the combined-effect Δ_B1→B2 (protocol simplification + T3 regularization).
2. **Phase D T3 sibling** `final_D_T3_L{l}_{m}_{d}` — gives the pure protocol-simplification Δ_D→B2 at fixed T3 regularization.
3. **Phase A clean sibling** `final_clean_{m}_{d}` — gives the absolute drop from clean baselines for the v4 axis-attribution surface.

Every Phase B2-nr cell `final_B2nr_L3_{m}_{d}` is compared against its Phase B1 L3 sibling `final_B_L3_{m}_{d}` only — that is the pure-protocol Δ_B1→B2nr metric at no regularization.

US-039 captures SHA-256 of all 66 reference `metrics.json` files (30 Phase B1 + 30 Phase D T3 + 6 Phase A clean) into `artifacts/validation/phase_b2_baseline_manifest.json` and re-verifies at US-048 plot rendering. The comparison set is frozen for the duration of the Phase B2 + B2-nr campaign.
