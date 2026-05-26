# Phase D — Regularization Sweep

**Status:** US-026 closed 2026-05-23, US-027 closed 2026-05-23 — treatments wired in [`src/experiments/cells.py`](../src/experiments/cells.py) + [`run_systematic.py`](../run_systematic.py); rationale documented here.

---

## What changed

| | Pre-Phase D (Phase B v2 baseline) | Post-Phase D (current) |
|---|---|---|
| Hparams | L3-Optuna winners frozen in `artifacts/best_hparams/{model}_{dataset}.json` | L3-Optuna winners **+** per-cell regularization delta (T1/T2/T3) |
| Regularization scope | Whatever Optuna picked (typically `dropout=0.0`, no mixup/cutmix) | Three explicit treatment arms — architectural dropout, label-mixing, combo |
| Cells | 186 (Phases A + B + C) | **276** when `include_phase_d=True` — Phase A/B/C byte-identical, 90 new Phase D cells appended |

## Why

*Does targeted regularization recover the Phase B v2 −12.49 pp mean collapse vs v1?*

This is the **operator-locked headline research question** from [PRD.md §1](../PRD.md). The v2 noise / salt-and-pepper pipeline fix (US-017, `PIPELINE_VERSION = 2`) corrected a pre-upsample noise application that had been bicubic-averaging high-frequency content out of the training distribution. The Phase B re-run under the corrected pipeline produced a **−12.49 pp mean val_acc collapse** across 30 cells (worst hit: `final_B_L3_transnext_tiny_cifar10` at −32.74 pp). The Optuna winners were tuned at L3 against the v1 pipeline and had no exposure to v2's coarse-grain noise structure (lag-1 spatial autocorrelation ≈ 0.993). Phase D layers three orthogonal regularization recipes on top of the frozen v2 baselines so each delta is attributable: T1 isolates architectural regularization (head dropout for CNNs, stochastic depth for TransNeXt), T2 isolates label-mixing (mixup only), T3 stacks both plus cutmix as the kitchen-sink ceiling. The 5-level × per-(model, dataset) breadth lets the recovery curve be read per-axis-severity rather than as a single scalar.

## Treatments

| Treatment | CNN delta | TransNeXt delta | Pattern |
|---|---|---|---|
| **T1 — Architectural dropout** | `dropout=0.2` (timm `drop_rate`) | `drop_path_rate=0.2` (stochastic depth) | Isolates architectural regularization. CNN dropout flows through timm's classifier head; TransNeXt routes to stochastic depth because timm CNNs do not honor `drop_path_rate`. |
| **T2 — Label-mixing** | `mixup_alpha=0.2, cutmix_alpha=0.0` | `mixup_alpha=0.2, cutmix_alpha=0.0` | Isolates label-mixing regularization. Model-agnostic; `timm.data.Mixup` wraps targets identically for all three architectures. |
| **T3 — Combo (kitchen-sink)** | `dropout=0.2` + `mixup_alpha=0.2` + `cutmix_alpha=1.0` | `drop_path_rate=0.2` + `mixup_alpha=0.2` + `cutmix_alpha=1.0` | Most aggressive combined regularization. T1 ∪ T2 with cutmix enabled at α=1.0. If recovery is achievable, T3 should show the strongest signal. |

Source of truth: [`run_systematic.py:_phase_d_treatment_deltas`](../run_systematic.py#L243-L262). The per-model routing (`is_transnext = model_name.startswith("transnext_")`) ensures byte-correct delta selection; mixup/cutmix keys are model-agnostic and applied uniformly.

## Tag scheme

`final_D_{T}_L{l}_{m}_{d}` — e.g. `final_D_T3_L5_resnet50_cifar10`. The `_D_` infix and explicit treatment token guarantee Phase D directories **cannot collide** with Phase B's `final_B_L{l}_{m}_{d}` scheme; `run_all_phases.py` enforces this with a defense-in-depth assertion at dispatch (`assert not bad` for any non-`final_D_*` tag in a `--phase D` matrix — see [`run_all_phases.py:380-390`](../run_all_phases.py#L380-L390)). Enumeration is delegated to [`iter_cells(include_phase_d=True)`](../src/experiments/cells.py#L79-L87).

## degradation_levels_hash invariance

Phase D does **NOT** modify [`src/data/degradation_levels.py`](../src/data/degradation_levels.py) or any `DegradeConfig` values. Every Phase D cell shares its Phase B sibling's `DegradeConfig` byte-for-byte ([`src/experiments/matrix.py:90-92`](../src/experiments/matrix.py#L90-L92): *"Phase D cells share Phase B's DegradeConfig (all 5 axes at level L); the treatment dimension is encoded only in `tag` and `treatment`, not in the pixel pipeline."*). `PIPELINE_VERSION = 2` carries forward verbatim. `degradation_levels_hash` does **not** rotate — Phase B v2 and Phase D siblings produce byte-identical per-sample noise patterns under the deterministic per-sample seeding (`seed = idx + SEED_OFFSET_VAL`). The only axis of variation between a `final_B_L3_resnet50_cifar10` cell and a `final_D_T3_L3_resnet50_cifar10` cell is the hparam blob layered on top of the L3-Optuna winners.

## Code paths

- [`src/experiments/cells.py:37-87`](../src/experiments/cells.py#L37-L87) — `PHASE_D_TREATMENTS = ("T1", "T2", "T3")`, `EXPECTED_COUNTS_WITH_D["D"] = 90`, `iter_cells(include_phase_d=True)` tag generation.
- [`src/experiments/matrix.py:78-114`](../src/experiments/matrix.py#L78-L114) — `build_final_matrix(include_phase_d=True)` returns 276 specs; treatment marker propagates into `CellSpec.treatment`; `degrade_config_for(level, axis=None)` produces the **same** Phase B config for the Phase D sibling.
- [`run_systematic.py:234-282`](../run_systematic.py#L234-L282) — `_phase_d_treatment_deltas(treatment, model_name)` returns the regularization-delta dict; `_apply_phase_d_treatment(spec, hparams)` merges deltas into `best_params` and stamps `phase_d_treatment` + `phase_d_deltas` onto the hparams blob.
- [`run_all_phases.py:260-264, 370-390`](../run_all_phases.py#L370-L390) — `FINAL_PHASES = ("A", "B", "C", "D")`, `include_phase_d = (phase == "D")`, and the `final_D_*` tag-prefix assertion that prevents Phase B directories from being overwritten.
- [`scripts/update_final_exp.py:223-311`](../scripts/update_final_exp.py#L223-L311) — `_phase_d_table(...)` renders the 90-cell `Final_Exp.md` table; `_phase_d_present_on_disk(runs_root)` gates rendering so the default 186-cell view stays byte-identical until at least one `final_D_*` directory exists.

## Migration note

Phase D has no v1 counterpart — it is a v3-era construct executed for the first time on the v2 pipeline. Every `metrics.json` under `runs/final/final_D_*` carries:
- `phase_d_treatment` (one of `"T1"`, `"T2"`, `"T3"`)
- `phase_d_deltas` (the dict literal returned by `_phase_d_treatment_deltas`)
- `pipeline_version = 2`

These three fields give post-hoc attribution: any cell's delta vs its Phase B sibling can be traced to the exact regularization recipe that produced it without re-reading the dispatch logs.

## Baseline comparison rule

Every Phase D cell `final_D_T{x}_L{l}_{m}_{d}` is compared **only** against its Phase B v2 sibling `final_B_L{l}_{m}_{d}`. v1 numbers are out of scope — the −12.49 pp collapse magnitude in [PRD.md §1](../PRD.md) is a v2-vs-v1 quantity, and Phase D measures recovery against the **v2** floor, not the v1 ceiling. The SHA-256 of all 30 Phase B v2 `metrics.json` files is captured at US-028 pre-flight into `artifacts/validation/phase_b_v2_baseline_manifest.json` and re-verified at US-031 plot rendering, so the comparison set is frozen for the duration of the Phase D campaign.
