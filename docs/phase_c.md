# Phase C — Single-Axis Isolation Semantics

**Status:** US-003 (2026-05-14) — change ratified by operator; implemented in [`src/data/degradation_levels.py`](../src/data/degradation_levels.py) and [`src/data/degrade.py`](../src/data/degrade.py).

---

## What changed

| | Pre-US-003 (legacy) | Post-US-003 (current) |
|---|---|---|
| Active axis | At level L | At level L |
| Inactive axes | At **L1 mild** values | At **identity** (no degradation) |
| Phase C L1 cell == Phase B L1 cell? | Yes (all 5 axes at L1) | **No** (1 axis at L1, 4 at identity) |

## Identity values

| Axis | Inactive value | Effect |
|---|---|---|
| `resolution` (`low_res`) | 224 | downsample stage short-circuits to a single bicubic upsample (matches Phase A clean baseline) |
| `blur` (`blur_kernel`, `blur_sigma`) | 1, 0.0 | gaussian-blur step returns `img` unchanged (`kernel<=1` guard) |
| `noise` (`noise_std`) | 0.0 | additive-noise step skipped (`>0` guard) |
| `salt_pepper` (`salt_pepper`) | 0.0 | S&P step skipped (`>0` guard) |
| `saturation` | 1.0 | saturation-lerp short-circuits (`s<1.0` guard) |

Source of truth: `IDENTITY_VALUES` in [`degradation_levels.py`](../src/data/degradation_levels.py).

## Why

Pre-US-003 a Phase C "blur at L5" cell still carried **L1 noise, L1 S&P, L1 resolution loss, and L1 saturation reduction** on top of the L5 blur. Measured accuracy drop attributed to "blur" was therefore contaminated by the L1 noise/S&P/resolution floor (small but non-zero on a 5-level curve with severe L5). The contamination grew at lower active-axis levels: at L1, the active axis was indistinguishable from the inactive floor, so the cell collapsed exactly to Phase B L1 — wasting 30 cells (the Phase C L1 row, 5 axes × 3 models × 2 datasets) on a redundant measurement.

Post-US-003 a Phase C cell isolates exactly **one** axis. The per-axis accuracy curves are now clean: Phase C `blur` at L1…L5 measures the marginal contribution of blur with every other degradation off. This is the contract the dashboard's per-axis trend graphs assume; the legacy semantics silently broke that contract.

## Consequences

- **`degradation_levels_hash` rotates.** Every Phase C cell's `metrics.json` will carry a new hash after US-003. Cross-batch comparisons against pre-US-003 Phase C cells (i.e. any run dir under `runs/final/final_C_*` that predates the US-003 commit) are **explicitly forbidden** — they measure a different quantity.
- **Phase A and Phase B unchanged.** Phase A clean baselines (`final_clean_*`) and Phase B combined cells (`final_B_L*_*`) are unaffected — `level_params(level, axis=None)` still returns the full L1…L5 dict.
- **Test invariant replaced.** `test_matrix.py::_check_phase_c_l1_collapses_to_phase_b_l1` is gone; the new `_check_phase_c_isolates_single_axis` asserts that every Phase C cell has exactly one non-identity axis matching its declared `axis` field at its declared `level`.
- **Determinism gate re-passes.** `test_degradation_determinism.py` is independent of `level_params` (it uses a fixed `DataConfig(low_res=16)`) so the MSE=0 contract at out_size=224 is preserved.

## Code paths

- `level_params(L, axis="blur")` → `{low_res: 224, blur_kernel: 41, blur_sigma: 8.00, noise_std: 0.0, salt_pepper: 0.0, saturation: 1.0}` (example at L=3, post-2026-05-14 blur rescale).
- `degrade_config_for(level=3, axis="blur", out_size=224)` → `DegradeConfig(low_res=224, blur_kernel=41, blur_sigma=8.00, gaussian_noise_std=0.0, salt_pepper_amount=0.0, saturation=1.0, degradation_type='all')`.
- `degrade_image(img, cfg, seed=…)` with the cfg above:
  1. saturation lerp: short-circuits (s == 1.0).
  2. downsample stage: `low_res == out_size == 224` → skip bilinear; single bicubic upsample 32→224 (or 28→224 for MNIST).
  3. blur: applied (kernel=41, σ=8.00 — see DEGRADATION_LEVELS L3 row).
  4. noise: short-circuits (std == 0.0).
  5. S&P: short-circuits (amount == 0.0).

So only the blur stage runs, against a clean bicubic-upsampled image — exactly the per-axis isolation Phase C is supposed to measure.

## Migration note

No `runs/final/final_C_*` cells have been executed on the RTX 5070 5070A branch as of US-003's landing — the 150-cell Phase C is the last phase of the 186-cell campaign (US-011…US-013, gated on US-004 driver). So the hash rotation has no downstream cleanup cost; the first Phase C run dirs will carry the post-US-003 hash from inception.
