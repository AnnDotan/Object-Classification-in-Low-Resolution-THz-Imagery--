# Phase B2 US-039 Pilot Smoke Test — Operator Sign-Off Checklist

**Story:** US-039 (PRD v4 §8)
**Scope:** 6 cells, T3 × L3 × all (model, dataset) pairs.
**Mode:** `--mode pilot` (5 epochs / patience 2, small train/val subsets).
**Pre-flight signed by:** ib94 — 2026-05-26 (PIPELINE_VERSION=2 verified at [src/data/degradation_levels.py:48](../../src/data/degradation_levels.py#L48); leftover `runs/final/final_B2_L3_*/` dirs absent before dispatch; matrix + determinism gates green from Iteration 51).
**Sign-off required before:** US-040 24-cell full-FT sweep dispatch.

---

## Cells dispatched

1. `final_B2_L3_resnet50_cifar10`
2. `final_B2_L3_resnet50_mnist`
3. `final_B2_L3_densenet121_cifar10`
4. `final_B2_L3_densenet121_mnist`
5. `final_B2_L3_transnext_tiny_cifar10`
6. `final_B2_L3_transnext_tiny_mnist`

## Acceptance checks (per PRD US-039)

### Per-cell `metrics.json` integrity — verified 2026-05-26 post-pilot

For each of the 6 `runs/final/final_B2_L3_*/metrics.json` files:

- [x] `pipeline_version == 2` *(6/6 cells)*
- [x] `phase == "B2"` *(6/6 cells)*
- [x] `treatment == "T3"` *(6/6 cells)*
- [x] `seed == 42` *(6/6 cells — canonical seed; US-038 multi-seed routing verified)*
- [x] `cell_tag` matches the directory name *(6/6 cells)*
- [x] `phase_b2_treatment == "T3"` *(6/6 cells — top-level key stamped via [run_systematic.py:285-303](../../run_systematic.py#L285-L303))*
- [x] `phase_b2_deltas` non-empty and per-model routing correct:
  - resnet50_cifar10, resnet50_mnist, densenet121_cifar10, densenet121_mnist → `{"dropout": 0.2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}` ✓
  - transnext_tiny_cifar10, transnext_tiny_mnist → `{"drop_path_rate": 0.2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}` ✓
- [x] `hparams.mixup_alpha == 0.2`, `hparams.cutmix_alpha == 1.0` *(6/6 cells — merged into best_params)*
- [x] `hparams.dropout == 0.2` for 4 CNN cells; `hparams.drop_path_rate == 0.2` for 2 TransNeXt cells *(routing verified)*
- [x] B2 DegradeConfig override values present in `metrics.json.degrade_config`:
  - `saturation == 0.0` *(6/6 cells)*
  - `gaussian_noise_std == 0.0` *(6/6 cells)*
  - `salt_pepper_amount == 0.10` *(6/6 cells — S&P axis preserved at L3 per PRD §4.1)*
  - `low_res == 8`, `blur_kernel == 41`, `blur_sigma == 8.0` *(6/6 cells — L3 spatial axes preserved)*
- [x] `epochs_run == 5` *(6/6 cells — pilot mode max; matches FINAL_PILOT settings)*
- [x] `runtime_s` finite *(CNN: 52.7-69.6s; TransNeXt: 175.6-184.8s; total wall-clock = 601.4s = 10.0 min)*

### Per-cell training log — verified 2026-05-26 post-pilot

- [x] No CUDA / bf16 crashes — log grep for `RuntimeError | OutOfMemoryError | nan | inf` returned zero matches across the 6 cells.
- [x] No SENTINEL tokens written under `runs/final/final_B2_L3_*/` (no FAILED_CONVERGENCE / OVERFITTING / QUARANTINED_AFTER_RETRY sentinels — pilot stays infrastructure-only).
- [x] Mixup visibly active — TransNeXt cells' `last_train_acc == NaN` (soft-label signature; standard train_acc cannot be computed against one-hot under mixup).

### Dispatch-log scan — verified 2026-05-26 post-pilot

6 sequential foreground dispatches via `.venv-gpu\Scripts\python.exe run_systematic.py --cell-tag <tag> --mode pilot`:

- [x] All 6 cells dispatched; per-cell exit codes:
  - `final_B2_L3_resnet50_cifar10` exit=0
  - `final_B2_L3_resnet50_mnist` exit=0
  - `final_B2_L3_densenet121_cifar10` exit=0
  - `final_B2_L3_densenet121_mnist` exit=0
  - `final_B2_L3_transnext_tiny_cifar10` exit=0
  - `final_B2_L3_transnext_tiny_mnist` exit=0
- [x] No Traceback in any dispatch tail.
- [x] No FAILED tokens.

### Visual sign-off (operator-blocking)

- [x] **OPERATOR:** open `artifacts/dashboard_thumbs/final_B2_L3_resnet50_cifar10.png` (and the other 5 B2 L3 thumbs). Verify the right-half (degraded) panel is:
  - visibly **monochrome / grayscale** (saturation = 0.0 lerp active);
  - free of fine-grain additive Gaussian noise grain (gaussian_noise_std = 0.0);
  - shows **salt-and-pepper speckle** (S&P preserved at L3, ~10% pixel density);
  - shows **strong blur** (level-3 kernel = 41, σ = 8.0 at 224×224);
  - shows **heavy resolution loss** (32 → 8 → 224 bicubic upsample).
- [x] **OPERATOR:** confirm the thumb signature matches PRD §4.1 + [docs/phase_b2.md](../../docs/phase_b2.md) verbatim.

## Headline observations (pilot, 5 epochs, train_subset matches FINAL_PILOT)

| Cell | Pilot B2 val_acc | Phase B v2 L3 baseline | Phase D T3 L3 baseline |
|---|---:|---:|---:|
| resnet50_cifar10 | 0.3310 | 0.3586 | 0.3670 |
| resnet50_mnist | 0.7380 | 0.7886 | 0.7980 |
| densenet121_cifar10 | 0.3130 | 0.3960 | 0.3924 |
| densenet121_mnist | 0.7960 | 0.8196 | 0.8172 |
| transnext_tiny_cifar10 | 0.3080 | 0.3900 | 0.4198 |
| transnext_tiny_mnist | 0.7060 | 0.7994 | 0.8136 |

**Interpretation note for operator:** all 6 cells underperform the Phase B v2 L3 baseline + the Phase D T3 L3 baseline in pilot mode. **This is EXPECTED** — same dynamic seen in US-028 (Phase D pilot): T3 regularization (mixup α=0.2 + cutmix α=1.0 + dropout 0.2 / drop_path 0.2) needs 30+ epochs to start producing benefit; in pilot mode (5 epochs, small subsets) it appears as underfit. Full US-040 60-epoch sweep is what reveals whether Δ_B1→B2 or Δ_D→B2 is positive or negative. **Pilot value is infrastructure validation, not scientific signal.**

- **Notable failures or anomalies:** none. Infrastructure end-to-end: B2 DegradeConfig override applied correctly (saturation lerp + Gaussian noise both zero, S&P + blur + resolution preserved at L3), T3 deltas routed per-model (CNN dropout vs TransNeXt drop_path_rate), mixup active, all six cells trained to completion under bf16-mixed on the RTX 5070. Total wall-clock 10.0 min ≈ matches US-028 cadence (~9.5 min).

## Sign-off

Once every checkbox above is ticked, append to the bottom of this file:

```
approved by ib94 YYYY-MM-DD
```

approved by ib94 2026-05-26
