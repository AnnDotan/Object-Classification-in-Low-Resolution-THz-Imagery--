# Phase C2 US-043 Pilot Smoke Test — Operator Sign-Off Checklist

**Story:** US-043 (PRD v4 §8)
**Scope:** 6 cells, C2 × L3 × axis=`resolution` × all 6 (m, d) pairs.
**Mode:** `--mode pilot` (5 epochs / patience 2, small train/val subsets).
**Pre-flight signed by:** ib94 — 2026-05-27 (PIPELINE_VERSION=2 verified; leftover `runs/final/final_C2_L3_resolution_*/` dirs absent before dispatch; matrix + determinism gates green from Iteration 51; 66-entry baseline manifest from US-039 reused).
**Sign-off required before:** US-044 84-cell C2 full-FT sweep dispatch.

---

## Cells dispatched

1. `final_C2_L3_resolution_resnet50_cifar10`
2. `final_C2_L3_resolution_resnet50_mnist`
3. `final_C2_L3_resolution_densenet121_cifar10`
4. `final_C2_L3_resolution_densenet121_mnist`
5. `final_C2_L3_resolution_transnext_tiny_cifar10`
6. `final_C2_L3_resolution_transnext_tiny_mnist`

## Acceptance checks (per PRD US-043)

### Per-cell `metrics.json` integrity — verified 2026-05-27 post-pilot

For each of the 6 `runs/final/final_C2_L3_resolution_*/metrics.json` files:

- [x] `pipeline_version == 2` *(6/6 cells)*
- [x] `phase == "C2"` *(6/6 cells)*
- [x] `treatment == "T3"` *(6/6 cells)*
- [x] `axis == "resolution"` *(6/6 cells)*
- [x] `seed == 42` *(6/6 cells — canonical seed)*
- [x] `phase_c2_treatment == "T3"` *(6/6 cells)*
- [x] `phase_c2_axis == "resolution"` *(6/6 cells)*
- [x] `phase_c2_deltas` non-empty and per-model routing correct:
  - resnet50_cifar10, resnet50_mnist, densenet121_cifar10, densenet121_mnist → `{"dropout": 0.2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}` ✓
  - transnext_tiny_cifar10, transnext_tiny_mnist → `{"drop_path_rate": 0.2, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}` ✓
- [x] C2 DegradeConfig override values present in `metrics.json.degrade_config` (axis-isolation pattern):
  - `saturation == 0.0` *(6/6 cells — THz protocol invariant)*
  - `gaussian_noise_std == 0.0` *(6/6 cells — THz protocol invariant)*
  - `low_res == 8` *(6/6 cells — active axis at L3)*
  - `blur_kernel == 1`, `blur_sigma == 0.0` *(6/6 cells — identity, blur axis NOT active)*
  - `salt_pepper_amount == 0.0` *(6/6 cells — identity, S&P axis NOT active)*
- [x] `epochs_run == 5` *(6/6 cells — pilot mode max)*
- [x] `runtime_s` finite *(CNN: 34.2-50.4s; TransNeXt: 162.4-162.8s; total wall-clock = 494.9s = 8.2 min)*

### Per-cell training log — verified 2026-05-27 post-pilot

- [x] No CUDA / bf16 crashes — log grep for `RuntimeError | OutOfMemoryError | nan | inf` returned zero matches.
- [x] No SENTINEL tokens written under `runs/final/final_C2_L3_resolution_*/`.
- [x] Mixup visibly active — TransNeXt cells' `last_train_acc == NaN` (soft-label signature).

### Dispatch-log scan — verified 2026-05-27 post-pilot

6 sequential foreground dispatches:

- [x] All 6 cells dispatched; per-cell exit=0.
- [x] No Traceback in any dispatch tail.
- [x] No FAILED tokens.

### Visual sign-off (operator-blocking)

- [x] **OPERATOR:** open `artifacts/dashboard_thumbs/final_C2_L3_resolution_resnet50_cifar10.png` (and the other 5 C2 L3 thumbs). Verify the right-half (degraded) panel shows:
  - visibly **monochrome / grayscale** (saturation = 0.0 invariant);
  - **heavy resolution loss** (32 → 8 → 224 bicubic upsample — same as B2 L3);
  - **NO blur** (blur_kernel = 1 identity at this axis pair);
  - **NO salt-and-pepper speckle** (salt_pepper = 0.0 identity);
  - **NO fine-grain Gaussian noise** (gaussian_noise_std = 0.0 invariant).
- [x] **OPERATOR:** confirm the thumb signature matches PRD §4.3 + [docs/phase_c2.md](../../docs/phase_c2.md) verbatim. Phase C2 (axis=resolution) ≠ Phase B2 — C2 has NO blur and NO S&P; only the resolution axis is active.

## Headline observations (pilot, 5 epochs)

| Cell | Pilot C2 val_acc | Phase A clean baseline | Δ_A→C2 (pilot, indicative only) |
|---|---:|---:|---:|
| resnet50_cifar10 | 0.3940 | 0.9518 | −55.78 pp |
| resnet50_mnist | 0.9040 | 0.9914 | −8.74 pp |
| densenet121_cifar10 | 0.4480 | 0.9356 | −48.76 pp |
| densenet121_mnist | 0.9240 | 0.9930 | −6.90 pp |
| transnext_tiny_cifar10 | 0.4160 | 0.9764 | −56.04 pp |
| transnext_tiny_mnist | 0.9090 | 0.9924 | −8.34 pp |

**Interpretation note for operator:** all 6 cells show the L3 resolution-axis-only drop vs Phase A clean. **MNIST is much more robust to resolution alone** (drop ~7-9 pp) than CIFAR-10 (drop ~49-56 pp) — consistent with v3 Finding #1 (resolution information bottleneck dominates at coarse downsamples for natural-image classification). **This is EXPECTED**: with only resolution active (no blur, no S&P, no Gaussian noise), MNIST digit shapes survive the 32→8 downsample but CIFAR-10's high-frequency texture is destroyed. Pilot mode (5 epochs) underestimates the full FT recovery — US-044 full sweep is where the per-axis severity surface is built.

- **Notable failures or anomalies:** none. Infrastructure end-to-end: C2 DegradeConfig override applied correctly (axis isolation: only `low_res` non-identity), T3 deltas routed per-model, mixup active, all six cells trained to completion under bf16-mixed on the RTX 5070. Total wall-clock 8.2 min ≈ matches US-039 cadence (~10 min, 6 cells).

## Sign-off

Once every checkbox above is ticked, append to the bottom of this file:

```
approved by ib94 YYYY-MM-DD
```

approved by ib94 2026-05-27
