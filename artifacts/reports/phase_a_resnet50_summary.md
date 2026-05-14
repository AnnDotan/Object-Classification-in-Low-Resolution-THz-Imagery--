# Phase A — ResNet50 Summary

**Story:** US-006 (Phase A · resnet50 × {cifar10, mnist}). Authoritative trackers: [`Final_Exp.json`](../../artifacts/Final_Exp.json), [`Final_Exp.md`](../../Final_Exp.md).

## Results

| Cell | best_val_acc | best_epoch | epochs_run | train@best (proxy) | gap@best | PSNR / SSIM |
|---|---:|---:|---:|---:|---:|---|
| `final_clean_resnet50_cifar10` | **0.9518** | 13 | 23 | 1.000 (last) | ~5 pp (last) | N/A (Phase A clean) |
| `final_clean_resnet50_mnist`   | **0.9914** | 11 | 21 | 1.000 (last) | ~1 pp (last) | N/A (Phase A clean) |

PSNR/SSIM are not defined for Phase A: the pipeline runs `clean → clean` so the degraded tensor is byte-identical to the source. `_measure_image_quality_for_cell` only writes `image_quality.json` for Phase B/C. The dashboard thumb at `artifacts/dashboard_thumbs/<tag>.png` is still produced and shows the clean side-by-side.

## Gap to paper baseline (TResNet)

The TResNet paper does not publish a directly-comparable resnet50 row on a 10K-train subset of CIFAR-10 / MNIST at 224×224. Anchoring qualitatively against the ImageNet-pretrained fine-tuning literature: full-50K-train CIFAR-10 with ImageNet-init ResNet50 typically lands in the **96–97 %** band at 224 input, MNIST saturates around **99.5 %**. Our 10K-train Phase A run reaches **95.18 % / 99.14 %** — within 1–2 pp of the literature band, consistent with the train-set reduction (1/5 of the full set). No paper-baseline regression flag is warranted at this point.

## NaN / divergence flags

- **NaN/Inf loss:** none in either history.json.
- **Late-stage divergence:** none — both cells early-stopped (patience 10), best_epoch ≪ epochs_run; `last_val_acc` is within 0.2 pp of `best_val_acc` on both cells.
- **Pathology guard (§6.3):** `healthy` on both (gap < 12 pp, val above 1.3× random, no NaN).

## Narrative (≤200 words)

ResNet50 Phase A baselines land in the expected band: 95.18 % on CIFAR-10 and 99.14 % on MNIST under the convergence-first protocol (60 ep / patience 10 / bf16 / AdamW + cosine; head 1e-3, backbone 1e-4, weight_decay 1e-4, label_smoothing 0.1, warmup 3). Both runs early-stopped well before the cap (best at epoch 13 / 11 of 23 / 21 trained), with last_train_acc saturating at 1.000 while val_acc held — consistent with healthy ImageNet-pretrained fine-tuning at 224×224. The pathology guard returns `healthy` on both; no NEEDS_FULL_FT sentinel.

**Provenance.** Run dirs are from the legacy 4050 box, preserved across the 2026-05-12 reset. They predate `degradation_levels_hash`, but Phase A is unaffected — the degradation table is not exercised on clean cells. The 5070 + bf16 + new blur curve will be picked up the first time a *degraded* cell trains.

**Outstanding for US-006.** VALIDATOR seed=43 re-run on `final_clean_resnet50_cifar10` (line 415, GPU-gated, ±0.5 pp tolerance); LIBRARIAN `docs/phase_a.md` + README "Best Results" row (417); DESIGNER per-US trend section in `Final_Exp.html` (418); HALT for operator approval (420).
