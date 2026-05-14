# Phase A — TransNeXt-tiny Summary

**Story:** US-008 (Phase A · transnext_tiny × {cifar10, mnist}). First Phase A pair executed under the RTX 5070 / bf16 / Lightning protocol via the US-005 driver.

## Results

| Cell | best_val_acc | best_epoch | epochs_run | last_train_acc | runtime | PSNR / SSIM |
|---|---:|---:|---:|---:|---:|---|
| `final_clean_transnext_tiny_cifar10` | **0.9764** | 5 | 15 | 1.0000 | 38.3 m | N/A (Phase A clean) |
| `final_clean_transnext_tiny_mnist`   | **0.9924** | 25 | 35 | 1.0000 | 89.6 m | N/A (Phase A clean) |

Both runs early-stopped via patience=10. cifar10 plateaued fast (5 → 15); mnist climbed gradually through epoch 25 before plateauing. Total wallclock for the pair: **2 h 8 m** on the 5070.

## Cross-model Phase A comparison

| Model | cifar10 best_val_acc | mnist best_val_acc |
|---|---:|---:|
| `resnet50`        | 0.9518 | 0.9914 |
| `densenet121`     | 0.9356 | 0.9930 |
| `transnext_tiny`  | **0.9764** | 0.9924 |

**TransNeXt-tiny tops cifar10 by +2.46 pp over resnet50 and +4.08 pp over densenet121** — consistent with the US-002 closure note where transnext_tiny_cifar10 also led at the convergence protocol (best_value=0.7336 under L3 degradation; the clean baseline scales the win up to 0.9764). MNIST clusters tightly at 0.991–0.993 across all three models — the dataset is too easy to differentiate.

## Gap to paper baseline (TransNeXt)

The TransNeXt paper does not publish a direct row on a 10K-train subset of CIFAR-10 / MNIST at 224×224. Anchoring qualitatively: ImageNet-pretrained transformer fine-tuning at 224 input typically reaches **97–98 %** on CIFAR-10 with the full 50K train, and saturates around **99.5 %** on MNIST. Our 10K-train Phase A run reaches **97.64 % / 99.24 %**, well within the literature band when adjusted for the train-set reduction.

## NaN / divergence flags

- **NaN/Inf loss:** epoch 1 train metrics are null on both cells (Lightning's standard sanity-check vs. first-train-epoch ordering; not a divergence). All subsequent epochs are clean.
- **Late-stage divergence:** none — both cells early-stopped well before the 60-epoch cap with `last_val_acc` within 1 pp of `best_val_acc`.
- **Pathology guard (§6.3):** `healthy` on both (gap < 12 pp, val above 1.3× random, no NaN at epoch ≥ 2).

## Narrative (≤200 words)

TransNeXt-tiny Phase A baselines top resnet50 on CIFAR-10 by a comfortable +2.46 pp (0.9764 vs 0.9518) and tie within 0.1 pp on MNIST. Both cells trained under the convergence-first protocol (60 ep / patience 10 / bf16 / AdamW + cosine; head 2.9e-3, backbone 3.0e-5, weight_decay 0.346, label_smoothing 0.009, warmup ≈ 1 — the US-002 Stage 1.5 winner hparams loaded automatically by `run_systematic.run_cell`). cifar10 plateaued fast (best @ epoch 5 of 15 trained); mnist climbed gradually through epoch 25 before patience exhausted at 35. Train accuracy saturated at 1.000 on both, with the generalization gap remaining well under the §6.3 overfitting threshold. The pathology guard returns `healthy` on both — no NEEDS_FULL_FT sentinel.

**Provenance.** First Phase A pair run end-to-end on the RTX 5070 / bf16 / Lightning. Required two Windows-specific fixes pre-launch: `num_workers=0` (the spawn-pickle pipeline truncates on TransNeXt-tiny + swattention payloads — see feedback memory), and a Lightning callback wallclock-fields patch so runtime/started_at/finished_at land in metrics.json.

**Outstanding for US-008.** VALIDATOR seed=43 re-run on `final_clean_transnext_tiny_cifar10` (PRD line 415, GPU-gated, ±0.5 pp tolerance); LIBRARIAN `docs/phase_a.md` + README "Best Results So Far" row.
