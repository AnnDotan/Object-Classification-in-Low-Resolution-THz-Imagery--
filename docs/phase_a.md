# Phase A — Clean Baselines

**Status:** 4 of 6 cells complete (ResNet50 ×2 from legacy 4050 box; TransNeXt-tiny ×2 first-on-5070); DenseNet121 ×2 legacy preserved. Each cell trains an ImageNet-pretrained model on the **clean** dataset (no degradation) at 224×224 under the convergence-first protocol. Phase A establishes the per-model upper bound that Phase B / Phase C drops are measured against.

Authoritative trackers: [`Final_Exp.md`](../Final_Exp.md), [`artifacts/Final_Exp.json`](../artifacts/Final_Exp.json). Per-model summaries: [`artifacts/reports/phase_a_<model>_summary.md`](../artifacts/reports/).

---

## ResNet50

**Story:** US-006. Summary: [`artifacts/reports/phase_a_resnet50_summary.md`](../artifacts/reports/phase_a_resnet50_summary.md). Pathology guard: `healthy` on both cells.

| Cell | best_val_acc | best_epoch | epochs_run | last_train_acc | runtime |
|---|---:|---:|---:|---:|---:|
| `final_clean_resnet50_cifar10` | **0.9518** | 13 | 23 | 1.0000 | 8.7 m |
| `final_clean_resnet50_mnist`   | **0.9914** | 11 | 21 | 1.0000 | 7.4 m |

Both runs were preserved across the 2026-05-12 state reset (legacy 4050 box; bf16/Lightning protocol). They predate `degradation_levels_hash` — Phase A clean cells do not exercise the degradation table, so the field's absence is not a defect. PSNR / SSIM are intentionally absent: `_measure_image_quality_for_cell` only emits `image_quality.json` for Phase B/C cells (clean-vs-clean is a trivial identity comparison).

**Hyperparameters (`PHASE_A_FROZEN_HPARAMS`):** AdamW + cosine, head LR 1e-3, backbone LR 1e-4, weight_decay 1e-4, label_smoothing 0.1, gradient clip 1.0, warmup 3 epochs, batch 32, bf16-mixed.

**Outstanding for US-006 closure.** VALIDATOR seed=43 reproducibility re-run on `final_clean_resnet50_cifar10` (±0.5 pp tolerance); DESIGNER per-US trend section in `Final_Exp.html`; operator-resolution on the `image_quality.json` criterion conflict (PRD line 413); then HALT for operator approval before US-007.

---

## DenseNet121

**Story:** US-007 (pending closure). Legacy 4050 baselines preserved; pathology guard returns `healthy` on both cells. Detailed write-up follows once US-007 lands.

| Cell | best_val_acc | epochs_run |
|---|---:|---:|
| `final_clean_densenet121_cifar10` | 0.9356 | 17 |
| `final_clean_densenet121_mnist`   | 0.9930 | — |

---

## TransNeXt-tiny

**Story:** US-008 (PARTIAL — REPORTER + pathology + LIBRARIAN done; DESIGNER + VALIDATOR seed=43 pending). Summary: [`artifacts/reports/phase_a_transnext_tiny_summary.md`](../artifacts/reports/phase_a_transnext_tiny_summary.md). First Phase A pair executed end-to-end on the RTX 5070 via the US-005 driver.

| Cell | best_val_acc | best_epoch | epochs_run | last_train_acc | runtime |
|---|---:|---:|---:|---:|---:|
| `final_clean_transnext_tiny_cifar10` | **0.9764** | 5 | 15 | 1.0000 | 38.3 m |
| `final_clean_transnext_tiny_mnist`   | **0.9924** | 25 | 35 | 1.0000 | 89.6 m |

Stage 1.5 winner hparams loaded automatically (head LR 2.9e-3, backbone LR 3.0e-5, weight_decay 0.346, label_smoothing 0.009, warmup ≈ 1).

---

## Cross-model snapshot

| Model | CIFAR-10 best_val_acc | MNIST best_val_acc |
|---|---:|---:|
| `resnet50`        | 0.9518 | 0.9914 |
| `densenet121`     | 0.9356 | 0.9930 |
| `transnext_tiny`  | **0.9764** | 0.9924 |

TransNeXt-tiny tops CIFAR-10 by +2.46 pp over resnet50 and +4.08 pp over densenet121. MNIST clusters within 0.16 pp across all three — the dataset saturates and does not differentiate models. These clean baselines define the per-model upper bound; the Phase B / Phase C drops are measured against the corresponding row here.
