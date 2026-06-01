# Object Classification in Low-Resolution THz-like Imagery — Final Research Report

**Status:** 276 / 276 cells complete (186 v2 + 90 Phase D, closed 2026-05-26) · **Generated:** 2026-06-01 · **Hardware:** RTX 5070 (Blackwell sm_120, 12 GB)

---

## Executive Summary

This report presents the final outcomes of a **276-cell** experimental campaign (186-cell v2
core + 90-cell Phase D regularization sweep) evaluating
the robustness of three deep-learning architectures —
**ResNet50** (CNN, ~25M params), **DenseNet121** (CNN, ~8M params), and
**TransNeXt-tiny** (attention, ~28M params) — under severe visual degradation that
simulates Terahertz (THz) imaging conditions: low resolution, blur, additive noise,
salt-and-pepper, and desaturation. All three models are trained at the same
**224 × 224** input resolution using a uniform protocol (full fine-tuning with
differential learning rates), so accuracy differences are attributable to architecture
and not to input shape.

**Pipeline version:** all numbers in this report are under
`PIPELINE_VERSION = 2` (see [`src/data/degradation_levels.py`](../src/data/degradation_levels.py)).
The 90 noise / salt-and-pepper / Phase B cells were re-run 2026-05-21 → 2026-05-23
after US-017 moved both stochastic pixel-level perturbations from POST-upsample
(at 224 × 224, where each `torch.randn` sample landed on a single output pixel)
to PRE-upsample (at `low_res`, where bicubic subsequently spreads each sample
across a wider footprint at 224 — sensor-realistic coarse-grain structure).
The 96 unaffected cells (Phase A clean + Phase C resolution / blur / saturation)
are byte-identical to the v1 pipeline. v1 ↔ v2 are not directly comparable on
the 90 re-run cells; v1 numbers survive only in git history (commit `3e1d089`).

**Three headline cross-architecture findings emerged from the 150-cell Phase C
single-axis isolation under pipeline v2:**

1. **Universal 3 × 3-downsample bottleneck.** At the most aggressive resolution
   axis level (L5 = 3 × 3 native pixels upsampled to 224), every architecture
   collapses to ~0.43-0.46 best-val-acc on CIFAR-10 and ~0.44-0.45 on MNIST,
   within ±3 pp across architectures. The pretrain receptive-field hierarchy
   fails uniformly when sub-class geometric structure is destroyed, regardless
   of whether the backbone is convolutional or attention-based. The resolution
   axis is untouched by US-017, so this finding carries forward verbatim from v1.

2. **TransNeXt softens but does not reverse the v2 CIFAR-10 perturbation
   collapse.** Under pipeline v2, TransNeXt-tiny retains a single-digit-to-low-
   double-digit robustness gap over both CNN backbones on every CIFAR-10
   perturbation axis at L5 (+13.2 pp on noise, +11.2 pp on blur, +7.1 pp on
   saturation, +10.0 pp on salt-and-pepper), but the underlying absolute floor
   is much lower than the v1 noise / S&P numbers suggested: tnx cifar10 noise
   drops to 0.80 at L5 (v1 reported 0.96 under the post-upsample bug). The v1
   "≥ 0.95 across L1 → L5" claim survives intact only on saturation (axis
   unchanged by US-017) and partially on salt-and-pepper (holds L1-L3, falls
   below 0.95 at L4-L5). Attention's adaptive receptive fields *soften* the
   coarse-grain perturbation collapse without preventing it — the texture-statistic
   destruction is too severe for TransNeXt-tiny to fully compensate at low_res = 3.

3. **MNIST geometric-axis robustness floor preserved under pipeline v2.** Across
   all three architectures, MNIST val_acc on noise / blur / saturation /
   salt-and-pepper stays flat at 0.984-0.993 across L1 → L5 — including the
   v2-hardened noise axis (where CIFAR-10 collapses by up to 22.9 pp) and the
   v2-hardened salt-and-pepper axis (CIFAR-10 drops up to 4.4 pp). Only the
   resolution axis bites on MNIST (and only at L4-L5). This is a 60-cell,
   two-axis empirical confirmation that the THz-like coarse-grain perturbations
   preserve digit-shape signal in MNIST regardless of severity — the thick digit
   strokes cover many bicubically-spread perturbation footprints, so the
   discriminative feature survives. Texture-dominant recognition (CIFAR-10) and
   geometry-dominant recognition (MNIST) diverge sharply under pipeline v2 in
   a way the v1 pipeline largely masked.

Combined-axis Phase B confirms that the resolution-axis collapse dominates the
overall L5 accuracy: under pipeline v2, with all five axes active at L5,
CIFAR-10 best-val-acc falls to 0.19-0.20 across all three models, while MNIST
falls to 0.27-0.28 — both bounded by the same resolution-axis floor exposed in
Phase C, with the v2-hardened noise / S&P now pulling the combined-axes
trajectory ~12.5 pp lower than v1.

---

## Campaign Statistics

- **Total cells:** 276 (6 Phase A + 30 Phase B + 150 Phase C + 90 Phase D)
- **Total accumulated GPU runtime:** ~261.4 GPU-hours (Phase D added 56.5 GPU-h)
- **Total training epochs across all cells:** 11,600
- **Architectures evaluated:** 3 (ResNet50, DenseNet121, TransNeXt-tiny)
- **Datasets:** 2 (CIFAR-10, MNIST upsampled 28 → 224)
- **Degradation severities:** 5 (L1 Mild → L5 Extreme)
- **Phase C single-axis isolation axes:** 5 (resolution, noise, blur, saturation, salt_pepper)
- **Phase D regularization treatments:** 3 (T1 architectural dropout/drop-path, T2 mixup, T3 combo)

---

## Methodology

### Architectures

| Model | Params | Input | Training Regime |
|-------|--------|-------|-----------------|
| ResNet50 | ~25M | 224 × 224 | Full fine-tune, AdamW, head LR 1e-3 / backbone LR 1e-4 |
| DenseNet121 | ~8M | 224 × 224 | Full fine-tune, AdamW, head LR 1e-3 / backbone LR 1e-4 |
| TransNeXt-tiny | ~28M | 224 × 224 | Full fine-tune, AdamW, head LR 5e-4 / backbone LR 5e-5, weight_decay 5e-2, drop_path 0.1 |

All three use ImageNet pretrained weights, a uniform 60-epoch cap with
`patience=10` early stopping (`min_delta=1e-4`, `monitor=val_acc`,
`mode=max`), gradient clipping at `max_norm=1.0`, batch size 32,
seed 42 with `pl.seed_everything(workers=True)`, and `bf16-mixed`
Blackwell precision.

### Datasets

- **CIFAR-10** — 10 object classes, 32 × 32 RGB, 10K train / 5K val, upsampled 32 → 224.
- **MNIST** — 10 digit classes, 28 × 28 grayscale broadcast to 3 channels, 10K train /
  5K val, upsampled 28 → 224 by the degradation pipeline.

### Degradation Pipeline

All cells share the same pipeline:

```
original → saturation lerp → downsample(low_res) → upsample 224 × 224 (bicubic)
        → blur → noise → salt&pepper → ImageNet normalize → model
```

### 5-Level Degradation Curve

| Level | low_res | blur kernel | blur σ | noise std | S&P | saturation |
|-------|---------|-------------|--------|-----------|-----|------------|
| L1 Mild | 18 | 13 | 2.50 | 0.04 | 0.03 | 0.95 |
| L2 Light | 12 | 25 | 5.00 | 0.08 | 0.06 | 0.65 |
| L3 Moderate | 8 | 41 | 8.00 | 0.12 | 0.10 | 0.40 |
| L4 Severe | 6 | 61 | 12.00 | 0.16 | 0.14 | 0.15 |
| L5 Extreme | 3 | 91 | 18.00 | 0.22 | 0.18 | 0.00 |

Saturation is a deterministic lerp `(1 − s) · gray + s · img`, applied **before**
noise / salt-and-pepper so injected noise stays color-correct. Validation
degradations are per-sample seeded
(`seed = idx + SEED_OFFSET_VAL`) — validation pixels are byte-identical
across dataset rebuilds, gated by `src/tests/test_degradation_determinism.py`.

### Phase Design

| Phase | Description | Count |
|-------|-------------|-------|
| **A — Clean baselines** | identity pipeline, no degradation | 6 |
| **B — Combined degradation** | all five axes active at the same level L | 30 |
| **C — Single-axis isolation** | one axis at L, the other four at identity | 150 |
| **Total** | | **186** |

Phase C single-axis isolation (US-003) was the diagnostic instrument
designed to attribute Phase B collapses to specific axes; the
headline findings above are direct outputs of this design.

---

## Phase A — Clean Baselines (6 cells)

Identity degradation pipeline; measures the upper bound for each (model, dataset)
pair before any THz-like distortion is applied.

| # | Model | Dataset | Best Val Acc | Epochs | Runtime |
|---|-------|---------|--------------|--------|---------|
| 1 | resnet50 | cifar10 | 0.9518 | 23 | 9m |
| 2 | resnet50 | mnist | 0.9914 | 21 | 7m |
| 3 | densenet121 | cifar10 | 0.9356 | 17 | 9m |
| 4 | densenet121 | mnist | 0.9930 | 29 | 14m |
| 5 | transnext⁠_⁠tiny | cifar10 | 0.9764 | 15 | 38m |
| 6 | transnext⁠_⁠tiny | mnist | 0.9924 | 35 | 90m |

**Reading:** TransNeXt-tiny leads CIFAR-10 by +2.46 pp over ResNet50 and +4.08 pp
over DenseNet121 at clean baseline; MNIST clusters tight at 0.991-0.993.

---

## Phase B — Combined Degradation (30 cells)

Every five axes are active at the same level L per cell. This is the
"realistic THz" regime — the model has to defeat all five degradations
simultaneously.

### Phase B Best-Val-Acc Curve (L1 → L5)

| Model | Dataset | L1 | L2 | L3 | L4 | L5 |
|-------|---------|----|----|----|----|----|
| resnet50 | cifar10 | 0.7646 | 0.5394 | 0.3586 | 0.2684 | 0.1906 |
| resnet50 | mnist | 0.9918 | 0.9636 | 0.7886 | 0.5644 | 0.2800 |
| densenet121 | cifar10 | 0.7848 | 0.5368 | 0.3960 | 0.2852 | 0.1958 |
| densenet121 | mnist | 0.9874 | 0.9654 | 0.8196 | 0.5808 | 0.2776 |
| transnext⁠_⁠tiny | cifar10 | 0.8758 | 0.6422 | 0.3900 | 0.2850 | 0.2010 |
| transnext⁠_⁠tiny | mnist | 0.9896 | 0.9600 | 0.7994 | 0.5724 | 0.2748 |

**Reading (pipeline v2).** Strictly monotonic on CIFAR-10 across all three
architectures (no inversions). MNIST holds ≥ 0.96 through L2 for all models,
then falls fast at L4 / L5 — driven by the resolution axis (see Phase C) and
amplified by the v2-hardened noise + S&P axes (mean ΔL3 vs v1 is roughly
−7 pp on MNIST). v2 Phase B mean Δ across the 30 cells is −12.49 pp vs v1
(operator-locked under US-020 close 2026-05-21). For comparison, Phase C
noise mean Δ vs v1 is −5.15 pp and Phase C salt-and-pepper mean Δ vs v1 is
−0.81 pp; the per-axis severity arithmetic is in §"Cross-Architecture
Headline Findings" below.

### Full Phase B Result Table

| # | Model | Dataset | Level | Best Val Acc | Epochs | Runtime |
|---|-------|---------|-------|--------------|--------|---------|
| 7 | resnet50 | cifar10 | L1 Mild | 0.7646 | 21 | 15m |
| 8 | resnet50 | mnist | L1 Mild | 0.9918 | 33 | 24m |
| 9 | densenet121 | cifar10 | L1 Mild | 0.7848 | 45 | 44m |
| 10 | densenet121 | mnist | L1 Mild | 0.9874 | 18 | 18m |
| 11 | transnext⁠_⁠tiny | cifar10 | L1 Mild | 0.8758 | 26 | 72m |
| 12 | transnext⁠_⁠tiny | mnist | L1 Mild | 0.9896 | 30 | 84m |
| 13 | resnet50 | cifar10 | L2 Light | 0.5394 | 31 | 23m |
| 14 | resnet50 | mnist | L2 Light | 0.9636 | 45 | 34m |
| 15 | densenet121 | cifar10 | L2 Light | 0.5368 | 14 | 14m |
| 16 | densenet121 | mnist | L2 Light | 0.9654 | 53 | 53m |
| 17 | transnext⁠_⁠tiny | cifar10 | L2 Light | 0.6422 | 22 | 61m |
| 18 | transnext⁠_⁠tiny | mnist | L2 Light | 0.9600 | 16 | 45m |
| 19 | resnet50 | cifar10 | L3 Moderate | 0.3586 | 13 | 10m |
| 20 | resnet50 | mnist | L3 Moderate | 0.7886 | 14 | 11m |
| 21 | densenet121 | cifar10 | L3 Moderate | 0.3960 | 15 | 15m |
| 22 | densenet121 | mnist | L3 Moderate | 0.8196 | 47 | 48m |
| 23 | transnext⁠_⁠tiny | cifar10 | L3 Moderate | 0.3900 | 13 | 36m |
| 24 | transnext⁠_⁠tiny | mnist | L3 Moderate | 0.7994 | 15 | 42m |
| 25 | resnet50 | cifar10 | L4 Severe | 0.2684 | 14 | 11m |
| 26 | resnet50 | mnist | L4 Severe | 0.5644 | 14 | 11m |
| 27 | densenet121 | cifar10 | L4 Severe | 0.2852 | 16 | 17m |
| 28 | densenet121 | mnist | L4 Severe | 0.5808 | 15 | 16m |
| 29 | transnext⁠_⁠tiny | cifar10 | L4 Severe | 0.2850 | 13 | 37m |
| 30 | transnext⁠_⁠tiny | mnist | L4 Severe | 0.5724 | 15 | 43m |
| 31 | resnet50 | cifar10 | L5 Extreme | 0.1906 | 15 | 12m |
| 32 | resnet50 | mnist | L5 Extreme | 0.2800 | 16 | 13m |
| 33 | densenet121 | cifar10 | L5 Extreme | 0.1958 | 21 | 23m |
| 34 | densenet121 | mnist | L5 Extreme | 0.2776 | 24 | 26m |
| 35 | transnext⁠_⁠tiny | cifar10 | L5 Extreme | 0.2010 | 15 | 43m |
| 36 | transnext⁠_⁠tiny | mnist | L5 Extreme | 0.2748 | 16 | 46m |

---

## Phase C — Single-Axis Isolation (150 cells)

The active axis is at level L; the other four axes are at **identity** (no
degradation). This isolates the contribution of each degradation axis from
the Phase B collapse.

At L1, the isolation cells differ from Phase B L1 cells because Phase B L1 has
**all five axes** at L1 mild while Phase C L1 has **one axis at L1, four at
identity** — see `docs/phase_c.md` for the scientific rationale and the
`degradation_levels_hash` rotation note.

### 5 × 5 Heatmaps per (model, dataset)

#### ResNet50

#### resnet50 · cifar10

| Axis | L1 | L2 | L3 | L4 | L5 |
|------|----|----|----|----|----|
| resolution | 0.8800 | 0.7898 | 0.6846 | 0.6050 | 0.4316 |
| noise | 0.8914 | 0.8470 | 0.7806 | 0.7276 | 0.6688 |
| blur | 0.9354 | 0.8988 | 0.8678 | 0.8146 | 0.7098 |
| saturation | 0.9206 | 0.9282 | 0.9256 | 0.9178 | 0.8822 |
| salt⁠_⁠pepper | 0.9042 | 0.8992 | 0.8652 | 0.8464 | 0.8330 |

#### resnet50 · mnist

| Axis | L1 | L2 | L3 | L4 | L5 |
|------|----|----|----|----|----|
| resolution | 0.9884 | 0.9850 | 0.9348 | 0.8122 | 0.4426 |
| noise | 0.9910 | 0.9916 | 0.9910 | 0.9910 | 0.9896 |
| blur | 0.9932 | 0.9914 | 0.9918 | 0.9920 | 0.9882 |
| saturation | 0.9894 | 0.9918 | 0.9936 | 0.9918 | 0.9910 |
| salt⁠_⁠pepper | 0.9908 | 0.9898 | 0.9888 | 0.9890 | 0.9862 |

#### DenseNet121

#### densenet121 · cifar10

| Axis | L1 | L2 | L3 | L4 | L5 |
|------|----|----|----|----|----|
| resolution | 0.8652 | 0.7960 | 0.6768 | 0.5776 | 0.4432 |
| noise | 0.8892 | 0.8364 | 0.7954 | 0.7506 | 0.6846 |
| blur | 0.9054 | 0.9040 | 0.8644 | 0.8204 | 0.7142 |
| saturation | 0.9184 | 0.9142 | 0.9166 | 0.9130 | 0.8834 |
| salt⁠_⁠pepper | 0.9072 | 0.8898 | 0.8796 | 0.8644 | 0.8508 |

#### densenet121 · mnist

| Axis | L1 | L2 | L3 | L4 | L5 |
|------|----|----|----|----|----|
| resolution | 0.9922 | 0.9862 | 0.9360 | 0.8132 | 0.4540 |
| noise | 0.9934 | 0.9926 | 0.9924 | 0.9934 | 0.9928 |
| blur | 0.9906 | 0.9934 | 0.9896 | 0.9928 | 0.9846 |
| saturation | 0.9932 | 0.9924 | 0.9926 | 0.9928 | 0.9914 |
| salt⁠_⁠pepper | 0.9918 | 0.9914 | 0.9906 | 0.9876 | 0.9872 |

#### TransNeXt-tiny

#### transnext_tiny · cifar10

| Axis | L1 | L2 | L3 | L4 | L5 |
|------|----|----|----|----|----|
| resolution | 0.9514 | 0.8862 | 0.7964 | 0.6840 | 0.4624 |
| noise | 0.9598 | 0.9334 | 0.8940 | 0.8598 | 0.8008 |
| blur | 0.9766 | 0.9724 | 0.9496 | 0.9088 | 0.8244 |
| saturation | 0.9792 | 0.9778 | 0.9754 | 0.9756 | 0.9534 |
| salt⁠_⁠pepper | 0.9728 | 0.9660 | 0.9548 | 0.9474 | 0.9422 |

#### transnext_tiny · mnist

| Axis | L1 | L2 | L3 | L4 | L5 |
|------|----|----|----|----|----|
| resolution | 0.9914 | 0.9808 | 0.9248 | 0.8008 | 0.4486 |
| noise | 0.9916 | 0.9928 | 0.9924 | 0.9916 | 0.9914 |
| blur | 0.9924 | 0.9896 | 0.9914 | 0.9928 | 0.9906 |
| saturation | 0.9936 | 0.9896 | 0.9918 | 0.9924 | 0.9904 |
| salt⁠_⁠pepper | 0.9900 | 0.9916 | 0.9882 | 0.9876 | 0.9894 |

### Full Phase C Result Table

| # | Model | Dataset | Level | Axis | Best Val Acc | Epochs | Runtime |
|---|-------|---------|-------|------|--------------|--------|---------|
| 37 | resnet50 | cifar10 | L1 Mild | resolution | 0.8800 | 52 | 25m |
| 38 | resnet50 | mnist | L1 Mild | resolution | 0.9884 | 13 | 6m |
| 39 | densenet121 | cifar10 | L1 Mild | resolution | 0.8652 | 35 | 26m |
| 40 | densenet121 | mnist | L1 Mild | resolution | 0.9922 | 41 | 31m |
| 41 | transnext⁠_⁠tiny | cifar10 | L1 Mild | resolution | 0.9514 | 16 | 41m |
| 42 | transnext⁠_⁠tiny | mnist | L1 Mild | resolution | 0.9914 | 29 | 74m |
| 43 | resnet50 | cifar10 | L1 Mild | noise | 0.8914 | 23 | 11m |
| 44 | resnet50 | mnist | L1 Mild | noise | 0.9910 | 31 | 16m |
| 45 | densenet121 | cifar10 | L1 Mild | noise | 0.8892 | 36 | 27m |
| 46 | densenet121 | mnist | L1 Mild | noise | 0.9934 | 27 | 21m |
| 47 | transnext⁠_⁠tiny | cifar10 | L1 Mild | noise | 0.9598 | 27 | 69m |
| 48 | transnext⁠_⁠tiny | mnist | L1 Mild | noise | 0.9916 | 23 | 59m |
| 49 | resnet50 | cifar10 | L1 Mild | blur | 0.9354 | 60 | 39m |
| 50 | resnet50 | mnist | L1 Mild | blur | 0.9932 | 31 | 20m |
| 51 | densenet121 | cifar10 | L1 Mild | blur | 0.9054 | 24 | 22m |
| 52 | densenet121 | mnist | L1 Mild | blur | 0.9906 | 20 | 19m |
| 53 | transnext⁠_⁠tiny | cifar10 | L1 Mild | blur | 0.9766 | 15 | 41m |
| 54 | transnext⁠_⁠tiny | mnist | L1 Mild | blur | 0.9924 | 27 | 74m |
| 55 | resnet50 | cifar10 | L1 Mild | saturation | 0.9206 | 14 | 7m |
| 56 | resnet50 | mnist | L1 Mild | saturation | 0.9894 | 13 | 7m |
| 57 | densenet121 | cifar10 | L1 Mild | saturation | 0.9184 | 37 | 28m |
| 58 | densenet121 | mnist | L1 Mild | saturation | 0.9932 | 28 | 22m |
| 59 | transnext⁠_⁠tiny | cifar10 | L1 Mild | saturation | 0.9792 | 17 | 44m |
| 60 | transnext⁠_⁠tiny | mnist | L1 Mild | saturation | 0.9936 | 30 | 77m |
| 61 | resnet50 | cifar10 | L1 Mild | salt⁠_⁠pepper | 0.9042 | 17 | 9m |
| 62 | resnet50 | mnist | L1 Mild | salt⁠_⁠pepper | 0.9908 | 37 | 19m |
| 63 | densenet121 | cifar10 | L1 Mild | salt⁠_⁠pepper | 0.9072 | 37 | 28m |
| 64 | densenet121 | mnist | L1 Mild | salt⁠_⁠pepper | 0.9918 | 27 | 21m |
| 65 | transnext⁠_⁠tiny | cifar10 | L1 Mild | salt⁠_⁠pepper | 0.9728 | 24 | 62m |
| 66 | transnext⁠_⁠tiny | mnist | L1 Mild | salt⁠_⁠pepper | 0.9900 | 16 | 41m |
| 67 | resnet50 | cifar10 | L2 Light | resolution | 0.7898 | 22 | 11m |
| 68 | resnet50 | mnist | L2 Light | resolution | 0.9850 | 37 | 18m |
| 69 | densenet121 | cifar10 | L2 Light | resolution | 0.7960 | 52 | 39m |
| 70 | densenet121 | mnist | L2 Light | resolution | 0.9862 | 52 | 40m |
| 71 | transnext⁠_⁠tiny | cifar10 | L2 Light | resolution | 0.8862 | 20 | 51m |
| 72 | transnext⁠_⁠tiny | mnist | L2 Light | resolution | 0.9808 | 20 | 51m |
| 73 | resnet50 | cifar10 | L2 Light | noise | 0.8470 | 20 | 10m |
| 74 | resnet50 | mnist | L2 Light | noise | 0.9916 | 26 | 13m |
| 75 | densenet121 | cifar10 | L2 Light | noise | 0.8364 | 39 | 29m |
| 76 | densenet121 | mnist | L2 Light | noise | 0.9926 | 25 | 19m |
| 77 | transnext⁠_⁠tiny | cifar10 | L2 Light | noise | 0.9334 | 19 | 49m |
| 78 | transnext⁠_⁠tiny | mnist | L2 Light | noise | 0.9928 | 32 | 82m |
| 79 | resnet50 | cifar10 | L2 Light | blur | 0.8988 | 17 | 11m |
| 80 | resnet50 | mnist | L2 Light | blur | 0.9914 | 24 | 16m |
| 81 | densenet121 | cifar10 | L2 Light | blur | 0.9040 | 44 | 41m |
| 82 | densenet121 | mnist | L2 Light | blur | 0.9934 | 24 | 23m |
| 83 | transnext⁠_⁠tiny | cifar10 | L2 Light | blur | 0.9724 | 14 | 38m |
| 84 | transnext⁠_⁠tiny | mnist | L2 Light | blur | 0.9896 | 17 | 47m |
| 85 | resnet50 | cifar10 | L2 Light | saturation | 0.9282 | 14 | 7m |
| 86 | resnet50 | mnist | L2 Light | saturation | 0.9918 | 21 | 11m |
| 87 | densenet121 | cifar10 | L2 Light | saturation | 0.9142 | 41 | 32m |
| 88 | densenet121 | mnist | L2 Light | saturation | 0.9924 | 22 | 17m |
| 89 | transnext⁠_⁠tiny | cifar10 | L2 Light | saturation | 0.9778 | 13 | 33m |
| 90 | transnext⁠_⁠tiny | mnist | L2 Light | saturation | 0.9896 | 14 | 36m |
| 91 | resnet50 | cifar10 | L2 Light | salt⁠_⁠pepper | 0.8992 | 33 | 17m |
| 92 | resnet50 | mnist | L2 Light | salt⁠_⁠pepper | 0.9898 | 27 | 14m |
| 93 | densenet121 | cifar10 | L2 Light | salt⁠_⁠pepper | 0.8898 | 45 | 34m |
| 94 | densenet121 | mnist | L2 Light | salt⁠_⁠pepper | 0.9914 | 41 | 32m |
| 95 | transnext⁠_⁠tiny | cifar10 | L2 Light | salt⁠_⁠pepper | 0.9660 | 15 | 38m |
| 96 | transnext⁠_⁠tiny | mnist | L2 Light | salt⁠_⁠pepper | 0.9916 | 46 | 2.0h |
| 97 | resnet50 | cifar10 | L3 Moderate | resolution | 0.6846 | 23 | 11m |
| 98 | resnet50 | mnist | L3 Moderate | resolution | 0.9348 | 49 | 24m |
| 99 | densenet121 | cifar10 | L3 Moderate | resolution | 0.6768 | 19 | 14m |
| 100 | densenet121 | mnist | L3 Moderate | resolution | 0.9360 | 53 | 41m |
| 101 | transnext⁠_⁠tiny | cifar10 | L3 Moderate | resolution | 0.7964 | 37 | 1.6h |
| 102 | transnext⁠_⁠tiny | mnist | L3 Moderate | resolution | 0.9248 | 16 | 41m |
| 103 | resnet50 | cifar10 | L3 Moderate | noise | 0.7806 | 17 | 8m |
| 104 | resnet50 | mnist | L3 Moderate | noise | 0.9910 | 25 | 13m |
| 105 | densenet121 | cifar10 | L3 Moderate | noise | 0.7954 | 34 | 26m |
| 106 | densenet121 | mnist | L3 Moderate | noise | 0.9924 | 42 | 32m |
| 107 | transnext⁠_⁠tiny | cifar10 | L3 Moderate | noise | 0.8940 | 16 | 41m |
| 108 | transnext⁠_⁠tiny | mnist | L3 Moderate | noise | 0.9924 | 32 | 82m |
| 109 | resnet50 | cifar10 | L3 Moderate | blur | 0.8678 | 31 | 21m |
| 110 | resnet50 | mnist | L3 Moderate | blur | 0.9918 | 31 | 22m |
| 111 | densenet121 | cifar10 | L3 Moderate | blur | 0.8644 | 36 | 35m |
| 112 | densenet121 | mnist | L3 Moderate | blur | 0.9896 | 23 | 22m |
| 113 | transnext⁠_⁠tiny | cifar10 | L3 Moderate | blur | 0.9496 | 15 | 41m |
| 114 | transnext⁠_⁠tiny | mnist | L3 Moderate | blur | 0.9914 | 22 | 61m |
| 115 | resnet50 | cifar10 | L3 Moderate | saturation | 0.9256 | 17 | 8m |
| 116 | resnet50 | mnist | L3 Moderate | saturation | 0.9936 | 19 | 10m |
| 117 | densenet121 | cifar10 | L3 Moderate | saturation | 0.9166 | 45 | 34m |
| 118 | densenet121 | mnist | L3 Moderate | saturation | 0.9926 | 37 | 29m |
| 119 | transnext⁠_⁠tiny | cifar10 | L3 Moderate | saturation | 0.9754 | 14 | 36m |
| 120 | transnext⁠_⁠tiny | mnist | L3 Moderate | saturation | 0.9918 | 22 | 57m |
| 121 | resnet50 | cifar10 | L3 Moderate | salt⁠_⁠pepper | 0.8652 | 18 | 9m |
| 122 | resnet50 | mnist | L3 Moderate | salt⁠_⁠pepper | 0.9888 | 31 | 16m |
| 123 | densenet121 | cifar10 | L3 Moderate | salt⁠_⁠pepper | 0.8796 | 49 | 37m |
| 124 | densenet121 | mnist | L3 Moderate | salt⁠_⁠pepper | 0.9906 | 46 | 35m |
| 125 | transnext⁠_⁠tiny | cifar10 | L3 Moderate | salt⁠_⁠pepper | 0.9548 | 13 | 33m |
| 126 | transnext⁠_⁠tiny | mnist | L3 Moderate | salt⁠_⁠pepper | 0.9882 | 18 | 46m |
| 127 | resnet50 | cifar10 | L4 Severe | resolution | 0.6050 | 60 | 29m |
| 128 | resnet50 | mnist | L4 Severe | resolution | 0.8122 | 43 | 21m |
| 129 | densenet121 | cifar10 | L4 Severe | resolution | 0.5776 | 17 | 13m |
| 130 | densenet121 | mnist | L4 Severe | resolution | 0.8132 | 30 | 23m |
| 131 | transnext⁠_⁠tiny | cifar10 | L4 Severe | resolution | 0.6840 | 33 | 84m |
| 132 | transnext⁠_⁠tiny | mnist | L4 Severe | resolution | 0.8008 | 16 | 41m |
| 133 | resnet50 | cifar10 | L4 Severe | noise | 0.7276 | 15 | 7m |
| 134 | resnet50 | mnist | L4 Severe | noise | 0.9910 | 27 | 14m |
| 135 | densenet121 | cifar10 | L4 Severe | noise | 0.7506 | 36 | 27m |
| 136 | densenet121 | mnist | L4 Severe | noise | 0.9934 | 42 | 32m |
| 137 | transnext⁠_⁠tiny | cifar10 | L4 Severe | noise | 0.8598 | 15 | 38m |
| 138 | transnext⁠_⁠tiny | mnist | L4 Severe | noise | 0.9916 | 40 | 1.7h |
| 139 | resnet50 | cifar10 | L4 Severe | blur | 0.8146 | 44 | 31m |
| 140 | resnet50 | mnist | L4 Severe | blur | 0.9920 | 34 | 25m |
| 141 | densenet121 | cifar10 | L4 Severe | blur | 0.8204 | 41 | 41m |
| 142 | densenet121 | mnist | L4 Severe | blur | 0.9928 | 44 | 44m |
| 143 | transnext⁠_⁠tiny | cifar10 | L4 Severe | blur | 0.9088 | 29 | 81m |
| 144 | transnext⁠_⁠tiny | mnist | L4 Severe | blur | 0.9928 | 36 | 1.7h |
| 145 | resnet50 | cifar10 | L4 Severe | saturation | 0.9178 | 18 | 9m |
| 146 | resnet50 | mnist | L4 Severe | saturation | 0.9918 | 13 | 7m |
| 147 | densenet121 | cifar10 | L4 Severe | saturation | 0.9130 | 35 | 27m |
| 148 | densenet121 | mnist | L4 Severe | saturation | 0.9928 | 45 | 35m |
| 149 | transnext⁠_⁠tiny | cifar10 | L4 Severe | saturation | 0.9756 | 14 | 36m |
| 150 | transnext⁠_⁠tiny | mnist | L4 Severe | saturation | 0.9924 | 19 | 49m |
| 151 | resnet50 | cifar10 | L4 Severe | salt⁠_⁠pepper | 0.8464 | 21 | 11m |
| 152 | resnet50 | mnist | L4 Severe | salt⁠_⁠pepper | 0.9890 | 31 | 16m |
| 153 | densenet121 | cifar10 | L4 Severe | salt⁠_⁠pepper | 0.8644 | 44 | 34m |
| 154 | densenet121 | mnist | L4 Severe | salt⁠_⁠pepper | 0.9876 | 20 | 15m |
| 155 | transnext⁠_⁠tiny | cifar10 | L4 Severe | salt⁠_⁠pepper | 0.9474 | 28 | 72m |
| 156 | transnext⁠_⁠tiny | mnist | L4 Severe | salt⁠_⁠pepper | 0.9876 | 19 | 49m |
| 157 | resnet50 | cifar10 | L5 Extreme | resolution | 0.4316 | 15 | 7m |
| 158 | resnet50 | mnist | L5 Extreme | resolution | 0.4426 | 20 | 10m |
| 159 | densenet121 | cifar10 | L5 Extreme | resolution | 0.4432 | 19 | 14m |
| 160 | densenet121 | mnist | L5 Extreme | resolution | 0.4540 | 21 | 16m |
| 161 | transnext⁠_⁠tiny | cifar10 | L5 Extreme | resolution | 0.4624 | 13 | 33m |
| 162 | transnext⁠_⁠tiny | mnist | L5 Extreme | resolution | 0.4486 | 17 | 44m |
| 163 | resnet50 | cifar10 | L5 Extreme | noise | 0.6688 | 30 | 15m |
| 164 | resnet50 | mnist | L5 Extreme | noise | 0.9896 | 19 | 10m |
| 165 | densenet121 | cifar10 | L5 Extreme | noise | 0.6846 | 36 | 27m |
| 166 | densenet121 | mnist | L5 Extreme | noise | 0.9928 | 49 | 38m |
| 167 | transnext⁠_⁠tiny | cifar10 | L5 Extreme | noise | 0.8008 | 60 | 2.5h |
| 168 | transnext⁠_⁠tiny | mnist | L5 Extreme | noise | 0.9914 | 36 | 1.5h |
| 169 | resnet50 | cifar10 | L5 Extreme | blur | 0.7098 | 39 | 30m |
| 170 | resnet50 | mnist | L5 Extreme | blur | 0.9882 | 31 | 24m |
| 171 | densenet121 | cifar10 | L5 Extreme | blur | 0.7142 | 21 | 22m |
| 172 | densenet121 | mnist | L5 Extreme | blur | 0.9846 | 19 | 20m |
| 173 | transnext⁠_⁠tiny | cifar10 | L5 Extreme | blur | 0.8244 | 22 | 62m |
| 174 | transnext⁠_⁠tiny | mnist | L5 Extreme | blur | 0.9906 | 60 | 2.8h |
| 175 | resnet50 | cifar10 | L5 Extreme | saturation | 0.8822 | 16 | 8m |
| 176 | resnet50 | mnist | L5 Extreme | saturation | 0.9910 | 13 | 7m |
| 177 | densenet121 | cifar10 | L5 Extreme | saturation | 0.8834 | 37 | 28m |
| 178 | densenet121 | mnist | L5 Extreme | saturation | 0.9914 | 20 | 16m |
| 179 | transnext⁠_⁠tiny | cifar10 | L5 Extreme | saturation | 0.9534 | 44 | 1.9h |
| 180 | transnext⁠_⁠tiny | mnist | L5 Extreme | saturation | 0.9904 | 18 | 46m |
| 181 | resnet50 | cifar10 | L5 Extreme | salt⁠_⁠pepper | 0.8330 | 23 | 12m |
| 182 | resnet50 | mnist | L5 Extreme | salt⁠_⁠pepper | 0.9862 | 31 | 16m |
| 183 | densenet121 | cifar10 | L5 Extreme | salt⁠_⁠pepper | 0.8508 | 44 | 33m |
| 184 | densenet121 | mnist | L5 Extreme | salt⁠_⁠pepper | 0.9872 | 22 | 17m |
| 185 | transnext⁠_⁠tiny | cifar10 | L5 Extreme | salt⁠_⁠pepper | 0.9422 | 59 | 2.5h |
| 186 | transnext⁠_⁠tiny | mnist | L5 Extreme | salt⁠_⁠pepper | 0.9894 | 40 | 1.7h |

---

## Cross-Architecture Headline Findings

### Finding 1 — Universal 3 × 3-downsample bottleneck at L5 resolution

At the most aggressive resolution axis level (L5 = 3 × 3 native pixels
upsampled to 224 × 224 bicubic), every architecture lands within ±3 pp
on CIFAR-10 and ±1 pp on MNIST:

| Axis @ L5 — resolution | resnet50 | densenet121 | transnext⁠_⁠tiny |
|------------------------|----------|-------------|----------------|
| CIFAR-10 | 0.4316 | 0.4432 | 0.4624 |
| MNIST | 0.4426 | 0.4540 | 0.4486 |

The pretrain receptive-field hierarchy fails uniformly when sub-class geometric
structure is destroyed, regardless of whether the backbone is convolutional or
attention-based. The L5 resolution axis is the rate-limiting degradation for
both Phase B (combined) collapses and the campaign's overall worst-case cells.

### Finding 2 — TransNeXt softens but does not reverse the v2 CIFAR-10 perturbation collapse

At L5 on CIFAR-10 under pipeline v2, TransNeXt-tiny retains a robustness gap over
both CNN backbones on every perturbation axis, but the absolute floor is much
lower than the v1 noise / S&P numbers suggested. The v1 "≥ 0.95 across L1 → L5"
claim now survives intact only on saturation (axis unchanged by US-017) and
partially on salt-and-pepper (holds L1-L3, falls below 0.95 at L4-L5):

| Axis @ L5 — CIFAR-10 | resnet50 | densenet121 | transnext⁠_⁠tiny | TransNeXt gap vs avg CNN |
|----------------------|----------|-------------|----------------|--------------------------|
| noise (v2) | 0.6688 | 0.6846 | 0.8008 | ~+13.2 pp |
| saturation | 0.8822 | 0.8834 | 0.9534 | ~+7.1 pp |
| salt_pepper (v2) | 0.8330 | 0.8508 | 0.9422 | ~+10.0 pp |
| blur | 0.7098 | 0.7142 | 0.8244 | ~+11.2 pp |
| resolution | 0.4316 | 0.4432 | 0.4624 | +3 pp |

Read-out: attention's adaptive receptive fields *soften* the coarse-grain
perturbation collapse without preventing it. The texture-statistic destruction
under pipeline v2 (where noise samples every low_res² pixel and bicubic spreads
each draw into a ~74-px footprint at low_res = 3) is too severe for
TransNeXt-tiny to fully compensate. Under the v1 post-upsample bug the same
TransNeXt-tiny architecture appeared to "hold ≥ 0.95" on noise — that was an
artefact of the bug, not architectural robustness, and is corrected here.

### Finding 3 — MNIST geometric-axis robustness floor preserved under pipeline v2

For all three architectures, MNIST val_acc on noise / blur / saturation /
salt-and-pepper stays flat at 0.984-0.993 across L1 → L5 under pipeline v2 —
including the v2-hardened noise axis (where CIFAR-10 collapses by up to 22.9 pp)
and the v2-hardened salt-and-pepper axis (CIFAR-10 drops up to 4.4 pp). Only the
resolution axis bites on MNIST (and only at L4-L5). This is now a 60-cell,
two-axis empirical confirmation that the THz-like coarse-grain perturbations
preserve digit-shape signal in MNIST regardless of severity — the thick digit
strokes cover many bicubically-spread perturbation footprints, so the
discriminative feature survives. Texture-dominant recognition (CIFAR-10) and
geometry-dominant recognition (MNIST) diverge sharply under pipeline v2 in a way
the v1 pipeline largely masked.

The v2 severity ordering on CIFAR-10 at L5 (worst → mildest) becomes
`resolution > noise > blur > salt_pepper > saturation` — noise leapfrogs blur as
the second-worst axis under v2, masked in v1 because post-upsample noise was
effectively averaged out by the bicubic kernel.

### Complete cross-model L5 axis table

| Axis @ L5 | resnet50 cifar10 | resnet50 mnist | densenet121 cifar10 | densenet121 mnist | transnext_tiny cifar10 | transnext_tiny mnist |
|-----------|------------------|----------------|---------------------|-------------------|------------------------|----------------------|
| resolution | 0.4316 | 0.4426 | 0.4432 | 0.4540 | 0.4624 | 0.4486 |
| noise | 0.6688 | 0.9896 | 0.6846 | 0.9928 | 0.8008 | 0.9914 |
| blur | 0.7098 | 0.9882 | 0.7142 | 0.9846 | 0.8244 | 0.9906 |
| saturation | 0.8822 | 0.9910 | 0.8834 | 0.9914 | 0.9534 | 0.9904 |
| salt⁠_⁠pepper | 0.8330 | 0.9862 | 0.8508 | 0.9872 | 0.9422 | 0.9894 |

---

## Research Conclusions

1. **Architecture matters more for noise than for geometry.** When the
   degradation is pixel-level (noise / saturation / salt-and-pepper), TransNeXt's
   attention mechanism provides a substantial robustness margin over CNNs. When
   the degradation is geometric (low resolution), all three architectures
   converge to the same failure mode at L5.

2. **The single-axis isolation design (Phase C) is the right diagnostic.**
   Without Phase C, the campaign's two headline cross-architecture findings
   would be invisible — Phase B's combined-axes L5 cells average over the
   resolution collapse and the perturbation-axis robustness gap, giving a
   misleading uniform "all CNN backbones fail" picture. The 150-cell investment
   in single-axis isolation surfaced both findings.

3. **CIFAR-10 vs MNIST gap at L5 quantifies natural-image vs digit-shape
   robustness.** Phase B L5 CIFAR-10 collapses to 0.19-0.20 across all three
   models under pipeline v2, while MNIST falls to 0.27-0.28 — a +7 to +8 pp
   gap that persists across architectures. The gap is driven entirely by the
   resolution axis; for all other axes, MNIST is essentially uniform at all
   severities (Finding 3). Under v1 the gap was wider (+12-15 pp) because v1
   noise / S&P did not depress MNIST val_acc, but v2 confirms that even under
   genuinely sensor-realistic coarse-grain perturbations MNIST geometry survives
   while CIFAR-10 texture statistics do not.

4. **§6.4 pathology-guard retry path is ineffective on Optuna-tuned models.**
   Across the entire campaign, the pathology guard flagged 13 cells (1 in
   Phase B resnet50, 2 in Phase B densenet121, 6 in Phase C resnet50, 3 in
   Phase C densenet121, 2 in Phase C transnext_tiny). All 13 were
   monotonic with their surrounding L-curve. The operator declined the
   §6.4 retry on all 13 (Iter 12 / 14 / 21 / 24 / Iter 25 closure) because
   the §6.4 deltas (`wd × 2` + `dropout = 0.1` + `backbone_lr ÷ 20`)
   consistently underfit tightly-tuned Optuna winners. Recommendation
   for future campaigns: replace §6.4 with an Optuna re-tune on the
   flagged cell, or tighten the pathology-guard threshold so it only
   fires on genuinely non-monotonic readings.

5. **Cross-architecture L5 convergence justifies the 224 × 224 uniform
   protocol decision.** When the V3 native-resolution refactor was rolled
   back (2026-05-13 protocol ratification), the fair-comparison invariant
   became "same tensor shape (224 × 224 after upsample)". The cross-model
   L5 resolution finding (Finding 1) is meaningful precisely because all
   three architectures see the same pixel budget — the convergence is
   evidence about the architectures, not about the input shape.

6. **Phase D regularization partially softens but does not reverse the v2
   collapse — confirming the information-bottleneck interpretation.**
   Across the 90-cell Phase D sweep (T1 architectural dropout/drop-path,
   T2 mixup α=0.2, T3 the T1 ∪ T2 ∪ cutmix α=1.0 combo, layered atop the
   frozen v2 L3-Optuna winners with no re-tune), mean Δ vs the Phase B v2
   baseline is +0.17 pp (T1), +0.31 pp (T2), and +0.77 pp (T3) — monotonic
   in regularization scope. TransNeXt × CIFAR-10 reaches +2.20 pp mean
   recovery under T3 (the strongest segment, consistent with Finding 2's
   attention-softens hypothesis), while DenseNet121 × CIFAR-10 sits at just
   +0.22 pp mean. The CIFAR-10 L5 cells under T3 average only +0.27 pp —
   the 3 × 3-downsample bottleneck (Finding 1) is unrescuable. Operator-
   locked answer to the §1 research question: **Regularization recovers a
   small fraction of the v2 collapse (mean Δ = +0.77 pp under T3 vs the
   −12.49 pp v2 drop). The collapse is largely a fundamental information
   bottleneck, not an overfit signal.**

---

## Operational Notes

- **Reproducibility lock:** `pl.seed_everything(42, workers=True)` + per-sample
  validation seeds (`seed = idx + SEED_OFFSET_VAL`). Determinism gate
  (`src/tests/test_degradation_determinism.py`) verifies byte-identical
  validation pixels across dataset rebuilds.

- **Optuna pre-tuning:** Six (model, dataset) pairs were tuned at L3 Moderate
  before the 186-cell campaign launched; the winning hyperparameters were
  frozen for every Phase A/B/C cell of that pair. Winners under
  `artifacts/best_hparams/{model}_{dataset}.json`.

- **Engine:** PyTorch Lightning in `src/lightning/` (`THzClassifier`,
  `THzDataModule`). `bf16-mixed` on Blackwell (sm_120) RTX 5070;
  `32-true` on CPU.

- **Weight privacy:** Trained checkpoints (`*.ckpt`), the Optuna SQLite
  store, and dashboard thumbnail caches are local-only, gitignored, and
  claudeignored. Analysis tooling reads `metrics.json` / `metrics.csv` /
  the tracker artifacts only — never the binary weights.

---

## Appendix A — Full 186-cell Result Table

See `Final_Exp.md` (auto-regenerated from `runs/final/` by
`scripts/update_final_exp.py`) for the canonical 186-cell tracker with PSNR /
SSIM / Started / Duration columns. This report's tables summarize the same
data with focus on val_acc.

## Appendix B — Story-Driven Methodology Timeline

The campaign executed under a 9-story execution roadmap (US-006 through US-014),
each closing a (phase, model) sub-campaign with operator-gated HALTs between
stories. After US-014 closure, US-015 (end-of-campaign verification + three
phase-boundary SYNCHRONIZER pushes + SECURITY leak audit) completes the campaign.

Closure decisions were recorded in `progress.txt` as `Iteration N` blocks
(43 iterations across the v1 186-cell campaign and the v2 90-cell noise-fix
re-run), each with
the dispatch command, background task id, first-pass readings, pathology-guard
outcomes, and operator decisions. The §6.4 retry decline streak (5-for-5
across all Phase B/C CNN dispatches + the one TransNeXt Phase C dispatch
with sentinels) is documented inline.

## Appendix C — Repository Pointers

- `Final_Exp.md` — 186-cell tracker (auto-regenerated)
- `artifacts/Final_Exp.json` — JSON-shaped tracker (dashboard data source)
- `artifacts/Final_Exp.html` — interactive dashboard with sortable rows + filters
- `artifacts/reports/phase_a_{model}_summary.md` — Phase A REPORTER summaries
- `docs/phase_a.md` / `docs/phase_c.md` — LIBRARIAN per-phase docs
- `progress.txt` — full iteration log (Iterations 1-43 — covers v1 + v2)
- `PRD.md` — v2 roadmap (US-017 through US-025; v1 US-001..US-016 closed)
- `CLAUDE.md` — protocol invariants + multi-agent guidance
- `scripts/build_final_exp_report.py` — this report generator
- `scripts/render_final_exp_pdf.py` — Markdown → PDF renderer

---

*Report generated 2026-06-01 from `artifacts/Final_Exp.json` (state of disk under
`PIPELINE_VERSION = 2`, after the v2 noise-fix 90-cell re-run closed
2026-05-23 / Iteration 40). Regenerate with* `python scripts/build_final_exp_report.py`.
