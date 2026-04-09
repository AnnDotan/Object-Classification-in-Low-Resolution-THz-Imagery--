# TODO — Project Execution Plan

## Deadlines
- Poster & Abstract: **31/05/2026**
- Final Presentation: **21/06/2026**
- Final Submission: **26/07/2026**

> Full experiment details: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)

---

## STAGE 1 — TransNeXt Validation ✅ COMPLETE
- TransNeXt LP: 68.75% vs ResNet50: 54.5% (+14.25%)
- Frozen backbone + head-only training works best

## STAGE 2 — Controlled Experiments ✅ COMPLETE
- [x] Official runs: ResNet50, DenseNet121, TransNeXt
- [x] Combined degradation: DenseNet 80.7%, ResNet 78.8%, TransNeXt 64.2%
- [x] Single-degradation isolation: 7/12 done
- [x] Dashboards with filtering and sample images

## STAGE 3 — Systematic Experiments (36 total) 🟡 IN PROGRESS

### Phase A: CIFAR-10 Systematic (9/9) — ✅ COMPLETE
| Level | ResNet50 | DenseNet121 | TransNeXt Micro |
|-------|----------|-------------|-----------------|
| L1 Mild | 81.2% | **81.9%** | 68.2% |
| L2 Moderate | 78.7% | **80.4%** | 63.6% |
| L3 Severe | 61.5% | **63.4%** | 44.9% |

### Phase B: MNIST Systematic (6/9) — 🟡 IN PROGRESS
| Level | ResNet50 | DenseNet121 | TransNeXt Micro |
|-------|----------|-------------|-----------------|
| L1 Mild | **99.1%** | 98.7% | 92.5% |
| L2 Moderate | **99.1%** | 99.0% | 92.8% |
| L3 Severe | Pending | Pending | Pending |

### Phase C: Single-Degradation Isolation (0/12) — 🔲 TODO
- [ ] Downsampling only: ResNet50, DenseNet121, TransNeXt Micro
- [ ] Blur only: ResNet50, DenseNet121, TransNeXt Micro
- [ ] Noise only: ResNet50, DenseNet121, TransNeXt Micro
- [ ] Salt & Pepper only: ResNet50, DenseNet121, TransNeXt Micro

### Phase D: Clean Baselines (0/6) — 🔲 TODO
- [ ] CIFAR-10 clean: ResNet50, DenseNet121, TransNeXt Micro
- [ ] MNIST clean: ResNet50, DenseNet121, TransNeXt Micro

### Infrastructure ✅ COMPLETE
- [x] run_all_phases.py — runs all 36 experiments sequentially
- [x] run_systematic.py — systematic launcher (--level, --mode, --dataset)
- [x] THzLikeMNIST dataset class
- [x] Auto-dashboard refresh after each experiment
- [x] Experiment plan dashboard (dark theme, original+degraded images, conclusions)

## STAGE 4 — Analysis 🔲 TODO
- [ ] Generate tables: model x level matrices (CIFAR-10 and MNIST)
- [ ] Generate figures: bar charts, learning curves, heatmaps
- [ ] Cross-dataset comparison (CIFAR-10 vs MNIST)
- [ ] Accuracy drop analysis (clean baseline vs degraded)
- [ ] Single-degradation contribution analysis
- [ ] Write conclusions with supporting evidence

## STAGE 5 — Poster & Abstract (31/05) 🔲 TODO
- [ ] Select best figures
- [ ] Write abstract
- [ ] Design poster layout

## STAGE 6 — Final Report & Presentation 🔲 TODO
- [ ] Write full report
- [ ] Prepare presentation slides
- [ ] **FINAL PRESENTATION (21/06)**

## STAGE 7 — Final Submission (26/07) 🔲 TODO
- [ ] Clean repository
- [ ] Verify reproducibility
- [ ] Submit

---

## Execution Timeline

| Week | Dates | Tasks |
|------|-------|-------|
| 1 | Apr 8-14 | Phase A (CIFAR-10 systematic) ✅, MNIST pipeline ✅ |
| 2 | Apr 15-21 | Phase B (MNIST systematic) 🟡, Phase D (baselines) |
| 3 | Apr 22-28 | Phase C (isolation experiments). Generate dashboards. |
| 4 | Apr 29-May 5 | Analysis: tables, figures, key findings. |
| 5 | May 6-12 | Verify reproducibility. Draft analysis. |
| 6-7 | May 13-26 | Poster design + abstract. |
| **8** | **May 27-31** | **POSTER & ABSTRACT DEADLINE** |
| 9-11 | Jun 1-21 | Report + **PRESENTATION (21/06)** |
| 12-16 | Jun 22-Jul 26 | Polish + **SUBMISSION (26/07)** |
