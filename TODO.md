# TODO — Project Execution Plan

## Deadlines
- Poster & Abstract: **31/05/2026**
- Final Presentation: **21/06/2026**
- Final Submission: **26/07/2026**

> Full experiment details: [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)

---

## STAGE 1 — TransNeXt Validation ✅ COMPLETE
- TransNeXt LP: 68.75% vs ResNet50: 54.5% (+14.25%)
- Frozen backbone + head-only training works best

## STAGE 2 — Controlled Experiments ✅ MOSTLY COMPLETE
- [x] Official runs: ResNet50, DenseNet121, TransNeXt
- [x] Combined degradation: DenseNet 80.7%, ResNet 78.8%, TransNeXt 64.2%
- [x] Single-degradation isolation: 7/9 done (TransNeXt pending)
- [x] Dashboards with filtering and sample images
- [ ] Run systematic experiments (3 levels × 3 models) via `run_systematic.py`

## STAGE 3 — Systematic Experiments (36 total) � IN PROGRESS

### Phase A: CIFAR-10 Systematic (9 experiments) — 🟡 Running
3 degradation levels × 3 models, identical protocol:
- [ ] Level 1 (Mild): ResNet50, DenseNet121, TransNeXt Micro
- [ ] Level 2 (Moderate): ResNet50, DenseNet121, TransNeXt Micro
- [ ] Level 3 (Severe): ResNet50, DenseNet121, TransNeXt Micro
```bash
python run_systematic.py --level all --mode full
```

### Setup: Experiment Plan Dashboard ✅ COMPLETE
- [x] Created `src/tools/generate_experiment_plan_dashboard.py`
- [x] Dark theme with white text, tracks all 36 planned experiments
- [x] Scans only `runs/systematic/` — matches by tag
- [x] Comparative bar charts per degradation level, cross-level chart
- [x] Clickable rows for learning curves, filters by phase/model/dataset/status
```bash
python src/tools/generate_experiment_plan_dashboard.py
```

### Phase B: MNIST Systematic (9 experiments) — 🔴 Critical
Same protocol as Phase A on MNIST dataset:
- [ ] Implement MNIST pipeline (`THzLikeMNIST` class in `src/data/datasets.py`)
- [ ] Add `--dataset` argument to `run_systematic.py` and `run_experiment()`
- [ ] Level 1 (Mild): ResNet50, DenseNet121, TransNeXt Micro
- [ ] Level 2 (Moderate): ResNet50, DenseNet121, TransNeXt Micro
- [ ] Level 3 (Severe): ResNet50, DenseNet121, TransNeXt Micro
```bash
python run_systematic.py --level all --mode full --dataset mnist
```

### Phase C: Single-Degradation Isolation (12 experiments) — 🟡 Important
4 degradation types × 3 models (CIFAR-10, Level 2 params):
- [ ] Downsampling only: ResNet50, DenseNet121, TransNeXt Micro
- [ ] Blur only: ResNet50, DenseNet121, TransNeXt Micro
- [ ] Noise only: ResNet50, DenseNet121, TransNeXt Micro
- [ ] Salt & Pepper only: ResNet50, DenseNet121, TransNeXt Micro
> Note: 7/12 already done in `runs/official/`. Check before running duplicates.

### Phase D: Clean Baselines (6 experiments) — 🟡 Important
No degradation — establishes upper bounds for accuracy drop analysis:
- [ ] CIFAR-10 clean: ResNet50, DenseNet121, TransNeXt Micro
- [ ] MNIST clean: ResNet50, DenseNet121, TransNeXt Micro

## STAGE 4 — Analysis
- [ ] Generate tables: model × level matrices for CIFAR-10 and MNIST
- [ ] Generate figures: bar charts, learning curves, heatmaps
- [ ] Cross-dataset comparison (CIFAR-10 vs MNIST)
- [ ] Accuracy drop analysis (clean baseline vs degraded)
- [ ] Single-degradation contribution analysis
- [ ] Explain WHY results occur (feature reuse vs attention)
- [ ] Write conclusions with supporting evidence

## STAGE 5 — Poster & Abstract (31/05)
- [ ] Select best figures (pipeline diagram, degradation examples, comparison graphs)
- [ ] Write abstract (problem, method, results, conclusion)
- [ ] Design poster layout
- [ ] Review and finalize

## STAGE 6 — Final Report & Presentation
- [ ] Write methodology, experiments, results, discussion sections
- [ ] Include all plots, tables, comparisons
- [ ] Prepare presentation slides
- [ ] **FINAL PRESENTATION (21/06)**

## STAGE 7 — Final Submission (26/07)
- [ ] Clean repository
- [ ] Verify reproducibility
- [ ] Finalize report
- [ ] Submit

---

## Execution Timeline

| Week | Dates | Tasks |
|------|-------|-------|
| 1 | Apr 8–14 | Phase A (CIFAR-10 systematic). Implement MNIST pipeline. |
| 2 | Apr 15–21 | Phase B (MNIST systematic). Phase D (baselines). |
| 3 | Apr 22–28 | Phase C (isolation experiments). Generate dashboards. |
| 4 | Apr 29–May 5 | Analysis: tables, figures, key findings. |
| 5 | May 6–12 | Verify reproducibility. Draft analysis. |
| 6–7 | May 13–26 | Poster design + abstract. |
| **8** | **May 27–31** | **POSTER & ABSTRACT DEADLINE** |
| 9–11 | Jun 1–21 | Report + **PRESENTATION (21/06)** |
| 12–16 | Jun 22–Jul 26 | Polish + **SUBMISSION (26/07)** |
