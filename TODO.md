# TODO — Project Execution Plan

## Deadlines
- Poster & Abstract: **31/05/2026**
- Final Presentation: **21/06/2026**
- Final Submission: **26/07/2026**

---

## STAGE 1 — TransNeXt Validation ✅ COMPLETE
- TransNeXt LP: 68.75% vs ResNet50: 54.5% (+14.25%)
- Frozen backbone + head-only training works best

## STAGE 2 — Controlled Experiments ✅ MOSTLY COMPLETE
- [x] Official runs: ResNet50, DenseNet121, TransNeXt
- [x] Combined degradation: DenseNet 80.7%, ResNet 78.8%, TransNeXt 64.2%
- [x] Single-degradation isolation: 7/9 done (TransNeXt pending)
- [x] Dashboards with filtering and sample images
- [ ] Run systematic experiments (3 levels x 3 models) via `run_systematic.py`

## STAGE 3 — Analysis
- [ ] Compare model robustness across degradation levels
- [ ] Explain WHY results occur (feature reuse vs attention)
- [ ] Write conclusions with supporting evidence

## STAGE 4 — Poster & Abstract (31/05)
- [ ] Select best figures (pipeline diagram, degradation examples, comparison graph)
- [ ] Write abstract (problem, method, results, conclusion)

## STAGE 5 — Final Experiments (June)
- [ ] Run missing experiments, verify reproducibility
- [ ] Lock final results, save best weights

## STAGE 6 — Final Report
- [ ] Write methodology, experiments, results, discussion
- [ ] Include plots, tables, comparisons

## STAGE 7 — Final Presentation (21/06)
- [ ] Update slides with final results and conclusions

## STAGE 8 — Final Submission (26/07)
- [ ] Clean repository, finalize report, submit
