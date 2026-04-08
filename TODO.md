# TODO – Final Project Execution Plan

## Current Phase
✅ STAGE 1 COMPLETE → Proceeding to STAGE 2

---

## Deadlines

- Poster & Abstracts: 31/05/2026
- Final Presentation: 21/06/2026
- Final Submission: 26/07/2026

---

## MAIN OBJECTIVE NOW

Turn the project from:
"working benchmark"

into:
"clear scientific conclusion with strong evidence"

---

## STAGE 1 – Fix TransNeXt (CRITICAL) ✅ COMPLETE

Goal: determine if TransNeXt is viable

Tasks:

[x] Run head-only experiment
    - freeze backbone
    - train classifier only
    - ✅ RESULT: 68.75% val_acc (epoch 11)

[x] Run linear probe
    - extract features
    - train linear classifier
    - ✅ GPU-accelerated training on RTX 4050

[x] Compare to ResNet baseline
    - ✅ 68.75% TransNeXt vs 54.5% ResNet50
    - ✅ 14.25% IMPROVEMENT

[x] Answer:
    → does pretrained TransNeXt contain useful features?
    → YES! Frozen backbone + head training is highly effective

Decision:

✅ SIGNAL EXISTS:
    → TransNeXt is VIABLE and outperforms baselines
    → proceed to STAGE 2 with confidence
    → TransNeXt = PRIMARY MODEL

---

## STAGE 2 – Controlled Experiments (IN PROGRESS)

Goal: build strong comparison across degradation levels

Tasks:

[x] Run official experiments for:
    - ResNet50 ✓
    - DenseNet121 ✓
    - TransNeXt ✓ (validated in STAGE 1)

[ ] Complete robustness evaluation:
    - [x] TransNeXt low_res=16: 68.75% ✓
    - [ ] TransNeXt low_res=8: (initialize and complete)
    - [x] ResNet50: 54.5% (low_res=16), 47.95% (low_res=8) ✓
    - [x] DenseNet121: 51.15% (low_res=16), 45.05% (low_res=8) ✓

[x] Single-degradation isolation experiments:
    - [x] 9 systematic runs (3 models x 3 types) - 7/9 done
    - [x] Downsampling-only: ResNet 80.75%, DenseNet 77.40%
    - [x] Blur-only: DenseNet 66.40%, ResNet 65.65%
    - [x] Salt & Pepper: ResNet 64.60%, DenseNet 62.05%
    - [ ] TransNeXt single-deg runs (3 pending)

[x] Generate:
    - [x] accuracy vs epoch ✓
    - [x] accuracy vs degradation ✓
    - [x] robustness curves ✓

[x] Export:
    - [x] tables (CSV → 56 runs aggregated) ✓
    - [x] plots (3 publication-ready figures) ✓

[x] Dashboards:
    - [x] Separate full-pipeline vs single-degradation experiments ✓
    - [x] Color-coded degradation groups for fair comparison ✓
    - [x] Sample image viewer (original vs degraded) ✓

---

## STAGE 3 – Analysis (VERY IMPORTANT)

Goal: produce real insight

Tasks:

[ ] Compare models:
    - who is most robust?
    - where does each fail?

[ ] Analyze:
    - effect of resolution
    - effect of noise
    - behavior trends

[ ] Validate hypothesis:
    - feature reuse vs attention

[ ] Write conclusions:
    - NOT just numbers
    - explain WHY results happen

---

## STAGE 4 – Poster & Abstract (DEADLINE: 31/05)

Tasks:

[ ] Select best figures:
    - pipeline diagram
    - degradation examples
    - main comparison graph

[ ] Write abstract:
    - problem
    - method
    - key results
    - conclusion

[ ] Keep it:
    - visual
    - minimal text
    - strong message

---

## STAGE 5 – Final Experiments (JUNE)

Goal: close all gaps

Tasks:

[ ] Run any missing experiments
[ ] Verify reproducibility
[ ] Lock final results
[ ] Save best model weights

---

## STAGE 6 – Final Report

Tasks:

[ ] Write:
    - methodology
    - experiments
    - results
    - discussion

[ ] Include:
    - plots
    - tables
    - comparisons

[ ] Emphasize:
    - robustness
    - architecture insight

---

## STAGE 7 – Final Presentation (21/06)

Tasks:

[ ] Update slides from progress version
[ ] Replace:
    - interim results → final results
    - open questions → conclusions

[ ] Prepare:
    - strong narrative
    - clear takeaway

---

## STAGE 8 – Final Submission (26/07)

Tasks:

[ ] Clean repository
[ ] Organize runs and artifacts
[ ] Finalize report
[ ] Submit all materials

---

## KEY STRATEGY

DO NOT:
- waste time on random experiments
- blindly fine-tune models

DO:
- run controlled experiments
- compare fairly
- extract insight

---

## SUCCESS CONDITION

Project is successful if:

- experiments are reproducible
- comparison is fair
- conclusions are clear and justified

Even if:
TransNeXt fails

The project is still strong if:
you explain WHY