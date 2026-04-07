# TODO – Final Project Execution Plan

## Current Phase
Post-progress stage → entering Advanced Research + Finalization

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

## STAGE 1 – Fix TransNeXt (CRITICAL)

Goal: determine if TransNeXt is viable

Tasks:

[ ] Run head-only experiment
    - freeze backbone
    - train classifier only

[ ] Run linear probe
    - extract features
    - train linear classifier

[ ] Compare to ResNet baseline

[ ] Answer:
    → does pretrained TransNeXt contain useful features?

Decision:

IF no signal:
    → stop investing in TransNeXt
    → project becomes CNN-focused

IF signal exists:
    → proceed to fine-tuning

---

## STAGE 2 – Controlled Experiments

Goal: build strong comparison

Tasks:

[ ] Run official experiments for:
    - ResNet50
    - DenseNet121
    - TransNeXt (if valid)

[ ] Evaluate across:
    - low_res = 16, 8
    - different noise levels

[ ] Generate:
    - accuracy vs epoch
    - accuracy vs degradation
    - robustness curves

[ ] Export:
    - tables (CSV → report-ready)
    - plots (publication-ready)

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