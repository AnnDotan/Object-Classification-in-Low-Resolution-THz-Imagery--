# PRD V2 (refined) - Final Summary Report Production Program (`fin-2026-061`)

**Project:** Classification of Objects at a Very Low Resolution (THz-like)
**Project numbers:** `s-2026-045` / `p-2026-061`
**Students:** Bahat Itamar (205805856), Dotan Ann Hanna (211542311)
**Supervisors:** Yitzhaky Yitzhak, Lerner Rinat
**Target artifact:** `Final_Report/fin-2026-061.docx`
**PRD status:** PLAN ONLY - no implementation in this session
**Build pipeline:** LaTeX (IEEEtran-compatible source) -> DOCX via Pandoc + BGU `reference.docx` style template
**Submission deadline:** 26 / 07 / 2026
**V1 archive:** [archive/V1/](archive/V1/) (frozen 2026-06-02)

---

## 0. V2 decisions log (operator-confirmed)

These ten decisions drove the refinement and are reflected throughout this PRD.

| # | Decision (operator answer) | Effect on the PRD |
|---|---|---|
| Q1 | **C** - Hybrid structure: phase-major Acceptance Tests, plus a new dedicated chapter "Cross-model robustness" right before Conclusions. | New section §7.8 added; §7.7 stays phase-major; figure stream re-numbered. |
| Q2 | **B** - Two separate figures per (phase, dataset) for PSNR and SSIM. | §4.2 produces 8 body figures (PSNR / SSIM x Phase C / Phase C2 x CIFAR-10 / MNIST). Each figure carries three model traces. |
| Q3 | **B** - Keep only the Phase B1 + Phase D-T3 overlay L1..L5 line plot in the appendix; delete Phase C L1..L5 line plots entirely. | Appendix F shrinks to two files (one per dataset). Per-axis L1..L5 plots deleted from the figure tree. |
| Q4 | **A + B** - Lock the cross-model numbers pending one operator number check, frame the prose neutrally (no "TransNeXt wins" rhetoric). | Abstract states the +5 / +7 / <1 pp magnitudes verbatim, but the surrounding sentences use language like "TransNeXt-tiny holds the highest margin at mild and moderate degradation" rather than ranking verbs. |
| Q5 | **B** - Implementation agent drafts a fresh Hebrew translation matching the new English framing; operator reviews. | V1 `hebrew_abstract.txt` is the starting point but is re-written end-to-end. Inline in `01_abstract.tex` per R5. |
| Q6 | **A** - Compress to ~25 pages. | Aggressive prose compression: bullets replace paragraphs in §7.5 spec sheet, §7.7 acceptance tests, §7.9 problems; phase definitions live only in §7.4 (no re-explanation downstream); captions tightened; calibration panels combined into two grid PNGs (FX-05) to count as two embeds instead of twelve. Hard ceiling: 30 pages; aspirational target: 25. |
| Q7 | **A** - Keep all 12 reliability panels in the body. | FX-05 combines them into `B_L3_grid.png` and `B_L5_grid.png` (3x2 each) so they cost two figure embeds in page count. |
| Q8 | **AS IS** - V1 sample-grid layout: one row per condition, six fixed validation indices per row (mixed classes). | No re-render; only FX-10 ASCII-safe sub-caption is applied. |
| Q9 | **C** - Time-only cost summary; supervisor signs a duration attestation. No money figures. | Table XVI becomes a duration table (per phase, planned vs actual GPU-h + operator-h); the signature line is a duration attestation, not a cost statement. |
| Q10 | **D** - Maximal equation typesetting: every architecture gets a formal numbered equation; plus the loss, plus the optimizer update rule. | §7.6 design section expands to ~7 numbered equation blocks (ResNet residual, DenseNet concat, TransNeXt attention with foveal mask, label-smoothed cross-entropy, AdamW update, cosine LR schedule, PSNR / SSIM / ECE definitions). |

---

## 1. Purpose

Produce the final summary report (`דוח מסכם`, §10.7 of *Fourth Year Engineering Project Procedure*, edition 2020-1, 22 / 10 / 2019) that:

1. Presents a head-to-head **robustness comparison of ResNet50, DenseNet121, and TransNeXt-tiny** under controlled synthetic THz-like degradation. The cross-model comparison is the primary scientific contribution and gets a dedicated chapter (§7.8).
2. Decomposes the accuracy loss into per-axis contributions, anchored on mean PSNR and mean SSIM. Every single-axis comparison plot uses PSNR or SSIM on the x-axis (R2 / Q2). Level indices L1..L5 are demoted to the appendix.
3. Compiles to a single DOCX (`fin-2026-061.docx`) that conforms exactly to BGU School of ECE regulations.
4. Re-uses the 402 canonical training cells + 48 multi-seed audit replicates already on disk. **No re-training.**
5. Reads as a self-contained engineering report - a reviewer with no prior contact with the repository must be able to read it end-to-end without encountering any internal-process language (R3).
6. Targets ~25 pages, hard ceiling 30 (Q6).

Implementation will happen in a separate session. This document is the contract between sessions.

---

## 2. Authoritative regulation extract (BGU §10.7 + §10.2)

The report is a single DOCX file named `fin-2026-061.docx`. Per §10.7.2 it MUST contain, in this order:

1. Cover page (per appendix 12.1)
2. Abstract / `תקציר סיום פרויקט` in English **and** Hebrew (§10.7.2.3)
3. Introduction (§10.7.2.4)
4. Final technical spec sheet (§10.7.2.5)
5. Engineering solution and design approach (§10.7.2.6)
6. Final acceptance tests (§10.7.2.7)
7. Problems and solutions (§10.7.2.8)
8. Conclusions and recommendations (§10.7.2.9)
9. References (§10.7.2.10)
10. Appendices (§10.7.2.11)

V2 inserts one extra analysis chapter (§7.8, Cross-model robustness) between Problems and Conclusions. The ten regulation-mandated components remain present and in order; the insertion is allowed because the regulation lists minima, not maxima.

Style requirements (§10.2) - unchanged from V1: Times New Roman 12 pt, 1.5 line spacing, 2.5 cm margins all sides, justified, Roman front-matter pagination, Arabic body pagination, IEEE figure / table captions with separate numbering streams, IEEE-style equations.

Forbidden in the body (unchanged from V1): em-dash / en-dash, fabricated numbers, marketing language, emoji, inline TODO / FIXME markers, copy-paste from the preliminary report, screenshots in lieu of data.

**V2 additional bans (R3).** None of the following may appear anywhere in the report body or its captions:
`PRD`, `PRD v4`, `PRD v5`, `US-001` through `US-099`, `Ralph loop`, `Ralph`, `MASTER`, `Claude`, `Codex`, `sub-agent`, `agent spec`, `sprint`, `commit`, `git`, `branch`, `PHASE_B2`, `operator-locked`, `BLOCKED-OPERATOR`. The new `scripts/process_term_lint.py` fails the build if any term appears in the rendered DOCX body text.

---

## 3. Inputs available (audited 2026-06-02)

| Source | Path | Purpose |
|---|---|---|
| Master tracker (402 rows) | [Final_Exp.md](../Final_Exp.md) | Per-cell ID, tag, val-acc, GPU-h |
| Aggregated JSON (schema v3) | [artifacts/Final_Exp.json](../artifacts/Final_Exp.json) | Multi-seed `[mean +/- std]` per cell |
| Per-pipeline image quality | [Final_Report/data/image_quality/_index.csv](data/image_quality/_index.csv) | 102 pipelines x mean PSNR / SSIM (built in V1) |
| V1 figures (reused unless on §8 fix list) | [Final_Report/figures/](figures/) | V1 baseline |
| V1 LaTeX sources | [Final_Report/source/](source/) | V1 baseline; V2 edits in place |
| V1 PRD (frozen reference) | [archive/V1/PRD_FinalReport.md](archive/V1/PRD_FinalReport.md) | History only |
| V1 deliverable (frozen reference) | [archive/V1/fin-2026-061.docx](archive/V1/fin-2026-061.docx) | History only |
| BGU regulations | [nohal_2026_ 21-10-2025תשפו.pdf](nohal_2026_%2021-10-2025%D7%AA%D7%A9%D7%A4%D7%95.pdf) | Authoritative ruleset |
| Degradation table | [src/data/degradation_levels.py](../src/data/degradation_levels.py) | L1..L5 single source of truth |

V2 reuses the V1 PSNR / SSIM probe verbatim. No re-probe.

---

## 4. Net-new analysis required for V2

None of these re-train.

### 4.1 Per-cell join (`per_cell_with_iq.csv`)

Single long-form table joining accuracy with image quality.

Columns: `tag, model, dataset, phase, level, axis, regularization, seed, val_acc, psnr_mean, psnr_std, ssim_mean, ssim_std`.

Join keys: `(dataset, phase, level, axis)` for image quality; `tag` for accuracy. Phase D rows reuse the Phase B1 image-quality numbers (pixels identical; regularization is training-side). The table footnote documents the re-use.

### 4.2 Per-axis val-acc plots vs PSNR and vs SSIM (R2, Q2)

**Why.** L1..L5 on the x-axis hides the fact that L3 resolution (PSNR ~= 12 dB on CIFAR-10), L3 blur (PSNR ~= 18 dB), and L3 saturation (PSNR ~= 28 dB) are radically different image-quality operating points.

**Required output (Q2 = two separate figures per (phase, dataset)):**

```
figures/curves/acc_vs_psnr_phaseC_cifar10.{png,pdf}
figures/curves/acc_vs_ssim_phaseC_cifar10.{png,pdf}
figures/curves/acc_vs_psnr_phaseC_mnist.{png,pdf}
figures/curves/acc_vs_ssim_phaseC_mnist.{png,pdf}
figures/curves/acc_vs_psnr_phaseC2_cifar10.{png,pdf}
figures/curves/acc_vs_ssim_phaseC2_cifar10.{png,pdf}
figures/curves/acc_vs_psnr_phaseC2_mnist.{png,pdf}
figures/curves/acc_vs_ssim_phaseC2_mnist.{png,pdf}
```

Eight body figures total. Plot spec:

* x-axis: `psnr_mean` (or `ssim_mean`).
* y-axis: validation accuracy in percent.
* Five axes traced for Phase C (resolution, blur, noise, saturation, S&P); three axes for Phase C2 (resolution, blur, S&P).
* Three model traces overlaid per plot (ResNet50 = blue circle, DenseNet121 = green square, TransNeXt-tiny = red triangle). Per-axis line styles distinguish the five axes within a model.
* Five points per axis line (the five levels).
* Same y-range across CIFAR-10 vs MNIST so the cross-dataset comparison is visual.
* Caption reports per-axis Spearman rho between accuracy and the metric, plus a one-sentence ranking note.

The cross-phase scatters (`accuracy_vs_psnr.png` and `accuracy_vs_ssim.png`, all 102 pipelines) stay in the body and precede the per-axis plots.

**L1..L5 plots demoted (Q3).** Per Q3 = B, only the Phase B1 + Phase D-T3 overlay survives, in Appendix F. The per-axis L1..L5 plots are deleted from `figures/curves/`.

### 4.3 Re-cuts of accuracy data (unchanged from V1 unless noted)

| Cut | V1 file | V2 action |
|---|---|---|
| Cross-phase accuracy vs PSNR scatter (102 pipelines) | [figures/curves/accuracy_vs_psnr.png](figures/curves/accuracy_vs_psnr.png) | Keep in body, FX-01 cleanup |
| Cross-phase accuracy vs SSIM scatter (102 pipelines) | [figures/curves/accuracy_vs_ssim.png](figures/curves/accuracy_vs_ssim.png) | Keep in body, FX-01 cleanup |
| Per-axis attribution at L3 bar chart | [figures/attribution/](figures/attribution/) | Keep as a complement to §4.2 (a single-glance ranking); FX-02 |
| Phase D recovery summary | [figures/phase_d/recovery_summary.png](figures/phase_d/recovery_summary.png) | Keep; caption rewrite removes V1 jargon; FX-08 |
| Phase B2 protocol-simplification decomposition summary | [figures/phase_b2/comparison_summary.png](figures/phase_b2/comparison_summary.png) | Re-render at publication size; FX-03 |
| Phase C2 per-axis THz attribution summary | [figures/phase_c2/attribution_summary.png](figures/phase_c2/attribution_summary.png) | Keep; caption notes overlap with §4.2 plots |
| Calibration reliability diagrams (12 panels) | [figures/calibration/](figures/calibration/) | Per Q7 = keep all 12; FX-05 combines into two grid PNGs (`B_L3_grid.png`, `B_L5_grid.png`) so they cost two figure embeds in page count |
| ECE vs level line chart | NEW (derived from `calibration.json`) | NEW per V2 §4.5 |
| Multi-seed variance boxplot at L3 | [figures/multiseed/](figures/multiseed/) | Re-render with rotated x-labels and external legend; FX-04 |
| Inference throughput bar | `inference_throughput.png` | Keep; FX-07 |
| Confusion focus at L5 | [figures/confusion/](figures/confusion/) | Re-render as one grid PNG with shared colorbar; FX-06 |

### 4.4 Tables

| # | Table | V2 action |
|---|---|---|
| I | Final technical spec sheet | Rewrite as as-built spec narrative; bullet-dense, R3-clean. |
| II | **Phases at a glance** (NEW per R4) | NEW. One row per phase: cells, axes-active, levels, regularization, research question. |
| III | Degradation level schedule | Unchanged from V1. |
| IV | Per-pipeline mean PSNR / SSIM (102 rows) | Unchanged. |
| V | Phase A clean baseline | Unchanged. |
| VI | Phase B1 combined-degradation accuracy | Caption rewrite (per-level PSNR / SSIM single-pair column group + footnote). |
| VII | Phase C single-axis accuracy | Add two columns: mean PSNR, mean SSIM per row (FX-12). |
| VIII | Phase D regularization-recovery accuracy | Caption rewrite (T1 / T2 / T3 legend at top of table). |
| IX | Phase B2 + B2-nr accuracy | Caption rewrite. |
| X | Phase C2 single-axis under THz protocol | Add two columns: mean PSNR, mean SSIM per row. |
| XI | Multi-seed audit summary (mean +/- sigma across {42, 43, 44}) | Unchanged. |
| XII | Per-axis attribution at L3 | Unchanged. |
| XIII | Calibration ECE per (model, dataset, level) | Unchanged. |
| XIV | Inference throughput | Unchanged. |
| XV | Goals vs achieved | Rewrite with the cross-model framing (Q1 + Q4). |
| XVI | **Duration attestation** (signed by supervisor, Q9) | Replaces V1 cost summary. Columns: phase, planned GPU-h, actual GPU-h, planned operator-h, actual operator-h. No money figures. Signature line attests to duration only. |

### 4.5 Figures (V2 numbered list)

| # | Figure | Source | V2 action |
|---|---|---|---|
| 1 | System block diagram | [block_diagrams/system_overview.png](figures/block_diagrams/system_overview.png) | Unchanged |
| 2 | Pipeline block diagram | [block_diagrams/pipeline_overview.png](figures/block_diagrams/pipeline_overview.png) | Unchanged |
| 3 | Sample grid: Phase B1 levels, CIFAR-10 | [samples/grid_phase_B_levels_cifar10.png](figures/samples/grid_phase_B_levels_cifar10.png) | Q8 = as-is; FX-10 caption fix |
| 4 | Sample grid: Phase B1 levels, MNIST | [samples/grid_phase_B_levels_mnist.png](figures/samples/grid_phase_B_levels_mnist.png) | Q8 = as-is; FX-10 caption fix |
| 5 | Sample grid: per-axis isolation at L3, CIFAR-10 | [samples/grid_phase_C_axes_L3_cifar10.png](figures/samples/grid_phase_C_axes_L3_cifar10.png) | Q8 = as-is |
| 6 | Cross-phase accuracy vs PSNR (102 pipelines) | [curves/accuracy_vs_psnr.png](figures/curves/accuracy_vs_psnr.png) | FX-01 |
| 7 | Cross-phase accuracy vs SSIM (102 pipelines) | [curves/accuracy_vs_ssim.png](figures/curves/accuracy_vs_ssim.png) | FX-01 |
| **8 NEW** | Per-axis acc vs PSNR, Phase C, CIFAR-10 | §4.2 | NEW |
| **9 NEW** | Per-axis acc vs SSIM, Phase C, CIFAR-10 | §4.2 | NEW |
| **10 NEW** | Per-axis acc vs PSNR, Phase C, MNIST | §4.2 | NEW |
| **11 NEW** | Per-axis acc vs SSIM, Phase C, MNIST | §4.2 | NEW |
| **12 NEW** | Per-axis acc vs PSNR, Phase C2, CIFAR-10 | §4.2 | NEW |
| **13 NEW** | Per-axis acc vs SSIM, Phase C2, CIFAR-10 | §4.2 | NEW |
| **14 NEW** | Per-axis acc vs PSNR, Phase C2, MNIST | §4.2 | NEW |
| **15 NEW** | Per-axis acc vs SSIM, Phase C2, MNIST | §4.2 | NEW |
| 16 | Per-axis attribution at L3 bar chart, both datasets | [attribution/](figures/attribution/) | FX-02 |
| 17 | Phase B2 protocol-simplification decomposition summary | [phase_b2/comparison_summary.png](figures/phase_b2/comparison_summary.png) | FX-03 |
| 18 | Phase D recovery summary | [phase_d/recovery_summary.png](figures/phase_d/recovery_summary.png) | FX-08 |
| 19 | Reliability diagrams grid at L3 (3x2) | NEW combined grid `calibration/B_L3_grid.png` | NEW (Q7 + FX-05) |
| 20 | Reliability diagrams grid at L5 (3x2) | NEW combined grid `calibration/B_L5_grid.png` | NEW (Q7 + FX-05) |
| 21 | ECE vs level line chart | NEW from `calibration.json` | NEW |
| 22 | Confusion matrices grid at L5 (3x2) | NEW combined grid `confusion/B_L5_grid.png` | NEW (FX-06) |
| 23 | Multi-seed variance boxplot, both datasets | [multiseed/](figures/multiseed/) | FX-04 |
| 24 | Inference throughput bar | `inference_throughput.png` | FX-07 |
| 25 | Gantt: planned vs actual schedule | [gantt/](figures/gantt/) | FX-09 |
| A1 (appendix F) | Phase B1 + Phase D-T3 overlay L1..L5, CIFAR-10 | [curves/accuracy_vs_level_cifar10.png](figures/curves/accuracy_vs_level_cifar10.png) | Demoted from body (Q3 = B) |
| A2 (appendix F) | Phase B1 + Phase D-T3 overlay L1..L5, MNIST | [curves/accuracy_vs_level_mnist.png](figures/curves/accuracy_vs_level_mnist.png) | Demoted from body (Q3 = B) |

Per-axis L1..L5 line plots (any new file matching `acc_vs_level_phaseC*.png`) are NOT produced (Q3 = B).

---

## 5. Folder layout

```
Final_Report/
├── PRD_FinalReport.md                ← this V2 PRD
├── PROGRESS.txt                      ← V2 authoring progress log
├── fin-2026-061.docx                 ← V2 deliverable (rebuilt next session)
├── fin-2026-061.pdf                  ← V2 reference render (rebuilt next session)
├── source/
│   ├── fin-2026-061.tex
│   ├── refs.bib
│   ├── bgu_reference.docx
│   ├── sections/                     ← V2 edits + new 04_phases_overview.tex + new 08_cross_model.tex; 01_abstract.tex inlines Hebrew (R5)
│   ├── tables/                       ← V2 adds tab_II_phases_overview.tex; PSNR / SSIM columns on tab_VI_phase_c.tex, tab_X_phase_c2.tex; tab_XVI rewritten as duration attestation (Q9)
│   └── filters/                      ← dash_sanitizer.lua, ieee_caption.lua + NEW process_term_lint.lua
├── figures/
│   ├── curves/                       ← + acc_vs_psnr_phase{C,C2}_<dataset>.{png,pdf}, acc_vs_ssim_phase{C,C2}_<dataset>.{png,pdf}, ece_vs_level_<dataset>.{png,pdf}
│   ├── calibration/                  ← + B_L3_grid.png, B_L5_grid.png (FX-05 combined)
│   ├── confusion/                    ← + B_L5_grid.png (FX-06 combined)
│   ├── multiseed/, attribution/, phase_b2/, phase_c2/, phase_d/, samples/, gantt/, block_diagrams/
├── data/
│   ├── image_quality/                ← V1 baseline (102 pipelines)
│   ├── tables/                       ← V1 baseline + per_cell_with_iq.csv
│   └── per_cell_metrics.csv
├── build/
│   ├── Makefile, build_all.ps1, tex/
├── scripts/
│   └── V1 scripts + build_psnr_axis_curves.py + build_ece_vs_level.py + build_per_cell_with_iq.py + process_term_lint.py + combine_calibration_grids.py + combine_confusion_grid.py
└── archive/
    └── V1/                           ← FROZEN V1 snapshot
```

---

## 6. Build pipeline (V2 deltas only)

### 6.1 Tooling

V1 stack + XeLaTeX with `polyglossia` + a Hebrew Unicode font (Culmus David CLM, Frank Ruhl CLM, or any installed system Hebrew font). The build script auto-detects the font and substitutes; if none is found, it emits a warning and falls back to pdfLaTeX with a transliteration placeholder (the DOCX still ships correct Hebrew because Pandoc renders the polyglossia environment).

### 6.2 Build steps

1. Analysis - V1 chain unchanged, then:
   * `scripts/build_per_cell_with_iq.py` -> `data/tables/per_cell_with_iq.csv`
   * `scripts/build_psnr_axis_curves.py` -> 8 PNG/PDF pairs + Spearman rho metadata JSON for caption injection
   * `scripts/build_ece_vs_level.py` -> ECE vs level line chart
   * `scripts/combine_calibration_grids.py` -> `B_L3_grid.png`, `B_L5_grid.png`
   * `scripts/combine_confusion_grid.py` -> `B_L5_grid.png`
2. Re-render the figures on the §8 fix list.
3. LaTeX edits:
   * Inline Hebrew abstract in `01_abstract.tex` (R5, Q5).
   * Strip banned terms from all `0?_*.tex` (R3).
   * Insert new `04_phases_overview.tex` between Introduction and Spec Sheet (R4).
   * Insert new `08_cross_model.tex` between Problems and Conclusions (Q1).
   * Renumber all `\section{...}` callsites and figure / table cross-refs.
   * Add the maximal equation block to the Design section (Q10).
4. Validation:
   * Existing `validate_report.py` (fonts, spacing, margins, em-dash, en-dash, emoji) - unchanged.
   * NEW `process_term_lint.py fin-2026-061.docx` - exit code != 0 if any banned term appears.
   * NEW page-count gate: `pdfinfo fin-2026-061.pdf | grep Pages` reports <= 30; warn at > 25 (Q6 aspirational, hard ceiling at 30).
   * NEW Hebrew-presence gate: at least one U+0590..U+05FF character in both DOCX body text and PDF text layer.
   * NEW "80 %" gate: the string "80 %" / "80%" / "80 percent" appears at most once in the body and never in the abstract (R1, Q4).
5. Hand QA pass - Hebrew readability, Spearman rho values, banned-term lint exit code, page count, §8 fix log closed.

### 6.3 Reproducibility

`Makefile` targets unchanged. `make all` end-to-end under 5 minutes on the 5070 host.

---

## 7. Section-by-section content brief (V2)

### 7.1 Cover page

Unchanged from V1.

### 7.2 Abstract - English + Hebrew (R1, R5, Q4, Q5)

**English (structured, ~250 words):**

* **Background (~25 w).** THz imaging produces low-resolution, low-SNR, colour-poor frames that defeat ImageNet-pretrained classifiers. The magnitude and the mechanism of the loss across modern deep-learning architectures are not documented quantitatively.
* **Aim (~25 w).** Compare the robustness of three representative deep architectures - a residual CNN, a densely connected CNN, and a foveal vision transformer - under controlled synthetic THz-like degradation, decompose the loss into per-axis contributions, and quantify how much of it can be recovered by training-time regularization.
* **Methods (~50 w).** 402-cell campaign on ResNet50, DenseNet121, TransNeXt-tiny x CIFAR-10, MNIST under a five-level degradation curve (saturation, downsample, blur, additive noise, salt-and-pepper). Seven phases: A clean, B1 combined, C single-axis, D regularization-recovery, B2 / B2-nr THz-protocol simplification, C2 single-axis under the THz protocol. 48-replicate multi-seed audit. Every pipeline anchored on PSNR / SSIM over a 256-image deterministic probe.
* **Findings (~50 w).** Under Phase B1 at L3 on CIFAR-10, TransNeXt-tiny holds a +5.0 pp margin over ResNet50 and DenseNet121; the margin widens to +7.0 pp at L2 and narrows to under 1 pp at L5. Resolution dominates the per-axis attribution by approximately 4x (mean L1..L5 effect on CIFAR-10: -27.45 pp for resolution vs -6.58 pp for blur and -5.13 pp for salt-and-pepper). Regularization (T3 in Phase D) recovers ~0.77 pp of the ~12.49 pp Phase B v2 collapse. The multi-seed audit confirms sigma < 1.5 pp on every L3 headline cell. Expected calibration error inflates from below 5 pp at L3 to above 15 pp at L5 on CIFAR-10.
* **Conclusion (~25 w).** The accuracy collapse is best modelled as an information bottleneck, not as an overfit. The foveal transformer holds the highest robustness margin at mild and moderate degradation; all three architectures converge at the extreme end. Future THz deployments should invest in resolution restoration before denoising or deblurring.
* **Keywords (~10 w).** Object classification, deep learning, low resolution, THz imaging, robustness, PSNR, SSIM, calibration, transformer, residual network, dense network.

The 80 % CIFAR-10 acceptance criterion from the preliminary report is NOT mentioned in the Abstract (R1). It appears once, in §7.9 Conclusions, against the as-measured numbers.

Q4 = A + B implementation: the +5.0 / +7.0 / <1 pp magnitudes are locked numerically (operator confirms against `per_cell_with_iq.csv` before the implementation session starts; if any number is materially off, the abstract is updated to match). The framing language stays neutral - "holds a margin", "narrows to under 1 pp", "holds the highest robustness margin" - and avoids ranking verbs like "wins" or "beats".

**Hebrew (Q5 = B implementation).** The implementation agent drafts a fresh Hebrew translation matching the new English structure. Starting point: V1 `hebrew_abstract.txt`. Re-written end-to-end so the cross-model framing matches the English. Inlined directly in `01_abstract.tex` immediately after the English abstract, in a `polyglossia` hebrew environment with `\begin{hebrew} ... \end{hebrew}`. Six labelled paragraphs (`רקע`, `מטרה`, `שיטות`, `ממצאים`, `מסקנות`, `מילות מפתח`) mirror the English structure. Operator reviews the draft before D5 QA.

### 7.3 Introduction (R3, Q6 compression)

Three paragraphs only (compressed from V1's six):

1. **Motivation and prior art.** THz sensing context; ImageNet-pretrained classifiers vs degraded inputs; one paragraph that cites [Sharma & Guleria, DenseNet, TResNet, EfficientNetV2, TransNeXt].
2. **Project goal and contributions.** Head-to-head robustness comparison of three architecture families. Four numbered contributions: (i) full per-axis attribution on three modern backbones; (ii) PSNR / SSIM-anchored cross-pipeline comparison; (iii) decomposition of the Phase B accuracy collapse into pure-protocol vs regularization-recoverable components; (iv) calibration assessment under extreme degradation.
3. **Survey of alternatives.** Bullet list - denoise-then-classify; super-resolution-then-classify; augmentation-only training; synthetic-degradation training-time exposure (this work). One short sentence each.

No mention of PRD, US-XXX, Ralph, Claude, Codex, MASTER, sub-agents, sprint, commit, git, or branch names (R3).

### 7.4 NEW - Experimental phases at a glance (R4)

The new sub-section. One short paragraph per phase, each opening with the pipeline used and closing with the research question motivating it. Followed by Table II.

* **Phase A (Clean baseline).** Pipeline: identity. Six cells. Upper bound for each (model, dataset).
* **Phase B1 (Combined degradation).** Pipeline: `Sat -> Downsample -> Upsample_224 -> Blur -> Noise -> S&P`, all five axes active at the same level. 30 cells (3 x 2 x 5). Quantifies the realistic worst-case degradation curve.
* **Phase C (Single-axis isolation).** Pipeline: one axis active at level L, four others at identity. 150 cells (3 x 2 x 5 x 5). Decomposes the Phase B1 collapse into per-axis contributions. Plotted vs PSNR and SSIM in §7.7.
* **Phase D (Regularization-recovery sweep).** Pipeline: identical to Phase B1, plus three training-time regularization treatments. 90 cells (3 x 2 x 3 x 5). Asks whether the Phase B1 collapse is an overfit (regularization should recover it) or an information bottleneck (recovery should be marginal).
* **Phase B2 (THz-protocol simplification).** Pipeline: Phase B1 minus saturation and additive noise; regularization treatment T3 active. 30 cells. Models the realistic THz acquisition (single-channel low-res blurry frame, no chroma noise).
* **Phase B2-nr (No-regularization arm).** Same pipeline as Phase B2 at L3 only, T3 off. Six cells. Decomposes the Phase B1 -> Phase B2 delta into pure-protocol and regularization components.
* **Phase C2 (Single-axis isolation under the THz protocol).** Same identity-elsewhere logic as Phase C, restricted to the three THz-relevant axes (resolution, blur, S&P). 90 cells (3 x 2 x 3 x 5). Per-axis attribution under deployment-realistic conditions.

**Table II (Phases at a glance)** columns: phase, cells, axes active at level L, levels swept, regularization, primary research question. One row per phase + one row for the multi-seed audit.

Every later section refers back to §7.4 for pipeline definitions and does NOT re-explain them (Q6 compression).

### 7.5 Final technical spec sheet (§10.7.2.5, R3, Q6)

Bullet-dense, narrative-light:

* Product: a Python package that synthetically degrades a dataset on a five-level curve, fine-tunes a backbone, and reports val-accuracy + calibration + throughput, all from a single config.
* As-built drift list (numbered, scientific framing - no US history, no Ralph, no commit hashes):
  1. Vision transformer variant: an early prototype used `transnext_small` (50 M params); the as-built uses `transnext_tiny` (28 M params) to match the ResNet50 parameter budget.
  2. Input resolution: a brief native-resolution detour was rolled back in favour of uniform 224 x 224 input across all three models (fair-comparison invariant).
  3. Blur kernel rescaling: kernel and sigma values are pixel-domain at 224 x 224, not at the native input size.
  4. Hyper-parameter freeze: per (model, dataset) the L3 winner from a 20-trial Optuna sweep is reused across all phases; per-phase re-tuning is out of scope.
  5. Multi-seed audit: the 24 L3 headline cells are replicated under two extra seeds for reproducibility bounds.
* Functional spec: a single compact subsection listing inputs, outputs, metrics, acceptance gates, with the as-built numbers.

### 7.6 Engineering solution and design approach (§10.7.2.6, Q10 maximal equations)

Seven numbered equation blocks:

(1) **Composite degradation pipeline.**
$$\tilde{x} = \mathrm{SP}_{p_5} \!\big( \mathrm{N}_{\sigma_4} \!\big( \mathrm{B}_{k_3, \sigma_3} \!\big( \mathrm{U}_{224} \!\big( \mathrm{D}_{r_2} \!\big( \mathrm{Sat}_{s_1}(x) \big) \big) \big) \big) \big)$$

(2) **ResNet residual block.**
$$y = \mathcal{F}(x, \{W_i\}) + x$$

(3) **DenseNet dense block.**
$$x_\ell = H_\ell \!\big( [x_0, x_1, \ldots, x_{\ell-1}] \big)$$

(4) **TransNeXt foveal attention.**
$$\mathrm{Attn}(Q, K, V) = \mathrm{softmax} \!\left( \frac{QK^\top \odot M_{\mathrm{foveal}}}{\sqrt{d_k}} \right) V$$

where $M_{\mathrm{foveal}}$ is the pixel-focused window mask.

(5) **Label-smoothed cross-entropy loss.**
$$\mathcal{L}_{\mathrm{LS}} = - \sum_{k=1}^{K} \!\Big( (1 - \alpha) \mathbb{1}[k=y] + \frac{\alpha}{K} \Big) \log p_k$$

(6) **AdamW parameter update.**
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \eta_t \!\left( \frac{m_t / (1 - \beta_1^t)}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon} + \lambda \theta_{t-1} \right)$$

(7) **Cosine LR schedule.**
$$\eta_t = \eta_{\min} + \tfrac{1}{2} (\eta_{\max} - \eta_{\min}) \!\left( 1 + \cos \!\left( \frac{t}{T} \pi \right) \right)$$

Followed by the PSNR / SSIM / ECE definition block (eqs. 8, 9, 10) carried over from V1.

Each equation gets one short caption sentence; no narrative paragraphs between equations (Q6 compression). Block diagrams (Figs. 1, 2) precede the equation block; per-block design considerations (why bicubic upsample, why saturation before noise, why ImageNet normalization after S&P, why label smoothing on TransNeXt FT) sit as a single bullet list after equation 10.

### 7.7 Final acceptance tests - phase-major (§10.7.2.7, Q1, Q6)

Compressed compared to V1. One sub-section per phase, no more than half a page each, structured as:

* One-sentence research question (lifted from §7.4 Table II).
* The relevant table + figure embedded inline.
* One interpretive paragraph (3-4 sentences max).

Sub-sections: §7.7.1 Phase A clean baseline; §7.7.2 Phase B1 combined; §7.7.3 Phase C single-axis (carries Figs. 8-11 and Table VII); §7.7.4 Phase D regularization recovery; §7.7.5 Phase B2 protocol simplification; §7.7.6 Phase C2 THz axes (carries Figs. 12-15 and Table X); §7.7.7 Multi-seed reproducibility; §7.7.8 Calibration (Figs. 19, 20, 21; Table XIII); §7.7.9 Confusion focus (Fig. 22); §7.7.10 Inference throughput (Fig. 24; Table XIV).

No re-explanation of phase pipelines; every reference points back to §7.4. The cross-model interpretation lives in §7.8, not here - this section only states phase-by-phase outcomes.

### 7.8 NEW - Cross-model robustness comparison (Q1 = C, R1 foreground)

The dedicated model-major chapter. Three sub-sections, one per model:

* **§7.8.1 ResNet50 robustness profile.** One paragraph synthesising the model's behaviour across Phases B1, C, D, B2, C2. Reads from `per_cell_with_iq.csv`. Inline references to Figs. 6-15 (no new figures). One small table summarising the model's mean accuracy at each level on each dataset.
* **§7.8.2 DenseNet121 robustness profile.** Same structure.
* **§7.8.3 TransNeXt-tiny robustness profile.** Same structure.
* **§7.8.4 Side-by-side reading.** Two paragraphs:
  * Paragraph 1: at matched PSNR (resp. SSIM), the three architectures' accuracy ordering across the operating range. State the +5.0 / +7.0 / <1 pp magnitudes verbatim (Q4 = A) using neutral language (Q4 = B). Cite Figs. 8-15.
  * Paragraph 2: convergence at L5. At PSNR ~= 11 dB on CIFAR-10, all three models land in [19, 21] pp accuracy. Implication for THz deployment: at extreme degradation, the architectural choice is irrelevant; the spatial information is gone.

This chapter is the report's headline contribution. ~2-3 pages including embedded tables.

### 7.9 Problems and solutions (§10.7.2.8, R3, Q6)

Compressed to four entries (V1 had seven). Each one line: symptom -> remedy.

1. Vision-transformer hardware bring-up delayed by Blackwell GPU driver gap; resolved by upstream PyTorch 2.x release for Blackwell.
2. Vision-transformer parameter budget initially over-spec; resolved by swapping to a capacity-matched 28 M-parameter variant.
3. Per-pixel blur invisibility on the upsampled tensor; resolved by rescaling kernel and sigma to pixel-domain at 224 x 224.
4. Single-seed reproducibility exposure; resolved by the 48-replicate multi-seed audit at L3.

The Windows DataLoader, the native-resolution detour, and the Phase B v2 collapse motivation are absorbed into §7.5 as "drift list" entries (avoid duplication).

### 7.10 Conclusions and recommendations (§10.7.2.9, R1, Q4)

Mandatory sub-elements per §10.7.2.9, each terse:

1. Work summary (one paragraph).
2. Preliminary spec vs delivered - Table XV. The 80 % CIFAR-10 acceptance criterion is reported here once: met at L1 / L2 on TransNeXt-tiny under Phase D-T3, not met at L4 / L5 by any model. One sentence, not a paragraph.
3. Pros / cons of the product and the idea (bullet list).
4. Success / failure analysis - one paragraph that points at §7.8 for the cross-model reading and §7.7 for the per-phase outcomes.
5. Plan vs execution - feasibility and timeline (Fig. 25); duration attestation Table XVI (Q9).
6. Recommendations: real THz data acquisition; super-resolution pre-stage; transformer-with-Swin-window backbone; reproduction on ImageNet-A.

### 7.11 References (§10.7.2.10)

Unchanged from V1.

### 7.12 Appendices (§10.7.2.11)

* A. Degradation pipeline source code listing (one page).
* B. Full per-cell metrics CSV (referenced; not embedded).
* C. Sample-strip gallery (figures index).
* D. Multi-seed raw numbers per cell.
* E. PSNR / SSIM per-image distribution histograms.
* **F. Phase B1 + Phase D-T3 overlay L1..L5 line plots** - two figures only, one per dataset (Q3 = B).
* G. IP-rights form (signed).
* H. Evaluation sheets copies.

---

## 8. Figure and table fix log (R6, per-asset)

| Code | Asset | Symptom (V1) | Remedy (V2) |
|---|---|---|---|
| FX-01 | [curves/accuracy_vs_psnr.png](figures/curves/accuracy_vs_psnr.png), [curves/accuracy_vs_ssim.png](figures/curves/accuracy_vs_ssim.png) | Spearman rho box overlaps top-left points on MNIST; legend overlaps lower-right TransNeXt cluster on CIFAR-10. | Move rho annotation to bottom-right (0.97, 0.03 axes coords). Legend at upper-left with `bbox_to_anchor=(0.02, 0.98)` and `framealpha=0.85`. Add `subplots_adjust(wspace=0.18)`. |
| FX-02 | [attribution/axis_attribution_L3_<dataset>.png](figures/attribution/) | Resolution bar on CIFAR-10 rides the bottom edge; legend covers Resolution group. | `ylim(bottom=-32, top=2)`. Legend at `bbox_to_anchor=(1.02, 1.0), loc='upper left'`. `tight_layout()`. `rcParams['axes.unicode_minus']=False`. |
| FX-03 | [phase_b2/comparison_summary.png](figures/phase_b2/comparison_summary.png) | Thumbnail-size, unreadable; sub-titles overflow. | Re-render at `figsize=(10, 6), dpi=200`. 2x3 grid, `sharey='row'`, `constrained_layout=True`. Single shared legend at `fig.legend(..., loc='upper center', ncols=5, frameon=False)`. |
| FX-04 | [multiseed/L3_variance_<dataset>.png](figures/multiseed/) | x-tick labels clipped (90° rotation); legend stack overlaps rightmost bar. | `figsize=(11, 5.5)`, x-labels at 45° right-anchored. Legend outside data box at `(1.02, 1.0)`. `tight_layout(rect=[0, 0, 0.88, 1])`. Use human-readable tag labels (`B / DenseNet121 / no reg`). |
| FX-05 | [calibration/B_L3_*.png](figures/calibration/) (x6), [calibration/B_L5_*.png](figures/calibration/) (x6) | Six independent PNGs per level; Pandoc resizes each independently; colorbars do not align. | NEW combined `B_L3_grid.png`, `B_L5_grid.png` - 3x2 subplots, `sharex='all', sharey='all'`, single shared colorbar. Q7-driven (keep all 12 panels, but as two embeds). |
| FX-06 | [confusion/B_L5_*.png](figures/confusion/) (x6) | Per-panel `vmax` differs; cross-model comparison misleading. | NEW combined `B_L5_grid.png` - shared `vmin=0, vmax=1` (normalized), single colorbar. |
| FX-07 | `inference_throughput.png` | Top bar rides the top axis; value label clipped. | `ax.set_ylim(0, max_value * 1.1)`; annotate each bar value above the bar. |
| FX-08 | [phase_d/recovery_summary.png](figures/phase_d/recovery_summary.png) | Caption uses "v1 / v2 baseline" without defining them. | Caption rewrite: define "Phase B baseline" as "Phase B1 cells under the frozen hyperparameters" once; refer to "Phase B" throughout. |
| FX-09 | [gantt/gantt_planned_vs_actual.png](figures/gantt/gantt_planned_vs_actual.png) | Bars labeled by US codes (US-001 .. US-053) - violates R3. | Re-label bars by milestone names: "Hardware bring-up", "Hyper-parameter sweep", "Phase A baselines", "Phase B1 combined", "Phase C single-axis", "Phase D regularization", "Phase B2 THz protocol", "Phase C2 THz axes", "Multi-seed audit", "Report authoring". |
| FX-10 | All sample grids under [samples/](figures/samples/) | Sub-captions use `|diff|` which renders as table delimiters in some Word renderers. | Replace `|diff|` with `Δ` in rendered captions; add ASCII `alt` attribute "diff". |
| FX-11 | Table III (degradation level schedule) | "blur kernel" auto-formatted as float on some renderers ("13.0"). | Force integer with `\num{13}` (siunitx) and `S[table-format=2.0]` column spec. |
| FX-12 | Table VII / Table X (Phase C / C2 per-axis accuracy) | Two new PSNR / SSIM columns must not push the table off-page. | `{ltablex}` package, `\setlength{\tabcolsep}{4pt}`, `\small` body. Column spec: `l l c c c c c S[table-format=2.1] S[table-format=1.3]`. |
| FX-13 | All figure captions referencing Spearman rho | V1 captions show "rho" as literal letters. | LaTeX side `\rho`; Pandoc carries OMML glyph. Validator fails if the literal string "rho" appears in any caption without a preceding `\`. |
| FX-14 | Two-column layouts in §7.6 design | Equation (4) TransNeXt attention overflows two-column boundary. | Wrap in `\begin{equation*}\begin{aligned} ... \end{aligned}\end{equation*}` and use the `figure*` / `equation*` full-width float (`[!t]` placement) so it spans both columns. |
| FX-15 (NEW V2) | All §4.2 PSNR / SSIM per-axis figures (Figs. 8-15) | n/a - new in V2. | Build target: matplotlib `figsize=(8, 5), dpi=160`. Shared y-range across (Phase, dataset) pair. Three model markers (circle / square / triangle). Five axis line-styles (solid / dashed / dotted / dash-dot / dotted-large). Spearman rho per-axis displayed in caption metadata JSON for caption injection. |
| FX-16 (NEW V2) | Hebrew abstract environment | n/a - new in V2. | `polyglossia` hebrew environment, font fallback chain (Culmus David CLM -> Frank Ruhl CLM -> SBL Hebrew -> system default). Six labelled paragraphs (`רקע`, `מטרה`, `שיטות`, `ממצאים`, `מסקנות`, `מילות מפתח`). Visible in both PDF (XeLaTeX) and DOCX (Pandoc polyglossia). |

---

## 9. Schedule

| Day | Task | Owner |
|---|---|---|
| D0 (today) | V2 PRD approved | operator |
| D1 | Net-new analysis: per-cell join CSV, 8 PSNR / SSIM per-axis plots, ECE-vs-level, combined calibration / confusion grids | implementation agent |
| D2 | Re-render the §8 FX-NN figures (FX-01 through FX-09) | implementation agent |
| D3 | LaTeX edits: inline Hebrew abstract; insert §7.4 Phases overview; insert §7.8 Cross-model chapter; strip banned terms; foreground robustness comparison in abstract + conclusions; renumber figures and tables; expand §7.6 equations to Q10 maximal | implementation agent |
| D4 | Build + validation: page-count gate (<= 30, target ~25), Hebrew-presence gate, banned-term lint, dash / emoji gate, 80 % gate | implementation agent |
| D5 | Operator QA pass: Hebrew readability, Spearman rho values, caption rewrites, FX-NN log closed; supervisor pre-review | operator + supervisor |
| D6 | Final freeze, sign duration attestation (Q9) and IP-rights form, ship V2 DOCX | operator + supervisor |
| -> 26 / 07 / 2026 | BGU submission | operator |

---

## 10. QA gates (V2)

Existing V1 gates remain (determinism, no fabrication, no AI-disclosure dashes, TNR / 1.5 spacing / 2.5 cm margins, IEEE captions, citation pass, PSNR / SSIM coverage, multi-seed disclosure, honest negative findings, build reproducibility).

V2 adds:

11. **Per-axis curves vs PSNR / SSIM** (R2 / Q2). Eight body figures exist, each with exactly five points per axis line and three model traces.
12. **Phases overview** (R4). Body contains one "Experimental phases at a glance" sub-section; Table II has at least seven phase rows.
13. **Banned-term lint** (R3). `process_term_lint.py` exit code 0.
14. **Hebrew abstract inlined** (R5 / Q5). At least one U+0590..U+05FF character in both DOCX body text and PDF text layer. The python-docx post-step is removed.
15. **80 % framing demoted** (R1 / Q4). "80 %" / "80%" / "80 percent" appears in §7.10 Conclusions at most once; absent from §7.2 Abstract.
16. **§8 fix log closed** (R6). Every `FX-NN` entry has a matching closing note in the implementation handoff (which V1 file was replaced, which parameters were used).
17. **Cross-model chapter present** (Q1). One sub-section titled "Cross-model robustness" between Problems and Conclusions, with three model sub-sections plus a side-by-side reading sub-section.
18. **Page count** (Q6). PDF page count <= 30. Warn at > 25.
19. **Maximal equations** (Q10). §7.6 contains at least seven numbered equation blocks (composite pipeline, ResNet, DenseNet, TransNeXt, label-smoothed CE, AdamW, cosine LR), plus the three definition equations (PSNR, SSIM, ECE).
20. **Duration attestation** (Q9). Table XVI exists; columns include planned and actual GPU-h and operator-h; no money figures; signature line attests to duration.
21. **Cross-model headline numbers** (Q4 = A). The +5.0 / +7.0 / <1 pp magnitudes in §7.2 Abstract match `data/tables/per_cell_with_iq.csv` to within 0.5 pp. Validator computes and asserts.

---

## 11. Out of scope (for V2)

* No re-training.
* No new architectures.
* No real THz hardware data.
* No human-perception study.
* No deployment / packaging beyond the DOCX deliverable.
* No translation of the body to Hebrew (only the abstract is bilingual).
* No re-probe of PSNR / SSIM (V1 numbers reused).
* No re-numbering of the V1 audit-trail CSV files in `data/tables/` (LaTeX renumbers; CSVs keep V1 names so the V1 DOCX in `archive/V1/` stays reproducible).
* No money figures (Q9). All cost framing is duration-only.

---

## 12. Open questions for the operator (V2)

Down from five to two:

1. **Hebrew font.** Which Unicode Hebrew font is installed on the build host (Culmus David CLM, Frank Ruhl CLM, SBL Hebrew, or other)? The build script auto-detects via the FX-16 fallback chain; advance knowledge lets the implementation agent pre-commit the font name.
2. **TransNeXt cross-model headline number check** (Q4 = A pre-condition). Operator confirms +5.0 / +7.0 / <1 pp against `data/tables/per_cell_with_iq.csv` before D1. If any magnitude is materially off, the abstract is updated to match the as-measured number; the framing language stays neutral either way.

---

## 13. Appendix: degradation level table (single source of truth)

| Level | Name | low_res | blur kernel | blur σ | noise std | S&P | saturation |
|---|---|---|---|---|---|---|---|
| L1 | Mild | 18 | 13 | 2.50 | 0.04 | 0.03 | 0.95 |
| L2 | Light | 12 | 25 | 5.00 | 0.08 | 0.06 | 0.65 |
| L3 | Moderate | 8 | 41 | 8.00 | 0.12 | 0.10 | 0.40 |
| L4 | Severe | 6 | 61 | 12.00 | 0.16 | 0.14 | 0.15 |
| L5 | Extreme | 3 | 91 | 18.00 | 0.22 | 0.18 | 0.00 |

Reproduced from [src/data/degradation_levels.py](../src/data/degradation_levels.py); any drift in the body of the report must update both this PRD and the source file in lockstep.

---

*End of PRD V2 (refined).*
