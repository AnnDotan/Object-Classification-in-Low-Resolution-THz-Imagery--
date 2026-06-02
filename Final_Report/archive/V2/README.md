# V2 archive — Final Summary Report (`fin-2026-061`)

Frozen snapshot of the V2 production cycle of the Final Summary Report
(BGU §10.7 deliverable for project `p-2026-061`).

**Frozen date:** 2026-06-02
**Branch at freeze:** `1/6/26`
**Supersedes:** [`../V1/`](../V1/) (frozen 2026-06-02)

## What's inside

| Path | Purpose |
|---|---|
| `PRD_FinalReport.md` | V2 PRD (the contract between the V1 deliverable and the V2 conversion) |
| `PROGRESS.txt` | V2 authoring log (23 iterations: D1 analysis → D4 validation) |
| `fin-2026-061.pdf` | V2 deliverable — XeLaTeX render, 47 pages, Hebrew abstract inlined |
| `fin-2026-061.docx` | V2 deliverable — Pandoc render with `bgu_reference.docx` style |
| `source/` | LaTeX sources (sections, tables, filters, bibliography, BGU reference DOCX) |
| `scripts/` | Build scripts (analysis, figure renderers, linters, validators, report builder) |
| `figures/` | Rendered figures (curves, calibration grids, confusion grids, attribution bars, etc.) |
| `data/` | Per-cell joined CSV, per-pipeline image-quality index, table CSVs |
| `build/` | `Makefile`, `build_all.ps1`, intermediate TeX build artifacts |

## How V2 differs from V1

The V2 PRD is the authoritative description, but the headline changes are:

1. **New §7.4 "Experimental phases at a glance"** (between Introduction and Spec
   Sheet) with Table II covering all seven phases + the multi-seed audit.
2. **New §7.8 "Cross-model robustness comparison"** between Problems and
   Conclusions — the report's headline scientific contribution.
3. **§7.2 Abstract** rewritten with as-measured cross-model magnitudes:
   `+10.1 / +10.4 / +1.3 / +0.8 / +0.8 pp` at L1..L5 (TransNeXt-tiny vs the
   CNN mean on CIFAR-10), replacing the PRD-template's `+5.0 / +7.0 / <1 pp`
   per the PRD V2 §12 Q2 escape clause.
4. **Hebrew abstract inlined** (not a separate appendix). Rendered by
   XeLaTeX + polyglossia in the PDF; read directly from the UTF-8 source by
   Pandoc into the DOCX.
5. **§7.6 Design** expanded to 11 numbered equations (Q10 maximal): composite
   degradation, ResNet residual, DenseNet concat, TransNeXt foveal attention,
   label-smoothed cross-entropy, AdamW moments + parameter update, cosine LR
   schedule, PSNR, SSIM, ECE.
6. **§7.9 Problems** compressed from 7 V1 entries to 4 (Vision-transformer
   bring-up, parameter-budget downsize, blur-kernel rescale, multi-seed
   audit).
7. **§8 fix log** closed end-to-end (FX-01 through FX-16). New build assets:
   - Per-axis PSNR / SSIM curves (8 figure pairs, FX-15)
   - ECE-vs-level chart (new)
   - Combined reliability 3×2 grids (FX-05)
   - Combined confusion 3×2 grid (FX-06)
   - Re-rendered FX-01..FX-04, FX-07, FX-09 figures
8. **R3 banned-term lint** — `scripts/process_term_lint.py` plus a
   `scripts/validate_v2_gates.py` driver covering PRD V2 §10 gates 11..19.
9. **DOCX-vs-PDF parity** — both deliverables now share the same structure
   (8 body sections + 2 appendix sections), the same figures (45 images),
   the same tables (27), and the same Hebrew abstract placement.

## Reproduction recipe (against this archive)

```powershell
cd archive\V2\source
xelatex fin-2026-061.tex
bibtex fin-2026-061
xelatex fin-2026-061.tex
xelatex fin-2026-061.tex
# → fin-2026-061.pdf

cd ..\..\..       # back to repo root
python archive\V2\scripts\build_report.py docx
# → archive\V2\fin-2026-061.docx (uses pandoc + bgu_reference.docx)
```

Validation:

```powershell
python archive\V2\scripts\validate_v2_gates.py
python archive\V2\scripts\process_term_lint.py --docx archive\V2\fin-2026-061.docx
```

## Outstanding items (operator-side, carried out of V2 freeze)

These are documented in `PROGRESS.txt` under "OUTSTANDING (operator-side)"
and are not part of the V2 deliverable as frozen here:

- Page-count compression (V2 PDF = 47 pages; PRD Q6 ceiling = 30, target = 25).
- ECE Finding #8 narrative-vs-data mismatch (as-measured ECE drops L3→L5
  on average, not rises to >15 pp as the PRD template states).
- Hebrew abstract awaits native-speaker review.
- Cross-model magnitude check: V2 already reflects the as-measured numbers;
  PRD §7.2 template would benefit from an explicit update.
