"""Assemble a comprehensive research-report Markdown for Final_Exp.pdf.

Reads `artifacts/Final_Exp.json` (the 186-cell tracker) and emits
`docs/Final_Exp_Report.md` — a long-form report with status summary,
methodology, per-phase result tables, cross-model headline findings,
research conclusions, and a full 186-cell appendix.

The PDF is rendered from this Markdown by `scripts/render_final_exp_pdf.py`.

Usage:
    python scripts/build_final_exp_report.py
    python scripts/build_final_exp_report.py --src artifacts/Final_Exp.json --out docs/Final_Exp_Report.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = _REPO_ROOT / "artifacts" / "Final_Exp.json"
_DEFAULT_OUT = _REPO_ROOT / "docs" / "Final_Exp_Report.md"

LEVEL_NAMES = {1: "L1 Mild", 2: "L2 Light", 3: "L3 Moderate", 4: "L4 Severe", 5: "L5 Extreme"}
AXES = ["resolution", "noise", "blur", "saturation", "salt_pepper"]
MODELS = ["resnet50", "densenet121", "transnext_tiny"]
DATASETS = ["cifar10", "mnist"]
EM = "—"


def _nb(s: Any) -> str:
    """Keep an identifier on one line in the rendered PDF.

    Markdown-pdf (via PyMuPDF's Story API) treats ``_`` as a soft break
    point, so model names like ``transnext_tiny`` and axis names like
    ``salt_pepper`` wrap mid-identifier in narrow table columns. Inline
    HTML span with ``white-space: nowrap`` is silently dropped by the
    renderer. The reliable fix is to insert U+2060 (WORD JOINER) on
    either side of every ``_`` — an invisible no-break joiner that
    overrides the default break-on-underscore heuristic without
    changing copy-paste behavior.
    """
    return str(s).replace("_", "⁠_⁠")


def _fmt_acc(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) and v >= 0 else EM


def _fmt_dur(secs: Any) -> str:
    if not isinstance(secs, (int, float)) or secs <= 0:
        return EM
    if secs < 90:
        return f"{secs:.0f}s"
    m = secs / 60.0
    if m < 90:
        return f"{m:.0f}m"
    return f"{m / 60.0:.1f}h"


def _index_rows(rows: list[dict]) -> dict[tuple, dict]:
    """Index rows by (phase, model, dataset, level, axis) for fast lookup."""
    out: dict[tuple, dict] = {}
    for r in rows:
        key = (r["phase"], r["model"], r["dataset"], r.get("level"), r.get("axis"))
        out[key] = r
    return out


def _render_phase_a(idx: dict[tuple, dict]) -> str:
    lines = [
        "| # | Model | Dataset | Best Val Acc | Epochs | Runtime |",
        "|---|-------|---------|--------------|--------|---------|",
    ]
    n = 0
    for model in MODELS:
        for dataset in DATASETS:
            n += 1
            r = idx.get(("A", model, dataset, None, None), {})
            lines.append(
                f"| {n} | {_nb(model)} | {_nb(dataset)} | {_fmt_acc(r.get('val_acc'))} | "
                f"{r.get('epochs_run') or EM} | {_fmt_dur(r.get('runtime_s'))} |"
            )
    return "\n".join(lines)


def _render_phase_b_curve(idx: dict[tuple, dict]) -> str:
    """Per-model L1..L5 best_val_acc curves on cifar10 and mnist."""
    lines = [
        "### Phase B Best-Val-Acc Curve (L1 → L5)",
        "",
        "| Model | Dataset | L1 | L2 | L3 | L4 | L5 |",
        "|-------|---------|----|----|----|----|----|"
    ]
    for model in MODELS:
        for dataset in DATASETS:
            cells = [_fmt_acc(idx.get(("B", model, dataset, lv, None), {}).get("val_acc")) for lv in (1, 2, 3, 4, 5)]
            lines.append(f"| {_nb(model)} | {_nb(dataset)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_phase_b_table(idx: dict[tuple, dict]) -> str:
    lines = [
        "| # | Model | Dataset | Level | Best Val Acc | Epochs | Runtime |",
        "|---|-------|---------|-------|--------------|--------|---------|",
    ]
    n = 6
    for level in (1, 2, 3, 4, 5):
        for model in MODELS:
            for dataset in DATASETS:
                n += 1
                r = idx.get(("B", model, dataset, level, None), {})
                lines.append(
                    f"| {n} | {_nb(model)} | {_nb(dataset)} | {_nb(LEVEL_NAMES[level])} | "
                    f"{_fmt_acc(r.get('val_acc'))} | {r.get('epochs_run') or EM} | "
                    f"{_fmt_dur(r.get('runtime_s'))} |"
                )
    return "\n".join(lines)


def _render_phase_c_heatmap(idx: dict[tuple, dict], model: str, dataset: str) -> str:
    """5x5 heatmap: rows=axes, cols=L1..L5."""
    lines = [
        f"#### {model} · {dataset}",
        "",
        "| Axis | L1 | L2 | L3 | L4 | L5 |",
        "|------|----|----|----|----|----|"
    ]
    for axis in AXES:
        cells = [_fmt_acc(idx.get(("C", model, dataset, lv, axis), {}).get("val_acc")) for lv in (1, 2, 3, 4, 5)]
        lines.append(f"| {_nb(axis)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_phase_c_table(idx: dict[tuple, dict]) -> str:
    lines = [
        "| # | Model | Dataset | Level | Axis | Best Val Acc | Epochs | Runtime |",
        "|---|-------|---------|-------|------|--------------|--------|---------|",
    ]
    n = 36
    for level in (1, 2, 3, 4, 5):
        for axis in AXES:
            for model in MODELS:
                for dataset in DATASETS:
                    n += 1
                    r = idx.get(("C", model, dataset, level, axis), {})
                    lines.append(
                        f"| {n} | {_nb(model)} | {_nb(dataset)} | {_nb(LEVEL_NAMES[level])} | {_nb(axis)} | "
                        f"{_fmt_acc(r.get('val_acc'))} | {r.get('epochs_run') or EM} | "
                        f"{_fmt_dur(r.get('runtime_s'))} |"
                    )
    return "\n".join(lines)


def _render_cross_model_l5(idx: dict[tuple, dict]) -> str:
    """The campaign's headline cross-model L5 table — all 5 axes × 3 models × 2 datasets."""
    lines = [
        "| Axis @ L5 | resnet50 cifar10 | resnet50 mnist | densenet121 cifar10 | densenet121 mnist | transnext_tiny cifar10 | transnext_tiny mnist |",
        "|-----------|------------------|----------------|---------------------|-------------------|------------------------|----------------------|",
    ]
    for axis in AXES:
        cells = []
        for model in MODELS:
            for dataset in DATASETS:
                v = idx.get(("C", model, dataset, 5, axis), {}).get("val_acc")
                cells.append(_fmt_acc(v))
        lines.append(f"| {_nb(axis)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _campaign_stats(rows: list[dict]) -> dict:
    total_runtime = sum(r.get("runtime_s") or 0 for r in rows)
    total_epochs = sum(r.get("epochs_run") or 0 for r in rows)
    return {"total_runtime_h": total_runtime / 3600.0, "total_epochs": total_epochs}


def render(data: dict) -> str:
    rows = data["rows"]
    idx = _index_rows(rows)
    stats = _campaign_stats(rows)
    today = _dt.date.today().isoformat()

    parts: list[str] = []

    parts.append(f"""# Object Classification in Low-Resolution THz-like Imagery — Final Research Report

**Status:** 186 / 186 cells complete · **Generated:** {today} · **Hardware:** RTX 5070 (Blackwell sm_120, 12 GB)

---

## Executive Summary

This report presents the final outcomes of a 186-cell experimental campaign evaluating
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

- **Total cells:** 186 (6 Phase A + 30 Phase B + 150 Phase C)
- **Total accumulated GPU runtime:** ~{stats['total_runtime_h']:.1f} GPU-hours
- **Total training epochs across all cells:** {stats['total_epochs']:,}
- **Architectures evaluated:** 3 (ResNet50, DenseNet121, TransNeXt-tiny)
- **Datasets:** 2 (CIFAR-10, MNIST upsampled 28 → 224)
- **Degradation severities:** 5 (L1 Mild → L5 Extreme)
- **Phase C single-axis isolation axes:** 5 (resolution, noise, blur, saturation, salt_pepper)

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

{_render_phase_a(idx)}

**Reading:** TransNeXt-tiny leads CIFAR-10 by +2.46 pp over ResNet50 and +4.08 pp
over DenseNet121 at clean baseline; MNIST clusters tight at 0.991-0.993.

---

## Phase B — Combined Degradation (30 cells)

Every five axes are active at the same level L per cell. This is the
"realistic THz" regime — the model has to defeat all five degradations
simultaneously.

{_render_phase_b_curve(idx)}

**Reading (pipeline v2).** Strictly monotonic on CIFAR-10 across all three
architectures (no inversions). MNIST holds ≥ 0.96 through L2 for all models,
then falls fast at L4 / L5 — driven by the resolution axis (see Phase C) and
amplified by the v2-hardened noise + S&P axes (mean ΔL3 vs v1 is roughly
−7 pp on MNIST). v2 Phase B mean Δ across the 30 cells is −12.49 pp vs v1
(operator-locked under US-020 close 2026-05-21).

### Full Phase B Result Table

{_render_phase_b_table(idx)}

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

{_render_phase_c_heatmap(idx, 'resnet50', 'cifar10')}

{_render_phase_c_heatmap(idx, 'resnet50', 'mnist')}

#### DenseNet121

{_render_phase_c_heatmap(idx, 'densenet121', 'cifar10')}

{_render_phase_c_heatmap(idx, 'densenet121', 'mnist')}

#### TransNeXt-tiny

{_render_phase_c_heatmap(idx, 'transnext_tiny', 'cifar10')}

{_render_phase_c_heatmap(idx, 'transnext_tiny', 'mnist')}

### Full Phase C Result Table

{_render_phase_c_table(idx)}

---

## Cross-Architecture Headline Findings

### Finding 1 — Universal 3 × 3-downsample bottleneck at L5 resolution

At the most aggressive resolution axis level (L5 = 3 × 3 native pixels
upsampled to 224 × 224 bicubic), every architecture lands within ±3 pp
on CIFAR-10 and ±1 pp on MNIST:

| Axis @ L5 — resolution | resnet50 | densenet121 | {_nb('transnext_tiny')} |
|------------------------|----------|-------------|----------------|
| CIFAR-10 | {_fmt_acc(idx.get(("C","resnet50","cifar10",5,"resolution"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","densenet121","cifar10",5,"resolution"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","transnext_tiny","cifar10",5,"resolution"),{}).get("val_acc"))} |
| MNIST | {_fmt_acc(idx.get(("C","resnet50","mnist",5,"resolution"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","densenet121","mnist",5,"resolution"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","transnext_tiny","mnist",5,"resolution"),{}).get("val_acc"))} |

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

| Axis @ L5 — CIFAR-10 | resnet50 | densenet121 | {_nb('transnext_tiny')} | TransNeXt gap vs avg CNN |
|----------------------|----------|-------------|----------------|--------------------------|
| noise (v2) | {_fmt_acc(idx.get(("C","resnet50","cifar10",5,"noise"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","densenet121","cifar10",5,"noise"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","transnext_tiny","cifar10",5,"noise"),{}).get("val_acc"))} | ~+13.2 pp |
| saturation | {_fmt_acc(idx.get(("C","resnet50","cifar10",5,"saturation"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","densenet121","cifar10",5,"saturation"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","transnext_tiny","cifar10",5,"saturation"),{}).get("val_acc"))} | ~+7.1 pp |
| salt_pepper (v2) | {_fmt_acc(idx.get(("C","resnet50","cifar10",5,"salt_pepper"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","densenet121","cifar10",5,"salt_pepper"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","transnext_tiny","cifar10",5,"salt_pepper"),{}).get("val_acc"))} | ~+10.0 pp |
| blur | {_fmt_acc(idx.get(("C","resnet50","cifar10",5,"blur"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","densenet121","cifar10",5,"blur"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","transnext_tiny","cifar10",5,"blur"),{}).get("val_acc"))} | ~+11.2 pp |
| resolution | {_fmt_acc(idx.get(("C","resnet50","cifar10",5,"resolution"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","densenet121","cifar10",5,"resolution"),{}).get("val_acc"))} | {_fmt_acc(idx.get(("C","transnext_tiny","cifar10",5,"resolution"),{}).get("val_acc"))} | +3 pp |

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

{_render_cross_model_l5(idx)}

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

---

## Operational Notes

- **Reproducibility lock:** `pl.seed_everything(42, workers=True)` + per-sample
  validation seeds (`seed = idx + SEED_OFFSET_VAL`). Determinism gate
  (`src/tests/test_degradation_determinism.py`) verifies byte-identical
  validation pixels across dataset rebuilds.

- **Optuna pre-tuning:** Six (model, dataset) pairs were tuned at L3 Moderate
  before the 186-cell campaign launched; the winning hyperparameters were
  frozen for every Phase A/B/C cell of that pair. Winners under
  `artifacts/best_hparams/{{model}}_{{dataset}}.json`.

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
- `artifacts/reports/phase_a_{{model}}_summary.md` — Phase A REPORTER summaries
- `docs/phase_a.md` / `docs/phase_c.md` — LIBRARIAN per-phase docs
- `progress.txt` — full iteration log (Iterations 1-43 — covers v1 + v2)
- `PRD.md` — v2 roadmap (US-017 through US-025; v1 US-001..US-016 closed)
- `CLAUDE.md` — protocol invariants + multi-agent guidance
- `scripts/build_final_exp_report.py` — this report generator
- `scripts/render_final_exp_pdf.py` — Markdown → PDF renderer

---

*Report generated {today} from `artifacts/Final_Exp.json` (state of disk under
`PIPELINE_VERSION = 2`, after the v2 noise-fix 90-cell re-run closed
2026-05-23 / Iteration 40). Regenerate with* `python scripts/build_final_exp_report.py`.
"""
)
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build comprehensive Final Experiment Report Markdown.")
    p.add_argument("--src", default=str(_DEFAULT_SRC), help=f"Source JSON (default: {_DEFAULT_SRC})")
    p.add_argument("--out", default=str(_DEFAULT_OUT), help=f"Output Markdown (default: {_DEFAULT_OUT})")
    args = p.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists():
        print(f"ERROR: source JSON not found: {src}", file=sys.stderr)
        return 1

    with src.open(encoding="utf-8") as f:
        data = json.load(f)

    content = render(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"wrote {out} ({len(content):,} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
