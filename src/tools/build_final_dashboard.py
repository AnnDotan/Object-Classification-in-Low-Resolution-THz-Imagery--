"""Build artifacts/Final_Exp.html — the 186-cell campaign dashboard (US-015).

Static HTML (no server, no JS framework). Three phase sections (Phase A / B / C),
each rendered as a <table> with one <tr> per cell and a follow-up detail row
expanded via <details>. Replaces the legacy tile-grid (US-011) and the separate
Phase A operator view (US-006); they were deleted in US-015 — single canonical
artifact at artifacts/Final_Exp.html.

Table columns (in order):
    ID | Model | Level | Parameters | PSNR/SSIM | Accuracy | Visuals

Per-row detail panel (HTML <details>):
    - Side-by-side Original|Degraded thumbnail (PNG from US-010 thumbs).
    - Click-to-expand learning curves (PNG from RunVisualizer).
    - Hyperparameters dump (from metrics.json["hparams"]).
    - Gate verdict (Phase A only): band, decision, reasons, manual_override note.

CLI:
    python -m src.tools.build_final_dashboard
    python -m src.tools.build_final_dashboard --out artifacts/Final_Exp.html
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Optional

from src.experiments.cells import EXPECTED_COUNTS, EXPECTED_TOTAL
from src.experiments.matrix import build_final_matrix
from src.experiments.run_status import detect_status, read_image_quality, read_metrics


_DEFAULT_OUT = Path("artifacts/Final_Exp.html")
_RUNS_ROOT = Path("runs/final")
_THUMBS_DIR = Path("artifacts/dashboard_thumbs")

# Levels reused by the Parameters column formatter.
_LEVEL_NAMES: dict[int, str] = {
    1: "Mild", 2: "Light", 3: "Moderate", 4: "Severe", 5: "Extreme",
}


# CSS lives inline so the file works directly via file:// without a server.
_CSS = """
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
       margin: 0; padding: 1.5rem; background: #0f1115; color: #e4e6ea; }
h1 { margin: 0 0 0.4rem; font-size: 1.4rem; }
.subtitle { color: #9aa3b2; margin-bottom: 1.2rem; font-size: 0.9rem; max-width: 70ch; }
.phase-section { margin-bottom: 1.5rem; background: #181b22; border-radius: 8px;
                  padding: 0.7rem 1rem 0.9rem; }
.phase-section > h2 { margin: 0.2rem 0 0.6rem; font-size: 1rem;
                       display: flex; justify-content: space-between; align-items: baseline; }
.phase-section > h2 .counts { font-size: 0.78rem; color: #9aa3b2; font-weight: normal; }
table.matrix { border-collapse: collapse; width: 100%; font-size: 0.82rem;
               background: #181b22; }
table.matrix th, table.matrix td { padding: 0.45rem 0.6rem;
                                    border-bottom: 1px solid #2a3142;
                                    text-align: left; vertical-align: top; }
table.matrix th { background: #1e2330; color: #cdd3df; font-weight: 600;
                   text-transform: uppercase; font-size: 0.68rem;
                   letter-spacing: 0.04em; position: sticky; top: 0; }
table.matrix td.id-cell { font-family: ui-monospace, Menlo, monospace;
                           color: #6b7280; width: 3rem; }
table.matrix td.params { font-family: ui-monospace, Menlo, monospace;
                          font-size: 0.72rem; color: #9aa3b2;
                          max-width: 22rem; word-break: break-word; }
table.matrix td.psnr-ssim { font-family: ui-monospace, Menlo, monospace;
                             font-size: 0.72rem; color: #cdd3df; }
table.matrix td.accuracy { font-family: ui-monospace, Menlo, monospace;
                            color: #a5f3a8; white-space: nowrap; }
table.matrix tr.detail-row td { padding: 0; background: #14181f;
                                 border-bottom: 2px solid #2a3142; }
table.matrix tr.detail-row > td > details { padding: 0.7rem 0.9rem; }
table.matrix tr.detail-row summary { cursor: pointer; color: #9aa3b2;
                                      font-size: 0.75rem; padding: 0.2rem 0;
                                      list-style: none; }
table.matrix tr.detail-row summary::-webkit-details-marker { display: none; }
.tag { font-family: ui-monospace, Menlo, monospace; color: #cdd3df;
       font-size: 0.72rem; }
.thumb { display: block; width: 11rem; height: auto; image-rendering: pixelated;
         border-radius: 3px; background: #0a0c10; border: 1px solid #2a3142; }
.thumb-caption { display: flex; justify-content: space-between;
                 font-size: 0.62rem; color: #8a93a3; padding: 0 2px;
                 margin: 0.2rem 0 0; letter-spacing: 0.04em;
                 text-transform: uppercase; width: 11rem; }
.thumb-caption span { flex: 1; text-align: center; }
.thumb-caption span + span { border-left: 1px solid #2a3142; }
.thumb-placeholder { width: 11rem; height: 5rem; border: 1px dashed #2a3142;
                      border-radius: 3px; display: flex; align-items: center;
                      justify-content: center; color: #5a6478; font-size: 0.7rem; }

.badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
         font-size: 0.66rem; font-weight: 600; vertical-align: middle; }
.badge.complete { background: #14532d; color: #bbf7d0; }
.badge.running  { background: #422006; color: #fed7aa; }
.badge.pending  { background: #1f2937; color: #9ca3af; }
.badge.failed   { background: #7f1d1d; color: #fecaca; }
.band { display: inline-block; padding: 1px 6px; border-radius: 3px;
        font-size: 0.66rem; font-weight: 600; margin-left: 0.3rem; }
.band.green   { background: #14532d; color: #bbf7d0; }
.band.yellow  { background: #422006; color: #fde68a; }
.band.red     { background: #7f1d1d; color: #fecaca; }
.decision { font-size: 0.66rem; font-weight: 600; }
.decision.continue { color: #a5f3a8; }
.decision.halt     { color: #fca5a5; }
.decision.pending  { color: #9ca3af; }

.detail-grid { display: grid; gap: 0.8rem; margin-top: 0.6rem;
               grid-template-columns: 1.2fr 1fr 1fr; }
.detail-grid h4 { margin: 0 0 0.35rem; font-size: 0.72rem;
                  color: #cdd3df; text-transform: uppercase; letter-spacing: 0.03em; }
.detail-grid img { width: 100%; height: auto; border-radius: 3px;
                   background: #0a0c10; border: 1px solid #2a3142; }
.detail-grid pre { margin: 0; padding: 0.5rem 0.6rem; background: #0a0c10;
                   border: 1px solid #2a3142; border-radius: 3px;
                   font-size: 0.7rem; overflow-x: auto; max-height: 280px; }
.detail-grid ul { margin: 0; padding-left: 1.1rem; font-size: 0.72rem; }
.detail-grid ul li { margin-bottom: 0.18rem; color: #fde68a; }
.detail-placeholder { padding: 0.4rem 0.6rem; font-size: 0.7rem;
                      color: #5a6478; border: 1px dashed #2a3142;
                      border-radius: 3px; text-align: center; }
.curve-toggle { background: transparent; border: 1px solid #2a3142;
                border-radius: 3px; padding: 0; margin-top: 0.4rem; }
.curve-toggle > summary { font-size: 0.7rem; padding: 0.3rem 0.5rem;
                           color: #cdd3df; cursor: pointer; list-style: none; }
.curve-toggle > summary::-webkit-details-marker { display: none; }
.curve-toggle img { display: block; width: 100%; height: auto;
                    border-top: 1px solid #2a3142; image-rendering: auto; }
.manual-override { margin-top: 0.4rem; padding: 0.4rem 0.6rem;
                    background: #1e2330; border-left: 3px solid #fde68a;
                    border-radius: 0 3px 3px 0; font-size: 0.7rem;
                    color: #fde68a; }
.manual-override strong { color: #fbbf24; }
"""


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def _read_gate_verdict(tag: str, runs_root: Path) -> Optional[dict]:
    path = runs_root / tag / "gate_verdict.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_train_acc_at_best(tag: str, runs_root: Path,
                             best_epoch: Optional[int]) -> Optional[float]:
    if best_epoch is None:
        return None
    csv_path = runs_root / tag / "metrics.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    if int(row["epoch"]) == int(best_epoch):
                        return float(row["train_acc"])
                except (ValueError, KeyError):
                    continue
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# Cell formatters
# ---------------------------------------------------------------------------


def _format_parameters_for_phase(spec) -> str:
    """Human-readable degradation parameters for the Parameters column."""
    deg = spec.degrade_config
    if spec.phase == "A":
        return "clean (no degradation)"

    base = (
        f"low_res={deg.low_res}, "
        f"blur {deg.blur_kernel}x{deg.blur_sigma:.2f}, "
        f"noise={deg.gaussian_noise_std:.2f}, "
        f"S&P={deg.salt_pepper_amount:.2f}, "
        f"sat={deg.saturation:.2f}"
    )
    if spec.phase == "B":
        return base
    # Phase C: highlight which axis is at level L; the rest are at L1 mild.
    return f"axis={spec.axis}@L{spec.level}; others=L1 (mild) — {base}"


def _format_level(spec) -> str:
    if spec.phase == "A":
        return "clean"
    name = _LEVEL_NAMES.get(spec.level or 0, "?")
    if spec.phase == "B":
        return f"L{spec.level} {name}"
    return f"L{spec.level} {name} / {spec.axis}"


def _format_psnr_ssim(quality: Optional[dict]) -> str:
    if quality is None:
        return "&mdash;"
    psnr_mean = quality.get("psnr_mean")
    psnr_std = quality.get("psnr_std")
    ssim_mean = quality.get("ssim_mean")
    ssim_std = quality.get("ssim_std")
    if not isinstance(psnr_mean, (int, float)) or not isinstance(ssim_mean, (int, float)):
        return "&mdash;"
    psnr = (
        f"{psnr_mean:.2f}±{psnr_std:.2f}" if isinstance(psnr_std, (int, float))
        else f"{psnr_mean:.2f}"
    )
    ssim = (
        f"{ssim_mean:.4f}±{ssim_std:.4f}" if isinstance(ssim_std, (int, float))
        else f"{ssim_mean:.4f}"
    )
    return f"PSNR {psnr}<br>SSIM {ssim}"


def _format_accuracy(spec, val_acc: Optional[float], verdict: Optional[dict]) -> str:
    """Numeric accuracy + Phase A green/yellow/red band badge."""
    if val_acc is None:
        return "&mdash;"
    base = f'<span class="acc">{val_acc:.4f}</span>'
    if spec.phase == "A" and verdict is not None and verdict.get("gate_band"):
        band = verdict["gate_band"]
        cls = band.lower()
        base += f' <span class="band {cls}">{html.escape(band.upper())}</span>'
    return base


def _read_best_val_acc(tag: str, runs_root: Path) -> Optional[float]:
    metrics = read_metrics(tag, runs_root)
    if metrics is None:
        return None
    for k in ("best_val_acc", "final_val_acc", "last_val_acc"):
        v = metrics.get(k)
        if isinstance(v, (int, float)) and v >= 0.0:
            return float(v)
    return None


# ---------------------------------------------------------------------------
# Visuals + detail row helpers
# ---------------------------------------------------------------------------


def _thumb_html(tag: str, thumbs_rel: Path, thumbs_root: Path) -> str:
    thumb_file = thumbs_root / f"{tag}.png"
    if not thumb_file.exists():
        return '<div class="thumb-placeholder">no thumb</div>'
    rel = (thumbs_rel / f"{tag}.png").as_posix()
    return (
        f'<img class="thumb" src="{html.escape(rel)}" alt="{html.escape(tag)}" loading="lazy">'
        '<div class="thumb-caption"><span>Clean</span><span>Degraded</span></div>'
    )


def _curve_html(tag: str, thumbs_rel: Path, thumbs_root: Path) -> str:
    curve_file = thumbs_root / f"{tag}_curve.png"
    if curve_file.exists():
        rel = (thumbs_rel / f"{tag}_curve.png").as_posix()
        return (
            '<details class="curve-toggle" open>'
            '<summary>Learning curves (val acc / loss vs. epoch)</summary>'
            f'<img src="{html.escape(rel)}" alt="{html.escape(tag)} curves" loading="lazy">'
            '</details>'
        )
    return '<div class="detail-placeholder">no learning curves yet</div>'


def _hparams_html(metrics: Optional[dict]) -> str:
    if metrics is None:
        return '<div class="detail-placeholder">no metrics.json yet</div>'
    hp = metrics.get("hparams")
    if not isinstance(hp, dict) or not hp:
        return '<div class="detail-placeholder">no hparams in metrics.json</div>'
    src = metrics.get("hparams_source") or {}
    return f"<pre>{html.escape(json.dumps({'best_params': hp, 'source': src}, indent=2))}</pre>"


def _verdict_html(verdict: Optional[dict]) -> str:
    if verdict is None:
        return '<div class="detail-placeholder">cell not run yet</div>'
    parts: list[str] = []

    decision = verdict.get("decision") or "pending"
    band = verdict.get("gate_band") or "n/a"
    parts.append(
        f'<div style="margin-bottom:0.4rem;"><strong>Decision:</strong> '
        f'<span class="decision {decision}">{html.escape(decision.upper())}</span> · '
        f'<strong>Band:</strong> <span class="band {band.lower()}">{html.escape(band.upper())}</span></div>'
    )

    explanation = verdict.get("gate_explanation") or ""
    if explanation:
        parts.append(
            f'<div style="font-size:0.7rem;color:#9aa3b2;margin-bottom:0.4rem;">'
            f'{html.escape(explanation)}</div>'
        )

    reasons = verdict.get("reasons") or []
    if reasons:
        items = "".join(f"<li>{html.escape(str(r))}</li>" for r in reasons)
        parts.append(f"<ul>{items}</ul>")
    else:
        parts.append('<div class="detail-placeholder">no halt reasons</div>')

    override = verdict.get("manual_override")
    if isinstance(override, dict):
        parts.append(
            '<div class="manual-override">'
            f'<strong>MANUAL OVERRIDE</strong> ({html.escape(override.get("date", ""))} · '
            f'{html.escape(override.get("operator", ""))}). '
            f'Previous decision: {html.escape(override.get("previous_decision", ""))}.<br>'
            f'{html.escape(override.get("reason", ""))}'
            '</div>'
        )
    return "".join(parts)


def _row_html(idx: int, spec, runs_root: Path,
              thumbs_rel: Path, thumbs_root: Path) -> str:
    metrics = read_metrics(spec.tag, runs_root)
    quality = read_image_quality(spec.tag, runs_root)
    verdict = _read_gate_verdict(spec.tag, runs_root)
    val_acc = _read_best_val_acc(spec.tag, runs_root)
    status = detect_status(spec.tag, runs_root, metrics)

    summary_row = f"""
<tr data-tag="{html.escape(spec.tag)}">
  <td class="id-cell">{idx}</td>
  <td>{html.escape(spec.model)}<br><span style="font-size:0.65rem;color:#6b7280;">{html.escape(spec.dataset)}</span></td>
  <td>{html.escape(_format_level(spec))}</td>
  <td class="params">{html.escape(_format_parameters_for_phase(spec))}</td>
  <td class="psnr-ssim">{_format_psnr_ssim(quality)}</td>
  <td class="accuracy">{_format_accuracy(spec, val_acc, verdict)}<br><span class="badge {status.lower()}">{html.escape(status)}</span></td>
  <td>{_thumb_html(spec.tag, thumbs_rel, thumbs_root)}</td>
</tr>
""".strip()

    detail_row = f"""
<tr class="detail-row"><td colspan="7">
  <details>
    <summary><span class="tag">{html.escape(spec.tag)}</span> &mdash; click for curves, hparams, gate verdict</summary>
    <div class="detail-grid">
      <div>
        <h4>Learning curves</h4>
        {_curve_html(spec.tag, thumbs_rel, thumbs_root)}
      </div>
      <div>
        <h4>Hyperparameters</h4>
        {_hparams_html(metrics)}
      </div>
      <div>
        <h4>Gate verdict</h4>
        {_verdict_html(verdict) if spec.phase == 'A' else '<div class="detail-placeholder">gate verdict only tracked for Phase A</div>'}
      </div>
    </div>
  </details>
</td></tr>
""".strip()

    return summary_row + "\n" + detail_row


# ---------------------------------------------------------------------------
# Phase section + dashboard composition
# ---------------------------------------------------------------------------


_PHASE_TITLES = {
    "A": "Phase A — Clean Baselines",
    "B": "Phase B — Combined Degradation",
    "C": "Phase C — Single-Axis Isolation",
}


def _phase_section_html(phase: str, cells: list, runs_root: Path,
                         thumbs_rel: Path, thumbs_root: Path,
                         start_idx: int) -> str:
    counts = {"Pending": 0, "Running": 0, "Complete": 0, "Failed": 0}
    rows: list[str] = []
    for offset, spec in enumerate(cells):
        idx = start_idx + offset
        m = read_metrics(spec.tag, runs_root)
        counts[detect_status(spec.tag, runs_root, m)] += 1
        rows.append(_row_html(idx, spec, runs_root, thumbs_rel, thumbs_root))

    counts_str = (
        f"{counts['Complete']} complete &middot; {counts['Running']} running &middot; "
        f"{counts['Pending']} pending &middot; {counts['Failed']} failed"
    )
    title = f"{_PHASE_TITLES[phase]} ({len(cells)} cells)"

    return f"""
<div class="phase-section" id="phase-{phase.lower()}">
  <h2>{html.escape(title)} <span class="counts">{counts_str}</span></h2>
  <table class="matrix">
    <thead>
      <tr>
        <th>ID</th><th>Model</th><th>Level</th><th>Parameters</th>
        <th>PSNR / SSIM</th><th>Accuracy</th><th>Visuals</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</div>
""".strip()


def build_dashboard(
    out_path: Path = _DEFAULT_OUT,
    thumbs_dir: Path = _THUMBS_DIR,
    runs_root: Path = _RUNS_ROOT,
) -> dict:
    """Render the Gold Standard dashboard. Returns a row-count summary."""
    by_phase: dict[str, list] = {"A": [], "B": [], "C": []}
    for c in build_final_matrix(out_size=224):
        by_phase[c.phase].append(c)
    assert sum(len(v) for v in by_phase.values()) == EXPECTED_TOTAL

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        thumbs_rel = Path(thumbs_dir).resolve().relative_to(out_path.parent.resolve())
    except ValueError:
        thumbs_rel = Path(thumbs_dir).resolve()
    thumbs_root = Path(thumbs_dir).resolve()

    section_a = _phase_section_html(
        "A", by_phase["A"], runs_root, thumbs_rel, thumbs_root, 1
    )
    b_start = 1 + EXPECTED_COUNTS["A"]
    section_b = _phase_section_html(
        "B", by_phase["B"], runs_root, thumbs_rel, thumbs_root, b_start
    )
    c_start = b_start + EXPECTED_COUNTS["B"]
    section_c = _phase_section_html(
        "C", by_phase["C"], runs_root, thumbs_rel, thumbs_root, c_start
    )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>THz Final Experiment Dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<h1>THz Final Experiment Dashboard</h1>
<div class="subtitle">
  186 cells &middot; 3 architectures (ResNet50, DenseNet121, TransNeXt) &middot;
  2 datasets (CIFAR-10, MNIST). Each row's Visuals cell holds the Original|Degraded
  thumbnail; click the row's expand control for learning curves, hyperparameters,
  and (Phase A) gate-verdict reasoning.
</div>
{section_a}
{section_b}
{section_c}
</body>
</html>
"""
    out_path.write_text(body, encoding="utf-8")

    return {
        "rows": EXPECTED_TOTAL,
        "phase_a": len(by_phase["A"]),
        "phase_b": len(by_phase["B"]),
        "phase_c": len(by_phase["C"]),
        "out": str(out_path),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build artifacts/Final_Exp.html (Gold Standard).")
    p.add_argument("--out", default=str(_DEFAULT_OUT),
                   help=f"Output HTML path (default: {_DEFAULT_OUT}).")
    p.add_argument("--thumbs-dir", default=str(_THUMBS_DIR),
                   help=f"Thumb directory used by <img> srcs (default: {_THUMBS_DIR}).")
    p.add_argument("--runs-root", default=str(_RUNS_ROOT),
                   help=f"Root with per-cell metrics + verdicts (default: {_RUNS_ROOT}).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    # Aggregate per-run metrics into artifacts/Final_Exp.json before rendering
    # the HTML; the new browser dashboard (US-004..) consumes this JSON via
    # fetch + 30 s polling. Failure here must not block HTML render.
    from src.tools.build_final_exp_json import build_final_exp_json  # lazy import
    try:
        json_summary = build_final_exp_json(runs_root=Path(args.runs_root))
        print(
            f"final_exp_json: {json_summary['rows']} rows -> {json_summary['out']}"
        )
    except Exception as e:  # noqa: BLE001 — keep HTML render alive on aggregator failure
        print(f"[warn] final_exp_json aggregator failed: {type(e).__name__}: {e}", file=sys.stderr)

    summary = build_dashboard(
        out_path=Path(args.out),
        thumbs_dir=Path(args.thumbs_dir),
        runs_root=Path(args.runs_root),
    )
    print(
        f"dashboard: {summary['rows']} rows "
        f"(A={summary['phase_a']}, B={summary['phase_b']}, C={summary['phase_c']}) "
        f"-> {summary['out']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
