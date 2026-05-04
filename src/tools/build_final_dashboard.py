"""Build artifacts/Final_Exp.html — the 186-cell campaign dashboard (US-011).

Static HTML (no server, no JS framework). Three collapsible sections
(Phase A / B / C), each containing a grid of 1 tile per cell:
  - cell tag, model, dataset, level, axis (Phase C only)
  - thumbnail PNG (Original vs Degraded, pre-rendered by US-010)
  - status badge (Pending / Running / Complete / Failed)
  - final val_acc if complete
  - iframe to the W&B run page if `wandb_run_id` is present in
    runs/final/<tag>/metrics.json; otherwise a placeholder card

CLI:
    python -m src.tools.build_final_dashboard
    python -m src.tools.build_final_dashboard --out artifacts/Final_Exp.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Optional

from src.experiments.matrix import EXPECTED_TOTAL, build_final_matrix


_DEFAULT_OUT = Path("artifacts/Final_Exp.html")
_RUNS_ROOT = Path("runs/final")
_THUMBS_DIR = Path("artifacts/dashboard_thumbs")


# CSS lives inline so the file works directly via file:// without a server.
_CSS = """
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
       margin: 0; padding: 1.5rem; background: #0f1115; color: #e4e6ea; }
h1 { margin: 0 0 0.5rem; font-size: 1.4rem; }
.subtitle { color: #9aa3b2; margin-bottom: 1.2rem; font-size: 0.9rem; }
details { margin-bottom: 1rem; background: #181b22; border-radius: 8px; padding: 0.6rem 1rem; }
details > summary { cursor: pointer; font-weight: 600; padding: 0.3rem 0;
                    list-style: none; display: flex; justify-content: space-between; }
details > summary::-webkit-details-marker { display: none; }
.grid { display: grid; gap: 0.75rem;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        margin-top: 0.6rem; }
.tile { background: #1e2330; border: 1px solid #2a3142; border-radius: 6px;
        padding: 0.5rem; font-size: 0.78rem; }
.tile h3 { margin: 0 0 0.25rem; font-size: 0.78rem; font-family: ui-monospace, Menlo, monospace;
           color: #cdd3df; word-break: break-all; }
.tile .meta { color: #8a93a3; font-size: 0.72rem; margin-bottom: 0.4rem; }
.tile img { width: 100%; height: auto; image-rendering: pixelated;
            border-radius: 3px; background: #0a0c10; }
.tile .row { display: flex; justify-content: space-between; align-items: center;
             margin-top: 0.3rem; }
.badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
         font-size: 0.7rem; font-weight: 600; }
.badge.complete { background: #14532d; color: #bbf7d0; }
.badge.running  { background: #422006; color: #fed7aa; }
.badge.pending  { background: #1f2937; color: #9ca3af; }
.badge.failed   { background: #7f1d1d; color: #fecaca; }
.acc { font-family: ui-monospace, Menlo, monospace; color: #a5f3a8; }
.iframe-wrap { margin-top: 0.4rem; }
.iframe-wrap iframe { width: 100%; height: 200px; border: 0; border-radius: 3px;
                      background: #0a0c10; }
.iframe-placeholder { height: 200px; display: flex; align-items: center;
                      justify-content: center; color: #5a6478;
                      background: #14181f; border: 1px dashed #2a3142;
                      border-radius: 3px; font-size: 0.72rem; margin-top: 0.4rem; }
.summary-counts { color: #9aa3b2; font-weight: normal; font-size: 0.85rem; }
"""


def _read_metrics(tag: str) -> Optional[dict]:
    path = _RUNS_ROOT / tag / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _detect_status(tag: str, metrics: Optional[dict]) -> str:
    """Pending | Running | Complete | Failed.

    Failed = a log.txt exists in the run dir and contains a Python traceback
    or '[ERROR]' marker but no completed metrics.
    """
    run_dir = _RUNS_ROOT / tag
    if metrics is not None:
        for key in ("final_val_acc", "best_val_acc", "last_val_acc"):
            v = metrics.get(key)
            if isinstance(v, (int, float)) and v >= 0.0:
                return "Complete"

    if run_dir.exists():
        log = run_dir / "log.txt"
        if log.exists():
            try:
                tail = log.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                tail = ""
            if "traceback" in tail or "[error]" in tail:
                return "Failed"
        return "Running"
    return "Pending"


def _wb_iframe_html(metrics: Optional[dict]) -> str:
    """Build the iframe HTML or a placeholder card."""
    if metrics is None:
        return '<div class="iframe-placeholder">No W&amp;B run yet</div>'
    run_id = metrics.get("wandb_run_id")
    entity = metrics.get("wandb_entity")
    project = metrics.get("wandb_project")
    if not (run_id and entity and project):
        return '<div class="iframe-placeholder">W&amp;B run id not captured</div>'
    src = f"https://wandb.ai/{html.escape(entity)}/{html.escape(project)}/runs/{html.escape(run_id)}"
    return f'<div class="iframe-wrap"><iframe src="{src}" loading="lazy" title="{html.escape(run_id)}"></iframe></div>'


def _thumb_html(tag: str, thumbs_dir: Path) -> str:
    rel = (thumbs_dir / f"{tag}.png").as_posix()
    return f'<img src="{html.escape(rel)}" alt="{html.escape(tag)}" loading="lazy">'


def _val_acc_str(metrics: Optional[dict]) -> str:
    if metrics is None:
        return ""
    for key in ("final_val_acc", "best_val_acc"):
        v = metrics.get(key)
        if isinstance(v, (int, float)) and v >= 0.0:
            return f'<span class="acc">val_acc={v:.4f}</span>'
    return ""


def _tile_html(spec, thumbs_dir: Path) -> str:
    metrics = _read_metrics(spec.tag)
    status = _detect_status(spec.tag, metrics)
    badge_cls = status.lower()
    meta_parts = [f"model={spec.model}", f"dataset={spec.dataset}"]
    if spec.level is not None:
        meta_parts.append(f"L{spec.level}")
    if spec.axis is not None:
        meta_parts.append(f"axis={spec.axis}")

    return f"""
<div class="tile" data-phase="{spec.phase}" data-status="{status}">
  <h3>{html.escape(spec.tag)}</h3>
  <div class="meta">{html.escape(' · '.join(meta_parts))}</div>
  {_thumb_html(spec.tag, thumbs_dir)}
  <div class="row">
    <span class="badge {badge_cls}">{status}</span>
    {_val_acc_str(metrics)}
  </div>
  {_wb_iframe_html(metrics)}
</div>
""".strip()


def _section_html(phase: str, cells: list, thumbs_dir: Path) -> str:
    counts = {"Pending": 0, "Running": 0, "Complete": 0, "Failed": 0}
    tiles_html = []
    for spec in cells:
        m = _read_metrics(spec.tag)
        counts[_detect_status(spec.tag, m)] += 1
        tiles_html.append(_tile_html(spec, thumbs_dir))

    summary_counts = (
        f"{counts['Complete']} complete · {counts['Running']} running · "
        f"{counts['Pending']} pending · {counts['Failed']} failed"
    )
    title = {
        "A": f"Phase A — clean baselines ({len(cells)})",
        "B": f"Phase B — combined degradation ({len(cells)})",
        "C": f"Phase C — single-axis isolation ({len(cells)})",
    }[phase]

    return f"""
<details {'open' if phase == 'A' else ''}>
  <summary>
    <span>{html.escape(title)}</span>
    <span class="summary-counts">{html.escape(summary_counts)}</span>
  </summary>
  <div class="grid">
    {''.join(tiles_html)}
  </div>
</details>
""".strip()


def build_dashboard(
    out_path: Path = _DEFAULT_OUT,
    thumbs_dir: Path = _THUMBS_DIR,
) -> dict:
    """Render the 186-cell dashboard. Returns a tile-count summary."""
    matrix = build_final_matrix()
    by_phase = {"A": [], "B": [], "C": []}
    for c in matrix:
        by_phase[c.phase].append(c)
    assert sum(len(v) for v in by_phase.values()) == EXPECTED_TOTAL

    # Resolve the thumbs dir relative to the dashboard file so img src
    # works under file:// without baking absolute paths.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        thumbs_rel = Path(thumbs_dir).resolve().relative_to(
            out_path.parent.resolve()
        )
    except ValueError:
        # thumbs_dir is not under out_path's parent — fall back to absolute
        thumbs_rel = Path(thumbs_dir).resolve()

    sections = [_section_html(ph, by_phase[ph], thumbs_rel) for ph in ("A", "B", "C")]

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
  186 cells · 3 architectures (ResNet50, DenseNet121, TransNeXt-Base) ·
  2 datasets (CIFAR-10, MNIST). Live W&amp;B iframes render where a run id is captured.
</div>
{''.join(sections)}
</body>
</html>
"""
    out_path.write_text(body, encoding="utf-8")

    return {
        "tiles": EXPECTED_TOTAL,
        "phase_a": len(by_phase["A"]),
        "phase_b": len(by_phase["B"]),
        "phase_c": len(by_phase["C"]),
        "out": str(out_path),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build artifacts/Final_Exp.html.")
    p.add_argument("--out", default=str(_DEFAULT_OUT),
                   help=f"Output HTML path (default: {_DEFAULT_OUT}).")
    p.add_argument("--thumbs-dir", default=str(_THUMBS_DIR),
                   help=f"Thumb directory used by <img> srcs (default: {_THUMBS_DIR}).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    summary = build_dashboard(
        out_path=Path(args.out),
        thumbs_dir=Path(args.thumbs_dir),
    )
    print(
        f"dashboard: {summary['tiles']} tiles "
        f"(A={summary['phase_a']}, B={summary['phase_b']}, C={summary['phase_c']}) "
        f"-> {summary['out']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
