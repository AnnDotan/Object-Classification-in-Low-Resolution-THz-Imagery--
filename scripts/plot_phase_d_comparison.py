"""Plot Phase D (regularization sweep) vs Phase B (baseline) — 6 small multiples + per-panel PNGs.

Reads `runs/final/final_B_L{l}_{m}_{d}/metrics.json` (baseline) and the three
Phase D treatment variants `runs/final/final_D_T{1,2,3}_L{l}_{m}_{d}/metrics.json`
for each (model, dataset, level) triple, then renders a 3-row x 2-column
small-multiple grid of `val_acc vs degradation level` with one line per
{baseline, T1, T2, T3}.

Outputs (US-031):
  - `artifacts/figures/phase_d_recovery_summary.png` (+ .pdf) — 3x2 composite
    (the canonical Final_Report rev2 figure; renamed from the US-026 working
    name `phase_d_comparison`).
  - `artifacts/figures/phase_d_comparison.png` (+ .pdf) — back-compat alias
    of the summary, byte-identical content. Emitted so the US-030 dashboard
    (`src/tools/build_final_dashboard.py`) continues to resolve its embedded
    `<img src="figures/phase_d_comparison.png">` until that consumer is
    migrated to the new name in a follow-up commit.
  - `artifacts/figures/phase_d_recovery_{model}_{dataset}.png` — six standalone
    per-panel PNGs (figsize=(5, 3.5)) consumed individually by the §VII
    appendix `\\includegraphics` chain in `docs/Final_Report.tex`.
  - `docs/_autogen/phase_d_recovery_table.tex` — 6-row `tabular` snippet with
    L3-anchored Δ_T{1,2,3} columns (booktabs, no outer `\\begin{table}` wrapper).

Pre-flight: re-verifies every Phase B v2 baseline's metrics.json against
`artifacts/validation/phase_b_v2_baseline_manifest.json` (PRD v3 §7
baseline-comparability invariant). Any drift aborts before writing plots.

Determinism: matplotlib pdf.fonttype=42 + svg.hashsalt fixed + `metadata=
{"CreationDate": None}` on every savefig; no `datetime.now()` overlays.
Two consecutive runs produce byte-identical PNGs (validated by
`src/tests/test_phase_d_plot.py::_check_byte_identical_across_runs`).

Idempotent — skips cells that have no `metrics.json` yet (NaN line segment).
Torch-free; depends only on matplotlib + the existing metrics-loading helpers.

US-026 (composite, 2026-05-23) → US-031 (per-panel + LaTeX, 2026-05-26).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
# Determinism pins (US-031).
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["svg.hashsalt"] = "phase_d_recovery"
import matplotlib.pyplot as plt  # noqa: E402

# Make the repo root importable when invoked as `python scripts/plot_phase_d_comparison.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.cells import (  # noqa: E402
    DATASETS,
    LEVELS,
    MODELS,
    PHASE_D_TREATMENTS,
)


RUNS_ROOT = _REPO_ROOT / "runs" / "final"
OUT_DIR = _REPO_ROOT / "artifacts" / "figures"
AUTOGEN_DIR = _REPO_ROOT / "docs" / "_autogen"
MANIFEST_PATH = _REPO_ROOT / "artifacts" / "validation" / "phase_b_v2_baseline_manifest.json"

# Anchor level for the LaTeX recovery table (PRD v3 §8 US-031).
L3_ANCHOR: int = 3

MODEL_LABEL = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "transnext_tiny": "TransNeXt-tiny",
}
DATASET_LABEL = {"cifar10": "CIFAR-10", "mnist": "MNIST"}

# Color + style per series — baseline is grey/dashed, treatments are saturated.
SERIES_STYLE = {
    "baseline": {"color": "#777777", "linestyle": "--", "marker": "o", "label": "Baseline (Phase B)"},
    "T1":       {"color": "#1f77b4", "linestyle": "-",  "marker": "s", "label": "T1 — dropout/drop-path"},
    "T2":       {"color": "#2ca02c", "linestyle": "-",  "marker": "^", "label": "T2 — mixup"},
    "T3":       {"color": "#d62728", "linestyle": "-",  "marker": "D", "label": "T3 — combo"},
}

# Savefig metadata — kills the embedded CreationDate that would otherwise differ
# byte-for-byte between two consecutive runs. matplotlib's PDF backend treats
# CreationDate=None as the documented signal to omit the timestamp; the PNG
# backend (via PIL pnginfo) silently skips None-valued keys, so the same dict
# is safe on both backends.
_SAVE_METADATA = {"CreationDate": None}


# ----------------------------------------------------------------------------
# Baseline-manifest re-verification (PRD v3 §7).
# ----------------------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_baseline_manifest(
    manifest_path: Optional[Path] = None,
    runs_root: Optional[Path] = None,
) -> tuple[bool, list[str]]:
    """Return (ok, drift_messages). If ok is False, the caller MUST exit 1
    before writing any plot output.

    Defaults resolve from the module-level constants at CALL time (not at
    def-time), so the test suite can monkeypatch `MANIFEST_PATH` / `RUNS_ROOT`
    on the module and have `main()` honour the override transparently."""
    if manifest_path is None:
        manifest_path = MANIFEST_PATH
    if runs_root is None:
        runs_root = RUNS_ROOT
    if not manifest_path.exists():
        return False, [f"manifest missing at {manifest_path}"]
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"manifest unreadable: {exc}"]
    entries = payload.get("entries", {})
    if not isinstance(entries, dict) or not entries:
        return False, ["manifest has no 'entries' map"]
    drifts: list[str] = []
    for tag, meta in sorted(entries.items()):
        expected = meta.get("metrics_sha256")
        rel = meta.get("metrics_path")
        if expected is None or rel is None:
            drifts.append(f"{tag}: malformed manifest entry")
            continue
        # Resolve the path relative to the repo root (manifest stores POSIX paths).
        metrics_path = _REPO_ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        # Fallback: tag-derived path under the provided runs_root (test helper).
        if not metrics_path.exists():
            metrics_path = runs_root / tag / "metrics.json"
        if not metrics_path.exists():
            drifts.append(f"{tag}: metrics.json deleted ({rel})")
            continue
        actual = _sha256_of(metrics_path)
        if actual != expected:
            drifts.append(
                f"{tag}: hash drift ({expected[:12]}... -> {actual[:12]}...)"
            )
    return (not drifts), drifts


# ----------------------------------------------------------------------------
# Series + panel construction.
# ----------------------------------------------------------------------------


def _read_best_val_acc(tag: str) -> Optional[float]:
    """Return best_val_acc for a cell, or None if metrics.json is missing/invalid."""
    p = RUNS_ROOT / tag / "metrics.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("best_val_acc", "final_val_acc", "last_val_acc"):
        v = m.get(key)
        if isinstance(v, (int, float)) and v >= 0.0:
            return float(v)
    return None


def _series_for_panel(model: str, dataset: str) -> dict[str, list[Optional[float]]]:
    """Return {series_key: [val_acc_at_L1, ..., val_acc_at_L5]} for a panel."""
    out: dict[str, list[Optional[float]]] = {}
    out["baseline"] = [
        _read_best_val_acc(f"final_B_L{l}_{model}_{dataset}") for l in LEVELS
    ]
    for t in PHASE_D_TREATMENTS:
        out[t] = [
            _read_best_val_acc(f"final_D_{t}_L{l}_{model}_{dataset}") for l in LEVELS
        ]
    return out


def _plot_panel(ax, model: str, dataset: str) -> None:
    series = _series_for_panel(model, dataset)
    for key in ("baseline", "T1", "T2", "T3"):
        ys = series[key]
        # Replace None with NaN so matplotlib skips missing segments.
        ys_plot = [float("nan") if y is None else y for y in ys]
        ax.plot(list(LEVELS), ys_plot, **SERIES_STYLE[key], linewidth=1.6, markersize=5)
    ax.set_xticks(list(LEVELS))
    ax.set_xticklabels([f"L{l}" for l in LEVELS])
    ax.set_xlabel("Degradation level")
    ax.set_ylabel("Best val_acc")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.set_title(f"{MODEL_LABEL.get(model, model)} · {DATASET_LABEL.get(dataset, dataset)}",
                 fontsize=10)


# ----------------------------------------------------------------------------
# Rendering.
# ----------------------------------------------------------------------------


def _render_summary(out_dir: Path) -> Path:
    """3x2 composite — the canonical Final_Report rev2 figure.

    Writes the canonical `phase_d_recovery_summary.{png,pdf}` AND a
    back-compat alias `phase_d_comparison.{png,pdf}` (US-030 dashboard
    consumer still references the old name; can be retired once the
    dashboard is migrated in a follow-up commit).
    """
    fig, axes = plt.subplots(
        nrows=len(MODELS), ncols=len(DATASETS),
        figsize=(10, 9), sharey=True, sharex=True,
    )
    for r, model in enumerate(MODELS):
        for c, dataset in enumerate(DATASETS):
            ax = axes[r][c]
            _plot_panel(ax, model, dataset)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Phase D — Regularization Sweep vs Phase B Baseline\n"
        "(US-026: dropout/drop-path, mixup, combo)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "phase_d_recovery_summary.png"
    pdf_path = out_dir / "phase_d_recovery_summary.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", metadata=_SAVE_METADATA)
    fig.savefig(pdf_path, bbox_inches="tight", metadata=_SAVE_METADATA)
    # Back-compat aliases — US-030 dashboard (src/tools/build_final_dashboard.py)
    # currently embeds artifacts/figures/phase_d_comparison.png by literal name.
    # Until that consumer is migrated, mirror the canonical render under the
    # old filenames so the dashboard does not 404.
    fig.savefig(out_dir / "phase_d_comparison.png", dpi=150, bbox_inches="tight",
                metadata=_SAVE_METADATA)
    fig.savefig(out_dir / "phase_d_comparison.pdf", bbox_inches="tight",
                metadata=_SAVE_METADATA)
    plt.close(fig)
    return png_path


def _render_per_panel(model: str, dataset: str, out_dir: Path) -> Path:
    """Single-panel figure mirroring the composite cell — figsize (5, 3.5)."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _plot_panel(ax, model, dataset)
    # Legend INSIDE the panel since there's no shared bottom legend here.
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase_d_recovery_{model}_{dataset}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", metadata=_SAVE_METADATA)
    plt.close(fig)
    return out_path


def render(out_dir: Path = OUT_DIR) -> list[Path]:
    """Render all 7 figures: 1 summary composite + 6 per-panel PNGs.

    Returns the list of written paths in order (summary PNG first, then the
    six per-panel PNGs in canonical (model, dataset) order).
    """
    paths: list[Path] = [_render_summary(out_dir)]
    for model in MODELS:
        for dataset in DATASETS:
            paths.append(_render_per_panel(model, dataset, out_dir))
    return paths


# ----------------------------------------------------------------------------
# LaTeX recovery table.
# ----------------------------------------------------------------------------


def _delta_at_level(model: str, dataset: str, level: int) -> dict[str, Optional[float]]:
    """Return {T1, T2, T3} delta in PERCENTAGE POINTS vs Phase B baseline at `level`.

    Missing baseline -> all None. Missing treatment -> that one is None.
    """
    base = _read_best_val_acc(f"final_B_L{level}_{model}_{dataset}")
    out: dict[str, Optional[float]] = {"T1": None, "T2": None, "T3": None}
    if base is None:
        return out
    for t in PHASE_D_TREATMENTS:
        v = _read_best_val_acc(f"final_D_{t}_L{level}_{model}_{dataset}")
        out[t] = None if v is None else (v - base) * 100.0
    return out


def _fmt_pp(v: Optional[float]) -> str:
    if v is None:
        return "---"
    return f"{v:+.2f}"


def _render_recovery_table(out_path: Path, level: int = L3_ANCHOR) -> Path:
    """Emit a 6-row booktabs `tabular` (no `\\begin{table}` wrapper)."""
    lines: list[str] = []
    lines.append(
        f"% AUTO-GENERATED — do not edit. Regenerate via scripts/plot_phase_d_comparison.py."
    )
    lines.append(
        f"% Phase D recovery — Δ vs Phase B baseline at L{level}, in percentage points."
    )
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"  \toprule")
    lines.append(
        r"  Model & Dataset & $\Delta_{T1}$@L"
        + str(level)
        + r" & $\Delta_{T2}$@L"
        + str(level)
        + r" & $\Delta_{T3}$@L"
        + str(level)
        + r" \\"
    )
    lines.append(r"  \midrule")
    for model in MODELS:
        for dataset in DATASETS:
            d = _delta_at_level(model, dataset, level)
            m_pretty = MODEL_LABEL.get(model, model)
            d_pretty = DATASET_LABEL.get(dataset, dataset)
            lines.append(
                f"  {m_pretty} & {d_pretty} & {_fmt_pp(d['T1'])} & "
                f"{_fmt_pp(d['T2'])} & {_fmt_pp(d['T3'])} \\\\"
            )
    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ----------------------------------------------------------------------------
# Entry point.
# ----------------------------------------------------------------------------


def main() -> int:
    # Pre-flight: re-verify the Phase B v2 baseline manifest BEFORE writing
    # any output. Any drift aborts with exit code 1 (PRD v3 §7).
    ok, drifts = _verify_baseline_manifest()
    if not ok:
        print(
            "[plot_phase_d_comparison] ABORT — Phase B v2 baseline drift detected:",
            file=sys.stderr,
        )
        for d in drifts:
            print(f"  - {d}", file=sys.stderr)
        return 1

    paths = render(OUT_DIR)
    table_path = _render_recovery_table(
        AUTOGEN_DIR / "phase_d_recovery_table.tex",
        level=L3_ANCHOR,
    )

    summary_png = paths[0]
    per_panel = paths[1:]
    print(f"[plot_phase_d_comparison] wrote {summary_png}")
    print(f"[plot_phase_d_comparison] wrote {summary_png.with_suffix('.pdf')}")
    # Back-compat aliases (US-030 dashboard consumer; see _render_summary).
    legacy_png = summary_png.with_name("phase_d_comparison.png")
    print(f"[plot_phase_d_comparison] wrote {legacy_png} (back-compat alias)")
    print(f"[plot_phase_d_comparison] wrote {legacy_png.with_suffix('.pdf')} (back-compat alias)")
    for p in per_panel:
        print(f"[plot_phase_d_comparison] wrote {p}")
    print(f"[plot_phase_d_comparison] wrote {table_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
