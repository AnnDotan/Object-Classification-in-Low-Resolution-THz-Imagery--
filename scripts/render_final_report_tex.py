"""Two-stage builder for the IEEEtran scientific report (US-024).

Stage 1: read artifacts/Final_Exp.json and emit booktabs LaTeX snippets to
         docs/_autogen/{phase_a_table,phase_b_table,phase_c_l5_table}.tex.
         These are \\input{}'d by docs/Final_Report.tex.

Stage 2: invoke pdflatex + bibtex + pdflatex + pdflatex on docs/Final_Report.tex,
         producing artifacts/Final_Report.pdf.

Idempotent: stage 1 produces byte-identical files on re-run (sorted iteration
over Final_Exp.json, deterministic formatting).

Usage:
    python scripts/render_final_report_tex.py                 # both stages
    python scripts/render_final_report_tex.py --tables-only   # stage 1 only
    python scripts/render_final_report_tex.py --build-only    # stage 2 only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = _REPO_ROOT / "artifacts" / "Final_Exp.json"
_AUTOGEN_DIR = _REPO_ROOT / "docs" / "_autogen"
_TEX_MAIN = _REPO_ROOT / "docs" / "Final_Report.tex"
_PDF_OUT = _REPO_ROOT / "artifacts" / "Final_Report.pdf"

MODELS = ["resnet50", "densenet121", "transnext_tiny"]
DATASETS = ["cifar10", "mnist"]
AXES = ["resolution", "noise", "blur", "saturation", "salt_pepper"]
LEVELS = [1, 2, 3, 4, 5]

MODEL_PRETTY = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "transnext_tiny": "TransNeXt-tiny",
}
DATASET_PRETTY = {"cifar10": "CIFAR-10", "mnist": "MNIST"}
AXIS_PRETTY = {
    "resolution": "resolution",
    "noise": "noise (v2)",
    "blur": "blur",
    "saturation": "saturation",
    "salt_pepper": "salt-and-pepper (v2)",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _load_rows(src: Path) -> list[dict]:
    with src.open(encoding="utf-8") as f:
        return json.load(f)["rows"]


def _index(rows: list[dict]) -> dict[str, dict]:
    return {r["tag"]: r for r in rows}


def _fmt(v) -> str:
    if isinstance(v, (int, float)) and v >= 0:
        return f"{v:.4f}"
    return "---"


def _bold_max(values: list[float | None]) -> list[str]:
    """Format a row of values, bolding the max."""
    nums = [v for v in values if isinstance(v, (int, float)) and v >= 0]
    if not nums:
        return ["---" if v is None else _fmt(v) for v in values]
    mx = max(nums)
    out = []
    for v in values:
        if isinstance(v, (int, float)) and v >= 0:
            cell = f"{v:.4f}"
            if abs(v - mx) < 1e-9:
                cell = f"\\textbf{{{cell}}}"
            out.append(cell)
        else:
            out.append("---")
    return out


# --------------------------------------------------------------------------
# Table renderers
# --------------------------------------------------------------------------


def render_phase_a(idx: dict) -> str:
    c10_max = max(
        (idx.get(f"final_clean_{m}_cifar10", {}).get("val_acc") or -1.0) for m in MODELS
    )
    mn_max = max(
        (idx.get(f"final_clean_{m}_mnist", {}).get("val_acc") or -1.0) for m in MODELS
    )
    body_lines = []
    for model in MODELS:
        c10 = idx.get(f"final_clean_{model}_cifar10", {}).get("val_acc")
        mn = idx.get(f"final_clean_{model}_mnist", {}).get("val_acc")
        c10_cell = (
            f"\\textbf{{{c10:.4f}}}"
            if isinstance(c10, (int, float)) and abs(c10 - c10_max) < 1e-9
            else _fmt(c10)
        )
        mn_cell = (
            f"\\textbf{{{mn:.4f}}}"
            if isinstance(mn, (int, float)) and abs(mn - mn_max) < 1e-9
            else _fmt(mn)
        )
        body_lines.append(f"    {MODEL_PRETTY[model]} & {c10_cell} & {mn_cell} \\\\")
    out = [
        r"% AUTO-GENERATED — do not edit. Regenerate via scripts/render_final_report_tex.py.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Phase A clean-baseline best-val-acc per (model, dataset). All cells use the identity degradation pipeline at $224{\times}224$.}",
        r"  \label{tab:phase_a}",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    Model & CIFAR-10 & MNIST \\",
        r"    \midrule",
        *body_lines,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(out) + "\n"


def render_phase_b(idx: dict) -> str:
    lines = []
    lines.append(r"% AUTO-GENERATED — do not edit. Regenerate via scripts/render_final_report_tex.py.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Phase B combined-degradation best-val-acc, pipeline v2. All five degradation axes are active at level $L$ simultaneously. Boldface marks the best-of-three architectures per $(\text{dataset}, L)$ column. Mean $\Delta$ across the 30 v2 cells is $-12.49$ pp vs the pre-US-017 v1 pipeline.}")
    lines.append(r"  \label{tab:phase_b}")
    lines.append(r"  \begin{tabular}{llccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Model & Dataset & L1 & L2 & L3 & L4 & L5 \\")
    lines.append(r"    \midrule")
    # per (dataset, level), find column max across the 3 models for bolding
    col_max: dict[tuple[str, int], float] = {}
    for ds in DATASETS:
        for L in LEVELS:
            vals = []
            for m in MODELS:
                v = idx.get(f"final_B_L{L}_{m}_{ds}", {}).get("val_acc")
                if isinstance(v, (int, float)) and v >= 0:
                    vals.append(v)
            if vals:
                col_max[(ds, L)] = max(vals)
    for model in MODELS:
        for ds in DATASETS:
            cells = []
            for L in LEVELS:
                v = idx.get(f"final_B_L{L}_{model}_{ds}", {}).get("val_acc")
                if isinstance(v, (int, float)) and v >= 0:
                    cell = f"{v:.4f}"
                    if (ds, L) in col_max and abs(v - col_max[(ds, L)]) < 1e-9:
                        cell = f"\\textbf{{{cell}}}"
                    cells.append(cell)
                else:
                    cells.append("---")
            lines.append(f"    {MODEL_PRETTY[model]} & {DATASET_PRETTY[ds]} & " + " & ".join(cells) + r" \\")
        if model != MODELS[-1]:
            lines.append(r"    \midrule")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def render_phase_c_l5(idx: dict) -> str:
    lines = []
    lines.append(r"% AUTO-GENERATED — do not edit. Regenerate via scripts/render_final_report_tex.py.")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Phase C single-axis isolation at $L=5$ (worst case), pipeline v2 on \texttt{noise} and \texttt{salt\_pepper} rows. Only the named axis is active at L5; the other four are at identity. Boldface marks the best-of-three architectures per $(\text{dataset}, \text{axis})$ cell. The \texttt{resolution}, \texttt{blur}, \texttt{saturation} rows are byte-identical to the pre-US-017 v1 pipeline.}")
    lines.append(r"  \label{tab:phase_c_l5}")
    lines.append(r"  \begin{tabular}{lcccccc}")
    lines.append(r"    \toprule")
    lines.append(r"     & \multicolumn{3}{c}{CIFAR-10} & \multicolumn{3}{c}{MNIST} \\")
    lines.append(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"    Axis @ L5 & ResNet50 & DenseNet121 & TransNeXt-tiny & ResNet50 & DenseNet121 & TransNeXt-tiny \\")
    lines.append(r"    \midrule")
    # find column max per (dataset)-restricted to 3 models per row
    for axis in AXES:
        cells = []
        for ds in DATASETS:
            vals = []
            for m in MODELS:
                v = idx.get(f"final_C_L5_{axis}_{m}_{ds}", {}).get("val_acc")
                vals.append(v if isinstance(v, (int, float)) and v >= 0 else None)
            mx = max((v for v in vals if v is not None), default=None)
            for v in vals:
                if v is None:
                    cells.append("---")
                else:
                    cell = f"{v:.4f}"
                    if mx is not None and abs(v - mx) < 1e-9:
                        cell = f"\\textbf{{{cell}}}"
                    cells.append(cell)
        axis_label = AXIS_PRETTY[axis].replace("_", "\\_")
        lines.append(f"    \\texttt{{{axis_label}}} & " + " & ".join(cells) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Stage 1: emit tables
# --------------------------------------------------------------------------


def emit_tables(src: Path, out_dir: Path) -> dict[str, Path]:
    rows = _load_rows(src)
    idx = _index(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, content in [
        ("phase_a_table.tex", render_phase_a(idx)),
        ("phase_b_table.tex", render_phase_b(idx)),
        ("phase_c_l5_table.tex", render_phase_c_l5(idx)),
    ]:
        path = out_dir / name
        # write with LF only for byte-identity across re-runs
        path.write_bytes(content.encode("utf-8"))
        written[name] = path
        print(f"wrote {path.relative_to(_REPO_ROOT)} ({len(content):,} chars)")
    return written


# --------------------------------------------------------------------------
# Stage 2: build PDF via pdflatex + bibtex
# --------------------------------------------------------------------------


def find_tool(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p
    miktex = Path("C:/Users/ib94/AppData/Local/Programs/MiKTeX/miktex/bin/x64") / f"{name}.exe"
    if miktex.exists():
        return str(miktex)
    return None


def build_pdf(tex_main: Path, out_pdf: Path) -> int:
    pdflatex = find_tool("pdflatex")
    bibtex = find_tool("bibtex")
    if not pdflatex:
        print("ERROR: pdflatex not found on PATH or at MiKTeX default location.", file=sys.stderr)
        return 1
    if not bibtex:
        print("WARN: bibtex not found; references may not resolve.", file=sys.stderr)

    workdir = tex_main.parent
    base = tex_main.stem  # "Final_Report"

    common_args = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={workdir}",
        str(tex_main),
    ]

    def run(cmd: list[str], label: str) -> int:
        print(f"  -> {label}: {Path(cmd[0]).name} {' '.join(cmd[1:])}")
        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            print(f"     TIMEOUT after 300s", file=sys.stderr)
            return 124
        # Always show last lines of output for debugging
        if result.returncode != 0:
            print(result.stdout[-3000:], file=sys.stderr)
            print(result.stderr[-1500:], file=sys.stderr)
        return result.returncode

    # pass 1
    rc = run(common_args, "pdflatex pass 1")
    if rc != 0:
        return rc

    # bibtex
    if bibtex:
        rc = run([bibtex, base], "bibtex")
        # bibtex returns nonzero on warnings; only fatal-out on missing aux
        # tolerate nonzero from bibtex if Final_Report.bbl was produced
        bbl = workdir / f"{base}.bbl"
        if not bbl.exists():
            print(f"ERROR: bibtex did not produce {bbl}", file=sys.stderr)
            return rc

    # pass 2 + 3 to resolve refs
    for i in (2, 3):
        rc = run(common_args, f"pdflatex pass {i}")
        if rc != 0:
            return rc

    built_pdf = workdir / f"{base}.pdf"
    if not built_pdf.exists():
        print(f"ERROR: build did not produce {built_pdf}", file=sys.stderr)
        return 1

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_pdf, out_pdf)
    print(f"wrote {out_pdf.relative_to(_REPO_ROOT)} ({out_pdf.stat().st_size:,} bytes)")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=str(_DEFAULT_SRC))
    p.add_argument("--out-dir", default=str(_AUTOGEN_DIR))
    p.add_argument("--tex", default=str(_TEX_MAIN))
    p.add_argument("--pdf", default=str(_PDF_OUT))
    p.add_argument("--tables-only", action="store_true")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args(argv)

    if not args.build_only:
        emit_tables(Path(args.src), Path(args.out_dir))

    if args.tables_only:
        return 0

    return build_pdf(Path(args.tex), Path(args.pdf))


if __name__ == "__main__":
    raise SystemExit(main())
