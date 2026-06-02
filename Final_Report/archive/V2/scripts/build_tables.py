"""Generate IEEE-style LaTeX tables and matching CSVs from Final_Exp.json.

Outputs (under Final_Report/):
  source/tables/tab_II_levels.tex          - degradation level schedule
  source/tables/tab_III_pipelines.tex      - per-pipeline PSNR / SSIM
  source/tables/tab_IV_phase_a.tex         - clean baseline accuracy
  source/tables/tab_V_phase_b.tex          - combined-degradation accuracy
  source/tables/tab_VI_phase_c.tex         - single-axis isolation (compact)
  source/tables/tab_VII_phase_d.tex        - regularization-recovery
  source/tables/tab_VIII_phase_b2.tex      - THz-protocol B2 + B2nr
  source/tables/tab_IX_phase_c2.tex        - single-axis THz protocol
  source/tables/tab_X_multiseed.tex        - multi-seed audit summary
  data/tables/*.csv                        - one CSV per table (audit trail)

All tables use the booktabs three-line format. All numbers come from
artifacts/Final_Exp.json (no synthesis, no fabrication).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "Final_Exp.json"
TEX_DIR = ROOT / "Final_Report" / "source" / "tables"
CSV_DIR = ROOT / "Final_Report" / "data" / "tables"
TEX_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)


MODEL_LABELS = {
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "transnext_tiny": "TransNeXt-tiny",
}
DATASET_LABELS = {"cifar10": "CIFAR-10", "mnist": "MNIST"}
AXIS_LABELS = {
    "resolution": "Resolution",
    "blur": "Blur",
    "noise": "Noise",
    "salt_pepper": "S\\&P",
    "saturation": "Saturation",
}
MODELS_ORDER = ("resnet50", "densenet121", "transnext_tiny")
DATASETS_ORDER = ("cifar10", "mnist")
LEVELS = (1, 2, 3, 4, 5)


def _fmt_acc(x: float | None) -> str:
    if x is None:
        return "--"
    return f"{x * 100:.2f}"


def _fmt_acc_pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    if std is None or std == 0:
        return f"{mean * 100:.2f}"
    return f"{mean * 100:.2f} $\\pm$ {std * 100:.2f}"


def _fmt_num(x: float | None, fmt: str = "{:.2f}") -> str:
    return "--" if x is None else fmt.format(x)


def _load_rows() -> list[dict]:
    with SRC.open("r", encoding="utf-8") as f:
        return json.load(f)["rows"]


def _csv(name: str, fields: list[str], rows: list[dict]) -> None:
    p = CSV_DIR / name
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _tex_table(
    out_name: str,
    caption: str,
    label: str,
    header_cells: list[str],
    body_rows: list[list[str]],
    column_spec: str,
) -> None:
    head = " & ".join(header_cells) + " \\\\"
    body = "\n".join(" & ".join(row) + " \\\\" for row in body_rows)
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\renewcommand{{\\arraystretch}}{{1.1}}
\\begin{{tabular}}{{{column_spec}}}
\\toprule
{head}
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    (TEX_DIR / out_name).write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# Table II: Degradation level schedule
# ---------------------------------------------------------------------------

def table_II_levels() -> None:
    from src.data.degradation_levels import DEGRADATION_LEVELS, LEVEL_NAMES

    header = ["Level", "Name", "low\\_res", "Blur K", "Blur $\\sigma$", "Noise std", "S\\&P", "Saturation"]
    body_rows = []
    csv_rows = []
    for L in LEVELS:
        p = DEGRADATION_LEVELS[L]
        body_rows.append([
            f"L{L}",
            LEVEL_NAMES[L],
            f"{int(p['low_res'])}",
            f"{int(p['blur_kernel'])}",
            f"{p['blur_sigma']:.2f}",
            f"{p['noise_std']:.2f}",
            f"{p['salt_pepper']:.2f}",
            f"{p['saturation']:.2f}",
        ])
        csv_rows.append({
            "level": L,
            "name": LEVEL_NAMES[L],
            "low_res": p["low_res"],
            "blur_kernel": p["blur_kernel"],
            "blur_sigma": p["blur_sigma"],
            "noise_std": p["noise_std"],
            "salt_pepper": p["salt_pepper"],
            "saturation": p["saturation"],
        })

    _tex_table(
        out_name="tab_II_levels.tex",
        caption="Five-level synthetic degradation schedule. Blur kernel and $\\sigma$ are pixel-domain values at the $224 \\times 224$ model input.",
        label="tab:levels",
        header_cells=header,
        body_rows=body_rows,
        column_spec="llrrrrrr",
    )
    _csv(
        "tab_II_levels.csv",
        ["level", "name", "low_res", "blur_kernel", "blur_sigma", "noise_std", "salt_pepper", "saturation"],
        csv_rows,
    )


# ---------------------------------------------------------------------------
# Table III: Per-pipeline PSNR / SSIM
# ---------------------------------------------------------------------------

def table_III_pipelines(rows: list[dict]) -> None:
    # Collapse to unique (dataset, phase, level, axis) -> PSNR/SSIM (identical across cells).
    seen: dict[tuple, dict] = {}
    for r in rows:
        if r.get("psnr_mean") is None:
            continue
        phase = "B" if r["phase"] == "D" else r["phase"]
        key = (r["dataset"], phase, r.get("level"), r.get("axis"))
        if key in seen:
            continue
        seen[key] = {
            "dataset": r["dataset"],
            "phase": phase,
            "level": r.get("level"),
            "axis": r.get("axis") or "",
            "psnr_mean": r["psnr_mean"],
            "psnr_std": r["psnr_std"],
            "ssim_mean": r["ssim_mean"],
            "ssim_std": r["ssim_std"],
        }
    pipelines = sorted(seen.values(), key=lambda d: (d["dataset"], d["phase"], d["level"] or 0, d["axis"]))

    # Compact: emit a 4-column block per dataset (level rows) for B / B2 / B2nr / C2 (with axis)
    # For Phase C we use a separate table later. Here we focus on the
    # protocol-level pipelines.
    header = ["Dataset", "Phase", "Level", "Axis", "PSNR (dB)", "SSIM"]
    body_rows = []
    for p in pipelines:
        if p["phase"] == "A":
            continue  # clean baseline = reference, PSNR = inf
        body_rows.append([
            DATASET_LABELS[p["dataset"]],
            p["phase"],
            f"L{p['level']}" if p["level"] else "--",
            AXIS_LABELS.get(p["axis"], p["axis"]) if p["axis"] else "--",
            f"{p['psnr_mean']:.2f} $\\pm$ {p['psnr_std']:.2f}",
            f"{p['ssim_mean']:.3f} $\\pm$ {p['ssim_std']:.3f}",
        ])

    _tex_table(
        out_name="tab_III_pipelines.tex",
        caption="Mean PSNR and SSIM per degradation pipeline. Each value is computed over a fixed deterministic 256-sample validation subset using torchmetrics with data\\_range=1, sample seed offset = $2 \\times 10^{7}$, clean reference upsampled to $224 \\times 224$ by bilinear interpolation.",
        label="tab:pipelines",
        header_cells=header,
        body_rows=body_rows,
        column_spec="llllrr",
    )
    _csv(
        "tab_III_pipelines.csv",
        ["dataset", "phase", "level", "axis", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"],
        list(seen.values()),
    )


# ---------------------------------------------------------------------------
# Per-phase accuracy tables.
# Helper that picks (mean+/-std, n_seeds) for a (phase, dataset, level, axis,
# treatment, model) cell.
# ---------------------------------------------------------------------------

def _cell_lookup(rows: list[dict]) -> dict[tuple, dict]:
    """Build a dict keyed by (phase, dataset, level, axis, treatment, model)."""
    out: dict[tuple, dict] = {}
    for r in rows:
        key = (
            r["phase"],
            r["dataset"],
            r.get("level"),
            r.get("axis"),
            r.get("treatment"),
            r["model"],
        )
        # If multi-seed, val_acc_mean is filled; otherwise val_acc is the single-seed value.
        out[key] = r
    return out


def _acc_str(r: dict | None) -> str:
    if r is None:
        return "--"
    if r.get("val_acc_mean") is not None:
        return _fmt_acc_pm(r["val_acc_mean"], r.get("val_acc_std"))
    return _fmt_acc(r.get("val_acc"))


def table_IV_phase_a(rows: list[dict]) -> None:
    L = _cell_lookup(rows)
    header = ["Model"] + [DATASET_LABELS[d] for d in DATASETS_ORDER]
    body_rows = []
    csv_rows = []
    for m in MODELS_ORDER:
        line = [MODEL_LABELS[m]]
        for d in DATASETS_ORDER:
            r = L.get(("A", d, None, None, None, m))
            line.append(_acc_str(r))
            csv_rows.append({
                "phase": "A",
                "model": m,
                "dataset": d,
                "val_acc": r.get("val_acc") if r else None,
                "val_acc_mean": r.get("val_acc_mean") if r else None,
                "val_acc_std": r.get("val_acc_std") if r else None,
            })
        body_rows.append(line)

    _tex_table(
        out_name="tab_IV_phase_a.tex",
        caption="Phase A clean-baseline accuracy (\\%) on the held-out validation split. No degradation applied; values are upper bounds for every subsequent phase.",
        label="tab:phase_a",
        header_cells=header,
        body_rows=body_rows,
        column_spec="lrr",
    )
    _csv("tab_IV_phase_a.csv", ["phase", "model", "dataset", "val_acc", "val_acc_mean", "val_acc_std"], csv_rows)


def _table_levels_block(
    out_name: str,
    caption: str,
    label: str,
    rows: list[dict],
    selector,
) -> None:
    """Generic per-(model, dataset, level) table.

    `selector(level, model, dataset)` returns the lookup key tuple
    or None to skip the entry.
    """
    L = _cell_lookup(rows)
    header = ["Model", "Dataset"] + [f"L{lv}" for lv in LEVELS]
    body_rows = []
    csv_rows = []
    for m in MODELS_ORDER:
        for d in DATASETS_ORDER:
            line = [MODEL_LABELS[m], DATASET_LABELS[d]]
            for lv in LEVELS:
                key = selector(lv, m, d)
                r = L.get(key) if key else None
                line.append(_acc_str(r))
                csv_rows.append({
                    "model": m,
                    "dataset": d,
                    "level": lv,
                    "val_acc": (r or {}).get("val_acc"),
                    "val_acc_mean": (r or {}).get("val_acc_mean"),
                    "val_acc_std": (r or {}).get("val_acc_std"),
                })
            body_rows.append(line)
    _tex_table(out_name, caption, label, header, body_rows, "ll" + "r" * len(LEVELS))
    _csv(out_name.replace(".tex", ".csv"), ["model", "dataset", "level", "val_acc", "val_acc_mean", "val_acc_std"], csv_rows)


def table_V_phase_b(rows: list[dict]) -> None:
    _table_levels_block(
        "tab_V_phase_b.tex",
        "Phase B (combined degradation) validation accuracy (\\%). All five axes active at the same level L. Cells with $\\pm$ values are multi-seed audited (seeds $\\{42, 43, 44\\}$).",
        "tab:phase_b",
        rows,
        lambda lv, m, d: ("B", d, lv, None, None, m),
    )


def table_VI_phase_c(rows: list[dict]) -> None:
    """Per-axis isolation: one block per axis."""
    L = _cell_lookup(rows)
    axes = ("resolution", "blur", "noise", "salt_pepper", "saturation")
    # Wide format: Model x Dataset x Axis -> rows of L1..L5
    header = ["Model", "Dataset", "Axis"] + [f"L{lv}" for lv in LEVELS]
    body_rows = []
    csv_rows = []
    for m in MODELS_ORDER:
        for d in DATASETS_ORDER:
            for ax in axes:
                line = [MODEL_LABELS[m], DATASET_LABELS[d], AXIS_LABELS[ax]]
                for lv in LEVELS:
                    r = L.get(("C", d, lv, ax, None, m))
                    line.append(_acc_str(r))
                    csv_rows.append({"model": m, "dataset": d, "axis": ax, "level": lv,
                                     "val_acc": (r or {}).get("val_acc")})
                body_rows.append(line)

    _tex_table(
        out_name="tab_VI_phase_c.tex",
        caption="Phase C single-axis isolation validation accuracy (\\%). The named axis is at level L; every other axis is at its identity value.",
        label="tab:phase_c",
        header_cells=header,
        body_rows=body_rows,
        column_spec="lll" + "r" * len(LEVELS),
    )
    _csv("tab_VI_phase_c.csv", ["model", "dataset", "axis", "level", "val_acc"], csv_rows)


def table_VII_phase_d(rows: list[dict]) -> None:
    """T1 / T2 / T3 regularization recovery, per (model, dataset, level)."""
    L = _cell_lookup(rows)
    treatments = ("T1", "T2", "T3")
    header = ["Model", "Dataset", "Treatment"] + [f"L{lv}" for lv in LEVELS]
    body_rows = []
    csv_rows = []
    for m in MODELS_ORDER:
        for d in DATASETS_ORDER:
            for t in treatments:
                line = [MODEL_LABELS[m], DATASET_LABELS[d], t]
                for lv in LEVELS:
                    r = L.get(("D", d, lv, None, t, m))
                    line.append(_acc_str(r))
                    csv_rows.append({"model": m, "dataset": d, "treatment": t, "level": lv,
                                     "val_acc": (r or {}).get("val_acc")})
                body_rows.append(line)

    _tex_table(
        out_name="tab_VII_phase_d.tex",
        caption="Phase D regularization-recovery validation accuracy (\\%). T1 = architectural regularization only; T2 = label-mixing only; T3 = combination. Degradation pipeline matches Phase B at the same level.",
        label="tab:phase_d",
        header_cells=header,
        body_rows=body_rows,
        column_spec="lll" + "r" * len(LEVELS),
    )
    _csv("tab_VII_phase_d.csv", ["model", "dataset", "treatment", "level", "val_acc"], csv_rows)


def table_VIII_phase_b2(rows: list[dict]) -> None:
    """Phase B2 (T3 reg, sat=0, noise=0) and B2-nr (L3 only, no reg)."""
    L = _cell_lookup(rows)
    header = ["Model", "Dataset", "Arm"] + [f"L{lv}" for lv in LEVELS]
    body_rows = []
    csv_rows = []
    for m in MODELS_ORDER:
        for d in DATASETS_ORDER:
            # B2 line (all 5 levels, T3)
            line = [MODEL_LABELS[m], DATASET_LABELS[d], "B2 (T3)"]
            for lv in LEVELS:
                r = L.get(("B2", d, lv, None, "T3", m))
                line.append(_acc_str(r))
                csv_rows.append({"arm": "B2", "model": m, "dataset": d, "level": lv,
                                 "val_acc": (r or {}).get("val_acc")})
            body_rows.append(line)
            # B2nr line (L3 only)
            line = [MODEL_LABELS[m], DATASET_LABELS[d], "B2nr"]
            for lv in LEVELS:
                if lv == 3:
                    r = L.get(("B2nr", d, lv, None, None, m))
                    line.append(_acc_str(r))
                    csv_rows.append({"arm": "B2nr", "model": m, "dataset": d, "level": lv,
                                     "val_acc": (r or {}).get("val_acc")})
                else:
                    line.append("--")
            body_rows.append(line)

    _tex_table(
        out_name="tab_VIII_phase_b2.tex",
        caption="Phase B2 (THz-protocol simplification, saturation $=0$, noise $=0$) and Phase B2-nr (L3 only, no regularization) validation accuracy (\\%).",
        label="tab:phase_b2",
        header_cells=header,
        body_rows=body_rows,
        column_spec="lll" + "r" * len(LEVELS),
    )
    _csv("tab_VIII_phase_b2.csv", ["arm", "model", "dataset", "level", "val_acc"], csv_rows)


def table_IX_phase_c2(rows: list[dict]) -> None:
    L = _cell_lookup(rows)
    axes = ("resolution", "blur", "salt_pepper")
    header = ["Model", "Dataset", "Axis"] + [f"L{lv}" for lv in LEVELS]
    body_rows = []
    csv_rows = []
    for m in MODELS_ORDER:
        for d in DATASETS_ORDER:
            for ax in axes:
                line = [MODEL_LABELS[m], DATASET_LABELS[d], AXIS_LABELS[ax]]
                for lv in LEVELS:
                    r = L.get(("C2", d, lv, ax, "T3", m))
                    line.append(_acc_str(r))
                    csv_rows.append({"model": m, "dataset": d, "axis": ax, "level": lv,
                                     "val_acc": (r or {}).get("val_acc")})
                body_rows.append(line)

    _tex_table(
        out_name="tab_IX_phase_c2.tex",
        caption="Phase C2 single-axis isolation under the THz protocol (saturation $=0$, noise $=0$, T3 regularization) validation accuracy (\\%).",
        label="tab:phase_c2",
        header_cells=header,
        body_rows=body_rows,
        column_spec="lll" + "r" * len(LEVELS),
    )
    _csv("tab_IX_phase_c2.csv", ["model", "dataset", "axis", "level", "val_acc"], csv_rows)


def table_X_multiseed(rows: list[dict]) -> None:
    """Multi-seed audit summary: mean +/- std across seeds {42, 43, 44} at L3."""
    header = ["Phase", "Model", "Dataset", "Level", "Axis / Tr.", "Mean (\\%)", "$\\sigma$ (pp)", "$n$"]
    body_rows = []
    csv_rows = []
    seen = set()
    for r in rows:
        if not r.get("val_acc_mean"):
            continue
        if r.get("val_acc_std") is None:
            continue
        n = len(r.get("seeds_observed", []) or [])
        if n < 2:
            continue
        key = (r["phase"], r["model"], r["dataset"], r.get("level"), r.get("axis"), r.get("treatment"))
        if key in seen:
            continue
        seen.add(key)
        body_rows.append([
            r["phase"],
            MODEL_LABELS[r["model"]],
            DATASET_LABELS[r["dataset"]],
            f"L{r['level']}" if r["level"] else "--",
            r.get("axis") or r.get("treatment") or "--",
            f"{r['val_acc_mean'] * 100:.2f}",
            f"{r['val_acc_std'] * 100:.2f}",
            str(n),
        ])
        csv_rows.append({
            "phase": r["phase"],
            "model": r["model"],
            "dataset": r["dataset"],
            "level": r["level"],
            "axis_or_treatment": r.get("axis") or r.get("treatment") or "",
            "val_acc_mean": r["val_acc_mean"],
            "val_acc_std": r["val_acc_std"],
            "n_seeds": n,
        })

    # Sort: phase, model, dataset
    body_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    _tex_table(
        out_name="tab_X_multiseed.tex",
        caption="Multi-seed audit: validation accuracy at L3 across seeds $\\{42, 43, 44\\}$, on the 24 L3 headline cells. Standard deviation is reported in percentage points.",
        label="tab:multiseed",
        header_cells=header,
        body_rows=body_rows,
        column_spec="lllllrrr",
    )
    _csv("tab_X_multiseed.csv",
         ["phase", "model", "dataset", "level", "axis_or_treatment", "val_acc_mean", "val_acc_std", "n_seeds"],
         csv_rows)


# ---------------------------------------------------------------------------
# Phase D recovery delta table (T3 effect on L3): mean Delta vs Phase B L3.
# ---------------------------------------------------------------------------

def table_XI_attribution(rows: list[dict]) -> None:
    """Per-axis attribution at L3: how much does each axis cost in accuracy?"""
    L = _cell_lookup(rows)
    axes = ("resolution", "blur", "noise", "salt_pepper", "saturation")
    header = ["Model", "Dataset"] + [AXIS_LABELS[a] for a in axes] + ["Combined (B L3)"]
    body_rows = []
    csv_rows = []
    for m in MODELS_ORDER:
        for d in DATASETS_ORDER:
            line = [MODEL_LABELS[m], DATASET_LABELS[d]]
            clean = L.get(("A", d, None, None, None, m))
            clean_acc = (clean or {}).get("val_acc")
            for ax in axes:
                r = L.get(("C", d, 3, ax, None, m))
                acc = (r or {}).get("val_acc")
                if clean_acc is None or acc is None:
                    line.append("--")
                else:
                    delta = (acc - clean_acc) * 100
                    line.append(f"{delta:+.2f}")
                    csv_rows.append({"model": m, "dataset": d, "axis": ax,
                                     "val_acc_clean": clean_acc,
                                     "val_acc_axis": acc,
                                     "delta_pp": delta})
            comb = L.get(("B", d, 3, None, None, m))
            comb_acc = (comb or {}).get("val_acc")
            if clean_acc is None or comb_acc is None:
                line.append("--")
            else:
                line.append(f"{(comb_acc - clean_acc) * 100:+.2f}")
            body_rows.append(line)

    _tex_table(
        out_name="tab_XI_attribution.tex",
        caption="Per-axis accuracy attribution at L3: validation-accuracy drop (percentage points) relative to the Phase A clean baseline. \"Combined\" is the Phase B L3 cell (all five axes active simultaneously).",
        label="tab:attribution",
        header_cells=header,
        body_rows=body_rows,
        column_spec="ll" + "r" * (len(axes) + 1),
    )
    _csv("tab_XI_attribution.csv",
         ["model", "dataset", "axis", "val_acc_clean", "val_acc_axis", "delta_pp"], csv_rows)


def main() -> None:
    rows = _load_rows()
    table_II_levels()
    table_III_pipelines(rows)
    table_IV_phase_a(rows)
    table_V_phase_b(rows)
    table_VI_phase_c(rows)
    table_VII_phase_d(rows)
    table_VIII_phase_b2(rows)
    table_IX_phase_c2(rows)
    table_X_multiseed(rows)
    table_XI_attribution(rows)
    print(f"[build_tables] wrote {len(list(TEX_DIR.glob('*.tex')))} .tex files to {TEX_DIR}")
    print(f"[build_tables] wrote {len(list(CSV_DIR.glob('*.csv')))} .csv files to {CSV_DIR}")


if __name__ == "__main__":
    main()
