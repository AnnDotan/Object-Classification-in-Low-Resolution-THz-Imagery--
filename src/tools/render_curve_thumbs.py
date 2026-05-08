"""Pre-render learning-curve PNGs for the 186-cell dashboard.

For every `runs/final/<tag>/metrics.csv` written by Lightning's
`CSVLogger`, draw a small (480x180) two-panel chart of `val_acc` and
`val_loss` vs. epoch, saved alongside the Original|Degraded thumb at
`artifacts/dashboard_thumbs/<tag>_curve.png`. The dashboard's
click-to-toggle `<details>` element loads it on demand.

This module is torch-free; it only needs pandas + matplotlib.

CLI:
    python -m src.tools.render_curve_thumbs              # all available, skip existing
    python -m src.tools.render_curve_thumbs --force      # regenerate
    python -m src.tools.render_curve_thumbs --tags T1 T2 # subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

# Configure matplotlib for headless use before pyplot is imported elsewhere.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.cells import all_cells  # noqa: E402


THUMB_DIR = Path("artifacts/dashboard_thumbs")
RUNS_ROOT = Path("runs/final")


def _read_curve_data(metrics_csv: Path) -> Optional[dict]:
    """Parse metrics.csv into per-epoch val_acc / val_loss arrays.

    Returns None if the file is missing, unreadable, or has no val rows.
    """
    if not metrics_csv.exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        # Fallback: hand-parse with csv.DictReader
        return _read_curve_data_stdlib(metrics_csv)

    try:
        df = pd.read_csv(metrics_csv)
    except (OSError, ValueError):
        return None

    if "epoch" not in df.columns:
        return None

    cols = {c: c for c in df.columns}
    val_acc_col = cols.get("val_acc") or cols.get("val/acc")
    val_loss_col = cols.get("val_loss") or cols.get("val/loss")
    train_acc_col = cols.get("train_acc") or cols.get("train/acc")
    train_loss_col = cols.get("train_loss") or cols.get("train/loss")

    if val_acc_col is None and val_loss_col is None:
        return None

    out: dict = {"epoch": df["epoch"].astype(float).to_numpy()}
    for label, col in [
        ("val_acc", val_acc_col), ("val_loss", val_loss_col),
        ("train_acc", train_acc_col), ("train_loss", train_loss_col),
    ]:
        if col is not None:
            mask = df[col].notna()
            out[label] = (df.loc[mask, "epoch"].astype(float).to_numpy(),
                          df.loc[mask, col].astype(float).to_numpy())
    return out


def _read_curve_data_stdlib(metrics_csv: Path) -> Optional[dict]:
    import csv

    rows: list[dict] = []
    try:
        with open(metrics_csv, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    except OSError:
        return None
    if not rows or "epoch" not in rows[0]:
        return None

    out: dict = {"epoch": []}
    series: dict[str, tuple[list, list]] = {
        "val_acc": ([], []), "val_loss": ([], []),
        "train_acc": ([], []), "train_loss": ([], []),
    }
    for r in rows:
        try:
            ep = float(r["epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        for key in series:
            v = r.get(key) or r.get(key.replace("_", "/"))
            if v in (None, "", "NaN", "nan"):
                continue
            try:
                series[key][0].append(ep)
                series[key][1].append(float(v))
            except ValueError:
                series[key][0].pop()  # roll back the epoch we just appended

    if not any(xs for xs, _ in series.values()):
        return None
    return {
        "epoch": [],
        **{k: (xs, ys) for k, (xs, ys) in series.items() if xs},
    }


def _plot(data: dict, out_path: Path) -> None:
    fig, (ax_acc, ax_loss) = plt.subplots(
        1, 2, figsize=(4.8, 1.8), dpi=100, facecolor="#181b22",
    )
    for ax in (ax_acc, ax_loss):
        ax.set_facecolor("#0f1115")
        ax.tick_params(colors="#9aa3b2", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#2a3142")
        ax.grid(True, color="#2a3142", linewidth=0.5, alpha=0.6)

    if "train_acc" in data:
        xs, ys = data["train_acc"]
        ax_acc.plot(xs, ys, color="#a5f3a8", linewidth=1.0, label="train")
    if "val_acc" in data:
        xs, ys = data["val_acc"]
        ax_acc.plot(xs, ys, color="#60a5fa", linewidth=1.2, label="val")
    ax_acc.set_title("accuracy", color="#cdd3df", fontsize=8)
    ax_acc.set_ylim(0, 1.0)
    ax_acc.legend(facecolor="#181b22", edgecolor="#2a3142",
                  labelcolor="#cdd3df", fontsize=6, loc="lower right")

    if "train_loss" in data:
        xs, ys = data["train_loss"]
        ax_loss.plot(xs, ys, color="#fed7aa", linewidth=1.0, label="train")
    if "val_loss" in data:
        xs, ys = data["val_loss"]
        ax_loss.plot(xs, ys, color="#fca5a5", linewidth=1.2, label="val")
    ax_loss.set_title("loss", color="#cdd3df", fontsize=8)
    ax_loss.legend(facecolor="#181b22", edgecolor="#2a3142",
                   labelcolor="#cdd3df", fontsize=6, loc="upper right")

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_curves(
    tags: Optional[Iterable[str]] = None,
    force: bool = False,
    out_dir: Path = THUMB_DIR,
    runs_root: Path = RUNS_ROOT,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = all_cells()
    if tags is not None:
        tag_set = set(tags)
        cells = [c for c in cells if c.tag in tag_set]
        if missing := tag_set - {c.tag for c in cells}:
            raise ValueError(f"unknown tags: {sorted(missing)[:3]}")

    written = skipped = no_data = failed = 0
    for spec in cells:
        target = out_dir / f"{spec.tag}_curve.png"
        if target.exists() and not force:
            skipped += 1
            continue
        data = _read_curve_data(runs_root / spec.tag / "metrics.csv")
        if data is None:
            no_data += 1
            continue
        try:
            _plot(data, target)
            written += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {spec.tag}: {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1

    return {
        "written": written, "skipped": skipped, "no_data": no_data,
        "failed": failed, "total": len(cells), "out_dir": str(out_dir),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-render val_acc/val_loss curve thumbnails for the 186-cell dashboard."
    )
    p.add_argument("--tags", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-dir", type=str, default=str(THUMB_DIR))
    p.add_argument("--runs-root", type=str, default=str(RUNS_ROOT))
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    summary = render_curves(
        tags=args.tags, force=args.force,
        out_dir=Path(args.out_dir), runs_root=Path(args.runs_root),
    )
    print(
        f"curves: {summary['written']} written, {summary['skipped']} skipped, "
        f"{summary['no_data']} no-data, {summary['failed']} failed "
        f"(of {summary['total']}) -> {summary['out_dir']}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
