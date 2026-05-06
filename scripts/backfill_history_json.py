"""Backfill `runs/final/<tag>/history.json` from Lightning's `metrics.csv` (US-018).

The 4 Phase A CNN cells were trained before `HistoryJSONCallback` existed,
so their per-epoch curves only live in `metrics.csv`. This script reads
that CSV (canonical schema: `epoch,train_loss,train_acc,val_loss,val_acc`)
and emits a `history.json` matching the new callback's schema, so the
dashboard's lazy Plotly drawer can render them too.

Idempotent: skips cells whose `history.json` already exists. Pass
`--force` to overwrite. Cells with no `metrics.csv` (Pending / TransNeXt
deferred) are silently skipped.

Privacy: never opens `*.ckpt`/`*.pt`/`*.pth`; reads only `metrics.csv` and
writes `history.json` next to it.

CLI:
    python scripts/backfill_history_json.py
    python scripts/backfill_history_json.py --force
    python scripts/backfill_history_json.py --tag final_clean_resnet50_cifar10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_RUNS_ROOT_DEFAULT = _REPO_ROOT / "runs" / "final"


def _safe_float(s: str) -> Optional[float]:
    """Coerce CSV cell to float-or-None; NaN/inf -> None for valid JSON."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _read_history_from_csv(csv_path: Path) -> list[dict]:
    """Parse Lightning metrics.csv into the HistoryJSONCallback row schema."""
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                epoch = int(r.get("epoch", "").strip())
            except ValueError:
                continue
            rows.append({
                "epoch": epoch,
                "train_loss": _safe_float(r.get("train_loss", "")),
                "val_loss":   _safe_float(r.get("val_loss", "")),
                "train_acc":  _safe_float(r.get("train_acc", "")),
                "val_acc":    _safe_float(r.get("val_acc", "")),
            })
    rows.sort(key=lambda r: r["epoch"])
    return rows


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tf:
        tmp = Path(tf.name)
        json.dump(payload, tf, indent=2, ensure_ascii=False)
        tf.write("\n")
    os.replace(tmp, path)


def backfill(
    *,
    runs_root: Path = _RUNS_ROOT_DEFAULT,
    tags: Optional[Iterable[str]] = None,
    force: bool = False,
) -> dict:
    """Backfill `history.json` for every cell directory under `runs_root`
    that has `metrics.csv` but no `history.json` (or all of them when
    `--force`). Returns a {written, skipped, missing_csv} summary."""
    if not runs_root.exists():
        return {"written": 0, "skipped": 0, "missing_csv": 0, "total": 0}

    if tags is None:
        cell_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir())
    else:
        cell_dirs = [runs_root / t for t in tags]

    written = skipped = missing_csv = 0
    for cell_dir in cell_dirs:
        tag = cell_dir.name
        history_path = cell_dir / "history.json"
        csv_path = cell_dir / "metrics.csv"
        if not csv_path.exists():
            missing_csv += 1
            continue
        if history_path.exists() and not force:
            skipped += 1
            continue
        rows = _read_history_from_csv(csv_path)
        if not rows:
            # Empty CSV — don't write a useless 0-row history.json.
            missing_csv += 1
            continue
        _write_atomic(history_path, {
            "schema_version": 1,
            "tag": tag,
            "history": rows,
        })
        written += 1
    return {
        "written": written,
        "skipped": skipped,
        "missing_csv": missing_csv,
        "total": len(cell_dirs),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backfill runs/final/<tag>/history.json from metrics.csv (US-018).",
    )
    p.add_argument(
        "--runs-root", default=str(_RUNS_ROOT_DEFAULT),
        help=f"Per-cell metrics root (default: {_RUNS_ROOT_DEFAULT}).",
    )
    p.add_argument(
        "--tag", action="append", dest="tags", default=None,
        help="Restrict to specific tag(s); repeat for multiple. Default: all.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing history.json files.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    summary = backfill(
        runs_root=Path(args.runs_root),
        tags=args.tags,
        force=args.force,
    )
    print(
        f"backfill: {summary['written']} written, {summary['skipped']} skipped "
        f"(history.json present), {summary['missing_csv']} skipped (no metrics.csv) "
        f"-> {args.runs_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
