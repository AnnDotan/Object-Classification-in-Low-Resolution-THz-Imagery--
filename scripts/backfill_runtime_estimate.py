"""Backfill `runtime_s` / `started_at` / `finished_at` for legacy run dirs.

The four Phase A baselines preserved across the 2026-05-12 state reset were
trained on the 4050 box, where `LegacyJSONMetricsCallback` did not yet
record wallclock fields. Their `metrics.json` therefore reads `runtime_s:
null`, and the dashboard's Runtime column renders "—" on those rows.

This script derives an approximate runtime from filesystem mtimes:

  started_at  := mtime(run_config.txt)   # written at the top of run_experiment
  finished_at := mtime(metrics.json)     # written by on_train_end
  runtime_s   := finished_at - started_at

The values are marked with `runtime_source: "file_mtime_estimate"` so a
downstream reader can tell them apart from the precise values Lightning
writes on freshly-run cells. Cells that already carry `runtime_s` are
skipped — the script never overwrites real data.

Usage:
    python scripts/backfill_runtime_estimate.py
    python scripts/backfill_runtime_estimate.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNS_ROOT = _REPO_ROOT / "runs" / "final"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat()


def backfill_cell(run_dir: Path, dry_run: bool) -> tuple[str, str]:
    """Return (status, message). Status is one of: backfilled / skipped / missing."""
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.txt"
    if not metrics_path.exists():
        return ("missing", f"no metrics.json in {run_dir.name}")
    if not config_path.exists():
        return ("missing", f"no run_config.txt in {run_dir.name}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    has_runtime = isinstance(metrics.get("runtime_s"), (int, float))
    if has_runtime:
        return ("skipped", f"{run_dir.name}: already has runtime_s={metrics['runtime_s']}")

    started_ts = config_path.stat().st_mtime
    finished_ts = metrics_path.stat().st_mtime
    runtime_s = float(finished_ts - started_ts)
    if runtime_s <= 0:
        return ("missing", f"{run_dir.name}: non-positive runtime delta ({runtime_s})")

    metrics["runtime_s"] = runtime_s
    metrics["started_at"] = _iso(started_ts)
    metrics["finished_at"] = _iso(finished_ts)
    metrics["runtime_source"] = "file_mtime_estimate"

    if not dry_run:
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return (
        "backfilled",
        f"{run_dir.name}: runtime_s={runtime_s:.0f}s "
        f"({runtime_s/60:.1f}m), started_at={metrics['started_at']}",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-root", default=str(_RUNS_ROOT))
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be patched but do not write.")
    args = p.parse_args(argv)

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"ERROR: runs-root does not exist: {runs_root}", file=sys.stderr)
        return 1

    n_back = n_skip = n_miss = 0
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        status, msg = backfill_cell(run_dir, dry_run=args.dry_run)
        if status == "backfilled":
            n_back += 1
        elif status == "skipped":
            n_skip += 1
        else:
            n_miss += 1
        print(f"[{status}] {msg}")

    suffix = " (dry-run)" if args.dry_run else ""
    print(f"\nbackfilled={n_back}, skipped={n_skip}, missing={n_miss}{suffix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
