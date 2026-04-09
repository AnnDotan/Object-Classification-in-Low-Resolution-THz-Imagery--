#!/usr/bin/env python3
"""
Dashboard auto-updater — watches runs/systematic/ for new completed experiments
and regenerates the experiment plan dashboard automatically.

Usage:
    python watch_dashboard.py              # check every 60 seconds (default)
    python watch_dashboard.py --interval 30  # check every 30 seconds

Press Ctrl+C to stop.
"""

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def count_completed_runs():
    """Count runs in runs/systematic/ that have metrics.csv with data."""
    sys_dir = Path("runs") / "systematic"
    if not sys_dir.exists():
        return 0, set()
    completed = set()
    for d in sys_dir.iterdir():
        if d.is_dir():
            metrics = d / "metrics.csv"
            log = d / "log.txt"
            # A run is "complete" if log contains "Total time:"
            if metrics.exists() and metrics.stat().st_size > 50:
                if log.exists() and "Total time:" in log.read_text(encoding="utf-8", errors="ignore"):
                    completed.add(d.name)
    return len(completed), completed


def regenerate_dashboard():
    """Call the experiment plan dashboard generator."""
    from src.tools.generate_experiment_plan_dashboard import main as gen
    gen()


def main():
    parser = argparse.ArgumentParser(description="Watch and auto-update experiment plan dashboard")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between checks (default: 60)")
    args = parser.parse_args()

    print("=" * 60)
    print("  DASHBOARD AUTO-UPDATER")
    print(f"  Watching: runs/systematic/")
    print(f"  Output:   artifacts/dashboard_experiment_plan.html")
    print(f"  Interval: {args.interval}s")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    last_count, last_runs = count_completed_runs()
    print(f"\n[INIT] {last_count} completed runs detected")

    while True:
        time.sleep(args.interval)
        current_count, current_runs = count_completed_runs()

        if current_runs != last_runs:
            new_runs = current_runs - last_runs
            print(f"\n[UPDATE] {current_count} completed runs "
                  f"(+{len(new_runs)} new: {', '.join(sorted(new_runs))})")
            try:
                regenerate_dashboard()
                print(f"[OK] Dashboard updated at {time.strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"[ERROR] Dashboard generation failed: {e}")
            last_count = current_count
            last_runs = current_runs
        else:
            print(f"  [{time.strftime('%H:%M:%S')}] No new completed runs ({current_count} total)", end="\r")


if __name__ == "__main__":
    main()
