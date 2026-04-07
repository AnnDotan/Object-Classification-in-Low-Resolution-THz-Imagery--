#!/usr/bin/env python3
"""
Refresh all experiment dashboards.

Usage:
    python src/tools/refresh_dashboards.py

    (Must be run from the repository root directory)

This script:
1. Updates the run summary CSV from the runs directory
2. Regenerates the basic interactive dashboard
3. Regenerates the advanced dashboard with learning curves
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and report status."""
    print(f"\n[STEP] {description}")
    print(f"       Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        if result.returncode == 0:
            print(f"       [SUCCESS]")
            return True
        else:
            print(f"       [FAILED] Exit code: {result.returncode}")
            return False
    except Exception as e:
        print(f"       [ERROR] {e}")
        return False


def main():
    print("=" * 70)
    print("DASHBOARD REFRESH TOOL")
    print("=" * 70)

    steps = [
        (["python", "src/tools/summarize_runs.py"], "Generate run summary from experiments"),
        (["python", "src/tools/generate_dashboard.py"], "Generate basic interactive dashboard"),
        (["python", "src/tools/generate_advanced_dashboard.py"], "Generate advanced dashboard with learning curves"),
    ]

    success_count = 0
    for cmd, description in steps:
        if run_command(cmd, description):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"COMPLETED: {success_count}/{len(steps)} steps successful")
    print("=" * 70)

    if success_count == len(steps):
        print("\n[SUCCESS] All dashboards refreshed!")
        print("\nOpen in your browser:")
        print("  - Basic:    artifacts/dashboard.html")
        print("  - Advanced: artifacts/dashboard_advanced.html")
        return 0
    else:
        print("\n[WARNING] Some steps failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
