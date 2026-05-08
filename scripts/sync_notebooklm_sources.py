"""One-shot helper: refresh THz Project notebook sources from local files.

For each path in FILES_TO_SYNC:
  1. If a source with the same title already exists in the notebook, delete it.
  2. Upload the file as a text source with the relative path as its title.

Usage:
    python scripts/sync_notebooklm_sources.py [--cleanup-only]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FILES_TO_SYNC: list[str] = [
    "Final_Exp.md",
    "README.md",
    "src/config/__init__.py",
    "src/config/paths.py",
    "src/experiments/matrix.py",
    "src/models/transnext_weights.py",
    "src/tests/test_dashboard.py",
    "src/tools/build_final_dashboard.py",
    "scripts/fetch_transnext_weights.py",
    "scripts/update_final_exp.py",
    "src/experiments/cells.py",
    "src/experiments/run_status.py",
    "src/tests/test_update_final_exp.py",
    "src/tools/render_curve_thumbs.py",
    "agents/DATA_ARCHITECT.md",
    "agents/DESIGNER.md",
    "agents/EXECUTOR.md",
    "agents/LIBRARIAN.md",
    "agents/MASTER.md",
    "agents/NOTEBOOKLM_SYNC.md",
    "agents/OPTIMIZER.md",
    "agents/REPORTER.md",
    "agents/SECURITY.md",
    "agents/SYNCHRONIZER.md",
    "agents/VALIDATOR.md",
]


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=capture,
        encoding="utf-8",
        errors="replace",
    )


def list_sources() -> list[dict]:
    proc = run(["python", "-m", "notebooklm", "source", "list", "--json"], capture=True)
    if proc.returncode != 0:
        print("Failed to list sources:", proc.stderr, file=sys.stderr)
        sys.exit(1)
    out = proc.stdout
    start = out.find("{")
    return json.loads(out[start:]).get("sources", [])


def delete_id(source_id: str, label: str) -> None:
    proc = run(
        ["python", "-m", "notebooklm", "source", "delete", "-y", source_id],
        capture=True,
    )
    if proc.returncode == 0:
        print(f"  deleted: {label} ({source_id[:8]})")
    else:
        print(f"  delete FAILED: {label} ({source_id[:8]}) -> {(proc.stderr or proc.stdout).strip()}")


def add_file(rel_path: str, title: str) -> bool:
    abs_path = (REPO_ROOT / rel_path).as_posix()
    proc = run(
        [
            "python", "-m", "notebooklm", "source", "add",
            abs_path, "--type", "text", "--title", title,
        ],
        capture=True,
    )
    if proc.returncode == 0:
        print(f"  added: {title}")
        return True
    print(f"  FAILED: {title}\n    {(proc.stderr or proc.stdout).strip()}")
    return False


def cleanup_duplicates(targets: set[str]) -> None:
    """For each target title, keep only the most recently created source; delete the rest."""
    sources = list_sources()
    by_title: dict[str, list[dict]] = defaultdict(list)
    for s in sources:
        if s["title"] in targets:
            by_title[s["title"]].append(s)
    for title, group in by_title.items():
        if len(group) <= 1:
            continue
        # Newest first
        group.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        keep = group[0]
        print(f"\n{title}: keep {keep['id'][:8]} ({keep['created_at']}); deleting {len(group) - 1} older")
        for s in group[1:]:
            delete_id(s["id"], title)


def upload() -> None:
    existing_titles = {s["title"] for s in list_sources()}
    print(f"Existing sources: {len(existing_titles)}")
    for rel in FILES_TO_SYNC:
        path = REPO_ROOT / rel
        if not path.exists():
            print(f"SKIP (missing): {rel}")
            continue
        title = rel
        print(f"\n--> {rel}")
        # Skip; cleanup pass will handle duplicates afterwards
        add_file(rel, title)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-only", action="store_true",
                        help="Only delete duplicate older sources for the listed titles.")
    args = parser.parse_args()

    if not args.cleanup_only:
        upload()
    cleanup_duplicates(set(FILES_TO_SYNC))
    return 0


if __name__ == "__main__":
    sys.exit(main())
