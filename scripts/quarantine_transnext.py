"""TransNeXt quarantine utility (US-014).

Hard-quarantine TransNeXt cells from the 186-cell campaign:

1. Detect any live python processes that look like a TransNeXt training run
   (best-effort via `psutil`; warns + aborts unless `--force`).
2. `rmtree` every directory under `runs/final/` whose name contains
   ``transnext`` (covers the canonical Phase A/B/C tags AND any sibling
   ``__v2``/timestamped artifacts).
3. Re-run the trackers so [Final_Exp.json] / [Final_Exp.md] /
   [Final_Exp.html] reflect ``Deferred`` for the 62 TransNeXt rows.

Privacy: never opens ``*.ckpt``/``*.pt``/``*.pth`` / files under
``artifacts/weights/``. Removal is by directory name only.

CLI:
    python scripts/quarantine_transnext.py --dry-run
    python scripts/quarantine_transnext.py            # interactive confirm
    python scripts/quarantine_transnext.py --force    # no prompts; kill running
    python scripts/quarantine_transnext.py --no-refresh  # skip tracker refresh
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_RUNS_ROOT = _REPO_ROOT / "runs" / "final"
# Substring match (case-insensitive) on directory name; covers
# `final_clean_transnext_base_cifar10`, `final_B_L1_transnext_*`,
# `final_C_L*_*_transnext_*`, plus any `__v2` siblings.
_QUARANTINE_NEEDLE = "transnext"


def find_quarantine_dirs(runs_root: Path = _RUNS_ROOT) -> list[Path]:
    """Return every direct child of `runs_root` whose name contains 'transnext'."""
    if not runs_root.exists():
        return []
    return sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and _QUARANTINE_NEEDLE in d.name.lower()
    )


def find_running_transnext_processes() -> list[tuple[int, str]]:
    """Return (pid, cmdline) tuples for python processes whose cmdline
    references TransNeXt training. Returns [] if psutil is unavailable
    (the script proceeds, treating "no detection" as "none running")."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return []
    hits: list[tuple[int, str]] = []
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if "python" not in name:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if not cmdline:
                continue
            if _QUARANTINE_NEEDLE in cmdline.lower():
                hits.append((int(proc.info["pid"]), cmdline))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return hits


def kill_processes(pids: list[int]) -> list[tuple[int, str]]:
    """Best-effort terminate the listed PIDs. Returns [(pid, error)] for failures."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return [(pid, "psutil unavailable") for pid in pids]
    failures: list[tuple[int, str]] = []
    for pid in pids:
        try:
            p = psutil.Process(pid)
            p.terminate()
            try:
                p.wait(timeout=5)
            except psutil.TimeoutExpired:
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            failures.append((pid, type(e).__name__))
    return failures


def _refresh_trackers() -> dict:
    """Invoke shared scripts/refresh_trackers.py module to regenerate trackers."""
    src = _REPO_ROOT / "scripts" / "refresh_trackers.py"
    spec = importlib.util.spec_from_file_location("refresh_trackers", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.refresh_all()


def quarantine(
    *,
    runs_root: Path = _RUNS_ROOT,
    dry_run: bool = False,
    force: bool = False,
    refresh: bool = True,
) -> dict:
    """Execute the quarantine. Returns a result dict suitable for tests.

    Schema:
        {
          "dirs_planned":    list[str],   # candidate dirs (always populated)
          "dirs_removed":    list[str],   # actually removed (empty when dry-run)
          "processes_found": list[(pid, cmdline)],
          "processes_killed":list[int],   # PIDs we asked to terminate
          "kill_failures":   list[(pid, error)],
          "refresh":         dict | None, # refresh_all() result, or None
          "dry_run":         bool,
          "force":           bool,
        }
    """
    dirs_planned = find_quarantine_dirs(runs_root)
    procs = find_running_transnext_processes()

    result: dict = {
        "dirs_planned": [str(p) for p in dirs_planned],
        "dirs_removed": [],
        "processes_found": [(pid, cmd) for pid, cmd in procs],
        "processes_killed": [],
        "kill_failures": [],
        "refresh": None,
        "dry_run": dry_run,
        "force": force,
    }

    if procs and not force and not dry_run:
        raise RuntimeError(
            f"Found {len(procs)} TransNeXt-like running process(es). "
            f"Re-run with --force to terminate, or stop them manually first. "
            f"PIDs: {[p[0] for p in procs]}"
        )

    if dry_run:
        return result

    if procs and force:
        pids = [p[0] for p in procs]
        result["processes_killed"] = pids
        result["kill_failures"] = kill_processes(pids)

    for d in dirs_planned:
        # Defensive: refuse to delete anything outside the configured runs_root.
        try:
            d.resolve().relative_to(runs_root.resolve())
        except ValueError:
            raise RuntimeError(
                f"Refusing to remove {d} — not under {runs_root}"
            )
        shutil.rmtree(d, ignore_errors=False)
        result["dirs_removed"].append(str(d))

    if refresh:
        try:
            result["refresh"] = _refresh_trackers()
        except Exception as e:
            result["refresh"] = {"errors": [f"{type(e).__name__}: {e}"]}

    return result


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hard-quarantine TransNeXt cells (US-014, PHASE_B_VISUAL_CORE.md).",
    )
    p.add_argument("--runs-root", default=str(_RUNS_ROOT),
                   help=f"Per-cell metrics root (default: {_RUNS_ROOT}).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be removed and exit (no side effects).")
    p.add_argument("--force", action="store_true",
                   help="Kill detected TransNeXt processes; skip interactive confirm.")
    p.add_argument("--no-refresh", action="store_true",
                   help="Skip the post-removal refresh of Final_Exp.{md,json,html}.")
    p.add_argument("--yes", action="store_true",
                   help="Bypass interactive confirmation prompt.")
    return p


def _confirm(prompt: str) -> bool:
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    runs_root = Path(args.runs_root)

    dirs = find_quarantine_dirs(runs_root)
    procs = find_running_transnext_processes()

    print(f"[quarantine] runs_root: {runs_root}")
    print(f"[quarantine] dirs to remove: {len(dirs)}")
    for d in dirs:
        print(f"  - {d}")
    print(f"[quarantine] running TransNeXt-like processes: {len(procs)}")
    for pid, cmd in procs:
        print(f"  - pid={pid}: {cmd[:120]}")

    if args.dry_run:
        print("[quarantine] --dry-run: no side effects.")
        return 0

    if not dirs and not procs:
        print("[quarantine] nothing to do.")
        if not args.no_refresh:
            r = _refresh_trackers()
            for err in r.get("errors", []):
                print(f"[refresh][WARN] {err}", file=sys.stderr)
        return 0

    if not args.yes and not args.force:
        if not _confirm(f"Remove {len(dirs)} dir(s) and refresh trackers? [y/N] "):
            print("[quarantine] aborted by user.")
            return 1

    try:
        result = quarantine(
            runs_root=runs_root,
            dry_run=False,
            force=args.force,
            refresh=not args.no_refresh,
        )
    except RuntimeError as e:
        print(f"[quarantine][ERROR] {e}", file=sys.stderr)
        return 2

    print(f"[quarantine] removed {len(result['dirs_removed'])} dir(s).")
    if result["processes_killed"]:
        print(f"[quarantine] terminated PIDs: {result['processes_killed']}")
    for pid, err in result["kill_failures"]:
        print(f"[quarantine][WARN] kill pid={pid}: {err}", file=sys.stderr)
    if result["refresh"]:
        for err in result["refresh"].get("errors", []):
            print(f"[refresh][WARN] {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
