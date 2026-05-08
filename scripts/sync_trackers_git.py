"""SYNCHRONIZER phase-boundary git sync (US-016).

After every Phase A/B/C completes, the runner asks this module to:
  1. Stage tracker files (Final_Exp.md, artifacts/Final_Exp.html, progress.txt).
  2. Commit them with a deterministic message.
  3. Push to the current branch's upstream.

Constraints (PRD US-016):
  - Pathspec is explicit — never `git add .` or `git add -A` (would risk
    committing weight files or runs/ artifacts the .gitignore should be
    excluding but which we don't want to depend on).
  - No --force, --no-verify, or interactive flags.
  - Fail-soft: any subprocess failure logs a [sync][WARN] and is recorded
    in the result dict, never raises, never halts the campaign.
  - No-op when no tracker files are staged after `git add` (nothing to commit).

Forbidden paths (verified by the explicit pathspec — these are NEVER staged):
  - *.ckpt / *.pt / *.pth
  - artifacts/weights/**
  - runs/final/**
  - artifacts/optuna_thz.db
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


# Explicit pathspec — never expand with `git add -A`.
TRACKER_PATHS: tuple[str, ...] = (
    "Final_Exp.md",
    "artifacts/Final_Exp.html",
    "progress.txt",
)

# Forbidden flags — assert they never appear in any subprocess call we make.
FORBIDDEN_FLAGS: frozenset[str] = frozenset({"--force", "-f", "--no-verify", "-i"})


def _run(cmd: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Wrapper around subprocess.run with our standard fail-soft contract.

    `check=False` so non-zero rc never raises; `capture_output=True` so we can
    surface stderr in the result dict without polluting the runner's stdout.
    """
    # Defensive guard against accidental forbidden-flag injection.
    bad = [a for a in cmd if a in FORBIDDEN_FLAGS]
    assert not bad, f"refusing forbidden flag(s) in git command: {bad}"
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        check=False, capture_output=True, text=True,
    )


def _git_add(paths: tuple[str, ...], cwd: Optional[Path]) -> tuple[bool, str]:
    """Stage the explicit pathspec. Returns (ok, error_msg)."""
    # Build `git add -- <p1> <p2> ...`. `--` separates paths from options.
    cmd = ["git", "add", "--", *paths]
    cp = _run(cmd, cwd)
    if cp.returncode != 0:
        return False, f"git add failed (rc={cp.returncode}): {cp.stderr.strip()}"
    return True, ""


def _has_staged_changes(paths: tuple[str, ...], cwd: Optional[Path]) -> bool:
    """Return True iff any of `paths` has staged differences.

    Uses `git diff --cached --quiet -- <paths>` whose rc is:
      0 -> no staged diffs (no-op)
      1 -> staged diffs present
      other -> error (treat as no diffs to be safe; caller logs)
    """
    cmd = ["git", "diff", "--cached", "--quiet", "--", *paths]
    cp = _run(cmd, cwd)
    return cp.returncode == 1


def _git_commit(message: str, cwd: Optional[Path]) -> tuple[bool, str]:
    cmd = ["git", "commit", "-m", message]
    cp = _run(cmd, cwd)
    if cp.returncode != 0:
        return False, f"git commit failed (rc={cp.returncode}): {cp.stderr.strip()}"
    return True, ""


def _git_push(cwd: Optional[Path]) -> tuple[bool, str]:
    cmd = ["git", "push"]  # uses upstream tracking; no remote/branch override
    cp = _run(cmd, cwd)
    if cp.returncode != 0:
        return False, f"git push failed (rc={cp.returncode}): {cp.stderr.strip()}"
    return True, ""


def _commit_message(phase: str, n_cells: int) -> str:
    return f"chore(trackers): refresh after Phase {phase} ({n_cells} cells)"


def commit_and_push_phase_boundary(
    phase: str,
    n_cells: int,
    *,
    cwd: Optional[Path] = None,
    run_fn=None,  # injectable for tests; receives (cmd, cwd) and returns CompletedProcess
) -> dict:
    """Stage tracker files, commit if dirty, push upstream. Fail-soft.

    Args:
        phase: "A" | "B" | "C".
        n_cells: number of cells the phase contained (informational only).
        cwd: optional working directory override (for tests).
        run_fn: optional injected runner; used by tests to mock subprocess.

    Returns:
        {
          "phase":     str,
          "n_cells":   int,
          "staged":    bool,   # True if `git add` succeeded for at least one path
          "committed": bool,   # True if commit was created (False if no-op or failure)
          "pushed":    bool,   # True if push succeeded
          "errors":    list[str],
        }

    Never raises. The runner inspects `errors` to decide whether to log a WARN.
    """
    if phase not in ("A", "B", "C"):
        return {
            "phase": phase, "n_cells": n_cells,
            "staged": False, "committed": False, "pushed": False,
            "errors": [f"unknown phase {phase!r}; expected one of A/B/C"],
        }

    # Allow tests to swap subprocess.run via run_fn.
    global _run
    saved_run = None
    if run_fn is not None:
        saved_run = _run
        _run = run_fn

    out = {
        "phase": phase,
        "n_cells": int(n_cells),
        "staged": False,
        "committed": False,
        "pushed": False,
        "errors": [],
    }
    try:
        ok, err = _git_add(TRACKER_PATHS, cwd)
        if not ok:
            out["errors"].append(err)
            print(f"[sync][WARN] {err}")
            return out
        out["staged"] = True

        if not _has_staged_changes(TRACKER_PATHS, cwd):
            # Nothing to commit — no-op success.
            print(f"[sync] no tracker changes after Phase {phase}; skipping commit + push.")
            return out

        ok, err = _git_commit(_commit_message(phase, n_cells), cwd)
        if not ok:
            out["errors"].append(err)
            print(f"[sync][WARN] {err}")
            return out
        out["committed"] = True

        ok, err = _git_push(cwd)
        if not ok:
            out["errors"].append(err)
            print(f"[sync][WARN] {err}")
            return out
        out["pushed"] = True
        print(f"[sync] Phase {phase} tracker refresh committed and pushed "
              f"({n_cells} cells).")
        return out
    finally:
        if saved_run is not None:
            _run = saved_run
