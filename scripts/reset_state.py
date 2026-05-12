"""State reset for the RTX 5070 RALPH Loop (PRD US-040).

Wipes the four stateful artifact surfaces that accumulated on the prior
RTX 4050 box so the 5070 campaign starts from a clean baseline, while
*preserving* the paper-anchored priors, the lock file, and the Phase A
clean-baseline run directories.

Reset surfaces (see PRD §4.1):

| Target                                | Action                              |
|---------------------------------------|-------------------------------------|
| ``artifacts/optuna_thz.db``           | Move to ``...db.pre-5070.bak``      |
| ``artifacts/best_hparams/*.json``     | Move to ``_archive_pre_5070/``      |
| ``artifacts/validation/*.json``       | Delete                              |
| ``artifacts/Final_Exp.json``          | Delete (regenerated downstream)     |
| ``artifacts/Final_Exp.html``          | Delete (regenerated downstream)     |

Preserved (never touched):

- ``artifacts/priors/`` (paper-anchored, frozen)
- ``artifacts/priors/**`` SHA-256s round-trip identical pre/post.
- ``requirements.lock.txt`` (cross-box pin)
- ``runs/final/final_clean_*`` (Phase A clean baselines)
- ``runs/final/final_B_*`` / ``final_C_*`` (gitignored; loop's
  ``--skip-existing`` handles re-attempts)

Usage:

    .venv-gpu\\Scripts\\python.exe scripts\\reset_state.py --dry-run
    .venv-gpu\\Scripts\\python.exe scripts\\reset_state.py --yes

Idempotency: ``--dry-run`` is always safe. ``--yes`` is destructive but
re-runnable: if a backup already exists at the destination, the script
appends a numeric suffix (``.bak.1``, ``.bak.2``, ...) so no prior backup
is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Path constants (importable from tests)
# -----------------------------------------------------------------------------

ARTIFACTS = _REPO_ROOT / "artifacts"

OPTUNA_DB = ARTIFACTS / "optuna_thz.db"
OPTUNA_DB_BACKUP = ARTIFACTS / "optuna_thz.db.pre-5070.bak"

BEST_HPARAMS_DIR = ARTIFACTS / "best_hparams"
BEST_HPARAMS_ARCHIVE = BEST_HPARAMS_DIR / "_archive_pre_5070"

VALIDATION_DIR = ARTIFACTS / "validation"

FINAL_EXP_JSON = ARTIFACTS / "Final_Exp.json"
FINAL_EXP_HTML = ARTIFACTS / "Final_Exp.html"

PRIORS_DIR = ARTIFACTS / "priors"
LOCK_FILE = _REPO_ROOT / "requirements.lock.txt"


# -----------------------------------------------------------------------------
# Action plan dataclasses
# -----------------------------------------------------------------------------


@dataclass
class Action:
    kind: str  # "move" | "delete"
    src: Path
    dst: Optional[Path] = None  # only for "move"
    note: str = ""

    def describe(self) -> str:
        if self.kind == "move":
            dst = self.dst if self.dst is not None else Path("<unknown>")
            return f"MOVE   {self.src} -> {dst}" + (f"   # {self.note}" if self.note else "")
        if self.kind == "delete":
            return f"DELETE {self.src}" + (f"   # {self.note}" if self.note else "")
        return f"UNKNOWN({self.kind}) {self.src}"


@dataclass
class ResetPlan:
    actions: list[Action] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already-clean targets

    def is_empty(self) -> bool:
        return not self.actions


# -----------------------------------------------------------------------------
# Plan construction (pure: no filesystem mutation)
# -----------------------------------------------------------------------------


def _next_backup_path(dst: Path) -> Path:
    """If ``dst`` exists, return ``dst.with_suffix(suffix + '.N')`` for the
    smallest non-colliding N >= 1; else return ``dst`` unchanged.

    Treats the full filename as the "stem" since the input may already
    carry a multi-dot extension (e.g. ``optuna_thz.db.pre-5070.bak``).
    """
    if not dst.exists():
        return dst
    n = 1
    while True:
        candidate = dst.with_name(f"{dst.name}.{n}")
        if not candidate.exists():
            return candidate
        n += 1


def build_plan(
    *,
    repo_root: Path = _REPO_ROOT,
) -> ResetPlan:
    """Inspect the filesystem and produce the action list. No I/O writes."""
    artifacts = repo_root / "artifacts"
    plan = ResetPlan()

    # 1) Optuna DB -> backup
    optuna = artifacts / "optuna_thz.db"
    if optuna.exists():
        plan.actions.append(
            Action(
                kind="move",
                src=optuna,
                dst=_next_backup_path(artifacts / "optuna_thz.db.pre-5070.bak"),
                note="Pre-5070 study; preserved for audit",
            )
        )
    else:
        plan.skipped.append(f"{optuna} (already absent)")

    # 2) best_hparams/*.json -> _archive_pre_5070/
    best_hparams = artifacts / "best_hparams"
    archive = best_hparams / "_archive_pre_5070"
    if best_hparams.is_dir():
        winner_files = sorted(p for p in best_hparams.glob("*.json") if p.is_file())
        if winner_files:
            for f in winner_files:
                plan.actions.append(
                    Action(
                        kind="move",
                        src=f,
                        dst=_next_backup_path(archive / f.name),
                        note="Pre-5070 fast-tune winner",
                    )
                )
        else:
            plan.skipped.append(f"{best_hparams}/*.json (no winner JSONs)")
    else:
        plan.skipped.append(f"{best_hparams} (directory absent)")

    # 3) validation/*.json -> delete
    validation = artifacts / "validation"
    if validation.is_dir():
        cache_files = sorted(p for p in validation.glob("*.json") if p.is_file())
        if cache_files:
            for f in cache_files:
                plan.actions.append(
                    Action(kind="delete", src=f, note="Stage-1.5 rank cache")
                )
        else:
            plan.skipped.append(f"{validation}/*.json (cache empty)")
    else:
        plan.skipped.append(f"{validation} (directory absent)")

    # 4) Final_Exp.json -> delete (regenerated by refresh_trackers)
    for target, note in (
        (artifacts / "Final_Exp.json", "regenerated by build_final_exp_json"),
        (artifacts / "Final_Exp.html", "regenerated by build_final_dashboard"),
    ):
        if target.exists():
            plan.actions.append(Action(kind="delete", src=target, note=note))
        else:
            plan.skipped.append(f"{target} (already absent)")

    return plan


# -----------------------------------------------------------------------------
# Plan execution
# -----------------------------------------------------------------------------


def execute_plan(plan: ResetPlan) -> list[str]:
    """Mutate the filesystem per ``plan``. Returns list of human-readable
    confirmations for stdout. Raises on I/O failure (caller decides policy).
    """
    confirmations: list[str] = []
    for action in plan.actions:
        if action.kind == "move":
            assert action.dst is not None  # constructed by build_plan
            action.dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(action.src), str(action.dst))
            confirmations.append(f"moved   {action.src.name} -> {action.dst}")
        elif action.kind == "delete":
            action.src.unlink()
            confirmations.append(f"deleted {action.src}")
        else:
            raise ValueError(f"Unknown action kind: {action.kind}")
    return confirmations


# -----------------------------------------------------------------------------
# Preservation guard (post-execution sanity check)
# -----------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_preserved(repo_root: Path = _REPO_ROOT) -> dict[str, str]:
    """SHA-256 every file that must survive the reset. Caller compares
    the pre/post snapshots to detect any accidental mutation.

    Files: ``artifacts/priors/**`` and ``requirements.lock.txt``.
    """
    snapshot: dict[str, str] = {}
    priors = repo_root / "artifacts" / "priors"
    if priors.is_dir():
        for f in sorted(priors.rglob("*")):
            if f.is_file():
                snapshot[str(f.relative_to(repo_root))] = _sha256_file(f)
    lock = repo_root / "requirements.lock.txt"
    if lock.is_file():
        snapshot[str(lock.relative_to(repo_root))] = _sha256_file(lock)
    return snapshot


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Reset Final_Exp state, Optuna DB, validation cache, and "
            "pre-5070 best_hparams winners. Preserves priors, lock file, "
            "and Phase A clean baselines. PRD US-040."
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Default. Print every file that would be touched and exit 0.",
    )
    g.add_argument(
        "--yes",
        action="store_true",
        help="Execute the plan. Required for any destructive action.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    # Default to dry-run unless --yes is given.
    destructive = bool(args.yes)

    # Resolve `_REPO_ROOT` at call time so tests can monkeypatch it on the module.
    plan = build_plan(repo_root=_REPO_ROOT)

    print("# US-040 Reset Plan")
    print(f"# repo_root: {_REPO_ROOT}")
    print(f"# mode: {'EXECUTE' if destructive else 'DRY RUN'}")
    print()

    if plan.is_empty():
        print("Nothing to do — state is already at the post-reset baseline.")
    else:
        for action in plan.actions:
            print(action.describe())

    if plan.skipped:
        print()
        print("# Skipped (already at baseline):")
        for s in plan.skipped:
            print(f"#   - {s}")

    print()
    print("# Preserved (never touched):")
    print(f"#   - {PRIORS_DIR}/")
    print(f"#   - {LOCK_FILE}")
    print("#   - runs/final/final_clean_* (Phase A clean baselines)")
    print("#   - runs/final/final_B_* / final_C_* (gitignored; --skip-existing handles)")

    if not destructive:
        print()
        print(
            "[dry-run] No filesystem changes made. "
            "Re-run with `--yes` to execute the plan."
        )
        return 0

    # Destructive path: snapshot preserved files, execute, verify they're unchanged.
    pre = snapshot_preserved(repo_root=_REPO_ROOT)
    confirmations = execute_plan(plan)
    post = snapshot_preserved(repo_root=_REPO_ROOT)

    print()
    print("# Executed:")
    for c in confirmations:
        print(f"  {c}")

    drifted = [k for k in pre if pre.get(k) != post.get(k)]
    missing = [k for k in pre if k not in post]
    if drifted or missing:
        print()
        print("[ERROR] Preservation contract violated:", file=sys.stderr)
        for k in drifted:
            print(f"  CHANGED:  {k}", file=sys.stderr)
        for k in missing:
            print(f"  REMOVED:  {k}", file=sys.stderr)
        return 2

    print()
    print(
        f"OK: {len(confirmations)} action(s) applied; "
        f"{len(pre)} preserved file(s) byte-identical."
    )
    print("Next: scripts/refresh_trackers.py  (rebuilds Final_Exp.{json,html})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
