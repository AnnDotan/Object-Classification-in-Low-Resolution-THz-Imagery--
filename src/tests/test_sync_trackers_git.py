"""SYNCHRONIZER phase-boundary git sync test (US-016).

Mocks subprocess.run via the module's run_fn injection point and verifies:
  (a) Happy path: git add (rc=0) -> diff --cached --quiet (rc=1, staged) ->
      git commit -m "chore(trackers): refresh after Phase A (6 cells)" -> git push.
  (b) Forbidden flags (--force, --no-verify, -f, -i) NEVER appear in any cmd.
  (c) No-op path: git add (rc=0) -> diff --cached --quiet (rc=0, no diffs) ->
      no commit, no push, no errors.
  (d) Push-failure: git add ok, commit ok, push rc=1 -> committed=True,
      pushed=False, errors populated, no exception raised.
  (e) Unknown phase: returns errors list, no subprocess invoked.
  (f) Pathspec includes the 3 tracker paths and excludes weight files.
  (g) Commit message format matches "chore(trackers): refresh after Phase {X} ({n} cells)".

Run: ``python -m src.tests.test_sync_trackers_git``
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scripts_sync_trackers_git",
        _REPO_ROOT / "scripts" / "sync_trackers_git.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_recorder(rc_map: dict[str, int]):
    """Return (run_fn, calls) where run_fn dispatches based on subcommand.

    `rc_map` maps subcommand keys ("add", "diff", "commit", "push") to return codes.
    """
    calls: list[list[str]] = []

    def run_fn(cmd, cwd=None):
        calls.append(list(cmd))
        # Validate forbidden flags
        bad = [a for a in cmd if a in mod.FORBIDDEN_FLAGS]
        if bad:
            raise AssertionError(f"forbidden flag in cmd: {bad}")
        # Dispatch on git subcommand
        sub = cmd[1] if len(cmd) > 1 and cmd[0] == "git" else None
        if sub == "add":
            rc = rc_map.get("add", 0)
        elif sub == "diff":
            rc = rc_map.get("diff", 1)  # default: changes staged
        elif sub == "commit":
            rc = rc_map.get("commit", 0)
        elif sub == "push":
            rc = rc_map.get("push", 0)
        else:
            rc = 0
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout="", stderr="")

    return run_fn, calls


# Module reference set in main() before any test uses it.
mod = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _check_happy_path() -> None:
    run_fn, calls = _make_recorder({"add": 0, "diff": 1, "commit": 0, "push": 0})
    result = mod.commit_and_push_phase_boundary("A", 6, run_fn=run_fn)

    assert result["phase"] == "A"
    assert result["n_cells"] == 6
    assert result["staged"] is True
    assert result["committed"] is True
    assert result["pushed"] is True
    assert result["errors"] == []
    # Sequence: add, diff, commit, push
    sequence = [c[1] for c in calls]
    assert sequence == ["add", "diff", "commit", "push"], f"sequence wrong: {sequence}"
    print("OK [happy] -- add -> diff(staged) -> commit -> push, all rc=0.")


def _check_no_forbidden_flags() -> None:
    """Forbidden flags must NEVER appear in any subprocess call."""
    run_fn, calls = _make_recorder({"add": 0, "diff": 1, "commit": 0, "push": 0})
    mod.commit_and_push_phase_boundary("B", 30, run_fn=run_fn)
    for cmd in calls:
        assert not any(a in mod.FORBIDDEN_FLAGS for a in cmd), (
            f"forbidden flag found in cmd: {cmd}"
        )
    print("OK [no-forbidden-flags] -- no --force/--no-verify/-f/-i in any cmd.")


def _check_pathspec_explicit_and_safe() -> None:
    """git add must use the explicit 3-path pathspec, never `git add .` / `-A`."""
    run_fn, calls = _make_recorder({"add": 0, "diff": 1, "commit": 0, "push": 0})
    mod.commit_and_push_phase_boundary("A", 6, run_fn=run_fn)
    add_cmd = next(c for c in calls if c[1] == "add")
    assert add_cmd[2] == "--", f"expected '--' separator before paths: {add_cmd}"
    paths_in_cmd = add_cmd[3:]
    expected = list(mod.TRACKER_PATHS)
    assert paths_in_cmd == expected, f"pathspec mismatch: {paths_in_cmd} vs {expected}"
    # Verify forbidden expansions absent
    assert "." not in add_cmd
    assert "-A" not in add_cmd
    assert "--all" not in add_cmd
    # Verify weight files / runs/ never staged
    forbidden_path_substrings = (".ckpt", ".pt", ".pth", "artifacts/weights",
                                  "runs/final", "optuna_thz.db")
    for p in paths_in_cmd:
        assert not any(s in p for s in forbidden_path_substrings), (
            f"tracker pathspec includes forbidden path: {p}"
        )
    print(f"OK [pathspec] -- explicit {len(expected)}-path pathspec; no -A / no weight paths.")


def _check_no_op_when_nothing_staged() -> None:
    """diff --cached --quiet rc=0 means no diffs -> skip commit + push."""
    run_fn, calls = _make_recorder({"add": 0, "diff": 0, "commit": 0, "push": 0})
    result = mod.commit_and_push_phase_boundary("C", 150, run_fn=run_fn)

    assert result["staged"] is True
    assert result["committed"] is False  # no-op
    assert result["pushed"] is False
    assert result["errors"] == []
    sequence = [c[1] for c in calls]
    assert sequence == ["add", "diff"], f"expected only add+diff, got {sequence}"
    print("OK [no-op] -- nothing staged -> skip commit + push, no error.")


def _check_push_failure_is_fail_soft() -> None:
    """Push rc != 0 must NOT raise. errors populated, returns the dict."""
    run_fn, calls = _make_recorder({"add": 0, "diff": 1, "commit": 0, "push": 1})
    result = mod.commit_and_push_phase_boundary("A", 6, run_fn=run_fn)

    assert result["staged"] is True
    assert result["committed"] is True
    assert result["pushed"] is False
    assert any("git push failed" in e for e in result["errors"]), result["errors"]
    print("OK [push-fail-soft] -- push rc=1 returns errors, never raises.")


def _check_commit_failure_is_fail_soft() -> None:
    run_fn, calls = _make_recorder({"add": 0, "diff": 1, "commit": 1, "push": 0})
    result = mod.commit_and_push_phase_boundary("B", 30, run_fn=run_fn)

    assert result["staged"] is True
    assert result["committed"] is False
    assert result["pushed"] is False
    assert any("git commit failed" in e for e in result["errors"])
    # Push must NOT have run after commit failed
    sequence = [c[1] for c in calls]
    assert "push" not in sequence
    print("OK [commit-fail-soft] -- commit rc=1 -> push skipped, no exception.")


def _check_add_failure_is_fail_soft() -> None:
    run_fn, calls = _make_recorder({"add": 1, "diff": 0, "commit": 0, "push": 0})
    result = mod.commit_and_push_phase_boundary("A", 6, run_fn=run_fn)

    assert result["staged"] is False
    assert result["committed"] is False
    assert result["pushed"] is False
    assert any("git add failed" in e for e in result["errors"])
    sequence = [c[1] for c in calls]
    assert sequence == ["add"]
    print("OK [add-fail-soft] -- git add rc=1 -> short-circuits cleanly.")


def _check_unknown_phase_rejected_without_subprocess() -> None:
    run_fn, calls = _make_recorder({})
    result = mod.commit_and_push_phase_boundary("D", 99, run_fn=run_fn)
    assert result["staged"] is False
    assert result["committed"] is False
    assert any("unknown phase" in e for e in result["errors"])
    assert calls == [], f"no subprocess should fire on unknown phase, got {calls}"
    print("OK [unknown-phase] -- phase != A/B/C rejected without subprocess fire.")


def _check_commit_message_format() -> None:
    run_fn, calls = _make_recorder({"add": 0, "diff": 1, "commit": 0, "push": 0})
    mod.commit_and_push_phase_boundary("A", 6, run_fn=run_fn)
    commit_cmd = next(c for c in calls if c[1] == "commit")
    msg = commit_cmd[commit_cmd.index("-m") + 1]
    assert msg == "chore(trackers): refresh after Phase A (6 cells)", f"got: {msg!r}"

    run_fn, calls = _make_recorder({"add": 0, "diff": 1, "commit": 0, "push": 0})
    mod.commit_and_push_phase_boundary("C", 150, run_fn=run_fn)
    commit_cmd = next(c for c in calls if c[1] == "commit")
    msg = commit_cmd[commit_cmd.index("-m") + 1]
    assert msg == "chore(trackers): refresh after Phase C (150 cells)", f"got: {msg!r}"
    print("OK [commit-msg] -- 'chore(trackers): refresh after Phase X (N cells)'.")


def main() -> int:
    global mod
    mod = _load_module()
    _check_happy_path()
    _check_no_forbidden_flags()
    _check_pathspec_explicit_and_safe()
    _check_no_op_when_nothing_staged()
    _check_push_failure_is_fail_soft()
    _check_commit_failure_is_fail_soft()
    _check_add_failure_is_fail_soft()
    _check_unknown_phase_rejected_without_subprocess()
    _check_commit_message_format()
    return 0


if __name__ == "__main__":
    sys.exit(main())
