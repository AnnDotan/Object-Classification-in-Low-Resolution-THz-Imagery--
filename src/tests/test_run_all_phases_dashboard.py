"""run_all_phases.py US-016 integration test.

Verifies the autonomous trigger + phase-boundary push wiring:
  (a) refresh_fn fires after EVERY attempted cell (success / failure / skip).
  (b) commit_and_push_phase_boundary fires EXACTLY ONCE per phase boundary,
      never per-cell.
  (c) phase boundary fires with the correct (phase, n_cells_in_phase).
  (d) Skipped cells still count toward the phase boundary (so a fully-skipped
      phase still triggers a sync).
  (e) A failure in run_cell does NOT skip refresh_fn or the phase boundary.
  (f) phase_boundary_fn raising does NOT halt the campaign.

Run: ``python -m src.tests.test_run_all_phases_dashboard``
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_all_phases_module",
        _REPO_ROOT / "run_all_phases.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------


def _check_phase_a_only_fires_one_boundary() -> None:
    """Run Phase A (6 cells) end-to-end with mocks. Exactly 6 refreshes,
    1 phase-boundary fire for Phase A."""
    runner = _load_runner()

    refresh_calls = {"n": 0}
    sync_calls: list[tuple] = []

    def fake_run_cell(tag, *, mode="full", engine="lightning"):
        return Path("/tmp/fake")  # not actually used by the runner

    def fake_refresh():
        refresh_calls["n"] += 1

    def fake_sync(phase, n_cells, **kw):
        sync_calls.append((phase, n_cells))
        return {"phase": phase, "n_cells": n_cells, "errors": [],
                "staged": True, "committed": True, "pushed": True}

    rc = runner.run_final_plan(
        phase="A", mode="full",
        skip_existing=False,
        run_cell_fn=fake_run_cell,
        refresh_fn=fake_refresh,
        phase_boundary_fn=fake_sync,
    )

    assert rc == 0
    assert refresh_calls["n"] == 6, (
        f"expected 6 refreshes (one per cell), got {refresh_calls['n']}"
    )
    assert sync_calls == [("A", 6)], f"expected single ('A', 6) fire, got {sync_calls}"
    print("OK [phase-a-one-boundary] -- 6 refreshes, 1 sync fire for Phase A.")


def _check_per_cell_refresh_never_pushes() -> None:
    """The refresh hook must run per cell; the phase boundary must NOT fire per cell."""
    runner = _load_runner()
    refresh_calls = {"n": 0}
    sync_calls: list[tuple] = []

    def fake_run_cell(tag, *, mode="full", engine="lightning"):
        return Path("/tmp/fake")

    def fake_refresh():
        refresh_calls["n"] += 1
        # No git commands here — verifies push is decoupled from per-cell refresh.

    def fake_sync(phase, n_cells, **kw):
        sync_calls.append((phase, n_cells))
        return {"errors": []}

    runner.run_final_plan(
        phase="A", mode="full", skip_existing=False,
        run_cell_fn=fake_run_cell, refresh_fn=fake_refresh,
        phase_boundary_fn=fake_sync,
    )

    # 6 refreshes; only 1 sync fire (at the boundary).
    assert refresh_calls["n"] == 6
    assert len(sync_calls) == 1
    print("OK [per-cell-refresh-no-push] -- per-cell refresh runs without invoking sync.")


def _check_skipped_cells_count_toward_phase_boundary() -> None:
    """If skip_existing=True and all 6 Phase A cells are 'complete' on disk, the
    phase boundary must STILL fire (so a fully-skipped phase still gets a sync)."""
    runner = _load_runner()
    sync_calls: list[tuple] = []

    def fake_run_cell(tag, *, mode="full", engine="lightning"):
        raise AssertionError(f"run_cell should not be called for skipped {tag}")

    def fake_refresh():
        pass

    def fake_sync(phase, n_cells, **kw):
        sync_calls.append((phase, n_cells))
        return {"errors": []}

    # Stub _has_completed_metrics to claim all cells are complete.
    original = runner._has_completed_metrics
    runner._has_completed_metrics = lambda _path: True
    try:
        runner.run_final_plan(
            phase="A", mode="full", skip_existing=True,
            run_cell_fn=fake_run_cell, refresh_fn=fake_refresh,
            phase_boundary_fn=fake_sync,
        )
    finally:
        runner._has_completed_metrics = original

    assert sync_calls == [("A", 6)], (
        f"phase boundary must fire even when all cells skipped; got {sync_calls}"
    )
    print("OK [skipped-still-fires] -- fully-skipped phase still triggers sync.")


def _check_run_cell_failure_does_not_skip_refresh() -> None:
    """A failed cell still invokes refresh_fn AND counts toward the phase boundary."""
    runner = _load_runner()
    refresh_calls = {"n": 0}
    sync_calls: list[tuple] = []

    cell_count = {"n": 0}
    def fake_run_cell(tag, *, mode="full", engine="lightning"):
        cell_count["n"] += 1
        if cell_count["n"] == 3:
            raise RuntimeError("synthetic training failure")
        return Path("/tmp/fake")

    def fake_refresh():
        refresh_calls["n"] += 1

    def fake_sync(phase, n_cells, **kw):
        sync_calls.append((phase, n_cells))
        return {"errors": []}

    rc = runner.run_final_plan(
        phase="A", mode="full", skip_existing=False,
        run_cell_fn=fake_run_cell, refresh_fn=fake_refresh,
        phase_boundary_fn=fake_sync,
    )

    assert rc == 1, "expected rc=1 due to one failed cell"
    assert refresh_calls["n"] == 6, "refresh must run for the failed cell too"
    assert sync_calls == [("A", 6)]
    print("OK [failure-does-not-skip] -- failed cell still triggers refresh + boundary.")


def _check_phase_boundary_fail_soft() -> None:
    """phase_boundary_fn raising must NOT halt the campaign (defense in depth)."""
    runner = _load_runner()

    def fake_run_cell(tag, *, mode="full", engine="lightning"):
        return Path("/tmp/fake")

    def fake_refresh():
        pass

    def explosive_sync(phase, n_cells, **kw):
        raise RuntimeError("synthetic sync explosion")

    rc = runner.run_final_plan(
        phase="A", mode="full", skip_existing=False,
        run_cell_fn=fake_run_cell, refresh_fn=fake_refresh,
        phase_boundary_fn=explosive_sync,
    )
    # Cells succeeded; sync raised but was contained.
    assert rc == 0, f"sync raising should not flip rc; got {rc}"
    print("OK [boundary-fail-soft] -- phase_boundary_fn raising doesn't halt the runner.")


def _check_all_phases_fires_three_boundaries() -> None:
    """phase='all' (186 cells) -> exactly 3 boundary fires: ('A',6), ('B',30), ('C',150)."""
    runner = _load_runner()
    sync_calls: list[tuple] = []

    def fake_run_cell(tag, *, mode="full", engine="lightning"):
        return Path("/tmp/fake")

    def fake_refresh():
        pass

    def fake_sync(phase, n_cells, **kw):
        sync_calls.append((phase, n_cells))
        return {"errors": []}

    rc = runner.run_final_plan(
        phase="all", mode="full", skip_existing=False,
        run_cell_fn=fake_run_cell, refresh_fn=fake_refresh,
        phase_boundary_fn=fake_sync,
    )

    assert rc == 0
    assert sync_calls == [("A", 6), ("B", 30), ("C", 150)], (
        f"expected 3 boundary fires in A->B->C order, got {sync_calls}"
    )
    print("OK [all-phases-three-boundaries] -- 3 boundary fires, 186 cells, A/B/C order.")


def main() -> int:
    _check_phase_a_only_fires_one_boundary()
    _check_per_cell_refresh_never_pushes()
    _check_skipped_cells_count_toward_phase_boundary()
    _check_run_cell_failure_does_not_skip_refresh()
    _check_phase_boundary_fail_soft()
    _check_all_phases_fires_three_boundaries()
    return 0


if __name__ == "__main__":
    sys.exit(main())
