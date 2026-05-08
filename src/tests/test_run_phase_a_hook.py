"""Phase A Final_Exp.md refresh hook (US-005).

Verifies:
  (a) refresh_final_exp_hook calls update_final_exp.main exactly once per cell.
  (b) On rc != 0 the hook raises (so the runner logs a visible WARN).
  (c) End-to-end: running 1 cell with the real hook + a fake runs/final/ tree
      writes Final_Exp.md at the expected path.
  (d) Hook failure does NOT abort subsequent cells (re-asserts the runner's
      post_hook resilience contract from US-004 with the production hook).

Run: ``python -m src.tests.test_run_phase_a_hook``
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "scripts_run_phase_a",
        _REPO_ROOT / "scripts" / "run_phase_a.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _good_rows(best: float) -> list[dict]:
    losses = [1.5, 1.3, 1.1, 0.95, 0.85, 0.78, 0.74, 0.72]
    accs = [0.40, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85, best]
    return [{
        "epoch": i + 1,
        "train_loss": losses[i] - 0.05,
        "train_acc": accs[i] + (0.05 if i == 7 else 0.02),
        "val_loss": losses[i],
        "val_acc": accs[i],
        "lr": 1e-3,
    } for i in range(8)]


def _stage_run_dir(runs_root: Path, tag: str, *, best_val_acc: float) -> Path:
    rd = runs_root / tag
    rd.mkdir(parents=True, exist_ok=True)
    rows = _good_rows(best_val_acc)
    _write_metrics_csv(rd / "metrics.csv", rows)
    (rd / "metrics.json").write_text(json.dumps({
        "best_val_acc": best_val_acc,
        "best_epoch": 8,
        "last_val_acc": best_val_acc,
        "last_train_acc": rows[-1]["train_acc"],
        "epochs_run": 8,
    }), encoding="utf-8")
    (rd / "log.txt").write_text("[lightning] run_name=stub\n", encoding="utf-8")
    return rd


def _check_hook_calls_update_final_exp_once() -> None:
    """Patch _load_update_final_exp so we can count main() calls."""
    runner = _load_runner()
    calls: list = []

    class _FakeMod:
        @staticmethod
        def main(argv):
            calls.append(argv)
            return 0

    original = runner._load_update_final_exp
    runner._load_update_final_exp = lambda: _FakeMod()
    try:
        runner.refresh_final_exp_hook({"tag": "final_clean_resnet50_mnist"})
    finally:
        runner._load_update_final_exp = original

    assert len(calls) == 1, f"expected 1 call, got {len(calls)}"
    assert calls[0] == [], f"hook should pass empty argv (use defaults), got {calls[0]!r}"
    print("OK [hook-calls-once] — refresh_final_exp_hook invokes update_final_exp.main once.")


def _check_hook_reraises_on_nonzero_rc() -> None:
    runner = _load_runner()

    class _FailingMod:
        @staticmethod
        def main(_argv):
            return 1

    original = runner._load_update_final_exp
    runner._load_update_final_exp = lambda: _FailingMod()
    try:
        runner.refresh_final_exp_hook({"tag": "final_clean_resnet50_mnist"})
    except RuntimeError as e:
        assert "rc=1" in str(e), f"error should name rc, got: {e}"
        assert "final_clean_resnet50_mnist" in str(e)
        print("OK [hook-reraises-on-nonzero] — hook raises RuntimeError when main returns rc != 0.")
        runner._load_update_final_exp = original
        return
    finally:
        runner._load_update_final_exp = original
    raise AssertionError("expected RuntimeError on rc != 0")


def _check_end_to_end_writes_final_exp_md() -> None:
    """Real hook writes Final_Exp.md when invoked with a staged runs/final/ tree.

    update_final_exp.main uses CWD-relative defaults (runs/final, Final_Exp.md),
    so we chdir into a temp workspace.
    """
    runner = _load_runner()

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        runs_root = ws / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Stage one good cell
        _stage_run_dir(runs_root, "final_clean_resnet50_mnist", best_val_acc=0.998)

        old_cwd = os.getcwd()
        os.chdir(ws)
        try:
            # Drive the runner's path with a fake run_cell that just returns the staged dir
            def fake_run_cell(tag, *, mode="full", engine="lightning"):
                return runs_root / tag

            exit_code, verdicts = runner.run_phase_a(
                ["final_clean_resnet50_mnist"],
                mode="pilot", engine="lightning",
                runs_root=runs_root,
                run_cell_fn=fake_run_cell,
                post_cell_hook=runner.refresh_final_exp_hook,
            )
        finally:
            os.chdir(old_cwd)

        final_exp = ws / "Final_Exp.md"
        assert exit_code == 0, f"expected continue, got {exit_code}"
        assert final_exp.exists(), f"Final_Exp.md not written at {final_exp}"
        body = final_exp.read_text(encoding="utf-8")
        assert "Phase A" in body
        assert "final_clean_resnet50_mnist" in body
        assert "Complete" in body, "expected Complete status for the staged cell"
    print("OK [end-to-end] — real hook writes Final_Exp.md with the staged cell as Complete.")


def _check_hook_failure_does_not_halt_runner() -> None:
    """Re-assert post-hook resilience with the production hook + a forced failure."""
    runner = _load_runner()

    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)

        cells = ["final_clean_resnet50_mnist", "final_clean_densenet121_mnist"]
        plan = {t: 0.998 for t in cells}

        def fake_run_cell(tag, *, mode="full", engine="lightning"):
            return _stage_run_dir(runs_root, tag, best_val_acc=plan[tag])

        # Force the hook to raise — runner must keep going.
        def failing_hook(_v):
            raise RuntimeError("synthetic update_final_exp failure")

        exit_code, verdicts = runner.run_phase_a(
            cells, mode="pilot",
            runs_root=runs_root,
            run_cell_fn=fake_run_cell,
            post_cell_hook=failing_hook,
        )

        assert exit_code == 0, "hook failure should not halt the runner"
        assert len(verdicts) == 2, f"both cells should have run, got {len(verdicts)}"
    print("OK [hook-failure-tolerated] — hook raising does not abort the next cell.")


def main() -> int:
    _check_hook_calls_update_final_exp_once()
    _check_hook_reraises_on_nonzero_rc()
    _check_end_to_end_writes_final_exp_md()
    _check_hook_failure_does_not_halt_runner()
    return 0


if __name__ == "__main__":
    sys.exit(main())
