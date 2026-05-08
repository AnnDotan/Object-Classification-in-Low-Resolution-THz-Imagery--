"""Phase A sequential runner test (US-004).

Mocks run_cell so the orchestration logic can be tested without actually
training. Verifies:
  (a) full-order dispatch when --only is absent (4 cells)
  (b) --only restricts to a single cell
  (c) --dry-run prints the plan and returns 0 without dispatching
  (d) green-band cell with continue -> proceeds to next cell
  (e) red-band cell halts the runner (exit 1) and skips later cells
  (f) continuation halt (e.g., overfit gap) overrides green band -> halt
  (g) run_cell raising -> captured as run_error, halts gracefully
  (h) gate_verdict.json is written at runs/final/<tag>/

Run: ``python -m src.tests.test_run_phase_a``
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib.util

_runner_spec = importlib.util.spec_from_file_location(
    "scripts_run_phase_a",
    _REPO_ROOT / "scripts" / "run_phase_a.py",
)
runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(runner)


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _make_good_csv_rows(best_val_acc: float, gap: float = 0.05) -> list[dict]:
    """8 epochs, monotone val_loss, peak at last epoch with given gap."""
    losses = [1.5, 1.3, 1.1, 0.95, 0.85, 0.78, 0.74, 0.72]
    accs = [0.40, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85, best_val_acc]
    rows = []
    for i in range(8):
        rows.append({
            "epoch": i + 1,
            "train_loss": losses[i] - 0.05,
            "train_acc": accs[i] + (gap if i == 7 else 0.02),
            "val_loss": losses[i],
            "val_acc": accs[i],
            "lr": 1e-3,
        })
    return rows


def _make_bad_trend_csv_rows(best_val_acc: float, gap: float = 0.007) -> list[dict]:
    """10 epochs simulating a converged plateau with rising val_loss in the tail.

    Best epoch is epoch 4 (val_acc=best_val_acc); epochs 5..10 plateau slightly
    below. Last 5 transitions (5..6, 6..7, 7..8, 8..9, 9..10) yield 1 of 5
    non-increasing -> trend rule fires.
    """
    losses = [1.0, 0.8, 0.6, 0.50, 0.51, 0.515, 0.52, 0.523, 0.525, 0.528]
    accs   = [0.40, 0.55, 0.70, best_val_acc,
              best_val_acc - 0.001, best_val_acc - 0.002,
              best_val_acc - 0.003, best_val_acc - 0.0035,
              best_val_acc - 0.004, best_val_acc - 0.0045]
    rows = []
    for i in range(10):
        train_acc = accs[i] + (gap if i == 3 else 0.005)
        rows.append({
            "epoch": i + 1,
            "train_loss": losses[i] - 0.05,
            "train_acc": train_acc,
            "val_loss": losses[i],
            "val_acc": accs[i],
            "lr": 1e-3,
        })
    return rows


def _stage_run_dir(runs_root: Path, tag: str, *,
                   best_val_acc: float, gap: float = 0.05,
                   include_log: bool = True, log_text: str = "",
                   bad_trend: bool = False) -> Path:
    run_dir = runs_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = (_make_bad_trend_csv_rows(best_val_acc, gap=gap)
            if bad_trend else _make_good_csv_rows(best_val_acc, gap=gap))
    _write_metrics_csv(run_dir / "metrics.csv", rows)
    # Find the actual best-val-acc epoch in the synthetic rows (good vs bad-trend
    # have different best-epoch positions).
    best_row = max(rows, key=lambda r: r["val_acc"])
    metrics = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_row["epoch"],
        "last_val_acc": rows[-1]["val_acc"],
        "last_train_acc": rows[-1]["train_acc"],
        "epochs_run": len(rows),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if include_log:
        (run_dir / "log.txt").write_text(log_text or "[lightning] run_name=stub\n",
                                          encoding="utf-8")
    return run_dir


def _make_fake_run_cell(runs_root: Path, plan: dict[str, dict]):
    """Return a fake run_cell that stages a run_dir per plan[tag] and returns it.

    plan[tag] keys: {best_val_acc, gap, raise_with (optional)}
    """
    def fake(tag: str, *, mode: str = "full", engine: str = "lightning"):
        cfg = plan[tag]
        if "raise_with" in cfg:
            raise RuntimeError(cfg["raise_with"])
        return _stage_run_dir(runs_root, tag,
                              best_val_acc=cfg["best_val_acc"],
                              gap=cfg.get("gap", 0.05))
    return fake


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def _check_full_order_dispatches_all_4_when_all_green() -> None:
    """All 4 cells return green -> exit 0, all 4 verdicts present in order."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            "final_clean_resnet50_mnist":     {"best_val_acc": 0.998},
            "final_clean_densenet121_mnist":  {"best_val_acc": 0.998},
            "final_clean_resnet50_cifar10":   {"best_val_acc": 0.92},
            "final_clean_densenet121_cifar10":{"best_val_acc": 0.94},
        }
        fake = _make_fake_run_cell(runs_root, plan)
        exit_code, verdicts = runner.run_phase_a(
            list(runner.PHASE_A_CELL_ORDER),
            mode="pilot", engine="lightning",
            runs_root=runs_root, run_cell_fn=fake,
        )

    assert exit_code == 0, f"expected 0, got {exit_code}"
    assert [v["tag"] for v in verdicts] == list(runner.PHASE_A_CELL_ORDER)
    for v in verdicts:
        assert v["decision"] == "continue"
        assert v["gate_band"] == "green"
    print("OK [all-green] — 4-cell green run completes with exit 0.")


def _check_red_band_halts_remaining_cells() -> None:
    """First cell returns red -> halt, remaining 3 cells skipped."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            "final_clean_resnet50_mnist":     {"best_val_acc": 0.85},  # red on mnist
            "final_clean_densenet121_mnist":  {"best_val_acc": 0.998},
            "final_clean_resnet50_cifar10":   {"best_val_acc": 0.92},
            "final_clean_densenet121_cifar10":{"best_val_acc": 0.94},
        }
        fake = _make_fake_run_cell(runs_root, plan)
        exit_code, verdicts = runner.run_phase_a(
            list(runner.PHASE_A_CELL_ORDER),
            mode="pilot", runs_root=runs_root, run_cell_fn=fake,
        )

    assert exit_code == 1
    assert len(verdicts) == 1, f"halt should stop after first cell, got {len(verdicts)}"
    assert verdicts[0]["gate_band"] == "red"
    assert verdicts[0]["decision"] == "halt"
    assert any("gate_band:red" in r for r in verdicts[0]["reasons"])
    print("OK [red-halts] — red band halts after first cell, later cells skipped.")


def _check_continuation_halt_overrides_green() -> None:
    """Green band but overfit gap -> halt (PRD §5 precedence)."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            "final_clean_resnet50_mnist": {"best_val_acc": 0.998, "gap": 0.20},
        }
        fake = _make_fake_run_cell(runs_root, plan)
        exit_code, verdicts = runner.run_phase_a(
            ["final_clean_resnet50_mnist"],
            mode="pilot", runs_root=runs_root, run_cell_fn=fake,
        )

    assert exit_code == 1
    v = verdicts[0]
    assert v["gate_band"] == "green"  # band is still green
    assert v["decision"] == "halt"     # but continuation halts
    assert any(r.startswith("overfit_gap") for r in v["reasons"])
    print("OK [overfit-overrides-green] — overfit gap halts even when band is green.")


def _check_run_cell_exception_captured() -> None:
    """run_cell raising -> halt with run_cell_raised reason, not crash."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            "final_clean_resnet50_mnist": {"raise_with": "synthetic CUDA OOM"},
        }
        fake = _make_fake_run_cell(runs_root, plan)
        exit_code, verdicts = runner.run_phase_a(
            ["final_clean_resnet50_mnist"],
            mode="pilot", runs_root=runs_root, run_cell_fn=fake,
        )

        assert exit_code == 1
        v = verdicts[0]
        assert v["decision"] == "halt"
        assert v["had_errors"] is True
        assert any("run_cell_raised" in r for r in v["reasons"])
        # gate_verdict.json should still be written even with no metrics files
        verdict_file = runs_root / "final_clean_resnet50_mnist" / "gate_verdict.json"
        assert verdict_file.exists(), "gate_verdict.json must be written even on crash"
    print("OK [run_cell-raises] -> exception captured, halt + verdict written.")


def _check_yellow_band_continues() -> None:
    """Yellow band (within 7pp but below 3pp) — auto-continue with warning."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            # paper(resnet50,cifar10)=0.93 -> yellow at 0.88 (-5pp).
            "final_clean_resnet50_cifar10": {"best_val_acc": 0.88},
        }
        fake = _make_fake_run_cell(runs_root, plan)
        exit_code, verdicts = runner.run_phase_a(
            ["final_clean_resnet50_cifar10"],
            mode="pilot", runs_root=runs_root, run_cell_fn=fake,
        )

    assert exit_code == 0
    assert verdicts[0]["gate_band"] == "yellow"
    assert verdicts[0]["decision"] == "continue"
    print("OK [yellow-continues] — yellow band proceeds without halting.")


def _check_verdict_file_shape() -> None:
    """gate_verdict.json must contain the required keys."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            "final_clean_densenet121_cifar10": {"best_val_acc": 0.94},
        }
        fake = _make_fake_run_cell(runs_root, plan)
        runner.run_phase_a(
            ["final_clean_densenet121_cifar10"],
            mode="pilot", runs_root=runs_root, run_cell_fn=fake,
        )
        path = runs_root / "final_clean_densenet121_cifar10" / "gate_verdict.json"
        assert path.exists()
        verdict = json.loads(path.read_text(encoding="utf-8"))

    required_keys = {
        "tag", "model", "dataset", "phase", "decision", "reasons",
        "gate_band", "gate_explanation", "paper_baseline",
        "best_val_acc", "gap", "best_epoch", "epochs_run",
        "had_errors", "error_marker", "run_dir",
    }
    missing = required_keys - set(verdict.keys())
    assert not missing, f"verdict missing keys: {missing}"
    print("OK [verdict-shape] — gate_verdict.json contains all required fields.")


def _check_trend_bad_green_small_gap_continues() -> None:
    """Conditional rule (2026-05-05): bad trend + green band + gap<=5pp -> CONTINUE.

    This is exactly the cell-2 false-positive scenario (densenet121_mnist:
    val_acc=0.9928, gap=0.7pp, plateau val_loss noise) that prompted the rule
    refinement. The runner must now auto-continue.
    """
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            # paper(densenet121,mnist)=0.995; val_acc=0.993 -> green (-0.2pp); gap=0.7pp
            "final_clean_densenet121_mnist": {"best_val_acc": 0.993, "gap": 0.007,
                                                 "bad_trend": True},
        }
        def fake(tag, *, mode="full", engine="lightning"):
            cfg = plan[tag]
            return _stage_run_dir(runs_root, tag,
                                  best_val_acc=cfg["best_val_acc"],
                                  gap=cfg["gap"],
                                  bad_trend=cfg["bad_trend"])
        exit_code, verdicts = runner.run_phase_a(
            ["final_clean_densenet121_mnist"],
            mode="pilot", runs_root=runs_root, run_cell_fn=fake,
        )

    v = verdicts[0]
    assert exit_code == 0, f"expected continue, got {exit_code}; verdict={v}"
    assert v["gate_band"] == "green"
    assert v["val_loss_trend_ok"] is False
    assert v["decision"] == "continue", (
        f"trend bad + green + small gap should continue, got: {v}"
    )
    # Warning reason should be present (non-halt) for audit trail.
    assert any(r.startswith("val_loss_trend_warning") for r in v["reasons"]), v["reasons"]
    print("OK [trend-green-small-gap] -- cell-2 plateau scenario auto-continues.")


def _check_trend_bad_green_high_gap_halts() -> None:
    """Bad trend + green band + gap > 5pp -> HALT (overfit signal in plateau)."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            # green band (-0.5pp) but gap=8pp -> halt under conditional rule
            "final_clean_densenet121_mnist": {"best_val_acc": 0.99, "gap": 0.08,
                                                 "bad_trend": True},
        }
        def fake(tag, *, mode="full", engine="lightning"):
            cfg = plan[tag]
            return _stage_run_dir(runs_root, tag,
                                  best_val_acc=cfg["best_val_acc"],
                                  gap=cfg["gap"],
                                  bad_trend=cfg["bad_trend"])
        exit_code, verdicts = runner.run_phase_a(
            ["final_clean_densenet121_mnist"],
            mode="pilot", runs_root=runs_root, run_cell_fn=fake,
        )

    v = verdicts[0]
    assert exit_code == 1
    assert v["gate_band"] == "green"  # band still green
    assert v["decision"] == "halt"
    assert any(r.startswith("val_loss_trend_with_off_band_or_high_gap") for r in v["reasons"]), v["reasons"]
    print("OK [trend-green-high-gap] -- bad trend + gap>5pp halts even when band is green.")


def _check_trend_bad_yellow_band_halts() -> None:
    """Bad trend + yellow band -> HALT regardless of gap."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            # paper(resnet50,cifar10)=0.93 -> 0.88 = yellow (-5pp); small gap; bad trend
            "final_clean_resnet50_cifar10": {"best_val_acc": 0.88, "gap": 0.02,
                                                 "bad_trend": True},
        }
        def fake(tag, *, mode="full", engine="lightning"):
            cfg = plan[tag]
            return _stage_run_dir(runs_root, tag,
                                  best_val_acc=cfg["best_val_acc"],
                                  gap=cfg["gap"],
                                  bad_trend=cfg["bad_trend"])
        exit_code, verdicts = runner.run_phase_a(
            ["final_clean_resnet50_cifar10"],
            mode="pilot", runs_root=runs_root, run_cell_fn=fake,
        )

    v = verdicts[0]
    assert exit_code == 1
    assert v["gate_band"] == "yellow"
    assert v["decision"] == "halt"
    assert any(r.startswith("val_loss_trend_with_off_band_or_high_gap") for r in v["reasons"]), v["reasons"]
    print("OK [trend-yellow] -- bad trend + yellow band halts regardless of gap.")


def _check_post_cell_hook_invoked() -> None:
    """post_cell_hook is called after each cell; failures don't abort the runner."""
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        plan = {
            "final_clean_resnet50_mnist":     {"best_val_acc": 0.998},
            "final_clean_densenet121_mnist":  {"best_val_acc": 0.998},
        }
        fake = _make_fake_run_cell(runs_root, plan)
        hook_calls: list[str] = []

        def hook(verdict):
            hook_calls.append(verdict["tag"])
            if verdict["tag"] == "final_clean_resnet50_mnist":
                raise RuntimeError("synthetic tracker failure")  # must not abort

        exit_code, verdicts = runner.run_phase_a(
            ["final_clean_resnet50_mnist", "final_clean_densenet121_mnist"],
            mode="pilot", runs_root=runs_root, run_cell_fn=fake, post_cell_hook=hook,
        )

    assert exit_code == 0, "hook failure should not halt the runner"
    assert len(verdicts) == 2
    assert hook_calls == ["final_clean_resnet50_mnist", "final_clean_densenet121_mnist"]
    print("OK [post-hook] — hook fires per cell; its failure doesn't abort the run.")


def _check_only_resolves_to_single_cell() -> None:
    tags = runner._resolve_only("resnet50_mnist")
    assert tags == ["final_clean_resnet50_mnist"]

    tags_full = runner._resolve_only(None)
    assert tags_full == list(runner.PHASE_A_CELL_ORDER)
    print("OK [only-resolves] — --only restricts to one cell; absent -> full order.")


def _check_only_rejects_unknown() -> None:
    try:
        runner._resolve_only("bogus_dataset")
    except SystemExit as e:
        assert "bogus_dataset" in str(e)
        print("OK [only-rejects] — --only with unknown pair raises SystemExit.")
        return
    raise AssertionError("expected SystemExit for unknown --only target")


def main() -> int:
    _check_only_resolves_to_single_cell()
    _check_only_rejects_unknown()
    _check_full_order_dispatches_all_4_when_all_green()
    _check_red_band_halts_remaining_cells()
    _check_continuation_halt_overrides_green()
    _check_run_cell_exception_captured()
    _check_yellow_band_continues()
    _check_verdict_file_shape()
    _check_trend_bad_green_small_gap_continues()
    _check_trend_bad_green_high_gap_halts()
    _check_trend_bad_yellow_band_halts()
    _check_post_cell_hook_invoked()
    return 0


if __name__ == "__main__":
    sys.exit(main())
