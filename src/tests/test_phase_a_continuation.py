"""Phase A auto-continue evaluator test (US-003).

Builds synthetic metrics.csv files covering each halt reason and the happy
path, then asserts evaluate_continuation() returns the expected decision +
reasons.

Halt scenarios pinned:
  (a) error marker in log file
  (b) metrics.csv missing
  (c) fewer than 5 epochs (truncated run)
  (d) overfit gap >= 0.15 at best epoch
  (e) val_loss trend: fewer than 3 of last 5 transitions non-increasing
  (f) combined: multiple halt reasons stack

Continue scenario pinned:
  (g) >= 5 epochs, gap < 15pp, trend ok, no errors

Run: ``python -m src.tests.test_phase_a_continuation``
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.phase_a_gate import (
    MIN_EPOCHS,
    OVERFIT_GAP_THRESHOLD,
    TREND_NON_INCREASING_REQUIRED,
    TREND_WINDOW,
    evaluate_continuation,
)


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(epoch: int, train_loss: float, train_acc: float,
         val_loss: float, val_acc: float, lr: float = 1e-3) -> dict:
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": lr,
    }


def _happy_path_rows() -> list[dict]:
    """8 epochs of monotone-improving training, gap ~5pp at best epoch."""
    rows = []
    val_losses = [1.5, 1.3, 1.1, 0.95, 0.85, 0.78, 0.74, 0.72]
    val_accs   = [0.40, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85, 0.86]
    train_accs = [0.45, 0.60, 0.70, 0.77, 0.83, 0.87, 0.89, 0.91]
    train_losses = [1.6, 1.35, 1.1, 0.9, 0.78, 0.68, 0.62, 0.58]
    for i in range(len(val_losses)):
        rows.append(_row(i + 1, train_losses[i], train_accs[i], val_losses[i], val_accs[i]))
    return rows


def _check_happy_path_continues() -> None:
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        _write_metrics_csv(m, _happy_path_rows())
        result = evaluate_continuation(m, error_log_path=Path(td) / "log.txt")

    assert result["decision"] == "continue", f"expected continue, got: {result}"
    assert result["reasons"] == [], f"expected no reasons, got: {result['reasons']}"
    assert result["had_errors"] is False
    assert result["epochs_run"] == 8
    assert result["best_epoch"] == 8  # last epoch has highest val_acc
    # gap = 0.91 - 0.86 = 0.05 < 0.15
    assert abs(result["gap"] - 0.05) < 1e-6
    assert result["val_loss_trend_ok"] is True
    assert result["trend_non_increasing"] == TREND_WINDOW  # all 5 deltas non-increasing
    print("OK [happy] — clean run with gap=5pp + monotone val_loss returns continue.")


def _check_too_few_epochs() -> None:
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        _write_metrics_csv(m, [_row(i + 1, 1.0, 0.5, 1.0, 0.5) for i in range(MIN_EPOCHS - 1)])
        result = evaluate_continuation(m, error_log_path=None)

    assert result["decision"] == "halt"
    assert any("too_few_epochs" in r for r in result["reasons"]), result["reasons"]
    assert result["epochs_run"] == MIN_EPOCHS - 1
    print("OK [too-few] — fewer than 5 epochs halts with too_few_epochs reason.")


def _check_metrics_csv_missing() -> None:
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics_does_not_exist.csv"
        result = evaluate_continuation(m, error_log_path=None)
    assert result["decision"] == "halt"
    assert "metrics_csv_missing" in result["reasons"]
    print("OK [missing-csv] — missing metrics.csv halts.")


def _check_overfit_gap() -> None:
    """Best epoch: train_acc=0.99, val_acc=0.80 → gap=0.19 > 0.15."""
    rows = _happy_path_rows()
    rows[-1]["train_acc"] = 0.99
    rows[-1]["val_acc"] = 0.80   # still highest val_acc among rows
    rows[-1]["val_loss"] = 0.70  # keep trend OK so this is the only halt reason
    # Bump other val_accs so this row stays the best
    for r in rows[:-1]:
        r["val_acc"] = min(r["val_acc"], 0.78)

    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        _write_metrics_csv(m, rows)
        result = evaluate_continuation(m, error_log_path=None)

    assert result["decision"] == "halt"
    assert any(r.startswith("overfit_gap") for r in result["reasons"]), result["reasons"]
    assert result["gap"] > OVERFIT_GAP_THRESHOLD
    assert result["had_errors"] is False
    print("OK [overfit] — gap >= 15pp at best epoch halts with overfit_gap reason.")


def _check_val_loss_trend_alone_does_not_halt() -> None:
    """Conditional rule (2026-05-05): val_loss trend alone no longer halts in evaluate_continuation.

    The runner's _evaluate_cell applies the conditional (gap > 5pp OR band != green);
    evaluate_continuation only reports the trend status as a signal. See
    test_run_phase_a.py for the conditional-rule scenarios.
    """
    val_losses = [1.0, 0.8, 0.7, 0.6, 0.7, 0.85, 0.95, 1.05]
    val_accs   = [0.40, 0.55, 0.62, 0.68, 0.66, 0.60, 0.55, 0.50]
    train_accs = [0.45, 0.60, 0.66, 0.72, 0.71, 0.66, 0.61, 0.56]
    rows = []
    for i, (vl, va, ta) in enumerate(zip(val_losses, val_accs, train_accs)):
        rows.append(_row(i + 1, vl, ta, vl, va))

    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        _write_metrics_csv(m, rows)
        result = evaluate_continuation(m, error_log_path=None)

    # Trend signal still surfaces in the output...
    assert result["val_loss_trend_ok"] is False
    assert result["trend_non_increasing"] < TREND_NON_INCREASING_REQUIRED
    # ...but it no longer halts on its own (gap is small here, no other hard rules tripped).
    assert result["decision"] == "continue", (
        f"trend-only halt was removed; expected continue, got: {result}"
    )
    assert not any(r.startswith("val_loss_trend:") for r in result["reasons"]), (
        f"trend should not appear as a halt reason in evaluate_continuation: {result['reasons']}"
    )
    assert result["trend_window_size"] == TREND_WINDOW
    # gap should be small here (best epoch is epoch 4 with train=0.72, val=0.68 -> 0.04)
    assert result["gap"] < OVERFIT_GAP_THRESHOLD
    print("OK [trend-alone-no-halt] -- val_loss trend alone no longer halts (conditional rule).")


def _check_error_marker_halts() -> None:
    """A Traceback in the error log halts even when metrics look fine."""
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        _write_metrics_csv(m, _happy_path_rows())
        log = Path(td) / "log.txt"
        log.write_text(
            "[lightning] run_name=foo\n"
            "[lightning] dataset=cifar10\n"
            "Traceback (most recent call last):\n"
            "  File \"x.py\", line 1, in <module>\n"
            "    raise RuntimeError('boom')\n"
            "RuntimeError: boom\n",
            encoding="utf-8",
        )
        result = evaluate_continuation(m, error_log_path=log)

    assert result["decision"] == "halt"
    assert result["had_errors"] is True
    assert result["error_marker"] is not None
    assert "Traceback" in result["error_marker"]
    assert any(r.startswith("errors_in_log") for r in result["reasons"]), result["reasons"]
    print("OK [error-marker] — Traceback in log halts even with otherwise-good metrics.")


def _check_oom_marker_halts() -> None:
    """CUDA OOM marker also trips the error gate."""
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        _write_metrics_csv(m, _happy_path_rows())
        log = Path(td) / "log.txt"
        log.write_text(
            "epoch 3: CUDA out of memory. Tried to allocate 2 GiB.\n", encoding="utf-8"
        )
        result = evaluate_continuation(m, error_log_path=log)

    assert result["decision"] == "halt"
    assert result["had_errors"] is True
    print("OK [oom-marker] — 'out of memory' marker triggers halt.")


def _check_combined_halt_reasons_stack() -> None:
    """Multiple problems → multiple reasons in output, single halt decision."""
    rows = _happy_path_rows()
    # Inject an overfit gap at best epoch.
    rows[-1]["train_acc"] = 0.99
    rows[-1]["val_acc"] = 0.80
    for r in rows[:-1]:
        r["val_acc"] = min(r["val_acc"], 0.78)
    # And inject rising val_loss tail.
    for i in range(-5, 0):
        rows[i]["val_loss"] = 0.5 + i * -0.1  # 1.0, 0.9, 0.8, 0.7, 0.6 — increasing back?
    # Actually make them strictly rising:
    for i, vl in zip(range(-5, 0), [0.5, 0.7, 0.9, 1.1, 1.3]):
        rows[i]["val_loss"] = vl

    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        _write_metrics_csv(m, rows)
        log = Path(td) / "log.txt"
        log.write_text("[ERROR] something bad\n", encoding="utf-8")
        result = evaluate_continuation(m, error_log_path=log)

    assert result["decision"] == "halt"
    # Two hard halt reasons stack now (trend is no longer a halt-causing reason
    # in evaluate_continuation; the runner re-evaluates it conditionally).
    assert len(result["reasons"]) >= 2, f"expected stacked reasons, got: {result['reasons']}"
    assert any(r.startswith("errors_in_log") for r in result["reasons"])
    assert any(r.startswith("overfit_gap") for r in result["reasons"])
    # Trend signal still surfaces in the output dict even though it's not a halt reason.
    assert result["val_loss_trend_ok"] is False
    print("OK [stacked] -- error + overfit_gap stack; trend reported as signal only.")


def _check_no_log_file_is_not_an_error() -> None:
    """Missing error_log_path is not the same as having errors."""
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        _write_metrics_csv(m, _happy_path_rows())
        result = evaluate_continuation(m, error_log_path=Path(td) / "no_such_log.txt")
    assert result["had_errors"] is False
    assert result["decision"] == "continue"
    print("OK [no-log] -- missing error_log_path is treated as 'no errors observed'.")


def _check_csv_without_lr_column_works() -> None:
    """LegacyMetricsCSVCallback emits 5-column CSVs (no lr); the reader must tolerate that."""
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        m.parent.mkdir(parents=True, exist_ok=True)
        # Schema as actually emitted by the current Lightning engine: no lr column.
        with open(m, "w", encoding="utf-8", newline="") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
            for i in range(8):
                f.write(f"{i+1},{1.5-0.1*i:.3f},{0.4+0.05*i:.3f},{1.4-0.1*i:.3f},{0.5+0.04*i:.3f}\n")
        result = evaluate_continuation(m, error_log_path=None)
    assert result["decision"] == "continue", f"reader rejected lr-less CSV: {result}"
    assert result["epochs_run"] == 8
    print("OK [no-lr-column] -- 5-column metrics.csv (no lr) accepted; was the runtime crash bug.")


def _check_csv_with_nan_train_metrics_in_epoch_1() -> None:
    """Real Lightning runs emit train_loss=nan in epoch 1 (display artifact)."""
    with tempfile.TemporaryDirectory() as td:
        m = Path(td) / "metrics.csv"
        m.parent.mkdir(parents=True, exist_ok=True)
        with open(m, "w", encoding="utf-8", newline="") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
            f.write("1,nan,0.10,1.865,0.674\n")
            for i in range(2, 9):
                f.write(f"{i},{1.5-0.1*i:.3f},{0.4+0.05*i:.3f},{1.4-0.1*i:.3f},{0.5+0.04*i:.3f}\n")
        result = evaluate_continuation(m, error_log_path=None)
    assert result["epochs_run"] == 8
    assert result["best_epoch"] == 8
    assert result["decision"] == "continue"
    print("OK [nan-epoch1] -- NaN train metrics in epoch 1 don't crash the reader.")


def main() -> int:
    _check_happy_path_continues()
    _check_too_few_epochs()
    _check_metrics_csv_missing()
    _check_overfit_gap()
    _check_val_loss_trend_alone_does_not_halt()
    _check_error_marker_halts()
    _check_oom_marker_halts()
    _check_combined_halt_reasons_stack()
    _check_no_log_file_is_not_an_error()
    _check_csv_without_lr_column_works()
    _check_csv_with_nan_train_metrics_in_epoch_1()
    return 0


if __name__ == "__main__":
    sys.exit(main())
