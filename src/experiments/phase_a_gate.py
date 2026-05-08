"""Phase A alignment-gate thresholds (US-002) + auto-continue evaluator (US-003).

Two responsibilities, kept in the same module so scripts/run_phase_a.py has
a single import surface:

(1) evaluate_gate / gate_explanation — band the val_acc against paper baselines.
    green  : val_acc >= paper - 0.03
    yellow : paper - 0.07 <= val_acc < paper - 0.03
    red    : val_acc < paper - 0.07

(2) evaluate_continuation — decide continue / halt from metrics.csv + log.txt.
    Halts on any of: error markers in log, train/val gap >= 15pp at best epoch,
    fewer than 3 of the last 5 val_loss transitions non-increasing, or fewer
    than 5 epochs total (truncated run).

Baselines (per CLAUDE.md / PRD §5):
    (resnet50,    cifar10) = 0.93   — full-data CIFAR-10 paper number
    (densenet121, cifar10) = 0.95   — full-data CIFAR-10 paper number
    (resnet50,    mnist)   = 0.995  — saturated benchmark floor
    (densenet121, mnist)   = 0.995  — saturated benchmark floor

The 10K-train subset will under-shoot full-data paper numbers; the −3 / −7pp
bands acknowledge that — yellow is "noisy but acceptable", red is a halt.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, Optional

GateBand = Literal["green", "yellow", "red"]

PAPER_BASELINES: dict[tuple[str, str], float] = {
    ("resnet50", "cifar10"): 0.93,
    ("densenet121", "cifar10"): 0.95,
    ("resnet50", "mnist"): 0.995,
    ("densenet121", "mnist"): 0.995,
}

GREEN_TOLERANCE: float = 0.03  # within −3pp of paper
YELLOW_TOLERANCE: float = 0.07  # within −7pp of paper


def evaluate_gate(model: str, dataset: str, val_acc: float) -> GateBand:
    """Return the gate band for a completed Phase A cell.

    Raises KeyError if (model, dataset) is not in PAPER_BASELINES — Phase A
    only covers the 4 active CNN cells; TransNeXt is intentionally absent
    until the GPU upgrade (per PRD non-goals).
    """
    if (model, dataset) not in PAPER_BASELINES:
        raise KeyError(
            f"no Phase A baseline registered for ({model!r}, {dataset!r}). "
            f"Known pairs: {sorted(PAPER_BASELINES.keys())}"
        )
    paper = PAPER_BASELINES[(model, dataset)]
    if val_acc >= paper - GREEN_TOLERANCE:
        return "green"
    if val_acc >= paper - YELLOW_TOLERANCE:
        return "yellow"
    return "red"


def gate_explanation(model: str, dataset: str, val_acc: float) -> str:
    """One-line human-readable rationale for the gate band.

    Used by gate_verdict.json and the dashboard's collapsible per-run panel.
    """
    paper = PAPER_BASELINES[(model, dataset)]
    delta = val_acc - paper
    band = evaluate_gate(model, dataset, val_acc)
    sign = "+" if delta >= 0 else ""
    return (
        f"{band.upper()}: val_acc={val_acc:.4f} vs paper={paper:.3f} "
        f"({sign}{delta * 100:.2f}pp); "
        f"green if >=-{GREEN_TOLERANCE * 100:.0f}pp, "
        f"yellow if >=-{YELLOW_TOLERANCE * 100:.0f}pp, else red"
    )


# --------------------------------------------------------------------------
# US-003: auto-continue evaluator
# --------------------------------------------------------------------------

OVERFIT_GAP_THRESHOLD: float = 0.15  # hard halt: train_acc - val_acc >= this
MIN_EPOCHS: int = 5                  # fewer than this -> halt, "too_few_epochs"
TREND_WINDOW: int = 5                # look at last N val_loss transitions
TREND_NON_INCREASING_REQUIRED: int = 3  # at least M of N must be non-increasing

# Conditional trend rule (operator decision 2026-05-05):
# A non-decreasing val_loss trend halts the runner ONLY when combined with a
# high overfitting gap (>5pp) OR an off-green band (deviation >3pp from paper).
# A converged green run with a small gap and noisy late-epoch val_loss is a
# plateau, not divergence; previously this fired a false halt on
# final_clean_densenet121_mnist (val_acc=0.9928, gap=0.7pp). Decision lives in
# scripts/run_phase_a.py:_evaluate_cell because the band requires (model, dataset)
# context that evaluate_continuation does not have.
TREND_HALT_GAP_THRESHOLD: float = 0.05  # >5pp gap -> trend rule re-enables

# Markers we treat as evidence of a hard error in log.txt or a dedicated error file.
ERROR_MARKERS: tuple[str, ...] = ("Traceback", "[ERROR]", "CUDA error", "out of memory")


def _read_metrics_csv(path: Path) -> list[dict]:
    """Parse metrics.csv into a list of per-epoch dicts.

    Required columns: epoch, train_acc, val_loss, val_acc.
    Optional columns: train_loss, lr (older runs include them; LegacyMetricsCSVCallback
    in the current Lightning engine omits lr). Missing optional fields are recorded
    as NaN; missing required fields still raise via KeyError.

    NaN values in any column are tolerated (Lightning emits NaN in epoch 1 of some
    runs as a display artifact before the first train step settles). Downstream
    halt logic operates on best-val-acc epoch metrics and val_loss-only trend, both
    of which are unaffected by NaN train metrics in early warmup epochs.
    """
    def _opt_float(r: dict, key: str) -> float:
        v = r.get(key)
        if v is None or v == "":
            return float("nan")
        try:
            return float(v)
        except ValueError:
            return float("nan")

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict] = []
        for r in reader:
            rows.append({
                "epoch": int(r["epoch"]),
                "train_loss": _opt_float(r, "train_loss"),
                "train_acc": float(r["train_acc"]),
                "val_loss": float(r["val_loss"]),
                "val_acc": float(r["val_acc"]),
                "lr": _opt_float(r, "lr"),
            })
    return rows


def _scan_error_log(path: Optional[Path]) -> tuple[bool, Optional[str]]:
    """Return (had_errors, first_marker_line).

    If `path` is None or doesn't exist → (False, None). Otherwise scan for any
    ERROR_MARKERS substring; first hit wins.
    """
    if path is None or not path.exists():
        return False, None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, None
    for marker in ERROR_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            # Return the line containing the first marker hit.
            line_start = text.rfind("\n", 0, idx) + 1
            line_end = text.find("\n", idx)
            line = text[line_start:line_end if line_end != -1 else None].strip()
            return True, line[:200]
    return False, None


def _val_loss_trend_ok(rows: list[dict]) -> tuple[bool, int, int]:
    """Inspect the last TREND_WINDOW val_loss transitions.

    A transition is "non-increasing" iff val_loss[i] <= val_loss[i-1].
    Returns (ok, n_non_increasing, n_transitions_evaluated). If fewer than
    TREND_WINDOW transitions are available, evaluate whatever exists; the
    required count remains TREND_NON_INCREASING_REQUIRED.
    """
    if len(rows) < 2:
        return False, 0, 0
    losses = [r["val_loss"] for r in rows]
    deltas = [losses[i] - losses[i - 1] for i in range(1, len(losses))]
    window = deltas[-TREND_WINDOW:]
    n_non_inc = sum(1 for d in window if d <= 0.0)
    return (n_non_inc >= TREND_NON_INCREASING_REQUIRED), n_non_inc, len(window)


def evaluate_continuation(
    metrics_csv_path: Path | str,
    error_log_path: Path | str | None = None,
) -> dict:
    """Decide continue | halt for a completed Phase A cell on hard rules only.

    Hard halts (always halt regardless of accuracy):
      - error markers found in error_log_path,
      - metrics.csv missing,
      - epochs_run < MIN_EPOCHS (truncated run),
      - (train_acc - val_acc) at the best-val-acc epoch >= OVERFIT_GAP_THRESHOLD.

    Soft signal (reported, not halt-triggering on its own):
      - val_loss trend (val_loss_trend_ok). The runner combines this with the
        gate band + gap to decide whether to halt — see _evaluate_cell in
        scripts/run_phase_a.py for the conditional rule.

    Returns:
        {
          "decision": "continue" | "halt",
          "reasons": list[str],          # empty on continue
          "gap": float,                  # train_acc - val_acc at best epoch (NaN if no rows)
          "best_epoch": int | None,
          "epochs_run": int,
          "val_loss_trend_ok": bool,
          "trend_non_increasing": int,   # count in window
          "trend_window_size": int,      # actual window length evaluated
          "had_errors": bool,
          "error_marker": str | None,    # first marker line if any
        }
    """
    metrics_csv_path = Path(metrics_csv_path)
    error_log_path = Path(error_log_path) if error_log_path is not None else None

    reasons: list[str] = []
    out: dict = {
        "decision": "continue",
        "reasons": reasons,
        "gap": float("nan"),
        "best_epoch": None,
        "epochs_run": 0,
        "val_loss_trend_ok": False,
        "trend_non_increasing": 0,
        "trend_window_size": 0,
        "had_errors": False,
        "error_marker": None,
    }

    had_errors, marker = _scan_error_log(error_log_path)
    out["had_errors"] = had_errors
    out["error_marker"] = marker
    if had_errors:
        reasons.append(f"errors_in_log:{marker!r}")

    if not metrics_csv_path.exists():
        reasons.append("metrics_csv_missing")
        out["decision"] = "halt"
        return out

    rows = _read_metrics_csv(metrics_csv_path)
    out["epochs_run"] = len(rows)

    if len(rows) < MIN_EPOCHS:
        reasons.append(f"too_few_epochs:{len(rows)}<{MIN_EPOCHS}")
        out["decision"] = "halt"
        # Still try to fill best_epoch / gap if any rows exist, but return halt.
        if rows:
            best = max(rows, key=lambda r: r["val_acc"])
            out["best_epoch"] = best["epoch"]
            out["gap"] = best["train_acc"] - best["val_acc"]
        return out

    best = max(rows, key=lambda r: r["val_acc"])
    out["best_epoch"] = best["epoch"]
    gap = best["train_acc"] - best["val_acc"]
    out["gap"] = gap
    if gap >= OVERFIT_GAP_THRESHOLD:
        reasons.append(f"overfit_gap:{gap:.4f}>={OVERFIT_GAP_THRESHOLD}")

    trend_ok, n_non_inc, win_size = _val_loss_trend_ok(rows)
    out["val_loss_trend_ok"] = trend_ok
    out["trend_non_increasing"] = n_non_inc
    out["trend_window_size"] = win_size
    # Note: val_loss trend is intentionally NOT a halt reason here. It is
    # reported as a signal; the runner's _evaluate_cell applies the conditional
    # rule (halt only when also gap > 5pp OR band != green). See module docstring.

    if had_errors or reasons:
        out["decision"] = "halt"
    else:
        out["decision"] = "continue"
    return out
