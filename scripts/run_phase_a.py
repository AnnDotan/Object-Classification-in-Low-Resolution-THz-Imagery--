"""Phase A sequential runner (US-004).

Executes the 4 active Phase A clean-baseline cells in order:
    1. final_clean_resnet50_mnist
    2. final_clean_densenet121_mnist
    3. final_clean_resnet50_cifar10
    4. final_clean_densenet121_cifar10

(TransNeXt rows are intentionally skipped — they remain Pending in
Final_Exp.md per the active PRD.)

For each cell:
  1. Dispatch via ``run_systematic.run_cell(tag, mode, engine)`` — never
     bypass the canonical runner.
  2. Read ``runs/final/<tag>/metrics.json`` for best_val_acc and
     ``runs/final/<tag>/metrics.csv`` + ``log.txt`` for trend / errors.
  3. Compute the gate band (US-002) and continuation decision (US-003).
  4. Write ``runs/final/<tag>/gate_verdict.json``.
  5. If band == 'red' OR decision == 'halt' → print summary, exit non-zero.
     Otherwise → proceed to the next cell automatically.

Decision precedence (PRD §5):
  1. Hard errors (caught from run_cell or detected via log markers) → halt.
  2. Continuation rule (gap, val_loss trend, epoch count) → halt overrides
     a green band.
  3. Alignment gate (red) → halt.
  4. Otherwise → continue (green or yellow).

CLI:
    python scripts/run_phase_a.py                         # all 4 cells, full mode
    python scripts/run_phase_a.py --only resnet50_mnist   # single cell
    python scripts/run_phase_a.py --dry-run               # print plan, don't train
    python scripts/run_phase_a.py --mode pilot            # 5-epoch smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.matrix import cells_by_tag  # noqa: E402
from src.experiments.phase_a_gate import (  # noqa: E402
    PAPER_BASELINES,
    TREND_HALT_GAP_THRESHOLD,
    evaluate_continuation,
    evaluate_gate,
    gate_explanation,
)


PHASE_A_CELL_ORDER: list[str] = [
    "final_clean_resnet50_mnist",
    "final_clean_densenet121_mnist",
    "final_clean_resnet50_cifar10",
    "final_clean_densenet121_cifar10",
]


def _load_update_final_exp():
    """Load scripts/update_final_exp.py as a module (no scripts/__init__.py).

    Returns the loaded module; cached on first call. Failure to load is the
    caller's responsibility — typically the runner's post_cell_hook catches it.
    """
    import importlib.util as _ilu
    src = Path(__file__).resolve().parent / "update_final_exp.py"
    spec = _ilu.spec_from_file_location("scripts_update_final_exp", src)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def refresh_final_exp_hook(verdict: dict) -> None:
    """Default post_cell_hook (US-005): regenerate Final_Exp.md after each cell.

    Calls scripts/update_final_exp.py with default args. Raises on non-zero
    return so the runner's hook-failure handling logs a visible WARN — but
    the runner does not abort subsequent cells on tracker-write failures
    (gate/continuation decision is the source of truth, per PRD US-005).
    """
    mod = _load_update_final_exp()
    rc = mod.main([])  # default --runs-root runs/final, --out Final_Exp.md
    if rc != 0:
        raise RuntimeError(f"update_final_exp.main returned rc={rc} for tag={verdict.get('tag')!r}")


def refresh_phase_a_dashboard_hook(verdict: dict) -> None:
    """Post-cell hook (US-015): regenerate the unified Final_Exp.html.

    The Phase A operator view (US-006, Final_Exp_PhaseA.html) was folded into
    the Gold Standard table dashboard in US-015 — there is now a single
    canonical artifact at artifacts/Final_Exp.html. Phase A rows still surface
    gate-verdict info via per-row <details> panels.
    """
    from src.tools.build_final_dashboard import main as _build  # lazy import
    rc = _build([])
    if rc != 0:
        raise RuntimeError(
            f"build_final_dashboard returned rc={rc} "
            f"for tag={verdict.get('tag')!r}"
        )


def refresh_all_trackers_hook(verdict: dict) -> None:
    """Combined hook: refresh Final_Exp.md AND Final_Exp_PhaseA.html in order.

    Failures in either step are isolated — the second step always runs even if
    the first raises, and the original exception is re-raised at the end so
    the runner logs a visible WARN. Both refreshes are idempotent so a partial
    failure does not corrupt either tracker.
    """
    first_error: Optional[Exception] = None
    try:
        refresh_final_exp_hook(verdict)
    except Exception as e:
        first_error = e
        print(f"[phase-a][WARN] Final_Exp.md refresh failed: {e}")
    try:
        refresh_phase_a_dashboard_hook(verdict)
    except Exception as e:
        print(f"[phase-a][WARN] Final_Exp_PhaseA.html refresh failed: {e}")
        if first_error is None:
            first_error = e
    if first_error is not None:
        raise first_error


def _resolve_only(only: Optional[str]) -> list[str]:
    """Map --only "<model>_<dataset>" to a single tag, or full order if None."""
    if only is None:
        return list(PHASE_A_CELL_ORDER)
    candidate = f"final_clean_{only}"
    if candidate in PHASE_A_CELL_ORDER:
        return [candidate]
    raise SystemExit(
        f"--only must be one of "
        f"{[t.replace('final_clean_', '') for t in PHASE_A_CELL_ORDER]}, got {only!r}"
    )


def _read_best_val_acc(metrics_json: Path) -> Optional[float]:
    """Pull best_val_acc (or fallback) from metrics.json. Returns None if absent."""
    if not metrics_json.exists():
        return None
    try:
        blob = json.loads(metrics_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for key in ("best_val_acc", "final_val_acc", "last_val_acc"):
        v = blob.get(key)
        if isinstance(v, (int, float)) and v >= 0.0:
            return float(v)
    return None


def _resolve_train_acc_at_best(metrics_json: Path, best_epoch: Optional[int]) -> Optional[float]:
    """Read train_acc at best_epoch from metrics.json if present, else None."""
    if not metrics_json.exists() or best_epoch is None:
        return None
    try:
        blob = json.loads(metrics_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    # LegacyJSONMetricsCallback writes last_train_acc, not per-epoch — fine,
    # the gap is also recoverable from the continuation evaluator.
    v = blob.get("last_train_acc")
    if isinstance(v, (int, float)) and v >= 0.0:
        return float(v)
    return None


def _evaluate_cell(spec, run_dir: Path, run_error: Optional[str]) -> dict:
    """Combine US-002 gate + US-003 continuation into one verdict dict.

    `run_error` is non-None when run_cell raised during training; its content
    seeds had_errors so the verdict halts even if metrics files are missing.
    """
    metrics_json = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics.csv"
    log_path = run_dir / "log.txt"

    cont = evaluate_continuation(metrics_csv, error_log_path=log_path)

    val_acc = _read_best_val_acc(metrics_json)
    train_acc = _resolve_train_acc_at_best(metrics_json, cont.get("best_epoch"))

    band: Optional[str]
    band_explanation: Optional[str]
    if val_acc is None or (spec.model, spec.dataset) not in PAPER_BASELINES:
        band = None
        band_explanation = (
            "no_val_acc_available" if val_acc is None
            else f"no_paper_baseline_for_{spec.model}_{spec.dataset}"
        )
    else:
        band = evaluate_gate(spec.model, spec.dataset, val_acc)
        band_explanation = gate_explanation(spec.model, spec.dataset, val_acc)

    # Decision precedence (PRD §5, refined 2026-05-05):
    #   1. run_cell exception -> always halt
    #   2. continuation hard halt (errors / too_few_epochs / gap >= 15pp) -> halt
    #   3. red band -> halt
    #   4. CONDITIONAL trend rule: val_loss_trend_ok == False halts ONLY when
    #      gap > 5pp OR band is not green. A converged green run with small gap
    #      and noisy late-epoch val_loss is a plateau, not divergence.
    #   5. Otherwise -> continue
    decision = "continue"
    reasons = list(cont.get("reasons", []))
    gap_val = cont.get("gap")
    gap_safe = (
        float(gap_val)
        if isinstance(gap_val, (int, float)) and gap_val == gap_val  # not NaN
        else 0.0
    )
    trend_ok = bool(cont.get("val_loss_trend_ok"))
    n_non_inc = cont.get("trend_non_increasing", 0)
    win_size = cont.get("trend_window_size", 0)

    if run_error is not None:
        decision = "halt"
        reasons.insert(0, f"run_cell_raised:{run_error[:200]}")
    elif cont.get("decision") == "halt":
        decision = "halt"
    elif band == "red":
        decision = "halt"
        reasons.append(f"gate_band:red ({band_explanation})")
    elif (not trend_ok) and (gap_safe > TREND_HALT_GAP_THRESHOLD or band != "green"):
        decision = "halt"
        reasons.append(
            f"val_loss_trend_with_off_band_or_high_gap:"
            f"{n_non_inc}/{win_size}_non_increasing,gap={gap_safe:+.4f},band={band}"
        )
    elif not trend_ok:
        # Trend bad but green band + small gap = plateau noise. Note as warning.
        reasons.append(
            f"val_loss_trend_warning:{n_non_inc}/{win_size}_non_increasing"
            f" (gap={gap_safe:+.4f},band={band}; trend rule suppressed by conditional)"
        )

    paper_baseline = PAPER_BASELINES.get((spec.model, spec.dataset))

    return {
        "tag": spec.tag,
        "model": spec.model,
        "dataset": spec.dataset,
        "phase": spec.phase,
        "decision": decision,
        "reasons": reasons,
        "gate_band": band,
        "gate_explanation": band_explanation,
        "paper_baseline": paper_baseline,
        "best_val_acc": val_acc,
        "train_acc_at_best": train_acc,
        "gap": cont.get("gap"),
        "best_epoch": cont.get("best_epoch"),
        "epochs_run": cont.get("epochs_run"),
        "val_loss_trend_ok": cont.get("val_loss_trend_ok"),
        "trend_non_increasing": cont.get("trend_non_increasing"),
        "trend_window_size": cont.get("trend_window_size"),
        "had_errors": cont.get("had_errors") or (run_error is not None),
        "error_marker": cont.get("error_marker"),
        "run_dir": str(run_dir),
    }


def _write_verdict(verdict: dict, run_dir: Path) -> Path:
    """Persist verdict to runs/final/<tag>/gate_verdict.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "gate_verdict.json"
    path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return path


def _print_verdict(verdict: dict) -> None:
    band = verdict["gate_band"] or "n/a"
    decision = verdict["decision"]
    val_acc = verdict["best_val_acc"]
    paper = verdict["paper_baseline"]
    gap = verdict["gap"]
    epochs = verdict["epochs_run"]
    sep = "-" * 70
    print(sep)
    print(f"  CELL {verdict['tag']} -> decision={decision.upper()}, band={band}")
    if val_acc is not None and paper is not None:
        delta = (val_acc - paper) * 100
        print(f"    val_acc={val_acc:.4f}  paper={paper:.3f}  delta={delta:+.2f}pp")
    if gap is not None and isinstance(gap, (int, float)):
        print(f"    train/val gap @ best epoch = {gap:+.4f}  (epochs_run={epochs})")
    if verdict["reasons"]:
        print(f"    reasons:")
        for r in verdict["reasons"]:
            print(f"      - {r}")
    print(sep)


def run_phase_a(
    tags: list[str],
    *,
    mode: str = "full",
    engine: str = "lightning",
    runs_root: Path = _REPO_ROOT / "runs" / "final",
    run_cell_fn: Optional[Callable] = None,  # injectable for tests
    post_cell_hook: Optional[Callable[[dict], None]] = None,  # US-005/US-006 hooks
    skip_existing: bool = False,
) -> tuple[int, list[dict]]:
    """Run the requested Phase A cells sequentially.

    Returns ``(exit_code, verdicts)``. exit_code is 0 iff every cell continued;
    1 if any cell halted. ``verdicts`` is in the order cells were attempted
    (skipped cells produce no verdict entry).

    skip_existing: if True, cells whose runs/final/<tag>/gate_verdict.json
    already exists are skipped (used for resume-after-crash). Default False
    preserves the original behavior.
    """
    if run_cell_fn is None:
        from run_systematic import run_cell as _rc
        run_cell_fn = _rc

    by_tag = cells_by_tag()
    verdicts: list[dict] = []

    for tag in tags:
        if tag not in by_tag:
            raise SystemExit(f"unknown Phase A tag: {tag!r}")
        spec = by_tag[tag]

        if skip_existing:
            verdict_path = runs_root / tag / "gate_verdict.json"
            if verdict_path.exists():
                print(f"[phase-a] SKIP {tag} (gate_verdict.json exists -> --skip-existing)")
                continue

        print(f"\n[phase-a] dispatching {tag}  (mode={mode}, engine={engine})")

        run_error: Optional[str] = None
        run_dir: Optional[Path] = None
        try:
            run_dir = Path(run_cell_fn(tag, mode=mode, engine=engine))
        except Exception as e:
            run_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            run_dir = runs_root / tag  # best-effort path for verdict write
            print(f"[phase-a][ERROR] run_cell raised for {tag}: {e}")

        verdict = _evaluate_cell(spec, run_dir, run_error)
        _write_verdict(verdict, run_dir)
        _print_verdict(verdict)
        verdicts.append(verdict)

        if post_cell_hook is not None:
            try:
                post_cell_hook(verdict)
            except Exception as e:
                # Per PRD US-005: tracker-write failure must NOT abort the
                # next cell on its own. Log and proceed.
                print(f"[phase-a][WARN] post_cell_hook failed: {e}")

        if verdict["decision"] == "halt":
            print(f"[phase-a] HALT after {tag} — operator review required.")
            return 1, verdicts

    print(f"\n[phase-a] DONE — all {len(verdicts)} cell(s) continued.")
    return 0, verdicts


def main() -> int:
    p = argparse.ArgumentParser(description="Phase A sequential runner (US-004)")
    p.add_argument("--mode", default="full", choices=["pilot", "full"],
                   help="pilot=5 epochs/2k subset, full=60 epochs/10k subset (CLAUDE.md).")
    p.add_argument("--engine", default="lightning", choices=["lightning", "legacy"])
    p.add_argument("--only", default=None,
                   help="Run a single cell by '<model>_<dataset>' suffix, e.g. resnet50_mnist.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned cell order and exit; do not train.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip any cell whose runs/final/<tag>/gate_verdict.json "
                        "already exists. Used to resume after a crash mid-sweep.")
    args = p.parse_args()

    tags = _resolve_only(args.only)

    if args.dry_run:
        print("[phase-a][dry-run] planned cell order:")
        for i, t in enumerate(tags, 1):
            print(f"  {i}. {t}")
        return 0

    exit_code, _ = run_phase_a(
        tags, mode=args.mode, engine=args.engine,
        post_cell_hook=refresh_all_trackers_hook,
        skip_existing=args.skip_existing,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
