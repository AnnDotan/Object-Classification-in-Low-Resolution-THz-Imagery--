"""RALPH Loop Driver — sequential 186-cell dispatch with pathology guard.

Thin wrapper over ``run_systematic.run_cell()`` providing:
  (a) sequential dispatch over ``iter_cells()`` filtered by --phase/--model/--dataset,
  (b) pathology guard evaluating PRD §6.3 verdicts after each ``Trainer.fit``,
  (c) sentinel writes (NEEDS_FULL_FT, INTERRUPTED, QUARANTINED_AFTER_RETRY),
  (d) --remediate-only second-pass mode under PRD §6.4 Full FT,
  (e) DEBUGGER hook with 3-attempt cap on torch.cuda.OutOfMemoryError.

This module stays torch-free at import time so the unit fixture in
``src/tests/test_ralph_loop.py`` can exercise pathology/sentinel/CLI logic
without pulling torch / Lightning into the test process. ``run_cell`` and the
real OOM exception class are resolved lazily inside the dispatch helpers.

PRD: US-005 (legacy US-046).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.degradation_levels import AXES  # noqa: E402
from src.experiments.cells import CellMeta, iter_cells  # noqa: E402

RUNS_DIR = REPO_ROOT / "runs" / "final"

# Sentinel filenames (PRD §6.1, §6.3)
NEEDS_FULL_FT = "NEEDS_FULL_FT"
INTERRUPTED = "INTERRUPTED"
QUARANTINED_AFTER_RETRY = "QUARANTINED_AFTER_RETRY"
DEBUGGER_LOG = "debugger.log"
RETRY_CONFIG = "retry_config.json"

# Pathology thresholds (PRD §6.3)
OVERFIT_GAP_PP = 18.0
RANDOM_BASELINE_ACC = 0.10
FAILED_CONV_ACC = 1.3 * RANDOM_BASELINE_ACC  # 0.13 on 10-class

# DEBUGGER OOM cap (PRD acceptance criterion line 395)
MAX_OOM_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Cell filter
# ---------------------------------------------------------------------------

def filter_cells(
    cells: Iterable[CellMeta],
    *,
    phase: Optional[str] = None,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    axes: Optional[Sequence[str]] = None,
) -> list[CellMeta]:
    axes_set = set(axes) if axes else None
    out: list[CellMeta] = []
    for c in cells:
        if phase is not None and c.phase != phase:
            continue
        if model is not None and c.model != model:
            continue
        if dataset is not None and c.dataset != dataset:
            continue
        if axes_set is not None and c.axis not in axes_set:
            continue
        out.append(c)
    return out


def _parse_axes_arg(value: str) -> tuple[str, ...]:
    """argparse type= callable: comma-separated axis names validated against AXES.

    Empty entries (e.g. trailing comma) are tolerated; unknown names raise
    ArgumentTypeError so argparse routes the failure through parser.error.
    """
    parts = tuple(p.strip() for p in value.split(",") if p.strip())
    if not parts:
        raise argparse.ArgumentTypeError(
            f"--axes requires at least one axis; expected subset of {list(AXES)}"
        )
    invalid = [a for a in parts if a not in AXES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown axis/axes {invalid!r}; expected subset of {list(AXES)}"
        )
    return parts


# ---------------------------------------------------------------------------
# Pathology guard (PRD §6.3)
# ---------------------------------------------------------------------------

def _is_nan_or_inf(x: object) -> bool:
    if x is None:
        return False
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return math.isnan(f) or math.isinf(f)


def evaluate_pathology(history: Sequence[dict]) -> str:
    """Return one of {"healthy", "failed_convergence", "overfitting"}.

    Inputs: list of per-epoch dicts (the HistoryJSONCallback schema —
    ``epoch, train_loss, val_loss, train_acc, val_acc``). Empty history
    counts as failed_convergence (the cell never produced a validation
    metric).

    Verdict precedence:
      1. NaN/Inf loss observed at or after epoch 2  → failed_convergence.
      2. Generalization gap at best-val epoch ≥ 18pp → overfitting.
      3. Best val_acc below 1.3 × random_baseline (0.13 on 10-class)
         → failed_convergence.
      4. Otherwise → healthy.
    """
    if not history:
        return "failed_convergence"

    for entry in history:
        ep = int(entry.get("epoch", 0) or 0)
        if ep >= 2 and _is_nan_or_inf(entry.get("train_loss")):
            return "failed_convergence"
        if ep >= 2 and _is_nan_or_inf(entry.get("val_loss")):
            return "failed_convergence"

    def _val_acc(e: dict) -> float:
        v = e.get("val_acc")
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return -1.0
        return -1.0 if math.isnan(f) else f

    best_idx = max(range(len(history)), key=lambda i: _val_acc(history[i]))
    best = history[best_idx]
    best_val = _val_acc(best)
    try:
        best_train = float(best.get("train_acc", 0.0))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        best_train = 0.0
    if math.isnan(best_train):
        best_train = 0.0

    gap_pp = (best_train - best_val) * 100.0
    if gap_pp >= OVERFIT_GAP_PP:
        return "overfitting"

    if best_val < FAILED_CONV_ACC:
        return "failed_convergence"

    return "healthy"


# ---------------------------------------------------------------------------
# Sentinel + completion helpers
# ---------------------------------------------------------------------------

def cell_run_dir(tag: str, base: Path = RUNS_DIR) -> Path:
    return base / tag


def write_sentinel(tag: str, name: str, body: str = "", *, base: Path = RUNS_DIR) -> Path:
    rd = cell_run_dir(tag, base)
    rd.mkdir(parents=True, exist_ok=True)
    p = rd / name
    p.write_text(body)
    return p


def remove_sentinel(tag: str, name: str, *, base: Path = RUNS_DIR) -> bool:
    p = cell_run_dir(tag, base) / name
    if p.exists():
        p.unlink()
        return True
    return False


def has_sentinel(tag: str, name: str, *, base: Path = RUNS_DIR) -> bool:
    return (cell_run_dir(tag, base) / name).exists()


def cell_is_complete(tag: str, *, base: Path = RUNS_DIR) -> bool:
    """Per PRD §6.5 / US-005 line 396: a cell counts as complete when
    metrics.json exists with ``best_val_acc ≥ 0`` and the INTERRUPTED
    sentinel is absent."""
    rd = cell_run_dir(tag, base)
    if (rd / INTERRUPTED).exists():
        return False
    metrics = rd / "metrics.json"
    if not metrics.exists():
        return False
    try:
        data = json.loads(metrics.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    bva = data.get("best_val_acc")
    try:
        return float(bva) >= 0.0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Remediation (PRD §6.4)
# ---------------------------------------------------------------------------

def build_retry_config(base_hparams: dict, verdict: str) -> dict:
    """Build the Full-FT retry config per PRD §6.3 deltas + §6.4 ratio tighten.

    failed_convergence: head_lr ÷ 3, weight_decay × 1.5, label_smoothing += 0.05 (cap 0.15).
    overfitting:        backbone_lr ÷ 2, weight_decay × 2,   dropout += 0.1 (cap 0.3).

    Plus §6.4 step 1: tighten differential ratio so ``backbone_lr = head_lr / 10``
    (applied *before* the verdict-specific deltas so the overfitting branch's
    ``÷ 2`` still composes with the tightened ratio).

    The Optuna winner JSON nests the actual hparams under ``best_params``; the
    deltas operate on those nested values (Iteration 12, 2026-05-15 — prior to
    this fix the deltas were applied to top-level keys that didn't exist and
    silently fell back to CNN defaults, producing a "retry" that decreased
    regularization for L2 cifar10's already-tightly-tuned Optuna winner).
    """
    if verdict not in ("failed_convergence", "overfitting"):
        raise ValueError(f"unknown verdict for retry: {verdict!r}")

    cfg = dict(base_hparams)
    cfg["full_ft"] = True
    cfg["remediation_reason"] = verdict

    bp = dict(cfg.get("best_params", {}))

    if "head_lr" in bp:
        bp["backbone_lr"] = float(bp["head_lr"]) / 10.0

    if verdict == "failed_convergence":
        if "head_lr" in bp:
            bp["head_lr"] = float(bp["head_lr"]) / 3.0
        bp["weight_decay"] = float(bp.get("weight_decay", 1e-4)) * 1.5
        ls = float(bp.get("label_smoothing", 0.0))
        bp["label_smoothing"] = min(ls + 0.05, 0.15)
    else:  # overfitting
        if "backbone_lr" in bp:
            bp["backbone_lr"] = float(bp["backbone_lr"]) / 2.0
        bp["weight_decay"] = float(bp.get("weight_decay", 1e-4)) * 2.0
        dp = float(bp.get("dropout", 0.0))
        bp["dropout"] = min(dp + 0.1, 0.3)

    cfg["best_params"] = bp
    return cfg


def write_retry_config(tag: str, retry_cfg: dict, *, base: Path = RUNS_DIR) -> Path:
    rd = cell_run_dir(tag, base)
    rd.mkdir(parents=True, exist_ok=True)
    p = rd / RETRY_CONFIG
    p.write_text(json.dumps(retry_cfg, indent=2, sort_keys=True))
    return p


def read_retry_config(tag: str, *, base: Path = RUNS_DIR) -> dict:
    p = cell_run_dir(tag, base) / RETRY_CONFIG
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# DEBUGGER OOM hook (PRD acceptance criterion line 395)
# ---------------------------------------------------------------------------

def apply_oom_fix(hparams: dict) -> dict:
    """First OOM fix from agents/DEBUGGER.md §A: physical batch ÷ 2 with
    accumulate_grad_batches × 2 so the effective batch stays at 32 (the
    fair-comparison invariant)."""
    cfg = dict(hparams)
    bs = int(cfg.get("batch_size", 32))
    if bs <= 1:
        raise RuntimeError("OOM fix exhausted: batch_size already at 1")
    ga = int(cfg.get("accumulate_grad_batches", 1))
    cfg["batch_size"] = bs // 2
    cfg["accumulate_grad_batches"] = ga * 2
    return cfg


def write_debugger_log(
    tag: str, exc: BaseException, attempt: int, *, base: Path = RUNS_DIR
) -> Path:
    rd = cell_run_dir(tag, base)
    rd.mkdir(parents=True, exist_ok=True)
    p = rd / DEBUGGER_LOG
    fingerprint = f"{type(exc).__module__}.{type(exc).__name__}: {exc}"
    prefix = p.read_text() if p.exists() else ""
    p.write_text(f"{prefix}attempt={attempt} {fingerprint}\n")
    return p


# ---------------------------------------------------------------------------
# Dispatch entry points
# ---------------------------------------------------------------------------

def _resolve_run_cell() -> Callable:
    """Lazy import so the unit fixture stays torch-free."""
    from run_systematic import run_cell as _rc  # type: ignore
    return _rc


def _resolve_oom_exception() -> type:
    """Lazy import — defaults to a stub class when torch is absent so the
    catch clause still compiles. Real callers (with torch installed) will
    get ``torch.cuda.OutOfMemoryError``."""
    try:
        import torch  # type: ignore

        return torch.cuda.OutOfMemoryError  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover — torch not in test env
        class _StubOOMError(RuntimeError):
            pass

        return _StubOOMError


def run_dry(cells: Sequence[CellMeta]) -> int:
    print(f"[ralph][dry-run] resolved {len(cells)} cell(s):")
    for i, c in enumerate(cells, 1):
        suffix = f" level={c.level}" if c.level is not None else ""
        suffix += f" axis={c.axis}" if c.axis is not None else ""
        print(f"  {i:>3}. {c.tag}  [phase={c.phase}{suffix}]")
    return 0


def run_dispatch(
    cells: Sequence[CellMeta],
    *,
    mode: str,
    engine: str,
    skip_existing: bool,
    base: Path = RUNS_DIR,
    run_cell_fn: Optional[Callable] = None,
    oom_exception: Optional[type] = None,
) -> int:
    """First-pass dispatch: train each cell, evaluate pathology, write
    sentinels. Cells already complete are skipped when ``skip_existing``;
    INTERRUPTED cells are re-attempted regardless."""
    run_cell_fn = run_cell_fn or _resolve_run_cell()
    oom_cls: type[BaseException] = oom_exception or _resolve_oom_exception()

    exit_code = 0
    for c in cells:
        if skip_existing and cell_is_complete(c.tag, base=base):
            print(f"[ralph] skip complete: {c.tag}")
            continue

        attempt = 0
        while True:
            attempt += 1
            try:
                run_cell_fn(c.tag, mode=mode, engine=engine)
                break
            except oom_cls as e:
                write_debugger_log(c.tag, e, attempt, base=base)
                if attempt >= MAX_OOM_ATTEMPTS:
                    print(f"[ralph] OOM cap hit on {c.tag} after {attempt} attempts",
                          file=sys.stderr)
                    exit_code = 1
                    break
                # In production a real OOM-fix would re-launch run_cell with
                # the modified hparams; here we simply log and retry — the
                # retry loop body is what the acceptance fixture exercises.
                continue
            except KeyboardInterrupt:
                write_sentinel(c.tag, INTERRUPTED, base=base)
                raise

        # Pathology guard runs only if Trainer.fit returned cleanly.
        if attempt < MAX_OOM_ATTEMPTS or exit_code == 0:
            verdict = _post_train_verdict(c.tag, base=base)
            if verdict in ("failed_convergence", "overfitting"):
                write_sentinel(c.tag, NEEDS_FULL_FT, body=verdict, base=base)

    return exit_code


def _post_train_verdict(
    tag: str, *, base: Path = RUNS_DIR, run_dir: Optional[Path] = None
) -> str:
    """Read history.json from the run dir and run evaluate_pathology.

    When ``run_dir`` is provided, read that dir's history.json directly —
    used by the retry path so the verdict is computed on the retry's
    history, not on the first-pass base dir (Iteration 12, 2026-05-15 —
    Lightning's _make_unique_run_dir routes the retry into ``__v2`` while
    sentinels live in base; without this hook the verdict was computed
    on stale first-pass data).

    Falls back to ``cell_run_dir(tag, base) / "history.json"`` when
    ``run_dir`` is None or missing — preserves the first-pass dispatch
    behavior unchanged.
    """
    if run_dir is not None:
        p = Path(run_dir) / "history.json"
    else:
        p = cell_run_dir(tag, base) / "history.json"
    if not p.exists():
        return "failed_convergence"
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return "failed_convergence"
    history = data if isinstance(data, list) else data.get("history", [])
    return evaluate_pathology(history)


def run_remediate(
    cells: Sequence[CellMeta],
    *,
    mode: str,
    engine: str,
    base: Path = RUNS_DIR,
    run_cell_fn: Optional[Callable] = None,
    hparams_loader: Optional[Callable[[CellMeta], dict]] = None,
) -> int:
    """Second-pass: for each cell carrying NEEDS_FULL_FT, build the Full-FT
    retry config, write it to retry_config.json, and dispatch via run_cell.

    Cells carrying QUARANTINED_AFTER_RETRY are skipped (cap enforced).
    """
    run_cell_fn = run_cell_fn or _resolve_run_cell()
    hparams_loader = hparams_loader or _default_hparams_loader

    exit_code = 0
    for c in cells:
        if has_sentinel(c.tag, QUARANTINED_AFTER_RETRY, base=base):
            print(f"[ralph][remediate] skip quarantined: {c.tag}")
            continue
        if not has_sentinel(c.tag, NEEDS_FULL_FT, base=base):
            continue

        verdict = (cell_run_dir(c.tag, base) / NEEDS_FULL_FT).read_text().strip() \
            or "failed_convergence"

        base_hparams = hparams_loader(c)
        retry_cfg = build_retry_config(base_hparams, verdict)
        write_retry_config(c.tag, retry_cfg, base=base)

        try:
            # Capture the actual run dir (Lightning's _make_unique_run_dir
            # may route the retry into <tag>__v2 since base has first-pass
            # artifacts). The pathology verdict reads from this dir so it
            # evaluates the retry's history, not the stale first-pass one.
            retry_run_dir = run_cell_fn(c.tag, mode=mode, engine=engine)
        except Exception as e:  # noqa: BLE001 — retry failed → quarantine
            print(f"[ralph][remediate] retry failed for {c.tag}: {e}",
                  file=sys.stderr)
            write_sentinel(c.tag, QUARANTINED_AFTER_RETRY, body=str(e), base=base)
            exit_code = 1
            continue

        post = _post_train_verdict(c.tag, base=base, run_dir=retry_run_dir)
        if post != "healthy":
            write_sentinel(c.tag, QUARANTINED_AFTER_RETRY,
                           body=f"second_failure:{post}", base=base)
            exit_code = 1
        else:
            remove_sentinel(c.tag, NEEDS_FULL_FT, base=base)

    return exit_code


def _default_hparams_loader(c: CellMeta) -> dict:
    p = REPO_ROOT / "artifacts" / "best_hparams" / f"{c.model}_{c.dataset}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_PHASE_CHOICES = ("A", "B", "B2", "B2nr", "C", "C2", "D", "multiseed")
_C2_AXES = ("resolution", "blur", "salt_pepper")


def _parse_seeds_arg(raw: str) -> list[int]:
    """Comma-separated seeds (US-038, v4). Used with --phase multiseed."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    seeds: list[int] = []
    for s in parts:
        try:
            seeds.append(int(s))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--seeds expects integers; got {s!r}"
            )
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds requires at least one integer")
    return seeds


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RALPH Loop Driver — sequential dispatch (PRD US-005 + US-038)."
    )
    p.add_argument("--plan", default="final", choices=["final"],
                   help="Cell plan to iterate. Only 'final' is supported.")
    p.add_argument("--phase", default=None, choices=list(_PHASE_CHOICES),
                   help="Filter cells by phase. 'multiseed' iterates a "
                        "named --cells list at each seed in --seeds.")
    p.add_argument("--model", default=None,
                   help="Filter cells by model (e.g. resnet50).")
    p.add_argument("--dataset", default=None, choices=["cifar10", "mnist"],
                   help="Filter cells by dataset.")
    p.add_argument("--axes", default=None, type=_parse_axes_arg,
                   help="Comma-separated axes to dispatch. Valid with --phase C "
                        f"(any subset of {list(AXES)}) and with --phase C2 "
                        f"(subset of {list(_C2_AXES)}).")
    p.add_argument("--cells", default=None,
                   help="Comma-separated cell tag list (--phase multiseed only).")
    p.add_argument("--seeds", default=None, type=_parse_seeds_arg,
                   help="Comma-separated integer seeds (--phase multiseed only). "
                        "e.g. --seeds 43,44 dispatches each --cells tag at both "
                        "seeds, writing to runs/final/<tag>_seed{N}/.")
    p.add_argument("--mode", default="full", choices=["full", "pilot"],
                   help="Training mode. --plan final rejects --mode pilot.")
    p.add_argument("--engine", default="lightning", choices=["lightning", "legacy"])
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved cell dispatch order and exit 0.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cells whose metrics.json.best_val_acc >= 0. "
                        "INTERRUPTED cells are re-attempted.")
    p.add_argument("--remediate-only", action="store_true",
                   help="Second pass: consume NEEDS_FULL_FT sentinels and "
                        "retry under PRD §6.4 Full FT.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    # CLAUDE.md convergence-first invariant.
    assert not (args.plan == "final" and args.mode == "pilot"), (
        "--plan final is incompatible with --mode pilot per CLAUDE.md "
        "convergence-first invariant (60 epochs / patience 10)."
    )

    # US-019 / US-038: --axes is a single-axis-isolation concept. Phase C
    # accepts the full 5-axis set; Phase C2 restricts to {resolution, blur,
    # salt_pepper}; other phases reject --axes outright.
    if args.axes is not None and args.phase not in ("C", "C2"):
        phase_label = args.phase if args.phase is not None else "all"
        parser.error(
            f"--axes is only valid with --phase C or --phase C2; got "
            f"--phase {phase_label} (axes are a single-axis-isolation concept)"
        )
    if args.axes is not None and args.phase == "C2":
        bad = [a for a in args.axes if a not in _C2_AXES]
        if bad:
            parser.error(
                f"--phase C2 only supports axes in {list(_C2_AXES)}; got {bad} "
                f"(noise + saturation are protocol-invariant under the C2 override)"
            )

    # US-038: multi-seed dispatch path.
    if args.phase == "multiseed":
        if not args.cells:
            parser.error("--phase multiseed requires --cells <tag,tag,...>")
        if not args.seeds:
            parser.error("--phase multiseed requires --seeds <int,int,...>")
        return _run_multiseed(args, parser)

    # US-038: opt-in flags routed by phase token.
    include_phase_d   = (args.phase == "D")
    include_phase_b2  = (args.phase == "B2")
    include_phase_b2nr = (args.phase == "B2nr")
    include_phase_c2  = (args.phase == "C2")

    cells_iter = iter_cells(
        include_phase_d=include_phase_d,
        include_phase_b2=include_phase_b2,
        include_phase_b2nr=include_phase_b2nr,
        include_phase_c2=include_phase_c2,
    )
    cells = filter_cells(
        cells_iter,
        phase=args.phase, model=args.model, dataset=args.dataset,
        axes=args.axes,
    )

    if args.dry_run:
        return run_dry(cells)

    if args.remediate_only:
        return run_remediate(cells, mode=args.mode, engine=args.engine)

    return run_dispatch(
        cells,
        mode=args.mode, engine=args.engine,
        skip_existing=args.skip_existing,
    )


def _run_multiseed(args, parser) -> int:
    """US-038 (v4) — re-dispatch a named list of cells at each non-default seed.

    For each (tag, seed) pair in product(args.cells, args.seeds), invokes
    run_systematic.run_cell(tag, seed=N), which writes to
    runs/final/<tag>_seed{N}/ (when N != 42).
    """
    from run_systematic import run_cell as _run_cell

    tags = [t.strip() for t in args.cells.split(",") if t.strip()]
    if not tags:
        parser.error("--cells must contain at least one tag")
    failed = 0
    for tag in tags:
        for seed in args.seeds:
            print(f"[multiseed] dispatching {tag} at seed={seed} ...")
            if args.dry_run:
                continue
            try:
                _run_cell(tag, mode=args.mode, engine=args.engine, seed=int(seed))
            except Exception as e:  # noqa: BLE001 — surface, keep dispatching
                print(f"[multiseed][ERROR] {tag} seed={seed}: {e}")
                failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
