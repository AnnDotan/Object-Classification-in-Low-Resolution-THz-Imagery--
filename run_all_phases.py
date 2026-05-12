#!/usr/bin/env python3
"""
Run ALL 36 experiments from the experiment plan sequentially.

Phase A: CIFAR-10 systematic (9 experiments) — 3 levels × 3 models
Phase B: MNIST systematic (9 experiments) — 3 levels × 3 models
Phase C: Single-degradation isolation (12 experiments) — 4 types × 3 models
Phase D: Clean baselines (6 experiments) — 2 datasets × 3 models

Usage:
    python run_all_phases.py                    # run everything
    python run_all_phases.py --phase A          # Phase A only
    python run_all_phases.py --phase A,B        # Phase A + B
    python run_all_phases.py --phase C,D        # Phase C + D
    python run_all_phases.py --skip-existing    # skip experiments whose run dir exists
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def _resolve_run_experiment(engine: str):
    """Pick the legacy or Lightning training entry. Same call signature."""
    if engine == "lightning":
        from src.lightning.train import run_experiment
    else:
        from src.runner import run_experiment
    return run_experiment

# ── Shared config ──
COMMON = {
    "pretrained": True,
    "out_size": 224,
    "batch_size": 32,
    "scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "group": "systematic",
    "epochs": 30,
    "train_subset": 10000,
    "val_subset": 5000,
    "early_stopping_patience": 5,
}

MODEL_CONFIGS = {
    # ResNet50 — TResNet/EfficientNetV2 paper recommendations
    # Lower backbone_lr to reduce overfitting (14% train-val gap at L1)
    # Stronger weight_decay + label_smoothing for regularization
    # Warmup per EfficientNetV2 progressive training strategy
    "resnet50": {
        "model_name": "resnet50",
        "lr": 1e-3,
        "backbone_lr": 5e-5,          # was 1e-4: slower adaptation, less forgetting
        "freeze_backbone": False,
        "weight_decay": 5e-4,          # was 1e-4: stronger L2 regularization
        "label_smoothing": 0.15,       # was 0.1: moderate increase
        "warmup_epochs": 3,            # EfficientNetV2 warmup recommendation
    },
    # DenseNet121 — DenseNet paper recommendations
    # Train acc hits 99%+ → needs strong label smoothing
    # Dense feature reuse already regularizes, but gap still ~19%
    # Lower backbone_lr + stronger weight_decay to combat overfitting
    "densenet121": {
        "model_name": "densenet121",
        "lr": 1e-3,
        "backbone_lr": 5e-5,          # was 1e-4: more conservative
        "freeze_backbone": False,
        "weight_decay": 5e-4,          # was 1e-4: stronger regularization
        "label_smoothing": 0.2,        # was 0.1: strong smoothing for 99%+ train acc
        "warmup_epochs": 2,            # shorter warmup (DenseNet converges fast)
    },
    # TransNeXt Micro — TransNeXt paper recommendations
    # Frozen backbone gave very low overfitting but low accuracy (68% L1)
    # Paper uses backbone_lr=5e-5 for fine-tuning; we use 1e-5 (conservative)
    # weight_decay=0.05 per TransNeXt paper fine-tuning recipe
    # drop_path_rate=0.1 per TransNeXt paper
    # 5-epoch warmup per TransNeXt training protocol
    "transnext_micro": {
        "model_name": "transnext_micro",
        "lr": 1e-3,                    # head LR
        "backbone_lr": 1e-5,           # was None/frozen: very conservative unfreeze
        "freeze_backbone": False,      # was True: allow backbone adaptation
        "weight_decay": 0.05,          # TransNeXt paper fine-tuning value
        "label_smoothing": 0.1,        # was 0.0: light smoothing for fine-tuning
        "warmup_epochs": 5,            # TransNeXt paper training protocol
        "drop_path_rate": 0.1,         # TransNeXt paper stochastic depth
    },
}

DEGRADATION_LEVELS = {
    1: {"name": "mild",     "low_res": 16, "blur_kernel": 3, "blur_sigma": 0.5,
        "gaussian_noise_std": 0.04, "salt_pepper_amount": 0.02, "p_grayscale": 0.3},
    2: {"name": "moderate", "low_res": 16, "blur_kernel": 5, "blur_sigma": 1.0,
        "gaussian_noise_std": 0.08, "salt_pepper_amount": 0.05, "p_grayscale": 0.3},
    3: {"name": "severe",   "low_res": 8,  "blur_kernel": 7, "blur_sigma": 1.5,
        "gaussian_noise_std": 0.12, "salt_pepper_amount": 0.08, "p_grayscale": 0.3},
}

MODELS = ["resnet50", "densenet121", "transnext_micro"]


def build_all_experiments():
    """Build list of all 36 experiments."""
    experiments = []

    # ── Phase A: CIFAR-10 systematic (9) ──
    for level_id, level in DEGRADATION_LEVELS.items():
        for model in MODELS:
            tag = f"sys_L{level_id}_{level['name']}_{model}"
            experiments.append({
                "phase": "A",
                "tag": tag,
                "dataset": "cifar10",
                "degradation_type": "all",
                **COMMON,
                **MODEL_CONFIGS[model],
                "low_res": level["low_res"],
                "blur_kernel": level["blur_kernel"],
                "blur_sigma": level["blur_sigma"],
                "gaussian_noise_std": level["gaussian_noise_std"],
                "salt_pepper_amount": level["salt_pepper_amount"],
                "p_grayscale": level["p_grayscale"],
            })

    # ── Phase B: MNIST systematic (9) ──
    for level_id, level in DEGRADATION_LEVELS.items():
        for model in MODELS:
            tag = f"sys_L{level_id}_{level['name']}_{model}_mnist"
            experiments.append({
                "phase": "B",
                "tag": tag,
                "dataset": "mnist",
                "degradation_type": "all",
                **COMMON,
                **MODEL_CONFIGS[model],
                "low_res": level["low_res"],
                "blur_kernel": level["blur_kernel"],
                "blur_sigma": level["blur_sigma"],
                "gaussian_noise_std": level["gaussian_noise_std"],
                "salt_pepper_amount": level["salt_pepper_amount"],
                "p_grayscale": level["p_grayscale"],
            })

    # ── Phase C: Single-degradation isolation (12) — CIFAR-10, Level 2 params ──
    isolation_configs = {
        "downsampling": {
            "low_res": 16, "blur_kernel": None, "blur_sigma": None,
            "gaussian_noise_std": None, "salt_pepper_amount": None, "p_grayscale": None,
        },
        "blur": {
            "low_res": 16, "blur_kernel": 5, "blur_sigma": 1.0,
            "gaussian_noise_std": None, "salt_pepper_amount": None, "p_grayscale": None,
        },
        "noise": {
            "low_res": 16, "blur_kernel": None, "blur_sigma": None,
            "gaussian_noise_std": 0.08, "salt_pepper_amount": None, "p_grayscale": None,
        },
        "salt_pepper": {
            "low_res": 16, "blur_kernel": None, "blur_sigma": None,
            "gaussian_noise_std": None, "salt_pepper_amount": 0.05, "p_grayscale": None,
        },
    }
    for iso_type, iso_params in isolation_configs.items():
        for model in MODELS:
            tag = f"iso_{iso_type}_{model}"
            exp = {
                "phase": "C",
                "tag": tag,
                "dataset": "cifar10",
                "degradation_type": iso_type,
                **COMMON,
                **MODEL_CONFIGS[model],
                "low_res": iso_params["low_res"],
            }
            # Only set non-None params
            for k in ["blur_kernel", "blur_sigma", "gaussian_noise_std",
                       "salt_pepper_amount", "p_grayscale"]:
                if iso_params[k] is not None:
                    exp[k] = iso_params[k]
            experiments.append(exp)

    # ── Phase D: Clean baselines (6) — no degradation ──
    for ds in ["cifar10", "mnist"]:
        for model in MODELS:
            ds_suffix = f"_{ds}" if ds == "mnist" else ""
            tag = f"clean_{model}{ds_suffix}"
            experiments.append({
                "phase": "D",
                "tag": tag,
                "dataset": ds,
                "degradation_type": "all",
                **COMMON,
                **MODEL_CONFIGS[model],
                "low_res": 224,   # no downsampling
                "blur_kernel": 1,
                "blur_sigma": 0.0,
                "gaussian_noise_std": 0.0,
                "salt_pepper_amount": 0.0,
                "p_grayscale": 0.0,
            })

    return experiments


def find_existing_run(tag: str) -> bool:
    """Check if a run with this tag already exists in runs/systematic/."""
    sys_dir = Path("runs") / "systematic"
    if not sys_dir.exists():
        return False
    for d in sys_dir.iterdir():
        if d.is_dir() and d.name.startswith(tag):
            # Check if it has metrics.csv with >0 rows
            metrics = d / "metrics.csv"
            if metrics.exists() and metrics.stat().st_size > 50:
                return True
    return False


def run_single(exp: dict, exp_num: int, total: int, engine: str = "lightning"):
    """Run a single experiment, stripping out internal keys."""
    tag = exp["tag"]
    phase = exp["phase"]
    run_experiment = _resolve_run_experiment(engine)

    # Remove 'phase' key before passing to run_experiment
    config = {k: v for k, v in exp.items() if k != "phase"}

    print(f"\n{'━' * 70}")
    print(f"  [{exp_num}/{total}] Phase {phase} | {config['model_name']} | {tag}  [engine={engine}]")
    print(f"  dataset={config['dataset']}, degradation={config['degradation_type']}, "
          f"low_res={config['low_res']}")
    print(f"{'━' * 70}\n")

    t0 = time.time()
    try:
        run_experiment(**config)
        dt = time.time() - t0
        return {"tag": tag, "phase": phase, "status": "OK", "time": f"{dt:.1f}s"}
    except Exception as e:
        dt = time.time() - t0
        print(f"\n[ERROR] {tag}: {e}")
        return {"tag": tag, "phase": phase, "status": f"FAILED: {e}", "time": f"{dt:.1f}s"}


# ============================================================
# 186-CELL FINAL PLAN orchestration (US-009)
# ============================================================

# Phases that exist in the 186-cell matrix (build_final_matrix). Distinct
# from the legacy A/B/C/D phases in this file (which mean different things).
FINAL_PHASES = ("A", "B", "C")


def _has_completed_metrics(metrics_path: Path) -> bool:
    """A cell counts as 'done' if metrics.json exists and contains a finite
    best_val_acc (the LegacyJSONMetricsCallback writes -1.0 if no epoch ran).
    Acceptance criterion uses 'final_val_acc' but our schema uses
    'best_val_acc' / 'last_val_acc'; we accept either to be forgiving."""
    if not metrics_path.exists():
        return False
    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for key in ("final_val_acc", "best_val_acc", "last_val_acc"):
        v = m.get(key)
        if isinstance(v, (int, float)) and v >= 0.0:
            return True
    return False


def _required_hparams_present(matrix) -> tuple[bool, list[str]]:
    """Return (all_present, missing_paths) for the (model, dataset) pairs
    needed by the matrix."""
    needed = {(c.model, c.dataset) for c in matrix}
    missing = []
    for model, dataset in sorted(needed):
        path = Path("artifacts/best_hparams") / f"{model}_{dataset}.json"
        if not path.exists():
            missing.append(str(path))
    return (not missing, missing)


def _load_refresh_trackers():
    """Load scripts/refresh_trackers.py via importlib (no scripts/__init__.py)."""
    import importlib.util
    src = Path(__file__).resolve().parent / "scripts" / "refresh_trackers.py"
    spec = importlib.util.spec_from_file_location("scripts_refresh_trackers", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_sync_trackers_git():
    import importlib.util
    src = Path(__file__).resolve().parent / "scripts" / "sync_trackers_git.py"
    spec = importlib.util.spec_from_file_location("scripts_sync_trackers_git", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _refresh_trackers_default(cell: Optional[str] = None) -> None:
    """Default per-cell refresh hook (US-019).

    When `cell` is provided, runs the incremental refresh path: patches only
    that row in Final_Exp.json and rebuilds the HTML; MD regen is skipped.
    Errors are logged but never raised — mirrors the previous behavior.
    """
    try:
        mod = _load_refresh_trackers()
        result = mod.refresh_all(cell=cell) if cell else mod.refresh_all()
        for err in result.get("errors", []):
            print(f"[refresh][WARN] {err}", file=sys.stderr)
    except Exception as e:
        print(f"[refresh][WARN] refresh_all raised: {e}", file=sys.stderr)


# ============================================================
# V3 Insurance Trial (Phase-B sanity check)
# ============================================================
# The 186-cell matrix runs CNNs at 224 (their ImageNet-pretrain regime) and
# TransNeXt at native 32 (V3 path). This trial trains a single ResNet50 cell
# at native 32x32 / L1 / CIFAR-10 so the paper can quote the empirical penalty
# of forcing a CNN out of its training resolution — justifying the asymmetric
# design instead of arguing it from architecture alone.
#
# Lives OUTSIDE the 186-cell matrix (matrix denominator stays 186). Output
# goes to runs/insurance/<tag>/ so dashboards that scan runs/final/ don't pick
# it up by accident.

INSURANCE_TRIAL_TAG = "insurance_resnet50_native32_L1_cifar10"


def _build_insurance_trial_config() -> dict:
    """Hardcoded config for the V3 insurance trial.

    L1 (Mild) degradation parameters lifted verbatim from
    src/data/degradation_levels.py so this stays in sync with the main
    plan's L1 cells. ResNet50 hyperparameters mirror the main matrix's
    resnet50 row so the only experimental variable is `out_size`.
    """
    return dict(
        model_name="resnet50",
        pretrained=True,
        dataset="cifar10",
        # The variable under test: native 32 vs the matrix's 224.
        out_size=32,
        img_size=32,
        low_res=20,           # L1 from degradation_levels.py
        blur_kernel=3,
        blur_sigma=0.70,
        gaussian_noise_std=0.03,
        salt_pepper_amount=0.02,
        saturation=1.00,
        degradation_type="all",
        # ResNet50 row from MODEL_CONFIGS — keep identical so out_size is
        # the only confound.
        lr=1e-3,
        backbone_lr=5e-5,
        freeze_backbone=False,
        weight_decay=5e-4,
        label_smoothing=0.15,
        warmup_epochs=3,
        scheduler_type="cosine",
        max_grad_norm=1.0,
        # Final-plan training budget per CLAUDE.md.
        epochs=60,
        batch_size=32,
        train_subset=10000,
        val_subset=5000,
        early_stopping_patience=10,
        # Output routing.
        group="insurance",
        run_name_override=INSURANCE_TRIAL_TAG,
    )


def _run_insurance_trial(
    *,
    engine: str = "lightning",
    skip_existing: bool = True,
    run_experiment_fn=None,
) -> dict:
    """Fire the V3 insurance trial. Fail-soft: errors are logged, never raised.

    Returns a status dict {tag, status, time?} suitable for inclusion in the
    run_final_plan results JSON.
    """
    tag = INSURANCE_TRIAL_TAG
    run_dir = Path("runs") / "insurance" / tag
    if skip_existing and _has_completed_metrics(run_dir / "metrics.json"):
        print(f"[insurance] SKIP {tag} — metrics.json already complete")
        return {"tag": tag, "phase": "INSURANCE", "status": "SKIPPED"}

    config = _build_insurance_trial_config()
    run_experiment = run_experiment_fn or _resolve_run_experiment(engine)

    print("\n" + "=" * 72)
    print(f"  [INSURANCE TRIAL] {tag}")
    print(f"  ResNet50 @ native 32x32 / L1 Mild / CIFAR-10")
    print(f"  Purpose: quantify CNN penalty at non-native resolution.")
    print("=" * 72)

    t0 = time.time()
    try:
        run_experiment(**config)
        dt = time.time() - t0
        print(f"[insurance] OK {tag} ({dt:.1f}s)")
        return {"tag": tag, "phase": "INSURANCE", "status": "OK", "time": f"{dt:.1f}s"}
    except Exception as e:
        dt = time.time() - t0
        print(f"[insurance][WARN] {tag} failed (continuing main plan): "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return {
            "tag": tag, "phase": "INSURANCE",
            "status": f"FAILED: {type(e).__name__}: {e}",
            "time": f"{dt:.1f}s",
        }


def run_final_plan(
    *,
    phase: str = "all",
    mode: str = "full",
    skip_existing: bool = True,
    tune_first: bool = False,
    engine: str = "lightning",
    model: Optional[str] = None,    # PRD US-042: filter matrix by model name
    dataset: Optional[str] = None,  # PRD US-042: filter matrix by dataset
    dry_run: bool = False,          # PRD US-042: resolve + print CellSpec(s), no training
    run_cell_fn=None,           # injectable for testing
    tune_all_fn=None,           # injectable for testing
    refresh_fn=None,            # injectable for testing (US-016: refreshes BOTH trackers)
    phase_boundary_fn=None,     # injectable for testing (US-016: SYNCHRONIZER push)
    insurance_trial_fn=None,    # injectable for testing (V3 insurance trial)
) -> int:
    """Iterate the 186-cell matrix, dispatching each cell via run_cell.

    Returns 0 if every cell completed (or was skipped), 1 if any cell failed.

    Per-cell: refresh_fn() updates Final_Exp.md AND Final_Exp.html.
    Per-phase boundary: phase_boundary_fn(phase, n_cells) commits + pushes
    tracker files via SYNCHRONIZER (fail-soft).
    """
    assert mode != "pilot", (
        "--plan final --mode pilot is rejected per CLAUDE.md "
        "(quality-over-speed is non-negotiable for the campaign)."
    )

    # Lazy imports so unit tests can swap in mocks without dragging in Lightning
    if run_cell_fn is None:
        from run_systematic import run_cell as run_cell_fn
    if refresh_fn is None:
        refresh_fn = _refresh_trackers_default
    if phase_boundary_fn is None:
        _sync = _load_sync_trackers_git()
        phase_boundary_fn = _sync.commit_and_push_phase_boundary
    from src.experiments.matrix import build_final_matrix

    matrix = build_final_matrix()
    if phase != "all":
        if phase not in FINAL_PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {FINAL_PHASES} or 'all'")
        matrix = [c for c in matrix if c.phase == phase]

    # PRD US-042: optional model/dataset filters. Used by `--dry-run` smoke
    # tests and by ad-hoc single-cell dispatches. The filters are AND-ed.
    if model is not None:
        matrix = [c for c in matrix if c.model == model]
    if dataset is not None:
        matrix = [c for c in matrix if c.dataset == dataset]

    # PRD US-042: `--dry-run` resolves CellSpecs (no training) and exits.
    # If the requested (model, dataset) is a `_native` alias not present in
    # the matrix (matrix uses bare `transnext_*` names under V3), synthesize
    # the spec from `_v3_cell_settings` so the operator can preview the
    # resolved DegradeConfig before the real training command runs.
    if dry_run:
        if not matrix and model is not None and dataset is not None:
            from src.experiments.matrix import _v3_cell_settings, CellSpec
            from src.data.degrade import degrade_config_for
            v3 = _v3_cell_settings(model, dataset)
            # Default to phase A clean baseline if no phase specified.
            phases_to_show = [phase] if phase != "all" else ["A"]
            for ph in phases_to_show:
                if ph == "A":
                    tag = f"final_clean_{model}_{dataset}"
                    lvl, ax = None, None
                else:
                    tag = f"final_{ph}_L3_{model}_{dataset}"
                    lvl, ax = 3, None
                deg = degrade_config_for(lvl, axis=ax, out_size=v3["out_size"])
                synthesized = CellSpec(
                    tag=tag, phase=ph, model=model, dataset=dataset,
                    level=lvl, axis=ax, degrade_config=deg, **v3,
                )
                matrix.append(synthesized)
        print("=" * 72)
        print(f"  DRY RUN — phase={phase}, model={model}, dataset={dataset}, "
              f"resolved {len(matrix)} cell(s)")
        print("=" * 72)
        for c in matrix:
            print(f"\n  tag             : {c.tag}")
            print(f"  phase           : {c.phase}")
            print(f"  model / dataset : {c.model} / {c.dataset}")
            print(f"  level / axis    : {c.level} / {c.axis}")
            print(f"  out_size        : {c.out_size}")
            print(f"  img_size        : {c.img_size}")
            print(f"  patch_size      : {c.patch_size}")
            print(f"  pretrain_size   : {c.pretrain_size}")
            print(f"  precision       : {c.precision}")
            print(f"  compile_mode    : {c.compile_mode}")
            print(f"  mnist_pad_to_32 : {c.mnist_pad_to_32}")
            print(f"  degrade_config  : {c.degrade_config}")
        return 0

    # V3 quarantine lift (ratified 2026-05-12): TransNeXt cells now dispatch
    # alongside CNN cells. The legacy US-014 filter block that filtered out
    # `is_quarantined(c.model)` here was removed when the RTX 5070 / Blackwell
    # hardware came online. `is_quarantined` itself is now a no-op predicate
    # (always returns False) so tracker / Optuna code that still calls it
    # continues to work without quarantining anything.

    if tune_first:
        all_present, missing = _required_hparams_present(matrix)
        if not all_present:
            print(f"[tune-first] missing {len(missing)} best_hparams files; "
                  f"running tune_all.py first.", file=sys.stderr)
            if tune_all_fn is None:
                import subprocess
                rc = subprocess.run(
                    [sys.executable, "tune_all.py", "--n-trials", "20"],
                    check=False,
                ).returncode
            else:
                rc = tune_all_fn()
            if rc != 0:
                print(f"[tune-first] tune_all.py failed (rc={rc}); aborting.",
                      file=sys.stderr)
                return rc

    # V3 insurance trial: ResNet50 @ native 32 / L1 / CIFAR-10. Runs
    # alongside Phase B (or the full "all" pass) but is NOT part of the
    # 186-cell matrix. Fail-soft — its result is recorded but never aborts
    # the main plan.
    insurance_result: Optional[dict] = None
    if phase in ("B", "all"):
        _insurance = insurance_trial_fn or _run_insurance_trial
        try:
            insurance_result = _insurance(engine=engine, skip_existing=skip_existing)
        except Exception as e:
            print(f"[insurance][WARN] insurance_trial_fn raised: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            insurance_result = {
                "tag": INSURANCE_TRIAL_TAG,
                "phase": "INSURANCE",
                "status": f"FAILED: {type(e).__name__}: {e}",
            }

    print("=" * 72)
    print(f"  186-CELL FINAL PLAN — phase={phase}, mode={mode}, "
          f"skip_existing={skip_existing}, n={len(matrix)}")
    print("=" * 72)

    # Phase boundary tracker: count attempted (run + skipped) cells per phase so
    # we can fire commit_and_push_phase_boundary once each phase reaches its
    # full expected count.
    phase_attempted: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    phase_total: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for c in matrix:
        phase_total[c.phase] += 1

    # SIGINT handler: on Ctrl-C, drop an INTERRUPTED sentinel under the
    # currently-running cell (US-016) so detect_status flips it to Failed,
    # then flush a tracker refresh before re-raising so the dashboard
    # reflects the partial state.
    import signal
    from src.experiments.run_status import INTERRUPTED_SENTINEL
    _interrupted = {"flag": False}
    _current_cell: dict[str, Optional[str]] = {"tag": None}
    def _on_sigint(_sig, _frm):
        if _interrupted["flag"]:
            return
        _interrupted["flag"] = True
        active = _current_cell["tag"]
        if active:
            sentinel = Path("runs") / "final" / active / INTERRUPTED_SENTINEL
            try:
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                sentinel.write_text(
                    f"sigint at cell {active}\n", encoding="utf-8",
                )
                print(f"\n[run_all_phases] SIGINT — wrote {sentinel}",
                      file=sys.stderr)
            except OSError as e:
                print(f"[run_all_phases][WARN] could not write sentinel: {e}",
                      file=sys.stderr)
        else:
            print("\n[run_all_phases] SIGINT received between cells.",
                  file=sys.stderr)
        try:
            # Incremental refresh on SIGINT (US-019): patch only the
            # interrupted cell's row so the dashboard surfaces Failed
            # without re-scanning the full 186-cell matrix.
            if active:
                refresh_fn(cell=active)
            else:
                refresh_fn()
        except Exception as e:
            print(f"[run_all_phases][WARN] refresh on SIGINT raised: {e}",
                  file=sys.stderr)
        raise KeyboardInterrupt
    _prev_handler = signal.signal(signal.SIGINT, _on_sigint)

    results: list[dict] = []
    failed = 0
    skipped = 0
    for i, spec in enumerate(matrix, start=1):
        run_dir = Path("runs") / "final" / spec.tag
        if skip_existing and _has_completed_metrics(run_dir / "metrics.json"):
            print(f"  [SKIP {i}/{len(matrix)}] {spec.tag} — metrics.json already complete")
            results.append({"tag": spec.tag, "phase": spec.phase, "status": "SKIPPED"})
            skipped += 1
            # Still count toward phase boundary so the SYNCHRONIZER fires when
            # the LAST phase cell is reached, even if it was skipped.
            phase_attempted[spec.phase] += 1
            if phase_attempted[spec.phase] == phase_total[spec.phase]:
                try:
                    sync_result = phase_boundary_fn(spec.phase, phase_total[spec.phase])
                    if sync_result.get("errors"):
                        for err in sync_result["errors"]:
                            print(f"[sync][WARN] {err}", file=sys.stderr)
                except Exception as e:
                    print(f"[sync][WARN] phase_boundary_fn raised: {e}", file=sys.stderr)
            continue

        print(f"\n  [{i}/{len(matrix)}] dispatch {spec.tag} (phase {spec.phase}) ...")
        t0 = time.time()
        _current_cell["tag"] = spec.tag
        try:
            run_cell_fn(spec.tag, mode=mode, engine=engine)
            dt = time.time() - t0
            results.append({
                "tag": spec.tag, "phase": spec.phase,
                "status": "OK", "time": f"{dt:.1f}s",
            })
            print(f"    [OK] {spec.tag} ({dt:.1f}s)")
        except Exception as e:
            dt = time.time() - t0
            failed += 1
            print(f"    [FAIL] {spec.tag}: {type(e).__name__}: {e}", file=sys.stderr)
            results.append({
                "tag": spec.tag, "phase": spec.phase,
                "status": f"FAILED: {type(e).__name__}: {e}",
                "time": f"{dt:.1f}s",
            })
            # Continue to next cell — no crash-stop.
        finally:
            _current_cell["tag"] = None

        # Incremental tracker refresh after every cell (US-019): patch only
        # this row's hydration in Final_Exp.json + rebuild the HTML; MD is
        # skipped here because the next phase boundary (or a manual full
        # refresh) will regenerate it.
        try:
            refresh_fn(cell=spec.tag)
        except TypeError:
            # Test stubs may not accept the kwarg; fall back to full refresh.
            try:
                refresh_fn()
            except Exception as e:
                print(f"[WARN] tracker refresh raised: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] tracker refresh raised: {e}", file=sys.stderr)

        # Phase boundary check: if this cell completed the phase, fire SYNCHRONIZER.
        phase_attempted[spec.phase] += 1
        if phase_attempted[spec.phase] == phase_total[spec.phase]:
            try:
                sync_result = phase_boundary_fn(spec.phase, phase_total[spec.phase])
                if sync_result.get("errors"):
                    for err in sync_result["errors"]:
                        print(f"[sync][WARN] {err}", file=sys.stderr)
            except Exception as e:
                # phase_boundary_fn is supposed to be fail-soft, but defend anyway.
                print(f"[sync][WARN] phase_boundary_fn raised: {e}", file=sys.stderr)

    # Restore prior SIGINT handler.
    try:
        signal.signal(signal.SIGINT, _prev_handler)
    except Exception:
        pass

    completed = sum(1 for r in results if r["status"] == "OK")
    print("\n" + "=" * 72)
    print(f"  FINAL PLAN SUMMARY: {completed} OK, {failed} failed, "
          f"{skipped} skipped (of {len(matrix)} matrix cells)")
    if insurance_result is not None:
        print(f"  INSURANCE TRIAL: {insurance_result['status']}")
    print("=" * 72)

    # Append the insurance trial to the results JSON so it's visible to the
    # paper-figure scripts, but keep it tagged phase="INSURANCE" so they can
    # filter it out of 186-cell rollups.
    persisted_results = list(results)
    if insurance_result is not None:
        persisted_results.append(insurance_result)

    out = Path("artifacts/final_plan_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(persisted_results, indent=2), encoding="utf-8")
    print(f"  -> {out}")

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Run all experiment plan phases")
    parser.add_argument("--plan", default="legacy", choices=["legacy", "final"],
                        help="legacy: 36-run plan in this file; final: 186-cell campaign "
                             "from src/experiments/matrix.py.")
    parser.add_argument("--phase", default=None,
                        help="legacy plan: comma list from {A,B,C,D}; "
                             "final plan: one of A | B | C | all")
    parser.add_argument("--mode", default="full", choices=["pilot", "full"],
                        help="final plan: 'full' is required (pilot is rejected). "
                             "ignored by legacy plan.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments that already have results")
    parser.add_argument("--tune-first", action="store_true",
                        help="final plan: run tune_all.py first if any best_hparams "
                             "is missing.")
    parser.add_argument("--engine", default="lightning", choices=["lightning", "legacy"],
                        help="Training engine: lightning (default) or legacy hand-rolled trainer")
    parser.add_argument("--model", default=None,
                        help="final plan: filter matrix to one model (e.g. transnext_small_native). "
                             "Used by --dry-run smoke tests; ignored by legacy plan.")
    parser.add_argument("--dataset", default=None,
                        help="final plan: filter matrix to one dataset (cifar10 | mnist). "
                             "Used by --dry-run smoke tests; ignored by legacy plan.")
    parser.add_argument("--dry-run", action="store_true",
                        help="final plan: resolve filtered CellSpec(s) and print "
                             "the DegradeConfig + V3 routing; no training.")
    args = parser.parse_args()

    if args.plan == "final":
        # Reject --plan final --mode pilot with assertion (per CLAUDE.md).
        assert args.mode != "pilot", (
            "--plan final --mode pilot is rejected. "
            "Use --plan legacy for ad-hoc pilot smoke tests."
        )
        rc = run_final_plan(
            phase=(args.phase or "all"),
            mode=args.mode,
            skip_existing=args.skip_existing,
            tune_first=args.tune_first,
            engine=args.engine,
            model=args.model,
            dataset=args.dataset,
            dry_run=args.dry_run,
        )
        sys.exit(rc)

    # Legacy 36-run plan — original behavior preserved verbatim.
    if args.phase is None:
        args.phase = "A,B,C,D"

    phases = [p.strip().upper() for p in args.phase.split(",")]

    all_experiments = build_all_experiments()
    experiments = [e for e in all_experiments if e["phase"] in phases]

    print("\n" + "=" * 70)
    print("  FULL EXPERIMENT PLAN RUNNER")
    print("=" * 70)
    print(f"  Engine: {args.engine}")
    print(f"  Phases: {', '.join(phases)}")
    print(f"  Total experiments: {len(experiments)}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Epochs per experiment: {COMMON['epochs']}")
    print(f"  Early stopping patience: {COMMON['early_stopping_patience']}")
    print("=" * 70)

    results = []
    skipped = 0

    for i, exp in enumerate(experiments, 1):
        if args.skip_existing and find_existing_run(exp["tag"]):
            print(f"  [SKIP] {exp['tag']} — already exists")
            skipped += 1
            continue

        result = run_single(exp, i - skipped, len(experiments) - skipped, engine=args.engine)
        results.append(result)

        # Print running tally
        ok = sum(1 for r in results if r["status"] == "OK")
        failed = sum(1 for r in results if r["status"] != "OK")
        remaining = len(experiments) - skipped - len(results)
        print(f"\n  Progress: {ok} OK, {failed} failed, {remaining} remaining, {skipped} skipped")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("  EXPERIMENT PLAN — FINAL SUMMARY")
    print("=" * 70)
    for phase in phases:
        phase_results = [r for r in results if r["phase"] == phase]
        if phase_results:
            ok = sum(1 for r in phase_results if r["status"] == "OK")
            print(f"\n  Phase {phase}: {ok}/{len(phase_results)} completed")
            for r in phase_results:
                icon = "✓" if r["status"] == "OK" else "✗"
                print(f"    {icon} {r['tag']:<45} {r['status']:<10} {r['time']}")
    if skipped:
        print(f"\n  Skipped (already existed): {skipped}")
    print("=" * 70)

    # Save results
    summary_path = Path("artifacts") / "all_phases_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Full results saved to {summary_path}")


if __name__ == "__main__":
    main()
