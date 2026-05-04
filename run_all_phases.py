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


def _refresh_final_exp_md() -> None:
    """Best-effort call to scripts/update_final_exp.py after each cell.
    Missing script -> warn-and-continue (acceptance: 'no crash-stop')."""
    script = Path("scripts/update_final_exp.py")
    if not script.exists():
        # Don't spam every iteration — print once.
        if not getattr(_refresh_final_exp_md, "_warned", False):
            print(f"[WARN] {script} not found; Final_Exp.md will not auto-refresh.",
                  file=sys.stderr)
            _refresh_final_exp_md._warned = True
        return
    try:
        import subprocess
        subprocess.run(
            [sys.executable, str(script)],
            check=False, capture_output=True, timeout=30,
        )
    except Exception as e:
        print(f"[WARN] Final_Exp.md refresh failed: {e}", file=sys.stderr)


def run_final_plan(
    *,
    phase: str = "all",
    mode: str = "full",
    skip_existing: bool = True,
    tune_first: bool = False,
    engine: str = "lightning",
    run_cell_fn=None,           # injectable for testing
    tune_all_fn=None,           # injectable for testing
    refresh_fn=None,            # injectable for testing
) -> int:
    """Iterate the 186-cell matrix, dispatching each cell via run_cell.

    Returns 0 if every cell completed (or was skipped), 1 if any cell failed.
    """
    assert mode != "pilot", (
        "--plan final --mode pilot is rejected per CLAUDE.md "
        "(quality-over-speed is non-negotiable for the campaign)."
    )

    # Lazy imports so unit tests can swap in mocks without dragging in Lightning
    if run_cell_fn is None:
        from run_systematic import run_cell as run_cell_fn
    if refresh_fn is None:
        refresh_fn = _refresh_final_exp_md
    from src.experiments.matrix import build_final_matrix

    matrix = build_final_matrix()
    if phase != "all":
        if phase not in FINAL_PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {FINAL_PHASES} or 'all'")
        matrix = [c for c in matrix if c.phase == phase]

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

    print("=" * 72)
    print(f"  186-CELL FINAL PLAN — phase={phase}, mode={mode}, "
          f"skip_existing={skip_existing}, n={len(matrix)}")
    print("=" * 72)

    results: list[dict] = []
    failed = 0
    skipped = 0
    for i, spec in enumerate(matrix, start=1):
        run_dir = Path("runs") / "final" / spec.tag
        if skip_existing and _has_completed_metrics(run_dir / "metrics.json"):
            print(f"  [SKIP {i}/{len(matrix)}] {spec.tag} — metrics.json already complete")
            results.append({"tag": spec.tag, "phase": spec.phase, "status": "SKIPPED"})
            skipped += 1
            continue

        print(f"\n  [{i}/{len(matrix)}] dispatch {spec.tag} (phase {spec.phase}) ...")
        t0 = time.time()
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

        # Refresh Final_Exp.md after every cell (success OR fail), best-effort.
        try:
            refresh_fn()
        except Exception as e:
            print(f"[WARN] Final_Exp.md refresh raised: {e}", file=sys.stderr)

    completed = sum(1 for r in results if r["status"] == "OK")
    print("\n" + "=" * 72)
    print(f"  FINAL PLAN SUMMARY: {completed} OK, {failed} failed, "
          f"{skipped} skipped (of {len(matrix)} matrix cells)")
    print("=" * 72)

    out = Path("artifacts/final_plan_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
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
