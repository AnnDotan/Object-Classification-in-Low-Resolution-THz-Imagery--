"""Option C validation: re-train top-K Optuna trials per pair at full convergence.

Why this exists:
    `tune_all.py` (Stage 1) runs an Optuna pre-tune at low budget (5 epochs / 2k
    train subset / 1k val subset) for fast hparam ranking. The PRD-mandated
    Phase B sweep runs at 60 epochs / 10k train / 5k val with patience=10. This
    script bridges the gap: it loads the top-K candidates from the SQLite study,
    re-trains each at the production protocol, and writes the validated winner
    to artifacts/best_hparams/{model}_{dataset}.json.

Idempotency:
    - Per-pair: skipped if best_hparams JSON already has
      `validated_at_full_convergence: true`.
    - Per-trial: result cached at
      artifacts/validation/{model}_{dataset}_rank{N}.json (small JSON, no weights).

Fail-soft:
    Per-pair exceptions are logged + the next pair continues.

Privacy:
    Opens only Optuna SQLite, priors JSON, and dataset files via
    THzLikeCIFAR10/THzLikeMNIST. Never opens *.ckpt/*.pt/*.pth.

CLI:
    python scripts/validate_top3.py
    python scripts/validate_top3.py --model resnet50 --dataset cifar10
    python scripts/validate_top3.py --top-k 3 --max-epochs 60 --patience 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Running `python scripts/validate_top3.py` only adds `scripts/` to sys.path,
# but `src.*` imports inside _train_at_full_convergence need the repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_PAIRS = [
    ("resnet50",      "cifar10"),
    ("densenet121",   "cifar10"),
    ("transnext_base","cifar10"),
    ("resnet50",      "mnist"),
    ("densenet121",   "mnist"),
    ("transnext_base","mnist"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=("resnet50", "densenet121", "transnext_base"),
                        default=None, help="Restrict to one model (default: all).")
    parser.add_argument("--dataset", choices=("cifar10", "mnist"),
                        default=None, help="Restrict to one dataset.")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Re-train the top-K trials by fast val_acc.")
    parser.add_argument("--max-epochs", type=int, default=60,
                        help="Max epochs per validation training (CLAUDE.md production).")
    parser.add_argument("--train-subset", type=int, default=10000,
                        help="Full Phase B train subset.")
    parser.add_argument("--val-subset", type=int, default=5000,
                        help="Full Phase B val subset.")
    parser.add_argument("--patience", type=int, default=10,
                        help="EarlyStopping patience (CLAUDE.md production).")
    parser.add_argument("--storage", default="sqlite:///artifacts/optuna_thz.db",
                        help="Optuna storage URL.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "best_hparams",
                        help="Where to write validated winner JSONs.")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "artifacts" / "validation",
                        help="Where to cache per-trial validation results.")
    args = parser.parse_args(argv)

    pairs = DEFAULT_PAIRS
    if args.model and args.dataset:
        pairs = [(args.model, args.dataset)]
    elif args.model:
        pairs = [(args.model, d) for d in ("cifar10", "mnist")]
    elif args.dataset:
        pairs = [(m, args.dataset) for m in ("resnet50", "densenet121", "transnext_base")]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    rc = 0
    for model, dataset in pairs:
        try:
            validate_pair(model, dataset, args)
        except Exception as exc:
            print(f"FAIL {model}_{dataset}: {type(exc).__name__}: {exc}", file=sys.stderr)
            rc = 1
    return rc


def validate_pair(model: str, dataset: str, args) -> None:
    import optuna  # lazy import (heavy)

    out_path = args.out_dir / f"{model}_{dataset}.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if existing.get("validated_at_full_convergence"):
            print(f"SKIP {model}_{dataset}: already validated "
                  f"(best_value={existing.get('best_value', 'n/a')})")
            return

    study_name = f"{model}_{dataset}_L3"
    study = optuna.load_study(study_name=study_name, storage=args.storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < args.top_k:
        raise RuntimeError(
            f"Only {len(completed)} completed trials in {study_name}; "
            f"need >= {args.top_k}. Run `python tune_all.py --n-trials 20 "
            f"--model {model} --dataset {dataset}` first."
        )
    completed.sort(key=lambda t: t.value, reverse=True)  # direction=maximize
    top = completed[:args.top_k]

    print(f"=== Validating top-{args.top_k} trials for {model}_{dataset} ===")
    for rank, trial in enumerate(top):
        print(f"  rank {rank} (trial #{trial.number}): fast_val_acc={trial.value:.4f}")

    results = []
    for rank, trial in enumerate(top):
        cache_file = args.cache_dir / f"{model}_{dataset}_rank{rank}.json"
        if cache_file.exists():
            entry = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"  rank {rank}: cached full_val_acc={entry['full_val_acc']:.4f}")
        else:
            print(f"  rank {rank}: training trial #{trial.number} at "
                  f"{args.max_epochs}ep / {args.train_subset}train / {args.val_subset}val / "
                  f"patience={args.patience}", flush=True)
            full_acc = _train_at_full_convergence(model, dataset, dict(trial.params), args)
            entry = {
                "model": model,
                "dataset": dataset,
                "rank": rank,
                "trial_number": trial.number,
                "fast_val_acc": float(trial.value),
                "full_val_acc": full_acc,
                "params": dict(trial.params),
                "max_epochs": args.max_epochs,
                "train_subset": args.train_subset,
                "val_subset": args.val_subset,
                "patience": args.patience,
            }
            cache_file.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            print(f"  rank {rank}: full_val_acc={full_acc:.4f} -> {cache_file}")
        results.append(entry)

    winner = max(results, key=lambda e: e["full_val_acc"])

    priors_path = REPO_ROOT / "artifacts" / "priors" / f"{model}.json"
    priors_hash = hashlib.sha256(priors_path.read_bytes()).hexdigest()

    final = {
        "model": model,
        "dataset": dataset,
        "study_name": study_name,
        "best_value": winner["full_val_acc"],
        "best_params": winner["params"],
        "n_trials_completed": len(completed),
        "priors_file_hash": priors_hash,
        "phase": "B",
        "level": 3,
        "validated_at_full_convergence": True,
        "validation_top_k": args.top_k,
        "validation_max_epochs": args.max_epochs,
        "validation_train_subset": args.train_subset,
        "validation_val_subset": args.val_subset,
        "validation_patience": args.patience,
        "validation_winner_trial": winner["trial_number"],
        "validation_results": [
            {"rank": r["rank"], "trial_number": r["trial_number"],
             "fast_val_acc": r["fast_val_acc"], "full_val_acc": r["full_val_acc"]}
            for r in results
        ],
    }
    out_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"OK {model}_{dataset}: winner=trial#{winner['trial_number']} "
          f"full_val_acc={winner['full_val_acc']:.4f} -> {out_path}")


def _train_at_full_convergence(model: str, dataset: str, params: dict, args) -> float:
    """Train one (model, dataset, hparams) at production protocol. Returns val_acc."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("WANDB_MODE", "offline")
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping

    from src.data.degradation_levels import level_params
    from src.lightning.datamodule import THzDataModule
    from src.lightning.module import THzClassifier

    p = level_params(3, axis=None)
    pl.seed_everything(42, workers=True)

    # US-043: align Stage 1.5 validation with the 5070 production protocol that
    # `src/lightning/train.py` already enforces — bf16-mixed on Blackwell, TF32
    # on the FP32 matmul path, and a non-zero num_workers with the datamodule's
    # pin_memory + persistent_workers + prefetch_factor knobs.
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cpu_count = os.cpu_count() or 4
        num_workers = min(max(cpu_count // 2, 2), 8)
    else:
        num_workers = 0

    classifier = THzClassifier(
        model_name=model,
        num_classes=10,
        pretrained=True,
        lr=params["head_lr"],
        backbone_lr=params["backbone_lr"],
        weight_decay=params["weight_decay"],
        label_smoothing=params["label_smoothing"],
        warmup_epochs=int(round(params["warmup_epochs"])),
        scheduler_type="cosine",
        max_grad_norm=1.0,
        epochs=args.max_epochs,
        freeze_backbone=False,
    )
    dm = THzDataModule(
        dataset=dataset,
        out_size=224,
        low_res=int(p["low_res"]),
        batch_size=32,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        degradation_type="all",
        blur_kernel=int(p["blur_kernel"]),
        blur_sigma=float(p["blur_sigma"]),
        gaussian_noise_std=float(p["noise_std"]),
        salt_pepper_amount=float(p["salt_pepper"]),
        saturation=float(p["saturation"]),
        num_workers=num_workers,
    )

    accelerator = "gpu" if cuda_available else "cpu"
    precision = "bf16-mixed" if cuda_available else "32-true"
    if cuda_available:
        torch.set_float32_matmul_precision("high")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=[
            EarlyStopping(monitor="val_acc", mode="max",
                          patience=args.patience, min_delta=1e-4),
        ],
    )
    trainer.fit(classifier, datamodule=dm)
    val_acc = trainer.callback_metrics.get("val_acc", None)
    return float(val_acc) if val_acc is not None else 0.0


if __name__ == "__main__":
    sys.exit(main())
