#!/usr/bin/env python3
"""
Systematic Degradation Experiments
====================================

For each degradation level combination, run all 3 models with
optimized hyperparameters and overfitting prevention.

Degradation Levels (combined pipeline):
  Level 1 (Mild):     low_res=16, blur_kernel=3, blur_sigma=0.5, noise_std=0.04, salt_pepper=0.02
  Level 2 (Moderate):  low_res=16, blur_kernel=5, blur_sigma=1.0, noise_std=0.08, salt_pepper=0.05
  Level 3 (Severe):    low_res=8,  blur_kernel=7, blur_sigma=1.5, noise_std=0.12, salt_pepper=0.08

Overfitting Prevention:
  - Early stopping (patience=5 epochs)
  - Cosine LR scheduler
  - Weight decay 1e-4
  - Gradient clipping (max_norm=1.0)
  - Label smoothing 0.1 (ResNet50/DenseNet121)
  - TransNeXt: frozen backbone (linear probe) to prevent overfitting

Models:
  1. ResNet50       - differential LR fine-tuning
  2. DenseNet121    - differential LR fine-tuning
  3. TransNeXt Micro - linear probe (frozen backbone)

Usage:
  python run_systematic.py --level 2 --mode pilot     # quick test, level 2
  python run_systematic.py --level all --mode full     # all levels, full training
  python run_systematic.py --level 1 --mode full       # level 1 only, full training
"""

import os
import sys
import csv
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

# ============================================================
# DEGRADATION LEVELS
# ============================================================

DEGRADATION_LEVELS = {
    1: {
        "name": "mild",
        "low_res": 16,
        "blur_kernel": 3,
        "blur_sigma": 0.5,
        "gaussian_noise_std": 0.04,
        "salt_pepper_amount": 0.02,
        "p_grayscale": 0.3,
    },
    2: {
        "name": "moderate",
        "low_res": 16,
        "blur_kernel": 5,
        "blur_sigma": 1.0,
        "gaussian_noise_std": 0.08,
        "salt_pepper_amount": 0.05,
        "p_grayscale": 0.3,
    },
    3: {
        "name": "severe",
        "low_res": 8,
        "blur_kernel": 7,
        "blur_sigma": 1.5,
        "gaussian_noise_std": 0.12,
        "salt_pepper_amount": 0.08,
        "p_grayscale": 0.3,
    },
}

# ============================================================
# MODEL CONFIGURATIONS (with overfitting prevention)
# ============================================================

MODEL_CONFIGS = [
    # ResNet50 — TResNet/EfficientNetV2 paper recommendations
    {
        "model_name": "resnet50",
        "lr": 1e-3,
        "backbone_lr": 5e-5,          # was 1e-4: slower adaptation, less forgetting
        "freeze_backbone": False,
        "weight_decay": 5e-4,          # was 1e-4: stronger L2 regularization
        "label_smoothing": 0.15,       # was 0.1: moderate increase
        "warmup_epochs": 3,            # EfficientNetV2 warmup recommendation
    },
    # DenseNet121 — DenseNet paper recommendations
    {
        "model_name": "densenet121",
        "lr": 1e-3,
        "backbone_lr": 5e-5,          # was 1e-4: more conservative
        "freeze_backbone": False,
        "weight_decay": 5e-4,          # was 1e-4: stronger regularization
        "label_smoothing": 0.2,        # was 0.1: strong smoothing for 99%+ train acc
        "warmup_epochs": 2,            # shorter warmup (DenseNet converges fast)
    },
    # TransNeXt Micro — TransNeXt paper recommendations
    {
        "model_name": "transnext_micro",
        "lr": 1e-3,
        "backbone_lr": 1e-5,           # was None/frozen: very conservative unfreeze
        "freeze_backbone": False,      # was True: allow backbone adaptation
        "weight_decay": 0.05,          # TransNeXt paper fine-tuning value
        "label_smoothing": 0.1,        # was 0.0: light smoothing for fine-tuning
        "warmup_epochs": 5,            # TransNeXt paper training protocol
        "drop_path_rate": 0.1,         # TransNeXt paper stochastic depth
    },
]

# ============================================================
# TRAINING MODES
# ============================================================

PILOT_SETTINGS = {
    "epochs": 5,
    "train_subset": 2000,
    "val_subset": 1000,
}

FULL_SETTINGS = {
    "epochs": 30,
    "train_subset": 10000,
    "val_subset": 5000,
}

COMMON = {
    "pretrained": True,
    "out_size": 224,
    "batch_size": 32,
    "degradation_type": "all",
    "scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "group": "systematic",
}


# ============================================================
# 186-CELL CAMPAIGN: single-cell dispatch (US-008)
# ============================================================

# Mode -> training settings for the final campaign. `pilot` is for ad-hoc
# smoke tests only; run_all_phases.py rejects it for --plan final.
FINAL_PILOT = {
    "epochs": 5,
    "train_subset": 2000,
    "val_subset": 1000,
    "early_stopping_patience": 2,
}
FINAL_FULL = {
    "epochs": 60,                    # CLAUDE.md: max_epochs 60
    "train_subset": 10000,
    "val_subset": 5000,
    "early_stopping_patience": 10,   # CLAUDE.md: patience 10, min_delta 1e-4
}


def _load_best_hparams(model: str, dataset: str) -> dict:
    """Load Optuna winner JSON for a (model, dataset) pair."""
    path = Path("artifacts/best_hparams") / f"{model}_{dataset}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing best_hparams: {path}\n"
            f"  Run `python tune_all.py --n-trials 20 --model {model} --dataset {dataset}` "
            f"first, or `python tune_all.py --n-trials 20` for all 6 pairs."
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    if "best_params" not in blob:
        raise ValueError(
            f"{path}: missing 'best_params' key — file may be from an interrupted study"
        )
    return blob


# Phase A frozen hyperparameters from CLAUDE.md "Training Hyperparameters" table.
# Used as a fallback when artifacts/best_hparams/{model}_{dataset}.json is missing
# AND the cell is Phase A (clean baseline). For Phase B/C CNN cells, missing
# hparams remains a hard error — Optuna tuning is mandatory there.
#
# V3 ratification (2026-05-12): TransNeXt entries were rewritten from LP
# (backbone frozen, label_smoothing=0) to full-FT priors. TransNeXt cells use
# V3_TRANSNEXT_FT_PRIORS below for ALL phases when no Optuna JSON exists.
PHASE_A_FROZEN_HPARAMS: dict[str, dict] = {
    "resnet50": {
        "head_lr": 1e-3,
        "backbone_lr": 1e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "warmup_epochs": 3,
    },
    "densenet121": {
        "head_lr": 1e-3,
        "backbone_lr": 1e-4,
        "weight_decay": 1e-4,
        "label_smoothing": 0.1,
        "warmup_epochs": 2,
    },
}


# V3 TransNeXt full-FT priors (ratified by MASTER 2026-05-12).
#
# Paper-anchored: TransNeXt §A.3 fine-tune table + the CNN convention of
# 10x differential backbone vs head LR. Applied as a hardcoded fallback when
# artifacts/best_hparams/{transnext_*}_{dataset}.json is absent — covers both
# the "tune-free start" the V3 plan calls for and any future TransNeXt sizes
# we add (priors are size-agnostic for the small/base/micro variants).
V3_TRANSNEXT_FT_PRIORS: dict[str, float | int] = {
    "head_lr": 5e-4,
    "backbone_lr": 5e-5,
    "weight_decay": 5e-2,
    "label_smoothing": 0.1,
    "warmup_epochs": 5,
    "drop_path_rate": 0.1,
}


# Phase D — Regularization Sweep treatment deltas (US-026, 2026-05-23).
#
# Each treatment is a *delta* on top of the L3-Optuna-tuned baseline hparams
# loaded from artifacts/best_hparams/. Phase D does NOT re-tune; treatments
# inject regularization on top of frozen baselines so deltas are attributable.
#
# CNN dropout flows through timm's `drop_rate` (classifier-head); TransNeXt
# does not honor it, so we route to `drop_path_rate` (stochastic depth) for
# architectural regularization on TransNeXt. Mixup/cutmix are model-agnostic.
def _phase_d_treatment_deltas(treatment: str, model_name: str) -> dict[str, float]:
    """Return the regularization-delta dict to merge into best_params for a
    Phase D (treatment, model) pair.

    T1 — architectural dropout (dropout=0.2 for CNN, drop_path_rate=0.2 for TransNeXt)
    T2 — label-mixing (mixup_alpha=0.2, cutmix off)
    T3 — combo (T1 + T2 + cutmix_alpha=1.0)
    """
    is_transnext = model_name.startswith("transnext_")
    if treatment == "T1":
        return {"drop_path_rate": 0.2} if is_transnext else {"dropout": 0.2}
    if treatment == "T2":
        return {"mixup_alpha": 0.2, "cutmix_alpha": 0.0}
    if treatment == "T3":
        arch = {"drop_path_rate": 0.2} if is_transnext else {"dropout": 0.2}
        return {**arch, "mixup_alpha": 0.2, "cutmix_alpha": 1.0}
    raise ValueError(
        f"unknown Phase D treatment: {treatment!r}; "
        f"expected one of T1, T2, T3"
    )


def _apply_phase_d_treatment(spec, hparams: dict) -> dict:
    """Return a new hparams blob with Phase D treatment deltas layered on top
    of best_params. For non-Phase-D cells the input is returned unchanged.

    The blob is shallow-copied so the caller's reference (and any cached
    Optuna JSON) is not mutated.
    """
    treatment = getattr(spec, "treatment", None)
    if spec.phase != "D" or not treatment:
        return hparams
    deltas = _phase_d_treatment_deltas(treatment, spec.model)
    bp = dict(hparams.get("best_params", {}))
    bp.update(deltas)
    out = dict(hparams)
    out["best_params"] = bp
    out["phase_d_treatment"] = treatment
    out["phase_d_deltas"] = deltas
    return out


def _apply_b2_or_c2_treatment(spec, hparams: dict) -> dict:
    """Return a new hparams blob with T3 deltas applied for Phase B2 / C2.

    US-038 (v4, 2026-05-26). Phase B2 and Phase C2 are T3 carriers — the
    dispatcher reuses `_phase_d_treatment_deltas("T3", model)` byte-for-byte.
    Phase B2nr is a no-treatment carrier: hparams are returned unchanged
    AND the `phase_b2nr_deltas: {}` marker is stamped explicitly for schema
    stability.

    No-op for Phase A / B / C / D cells (D is routed via
    `_apply_phase_d_treatment`).
    """
    phase = getattr(spec, "phase", None)
    if phase not in ("B2", "B2nr", "C2"):
        return hparams
    out = dict(hparams)
    if phase == "B2nr":
        out["phase_b2nr_deltas"] = {}
        return out

    treatment = getattr(spec, "treatment", None) or "T3"
    deltas = _phase_d_treatment_deltas(treatment, spec.model)
    bp = dict(hparams.get("best_params", {}))
    bp.update(deltas)
    out["best_params"] = bp
    if phase == "B2":
        out["phase_b2_treatment"] = treatment
        out["phase_b2_deltas"] = deltas
    else:  # C2
        out["phase_c2_treatment"] = treatment
        out["phase_c2_deltas"] = deltas
        out["phase_c2_axis"] = getattr(spec, "axis", None)
    return out


def _load_hparams_for_cell(spec) -> dict:
    """Load best_hparams for a CellSpec, with Phase A / V3 TransNeXt fallback.

    Resolution order:
      1. ``artifacts/best_hparams/{model}_{dataset}.json`` if present (Optuna winner).
      2. If absent AND model is TransNeXt: return V3_TRANSNEXT_FT_PRIORS
         (paper-anchored full-FT priors — V3 ratified tune-free start).
      3. If absent AND ``spec.phase == 'A'`` AND CNN model has frozen defaults:
         return a synthetic blob carrying the CLAUDE.md frozen hparams.
      4. Otherwise (Phase B/C CNN, no Optuna JSON): re-raise FileNotFoundError.
    """
    path = Path("artifacts/best_hparams") / f"{spec.model}_{spec.dataset}.json"
    if path.exists():
        return _load_best_hparams(spec.model, spec.dataset)
    if spec.model.startswith("transnext_"):
        return {
            "model": spec.model,
            "dataset": spec.dataset,
            "study_name": f"v3_transnext_ft_priors_{spec.model}_{spec.dataset}",
            "best_value": None,
            "best_params": dict(V3_TRANSNEXT_FT_PRIORS),
            "n_trials_completed": 0,
            "priors_file_hash": None,
            "phase": spec.phase,
            "level": spec.level,
            "source": "v3_transnext_ft_priors",
        }
    if spec.phase == "A" and spec.model in PHASE_A_FROZEN_HPARAMS:
        return {
            "model": spec.model,
            "dataset": spec.dataset,
            "study_name": f"phase_a_frozen_{spec.model}_{spec.dataset}",
            "best_value": None,
            "best_params": dict(PHASE_A_FROZEN_HPARAMS[spec.model]),
            "n_trials_completed": 0,
            "priors_file_hash": None,
            "phase": "A",
            "level": None,
            "source": "claude_md_frozen",
        }
    return _load_best_hparams(spec.model, spec.dataset)


def _cell_config(
    spec,
    hparams: dict,
    mode: str,
    seed: int = 42,
    effective_tag: Optional[str] = None,
    log_logits: bool = False,
) -> dict:
    """Build run_experiment kwargs from a CellSpec + best_hparams.

    The DegradeConfig fields override the legacy CLI flag names because
    that's what THzDataModule consumes. `saturation` is included now that
    train.py forwards it to THzDataModule (US-008 prerequisite).

    Resolution / precision / compile fields (out_size, img_size, patch_size,
    pretrain_size, compile_mode, precision) are read from CellSpec — they are
    fixed at 224x224 / bf16-mixed / no-compile but still flow through the
    CellSpec so future overrides have a single source of truth.

    US-038 (v4): `seed` overrides the default `pl.seed_everything(42)` for
    multi-seed audit runs. `effective_tag` is the on-disk tag (may include a
    `_seed{N}` suffix); when omitted it defaults to `spec.tag`. `log_logits`
    threads down to the trainer so US-045 diagnostics can dump per-batch
    val/test logits to `runs/final/<tag>/logits/`.
    """
    settings = FINAL_PILOT if mode == "pilot" else FINAL_FULL
    deg = spec.degrade_config
    bp = hparams["best_params"]
    tag_on_disk = effective_tag or spec.tag

    return {
        # Model + training
        "model_name": spec.model,
        "pretrained": True,
        "out_size": spec.out_size,
        "batch_size": 32,
        "freeze_backbone": False,        # all 3 models full FT in final campaign
        "scheduler_type": "cosine",
        "max_grad_norm": 1.0,
        "lr": float(bp["head_lr"]),
        "backbone_lr": float(bp["backbone_lr"]),
        "weight_decay": float(bp["weight_decay"]),
        "label_smoothing": float(bp["label_smoothing"]),
        "warmup_epochs": int(round(float(bp["warmup_epochs"]))),
        "drop_path_rate": float(bp.get("drop_path_rate", 0.0)),
        "dropout": float(bp.get("dropout", 0.0)),
        # Phase D (US-026): mixup / cutmix flow through bp when a treatment
        # injects them. Default 0.0 preserves Phase A/B/C behavior exactly.
        "mixup_alpha": float(bp.get("mixup_alpha", 0.0)),
        "cutmix_alpha": float(bp.get("cutmix_alpha", 0.0)),

        # Blackwell + TransNeXt knobs (CellSpec is SoT; all 224x224 now)
        "img_size": spec.img_size,
        "patch_size": spec.patch_size,
        "pretrain_size": spec.pretrain_size,
        "compile_mode": spec.compile_mode,
        "precision": spec.precision,

        # Degradation (from CellSpec.degrade_config)
        "low_res": deg.low_res,
        "blur_kernel": deg.blur_kernel,
        "blur_sigma": deg.blur_sigma,
        "gaussian_noise_std": deg.gaussian_noise_std,
        "salt_pepper_amount": deg.salt_pepper_amount,
        "saturation": deg.saturation,
        "degradation_type": deg.degradation_type,

        # Bookkeeping
        "tag": tag_on_disk,
        "dataset": spec.dataset,
        "group": "final",
        "run_name_override": tag_on_disk,    # forces runs/final/<tag_on_disk>/

        # Phase-aware training schedule
        "epochs": settings["epochs"],
        "train_subset": settings["train_subset"],
        "val_subset": settings["val_subset"],
        "early_stopping_patience": settings["early_stopping_patience"],

        # US-038 (v4): multi-seed audit + logit-logging
        "seed": int(seed),
        "log_logits": bool(log_logits),
    }


def _merge_metadata_into_metrics_json(
    run_dir: Path, spec, hparams: dict, seed: int = 42,
) -> None:
    """Extend metrics.json with hparams + cell metadata for round-trip reproducibility.

    Acceptance: "All hparam values logged to metrics.json under a `hparams` key."

    US-038 (v4): adds `seed` (first-class field for multi-seed audits) and
    Phase B2 / B2nr / C2 treatment + axis fields.
    """
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        return  # training crashed before LegacyJSONMetricsCallback fired
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["hparams"] = hparams.get("best_params", {})
    metrics["hparams_source"] = {
        "study_name": hparams.get("study_name"),
        "best_value": hparams.get("best_value"),
        "n_trials_completed": hparams.get("n_trials_completed"),
        "priors_file_hash": hparams.get("priors_file_hash"),
    }
    metrics["cell_tag"] = spec.tag
    metrics["phase"] = spec.phase
    metrics["level"] = spec.level
    metrics["axis"] = spec.axis
    metrics["model"] = spec.model
    metrics["dataset"] = spec.dataset
    metrics["seed"] = int(seed)
    # Treatment field (PRD §4.7): Phase A/B/B2nr/C carry None; B2/C2 carry "T3";
    # Phase D carries one of T1/T2/T3.
    metrics["treatment"] = getattr(spec, "treatment", None)
    # US-040 (v4): stamp the resolved DegradeConfig values as a top-level block
    # so downstream auditors can confirm the B2 / B2nr / C2 override survived
    # the dispatcher (matches PRD US-040 AC #2: "B2 DegradeConfig override").
    deg = getattr(spec, "degrade_config", None)
    if deg is not None:
        metrics["degrade_config"] = {
            "low_res": int(deg.low_res),
            "out_size": int(deg.out_size),
            "blur_kernel": int(deg.blur_kernel),
            "blur_sigma": float(deg.blur_sigma),
            "gaussian_noise_std": float(deg.gaussian_noise_std),
            "salt_pepper_amount": float(deg.salt_pepper_amount),
            "saturation": float(deg.saturation),
            "degradation_type": str(deg.degradation_type),
        }
    # Phase D (US-026/US-028): persist treatment + deltas as top-level keys
    # so PRD US-028 acceptance criteria are literally satisfied (the merged
    # delta values already flow through metrics["hparams"], but the explicit
    # treatment label is what the post-hoc attribution scripts read).
    if hparams.get("phase_d_treatment") is not None:
        metrics["phase_d_treatment"] = hparams["phase_d_treatment"]
        metrics["phase_d_deltas"] = hparams.get("phase_d_deltas", {})
    # Phase B2 / B2nr / C2 (US-038): mirror the Phase D convention.
    if hparams.get("phase_b2_treatment") is not None:
        metrics["phase_b2_treatment"] = hparams["phase_b2_treatment"]
        metrics["phase_b2_deltas"] = hparams.get("phase_b2_deltas", {})
    if "phase_b2nr_deltas" in hparams:
        metrics["phase_b2nr_deltas"] = hparams["phase_b2nr_deltas"]
    if hparams.get("phase_c2_treatment") is not None:
        metrics["phase_c2_treatment"] = hparams["phase_c2_treatment"]
        metrics["phase_c2_deltas"] = hparams.get("phase_c2_deltas", {})
        metrics["phase_c2_axis"] = hparams.get("phase_c2_axis")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _measure_image_quality_for_cell(spec, run_dir: Path, n_samples: int = 256) -> Optional[str]:
    """US-016: write `runs/final/<tag>/image_quality.json` after a cell trains.

    Skipped for Phase A (clean) — PSNR/SSIM against an identity pipeline is
    degenerate (inf / 1.0). Idempotent: returns early if the file already
    exists. Soft-fail: returns the error string instead of raising so the
    runner can continue to the next cell.
    """
    if spec.phase == "A":
        return None
    out = run_dir / "image_quality.json"
    if out.exists():
        return None
    try:
        from src.tools.measure_image_quality import measure  # lazy
        result = measure(
            dataset=spec.dataset,
            deg_cfg=spec.degrade_config,
            n_samples=n_samples,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return None
    except Exception as e:  # noqa: BLE001 — best-effort post-train side-channel
        return f"{type(e).__name__}: {e}"


_MULTISEED_SUFFIX_RE = __import__("re").compile(r"_seed(\d+)$")


def _split_multiseed_tag(cell_tag: str) -> tuple[str, int]:
    """Return (base_tag, seed). If `cell_tag` carries `_seed{N}` (N != 42),
    strip it and return (base, N). Otherwise return (cell_tag, 42).

    US-038 (v4). Multi-seed dispatches use the suffix to route to a unique
    `runs/final/<base_tag>_seed{N}/` directory while still resolving the
    DegradeConfig + hparams against the canonical base tag in
    `build_final_matrix()`.
    """
    m = _MULTISEED_SUFFIX_RE.search(cell_tag)
    if m is None:
        return cell_tag, 42
    seed = int(m.group(1))
    base = cell_tag[: m.start()]
    return base, seed


def run_cell(
    cell_tag: str,
    mode: str = "full",
    engine: str = "lightning",
    seed: int = 42,
    *,
    run_experiment_fn=None,         # injectable for testing
    measure_quality_fn=None,        # injectable for testing
    log_logits: bool = False,
):
    """Dispatch a single cell from the campaign matrix.

    Resolves the tag against build_final_matrix(), loads the (model, dataset)
    Optuna winner, calls run_experiment with run_name_override=tag so output
    lands at runs/final/<tag>/, then:
      1. Merges hparams + cell metadata into metrics.json (US-008).
      2. Writes image_quality.json with PSNR/SSIM (US-016) for Phase B/C cells.

    Multi-seed (US-038, v4): if `cell_tag` ends in `_seed{N}` (N != 42) OR
    the `seed` kwarg overrides 42, the run directory uses the suffixed tag
    and `pl.seed_everything(N, workers=True)` is invoked in the trainer.
    The DegradeConfig + hparams are resolved against the BASE tag (without
    the `_seed{N}` suffix), so multi-seed runs share the canonical cell's
    DegradeConfig byte-for-byte.

    `log_logits` (US-038, v4): when True, the trainer dumps per-batch
    val/test logits to `runs/final/<tag>/logits/` for downstream diagnostics
    (US-045). Defaults to False to preserve the legacy behavior.

    The PSNR/SSIM step is best-effort — failure is logged to stderr and the
    return value is unaffected. `measure_quality_fn` is injectable so unit
    tests can avoid loading torchvision.
    """
    from src.experiments.matrix import cells_by_tag

    by_tag = cells_by_tag()

    # Resolve base tag (strip _seed{N} suffix) for matrix lookup.
    base_tag, parsed_seed = _split_multiseed_tag(cell_tag)
    if seed != 42 and parsed_seed == 42:
        # Caller passed --seed but used the bare base_tag; promote to suffixed form.
        parsed_seed = seed
        effective_tag = f"{base_tag}_seed{seed}" if seed != 42 else base_tag
    else:
        effective_tag = cell_tag if parsed_seed != 42 else base_tag
        seed = parsed_seed

    if base_tag not in by_tag:
        raise ValueError(
            f"unknown cell tag: {cell_tag!r} (base: {base_tag!r}). "
            f"Expected one of the canonical tags from build_final_matrix()."
        )
    spec = by_tag[base_tag]
    hparams = _load_hparams_for_cell(spec)
    # §6.4 retry plumbing (Iteration 12, 2026-05-15): if the ralph driver
    # wrote a retry_config.json for this cell, it overrides best_hparams so
    # the deltas (lr_backbone÷2, weight_decay×2, dropout+=0.1 for overfitting;
    # head_lr÷3, weight_decay×1.5, label_smoothing+=0.05 for failed_convergence)
    # actually reach the trainer. Without this hook the retry trained with the
    # original Optuna winner hparams and produced identical results.
    retry_path = Path("runs/final") / effective_tag / "retry_config.json"
    if retry_path.exists():
        hparams = json.loads(retry_path.read_text(encoding="utf-8"))
    # Phase D (US-026): layer treatment-specific regularization deltas onto
    # the loaded best_params. No-op for Phase A/B/B2/B2nr/C/C2 cells.
    hparams = _apply_phase_d_treatment(spec, hparams)
    # Phase B2 / B2nr / C2 (US-038): same delta-merging pattern.
    hparams = _apply_b2_or_c2_treatment(spec, hparams)
    config = _cell_config(spec, hparams, mode, seed=seed,
                          effective_tag=effective_tag, log_logits=log_logits)

    run_experiment = run_experiment_fn or _resolve_run_experiment(engine)
    run_dir = Path(run_experiment(**config))
    _merge_metadata_into_metrics_json(run_dir, spec, hparams, seed=seed)

    quality_fn = measure_quality_fn or _measure_image_quality_for_cell
    quality_err = quality_fn(spec, run_dir)
    if quality_err:
        print(f"[run_cell][WARN] image_quality.json for {effective_tag}: {quality_err}",
              file=sys.stderr)

    return run_dir


def run_systematic(levels: list[int], mode: str = "pilot", dataset: str = "cifar10",
                   engine: str = "lightning"):
    """Run all 3 models for each specified degradation level."""
    settings = PILOT_SETTINGS if mode == "pilot" else FULL_SETTINGS
    ds_label = "CIFAR-10" if dataset == "cifar10" else "MNIST"
    run_experiment = _resolve_run_experiment(engine)

    total_experiments = len(levels) * len(MODEL_CONFIGS)

    print("\n" + "=" * 70)
    print(f"  SYSTEMATIC DEGRADATION EXPERIMENTS ({mode.upper()} MODE)")
    print("=" * 70)
    print(f"  Engine: {engine}")
    print(f"  Dataset: {ds_label}")
    print(f"  Degradation levels: {levels}")
    print(f"  Models: ResNet50, DenseNet121, TransNeXt Micro")
    print(f"  Epochs: {settings['epochs']}")
    print(f"  Total experiments: {total_experiments}")
    print(f"  Overfitting prevention: early stopping, cosine LR, weight decay,")
    print(f"                          gradient clipping, label smoothing")
    print("=" * 70)

    results = []
    exp_num = 0

    for level_id in levels:
        level = DEGRADATION_LEVELS[level_id]
        level_name = level["name"]

        print(f"\n{'━' * 70}")
        print(f"  DEGRADATION LEVEL {level_id} ({level_name.upper()})")
        print(f"  low_res={level['low_res']}, blur_kernel={level['blur_kernel']}, "
              f"blur_sigma={level['blur_sigma']}")
        print(f"  noise_std={level['gaussian_noise_std']}, "
              f"salt_pepper={level['salt_pepper_amount']}, "
              f"grayscale_p={level['p_grayscale']}")
        print(f"{'━' * 70}")

        for model_cfg in MODEL_CONFIGS:
            exp_num += 1
            model = model_cfg["model_name"]
            ds_suffix = f"_{dataset}" if dataset != "cifar10" else ""
            tag = f"sys_L{level_id}_{level_name}_{model}{ds_suffix}"

            print(f"\n{'─' * 70}")
            print(f"  [{exp_num}/{total_experiments}] {model} @ Level {level_id} ({level_name})")
            print(f"  LR={model_cfg['lr']}, backbone_lr={model_cfg.get('backbone_lr')}")
            print(f"  freeze={model_cfg['freeze_backbone']}, wd={model_cfg['weight_decay']}")
            print(f"{'─' * 70}\n")

            config = {
                **COMMON,
                **settings,
                **model_cfg,
                "low_res": level["low_res"],
                "blur_kernel": level["blur_kernel"],
                "blur_sigma": level["blur_sigma"],
                "gaussian_noise_std": level["gaussian_noise_std"],
                "salt_pepper_amount": level["salt_pepper_amount"],
                "p_grayscale": level["p_grayscale"],
                "early_stopping_patience": 5,
                "tag": tag,
                "dataset": dataset,
            }

            t0 = time.time()
            try:
                run_experiment(**config)
                dt = time.time() - t0
                results.append({
                    "level": level_id,
                    "level_name": level_name,
                    "model": model,
                    "tag": tag,
                    "status": "OK",
                    "time": f"{dt:.1f}s",
                })
            except Exception as e:
                dt = time.time() - t0
                print(f"[ERROR] {model} @ Level {level_id}: {e}")
                results.append({
                    "level": level_id,
                    "level_name": level_name,
                    "model": model,
                    "tag": tag,
                    "status": f"FAILED: {e}",
                    "time": f"{dt:.1f}s",
                })

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  SYSTEMATIC EXPERIMENTS - RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Level':<10} {'Model':<20} {'Status':<10} {'Time':<10}")
    print(f"  {'─'*10} {'─'*20} {'─'*10} {'─'*10}")
    for r in results:
        icon = "✓" if r["status"] == "OK" else "✗"
        print(f"  {icon} L{r['level']} ({r['level_name']:<8}) {r['model']:<20} {r['status']:<10} {r['time']}")
    print("=" * 70)

    # Save summary JSON for dashboard
    summary_path = Path("artifacts") / "systematic_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Summary saved to {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Systematic degradation experiments")
    p.add_argument("--level", default="all",
                   help="Degradation level: 1, 2, 3, or 'all'")
    p.add_argument("--mode", default="pilot", choices=["pilot", "full"],
                   help="pilot (5 epochs, small data) or full (30 epochs)")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"],
                   help="Dataset to use: cifar10 or mnist")
    p.add_argument("--engine", default="lightning", choices=["lightning", "legacy"],
                   help="Training engine: lightning (default) or legacy hand-rolled trainer")
    p.add_argument(
        "--transnext_size",
        default="small",
        choices=["micro", "tiny", "small", "base"],
        help="TransNeXt size variant. Default 'small' (CLAUDE.md baseline); "
             "use 'base' for the final 186-cell campaign per US-006.",
    )
    p.add_argument(
        "--transnext_mode",
        default="ft",
        choices=["lp", "ft"],
        help="TransNeXt training mode: 'lp' (linear probe, frozen backbone) "
             "or 'ft' (full fine-tuning). Default 'ft' for the final campaign.",
    )
    p.add_argument(
        "--cell-tag",
        default=None,
        help="Run a single cell from the campaign matrix (US-008). Tag form: "
             "final_clean_{m}_{d} | final_B_L{l}_{m}_{d} | final_C_L{l}_{ax}_{m}_{d} | "
             "final_B2_L{l}_{m}_{d} | final_B2nr_L3_{m}_{d} | "
             "final_C2_L{l}_{ax}_{m}_{d} | final_D_{T}_L{l}_{m}_{d}. "
             "Loads hparams from artifacts/best_hparams/{m}_{d}.json. When set, "
             "the legacy --level loop is bypassed.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Lightning seed (US-038, v4). Default 42 — the canonical "
             "campaign seed. Non-default values append a `_seed{N}` suffix "
             "to the cell tag (`runs/final/<tag>_seed{N}/`) for multi-seed "
             "audit dispatches; per-sample degradation seeds (SEED_OFFSET_*) "
             "are independent of this knob.",
    )
    p.add_argument(
        "--log-logits", action="store_true", default=False,
        help="US-038 (v4) — dump per-batch val/test logits under "
             "runs/final/<tag>/logits/ for downstream diagnostics (US-045).",
    )
    args = p.parse_args()

    # Re-target the TransNeXt entry of MODEL_CONFIGS based on CLI flags.
    for _cfg in MODEL_CONFIGS:
        if _cfg["model_name"].startswith("transnext_"):
            _cfg["model_name"] = f"transnext_{args.transnext_size}"
            _cfg["freeze_backbone"] = (args.transnext_mode == "lp")
            break

    # 186-cell single-cell dispatch (US-008): --cell-tag bypasses the legacy loop.
    if args.cell_tag:
        run_cell(
            args.cell_tag,
            mode=args.mode,
            engine=args.engine,
            seed=args.seed,
            log_logits=args.log_logits,
        )
        sys.exit(0)

    if args.level == "all":
        levels = [1, 2, 3]
    else:
        levels = [int(x) for x in args.level.split(",")]
        for lv in levels:
            if lv not in DEGRADATION_LEVELS:
                print(f"[ERROR] Invalid level {lv}. Choose from 1, 2, 3")
                sys.exit(1)

    run_systematic(levels, args.mode, args.dataset, args.engine)
