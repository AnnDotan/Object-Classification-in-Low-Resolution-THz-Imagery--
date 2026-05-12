"""Optuna pre-tuning entry point — paper-anchored priors per (model, dataset).

This file is the canonical CLI for the THz hyperparameter sweep referenced
in CLAUDE.md and PRD US-004 / US-005:

    python tune_all.py --validate-only          # sanity-check priors files
    python tune_all.py --n-trials 20            # tune all 6 (model, dataset) pairs
    python tune_all.py --n-trials 20 --model resnet50 --dataset cifar10

Priors live at `artifacts/priors/{model}.json`. The schema is documented in
`artifacts/priors/_schema.json`; this module validates priors in-process so
the project does not need the jsonschema dependency at runtime.

Optuna study + runner code (US-005) is loaded lazily on demand to keep
`--validate-only` fast and dependency-light.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
PRIORS_DIR = REPO_ROOT / "artifacts" / "priors"
SCHEMA_PATH = PRIORS_DIR / "_schema.json"

SUPPORTED_MODELS = (
    "resnet50",
    "densenet121",
    "transnext_base",
    # PRD US-042 native-resolution aliases (V3-routed). Priors files exist
    # only when paper-derived priors warrant a separate JSON; we ship
    # `transnext_small_native.json` per the US-042 acceptance criterion.
    "transnext_small_native",
)
SUPPORTED_DATASETS = ("cifar10", "mnist")

REQUIRED_HPARAMS = (
    "head_lr",
    "backbone_lr",
    "weight_decay",
    "label_smoothing",
    "warmup_epochs",
)
SUPPORTED_DISTRIBUTIONS = ("uniform", "loguniform", "categorical")

# Project convention (CLAUDE.md "narrowed to paper-derived priors ± 1 decade
# max", PRD US-015): a `loguniform` band may extend at most one decade above
# AND one decade below the paper anchor — total span up to 2 decades, i.e.
# `high / low ≤ MAX_LOGUNIFORM_RATIO`. Any wider band is "blind exploration"
# and the validator rejects it.
MAX_LOGUNIFORM_RATIO: float = 100.0


class PriorsValidationError(ValueError):
    """Raised when a priors JSON fails validation. The message names the
    offending field path so callers can fix it without grepping."""


def _validate_hparam(name: str, hp: Any) -> list[str]:
    """Return a list of error strings for a single hparam entry. Empty = OK."""
    errors: list[str] = []
    path = f"hparams.{name}"

    if not isinstance(hp, dict):
        return [f"{path}: expected object, got {type(hp).__name__}"]

    dist = hp.get("distribution")
    if dist not in SUPPORTED_DISTRIBUTIONS:
        errors.append(
            f"{path}.distribution: expected one of {SUPPORTED_DISTRIBUTIONS}, got {dist!r}"
        )
        return errors  # downstream checks assume a valid distribution

    citation = hp.get("citation")
    if not isinstance(citation, str) or not citation.strip():
        errors.append(f"{path}.citation: must be a non-empty string")

    if dist in ("uniform", "loguniform"):
        for key in ("low", "high"):
            if key not in hp:
                errors.append(f"{path}.{key}: required for {dist} distribution")
                continue
            if not isinstance(hp[key], (int, float)) or isinstance(hp[key], bool):
                errors.append(f"{path}.{key}: must be a number, got {hp[key]!r}")
        if "low" in hp and "high" in hp and isinstance(hp["low"], (int, float)) \
                and isinstance(hp["high"], (int, float)):
            if hp["low"] > hp["high"]:
                errors.append(
                    f"{path}: low ({hp['low']}) > high ({hp['high']}) — empty range"
                )
            if dist == "loguniform" and (hp["low"] <= 0 or hp["high"] <= 0):
                errors.append(
                    f"{path}: loguniform requires strictly positive low/high "
                    f"(got low={hp['low']}, high={hp['high']})"
                )
            elif dist == "loguniform":
                ratio = hp["high"] / hp["low"]
                if ratio > MAX_LOGUNIFORM_RATIO:
                    errors.append(
                        f"{path}: loguniform band too wide — high/low={ratio:.4g} "
                        f"exceeds MAX_LOGUNIFORM_RATIO={MAX_LOGUNIFORM_RATIO} "
                        f"(±1 decade from anchor; PRD US-015)"
                    )
    else:  # categorical
        choices = hp.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            errors.append(f"{path}.choices: must be a non-empty list")

    allowed_keys = {"distribution", "low", "high", "choices", "citation", "note"}
    extra = set(hp.keys()) - allowed_keys
    if extra:
        errors.append(f"{path}: unexpected fields {sorted(extra)}")

    return errors


def validate_priors(data: Any) -> list[str]:
    """Return a list of validation errors. Empty list means OK."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return [f"root: expected object, got {type(data).__name__}"]

    if "model" not in data:
        errors.append("model: required field is missing")
    elif not isinstance(data["model"], str) or not data["model"].strip():
        errors.append("model: must be a non-empty string")

    if "hparams" not in data:
        errors.append("hparams: required field is missing")
        return errors  # nothing else to check

    hparams = data["hparams"]
    if not isinstance(hparams, dict):
        errors.append(f"hparams: expected object, got {type(hparams).__name__}")
        return errors

    for required in REQUIRED_HPARAMS:
        if required not in hparams:
            errors.append(f"hparams.{required}: required hparam is missing")

    for name, hp in hparams.items():
        errors.extend(_validate_hparam(name, hp))

    return errors


def load_priors(model_name: str) -> dict:
    """Load and validate the priors JSON for a model.

    Raises:
        FileNotFoundError: if the priors file does not exist.
        PriorsValidationError: if the priors file is malformed; message
            lists every offending field path.
    """
    path = PRIORS_DIR / f"{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"priors file not found: {path}. "
            f"Expected one of {SUPPORTED_MODELS}."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_priors(data)
    if errors:
        msg = f"priors file {path} failed validation:\n  - " + "\n  - ".join(errors)
        raise PriorsValidationError(msg)
    if data.get("model") != model_name:
        raise PriorsValidationError(
            f"priors file {path}: model field {data.get('model')!r} "
            f"does not match requested {model_name!r}"
        )
    return data


def priors_file_hash(model_name: str) -> str:
    """Return SHA-256 hex digest of the on-disk priors file for `model_name`.

    Used by `metrics.json.hparams_source.priors_file_hash` (PRD US-008) so a
    completed run can be traced back to the exact priors recipe that seeded
    its Optuna study. Hashing is byte-stream: any change to the file (even
    a trailing newline tweak) produces a new digest.
    """
    path = PRIORS_DIR / f"{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"priors file not found: {path}. "
            f"Expected one of {SUPPORTED_MODELS}."
        )
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _cmd_validate_only() -> int:
    """Validate every supported model's priors. Exit 0 if all pass, 1 otherwise."""
    failures: list[str] = []
    for model in SUPPORTED_MODELS:
        try:
            load_priors(model)
            print(f"OK  {model}")
        except (FileNotFoundError, PriorsValidationError) as e:
            failures.append(f"{model}: {e}")
            print(f"FAIL {model}: {e}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)}/{len(SUPPORTED_MODELS)} priors files failed.",
              file=sys.stderr)
        return 1
    print(f"\nAll {len(SUPPORTED_MODELS)} priors files valid.")
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run paper-anchored Optuna tuning.")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate every priors file under artifacts/priors/ and exit.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials per (model, dataset) pair.",
    )
    p.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        default=None,
        help="Restrict tuning to a single model (default: all).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=SUPPORTED_DATASETS,
        default=None,
        help="Restrict tuning to a single dataset (default: all).",
    )
    p.add_argument(
        "--storage",
        type=str,
        default="sqlite:///artifacts/optuna_thz.db",
        help="Optuna storage URL (default: SQLite under artifacts/).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/best_hparams",
        help="Where to write best_hparams JSONs (default: artifacts/best_hparams).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.validate_only:
        return _cmd_validate_only()

    # Lazy import: only US-005 path needs Optuna + Lightning.
    try:
        from src.tune_hyperparams import run_studies
    except ImportError as e:
        print(
            f"error: failed to import tuning runner ({e}). "
            "Install requirements.txt and re-run.",
            file=sys.stderr,
        )
        return 2

    models = [args.model] if args.model else list(SUPPORTED_MODELS)
    datasets = [args.dataset] if args.dataset else list(SUPPORTED_DATASETS)

    # US-014 quarantine guard: skip TransNeXt when iterating over all
    # SUPPORTED_MODELS. An explicit `--model transnext_*` bypasses the guard
    # so the guard cannot silently override an operator-stated intent.
    from src.experiments.run_status import is_quarantined as _is_quarantined
    if not args.model:
        before = list(models)
        models = [m for m in models if not _is_quarantined(m)]
        skipped = [m for m in before if m not in models]
        if skipped:
            print(
                f"[quarantine] skipping {len(skipped)} quarantined model(s) "
                f"({', '.join(skipped)}); pass --model {skipped[0]} to override.",
                file=sys.stderr,
            )

    return run_studies(
        models=models,
        datasets=datasets,
        n_trials=args.n_trials,
        storage=args.storage,
        out_dir=Path(args.out_dir),
        load_priors=load_priors,
    )


if __name__ == "__main__":
    sys.exit(main())
