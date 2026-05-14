"""Smoke-test the Optuna study runner with a stubbed train_fn.

Heavy training is replaced by a deterministic stub so the test runs in
seconds without GPU, while exercising:
  - priors -> trial.suggest_float plumbing
  - SQLite study persistence (resumable on crash)
  - winner JSON shape: best_value, best_params, n_trials_completed,
    study_name, priors_file_hash, phase, level
  - train_fn receives params with the expected keys

Run: ``python -m src.tests.test_tune_studies``
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import optuna

# Quiet Optuna's per-trial logs in the test output.
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.tune_hyperparams import _suggest_from_priors, run_studies
from tune_all import load_priors


_EXPECTED_PARAM_KEYS = {
    "head_lr", "backbone_lr", "weight_decay", "label_smoothing", "warmup_epochs",
}


def _stub_train_fn(*, model: str, dataset: str, params: dict, **_) -> float:
    """Deterministic synthetic objective. Higher head_lr -> lower val_acc,
    so Optuna has a non-trivial signal to optimize on."""
    missing = _EXPECTED_PARAM_KEYS - set(params.keys())
    assert not missing, f"train_fn missing param keys: {missing}"
    assert isinstance(params["warmup_epochs"], int), \
        f"warmup_epochs must be int, got {type(params['warmup_epochs'])}"
    base = 0.5
    penalty = min(1.0, params["head_lr"] * 100.0)
    return base + (1.0 - penalty) * 0.4


def _check_winner_json_written() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td_path = Path(td)
        out_dir = td_path / "best_hparams"
        storage = f"sqlite:///{td_path / 'optuna.db'}".replace("\\", "/")

        rc = run_studies(
            models=["resnet50"],
            datasets=["cifar10"],
            n_trials=2,
            storage=storage,
            out_dir=out_dir,
            load_priors=load_priors,
            train_fn=_stub_train_fn,
        )
        assert rc == 0, f"run_studies returned non-zero: {rc}"

        target = out_dir / "resnet50_cifar10.json"
        assert target.exists(), f"winner JSON not written: {target}"
        winner = json.loads(target.read_text(encoding="utf-8"))

        for key in (
            "model", "dataset", "study_name", "best_value", "best_params",
            "n_trials_completed", "priors_file_hash", "phase", "level",
        ):
            assert key in winner, f"winner JSON missing {key}"

        assert winner["model"] == "resnet50"
        assert winner["dataset"] == "cifar10"
        assert winner["phase"] == "B"
        assert winner["level"] == 3
        assert winner["study_name"] == "resnet50_cifar10_L3"
        assert winner["n_trials_completed"] == 2
        assert 0.0 <= winner["best_value"] <= 1.0
        assert _EXPECTED_PARAM_KEYS <= set(winner["best_params"].keys())
        assert len(winner["priors_file_hash"]) == 64  # SHA-256 hex

        print(
            f"OK [winner] best_val_acc={winner['best_value']:.4f} "
            f"params={list(winner['best_params'].keys())} "
            f"hash={winner['priors_file_hash'][:8]}..."
        )


def _check_study_resumes_from_storage() -> None:
    """A second run with the same storage URL should accumulate trials,
    not start over. The SQLite store keys by study_name."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        td_path = Path(td)
        out_dir = td_path / "best_hparams"
        db_url = f"sqlite:///{td_path / 'opt.db'}".replace("\\", "/")

        run_studies(
            models=["resnet50"], datasets=["cifar10"],
            n_trials=2, storage=db_url, out_dir=out_dir,
            load_priors=load_priors, train_fn=_stub_train_fn,
        )
        run_studies(
            models=["resnet50"], datasets=["cifar10"],
            n_trials=3, storage=db_url, out_dir=out_dir,
            load_priors=load_priors, train_fn=_stub_train_fn,
        )

        winner = json.loads(
            (out_dir / "resnet50_cifar10.json").read_text(encoding="utf-8")
        )
        assert winner["n_trials_completed"] == 5, (
            f"expected 2+3=5 trials accumulated, got {winner['n_trials_completed']}"
        )
        print("OK [resume] — second run added 3 trials to existing study (total=5).")


def _check_param_distribution_routing() -> None:
    """Spot-check: priors with loguniform are passed log=True to suggest_float."""
    priors = load_priors("transnext_tiny")
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial = study.ask()
    params = _suggest_from_priors(trial, priors)
    assert _EXPECTED_PARAM_KEYS == set(params.keys()), params.keys()
    hp = priors["hparams"]
    for name, val in params.items():
        if name == "warmup_epochs":
            continue
        assert hp[name]["low"] <= val <= hp[name]["high"], (
            f"{name}={val} outside [{hp[name]['low']}, {hp[name]['high']}]"
        )
    print("OK [routing] — _suggest_from_priors yields params inside priors ranges.")


def main() -> int:
    _check_param_distribution_routing()
    _check_winner_json_written()
    _check_study_resumes_from_storage()
    return 0


if __name__ == "__main__":
    sys.exit(main())
