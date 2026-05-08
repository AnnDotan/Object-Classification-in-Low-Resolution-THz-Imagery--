"""Phase A best_hparams fallback contract.

When ``artifacts/best_hparams/{model}_{dataset}.json`` is absent, ``run_cell``
must:
  (a) For Phase A cells with a known model → return CLAUDE.md frozen defaults.
  (b) For Phase B/C cells → still raise FileNotFoundError (Optuna mandatory).
  (c) For Phase A cells with an unknown model → still raise (no silent fallback).

This pins the policy decision the operator chose ("option B") so that future
refactors cannot accidentally weaken Phase B/C's tuning requirement.

Run: ``python -m src.tests.test_phase_a_hparams_fallback``
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from run_systematic import (
    PHASE_A_FROZEN_HPARAMS,
    _load_hparams_for_cell,
)
from src.experiments.matrix import cells_by_tag


def _check_phase_a_resnet50_cifar10_fallback() -> None:
    """ResNet50/CIFAR-10 Phase A: best_hparams missing → frozen defaults."""
    spec = cells_by_tag()["final_clean_resnet50_cifar10"]
    bh_path = _REPO_ROOT / "artifacts" / "best_hparams" / "resnet50_cifar10.json"
    if bh_path.exists():
        print("SKIP [phase-a-fallback-resnet] — best_hparams JSON exists; can't test fallback path")
        return

    blob = _load_hparams_for_cell(spec)
    bp = blob["best_params"]
    expected = PHASE_A_FROZEN_HPARAMS["resnet50"]
    for k, v in expected.items():
        assert bp[k] == v, f"resnet50 frozen hparam {k}: expected {v}, got {bp[k]}"
    assert blob["source"] == "claude_md_frozen"
    assert blob["phase"] == "A"
    assert blob["best_value"] is None
    print("OK [phase-a-fallback-resnet] — Phase A ResNet50 falls back to CLAUDE.md frozen.")


def _check_phase_a_densenet121_mnist_fallback() -> None:
    """DenseNet121/MNIST Phase A: best_hparams missing → frozen defaults."""
    spec = cells_by_tag()["final_clean_densenet121_mnist"]
    bh_path = _REPO_ROOT / "artifacts" / "best_hparams" / "densenet121_mnist.json"
    if bh_path.exists():
        print("SKIP [phase-a-fallback-densenet] — best_hparams JSON exists; can't test fallback path")
        return

    blob = _load_hparams_for_cell(spec)
    bp = blob["best_params"]
    expected = PHASE_A_FROZEN_HPARAMS["densenet121"]
    for k, v in expected.items():
        assert bp[k] == v, f"densenet121 frozen hparam {k}: expected {v}, got {bp[k]}"
    assert blob["source"] == "claude_md_frozen"
    print("OK [phase-a-fallback-densenet] — Phase A DenseNet121 falls back to CLAUDE.md frozen.")


def _check_phase_b_still_raises_when_missing() -> None:
    """Phase B with missing best_hparams must still raise — no silent fallback."""
    spec = cells_by_tag()["final_B_L3_resnet50_cifar10"]
    bh_path = _REPO_ROOT / "artifacts" / "best_hparams" / "resnet50_cifar10.json"
    if bh_path.exists():
        print("SKIP [phase-b-no-fallback] — best_hparams JSON exists; can't test missing path")
        return

    try:
        _load_hparams_for_cell(spec)
    except FileNotFoundError as e:
        msg = str(e)
        assert "resnet50_cifar10.json" in msg
        assert "tune_all.py" in msg, f"missing remediation hint in: {msg}"
        print("OK [phase-b-no-fallback] — Phase B still raises FileNotFoundError when hparams missing.")
        return
    raise AssertionError("expected FileNotFoundError for missing Phase B best_hparams")


def _check_phase_c_still_raises_when_missing() -> None:
    """Phase C with missing best_hparams must still raise."""
    spec = cells_by_tag()["final_C_L4_noise_densenet121_cifar10"]
    bh_path = _REPO_ROOT / "artifacts" / "best_hparams" / "densenet121_cifar10.json"
    if bh_path.exists():
        print("SKIP [phase-c-no-fallback] — best_hparams JSON exists; can't test missing path")
        return

    try:
        _load_hparams_for_cell(spec)
    except FileNotFoundError:
        print("OK [phase-c-no-fallback] — Phase C still raises FileNotFoundError when hparams missing.")
        return
    raise AssertionError("expected FileNotFoundError for missing Phase C best_hparams")


def _check_existing_json_takes_precedence() -> None:
    """A real best_hparams JSON must override the Phase A frozen fallback."""
    bh_dir = _REPO_ROOT / "artifacts" / "best_hparams"
    bh_dir.mkdir(parents=True, exist_ok=True)
    bh_path = bh_dir / "resnet50_cifar10.json"
    pre_existed = bh_path.exists()
    backup = bh_path.read_text(encoding="utf-8") if pre_existed else None

    sentinel = {
        "model": "resnet50",
        "dataset": "cifar10",
        "study_name": "sentinel_real_optuna_winner",
        "best_value": 0.987,
        "best_params": {
            "head_lr": 7.7e-4,
            "backbone_lr": 8.8e-5,
            "weight_decay": 9.9e-4,
            "label_smoothing": 0.13,
            "warmup_epochs": 4,
        },
        "n_trials_completed": 20,
        "priors_file_hash": "f" * 64,
        "phase": "B",
        "level": 3,
    }
    bh_path.write_text(json.dumps(sentinel), encoding="utf-8")
    try:
        spec = cells_by_tag()["final_clean_resnet50_cifar10"]
        blob = _load_hparams_for_cell(spec)
        # The real Optuna winner must win — not the frozen fallback.
        assert blob["best_params"]["head_lr"] == 7.7e-4, (
            "real best_hparams JSON did not take precedence over frozen fallback"
        )
        assert blob["study_name"] == "sentinel_real_optuna_winner"
        assert blob.get("source") != "claude_md_frozen"
        print("OK [json-precedence] — real best_hparams JSON overrides Phase A frozen fallback.")
    finally:
        if pre_existed:
            bh_path.write_text(backup, encoding="utf-8")
        else:
            bh_path.unlink(missing_ok=True)


def main() -> int:
    _check_phase_a_resnet50_cifar10_fallback()
    _check_phase_a_densenet121_mnist_fallback()
    _check_phase_b_still_raises_when_missing()
    _check_phase_c_still_raises_when_missing()
    _check_existing_json_takes_precedence()
    return 0


if __name__ == "__main__":
    sys.exit(main())
