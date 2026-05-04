"""run_cell() orchestration test (US-008).

The actual ResNet50/DenseNet121/TransNeXt training is too slow for unit
tests (5+ minutes even in pilot mode on CPU), so this test injects a
mock run_experiment_fn and verifies the orchestration:
  - tag -> CellSpec resolution
  - hparams loaded from artifacts/best_hparams/{model}_{dataset}.json
  - DegradeConfig fields land in run_experiment kwargs (saturation included)
  - run_name_override forced to the cell tag
  - missing best_hparams raises FileNotFoundError with a clear remediation hint
  - metrics.json is rewritten to include hparams + cell metadata

Run: ``python -m src.tests.test_run_cell``
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Inserting repo root so `import run_systematic` finds the file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import run_systematic
from run_systematic import (
    _cell_config,
    _load_best_hparams,
    _merge_metadata_into_metrics_json,
    run_cell,
)
from src.experiments.matrix import cells_by_tag


_FAKE_BEST = {
    "model": "resnet50",
    "dataset": "cifar10",
    "study_name": "resnet50_cifar10_L3",
    "best_value": 0.6234,
    "best_params": {
        "head_lr": 1.5e-3,
        "backbone_lr": 2e-4,
        "weight_decay": 5e-4,
        "label_smoothing": 0.1,
        "warmup_epochs": 3.4,
    },
    "n_trials_completed": 20,
    "priors_file_hash": "a" * 64,
    "phase": "B",
    "level": 3,
}


def _check_cell_config_shape() -> None:
    """Verify _cell_config maps a Phase C cell + best_hparams to the right kwargs."""
    spec = cells_by_tag()["final_C_L4_noise_resnet50_cifar10"]
    cfg = _cell_config(spec, _FAKE_BEST, mode="full")

    assert cfg["model_name"] == "resnet50"
    assert cfg["dataset"] == "cifar10"
    assert cfg["group"] == "final"
    assert cfg["run_name_override"] == "final_C_L4_noise_resnet50_cifar10"
    assert cfg["tag"] == "final_C_L4_noise_resnet50_cifar10"

    # Hparams round-tripped (warmup rounded to int)
    assert cfg["lr"] == 1.5e-3
    assert cfg["backbone_lr"] == 2e-4
    assert cfg["weight_decay"] == 5e-4
    assert cfg["label_smoothing"] == 0.1
    assert cfg["warmup_epochs"] == 3
    assert isinstance(cfg["warmup_epochs"], int)

    # Degradation: noise at L4, others at L1 (Phase C isolation)
    assert cfg["degradation_type"] == "all"
    assert cfg["gaussian_noise_std"] == spec.degrade_config.gaussian_noise_std
    assert cfg["saturation"] == spec.degrade_config.saturation
    assert cfg["low_res"] == spec.degrade_config.low_res
    assert cfg["blur_kernel"] == spec.degrade_config.blur_kernel

    # Final-mode training schedule (CLAUDE.md: 60 epochs / patience 10)
    assert cfg["epochs"] == 60
    assert cfg["early_stopping_patience"] == 10
    assert cfg["freeze_backbone"] is False
    assert cfg["scheduler_type"] == "cosine"

    print("OK [config] — _cell_config maps spec + hparams to run_experiment kwargs.")


def _check_pilot_mode_shorter() -> None:
    spec = cells_by_tag()["final_clean_resnet50_cifar10"]
    cfg = _cell_config(spec, _FAKE_BEST, mode="pilot")
    assert cfg["epochs"] == 5
    assert cfg["early_stopping_patience"] == 2
    assert cfg["train_subset"] == 2000
    print("OK [pilot] — pilot mode shrinks epochs/subset/patience as expected.")


def _check_missing_hparams_raises() -> None:
    """Override PRIORS_DIR so we can stage missing best_hparams cleanly."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Pretend artifacts/best_hparams is empty by pointing CWD-relative
        # path resolution at a fresh dir.
        cwd_swap = _REPO_ROOT / "artifacts" / "best_hparams"
        # The function reads "artifacts/best_hparams/<model>_<dataset>.json"
        # relative to CWD. We can't easily swap CWD here; instead we hit a
        # known-missing pair. ResNet50/MNIST hasn't been tuned yet, so its
        # best_hparams JSON shouldn't exist.
        target = cwd_swap / "resnet50_mnist.json"
        if target.exists():
            print("SKIP [missing-hparams] — resnet50_mnist.json exists; can't test missing path here")
            return

        try:
            _load_best_hparams("resnet50", "mnist")
        except FileNotFoundError as e:
            msg = str(e)
            assert "resnet50_mnist.json" in msg, f"error doesn't name file: {msg}"
            assert "tune_all.py" in msg, f"error lacks remediation hint: {msg}"
            print("OK [missing-hparams] — raises FileNotFoundError naming file + remediation.")
            return
        raise AssertionError("expected FileNotFoundError for missing best_hparams")


def _check_metrics_merge_shape() -> None:
    """_merge_metadata_into_metrics_json overlays hparams + cell metadata."""
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        # Simulate what LegacyJSONMetricsCallback writes
        (run_dir / "metrics.json").write_text(json.dumps({
            "best_val_acc": 0.834,
            "best_epoch": 17,
            "last_val_acc": 0.831,
            "epochs_run": 23,
        }), encoding="utf-8")

        spec = cells_by_tag()["final_B_L3_resnet50_cifar10"]
        _merge_metadata_into_metrics_json(run_dir, spec, _FAKE_BEST)

        merged = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    # Original keys preserved
    assert merged["best_val_acc"] == 0.834
    assert merged["epochs_run"] == 23
    # New keys added
    assert merged["hparams"] == _FAKE_BEST["best_params"]
    assert merged["hparams_source"]["study_name"] == "resnet50_cifar10_L3"
    assert merged["hparams_source"]["best_value"] == 0.6234
    assert merged["hparams_source"]["priors_file_hash"] == "a" * 64
    assert merged["cell_tag"] == "final_B_L3_resnet50_cifar10"
    assert merged["phase"] == "B"
    assert merged["level"] == 3
    assert merged["axis"] is None
    assert merged["model"] == "resnet50"
    assert merged["dataset"] == "cifar10"
    print("OK [merge] — metrics.json gains hparams + cell metadata round-trip.")


def _check_unknown_tag_rejected() -> None:
    try:
        run_cell("not_a_real_tag", mode="pilot",
                 run_experiment_fn=lambda **_: Path("/tmp/never"))
    except ValueError as e:
        assert "not_a_real_tag" in str(e)
        print("OK [unknown-tag] — run_cell rejects an unknown cell tag.")
        return
    raise AssertionError("expected ValueError for unknown cell tag")


def _check_run_cell_invokes_with_correct_kwargs(monkeypatch_path: Path) -> None:
    """End-to-end with mocked training: assert the kwargs run_experiment receives."""
    captured: dict = {}

    def fake_run_experiment(**kwargs):
        captured.update(kwargs)
        run_dir = monkeypatch_path / kwargs["run_name_override"]
        run_dir.mkdir(parents=True, exist_ok=True)
        # Mimic what LegacyJSONMetricsCallback writes at end of training
        (run_dir / "metrics.json").write_text(json.dumps({
            "best_val_acc": 0.5, "best_epoch": 1, "last_val_acc": 0.5,
            "last_train_acc": 0.7, "epochs_run": 1,
        }), encoding="utf-8")
        return run_dir

    # Stage a synthetic best_hparams file in the actual artifacts dir so
    # _load_best_hparams finds it (it's relative to CWD).
    bh_dir = _REPO_ROOT / "artifacts" / "best_hparams"
    bh_dir.mkdir(parents=True, exist_ok=True)
    bh_path = bh_dir / "resnet50_cifar10.json"
    pre_existed = bh_path.exists()
    backup = bh_path.read_text(encoding="utf-8") if pre_existed else None
    bh_path.write_text(json.dumps(_FAKE_BEST), encoding="utf-8")
    try:
        run_dir = run_cell(
            "final_clean_resnet50_cifar10",
            mode="pilot",
            run_experiment_fn=fake_run_experiment,
        )
        # Override actual run_dir to the temp path the fake created
        run_dir = monkeypatch_path / "final_clean_resnet50_cifar10"
        # We bypassed default run_dir logic but test the kwargs that flowed in
        assert captured["run_name_override"] == "final_clean_resnet50_cifar10"
        assert captured["model_name"] == "resnet50"
        assert captured["dataset"] == "cifar10"
        assert captured["group"] == "final"
        assert captured["lr"] == 1.5e-3
        # Phase A clean: degradation_type='none'
        assert captured["degradation_type"] == "none"
        # Even in pilot, hparams round-trip
        assert captured["weight_decay"] == 5e-4
        # warmup rounded to int
        assert captured["warmup_epochs"] == 3
        # The post-call merge should have run on the fake's metrics.json
        merged = json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        assert merged["cell_tag"] == "final_clean_resnet50_cifar10"
        assert merged["phase"] == "A"
        assert merged["hparams"]["head_lr"] == 1.5e-3
        print("OK [dispatch] — run_cell invokes run_experiment with correct kwargs "
              "and merges hparams into metrics.json.")
    finally:
        if pre_existed:
            bh_path.write_text(backup, encoding="utf-8")
        else:
            bh_path.unlink(missing_ok=True)


def main() -> int:
    _check_cell_config_shape()
    _check_pilot_mode_shorter()
    _check_missing_hparams_raises()
    _check_metrics_merge_shape()
    _check_unknown_tag_rejected()
    with tempfile.TemporaryDirectory() as td:
        _check_run_cell_invokes_with_correct_kwargs(Path(td))
    return 0


if __name__ == "__main__":
    sys.exit(main())
