"""W&B run-id capture in metrics.json (US-012).

Verifies LegacyJSONMetricsCallback:
  - extracts (run_id, entity, project) from a WandbLogger on the trainer
  - writes wandb_* as null when no WandbLogger is present
  - writes wandb_* as null when the WandbLogger raises on .experiment access
    (W&B offline init failures must not kill metrics.json)
  - never omits the three keys — dashboard relies on stable schema

Run: ``python -m src.tests.test_wandb_capture``
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.lightning.callbacks import LegacyJSONMetricsCallback


class _FakeWandbLogger:
    """Mimics pytorch_lightning.loggers.WandbLogger for this test."""
    def __init__(self, run_id: str | None, entity: str | None, project: str | None,
                 raise_on_experiment: bool = False):
        self._exp = SimpleNamespace(id=run_id, entity=entity, project=project)
        self._raise = raise_on_experiment

    @property
    def experiment(self):
        if self._raise:
            raise RuntimeError("simulated WandbLogger experiment failure")
        return self._exp


_FakeWandbLogger.__name__ = "WandbLogger"  # callback matches by class name


class _FakeOtherLogger:
    """A non-WandB logger; should be ignored."""
    pass


def _make_trainer(*loggers):
    return SimpleNamespace(loggers=list(loggers), sanity_checking=False)


def _flush(cb: LegacyJSONMetricsCallback, trainer) -> dict:
    """Trigger on_train_end and read back the JSON."""
    cb._best_val_acc = 0.81
    cb._best_epoch = 12
    cb._last_val_acc = 0.79
    cb._last_train_acc = 0.85
    cb._last_train_loss = 0.4
    cb._last_val_loss = 0.5
    cb._epochs_run = 12
    cb.on_train_end(trainer, None)
    return json.loads(cb.json_path.read_text(encoding="utf-8"))


_REQUIRED_KEYS = (
    "best_val_acc", "best_epoch", "last_val_acc", "last_train_acc",
    "last_val_loss", "last_train_loss", "epochs_run",
    "wandb_run_id", "wandb_entity", "wandb_project",
)


def _check_keys_present(d: dict, label: str) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in d]
    assert not missing, f"[{label}] missing keys: {missing}"


def _check_with_wandb_logger(td: Path) -> None:
    cb = LegacyJSONMetricsCallback(td)
    trainer = _make_trainer(
        _FakeOtherLogger(),
        _FakeWandbLogger("run_xyz", "thz-team", "thz-final"),
    )
    payload = _flush(cb, trainer)
    _check_keys_present(payload, "with-wandb")
    assert payload["wandb_run_id"] == "run_xyz"
    assert payload["wandb_entity"] == "thz-team"
    assert payload["wandb_project"] == "thz-final"
    assert payload["best_val_acc"] == 0.81
    print("OK [with-wandb] — run_id/entity/project captured into metrics.json.")


def _check_without_wandb_logger(td: Path) -> None:
    cb = LegacyJSONMetricsCallback(td)
    trainer = _make_trainer(_FakeOtherLogger())
    payload = _flush(cb, trainer)
    _check_keys_present(payload, "no-wandb")
    assert payload["wandb_run_id"] is None
    assert payload["wandb_entity"] is None
    assert payload["wandb_project"] is None
    print("OK [no-wandb] — wandb_* fields written as null when no WandbLogger.")


def _check_wandb_logger_raises(td: Path) -> None:
    cb = LegacyJSONMetricsCallback(td)
    trainer = _make_trainer(
        _FakeWandbLogger("x", "y", "z", raise_on_experiment=True),
    )
    payload = _flush(cb, trainer)
    _check_keys_present(payload, "raise")
    assert payload["wandb_run_id"] is None
    assert payload["wandb_entity"] is None
    assert payload["wandb_project"] is None
    print("OK [raise] — WandbLogger.experiment failure -> nulls, no exception escapes.")


def _check_no_loggers_attr(td: Path) -> None:
    """trainer.loggers None should be tolerated."""
    cb = LegacyJSONMetricsCallback(td)
    trainer = SimpleNamespace(loggers=None, sanity_checking=False)
    payload = _flush(cb, trainer)
    _check_keys_present(payload, "loggers-none")
    assert payload["wandb_run_id"] is None
    print("OK [loggers-none] — trainer.loggers=None tolerated, fields null.")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        _check_with_wandb_logger(Path(td))
    with tempfile.TemporaryDirectory() as td:
        _check_without_wandb_logger(Path(td))
    with tempfile.TemporaryDirectory() as td:
        _check_wandb_logger_raises(Path(td))
    with tempfile.TemporaryDirectory() as td:
        _check_no_loggers_attr(Path(td))
    return 0


if __name__ == "__main__":
    sys.exit(main())
