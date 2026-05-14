"""Lightning callbacks that keep the legacy artifact schema intact.

Why these exist:
- The four dashboards in src/tools/ scan runs/<group>/<run_name>/metrics.csv
  and metrics.json. Lightning's own loggers write to lightning_logs/ in a
  different schema. These callbacks bridge that gap so dashboards keep working.
"""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch


class LegacyMetricsCSVCallback(pl.Callback):
    """Append [epoch, train_loss, train_acc, val_loss, val_acc] to <run_dir>/metrics.csv
    after every validation epoch — matches src/runner.py's schema verbatim."""

    HEADER = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / "metrics.csv"
        if not self.metrics_path.exists():
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADER)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        epoch = trainer.current_epoch + 1
        row = [
            epoch,
            f"{float(m.get('train_loss', float('nan'))):.6f}",
            f"{float(m.get('train_acc',  float('nan'))):.6f}",
            f"{float(m.get('val_loss',   float('nan'))):.6f}",
            f"{float(m.get('val_acc',    float('nan'))):.6f}",
        ]
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


class LegacyJSONMetricsCallback(pl.Callback):
    """Write a metrics.json summary at end of training (best/last val acc, epochs).

    Also captures W&B (run_id, entity, project) when a WandbLogger is wired
    into the Trainer (US-012). Fields are written as null when W&B is offline
    or absent — never omitted, so the dashboard's tag-based lookup is stable.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.json_path = self.run_dir / "metrics.json"
        self._best_val_acc = -1.0
        self._best_epoch = 0
        self._last_val_acc = float("nan")
        self._last_train_acc = float("nan")
        self._last_train_loss = float("nan")
        self._last_val_loss = float("nan")
        self._epochs_run = 0
        # Dashboard fields populated by on_train_start / on_train_end.
        self._started_monotonic: Optional[float] = None
        self._started_at: Optional[str] = None
        self._finished_at: Optional[str] = None
        self._runtime_s: Optional[float] = None

    @staticmethod
    def _utc_isoformat() -> str:
        # ISO-8601, UTC, second precision — round-trips through json + the
        # final_exp_schema str field with no parsing on the dashboard side.
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def on_train_start(self, trainer: pl.Trainer, _pl_module) -> None:
        # Lightning's sanity-check validation runs BEFORE on_train_start
        # so this is the right hook to anchor wall-clock start.
        if self._started_monotonic is None:
            self._started_monotonic = time.monotonic()
            self._started_at = self._utc_isoformat()

    def on_validation_epoch_end(self, trainer: pl.Trainer, _pl_module) -> None:
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        va = float(m.get("val_acc", float("nan")))
        epoch = trainer.current_epoch + 1
        self._epochs_run = epoch
        self._last_val_acc = va
        self._last_val_loss = float(m.get("val_loss", float("nan")))
        self._last_train_acc = float(m.get("train_acc", float("nan")))
        self._last_train_loss = float(m.get("train_loss", float("nan")))
        if va > self._best_val_acc:
            self._best_val_acc = va
            self._best_epoch = epoch

    @staticmethod
    def _extract_wandb_ids(trainer: pl.Trainer) -> tuple:
        """Find the (run_id, entity, project) for any WandbLogger on the Trainer.

        Returns (None, None, None) if W&B is absent. We never raise — even if
        the WandbLogger errored, the campaign metrics.json must still be written.
        """
        loggers = trainer.loggers or []
        for logger in loggers:
            if logger.__class__.__name__ != "WandbLogger":
                continue
            try:
                exp = logger.experiment
                run_id = getattr(exp, "id", None)
                entity = getattr(exp, "entity", None)
                project = getattr(exp, "project", None)
                return run_id, entity, project
            except Exception:
                return None, None, None
        return None, None, None

    def on_train_end(self, trainer: pl.Trainer, _pl_module) -> None:
        wandb_run_id, wandb_entity, wandb_project = self._extract_wandb_ids(trainer)
        # Wallclock: if on_train_start fired, runtime_s = monotonic delta; if
        # the run aborted before on_train_start (unlikely under PL), the field
        # stays null and the dashboard renders "—".
        if self._started_monotonic is not None:
            self._runtime_s = float(time.monotonic() - self._started_monotonic)
            self._finished_at = self._utc_isoformat()
        payload = {
            "best_val_acc": self._best_val_acc,
            "best_epoch": self._best_epoch,
            "last_val_acc": self._last_val_acc,
            "last_train_acc": self._last_train_acc,
            "last_val_loss": self._last_val_loss,
            "last_train_loss": self._last_train_loss,
            "epochs_run": self._epochs_run,
            # US-012: always written, even when null (dashboard relies on this).
            "wandb_run_id": wandb_run_id,
            "wandb_entity": wandb_entity,
            "wandb_project": wandb_project,
            # Dashboard runtime/timing fields — always written (null when
            # train start never fired). Consumed by build_final_exp_json.py.
            "runtime_s": self._runtime_s,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


class HistoryJSONCallback(pl.Callback):
    """Per-epoch learning curve writer for the dashboard's lazy Plotly drawer (US-018).

    Schema (`runs/final/<tag>/history.json`):
        {
          "schema_version": 1,
          "tag":     "<tag>",
          "history": [
            {"epoch": 1, "train_loss": ..., "val_loss": ..., "train_acc": ..., "val_acc": ...},
            ...
          ]
        }

    Writes atomically (tmp + os.replace) on every `on_validation_epoch_end`
    so a SIGINT'd run still leaves a usable curve. NaN values are coerced
    to None so `JSON.parse` succeeds in the browser (raw NaN is invalid JSON).

    Privacy: never reads checkpoints; all values come from
    `trainer.callback_metrics`. The schema deliberately excludes
    paths, model state, or anything not directly plotted.
    """

    def __init__(self, run_dir: Path, tag: str):
        self.run_dir = Path(run_dir)
        self.tag = tag
        self.json_path = self.run_dir / "history.json"
        self._history: list[dict] = []

    @staticmethod
    def _safe(v) -> Optional[float]:
        """Coerce torch tensors / NaN to JSON-friendly numbers or None."""
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    def _write_atomic(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "tag": self.tag,
            "history": list(self._history),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self.run_dir),
            prefix=self.json_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tmp = Path(tf.name)
            json.dump(payload, tf, indent=2, ensure_ascii=False)
            tf.write("\n")
        os.replace(tmp, self.json_path)

    def on_validation_epoch_end(self, trainer: pl.Trainer, _pl_module) -> None:
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        epoch = trainer.current_epoch + 1
        self._history.append({
            "epoch": epoch,
            "train_loss": self._safe(m.get("train_loss")),
            "val_loss":   self._safe(m.get("val_loss")),
            "train_acc":  self._safe(m.get("train_acc")),
            "val_acc":    self._safe(m.get("val_acc")),
        })
        # Atomic write per epoch — on SIGINT we still leave a valid file.
        try:
            self._write_atomic()
        except OSError:
            # Disk full or similar; one missed atomic write must not
            # crash training. Next epoch will retry.
            pass


class LegacyCheckpointCallback(pl.Callback):
    """Save best.pt and model_last.pt with the legacy state-dict schema so existing
    tooling that loads checkpoints (visualize_run.py, dashboard inspectors) keeps
    working without modification."""

    def __init__(self, run_dir: Path, run_meta: dict):
        self.run_dir = Path(run_dir)
        self.run_meta = run_meta
        self._best_val_acc = -1.0

    def _save(self, pl_module: pl.LightningModule, path: Path, val_acc: float, epoch: int) -> None:
        torch.save(
            {
                "model": pl_module.model.state_dict(),
                "best_val_acc": val_acc,
                "epoch": epoch,
                **self.run_meta,
            },
            path,
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        va = float(trainer.callback_metrics.get("val_acc", -1.0))
        if va > self._best_val_acc:
            self._best_val_acc = va
            self._save(pl_module, self.run_dir / "best.pt", va, trainer.current_epoch + 1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        last_va = float(trainer.callback_metrics.get("val_acc", float("nan")))
        self._save(pl_module, self.run_dir / "model_last.pt", last_va, trainer.current_epoch + 1)


class LogSampleImagesCallback(pl.Callback):
    """Once at training start: grab one batch, un-normalize, save a sample-input
    montage to <run_dir>/sample_inputs.png and (if W&B online) upload."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, run_dir: Path, n: int = 8):
        self.run_dir = Path(run_dir)
        self.n = n

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.IMAGENET_MEAN, device=x.device).view(3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, device=x.device).view(3, 1, 1)
        return (x * std + mean).clamp(0, 1)

    def on_train_start(self, trainer: pl.Trainer, _pl_module) -> None:
        try:
            batch = next(iter(trainer.datamodule.train_dataloader()))
        except Exception:
            return
        x, _y = batch
        x = x[: self.n]
        x_vis = self._denorm(x).cpu()
        try:
            from torchvision.utils import save_image, make_grid
            grid = make_grid(x_vis, nrow=min(self.n, 4))
            save_image(grid, self.run_dir / "sample_inputs.png")
        except Exception:
            return
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({"sample_inputs": [wandb.Image(x_vis[i]) for i in range(x_vis.size(0))]})
        except Exception:
            pass


class DashboardRefreshCallback(pl.Callback):
    """At end of training, regenerate the experiment-plan dashboard (mirrors
    src/runner.py:415-427)."""

    def on_train_end(self, _trainer, _pl_module) -> None:
        try:
            from src.tools.generate_experiment_plan_dashboard import main as gen_dash
            gen_dash()
        except Exception as e:
            print(f"[WARN] dashboard refresh failed: {e}")
