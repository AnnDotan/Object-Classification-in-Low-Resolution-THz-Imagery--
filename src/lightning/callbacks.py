"""Lightning callbacks that keep the legacy artifact schema intact.

Why these exist:
- The four dashboards in src/tools/ scan runs/<group>/<run_name>/metrics.csv
  and metrics.json. Lightning's own loggers write to lightning_logs/ in a
  different schema. These callbacks bridge that gap so dashboards keep working.
"""
from __future__ import annotations

import csv
import json
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
    """Write a metrics.json summary at end of training (best/last val acc, epochs)."""

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

    def on_train_end(self, trainer: pl.Trainer, _pl_module) -> None:
        payload = {
            "best_val_acc": self._best_val_acc,
            "best_epoch": self._best_epoch,
            "last_val_acc": self._last_val_acc,
            "last_train_acc": self._last_train_acc,
            "last_val_loss": self._last_val_loss,
            "last_train_loss": self._last_train_loss,
            "epochs_run": self._epochs_run,
        }
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


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
