"""PyTorch Lightning training engine for the THz-like degradation project.

Public surface:
    THzClassifier      LightningModule wrapping ResNet/DenseNet/TransNeXt.
    THzDataModule      LightningDataModule wrapping THzLikeCIFAR10 / THzLikeMNIST.
    run_experiment     Drop-in replacement for src.runner.run_experiment(...).

The legacy hand-rolled trainer in src/runner.py is preserved for parity validation
and is selected via the --engine legacy flag in run_systematic.py / run_all_phases.py.
"""

from .module import THzClassifier
from .datamodule import THzDataModule
from .train import run_experiment

__all__ = ["THzClassifier", "THzDataModule", "run_experiment"]
