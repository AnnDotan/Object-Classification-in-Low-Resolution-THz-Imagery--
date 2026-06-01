"""US-045 shared helpers — checkpoint loading + dataset reconstruction.

Used by:
  - scripts/run_test_set_inference.py
  - scripts/generate_confusion_matrices.py
  - scripts/generate_calibration_diagrams.py

NOT used by:
  - scripts/measure_inference_throughput.py (independent, no checkpoint load)

Design contract:
  - Reconstruct the THzClassifier from runs/final/<tag>/metrics.json (the
    only structured cell-config artifact). The metrics.json carries
    `model`, `dataset`, `degrade_config`, `hparams`, `seed`, `phase`,
    `treatment`, `level`, `axis`. We use degrade_config to build a
    DataConfig (val=True branch since the held-out test partition lives
    behind train=False).
  - The checkpoint format is LegacyCheckpointCallback's:
        {"model": state_dict, "best_val_acc": ..., "epoch": ..., **run_meta}
  - We load `model` into a fresh THzClassifier built with the right model
    name and instantiation params. We use eval() + torch.no_grad() for
    inference. bf16-mixed precision matches training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.datasets import DataConfig, THzLikeCIFAR10, THzLikeMNIST
from src.data.degradation_levels import IDENTITY_VALUES, level_params
from src.lightning.module import THzClassifier


# DEGRADATION_LEVELS keys -> DegradeConfig keys
_LEVELS_TO_DC = {
    "low_res": "low_res",
    "blur_kernel": "blur_kernel",
    "blur_sigma": "blur_sigma",
    "noise_std": "gaussian_noise_std",
    "salt_pepper": "salt_pepper_amount",
    "saturation": "saturation",
}


def _reconstruct_degrade_config(phase: str, level: int, axis: Optional[str]) -> dict:
    """Legacy fallback when metrics.json predates the degrade_config field.

    Translates the DEGRADATION_LEVELS table (degradation_levels.py) into the
    DegradeConfig field-name convention used by DataConfig and the
    metrics.json `degrade_config` block on newer cells.
    """
    if phase == "A":
        # Clean baseline — no degradation
        params = dict(IDENTITY_VALUES)
        deg_type = "none"
    elif phase in ("B", "D"):
        # Combined Phase B / Phase D (D inherits B's DegradeConfig at level L)
        params = level_params(level)
        deg_type = "all"
    elif phase == "C":
        # Single-axis isolation
        if axis is None:
            raise ValueError(f"Phase C requires axis (got None) at L{level}")
        params = level_params(level, axis=axis)
        deg_type = axis
    else:
        raise ValueError(
            f"Cannot reconstruct degrade_config for phase={phase!r} L{level} axis={axis!r} "
            f"— newer phases (B2, B2nr, C2) MUST carry the degrade_config field in metrics.json"
        )

    dc = {_LEVELS_TO_DC[k]: v for k, v in params.items()}
    dc["out_size"] = 224
    dc["degradation_type"] = deg_type
    return dc


@dataclass
class CellSpec:
    """Lightweight cell descriptor parsed from metrics.json."""

    tag: str  # e.g. "final_B2_L3_resnet50_cifar10" (BASE tag, without _seedN)
    seed: int  # 42 / 43 / 44
    run_dir: Path  # actual on-disk dir (may have _seed{N} suffix)
    model_name: str
    dataset: str  # "cifar10" | "mnist"
    phase: str
    treatment: Optional[str]
    level: int
    axis: Optional[str]
    degrade_config: dict
    best_val_acc: float


def load_cell_spec(run_dir: Path) -> CellSpec:
    mj = run_dir / "metrics.json"
    m = json.loads(mj.read_text())
    phase = m.get("phase")
    level = int(m.get("level") or 0)
    axis = m.get("axis")
    if "degrade_config" in m:
        dc = dict(m["degrade_config"])
    else:
        # Legacy B1 / D-T3 / C / A cells predate the degrade_config field.
        dc = _reconstruct_degrade_config(phase, level, axis)
    return CellSpec(
        tag=m.get("cell_tag", run_dir.name.rsplit("_seed", 1)[0] if "_seed" in run_dir.name else run_dir.name),
        seed=int(m.get("seed") or 42),
        run_dir=run_dir,
        model_name=m["model"],
        dataset=m["dataset"],
        phase=phase,
        treatment=m.get("treatment") or m.get("phase_d_treatment"),
        level=level,
        axis=axis,
        degrade_config=dc,
        best_val_acc=float(m.get("best_val_acc", 0.0)),
    )


def build_classifier(spec: CellSpec, device: torch.device) -> THzClassifier:
    """Reconstruct THzClassifier, load state_dict from <run_dir>/best.pt."""
    ckpt = torch.load(spec.run_dir / "best.pt", map_location=device, weights_only=False)
    state_dict = ckpt["model"]

    # Minimal init kwargs — we only need forward(). Training-only knobs (lr,
    # weight_decay, scheduler) are no-ops in eval. Drop_path / dropout are
    # zeroed for inference reproducibility (state_dict matches regardless).
    is_transnext = spec.model_name.startswith("transnext_")
    init_kwargs = dict(
        model_name=spec.model_name,
        num_classes=10,
        pretrained=is_transnext,  # CNN pretrained weights are overridden by ckpt; TransNeXt needs the structural skeleton
        lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
        label_smoothing=0.0,
        drop_path_rate=0.0,
        dropout=0.0,
        pretrain_size=224 if is_transnext else None,
    )
    classifier = THzClassifier(**init_kwargs)
    missing, unexpected = classifier.model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading {spec.run_dir}: {unexpected[:5]}")
    if missing:
        # TransNeXt's swattention buffer keys are non-persistent and missing is OK
        non_buffer_missing = [k for k in missing if "buffer" not in k.lower() and "running" not in k.lower()]
        if non_buffer_missing:
            print(f"  [warn] missing non-buffer keys for {spec.run_dir.name}: {non_buffer_missing[:5]}")
    classifier.eval()
    classifier.to(device)
    return classifier


def build_test_loader(
    spec: CellSpec,
    test_indices: list[int],
    batch_size: int = 64,
) -> DataLoader:
    """Build a DataLoader for the held-out test split.

    Uses `train=False` on the official torchvision dataset (the test
    partition), then Subsets to `test_indices`. DegradeConfig is rebuilt
    from spec.degrade_config so the test-set inference uses the same per-
    sample DegradeConfig (and the same per-sample seed offsets) as the
    training-time validation.
    """
    dc = spec.degrade_config
    cfg = DataConfig(
        dataset=spec.dataset,
        train=False,  # official test partition
        out_size=int(dc["out_size"]),
        low_res=int(dc["low_res"]),
        degradation_type=str(dc.get("degradation_type", "all")),
        blur_kernel=int(dc["blur_kernel"]) if dc.get("blur_kernel") is not None else None,
        blur_sigma=float(dc["blur_sigma"]) if dc.get("blur_sigma") is not None else None,
        gaussian_noise_std=float(dc["gaussian_noise_std"]) if dc.get("gaussian_noise_std") is not None else None,
        salt_pepper_amount=float(dc["salt_pepper_amount"]) if dc.get("salt_pepper_amount") is not None else None,
        saturation=float(dc["saturation"]) if dc.get("saturation") is not None else None,
    )
    cls = THzLikeMNIST if spec.dataset == "mnist" else THzLikeCIFAR10
    full = cls(cfg)
    subset = Subset(full, sorted(test_indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return loader


def build_val_loader(spec: CellSpec, val_subset: int = 5000, batch_size: int = 64) -> DataLoader:
    """Match the training-time val loader for diagnostics (indices 0..val_subset-1)."""
    dc = spec.degrade_config
    cfg = DataConfig(
        dataset=spec.dataset,
        train=False,
        out_size=int(dc["out_size"]),
        low_res=int(dc["low_res"]),
        degradation_type=str(dc.get("degradation_type", "all")),
        blur_kernel=int(dc["blur_kernel"]) if dc.get("blur_kernel") is not None else None,
        blur_sigma=float(dc["blur_sigma"]) if dc.get("blur_sigma") is not None else None,
        gaussian_noise_std=float(dc["gaussian_noise_std"]) if dc.get("gaussian_noise_std") is not None else None,
        salt_pepper_amount=float(dc["salt_pepper_amount"]) if dc.get("salt_pepper_amount") is not None else None,
        saturation=float(dc["saturation"]) if dc.get("saturation") is not None else None,
    )
    cls = THzLikeMNIST if spec.dataset == "mnist" else THzLikeCIFAR10
    full = cls(cfg)
    subset = Subset(full, list(range(min(val_subset, len(full)))))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)


@torch.no_grad()
def run_inference(
    classifier: THzClassifier,
    loader: DataLoader,
    device: torch.device,
    use_bf16: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits, labels) as numpy float32 / int64."""
    all_logits = []
    all_labels = []
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else _NullCtx()
    )
    with autocast_ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = classifier(x)
            all_logits.append(logits.detach().float().cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def get_or_run_val_logits(spec: CellSpec, device: torch.device, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate val-set logits cache for a cell.

    Cache lives at runs/final/<dir>/logits/val.npz. Used by confusion
    matrix + calibration scripts when test.npz is not available (L5 cells
    are not in the test-set inference set; they're scored on val).
    """
    cache = spec.run_dir / "logits" / "val.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        return z["logits"], z["labels"]
    classifier = build_classifier(spec, device)
    loader = build_val_loader(spec, val_subset=5000, batch_size=64)
    logits, labels = run_inference(classifier, loader, device, use_bf16=True)
    del classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, logits=logits.astype(np.float32), labels=labels.astype(np.int64))
    return logits, labels


def get_or_run_test_logits(spec: CellSpec, test_indices: list[int], device: torch.device, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate held-out test-split logits cache for a cell."""
    cache = spec.run_dir / "logits" / "test.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        return z["logits"], z["labels"]
    classifier = build_classifier(spec, device)
    loader = build_test_loader(spec, test_indices, batch_size=64)
    logits, labels = run_inference(classifier, loader, device, use_bf16=True)
    del classifier
    if device.type == "cuda":
        torch.cuda.empty_cache()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, logits=logits.astype(np.float32), labels=labels.astype(np.int64))
    return logits, labels


# ---------------------------------------------------------------------------
# Headline cell lists


HEADLINE_BASES = [
    ("B", None),
    ("D", "T3"),
    ("B2", "T3"),
    ("B2nr", None),
]


def headline_l3_bases() -> list[str]:
    """24 L3 headline cells across B1 + D-T3 + B2 + B2-nr × 6 (m, d) pairs."""
    out: list[str] = []
    for phase_prefix in ("final_B_L3_", "final_D_T3_L3_", "final_B2_L3_", "final_B2nr_L3_"):
        for m in ("resnet50", "densenet121", "transnext_tiny"):
            for d in ("cifar10", "mnist"):
                out.append(f"{phase_prefix}{m}_{d}")
    return out


def headline_l5_bases() -> list[str]:
    """L5 cells used for confusion matrices.

    PRD US-045 §(b) lists 24 cells across B1 + D-T3 + B2 + B2-nr, but B2-nr
    is L3-only per US-041 scope lock — no B2-nr L5 cells exist on disk.
    Returns the 18 L5 cells that actually exist: B1 + D-T3 + B2, each
    × 6 (model, dataset) pairs. The PRD inconsistency is documented
    in the US-045 close iter and §VIII outcome block.
    """
    out: list[str] = []
    for phase_prefix in ("final_B_L5_", "final_D_T3_L5_", "final_B2_L5_"):
        for m in ("resnet50", "densenet121", "transnext_tiny"):
            for d in ("cifar10", "mnist"):
                out.append(f"{phase_prefix}{m}_{d}")
    return out
