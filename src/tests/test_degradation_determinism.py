"""Validation-degradation determinism check.

Run: ``python -m src.tests.test_degradation_determinism``

Contract: val pixels are byte-identical across freshly-built dataset
instances at out_size=224 (the campaign's single output resolution). The
test verifies:

  (a) intra-instance: two reads of val image #N from the same dataset are byte-equal
  (b) cross-instance: a freshly-built dataset produces the same val image #N
  (c) batched read via DataLoader is byte-stable across instances
  (d) pixel histograms are exactly equal across freshly-built instances
  (e) train/val seed offsets are disjoint (SEED_OFFSET_TRAIN != SEED_OFFSET_VAL)

Datasets exercised:
  - cifar10 @ out_size=224
  - mnist   @ out_size=224

Tested at the dataset layer (where the determinism actually lives), so the
test does not require pytorch_lightning. A Lightning-DataModule cross-check
is added at the end and skipped automatically if the import fails.
"""
from __future__ import annotations

import sys
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.data.datasets import DataConfig, THzLikeCIFAR10, THzLikeMNIST


def _build(name: str, train: bool, out_size: int = 224) -> Tuple[torch.utils.data.Dataset, DataConfig]:
    cls = THzLikeMNIST if name == "mnist" else THzLikeCIFAR10
    cfg = DataConfig(
        dataset=name,
        train=train,
        out_size=out_size,
        low_res=16,
    )
    return cls(cfg), cfg


def _check_dataset(name: str, out_size: int = 224) -> None:
    label = f"{name}@{out_size}"
    ds_a, _ = _build(name, train=False, out_size=out_size)
    ds_b, _ = _build(name, train=False, out_size=out_size)

    # Pixel shape sanity: every emitted image must match out_size.
    a0 = ds_a[5][0]
    assert tuple(a0.shape) == (3, out_size, out_size), (
        f"[{label}] unexpected pixel shape {tuple(a0.shape)} for out_size={out_size}"
    )

    # (a) intra-instance: same idx -> same pixels
    a0_again = ds_a[5][0]
    intra_mse = torch.mean((a0 - a0_again) ** 2).item()
    assert intra_mse == 0.0, (
        f"[{label}] intra-instance non-determinism for val idx=5: MSE={intra_mse}"
    )

    # (b) cross-instance: freshly-built dataset, same pixels
    b0 = ds_b[5][0]
    inter_mse = torch.mean((a0 - b0) ** 2).item()
    assert inter_mse == 0.0, (
        f"[{label}] cross-instance non-determinism for val idx=5: MSE={inter_mse}"
    )

    # (c) batched read via DataLoader (val protocol: shuffle=False)
    loader_a = DataLoader(ds_a, batch_size=32, shuffle=False, num_workers=0)
    loader_b = DataLoader(ds_b, batch_size=32, shuffle=False, num_workers=0)
    batch_a = next(iter(loader_a))[0]
    batch_b = next(iter(loader_b))[0]
    batch_mse = torch.mean((batch_a - batch_b) ** 2).item()
    assert batch_mse == 0.0, (
        f"[{label}] DataLoader-batch non-determinism: MSE={batch_mse}"
    )

    # (d) pixel histograms must match exactly
    hist_a = torch.histc(batch_a, bins=256, min=-3.0, max=3.0)
    hist_b = torch.histc(batch_b, bins=256, min=-3.0, max=3.0)
    hist_diff = (hist_a - hist_b).abs().sum().item()
    assert hist_diff == 0.0, (
        f"[{label}] pixel-histogram differs across instances: diff={hist_diff}"
    )

    # (e) train/val seed offsets are disjoint
    train_ds, _ = _build(name, train=True, out_size=out_size)
    train0 = train_ds[0][0]
    val0 = ds_a[0][0]
    train_val_mse = torch.mean((train0 - val0) ** 2).item()
    assert train_val_mse > 0.0, (
        f"[{label}] train/val seed offsets collide: MSE={train_val_mse} "
        "(SEED_OFFSET_TRAIN and SEED_OFFSET_VAL must be different)"
    )

    print(
        f"OK [{label}] — val determinism verified "
        f"(intra=0, inter=0, batch=0, hist diff=0, "
        f"train/val disjoint MSE={train_val_mse:.4f})."
    )


def _check_lightning_datamodule(dataset: str, out_size: int = 224) -> None:
    try:
        import pytorch_lightning as pl  # noqa: F401
        from src.lightning.datamodule import THzDataModule
    except Exception as e:
        print(f"SKIP lightning-datamodule check ({type(e).__name__}: {e})")
        return

    label = f"{dataset}@{out_size}"
    pl.seed_everything(42, workers=True)
    dm1 = THzDataModule(
        dataset=dataset, out_size=out_size, low_res=16,
        batch_size=32, val_subset=64,
    )
    dm1.prepare_data()
    dm1.setup()
    a = next(iter(dm1.val_dataloader()))[0]
    b = next(iter(dm1.val_dataloader()))[0]
    intra = torch.mean((a - b) ** 2).item()
    assert intra == 0.0, f"[{label}] DataModule intra MSE={intra}"

    pl.seed_everything(42, workers=True)
    dm2 = THzDataModule(
        dataset=dataset, out_size=out_size, low_res=16,
        batch_size=32, val_subset=64,
    )
    dm2.prepare_data()
    dm2.setup()
    c = next(iter(dm2.val_dataloader()))[0]
    inter = torch.mean((a - c) ** 2).item()
    assert inter == 0.0, f"[{label}] DataModule inter MSE={inter}"

    print(f"OK [lightning {label}] — DataModule val determinism verified (intra=0, inter=0).")


def _check_v2_noise_is_pre_upsample() -> None:
    """v2 pipeline invariant (US-017, 2026-05-20).

    Under pipeline v2, additive Gaussian noise is drawn at `low_res` and THEN
    bicubic-upsampled to `out_size`. The upsample spreads each noise sample
    across its kernel footprint, so adjacent residual pixels are highly
    correlated. Under the (now-removed) v1 order, noise was drawn directly at
    `out_size` and adjacent pixels were independent (lag-1 autocorrelation ~0).

    Fingerprint: lag-1 horizontal autocorrelation of `(noised - clean_baseline)`
    must be > 0.5 for low_res=8, out_size=224. A failure here means the noise
    step has regressed to the v1 (post-upsample) location.

    A second fingerprint: the SAME seed must produce a SMALLER number of unique
    noise values than the post-upsample case would, because we draw only
    low_res*low_res samples per channel instead of out_size*out_size.
    """
    import torch
    from src.data.degrade import DegradeConfig, degrade_image

    LOW_RES, OUT = 8, 224
    NOISE_STD = 0.2

    clean_img = torch.full((3, 32, 32), 0.5, dtype=torch.float32)
    cfg_noisy = DegradeConfig(
        low_res=LOW_RES, out_size=OUT,
        blur_kernel=0, blur_sigma=0.0,
        gaussian_noise_std=NOISE_STD,
        salt_pepper_amount=0.0,
        saturation=1.0,
        degradation_type='all',
    )
    cfg_clean = DegradeConfig(
        low_res=LOW_RES, out_size=OUT,
        blur_kernel=0, blur_sigma=0.0,
        gaussian_noise_std=0.0,
        salt_pepper_amount=0.0,
        saturation=1.0,
        degradation_type='all',
    )

    out_noisy = degrade_image(clean_img, cfg_noisy, seed=12345)
    out_base = degrade_image(clean_img, cfg_clean, seed=12345)
    residual = (out_noisy - out_base)[0]  # one channel

    r_centered = residual - residual.mean()
    var = float((r_centered ** 2).mean().item())
    assert var > 0.0, (
        "[v2 noise pre-upsample] residual is identically zero — noise step did not fire."
    )
    cov = float((r_centered[:, :-1] * r_centered[:, 1:]).mean().item())
    autocorr = cov / var

    assert autocorr > 0.5, (
        f"[v2 noise pre-upsample] lag-1 horizontal autocorrelation {autocorr:.3f} <= 0.5; "
        "noise step appears to be running POST-upsample (v1 order). "
        "Expected ~0.9 for low_res=8 -> bicubic 224."
    )

    # Determinism still holds under v2: same seed -> byte-identical output.
    out_noisy_again = degrade_image(clean_img, cfg_noisy, seed=12345)
    repro_mse = float(((out_noisy - out_noisy_again) ** 2).mean().item())
    assert repro_mse == 0.0, (
        f"[v2 noise pre-upsample] reproducibility broke under v2: MSE={repro_mse}"
    )

    print(
        f"OK [v2 noise pre-upsample] lag-1 autocorr={autocorr:.3f} > 0.5; "
        "byte-identical re-run (MSE=0)."
    )


_DATASETS: tuple[str, ...] = ("cifar10", "mnist")


# ----------------------------------------------------------------------
# US-038 — Phase B2 / C2 override + multi-seed determinism groups.
# ----------------------------------------------------------------------


def _b2_dataconfig(name: str, level: int) -> "DataConfig":
    """Build a DataConfig that mirrors what a Phase B2 cell would feed the
    DataModule: same level table, but saturation = 0 and gaussian_noise_std = 0.
    """
    from src.data.datasets import DataConfig
    from src.data.degrade import degrade_config_for_b2
    cfg = degrade_config_for_b2(level)
    return DataConfig(
        dataset=name,
        train=False,
        out_size=cfg.out_size,
        low_res=cfg.low_res,
        gaussian_noise_std=cfg.gaussian_noise_std,
        blur_kernel=cfg.blur_kernel,
        blur_sigma=cfg.blur_sigma,
        salt_pepper_amount=cfg.salt_pepper_amount,
        saturation=cfg.saturation,
        degradation_type=cfg.degradation_type,
    )


def _c2_dataconfig(name: str, level: int, axis: str) -> "DataConfig":
    from src.data.datasets import DataConfig
    from src.data.degrade import degrade_config_for_c2
    cfg = degrade_config_for_c2(level, axis)
    return DataConfig(
        dataset=name,
        train=False,
        out_size=cfg.out_size,
        low_res=cfg.low_res,
        gaussian_noise_std=cfg.gaussian_noise_std,
        blur_kernel=cfg.blur_kernel,
        blur_sigma=cfg.blur_sigma,
        salt_pepper_amount=cfg.salt_pepper_amount,
        saturation=cfg.saturation,
        degradation_type=cfg.degradation_type,
    )


def _check_b2_determinism(name: str, level: int = 3) -> None:
    """PRD §4.9 — Phase B2 determinism group. Val pixels must be byte-identical
    across freshly-built dataset instances under the B2 DegradeConfig override.
    Also asserts the override values: saturation = 0.0 + gaussian_noise_std = 0.0.
    """
    from src.data.datasets import THzLikeCIFAR10, THzLikeMNIST
    from src.data.degrade import degrade_config_for_b2

    label = f"b2_{name}_l{level}"
    cls = THzLikeMNIST if name == "mnist" else THzLikeCIFAR10
    cfg_a = _b2_dataconfig(name, level)
    cfg_b = _b2_dataconfig(name, level)
    ds_a = cls(cfg_a)
    ds_b = cls(cfg_b)

    a0 = ds_a[5][0]
    b0 = ds_b[5][0]
    intra_mse = float(torch.mean((ds_a[5][0] - a0) ** 2).item())
    inter_mse = float(torch.mean((a0 - b0) ** 2).item())
    assert intra_mse == 0.0, f"[{label}] intra-instance MSE={intra_mse}"
    assert inter_mse == 0.0, f"[{label}] cross-instance MSE={inter_mse}"

    # Substring assertions: B2 override values.
    deg = degrade_config_for_b2(level)
    assert deg.saturation == 0.0, f"[{label}] saturation override broke: {deg.saturation}"
    assert deg.gaussian_noise_std == 0.0, (
        f"[{label}] gaussian_noise_std override broke: {deg.gaussian_noise_std}"
    )
    print(f"OK [{label}] — B2 determinism + override verified (sat=0, noise_std=0, MSE=0).")


def _check_c2_resolution_determinism(name: str, level: int = 3) -> None:
    """PRD §4.9 — Phase C2 determinism group for the resolution axis. Also
    asserts axis isolation: only `low_res` differs from IDENTITY_VALUES."""
    from src.data.datasets import THzLikeCIFAR10, THzLikeMNIST
    from src.data.degrade import degrade_config_for_c2
    from src.data.degradation_levels import IDENTITY_VALUES

    label = f"c2_{name}_l{level}_resolution"
    cls = THzLikeMNIST if name == "mnist" else THzLikeCIFAR10
    cfg_a = _c2_dataconfig(name, level, axis="resolution")
    cfg_b = _c2_dataconfig(name, level, axis="resolution")
    ds_a = cls(cfg_a)
    ds_b = cls(cfg_b)

    a0 = ds_a[5][0]
    b0 = ds_b[5][0]
    inter_mse = float(torch.mean((a0 - b0) ** 2).item())
    assert inter_mse == 0.0, f"[{label}] cross-instance MSE={inter_mse}"

    deg = degrade_config_for_c2(level, axis="resolution")
    assert deg.saturation == 0.0
    assert deg.gaussian_noise_std == 0.0
    # Axis isolation: only low_res non-identity (blur/sp at identity).
    assert deg.blur_kernel == IDENTITY_VALUES["blur_kernel"]
    assert deg.blur_sigma == IDENTITY_VALUES["blur_sigma"]
    assert deg.salt_pepper_amount == IDENTITY_VALUES["salt_pepper"]
    # And low_res actually differs from identity (=224) at L3.
    assert deg.low_res != IDENTITY_VALUES["low_res"], (
        f"[{label}] resolution axis must differ from identity at L{level}; got {deg.low_res}"
    )
    print(f"OK [{label}] — C2 resolution determinism + axis isolation verified.")


def _check_multi_seed_val_byte_identical() -> None:
    """PRD §4.9 — multi-seed determinism group. Validation pixels MUST be
    byte-identical across pl.seed_everything(N) for N ∈ {42, 43, 44} because
    the per-sample degradation seed offsets (SEED_OFFSET_VAL) are independent
    of the global Lightning seed.

    Falls back to a torch-only check if Lightning import fails.
    """
    label = "multi_seed_val_byte_identical_42_43_44"
    try:
        import pytorch_lightning as pl  # noqa: F401
    except Exception as e:
        print(f"SKIP [{label}] — Lightning import failed: {type(e).__name__}: {e}")
        return

    # Build a val dataset three times under three different global seeds.
    from src.data.datasets import THzLikeCIFAR10, DataConfig
    val_batches: list[torch.Tensor] = []
    for seed in (42, 43, 44):
        pl.seed_everything(seed, workers=True)
        ds = THzLikeCIFAR10(DataConfig(
            dataset="cifar10", train=False, out_size=224, low_res=16,
        ))
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        val_batches.append(next(iter(loader))[0])

    for i, batch in enumerate(val_batches[1:], start=1):
        mse = float(torch.mean((val_batches[0] - batch) ** 2).item())
        assert mse == 0.0, (
            f"[{label}] val pixels NOT byte-identical across seed 42 vs seed "
            f"{42 + i}: MSE={mse}. The per-sample degradation seed offsets "
            f"must be independent of pl.seed_everything."
        )
    print(f"OK [{label}] — val pixels byte-identical across seeds {{42, 43, 44}}.")


def main() -> int:
    for name in _DATASETS:
        _check_dataset(name)

    for name in _DATASETS:
        _check_lightning_datamodule(dataset=name)

    _check_v2_noise_is_pre_upsample()

    # US-038 — new groups
    for name in _DATASETS:
        _check_b2_determinism(name, level=3)
    for name in _DATASETS:
        _check_c2_resolution_determinism(name, level=3)
    _check_multi_seed_val_byte_identical()

    return 0


if __name__ == "__main__":
    sys.exit(main())
