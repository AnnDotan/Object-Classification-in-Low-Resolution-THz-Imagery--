"""Smoke-test the offline PSNR/SSIM measurement script.

Runs the script on a Phase B L3 cifar10 cell with n_samples=8 and asserts:
  - JSON has the required keys
  - PSNR is non-negative, SSIM is in [0, 1] (heavy degradation can suppress
    SSIM but it should not go negative for typical natural images)
  - sample_indices is the deterministic 0..n-1 range
  - degrade_config round-trips fields the campaign relies on
  - Tag and levels-spec parsers agree on the same DegradeConfig

Run: ``python -m src.tests.test_image_quality``
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from src.data.degrade import degrade_config_for
from src.tools.measure_image_quality import (
    main as measure_main,
    parse_cell_tag,
    parse_levels_spec,
)


def _check_smoke_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "image_quality.json"
        rc = measure_main(
            [
                "--cell-tag",
                "final_B_L3_resnet50_cifar10",
                "--n-samples",
                "8",
                "--out",
                str(out),
            ]
        )
        assert rc == 0, f"measure script returned non-zero: {rc}"
        assert out.exists(), "image_quality.json not written"

        data = json.loads(out.read_text(encoding="utf-8"))

    required = {
        "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
        "n_samples", "sample_indices", "degrade_config",
        "cell_tag", "dataset", "level", "axis",
    }
    missing = required - data.keys()
    assert not missing, f"missing JSON keys: {missing}"

    assert data["n_samples"] == 8
    assert data["sample_indices"] == list(range(8)), (
        f"sample_indices not deterministic 0..7: {data['sample_indices']}"
    )
    assert data["dataset"] == "cifar10"
    assert data["level"] == 3
    assert data["axis"] is None  # Phase B is combined, no single axis

    assert data["psnr_mean"] >= 0.0, f"PSNR went negative: {data['psnr_mean']}"
    assert 0.0 <= data["ssim_mean"] <= 1.0, (
        f"SSIM out of [0,1]: {data['ssim_mean']}"
    )

    deg = data["degrade_config"]
    for key in ("low_res", "blur_kernel", "blur_sigma",
                "gaussian_noise_std", "salt_pepper_amount",
                "saturation", "degradation_type"):
        assert key in deg, f"degrade_config missing {key}"

    print(
        f"OK [smoke] — final_B_L3_resnet50_cifar10 n=8 "
        f"PSNR={data['psnr_mean']:.2f}±{data['psnr_std']:.2f} "
        f"SSIM={data['ssim_mean']:.4f}±{data['ssim_std']:.4f}"
    )


def _check_tag_vs_levels_spec_agreement() -> None:
    """Cell-tag parsing and --levels-spec parsing must yield the same config."""
    info = parse_cell_tag("final_C_L4_noise_densenet121_mnist")
    assert info == {
        "phase": "C", "level": 4, "axis": "noise",
        "model": "densenet121", "dataset": "mnist",
    }, info

    level_a, axis_a = parse_levels_spec("L4:noise")
    assert (level_a, axis_a) == (4, "noise")

    cfg_tag = degrade_config_for(info["level"], axis=info["axis"])
    cfg_spec = degrade_config_for(level_a, axis=axis_a)
    assert cfg_tag == cfg_spec, (
        f"tag-derived config differs from levels-spec-derived: {cfg_tag} != {cfg_spec}"
    )

    info_clean = parse_cell_tag("final_clean_resnet50_cifar10")
    assert info_clean["phase"] == "A" and info_clean["level"] is None

    print("OK [parsers] — cell-tag and --levels-spec agree on DegradeConfig.")


def main() -> int:
    _check_tag_vs_levels_spec_agreement()
    _check_smoke_run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
