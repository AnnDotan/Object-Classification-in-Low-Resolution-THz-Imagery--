#!/usr/bin/env python3
"""
Generate base64-encoded sample degradation images for dashboard embedding.
For each unique degradation config found in runs, produces a side-by-side
comparison: original CIFAR-10 image vs degraded version.
"""

import base64
import csv
import io
import json
from pathlib import Path

import torch
from torchvision import datasets, transforms
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.degrade import DegradeConfig, degrade_image


SAMPLE_IDX = 7  # A recognizable CIFAR-10 image (horse)


def get_cifar10_sample(idx: int = SAMPLE_IDX):
    """Get a single CIFAR-10 image as tensor [3,32,32] in [0,1]."""
    ds = datasets.CIFAR10(root="./data", train=False, download=True,
                          transform=transforms.ToTensor())
    img, label = ds[idx]
    class_names = ds.classes
    return img, class_names[label]


def tensor_to_base64(t: torch.Tensor, size: int = 128) -> str:
    """Convert [C,H,W] tensor in [0,1] to base64-encoded PNG string."""
    t = t.clamp(0, 1)
    # Resize for display
    t = t.unsqueeze(0)
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="nearest")
    t = t.squeeze(0)
    # To PIL
    arr = (t.permute(1, 2, 0).numpy() * 255).astype("uint8")
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def get_run_degrade_config(run_dir: str) -> dict:
    """Read run_config.txt and extract degradation parameters."""
    config_path = Path(run_dir) / "run_config.txt"
    data = {}
    if config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                data[k.strip()] = v.strip()
    return data


def make_config_key(low_res, out_size, deg_type):
    """Create a unique key for a degradation configuration."""
    return f"lr{low_res}_out{out_size}_deg{deg_type}"


def generate_sample_images(csv_path: Path) -> dict:
    """
    Generate sample images for each unique degradation config.
    Returns dict: config_key -> {"original_b64": ..., "degraded_b64": ..., "label": ...}
    """
    # Load run summary
    runs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row)

    # Find unique degradation configs
    configs_seen = {}
    for run in runs:
        run_dir = run.get("run_dir", "")
        low_res = run.get("low_res", "16")
        out_size = run.get("out_size", "32")

        # Try to get degradation_type from run_config.txt
        cfg_data = get_run_degrade_config(run_dir)
        deg_type = cfg_data.get("degradation_type", "all")

        if not low_res or not out_size:
            continue

        key = make_config_key(low_res, out_size, deg_type)
        if key not in configs_seen:
            configs_seen[key] = {
                "low_res": int(low_res),
                "out_size": int(out_size),
                "degradation_type": deg_type,
            }

    # Get sample image
    original_img, label = get_cifar10_sample()

    # Generate degraded versions
    results = {}
    display_size = 128
    original_b64 = tensor_to_base64(original_img, size=display_size)

    for key, cfg_params in configs_seen.items():
        deg_cfg = DegradeConfig(
            low_res=cfg_params["low_res"],
            out_size=cfg_params["out_size"],
            degradation_type=cfg_params["degradation_type"],
            p_grayscale=0.0,  # deterministic: no random grayscale
        )
        degraded = degrade_image(original_img.clone(), deg_cfg, seed=42)
        degraded_b64 = tensor_to_base64(degraded, size=display_size)

        results[key] = {
            "original_b64": original_b64,
            "degraded_b64": degraded_b64,
            "label": label,
            "low_res": cfg_params["low_res"],
            "out_size": cfg_params["out_size"],
            "degradation_type": cfg_params["degradation_type"],
        }

    return results


def main():
    csv_path = Path("artifacts/tables/run_summary.csv")
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found")
        return

    print("[GENERATE] Generating sample degradation images...")
    samples = generate_sample_images(csv_path)

    out_path = Path("artifacts/tables/sample_images.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f)

    print(f"[OK] Generated {len(samples)} sample image pairs -> {out_path}")
    for key, info in samples.items():
        print(f"     {key}: low_res={info['low_res']}, out_size={info['out_size']}, deg={info['degradation_type']}")


if __name__ == "__main__":
    main()
