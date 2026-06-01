"""US-045 (d) — Per-image inference throughput on RTX 5070 at bf16-mixed.

For each of the 3 model families:
  - Build a freshly-initialized model at the canonical 224x224 input size
    (matches the THzClassifier construction used during training)
  - Run 100 warmup forward passes at batch_size=1
  - Run 1000 measurement forward passes at batch_size=1
  - Record mean ± std (ms / image) and 50/95/99 percentile latencies

Outputs:
  - artifacts/figures/inference_throughput.csv
  - artifacts/figures/inference_throughput.png  (horizontal bar, mean ± std)
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lightning.module import THzClassifier

OUT_CSV = ROOT / "artifacts" / "figures" / "inference_throughput.csv"
OUT_PNG = ROOT / "artifacts" / "figures" / "inference_throughput.png"
WARMUP_BATCHES = 100
MEASURE_BATCHES = 1000
BATCH_SIZE = 1
INPUT_HW = 224
MODELS = ("resnet50", "densenet121", "transnext_tiny")


def _build(model_name: str, device: torch.device) -> THzClassifier:
    is_tn = model_name.startswith("transnext_")
    classifier = THzClassifier(
        model_name=model_name,
        num_classes=10,
        pretrained=is_tn,
        lr=1e-3,
        backbone_lr=1e-4,
        weight_decay=1e-4,
        label_smoothing=0.0,
        drop_path_rate=0.0,
        dropout=0.0,
        pretrain_size=224 if is_tn else None,
    )
    classifier.eval()
    classifier.to(device)
    return classifier


@torch.no_grad()
def _measure(classifier: THzClassifier, device: torch.device) -> dict:
    x = torch.randn(BATCH_SIZE, 3, INPUT_HW, INPUT_HW, device=device)
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    )
    # Warmup
    with autocast_ctx:
        for _ in range(WARMUP_BATCHES):
            classifier(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Measurement
        latencies_ms = np.zeros(MEASURE_BATCHES, dtype=np.float64)
        for i in range(MEASURE_BATCHES):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                classifier(x)
                end.record()
                torch.cuda.synchronize()
                latencies_ms[i] = start.elapsed_time(end)
            else:
                t0 = time.perf_counter()
                classifier(x)
                latencies_ms[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "mean_ms": float(latencies_ms.mean()),
        "std_ms": float(latencies_ms.std()),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "min_ms": float(latencies_ms.min()),
        "max_ms": float(latencies_ms.max()),
        "throughput_images_per_s": float(1000.0 / latencies_ms.mean()),
    }


def _render(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=140)
    names = [r["model"] for r in rows]
    means = [r["mean_ms"] for r in rows]
    stds = [r["std_ms"] for r in rows]
    y = np.arange(len(names))
    ax.barh(y, means, xerr=stds, color=["#4c72b0", "#55a868", "#c44e52"], edgecolor="black", linewidth=0.5, capsize=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Latency per image (ms, batch_size=1, bf16-mixed)")
    ax.set_title(f"RTX 5070 inference throughput @ {INPUT_HW}x{INPUT_HW}\n(warmup={WARMUP_BATCHES}, measure={MEASURE_BATCHES} forward passes)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    for i, (m, s, r) in enumerate(zip(means, stds, rows)):
        ax.text(m + s + max(means) * 0.02, i, f"{m:.2f} ± {s:.2f} ms  |  {r['throughput_images_per_s']:.0f} img/s",
                va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[throughput] device={dev}  warmup={WARMUP_BATCHES}  measure={MEASURE_BATCHES}  input={INPUT_HW}")
    if dev.type == "cuda":
        cap = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
        print(f"[throughput] GPU: {name}  cap={cap}")
    rows = []
    t0 = time.time()
    for model_name in MODELS:
        print(f"[throughput] measuring {model_name}...")
        classifier = _build(model_name, dev)
        result = _measure(classifier, dev)
        del classifier
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        result["model"] = model_name
        rows.append(result)
        print(f"  mean={result['mean_ms']:.2f}±{result['std_ms']:.2f} ms  p50={result['p50_ms']:.2f}  p95={result['p95_ms']:.2f}  p99={result['p99_ms']:.2f}  throughput={result['throughput_images_per_s']:.0f} img/s")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "throughput_images_per_s"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"[throughput] wrote {OUT_CSV}")

    _render(rows)
    print(f"[throughput] wrote {OUT_PNG}; elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
