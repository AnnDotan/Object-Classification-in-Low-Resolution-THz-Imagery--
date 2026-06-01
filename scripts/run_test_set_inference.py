"""US-045 (a) — Held-out test-set inference for 24 L3 headline cells x 3 seeds.

Inputs:
  - artifacts/validation/test_split_indices.json (US-045 pre-flight, seed=99)
  - runs/final/<base_tag>{,_seed43,_seed44}/best.pt + metrics.json

Outputs:
  - artifacts/validation/test_set_results.json — 72 entries with
      {tag, seed, run_dir, val_acc, test_acc, gap, n_test_samples}
    + summary: n_cells, n_gap_over_2pp, max_gap, max_gap_cell.
  - runs/final/<dir>/logits/test.npz — (logits, labels) tensor cache for
    re-use in confusion + calibration scripts.

Idempotency:
  - If --skip-existing, cells with a test.npz already are skipped.
  - The cache file is overwritten cleanly otherwise.

Usage:
  python scripts/run_test_set_inference.py [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.diagnostics_common import (
    build_classifier,
    build_test_loader,
    headline_l3_bases,
    load_cell_spec,
    run_inference,
)

OUT_JSON = ROOT / "artifacts" / "validation" / "test_set_results.json"
TEST_SPLIT = ROOT / "artifacts" / "validation" / "test_split_indices.json"
SEEDS = (42, 43, 44)


def resolve_run_dir(base_tag: str, seed: int) -> Path:
    if seed == 42:
        return ROOT / "runs" / "final" / base_tag
    return ROOT / "runs" / "final" / f"{base_tag}_seed{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true", help="Skip cells whose logits/test.npz already exists.")
    args = ap.parse_args()

    test_split = json.loads(TEST_SPLIT.read_text())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run_test_set_inference] device={dev}")

    bases = headline_l3_bases()
    assert len(bases) == 24, f"expected 24 L3 bases, got {len(bases)}"

    entries = []
    total_t0 = time.time()
    n_total = 0
    n_skipped = 0
    for base in bases:
        for seed in SEEDS:
            n_total += 1
            run_dir = resolve_run_dir(base, seed)
            if not (run_dir / "metrics.json").exists():
                print(f"[skip] {run_dir.name} — no metrics.json")
                continue
            cache = run_dir / "logits" / "test.npz"
            if args.skip_existing and cache.exists():
                # Reuse cached logits + cell spec for accuracy compute
                cached = np.load(cache)
                logits, labels = cached["logits"], cached["labels"]
                spec = load_cell_spec(run_dir)
                test_acc = float((logits.argmax(axis=1) == labels).mean())
                gap = test_acc - spec.best_val_acc
                print(f"[cached] {run_dir.name}: val={spec.best_val_acc*100:.2f}% test={test_acc*100:.2f}% gap={gap*100:+.2f}pp")
                n_skipped += 1
            else:
                spec = load_cell_spec(run_dir)
                idxs = test_split["datasets"][spec.dataset]["indices"]
                t0 = time.time()
                classifier = build_classifier(spec, dev)
                loader = build_test_loader(spec, idxs, batch_size=64)
                logits, labels = run_inference(classifier, loader, dev, use_bf16=True)
                # Free model + cache before next cell
                del classifier
                torch.cuda.empty_cache() if dev.type == "cuda" else None

                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, logits=logits.astype(np.float32), labels=labels.astype(np.int64))
                test_acc = float((logits.argmax(axis=1) == labels).mean())
                gap = test_acc - spec.best_val_acc
                elapsed = time.time() - t0
                print(f"[done] {run_dir.name}: val={spec.best_val_acc*100:.2f}% test={test_acc*100:.2f}% gap={gap*100:+.2f}pp ({elapsed:.0f}s, n={len(labels)})")

            entries.append({
                "tag": base,
                "seed": int(spec.seed),
                "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
                "model": spec.model_name,
                "dataset": spec.dataset,
                "phase": spec.phase,
                "treatment": spec.treatment,
                "level": spec.level,
                "val_acc": spec.best_val_acc,
                "test_acc": test_acc,
                "gap_pp": (test_acc - spec.best_val_acc) * 100.0,
                "n_test_samples": int(len(labels)),
            })

    # Summary
    if entries:
        gaps = [abs(e["gap_pp"]) for e in entries]
        max_gap = max(gaps)
        max_gap_idx = gaps.index(max_gap)
        n_over_2pp = sum(1 for g in gaps if g > 2.0)
    else:
        max_gap = 0.0
        max_gap_idx = -1
        n_over_2pp = 0

    summary = {
        "n_cells": len(entries),
        "n_dispatched": n_total,
        "n_cached_skip": n_skipped,
        "n_gap_over_2pp": n_over_2pp,
        "max_abs_gap_pp": round(max_gap, 4),
        "max_abs_gap_cell": entries[max_gap_idx]["run_dir"] if entries else None,
        "elapsed_s": round(time.time() - total_t0, 1),
        "test_split_indices_path": "artifacts/validation/test_split_indices.json",
        "test_split_seed": int(test_split["seed"]),
        "test_split_per_class": int(test_split["per_class"]),
    }

    payload = {"summary": summary, "entries": entries}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\n[run_test_set_inference] wrote {OUT_JSON}")
    print(f"  n_cells={len(entries)}  n_gap_over_2pp={n_over_2pp}  max_abs_gap={max_gap:.2f}pp  elapsed={summary['elapsed_s']}s")


if __name__ == "__main__":
    main()
