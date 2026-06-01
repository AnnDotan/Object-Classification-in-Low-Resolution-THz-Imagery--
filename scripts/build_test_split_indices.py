"""US-045 pre-flight — Build deterministic stratified held-out test split.

Constraint (PRD v4 §7):
  - For each dataset (cifar10, mnist), sample 1000 indices per class from
    the official test partition (train=False) using seed=99 stratified
    sampling. Where a class has fewer than 1000 samples in the test
    partition, take all of them. Indices are 0-based into the *official*
    torchvision test partition for the dataset.
  - The output JSON is the authoritative source for US-045's test-set
    inference pass. Idempotent (re-running produces a byte-identical file).

Output: artifacts/validation/test_split_indices.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from torchvision import datasets

SEED = 99
PER_CLASS = 1000
ROOT = Path("./data")
OUT = Path("artifacts/validation/test_split_indices.json")


def _labels_for(name: str) -> list[int]:
    if name == "cifar10":
        ds = datasets.CIFAR10(root=str(ROOT), train=False, download=True)
        return list(ds.targets)
    if name == "mnist":
        ds = datasets.MNIST(root=str(ROOT), train=False, download=True)
        return ds.targets.tolist()
    raise ValueError(name)


def stratified_split(labels: list[int], seed: int, per_class: int) -> list[int]:
    """Stratified deterministic sample. Within each class, sort the official
    indices then permute via numpy.random.RandomState(seed) and take the
    first per_class (or all if fewer).
    """
    by_cls: dict[int, list[int]] = defaultdict(list)
    for idx, y in enumerate(labels):
        by_cls[int(y)].append(idx)
    rng = np.random.RandomState(seed)
    picked: list[int] = []
    for cls in sorted(by_cls):
        pool = sorted(by_cls[cls])
        if len(pool) <= per_class:
            picked.extend(pool)
        else:
            perm = rng.permutation(len(pool))[:per_class]
            picked.extend(sorted(pool[i] for i in perm))
    return sorted(picked)


def main() -> None:
    out = {
        "seed": SEED,
        "per_class": PER_CLASS,
        "datasets": {},
    }
    for name in ("cifar10", "mnist"):
        labels = _labels_for(name)
        idxs = stratified_split(labels, SEED, PER_CLASS)
        per_cls_counts = defaultdict(int)
        for i in idxs:
            per_cls_counts[int(labels[i])] += 1
        out["datasets"][name] = {
            "n_total_test": len(labels),
            "n_selected": len(idxs),
            "per_class_counts": {str(k): v for k, v in sorted(per_cls_counts.items())},
            "indices": idxs,
        }
        print(f"[test_split] {name}: {len(idxs)}/{len(labels)} selected; per-class={dict(per_cls_counts)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"[test_split] wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
