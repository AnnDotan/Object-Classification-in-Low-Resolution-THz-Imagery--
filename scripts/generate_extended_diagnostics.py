"""Extended diagnostics — confusion + calibration PNGs for the phases US-045 skipped.

US-045 rendered confusion (L5) and calibration (L3+L5) diagrams only for the
headline phases (B1, D-T3, B2, B2-nr). The dashboard galleries now cover every
phase, so this script fills in the remaining cells:

  - Phase C   (5 axes)  : final_C_L{3,5}_{axis}_{m}_{d}   — 30 per level
  - Phase C2  (3 axes)  : final_C2_L{3,5}_{axis}_{m}_{d}  — 18 per level
  - Phase D T1/T2       : final_D_T{1,2}_L{3,5}_{m}_{d}   — 12 per level

Same conventions as US-045: confusion at L5 from val-set logits; calibration
at L3 from the held-out test split (seed=99), at L5 from val logits. Logit
caches land in runs/final/<tag>/logits/{val,test}.npz exactly like the
US-045 scripts, so re-runs are cheap.

Deliberately does NOT touch docs/_autogen/calibration_table.tex or the
US-045 _index.json files (those are report artifacts scoped to the headline
cells). Extended indexes are written to *_index_extended.json instead.

Usage:
  python scripts/generate_extended_diagnostics.py [--skip-existing]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.diagnostics_common import (
    get_or_run_test_logits,
    get_or_run_val_logits,
    load_cell_spec,
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the exact render/measure functions so extended PNGs are visually
# identical to the US-045 ones.
_conf = _load_script("generate_confusion_matrices")
_cal = _load_script("generate_calibration_diagrams")

CONF_DIR = ROOT / "artifacts" / "figures" / "confusion"
CAL_DIR = ROOT / "artifacts" / "figures" / "calibration"
TEST_SPLIT = ROOT / "artifacts" / "validation" / "test_split_indices.json"

MODELS = ("resnet50", "densenet121", "transnext_tiny")
DATASETS = ("cifar10", "mnist")
C_AXES = ("resolution", "noise", "blur", "saturation", "salt_pepper")
C2_AXES = ("resolution", "blur", "salt_pepper")


def extended_bases(level: int) -> list[str]:
    out: list[str] = []
    for ax in C_AXES:
        for m in MODELS:
            for d in DATASETS:
                out.append(f"final_C_L{level}_{ax}_{m}_{d}")
    for ax in C2_AXES:
        for m in MODELS:
            for d in DATASETS:
                out.append(f"final_C2_L{level}_{ax}_{m}_{d}")
    for t in ("T1", "T2"):
        for m in MODELS:
            for d in DATASETS:
                out.append(f"final_D_{t}_L{level}_{m}_{d}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extended-diag] device={dev}")
    test_split = json.loads(TEST_SPLIT.read_text())

    CONF_DIR.mkdir(parents=True, exist_ok=True)
    CAL_DIR.mkdir(parents=True, exist_ok=True)

    conf_entries: list[dict] = []
    cal_entries: list[dict] = []
    t0 = time.time()

    l3_bases = extended_bases(3)
    l5_bases = extended_bases(5)
    n_total = len(l3_bases) + len(l5_bases)
    print(f"[extended-diag] {len(l3_bases)} L3 cells (calibration/test-split) + "
          f"{len(l5_bases)} L5 cells (confusion + calibration/val)")

    done = 0

    # ---- L5: confusion + calibration from val logits (one inference each) --
    for base in l5_bases:
        done += 1
        run_dir = ROOT / "runs" / "final" / base
        if not (run_dir / "metrics.json").exists():
            print(f"[skip] {base} — no metrics.json")
            continue
        spec = load_cell_spec(run_dir)
        conf_png = CONF_DIR / f"{base}_L5.png"
        cal_png = CAL_DIR / f"{base}_L5.png"
        if args.skip_existing and conf_png.exists() and cal_png.exists():
            print(f"[cached-png] {base} ({done}/{n_total})")
            continue
        t_cell = time.time()
        logits, labels = get_or_run_val_logits(spec, dev, force=False)

        if not (args.skip_existing and conf_png.exists()):
            cm = _conf._confusion(logits, labels, n_classes=10)
            classes = _conf.MNIST_CLASSES if spec.dataset == "mnist" else _conf.CIFAR_CLASSES
            diag_acc = float(np.trace(cm) / cm.sum())
            title = f"{base} — {spec.model_name} / {spec.dataset} L5  (n={int(cm.sum())}, acc={diag_acc*100:.2f}%)"
            _conf._render(cm, classes, title, conf_png)
            recalls = (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).tolist()
            cm2 = cm.copy()
            np.fill_diagonal(cm2, 0)
            ridx, cidx = np.unravel_index(np.argmax(cm2), cm2.shape)
            conf_entries.append({
                "tag": base, "phase": spec.phase, "treatment": spec.treatment,
                "axis": spec.axis, "model": spec.model_name, "dataset": spec.dataset,
                "level": 5, "val_acc_diag": diag_acc, "per_class_recall": recalls,
                "most_confused": {"true": classes[int(ridx)], "predicted": classes[int(cidx)],
                                  "count": int(cm2[ridx, cidx])},
                "png": str(conf_png.relative_to(ROOT)).replace("\\", "/"),
            })

        if not (args.skip_existing and cal_png.exists()):
            z = logits.astype(np.float64)
            z -= z.max(axis=1, keepdims=True)
            e = np.exp(z)
            probs = e / e.sum(axis=1, keepdims=True)
            ece, bcent, bconf, bacc, bcnt = _cal._ece_and_bins(probs, labels, n_bins=_cal.N_BINS)
            title = f"{base} — {spec.model_name} / {spec.dataset} L5  (val split, n={int(len(labels))})"
            _cal._render(title, ece, bcent, bacc, bconf, bcnt, cal_png)
            cal_entries.append({
                "tag": base, "phase": spec.phase, "treatment": spec.treatment,
                "axis": spec.axis, "model": spec.model_name, "dataset": spec.dataset,
                "level": 5, "split": "val", "ece_pp": ece * 100.0,
                "png": str(cal_png.relative_to(ROOT)).replace("\\", "/"),
            })
        print(f"[done] {base} L5 ({done}/{n_total}, {time.time()-t_cell:.0f}s)")

    # ---- L3: calibration from held-out test-split logits -------------------
    for base in l3_bases:
        done += 1
        run_dir = ROOT / "runs" / "final" / base
        if not (run_dir / "metrics.json").exists():
            print(f"[skip] {base} — no metrics.json")
            continue
        spec = load_cell_spec(run_dir)
        cal_png = CAL_DIR / f"{base}_L3.png"
        if args.skip_existing and cal_png.exists():
            print(f"[cached-png] {base} ({done}/{n_total})")
            continue
        t_cell = time.time()
        idxs = test_split["datasets"][spec.dataset]["indices"]
        logits, labels = get_or_run_test_logits(spec, idxs, dev, force=False)
        z = logits.astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=1, keepdims=True)
        ece, bcent, bconf, bacc, bcnt = _cal._ece_and_bins(probs, labels, n_bins=_cal.N_BINS)
        title = f"{base} — {spec.model_name} / {spec.dataset} L3  (test split, n={int(len(labels))})"
        _cal._render(title, ece, bcent, bacc, bconf, bcnt, cal_png)
        cal_entries.append({
            "tag": base, "phase": spec.phase, "treatment": spec.treatment,
            "axis": spec.axis, "model": spec.model_name, "dataset": spec.dataset,
            "level": 3, "split": "test", "ece_pp": ece * 100.0,
            "png": str(cal_png.relative_to(ROOT)).replace("\\", "/"),
        })
        print(f"[done] {base} L3 ({done}/{n_total}, {time.time()-t_cell:.0f}s)")

    elapsed = round(time.time() - t0, 1)
    (CONF_DIR / "_index_extended.json").write_text(json.dumps(
        {"summary": {"n_rendered": len(conf_entries), "elapsed_s": elapsed}, "entries": conf_entries},
        indent=2, sort_keys=True))
    (CAL_DIR / "_index_extended.json").write_text(json.dumps(
        {"summary": {"n_rendered": len(cal_entries), "elapsed_s": elapsed}, "entries": cal_entries},
        indent=2, sort_keys=True))
    print(f"\n[extended-diag] confusion +{len(conf_entries)} PNGs, calibration +{len(cal_entries)} PNGs, elapsed={elapsed}s")


if __name__ == "__main__":
    main()
