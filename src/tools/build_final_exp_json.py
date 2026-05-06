"""Aggregate per-run metrics into `artifacts/Final_Exp.json` (FINAL_EXP Dashboard, US-002).

Reads `build_final_matrix()` for plan enumeration (186 cells) and
`runs/final/<tag>/metrics.json` for completed-run hydration. Emits a single
JSON document conforming to `src.tools.final_exp_schema.FinalExpDoc`.

The browser-side dashboard (`artifacts/Final_Exp.html`, US-004..) consumes
this document via `fetch` and 30 s polling; it is the only data contract
between the Python build pipeline and the browser. Atomic writes
(`tmp + os.replace`) ensure the browser never reads a half-written file
mid-poll.

Privacy: never opens `*.ckpt`, `*.pt`, `*.pth`, `runs/**/*.ckpt`, or
`artifacts/weights/*`. Reads only `metrics.json` via
`src.experiments.run_status.read_metrics`.

CLI:
    python -m src.tools.build_final_exp_json
    python -m src.tools.build_final_exp_json --out artifacts/Final_Exp.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.data.degradation_levels import level_params
from src.experiments.cells import EXPECTED_TOTAL, CellMeta, iter_cells
from src.experiments.run_status import detect_status, read_metrics
from src.tools.final_exp_schema import (
    SCHEMA_VERSION,
    CountsDict,
    FinalExpDoc,
    FinalExpRow,
    ParamsDict,
)

_DEFAULT_OUT = Path("artifacts/Final_Exp.json")
_RUNS_ROOT = Path("runs/final")

# Sentinel "no-op" params for Phase A (clean, no degradation step). Each value
# is the identity for its operation: low_res=224 (= model input, no downsample),
# blur kernel 1x1, σ=0, no noise, no S&P, full saturation. The renderer
# suppresses params display when level is None — these values exist only to
# satisfy the FinalExpRow.params TypedDict contract.
_CLEAN_PARAMS: ParamsDict = {
    "low_res": 224,
    "blur_kernel": 1,
    "blur_sigma": 0.0,
    "noise_std": 0.0,
    "salt_pepper": 0.0,
    "saturation": 1.0,
}


def _params_for(meta: CellMeta) -> ParamsDict:
    if meta.level is None:
        return dict(_CLEAN_PARAMS)  # type: ignore[return-value]
    raw = level_params(meta.level, axis=meta.axis)
    return {
        "low_res": int(raw["low_res"]),
        "blur_kernel": int(raw["blur_kernel"]),
        "blur_sigma": float(raw["blur_sigma"]),
        "noise_std": float(raw["noise_std"]),
        "salt_pepper": float(raw["salt_pepper"]),
        "saturation": float(raw["saturation"]),
    }


def _row_for(meta: CellMeta, runs_root: Path) -> FinalExpRow:
    metrics = read_metrics(meta.tag, runs_root)
    status = detect_status(meta.tag, runs_root, metrics)
    val_acc: Optional[float] = None
    val_loss: Optional[float] = None
    epochs_run: Optional[int] = None
    runtime_s: Optional[float] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    if metrics is not None:
        for key in ("best_val_acc", "final_val_acc", "last_val_acc"):
            v = metrics.get(key)
            if isinstance(v, (int, float)) and v >= 0.0:
                val_acc = float(v)
                break
        v = metrics.get("last_val_loss")
        val_loss = float(v) if isinstance(v, (int, float)) else None
        v = metrics.get("epochs_run")
        epochs_run = int(v) if isinstance(v, (int, float)) else None
        v = metrics.get("runtime_s")
        runtime_s = float(v) if isinstance(v, (int, float)) else None
        v = metrics.get("started_at")
        started_at = str(v) if isinstance(v, str) else None
        v = metrics.get("finished_at")
        finished_at = str(v) if isinstance(v, str) else None
    return {
        "tag": meta.tag,
        "phase": meta.phase,  # type: ignore[typeddict-item]
        "model": meta.model,
        "dataset": meta.dataset,
        "level": meta.level,
        "axis": meta.axis,
        "params": _params_for(meta),
        "status": status,  # type: ignore[typeddict-item]
        "val_acc": val_acc,
        "val_loss": val_loss,
        "epochs_run": epochs_run,
        "runtime_s": runtime_s,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def build_doc(runs_root: Path = _RUNS_ROOT) -> FinalExpDoc:
    """Build the FinalExpDoc by enumerating the 186 cells and hydrating completed runs."""
    rows: list[FinalExpRow] = [_row_for(meta, runs_root) for meta in iter_cells()]
    rows.sort(key=lambda r: r["tag"])
    assert len(rows) == EXPECTED_TOTAL, f"expected {EXPECTED_TOTAL} rows, got {len(rows)}"
    counts: CountsDict = {
        "total": len(rows),
        "pending": sum(1 for r in rows if r["status"] == "Pending"),
        "running": sum(1 for r in rows if r["status"] == "Running"),
        "complete": sum(1 for r in rows if r["status"] == "Complete"),
        "failed": sum(1 for r in rows if r["status"] == "Failed"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
        "counts": counts,
    }


def write_doc_atomic(doc: FinalExpDoc, out_path: Path) -> None:
    """Write the doc to `out_path` via tmp + os.replace so partial files never reach readers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(out_path.parent),
        prefix=out_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tf:
        tmp_path = Path(tf.name)
        json.dump(doc, tf, indent=2, sort_keys=False, ensure_ascii=False)
        tf.write("\n")
    os.replace(tmp_path, out_path)


def build_final_exp_json(
    out_path: Path = _DEFAULT_OUT,
    runs_root: Path = _RUNS_ROOT,
) -> dict:
    """Build the JSON aggregate and write it atomically. Returns a small summary dict."""
    doc = build_doc(runs_root=runs_root)
    write_doc_atomic(doc, out_path)
    return {
        "rows": doc["counts"]["total"],
        "counts": dict(doc["counts"]),
        "out": str(out_path),
        "generated_at": doc["generated_at"],
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate runs/final/<tag>/metrics.json into artifacts/Final_Exp.json.")
    p.add_argument("--out", default=str(_DEFAULT_OUT), help=f"Output JSON path (default: {_DEFAULT_OUT}).")
    p.add_argument("--runs-root", default=str(_RUNS_ROOT), help=f"Per-cell metrics root (default: {_RUNS_ROOT}).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    summary = build_final_exp_json(out_path=Path(args.out), runs_root=Path(args.runs_root))
    c = summary["counts"]
    print(
        f"final_exp_json: {summary['rows']} rows "
        f"(pending={c['pending']}, running={c['running']}, complete={c['complete']}, failed={c['failed']}) "
        f"-> {summary['out']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
