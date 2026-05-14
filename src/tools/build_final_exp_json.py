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
from src.experiments.run_status import (
    QUARANTINE_REASON,
    detect_status,
    is_quarantined,
    read_image_quality,
    read_metrics,
)
from src.tools.final_exp_schema import (
    SCHEMA_VERSION,
    CountsDict,
    FinalExpDoc,
    FinalExpRow,
    ParamsDict,
)

_DEFAULT_OUT = Path("artifacts/Final_Exp.json")
_RUNS_ROOT = Path("runs/final")
# US-017 Visual Core: directory of pre-rendered side-by-side Original|Degraded
# PNGs (one per cell tag). The aggregator only checks for PNG presence — it
# never opens the file — so this path lookup is byte-stream-free.
_VISUAL_CORE_DIR = Path("artifacts/dashboard_thumbs")
# Path the BROWSER uses to load the PNG. The dashboard sits at
# `artifacts/Final_Exp.html`, so the relative path is `dashboard_thumbs/<tag>.png`.
_VISUAL_CORE_REL = "dashboard_thumbs"

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


def _has_history(tag: str, runs_root: Path) -> bool:
    """US-018: True iff `runs/final/<tag>/history.json` exists. Presence-only
    check — never opens the file (which keeps the lazy-fetch contract intact
    and avoids parsing on every aggregator pass)."""
    return (runs_root / tag / "history.json").exists()


def _visual_core_for(tag: str, visual_dir: Optional[Path] = None) -> Optional[str]:
    """Return the dashboard-relative PNG path if it exists on disk, else None.

    Privacy-by-design: we *only* call `Path.exists()` — never `open()` — so
    the aggregator cannot accidentally read pixels or weight blobs while
    populating the schema field.

    `visual_dir` defaults to the module-level `_VISUAL_CORE_DIR` *resolved
    at call time*, so tests can monkeypatch the module attribute.
    """
    base = visual_dir if visual_dir is not None else _VISUAL_CORE_DIR
    png = base / f"{tag}.png"
    if png.exists():
        return f"{_VISUAL_CORE_REL}/{tag}.png"
    return None


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

    # US-014: quarantine override — TransNeXt rows are Deferred regardless
    # of any leftover on-disk artifacts. The override is intentionally
    # last so it cannot be defeated by a stale Running/Complete dir.
    quarantined = is_quarantined(meta.model)
    quarantine_reason: Optional[str] = QUARANTINE_REASON if quarantined else None
    if quarantined:
        status = "Deferred"

    # PSNR/SSIM (US-002): sourced from runs/final/<tag>/image_quality.json so
    # the dashboard can render them under the Visual Core thumbnail. Null on
    # Phase A clean cells (the measurement step is skipped — clean-vs-clean
    # is identity) and on any cell whose run dir hasn't been measured yet.
    iq = read_image_quality(meta.tag, runs_root)
    psnr_mean: Optional[float] = None
    psnr_std: Optional[float] = None
    ssim_mean: Optional[float] = None
    ssim_std: Optional[float] = None
    if iq is not None:
        v = iq.get("psnr_mean")
        psnr_mean = float(v) if isinstance(v, (int, float)) else None
        v = iq.get("psnr_std")
        psnr_std = float(v) if isinstance(v, (int, float)) else None
        v = iq.get("ssim_mean")
        ssim_mean = float(v) if isinstance(v, (int, float)) else None
        v = iq.get("ssim_std")
        ssim_std = float(v) if isinstance(v, (int, float)) else None

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
        "quarantined": quarantined,
        "quarantine_reason": quarantine_reason,
        "visual_core": _visual_core_for(meta.tag),
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "has_history": _has_history(meta.tag, runs_root),
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
        "deferred": sum(1 for r in rows if r["status"] == "Deferred"),
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


def update_cell(
    tag: str,
    out_path: Path = _DEFAULT_OUT,
    runs_root: Path = _RUNS_ROOT,
) -> dict:
    """Patch a single row's hydration into an existing `Final_Exp.json` (US-019).

    Avoids re-scanning all 186 cells when only one row changed. Loads the
    on-disk JSON, replaces the matching row's metric/status fields, and
    re-derives `counts` + `generated_at` before writing atomically.

    Falls back to a full `build_final_exp_json` rebuild when the on-disk
    file is missing, schema-mismatched, or doesn't contain `tag`.
    """
    from src.experiments.cells import iter_cells

    # Validate the tag up front so a typo cannot silently trigger a full
    # rebuild (which would obscure the real wiring bug).
    meta = next((m for m in iter_cells() if m.tag == tag), None)
    if meta is None:
        raise ValueError(
            f"unknown cell tag: {tag!r}. "
            "Expected one of the 186 tags from src.experiments.cells.iter_cells()."
        )

    if not out_path.exists():
        return build_final_exp_json(out_path=out_path, runs_root=runs_root)

    try:
        doc = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return build_final_exp_json(out_path=out_path, runs_root=runs_root)

    if doc.get("schema_version") != SCHEMA_VERSION or "rows" not in doc:
        return build_final_exp_json(out_path=out_path, runs_root=runs_root)

    new_row = _row_for(meta, runs_root)
    rows = doc["rows"]
    replaced = False
    for i, r in enumerate(rows):
        if r.get("tag") == tag:
            rows[i] = new_row
            replaced = True
            break
    if not replaced:
        # Schema drifted (row count / order). Safer to rebuild than to append.
        return build_final_exp_json(out_path=out_path, runs_root=runs_root)

    counts: CountsDict = {
        "total": len(rows),
        "pending": sum(1 for r in rows if r["status"] == "Pending"),
        "running": sum(1 for r in rows if r["status"] == "Running"),
        "complete": sum(1 for r in rows if r["status"] == "Complete"),
        "failed": sum(1 for r in rows if r["status"] == "Failed"),
        "deferred": sum(1 for r in rows if r["status"] == "Deferred"),
    }
    doc["counts"] = counts
    doc["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    write_doc_atomic(doc, out_path)
    return {
        "rows": counts["total"],
        "counts": dict(counts),
        "out": str(out_path),
        "generated_at": doc["generated_at"],
        "updated_tag": tag,
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate runs/final/<tag>/metrics.json into artifacts/Final_Exp.json.")
    p.add_argument("--out", default=str(_DEFAULT_OUT), help=f"Output JSON path (default: {_DEFAULT_OUT}).")
    p.add_argument("--runs-root", default=str(_RUNS_ROOT), help=f"Per-cell metrics root (default: {_RUNS_ROOT}).")
    p.add_argument(
        "--cell", default=None,
        help="US-019 incremental mode: patch only this single tag's row in the existing JSON.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.cell:
        summary = update_cell(
            tag=args.cell, out_path=Path(args.out), runs_root=Path(args.runs_root),
        )
        c = summary["counts"]
        print(
            f"final_exp_json[--cell={args.cell}]: {summary['rows']} rows "
            f"(pending={c['pending']}, running={c['running']}, complete={c['complete']}, "
            f"failed={c['failed']}, deferred={c['deferred']}) "
            f"-> {summary['out']}"
        )
        return 0
    summary = build_final_exp_json(out_path=Path(args.out), runs_root=Path(args.runs_root))
    c = summary["counts"]
    print(
        f"final_exp_json: {summary['rows']} rows "
        f"(pending={c['pending']}, running={c['running']}, complete={c['complete']}, "
        f"failed={c['failed']}, deferred={c['deferred']}) "
        f"-> {summary['out']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
