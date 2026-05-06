"""Unit tests for scripts/update_final_exp.py.

Asserts:
  - Empty runs/final/ -> 186 Pending rows, total 0/186
  - A synthesised metrics.json flips its row to Complete and bumps totals
  - PSNR/SSIM render when image_quality.json is present
  - A traceback in log.txt without metrics.json produces Failed
  - Idempotency: render() called twice on the same disk state is byte-identical
  - --check exits 0 on no-drift, 1 on drift

Run: python -m src.tests.test_update_final_exp
"""
from __future__ import annotations

import datetime as _dt
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_updater():
    """Import scripts/update_final_exp.py as a module (it lives outside src/)."""
    path = _REPO_ROOT / "scripts" / "update_final_exp.py"
    spec = importlib.util.spec_from_file_location("update_final_exp", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


updater = _load_updater()


def _fixed_today() -> _dt.date:
    return _dt.date(2026, 5, 5)


def test_empty_runs_root_renders_with_quarantine_split():
    """US-014 quarantine: TransNeXt rows are Deferred regardless of disk state.
    Empty runs/final/ -> 124 Pending CNN rows + 62 Deferred TransNeXt rows.
    """
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        out = updater.render(runs_root, today=_fixed_today())
        # `| Pending |` only appears in CNN table rows.
        assert out.count("| Pending |") == 124, (
            f"expected 124 Pending (186 - 62 TransNeXt); "
            f"got {out.count('| Pending |')}"
        )
        # Quarantined rows show `Deferred — Pending Hardware`.
        assert out.count("Deferred — Pending Hardware") == 62, (
            f"expected 62 Deferred TransNeXt rows; "
            f"got {out.count('Deferred — Pending Hardware')}"
        )
        # Status summary still uses 186 denominator.
        assert "0/6" in out and "0/30" in out and "0/150" in out
        assert "**Total: 0/186**" in out
        assert "Deferred (TransNeXt — Pending Hardware): 62/186" in out
        assert "Last updated: 2026-05-05" in out


def test_complete_row_updates_status_and_totals():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        cell_dir = runs_root / "final_clean_resnet50_cifar10"
        cell_dir.mkdir(parents=True)
        (cell_dir / "metrics.json").write_text(
            json.dumps({"best_val_acc": 0.8731, "best_epoch": 24}),
            encoding="utf-8",
        )

        out = updater.render(runs_root, today=_fixed_today())
        # Row 1 must show Complete + 0.8731
        line = next(ln for ln in out.splitlines() if "final_clean_resnet50_cifar10" in ln)
        assert "Complete" in line
        assert "0.8731" in line
        # Status summary updated for Phase A only
        assert "Phase A (Clean): 1/6" in out
        assert "Phase B (Combined): 0/30" in out
        assert "**Total: 1/186**" in out


def test_image_quality_renders_psnr_ssim():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        cell_dir = runs_root / "final_B_L3_resnet50_cifar10"
        cell_dir.mkdir(parents=True)
        (cell_dir / "metrics.json").write_text(
            json.dumps({"best_val_acc": 0.42}), encoding="utf-8",
        )
        (cell_dir / "image_quality.json").write_text(
            json.dumps({
                "psnr_mean": 18.34, "psnr_std": 0.51,
                "ssim_mean": 0.4123, "ssim_std": 0.0287,
            }),
            encoding="utf-8",
        )
        out = updater.render(runs_root, today=_fixed_today())
        line = next(ln for ln in out.splitlines() if "final_B_L3_resnet50_cifar10" in ln)
        assert "18.34±0.51" in line
        assert "0.4123±0.0287" in line


def test_traceback_without_metrics_marks_failed():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        cell_dir = runs_root / "final_C_L5_blur_densenet121_mnist"
        cell_dir.mkdir(parents=True)
        (cell_dir / "log.txt").write_text(
            "epoch 0 ...\nTraceback (most recent call last):\n  ...\n",
            encoding="utf-8",
        )
        out = updater.render(runs_root, today=_fixed_today())
        line = next(ln for ln in out.splitlines() if "final_C_L5_blur_densenet121_mnist" in ln)
        assert "Failed" in line


def test_render_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        cell_dir = runs_root / "final_clean_transnext_base_mnist"
        cell_dir.mkdir(parents=True)
        (cell_dir / "metrics.json").write_text(
            json.dumps({"best_val_acc": 0.99}), encoding="utf-8",
        )
        a = updater.render(runs_root, today=_fixed_today())
        b = updater.render(runs_root, today=_fixed_today())
        assert a == b, "render must be deterministic for fixed disk state + date"


def test_check_mode_detects_drift():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        out_path = Path(td) / "Final_Exp.md"
        # Write a stale tracker that doesn't match disk state
        out_path.write_text("# stale tracker\n", encoding="utf-8")
        rc = updater.main([
            "--runs-root", str(runs_root),
            "--out", str(out_path),
            "--check",
        ])
        assert rc == 1, "drift must produce exit code 1"

        # Now make it match
        fresh = updater.render(runs_root)
        out_path.write_text(fresh, encoding="utf-8")
        rc2 = updater.main([
            "--runs-root", str(runs_root),
            "--out", str(out_path),
            "--check",
        ])
        assert rc2 == 0, "matching disk state must produce exit code 0"


def _run_all() -> int:
    fns = [
        test_empty_runs_root_renders_with_quarantine_split,
        test_complete_row_updates_status_and_totals,
        test_image_quality_renders_psnr_ssim,
        test_traceback_without_metrics_marks_failed,
        test_render_is_idempotent,
        test_check_mode_detects_drift,
    ]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
