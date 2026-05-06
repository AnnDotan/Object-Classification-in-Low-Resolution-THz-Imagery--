"""Unit tests for scripts/backfill_history_json.py (US-018).

Asserts:
  - Empty runs/final/ -> 0 written, 0 errors
  - A staged metrics.csv -> a matching history.json with the right schema
  - Existing history.json is preserved by default; --force overwrites
  - NaN cells are coerced to JSON null (not the string "NaN")
  - Privacy: never opens *.ckpt / *.pt / *.pth files

Run: ``python -m src.tests.test_backfill_history_json``
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "backfill_history_json",
        _REPO_ROOT / "scripts" / "backfill_history_json.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bf = _load_module()


_CSV_BODY = (
    "epoch,train_loss,train_acc,val_loss,val_acc\n"
    "1,nan,nan,1.966101,0.687000\n"
    "2,2.182377,0.394600,0.806598,0.893400\n"
    "3,1.105200,0.620100,0.512300,0.940800\n"
)


def _stage_cell(runs_root: Path, tag: str, csv_body: str = _CSV_BODY) -> Path:
    cell = runs_root / tag
    cell.mkdir(parents=True)
    (cell / "metrics.csv").write_text(csv_body, encoding="utf-8")
    return cell


def test_empty_runs_root_is_noop():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        summary = bf.backfill(runs_root=runs_root)
    assert summary == {"written": 0, "skipped": 0, "missing_csv": 0, "total": 0}


def test_csv_round_trips_into_history_json():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        tag = "final_clean_resnet50_cifar10"
        _stage_cell(runs_root, tag)

        summary = bf.backfill(runs_root=runs_root)
        assert summary["written"] == 1
        out = json.loads((runs_root / tag / "history.json").read_text(encoding="utf-8"))
        assert out["schema_version"] == 1
        assert out["tag"] == tag
        assert len(out["history"]) == 3
        # Epoch 1 had NaN train_loss / train_acc — must be JSON null, not "NaN".
        e1 = out["history"][0]
        assert e1["epoch"] == 1
        assert e1["train_loss"] is None
        assert e1["train_acc"] is None
        assert abs(e1["val_loss"] - 1.966101) < 1e-6
        assert abs(e1["val_acc"] - 0.687000) < 1e-6
        # Epoch 2 has full numbers.
        e2 = out["history"][1]
        assert abs(e2["train_loss"] - 2.182377) < 1e-6
        # Epoch 3 sanity — sorted by epoch.
        assert out["history"][2]["epoch"] == 3


def test_idempotent_skips_existing_history_json():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        tag = "final_clean_densenet121_mnist"
        cell = _stage_cell(runs_root, tag)
        # Pre-write a history.json with a sentinel value.
        (cell / "history.json").write_text(
            json.dumps({"schema_version": 1, "tag": tag,
                        "history": [{"epoch": 99, "val_acc": 0.9999}]}),
            encoding="utf-8",
        )

        summary = bf.backfill(runs_root=runs_root)
        assert summary["written"] == 0
        assert summary["skipped"] == 1
        # Sentinel still there.
        out = json.loads((cell / "history.json").read_text(encoding="utf-8"))
        assert out["history"][0]["val_acc"] == 0.9999

        # --force overwrites.
        summary2 = bf.backfill(runs_root=runs_root, force=True)
        assert summary2["written"] == 1
        out2 = json.loads((cell / "history.json").read_text(encoding="utf-8"))
        assert len(out2["history"]) == 3, "force did not regenerate from CSV"


def test_cell_without_csv_is_marked_missing_csv():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        tag = "final_B_L3_resnet50_cifar10"
        (runs_root / tag).mkdir()  # no metrics.csv

        summary = bf.backfill(runs_root=runs_root)
        assert summary["missing_csv"] == 1
        assert summary["written"] == 0
        assert not (runs_root / tag / "history.json").exists()


def test_backfill_never_opens_weight_files():
    """Weight Privacy: the script must never read *.ckpt / *.pt / *.pth."""
    opened: list[str] = []
    real_open = io.open

    def _recording_open(file, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            opened.append(str(file))
        except Exception:
            pass
        return real_open(file, *args, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        tag = "final_clean_resnet50_cifar10"
        cell = _stage_cell(runs_root, tag)
        # Decoy weight files that the backfiller must not read.
        (cell / "model.ckpt").write_bytes(b"\x00" * 16)
        (cell / "weights.pt").write_bytes(b"\x00" * 16)
        (cell / "snapshot.pth").write_bytes(b"\x00" * 16)

        with patch("builtins.open", side_effect=_recording_open), \
             patch("io.open", side_effect=_recording_open):
            bf.backfill(runs_root=runs_root)

    forbidden = (".ckpt", ".pt", ".pth")
    bad = [p for p in opened if any(p.endswith(s) for s in forbidden)]
    assert not bad, f"backfill opened weight file(s): {bad}"


def _run_all() -> int:
    fns = [
        test_empty_runs_root_is_noop,
        test_csv_round_trips_into_history_json,
        test_idempotent_skips_existing_history_json,
        test_cell_without_csv_is_marked_missing_csv,
        test_backfill_never_opens_weight_files,
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
