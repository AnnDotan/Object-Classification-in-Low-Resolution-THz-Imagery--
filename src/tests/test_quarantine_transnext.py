"""Unit tests for scripts/quarantine_transnext.py and the matching guard
in run_all_phases.py (US-014, docs/prds/PHASE_B_VISUAL_CORE.md).

Asserts:
  - find_quarantine_dirs picks up canonical TransNeXt tags + __v2 siblings
    and ignores CNN runs.
  - quarantine(dry_run=True) makes zero filesystem changes.
  - quarantine() actually removes the planned dirs.
  - is_quarantined(model) matches all three TransNeXt model strings and
    no CNN model.
  - run_all_phases.run_final_plan does not dispatch any TransNeXt cell.

Run: python -m src.tests.test_quarantine_transnext
"""
from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.experiments.run_status import is_quarantined  # noqa: E402

_FORBIDDEN_PATH_SUBSTRINGS: tuple[str, ...] = (
    ".ckpt", ".pt", ".pth", "artifacts/weights", "artifacts\\weights",
)


def _load_quarantine():
    path = _REPO_ROOT / "scripts" / "quarantine_transnext.py"
    spec = importlib.util.spec_from_file_location("quarantine_transnext", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


qmod = _load_quarantine()


def test_is_quarantined_matches_all_transnext_variants():
    assert is_quarantined("transnext_base") is True
    assert is_quarantined("transnext_micro") is True
    assert is_quarantined("transnext_small") is True
    assert is_quarantined("TransNeXt_Base") is True
    # Negative cases — CNNs and edge inputs.
    assert is_quarantined("resnet50") is False
    assert is_quarantined("densenet121") is False
    assert is_quarantined("") is False
    assert is_quarantined(None) is False  # type: ignore[arg-type]


def test_find_quarantine_dirs_picks_transnext_and_v2_siblings():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Plant a mix of CNN, TransNeXt, and __v2 dirs.
        wanted = [
            "final_clean_transnext_base_cifar10",
            "final_clean_transnext_base_mnist",
            "final_clean_transnext_base_cifar10__v2",
            "final_B_L3_transnext_base_mnist",
        ]
        unwanted = [
            "final_clean_resnet50_cifar10",
            "final_clean_densenet121_mnist",
            "final_B_L3_resnet50_cifar10",
        ]
        for name in wanted + unwanted:
            (runs_root / name).mkdir()

        found = sorted(d.name for d in qmod.find_quarantine_dirs(runs_root))
        assert found == sorted(wanted), f"unexpected: {found}"


def test_dry_run_does_not_remove_anything():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        d = runs_root / "final_clean_transnext_base_cifar10"
        d.mkdir()
        (d / "log.txt").write_text("epoch 0\n", encoding="utf-8")

        result = qmod.quarantine(runs_root=runs_root, dry_run=True, refresh=False)
        assert result["dry_run"] is True
        assert result["dirs_removed"] == []
        assert d.exists(), "dry-run must not remove the dir"


def test_quarantine_removes_dirs_and_skips_refresh():
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        keep = runs_root / "final_clean_resnet50_cifar10"
        kill1 = runs_root / "final_clean_transnext_base_cifar10"
        kill2 = runs_root / "final_clean_transnext_base_mnist__v2"
        for p in (keep, kill1, kill2):
            p.mkdir()
            (p / "log.txt").write_text("placeholder\n", encoding="utf-8")

        result = qmod.quarantine(runs_root=runs_root, dry_run=False, refresh=False)
        assert sorted(Path(p).name for p in result["dirs_removed"]) == sorted([kill1.name, kill2.name])
        assert keep.exists(), "CNN dir must be untouched"
        assert not kill1.exists()
        assert not kill2.exists()


def test_quarantine_never_opens_weight_files():
    """Weight Privacy: the script must not Read .ckpt/.pt/.pth/weights paths."""
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
        d = runs_root / "final_clean_transnext_base_cifar10"
        d.mkdir()
        # Decoy weight files that the quarantine script must not read.
        (d / "model.ckpt").write_bytes(b"\x00" * 16)
        (d / "weights.pt").write_bytes(b"\x00" * 16)
        (d / "snapshot.pth").write_bytes(b"\x00" * 16)

        with patch("builtins.open", side_effect=_recording_open), \
             patch("io.open", side_effect=_recording_open):
            qmod.quarantine(runs_root=runs_root, dry_run=False, refresh=False)

    bad = [p for p in opened if any(s in p for s in _FORBIDDEN_PATH_SUBSTRINGS)]
    assert not bad, f"quarantine opened forbidden paths: {bad}"


def test_iter_cells_filter_yields_124_cnn_cells():
    """The guard logic in run_all_phases.run_final_plan filters
    `is_quarantined(c.model)` from the matrix. We assert the predicate
    against the torch-free `iter_cells` enumeration (the actual
    `build_final_matrix` requires torch and is exercised in CI)."""
    from src.experiments.cells import iter_cells

    cells = list(iter_cells())
    assert len(cells) == 186, len(cells)
    cnn = [c for c in cells if not is_quarantined(c.model)]
    transnext = [c for c in cells if is_quarantined(c.model)]
    # 2 CNN models * 2 datasets * (1 clean + 5 B + 25 C) = 124
    assert len(cnn) == 124, f"expected 124 CNN cells; got {len(cnn)}"
    assert len(transnext) == 62, f"expected 62 TransNeXt cells; got {len(transnext)}"
    # Every quarantined cell's model contains 'transnext'.
    for c in transnext:
        assert "transnext" in c.model.lower(), c.model


def test_interrupted_sentinel_marks_cell_failed():
    """US-016: an INTERRUPTED sentinel under runs/final/<tag>/ flips the
    cell to Failed regardless of any partial metrics.json present."""
    from src.experiments.run_status import (
        INTERRUPTED_SENTINEL,
        detect_status,
    )
    import json as _json

    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        tag = "final_B_L3_resnet50_cifar10"
        rd = runs_root / tag
        rd.mkdir(parents=True)

        # Without sentinel + with completed metrics → Complete.
        (rd / "metrics.json").write_text(
            _json.dumps({"best_val_acc": 0.7}), encoding="utf-8",
        )
        assert detect_status(tag, runs_root=runs_root) == "Complete"

        # Drop sentinel → must override to Failed.
        (rd / INTERRUPTED_SENTINEL).write_text(
            "sigint at cell\n", encoding="utf-8",
        )
        assert detect_status(tag, runs_root=runs_root) == "Failed"

        # Sentinel without metrics → also Failed.
        rd2 = runs_root / "final_B_L3_densenet121_mnist"
        rd2.mkdir()
        (rd2 / INTERRUPTED_SENTINEL).write_text("sigint\n", encoding="utf-8")
        assert detect_status("final_B_L3_densenet121_mnist", runs_root=runs_root) == "Failed"


def test_tune_all_skips_transnext_when_iterating_all_models():
    """US-016: `tune_all.py` (no --model) must skip TransNeXt by default;
    explicit `--model transnext_*` bypasses the guard.

    `tune_all.main` lazy-imports `src.tune_hyperparams.run_studies`. We
    inject a fake module via `sys.modules` so the import resolves to our
    capture without pulling torch + Lightning into the test."""
    import sys as _sys
    import types as _types
    import tune_all as ta

    captured: dict = {}

    fake_module = _types.ModuleType("src.tune_hyperparams")

    def _fake_run_studies(**kwargs):
        captured.update(kwargs)
        return 0

    fake_module.run_studies = _fake_run_studies
    _sys.modules["src.tune_hyperparams"] = fake_module
    try:
        # No --model: TransNeXt should be filtered out.
        rc = ta.main(["--n-trials", "1", "--dataset", "cifar10"])
        assert rc == 0
        assert "transnext_base" not in captured["models"], (
            f"transnext_base must be filtered; got {captured['models']}"
        )
        assert set(captured["models"]) == {"resnet50", "densenet121"}, captured["models"]

        # Explicit --model transnext_base: guard must NOT override operator intent.
        captured.clear()
        rc = ta.main(["--n-trials", "1", "--model", "transnext_base", "--dataset", "cifar10"])
        assert rc == 0
        assert captured["models"] == ["transnext_base"], captured["models"]
    finally:
        _sys.modules.pop("src.tune_hyperparams", None)


def test_run_all_phases_has_quarantine_guard_in_source():
    """Static guard against accidental removal of the filter in run_all_phases.py.

    Importing run_all_phases.run_final_plan transitively pulls in torch
    (matrix.py), which isn't installed in every CI tier. Read the source
    instead to assert the guard predicate is present."""
    src = (_REPO_ROOT / "run_all_phases.py").read_text(encoding="utf-8")
    assert "is_quarantined" in src, (
        "run_all_phases.py must filter the matrix via is_quarantined(c.model) (US-014)"
    )
    assert "[c for c in matrix if not _is_quarantined(c.model)]" in src, (
        "run_all_phases.py must drop quarantined cells from the executable matrix"
    )


def _run_all() -> int:
    fns = [
        test_is_quarantined_matches_all_transnext_variants,
        test_find_quarantine_dirs_picks_transnext_and_v2_siblings,
        test_dry_run_does_not_remove_anything,
        test_quarantine_removes_dirs_and_skips_refresh,
        test_quarantine_never_opens_weight_files,
        test_iter_cells_filter_yields_124_cnn_cells,
        test_interrupted_sentinel_marks_cell_failed,
        test_tune_all_skips_transnext_when_iterating_all_models,
        test_run_all_phases_has_quarantine_guard_in_source,
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
