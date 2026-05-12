"""Unit tests for scripts/quarantine_transnext.py and the V3 quarantine lift.

Historically (US-014): the runner filtered TransNeXt cells via
is_quarantined() while the project waited for Blackwell hardware.

Current (V3 lift, ratified 2026-05-12): the predicate is a no-op and the
filter block in run_all_phases.py is gone. Tests now assert the LIFT:
  - is_quarantined returns False for every input.
  - The 186-cell matrix dispatches all 186 (62 TransNeXt + 124 CNN).
  - tune_all.py's default sweep includes TransNeXt.
  - run_all_phases.py source no longer contains the old filter block.

The quarantine SCRIPT (scripts/quarantine_transnext.py) is retained as a
manual-cleanup tool — useful for wiping stale TransNeXt run dirs before
re-running under V3. Tests below still exercise it (find/dry-run/remove)
to keep the cleanup path working.

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


def test_is_quarantined_returns_false_after_v3_lift():
    """V3 lift (ratified 2026-05-12): is_quarantined is a permanent no-op.

    Pre-V3 this predicate returned True for TransNeXt models so the runner
    skipped those cells while the project waited for Blackwell hardware. With
    the RTX 5070 online, TransNeXt cells dispatch and is_quarantined must
    return False for every input (callers that still invoke it just see "no
    quarantine" instead of having to be re-plumbed).
    """
    assert is_quarantined("transnext_base") is False
    assert is_quarantined("transnext_micro") is False
    assert is_quarantined("transnext_small") is False
    assert is_quarantined("TransNeXt_Base") is False
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


def test_v3_lift_iter_cells_dispatches_all_186():
    """V3 lift: all 186 cells (including the 62 TransNeXt rows) are
    dispatchable now that is_quarantined is a no-op."""
    from src.experiments.cells import iter_cells

    cells = list(iter_cells())
    assert len(cells) == 186, len(cells)
    dispatched = [c for c in cells if not is_quarantined(c.model)]
    assert len(dispatched) == 186, (
        f"V3 lift broken: only {len(dispatched)}/186 cells pass is_quarantined"
    )
    # Sanity: the matrix still contains TransNeXt rows (their existence is the
    # point of the lift — if they vanish, something else regressed).
    transnext_cells = [c for c in cells if "transnext" in c.model.lower()]
    assert len(transnext_cells) == 62, (
        f"matrix lost TransNeXt rows: {len(transnext_cells)}/62"
    )


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


def test_v3_lift_tune_all_includes_transnext_by_default():
    """V3 lift: `tune_all.py` (no --model) now includes TransNeXt in the
    default sweep because is_quarantined is a no-op."""
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
        rc = ta.main(["--n-trials", "1", "--dataset", "cifar10"])
        assert rc == 0
        assert "transnext_base" in captured["models"], (
            f"V3 lift broken: transnext_base missing from default tune sweep; "
            f"got {captured['models']}"
        )
        # Explicit --model transnext_base still works (operator intent path).
        captured.clear()
        rc = ta.main(["--n-trials", "1", "--model", "transnext_base", "--dataset", "cifar10"])
        assert rc == 0
        assert captured["models"] == ["transnext_base"], captured["models"]
    finally:
        _sys.modules.pop("src.tune_hyperparams", None)


def test_v3_lift_run_all_phases_no_longer_filters_quarantine():
    """V3 lift: the filter block in run_all_phases.py was removed. Asserting
    the *absence* of the old filter so a future regression that re-introduces
    it would be caught here."""
    src = (_REPO_ROOT / "run_all_phases.py").read_text(encoding="utf-8")
    assert "[c for c in matrix if not _is_quarantined(c.model)]" not in src, (
        "run_all_phases.py re-introduced the legacy US-014 quarantine filter; "
        "V3 ratified the lift on 2026-05-12 — see CLAUDE.md / matrix.py for the "
        "fair-comparison reframing."
    )


def test_us042_transnext_native_variants_reachable():
    """PRD US-042: `transnext_{micro,small,base}_native` are reachable through
    the wrapper specs and (for `_small_native`) the Optuna priors loader.

    Reconciliation with CLAUDE.md V3 (ratified 2026-05-12): the legacy
    224-upsample TransNeXt path was collapsed entirely by V3. The PRD's
    `--transnext_legacy_upsample` opt-in flag is therefore unneeded — the
    legacy path is blocked by nonexistence, not by an explicit gate.
    """
    from src.models.transnext_wrapper import (
        TRANSNEXT_NATIVE_VARIANTS,
        _TRANSNEXT_SPECS,
    )

    # Spec dict entry for every native variant.
    for v in TRANSNEXT_NATIVE_VARIANTS:
        assert v in _TRANSNEXT_SPECS, f"missing _TRANSNEXT_SPECS entry: {v}"
        spec = _TRANSNEXT_SPECS[v]
        assert spec["default_img_size"] == 32, (
            f"{v} default_img_size: expected 32, got {spec.get('default_img_size')}"
        )
        assert spec["default_patch_size"] == 2, (
            f"{v} default_patch_size: expected 2, got {spec.get('default_patch_size')}"
        )
        assert spec["default_pretrain_size"] is None, (
            f"{v} default_pretrain_size: expected None, got {spec.get('default_pretrain_size')}"
        )

    # Each native variant shares its arch dict with its base counterpart.
    for native, base in (
        ("transnext_micro_native", "transnext_micro"),
        ("transnext_small_native", "transnext_small"),
        ("transnext_base_native", "transnext_base"),
    ):
        for k in ("embed_dims", "num_heads", "depths"):
            assert _TRANSNEXT_SPECS[native][k] == _TRANSNEXT_SPECS[base][k], (
                f"{native}.{k} != {base}.{k}: native aliases must mirror their base architecture"
            )

    # `tune_all.py --validate-only` recognizes the new priors file. We
    # bypass the CLI and call the loader directly to keep this test fast.
    import tune_all
    assert "transnext_small_native" in tune_all.SUPPORTED_MODELS, (
        "tune_all.SUPPORTED_MODELS missing 'transnext_small_native'"
    )
    priors = tune_all.load_priors("transnext_small_native")
    assert "head_lr" in priors["hparams"]
    assert "backbone_lr" in priors["hparams"]


def test_us042_run_all_phases_dry_run_resolves_native_cell():
    """PRD US-042: `--plan final --phase A --model transnext_small_native
    --dataset cifar10 --dry-run` exits 0 and prints the resolved CellSpec."""
    import run_all_phases

    rc = run_all_phases.run_final_plan(
        phase="A",
        model="transnext_small_native",
        dataset="cifar10",
        dry_run=True,
        # Stub the heavy callables — dry-run shouldn't reach them, but pin
        # them anyway so an accidental code regression surfaces here.
        run_cell_fn=lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("dry-run reached run_cell")
        ),
        refresh_fn=lambda: (_ for _ in ()).throw(
            AssertionError("dry-run reached refresh")
        ),
        phase_boundary_fn=lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("dry-run reached phase boundary push")
        ),
        insurance_trial_fn=lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("dry-run reached insurance trial")
        ),
    )
    assert rc == 0, f"dry-run must exit 0, got rc={rc}"


def _run_all() -> int:
    fns = [
        test_is_quarantined_returns_false_after_v3_lift,
        test_find_quarantine_dirs_picks_transnext_and_v2_siblings,
        test_dry_run_does_not_remove_anything,
        test_quarantine_removes_dirs_and_skips_refresh,
        test_quarantine_never_opens_weight_files,
        test_v3_lift_iter_cells_dispatches_all_186,
        test_interrupted_sentinel_marks_cell_failed,
        test_v3_lift_tune_all_includes_transnext_by_default,
        test_v3_lift_run_all_phases_no_longer_filters_quarantine,
        test_us042_transnext_native_variants_reachable,
        test_us042_run_all_phases_dry_run_resolves_native_cell,
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
