"""Shared tracker-refresh helper test (US-016).

Verifies:
  (a) refresh_all() runs BOTH refresh functions.
  (b) Failure in one does NOT skip the other (failure isolation).
  (c) Errors surface in the result dict; nothing raises.
  (d) Real run end-to-end writes Final_Exp.md AND artifacts/Final_Exp.html.

Run: ``python -m src.tests.test_refresh_trackers``
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scripts_refresh_trackers",
        _REPO_ROOT / "scripts" / "refresh_trackers.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_refresh_all_invokes_both() -> None:
    mod = _load_module()

    md_calls = {"n": 0}
    html_calls = {"n": 0}
    thumbs_calls = {"n": 0}

    original_md = mod.refresh_final_exp_md
    original_html = mod.refresh_final_exp_html
    original_thumbs = mod.refresh_visual_core_thumbs
    mod.refresh_final_exp_md = lambda: (md_calls.update(n=md_calls["n"] + 1) or (True, None))
    mod.refresh_final_exp_html = lambda: (html_calls.update(n=html_calls["n"] + 1) or (True, None))
    mod.refresh_visual_core_thumbs = lambda: (thumbs_calls.update(n=thumbs_calls["n"] + 1) or (True, None))
    try:
        result = mod.refresh_all()
    finally:
        mod.refresh_final_exp_md = original_md
        mod.refresh_final_exp_html = original_html
        mod.refresh_visual_core_thumbs = original_thumbs

    assert md_calls["n"] == 1
    assert html_calls["n"] == 1
    assert thumbs_calls["n"] == 1
    assert result["md_ok"] is True
    assert result["html_ok"] is True
    assert result["thumbs_ok"] is True
    assert result["errors"] == []
    print("OK [all-invoked] -- refresh_all calls md + html + thumbs refresh once each.")


def _check_failure_isolation() -> None:
    mod = _load_module()

    html_calls = {"n": 0}
    thumbs_calls = {"n": 0}
    original_md = mod.refresh_final_exp_md
    original_html = mod.refresh_final_exp_html
    original_thumbs = mod.refresh_visual_core_thumbs

    mod.refresh_final_exp_md = lambda: (False, "synthetic md failure")
    mod.refresh_final_exp_html = lambda: (
        html_calls.update(n=html_calls["n"] + 1) or (True, None)
    )
    mod.refresh_visual_core_thumbs = lambda: (
        thumbs_calls.update(n=thumbs_calls["n"] + 1) or (True, None)
    )
    try:
        result = mod.refresh_all()
    finally:
        mod.refresh_final_exp_md = original_md
        mod.refresh_final_exp_html = original_html
        mod.refresh_visual_core_thumbs = original_thumbs

    assert html_calls["n"] == 1, "HTML refresh must run even when MD refresh failed"
    assert thumbs_calls["n"] == 1, "Thumbs refresh must run even when MD refresh failed"
    assert result["md_ok"] is False
    assert result["html_ok"] is True
    assert result["thumbs_ok"] is True
    assert any("synthetic md failure" in e for e in result["errors"])
    assert any("final_exp_md:" in e for e in result["errors"])
    print("OK [isolation] -- MD failure does not skip HTML/thumbs; error surfaced.")


def _check_does_not_raise_on_inner_exception() -> None:
    """Inner functions can themselves raise; refresh_all must catch via the
    (ok, error) contract from each refresh helper."""
    mod = _load_module()

    original_md = mod.refresh_final_exp_md
    original_html = mod.refresh_final_exp_html
    original_thumbs = mod.refresh_visual_core_thumbs

    # Simulate the wrappers' exception-catch behavior by returning (False, msg).
    mod.refresh_final_exp_md = lambda: (False, "RuntimeError: boom")
    mod.refresh_final_exp_html = lambda: (False, "RuntimeError: also boom")
    mod.refresh_visual_core_thumbs = lambda: (False, "RuntimeError: thumbs boom")
    try:
        result = mod.refresh_all()  # MUST NOT raise
    finally:
        mod.refresh_final_exp_md = original_md
        mod.refresh_final_exp_html = original_html
        mod.refresh_visual_core_thumbs = original_thumbs

    assert result["md_ok"] is False
    assert result["html_ok"] is False
    assert result["thumbs_ok"] is False
    assert len(result["errors"]) == 3
    print("OK [no-raise] -- refresh_all returns errors instead of raising.")


def _stage_complete_cell(runs_root: Path, tag: str) -> None:
    rd = runs_root / tag
    rd.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(8):
        rows.append({
            "epoch": i + 1,
            "train_loss": 1.5 - 0.1 * i,
            "train_acc": 0.4 + 0.05 * i,
            "val_loss": 1.4 - 0.1 * i,
            "val_acc": 0.5 + 0.04 * i,
        })
    with open(rd / "metrics.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                            "val_loss", "val_acc"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (rd / "metrics.json").write_text(json.dumps({
        "best_val_acc": 0.998,
        "best_epoch": 8,
        "last_val_acc": 0.998,
        "last_train_acc": 0.999,
        "epochs_run": 8,
    }), encoding="utf-8")


def _check_end_to_end_writes_both_trackers() -> None:
    mod = _load_module()
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        runs_root = ws / "runs" / "final"
        runs_root.mkdir(parents=True)
        _stage_complete_cell(runs_root, "final_clean_resnet50_mnist")

        old_cwd = os.getcwd()
        os.chdir(ws)
        try:
            (ws / "artifacts").mkdir(exist_ok=True)
            # Stub the thumbs refresh to keep the test torch-free; the real
            # render is exercised in src/tests/test_render_thumbs.py.
            original_thumbs = mod.refresh_visual_core_thumbs
            mod.refresh_visual_core_thumbs = lambda: (True, None)
            try:
                result = mod.refresh_all()
            finally:
                mod.refresh_visual_core_thumbs = original_thumbs
        finally:
            os.chdir(old_cwd)

        md = ws / "Final_Exp.md"
        html = ws / "artifacts" / "Final_Exp.html"
        assert result["md_ok"] is True, result
        assert result["html_ok"] is True, result
        assert result["thumbs_ok"] is True, result
        assert md.exists()
        assert html.exists()
        body = html.read_text(encoding="utf-8")
        assert "final_clean_resnet50_mnist" in body
        assert ">Complete<" in body
    print("OK [end-to-end] -- real refresh_all writes Final_Exp.md + Final_Exp.html.")


def _check_incremental_mode_skips_md_and_calls_json_cell() -> None:
    """US-019: `refresh_all(cell=tag)` must run the incremental JSON path
    (and HTML), skip MD by default, and report `mode == 'incremental'`."""
    mod = _load_module()

    md_calls = {"n": 0}
    html_calls = {"n": 0}
    json_cell_calls = {"n": 0, "tag": None}
    thumbs_calls = {"n": 0}
    original_md = mod.refresh_final_exp_md
    original_html = mod.refresh_final_exp_html
    original_json = mod.refresh_final_exp_json_cell
    original_thumbs = mod.refresh_visual_core_thumbs

    mod.refresh_final_exp_md = lambda: (md_calls.update(n=md_calls["n"] + 1) or (True, None))
    mod.refresh_final_exp_html = lambda: (html_calls.update(n=html_calls["n"] + 1) or (True, None))
    mod.refresh_final_exp_json_cell = lambda tag: (
        json_cell_calls.update(n=json_cell_calls["n"] + 1, tag=tag) or (True, None)
    )
    mod.refresh_visual_core_thumbs = lambda: (
        thumbs_calls.update(n=thumbs_calls["n"] + 1) or (True, None)
    )
    try:
        result = mod.refresh_all(cell="final_B_L3_resnet50_cifar10")
    finally:
        mod.refresh_final_exp_md = original_md
        mod.refresh_final_exp_html = original_html
        mod.refresh_final_exp_json_cell = original_json
        mod.refresh_visual_core_thumbs = original_thumbs

    assert md_calls["n"] == 0, "MD must be skipped in incremental mode by default"
    assert json_cell_calls["n"] == 1, "JSON cell-update must run in incremental mode"
    assert json_cell_calls["tag"] == "final_B_L3_resnet50_cifar10"
    assert html_calls["n"] == 1, "HTML must rebuild even in incremental mode"
    assert thumbs_calls["n"] == 1, "Thumbs run unless --no-thumbs is set"
    assert result["mode"] == "incremental"
    assert result["cell"] == "final_B_L3_resnet50_cifar10"
    assert result["md_ok"] is None
    assert result["json_ok"] is True
    assert result["html_ok"] is True
    assert result["thumbs_ok"] is True
    assert result["errors"] == []
    print("OK [incremental] -- cell mode skips MD, runs JSON cell-update + HTML.")


def _check_skip_thumbs_flag_skips_thumb_render() -> None:
    """`refresh_all(skip_thumbs=True)` must short-circuit the thumbs step."""
    mod = _load_module()

    thumbs_calls = {"n": 0}
    original_md = mod.refresh_final_exp_md
    original_html = mod.refresh_final_exp_html
    original_thumbs = mod.refresh_visual_core_thumbs

    mod.refresh_final_exp_md = lambda: (True, None)
    mod.refresh_final_exp_html = lambda: (True, None)
    mod.refresh_visual_core_thumbs = lambda: (
        thumbs_calls.update(n=thumbs_calls["n"] + 1) or (True, None)
    )
    try:
        result = mod.refresh_all(skip_thumbs=True)
    finally:
        mod.refresh_final_exp_md = original_md
        mod.refresh_final_exp_html = original_html
        mod.refresh_visual_core_thumbs = original_thumbs

    assert thumbs_calls["n"] == 0, "thumbs must NOT run when skip_thumbs=True"
    assert result["thumbs_ok"] is None
    assert result["mode"] == "full"
    print("OK [skip-thumbs] -- skip_thumbs=True short-circuits the thumb render.")


def _check_cli_main_routes_cell_argument() -> None:
    """`scripts/refresh_trackers.py --cell <tag> --no-md --no-thumbs` must
    invoke `refresh_all(cell=<tag>, skip_md=True, skip_thumbs=True)`."""
    mod = _load_module()

    captured: dict = {}

    def _fake_refresh_all(*, cell=None, skip_md=False, skip_thumbs=False):
        captured.update(cell=cell, skip_md=skip_md, skip_thumbs=skip_thumbs)
        return {"md_ok": None, "html_ok": True, "json_ok": True,
                "thumbs_ok": None, "mode": "incremental", "cell": cell, "errors": []}

    original = mod.refresh_all
    mod.refresh_all = _fake_refresh_all
    try:
        rc = mod.main(["--cell", "final_B_L3_densenet121_mnist",
                        "--no-md", "--no-thumbs"])
    finally:
        mod.refresh_all = original

    assert rc == 0, f"unexpected rc={rc}"
    assert captured["cell"] == "final_B_L3_densenet121_mnist"
    assert captured["skip_md"] is True
    assert captured["skip_thumbs"] is True
    print("OK [cli] -- CLI flags route through to refresh_all(cell, skip_md, skip_thumbs).")


def main() -> int:
    _check_refresh_all_invokes_both()
    _check_failure_isolation()
    _check_does_not_raise_on_inner_exception()
    _check_end_to_end_writes_both_trackers()
    _check_incremental_mode_skips_md_and_calls_json_cell()
    _check_skip_thumbs_flag_skips_thumb_render()
    _check_cli_main_routes_cell_argument()
    return 0


if __name__ == "__main__":
    sys.exit(main())
