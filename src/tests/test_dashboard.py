"""Final_Exp.html dashboard tests (FINAL_EXP Dashboard, US-004..US-007).

Replaces the legacy Gold-Standard <details>-row test (project US-015).
The new dashboard is JS-driven: rows are populated client-side from
artifacts/Final_Exp.json at load time, so the static HTML contains a
scaffold (header / stats bar / tab bar / empty tbody) but no row data.
Aggregate row enumeration is covered by test_build_final_exp_json.

Asserts:
  - build_dashboard returns the stable summary shape used by
    scripts/refresh_trackers.refresh_final_exp_html.
  - Static HTML scaffold structure: doctype, header, stats bar, sticky
    tab bar with three Phase A/B/C tabs, empty tbody, banner element,
    inline JS that fetches ./Final_Exp.json at runtime.
  - Required column headers in exact order.
  - Inline JS exposes the level-badge renderer used by US-007 (named
    helper levelBadgeHtml — sentinel for downstream stories).
  - No legacy layout leakage: phase-section divs / detail-row tables /
    PSNR-SSIM columns / gate-verdict markup are gone.

Run: ``python -m src.tests.test_dashboard``
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tools.build_final_dashboard import build_dashboard  # noqa: E402


def _check_summary_shape() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        summary = build_dashboard(out_path=out, runs_root=runs_root)
    assert summary["rows"] == 186, summary
    assert summary["phase_a"] == 6, summary
    assert summary["phase_b"] == 30, summary
    assert summary["phase_c"] == 150, summary
    assert summary["out"].endswith("Final_Exp.html"), summary
    print("OK [summary] -- build_dashboard returns rows / phase_a / phase_b / phase_c / out.")


def _check_static_scaffold() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert body.startswith("<!DOCTYPE html>"), "missing doctype"
    assert "<title>Final Experiment Dashboard" in body
    assert 'class="header"' in body
    assert 'class="stats-bar"' in body
    assert 'class="tab-bar"' in body
    assert 'class="exp-table"' in body
    # Tabs for all three phases
    for p in ("A", "B", "C"):
        assert f'data-phase="{p}"' in body, f"missing tab data-phase={p}"
        assert f'id="tab-count-{p}"' in body, f"missing tab-count-{p}"
    # Stats bar IDs
    for sid in (
        "stat-total", "stat-pending", "stat-running",
        "stat-complete", "stat-failed", "stat-best", "stat-all",
    ):
        assert f'id="{sid}"' in body, f"missing stat id {sid}"
    # tbody is JS-populated (empty in static HTML)
    assert '<tbody id="exp-tbody"></tbody>' in body, "tbody must be empty in static HTML"
    # Banner for fetch failures
    assert 'id="banner"' in body
    print("OK [scaffold] -- doctype, header, stats bar, tab bar, table shell, banner.")


def _check_column_headers_in_order() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    expected = ["Tag", "Model", "Dataset", "Phase", "Level", "Status", "val_acc", "Epochs", "Runtime"]
    positions = [body.index(f"<th>{h}</th>") for h in expected]
    assert positions == sorted(positions), f"column headers out of order: {expected} -> {positions}"
    print(f"OK [headers] -- {len(expected)} columns in order: {' | '.join(expected)}.")


def _check_inline_js_contract() -> None:
    """The inline JS must reference ./Final_Exp.json and expose level-badge logic."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert "./Final_Exp.json" in body, "JS must fetch ./Final_Exp.json"
    assert "levelBadgeHtml" in body, "JS must define the level-badge renderer (US-007)"
    assert "applyActivePhase" in body, "JS must expose tab-switch logic (US-006)"
    assert "renderRows" in body, "JS must expose row rendering (US-005)"
    # localStorage key for active phase persistence
    assert "final_exp.active_phase" in body, "missing localStorage key for active phase"
    print("OK [js] -- Final_Exp.json fetch + tab/badge/row helpers + localStorage key.")


def _check_no_legacy_markup_leaks() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    forbidden = [
        'class="phase-section"',  # legacy per-phase wrapper
        'class="detail-row"',     # legacy <details>-row
        'PSNR / SSIM',            # legacy column
        "gate_verdict",           # legacy Phase A overlay
        '<details>',
    ]
    for f in forbidden:
        assert f not in body, f"legacy markup leaked into new dashboard: {f!r}"
    print("OK [clean] -- no legacy phase-section / detail-row / PSNR-SSIM / gate-verdict leakage.")


def _check_thumbs_dir_back_compat() -> None:
    """build_dashboard must accept (and ignore) thumbs_dir for refresh_trackers back-compat."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        thumbs = Path(td) / "thumbs"
        thumbs.mkdir()
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Old call sites pass thumbs_dir kwarg; must not raise.
        summary = build_dashboard(out_path=out, runs_root=runs_root, thumbs_dir=thumbs)
        assert summary["rows"] == 186
    print("OK [back-compat] -- build_dashboard accepts (and ignores) thumbs_dir kwarg.")


def main() -> int:
    _check_summary_shape()
    _check_static_scaffold()
    _check_column_headers_in_order()
    _check_inline_js_contract()
    _check_no_legacy_markup_leaks()
    _check_thumbs_dir_back_compat()
    print("\nAll dashboard scaffold checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
