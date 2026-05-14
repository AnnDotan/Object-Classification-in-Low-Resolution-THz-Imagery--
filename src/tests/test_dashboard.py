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
    expected = ["Tag", "Model", "Dataset", "Phase", "Level", "Visual", "Status", "val_acc", "Epochs", "Runtime", "Curves"]
    positions = [body.index(f"<th>{h}</th>") for h in expected]
    assert positions == sorted(positions), f"column headers out of order: {expected} -> {positions}"
    print(f"OK [headers] -- {len(expected)} columns in order: {' | '.join(expected)}.")


def _check_inline_js_contract() -> None:
    """The inline JS must wire up bootstrap, fetch fallback, level-badge logic, polling, chips."""
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
    assert "readInlineDoc" in body, "JS must bootstrap from inline JSON (file:// compat)"
    assert "startPolling" in body and "stopPolling" in body, "JS must implement polling (US-013)"
    assert "POLL_INTERVAL_MS" in body, "polling interval constant missing"
    assert "visibilitychange" in body, "polling must pause on tab hidden (US-013)"
    assert "rowMatchesFilters" in body, "JS must implement chip filter logic (US-010)"
    # localStorage keys
    assert "final_exp.active_phase" in body, "missing localStorage key for active phase"
    assert "final_exp.filters." in body, "missing localStorage key prefix for chip filters"
    print("OK [js] -- inline bootstrap + fetch fallback + tab/badge/row + chips + polling.")


def _check_inline_initial_data_embedded() -> None:
    """Static HTML must embed the FinalExpDoc as <script type=application/json>."""
    import json as _json
    import re as _re
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    m = _re.search(
        r'<script type="application/json" id="initial-data">(.+?)</script>',
        body, _re.S,
    )
    assert m, "missing inline-data <script type=application/json>"
    # Reverse the </ -> <\/ defensive escape before parsing.
    raw = m.group(1).replace("<\\/", "</")
    doc = _json.loads(raw)
    assert doc["schema_version"] == 1
    assert doc["counts"]["total"] == 186
    assert len(doc["rows"]) == 186
    print(f"OK [inline-data] -- 186 rows embedded inline ({len(raw):,} bytes JSON).")


def _check_chip_filter_groups_present() -> None:
    """Filter row must render chip groups for model / dataset / status / axis (US-010)."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    for facet in ("model", "dataset", "status", "axis"):
        assert f'data-chip-group="{facet}"' in body, f"missing chip group {facet}"
    expected_values = (
        "resnet50", "densenet121", "transnext_tiny",
        "cifar10", "mnist",
        "Pending", "Running", "Complete", "Failed", "Deferred",
        "resolution", "blur", "noise", "saturation", "salt_pepper",
    )
    for v in expected_values:
        assert f'data-value="{v}"' in body, f"missing chip value {v}"
    assert 'id="clear-filters"' in body, "missing clear-all button"
    assert 'id="visible-count"' in body, "missing visible-count counter"
    print(f"OK [chips] -- 4 chip groups, {len(expected_values)} values, clear-all + count.")


def _check_polling_and_timestamp_ui() -> None:
    """Header must surface generated + last-polled timestamps (US-014a)."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert 'id="ts-generated"' in body
    assert 'id="ts-polled"' in body
    print("OK [timestamps] -- ts-generated + ts-polled IDs present.")


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


def _check_polling_cadence_is_10_minutes() -> None:
    """US-019: dashboard polling cadence must be 600,000 ms (10 min).

    Static-grep the rendered HTML for the constant assignment so a future
    edit can't silently drop us back to 30 s without breaking the test."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert "POLL_INTERVAL_MS = 600000" in body, (
        "POLL_INTERVAL_MS must be 600000 (10 min) per US-019"
    )
    # Belt-and-braces: 30000 should NOT be the assigned value any more.
    assert "POLL_INTERVAL_MS = 30000" not in body, (
        "stale 30 s polling cadence still present"
    )
    print("OK [polling] -- POLL_INTERVAL_MS=600000 (10 min) per US-019.")


def _check_manual_refresh_and_countdown_pill_present() -> None:
    """US-019: header must surface a 'Next poll in' countdown + a manual
    'Refresh now' button. The button forces a fetch via `loadJsonOnce()`
    without altering the polling timer."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert 'id="ts-countdown"' in body, "missing countdown span"
    assert 'id="manual-refresh"' in body, "missing manual-refresh button"
    assert "Refresh now" in body, "manual-refresh button label missing"
    assert "fmtCountdown" in body, "JS must define fmtCountdown helper"
    assert "bindManualRefresh" in body, "JS must wire the manual-refresh click handler"
    # The button must call loadJsonOnce, not setInterval — otherwise it
    # could create a parallel polling loop.
    bind_idx = body.index("bindManualRefresh")
    bind_block = body[bind_idx:bind_idx + 600]
    assert "loadJsonOnce" in bind_block, "manual refresh must call loadJsonOnce"
    assert "setInterval" not in bind_block, "manual refresh must NOT install a new polling interval"
    print("OK [manual-refresh] -- countdown pill + Refresh-now button + fmtCountdown wired.")


def _check_curves_drawer_lazy_loaded() -> None:
    """US-018: dashboard ships a learning-curve drawer that
    (a) declares Plotly via `defer` so initial paint is unblocked,
    (b) does NOT fetch any history.json from the bootstrap path,
    (c) wires a row-click handler that calls openCurvesDrawer,
    (d) caches fetched payloads in HISTORY_CACHE."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")

    # CDN load tag must be `defer` so it does not block first paint.
    assert "cdn.plot.ly/plotly" in body, "Plotly CDN script missing"
    assert "<script src=\"https://cdn.plot.ly/plotly" in body
    assert "defer></script>" in body, "Plotly script must be deferred"

    # Drawer markup + JS hooks present.
    assert 'id="curves-drawer"' in body
    assert 'id="drawer-title"' in body
    assert 'id="drawer-body"' in body
    assert 'id="drawer-close"' in body
    assert "openCurvesDrawer" in body
    assert "closeCurvesDrawer" in body
    assert "HISTORY_CACHE" in body, "JS must cache fetched history payloads"
    assert "renderCurvesFallback" in body, "fallback for offline/no-Plotly required"

    # Initial paint must NOT fetch any history.json. The string 'history.json'
    # may legitimately appear in the open-handler URL template, but the
    # `init` block (and the polling path) must not contain it.
    init_idx = body.find("function init()")
    assert init_idx > 0
    init_block = body[init_idx:init_idx + 2400]
    assert "history.json" not in init_block, (
        "init() must NOT fetch history.json; it is opened lazily on row click"
    )

    # has_history pill rendered.
    assert "historyIndicatorHtml" in body
    assert "history-indicator" in body
    print("OK [curves-drawer] -- Plotly deferred; drawer markup; row-click + cache wired.")


def _check_visual_core_column_renders() -> None:
    """US-017: Visual column renders an <img loading="lazy"> when row.visual_core
    is non-null, and a missing-state placeholder otherwise. The renderer also
    declares the CSS class for the thumbnail."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    # JS function + classes must be present.
    assert "visualCoreHtml" in body, "JS must define visualCoreHtml renderer (US-017)"
    assert "visual-core-thumb" in body, "missing .visual-core-thumb class"
    assert "visual-core-missing" in body, "missing .visual-core-missing placeholder"
    assert 'loading="lazy"' in body, "Visual Core <img> must use loading='lazy'"
    # The Visual <th> sits between Level and Status in the column order.
    pos_level = body.index("<th>Level</th>")
    pos_visual = body.index("<th>Visual</th>")
    pos_status = body.index("<th>Status</th>")
    assert pos_level < pos_visual < pos_status, "Visual column must sit between Level and Status"
    print("OK [visual-core] -- Visual column rendered with lazy <img> + missing placeholder.")


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


def _check_history_fetch_uses_parent_relative_path() -> None:
    """The dashboard HTML lives at artifacts/Final_Exp.html and the run dirs
    at <repo>/runs/final/<tag>/. The lazy history fetch URL must therefore
    walk UP one level out of artifacts/ before descending into runs/final/,
    otherwise the browser resolves it as artifacts/runs/final/<tag>/ and
    silently 404s — which is exactly the bug that hid all learning curves
    from the Phase A resnet50 + densenet121 rows."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    # The corrected URL must be parent-relative.
    assert "'../runs/final/'" in body, (
        "history fetch URL must be parent-relative; got something else"
    )
    # The buggy in-place form must NOT appear anywhere in the JS.
    assert "'./runs/final/'" not in body, (
        "stale ./runs/final/ URL still present — dashboard at artifacts/ will 404"
    )
    print("OK [history-url] -- learning-curve fetch uses ../runs/final/<tag>/.")


def main() -> int:
    _check_summary_shape()
    _check_static_scaffold()
    _check_column_headers_in_order()
    _check_inline_js_contract()
    _check_inline_initial_data_embedded()
    _check_chip_filter_groups_present()
    _check_polling_and_timestamp_ui()
    _check_no_legacy_markup_leaks()
    _check_polling_cadence_is_10_minutes()
    _check_manual_refresh_and_countdown_pill_present()
    _check_visual_core_column_renders()
    _check_curves_drawer_lazy_loaded()
    _check_thumbs_dir_back_compat()
    _check_history_fetch_uses_parent_relative_path()
    print("\nAll dashboard scaffold checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
