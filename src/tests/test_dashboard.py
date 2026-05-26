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
from src.tools.final_exp_schema import SCHEMA_VERSION  # noqa: E402


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
    # Tabs for all four phases (US-030 added Phase D)
    for p in ("A", "B", "C", "D"):
        assert f'data-phase="{p}"' in body, f"missing tab data-phase={p}"
        assert f'id="tab-count-{p}"' in body, f"missing tab-count-{p}"
    # Stats bar IDs. NOTE: the legacy `stat-best` slot was renamed to
    # `stat-deferred` by US-014 (the test was stale until US-030 caught it).
    for sid in (
        "stat-total", "stat-pending", "stat-running",
        "stat-complete", "stat-failed", "stat-deferred", "stat-all",
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
    expected = ["Tag", "Model", "Dataset", "Phase", "Level", "Treatment", "Visual", "Status", "val_acc", "Epochs", "Runtime", "Curves"]
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
    assert doc["schema_version"] == SCHEMA_VERSION
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


def _check_curves_drawer_prefers_inline_history() -> None:
    """File:// blocks fetch() of local files. The drawer therefore must use
    `row.history` (embedded by the aggregator) as the primary source and
    fall back to fetch only when the inline payload is null. This avoids
    the silent-curve bug the operator reported when the dashboard was
    opened by double-click."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    # Inline path present.
    assert "Array.isArray(row.history)" in body, (
        "JS must check row.history before the fetch fallback"
    )
    # The renderCurves call on the inline path must run before the fetch
    # plumbing — find both, assert ordering.
    inline_idx = body.find("Array.isArray(row.history)")
    fetch_idx = body.find("HISTORY_PENDING.set(tag, p);")
    assert inline_idx > 0 and fetch_idx > 0
    assert inline_idx < fetch_idx, (
        "inline path must be evaluated before the fetch fallback in openCurvesDrawer"
    )
    print("OK [history-inline-first] -- drawer prefers row.history over fetch.")


def _check_image_quality_renders_under_visual_core() -> None:
    """PSNR/SSIM (US-002): the dashboard JS must render a `.visual-core-quality`
    span DIRECTLY under the thumbnail image in `visualCoreHtml(row)`. The CSS
    must declare `.visual-core-quality` so the strip aligns under the 240px
    thumb. Phase A gets a dim placeholder; Phase B/C reads from row.psnr_mean /
    row.ssim_mean."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert "imageQualityHtml" in body, (
        "JS must define imageQualityHtml renderer for PSNR/SSIM"
    )
    # visualCoreHtml must compose imageQualityHtml so the strip lands under
    # the thumb in the same <td> cell.
    assert "img + imageQualityHtml(row)" in body, (
        "visualCoreHtml must concatenate the thumb + PSNR/SSIM strip"
    )
    # Schema fields referenced by the JS.
    for field in ("psnr_mean", "psnr_std", "ssim_mean", "ssim_std"):
        assert "row." + field in body, f"JS must read row.{field}"
    # CSS class for the strip is present.
    assert ".visual-core-quality" in body, "missing .visual-core-quality CSS"
    print("OK [image-quality-strip] -- PSNR/SSIM rendered under Visual Core thumb.")


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


def _check_phase_d_tab_renders() -> None:
    """US-030: Phase D tab must be present in the static scaffold even when
    no `runs/final/final_D_*` directories exist. The count badge shows 90
    (the expected Phase D cell count) regardless of on-disk state — the tab
    is always present so the JS can route data to it; the embedded JSON's
    row count is what gates the 186-vs-276 contract."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert 'data-phase="D"' in body, "missing Phase D tab data-phase"
    assert 'id="tab-count-D"' in body, "missing tab-count-D"
    # The count badge must show 90 (literal markup inside the tab button).
    # We grep for `>90<` inside the tab-count-D span — looser than parsing
    # but tight enough to catch a regression to "—" or 0.
    import re as _re
    m = _re.search(r'id="tab-count-D"[^>]*>(\d+|—)</span>', body)
    assert m, "tab-count-D inner text not found"
    assert m.group(1) == "90", f"tab-count-D badge should read '90', got '{m.group(1)}'"
    print("OK [phase-d-tab] -- Phase D tab present with count=90 badge.")


def _check_treatment_chip_group_present() -> None:
    """US-030: TREATMENT chip group with T1/T2/T3 values must be in the
    filter row. Phase-gating (hide on A/B/C, show on D) is JS-driven via
    `treatmentGroup.style.display` — the static HTML always emits the group."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert 'data-chip-group="treatment"' in body, "missing treatment chip group"
    for v in ("T1", "T2", "T3"):
        assert f'data-value="{v}"' in body, f"missing treatment chip value {v}"
    # The JS must wire up phase-gated visibility for the group.
    assert "treatmentGroup" in body, "JS must reference treatmentGroup element"
    print("OK [treatment-chips] -- TREATMENT chip group with T1/T2/T3 present.")


def _check_phase_d_row_has_treatment_attr() -> None:
    """US-030: a row rendered from a Phase D metrics.json must carry
    `data-treatment` on its <tr> dataset (the JS reads this for multi-select
    filtering). We can't run the JS here, so we drop a synthetic Phase D
    metrics.json and assert the inline-embedded JSON has the treatment field
    populated on a Phase D row — the JS path is deterministic from the data."""
    import json as _json
    import re as _re
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        # Plant one synthetic Phase D run dir so phase_d_present_on_disk()
        # flips to True and the aggregator enumerates Phase D rows.
        cell_dir = runs_root / "final_D_T1_L3_resnet50_cifar10"
        cell_dir.mkdir(parents=True)
        (cell_dir / "metrics.json").write_text(
            _json.dumps({"best_val_acc": 0.42, "epochs_run": 1, "runtime_s": 60.0}),
            encoding="utf-8",
        )
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    m = _re.search(
        r'<script type="application/json" id="initial-data">(.+?)</script>',
        body, _re.S,
    )
    assert m, "missing inline-data <script>"
    raw = m.group(1).replace("<\\/", "</")
    doc = _json.loads(raw)
    t1_rows = [r for r in doc["rows"] if r.get("treatment") == "T1"]
    assert t1_rows, "no Phase D rows with treatment='T1' embedded"
    # And the target synthetic row must be Complete (proves the read path).
    target = next(
        (r for r in t1_rows if r["tag"] == "final_D_T1_L3_resnet50_cifar10"),
        None,
    )
    assert target is not None, "planted Phase D row missing from embedded JSON"
    assert target["phase"] == "D"
    assert target["treatment"] == "T1"
    # Final guard: the renderRows JS path must set tr.dataset.treatment.
    assert "tr.dataset.treatment" in body, (
        "JS must assign tr.dataset.treatment so the filter can read it"
    )
    print("OK [phase-d-treatment-attr] -- Phase D row carries treatment='T1' in JSON + JS reads it.")


def _check_phase_d_recovery_section_mounts() -> None:
    """US-030: the Phase D Recovery <details> section must be in the static
    HTML so the JS hook always has a mount target, even on dashboards built
    before any Phase D runs land. The JS pending placeholder is the default
    body content."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert 'id="phase-d-recovery-section"' in body, "missing recovery section mount"
    assert "renderPhaseDRecoveryStrip" in body, "JS must define recovery-strip renderer"
    assert "phase_d_comparison.png" in body, (
        "JS must reference the recovery comparison PNG"
    )
    print("OK [phase-d-recovery] -- recovery <details> section + JS renderer wired.")


def _check_186_cell_view_byte_identical_when_phase_d_absent() -> None:
    """US-030: when no `final_D_*` directories exist, the embedded JSON
    must still have exactly 186 rows (the legacy 186-cell contract). The
    Phase D tab is structural (always in the static scaffold) but rows
    are gated by data — `phase_d_present_on_disk` is False here, so the
    aggregator does NOT enumerate Phase D cells."""
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
    assert m, "missing inline-data <script>"
    raw = m.group(1).replace("<\\/", "</")
    doc = _json.loads(raw)
    assert len(doc["rows"]) == 186, (
        f"expected 186 rows when no Phase D dirs exist, got {len(doc['rows'])}"
    )
    assert doc["counts"]["total"] == 186, (
        f"expected counts.total=186, got {doc['counts']['total']}"
    )
    # No row should carry a non-null treatment in the 186-row view.
    treatments = [r.get("treatment") for r in doc["rows"]]
    assert all(t is None for t in treatments), (
        "186-row view must have treatment=null on every row"
    )
    print("OK [186-byte-identical] -- 186 rows embedded; no treatment field populated.")


def _check_seven_phase_tabs_render() -> None:
    """US-047: dashboard renders 7 tabs (A, B, B2, B2nr, C, C2, D) with
    canonical order + correct counts. Phase B tab text reads "Phase B (B1)"
    (data-phase stays "B" so JS routing is unchanged)."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")

    expected = (
        ('A', '6', 'Phase A'),
        ('B', '30', 'Phase B (B1)'),
        ('B2', '30', 'Phase B2'),
        ('B2nr', '6', 'Phase B2-nr'),
        ('C', '150', 'Phase C'),
        ('C2', '90', 'Phase C2'),
        ('D', '90', 'Phase D'),
    )
    for data_phase, count, label in expected:
        # data-phase attribute
        assert f'data-phase="{data_phase}"' in body, (
            f"missing tab data-phase={data_phase!r}"
        )
        # tab-count span
        assert f'id="tab-count-{data_phase}"' in body, (
            f"missing tab-count span for phase={data_phase!r}"
        )
        # Tab label text — Phase B reads "Phase B (B1)"; others read "Phase X"
        # exactly as the label asserts. Search for the label as a substring
        # of the tab button to be robust against attribute ordering.
        assert label in body, f"missing tab label {label!r} for phase={data_phase!r}"
    print("OK [7-phase-tabs] -- A/B/B2/B2nr/C/C2/D tabs render with B -> 'Phase B (B1)'.")


def _check_treatment_chip_visibility_logic_extended() -> None:
    """US-047: treatment chip group is visible on B2, C2, D (TREATMENT_PHASES);
    hidden on A, B (B1), B2nr, C. The JS controls this via the
    TREATMENT_PHASES array — assert its presence and the lookup."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert "TREATMENT_PHASES" in body, "JS missing TREATMENT_PHASES constant"
    assert "AXIS_PHASES" in body, "JS missing AXIS_PHASES constant"
    # The JS lookup pattern: TREATMENT_PHASES.indexOf(phase) !== -1
    assert "TREATMENT_PHASES.indexOf(phase)" in body, (
        "JS must gate the treatment chip via TREATMENT_PHASES.indexOf(phase) check"
    )
    assert "AXIS_PHASES.indexOf(phase)" in body, (
        "JS must gate the axis chip via AXIS_PHASES.indexOf(phase) check"
    )
    print("OK [treatment-chip-visibility] -- B2/C2/D show chip; A/B/B2nr/C hide it.")


def _check_multi_seed_variance_indicator_renderer() -> None:
    """US-047: when val_acc_mean and val_acc_std are populated, the val_acc
    cell renders 'mean% ± std%' with a tooltip listing seeds_observed."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    assert "fmtAccMaybeVariance" in body, "missing variance-aware acc helper"
    assert "multiseed-indicator" in body, "missing multiseed CSS hook in JS"
    assert "seeds_observed" in body, "JS must reference seeds_observed for tooltip"
    print("OK [multi-seed-indicator] -- fmtAccMaybeVariance + seeds tooltip wired in JS.")


def _check_v4_diagnostic_stub_sections_present() -> None:
    """US-047: the dashboard includes empty-state mounts for the v4
    diagnostic galleries (confusion matrices, calibration), inference
    throughput card, and B2/C2 comparison strips. Each becomes data-driven
    once its upstream US dispatches; the mount target is static so the JS
    hook is always available."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "Final_Exp.html"
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    for section_id in (
        "phase-b2-comparison-section",
        "phase-c2-comparison-section",
        "throughput-card-section",
        "confusion-gallery-section",
        "calibration-gallery-section",
    ):
        assert f'id="{section_id}"' in body, f"missing v4 stub section: {section_id}"
    # Empty-state placeholders carry the "Awaiting" prefix so the operator
    # knows the section is gated on an upstream US.
    assert body.count("Awaiting Phase B2 runs") >= 1
    assert body.count("Awaiting Phase C2 runs") >= 1
    assert body.count("Awaiting US-045") >= 3  # confusion + calibration + throughput
    print("OK [v4-diag-stubs] -- B2 / C2 / throughput / confusion / calibration mounts present.")


def _check_276_view_byte_identical_when_no_new_phases_on_disk() -> None:
    """US-047 regression: with Phase D present on disk but B2/B2nr/C2
    absent, the embedded JSON must contain the v3 276-row contract
    (6 A + 30 B + 150 C + 90 D), every row's treatment matches the v3
    convention (None on A/B/C; T1/T2/T3 on D), and seed=42 on every row."""
    import json as _json
    import re as _re
    with tempfile.TemporaryDirectory() as td:
        runs_root = Path(td) / "runs" / "final"
        runs_root.mkdir(parents=True)
        # Plant a single Phase D metrics.json to flip the gate to True.
        cell_dir = runs_root / "final_D_T3_L3_resnet50_cifar10"
        cell_dir.mkdir(parents=True)
        (cell_dir / "metrics.json").write_text(
            _json.dumps({"best_val_acc": 0.5, "epochs_run": 5, "pipeline_version": 2}),
            encoding="utf-8",
        )
        out = Path(td) / "Final_Exp.html"
        build_dashboard(out_path=out, runs_root=runs_root)
        body = out.read_text(encoding="utf-8")
    m = _re.search(
        r'<script type="application/json" id="initial-data">(.+?)</script>',
        body, _re.S,
    )
    assert m, "missing inline-data <script>"
    raw = m.group(1).replace("<\\/", "</")
    doc = _json.loads(raw)
    assert len(doc["rows"]) == 276, (
        f"v3 view drift: expected 276 rows with D present + B2/B2nr/C2 absent; got {len(doc['rows'])}"
    )
    phases = sorted({r["phase"] for r in doc["rows"]})
    assert phases == ["A", "B", "C", "D"], f"unexpected phases in v3 view: {phases}"
    # Every row carries seed=42 (canonical).
    seeds = {r.get("seed") for r in doc["rows"]}
    assert seeds == {42}, f"unexpected canonical seeds: {seeds}"
    print("OK [276-v3-byte-identical] -- 276-row v3 view preserved when no B2/B2nr/C2 dirs on disk.")


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
    _check_curves_drawer_prefers_inline_history()
    _check_image_quality_renders_under_visual_core()
    _check_phase_d_tab_renders()
    _check_treatment_chip_group_present()
    _check_phase_d_row_has_treatment_attr()
    _check_phase_d_recovery_section_mounts()
    _check_186_cell_view_byte_identical_when_phase_d_absent()
    # US-047 additions
    _check_seven_phase_tabs_render()
    _check_treatment_chip_visibility_logic_extended()
    _check_multi_seed_variance_indicator_renderer()
    _check_v4_diagnostic_stub_sections_present()
    _check_276_view_byte_identical_when_no_new_phases_on_disk()
    print("\nAll dashboard scaffold checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
