"""Build artifacts/Final_Exp.html — the FINAL_EXP Dashboard (US-004..US-007).

The successor to the legacy <details>-row dashboard. UI parity with the
pilot's "Experiment Plan Dashboard" (artifacts/dashboard_experiment_plan.html)
plus three Phase A/B/C tabs, a Level badge column with parameter tooltip,
and JSON-polling refresh (filters/sort/search added in US-008..US-014).

Architecture:
    runs/final/<tag>/metrics.json
              │
              ▼
    build_final_exp_json.py  ──► artifacts/Final_Exp.json
              │                                 ▲
              │            (browser fetch + 30s polling, US-013)
              ▼                                 │
    build_final_dashboard.py  ──► artifacts/Final_Exp.html

The HTML is a static, single-file artifact that loads `./Final_Exp.json`
at runtime via fetch — no server, no framework, no external JS deps
beyond what the pilot already loaded. Rows are rendered client-side so
the static HTML is small and the same template handles updates without
a Python rebuild.

Switched to torch-free `src.experiments.cells.iter_cells()` for
plan-row enumeration (the legacy implementation pulled in
`src.experiments.matrix.build_final_matrix`, which transitively imports
torch). Tools that don't need DegradeConfig should stay torch-free.

CLI:
    python -m src.tools.build_final_dashboard
    python -m src.tools.build_final_dashboard --out artifacts/Final_Exp.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.experiments.cells import (
    EXPECTED_COUNTS,
    EXPECTED_COUNTS_WITH_ALL,
    EXPECTED_COUNTS_WITH_D,
    EXPECTED_TOTAL,
    PHASE_D_TREATMENTS,
)


_DEFAULT_OUT = Path("artifacts/Final_Exp.html")
_DEFAULT_RUNS_ROOT = Path("runs/final")
# Sibling JSON the browser fetches at runtime. Relative path so the HTML
# works equally well via file:// and a static server.
_JSON_SIBLING_NAME = "Final_Exp.json"


# CSS variable block + component styles. Lifted from the pilot dashboard
# (artifacts/dashboard_experiment_plan.html) for visual parity, plus
# tab-bar and level-badge additions for US-006/US-007.
_CSS = r"""
:root {
    --bg:        #0b0d10;
    --surface:   #12151a;
    --surface2:  #181c22;
    --surface3:  #1e232b;
    --border:    #2a3040;
    --border-hi: #3a4560;
    --text:      #e8eaf0;
    --text-dim:  #8890a0;
    --accent:    #ffffff;
    --accent2:   #d0d4dc;
    --accent-dim: rgba(255,255,255,0.08);
    --green:     #4caf50;
    --green2:    #81c784;
    --orange:    #ffab40;
    --red:       #ff5252;
    --blue:      #4fc3f7;
    --purple:    #b39ddb;
    --L1:        #4caf50;
    --L2:        #9ccc65;
    --L3:        #ffab40;
    --L4:        #ff7043;
    --L5:        #ff5252;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html {
    /* 2026-05-13 v3: dropped sticky thead entirely after two failed
       offset attempts (76px → 96px both still clipped rows under scroll).
       Sticky table headers in a page-level scroll context always overlap
       body rows once they scroll past — the only way to *not* overlap is
       to not be sticky. The tab-bar remains sticky for phase navigation;
       the column-header row now scrolls away with the body. */
}
body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.6;
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 28px;
    padding: 28px 20px 22px;
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface3) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
}
.header h1 {
    font-size: 1.85em;
    color: var(--accent);
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}
.header .subtitle {
    color: var(--text-dim);
    font-size: 0.92em;
}
.header .timestamps {
    margin-top: 10px;
    font-size: 0.78em;
    color: var(--text-dim);
    letter-spacing: 0.3px;
}
.header .timestamps .ts-fresh { color: var(--green2); }
.header .timestamps .ts-stale { color: var(--orange); }
.header .timestamps .ts-error { color: var(--red); }
/* US-019: manual "Refresh now" button — visually distinct from the
   ambient text in the timestamp pill so the operator can spot it. */
.header .timestamps .manual-refresh-btn {
    margin-left: 12px;
    padding: 2px 10px;
    font-size: 0.78em;
    font-family: inherit;
    color: var(--accent2);
    background: var(--surface3);
    border: 1px solid var(--border-hi);
    border-radius: 6px;
    cursor: pointer;
}
.header .timestamps .manual-refresh-btn:hover {
    background: var(--surface2);
    color: var(--accent);
}
.header .timestamps .manual-refresh-btn:disabled {
    opacity: 0.5;
    cursor: progress;
}

/* Stats bar */
.stats-bar {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: var(--accent); }
.stat-value {
    font-size: 1.8em;
    font-weight: 700;
    color: var(--accent);
    line-height: 1.2;
}
.stat-value.pending  { color: var(--orange); }
.stat-value.running  { color: var(--blue); }
.stat-value.complete { color: var(--green2); }
.stat-value.failed   { color: var(--red); }
.stat-value.deferred { color: var(--purple); }
.stat-label {
    font-size: 0.74em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Phase tabs */
.tab-bar {
    position: sticky;
    top: 0;
    z-index: 5;
    display: flex;
    gap: 4px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 6px;
    margin-bottom: 16px;
    /* Pin tab-bar height so the sticky thead offset below (top:96px)
       is deterministic across browsers / zoom levels. */
    min-height: 64px;
    box-sizing: border-box;
}
.tab-btn {
    flex: 1;
    background: transparent;
    color: var(--text-dim);
    border: 1px solid transparent;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.92em;
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: 0.3px;
    font-weight: 500;
}
.tab-btn:hover {
    color: var(--text);
    background: var(--surface2);
}
.tab-btn.active {
    color: var(--accent);
    background: var(--surface2);
    border-color: var(--border-hi);
    font-weight: 600;
}
.tab-btn .tab-count {
    color: var(--text-dim);
    font-weight: 400;
    margin-left: 6px;
    font-size: 0.85em;
}
.tab-btn.active .tab-count { color: var(--text); }

/* Filter shell — populated in US-008..US-013 */
.filters {
    display: flex;
    gap: 14px;
    padding: 14px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 18px;
    flex-wrap: wrap;
    align-items: center;
}
.filters .filters-placeholder {
    color: var(--text-dim);
    font-size: 0.82em;
    font-style: italic;
}

/* Experiment table */
.exp-table-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}
.exp-table {
    width: 100%;
    border-collapse: collapse;
}
.exp-table th {
    background: var(--surface3);
    padding: 10px 14px;
    text-align: left;
    font-size: 0.72em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    /* NOT sticky — see comment on `html` above. Sticky thead was
       clipping the first body row under scroll. */
    user-select: none;
    white-space: nowrap;
}
.exp-table th:hover {
    color: var(--text);
    background: var(--surface2);
}
.exp-table td {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 0.92em;
    vertical-align: middle;
}
.exp-table tbody tr.exp-row { min-height: 140px; }
.exp-table tr.exp-row:hover { background: var(--accent-dim); }
.exp-table tr.exp-row.pending td { opacity: 0.55; }
.exp-table tr.exp-row.complete td { opacity: 1; }
.exp-table tr.exp-row.running td { opacity: 0.85; }
.exp-table tr.exp-row.failed td { opacity: 0.85; color: var(--red); }
.exp-table .id-cell {
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
    font-weight: 600;
    color: var(--blue);
    font-size: 0.82em;
    white-space: nowrap;
}
.exp-table .num-cell {
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
    text-align: right;
    white-space: nowrap;
}
.exp-table .acc-cell { color: var(--green2); font-weight: 600; }
.exp-table .acc-cell.acc-mid { color: var(--orange); }
.exp-table .acc-cell.acc-low { color: var(--red); }

/* Visual Core thumbnail (US-017) — Original|Degraded side-by-side preview.
   Source PNG is 224x448 (2:1), so 240x120 here is a clean 1.07x downscale
   that keeps both halves clearly inspectable in-row. */
.visual-core-thumb {
    display: block;
    width: 240px;
    height: 120px;
    object-fit: cover;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface3);
}
.visual-core-missing {
    display: inline-block;
    width: 240px;
    height: 120px;
    color: var(--text-dim);
    font-size: 0.82em;
    text-align: center;
    border: 1px dashed var(--border);
    border-radius: 4px;
    padding: 48px 0;
    box-sizing: border-box;
}
/* PSNR/SSIM strip rendered under the Visual Core thumbnail (US-002).
   Shows degradation quality of the right (degraded) half vs the left
   (original). Phase A clean cells skip the measurement (clean-vs-clean
   is identity) and render a dim '—' instead. */
.visual-core-quality {
    display: block;
    margin-top: 4px;
    width: 240px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 0.78em;
    color: var(--text-dim);
    text-align: center;
    white-space: nowrap;
}
.visual-core-quality .vq-label { color: var(--text-dim); }
.visual-core-quality .vq-value { color: var(--text); }
.visual-core-quality .vq-sep { color: var(--border); margin: 0 6px; }

/* Inline degradation-parameter strip (re-fix 2026-05-13). Shows all six
   degradation values per row at a glance — replaces the level-badge
   tooltip as the primary surface (the tooltip stays for power users). */
.deg-params {
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
    font-size: 0.82em;
    line-height: 1.5;
    white-space: nowrap;
    color: var(--text);
}
.deg-params .deg-line { display: block; }
.deg-params .lbl { color: var(--text-dim); margin-right: 3px; }
.deg-params .sep { color: var(--text-dim); margin: 0 6px; }
.deg-params.clean { color: var(--text-dim); font-style: italic; }

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.74em;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    border: 1px solid var(--border);
}
.status-pill.pending  { color: var(--orange); border-color: var(--orange); }
.status-pill.running  { color: var(--blue);   border-color: var(--blue); }
.status-pill.complete { color: var(--green2); border-color: var(--green2); }
.status-pill.failed   { color: var(--red);    border-color: var(--red); }
/* US-018 history indicator — small chart icon shown on rows that have a
   `runs/final/<tag>/history.json` available for the lazy Plotly drawer. */
.history-indicator {
    display: inline-block;
    width: 18px;
    text-align: center;
    color: var(--accent2);
    cursor: pointer;
    user-select: none;
    font-size: 0.92em;
}
.history-indicator.absent {
    color: var(--text-dim);
    cursor: default;
}
.exp-row { cursor: pointer; }

/* US-018 — right-side drawer for the lazy-loaded learning curves. */
.curves-drawer {
    position: fixed;
    top: 0;
    right: 0;
    width: min(640px, 92vw);
    height: 100vh;
    background: var(--surface);
    border-left: 1px solid var(--border-hi);
    box-shadow: -8px 0 24px rgba(0, 0, 0, 0.45);
    transform: translateX(100%);
    transition: transform 0.18s ease;
    z-index: 100;
    display: flex;
    flex-direction: column;
}
.curves-drawer.open { transform: translateX(0); }
.curves-drawer .drawer-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
}
.curves-drawer .drawer-title {
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
    font-size: 0.92em;
    color: var(--accent);
    word-break: break-all;
}
.curves-drawer .drawer-close {
    background: transparent;
    border: 0;
    color: var(--text-dim);
    font-size: 1.4em;
    cursor: pointer;
    padding: 0 6px;
}
.curves-drawer .drawer-close:hover { color: var(--accent); }
.curves-drawer .drawer-body {
    flex: 1;
    overflow-y: auto;
    padding: 12px 16px;
}
.curves-drawer .drawer-body .placeholder {
    color: var(--text-dim);
    font-size: 0.88em;
    padding: 20px 0;
    text-align: center;
}
.curves-drawer .plot-container {
    width: 100%;
    height: 280px;
    margin-bottom: 14px;
}

/* US-014 quarantine — TransNeXt rows pending hardware. Muted purple +
   dashed border distinguishes "deliberately deferred" from "active". */
.status-pill.deferred {
    color: var(--purple);
    border-color: var(--purple);
    border-style: dashed;
    background: rgba(179, 157, 219, 0.08);
}
.exp-row.deferred { opacity: 0.62; }
.exp-row.deferred:hover { opacity: 0.92; }

/* Level badge — US-007 */
.level-badge {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    min-width: 44px;
    padding: 3px 8px;
    border-radius: 8px;
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
    font-size: 0.78em;
    font-weight: 700;
    color: #0b0d10;
    background: var(--surface3);
    border: 1px solid var(--border);
    cursor: help;
}
.level-badge.lvl-clean { background: var(--surface2); color: var(--green2);
                          border-color: var(--border); font-weight: 600; }
.level-badge.lvl-1 { background: var(--L1); }
.level-badge.lvl-2 { background: var(--L2); }
.level-badge.lvl-3 { background: var(--L3); }
.level-badge.lvl-4 { background: var(--L4); color: #fff; }
.level-badge.lvl-5 { background: var(--L5); color: #fff; }
.level-badge .axis-caption {
    margin-top: 1px;
    font-size: 0.78em;
    font-weight: 500;
    text-transform: lowercase;
    opacity: 0.85;
    letter-spacing: 0.2px;
}

/* Filter chip row (US-010) */
.filters {
    flex-direction: column;
    align-items: stretch;
    gap: 10px;
}
.chip-group {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.chip-group .chip-label {
    font-size: 0.72em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    min-width: 78px;
}
.chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--text-dim);
    font-size: 0.78em;
    cursor: pointer;
    user-select: none;
    transition: all 0.12s;
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
}
.chip:hover { border-color: var(--accent); color: var(--text); }
.chip.active {
    background: var(--accent);
    color: #0b0d10;
    border-color: var(--accent);
    font-weight: 600;
}
.filters .filter-actions {
    display: flex;
    gap: 10px;
    margin-top: 4px;
    align-items: center;
}
.filters .clear-all {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-dim);
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.78em;
    cursor: pointer;
}
.filters .clear-all:hover { color: var(--text); border-color: var(--accent); }
.filters .visible-count {
    color: var(--text-dim);
    font-size: 0.78em;
    margin-left: auto;
}

/* Banner for fetch / data failures */
.banner {
    margin-bottom: 18px;
    padding: 12px 16px;
    background: var(--surface2);
    border: 1px solid var(--red);
    border-radius: 10px;
    color: var(--red);
    font-size: 0.88em;
    display: none;
}
.banner.visible { display: block; }
.banner.warn { border-color: var(--orange); color: var(--orange); }

/* Execution US Trend (US-006 line 418). Renders one card per closed
   execution US (Phase A: 2 cells; Phase B: 10 cells; Phase C: 50 cells).
   Each card lists the cells the US executed and their best_val_acc, sorted
   level → ascending; the card is only emitted if ≥ 1 cell is Complete. */
.us-trend-section {
    margin: 0 24px 18px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0;
}
.us-trend-section > summary {
    cursor: pointer;
    padding: 12px 16px;
    font-size: 0.95em;
    color: var(--text);
    list-style: none;
    user-select: none;
}
.us-trend-section > summary::-webkit-details-marker { display: none; }
.us-trend-section > summary::before {
    content: '▸ ';
    color: var(--text-dim);
    display: inline-block;
    width: 1em;
    transition: transform 0.15s;
}
.us-trend-section[open] > summary::before { content: '▾ '; }
.us-trend-body {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 4px 16px 14px;
}
.us-trend-card {
    flex: 1 1 280px;
    min-width: 260px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
}
.us-trend-card .ut-title {
    font-weight: 600;
    color: var(--text);
    font-size: 0.92em;
    margin-bottom: 2px;
}
.us-trend-card .ut-subtitle {
    color: var(--text-dim);
    font-size: 0.78em;
    margin-bottom: 8px;
}
.us-trend-card table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82em;
}
.us-trend-card td {
    padding: 3px 6px;
    border-bottom: 1px dotted var(--border);
}
.us-trend-card tr:last-child td { border-bottom: none; }
.us-trend-card td.ut-cell-tag { color: var(--text-dim); font-family: monospace; }
.us-trend-card td.ut-cell-acc { text-align: right; font-variant-numeric: tabular-nums; }
.us-trend-card .ut-pending { color: var(--text-dim); font-style: italic; font-size: 0.82em; }

/* V6 restructure (2026-07-02) — the tab bar is split into two labeled
   groups: the phases the V6 summary report treats in depth (B1 + C2) and
   the supporting phases. Same .tab-btn elements, same data-phase routing. */
.tab-bar { gap: 14px; align-items: stretch; }
.tab-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 4px 6px;
    border-radius: 8px;
}
.tab-group-v6 {
    flex: 2 1 0;
    background: rgba(79, 195, 247, 0.05);
    border: 1px solid var(--border-hi);
}
.tab-group-support { flex: 5 1 0; border: 1px solid transparent; }
.tab-group-label {
    font-size: 0.66em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 2px 8px 0;
    white-space: nowrap;
}
.tab-group-v6 .tab-group-label { color: var(--blue); }
.tab-group-btns { display: flex; gap: 4px; flex: 1; }
.tab-group-btns .tab-btn { flex: 1; }
.tab-group-v6 .tab-btn.active { border-color: var(--blue); }

/* Per-phase intro block — motivation + V6 findings + figures + charts.
   Rendered by JS (renderPhaseIntro) whenever the active phase changes. */
.phase-intro {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 18px;
}
.phase-intro h2 {
    font-size: 1.15em;
    color: var(--accent);
    letter-spacing: -0.2px;
    margin-bottom: 2px;
}
.phase-intro .pi-badge {
    display: inline-block;
    margin-left: 10px;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.6em;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    vertical-align: middle;
}
.pi-badge.deep  { color: var(--blue);     border: 1px solid var(--blue); }
.pi-badge.brief { color: var(--text-dim); border: 1px solid var(--border-hi); }
.phase-intro .pi-cells { color: var(--text-dim); font-size: 0.8em; margin-bottom: 12px; }
.phase-intro .pi-cols {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 18px;
    margin-bottom: 6px;
}
.phase-intro .pi-block h3 {
    font-size: 0.78em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}
.phase-intro .pi-block p, .phase-intro .pi-block li {
    font-size: 0.88em;
    line-height: 1.55;
    color: var(--text);
}
.phase-intro .pi-block ul { padding-left: 18px; }
.phase-intro .pi-block li { margin-bottom: 4px; }
.phase-intro .pi-num { color: var(--green2); font-weight: 600; font-variant-numeric: tabular-nums; }
.phase-intro .pi-neg { color: var(--orange); font-weight: 600; font-variant-numeric: tabular-nums; }

/* Static V6 figure gallery inside the intro */
.pi-figures {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 14px;
}
.pi-figure {
    flex: 1 1 380px;
    max-width: 640px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
}
.pi-figure img {
    display: block;
    width: 100%;
    height: auto;
    border-radius: 4px;
    background: #fff;
}
.pi-figure .pi-caption {
    margin-top: 6px;
    font-size: 0.76em;
    color: var(--text-dim);
    line-height: 1.4;
}
.pi-figure a { color: inherit; text-decoration: none; }

/* Interactive chart panels (Plotly) */
.pi-charts {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 14px;
    margin-top: 14px;
}
.chart-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px 4px;
}
.chart-card .chart-title {
    font-size: 0.84em;
    color: var(--text);
    font-weight: 600;
    margin-bottom: 2px;
}
.chart-card .chart-sub { font-size: 0.72em; color: var(--text-dim); margin-bottom: 6px; }
.chart-card .chart-plot { width: 100%; height: 340px; }
.chart-card .chart-placeholder {
    color: var(--text-dim);
    font-size: 0.82em;
    font-style: italic;
    padding: 40px 0;
    text-align: center;
}
/* Axis selector for the C2 per-model chart */
.axis-select { display: flex; gap: 6px; margin: 4px 0 8px; }
.axis-select .axis-btn {
    padding: 3px 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--surface3);
    color: var(--text-dim);
    font-size: 0.76em;
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
    cursor: pointer;
}
.axis-select .axis-btn.active {
    color: var(--accent);
    border-color: var(--border-hi);
    background: var(--surface2);
    font-weight: 600;
}

/* Chart controls row (metric selector) */
.chart-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.chart-controls-label {
    font-size: 0.72em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.chart-controls .axis-select { margin: 0; }
.chart-controls-hint { font-size: 0.72em; color: var(--text-dim); flex: 1 1 320px; }

/* Overview tab */
.tab-group-ov {
    flex: 1 1 0;
    background: rgba(129, 199, 132, 0.05);
    border: 1px solid var(--border-hi);
}
.tab-group-ov .tab-group-label { color: var(--green2); }
.tab-group-ov .tab-btn.active { border-color: var(--green2); }
.ov-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 12px;
    margin: 12px 0;
}
.ov-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
}
.ov-card h4 { color: var(--accent); font-size: 0.92em; margin-bottom: 6px; }
.ov-card p { font-size: 0.84em; color: var(--text); line-height: 1.55; }
.ov-note { margin-top: 12px; font-size: 0.84em; color: var(--text-dim); }

/* Sortable V6 tables */
.pi-tables { margin-top: 14px; }
.pi-tables > summary {
    cursor: pointer;
    font-size: 0.84em;
    color: var(--text);
    padding: 8px 10px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    user-select: none;
}
.pi-tables[open] > summary { border-radius: 8px 8px 0 0; }
.vt-wrap {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-top: none;
    padding: 10px 12px;
    overflow-x: auto;
}
.vt-caption { font-size: 0.76em; color: var(--text-dim); margin-bottom: 6px; }
.v6-table { width: 100%; border-collapse: collapse; font-size: 0.82em; }
.v6-table th {
    background: var(--surface3);
    color: var(--text-dim);
    text-transform: uppercase;
    font-size: 0.82em;
    letter-spacing: 0.4px;
    padding: 6px 10px;
    text-align: left;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
    border-bottom: 1px solid var(--border);
}
.v6-table th:hover { color: var(--text); }
.v6-table th.vt-sorted { color: var(--blue); }
.v6-table td { padding: 5px 10px; border-bottom: 1px dotted var(--border); }
.v6-table .vt-num {
    text-align: right;
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.v6-table .vt-std { color: var(--text-dim); font-size: 0.86em; }
.v6-table .vt-neg { color: var(--orange); }
.v6-table tr.vt-sub td { color: var(--text-dim); }

/* Lightbox */
.lightbox {
    position: fixed;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(4, 6, 8, 0.88);
    z-index: 300;
    cursor: zoom-out;
}
.lightbox.open { display: flex; }
.lightbox figure {
    max-width: 94vw;
    max-height: 94vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}
.lightbox img {
    max-width: 94vw;
    max-height: 88vh;
    object-fit: contain;
    border: 1px solid var(--border-hi);
    border-radius: 6px;
    background: #fff;
}
.lightbox figcaption {
    color: var(--text-dim);
    font-size: 0.8em;
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
    text-align: center;
    word-break: break-all;
    max-width: 90vw;
}
/* zoom affordance on every lightbox-able image */
img.visual-core-thumb, .pi-figure img, .drawer-samples img { cursor: zoom-in; }

/* Drawer same-sample strip */
.drawer-samples-title {
    font-size: 0.76em;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.drawer-samples {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 8px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
.drawer-samples .ds-item { flex: 0 0 auto; text-align: center; }
.drawer-samples .ds-item img {
    width: 136px;
    height: 68px;
    object-fit: cover;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface3);
    display: block;
}
.drawer-samples .ds-item.current img { outline: 2px solid var(--blue); }
.drawer-samples .ds-lbl {
    font-size: 0.68em;
    color: var(--text-dim);
    margin-top: 2px;
    font-family: 'Consolas', 'Monaco', ui-monospace, monospace;
}
.drawer-samples .ds-item.current .ds-lbl { color: var(--blue); }
.drawer-samples .ds-missing { color: var(--text-dim); }
"""

# Inline JS: bootstraps from <script id="initial-data"> JSON (works on file://
# where fetch() of local files is CORS-blocked), then polls ./Final_Exp.json
# every 30 s for live updates (works when served via HTTP).
# US-005 (load + render), US-006 (tabs), US-007 (level badge),
# US-010 (chip filters), US-013 (polling), US-014a (last-refreshed timestamp).
_JS = r"""
(function () {
    'use strict';

    const JSON_PATH = './Final_Exp.json';
    const LS_ACTIVE_PHASE = 'final_exp.active_phase';
    const LS_FILTERS_PREFIX = 'final_exp.filters.';
    // US-047 (v4): 7 phases A → B → B2 → B2nr → C → C2 → D. Phase B is
    // labeled "Phase B (B1)" in the tab text but data-phase stays "B".
    // 2026-07-02: 'OV' is the synthetic Overview tab (no cells of its own —
    // it renders the campaign conclusions + model comparison instead of
    // the experiment table).
    const PHASES = ['OV', 'A', 'B', 'B2', 'B2nr', 'C', 'C2', 'D'];
    // Treatment chip is visible only on regularized phases (B2, C2, D).
    const TREATMENT_PHASES = ['B2', 'C2', 'D'];
    // Axis chip is meaningful on Phase C (legacy isolation) and Phase C2
    // (THz-protocol isolation; restricted axes).
    const AXIS_PHASES = ['C', 'C2'];
    // US-019: 10-minute polling cadence. The aggregator pushes per-cell
    // updates after every Trainer.fit (incremental --cell mode), so the
    // dashboard does NOT need a tight polling loop — once every 10 min is
    // sufficient and avoids burning CPU on an idle tab.
    const POLL_INTERVAL_MS = 600000;
    const STALE_THRESHOLD_MS = 1800000; // visibility-paused > 30 min -> orange

    const state = {
        rows: [],
        countsByPhase: { A: {}, B: {}, C: {}, D: {} },
        activePhase: 'A',
        filters: { A: {}, B: {}, C: {}, D: {} },
        lastGeneratedAt: null,
        lastPolledMs: null,
        pollTimer: null,
        relTimer: null,
        fetchEverSucceeded: false,
        bootstrapFromInline: false,
    };

    // -- Helpers -----------------------------------------------------------

    function fmtAcc(v) {
        if (v === null || v === undefined) return '—';
        return (v * 100).toFixed(2) + '%';
    }

    // US-047 (v4): multi-seed variance indicator. When val_acc_mean and
    // val_acc_std are populated (≥ 2 audit seeds), render as "mean ± std%"
    // with a tooltip listing seeds_observed. Falls back to fmtAcc(v).
    function fmtAccMaybeVariance(row) {
        var m = row.val_acc_mean, s = row.val_acc_std;
        if (m === null || m === undefined || s === null || s === undefined) {
            return fmtAcc(row.val_acc);
        }
        var seeds = (row.seeds_observed || []).join(', ');
        var html = (m * 100).toFixed(2) + '% ± ' + (s * 100).toFixed(2) + '%';
        return '<span class="multiseed-indicator" title="seeds: ' +
            seeds + '">' + html + '</span>';
    }

    function accClass(v) {
        if (v === null || v === undefined) return '';
        if (v >= 0.85) return '';
        if (v >= 0.60) return 'acc-mid';
        return 'acc-low';
    }

    function fmtRuntime(s) {
        if (s === null || s === undefined) return '—';
        if (s < 60) return s.toFixed(0) + 's';
        if (s < 3600) return (s / 60).toFixed(1) + 'm';
        return (s / 3600).toFixed(1) + 'h';
    }

    function levelBadgeHtml(row) {
        let cls, label, caption = '';
        if (row.level === null || row.level === undefined) {
            cls = 'lvl-clean';
            label = 'clean';
        } else {
            cls = 'lvl-' + row.level;
            label = 'L' + row.level;
            if (row.phase === 'C' && row.axis) {
                caption = '<span class="axis-caption">' + row.axis + '</span>';
            }
        }
        const p = row.params || {};
        const tip = (
            'low_res=' + p.low_res +
            ' | blur=' + p.blur_kernel + 'x' + p.blur_kernel +
            ' σ=' + (p.blur_sigma !== undefined ? p.blur_sigma : '?') +
            ' | noise=' + p.noise_std +
            ' | S&P=' + p.salt_pepper +
            ' | sat=' + p.saturation
        );
        return '<span class="level-badge ' + cls +
               '" title="' + tip + '">' + label + caption + '</span>';
    }

    function paramsHtml(row) {
        // Re-fix 2026-05-13: render all six degradation values inline
        // (was tooltip-only). Two-line compact strip; Phase A clean rows
        // show "— clean —".
        if (row.level === null || row.level === undefined) {
            return '<span class="deg-params clean">— clean —</span>';
        }
        const p = row.params || {};
        const sep = '<span class="sep">·</span>';
        const line1 = (
            '<span class="lbl">res</span>' + p.low_res +
            sep +
            '<span class="lbl">blur</span>' + p.blur_kernel + '×' + p.blur_kernel +
            ' σ' + (p.blur_sigma !== undefined ? p.blur_sigma.toFixed(2) : '?') +
            sep +
            '<span class="lbl">noise</span>' + p.noise_std.toFixed(2)
        );
        const line2 = (
            '<span class="lbl">S&amp;P</span>' + p.salt_pepper.toFixed(2) +
            sep +
            '<span class="lbl">sat</span>' + p.saturation.toFixed(2)
        );
        return (
            '<span class="deg-params">' +
              '<span class="deg-line">' + line1 + '</span>' +
              '<span class="deg-line">' + line2 + '</span>' +
            '</span>'
        );
    }

    function imageQualityHtml(row) {
        // PSNR/SSIM line shown under the Visual Core thumbnail. Reads
        // psnr_mean / psnr_std / ssim_mean / ssim_std from the row schema
        // (populated from runs/final/<tag>/image_quality.json). Phase A
        // clean baselines skip the measurement (clean-vs-clean is identity)
        // and render a dim placeholder so the row height stays uniform.
        const phaseAClean = (row.phase === 'A');
        const hasPSNR = typeof row.psnr_mean === 'number';
        const hasSSIM = typeof row.ssim_mean === 'number';
        if (phaseAClean && !hasPSNR && !hasSSIM) {
            return '<span class="visual-core-quality" title="Phase A is a clean baseline — PSNR/SSIM not measured (identity comparison)">' +
                   '<span class="vq-label">PSNR</span> <span class="vq-value">—</span>' +
                   '<span class="vq-sep">|</span>' +
                   '<span class="vq-label">SSIM</span> <span class="vq-value">—</span>' +
                   '</span>';
        }
        if (!hasPSNR && !hasSSIM) {
            return '<span class="visual-core-quality" title="Run scripts/refresh_trackers.py --cell '+ row.tag +' to populate PSNR/SSIM">' +
                   '<span class="vq-label">PSNR</span> <span class="vq-value">—</span>' +
                   '<span class="vq-sep">|</span>' +
                   '<span class="vq-label">SSIM</span> <span class="vq-value">—</span>' +
                   '</span>';
        }
        let psnrText = '—';
        if (hasPSNR) {
            psnrText = row.psnr_mean.toFixed(2);
            if (typeof row.psnr_std === 'number') {
                psnrText += '±' + row.psnr_std.toFixed(2);
            }
            psnrText += ' dB';
        }
        let ssimText = '—';
        if (hasSSIM) {
            ssimText = row.ssim_mean.toFixed(3);
            if (typeof row.ssim_std === 'number') {
                ssimText += '±' + row.ssim_std.toFixed(3);
            }
        }
        const tooltip = 'Image quality of the degraded sample vs the clean original (image_quality.json).';
        return '<span class="visual-core-quality" title="' + tooltip + '">' +
               '<span class="vq-label">PSNR</span> <span class="vq-value">' + psnrText + '</span>' +
               '<span class="vq-sep">|</span>' +
               '<span class="vq-label">SSIM</span> <span class="vq-value">' + ssimText + '</span>' +
               '</span>';
    }

    function visualCoreHtml(row) {
        // US-017: lazy-loaded side-by-side Original|Degraded preview, plus a
        // PSNR/SSIM strip rendered directly underneath (US-002).
        const alt = 'Original | Degraded preview for ' + row.tag;
        let img;
        if (!row.visual_core) {
            img = '<span class="visual-core-missing" title="Visual Core PNG not yet rendered — run scripts/refresh_trackers.py">—</span>';
        } else {
            img = '<img class="visual-core-thumb" loading="lazy" decoding="async" ' +
                  'src="' + row.visual_core + '" alt="' + alt + '" title="' + alt + '">';
        }
        return img + imageQualityHtml(row);
    }

    function historyIndicatorHtml(row) {
        // US-018: show 📈 on rows with a learning curve, ▫ otherwise. Click
        // is bound at the <tr> level (any cell click opens the drawer).
        if (row.has_history) {
            return '<span class="history-indicator" title="Click row to see learning curves">📈</span>';
        }
        return '<span class="history-indicator absent" title="No learning curves yet — cell is ' +
               row.status + '">▫</span>';
    }

    function statusPillHtml(status, row) {
        const cls = status.toLowerCase();
        if (row && row.quarantined) {
            const reason = row.quarantine_reason || 'Quarantined';
            const tip = 'Deferred &mdash; ' + reason +
                        ' (US-014: excluded from execution)';
            return '<span class="status-pill deferred" title="' + tip +
                   '">Deferred &middot; ' + reason + '</span>';
        }
        return '<span class="status-pill ' + cls + '">' + status + '</span>';
    }

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    // -- Rendering ---------------------------------------------------------

    function renderRows(rows) {
        const tbody = document.getElementById('exp-tbody');
        const frag = document.createDocumentFragment();
        rows.forEach(function (row) {
            const tr = document.createElement('tr');
            tr.className = 'exp-row ' + row.status.toLowerCase();
            tr.dataset.tag = row.tag;
            tr.dataset.phase = row.phase;
            tr.dataset.model = row.model;
            tr.dataset.dataset = row.dataset;
            tr.dataset.status = row.status;
            tr.dataset.level = row.level === null ? 'clean' : ('L' + row.level);
            tr.dataset.axis = row.axis || '';
            tr.dataset.treatment = row.treatment || '';
            tr.innerHTML = (
                '<td class="id-cell">' + escapeHtml(row.tag) + '</td>' +
                '<td>' + escapeHtml(row.model) + '</td>' +
                '<td>' + escapeHtml(row.dataset) + '</td>' +
                '<td>' + escapeHtml(row.phase) + '</td>' +
                '<td>' + levelBadgeHtml(row) + '</td>' +
                '<td>' + (row.treatment ? escapeHtml(row.treatment) : '—') + '</td>' +
                '<td>' + paramsHtml(row) + '</td>' +
                '<td>' + visualCoreHtml(row) + '</td>' +
                '<td>' + statusPillHtml(row.status, row) + '</td>' +
                '<td class="num-cell acc-cell ' + accClass(row.val_acc) + '">' +
                    fmtAccMaybeVariance(row) + '</td>' +
                '<td class="num-cell">' +
                    (row.epochs_run !== null && row.epochs_run !== undefined ? row.epochs_run : '—') +
                '</td>' +
                '<td class="num-cell">' + fmtRuntime(row.runtime_s) + '</td>' +
                '<td>' + historyIndicatorHtml(row) + '</td>'
            );
            frag.appendChild(tr);
        });
        tbody.replaceChildren(frag);
    }

    function recomputeCountsByPhase(rows) {
        const out = { A: {}, B: {}, C: {}, D: {} };
        PHASES.forEach(function (p) {
            out[p] = { total: 0, pending: 0, running: 0, complete: 0, failed: 0,
                       deferred: 0, best_val_acc: null };
        });
        rows.forEach(function (r) {
            const c = out[r.phase];
            if (!c) return;
            c.total += 1;
            const k = r.status.toLowerCase();
            if (c[k] !== undefined) c[k] += 1;
            if (r.val_acc !== null && r.val_acc !== undefined) {
                if (c.best_val_acc === null || r.val_acc > c.best_val_acc) {
                    c.best_val_acc = r.val_acc;
                }
            }
        });
        return out;
    }

    function renderStatsBar(activePhase) {
        let c;
        if (activePhase === 'OV') {
            // Overview aggregates the whole campaign.
            c = { total: state.rows.length, pending: 0, running: 0,
                  complete: 0, failed: 0, deferred: 0 };
            state.rows.forEach(function (r) {
                const k = r.status.toLowerCase();
                if (c[k] !== undefined) c[k] += 1;
            });
        } else {
            c = state.countsByPhase[activePhase] || {};
        }
        const setStat = function (id, v) {
            const el = document.getElementById(id);
            if (el) el.textContent = v;
        };
        setStat('stat-total',    c.total !== undefined ? c.total : '—');
        setStat('stat-pending',  c.pending !== undefined ? c.pending : '—');
        setStat('stat-running',  c.running !== undefined ? c.running : '—');
        setStat('stat-complete', c.complete !== undefined ? c.complete : '—');
        setStat('stat-failed',   c.failed !== undefined ? c.failed : '—');
        setStat('stat-deferred', c.deferred !== undefined ? c.deferred : '—');
        // Global "All 186" indicator
        const all = state.rows.length;
        setStat('stat-all', all);
    }

    function rowMatchesFilters(tr, filters) {
        const f = filters || {};
        // Empty array (or undefined) means "All" for that facet.
        function pass(facet, value) {
            const sel = f[facet];
            if (!sel || sel.length === 0) return true;
            return sel.indexOf(value) !== -1;
        }
        if (!pass('model', tr.dataset.model)) return false;
        if (!pass('dataset', tr.dataset.dataset)) return false;
        if (!pass('status', tr.dataset.status)) return false;
        // Axis filter only constrains Phase C rows; Phase A/B rows pass through.
        if (tr.dataset.phase === 'C') {
            const sel = f.axis || [];
            if (sel.length > 0 && sel.indexOf(tr.dataset.axis) === -1) return false;
        }
        // Treatment filter only constrains Phase D rows (US-030).
        if (tr.dataset.phase === 'D') {
            const sel = f.treatment || [];
            if (sel.length > 0 && sel.indexOf(tr.dataset.treatment) === -1) return false;
        }
        return true;
    }

    function applyVisibility() {
        const phase = state.activePhase;
        const f = state.filters[phase] || {};
        let visible = 0;
        document.querySelectorAll('tr.exp-row').forEach(function (tr) {
            const onPhase = (tr.dataset.phase === phase);
            const matched = onPhase && rowMatchesFilters(tr, f);
            tr.style.display = matched ? '' : 'none';
            if (matched) visible += 1;
        });
        const vc = document.getElementById('visible-count');
        if (vc) vc.textContent = visible + ' visible';
    }

    function applyActivePhase(phase) {
        if (PHASES.indexOf(phase) === -1) phase = 'OV';
        state.activePhase = phase;
        try { localStorage.setItem(LS_ACTIVE_PHASE, phase); } catch (e) { /* ignore */ }

        // Tab styling
        document.querySelectorAll('.tab-btn').forEach(function (b) {
            b.classList.toggle('active', b.dataset.phase === phase);
        });
        // Overview is a synthetic tab: it has no table rows, so the filter
        // strip + experiment table are hidden while it is active.
        const isOverview = (phase === 'OV');
        ['.filters', '.exp-table-wrap'].forEach(function (sel) {
            const n = document.querySelector(sel);
            if (n) n.style.display = isOverview ? 'none' : '';
        });
        // Axis chip group is meaningful on Phase C and Phase C2 — hide elsewhere.
        const axisGroup = document.querySelector('[data-chip-group="axis"]');
        if (axisGroup) axisGroup.style.display = (AXIS_PHASES.indexOf(phase) !== -1) ? '' : 'none';
        // Treatment chip group is meaningful on regularized phases (B2/C2/D).
        const treatmentGroup = document.querySelector('[data-chip-group="treatment"]');
        if (treatmentGroup) treatmentGroup.style.display = (TREATMENT_PHASES.indexOf(phase) !== -1) ? '' : 'none';

        // Restore this phase's chip selections in the UI.
        renderChipSelections();
        applyVisibility();
        renderStatsBar(phase);
        renderPhaseIntro(phase);
    }

    function renderTabCounts() {
        PHASES.forEach(function (p) {
            const el = document.getElementById('tab-count-' + p);
            if (!el) return;
            if (p === 'OV') { el.textContent = state.rows.length || '—'; return; }
            const c = state.countsByPhase[p] || {};
            el.textContent = c.total !== undefined ? c.total : '—';
        });
    }

    // -- Execution US Trend (US-006 line 418) ------------------------------
    // One card per execution US in EXECUTION_US_GROUPS that has ≥ 1 Complete
    // cell. Cells inside a card are sorted by level (clean < L1 < … < L5)
    // and then by dataset. Pending USs are skipped silently so the section
    // grows monotonically as each phase closes.
    const EXECUTION_US_GROUPS = [
        { id: 'US-006', phase: 'A', model: 'resnet50',       title: 'Phase A · resnet50' },
        { id: 'US-007', phase: 'A', model: 'densenet121',    title: 'Phase A · densenet121' },
        { id: 'US-008', phase: 'A', model: 'transnext_tiny', title: 'Phase A · transnext_tiny' },
        { id: 'US-009', phase: 'B', model: 'resnet50',       title: 'Phase B · resnet50' },
        { id: 'US-010', phase: 'B', model: 'densenet121',    title: 'Phase B · densenet121' },
        { id: 'US-011', phase: 'B', model: 'transnext_tiny', title: 'Phase B · transnext_tiny' },
        { id: 'US-012', phase: 'C', model: 'resnet50',       title: 'Phase C · resnet50' },
        { id: 'US-013', phase: 'C', model: 'densenet121',    title: 'Phase C · densenet121' },
        { id: 'US-014', phase: 'C', model: 'transnext_tiny', title: 'Phase C · transnext_tiny' }
    ];

    function levelSortKey(row) {
        // clean=−1, L1..L5 = 1..5, axis breaks ties alphabetically.
        const lvl = (row.level === null || row.level === undefined) ? -1 : row.level;
        const ax = row.axis || '';
        return lvl * 100 + (ax ? ax.charCodeAt(0) : 0);
    }

    function renderExecutionUsTrendCard(group, rows) {
        const cells = rows.filter(function (r) {
            return r.phase === group.phase
                && r.model === group.model
                && r.status === 'Complete';
        });
        if (!cells.length) return '';
        cells.sort(function (a, b) {
            const k = levelSortKey(a) - levelSortKey(b);
            if (k !== 0) return k;
            return a.dataset < b.dataset ? -1 : (a.dataset > b.dataset ? 1 : 0);
        });
        const rowsHtml = cells.map(function (r) {
            const accCls = accClass(r.val_acc);
            const lvlLabel = (r.level === null || r.level === undefined)
                ? 'clean'
                : ('L' + r.level + (r.phase === 'C' && r.axis ? '·' + r.axis : ''));
            return '<tr>'
                + '<td class="ut-cell-tag">' + escapeHtml(r.dataset) + '</td>'
                + '<td>' + escapeHtml(lvlLabel) + '</td>'
                + '<td class="ut-cell-acc ' + accCls + '">' + fmtAcc(r.val_acc) + '</td>'
                + '</tr>';
        }).join('');
        return '<div class="us-trend-card">'
            + '<div class="ut-title">' + escapeHtml(group.id) + ' — ' + escapeHtml(group.title) + '</div>'
            + '<div class="ut-subtitle">' + cells.length + ' complete cell' + (cells.length === 1 ? '' : 's') + '</div>'
            + '<table><tbody>' + rowsHtml + '</tbody></table>'
            + '</div>';
    }

    function renderExecutionUsTrend(rows) {
        const body = document.getElementById('us-trend-body');
        if (!body) return;
        const cards = EXECUTION_US_GROUPS
            .map(function (g) { return renderExecutionUsTrendCard(g, rows); })
            .filter(function (h) { return h.length > 0; });
        if (!cards.length) {
            body.innerHTML = '<span class="ut-pending">No closed execution US yet.</span>';
            return;
        }
        body.innerHTML = cards.join('');
    }

    // -- Phase D Recovery strip (US-030) -----------------------------------
    // Embeds artifacts/figures/phase_d_comparison.png once at least one
    // Phase D cell completes. Pending placeholder until then so the
    // disclosure-section still mounts even on zero-D dashboards.
    function renderPhaseDRecoveryStrip(rows) {
        const section = document.getElementById('phase-d-recovery-section');
        if (!section) return;
        const body = section.querySelector('.us-trend-body');
        if (!body) return;
        const completeD = rows.filter(function (r) {
            return r.phase === 'D' && r.status === 'Complete';
        });
        if (completeD.length === 0) {
            body.innerHTML = '<span class="ut-pending">Awaiting Phase D runs.</span>';
            return;
        }
        body.innerHTML =
            '<img src="figures/phase_d_comparison.png" ' +
                 'alt="Phase D recovery summary" ' +
                 'style="max-width: 100%; height: auto; border-radius: 6px;">';
    }

    // -- v4 strips (US-048): Phase B2, Phase C2, Throughput, Confusion, Calibration

    function renderPhaseB2ComparisonStrip(rows) {
        const section = document.getElementById('phase-b2-comparison-section');
        if (!section) return;
        const body = section.querySelector('.us-trend-body');
        if (!body) return;
        const hasB2 = rows.some(function (r) { return r.phase === 'B2' && r.status === 'Complete'; });
        if (!hasB2) {
            body.innerHTML = '<span class="ut-pending">Awaiting Phase B2 runs (US-040 / US-041).</span>';
            return;
        }
        body.innerHTML =
            '<img src="figures/phase_b2_comparison_summary.png" ' +
                 'alt="Phase B2 comparison summary" ' +
                 'style="max-width: 100%; height: auto; border-radius: 6px;">';
    }

    function renderPhaseC2AttributionStrip(rows) {
        const section = document.getElementById('phase-c2-comparison-section');
        if (!section) return;
        const body = section.querySelector('.us-trend-body');
        if (!body) return;
        const hasC2 = rows.some(function (r) { return r.phase === 'C2' && r.status === 'Complete'; });
        if (!hasC2) {
            body.innerHTML = '<span class="ut-pending">Awaiting Phase C2 runs (US-044).</span>';
            return;
        }
        body.innerHTML =
            '<img src="figures/phase_c2_attribution_summary.png" ' +
                 'alt="Phase C2 axis attribution summary" ' +
                 'style="max-width: 100%; height: auto; border-radius: 6px;">';
    }

    function renderThroughputCard() {
        // Probe via an <img> element rather than fetch(): fetch() of local
        // files is CORS-blocked on file://, which made this card show the
        // pending placeholder forever on double-click-opened dashboards
        // even though the PNG existed (fixed 2026-07-02).
        const body = document.getElementById('throughput-body');
        if (!body) return;
        const img = new Image();
        img.onload = function () {
            body.innerHTML =
                '<img src="figures/inference_throughput.png" ' +
                     'alt="Inference throughput" ' +
                     'style="max-width: 100%; height: auto; border-radius: 6px;">';
        };
        img.onerror = function () {
            body.innerHTML = '<span class="ut-pending">Awaiting US-045 throughput measurement.</span>';
        };
        img.src = 'figures/inference_throughput.png';
    }

    function renderConfusionGallery(rows) {
        const body = document.getElementById('confusion-gallery-body');
        if (!body) return;
        // Every Complete L5 cell has a confusion PNG on disk: US-045 rendered
        // B / B2 / D-T3; scripts/generate_extended_diagnostics.py (2026-07-02)
        // filled in C, C2 and D-T1/T2.
        const completeL5 = rows.filter(function (r) {
            return r.level === 5 && r.status === 'Complete';
        });
        if (completeL5.length === 0) {
            body.innerHTML = '<span class="ut-pending">Awaiting US-045 confusion-matrix dump (18 L5 cells × seed=42).</span>';
            return;
        }
        // Build a simple thumbnail grid; PNGs are at figures/confusion/<tag>_L5.png
        const items = completeL5.map(function (r) {
            const tag = r.tag;
            return '<a href="figures/confusion/' + tag + '_L5.png" target="_blank" ' +
                   'style="display:inline-block;margin:4px;text-align:center;font-size:11px;">' +
                   '<img src="figures/confusion/' + tag + '_L5.png" loading="lazy" ' +
                        'alt="' + tag + '" style="width:180px;height:auto;border:1px solid #ddd;border-radius:4px;">' +
                   '<div>' + tag + '</div></a>';
        }).join('');
        body.innerHTML = '<div style="display:flex;flex-wrap:wrap;">' + items + '</div>';
    }

    function renderCalibrationGallery(rows) {
        const body = document.getElementById('calibration-gallery-body');
        if (!body) return;
        // Every Complete L3/L5 cell has a calibration PNG on disk: US-045
        // rendered B / B2 / B2nr / D-T3; generate_extended_diagnostics.py
        // (2026-07-02) filled in C, C2 and D-T1/T2.
        const targets = rows.filter(function (r) {
            return r.status === 'Complete' && (r.level === 3 || r.level === 5);
        });
        if (targets.length === 0) {
            body.innerHTML = '<span class="ut-pending">Awaiting US-045 calibration / ECE diagnostics (42 cells).</span>';
            return;
        }
        const items = targets.map(function (r) {
            const tag = r.tag;
            const lvl = 'L' + r.level;
            return '<a href="figures/calibration/' + tag + '_' + lvl + '.png" target="_blank" ' +
                   'style="display:inline-block;margin:4px;text-align:center;font-size:11px;">' +
                   '<img src="figures/calibration/' + tag + '_' + lvl + '.png" loading="lazy" ' +
                        'alt="' + tag + ' ' + lvl + '" style="width:170px;height:auto;border:1px solid #ddd;border-radius:4px;">' +
                   '<div>' + tag + ' ' + lvl + '</div></a>';
        }).join('');
        body.innerHTML = '<div style="display:flex;flex-wrap:wrap;">' + items + '</div>';
    }

    // -- V6 per-phase intro blocks (2026-07-02) -----------------------------
    // Motivation + findings text and figure lists are lifted from the V6
    // summary report (Final_Report/V6/source/sections/04_phases_overview.tex
    // + 05_tests.tex). B1 and C2 are the two phases V6 treats in depth; the
    // rest get the report's short focused paragraph. Figures are the exact
    // PNGs the report renders, copied to artifacts/figures/v6/.
    var V6_FIG = 'figures/v6/';

    function piFig(src, caption) {
        return '<div class="pi-figure"><a href="' + V6_FIG + src + '" target="_blank">' +
               '<img loading="lazy" decoding="async" src="' + V6_FIG + src + '" alt="' + caption + '">' +
               '<div class="pi-caption">' + caption + '</div></a></div>';
    }

    function piLocalFig(src, caption) {
        return '<div class="pi-figure"><a href="' + src + '" target="_blank">' +
               '<img loading="lazy" decoding="async" src="' + src + '" alt="' + caption + '">' +
               '<div class="pi-caption">' + caption + '</div></a></div>';
    }

    var PHASE_INTROS = {
        A: {
            title: 'Phase A — Clean baselines',
            badge: 'brief',
            cells: '6 cells · 3 models × 2 datasets · pipeline at identity (upsample to 224×224 only)',
            motivation: '<p>Establish the upper-bound clean accuracy per (model, dataset) pair. ' +
                'Every accuracy drop reported anywhere else in the campaign is measured against ' +
                'these six reference numbers.</p>',
            findings: '<ul>' +
                '<li>CIFAR-10: ResNet50 <span class="pi-num">95.18%</span>, DenseNet121 <span class="pi-num">93.56%</span>, TransNeXt-tiny <span class="pi-num">97.64%</span>.</li>' +
                '<li>MNIST: all three models converge to <span class="pi-num">99.1–99.3%</span> (ResNet50 99.14, DenseNet121 99.30, TransNeXt 99.24).</li>' +
                '<li>Cross-cutting diagnostics (V6-corrected): inference at batch 1 on RTX 5070 — ResNet50 <span class="pi-num">4.2 ms</span>, DenseNet121 <span class="pi-num">10.7 ms</span>, TransNeXt-tiny <span class="pi-num">27.5 ms</span> (~6.5× ResNet50, still ≥ THz scanner acquisition rates).</li>' +
                '</ul>',
            figuresHtml:
                piFig('block_diagrams/system_overview.png', 'V6 Fig. — system overview: the full experiment system (data, degradation, training engine, trackers).') +
                piFig('block_diagrams/pipeline_overview.png', 'V6 Fig. — degradation pipeline: saturation lerp → downsample → upsample 224 → blur → noise → S&amp;P → normalize.')
        },
        B: {
            title: 'Phase B1 — Combined degradation',
            badge: 'deep',
            cells: '30 cells · all five axes active simultaneously at a common level L1–L5 · V6 report §5 (deep)',
            motivation: '<p>The deployment-shaped question: <em>how much accuracy survives when everything ' +
                'degrades at once?</em> All five degradation axes (resolution, blur, noise, saturation, ' +
                'salt-and-pepper) are applied together at a common severity level. Here the L1–L5 index is a ' +
                'legitimate x-axis because a single fixed recipe is swept and each level is one well-defined ' +
                'operating point.</p>',
            findings: '<ul>' +
                '<li><strong>Monotonic severe collapse on CIFAR-10:</strong> L1 already costs <span class="pi-neg">10–19 pp</span>; L3 costs <span class="pi-neg">54–59 pp</span>; at L5 all three models sit at <span class="pi-num">19–21%</span> vs the 10% random floor.</li>' +
                '<li><strong>Transformer leads only while images are readable:</strong> TransNeXt beats the CNN mean by <span class="pi-num">+10.11 pp</span> at L1 and <span class="pi-num">+10.41 pp</span> at L2; the lead collapses to <span class="pi-num">+2.05 pp</span> at L3 (≈1 seed-σ over the better CNN) and under 1 pp at L4–L5 — from L3 onward the architectures are practically interchangeable.</li>' +
                '<li><strong>MNIST resists, then falls:</strong> drops &lt; 3.3 pp at L1–L2, usable at L3 (<span class="pi-num">~79–81%</span>), collapses to <span class="pi-num">27–28%</span> at L5 once 3×3-pixel downsampling removes stroke structure.</li>' +
                '<li><strong>Multi-seed audit:</strong> all 24 L3 headline cells re-trained at seeds 43/44 — largest per-cell σ is <span class="pi-num">1.18 pp</span>; no cross-model ordering reverses (L3 rows below show mean ± std).</li>' +
                '</ul>',
            figuresHtml:
                piFig('curves/b1_acc_vs_level.png', 'V6 Fig. — B1 validation accuracy vs combined level; dotted lines are the per-model clean baselines (left CIFAR-10, right MNIST).') +
                piFig('samples/grid_B1_levels_cifar10.png', 'V6 Fig. — the B1 recipe at L1–L5 on six fixed CIFAR-10 validation images.') +
                piFig('samples/grid_B1_levels_mnist.png', 'V6 Fig. — the B1 recipe at L1–L5 on six fixed MNIST validation images.'),
            charts: 'B'
        },
        B2: {
            title: 'Phase B2 — THz-protocol simplification',
            badge: 'brief',
            cells: '30 cells · B1 recipe with saturation fixed at 0 (full grayscale) and Gaussian noise off · T3 regularization',
            motivation: '<p>The deployment-realistic protocol: a real THz frame is single-channel, and the ' +
                'dominant sensor noise is impulsive rather than Gaussian. Phase B2 re-runs the combined sweep ' +
                'under that simpler protocol and asks how much easier it is than the worst-case B1 recipe.</p>',
            findings: '<ul>' +
                '<li>The realistic protocol lifts accuracy by <span class="pi-num">+4.01 pp</span> on average over Phase B1, and by <span class="pi-num">+5.58 pp</span> at the moderate level (L3).</li>' +
                '<li>Combined with Phase B2-nr this decomposes into ~¾ pure protocol physics, ~¼ regularization (see the B2-nr tab).</li>' +
                '</ul>',
            figuresHtml:
                piLocalFig('figures/phase_b2_comparison_summary.png', 'B1 vs D-T3 vs B2 vs B2-nr comparison summary (campaign figure; also embedded in the strip above the tabs).')
        },
        B2nr: {
            title: 'Phase B2-nr — No-regularization arm',
            badge: 'brief',
            cells: '6 cells · B2 pipeline at L3 only, T3 off',
            motivation: '<p>Of the B1→B2 improvement, how much comes from the simpler protocol itself and how ' +
                'much from the T3 regularization that Phase B2 keeps on? Re-running B2 at L3 with regularization ' +
                'off isolates the two contributions.</p>',
            findings: '<ul>' +
                '<li>L3 decomposition: <span class="pi-num">+4.19 pp</span> from the simpler protocol itself vs only <span class="pi-num">+1.39 pp</span> from T3 — about three-quarters of the improvement is the physics of the protocol, not the training trick.</li>' +
                '<li>T3 helps ~twice as much under the simpler protocol (+1.39 pp) as under the full one (+0.77 pp): regularization becomes more useful as degradation becomes less severe.</li>' +
                '<li>Calibration link: the single worst miscalibration of the whole campaign (ECE <span class="pi-neg">12.7%</span>) occurs on this unregularized arm — regularization also buys calibration.</li>' +
                '</ul>',
            figuresHtml: ''
        },
        C: {
            title: 'Phase C — Single-axis isolation (full-color protocol)',
            badge: 'brief',
            cells: '150 cells · exactly one of the five axes active at level L, the other four at identity',
            motivation: '<p>Which individual degradation drives the Phase B1 collapse? Each cell activates one ' +
                'axis at level L and holds the other four at identity, over the full color pipeline. Note a ' +
                'Phase C L1 cell is <em>not</em> equivalent to Phase B1 L1 (which has all five axes at L1).</p>',
            findings: '<ul>' +
                '<li>Clear severity ordering: <strong>resolution loss dominates</strong>; blur and additive noise are intermediate; desaturation and salt-and-pepper are comparatively benign in isolation.</li>' +
                '<li>“Taking away the pixels hurts far more than taking away the color.” This ordering motivated the three-axis THz-relevant subset studied in depth as Phase C2.</li>' +
                '</ul>',
            figuresHtml: '',
            charts: 'C'
        },
        C2: {
            title: 'Phase C2 — Single-axis attribution under the THz protocol',
            badge: 'deep',
            cells: '90 cells · resolution / blur / salt-and-pepper, one at a time · grayscale, zero Gaussian noise, T3 · V6 report §5 (deep)',
            motivation: '<p>The engineering question behind the whole campaign: <em>of the degradations a real ' +
                'THz sensor produces, which one should be fought first?</em> Same identity-elsewhere logic as ' +
                'Phase C, restricted to the three THz-relevant spatial axes, with grayscale and zero Gaussian ' +
                'noise forced throughout. In the V6 report, cross-axis comparisons are drawn against ' +
                '<strong>measured PSNR/SSIM</strong> rather than the preset L1–L5 index, because a given level ' +
                'does unequal damage across axes (the interactive charts below use the L-index within each axis; ' +
                'the V6 figures give the measured-quality view).</p>',
            findings: '<ul>' +
                '<li><strong>Resolution is the axis that matters:</strong> mean L1–L5 accuracy cost is <span class="pi-neg">−27.45 pp</span> for resolution vs <span class="pi-neg">−6.58 pp</span> (blur) and <span class="pi-neg">−5.13 pp</span> (salt-and-pepper) — a factor of ~4. The ordering is identical under matched PSNR and SSIM.</li>' +
                '<li><strong>Accuracy tracks measured quality almost perfectly for resolution:</strong> Spearman ρ = <span class="pi-num">+0.98</span> on both datasets and both metrics; blur/S&amp;P correlate weaker because they are close to harmless (CIFAR-10 blur +0.84, S&amp;P +0.53; MNIST blur +0.42, S&amp;P +0.70).</li>' +
                '<li><strong>Model ordering:</strong> on CIFAR-10 TransNeXt-tiny is the most robust on every axis — most visibly on blur, holding <span class="pi-num">71.0%</span> at L5 vs 58.8% (ResNet50) / 64.6% (DenseNet121); the two CNNs essentially overlap.</li>' +
                '<li>On MNIST only resolution produces a meaningful fall (<span class="pi-num">~45%</span> at L5, all backbones); blur and S&amp;P leave digits readable at every level.</li>' +
                '</ul>',
            figuresHtml:
                piFig('curves/c2_type_psnr.png', 'V6 Fig. — axis comparison vs measured PSNR (one curve per axis, averaged over backbones; left CIFAR-10, right MNIST).') +
                piFig('curves/c2_type_ssim.png', 'V6 Fig. — axis comparison vs measured SSIM.') +
                piFig('curves/c2_model_psnr.png', 'V6 Fig. — model comparison vs PSNR, one panel per axis × dataset.') +
                piFig('curves/c2_model_ssim.png', 'V6 Fig. — model comparison vs SSIM.') +
                piFig('samples/grid_THz_L3_cifar10.png', 'V6 Fig. — clean / B1 / B2 grayscale / three C2 single-axis rows at L3 (CIFAR-10).') +
                piFig('samples/grid_THz_L3_mnist.png', 'V6 Fig. — the same L3 sample strip on MNIST.'),
            charts: 'C2'
        },
        D: {
            title: 'Phase D — Regularization recovery',
            badge: 'brief',
            cells: '90 cells · B1 pipeline + one of three treatments (T1 dropout/drop-path · T2 mixup · T3 combo)',
            motivation: '<p>Is the Phase B1 collapse over-fitting — which regularization should fix — or lost ' +
                'information, which it cannot? Three treatments layer on top of the frozen L3-Optuna winners with ' +
                'no re-tune.</p>',
            findings: '<ul>' +
                '<li>The best treatment (T3, the combination) recovers <span class="pi-num">+0.77 pp</span> on average — against a collapse roughly sixteen times larger (~12.5 pp).</li>' +
                '<li>Best single segment: TransNeXt-tiny on CIFAR-10 at <span class="pi-num">+2.20 pp</span> mean over 5 levels.</li>' +
                '<li>Conclusion: the collapse is an <strong>information bottleneck, not over-fitting</strong> — the detail the models need is no longer in the pixels.</li>' +
                '</ul>',
            figuresHtml:
                piLocalFig('figures/phase_d_comparison.png', 'Phase D recovery summary — Phase B1 baseline vs T1/T2/T3 (campaign figure).')
        }
    };

    // -- V6 tables as sortable HTML (2026-07-02) ----------------------------
    // Accuracy / delta / gap tables are computed client-side from state.rows
    // so they always agree with the table + charts; the Spearman table is
    // static (values from Final_Report/V6/source/tables/tab_c2_spearman.tex).

    function findCell(phase, model, dataset, level, axis, treatment) {
        for (var i = 0; i < state.rows.length; i++) {
            var r = state.rows[i];
            if (r.phase === phase && r.model === model && r.dataset === dataset
                && (level === null ? (r.level === null || r.level === undefined) : r.level === level)
                && (axis === undefined || (r.axis || null) === axis)
                && (treatment === undefined || (r.treatment || null) === treatment)) {
                return r;
            }
        }
        return null;
    }

    function cellAccHtml(r) {
        if (!r || r.val_acc === null || r.val_acc === undefined) return '—';
        if (r.val_acc_mean !== null && r.val_acc_mean !== undefined
            && r.val_acc_std !== null && r.val_acc_std !== undefined) {
            return (r.val_acc_mean * 100).toFixed(2) + '<span class="vt-std">±' +
                   (r.val_acc_std * 100).toFixed(2) + '</span>';
        }
        return (r.val_acc * 100).toFixed(2);
    }

    function v6Table(headers, rowsHtml, caption) {
        var ths = headers.map(function (h) {
            return '<th title="Click to sort">' + h + '</th>';
        }).join('');
        return '<div class="vt-wrap">' +
               (caption ? '<div class="vt-caption">' + caption + '</div>' : '') +
               '<table class="v6-table"><thead><tr>' + ths + '</tr></thead>' +
               '<tbody>' + rowsHtml + '</tbody></table></div>';
    }

    function b1TablesHtml() {
        var lv = ['L1', 'L2', 'L3', 'L4', 'L5'];
        // Table V — full accuracy block
        var accRows = '', deltaRows = '';
        MODEL_ORDER.forEach(function (m) {
            ['cifar10', 'mnist'].forEach(function (d) {
                var clean = findCell('A', m, d, null);
                var cleanAcc = clean && clean.val_acc !== null ? clean.val_acc * 100 : null;
                var acc = '<tr><td>' + MODEL_LABELS[m] + '</td><td>' + d + '</td>' +
                          '<td class="vt-num">' + (cleanAcc !== null ? cleanAcc.toFixed(2) : '—') + '</td>';
                var del = '<tr><td>' + MODEL_LABELS[m] + '</td><td>' + d + '</td>';
                LEVELS.forEach(function (l) {
                    var r = findCell('B', m, d, l);
                    acc += '<td class="vt-num">' + cellAccHtml(r) + '</td>';
                    var dv = (r && r.val_acc !== null && cleanAcc !== null)
                        ? (r.val_acc * 100 - cleanAcc) : null;
                    del += '<td class="vt-num vt-neg">' + (dv !== null ? dv.toFixed(2) : '—') + '</td>';
                });
                accRows += acc + '</tr>';
                deltaRows += del + '</tr>';
            });
        });
        // Gap table — TransNeXt margin over the CNNs per (dataset, level)
        var gapRows = '';
        ['cifar10', 'mnist'].forEach(function (d) {
            var tr = '<tr><td>' + d + '</td>';
            var tr2vals = [];
            LEVELS.forEach(function (l) {
                var t = findCell('B', 'transnext_tiny', d, l);
                var r1 = findCell('B', 'resnet50', d, l);
                var r2 = findCell('B', 'densenet121', d, l);
                if (t && r1 && r2 && t.val_acc !== null) {
                    var cnnMean = (r1.val_acc + r2.val_acc) / 2 * 100;
                    var cnnBest = Math.max(r1.val_acc, r2.val_acc) * 100;
                    var g1 = t.val_acc * 100 - cnnMean;
                    var g2 = t.val_acc * 100 - cnnBest;
                    tr += '<td class="vt-num">' + (g1 >= 0 ? '+' : '') + g1.toFixed(2) + '</td>';
                    tr2vals.push('<td class="vt-num">' + (g2 >= 0 ? '+' : '') + g2.toFixed(2) + '</td>');
                } else {
                    tr += '<td class="vt-num">—</td>';
                    tr2vals.push('<td class="vt-num">—</td>');
                }
            });
            gapRows += tr + '</tr>';
            gapRows += '<tr class="vt-sub"><td>' + d + ' (vs better CNN)</td>' + tr2vals.join('') + '</tr>';
        });
        return '<details class="pi-tables"><summary>V6 tables — Table V (accuracy), Δ vs clean, TransNeXt gap (click headers to sort)</summary>' +
            v6Table(['Model', 'Dataset', 'Clean'].concat(lv), accRows,
                    'Table V — Phase B1 val_acc (%); L3 shows multi-seed mean ± σ where audited.') +
            v6Table(['Model', 'Dataset'].concat(lv), deltaRows,
                    'Δ vs clean baseline (pp) — how much each level costs.') +
            v6Table(['Dataset'].concat(lv), gapRows,
                    'TransNeXt-tiny margin over the CNN mean (pp); second row per dataset = margin over the better CNN.') +
            '</details>';
    }

    function c2TablesHtml() {
        var lv = ['L1', 'L2', 'L3', 'L4', 'L5'];
        var accRows = '';
        AXIS_ORDER_C2.forEach(function (ax) {
            MODEL_ORDER.forEach(function (m) {
                ['cifar10', 'mnist'].forEach(function (d) {
                    var tr = '<tr><td>' + ax + '</td><td>' + MODEL_LABELS[m] + '</td><td>' + d + '</td>';
                    LEVELS.forEach(function (l) {
                        tr += '<td class="vt-num">' + cellAccHtml(findCell('C2', m, d, l, ax)) + '</td>';
                    });
                    accRows += tr + '</tr>';
                });
            });
        });
        // Spearman table — static values from tab_c2_spearman.tex (pooled n=15).
        var sp = [
            ['Resolution', '+0.98', '+0.98', '+0.98', '+0.98'],
            ['Blur', '+0.84', '+0.84', '+0.42', '+0.42'],
            ['Salt &amp; pepper', '+0.53', '+0.53', '+0.70', '+0.70'],
        ];
        var spRows = sp.map(function (r) {
            return '<tr><td>' + r[0] + '</td>' + r.slice(1).map(function (v) {
                return '<td class="vt-num">' + v + '</td>';
            }).join('') + '</tr>';
        }).join('');
        return '<details class="pi-tables"><summary>V6 tables — Table IX (accuracy block), Spearman ρ vs measured quality (click headers to sort)</summary>' +
            v6Table(['Axis', 'Model', 'Dataset'].concat(lv), accRows,
                    'Table IX — Phase C2 val_acc (%) per axis; L3 shows multi-seed mean ± σ where audited.') +
            v6Table(['Axis', 'ρ PSNR (CIFAR-10)', 'ρ SSIM (CIFAR-10)', 'ρ PSNR (MNIST)', 'ρ SSIM (MNIST)'], spRows,
                    'Spearman rank correlation between accuracy and measured quality, pooled over the 3 backbones (n = 15 per entry).') +
            '</details>';
    }

    function bindSortableTables(scope) {
        scope.querySelectorAll('table.v6-table th').forEach(function (th) {
            th.addEventListener('click', function () {
                var table = th.closest('table');
                var tbody = table.querySelector('tbody');
                var idx = th.cellIndex;
                var dir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
                table.querySelectorAll('th').forEach(function (h) { delete h.dataset.dir; h.classList.remove('vt-sorted'); });
                th.dataset.dir = dir;
                th.classList.add('vt-sorted');
                var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
                rows.sort(function (a, b) {
                    var ta = a.cells[idx] ? a.cells[idx].textContent.trim() : '';
                    var tb = b.cells[idx] ? b.cells[idx].textContent.trim() : '';
                    var na = parseFloat(ta.replace(/[+%]/g, ''));
                    var nb = parseFloat(tb.replace(/[+%]/g, ''));
                    var cmp;
                    if (!isNaN(na) && !isNaN(nb)) cmp = na - nb;
                    else cmp = ta.localeCompare(tb);
                    return dir === 'asc' ? cmp : -cmp;
                });
                rows.forEach(function (r) { tbody.appendChild(r); });
            });
        });
    }

    // -- Overview tab (2026-07-02) -----------------------------------------
    // The four headline scientific conclusions (V6 §7) + a model comparison
    // table. Throughput + parameter counts are static (measured by US-045 /
    // model cards); accuracies are computed live from state.rows.

    var MODEL_META = {
        resnet50:       { params: '25.6 M', ms: 4.2,  ips: 238 },
        densenet121:    { params: '8.0 M',  ms: 10.7, ips: 94 },
        transnext_tiny: { params: '28.3 M', ms: 27.5, ips: 36 },
    };

    function buildOverviewHtml() {
        var cards =
            '<div class="ov-grid">' +
            '<div class="ov-card"><h4>1 · An information bottleneck, not over-fitting</h4>' +
            '<p>The best regularization treatment (Phase D, T3) recovers only <span class="pi-num">+0.77 pp</span> of the ' +
            '~12.5 pp combined-degradation collapse. Phase B2 shows the deployment-realistic protocol is worth ' +
            '<span class="pi-num">+4.01 pp</span>, and the B2-nr arm attributes ~¾ of that to protocol physics rather than training tricks. ' +
            'The detail the models need is simply no longer in the pixels.</p></div>' +
            '<div class="ov-card"><h4>2 · The foveal transformer wins while images are readable</h4>' +
            '<p>TransNeXt-tiny beats the CNN mean by <span class="pi-num">+10.11 pp</span> (L1) and <span class="pi-num">+10.41 pp</span> (L2) ' +
            'on CIFAR-10 — the operationally relevant mild–moderate range. From L3 the lead shrinks to ' +
            '<span class="pi-num">+2.05 pp</span> (≈1 seed-σ over the better CNN).</p></div>' +
            '<div class="ov-card"><h4>3 · At the extreme, architecture stops mattering</h4>' +
            '<p>At L5 every backbone lands in the same band: <span class="pi-num">19–21%</span> on CIFAR-10 and ' +
            '<span class="pi-num">27–28%</span> on MNIST. Once 3×3-pixel downsampling removes the structure, no architecture can recover it.</p></div>' +
            '<div class="ov-card"><h4>4 · Resolution is the dominant THz axis</h4>' +
            '<p>Under the THz protocol (Phase C2), resolution costs <span class="pi-neg">−27.45 pp</span> mean L1–L5 vs ' +
            '<span class="pi-neg">−6.58 pp</span> (blur) and <span class="pi-neg">−5.13 pp</span> (salt-and-pepper) — a ~4× factor, with ' +
            'Spearman ρ = <span class="pi-num">+0.98</span> against measured PSNR/SSIM. Fight resolution first.</p></div>' +
            '</div>';

        // Model comparison table (computed + static columns)
        var rowsHtml = '';
        MODEL_ORDER.forEach(function (m) {
            var meta = MODEL_META[m];
            var cleanC = findCell('A', m, 'cifar10', null);
            var cleanM = findCell('A', m, 'mnist', null);
            var b1l3 = findCell('B', m, 'cifar10', 3);
            var b1l5 = findCell('B', m, 'cifar10', 5);
            var c2blur = findCell('C2', m, 'cifar10', 5, 'blur');
            rowsHtml += '<tr><td>' + MODEL_LABELS[m] + '</td>' +
                '<td class="vt-num">' + meta.params + '</td>' +
                '<td class="vt-num">' + cellAccHtml(cleanC) + '</td>' +
                '<td class="vt-num">' + cellAccHtml(cleanM) + '</td>' +
                '<td class="vt-num">' + cellAccHtml(b1l3) + '</td>' +
                '<td class="vt-num">' + cellAccHtml(b1l5) + '</td>' +
                '<td class="vt-num">' + cellAccHtml(c2blur) + '</td>' +
                '<td class="vt-num">' + meta.ms.toFixed(1) + '</td>' +
                '<td class="vt-num">' + meta.ips + '</td></tr>';
        });
        var table = v6Table(
            ['Model', 'Params', 'Clean CIFAR-10 (%)', 'Clean MNIST (%)', 'B1 L3 CIFAR-10 (%)',
             'B1 L5 CIFAR-10 (%)', 'C2 blur L5 CIFAR-10 (%)', 'ms / img', 'img / s'],
            rowsHtml,
            'Model comparison — accuracies computed live from the run data; throughput measured at batch 1, bf16-mixed, RTX 5070 (US-045). Click headers to sort.');

        var stats = state.rows.length
            ? state.rows.length + ' canonical cells · 48 multi-seed audit replicates · 3 architectures · 2 datasets · 5-level degradation curve'
            : '';
        return '<h2>Campaign Overview<span class="pi-badge deep">V6 conclusions</span></h2>' +
            '<div class="pi-cells">' + stats + '</div>' +
            cards + table +
            '<p class="ov-note">The two phases the V6 report treats in depth are ' +
            '<strong>Phase B1</strong> (combined degradation) and <strong>Phase C2</strong> (THz single-axis attribution) — ' +
            'open their tabs for interactive charts, V6 figures and the full per-cell table.</p>';
    }

    function renderPhaseIntro(phase) {
        var el = document.getElementById('phase-intro');
        if (!el) return;
        if (phase === 'OV') {
            el.style.display = '';
            el.innerHTML = buildOverviewHtml();
            bindSortableTables(el);
            return;
        }
        var info = PHASE_INTROS[phase];
        if (!info) { el.style.display = 'none'; return; }
        el.style.display = '';
        var badge = info.badge === 'deep'
            ? '<span class="pi-badge deep">V6 report — in depth</span>'
            : '<span class="pi-badge brief">V6 report — brief</span>';
        var chartsHtml = '';
        var metricSelHtml =
            '<div class="chart-controls">' +
            '<span class="chart-controls-label">Chart x-axis:</span>' +
            '<div class="axis-select" data-chart-group="metric"></div>' +
            '<span class="chart-controls-hint">L-index sweeps the preset levels; PSNR and SSIM plot accuracy against the measured image quality of each cell (the V6 cross-axis view).</span>' +
            '</div>';
        if (info.charts === 'B') {
            chartsHtml = metricSelHtml +
                '<div class="pi-charts">' +
                '<div class="chart-card"><div class="chart-title">Degradation-type impact — CIFAR-10</div>' +
                '<div class="chart-sub">Single-axis isolation (Phase C, full-color protocol), mean over the 3 models, vs the combined B1 recipe</div>' +
                '<div class="chart-plot" id="chart-B-type-cifar10"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '<div class="chart-card"><div class="chart-title">Degradation-type impact — MNIST</div>' +
                '<div class="chart-sub">Single-axis isolation (Phase C), mean over the 3 models, vs combined B1</div>' +
                '<div class="chart-plot" id="chart-B-type-mnist"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '<div class="chart-card"><div class="chart-title">Model robustness — CIFAR-10</div>' +
                '<div class="chart-sub">Phase B1 val_acc per model vs level; dotted lines are the Phase A clean baselines</div>' +
                '<div class="chart-plot" id="chart-B-model-cifar10"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '<div class="chart-card"><div class="chart-title">Model robustness — MNIST</div>' +
                '<div class="chart-sub">Phase B1 val_acc per model vs level; dotted lines are the Phase A clean baselines</div>' +
                '<div class="chart-plot" id="chart-B-model-mnist"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '</div>';
        } else if (info.charts === 'C') {
            chartsHtml = metricSelHtml +
                '<div class="pi-charts">' +
                '<div class="chart-card"><div class="chart-title">Degradation-type impact — CIFAR-10</div>' +
                '<div class="chart-sub">One curve per isolated axis, mean over the 3 models, vs combined B1</div>' +
                '<div class="chart-plot" id="chart-C-type-cifar10"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '<div class="chart-card"><div class="chart-title">Degradation-type impact — MNIST</div>' +
                '<div class="chart-sub">One curve per isolated axis, mean over the 3 models, vs combined B1</div>' +
                '<div class="chart-plot" id="chart-C-type-mnist"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '</div>';
        } else if (info.charts === 'C2') {
            chartsHtml = metricSelHtml +
                '<div class="pi-charts">' +
                '<div class="chart-card"><div class="chart-title">THz degradation-type impact — CIFAR-10</div>' +
                '<div class="chart-sub">One curve per C2 axis (grayscale protocol, T3), mean over the 3 models, vs combined B2</div>' +
                '<div class="chart-plot" id="chart-C2-type-cifar10"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '<div class="chart-card"><div class="chart-title">THz degradation-type impact — MNIST</div>' +
                '<div class="chart-sub">One curve per C2 axis, mean over the 3 models, vs combined B2</div>' +
                '<div class="chart-plot" id="chart-C2-type-mnist"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '<div class="chart-card"><div class="chart-title">Model robustness per axis — CIFAR-10</div>' +
                '<div class="chart-sub">Phase C2 val_acc per model vs level on the selected axis</div>' +
                '<div class="axis-select" data-chart-group="c2-model"></div>' +
                '<div class="chart-plot" id="chart-C2-model-cifar10"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '<div class="chart-card"><div class="chart-title">Model robustness per axis — MNIST</div>' +
                '<div class="chart-sub">Phase C2 val_acc per model vs level on the selected axis</div>' +
                '<div class="chart-plot" id="chart-C2-model-mnist"><div class="chart-placeholder">Loading Plotly…</div></div></div>' +
                '</div>';
        }
        var tablesHtml = '';
        if (phase === 'B') tablesHtml = b1TablesHtml();
        else if (phase === 'C2') tablesHtml = c2TablesHtml();
        el.innerHTML =
            '<h2>' + info.title + badge + '</h2>' +
            '<div class="pi-cells">' + info.cells + '</div>' +
            '<div class="pi-cols">' +
            '<div class="pi-block"><h3>Motivation</h3>' + info.motivation + '</div>' +
            '<div class="pi-block"><h3>Findings (V6 report)</h3>' + info.findings + '</div>' +
            '</div>' +
            chartsHtml +
            tablesHtml +
            (info.figuresHtml ? '<div class="pi-figures">' + info.figuresHtml + '</div>' : '');
        bindSortableTables(el);
        if (info.charts) renderPhaseCharts(phase);
    }

    // -- Interactive Plotly charts (B1 + C + C2) ---------------------------
    // Data comes straight from state.rows (the same inline JSON the table
    // renders from), so the charts always agree with the table.
    // Categorical colors: dataviz reference palette, dark-mode steps, fixed
    // slot order (blue, aqua, yellow, green, violet) — CVD-validated set.

    var MODEL_ORDER = ['resnet50', 'densenet121', 'transnext_tiny'];
    var MODEL_COLORS = { resnet50: '#3987e5', densenet121: '#199e70', transnext_tiny: '#c98500' };
    var MODEL_LABELS = { resnet50: 'ResNet50', densenet121: 'DenseNet121', transnext_tiny: 'TransNeXt-tiny' };
    var AXIS_ORDER_C  = ['resolution', 'blur', 'salt_pepper', 'noise', 'saturation'];
    var AXIS_ORDER_C2 = ['resolution', 'blur', 'salt_pepper'];
    var AXIS_COLORS = { resolution: '#3987e5', blur: '#199e70', salt_pepper: '#c98500',
                        noise: '#008300', saturation: '#9085e9' };
    var COMBINED_COLOR = '#c3c2b7';
    var LEVELS = [1, 2, 3, 4, 5];
    var c2SelectedAxis = 'resolution';
    // Chart x-axis metric: 'level' (preset L1–L5 index) | 'psnr' | 'ssim'
    // (measured image quality of each cell — the V6 cross-axis view).
    var chartMetric = 'level';
    var METRIC_LABELS = { level: 'L-index', psnr: 'PSNR', ssim: 'SSIM' };

    function completeRows(phase) {
        return state.rows.filter(function (r) {
            return r.phase === phase && r.status === 'Complete'
                && r.val_acc !== null && r.val_acc !== undefined;
        });
    }

    function meanOf(rows, pred, valueOf) {
        var vals = [];
        rows.forEach(function (r) {
            if (!pred(r)) return;
            var v = valueOf(r);
            if (typeof v === 'number' && !isNaN(v)) vals.push(v);
        });
        if (!vals.length) return null;
        var s = 0;
        vals.forEach(function (v) { s += v; });
        return s / vals.length;
    }

    function accAt(rows, pred) {
        return meanOf(rows, pred, function (r) { return r.val_acc * 100; });
    }

    // X coordinate of a (level, group) point under the active metric. For
    // 'level' it is the preset index; for psnr/ssim it is the measured
    // quality of the matching cells (identical across models for the same
    // pipeline — the mean collapses replicated values, not divergent ones).
    function metricXAt(rows, pred) {
        if (chartMetric === 'psnr') return meanOf(rows, pred, function (r) { return r.psnr_mean; });
        if (chartMetric === 'ssim') return meanOf(rows, pred, function (r) { return r.ssim_mean; });
        return null; // 'level' handled by caller
    }

    function chartLayout(yTitle) {
        var xaxis;
        if (chartMetric === 'psnr') {
            xaxis = { title: { text: 'Measured PSNR (dB) — higher is better quality' },
                      gridcolor: '#2a3040', zeroline: false };
        } else if (chartMetric === 'ssim') {
            xaxis = { title: { text: 'Measured SSIM — higher is better quality' },
                      gridcolor: '#2a3040', zeroline: false };
        } else {
            xaxis = { title: { text: 'Degradation level' }, tickvals: LEVELS,
                      ticktext: ['L1', 'L2', 'L3', 'L4', 'L5'],
                      gridcolor: '#2a3040', zeroline: false };
        }
        return {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#c3c2b7', size: 11,
                    family: "'Segoe UI', system-ui, sans-serif" },
            margin: { l: 52, r: 12, t: 8, b: 40 },
            hovermode: chartMetric === 'level' ? 'x unified' : 'closest',
            hoverlabel: { bgcolor: '#1e232b', bordercolor: '#3a4560',
                          font: { color: '#e8eaf0', size: 11 } },
            xaxis: xaxis,
            yaxis: { title: { text: yTitle }, gridcolor: '#2a3040',
                     zeroline: false, rangemode: 'tozero' },
            legend: { orientation: 'h', y: 1.14, font: { color: '#e8eaf0' } },
        };
    }

    var CHART_CFG = { displayModeBar: false, responsive: true };

    // Build one series: for each level, y = mean accuracy of the matching
    // cells and x = level index or measured quality. Null points are skipped.
    function seriesTrace(name, rows, predForLevel, color, dash) {
        var xs = [], ys = [], lvls = [];
        LEVELS.forEach(function (l) {
            var pred = predForLevel(l);
            var y = accAt(rows, pred);
            if (y === null) return;
            var x = chartMetric === 'level' ? l : metricXAt(rows, pred);
            if (x === null) return;
            xs.push(x); ys.push(y); lvls.push('L' + l);
        });
        var xfmt = chartMetric === 'psnr' ? ' · %{x:.1f} dB'
                 : chartMetric === 'ssim' ? ' · SSIM %{x:.3f}' : '';
        return {
            x: xs, y: ys, name: name, mode: 'lines+markers',
            customdata: lvls,
            line: { color: color, width: 2, dash: dash || 'solid' },
            marker: { size: 8, color: color },
            connectgaps: true,
            hovertemplate: name + ' (%{customdata}' + xfmt + '): %{y:.2f}%<extra></extra>',
        };
    }

    function renderTypeChart(elId, isoPhase, combinedPhase, dataset, axisOrder) {
        var el = document.getElementById(elId);
        if (!el) return;
        var iso = completeRows(isoPhase);
        var comb = completeRows(combinedPhase);
        var traces = [];
        axisOrder.forEach(function (ax) {
            traces.push(seriesTrace(ax, iso, function (l) {
                return function (r) {
                    return r.axis === ax && r.level === l && r.dataset === dataset;
                };
            }, AXIS_COLORS[ax]));
        });
        if (comb.length) {
            traces.push(seriesTrace(
                'combined (' + (combinedPhase === 'B' ? 'B1' : combinedPhase) + ')',
                comb, function (l) {
                    return function (r) { return r.level === l && r.dataset === dataset; };
                }, COMBINED_COLOR, 'dash'));
        }
        el.innerHTML = ''; // drop the "Loading Plotly…" placeholder
        window.Plotly.newPlot(elId, traces, chartLayout('val_acc (%)'), CHART_CFG);
    }

    function renderModelChart(elId, phase, dataset, axis) {
        var el = document.getElementById(elId);
        if (!el) return;
        var rows = completeRows(phase);
        var clean = completeRows('A');
        var traces = [];
        MODEL_ORDER.forEach(function (m) {
            traces.push(seriesTrace(MODEL_LABELS[m], rows, function (l) {
                return function (r) {
                    return r.model === m && r.level === l && r.dataset === dataset
                        && (!axis || r.axis === axis);
                };
            }, MODEL_COLORS[m]));
        });
        // Clean baselines as dotted reference lines (Phase A), spanning the
        // observed x-range of the model series.
        var xmin = null, xmax = null;
        traces.forEach(function (t) {
            t.x.forEach(function (v) {
                if (xmin === null || v < xmin) xmin = v;
                if (xmax === null || v > xmax) xmax = v;
            });
        });
        if (xmin !== null && xmax !== null && xmin !== xmax) {
            MODEL_ORDER.forEach(function (m) {
                var base = accAt(clean, function (r) {
                    return r.model === m && r.dataset === dataset;
                });
                if (base === null) return;
                traces.push({
                    x: [xmin, xmax], y: [base, base],
                    name: MODEL_LABELS[m] + ' clean', mode: 'lines',
                    line: { color: MODEL_COLORS[m], width: 1, dash: 'dot' },
                    showlegend: false, hoverinfo: 'skip',
                });
            });
        }
        el.innerHTML = ''; // drop the "Loading Plotly…" placeholder
        window.Plotly.newPlot(elId, traces, chartLayout('val_acc (%)'), CHART_CFG);
    }

    function renderMetricSelector() {
        document.querySelectorAll('[data-chart-group="metric"]').forEach(function (host) {
            host.innerHTML = ['level', 'psnr', 'ssim'].map(function (mkey) {
                return '<button type="button" class="axis-btn' +
                       (mkey === chartMetric ? ' active' : '') +
                       '" data-metric="' + mkey + '">' + METRIC_LABELS[mkey] + '</button>';
            }).join('');
            host.querySelectorAll('.axis-btn').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    chartMetric = btn.dataset.metric;
                    renderMetricSelector();
                    renderPhaseCharts(state.activePhase);
                });
            });
        });
    }

    function renderAxisSelector() {
        var host = document.querySelector('[data-chart-group="c2-model"]');
        if (!host) return;
        host.innerHTML = AXIS_ORDER_C2.map(function (ax) {
            return '<button type="button" class="axis-btn' +
                   (ax === c2SelectedAxis ? ' active' : '') +
                   '" data-axis="' + ax + '">' + ax + '</button>';
        }).join('');
        host.querySelectorAll('.axis-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                c2SelectedAxis = btn.dataset.axis;
                renderAxisSelector();
                whenPlotly(function () {
                    renderModelChart('chart-C2-model-cifar10', 'C2', 'cifar10', c2SelectedAxis);
                    renderModelChart('chart-C2-model-mnist',   'C2', 'mnist',   c2SelectedAxis);
                });
            });
        });
    }

    function whenPlotly(cb, tries) {
        if (typeof window.Plotly !== 'undefined') { cb(); return; }
        tries = tries === undefined ? 40 : tries;
        if (tries <= 0) {
            document.querySelectorAll('.chart-plot .chart-placeholder').forEach(function (p) {
                p.textContent = 'Plotly CDN unavailable (offline?) — interactive charts disabled; the V6 figures below carry the same story.';
            });
            return;
        }
        setTimeout(function () { whenPlotly(cb, tries - 1); }, 250);
    }

    function renderPhaseCharts(phase) {
        if (!state.rows.length) return;
        renderMetricSelector();
        whenPlotly(function () {
            if (phase === 'B') {
                renderTypeChart('chart-B-type-cifar10', 'C', 'B', 'cifar10', AXIS_ORDER_C);
                renderTypeChart('chart-B-type-mnist',   'C', 'B', 'mnist',   AXIS_ORDER_C);
                renderModelChart('chart-B-model-cifar10', 'B', 'cifar10', null);
                renderModelChart('chart-B-model-mnist',   'B', 'mnist',   null);
            } else if (phase === 'C') {
                renderTypeChart('chart-C-type-cifar10', 'C', 'B', 'cifar10', AXIS_ORDER_C);
                renderTypeChart('chart-C-type-mnist',   'C', 'B', 'mnist',   AXIS_ORDER_C);
            } else if (phase === 'C2') {
                renderTypeChart('chart-C2-type-cifar10', 'C2', 'B2', 'cifar10', AXIS_ORDER_C2);
                renderTypeChart('chart-C2-type-mnist',   'C2', 'B2', 'mnist',   AXIS_ORDER_C2);
                renderAxisSelector();
                renderModelChart('chart-C2-model-cifar10', 'C2', 'cifar10', c2SelectedAxis);
                renderModelChart('chart-C2-model-mnist',   'C2', 'mnist',   c2SelectedAxis);
            }
        });
    }

    function showBanner(msg) {
        const b = document.getElementById('banner');
        if (b) {
            b.textContent = msg;
            b.classList.add('visible');
        }
    }

    function hideBanner() {
        const b = document.getElementById('banner');
        if (b) b.classList.remove('visible');
    }

    // -- Chip filters (US-010) --------------------------------------------

    function loadFilters(phase) {
        try {
            const raw = localStorage.getItem(LS_FILTERS_PREFIX + phase);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (parsed && typeof parsed === 'object') return parsed;
            }
        } catch (e) { /* ignore */ }
        return {};
    }

    function saveFilters(phase) {
        try {
            localStorage.setItem(
                LS_FILTERS_PREFIX + phase,
                JSON.stringify(state.filters[phase] || {})
            );
        } catch (e) { /* ignore */ }
    }

    function toggleChip(facet, value) {
        const phase = state.activePhase;
        if (!state.filters[phase]) state.filters[phase] = {};
        const sel = state.filters[phase][facet] || [];
        const idx = sel.indexOf(value);
        if (idx === -1) sel.push(value);
        else sel.splice(idx, 1);
        state.filters[phase][facet] = sel;
        saveFilters(phase);
        renderChipSelections();
        applyVisibility();
    }

    function renderChipSelections() {
        const phase = state.activePhase;
        const f = state.filters[phase] || {};
        document.querySelectorAll('.chip').forEach(function (chip) {
            const facet = chip.dataset.facet;
            const value = chip.dataset.value;
            const sel = f[facet] || [];
            chip.classList.toggle('active', sel.indexOf(value) !== -1);
        });
    }

    function clearAllFilters() {
        const phase = state.activePhase;
        state.filters[phase] = {};
        saveFilters(phase);
        renderChipSelections();
        applyVisibility();
    }

    // -- Last-polled timestamp (US-014a) ----------------------------------

    function fmtRel(deltaMs) {
        const s = Math.floor(deltaMs / 1000);
        if (s < 5)    return 'just now';
        if (s < 60)   return s + 's ago';
        if (s < 3600) return Math.floor(s / 60) + 'm ago';
        return Math.floor(s / 3600) + 'h ago';
    }

    function fmtCountdown(ms) {
        if (ms < 0) ms = 0;
        const s = Math.floor(ms / 1000);
        const m = Math.floor(s / 60);
        const r = s - m * 60;
        return m + ':' + (r < 10 ? '0' + r : r);
    }

    function updateRelTs() {
        const el = document.getElementById('ts-polled');
        if (!el) return;
        if (state.lastPolledMs === null) {
            el.textContent = '—';
            el.className = '';
        } else {
            const delta = Date.now() - state.lastPolledMs;
            el.textContent = fmtRel(delta);
            el.className = (delta > STALE_THRESHOLD_MS) ? 'ts-stale' : 'ts-fresh';
        }
        // US-019: countdown to the next scheduled poll. While the tab is
        // hidden the timer is paused (visibilitychange handler); reflect
        // that with "(paused)" instead of a stale countdown.
        const cd = document.getElementById('ts-countdown');
        if (!cd) return;
        if (state.pollTimer === null) {
            cd.textContent = '(paused)';
            return;
        }
        if (state.lastPolledMs === null) {
            cd.textContent = '—';
            return;
        }
        const elapsed = Date.now() - state.lastPolledMs;
        const remaining = POLL_INTERVAL_MS - elapsed;
        cd.textContent = fmtCountdown(remaining);
    }

    // -- Data load + polling (US-013) -------------------------------------

    function ingest(doc) {
        if (!doc || !Array.isArray(doc.rows)) {
            showBanner('Final_Exp.json missing or malformed.');
            return;
        }
        // Skip no-op poll cycles.
        if (state.lastGeneratedAt && doc.generated_at === state.lastGeneratedAt
            && state.rows.length === doc.rows.length) {
            return;
        }
        hideBanner();
        state.lastGeneratedAt = doc.generated_at || null;
        state.rows = doc.rows;
        state.countsByPhase = recomputeCountsByPhase(doc.rows);
        renderRows(doc.rows);
        renderTabCounts();
        renderExecutionUsTrend(doc.rows);
        renderPhaseDRecoveryStrip(doc.rows);
        renderPhaseB2ComparisonStrip(doc.rows);
        renderPhaseC2AttributionStrip(doc.rows);
        renderThroughputCard();
        renderConfusionGallery(doc.rows);
        renderCalibrationGallery(doc.rows);
        applyActivePhase(state.activePhase);

        const gen = document.getElementById('ts-generated');
        if (gen && doc.generated_at) gen.textContent = doc.generated_at;
    }

    function readInlineDoc() {
        const tag = document.getElementById('initial-data');
        if (!tag) return null;
        try { return JSON.parse(tag.textContent); }
        catch (e) { return null; }
    }

    function loadJsonOnce() {
        return fetch(JSON_PATH + '?t=' + Date.now(), { cache: 'no-store' })
            .then(function (r) {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            })
            .then(function (doc) {
                state.fetchEverSucceeded = true;
                state.lastPolledMs = Date.now();
                ingest(doc);
                updateRelTs();
            })
            .catch(function (err) {
                // On file:// the browser blocks fetch of local files; if we
                // bootstrapped from inline data and never had a successful
                // fetch, stay silent (no banner spam). On HTTP, show a
                // soft warning.
                if (state.bootstrapFromInline && !state.fetchEverSucceeded) {
                    return; // silent on file:// — inline data is authoritative
                }
                const b = document.getElementById('banner');
                if (b) {
                    b.textContent = 'Polling failed: ' + err.message +
                        ' — showing last successful snapshot.';
                    b.classList.add('visible');
                    b.classList.add('warn');
                }
            });
    }

    function startPolling() {
        if (state.pollTimer !== null) return;
        state.pollTimer = setInterval(loadJsonOnce, POLL_INTERVAL_MS);
    }

    function stopPolling() {
        if (state.pollTimer !== null) {
            clearInterval(state.pollTimer);
            state.pollTimer = null;
        }
    }

    function bindVisibility() {
        document.addEventListener('visibilitychange', function () {
            if (document.hidden) {
                stopPolling();
            } else {
                startPolling();
                loadJsonOnce(); // catch up immediately on return
            }
        });
    }

    // -- Boot --------------------------------------------------------------

    function bindTabs() {
        document.querySelectorAll('.tab-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                applyActivePhase(btn.dataset.phase);
            });
        });
    }

    function bindChips() {
        document.querySelectorAll('.chip').forEach(function (chip) {
            chip.addEventListener('click', function () {
                toggleChip(chip.dataset.facet, chip.dataset.value);
            });
        });
        const clr = document.getElementById('clear-filters');
        if (clr) clr.addEventListener('click', clearAllFilters);
    }

    // -- US-018: Lazy learning-curve drawer ------------------------------

    const HISTORY_CACHE = new Map();   // tag -> parsed history doc
    const HISTORY_PENDING = new Map(); // tag -> in-flight Promise

    function findRowByTag(tag) {
        for (let i = 0; i < state.rows.length; i++) {
            if (state.rows[i].tag === tag) return state.rows[i];
        }
        return null;
    }

    // 2026-07-02: same-sample degradation strip shown at the top of the
    // drawer — the clean baseline plus every sibling level of the clicked
    // cell (same phase / model / dataset / axis / treatment), so the L1→L5
    // progression of the SAME validation image is visible at a glance.
    function buildDrawerSamples(row) {
        if (row.level === null || row.level === undefined) return '';
        var siblings = state.rows.filter(function (r) {
            return r.phase === row.phase && r.model === row.model
                && r.dataset === row.dataset
                && (r.axis || '') === (row.axis || '')
                && (r.treatment || '') === (row.treatment || '')
                && r.level !== null && r.level !== undefined;
        });
        siblings.sort(function (a, b) { return a.level - b.level; });
        var clean = state.rows.filter(function (r) {
            return r.phase === 'A' && r.model === row.model && r.dataset === row.dataset;
        })[0];
        var items = (clean ? [clean] : []).concat(siblings);
        if (items.length < 2) return '';
        var cells = items.map(function (r) {
            var lbl = (r.level === null || r.level === undefined) ? 'clean' : 'L' + r.level;
            var cur = r.tag === row.tag ? ' current' : '';
            var img = r.visual_core
                ? '<img src="' + r.visual_core + '" loading="lazy" decoding="async" alt="' +
                  escapeHtml(r.tag) + ' — original | degraded">'
                : '<span class="ds-missing">—</span>';
            return '<div class="ds-item' + cur + '" title="' + escapeHtml(r.tag) + '">' + img +
                   '<div class="ds-lbl">' + lbl + ' · ' + fmtAcc(r.val_acc) + '</div></div>';
        }).join('');
        return '<div class="drawer-samples-title">Same pipeline across levels — original | degraded (click to zoom)</div>' +
               '<div class="drawer-samples">' + cells + '</div>';
    }

    function openCurvesDrawer(tag) {
        const row = findRowByTag(tag);
        if (!row) return;
        const drawer = document.getElementById('curves-drawer');
        const titleEl = document.getElementById('drawer-title');
        const bodyEl = document.getElementById('drawer-body');
        if (!drawer || !titleEl || !bodyEl) return;
        titleEl.textContent = tag;
        drawer.classList.add('open');

        // Sample strip + a dedicated host for the curves so the strip
        // survives every curves-render path below.
        bodyEl.innerHTML = buildDrawerSamples(row) + '<div id="drawer-curves-host"></div>';
        const curvesEl = document.getElementById('drawer-curves-host');

        // Pending / Deferred / Failed rows: show the placeholder, don't fetch.
        if (!row.has_history) {
            curvesEl.innerHTML =
                '<div class="placeholder">No learning curves yet — cell is ' +
                escapeHtml(row.status) + '.</div>';
            return;
        }

        // INLINE PATH (primary, file://-safe): the aggregator embeds the
        // history list directly onto the row at build time. Every modern
        // browser blocks fetch() of file:// cross-origin local files, so
        // any dashboard opened via double-click MUST use the inline data —
        // there is no second chance.
        if (Array.isArray(row.history)) {
            const doc = { history: row.history };
            HISTORY_CACHE.set(tag, doc);
            renderCurves(curvesEl, doc);
            return;
        }

        // Cached: instant re-render, no network.
        if (HISTORY_CACHE.has(tag)) {
            renderCurves(curvesEl, HISTORY_CACHE.get(tag));
            return;
        }

        // FETCH FALLBACK (HTTP-served dashboards only): used when the
        // aggregator has not yet been rebuilt since the cell finished
        // training, so the row.history field is still null. Triggers a
        // file:// security error in offline mode — that error path is
        // surfaced to the user in the catch() below.
        curvesEl.innerHTML = '<div class="placeholder">Loading learning curves…</div>';

        if (!HISTORY_PENDING.has(tag)) {
            // The dashboard HTML lives at artifacts/Final_Exp.html and the run
            // dirs at <repo>/runs/final/<tag>/. Relative URL goes UP one level
            // out of artifacts/ before descending into runs/final/.
            const url = '../runs/final/' + tag + '/history.json?t=' + Date.now();
            const p = fetch(url, { cache: 'no-store' })
                .then(function (r) {
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    return r.json();
                })
                .then(function (doc) {
                    HISTORY_CACHE.set(tag, doc);
                    return doc;
                });
            HISTORY_PENDING.set(tag, p);
        }
        HISTORY_PENDING.get(tag)
            .then(function (doc) {
                // The drawer may have been closed or switched to another
                // tag before the fetch resolved; only render if still on tag.
                // Re-query the host — a re-open may have replaced the DOM.
                const host = document.getElementById('drawer-curves-host');
                if (host && titleEl.textContent === tag && drawer.classList.contains('open')) {
                    renderCurves(host, doc);
                }
            })
            .catch(function (err) {
                const host = document.getElementById('drawer-curves-host');
                if (host && titleEl.textContent === tag) {
                    host.innerHTML =
                        '<div class="placeholder">Curves not yet inlined — rebuild the dashboard (`python scripts/refresh_trackers.py --cell ' +
                        escapeHtml(tag) +
                        '`) so the history embeds into Final_Exp.json. File:// blocks fetch() of local files. Underlying error: ' +
                        escapeHtml(err.message) + '</div>';
                }
            })
            .finally(function () {
                HISTORY_PENDING.delete(tag);
            });
    }

    function closeCurvesDrawer() {
        const drawer = document.getElementById('curves-drawer');
        if (drawer) drawer.classList.remove('open');
    }

    function renderCurves(bodyEl, doc) {
        // If Plotly isn't available (offline file://, ad-blocker), fall back
        // to a tiny SVG-free table so the drawer still has *something*.
        if (typeof window.Plotly === 'undefined') {
            renderCurvesFallback(bodyEl, doc);
            return;
        }
        bodyEl.innerHTML =
            '<div id="plot-loss" class="plot-container"></div>' +
            '<div id="plot-acc"  class="plot-container"></div>';
        const hist = (doc && doc.history) || [];
        const epochs = hist.map(function (r) { return r.epoch; });
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor:  'rgba(0,0,0,0)',
            font: { color: '#e8eaf0', size: 11 },
            margin: { l: 48, r: 16, t: 32, b: 32 },
            xaxis: { title: 'Epoch', gridcolor: '#2a3040' },
            yaxis: { gridcolor: '#2a3040' },
            legend: { orientation: 'h', y: 1.16 },
        };
        const cfg = { displayModeBar: false, responsive: true };
        window.Plotly.newPlot('plot-loss', [
            { x: epochs, y: hist.map(function (r) { return r.train_loss; }),
              name: 'train_loss', mode: 'lines+markers',
              line: { color: '#4fc3f7' } },
            { x: epochs, y: hist.map(function (r) { return r.val_loss; }),
              name: 'val_loss',   mode: 'lines+markers',
              line: { color: '#ff7043' } },
        ], Object.assign({}, layout, { title: 'Loss' }), cfg);
        window.Plotly.newPlot('plot-acc', [
            { x: epochs, y: hist.map(function (r) { return r.train_acc; }),
              name: 'train_acc', mode: 'lines+markers',
              line: { color: '#9ccc65' } },
            { x: epochs, y: hist.map(function (r) { return r.val_acc; }),
              name: 'val_acc',   mode: 'lines+markers',
              line: { color: '#81c784' } },
        ], Object.assign({}, layout, { title: 'Accuracy' }), cfg);
    }

    function renderCurvesFallback(bodyEl, doc) {
        const hist = (doc && doc.history) || [];
        const rows = hist.map(function (r) {
            return '<tr><td>' + r.epoch +
                '</td><td>' + (r.train_loss === null ? '—' : r.train_loss.toFixed(4)) +
                '</td><td>' + (r.val_loss   === null ? '—' : r.val_loss.toFixed(4)) +
                '</td><td>' + (r.train_acc  === null ? '—' : (r.train_acc * 100).toFixed(2) + '%') +
                '</td><td>' + (r.val_acc    === null ? '—' : (r.val_acc   * 100).toFixed(2) + '%') +
                '</td></tr>';
        }).join('');
        bodyEl.innerHTML =
            '<div class="placeholder">Plotly not loaded — showing raw history.</div>' +
            '<table class="exp-table"><thead><tr><th>Ep</th><th>train_loss</th><th>val_loss</th><th>train_acc</th><th>val_acc</th></tr></thead><tbody>' +
            rows + '</tbody></table>';
    }

    function bindCurvesDrawer() {
        // Clicks bubble from <tr>; open the drawer for that row's tag.
        const tbody = document.getElementById('exp-tbody');
        if (tbody) {
            tbody.addEventListener('click', function (ev) {
                const tr = ev.target.closest('tr.exp-row');
                if (!tr || !tr.dataset.tag) return;
                openCurvesDrawer(tr.dataset.tag);
            });
        }
        const closeBtn = document.getElementById('drawer-close');
        if (closeBtn) closeBtn.addEventListener('click', closeCurvesDrawer);
        // Esc closes the lightbox first, then the drawer.
        document.addEventListener('keydown', function (ev) {
            if (ev.key !== 'Escape') return;
            const lb = document.getElementById('lightbox');
            if (lb && lb.classList.contains('open')) { closeLightbox(); return; }
            closeCurvesDrawer();
        });
    }

    // -- Lightbox (2026-07-02) ---------------------------------------------
    // Click any dashboard image (visual-core thumb, V6 figure, gallery PNG,
    // drawer sample) to zoom it full-screen. Bound in the capture phase so
    // the click does NOT also trigger the row drawer or the <a target=_blank>
    // wrapper the image sits inside.
    var LIGHTBOX_IMG_SELECTOR = [
        'img.visual-core-thumb',
        '.pi-figure img',
        '#confusion-gallery-body img',
        '#calibration-gallery-body img',
        '.us-trend-body img',
        '.drawer-samples img'
    ].join(', ');

    function openLightbox(src, caption) {
        var lb = document.getElementById('lightbox');
        if (!lb) return;
        document.getElementById('lightbox-img').src = src;
        document.getElementById('lightbox-cap').textContent = caption || '';
        lb.classList.add('open');
    }

    function closeLightbox() {
        var lb = document.getElementById('lightbox');
        if (lb) lb.classList.remove('open');
    }

    function bindLightbox() {
        document.addEventListener('click', function (ev) {
            var lb = document.getElementById('lightbox');
            if (lb && lb.classList.contains('open')) {
                // Any click while open closes it.
                ev.preventDefault();
                ev.stopPropagation();
                closeLightbox();
                return;
            }
            var t = ev.target;
            if (t && t.matches && t.matches(LIGHTBOX_IMG_SELECTOR)) {
                ev.preventDefault();
                ev.stopPropagation();
                openLightbox(t.currentSrc || t.src, t.alt || t.title || '');
            }
        }, true);
    }

    function bindManualRefresh() {
        // US-019: "🔄 Refresh now" forces an immediate JSON fetch but does
        // NOT touch the 10-min polling cadence (so an impatient user can't
        // accidentally turn the dashboard into a tight polling loop).
        const btn = document.getElementById('manual-refresh');
        if (!btn) return;
        btn.addEventListener('click', function () {
            btn.disabled = true;
            btn.textContent = '⟳ refreshing…';
            loadJsonOnce().finally(function () {
                btn.disabled = false;
                btn.textContent = '🔄 Refresh now';
            });
        });
    }

    function init() {
        // Restore active phase + per-phase filters from localStorage.
        let saved = 'OV';
        try {
            const v = localStorage.getItem(LS_ACTIVE_PHASE);
            if (v && PHASES.indexOf(v) !== -1) saved = v;
        } catch (e) { /* ignore */ }
        state.activePhase = saved;
        PHASES.forEach(function (p) { state.filters[p] = loadFilters(p); });

        bindTabs();
        bindChips();
        bindManualRefresh();
        bindCurvesDrawer();
        bindLightbox();
        bindVisibility();

        // Bootstrap from inline data (works on file://).
        const inline = readInlineDoc();
        if (inline) {
            state.bootstrapFromInline = true;
            state.lastPolledMs = Date.now();
            ingest(inline);
            updateRelTs();
        }

        // Start the relative-timestamp ticker (1 s).
        state.relTimer = setInterval(updateRelTs, 1000);

        // Attempt a fresh fetch (works under HTTP; silently no-ops on file://
        // unless inline bootstrap failed).
        loadJsonOnce();
        startPolling();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
"""


def _chip_html(facet: str, label: str, values: tuple[str, ...]) -> str:
    chips = "".join(
        f'<span class="chip" data-facet="{facet}" data-value="{v}">{v}</span>'
        for v in values
    )
    return (
        f'<div class="chip-group" data-chip-group="{facet}">'
        f'  <span class="chip-label">{label}</span>'
        f'  {chips}'
        f'</div>'
    )


def _html_template(initial_doc_json: str) -> str:
    """Static HTML scaffold with embedded initial data.

    The browser bootstraps from the inline `<script id="initial-data">` JSON
    (works on file:// where fetch of local files is CORS-blocked) and then
    polls `./Final_Exp.json` every 30 s for live updates (works when served
    via HTTP).
    """
    from src.experiments.cells import MODELS, DATASETS  # local — already torch-free
    from src.data.degradation_levels import AXES

    chip_model     = _chip_html("model",     "MODEL",     MODELS)
    chip_dataset   = _chip_html("dataset",   "DATASET",   DATASETS)
    chip_status    = _chip_html("status",    "STATUS",    ("Pending", "Running", "Complete", "Failed", "Deferred"))
    chip_treatment = _chip_html("treatment", "TREATMENT", PHASE_D_TREATMENTS)
    chip_axis      = _chip_html("axis",      "AXIS",      AXES)

    # JSON is embedded inside <script type="application/json"> so browsers
    # do not parse it; the bootstrap escapes "</" sequences just in case
    # (defense-in-depth — the schema cannot legally contain those).
    safe_json = initial_doc_json.replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Final Experiment Dashboard — 402 cells</title>
<style>{_CSS}</style>
<!-- US-018: Plotly is loaded with `defer` and used only when a row is
     clicked. Initial paint does NOT depend on the CDN, so an offline /
     CDN-blocked environment falls back to a raw-numbers table. -->
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" defer></script>
</head>
<body>
<div class="header">
  <h1>🔬 Final Experiment Dashboard</h1>
  <div class="subtitle">402-cell v4 campaign &middot; 3 architectures &middot; 2 datasets &middot; 5-level degradation curve &middot; V6 report phases: <strong>B1 + C2</strong></div>
  <div class="timestamps">
    Generated: <span id="ts-generated">—</span>
    &middot; Last polled: <span id="ts-polled">—</span>
    &middot; Next poll in: <span id="ts-countdown">—</span>
    <button type="button" id="manual-refresh" class="manual-refresh-btn"
            title="Force an immediate fetch (US-019). Does not change the 10-minute auto-poll cadence.">🔄 Refresh now</button>
  </div>
</div>

<div class="stats-bar">
  <div class="stat-card"><div class="stat-value"           id="stat-total">—</div>    <div class="stat-label">In Phase</div></div>
  <div class="stat-card"><div class="stat-value pending"   id="stat-pending">—</div>  <div class="stat-label">Pending</div></div>
  <div class="stat-card"><div class="stat-value running"   id="stat-running">—</div>  <div class="stat-label">Running</div></div>
  <div class="stat-card"><div class="stat-value complete"  id="stat-complete">—</div> <div class="stat-label">Complete</div></div>
  <div class="stat-card"><div class="stat-value failed"    id="stat-failed">—</div>   <div class="stat-label">Failed</div></div>
  <div class="stat-card"><div class="stat-value deferred"  id="stat-deferred">—</div> <div class="stat-label">Deferred</div></div>
  <div class="stat-card"><div class="stat-value"           id="stat-all">—</div>      <div class="stat-label">All Cells</div></div>
</div>

<!-- Execution US Trend — DESIGNER, US-006 line 418.
     Populated by JS (renderExecutionUsTrend) from the loaded doc; cards
     for execution USs (US-006…US-014) that have at least one Complete
     cell appear here. Pending USs are silently skipped so the section
     grows monotonically as each phase closes. -->
<details class="us-trend-section" id="us-trend-section">
  <summary>Execution US Trend</summary>
  <div class="us-trend-body" id="us-trend-body">
    <span class="ut-pending">No closed execution US yet.</span>
  </div>
</details>

<!-- US-030: Phase D Recovery strip — embeds artifacts/figures/phase_d_comparison.png
     once at least one Phase D cell is Complete. Until then, the body shows
     a pending placeholder. Section mounts regardless of D-on-disk state so
     the JS hook always has a target. -->
<details class="us-trend-section phase-d-recovery-section" id="phase-d-recovery-section">
  <summary>Phase D Recovery (Phase B v2 baseline vs T1/T2/T3)</summary>
  <div class="us-trend-body">
    <span class="ut-pending">Awaiting Phase D runs.</span>
  </div>
</details>

<!-- US-047 (v4) — Phase B2 comparison strip. Renders B1 vs D-T3 vs B2 vs B2nr
     once Phase B2 lands (US-040). Body shows a pending placeholder otherwise. -->
<details class="us-trend-section phase-b2-comparison-section" id="phase-b2-comparison-section">
  <summary>Phase B2 Comparison (B1 vs D-T3 vs B2 vs B2-nr)</summary>
  <div class="us-trend-body">
    <span class="ut-pending">Awaiting Phase B2 runs (US-040 / US-041).</span>
  </div>
</details>

<!-- US-047 (v4) — Phase C2 axis-attribution strip. Renders per-axis recovery
     surfaces once Phase C2 lands (US-044). -->
<details class="us-trend-section phase-c2-comparison-section" id="phase-c2-comparison-section">
  <summary>Phase C2 Axis Attribution (resolution / blur / salt_pepper)</summary>
  <div class="us-trend-body">
    <span class="ut-pending">Awaiting Phase C2 runs (US-044).</span>
  </div>
</details>

<!-- US-047 (v4) — Inference throughput card. Reads
     artifacts/figures/inference_throughput.csv once US-045 lands.
     Empty-state placeholder until then. -->
<details class="us-trend-section throughput-card-section" id="throughput-card-section">
  <summary>Inference Throughput (bf16-mixed, batch=1, 224×224)</summary>
  <div class="us-trend-body" id="throughput-body">
    <span class="ut-pending">Awaiting US-045 diagnostics (3 models × 100 warmup + 1000 measurement batches).</span>
  </div>
</details>

<!-- US-047 (v4) — Confusion-matrix gallery. Lazy-loads 24 L5 PNGs from
     artifacts/figures/confusion/ when the user expands the section. -->
<details class="us-trend-section confusion-gallery-section" id="confusion-gallery-section">
  <summary>Confusion Matrices — all L5 cells (78)</summary>
  <div class="us-trend-body" id="confusion-gallery-body">
    <span class="ut-pending">Awaiting US-045 confusion-matrix dump (24 L5 cells × seed=42).</span>
  </div>
</details>

<!-- US-047 (v4) — Calibration / ECE gallery. Lazy-loads 48 reliability
     diagrams (24 L3 + 24 L5) when the user expands the section. -->
<details class="us-trend-section calibration-gallery-section" id="calibration-gallery-section">
  <summary>Calibration Diagrams + ECE — all L3 + L5 cells (162)</summary>
  <div class="us-trend-body" id="calibration-gallery-body">
    <span class="ut-pending">Awaiting US-045 calibration / ECE diagnostics (48 cells).</span>
  </div>
</details>

<div class="tab-bar" role="tablist">
  <div class="tab-group tab-group-ov">
    <span class="tab-group-label">🏠 Start here</span>
    <div class="tab-group-btns">
      <button type="button" class="tab-btn" data-phase="OV" role="tab">Overview<span class="tab-count" id="tab-count-OV">402</span></button>
    </div>
  </div>
  <div class="tab-group tab-group-v6">
    <span class="tab-group-label">📄 V6 report phases — in depth</span>
    <div class="tab-group-btns">
      <button type="button" class="tab-btn" data-phase="B" role="tab">Phase B (B1)<span class="tab-count" id="tab-count-B">{EXPECTED_COUNTS['B']}</span></button>
      <button type="button" class="tab-btn" data-phase="C2" role="tab">Phase C2<span class="tab-count" id="tab-count-C2">{EXPECTED_COUNTS_WITH_ALL['C2']}</span></button>
    </div>
  </div>
  <div class="tab-group tab-group-support">
    <span class="tab-group-label">Supporting phases</span>
    <div class="tab-group-btns">
      <button type="button" class="tab-btn" data-phase="A" role="tab">Phase A<span class="tab-count" id="tab-count-A">{EXPECTED_COUNTS['A']}</span></button>
      <button type="button" class="tab-btn" data-phase="B2" role="tab">Phase B2<span class="tab-count" id="tab-count-B2">{EXPECTED_COUNTS_WITH_ALL['B2']}</span></button>
      <button type="button" class="tab-btn" data-phase="B2nr" role="tab">Phase B2-nr<span class="tab-count" id="tab-count-B2nr">{EXPECTED_COUNTS_WITH_ALL['B2nr']}</span></button>
      <button type="button" class="tab-btn" data-phase="C" role="tab">Phase C<span class="tab-count" id="tab-count-C">{EXPECTED_COUNTS['C']}</span></button>
      <button type="button" class="tab-btn" data-phase="D" role="tab">Phase D<span class="tab-count" id="tab-count-D">{EXPECTED_COUNTS_WITH_D['D']}</span></button>
    </div>
  </div>
</div>

<!-- V6 per-phase intro (2026-07-02): motivation + findings + V6 figures +
     interactive Plotly charts. Populated by renderPhaseIntro(phase). -->
<section id="phase-intro" class="phase-intro" style="display:none"></section>

<div class="filters">
  {chip_model}
  {chip_dataset}
  {chip_status}
  {chip_treatment}
  {chip_axis}
  <div class="filter-actions">
    <button type="button" class="clear-all" id="clear-filters">Clear all</button>
    <span class="visible-count" id="visible-count">— visible</span>
  </div>
</div>

<div id="banner" class="banner" role="alert"></div>

<div class="exp-table-wrap">
  <table class="exp-table">
    <thead>
      <tr>
        <th>Tag</th>
        <th>Model</th>
        <th>Dataset</th>
        <th>Phase</th>
        <th>Level</th>
        <th>Treatment</th>
        <th>Degradation</th>
        <th>Visual</th>
        <th>Status</th>
        <th>val_acc</th>
        <th>Epochs</th>
        <th>Runtime</th>
        <th>Curves</th>
      </tr>
    </thead>
    <tbody id="exp-tbody"></tbody>
  </table>
</div>

<!-- 2026-07-02: full-screen lightbox. Any dashboard image click zooms it;
     click anywhere / Esc closes. -->
<div id="lightbox" class="lightbox" role="dialog" aria-label="Image zoom">
  <figure>
    <img id="lightbox-img" src="" alt="">
    <figcaption id="lightbox-cap"></figcaption>
  </figure>
</div>

<!-- US-018: lazy-loaded learning-curve drawer. Hidden via CSS transform
     until a row is clicked; populated by `openCurvesDrawer(tag)`. -->
<aside id="curves-drawer" class="curves-drawer" aria-hidden="true">
  <div class="drawer-header">
    <span class="drawer-title" id="drawer-title">—</span>
    <button type="button" class="drawer-close" id="drawer-close"
            aria-label="Close learning curves drawer">×</button>
  </div>
  <div class="drawer-body" id="drawer-body">
    <div class="placeholder">Click a row to load learning curves.</div>
  </div>
</aside>

<script type="application/json" id="initial-data">{safe_json}</script>
<script>{_JS}</script>
</body>
</html>
"""


def build_dashboard(
    out_path: Path = _DEFAULT_OUT,
    runs_root: Path = _DEFAULT_RUNS_ROOT,
    *,
    thumbs_dir: Optional[Path] = None,  # accepted for back-compat; unused
) -> dict:
    """Render the FINAL_EXP Dashboard HTML to `out_path`.

    Embeds the current FinalExpDoc as inline JSON so the dashboard works
    without an HTTP server (browsers block fetch() of local files when
    the page is opened via file://). The same JSON is also written as a
    sibling file by `main()` for HTTP-served scenarios where polling
    needs a fresh source.

    Returns a small summary dict (stable shape across stories so existing
    callers — `scripts/refresh_trackers.refresh_final_exp_html` and
    `src/tests/test_dashboard.py` — keep working).
    """
    del thumbs_dir  # unused — pilot-styled dashboard does not embed thumbnails
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the doc directly (torch-free) so the inline embed reflects the
    # current run state, even if the sibling JSON write hasn't happened yet.
    import json as _json
    from src.tools.build_final_exp_json import build_doc  # lazy import
    doc = build_doc(runs_root=Path(runs_root))
    initial_doc_json = _json.dumps(doc, ensure_ascii=False)

    body = _html_template(initial_doc_json=initial_doc_json)
    out_path.write_text(body, encoding="utf-8")
    # US-030: derive Phase D inclusion from the row count the aggregator
    # actually produced (it already consulted `phase_d_present_on_disk`).
    # Keeps the legacy 186-row summary contract byte-identical when no
    # `final_D_*` directories exist, and reports 276 once they do.
    # US-048 (v4): count v4 phases from the actual row data.
    from collections import Counter as _Counter
    phase_count = _Counter(r.get("phase") for r in doc["rows"])
    phase_d_rows = phase_count.get("D", 0)
    return {
        "rows": len(doc["rows"]),
        "phase_a": phase_count.get("A", EXPECTED_COUNTS["A"]),
        "phase_b": phase_count.get("B", EXPECTED_COUNTS["B"]),
        "phase_c": phase_count.get("C", EXPECTED_COUNTS["C"]),
        "phase_d": phase_d_rows,
        "phase_b2": phase_count.get("B2", 0),
        "phase_b2nr": phase_count.get("B2nr", 0),
        "phase_c2": phase_count.get("C2", 0),
        "out": str(out_path),
        "embedded_rows": len(doc["rows"]),
        "generated_at": doc["generated_at"],
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build artifacts/Final_Exp.html (FINAL_EXP Dashboard).")
    p.add_argument("--out", default=str(_DEFAULT_OUT),
                   help=f"Output HTML path (default: {_DEFAULT_OUT}).")
    p.add_argument("--runs-root", default=str(_DEFAULT_RUNS_ROOT),
                   help=f"Per-cell metrics root, forwarded to the aggregator (default: {_DEFAULT_RUNS_ROOT}).")
    # Accepted for back-compat with refresh_trackers.refresh_final_exp_html /
    # legacy callers that still pass --thumbs-dir; the new dashboard ignores it.
    p.add_argument("--thumbs-dir", default=None, help=argparse.SUPPRESS)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)

    # 1. Aggregate per-cell metrics into Final_Exp.json (US-002 contract).
    #    Failure here logs to stderr but does not block HTML render — the
    #    browser will surface a banner if the JSON is missing.
    from src.tools.build_final_exp_json import build_final_exp_json  # lazy import
    try:
        json_summary = build_final_exp_json(runs_root=Path(args.runs_root))
        c = json_summary["counts"]
        print(
            f"final_exp_json: {json_summary['rows']} rows "
            f"(pending={c['pending']}, running={c['running']}, "
            f"complete={c['complete']}, failed={c['failed']}, "
            f"deferred={c.get('deferred', 0)}) "
            f"-> {json_summary['out']}"
        )
    except Exception as e:  # noqa: BLE001 — keep HTML render alive on aggregator failure
        print(
            f"[warn] final_exp_json aggregator failed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )

    # 2. Render the static HTML scaffold; rows are JS-rendered from the JSON.
    summary = build_dashboard(out_path=Path(args.out), runs_root=Path(args.runs_root))
    print(
        f"dashboard: shell rendered "
        f"(A={summary['phase_a']}, B={summary['phase_b']}, C={summary['phase_c']}, "
        f"D={summary['phase_d']}, "
        f"B2={summary.get('phase_b2', 0)}, "
        f"B2nr={summary.get('phase_b2nr', 0)}, "
        f"C2={summary.get('phase_c2', 0)}, "
        f"total={summary['rows']}) -> {summary['out']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
