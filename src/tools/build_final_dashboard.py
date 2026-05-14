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

from src.experiments.cells import EXPECTED_COUNTS, EXPECTED_TOTAL


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
    const PHASES = ['A', 'B', 'C'];
    // US-019: 10-minute polling cadence. The aggregator pushes per-cell
    // updates after every Trainer.fit (incremental --cell mode), so the
    // dashboard does NOT need a tight polling loop — once every 10 min is
    // sufficient and avoids burning CPU on an idle tab.
    const POLL_INTERVAL_MS = 600000;
    const STALE_THRESHOLD_MS = 1800000; // visibility-paused > 30 min -> orange

    const state = {
        rows: [],
        countsByPhase: { A: {}, B: {}, C: {} },
        activePhase: 'A',
        filters: { A: {}, B: {}, C: {} },
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
            tr.innerHTML = (
                '<td class="id-cell">' + escapeHtml(row.tag) + '</td>' +
                '<td>' + escapeHtml(row.model) + '</td>' +
                '<td>' + escapeHtml(row.dataset) + '</td>' +
                '<td>' + escapeHtml(row.phase) + '</td>' +
                '<td>' + levelBadgeHtml(row) + '</td>' +
                '<td>' + paramsHtml(row) + '</td>' +
                '<td>' + visualCoreHtml(row) + '</td>' +
                '<td>' + statusPillHtml(row.status, row) + '</td>' +
                '<td class="num-cell acc-cell ' + accClass(row.val_acc) + '">' +
                    fmtAcc(row.val_acc) + '</td>' +
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
        const out = { A: {}, B: {}, C: {} };
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
        const c = state.countsByPhase[activePhase] || {};
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
        if (PHASES.indexOf(phase) === -1) phase = 'A';
        state.activePhase = phase;
        try { localStorage.setItem(LS_ACTIVE_PHASE, phase); } catch (e) { /* ignore */ }

        // Tab styling
        document.querySelectorAll('.tab-btn').forEach(function (b) {
            b.classList.toggle('active', b.dataset.phase === phase);
        });
        // Axis chip group is only meaningful on Phase C — hide on A/B.
        const axisGroup = document.querySelector('[data-chip-group="axis"]');
        if (axisGroup) axisGroup.style.display = (phase === 'C') ? '' : 'none';

        // Restore this phase's chip selections in the UI.
        renderChipSelections();
        applyVisibility();
        renderStatsBar(phase);
    }

    function renderTabCounts() {
        PHASES.forEach(function (p) {
            const el = document.getElementById('tab-count-' + p);
            const c = state.countsByPhase[p] || {};
            if (el) el.textContent = c.total !== undefined ? c.total : '—';
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

    function openCurvesDrawer(tag) {
        const row = findRowByTag(tag);
        if (!row) return;
        const drawer = document.getElementById('curves-drawer');
        const titleEl = document.getElementById('drawer-title');
        const bodyEl = document.getElementById('drawer-body');
        if (!drawer || !titleEl || !bodyEl) return;
        titleEl.textContent = tag;
        drawer.classList.add('open');

        // Pending / Deferred / Failed rows: show the placeholder, don't fetch.
        if (!row.has_history) {
            bodyEl.innerHTML =
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
            renderCurves(bodyEl, doc);
            return;
        }

        // Cached: instant re-render, no network.
        if (HISTORY_CACHE.has(tag)) {
            renderCurves(bodyEl, HISTORY_CACHE.get(tag));
            return;
        }

        // FETCH FALLBACK (HTTP-served dashboards only): used when the
        // aggregator has not yet been rebuilt since the cell finished
        // training, so the row.history field is still null. Triggers a
        // file:// security error in offline mode — that error path is
        // surfaced to the user in the catch() below.
        bodyEl.innerHTML = '<div class="placeholder">Loading learning curves…</div>';

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
                if (titleEl.textContent === tag && drawer.classList.contains('open')) {
                    renderCurves(bodyEl, doc);
                }
            })
            .catch(function (err) {
                if (titleEl.textContent === tag) {
                    bodyEl.innerHTML =
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
        // Esc closes the drawer.
        document.addEventListener('keydown', function (ev) {
            if (ev.key === 'Escape') closeCurvesDrawer();
        });
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
        let saved = 'A';
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

    chip_model   = _chip_html("model",   "MODEL",   MODELS)
    chip_dataset = _chip_html("dataset", "DATASET", DATASETS)
    chip_status  = _chip_html("status",  "STATUS",  ("Pending", "Running", "Complete", "Failed", "Deferred"))
    chip_axis    = _chip_html("axis",    "AXIS",    AXES)

    # JSON is embedded inside <script type="application/json"> so browsers
    # do not parse it; the bootstrap escapes "</" sequences just in case
    # (defense-in-depth — the schema cannot legally contain those).
    safe_json = initial_doc_json.replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Final Experiment Dashboard — 186 cells</title>
<style>{_CSS}</style>
<!-- US-018: Plotly is loaded with `defer` and used only when a row is
     clicked. Initial paint does NOT depend on the CDN, so an offline /
     CDN-blocked environment falls back to a raw-numbers table. -->
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" defer></script>
</head>
<body>
<div class="header">
  <h1>🔬 Final Experiment Dashboard</h1>
  <div class="subtitle">186-cell Research Phase &middot; 3 architectures &middot; 2 datasets &middot; 5-level degradation curve</div>
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
<details class="us-trend-section" id="us-trend-section" open>
  <summary>Execution US Trend</summary>
  <div class="us-trend-body" id="us-trend-body">
    <span class="ut-pending">No closed execution US yet.</span>
  </div>
</details>

<div class="tab-bar" role="tablist">
  <button type="button" class="tab-btn" data-phase="A" role="tab">Phase A<span class="tab-count" id="tab-count-A">{EXPECTED_COUNTS['A']}</span></button>
  <button type="button" class="tab-btn" data-phase="B" role="tab">Phase B<span class="tab-count" id="tab-count-B">{EXPECTED_COUNTS['B']}</span></button>
  <button type="button" class="tab-btn" data-phase="C" role="tab">Phase C<span class="tab-count" id="tab-count-C">{EXPECTED_COUNTS['C']}</span></button>
</div>

<div class="filters">
  {chip_model}
  {chip_dataset}
  {chip_status}
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
    return {
        "rows": EXPECTED_TOTAL,
        "phase_a": EXPECTED_COUNTS["A"],
        "phase_b": EXPECTED_COUNTS["B"],
        "phase_c": EXPECTED_COUNTS["C"],
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
        f"total={summary['rows']}) -> {summary['out']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
