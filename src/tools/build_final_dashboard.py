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
    margin-bottom: 18px;
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
    position: sticky;
    top: 56px; /* below tab-bar */
    z-index: 1;
    user-select: none;
    white-space: nowrap;
}
.exp-table th:hover {
    color: var(--text);
    background: var(--surface2);
}
.exp-table td {
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 0.86em;
    vertical-align: middle;
}
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

/* Banner for fetch failures */
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
"""

# Inline JS: fetches Final_Exp.json on load and populates the table + stats.
# US-005 (load + render), US-006 (tab switching), US-007 (level badge).
# Filters / sort / polling / presets land in US-008..US-014.
_JS = r"""
(function () {
    'use strict';

    const JSON_PATH = './Final_Exp.json';
    const LS_ACTIVE_PHASE = 'final_exp.active_phase';
    const PHASES = ['A', 'B', 'C'];

    const state = {
        rows: [],
        countsByPhase: { A: {}, B: {}, C: {} },
        activePhase: 'A',
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

    function statusPillHtml(status) {
        const cls = status.toLowerCase();
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
                '<td>' + statusPillHtml(row.status) + '</td>' +
                '<td class="num-cell acc-cell ' + accClass(row.val_acc) + '">' +
                    fmtAcc(row.val_acc) + '</td>' +
                '<td class="num-cell">' +
                    (row.epochs_run !== null && row.epochs_run !== undefined ? row.epochs_run : '—') +
                '</td>' +
                '<td class="num-cell">' + fmtRuntime(row.runtime_s) + '</td>'
            );
            frag.appendChild(tr);
        });
        tbody.replaceChildren(frag);
    }

    function recomputeCountsByPhase(rows) {
        const out = { A: {}, B: {}, C: {} };
        PHASES.forEach(function (p) {
            out[p] = { total: 0, pending: 0, running: 0, complete: 0, failed: 0,
                       best_val_acc: null };
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
        setStat('stat-best',     c.best_val_acc === null || c.best_val_acc === undefined
                                  ? '—' : fmtAcc(c.best_val_acc));
        // Global "All 186" indicator
        const all = state.rows.length;
        setStat('stat-all', all);
    }

    function applyActivePhase(phase) {
        if (PHASES.indexOf(phase) === -1) phase = 'A';
        state.activePhase = phase;
        try { localStorage.setItem(LS_ACTIVE_PHASE, phase); } catch (e) { /* ignore */ }

        // Tab styling
        document.querySelectorAll('.tab-btn').forEach(function (b) {
            b.classList.toggle('active', b.dataset.phase === phase);
        });
        // Row visibility — display none/'' (parity with pilot's mechanism)
        document.querySelectorAll('tr.exp-row').forEach(function (tr) {
            tr.style.display = (tr.dataset.phase === phase) ? '' : 'none';
        });
        renderStatsBar(phase);
    }

    function renderTabCounts() {
        PHASES.forEach(function (p) {
            const el = document.getElementById('tab-count-' + p);
            const c = state.countsByPhase[p] || {};
            if (el) el.textContent = c.total !== undefined ? c.total : '—';
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

    // -- Data load ---------------------------------------------------------

    function ingest(doc) {
        if (!doc || !Array.isArray(doc.rows)) {
            showBanner('Final_Exp.json missing or malformed.');
            return;
        }
        hideBanner();
        state.rows = doc.rows;
        state.countsByPhase = recomputeCountsByPhase(doc.rows);
        renderRows(doc.rows);
        renderTabCounts();
        applyActivePhase(state.activePhase);

        // Header generated_at
        const gen = document.getElementById('ts-generated');
        if (gen && doc.generated_at) gen.textContent = doc.generated_at;
    }

    function loadJson() {
        return fetch(JSON_PATH, { cache: 'no-store' })
            .then(function (r) {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            })
            .then(ingest)
            .catch(function (err) {
                showBanner('Could not load ' + JSON_PATH +
                           ' — run scripts/refresh_trackers.py. (' + err.message + ')');
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

    function init() {
        let saved = 'A';
        try {
            const v = localStorage.getItem(LS_ACTIVE_PHASE);
            if (v && PHASES.indexOf(v) !== -1) saved = v;
        } catch (e) { /* ignore */ }
        state.activePhase = saved;
        bindTabs();
        loadJson();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
"""


def _html_template(json_sibling: str) -> str:
    """Static HTML scaffold. All rows + counts are populated client-side from JSON_PATH."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Final Experiment Dashboard — 186 cells</title>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
  <h1>🔬 Final Experiment Dashboard</h1>
  <div class="subtitle">186-cell Research Phase &middot; 3 architectures &middot; 2 datasets &middot; 5-level degradation curve</div>
  <div class="timestamps">
    Generated: <span id="ts-generated">—</span>
  </div>
</div>

<div class="stats-bar">
  <div class="stat-card"><div class="stat-value"           id="stat-total">—</div>    <div class="stat-label">In Phase</div></div>
  <div class="stat-card"><div class="stat-value pending"   id="stat-pending">—</div>  <div class="stat-label">Pending</div></div>
  <div class="stat-card"><div class="stat-value running"   id="stat-running">—</div>  <div class="stat-label">Running</div></div>
  <div class="stat-card"><div class="stat-value complete"  id="stat-complete">—</div> <div class="stat-label">Complete</div></div>
  <div class="stat-card"><div class="stat-value failed"    id="stat-failed">—</div>   <div class="stat-label">Failed</div></div>
  <div class="stat-card"><div class="stat-value"           id="stat-best">—</div>     <div class="stat-label">Best val_acc</div></div>
  <div class="stat-card"><div class="stat-value"           id="stat-all">—</div>      <div class="stat-label">All Cells</div></div>
</div>

<div class="tab-bar" role="tablist">
  <button type="button" class="tab-btn" data-phase="A" role="tab">Phase A<span class="tab-count" id="tab-count-A">{EXPECTED_COUNTS['A']}</span></button>
  <button type="button" class="tab-btn" data-phase="B" role="tab">Phase B<span class="tab-count" id="tab-count-B">{EXPECTED_COUNTS['B']}</span></button>
  <button type="button" class="tab-btn" data-phase="C" role="tab">Phase C<span class="tab-count" id="tab-count-C">{EXPECTED_COUNTS['C']}</span></button>
</div>

<div class="filters">
  <span class="filters-placeholder">Filters land in US-008+ (model / dataset / status / level / axis / search / presets).</span>
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
        <th>Status</th>
        <th>val_acc</th>
        <th>Epochs</th>
        <th>Runtime</th>
      </tr>
    </thead>
    <tbody id="exp-tbody"></tbody>
  </table>
</div>

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

    Returns a small summary dict (stable shape across stories so existing
    callers — `scripts/refresh_trackers.refresh_final_exp_html` and
    `src/tests/test_dashboard.py` — keep working).
    """
    del thumbs_dir  # unused — pilot-styled dashboard does not embed thumbnails
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = _html_template(json_sibling=_JSON_SIBLING_NAME)
    out_path.write_text(body, encoding="utf-8")
    return {
        "rows": EXPECTED_TOTAL,
        "phase_a": EXPECTED_COUNTS["A"],
        "phase_b": EXPECTED_COUNTS["B"],
        "phase_c": EXPECTED_COUNTS["C"],
        "out": str(out_path),
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
            f"complete={c['complete']}, failed={c['failed']}) "
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
